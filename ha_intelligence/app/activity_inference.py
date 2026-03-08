"""Activity inference from zone + device data for HA Intelligence."""

import json
import logging
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Fallback zone types per room when no EPL zone data available
ROOM_ZONE_DEFAULTS = {
    'kontor': 'skrivebord',
    'sovevarelse': 'seng',
    'alrum': 'sofa_tv',
    'darwins_vaerelse': 'sofa_tv',
    'koekken': 'koekken',
    'gang': 'indgang',
    'udestuen': 'sofa_tv',
    'badevaerelse': None,
}

# Default activity candidates per zone type
ZONE_ACTIVITIES = {
    'skrivebord': ['arbejder', 'lektier', 'computer'],
    'seng': ['sover', 'hviler', 'læser'],
    'sofa_tv': ['ser_tv', 'spiller', 'slapper_af'],
    'gulv_leg': ['leger', 'træner'],
    'laenestol': ['slapper_af', 'læser'],
    'spisebord': ['spiser', 'arbejder', 'lektier'],
    'indgang': ['ankommer', 'gaar'],
    'koekken': ['laver_mad', 'rydder_op'],
}

# Device → activity boost
DEVICE_ACTIVITIES = {
    'tv_on': 'ser_tv',
    'playstation_on': 'spiller',
    'pc_on': 'computer',
}


class ActivityInference:
    """Infers activities from EPL zone data + device states."""

    def __init__(self, options: dict, db, mqtt_publisher,
                 feedback_engine=None):
        self.db = db
        self.mqtt = mqtt_publisher
        self.feedback_engine = feedback_engine
        self.active = options.get('feedback_activity_enabled', True)

        # Parse zone config
        zone_raw = options.get('activity_zone_config', '{}')
        try:
            self.zone_config = json.loads(zone_raw) if isinstance(
                zone_raw, str) else zone_raw
        except (json.JSONDecodeError, TypeError):
            self.zone_config = {}

        # Parse device sensors
        device_raw = options.get('activity_device_sensors', '{}')
        try:
            self.device_sensors = json.loads(device_raw) if isinstance(
                device_raw, str) else device_raw
        except (json.JSONDecodeError, TypeError):
            self.device_sensors = {}

        # HA Supervisor API
        self._ha_url = os.environ.get(
            'SUPERVISOR_URL', 'http://supervisor/core')
        self._ha_token = os.environ.get('SUPERVISOR_TOKEN', '')

        # Current activities per person
        self._current = {}

        logger.info(
            f"ActivityInference initialized "
            f"(active={self.active}, "
            f"rooms={len(self.zone_config)}, "
            f"devices={len(self.device_sensors)})"
        )

    # ── HA API helpers ──────────────────────────────────────

    def _get_ha_state(self, entity_id: str) -> str | None:
        """Get a single entity state from HA Supervisor API."""
        if not self._ha_token:
            return None
        url = f"{self._ha_url}/api/states/{entity_id}"
        req = urllib.request.Request(url, headers={
            'Authorization': f'Bearer {self._ha_token}',
            'Content-Type': 'application/json',
        })
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data.get('state')
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError) as e:
            logger.debug(f"Failed to get state for {entity_id}: {e}")
            return None

    def _guess_zone(self, room_slug: str) -> str | None:
        """Determine zone type for a room.

        Priority: zone_config from options → ROOM_ZONE_DEFAULTS fallback.
        """
        room_config = self.zone_config.get(room_slug, {})
        zones = room_config.get('zones', {})
        if zones:
            # Return first zone value (primary zone)
            return next(iter(zones.values()), None)
        return ROOM_ZONE_DEFAULTS.get(room_slug)

    def _fetch_device_states(self, room_slug: str) -> dict:
        """Fetch device states for a room from HA.

        Checks device_sensors config and zone_config for device entities.
        Returns: {'tv_on': True/False, 'playstation_on': ..., 'pc_on': ...}
        """
        states = {}

        # Gather device entity IDs from both config sources
        device_entities = {}

        # From activity_device_sensors config (flat: {"tv": "sensor.tv_power", ...})
        room_devices = self.device_sensors.get(room_slug, {})
        if isinstance(room_devices, dict):
            device_entities.update(room_devices)

        # From zone_config devices list
        room_config = self.zone_config.get(room_slug, {})
        for entity_id in room_config.get('devices', []):
            # Infer device key from entity_id
            eid_lower = entity_id.lower()
            if 'playstation' in eid_lower or 'ps5' in eid_lower or 'ps4' in eid_lower:
                device_entities.setdefault('playstation', entity_id)
            elif 'tv' in eid_lower or 'samsung' in eid_lower:
                device_entities.setdefault('tv', entity_id)
            elif 'pc' in eid_lower or 'computer' in eid_lower:
                device_entities.setdefault('pc', entity_id)

        # Fetch states from HA
        for key, entity_id in device_entities.items():
            state = self._get_ha_state(entity_id)
            if state is None:
                continue

            # Determine if device is "on"
            is_on = False
            if state in ('on', 'playing', 'paused', 'idle', 'standby'):
                is_on = True
            elif state not in ('off', 'unavailable', 'unknown'):
                # Numeric state (e.g. power sensor in watts)
                try:
                    is_on = float(state) > 10  # >10W = on
                except (ValueError, TypeError):
                    pass

            # Map to standard device keys
            if key in ('tv', 'tv_power'):
                states['tv_on'] = is_on
            elif key in ('playstation', 'ps5', 'ps4'):
                states['playstation_on'] = is_on
            elif key in ('pc', 'computer'):
                states['pc_on'] = is_on

        return states

    # ── Activity inference ────────────────────────────────────

    def infer_activity(self, person_slug: str, person_name: str,
                        room_slug: str, room_name: str,
                        zone: str = None,
                        device_states: dict = None) -> dict:
        """Infer activity for a person in a room.

        Returns: {activity, confidence, source, candidates}
        """
        if not self.active:
            return {'activity': 'ukendt', 'confidence': 0.0,
                    'source': 'disabled', 'candidates': []}

        # Auto-detect zone if not provided
        if zone is None:
            zone = self._guess_zone(room_slug)

        # Auto-detect device states if not provided
        if device_states is None:
            device_states = self._fetch_device_states(room_slug)
        else:
            device_states = device_states or {}

        # 1. Get candidates from zone type
        candidates = list(ZONE_ACTIVITIES.get(zone or '', ['ukendt']))

        # 2. Device state boosts specific activities
        for device_key, activity in DEVICE_ACTIVITIES.items():
            if device_states.get(device_key):
                if activity not in candidates:
                    candidates.insert(0, activity)
                else:
                    candidates.remove(activity)
                    candidates.insert(0, activity)

        # 3. Look up learned activity from DB
        learned = self.db.lookup_activity(
            person_slug, room_slug, zone or '', device_states)

        if learned and learned['confirmed_count'] >= 2:
            activity = learned['activity']
            # Confidence based on confirmation count
            conf = min(0.5 + (learned['confirmed_count'] * 0.1), 0.95)
            source = 'learned'
        elif candidates:
            activity = candidates[0]
            conf = 0.3  # Low confidence for zone-only inference
            source = 'zone_inference'
        else:
            activity = 'ukendt'
            conf = 0.1
            source = 'fallback'

        result = {
            'activity': activity,
            'confidence': conf,
            'source': source,
            'candidates': candidates[:5],
            'zone': zone,
            'devices': device_states,
        }

        # 4. Ask user if confidence is low
        if (self.feedback_engine and conf < 0.5 and
                self.feedback_engine.should_ask(conf)):
            self.feedback_engine.ask_activity(
                person_slug, person_name, room_name,
                candidates[:4], conf,
                room_slug=room_slug, zone=zone or '',
                device_states=device_states)

        # 5. Publish activity sensor
        self._current[person_slug] = result
        self.mqtt.publish_activity(
            person_slug=person_slug,
            person_name=person_name,
            state=activity,
            attributes={
                'room': room_slug,
                'zone': zone or 'unknown',
                'confidence': conf,
                'source': source,
                'candidates': candidates[:5],
                'devices': device_states,
            }
        )

        return result

    def learn_from_feedback(self, person_slug: str, room_slug: str,
                             zone: str, devices_state: dict,
                             activity: str):
        """Store a confirmed activity in the learned_activities table."""
        self.db.upsert_learned_activity(
            person=person_slug,
            room=room_slug,
            zone=zone or '',
            devices_state=devices_state,
            activity=activity,
        )
        logger.info(
            f"Learned activity: {person_slug} in "
            f"{room_slug}/{zone} = {activity}"
        )

    def get_current_activities(self) -> dict:
        """Get current activities for all tracked persons."""
        return dict(self._current)
