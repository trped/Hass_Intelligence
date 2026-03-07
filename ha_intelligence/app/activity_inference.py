"""Activity inference from zone + device data for HA Intelligence."""

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

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

        # Current activities per person
        self._current = {}

        logger.info(
            f"ActivityInference initialized "
            f"(active={self.active}, "
            f"rooms={len(self.zone_config)}, "
            f"devices={len(self.device_sensors)})"
        )

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
                candidates[:4], conf)

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
