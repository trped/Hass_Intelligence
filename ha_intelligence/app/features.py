"""Feature extraction pipeline for HA Intelligence ML models.

Converts raw HA state changes into feature vectors suitable for
River online ML models. All features are returned as dict[str, float|int].
"""

import math
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

TWO_PI = 2 * math.pi


class FeatureExtractor:
    """Extracts features from HA states for ML models."""

    def __init__(self, registry=None):
        self.registry = registry
        # Motion frequency tracking: area_id -> list of timestamps
        self._motion_events = defaultdict(list)
        # Light/media state tracking: entity_id -> state
        self._context_states = {}

    # ── Time features ────────────────────────────────────────────

    @staticmethod
    def extract_time_features(timestamp: Optional[datetime] = None) -> dict:
        """Extract cyclic time features from a timestamp.

        Returns sin/cos encoded hour and weekday plus is_weekend flag.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        hour = timestamp.hour + timestamp.minute / 60.0
        weekday = timestamp.weekday()  # 0=Monday

        return {
            'hour_sin': math.sin(TWO_PI * hour / 24),
            'hour_cos': math.cos(TWO_PI * hour / 24),
            'weekday_sin': math.sin(TWO_PI * weekday / 7),
            'weekday_cos': math.cos(TWO_PI * weekday / 7),
            'is_weekend': 1 if weekday >= 5 else 0,
        }

    # ── Room features ────────────────────────────────────────────

    def extract_room_features(self, area_id: str, room_state: dict,
                               timestamp: Optional[datetime] = None) -> dict:
        """Extract features for room occupancy prediction.

        Args:
            area_id: The room area identifier
            room_state: Dict with 'sensors' and 'last_occupied' from SensorEngine
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        features = self.extract_time_features(timestamp)

        sensors = room_state.get('sensors', {})
        active = sum(1 for v in sensors.values() if v == 'on')
        total = len(sensors)

        features.update({
            'sensors_active': active,
            'sensors_total': total,
            'active_ratio': active / total if total > 0 else 0.0,
        })

        # Minutes since last motion
        last_occ = room_state.get('last_occupied')
        if last_occ:
            try:
                last_dt = datetime.fromisoformat(last_occ)
                delta = (timestamp - last_dt).total_seconds() / 60.0
                features['minutes_since_motion'] = min(delta, 1440)  # cap at 24h
            except (ValueError, TypeError):
                features['minutes_since_motion'] = 1440
        else:
            features['minutes_since_motion'] = 1440

        # Motion frequency (events in last 5 minutes)
        features['motion_freq_5min'] = self._get_motion_frequency(area_id, timestamp, minutes=5)

        # Room context from registry (lights, media)
        context = self._get_room_context(area_id)
        features.update(context)

        return features

    # ── Person features ──────────────────────────────────────────

    def extract_person_features(self, person_entity: str, person_state: dict,
                                 rooms_with_motion: int = 0,
                                 timestamp: Optional[datetime] = None,
                                 person_room: dict = None) -> dict:
        """Extract features for person activity prediction.

        Args:
            person_entity: The person entity_id
            person_state: Dict with 'ha_state', 'source' from SensorEngine
            rooms_with_motion: Number of rooms with active motion sensors
            timestamp: Optional timestamp (defaults to now)
            person_room: Optional BLE room data from SensorEngine._person_rooms
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        features = self.extract_time_features(timestamp)

        ha_state = person_state.get('ha_state', 'unknown')
        is_home = 1 if ha_state == 'home' else 0

        features.update({
            'is_home': is_home,
            'rooms_with_motion': rooms_with_motion,
        })

        # Minutes since last state change (if tracked)
        last_change = person_state.get('last_changed')
        if last_change:
            try:
                last_dt = datetime.fromisoformat(last_change)
                delta = (timestamp - last_dt).total_seconds() / 60.0
                features['minutes_since_change'] = min(delta, 1440)
            except (ValueError, TypeError):
                features['minutes_since_change'] = 1440
        else:
            features['minutes_since_change'] = 0

        # Tracker source type
        source = person_state.get('source', '')
        features['source_gps'] = 1 if 'gps' in str(source).lower() else 0
        features['source_router'] = 1 if 'router' in str(source).lower() else 0
        features['source_ble'] = 1 if 'ble' in str(source).lower() or 'bermuda' in str(source).lower() else 0

        # BLE person-room features (Phase 1)
        if person_room:
            room_source = person_room.get('source', 'none')
            features['ble_area_known'] = 1 if room_source in ('ble', 'motion_fallback') else 0
            ble_dist = person_room.get('distance')
            features['ble_distance'] = float(ble_dist) if ble_dist is not None else 0.0

            # Minutes in current room
            room_entered = person_room.get('room_entered_at')
            if room_entered:
                try:
                    entered_dt = datetime.fromisoformat(room_entered)
                    minutes = (timestamp - entered_dt).total_seconds() / 60.0
                    features['minutes_in_current_room'] = min(minutes, 1440)
                except (ValueError, TypeError):
                    features['minutes_in_current_room'] = 0.0
            else:
                features['minutes_in_current_room'] = 0.0
        else:
            features['ble_area_known'] = 0
            features['ble_distance'] = 0.0
            features['minutes_in_current_room'] = 0.0

        return features

    # ── Motion tracking ──────────────────────────────────────────

    def update_motion_tracking(self, area_id: str,
                                timestamp: Optional[datetime] = None):
        """Record a motion event for frequency calculation."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self._motion_events[area_id].append(timestamp)

        # Keep only last 30 minutes of events
        cutoff = timestamp - timedelta(minutes=30)
        self._motion_events[area_id] = [
            t for t in self._motion_events[area_id] if t > cutoff
        ]

    def _get_motion_frequency(self, area_id: str, timestamp: datetime,
                               minutes: int = 5) -> int:
        """Count motion events in the last N minutes."""
        cutoff = timestamp - timedelta(minutes=minutes)
        events = self._motion_events.get(area_id, [])
        return sum(1 for t in events if t > cutoff)

    # ── Context from registry ────────────────────────────────────

    def update_context_state(self, entity_id: str, state: str,
                              attributes: dict = None):
        """Track entity states and attributes for room context features."""
        if state in ('unavailable', 'unknown'):
            return
        self._context_states[entity_id] = {
            'state': state,
            'attributes': attributes or {},
        }

    def _get_room_context(self, area_id: str) -> dict:
        """Get comprehensive context features for a room from all area entities.

        Aggregates features by domain: lights (brightness, color_temp),
        sensors (by device_class), climate, media, switches, covers, etc.
        """
        features = {}

        if not self.registry:
            return features

        entities = self.registry.get_entities_in_area(area_id)

        # Collectors per domain
        lights = []
        media_players = []
        sensor_values = defaultdict(list)  # device_class -> [float]
        climate_entities = []
        switches_on = 0
        switches_total = 0
        covers_open = 0
        covers_total = 0
        fans_on = 0
        fans_total = 0
        binary_on = defaultdict(int)   # device_class -> count on
        binary_total = defaultdict(int)  # device_class -> count total

        for eid in entities:
            ctx = self._context_states.get(eid)
            if not ctx:
                continue

            domain = eid.split('.')[0]
            state = ctx['state']
            attrs = ctx.get('attributes', {})

            if domain == 'light':
                is_on = 1 if state == 'on' else 0
                lights.append({
                    'is_on': is_on,
                    'brightness': attrs.get('brightness'),
                    'color_temp': attrs.get('color_temp'),
                    'color_temp_kelvin': attrs.get('color_temp_kelvin'),
                })

            elif domain == 'media_player':
                is_playing = 1 if state == 'playing' else 0
                is_active = 1 if state in ('playing', 'paused', 'on', 'idle') else 0
                media_players.append({
                    'is_active': is_active,
                    'is_playing': is_playing,
                    'volume': attrs.get('volume_level'),
                })

            elif domain == 'sensor':
                try:
                    value = float(state)
                    dc = attrs.get('device_class', 'generic')
                    sensor_values[dc].append(value)
                except (ValueError, TypeError):
                    pass

            elif domain == 'climate':
                climate_entities.append({
                    'current_temp': attrs.get('current_temperature'),
                    'target_temp': attrs.get('temperature'),
                    'humidity': attrs.get('current_humidity'),
                    'hvac_on': 1 if state not in ('off', 'idle', 'unavailable') else 0,
                })

            elif domain == 'switch':
                switches_total += 1
                if state == 'on':
                    switches_on += 1

            elif domain == 'cover':
                covers_total += 1
                if state == 'open':
                    covers_open += 1

            elif domain == 'fan':
                fans_total += 1
                if state == 'on':
                    fans_on += 1

            elif domain == 'binary_sensor':
                dc = attrs.get('device_class', 'generic')
                binary_total[dc] += 1
                if state == 'on':
                    binary_on[dc] += 1

        # ── Aggregate features ──

        # Lights
        if lights:
            features['lights_count'] = len(lights)
            features['lights_on'] = sum(l['is_on'] for l in lights)
            features['lights_on_ratio'] = features['lights_on'] / len(lights)
            bright = [l['brightness'] for l in lights
                      if l['is_on'] and l['brightness'] is not None]
            if bright:
                features['avg_brightness'] = sum(bright) / len(bright) / 255.0
            ct = [l['color_temp_kelvin'] for l in lights
                  if l['is_on'] and l['color_temp_kelvin'] is not None]
            if ct:
                features['avg_color_temp_k'] = sum(ct) / len(ct)

        # Media players
        if media_players:
            features['media_count'] = len(media_players)
            features['media_active'] = sum(m['is_active'] for m in media_players)
            features['media_playing'] = sum(m['is_playing'] for m in media_players)
            vols = [m['volume'] for m in media_players if m['volume'] is not None]
            if vols:
                features['avg_volume'] = sum(vols) / len(vols)

        # Sensors by device_class (temperature, humidity, power, illuminance, etc.)
        for dc, vals in sensor_values.items():
            prefix = f'sensor_{dc}'
            features[f'{prefix}_avg'] = sum(vals) / len(vals)
            if len(vals) > 1:
                features[f'{prefix}_min'] = min(vals)
                features[f'{prefix}_max'] = max(vals)

        # Climate
        if climate_entities:
            features['climate_on'] = sum(c['hvac_on'] for c in climate_entities)
            temps = [c['current_temp'] for c in climate_entities
                     if c['current_temp'] is not None]
            if temps:
                features['climate_temp'] = sum(temps) / len(temps)
            targets = [c['target_temp'] for c in climate_entities
                       if c['target_temp'] is not None]
            if targets:
                features['climate_target'] = sum(targets) / len(targets)
            hums = [c['humidity'] for c in climate_entities
                    if c['humidity'] is not None]
            if hums:
                features['climate_humidity'] = sum(hums) / len(hums)

        # Switches
        if switches_total:
            features['switches_total'] = switches_total
            features['switches_on'] = switches_on

        # Covers
        if covers_total:
            features['covers_total'] = covers_total
            features['covers_open'] = covers_open

        # Fans
        if fans_total:
            features['fans_total'] = fans_total
            features['fans_on'] = fans_on

        # Binary sensors by device_class
        for dc in binary_total:
            prefix = f'bs_{dc}'
            features[f'{prefix}_total'] = binary_total[dc]
            features[f'{prefix}_on'] = binary_on[dc]

        return features
