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
                                 timestamp: Optional[datetime] = None) -> dict:
        """Extract features for person activity prediction.

        Args:
            person_entity: The person entity_id
            person_state: Dict with 'ha_state', 'source' from SensorEngine
            rooms_with_motion: Number of rooms with active motion sensors
            timestamp: Optional timestamp (defaults to now)
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

    def update_context_state(self, entity_id: str, state: str):
        """Track light/media states for room context features."""
        domain = entity_id.split('.')[0] if '.' in entity_id else ''
        if domain in ('light', 'media_player'):
            self._context_states[entity_id] = state

    def _get_room_context(self, area_id: str) -> dict:
        """Get context features for a room from tracked states."""
        has_light_on = 0
        has_media_playing = 0

        if self.registry:
            entities = self.registry.get_entities_in_area(area_id)
            for eid in entities:
                state = self._context_states.get(eid)
                if not state:
                    continue
                domain = eid.split('.')[0]
                if domain == 'light' and state == 'on':
                    has_light_on = 1
                elif domain == 'media_player' and state in ('playing', 'on'):
                    has_media_playing = 1

        return {
            'has_light_on': has_light_on,
            'has_media_playing': has_media_playing,
        }
