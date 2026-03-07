"""ML Engine - orchestrates feature extraction, training, and prediction.

Bridges the gap between raw HA state changes and ML model predictions.
Training runs in a ThreadPoolExecutor to avoid blocking the event loop.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional, Tuple

from features import FeatureExtractor
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)

# Minimum observations before ML predictions are used
ML_THRESHOLD = 50


class MLEngine:
    """Orchestrates ML training and prediction for HA Intelligence."""

    def __init__(self, db, registry=None):
        self.db = db
        self.registry = registry
        self.features = FeatureExtractor(registry=registry)
        self.models = ModelManager(db=db)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._person_rooms = {}  # person_entity -> room dict (synced from SensorEngine)

    def update_person_room(self, person_entity: str, room_data: dict):
        """Update BLE room data for a person (called from SensorEngine)."""
        self._person_rooms[person_entity] = room_data

    async def on_state_change(self, entity_id: str, old_state: str,
                               new_state: str, attributes: dict):
        """Process a state change for ML training (non-blocking).

        Called from SensorEngine.on_state_change(). Runs training
        in a thread pool to avoid blocking the event loop.
        """
        domain = entity_id.split('.')[0]

        # Track ALL entity states + attributes for room context features
        self.features.update_context_state(entity_id, new_state, attributes)

        # Motion events → train room model
        if domain == 'binary_sensor' and any(
            kw in entity_id for kw in ('motion', 'occupancy', 'presence', 'mmwave')
        ):
            area_id = None
            if self.registry:
                area_id = self.registry.get_area_id(entity_id)
            if area_id:
                self.features.update_motion_tracking(area_id)
                # Train in background
                loop = asyncio.get_event_loop()
                loop.run_in_executor(
                    self._executor,
                    self._train_room, area_id, new_state
                )

        # Person state changes → train person model
        if domain == 'person':
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                self._executor,
                self._train_person, entity_id, new_state, attributes
            )

    def _train_room(self, area_id: str, sensor_state: str):
        """Train room occupancy model (runs in thread)."""
        try:
            from main import SensorEngine  # avoid circular at module level
        except ImportError:
            pass

        try:
            # Determine label from sensor state
            label = 'occupied' if sensor_state == 'on' else 'empty'

            # Build room_state dict for feature extraction
            room_state = {
                'sensors': {f'_train_{area_id}': sensor_state},
                'last_occupied': datetime.now(timezone.utc).isoformat() if sensor_state == 'on' else None,
            }

            features = self.features.extract_room_features(area_id, room_state)
            model = self.models.get_or_create_room_model(area_id)
            model.learn(features, label, age_minutes=0.0)

            # Store observation in DB
            self.db.insert_observation(
                area_id=area_id,
                features=json.dumps(features),
                label=label,
                model_type='room',
            )
        except Exception as e:
            logger.debug(f"Room training error for {area_id}: {e}")

    def _train_person(self, person_entity: str, state: str, attributes: dict):
        """Train person activity model (runs in thread)."""
        try:
            # Infer label from HA state
            if state == 'home':
                label = 'active'
            elif state == 'not_home':
                label = 'away'
            else:
                label = state  # zone name, etc.

            person_state = {
                'ha_state': state,
                'source': attributes.get('source'),
            }

            # Include BLE room data in feature extraction
            person_room = self._person_rooms.get(person_entity)

            features = self.features.extract_person_features(
                person_entity, person_state, person_room=person_room
            )
            model = self.models.get_or_create_person_model(person_entity)
            model.learn(features, label, age_minutes=0.0)

            # Store observation
            slug = person_entity.replace('person.', '')
            self.db.insert_observation(
                person_id=slug,
                features=json.dumps(features),
                label=label,
                model_type='person',
            )
        except Exception as e:
            logger.debug(f"Person training error for {person_entity}: {e}")

    def predict_room(self, area_id: str,
                     room_state: dict) -> Optional[Tuple[str, float, str]]:
        """Predict room occupancy using ML model.

        Returns:
            Tuple of (state, confidence, 'ml_river') or None if ML
            doesn't have enough data yet.
        """
        model = self.models.room_models.get(area_id)
        if not model or model.samples_seen < ML_THRESHOLD:
            return None

        try:
            features = self.features.extract_room_features(area_id, room_state)
            label, confidence = model.predict(features)
            if label == 'unknown':
                return None
            return (label, confidence, 'ml_river')
        except Exception as e:
            logger.debug(f"Room predict error for {area_id}: {e}")
            return None

    def predict_person(self, person_entity: str,
                       person_state: dict,
                       rooms_with_motion: int = 0) -> Optional[Tuple[str, float, str]]:
        """Predict person activity using ML model.

        Returns:
            Tuple of (state, confidence, 'ml_river') or None if ML
            doesn't have enough data yet.
        """
        model = self.models.person_models.get(person_entity)
        if not model or model.samples_seen < ML_THRESHOLD:
            return None

        try:
            # Include BLE room data in feature extraction
            person_room = self._person_rooms.get(person_entity)

            features = self.features.extract_person_features(
                person_entity, person_state, rooms_with_motion,
                person_room=person_room
            )
            label, confidence = model.predict(features)
            if label == 'unknown':
                return None
            return (label, confidence, 'ml_river')
        except Exception as e:
            logger.debug(f"Person predict error for {person_entity}: {e}")
            return None

    def get_stats(self) -> dict:
        """Get ML engine stats."""
        model_stats = self.models.get_stats()
        model_stats['ml_active'] = model_stats['total_samples'] >= ML_THRESHOLD
        model_stats['ml_threshold'] = ML_THRESHOLD
        return model_stats
