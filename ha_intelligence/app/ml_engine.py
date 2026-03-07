"""ML Engine - orchestrates feature extraction, training, and prediction.

Bridges the gap between raw HA state changes and ML model predictions.
Training runs in a ThreadPoolExecutor to avoid blocking the event loop.

Phase 4 additions:
- Markov Chain movement prediction (person room transitions)
- Anomaly detection (HalfSpaceTrees on room features)
- Batch model fallback (scikit-learn GradientBoosting)
- Netatmo camera face detection as evidence source
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from features import FeatureExtractor
from models.model_manager import ModelManager
from priors import PriorCalculator

logger = logging.getLogger(__name__)

# Minimum observations before ML predictions are used
ML_THRESHOLD = 50

# Weight for combining ML prediction with prior probability
# final = ML_WEIGHT * ml_confidence + PRIOR_WEIGHT * prior_probability
ML_WEIGHT = 0.7
PRIOR_WEIGHT = 0.3


class MLEngine:
    """Orchestrates ML training and prediction for HA Intelligence."""

    def __init__(self, db, registry=None):
        self.db = db
        self.registry = registry
        self.features = FeatureExtractor(registry=registry)
        self.models = ModelManager(db=db)
        self.priors = PriorCalculator(db=db)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._person_rooms = {}  # person_entity -> room dict (synced from SensorEngine)
        self._room_states = {}   # area_id -> room_state dict (synced from SensorEngine)
        self._netatmo_rooms = {} # person_entity -> {'room': str, 'timestamp': str}
        self._previous_person_rooms = {}  # person_entity -> previous room for Markov

    def update_person_room(self, person_entity: str, room_data: dict):
        """Update BLE room data for a person (called from SensorEngine)."""
        old_room = None
        if person_entity in self._person_rooms:
            old_room = self._person_rooms[person_entity].get('room')

        self._person_rooms[person_entity] = room_data
        new_room = room_data.get('room')

        # Record Markov transition if room changed
        if old_room and new_room and old_room != new_room:
            self._record_markov_transition(person_entity, old_room, new_room)

    def update_netatmo_room(self, person_entity: str, room: str,
                            timestamp: str):
        """Update Netatmo camera face detection room for a person.

        Netatmo camera is the highest-confidence source (0.95) for
        person-room location — camera is "always right".

        Args:
            person_entity: e.g. 'person.troels'
            room: Room name from camera (e.g. 'alrum')
            timestamp: ISO timestamp of detection
        """
        old_room = None
        if person_entity in self._netatmo_rooms:
            old_room = self._netatmo_rooms[person_entity].get('room')

        self._netatmo_rooms[person_entity] = {
            'room': room,
            'timestamp': timestamp,
        }

        # Record Markov transition from Netatmo as well
        if old_room and room and old_room != room:
            self._record_markov_transition(person_entity, old_room, room)

    def _record_markov_transition(self, person_entity: str,
                                   from_room: str, to_room: str):
        """Record a room transition for Markov prediction."""
        try:
            hour = datetime.now(timezone.utc).hour
            markov = self.models.get_or_create_markov_model(person_entity)
            markov.record_transition(from_room, to_room, hour)
        except Exception as e:
            logger.debug(f"Markov transition error: {e}")

    def update_room_state(self, area_id: str, room_state: dict):
        """Sync room state from SensorEngine for rich feature extraction."""
        self._room_states[area_id] = room_state

    async def on_state_change(self, entity_id: str, old_state: str,
                               new_state: str, attributes: dict):
        """Process a state change for ML training (non-blocking).

        Called from SensorEngine.on_state_change(). Runs training
        in a thread pool to avoid blocking the event loop.
        """
        domain = entity_id.split('.')[0]

        # Track ALL entity states + attributes for room context features
        self.features.update_context_state(entity_id, new_state, attributes)

        # Motion events -> train room model + anomaly detection
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

        # Person state changes -> train person model
        if domain == 'person':
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                self._executor,
                self._train_person, entity_id, new_state, attributes
            )

    def _train_room(self, area_id: str, sensor_state: str):
        """Train room occupancy model + anomaly detection (runs in thread).

        Uses the full room_state synced from SensorEngine (if available)
        so features include actual sensor counts, EPL zones, lights, media etc.
        Falls back to minimal state if sync hasn't happened yet.
        """
        try:
            # Determine label from sensor state
            label = 'occupied' if sensor_state == 'on' else 'empty'

            # Phase 2: Use actual room_state from SensorEngine for rich features
            room_state = self._room_states.get(area_id)
            if not room_state:
                # Fallback: minimal room_state (pre-sync or unknown room)
                room_state = {
                    'sensors': {f'_train_{area_id}': sensor_state},
                    'last_occupied': datetime.now(timezone.utc).isoformat() if sensor_state == 'on' else None,
                }

            features = self.features.extract_room_features(area_id, room_state)
            model = self.models.get_or_create_room_model(area_id)
            model.learn(features, label, age_minutes=0.0)

            # Phase 4: Anomaly detection on room features
            anomaly_model = self.models.get_or_create_anomaly_model(area_id)
            anomaly_model.learn_and_score(features)

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
        """Predict room occupancy using ML model + prior weighting.

        Combines ML prediction confidence with historical state priors:
          final_confidence = ML_WEIGHT * ml_confidence + PRIOR_WEIGHT * prior_probability

        Falls back to batch model if online model not available.

        Returns:
            Tuple of (state, confidence, source_tag) or None if ML
            doesn't have enough data yet.
        """
        model = self.models.room_models.get(area_id)
        if not model or model.samples_seen < ML_THRESHOLD:
            # Phase 4: Try batch model fallback
            return self._predict_room_batch(area_id, room_state)

        try:
            features = self.features.extract_room_features(area_id, room_state)
            label, ml_confidence = model.predict(features)
            if label == 'unknown':
                return None

            # Phase 3: Combine with prior probability
            prior_info = self.priors.get_prior('room', area_id)
            if prior_info['has_data']:
                prior_prob = prior_info['priors'].get(label, 0.0)
                confidence = round(
                    ML_WEIGHT * ml_confidence + PRIOR_WEIGHT * prior_prob, 4
                )
                source = 'ml_river+prior'
            else:
                confidence = ml_confidence
                source = 'ml_river'

            return (label, confidence, source)
        except Exception as e:
            logger.debug(f"Room predict error for {area_id}: {e}")
            return None

    def _predict_room_batch(self, area_id: str,
                            room_state: dict) -> Optional[Tuple[str, float, str]]:
        """Fallback prediction using batch-trained model."""
        if not self.models.batch_trainer:
            return None
        try:
            features = self.features.extract_room_features(area_id, room_state)
            result = self.models.batch_trainer.predict('room', area_id, features)
            if result:
                label, confidence = result
                return (label, confidence, 'ml_batch')
        except Exception as e:
            logger.debug(f"Batch room predict error for {area_id}: {e}")
        return None

    def predict_person(self, person_entity: str,
                       person_state: dict,
                       rooms_with_motion: int = 0) -> Optional[Tuple[str, float, str]]:
        """Predict person activity using ML model.

        Falls back to batch model if online model not available.

        Returns:
            Tuple of (state, confidence, source_tag) or None if ML
            doesn't have enough data yet.
        """
        model = self.models.person_models.get(person_entity)
        if not model or model.samples_seen < ML_THRESHOLD:
            # Phase 4: Try batch model fallback
            return self._predict_person_batch(person_entity, person_state)

        try:
            # Include BLE room data in feature extraction
            person_room = self._person_rooms.get(person_entity)

            features = self.features.extract_person_features(
                person_entity, person_state, rooms_with_motion,
                person_room=person_room
            )
            label, ml_confidence = model.predict(features)
            if label == 'unknown':
                return None

            # Phase 3: Combine with prior probability
            slug = person_entity.replace('person.', '')
            prior_info = self.priors.get_prior('person', slug)
            if prior_info['has_data']:
                prior_prob = prior_info['priors'].get(label, 0.0)
                confidence = round(
                    ML_WEIGHT * ml_confidence + PRIOR_WEIGHT * prior_prob, 4
                )
                source = 'ml_river+prior'
            else:
                confidence = ml_confidence
                source = 'ml_river'

            return (label, confidence, source)
        except Exception as e:
            logger.debug(f"Person predict error for {person_entity}: {e}")
            return None

    def _predict_person_batch(self, person_entity: str,
                              person_state: dict) -> Optional[Tuple[str, float, str]]:
        """Fallback prediction using batch-trained model."""
        if not self.models.batch_trainer:
            return None
        try:
            person_room = self._person_rooms.get(person_entity)
            features = self.features.extract_person_features(
                person_entity, person_state, person_room=person_room
            )
            slug = person_entity.replace('person.', '')
            result = self.models.batch_trainer.predict('person', slug, features)
            if result:
                label, confidence = result
                return (label, confidence, 'ml_batch')
        except Exception as e:
            logger.debug(f"Batch person predict error for {person_entity}: {e}")
        return None

    def predict_next_room(self, person_entity: str,
                          current_room: str) -> Optional[List[Tuple[str, float]]]:
        """Predict next room for a person using Markov chain.

        Args:
            person_entity: e.g. 'person.troels'
            current_room: Current room area_id

        Returns:
            List of (room, probability) sorted by probability desc,
            or None if not enough data.
        """
        markov = self.models.markov_models.get(person_entity)
        if not markov:
            return None

        hour = datetime.now(timezone.utc).hour
        predictions = markov.predict_next(current_room, hour)
        return predictions if predictions else None

    def get_anomaly_score(self, area_id: str) -> Optional[dict]:
        """Get latest anomaly info for a room.

        Returns:
            Dict with 'score', 'ready', 'latest_anomaly' or None.
        """
        anomaly_model = self.models.anomaly_models.get(area_id)
        if not anomaly_model:
            return None
        return anomaly_model.get_stats()

    def get_room_evidence(self, area_id: str, room_state: dict) -> dict:
        """Analyze active evidence sources for a room.

        Returns dict with sources, count, detail for publishing
        as room sensor attributes.
        """
        try:
            features = self.features.extract_room_features(area_id, room_state)
            return self.features.analyze_evidence(features)
        except Exception as e:
            logger.debug(f"Evidence analysis error for {area_id}: {e}")
            return {'sources': [], 'count': 0, 'detail': {}}

    def get_person_best_room(self, person_entity: str) -> Optional[dict]:
        """Get the best room estimate for a person from all sources.

        Merges BLE (Bermuda), Netatmo camera, and motion inference.
        Returns dict with room, confidence, source.
        """
        best = None

        # Netatmo camera: highest confidence (0.95)
        netatmo = self._netatmo_rooms.get(person_entity)
        if netatmo and netatmo.get('room'):
            # Check age — only trust within 10 minutes
            try:
                ts = datetime.fromisoformat(
                    netatmo['timestamp'].replace('Z', '+00:00')
                )
                age_minutes = (
                    datetime.now(timezone.utc) - ts
                ).total_seconds() / 60
                if age_minutes < 10:
                    best = {
                        'room': netatmo['room'],
                        'confidence': 0.95,
                        'source': 'netatmo_camera',
                    }
            except Exception:
                pass

        # BLE/Bermuda: confidence 0.9 (lower than fresh Netatmo)
        ble_room = self._person_rooms.get(person_entity)
        if ble_room and ble_room.get('room'):
            ble_conf = ble_room.get('confidence', 0.9)
            if not best or ble_conf > best['confidence']:
                best = {
                    'room': ble_room['room'],
                    'confidence': ble_conf,
                    'source': 'bermuda_ble',
                }

        return best

    def get_stats(self) -> dict:
        """Get ML engine stats."""
        model_stats = self.models.get_stats()
        model_stats['ml_active'] = model_stats['total_samples'] >= ML_THRESHOLD
        model_stats['ml_threshold'] = ML_THRESHOLD
        return model_stats
