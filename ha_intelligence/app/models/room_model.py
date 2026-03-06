"""Room occupancy model using River online ML.

Uses GaussianNB for occupied/empty classification with memory decay
so recent observations weigh more than old ones.
"""

import logging
import math
import os
from typing import Optional, Tuple

import joblib
from river import naive_bayes

logger = logging.getLogger(__name__)

MODELS_DIR = '/data/models'
MIN_SAMPLES = 10  # Minimum samples before predictions are used
DECAY_LAMBDA = 0.1  # Fast decay for presence (e^(-0.1 * age_minutes))


class RoomOccupancyModel:
    """Online learning model for room occupancy prediction."""

    def __init__(self, area_id: str):
        self.area_id = area_id
        self.model = naive_bayes.GaussianNB()
        self.samples_seen = 0
        self._path = os.path.join(MODELS_DIR, f'room_{area_id}.joblib')

        # Try to load existing model
        self._load()

    def learn(self, features: dict, label: str, age_minutes: float = 0.0):
        """Train model with one observation.

        Args:
            features: Feature dict from FeatureExtractor
            label: 'occupied' or 'empty'
            age_minutes: How old this observation is (for decay weighting)
        """
        weight = math.exp(-DECAY_LAMBDA * age_minutes)
        # River's learn_one doesn't support sample_weight on GaussianNB,
        # but we can approximate by skipping very old/low-weight samples
        if weight < 0.01:
            return

        try:
            self.model.learn_one(features, label)
            self.samples_seen += 1
        except Exception as e:
            logger.debug(f"Room model learn error for {self.area_id}: {e}")

    def predict(self, features: dict) -> Tuple[str, float]:
        """Predict room occupancy.

        Returns:
            Tuple of (label, confidence). Returns ('unknown', 0.0) if
            not enough samples have been seen.
        """
        if self.samples_seen < MIN_SAMPLES:
            return ('unknown', 0.0)

        try:
            proba = self.model.predict_proba_one(features)
            if not proba:
                return ('unknown', 0.0)

            best_label = max(proba, key=proba.get)
            confidence = proba[best_label]
            return (best_label, round(confidence, 3))
        except Exception as e:
            logger.debug(f"Room model predict error for {self.area_id}: {e}")
            return ('unknown', 0.0)

    def save(self):
        """Persist model to disk."""
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            data = {
                'model': self.model,
                'samples_seen': self.samples_seen,
                'area_id': self.area_id,
            }
            joblib.dump(data, self._path)
        except Exception as e:
            logger.error(f"Failed to save room model {self.area_id}: {e}")

    def _load(self):
        """Load model from disk if available."""
        if not os.path.exists(self._path):
            return
        try:
            data = joblib.load(self._path)
            self.model = data['model']
            self.samples_seen = data.get('samples_seen', 0)
            logger.debug(f"Loaded room model {self.area_id} ({self.samples_seen} samples)")
        except Exception as e:
            logger.warning(f"Failed to load room model {self.area_id}: {e}")
            self.model = naive_bayes.GaussianNB()
            self.samples_seen = 0
