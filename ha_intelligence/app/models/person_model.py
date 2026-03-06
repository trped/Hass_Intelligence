"""Person activity model using River online ML.

Uses GaussianNB for activity classification (active/idle/sleeping/away)
with slower memory decay since activity patterns change more gradually.
"""

import logging
import math
import os
from typing import Tuple

import joblib
from river import naive_bayes

logger = logging.getLogger(__name__)

MODELS_DIR = '/data/models'
MIN_SAMPLES = 10
DECAY_LAMBDA = 0.005  # Slow decay for activity patterns


class PersonActivityModel:
    """Online learning model for person activity prediction."""

    def __init__(self, person_id: str):
        self.person_id = person_id
        self.model = naive_bayes.GaussianNB()
        self.samples_seen = 0
        # Sanitize person_id for filename (person.troels -> troels)
        slug = person_id.replace('person.', '').replace('.', '_')
        self._path = os.path.join(MODELS_DIR, f'person_{slug}.joblib')

        self._load()

    def learn(self, features: dict, label: str, age_minutes: float = 0.0):
        """Train model with one observation.

        Args:
            features: Feature dict from FeatureExtractor
            label: 'active', 'idle', 'sleeping', or 'away'
            age_minutes: How old this observation is (for decay weighting)
        """
        weight = math.exp(-DECAY_LAMBDA * age_minutes)
        if weight < 0.01:
            return

        try:
            self.model.learn_one(features, label)
            self.samples_seen += 1
        except Exception as e:
            logger.debug(f"Person model learn error for {self.person_id}: {e}")

    def predict(self, features: dict) -> Tuple[str, float]:
        """Predict person activity.

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
            logger.debug(f"Person model predict error for {self.person_id}: {e}")
            return ('unknown', 0.0)

    def save(self):
        """Persist model to disk."""
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            data = {
                'model': self.model,
                'samples_seen': self.samples_seen,
                'person_id': self.person_id,
            }
            joblib.dump(data, self._path)
        except Exception as e:
            logger.error(f"Failed to save person model {self.person_id}: {e}")

    def _load(self):
        """Load model from disk if available."""
        if not os.path.exists(self._path):
            return
        try:
            data = joblib.load(self._path)
            self.model = data['model']
            self.samples_seen = data.get('samples_seen', 0)
            logger.debug(f"Loaded person model {self.person_id} ({self.samples_seen} samples)")
        except Exception as e:
            logger.warning(f"Failed to load person model {self.person_id}: {e}")
            self.model = naive_bayes.GaussianNB()
            self.samples_seen = 0
