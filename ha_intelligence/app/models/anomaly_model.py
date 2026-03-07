"""Anomaly detection using River's HalfSpaceTrees.

Online Isolation Forest variant that scores room feature vectors for
unusual activity patterns (e.g. motion at 3 AM, unexpected sensor combos).
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import joblib
from river import anomaly

logger = logging.getLogger(__name__)

MODELS_DIR = '/data/models'
MIN_SAMPLES = 100  # Train on at least 100 normal samples before scoring
ANOMALY_THRESHOLD = 0.7  # Score > 0.7 = anomaly
WINDOW_SIZE = 250  # HalfSpaceTrees window size


class AnomalyDetector:
    """Online anomaly detection for room activity patterns.

    Uses River's HalfSpaceTrees — an online adaptation of Isolation Forest.
    Scores each observation from 0.0 (normal) to 1.0 (anomalous).
    """

    def __init__(self, area_id: str):
        self.area_id = area_id
        self.model = anomaly.HalfSpaceTrees(
            n_trees=10,
            height=8,
            window_size=WINDOW_SIZE,
            seed=42,
        )
        self.samples_seen = 0
        self.anomalies_detected = 0
        self.latest_anomaly = None  # dict with score, timestamp, features
        self._path = os.path.join(MODELS_DIR, f'anomaly_{area_id}.joblib')
        self._load()

    def learn_and_score(self, features: dict) -> Optional[float]:
        """Train on features and return anomaly score.

        Always trains (learn_one), then scores if enough samples seen.

        Args:
            features: Feature dict from FeatureExtractor

        Returns:
            Anomaly score (0.0-1.0) or None if not enough samples.
        """
        try:
            # Always learn
            self.model.learn_one(features)
            self.samples_seen += 1

            # Only score after enough training
            if self.samples_seen < MIN_SAMPLES:
                return None

            score = self.model.score_one(features)
            # River HalfSpaceTrees.score_one returns 0-1 where
            # higher = more anomalous
            score = round(score, 4)

            if score > ANOMALY_THRESHOLD:
                self.anomalies_detected += 1
                self.latest_anomaly = {
                    'score': score,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'area_id': self.area_id,
                }
                logger.info(
                    f"Anomaly detected in {self.area_id}: "
                    f"score={score:.3f}"
                )

            return score

        except Exception as e:
            logger.debug(f"Anomaly model error for {self.area_id}: {e}")
            return None

    def get_stats(self) -> dict:
        """Get anomaly model statistics."""
        return {
            'samples_seen': self.samples_seen,
            'anomalies_detected': self.anomalies_detected,
            'latest_anomaly': self.latest_anomaly,
            'ready': self.samples_seen >= MIN_SAMPLES,
            'threshold': ANOMALY_THRESHOLD,
        }

    def save(self):
        """Persist model to disk."""
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            data = {
                'model': self.model,
                'samples_seen': self.samples_seen,
                'anomalies_detected': self.anomalies_detected,
                'latest_anomaly': self.latest_anomaly,
                'area_id': self.area_id,
            }
            joblib.dump(data, self._path)
        except Exception as e:
            logger.error(f"Failed to save anomaly model {self.area_id}: {e}")

    def _load(self):
        """Load model from disk if available."""
        if not os.path.exists(self._path):
            return
        try:
            data = joblib.load(self._path)
            self.model = data['model']
            self.samples_seen = data.get('samples_seen', 0)
            self.anomalies_detected = data.get('anomalies_detected', 0)
            self.latest_anomaly = data.get('latest_anomaly')
            logger.debug(
                f"Loaded anomaly model {self.area_id} "
                f"({self.samples_seen} samples)"
            )
        except Exception as e:
            logger.warning(f"Failed to load anomaly model {self.area_id}: {e}")
            self.model = anomaly.HalfSpaceTrees(
                n_trees=10, height=8,
                window_size=WINDOW_SIZE, seed=42,
            )
            self.samples_seen = 0
            self.anomalies_detected = 0
            self.latest_anomaly = None
