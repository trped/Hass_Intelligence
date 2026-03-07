"""Batch retraining with scikit-learn GradientBoosting.

Runs nightly (04:00 UTC) on accumulated observations.
Compares batch model accuracy with online River model on last 24h data.
Winner becomes primary, loser becomes fallback.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = '/data/models'
MIN_OBSERVATIONS = 200  # Need at least 200 observations for batch training
TRAINING_DAYS = 14  # Use last 14 days of observations
EVAL_HOURS = 24  # Evaluate on last 24 hours


class BatchTrainer:
    """Nightly batch retraining using GradientBoosting.

    Trains on accumulated observations from the SQLite database,
    compares with the online River model, and stores the winner.
    """

    def __init__(self, db):
        self.db = db
        self.room_models = {}   # area_id -> fitted sklearn model
        self.person_models = {}  # person_id -> fitted sklearn model
        self.last_trained = None
        self.last_results = {}  # area_id/person_id -> comparison results

    def run_nightly_training(self, online_models: dict = None):
        """Run full batch training cycle.

        Args:
            online_models: Dict of ModelManager's room_models/person_models
                          for accuracy comparison.

        Returns:
            Dict with training results per target.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            logger.error("scikit-learn not installed, skipping batch training")
            return {}

        results = {}
        self.last_trained = datetime.now(timezone.utc).isoformat()

        # Train room models
        room_observations = self._get_grouped_observations('room')
        for target_id, obs_list in room_observations.items():
            result = self._train_one(
                target_id, obs_list, 'room',
                online_model=online_models.get('room', {}).get(target_id)
                if online_models else None
            )
            if result:
                results[f'room_{target_id}'] = result

        # Train person models
        person_observations = self._get_grouped_observations('person')
        for target_id, obs_list in person_observations.items():
            result = self._train_one(
                target_id, obs_list, 'person',
                online_model=online_models.get('person', {}).get(target_id)
                if online_models else None
            )
            if result:
                results[f'person_{target_id}'] = result

        self.last_results = results
        logger.info(
            f"Batch training complete: {len(results)} models trained"
        )
        return results

    def _train_one(self, target_id: str, observations: list,
                   model_type: str,
                   online_model=None) -> Optional[dict]:
        """Train a single batch model and compare with online.

        Returns comparison dict or None if insufficient data.
        """
        if len(observations) < MIN_OBSERVATIONS:
            logger.debug(
                f"Batch skip {model_type}/{target_id}: "
                f"only {len(observations)} observations"
            )
            return None

        try:
            from sklearn.ensemble import GradientBoostingClassifier

            # Parse features and labels
            X, y = self._prepare_data(observations)
            if X is None or len(set(y)) < 2:
                return None

            # Split: train on older data, eval on recent 24h
            train_X, train_y, eval_X, eval_y = self._split_train_eval(
                observations, X, y
            )
            if len(train_X) < 50 or len(eval_X) < 10:
                return None

            # Train GradientBoosting
            clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
            clf.fit(train_X, train_y)

            # Evaluate batch model
            batch_accuracy = clf.score(eval_X, eval_y)

            # Evaluate online model (if available)
            online_accuracy = None
            if online_model and hasattr(online_model, 'predict'):
                online_correct = 0
                online_total = 0
                for i, obs in enumerate(observations[-len(eval_X):]):
                    try:
                        features = json.loads(obs.get('features', '{}'))
                        pred_label, _ = online_model.predict(features)
                        if pred_label != 'unknown':
                            online_total += 1
                            if pred_label == eval_y[i]:
                                online_correct += 1
                    except Exception:
                        pass
                if online_total > 0:
                    online_accuracy = online_correct / online_total

            # Store model
            model_key = f'{model_type}_{target_id}'
            if model_type == 'room':
                self.room_models[target_id] = clf
            else:
                self.person_models[target_id] = clf

            # Save to disk
            self._save_model(clf, model_key)

            # Determine winner
            winner = 'batch'
            if online_accuracy is not None and online_accuracy > batch_accuracy:
                winner = 'online'

            result = {
                'batch_accuracy': round(batch_accuracy, 4),
                'online_accuracy': round(online_accuracy, 4) if online_accuracy is not None else None,
                'winner': winner,
                'train_samples': len(train_X),
                'eval_samples': len(eval_X),
                'trained_at': datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                f"Batch {model_type}/{target_id}: "
                f"batch={batch_accuracy:.3f} "
                f"online={online_accuracy:.3f if online_accuracy else 'N/A'} "
                f"winner={winner}"
            )
            return result

        except Exception as e:
            logger.error(f"Batch training error for {model_type}/{target_id}: {e}")
            return None

    def predict(self, model_type: str, target_id: str,
                features: dict) -> Optional[Tuple[str, float]]:
        """Predict using batch model.

        Args:
            model_type: 'room' or 'person'
            target_id: area_id or person_id
            features: Feature dict (will be converted to numpy array)

        Returns:
            Tuple of (label, confidence) or None if no batch model.
        """
        models = self.room_models if model_type == 'room' else self.person_models
        clf = models.get(target_id)
        if clf is None:
            return None

        try:
            # Convert feature dict to array matching trained feature order
            feature_names = clf.feature_names_in_ if hasattr(clf, 'feature_names_in_') else None
            if feature_names is not None:
                X = np.array([[features.get(f, 0.0) for f in feature_names]])
            else:
                X = np.array([list(features.values())])

            label = clf.predict(X)[0]
            proba = clf.predict_proba(X)[0]
            confidence = float(max(proba))
            return (label, round(confidence, 3))

        except Exception as e:
            logger.debug(f"Batch predict error for {model_type}/{target_id}: {e}")
            return None

    def _get_grouped_observations(self, model_type: str) -> dict:
        """Get observations grouped by target_id."""
        hours = TRAINING_DAYS * 24
        obs_list = self.db.get_recent_observations(
            model_type=model_type, hours=hours, limit=10000
        )
        grouped = {}
        key_field = 'area_id' if model_type == 'room' else 'person_id'
        for obs in obs_list:
            tid = obs.get(key_field)
            if tid:
                grouped.setdefault(tid, []).append(obs)
        return grouped

    def _prepare_data(self, observations: list):
        """Convert observations to numpy arrays."""
        try:
            # Collect all feature names from first observation
            first_features = json.loads(observations[0].get('features', '{}'))
            feature_names = sorted(first_features.keys())

            if not feature_names:
                return None, None

            X = []
            y = []
            for obs in observations:
                features = json.loads(obs.get('features', '{}'))
                row = [float(features.get(f, 0.0)) for f in feature_names]
                X.append(row)
                y.append(obs['label'])

            return np.array(X), np.array(y)

        except Exception as e:
            logger.debug(f"Data preparation error: {e}")
            return None, None

    def _split_train_eval(self, observations, X, y):
        """Split into train (older) and eval (last 24h) sets."""
        # observations are ordered DESC, so recent ones are first
        eval_count = min(len(X) // 5, 200)  # max 20% or 200 for eval
        eval_count = max(eval_count, 10)

        if eval_count >= len(X):
            return X, y, X, y

        # Recent = eval, older = train
        eval_X = X[:eval_count]
        eval_y = y[:eval_count]
        train_X = X[eval_count:]
        train_y = y[eval_count:]

        return train_X, train_y, eval_X, eval_y

    def _save_model(self, clf, model_key: str):
        """Save batch model to disk."""
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            path = os.path.join(MODELS_DIR, f'batch_{model_key}.joblib')
            joblib.dump(clf, path)
        except Exception as e:
            logger.error(f"Failed to save batch model {model_key}: {e}")

    def get_stats(self) -> dict:
        """Get batch trainer statistics."""
        return {
            'room_models': len(self.room_models),
            'person_models': len(self.person_models),
            'last_trained': self.last_trained,
            'results': self.last_results,
        }
