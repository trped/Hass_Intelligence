"""Model lifecycle management for HA Intelligence.

Manages creation, retrieval, saving and stats for all ML models:
- Room occupancy (River GaussianNB)
- Person activity (River GaussianNB)
- Movement prediction (Markov Chain)
- Anomaly detection (River HalfSpaceTrees)
- Batch retraining (scikit-learn GradientBoosting)
"""

import asyncio
import logging
from datetime import datetime, timezone

from models.room_model import RoomOccupancyModel
from models.person_model import PersonActivityModel
from models.markov_chain import MovementPredictor
from models.anomaly_model import AnomalyDetector
from models.batch_trainer import BatchTrainer

logger = logging.getLogger(__name__)

SAVE_INTERVAL = 300  # 5 minutes
BATCH_HOUR_UTC = 4   # Run batch training at 04:00 UTC


class ModelManager:
    """Manages all ML model instances."""

    def __init__(self, db=None):
        self.db = db
        self.room_models = {}      # area_id -> RoomOccupancyModel
        self.person_models = {}    # person_id -> PersonActivityModel
        self.markov_models = {}    # person_id -> MovementPredictor
        self.anomaly_models = {}   # area_id -> AnomalyDetector
        self.batch_trainer = BatchTrainer(db) if db else None

    def get_or_create_room_model(self, area_id: str) -> RoomOccupancyModel:
        """Get existing or create new room model."""
        if area_id not in self.room_models:
            self.room_models[area_id] = RoomOccupancyModel(area_id)
        return self.room_models[area_id]

    def get_or_create_person_model(self, person_id: str) -> PersonActivityModel:
        """Get existing or create new person model."""
        if person_id not in self.person_models:
            self.person_models[person_id] = PersonActivityModel(person_id)
        return self.person_models[person_id]

    def get_or_create_markov_model(self, person_id: str) -> MovementPredictor:
        """Get existing or create new Markov movement model."""
        if person_id not in self.markov_models:
            self.markov_models[person_id] = MovementPredictor(person_id)
        return self.markov_models[person_id]

    def get_or_create_anomaly_model(self, area_id: str) -> AnomalyDetector:
        """Get existing or create new anomaly detector."""
        if area_id not in self.anomaly_models:
            self.anomaly_models[area_id] = AnomalyDetector(area_id)
        return self.anomaly_models[area_id]

    def save_all(self):
        """Persist all models to disk and update DB stats."""
        saved = 0
        for area_id, model in self.room_models.items():
            model.save()
            if self.db:
                self.db.upsert_model_version(
                    f'room_{area_id}', samples_seen=model.samples_seen
                )
            saved += 1

        for person_id, model in self.person_models.items():
            model.save()
            if self.db:
                slug = person_id.replace('person.', '')
                self.db.upsert_model_version(
                    f'person_{slug}', samples_seen=model.samples_seen
                )
            saved += 1

        for person_id, model in self.markov_models.items():
            model.save()
            if self.db:
                slug = person_id.replace('person.', '')
                self.db.upsert_model_version(
                    f'markov_{slug}',
                    samples_seen=model.total_transitions
                )
            saved += 1

        for area_id, model in self.anomaly_models.items():
            model.save()
            if self.db:
                self.db.upsert_model_version(
                    f'anomaly_{area_id}',
                    samples_seen=model.samples_seen
                )
            saved += 1

        if saved:
            logger.info(f"Saved {saved} models to disk")

    async def periodic_save(self):
        """Async task: save all models every 5 minutes."""
        while True:
            await asyncio.sleep(SAVE_INTERVAL)
            try:
                self.save_all()
            except Exception as e:
                logger.error(f"Model save error: {e}")

    async def nightly_batch_training(self):
        """Async task: run batch retraining at 04:00 UTC each night."""
        while True:
            try:
                now = datetime.now(timezone.utc)
                # Calculate seconds until next BATCH_HOUR_UTC
                target_hour = BATCH_HOUR_UTC
                if now.hour >= target_hour:
                    # Next day
                    hours_until = 24 - now.hour + target_hour
                else:
                    hours_until = target_hour - now.hour
                seconds_until = (
                    hours_until * 3600
                    - now.minute * 60
                    - now.second
                )
                logger.info(
                    f"Batch training scheduled in {hours_until}h "
                    f"({seconds_until}s)"
                )
                await asyncio.sleep(seconds_until)

                # Run batch training
                if self.batch_trainer:
                    logger.info("Starting nightly batch training...")
                    online_models = {
                        'room': self.room_models,
                        'person': self.person_models,
                    }
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None,
                        self.batch_trainer.run_nightly_training,
                        online_models,
                    )
                    logger.info(
                        f"Batch training complete: "
                        f"{len(results)} models trained"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch training error: {e}")
                # Wait 1 hour before retry on error
                await asyncio.sleep(3600)

    def get_stats(self) -> dict:
        """Get ML stats for system sensor."""
        total_room_samples = sum(
            m.samples_seen for m in self.room_models.values()
        )
        total_person_samples = sum(
            m.samples_seen for m in self.person_models.values()
        )
        total_transitions = sum(
            m.total_transitions for m in self.markov_models.values()
        )
        total_anomaly_samples = sum(
            m.samples_seen for m in self.anomaly_models.values()
        )
        total_anomalies = sum(
            m.anomalies_detected for m in self.anomaly_models.values()
        )

        stats = {
            'room_models': len(self.room_models),
            'person_models': len(self.person_models),
            'markov_models': len(self.markov_models),
            'anomaly_models': len(self.anomaly_models),
            'total_room_samples': total_room_samples,
            'total_person_samples': total_person_samples,
            'total_transitions': total_transitions,
            'total_anomaly_samples': total_anomaly_samples,
            'total_anomalies': total_anomalies,
            'total_samples': (
                total_room_samples + total_person_samples
                + total_transitions + total_anomaly_samples
            ),
        }

        # Batch trainer stats
        if self.batch_trainer:
            stats['batch'] = self.batch_trainer.get_stats()

        return stats
