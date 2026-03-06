"""Model lifecycle management for HA Intelligence.

Manages creation, retrieval, saving and stats for all ML models.
"""

import asyncio
import logging

from models.room_model import RoomOccupancyModel
from models.person_model import PersonActivityModel

logger = logging.getLogger(__name__)

SAVE_INTERVAL = 300  # 5 minutes


class ModelManager:
    """Manages all ML model instances."""

    def __init__(self, db=None):
        self.db = db
        self.room_models = {}    # area_id -> RoomOccupancyModel
        self.person_models = {}  # person_id -> PersonActivityModel

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

    def get_stats(self) -> dict:
        """Get ML stats for system sensor."""
        total_room_samples = sum(m.samples_seen for m in self.room_models.values())
        total_person_samples = sum(m.samples_seen for m in self.person_models.values())

        return {
            'room_models': len(self.room_models),
            'person_models': len(self.person_models),
            'total_room_samples': total_room_samples,
            'total_person_samples': total_person_samples,
            'total_samples': total_room_samples + total_person_samples,
        }
