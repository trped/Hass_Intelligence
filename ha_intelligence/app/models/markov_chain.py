"""Markov Chain movement prediction for person-room transitions.

Predicts next room based on current room and time bucket using
transition probabilities learned from observed room changes.
"""

import json
import logging
import os
from collections import defaultdict
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

MODELS_DIR = '/data/models'
MIN_TRANSITIONS = 10  # Minimum before predictions are used


def _time_bucket(hour: int) -> str:
    """Map hour to time bucket.

    morgen: 06-10, dag: 10-17, aften: 17-22, nat: 22-06
    """
    if 6 <= hour < 10:
        return 'morgen'
    elif 10 <= hour < 17:
        return 'dag'
    elif 17 <= hour < 22:
        return 'aften'
    else:
        return 'nat'


class MovementPredictor:
    """Predicts next room from transition probabilities.

    Maintains a transition matrix per time bucket:
        P(next_room | current_room, time_bucket)

    Updated online at each observed room change.
    Persisted as JSON for easy inspection.
    """

    def __init__(self, person_id: str):
        self.person_id = person_id
        self.total_transitions = 0

        # transitions[time_bucket][from_room][to_room] = count
        self._transitions = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        slug = person_id.replace('person.', '').replace('.', '_')
        self._path = os.path.join(MODELS_DIR, f'markov_{slug}.json')
        self._load()

    def record_transition(self, from_room: str, to_room: str, hour: int):
        """Record a room-to-room transition.

        Args:
            from_room: Source room area_id (e.g. 'koekken')
            to_room: Destination room area_id (e.g. 'alrum')
            hour: Hour of day (0-23) for time bucket assignment
        """
        if from_room == to_room:
            return  # Self-transitions don't count

        bucket = _time_bucket(hour)
        self._transitions[bucket][from_room][to_room] += 1
        self.total_transitions += 1

    def predict_next(self, current_room: str,
                     hour: int) -> List[Tuple[str, float]]:
        """Predict next rooms with probabilities.

        Args:
            current_room: Current room area_id
            hour: Current hour (0-23)

        Returns:
            List of (room, probability) sorted by probability desc.
            Empty list if not enough data.
        """
        if self.total_transitions < MIN_TRANSITIONS:
            return []

        bucket = _time_bucket(hour)
        row = self._transitions.get(bucket, {}).get(current_room, {})
        if not row:
            # Fallback: try all time buckets merged
            row = self._merged_row(current_room)

        if not row:
            return []

        total = sum(row.values())
        if total == 0:
            return []

        predictions = [
            (room, round(count / total, 3))
            for room, count in row.items()
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:5]  # Top 5

    def get_top_prediction(self, current_room: str,
                           hour: int) -> Optional[Tuple[str, float]]:
        """Get single best next-room prediction.

        Returns:
            Tuple of (next_room, probability) or None.
        """
        preds = self.predict_next(current_room, hour)
        return preds[0] if preds else None

    def _merged_row(self, current_room: str) -> dict:
        """Merge transition counts across all time buckets for a room."""
        merged = defaultdict(int)
        for bucket_data in self._transitions.values():
            row = bucket_data.get(current_room, {})
            for to_room, count in row.items():
                merged[to_room] += count
        return dict(merged)

    def get_stats(self) -> dict:
        """Get Markov model statistics."""
        unique_rooms = set()
        for bucket_data in self._transitions.values():
            for from_room, targets in bucket_data.items():
                unique_rooms.add(from_room)
                unique_rooms.update(targets.keys())

        return {
            'total_transitions': self.total_transitions,
            'unique_rooms': len(unique_rooms),
            'time_buckets': list(self._transitions.keys()),
            'ready': self.total_transitions >= MIN_TRANSITIONS,
        }

    def save(self):
        """Persist transition matrix to JSON."""
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            # Convert nested defaultdicts to plain dicts for JSON
            data = {
                'person_id': self.person_id,
                'total_transitions': self.total_transitions,
                'transitions': {
                    bucket: {
                        from_room: dict(targets)
                        for from_room, targets in rooms.items()
                    }
                    for bucket, rooms in self._transitions.items()
                }
            }
            with open(self._path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Markov model {self.person_id}: {e}")

    def _load(self):
        """Load transition matrix from JSON if available."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self.total_transitions = data.get('total_transitions', 0)
            raw = data.get('transitions', {})
            for bucket, rooms in raw.items():
                for from_room, targets in rooms.items():
                    for to_room, count in targets.items():
                        self._transitions[bucket][from_room][to_room] = count
            logger.debug(
                f"Loaded Markov model {self.person_id} "
                f"({self.total_transitions} transitions)"
            )
        except Exception as e:
            logger.warning(f"Failed to load Markov model {self.person_id}: {e}")
            self._transitions = defaultdict(
                lambda: defaultdict(lambda: defaultdict(int))
            )
            self.total_transitions = 0
