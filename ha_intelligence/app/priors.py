"""State Priors — nightly calculation of P(state | hour, weekday).

Aggregates observations from the database to compute per-hour, per-weekday
probability distributions for rooms and persons. Used by MLEngine to
weight predictions with historical patterns.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Minimum days of data before priors are considered reliable
MIN_DAYS_FOR_PRIORS = 7


class PriorCalculator:
    """Calculates and stores state priors from historical observations."""

    def __init__(self, db):
        self.db = db
        self._last_run = None

    def calculate_all_priors(self):
        """Aggregate observations into P(state | hour, weekday) for all targets.

        Groups observations by (model_type, target_id, hour, weekday) and
        computes the frequency of each label (state) within each group.
        Results are upserted into the state_priors table.
        """
        logger.info("Calculating state priors...")

        # Check if we have enough data (at least MIN_DAYS_FOR_PRIORS days)
        rows = self.db.execute(
            """SELECT MIN(observed_at) as earliest, COUNT(*) as total
               FROM observations""",
            fetch=True
        )
        if not rows or rows[0]['total'] == 0:
            logger.info("No observations yet — skipping prior calculation")
            return 0

        earliest = rows[0]['earliest']
        total_obs = rows[0]['total']
        if earliest:
            try:
                earliest_dt = datetime.fromisoformat(earliest)
                days_of_data = (datetime.now(timezone.utc) - earliest_dt).days
                if days_of_data < MIN_DAYS_FOR_PRIORS:
                    logger.info(
                        f"Only {days_of_data} days of data "
                        f"(need {MIN_DAYS_FOR_PRIORS}) — skipping priors"
                    )
                    return 0
            except (ValueError, TypeError):
                pass

        # Aggregate: GROUP BY model_type, target_id, hour, weekday, label
        # Use SQLite strftime to extract hour and weekday from observed_at
        # SQLite weekday: 0=Sunday, we want 0=Monday → (weekday + 6) % 7
        agg_rows = self.db.execute(
            """SELECT
                 model_type,
                 CASE WHEN model_type = 'room' THEN area_id ELSE person_id END as target_id,
                 CAST(strftime('%%H', observed_at) AS INTEGER) as hour,
                 (CAST(strftime('%%w', observed_at) AS INTEGER) + 6) %% 7 as weekday,
                 label,
                 COUNT(*) as cnt
               FROM observations
               WHERE observed_at > datetime('now', '-30 days')
                 AND label IS NOT NULL
                 AND (area_id IS NOT NULL OR person_id IS NOT NULL)
               GROUP BY model_type, target_id, hour, weekday, label
               ORDER BY model_type, target_id, hour, weekday, cnt DESC""",
            fetch=True
        )

        if not agg_rows:
            logger.info("No valid observations for prior calculation")
            return 0

        # Group by (model_type, target_id, hour, weekday) → compute probabilities
        # counts[key] = {label: count, ...}
        counts = defaultdict(lambda: defaultdict(int))
        for row in agg_rows:
            key = (row['model_type'], row['target_id'], row['hour'], row['weekday'])
            counts[key][row['label']] += row['cnt']

        # Calculate probabilities and upsert
        prior_count = 0
        for (model_type, target_id, hour, weekday), label_counts in counts.items():
            total = sum(label_counts.values())
            if total < 3:  # Skip sparse bins
                continue
            for label, count in label_counts.items():
                probability = round(count / total, 4)
                self.db.upsert_state_prior(
                    target_type=model_type,
                    target_id=target_id,
                    hour=hour,
                    weekday=weekday,
                    state=label,
                    probability=probability,
                )
                prior_count += 1

        self._last_run = datetime.now(timezone.utc)
        logger.info(
            f"Priors calculated: {prior_count} entries from "
            f"{total_obs} observations ({len(counts)} time slots)"
        )
        return prior_count

    def get_prior(self, target_type: str, target_id: str,
                  hour: int = None, weekday: int = None) -> dict:
        """Get prior probability distribution for a target at given time.

        Args:
            target_type: 'room' or 'person'
            target_id: area_id or person slug
            hour: Hour of day (0-23). Defaults to current hour.
            weekday: Day of week (0=Mon, 6=Sun). Defaults to current.

        Returns:
            Dict with:
              - 'priors': {state: probability, ...} sorted by probability desc
              - 'best_state': most likely state
              - 'best_probability': probability of most likely state
              - 'has_data': whether any priors exist for this slot
        """
        if hour is None or weekday is None:
            now = datetime.now(timezone.utc)
            if hour is None:
                hour = now.hour
            if weekday is None:
                weekday = now.weekday()

        rows = self.db.get_state_prior(target_type, target_id, hour, weekday)

        if not rows:
            return {
                'priors': {},
                'best_state': None,
                'best_probability': 0.0,
                'has_data': False,
            }

        priors = {row['state']: row['probability'] for row in rows}
        best = rows[0]  # Already sorted DESC by probability in DB query
        return {
            'priors': priors,
            'best_state': best['state'],
            'best_probability': best['probability'],
            'has_data': True,
        }

    def get_heatmap(self, target_type: str, target_id: str,
                    state: str = 'occupied') -> list:
        """Get 24×7 heatmap of prior probabilities for a target/state.

        Returns list of {hour, weekday, probability} for building
        a time-of-week heatmap visualization.
        """
        rows = self.db.execute(
            """SELECT hour, weekday, probability
               FROM state_priors
               WHERE target_type = ? AND target_id = ? AND state = ?
               ORDER BY weekday, hour""",
            (target_type, target_id, state), fetch=True
        )

        # Build complete 24×7 grid (fill missing slots with 0)
        grid = {}
        for row in rows:
            grid[(row['hour'], row['weekday'])] = row['probability']

        result = []
        for weekday in range(7):
            for hour in range(24):
                result.append({
                    'hour': hour,
                    'weekday': weekday,
                    'probability': grid.get((hour, weekday), 0.0),
                })
        return result

    def get_all_targets(self) -> list:
        """List all targets that have priors calculated."""
        rows = self.db.execute(
            """SELECT DISTINCT target_type, target_id
               FROM state_priors
               ORDER BY target_type, target_id""",
            fetch=True
        )
        return rows or []

    async def nightly_job(self):
        """Async task that runs prior calculation at 03:00 every night."""
        logger.info("Prior nightly job task started")
        while True:
            try:
                now = datetime.now(timezone.utc)
                # Calculate seconds until next 03:00 UTC
                target_hour = 3
                if now.hour < target_hour:
                    hours_until = target_hour - now.hour
                else:
                    hours_until = 24 - now.hour + target_hour
                seconds_until = (
                    hours_until * 3600
                    - now.minute * 60
                    - now.second
                )
                # Minimum wait: 60 seconds (avoid tight loops)
                seconds_until = max(60, seconds_until)

                logger.debug(
                    f"Prior nightly job sleeping {seconds_until}s "
                    f"(next run at {target_hour}:00 UTC)"
                )
                await asyncio.sleep(seconds_until)

                # Run calculation in thread to avoid blocking
                loop = asyncio.get_event_loop()
                count = await loop.run_in_executor(
                    None, self.calculate_all_priors
                )
                logger.info(f"Nightly prior calculation done: {count} entries")

            except asyncio.CancelledError:
                logger.info("Prior nightly job cancelled")
                break
            except Exception as e:
                logger.error(f"Prior nightly job error: {e}")
                # Wait 1 hour before retrying on error
                await asyncio.sleep(3600)
