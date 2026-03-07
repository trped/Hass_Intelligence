"""Claude Haiku integration for natural language household summaries."""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# Try importing anthropic — graceful fallback if not installed
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.info("anthropic SDK not installed — Haiku engine will use template fallback")


class TokenBudget:
    """Track daily token usage against budget."""

    def __init__(self, max_tokens_day: int = 5000):
        self.max_tokens_day = max_tokens_day
        self._used_today = 0
        self._reset_date = datetime.now(timezone.utc).date()

    def _maybe_reset(self):
        today = datetime.now(timezone.utc).date()
        if today != self._reset_date:
            self._used_today = 0
            self._reset_date = today

    @property
    def remaining(self) -> int:
        self._maybe_reset()
        return max(0, self.max_tokens_day - self._used_today)

    @property
    def used_today(self) -> int:
        self._maybe_reset()
        return self._used_today

    def can_spend(self, estimated: int = 300) -> bool:
        return self.remaining >= estimated

    def record(self, tokens: int):
        self._maybe_reset()
        self._used_today += tokens


class HaikuEngine:
    """Generate natural language household summaries using Claude Haiku.

    Features:
    - Rate limiting: max 1 API call per interval (default 5 min)
    - Change detection: only call API on significant state changes
    - Template fallback: works without API key
    - Token budget: tracks daily usage against configurable limit
    - Feature flags: anomaly, low_confidence, daily_summary, energy
    """

    VERSION = '0.7.3'

    def __init__(self, options: dict, db=None, ml_engine=None):
        self.db = db
        self.ml_engine = ml_engine

        # Config from add-on options
        self.api_key = options.get('haiku_api_key', '')
        self.enabled = options.get('haiku_enabled', False)
        self.flag_anomaly = options.get('haiku_anomaly', True)
        self.flag_low_confidence = options.get('haiku_low_confidence', False)
        self.flag_daily_summary = options.get('haiku_daily_summary', True)
        self.flag_energy = options.get('haiku_energy', False)
        self.max_tokens_day = options.get('haiku_max_tokens_day', 5000)

        # Rate limiting
        self._min_interval = 300  # 5 minutes
        self._last_call_time = 0.0

        # Change detection
        self._last_state_hash = ''

        # Token budget
        self._budget = TokenBudget(self.max_tokens_day)

        # Cached results
        self._last_summary = ''
        self._last_summary_source = 'none'
        self._last_summary_time = None
        self._total_api_calls = 0
        self._total_template_calls = 0
        self._last_error = None

        # Anthropic client
        self._client = None
        if self.api_key and ANTHROPIC_AVAILABLE and self.enabled:
            try:
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Haiku engine initialized with Anthropic API")
            except Exception as e:
                logger.error(f"Failed to create Anthropic client: {e}")
                self._last_error = str(e)
        elif not self.enabled:
            logger.info("Haiku engine disabled in config")
        elif not self.api_key:
            logger.info("Haiku engine: no API key — template fallback only")
        elif not ANTHROPIC_AVAILABLE:
            logger.info("Haiku engine: anthropic SDK not available — template fallback only")

    @property
    def active(self) -> bool:
        return self.enabled and self._client is not None

    @property
    def api_available(self) -> bool:
        return ANTHROPIC_AVAILABLE and bool(self.api_key) and self._client is not None

    # ── State collection ─────────────────────────────────────────

    def _collect_state(self, room_states: dict, person_states: dict,
                       system_info: dict = None) -> dict:
        """Collect current household state for summary generation."""
        state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'rooms': {},
            'persons': {},
        }

        # Room states
        for area_id, info in room_states.items():
            state['rooms'][area_id] = {
                'state': info.get('state', 'unknown'),
                'confidence': info.get('confidence', 0.0),
                'source': info.get('source', 'unknown'),
                'sensors_active': info.get('sensors_active', 0),
            }

        # Person states
        for entity_id, info in person_states.items():
            name = entity_id.replace('person.', '').replace('_', ' ').title()
            state['persons'][entity_id] = {
                'name': name,
                'state': info.get('state', 'unknown'),
                'room': info.get('room', 'unknown'),
                'confidence': info.get('confidence', 0.0),
                'source': info.get('source', 'unknown'),
            }

        # ML stats
        if self.ml_engine:
            ml_stats = self.ml_engine.get_stats()
            state['ml'] = {
                'active': ml_stats.get('ml_active', False),
                'total_samples': ml_stats.get('total_samples', 0),
                'anomalies': ml_stats.get('total_anomalies', 0),
            }

        if system_info:
            state['system'] = system_info

        return state

    def _compute_state_hash(self, state: dict) -> str:
        """Hash significant parts of state for change detection."""
        significant = {
            'rooms': {k: v['state'] for k, v in state.get('rooms', {}).items()},
            'persons': {k: (v['state'], v['room']) for k, v in state.get('persons', {}).items()},
        }
        raw = json.dumps(significant, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    # ── Template fallback ────────────────────────────────────────

    def _generate_template_summary(self, state: dict) -> str:
        """Generate a simple template-based summary (no API needed)."""
        parts = []

        # Person summaries
        persons = state.get('persons', {})
        for pid, info in persons.items():
            name = info.get('name', 'Ukendt')
            pstate = info.get('state', 'unknown')
            room = info.get('room', 'unknown')

            if pstate == 'away':
                parts.append(f"{name} er ikke hjemme")
            elif pstate == 'sleeping':
                parts.append(f"{name} sover")
            elif room and room != 'unknown':
                room_name = room.replace('_', ' ').title()
                parts.append(f"{name} er i {room_name}")
            elif pstate == 'active':
                parts.append(f"{name} er hjemme")
            elif pstate == 'idle':
                parts.append(f"{name} er hjemme (inaktiv)")
            else:
                parts.append(f"{name}: {pstate}")

        # Room activity
        rooms = state.get('rooms', {})
        active_rooms = [k.replace('_', ' ').title() for k, v in rooms.items()
                        if v.get('state') in ('occupied', 'active')]
        if active_rooms:
            parts.append(f"Aktivitet i: {', '.join(active_rooms)}")

        if not parts:
            return "Ingen data tilgængelig"

        return '. '.join(parts) + '.'

    # ── Claude API call ──────────────────────────────────────────

    def _build_prompt(self, state: dict) -> str:
        """Build structured prompt for Claude Haiku."""
        lines = [
            "Du er en smart-home assistent for et dansk hjem (Hyggebo).",
            "Beskriv husstandens aktuelle tilstand i 1-3 korte sætninger på dansk.",
            "Vær naturlig og informativ. Brug personernes navne.",
            "",
            "Aktuel tilstand:",
        ]

        # Persons
        for pid, info in state.get('persons', {}).items():
            name = info.get('name', 'Ukendt')
            pstate = info.get('state', 'unknown')
            room = info.get('room', 'unknown')
            conf = info.get('confidence', 0)
            lines.append(f"  {name}: tilstand={pstate}, rum={room}, sikkerhed={conf:.0%}")

        # Rooms
        lines.append("")
        for rid, info in state.get('rooms', {}).items():
            rstate = info.get('state', 'unknown')
            sensors = info.get('sensors_active', 0)
            lines.append(f"  {rid}: {rstate} (aktive sensorer: {sensors})")

        # ML context
        ml = state.get('ml', {})
        if ml.get('active'):
            lines.append(f"\nML er aktiv med {ml.get('total_samples', 0)} samples.")
            if self.flag_anomaly and ml.get('anomalies', 0) > 0:
                lines.append(f"Der er registreret {ml['anomalies']} anomalier.")

        lines.append("\nSvar kun med beskrivelsen, ingen forklaringer eller ekstra tekst.")
        return '\n'.join(lines)

    async def _call_api(self, prompt: str) -> tuple:
        """Call Claude Haiku API. Returns (summary, tokens_used)."""
        if not self._client:
            return None, 0

        try:
            # Run synchronous API call in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            ))

            summary = response.content[0].text.strip()
            tokens_used = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)

            return summary, tokens_used

        except Exception as e:
            logger.error(f"Haiku API call failed: {e}")
            self._last_error = str(e)
            return None, 0

    # ── Main generation method ───────────────────────────────────

    async def generate_summary(self, room_states: dict, person_states: dict,
                                system_info: dict = None, force: bool = False) -> dict:
        """Generate household summary.

        Returns dict with: summary, source, tokens_used, changed
        """
        state = self._collect_state(room_states, person_states, system_info)
        state_hash = self._compute_state_hash(state)
        now = time.time()

        # Check if state changed
        changed = state_hash != self._last_state_hash
        if not changed and not force:
            return {
                'summary': self._last_summary,
                'source': self._last_summary_source,
                'tokens_used': 0,
                'changed': False,
            }

        # Rate limiting
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval and not force:
            return {
                'summary': self._last_summary,
                'source': self._last_summary_source,
                'tokens_used': 0,
                'changed': False,
                'rate_limited': True,
            }

        # Try API first if available and budget allows
        tokens_used = 0
        source = 'template'

        if self.api_available and self._budget.can_spend(300):
            prompt = self._build_prompt(state)
            summary, tokens_used = await self._call_api(prompt)

            if summary:
                source = 'haiku'
                self._budget.record(tokens_used)
                self._total_api_calls += 1
                self._last_error = None
            else:
                # Fallback to template
                summary = self._generate_template_summary(state)
                self._total_template_calls += 1
        else:
            summary = self._generate_template_summary(state)
            self._total_template_calls += 1

        # Update cache
        self._last_summary = summary
        self._last_summary_source = source
        self._last_summary_time = datetime.now(timezone.utc)
        self._last_state_hash = state_hash
        self._last_call_time = now

        return {
            'summary': summary,
            'source': source,
            'tokens_used': tokens_used,
            'changed': True,
        }

    # ── Periodic task ────────────────────────────────────────────

    async def periodic_summary(self, sensor_engine):
        """Periodic task: generate summary every 5 minutes."""
        logger.info("Haiku periodic summary task started")
        await asyncio.sleep(30)  # Wait for initial data collection

        while True:
            try:
                result = await self.generate_summary(
                    sensor_engine._room_states,
                    sensor_engine._person_states,
                )
                if result.get('changed') and result.get('summary'):
                    # Publish via MQTT
                    sensor_engine.mqtt.publish_summary(
                        state=result['source'],
                        attributes={
                            'summary': result['summary'],
                            'source': result['source'],
                            'tokens_used': result['tokens_used'],
                            'tokens_today': self._budget.used_today,
                            'tokens_remaining': self._budget.remaining,
                            'api_calls_total': self._total_api_calls,
                            'template_calls_total': self._total_template_calls,
                        }
                    )
                    logger.debug(f"Summary published ({result['source']}): {result['summary'][:80]}...")

            except Exception as e:
                logger.error(f"Haiku periodic summary error: {e}")

            await asyncio.sleep(self._min_interval)

    # ── Status for API/UI ────────────────────────────────────────

    def get_status(self) -> dict:
        """Get haiku engine status for API/system sensor."""
        return {
            'active': self.active,
            'enabled': self.enabled,
            'api_available': self.api_available,
            'anthropic_sdk': ANTHROPIC_AVAILABLE,
            'has_api_key': bool(self.api_key),
            'tokens_used_today': self._budget.used_today,
            'tokens_remaining': self._budget.remaining,
            'tokens_max_day': self._budget.max_tokens_day,
            'api_calls_total': self._total_api_calls,
            'template_calls_total': self._total_template_calls,
            'last_summary': self._last_summary,
            'last_summary_source': self._last_summary_source,
            'last_summary_time': self._last_summary_time.isoformat() if self._last_summary_time else None,
            'last_error': self._last_error,
            'flags': {
                'anomaly': self.flag_anomaly,
                'low_confidence': self.flag_low_confidence,
                'daily_summary': self.flag_daily_summary,
                'energy': self.flag_energy,
            },
        }
