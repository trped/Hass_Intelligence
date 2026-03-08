"""Human-in-the-loop feedback engine for HA Intelligence."""

import json
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

MAX_ACTIVE_QUESTIONS = 20


class FeedbackEngine:
    """Asks users via MQTT notifications when ML confidence is low.

    Modes:
      - bootstrap (first N days): asks more frequently to build training data
      - normal: only asks below confidence threshold
    """

    def __init__(self, options: dict, db, mqtt_publisher,
                 ml_engine=None, notification_engine=None):
        self.db = db
        self.mqtt = mqtt_publisher
        self.ml_engine = ml_engine
        self.notification_engine = notification_engine
        self.activity_inference = None  # Set after ActivityInference created

        # Config
        self.confidence_threshold = options.get(
            'feedback_confidence_threshold', 40) / 100.0
        self.bootstrap_days = options.get('feedback_bootstrap_days', 14)
        self.active = options.get('feedback_active', True)

        # Rate limiting reuses NotificationEngine config
        self._question_counter = 0
        self._question_context = {}  # question_id → context for learning

        logger.info(
            f"FeedbackEngine initialized "
            f"(active={self.active}, threshold={self.confidence_threshold}, "
            f"bootstrap_days={self.bootstrap_days})"
        )

    @property
    def is_bootstrap(self) -> bool:
        """Check if we're still in bootstrap learning period."""
        stats = self.db.get_feedback_stats()
        first = stats.get('first_answer_at')
        if not first:
            return True  # No answers yet = definitely bootstrap
        try:
            first_dt = datetime.fromisoformat(first)
            elapsed = datetime.now(timezone.utc) - first_dt.replace(
                tzinfo=timezone.utc)
            return elapsed.days < self.bootstrap_days
        except (ValueError, TypeError):
            return True

    def get_effective_threshold(self) -> float:
        """Higher threshold during bootstrap = asks more questions."""
        if self.is_bootstrap:
            return min(self.confidence_threshold + 0.25, 0.85)
        return self.confidence_threshold

    def should_ask(self, confidence: float) -> bool:
        """Determine if we should ask the user about this prediction."""
        if not self.active:
            return False
        if not self.notification_engine or not self.notification_engine._can_send():
            return False
        # Enforce max active questions
        stats = self.db.get_feedback_stats()
        if stats['pending'] >= MAX_ACTIVE_QUESTIONS:
            return False
        return confidence < self.get_effective_threshold()

    def _enforce_question_limit(self):
        """Evict oldest entries from in-memory question context if over limit."""
        if len(self._question_context) <= MAX_ACTIVE_QUESTIONS:
            return
        # Sort by key (question_id, ascending = oldest first)
        sorted_ids = sorted(self._question_context.keys())
        while len(self._question_context) > MAX_ACTIVE_QUESTIONS:
            oldest_id = sorted_ids.pop(0)
            del self._question_context[oldest_id]
            logger.debug(f"Evicted oldest feedback context: {oldest_id}")

    # ── Question generation ────────────────────────────────────

    def _timestamp_str(self) -> str:
        """Format current local time for question text."""
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo('Europe/Copenhagen'))
        return now.strftime('%d/%m kl. %H:%M')

    def ask_room_state(self, room_slug: str, room_name: str,
                       confidence: float, prediction_id: int = None):
        """Ask user about room occupancy."""
        options = ["Optaget", "Tomt", "Ved ikke"]
        ts = self._timestamp_str()
        question_text = f"Er {room_name} optaget? ({ts})"

        q_id = self.db.create_feedback_question(
            question_type='room_state',
            target=f'room_{room_slug}',
            question_text=question_text,
            options=options,
            prediction_id=prediction_id,
            confidence=confidence,
        )

        self._send_notification(q_id, question_text, options)
        return q_id

    def ask_person_location(self, person_slug: str, person_name: str,
                             rooms: list, confidence: float,
                             prediction_id: int = None):
        """Ask user about person location."""
        room_names = [r['name'] for r in rooms[:3]]
        options = room_names + ["Ude", "Ved ikke"]
        ts = self._timestamp_str()
        question_text = f"Hvor er {person_name}? ({ts})"

        q_id = self.db.create_feedback_question(
            question_type='person_location',
            target=f'person_{person_slug}',
            question_text=question_text,
            options=options,
            prediction_id=prediction_id,
            confidence=confidence,
        )

        self._send_notification(q_id, question_text, options)
        return q_id

    def ask_activity(self, person_slug: str, person_name: str,
                      room_name: str, candidates: list,
                      confidence: float, prediction_id: int = None,
                      room_slug: str = '', zone: str = '',
                      device_states: dict = None):
        """Ask user about person's current activity."""
        options = candidates[:4] + ["Andet"]
        ts = self._timestamp_str()
        question_text = f"Hvad laver {person_name} i {room_name}? ({ts})"

        q_id = self.db.create_feedback_question(
            question_type='activity',
            target=f'person_{person_slug}',
            question_text=question_text,
            options=options,
            prediction_id=prediction_id,
            confidence=confidence,
        )

        # Store context for learning when answer arrives
        self._enforce_question_limit()
        self._question_context[q_id] = {
            'person_slug': person_slug,
            'room_slug': room_slug,
            'zone': zone,
            'device_states': device_states or {},
        }

        self._send_notification(q_id, question_text, options)
        return q_id

    # ── Notification sending ───────────────────────────────────

    def _send_notification(self, question_id: int, text: str,
                            options: list):
        """Send actionable notification via NotificationEngine."""
        if not self.notification_engine:
            logger.warning("No notification_engine, cannot send feedback")
            return

        actions = []
        for i, opt in enumerate(options):
            slug = opt.lower().replace(' ', '_').replace('æ', 'ae') \
                .replace('ø', 'oe').replace('å', 'aa')
            actions.append({
                "action": f"hai_fb_{question_id}_{slug}",
                "title": opt,
            })

        self.notification_engine.send_actionable(
            title="Feedback",
            message=text,
            actions=actions,
            tag=f"hai_feedback_{question_id}",
        )
        self._question_counter += 1
        logger.info(f"Feedback question #{question_id}: {text}")

    # ── Answer handling ────────────────────────────────────────

    def on_feedback_message(self, client, userdata, msg):
        """Handle MQTT feedback answers from hai/feedback/#."""
        try:
            topic = msg.topic  # hai/feedback/{question_id}
            parts = topic.split('/')
            if len(parts) < 3:
                return

            question_id = int(parts[2])
            answer = msg.payload.decode('utf-8')

            self._process_answer(question_id, answer)

        except (ValueError, IndexError, Exception) as e:
            logger.error(f"Feedback message error: {e}")

    def _process_answer(self, question_id: int, answer: str):
        """Process a user's answer to a feedback question."""
        question = self.db.get_question_by_id(question_id)
        if not question:
            logger.warning(f"Unknown feedback question: {question_id}")
            return

        if question['answered_at']:
            logger.debug(f"Question {question_id} already answered")
            return

        # Store answer
        self.db.answer_feedback_question(question_id, answer)
        logger.info(
            f"Feedback answer: Q#{question_id} "
            f"type={question['question_type']} answer={answer}"
        )

        # Train ML with user feedback (3x weight)
        if self.ml_engine and question['prediction_id']:
            try:
                # Map answer to state label
                label = self._answer_to_label(
                    question['question_type'], answer)
                if label:
                    self.db.update_prediction_feedback(
                        question['prediction_id'], label,
                        'user_feedback')
                    logger.info(
                        f"ML trained with user feedback: {label} (3x weight)")
            except Exception as e:
                logger.error(f"ML feedback training error: {e}")

        # Update learned activities if activity question
        if question['question_type'] == 'activity' and answer != 'Ved ikke':
            ctx = self._question_context.pop(question_id, None)
            if ctx and self.activity_inference:
                self.activity_inference.learn_from_feedback(
                    person_slug=ctx['person_slug'],
                    room_slug=ctx['room_slug'],
                    zone=ctx['zone'],
                    devices_state=ctx['device_states'],
                    activity=answer.lower().replace(' ', '_'),
                )

    def _answer_to_label(self, question_type: str, answer: str) -> str | None:
        """Convert user answer to ML label."""
        if answer in ('Ved ikke', 'Andet'):
            return None

        mapping = {
            'room_state': {
                'Optaget': 'occupied', 'optaget': 'occupied',
                'Tomt': 'empty', 'tomt': 'empty',
            },
            'person_location': {
                'Ude': 'not_home', 'ude': 'not_home',
            },
        }

        type_map = mapping.get(question_type, {})
        if answer in type_map:
            return type_map[answer]

        # For person_location, room names map to themselves
        if question_type == 'person_location':
            return answer.lower()

        # For activity, the answer is the activity itself
        if question_type == 'activity':
            return answer.lower().replace(' ', '_')

        return None

    # ── Status ─────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get feedback system status for dashboard/MQTT."""
        stats = self.db.get_feedback_stats()
        return {
            'active': self.active,
            'mode': 'bootstrap' if self.is_bootstrap else 'normal',
            'threshold': self.get_effective_threshold(),
            'pending_questions': stats['pending'],
            'answered_today': stats['answered_today'],
            'answered_total': stats['answered_total'],
        }

    def publish_status(self):
        """Publish feedback system status to MQTT."""
        status = self.get_status()
        self.mqtt.publish_feedback_status(
            state=status['mode'],
            attributes=status,
        )
