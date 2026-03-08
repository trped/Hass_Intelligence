"""Smart notification engine for HA Intelligence.

Publishes actionable notifications to HA via MQTT → persistent_notification.
Supports anomaly alerts, prediction-based notifications, and low confidence warnings.
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class NotificationEngine:
    """Manages smart notifications with rate limiting and quiet hours."""

    def __init__(self, options: dict, mqtt_publisher=None):
        self.mqtt = mqtt_publisher
        self.active = options.get('feedback_active', True)
        self.max_daily = options.get('feedback_max_daily', 3)
        self.cooldown_min = options.get('feedback_cooldown_min', 120)
        self.quiet_start = self._parse_time(options.get('feedback_quiet_start', '22:00'))
        self.quiet_end = self._parse_time(options.get('feedback_quiet_end', '07:00'))
        self.notify_service = options.get('feedback_notify_service', '')

        # HA Supervisor API
        self._ha_url = os.environ.get(
            'SUPERVISOR_URL', 'http://supervisor/core')
        self._ha_token = os.environ.get('SUPERVISOR_TOKEN', '')

        # State
        self._sent_today = 0
        self._reset_date = datetime.now(timezone.utc).date()
        self._last_sent_time = 0.0
        self._history = []  # list of (timestamp, type, message)

    @staticmethod
    def _parse_time(time_str: str) -> tuple:
        """Parse 'HH:MM' to (hour, minute)."""
        try:
            parts = time_str.split(':')
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return (22, 0)

    def _maybe_reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if today != self._reset_date:
            self._sent_today = 0
            self._reset_date = today

    def _is_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours."""
        now = datetime.now(timezone.utc) + timedelta(hours=1)  # CET approx
        current = (now.hour, now.minute)
        start = self.quiet_start
        end = self.quiet_end

        if start <= end:
            return start <= current < end
        else:
            # Wraps midnight (e.g. 22:00 - 07:00)
            return current >= start or current < end

    def _can_send(self) -> bool:
        """Check all rate limits."""
        if not self.active:
            return False

        self._maybe_reset_daily()

        if self._sent_today >= self.max_daily:
            return False

        if self._is_quiet_hours():
            return False

        elapsed = time.time() - self._last_sent_time
        if elapsed < self.cooldown_min * 60:
            return False

        return True

    def _call_ha_service(self, domain: str, service: str,
                          data: dict) -> bool:
        """Call a Home Assistant service via Supervisor API."""
        url = f"{self._ha_url}/api/services/{domain}/{service}"
        body = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(
            url, data=body, method='POST',
            headers={
                'Authorization': f'Bearer {self._ha_token}',
                'Content-Type': 'application/json',
            })
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status < 400
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            logger.error(f"HA service call failed {domain}.{service}: {e}")
            return False

    def _send(self, title: str, message: str, notification_type: str = 'info'):
        """Send notification via HA persistent_notification + optional mobile."""
        # Persistent notification (always visible in HA dashboard)
        self._call_ha_service('persistent_notification', 'create', {
            'title': f"HAI: {title}",
            'message': message,
            'notification_id': f"hai_{notification_type}_{int(time.time())}",
        })

        # Mobile notification if service configured
        if self.notify_service:
            parts = self.notify_service.split('.', 1)
            domain = parts[0] if len(parts) == 2 else 'notify'
            service = parts[1] if len(parts) == 2 else self.notify_service
            self._call_ha_service(domain, service, {
                'title': f"HAI: {title}",
                'message': message,
            })

        self._sent_today += 1
        self._last_sent_time = time.time()
        self._history.append((
            datetime.now(timezone.utc).isoformat(),
            notification_type,
            f"{title}: {message}",
        ))
        if len(self._history) > 50:
            self._history = self._history[-50:]

        logger.info(f"Notification sent: [{notification_type}] {title}")
        return True

    def send_actionable(self, title: str, message: str,
                         actions: list, tag: str = None):
        """Send actionable notification with buttons to mobile app.

        Args:
            actions: list of {"action": "slug", "title": "Label"}
            tag: notification tag for replacement
        """
        if not self._can_send():
            return False

        if not self.notify_service:
            logger.warning("No notify_service configured, cannot send "
                           "actionable notification")
            return False

        parts = self.notify_service.split('.', 1)
        domain = parts[0] if len(parts) == 2 else 'notify'
        service = parts[1] if len(parts) == 2 else self.notify_service

        data = {
            'title': f"HAI: {title}",
            'message': message,
            'data': {
                'actions': actions,
            },
        }
        if tag:
            data['data']['tag'] = tag

        ok = self._call_ha_service(domain, service, data)

        if ok:
            self._sent_today += 1
            self._last_sent_time = time.time()
            self._history.append((
                datetime.now(timezone.utc).isoformat(),
                'feedback',
                f"{title}: {message}",
            ))
            if len(self._history) > 50:
                self._history = self._history[-50:]
            logger.info(f"Actionable notification sent: {title}")

        return ok

    # ── Notification types ────────────────────────────────────────

    def notify_anomaly(self, area_id: str, score: float, details: str = ''):
        """Send anomaly notification for a room."""
        if not self._can_send():
            return False

        room_name = area_id.replace('_', ' ').title()
        title = f"Usædvanlig aktivitet i {room_name}"
        message = f"Anomali-score: {score:.2f}."
        if details:
            message += f" {details}"

        return self._send(title, message, 'anomaly')

    def notify_prediction(self, person_name: str, prediction: str, confidence: float):
        """Send prediction-based notification."""
        if not self._can_send():
            return False

        title = f"Forudsigelse for {person_name}"
        message = f"{prediction} (sikkerhed: {confidence:.0%})"

        return self._send(title, message, 'prediction')

    def notify_low_confidence(self, area_id: str, confidence: float, source: str):
        """Notify about persistent low confidence in a room."""
        if not self._can_send():
            return False

        room_name = area_id.replace('_', ' ').title()
        title = f"Lav sikkerhed i {room_name}"
        message = f"Nuværende sikkerhed: {confidence:.0%} (kilde: {source}). ML har brug for flere data."

        return self._send(title, message, 'low_confidence')

    def notify_system(self, title: str, message: str):
        """Send generic system notification."""
        if not self._can_send():
            return False
        return self._send(title, message, 'system')

    # ── Check triggers (called from main loop) ───────────────────

    def check_anomalies(self, ml_engine):
        """Check all rooms for anomalies and notify if needed."""
        if not self.active or not ml_engine:
            return

        for area_id, model in ml_engine.models.anomaly_models.items():
            stats = model.get_stats()
            if not stats or not stats.get('ready'):
                continue

            score = stats.get('score')
            if score is not None and score > 0.7:
                self.notify_anomaly(area_id, score)

    def check_low_confidence(self, room_states: dict, threshold: float = 0.4):
        """Check for persistently low confidence rooms."""
        if not self.active:
            return

        for area_id, info in room_states.items():
            conf = info.get('confidence', 0)
            source = info.get('source', 'unknown')
            if 0 < conf < threshold and source != 'rule_based':
                self.notify_low_confidence(area_id, conf, source)

    # ── Status ────────────────────────────────────────────────────

    def get_status(self) -> dict:
        self._maybe_reset_daily()
        return {
            'active': self.active,
            'sent_today': self._sent_today,
            'max_daily': self.max_daily,
            'cooldown_min': self.cooldown_min,
            'quiet_hours': f"{self.quiet_start[0]:02d}:{self.quiet_start[1]:02d}-{self.quiet_end[0]:02d}:{self.quiet_end[1]:02d}",
            'is_quiet': self._is_quiet_hours(),
            'can_send': self._can_send(),
            'history': self._history[-10:],
        }
