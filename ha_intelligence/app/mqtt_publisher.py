"""Publish virtual sensors to Home Assistant via MQTT Discovery."""

import json
import logging
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

DISCOVERY_PREFIX = "homeassistant"
STATE_PREFIX = "hai"


class MQTTPublisher:
    """Publishes room and person sensors to HA via MQTT Discovery."""

    def __init__(self, host: str, port: int, username: str = None,
                 password: str = None):
        self.client = mqtt.Client(
            client_id="ha_intelligence",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        if username:
            self.client.username_pw_set(username, password)

        self._connected = False
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        try:
            self.client.connect(host, port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"MQTT connect failed: {e}")

    @property
    def connected(self) -> bool:
        return self._connected

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected")
        else:
            logger.error(f"MQTT connect failed with code {rc}")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        self._connected = False
        logger.warning(f"MQTT disconnected (rc={rc})")

    def _publish_discovery(self, component: str, object_id: str,
                           name: str, icon: str = None,
                           extra_config: dict = None):
        """Register a sensor via MQTT Discovery."""
        unique_id = f"hai_{object_id}"
        config = {
            "name": name,
            "unique_id": unique_id,
            "object_id": f"hai_{object_id}",
            "state_topic": f"{STATE_PREFIX}/{object_id}/state",
            "json_attributes_topic": f"{STATE_PREFIX}/{object_id}/attributes",
            "device": {
                "identifiers": ["ha_intelligence"],
                "name": "HA Intelligence",
                "manufacturer": "Hyggebo",
                "model": "Intelligence System",
                "sw_version": "0.8.6",
            },
        }
        if icon:
            config["icon"] = icon
        if extra_config:
            config.update(extra_config)

        topic = f"{DISCOVERY_PREFIX}/{component}/{unique_id}/config"
        self.client.publish(topic, json.dumps(config), retain=True)

    def _publish_state(self, object_id: str, state: str, attributes: dict):
        """Publish state + attributes for a sensor."""
        self.client.publish(
            f"{STATE_PREFIX}/{object_id}/state",
            state, retain=True
        )
        # Add timestamp to attributes
        attrs = {**attributes, 'last_updated': datetime.now(timezone.utc).isoformat()}
        self.client.publish(
            f"{STATE_PREFIX}/{object_id}/attributes",
            json.dumps(attrs), retain=True
        )

    # ── Room sensors ────────────────────────────────────────────

    def publish_room(self, slug: str, name: str, state: str,
                     attributes: dict):
        """
        Publish sensor.hai_room_[slug].
        state: occupied / empty / active / quiet
        """
        object_id = f"room_{slug}"
        self._publish_discovery(
            'sensor', object_id, f"HAI {name}",
            icon="mdi:floor-plan"
        )
        self._publish_state(object_id, state, attributes)

    # ── Person sensors ──────────────────────────────────────────

    def publish_person(self, slug: str, name: str, state: str,
                       attributes: dict):
        """
        Publish sensor.hai_person_[slug].
        state: active / idle / sleeping / away
        """
        object_id = f"person_{slug}"
        self._publish_discovery(
            'sensor', object_id, f"HAI {name}",
            icon="mdi:account"
        )
        self._publish_state(object_id, state, attributes)

    # ── System sensor ───────────────────────────────────────────

    def publish_system_status(self, status: str, attributes: dict):
        """Publish sensor.hai_system with overall system status."""
        self._publish_discovery(
            'sensor', 'system', 'HAI System Status',
            icon="mdi:brain"
        )
        self._publish_state('system', status, attributes)

    # ── Household sensor ─────────────────────────────────────────

    def publish_household(self, state: str, attributes: dict):
        """Publish sensor.hai_household with household mode."""
        self._publish_discovery(
            'sensor', 'household', 'HAI Household',
            icon="mdi:home-account"
        )
        self._publish_state('household', state, attributes)

    def remove_sensor(self, object_id: str):
        """Remove a sensor from HA by publishing empty config."""
        unique_id = f"hai_{object_id}"
        topic = f"{DISCOVERY_PREFIX}/sensor/{unique_id}/config"
        self.client.publish(topic, "", retain=True)
        logger.info(f"Removed sensor: {unique_id}")

    # ── Activity sensors ───────────────────────────────────────

    def publish_activity(self, person_slug: str, person_name: str,
                         state: str, attributes: dict):
        """Publish sensor.hai_activity_[person] with current activity."""
        object_id = f"activity_{person_slug}"
        self._publish_discovery(
            'sensor', object_id, f"HAI Aktivitet {person_name}",
            icon="mdi:run"
        )
        self._publish_state(object_id, state, attributes)

    # ── Feedback sensor ────────────────────────────────────────

    def publish_feedback_status(self, state: str, attributes: dict):
        """Publish sensor.hai_feedback with feedback system status."""
        self._publish_discovery(
            'sensor', 'feedback', 'HAI Feedback',
            icon="mdi:comment-question-outline"
        )
        self._publish_state('feedback', state, attributes)

    def subscribe_feedback(self, callback):
        """Subscribe to hai/feedback/# for user answers."""
        self.client.subscribe("hai/feedback/#")
        self.client.message_callback_add("hai/feedback/#", callback)
        logger.info("Subscribed to hai/feedback/#")

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
