"""HA Intelligence Add-on - Main entry point."""

import asyncio
import json
import logging
import os
import signal
import sys

import aiohttp
import uvicorn

from database import Database
from discovery import Discovery
from event_listener import EventListener
from mqtt_publisher import MQTTPublisher
from web_ui import create_app


def resolve_supervisor_token() -> str:
    """Get SUPERVISOR_TOKEN from env or S6 container environment file."""
    token = os.environ.get('SUPERVISOR_TOKEN', '')
    if token:
        return token
    # Fallback: S6 stores env vars in files
    try:
        with open('/run/s6/container_environment/SUPERVISOR_TOKEN') as f:
            token = f.read().strip()
            if token:
                os.environ['SUPERVISOR_TOKEN'] = token
                return token
    except (FileNotFoundError, PermissionError):
        pass
    return ''

async def discover_mqtt(token: str) -> dict | None:
    """Auto-discover MQTT broker via Supervisor services API."""
    url = 'http://supervisor/services/mqtt'
    headers = {'Authorization': f'Bearer {token}'}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = (await resp.json()).get('data', {})
                    if data.get('host'):
                        return {
                            'host': data['host'],
                            'port': data.get('port', 1883),
                            'username': data.get('username'),
                            'password': data.get('password'),
                        }
    except Exception as e:
        logger.warning(f"MQTT auto-discovery failed: {e}")
    return None


# ── Logging ─────────────────────────────────────────────────────

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'info').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger('ha_intelligence')


# ── Load add-on options ─────────────────────────────────────────

def load_options() -> dict:
    """Load options from /data/options.json (set by HA Supervisor)."""
    options_path = '/data/options.json'
    if os.path.exists(options_path):
        with open(options_path) as f:
            return json.load(f)
    logger.warning("No options.json found, using defaults")
    return {}


# ── Sensor publishing ───────────────────────────────────────────

class SensorEngine:
    """Simple rule-based sensor publisher for v0.1.
    Publishes room and person sensors based on observed HA states.
    ML models will be added in v0.2+.
    """

    def __init__(self, db: Database, mqtt: MQTTPublisher, discovery: Discovery):
        self.db = db
        self.mqtt = mqtt
        self.discovery = discovery
        self._room_states = {}   # area_id -> last known state info
        self._person_states = {} # entity_id -> last known state info
        self._publish_interval = 60  # seconds

    async def on_state_change(self, entity_id: str, old_state: str,
                               new_state: str, attributes: dict):
        """Called on every state_changed event. Updates internal state."""
        domain = entity_id.split('.')[0]

        # Track motion/occupancy sensors → room state
        if domain in ('binary_sensor',) and any(
            kw in entity_id for kw in ('motion', 'occupancy', 'presence', 'mmwave')
        ):
            area_id = attributes.get('area_id')
            if area_id:
                self._update_room_state(area_id, entity_id, new_state)

        # Track person entities → person state
        if domain == 'person':
            self._update_person_state(entity_id, new_state, attributes)

        # Track device_tracker → person state
        if domain == 'device_tracker':
            self._update_tracker_state(entity_id, new_state, attributes)

    def _update_room_state(self, area_id: str, sensor_id: str, state: str):
        """Update room state based on motion/occupancy sensor."""
        if area_id not in self._room_states:
            self._room_states[area_id] = {
                'sensors': {}, 'last_occupied': None
            }
        self._room_states[area_id]['sensors'][sensor_id] = state
        if state == 'on':
            from datetime import datetime, timezone
            self._room_states[area_id]['last_occupied'] = (
                datetime.now(timezone.utc).isoformat()
            )

    def _update_person_state(self, entity_id: str, state: str, attrs: dict):
        """Update person state from person entity."""
        self._person_states[entity_id] = {
            'ha_state': state,
            'source': attrs.get('source'),
        }

    def _update_tracker_state(self, entity_id: str, state: str, attrs: dict):
        """Update person state from device_tracker."""
        # Try to match tracker to person via user_id or naming
        # For now, store it and let the periodic publisher resolve it
        pass

    async def periodic_publish(self):
        """Periodically publish sensor states to MQTT."""
        while True:
            try:
                await self._publish_rooms()
                await self._publish_persons()
                await self._publish_system()
            except Exception as e:
                logger.error(f"Publish error: {e}")
            await asyncio.sleep(self._publish_interval)

    async def _publish_rooms(self):
        """Publish room sensors based on collected state."""
        rooms = self.db.get_rooms()
        for room in rooms:
            area_id = room['area_id']
            room_info = self._room_states.get(area_id, {})
            sensors = room_info.get('sensors', {})

            # Determine room state
            any_occupied = any(v == 'on' for v in sensors.values())
            state = 'occupied' if any_occupied else 'empty'

            self.mqtt.publish_room(
                slug=room['slug'],
                name=room['name'],
                state=state,
                attributes={
                    'area_id': area_id,
                    'motion_sensors': len(sensors),
                    'active_sensors': sum(1 for v in sensors.values() if v == 'on'),
                    'last_occupied': room_info.get('last_occupied'),
                    'confidence': 0.7 if sensors else 0.3,
                    'source': 'rule_based',
                }
            )

    async def _publish_persons(self):
        """Publish person sensors based on collected state."""
        persons = self.db.get_persons()
        for person in persons:
            entity_id = person['entity_id']
            info = self._person_states.get(entity_id, {})

            ha_state = info.get('ha_state', 'unknown')
            is_home = ha_state == 'home'

            # Simple v0.1 logic: home → active, not_home → away
            state = 'active' if is_home else 'away'

            self.mqtt.publish_person(
                slug=person['slug'],
                name=person['name'],
                state=state,
                attributes={
                    'home': is_home,
                    'ha_state': ha_state,
                    'location': 'home' if is_home else 'away',
                    'room': 'unknown',
                    'activity': 'unknown',
                    'confidence': 0.8 if ha_state != 'unknown' else 0.3,
                    'source': 'rule_based',
                }
            )

    async def _publish_system(self):
        """Publish system status sensor."""
        stats = self.db.get_stats()
        self.mqtt.publish_system_status(
            status='learning',
            attributes={
                'version': '0.1.6',
                'events_24h': stats['events_24h'],
                'events_total': stats['events_total'],
                'entities_discovered': stats['entities_discovered'],
                'rooms': stats['rooms'],
                'persons': stats['persons'],
                'ml_active': False,
                'haiku_active': False,
            }
        )


# ── Main ────────────────────────────────────────────────────────

async def main():
    logger.info("=" * 50)
    logger.info("HA Intelligence v0.1.6 starting...")
    logger.info("=" * 50)

    # Load config
    options = load_options()
    os.environ.setdefault('LOG_LEVEL', options.get('log_level', 'info'))

    # Resolve SUPERVISOR_TOKEN (may be in S6 env file, not Docker env)
    token = resolve_supervisor_token()
    logger.info(f"SUPERVISOR_TOKEN present: {bool(token)}")

    # Initialize components
    db = Database()
    logger.info("Database initialized")

    # Auto-discover MQTT broker via Supervisor API
    mqtt_host = options.get('mqtt_host', 'core-mosquitto')
    mqtt_port = options.get('mqtt_port', 1883)
    mqtt_user = options.get('mqtt_user') or None
    mqtt_pass = options.get('mqtt_password') or None

    if token:
        discovered = await discover_mqtt(token)
        if discovered:
            mqtt_host = discovered['host']
            mqtt_port = discovered['port']
            if discovered.get('username'):
                mqtt_user = discovered['username']
                mqtt_pass = discovered.get('password')
            logger.info(f"MQTT auto-discovered: {mqtt_host}:{mqtt_port}")

    mqtt = MQTTPublisher(
        host=mqtt_host, port=mqtt_port,
        username=mqtt_user, password=mqtt_pass,
    )
    logger.info("MQTT publisher initialized")

    discovery = Discovery(db)
    await discovery.discover_all()
    logger.info("Discovery complete")

    sensor_engine = SensorEngine(db, mqtt, discovery)
    event_listener = EventListener(db, on_state_change=sensor_engine.on_state_change)

    # Create web app
    app = create_app(db, event_listener, mqtt)

    # Start all tasks
    loop = asyncio.get_event_loop()

    # Graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: signal_handler())

    # Start uvicorn in background
    config = uvicorn.Config(
        app, host="0.0.0.0", port=8099,
        log_level=options.get('log_level', 'info'),
    )
    server = uvicorn.Server(config)

    # Run all tasks concurrently
    tasks = [
        asyncio.create_task(event_listener.start()),
        asyncio.create_task(sensor_engine.periodic_publish()),
        asyncio.create_task(server.serve()),
    ]

    logger.info("All systems running")

    # Wait for shutdown
    try:
        await shutdown_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down...")
        await event_listener.stop()
        mqtt.stop()
        for task in tasks:
            task.cancel()
        logger.info("Goodbye!")


if __name__ == '__main__':
    asyncio.run(main())
