"""HA Intelligence Add-on - Main entry point."""

import asyncio
import json
import logging
import math
import os
import signal
import sys
from datetime import datetime, timezone, timedelta

import uvicorn

from database import Database
from discovery import Discovery
from event_listener import EventListener
from ml_engine import MLEngine
from mqtt_publisher import MQTTPublisher
from registry import Registry
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
    """Rule-based sensor publisher with ML integration hooks.
    Publishes room and person sensors based on observed HA states.
    ML models enhance predictions when enough data is collected.
    """

    def __init__(self, db: Database, mqtt: MQTTPublisher, discovery: Discovery,
                 registry: Registry = None, ml_engine: MLEngine = None):
        self.db = db
        self.mqtt = mqtt
        self.discovery = discovery
        self.registry = registry
        self.ml_engine = ml_engine
        self._room_states = {}   # area_id -> last known state info
        self._person_states = {} # entity_id -> last known state info
        self._tracker_to_person = {}  # device_tracker entity_id -> person entity_id
        self._publish_interval = 60  # seconds
        self._stale_threshold = timedelta(minutes=30)

    async def on_state_change(self, entity_id: str, old_state: str,
                               new_state: str, attributes: dict):
        """Called on every state_changed event. Updates internal state."""
        domain = entity_id.split('.')[0]

        # Forward to ML engine for training (non-blocking)
        if self.ml_engine:
            try:
                await self.ml_engine.on_state_change(
                    entity_id, old_state, new_state, attributes
                )
            except Exception as e:
                logger.debug(f"ML engine error: {e}")

        # Track motion/occupancy sensors → room state
        # Use registry to resolve area_id (NOT from state attributes)
        if domain in ('binary_sensor',) and any(
            kw in entity_id for kw in ('motion', 'occupancy', 'presence', 'mmwave')
        ):
            area_id = None
            if self.registry:
                area_id = self.registry.get_area_id(entity_id)
            if not area_id:
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
        """Update person state from device_tracker.
        Matches tracker to person via source attribute on person entities.
        """
        # Check if we already know which person this tracker belongs to
        if entity_id in self._tracker_to_person:
            person_id = self._tracker_to_person[entity_id]
            if person_id in self._person_states:
                self._person_states[person_id]['ha_state'] = state
                self._person_states[person_id]['source'] = entity_id
                return

        # Try to match: person entities have 'source' attr pointing to their tracker
        for person_id, info in self._person_states.items():
            if info.get('source') == entity_id:
                self._tracker_to_person[entity_id] = person_id
                self._person_states[person_id]['ha_state'] = state
                self._person_states[person_id]['source'] = entity_id
                return

    def _cleanup_stale_states(self):
        """Remove stale room states where no sensors are active and
        last_occupied exceeds stale threshold."""
        now = datetime.now(timezone.utc)
        stale_rooms = []
        for area_id, info in self._room_states.items():
            sensors = info.get('sensors', {})
            any_active = any(v == 'on' for v in sensors.values())
            if any_active:
                continue
            last_occ = info.get('last_occupied')
            if last_occ:
                try:
                    last_dt = datetime.fromisoformat(last_occ)
                    if now - last_dt > self._stale_threshold:
                        stale_rooms.append(area_id)
                except (ValueError, TypeError):
                    stale_rooms.append(area_id)
        for area_id in stale_rooms:
            self._room_states[area_id]['sensors'] = {}
            logger.debug(f"Cleared stale sensors for room {area_id}")

    async def periodic_publish(self):
        """Periodically publish sensor states to MQTT."""
        while True:
            try:
                self._cleanup_stale_states()
                await self._publish_rooms()
                await self._publish_persons()
                await self._publish_system()
                await self._publish_time_context()
                await self._publish_household()
            except Exception as e:
                logger.error(f"Publish error: {e}")
            await asyncio.sleep(self._publish_interval)

    async def periodic_maintenance(self):
        """Run database pruning every 6 hours."""
        while True:
            await asyncio.sleep(6 * 3600)  # 6 hours
            try:
                pruned_events = self.db.prune_old_events(days=30)
                pruned_obs = self.db.prune_old_observations(days=14)
                pruned_pred = self.db.prune_old_predictions(days=7)
                logger.info(
                    f"Maintenance: pruned {pruned_events} events, "
                    f"{pruned_obs} observations, {pruned_pred} predictions"
                )
            except Exception as e:
                logger.error(f"Maintenance error: {e}")

    async def _publish_rooms(self):
        """Publish room sensors with ML prediction + rule-based fallback."""
        rooms = self.db.get_rooms()
        for room in rooms:
            area_id = room['area_id']
            room_info = self._room_states.get(area_id, {})
            sensors = room_info.get('sensors', {})

            # Rule-based state
            any_occupied = any(v == 'on' for v in sensors.values())
            rule_state = 'occupied' if any_occupied else 'empty'
            rule_confidence = 0.7 if sensors else 0.3

            # ML prediction (may return None if not enough data)
            ml_state = None
            ml_confidence = 0.0
            ml_samples = 0
            source = 'rule_based'

            if self.ml_engine:
                ml_result = self.ml_engine.predict_room(area_id, room_info)
                if ml_result:
                    ml_state, ml_confidence, source_tag = ml_result
                    model = self.ml_engine.models.room_models.get(area_id)
                    ml_samples = model.samples_seen if model else 0
                    # ML wins if higher confidence
                    if ml_confidence > rule_confidence:
                        source = source_tag

            # Choose best prediction
            if source == 'ml_river' and ml_state:
                state = ml_state
                confidence = ml_confidence
            else:
                state = rule_state
                confidence = rule_confidence

            self.mqtt.publish_room(
                slug=room['slug'],
                name=room['name'],
                state=state,
                attributes={
                    'area_id': area_id,
                    'motion_sensors': len(sensors),
                    'active_sensors': sum(1 for v in sensors.values() if v == 'on'),
                    'last_occupied': room_info.get('last_occupied'),
                    'confidence': confidence,
                    'source': source,
                    'rule_state': rule_state,
                    'rule_confidence': rule_confidence,
                    'ml_state': ml_state,
                    'ml_confidence': ml_confidence,
                    'ml_samples': ml_samples,
                }
            )

    async def _publish_persons(self):
        """Publish person sensors with ML prediction + rule-based fallback."""
        persons = self.db.get_persons()

        # Count rooms with active motion for person features
        rooms_with_motion = sum(
            1 for info in self._room_states.values()
            if any(v == 'on' for v in info.get('sensors', {}).values())
        )

        for person in persons:
            entity_id = person['entity_id']
            info = self._person_states.get(entity_id, {})

            ha_state = info.get('ha_state', 'unknown')
            is_home = ha_state == 'home'

            # Rule-based state
            rule_state = 'active' if is_home else 'away'
            rule_confidence = 0.8 if ha_state != 'unknown' else 0.3

            # ML prediction
            ml_state = None
            ml_confidence = 0.0
            ml_samples = 0
            source = 'rule_based'

            if self.ml_engine:
                ml_result = self.ml_engine.predict_person(
                    entity_id, info, rooms_with_motion
                )
                if ml_result:
                    ml_state, ml_confidence, source_tag = ml_result
                    model = self.ml_engine.models.person_models.get(entity_id)
                    ml_samples = model.samples_seen if model else 0
                    if ml_confidence > rule_confidence:
                        source = source_tag

            # Choose best prediction
            if source == 'ml_river' and ml_state:
                state = ml_state
                confidence = ml_confidence
            else:
                state = rule_state
                confidence = rule_confidence

            self.mqtt.publish_person(
                slug=person['slug'],
                name=person['name'],
                state=state,
                attributes={
                    'home': is_home,
                    'ha_state': ha_state,
                    'location': 'home' if is_home else 'away',
                    'room': 'unknown',
                    'activity': state,
                    'confidence': confidence,
                    'source': source,
                    'rule_state': rule_state,
                    'rule_confidence': rule_confidence,
                    'ml_state': ml_state,
                    'ml_confidence': ml_confidence,
                    'ml_samples': ml_samples,
                }
            )

    async def _publish_system(self):
        """Publish system status sensor."""
        stats = self.db.get_stats()
        reg_info = {}
        if self.registry:
            reg_info = {
                'registry_entities': self.registry.entity_count,
                'registry_devices': self.registry.device_count,
                'registry_mapped': self.registry.mapped_count,
            }

        # ML stats
        ml_info = {
            'ml_active': False,
            'ml_room_models': 0,
            'ml_person_models': 0,
            'ml_total_samples': 0,
        }
        if self.ml_engine:
            ml_stats = self.ml_engine.get_stats()
            ml_info = {
                'ml_active': ml_stats.get('ml_active', False),
                'ml_room_models': ml_stats.get('room_models', 0),
                'ml_person_models': ml_stats.get('person_models', 0),
                'ml_total_samples': ml_stats.get('total_samples', 0),
                'ml_threshold': ml_stats.get('ml_threshold', 50),
            }

        status = 'ml_active' if ml_info['ml_active'] else 'learning'
        self.mqtt.publish_system_status(
            status=status,
            attributes={
                'version': '0.3.0',
                'events_24h': stats['events_24h'],
                'events_total': stats['events_total'],
                'entities_discovered': stats['entities_discovered'],
                'rooms': stats['rooms'],
                'persons': stats['persons'],
                'haiku_active': False,
                **reg_info,
                **ml_info,
            }
        )

    async def _publish_time_context(self):
        """Publish time context sensor with time-of-day category."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()

        # Categorize time of day
        if 6 <= hour < 10:
            period = 'morgen'
        elif 10 <= hour < 17:
            period = 'dag'
        elif 17 <= hour < 22:
            period = 'aften'
        else:
            period = 'nat'

        # Cyclic time encoding
        hour_sin = round(math.sin(2 * math.pi * hour / 24), 4)
        hour_cos = round(math.cos(2 * math.pi * hour / 24), 4)
        weekday_sin = round(math.sin(2 * math.pi * weekday / 7), 4)
        weekday_cos = round(math.cos(2 * math.pi * weekday / 7), 4)
        is_weekend = weekday >= 5

        self.mqtt.publish_time_context(
            state=period,
            attributes={
                'hour': hour,
                'weekday': weekday,
                'weekday_name': ['mandag', 'tirsdag', 'onsdag', 'torsdag',
                                 'fredag', 'lørdag', 'søndag'][weekday],
                'is_weekend': is_weekend,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'weekday_sin': weekday_sin,
                'weekday_cos': weekday_cos,
            }
        )

    async def _publish_household(self):
        """Publish household mode sensor based on person states + time."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()

        # Count persons home
        persons_home = sum(
            1 for info in self._person_states.values()
            if info.get('ha_state') == 'home'
        )
        total_persons = len(self._person_states) or 1

        # Count rooms with active motion
        rooms_active = sum(
            1 for info in self._room_states.values()
            if any(v == 'on' for v in info.get('sensors', {}).values())
        )

        # Infer household mode
        if persons_home == 0:
            mode = 'tom'
        elif hour < 6 or hour >= 23:
            mode = 'nat'
        elif 6 <= hour < 10:
            mode = 'morgen'
        elif weekday >= 5:
            mode = 'weekend'
        else:
            mode = 'hverdag'

        self.mqtt.publish_household(
            state=mode,
            attributes={
                'persons_home': persons_home,
                'persons_total': total_persons,
                'rooms_active': rooms_active,
                'occupancy_ratio': round(persons_home / total_persons, 2),
            }
        )


# ── Main ────────────────────────────────────────────────────────

async def main():
    logger.info("=" * 50)
    logger.info("HA Intelligence v0.3.0 starting...")
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

    # Load entity/device registries
    registry = Registry(db)
    await registry.load_all()

    # MQTT config from add-on options (no auto-discovery)
    mqtt_host = options.get('mqtt_host', 'a0d7b954-emqx')
    mqtt_port = options.get('mqtt_port', 1883)
    mqtt_user = options.get('mqtt_user') or None
    mqtt_pass = options.get('mqtt_password') or None
    logger.info(f"MQTT config: {mqtt_host}:{mqtt_port} (user: {mqtt_user or 'none'})")

    mqtt = MQTTPublisher(
        host=mqtt_host, port=mqtt_port,
        username=mqtt_user, password=mqtt_pass,
    )
    logger.info("MQTT publisher initialized")

    discovery = Discovery(db)
    await discovery.discover_all()
    logger.info("Discovery complete")

    # Initialize ML engine
    ml_engine = MLEngine(db, registry=registry)
    logger.info("ML engine initialized")

    sensor_engine = SensorEngine(
        db, mqtt, discovery, registry=registry, ml_engine=ml_engine
    )
    event_listener = EventListener(
        db, registry=registry,
        on_state_change=sensor_engine.on_state_change
    )

    # Create web app
    app = create_app(db, event_listener, mqtt, registry=registry, ml_engine=ml_engine)

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
        asyncio.create_task(sensor_engine.periodic_maintenance()),
        asyncio.create_task(ml_engine.models.periodic_save()),
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
        # Save ML models before exit
        try:
            ml_engine.models.save_all()
            logger.info("ML models saved")
        except Exception as e:
            logger.error(f"Failed to save ML models: {e}")
        await event_listener.stop()
        mqtt.stop()
        for task in tasks:
            task.cancel()
        logger.info("Goodbye!")


if __name__ == '__main__':
    asyncio.run(main())
