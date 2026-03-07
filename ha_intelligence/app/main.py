"""HA Intelligence Add-on - Main entry point."""

import os
import sys
import traceback

# Early crash log helper — captures import-time failures
_CRASH_LOG = '/data/startup.log'

def _write_crash(msg, exc=None):
    try:
        with open(_CRASH_LOG, 'a') as f:
            f.write(f"{msg}\n")
            if exc:
                traceback.print_exc(file=f)
            f.flush()
    except Exception:
        pass

try:
    import asyncio
    import json
    import logging
    import math
    import signal
    from datetime import datetime, timezone, timedelta

    import uvicorn

    from database import Database
    from discovery import Discovery
    from event_listener import EventListener
    from haiku_engine import HaikuEngine
    from ml_engine import MLEngine
    from mqtt_publisher import MQTTPublisher
    from registry import Registry
    from web_ui import create_app
except Exception as e:
    _write_crash(f"IMPORT ERROR: {type(e).__name__}: {e}", exc=e)
    raise


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
                 registry: Registry = None, ml_engine: MLEngine = None,
                 haiku_engine: 'HaikuEngine' = None):
        self.db = db
        self.mqtt = mqtt
        self.discovery = discovery
        self.registry = registry
        self.ml_engine = ml_engine
        self.haiku_engine = haiku_engine
        self._room_states = {}   # area_id -> last known state info
        self._person_states = {} # entity_id -> last known state info
        self._tracker_to_person = {}  # device_tracker entity_id -> person entity_id
        self._person_rooms = {}  # person_entity -> {area_id, area_name, confidence, source, distance, updated_at}
        self._bermuda_to_person = {}  # bermuda_sensor_entity -> person_entity
        self._netatmo_to_person = {}  # 'troels' -> 'person.troels' (Netatmo camera face detection)
        self._publish_interval = 60  # seconds
        self._stale_threshold = timedelta(minutes=30)
        self._last_predictions = {}  # sensor_target -> prediction_id (for feedback loop)

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

        # Track Bermuda BLE area sensors → person room
        if entity_id in self._bermuda_to_person:
            person_entity = self._bermuda_to_person[entity_id]
            # new_state is the area name from Bermuda (e.g. "Køkken")
            # area_id comes from the sensor's attributes
            ble_area_id = attributes.get('area_id', '')
            ble_distance = attributes.get('distance')
            if ble_area_id and new_state not in ('unavailable', 'unknown', ''):
                self._update_person_room(
                    person_entity=person_entity,
                    area_id=ble_area_id,
                    area_name=new_state,  # Bermuda state = area name
                    confidence=0.9,
                    source='ble',
                    distance=float(ble_distance) if ble_distance is not None else None,
                )
            elif new_state in ('unavailable', 'unknown', ''):
                # BLE lost — clear room data or mark as unknown
                if person_entity in self._person_rooms:
                    self._person_rooms[person_entity]['confidence'] = 0.1
                    self._person_rooms[person_entity]['source'] = 'ble_stale'
                    logger.debug(f"BLE lost for {person_entity}")

            # Sync person room data to ML engine for BLE features
            if self.ml_engine and person_entity in self._person_rooms:
                self.ml_engine.update_person_room(
                    person_entity, self._person_rooms[person_entity]
                )

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

        # Phase 4: Netatmo camera face detection → person room (highest confidence)
        if entity_id.startswith('input_text.netatmo_sidst_set_'):
            self._handle_netatmo_update(entity_id, new_state)

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
        # Phase 2: Sync room state to ML engine for rich feature extraction
        if self.ml_engine:
            self.ml_engine.update_room_state(area_id, self._room_states[area_id])

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

    def init_bermuda_mapping(self, bermuda_sensors: dict):
        """Initialize Bermuda sensor → person mapping from config.

        Args:
            bermuda_sensors: Dict mapping person entity_id → bermuda sensor entity_id
                e.g. {"person.troels": "sensor.bermuda_..._area_last_seen"}
        """
        for person_entity, bermuda_entity in bermuda_sensors.items():
            self._bermuda_to_person[bermuda_entity] = person_entity
            logger.info(f"BLE mapping: {bermuda_entity} → {person_entity}")
        logger.info(f"BLE tracking configured for {len(bermuda_sensors)} persons")

    def _update_person_room(self, person_entity: str, area_id: str,
                            area_name: str = None, confidence: float = 0.9,
                            source: str = 'ble', distance: float = None):
        """Update person-room tracking from BLE or motion fallback.

        Args:
            person_entity: Person entity_id (e.g. "person.troels")
            area_id: HA area_id (e.g. "kokken")
            area_name: Human-readable area name
            confidence: 0.0-1.0 confidence in the room assignment
            source: "ble" or "motion_fallback"
            distance: BLE distance in meters (None if unknown)
        """
        now = datetime.now(timezone.utc)
        prev = self._person_rooms.get(person_entity, {})
        prev_area = prev.get('area_id')

        self._person_rooms[person_entity] = {
            'area_id': area_id,
            'area_name': area_name or area_id,
            'confidence': confidence,
            'source': source,
            'distance': distance,
            'updated_at': now.isoformat(),
            'room_entered_at': prev.get('room_entered_at', now.isoformat())
                if prev_area == area_id else now.isoformat(),
        }

        if prev_area and prev_area != area_id:
            logger.debug(
                f"Person {person_entity} moved: {prev_area} → {area_id} "
                f"(source={source}, confidence={confidence:.2f})"
            )

    def init_netatmo_mapping(self, persons: list):
        """Initialize Netatmo person name → person entity mapping.

        Netatmo entities follow pattern: input_text.netatmo_sidst_set_<name>
        State format: 'room_name|ISO_timestamp'

        Args:
            persons: List of person dicts from DB (with 'slug' and 'entity_id')
        """
        for p in persons:
            slug = p.get('slug', '')
            entity_id = p.get('entity_id', '')
            if slug and entity_id:
                self._netatmo_to_person[slug] = entity_id
        logger.info(f"Netatmo mapping configured for {len(self._netatmo_to_person)} persons")

    def _handle_netatmo_update(self, entity_id: str, state: str):
        """Handle Netatmo camera face detection state change.

        Entity: input_text.netatmo_sidst_set_<name>
        State format: 'room_name|ISO_timestamp'
        """
        try:
            # Extract person name from entity_id
            name = entity_id.replace('input_text.netatmo_sidst_set_', '')

            # Parse state: "room|timestamp"
            if '|' not in state:
                return
            parts = state.split('|', 1)
            room = parts[0].strip()
            timestamp = parts[1].strip()

            if not room or not timestamp:
                return

            # Resolve person entity
            person_entity = self._netatmo_to_person.get(name)
            if not person_entity:
                logger.debug(f"Netatmo: unknown person '{name}'")
                return

            # Forward to ML engine
            if self.ml_engine:
                self.ml_engine.update_netatmo_room(
                    person_entity, room, timestamp
                )
                logger.debug(
                    f"Netatmo camera: {person_entity} seen in {room} "
                    f"at {timestamp}"
                )

            # Also update person room tracking (high confidence)
            # Map room name to area_id if possible
            area_id = room  # Default: use room name as area_id
            area_name = room
            if self.registry:
                # Try to find matching area
                for eid, aid in self.registry._area_map.items():
                    if aid.lower() == room.lower():
                        area_id = aid
                        break

            self._update_person_room(
                person_entity=person_entity,
                area_id=area_id,
                area_name=area_name,
                confidence=0.95,
                source='netatmo_camera',
                distance=None,
            )

        except Exception as e:
            logger.debug(f"Netatmo parse error for {entity_id}: {e}")

    def _infer_room_from_motion(self, person_entity: str) -> dict:
        """Motion fallback: infer room from most recently active motion sensor.

        Only used when BLE data is unavailable. Requires person to be home.

        Returns:
            Dict with room info or empty dict if cannot infer.
        """
        info = self._person_states.get(person_entity, {})
        if info.get('ha_state') != 'home':
            return {}

        # Find room with most recent motion
        best_room = None
        best_time = None
        for area_id, room_info in self._room_states.items():
            last_occ = room_info.get('last_occupied')
            if not last_occ:
                continue
            try:
                last_dt = datetime.fromisoformat(last_occ)
                if best_time is None or last_dt > best_time:
                    best_time = last_dt
                    best_room = area_id
            except (ValueError, TypeError):
                continue

        if best_room and best_time:
            # Only use if motion was within last 10 minutes
            age = (datetime.now(timezone.utc) - best_time).total_seconds() / 60.0
            if age <= 10:
                area_name = best_room
                if self.registry:
                    # Try to get friendly area name
                    for eid, aid in self.registry._area_map.items():
                        if aid == best_room:
                            area_name = best_room
                            break
                return {
                    'area_id': best_room,
                    'area_name': area_name,
                    'confidence': max(0.3, 0.6 - age * 0.03),  # Decays with age
                    'source': 'motion_fallback',
                    'distance': None,
                }
        return {}

    def _store_prediction(self, target: str, state: str, confidence: float,
                          method: str):
        """Store a prediction and track its ID for feedback."""
        try:
            pred_id = self.db.insert_prediction(
                sensor_target=target,
                predicted_state=state,
                confidence=confidence,
                method=method,
            )
            if pred_id:
                self._last_predictions[target] = pred_id
        except Exception as e:
            logger.debug(f"Store prediction error for {target}: {e}")

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
            if source != 'rule_based' and ml_state:
                state = ml_state
                confidence = ml_confidence
            else:
                state = rule_state
                confidence = rule_confidence

            # Phase 2: Evidence analysis
            evidence = {'sources': [], 'count': 0, 'detail': {}}
            if self.ml_engine:
                evidence = self.ml_engine.get_room_evidence(area_id, room_info)

            # Phase 3: Prior probability info
            prior_info = {'best_state': None, 'best_probability': 0.0, 'has_data': False}
            if self.ml_engine:
                prior_info = self.ml_engine.priors.get_prior('room', area_id)

            # Phase 4: Anomaly detection
            anomaly_info = {'score': None, 'ready': False, 'anomalies_detected': 0}
            if self.ml_engine:
                anomaly_stats = self.ml_engine.get_anomaly_score(area_id)
                if anomaly_stats:
                    anomaly_info = anomaly_stats

            # Feedback loop: close previous prediction with current actual state
            target = f'room_{area_id}'
            prev_id = self._last_predictions.get(target)
            if prev_id:
                try:
                    self.db.update_prediction_feedback(
                        prev_id, rule_state, 'auto_publish')
                except Exception as e:
                    logger.debug(f"Feedback error for {target}: {e}")

            # Store new prediction for next cycle
            self._store_prediction(
                target=target,
                state=state,
                confidence=confidence,
                method=source,
            )

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
                    # Phase 2: Evidence sources
                    'evidence_sources': evidence['sources'],
                    'evidence_count': evidence['count'],
                    'evidence_detail': evidence['detail'],
                    # Phase 3: State priors
                    'prior_state': prior_info.get('best_state'),
                    'prior_probability': prior_info.get('best_probability', 0.0),
                    'prior_available': prior_info.get('has_data', False),
                    # Phase 4: Anomaly detection
                    'anomaly_score': anomaly_info.get('score'),
                    'anomaly_ready': anomaly_info.get('ready', False),
                    'anomalies_detected': anomaly_info.get('anomalies_detected', 0),
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
            if source != 'rule_based' and ml_state:
                state = ml_state
                confidence = ml_confidence
            else:
                state = rule_state
                confidence = rule_confidence

            # Feedback loop: close previous prediction with current actual state
            if ha_state == 'home':
                actual_state = 'active'
            elif ha_state == 'not_home':
                actual_state = 'away'
            else:
                actual_state = ha_state
            target = f'person_{entity_id}'
            prev_id = self._last_predictions.get(target)
            if prev_id:
                try:
                    self.db.update_prediction_feedback(
                        prev_id, actual_state, 'auto_publish')
                except Exception as e:
                    logger.debug(f"Feedback error for {target}: {e}")

            # Store new prediction for next cycle
            self._store_prediction(
                target=target,
                state=state,
                confidence=confidence,
                method=source,
            )

            # Phase 3: Prior probability info
            person_slug = entity_id.replace('person.', '')
            person_prior_info = {'best_state': None, 'best_probability': 0.0, 'has_data': False}
            if self.ml_engine:
                person_prior_info = self.ml_engine.priors.get_prior('person', person_slug)

            # Phase 4: Best room from ML engine (merges Netatmo + BLE)
            best_room_ml = None
            if self.ml_engine:
                best_room_ml = self.ml_engine.get_person_best_room(entity_id)

            # BLE room data with motion fallback
            room_data = self._person_rooms.get(entity_id, {})

            # Phase 4: If ML engine has a better room estimate, use it
            if best_room_ml and best_room_ml.get('confidence', 0) > room_data.get('confidence', 0):
                room_data = {
                    'area_id': best_room_ml['room'],
                    'area_name': best_room_ml['room'],
                    'confidence': best_room_ml['confidence'],
                    'source': best_room_ml['source'],
                    'distance': None,
                }

            if not room_data or room_data.get('confidence', 0) < 0.2:
                # Try motion fallback for home persons without BLE
                if is_home:
                    fallback = self._infer_room_from_motion(entity_id)
                    if fallback:
                        room_data = fallback

            room_name = room_data.get('area_name', 'unknown')
            room_id = room_data.get('area_id', '')
            room_confidence = room_data.get('confidence', 0.0)
            room_source = room_data.get('source', 'none')
            ble_distance = room_data.get('distance')

            # Calculate minutes in current room
            minutes_in_room = 0
            room_entered = room_data.get('room_entered_at')
            if room_entered:
                try:
                    entered_dt = datetime.fromisoformat(room_entered)
                    minutes_in_room = round(
                        (datetime.now(timezone.utc) - entered_dt).total_seconds() / 60, 1
                    )
                except (ValueError, TypeError):
                    pass

            # Phase 4: Markov movement prediction
            predicted_next_room = None
            next_room_probability = 0.0
            if self.ml_engine and room_id and is_home:
                markov_result = self.ml_engine.predict_next_room(
                    entity_id, room_id
                )
                if markov_result and len(markov_result) > 0:
                    predicted_next_room = markov_result[0][0]
                    next_room_probability = round(markov_result[0][1], 3)

            self.mqtt.publish_person(
                slug=person['slug'],
                name=person['name'],
                state=state,
                attributes={
                    'home': is_home,
                    'ha_state': ha_state,
                    'location': 'home' if is_home else 'away',
                    'room': room_name if is_home else 'away',
                    'room_id': room_id,
                    'room_confidence': round(room_confidence, 2),
                    'room_source': room_source,
                    'ble_distance': round(ble_distance, 2) if ble_distance is not None else None,
                    'minutes_in_room': minutes_in_room,
                    'activity': state,
                    'confidence': confidence,
                    'source': source,
                    'rule_state': rule_state,
                    'rule_confidence': rule_confidence,
                    'ml_state': ml_state,
                    'ml_confidence': ml_confidence,
                    'ml_samples': ml_samples,
                    # Phase 3: State priors
                    'prior_state': person_prior_info.get('best_state'),
                    'prior_probability': person_prior_info.get('best_probability', 0.0),
                    'prior_available': person_prior_info.get('has_data', False),
                    # Phase 4: Markov movement prediction
                    'predicted_next_room': predicted_next_room,
                    'next_room_probability': next_room_probability,
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
                # Phase 4: Markov, anomaly, batch stats
                'ml_markov_models': ml_stats.get('markov_models', 0),
                'ml_anomaly_models': ml_stats.get('anomaly_models', 0),
                'ml_total_transitions': ml_stats.get('total_transitions', 0),
                'ml_total_anomaly_samples': ml_stats.get('total_anomaly_samples', 0),
                'ml_total_anomalies': ml_stats.get('total_anomalies', 0),
            }
            # Batch training stats
            batch_stats = ml_stats.get('batch')
            if batch_stats:
                ml_info['batch_room_models'] = batch_stats.get('room_models', 0)
                ml_info['batch_person_models'] = batch_stats.get('person_models', 0)
                ml_info['batch_last_trained'] = batch_stats.get('last_trained')

        # Prediction accuracy stats
        try:
            accuracy = self.db.get_prediction_accuracy()
            ml_info['accuracy_total'] = accuracy.get('total', 0)
            ml_info['accuracy_correct'] = accuracy.get('correct', 0)
            ml_info['accuracy_pct'] = round(accuracy.get('accuracy', 0.0) * 100, 1)
        except Exception:
            ml_info['accuracy_total'] = 0
            ml_info['accuracy_correct'] = 0
            ml_info['accuracy_pct'] = 0.0

        # Phase 3: Prior stats
        prior_targets = []
        if self.ml_engine:
            prior_targets = self.ml_engine.priors.get_all_targets()
        ml_info['priors_calculated'] = len(prior_targets)
        ml_info['priors_last_run'] = (
            self.ml_engine.priors._last_run.isoformat()
            if self.ml_engine and self.ml_engine.priors._last_run
            else None
        )

        # Haiku status
        haiku_active = False
        haiku_info = {}
        if self.haiku_engine:
            haiku_active = self.haiku_engine.active
            haiku_info = {
                'haiku_tokens_today': self.haiku_engine._budget.used_today,
                'haiku_api_calls': self.haiku_engine._total_api_calls,
                'haiku_source': self.haiku_engine._last_summary_source,
            }

        status = 'ml_active' if ml_info['ml_active'] else 'learning'
        self.mqtt.publish_system_status(
            status=status,
            attributes={
                'version': HaikuEngine.VERSION,
                'events_24h': stats['events_24h'],
                'events_total': stats['events_total'],
                'entities_discovered': stats['entities_discovered'],
                'rooms': stats['rooms'],
                'persons': stats['persons'],
                'haiku_active': haiku_active,
                **reg_info,
                **ml_info,
                **haiku_info,
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
    logger.info("HA Intelligence v0.5.3 starting...")
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

    # Initialize Haiku engine
    haiku_engine = HaikuEngine(options, db=db, ml_engine=ml_engine)
    logger.info(f"Haiku engine initialized (active={haiku_engine.active})")

    sensor_engine = SensorEngine(
        db, mqtt, discovery, registry=registry, ml_engine=ml_engine,
        haiku_engine=haiku_engine,
    )

    # Initialize BLE person-room tracking from config
    bermuda_raw = options.get('bermuda_sensors', '')
    if bermuda_raw:
        try:
            bermuda_map = json.loads(bermuda_raw) if isinstance(bermuda_raw, str) else bermuda_raw
            sensor_engine.init_bermuda_mapping(bermuda_map)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse bermuda_sensors config: {e}")

    # Initialize Netatmo camera face detection mapping
    persons = db.get_persons()
    sensor_engine.init_netatmo_mapping(persons)

    event_listener = EventListener(
        db, registry=registry,
        on_state_change=sensor_engine.on_state_change
    )

    # Create web app
    app = create_app(db, event_listener, mqtt, registry=registry, ml_engine=ml_engine,
                     haiku_engine=haiku_engine)

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
        asyncio.create_task(ml_engine.priors.nightly_job()),
        asyncio.create_task(ml_engine.models.nightly_batch_training()),
        asyncio.create_task(haiku_engine.periodic_summary(sensor_engine)),
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
    try:
        asyncio.run(main())
    except Exception as e:
        # Write crash info to file readable via SSH
        import traceback
        crash_log = '/data/startup.log'
        try:
            with open(crash_log, 'w') as f:
                f.write(f"CRASH at {datetime.now()}\n")
                f.write(f"Error: {e}\n\n")
                traceback.print_exc(file=f)
            print(f"Crash logged to {crash_log}", file=sys.stderr)
        except Exception:
            pass
        raise
