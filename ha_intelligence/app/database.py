"""SQLite database for HA Intelligence."""

import sqlite3
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = "/data/ha_intelligence.db"
MAX_RETRIES = 3
RETRY_BACKOFF = 0.1  # seconds


def get_db_path() -> str:
    """Return DB path, using local path for development."""
    if os.path.exists("/data"):
        return DB_PATH
    # Development fallback
    path = os.path.join(os.path.dirname(__file__), "dev_data")
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "ha_intelligence.db")


class Database:
    def __init__(self):
        self.path = get_db_path()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        conn = self._connect()
        try:
            # 1) Create base tables (v0.1.x compatible - no new columns)
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            # 2) Migrate: add columns that may be missing from older versions
            self._migrate(conn)
            # 3) Create indexes on migrated columns (safe now that columns exist)
            self._create_post_migration_indexes(conn)
            logger.info(f"Database initialized at {self.path}")
        finally:
            conn.close()

    def _migrate(self, conn):
        """Add columns that may be missing from older schema versions."""
        migrations = [
            ("events", "event_type", "TEXT DEFAULT 'state_changed'"),
            ("discovered_entities", "device_class", "TEXT"),
            ("discovered_entities", "platform", "TEXT"),
            ("discovered_entities", "device_id", "TEXT"),
            ("rooms", "enabled", "INTEGER DEFAULT 1"),
            ("persons", "enabled", "INTEGER DEFAULT 1"),
        ]
        for table, column, col_type in migrations:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                conn.commit()
                logger.info(f"Migration: added {table}.{column}")
            except sqlite3.OperationalError:
                pass  # Column already exists

    def _create_post_migration_indexes(self, conn):
        """Create indexes on columns added by migrations."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_entities_area ON discovered_entities(area_id)",
        ]
        for sql in indexes:
            try:
                conn.execute(sql)
                conn.commit()
            except sqlite3.OperationalError as e:
                logger.warning(f"Index creation skipped: {e}")

    def execute(self, query: str, params: tuple = (), fetch: bool = False):
        """Execute query with retry on 'database is locked'."""
        for attempt in range(MAX_RETRIES):
            conn = self._connect()
            try:
                cur = conn.execute(query, params)
                conn.commit()
                if fetch:
                    return [dict(row) for row in cur.fetchall()]
                return cur.rowcount
            except sqlite3.OperationalError as e:
                conn.rollback()
                if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"DB locked, retry {attempt + 1}/{MAX_RETRIES} in {wait}s")
                    time.sleep(wait)
                    continue
                logger.error(f"DB error: {e} | Query: {query[:100]}")
                raise
            except Exception as e:
                conn.rollback()
                logger.error(f"DB error: {e} | Query: {query[:100]}")
                raise
            finally:
                conn.close()

    def execute_many(self, query: str, params_list: list):
        """Execute many with retry on 'database is locked'."""
        for attempt in range(MAX_RETRIES):
            conn = self._connect()
            try:
                conn.executemany(query, params_list)
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                conn.rollback()
                if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"DB locked (many), retry {attempt + 1}/{MAX_RETRIES}")
                    time.sleep(wait)
                    continue
                logger.error(f"DB executemany error: {e}")
                raise
            except Exception as e:
                conn.rollback()
                logger.error(f"DB executemany error: {e}")
                raise
            finally:
                conn.close()

    # ── Events ──────────────────────────────────────────────────

    def insert_event(self, entity_id: str, old_state: str, new_state: str,
                     attributes: dict, timestamp: str = None,
                     event_type: str = 'state_changed'):
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        self.execute(
            """INSERT INTO events
               (entity_id, old_state, new_state, attributes, recorded_at, event_type)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (entity_id, old_state, new_state, json.dumps(attributes), ts, event_type)
        )

    def insert_events_bulk(self, events: list):
        self.execute_many(
            """INSERT INTO events
               (entity_id, old_state, new_state, attributes, recorded_at, event_type)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(e['entity_id'], e.get('old_state', ''),
              e.get('new_state', ''), json.dumps(e.get('attributes', {})),
              e.get('timestamp', datetime.now(timezone.utc).isoformat()),
              e.get('event_type', 'state_changed'))
             for e in events]
        )

    def get_event_count(self, hours: int = 24) -> int:
        rows = self.execute(
            """SELECT COUNT(*) as cnt FROM events
               WHERE recorded_at > datetime('now', ?)""",
            (f'-{hours} hours',), fetch=True
        )
        return rows[0]['cnt'] if rows else 0

    # ── Entities (discovered) ───────────────────────────────────

    def upsert_entity(self, entity_id: str, domain: str, area_id: str = None,
                      friendly_name: str = None, device_class: str = None,
                      platform: str = None, device_id: str = None):
        self.execute(
            """INSERT INTO discovered_entities
               (entity_id, domain, area_id, friendly_name, device_class,
                platform, device_id, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(entity_id) DO UPDATE SET
                 area_id = COALESCE(excluded.area_id, area_id),
                 friendly_name = COALESCE(excluded.friendly_name, friendly_name),
                 device_class = COALESCE(excluded.device_class, device_class),
                 platform = COALESCE(excluded.platform, platform),
                 device_id = COALESCE(excluded.device_id, device_id),
                 last_seen = datetime('now'),
                 event_count = event_count + 1""",
            (entity_id, domain, area_id, friendly_name, device_class,
             platform, device_id)
        )

    def upsert_entities_bulk(self, entities: list):
        """Bulk upsert entities from registry data."""
        self.execute_many(
            """INSERT INTO discovered_entities
               (entity_id, domain, area_id, friendly_name, device_class,
                platform, device_id, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(entity_id) DO UPDATE SET
                 area_id = COALESCE(excluded.area_id, area_id),
                 friendly_name = COALESCE(excluded.friendly_name, friendly_name),
                 device_class = COALESCE(excluded.device_class, device_class),
                 platform = COALESCE(excluded.platform, platform),
                 device_id = COALESCE(excluded.device_id, device_id),
                 last_seen = datetime('now')""",
            [(e['entity_id'], e['domain'], e.get('area_id'),
              e.get('friendly_name'), e.get('device_class'),
              e.get('platform'), e.get('device_id'))
             for e in entities]
        )

    def get_discovered_entities(self, domain: str = None) -> list:
        if domain:
            return self.execute(
                "SELECT * FROM discovered_entities WHERE domain = ? ORDER BY event_count DESC",
                (domain,), fetch=True
            )
        return self.execute(
            "SELECT * FROM discovered_entities ORDER BY event_count DESC",
            fetch=True
        )

    def get_entities_with_area(self) -> list:
        """Get entities that have an area_id assigned."""
        return self.execute(
            "SELECT * FROM discovered_entities WHERE area_id IS NOT NULL ORDER BY area_id, domain",
            fetch=True
        )

    def get_event_type_stats(self) -> list:
        """Get event counts grouped by event_type."""
        return self.execute(
            """SELECT event_type, COUNT(*) as cnt FROM events
               WHERE recorded_at > datetime('now', '-24 hours')
               GROUP BY event_type ORDER BY cnt DESC""",
            fetch=True
        )

    # ── Rooms ───────────────────────────────────────────────────

    def upsert_room(self, area_id: str, name: str, slug: str):
        # Preserve enabled status for existing rooms
        existing = self.execute(
            "SELECT enabled FROM rooms WHERE area_id = ? OR slug = ?",
            (area_id, slug), fetch=True)
        if existing:
            enabled_val = existing[0]['enabled']
            # Remove stale rows (e.g. area_id changed for same slug)
            self.execute(
                "DELETE FROM rooms WHERE slug = ? AND area_id != ?",
                (slug, area_id))
            self.execute(
                """INSERT INTO rooms (area_id, name, slug, enabled)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(area_id) DO UPDATE SET
                     name = excluded.name, slug = excluded.slug""",
                (area_id, name, slug, enabled_val))
        else:
            self.execute(
                "INSERT INTO rooms (area_id, name, slug) VALUES (?, ?, ?)",
                (area_id, name, slug))

    def get_rooms(self, enabled_only: bool = True) -> list:
        if enabled_only:
            return self.execute(
                "SELECT * FROM rooms WHERE enabled = 1", fetch=True)
        return self.execute("SELECT * FROM rooms", fetch=True)

    # ── Persons ─────────────────────────────────────────────────

    def upsert_person(self, entity_id: str, name: str, slug: str):
        existing = self.execute(
            "SELECT enabled FROM persons WHERE entity_id = ? OR slug = ?",
            (entity_id, slug), fetch=True)
        if existing:
            enabled_val = existing[0]['enabled']
            self.execute(
                "DELETE FROM persons WHERE slug = ? AND entity_id != ?",
                (slug, entity_id))
            self.execute(
                """INSERT INTO persons (entity_id, name, slug, enabled)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(entity_id) DO UPDATE SET
                     name = excluded.name, slug = excluded.slug""",
                (entity_id, name, slug, enabled_val))
        else:
            self.execute(
                "INSERT INTO persons (entity_id, name, slug) VALUES (?, ?, ?)",
                (entity_id, name, slug))

    def get_persons(self, enabled_only: bool = True) -> list:
        if enabled_only:
            return self.execute(
                "SELECT * FROM persons WHERE enabled = 1", fetch=True)
        return self.execute("SELECT * FROM persons", fetch=True)

    def set_room_enabled(self, slug: str, enabled: bool) -> int:
        return self.execute(
            "UPDATE rooms SET enabled = ? WHERE slug = ?",
            (1 if enabled else 0, slug))

    def set_person_enabled(self, slug: str, enabled: bool) -> int:
        return self.execute(
            "UPDATE persons SET enabled = ? WHERE slug = ?",
            (1 if enabled else 0, slug))

    # ── Feedback Questions ───────────────────────────────────────

    def create_feedback_question(self, question_type: str, target: str,
                                  question_text: str, options: list,
                                  prediction_id: int = None,
                                  confidence: float = None) -> int:
        conn = self._connect()
        cur = conn.execute(
            """INSERT INTO feedback_questions
               (question_type, target, question_text, options,
                prediction_id, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (question_type, target, question_text,
             json.dumps(options), prediction_id, confidence))
        conn.commit()
        return cur.lastrowid

    def answer_feedback_question(self, question_id: int, answer: str):
        self.execute(
            """UPDATE feedback_questions
               SET answered_at = datetime('now'), answer = ?
               WHERE id = ?""",
            (answer, question_id))

    def get_pending_questions(self) -> list:
        return self.execute(
            """SELECT * FROM feedback_questions
               WHERE answered_at IS NULL
               ORDER BY created_at DESC""", fetch=True)

    def get_feedback_stats(self) -> dict:
        today = self.execute(
            """SELECT COUNT(*) as cnt FROM feedback_questions
               WHERE answered_at IS NOT NULL
               AND date(answered_at) = date('now')""", fetch=True)
        total = self.execute(
            """SELECT COUNT(*) as cnt FROM feedback_questions
               WHERE answered_at IS NOT NULL""", fetch=True)
        pending = self.execute(
            """SELECT COUNT(*) as cnt FROM feedback_questions
               WHERE answered_at IS NULL""", fetch=True)
        first = self.execute(
            """SELECT MIN(created_at) as first_q FROM feedback_questions
               WHERE answered_at IS NOT NULL""", fetch=True)
        return {
            'answered_today': today[0]['cnt'] if today else 0,
            'answered_total': total[0]['cnt'] if total else 0,
            'pending': pending[0]['cnt'] if pending else 0,
            'first_answer_at': first[0]['first_q'] if first else None,
        }

    def get_question_by_id(self, question_id: int) -> dict | None:
        rows = self.execute(
            "SELECT * FROM feedback_questions WHERE id = ?",
            (question_id,), fetch=True)
        return rows[0] if rows else None

    # ── Learned Activities ───────────────────────────────────────

    def upsert_learned_activity(self, person: str, room: str, zone: str,
                                 devices_state: dict, activity: str):
        existing = self.execute(
            """SELECT id, confirmed_count FROM learned_activities
               WHERE person = ? AND room = ? AND zone = ?
               AND devices_state = ? AND activity = ?""",
            (person, room, zone or '', json.dumps(devices_state), activity),
            fetch=True)
        if existing:
            self.execute(
                """UPDATE learned_activities
                   SET confirmed_count = confirmed_count + 1,
                       last_confirmed = datetime('now')
                   WHERE id = ?""",
                (existing[0]['id'],))
        else:
            self.execute(
                """INSERT INTO learned_activities
                   (person, room, zone, devices_state, activity)
                   VALUES (?, ?, ?, ?, ?)""",
                (person, room, zone or '', json.dumps(devices_state), activity))

    def lookup_activity(self, person: str, room: str, zone: str,
                         devices_state: dict) -> dict | None:
        rows = self.execute(
            """SELECT activity, confirmed_count FROM learned_activities
               WHERE person = ? AND room = ? AND zone = ?
               AND devices_state = ?
               ORDER BY confirmed_count DESC LIMIT 1""",
            (person, room, zone or '', json.dumps(devices_state)),
            fetch=True)
        return rows[0] if rows else None

    def get_learned_activities(self, limit: int = 50) -> list:
        return self.execute(
            """SELECT * FROM learned_activities
               ORDER BY last_confirmed DESC LIMIT ?""",
            (limit,), fetch=True)

    # ── Config ──────────────────────────────────────────────────

    def get_config(self, key: str) -> str | None:
        rows = self.execute(
            "SELECT value FROM system_config WHERE key = ?",
            (key,), fetch=True
        )
        return rows[0]['value'] if rows else None

    def set_config(self, key: str, value: str):
        self.execute(
            """INSERT INTO system_config (key, value)
               VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET
                 value = excluded.value, updated_at = datetime('now')""",
            (key, value)
        )

    # ── Observations (ML feature vectors) ──────────────────────

    def insert_observation(self, area_id: str = None, person_id: str = None,
                           features: dict = None, label: str = None,
                           model_type: str = 'room', weight: float = 1.0):
        self.execute(
            """INSERT INTO observations
               (observed_at, area_id, person_id, features, label, model_type, weight)
               VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)""",
            (area_id, person_id, json.dumps(features or {}), label, model_type, weight)
        )

    def get_recent_observations(self, model_type: str = None,
                                 target_id: str = None,
                                 hours: int = 24, limit: int = 500) -> list:
        query = """SELECT * FROM observations
                   WHERE observed_at > datetime('now', ?)"""
        params = [f'-{hours} hours']
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        if target_id:
            if model_type == 'person':
                query += " AND person_id = ?"
            else:
                query += " AND area_id = ?"
            params.append(target_id)
        query += " ORDER BY observed_at DESC LIMIT ?"
        params.append(limit)
        return self.execute(query, tuple(params), fetch=True)

    def get_observation_count(self, model_type: str = None,
                               target_id: str = None) -> int:
        query = "SELECT COUNT(*) as cnt FROM observations WHERE 1=1"
        params = []
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        if target_id:
            if model_type == 'person':
                query += " AND person_id = ?"
            else:
                query += " AND area_id = ?"
            params.append(target_id)
        rows = self.execute(query, tuple(params), fetch=True)
        return rows[0]['cnt'] if rows else 0

    # ── Predictions ─────────────────────────────────────────────

    def insert_prediction(self, sensor_target: str, predicted_state: str,
                          confidence: float, method: str, features: dict = None) -> int:
        """Insert prediction and return its row ID for feedback tracking."""
        conn = self._connect()
        try:
            cur = conn.execute(
                """INSERT INTO predictions
                   (sensor_target, predicted_state, confidence, method, features)
                   VALUES (?, ?, ?, ?, ?)""",
                (sensor_target, predicted_state, confidence, method,
                 json.dumps(features or {}))
            )
            conn.commit()
            return cur.lastrowid
        except Exception as e:
            conn.rollback()
            logger.error(f"insert_prediction error: {e}")
            return 0
        finally:
            conn.close()

    def get_latest_prediction(self, sensor_target: str) -> dict | None:
        """Get the most recent unresolved prediction for a target."""
        rows = self.execute(
            """SELECT id, predicted_state, confidence, method
               FROM predictions
               WHERE sensor_target = ? AND was_correct IS NULL
               ORDER BY predicted_at DESC LIMIT 1""",
            (sensor_target,), fetch=True
        )
        return rows[0] if rows else None

    def update_prediction_feedback(self, prediction_id: int, actual_state: str,
                                    feedback_source: str = 'auto'):
        was_correct = None
        # Look up predicted_state to compare
        rows = self.execute(
            "SELECT predicted_state FROM predictions WHERE id = ?",
            (prediction_id,), fetch=True
        )
        if rows:
            was_correct = 1 if rows[0]['predicted_state'] == actual_state else 0
        self.execute(
            """UPDATE predictions SET
               actual_state = ?, was_correct = ?,
               feedback_source = ?, feedback_at = datetime('now')
               WHERE id = ?""",
            (actual_state, was_correct, feedback_source, prediction_id)
        )

    def get_recent_predictions(self, limit: int = 50) -> list:
        return self.execute(
            """SELECT * FROM predictions
               ORDER BY predicted_at DESC LIMIT ?""",
            (limit,), fetch=True
        )

    def get_prediction_accuracy(self, method: str = None, hours: int = 24) -> dict:
        query = """SELECT
                     COUNT(*) as total,
                     SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                     SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as incorrect,
                     SUM(CASE WHEN was_correct IS NULL THEN 1 ELSE 0 END) as pending
                   FROM predictions
                   WHERE predicted_at > datetime('now', ?)"""
        params = [f'-{hours} hours']
        if method:
            query += " AND method = ?"
            params.append(method)
        rows = self.execute(query, tuple(params), fetch=True)
        if not rows or rows[0]['total'] == 0:
            return {'total': 0, 'accuracy': 0.0, 'correct': 0, 'incorrect': 0, 'pending': 0}
        r = rows[0]
        evaluated = r['correct'] + r['incorrect']
        accuracy = r['correct'] / evaluated if evaluated > 0 else 0.0
        return {
            'total': r['total'], 'accuracy': round(accuracy, 3),
            'correct': r['correct'], 'incorrect': r['incorrect'], 'pending': r['pending'],
        }

    # ── Model Versions ──────────────────────────────────────────

    def upsert_model_version(self, model_name: str, version: int = 1,
                              accuracy: float = 0.0, samples_seen: int = 0):
        self.execute(
            """INSERT INTO model_versions (model_name, version, accuracy, samples_seen)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(model_name) DO UPDATE SET
                 version = excluded.version,
                 accuracy = excluded.accuracy,
                 samples_seen = excluded.samples_seen,
                 updated_at = datetime('now')""",
            (model_name, version, accuracy, samples_seen)
        )

    def get_model_versions(self) -> list:
        return self.execute(
            "SELECT * FROM model_versions ORDER BY model_name", fetch=True
        )

    def get_model_version(self, model_name: str) -> dict | None:
        rows = self.execute(
            "SELECT * FROM model_versions WHERE model_name = ?",
            (model_name,), fetch=True
        )
        return rows[0] if rows else None

    # ── State Priors ────────────────────────────────────────────

    def upsert_state_prior(self, target_type: str, target_id: str,
                            hour: int, weekday: int,
                            state: str, probability: float):
        self.execute(
            """INSERT INTO state_priors
               (target_type, target_id, hour, weekday, state, probability)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(target_type, target_id, hour, weekday, state) DO UPDATE SET
                 probability = excluded.probability,
                 updated_at = datetime('now')""",
            (target_type, target_id, hour, weekday, state, probability)
        )

    def get_state_prior(self, target_type: str, target_id: str,
                         hour: int, weekday: int) -> list:
        return self.execute(
            """SELECT state, probability FROM state_priors
               WHERE target_type = ? AND target_id = ? AND hour = ? AND weekday = ?
               ORDER BY probability DESC""",
            (target_type, target_id, hour, weekday), fetch=True
        )

    # ── Pruning ─────────────────────────────────────────────────

    def prune_old_events(self, days: int = 30) -> int:
        return self.execute(
            "DELETE FROM events WHERE recorded_at < datetime('now', ?)",
            (f'-{days} days',)
        )

    def prune_old_observations(self, days: int = 14) -> int:
        return self.execute(
            "DELETE FROM observations WHERE observed_at < datetime('now', ?)",
            (f'-{days} days',)
        )

    def prune_old_predictions(self, days: int = 7) -> int:
        return self.execute(
            "DELETE FROM predictions WHERE predicted_at < datetime('now', ?)",
            (f'-{days} days',)
        )

    # ── Stats ───────────────────────────────────────────────────

    def get_stats(self) -> dict:
        events_24h = self.get_event_count(24)
        events_total = self.execute(
            "SELECT COUNT(*) as cnt FROM events", fetch=True
        )[0]['cnt']
        entities = self.execute(
            "SELECT COUNT(*) as cnt FROM discovered_entities", fetch=True
        )[0]['cnt']
        rooms = self.execute(
            "SELECT COUNT(*) as cnt FROM rooms", fetch=True
        )[0]['cnt']
        persons = self.execute(
            "SELECT COUNT(*) as cnt FROM persons", fetch=True
        )[0]['cnt']
        observations = self.execute(
            "SELECT COUNT(*) as cnt FROM observations", fetch=True
        )[0]['cnt']
        return {
            'events_24h': events_24h,
            'events_total': events_total,
            'entities_discovered': entities,
            'rooms': rooms,
            'persons': persons,
            'observations_total': observations,
        }


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL,
    old_state TEXT DEFAULT '',
    new_state TEXT DEFAULT '',
    attributes TEXT DEFAULT '{}',
    recorded_at TEXT NOT NULL,
    processed INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_entity ON events(entity_id);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(recorded_at);

CREATE TABLE IF NOT EXISTS discovered_entities (
    entity_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    area_id TEXT,
    friendly_name TEXT,
    first_seen TEXT DEFAULT (datetime('now')),
    last_seen TEXT DEFAULT (datetime('now')),
    event_count INTEGER DEFAULT 1,
    weight REAL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS idx_entities_domain ON discovered_entities(domain);

CREATE TABLE IF NOT EXISTS rooms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    area_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    entity_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    slug TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_at TEXT DEFAULT (datetime('now')),
    sensor_target TEXT NOT NULL,
    predicted_state TEXT NOT NULL,
    confidence REAL NOT NULL,
    method TEXT NOT NULL,
    features TEXT DEFAULT '{}',
    actual_state TEXT,
    was_correct INTEGER,
    feedback_source TEXT,
    feedback_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(predicted_at);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at TEXT DEFAULT (datetime('now')),
    area_id TEXT,
    person_id TEXT,
    features TEXT DEFAULT '{}',
    label TEXT,
    model_type TEXT NOT NULL DEFAULT 'room',
    weight REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_obs_time ON observations(observed_at);
CREATE INDEX IF NOT EXISTS idx_obs_area ON observations(area_id);
CREATE INDEX IF NOT EXISTS idx_obs_person ON observations(person_id);
CREATE INDEX IF NOT EXISTS idx_obs_type ON observations(model_type);

CREATE TABLE IF NOT EXISTS model_versions (
    model_name TEXT PRIMARY KEY,
    version INTEGER DEFAULT 1,
    accuracy REAL DEFAULT 0.0,
    samples_seen INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS state_priors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    hour INTEGER NOT NULL,
    weekday INTEGER NOT NULL,
    state TEXT NOT NULL,
    probability REAL NOT NULL,
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(target_type, target_id, hour, weekday, state)
);
CREATE INDEX IF NOT EXISTS idx_priors_lookup ON state_priors(target_type, target_id, hour, weekday);

CREATE TABLE IF NOT EXISTS system_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Default config
INSERT OR IGNORE INTO system_config (key, value) VALUES
    ('version', '1.0.2'),
    ('started_at', datetime('now')),
    ('confidence_threshold', '0.4');

CREATE TABLE IF NOT EXISTS feedback_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now')),
    question_type TEXT NOT NULL,
    target TEXT NOT NULL,
    question_text TEXT NOT NULL,
    options TEXT DEFAULT '[]',
    answered_at TEXT,
    answer TEXT,
    prediction_id INTEGER,
    confidence REAL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_time ON feedback_questions(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback_questions(question_type);

CREATE TABLE IF NOT EXISTS learned_activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person TEXT NOT NULL,
    room TEXT NOT NULL,
    zone TEXT,
    devices_state TEXT DEFAULT '{}',
    activity TEXT NOT NULL,
    confirmed_count INTEGER DEFAULT 1,
    last_confirmed TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_activity_lookup ON learned_activities(person, room, zone);
"""
