"""SQLite database for HA Intelligence."""

import sqlite3
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = "/data/ha_intelligence.db"


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
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        conn = self._connect()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            logger.info(f"Database initialized at {self.path}")
        finally:
            conn.close()

    def execute(self, query: str, params: tuple = (), fetch: bool = False):
        conn = self._connect()
        try:
            cur = conn.execute(query, params)
            conn.commit()
            if fetch:
                return [dict(row) for row in cur.fetchall()]
            return cur.rowcount
        except Exception as e:
            conn.rollback()
            logger.error(f"DB error: {e} | Query: {query[:100]}")
            raise
        finally:
            conn.close()

    def execute_many(self, query: str, params_list: list):
        conn = self._connect()
        try:
            conn.executemany(query, params_list)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"DB executemany error: {e}")
            raise
        finally:
            conn.close()

    # ── Events ──────────────────────────────────────────────────

    def insert_event(self, entity_id: str, old_state: str, new_state: str,
                     attributes: dict, timestamp: str = None):
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        self.execute(
            """INSERT INTO events
               (entity_id, old_state, new_state, attributes, recorded_at)
               VALUES (?, ?, ?, ?, ?)""",
            (entity_id, old_state, new_state, json.dumps(attributes), ts)
        )

    def insert_events_bulk(self, events: list):
        self.execute_many(
            """INSERT INTO events
               (entity_id, old_state, new_state, attributes, recorded_at)
               VALUES (?, ?, ?, ?, ?)""",
            [(e['entity_id'], e.get('old_state', ''),
              e.get('new_state', ''), json.dumps(e.get('attributes', {})),
              e.get('timestamp', datetime.now(timezone.utc).isoformat()))
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
                      friendly_name: str = None):
        self.execute(
            """INSERT INTO discovered_entities
               (entity_id, domain, area_id, friendly_name, last_seen)
               VALUES (?, ?, ?, ?, datetime('now'))
               ON CONFLICT(entity_id) DO UPDATE SET
                 area_id = COALESCE(excluded.area_id, area_id),
                 friendly_name = COALESCE(excluded.friendly_name, friendly_name),
                 last_seen = datetime('now'),
                 event_count = event_count + 1""",
            (entity_id, domain, area_id, friendly_name)
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

    # ── Rooms ───────────────────────────────────────────────────

    def upsert_room(self, area_id: str, name: str, slug: str):
        self.execute(
            """INSERT INTO rooms (area_id, name, slug)
               VALUES (?, ?, ?)
               ON CONFLICT(area_id) DO UPDATE SET
                 name = excluded.name, slug = excluded.slug""",
            (area_id, name, slug)
        )

    def get_rooms(self) -> list:
        return self.execute("SELECT * FROM rooms", fetch=True)

    # ── Persons ─────────────────────────────────────────────────

    def upsert_person(self, entity_id: str, name: str, slug: str):
        self.execute(
            """INSERT INTO persons (entity_id, name, slug)
               VALUES (?, ?, ?)
               ON CONFLICT(entity_id) DO UPDATE SET
                 name = excluded.name, slug = excluded.slug""",
            (entity_id, name, slug)
        )

    def get_persons(self) -> list:
        return self.execute("SELECT * FROM persons", fetch=True)

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
        return {
            'events_24h': events_24h,
            'events_total': events_total,
            'entities_discovered': entities,
            'rooms': rooms,
            'persons': persons,
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

CREATE TABLE IF NOT EXISTS system_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Default config
INSERT OR IGNORE INTO system_config (key, value) VALUES
    ('version', '0.1.5'),
    ('started_at', datetime('now')),
    ('confidence_threshold', '0.4');
"""
