"""Settings manager — writable app settings separate from HA Supervisor options.

HA Supervisor manages /data/options.json (read-only from app perspective).
This module manages /data/settings.json for user-configurable settings
that can be changed at runtime via the web UI.
"""

import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SETTINGS_PATH = '/data/settings.json'

# Default settings structure
DEFAULTS = {
    'hustilstand': {
        'enabled': True,
        'entity_id': 'input_select.hus_tilstand',
        'state_map': {
            'hjemme': 'hjemme',
            'nat': 'nat',
            'ude': 'ude',
            'kun_hunde': 'kun_hunde',
            'ferie': 'ferie',
        },
    },
    'persons': {
        # person slug -> config overrides
        # e.g. 'troels': {'bermuda_sensor': 'sensor.xxx', 'zones': {...}}
    },
    'rooms': {
        # area_id -> config overrides
        # e.g. 'alrum': {'name': 'Alrum', 'icon': 'mdi:sofa'}
    },
    'zones': {
        # HA zone -> activity mapping
        # e.g. 'zone.work': {'activity': 'arbejde', 'persons': ['troels']}
    },
    'activity': {
        'zone_map': {},
        'device_sensors': {},
        'epl_zone_config': {},
    },
    'feedback': {
        # Overrides for feedback settings (base comes from options.json)
    },
    'mqtt': {
        # Overrides for MQTT settings
    },
    'ml': {
        'threshold': 50,
        'ml_weight': 0.7,
        'prior_weight': 0.3,
    },
    'system': {
        'publish_interval': 60,
    },
    'entity_selections': {
        'persons': [],
        'presence': [],
        'ble_tracking': [],
        'lights': [],
        'climate': [],
        'media': [],
        'energy': [],
        'appliances': [],
        'security': [],
        'calendars': [],
        'cameras': [],
    },
    'entity_suggestions_dismissed': [],
}


class SettingsManager:
    """Manages writable app settings with persistence to /data/settings.json."""

    def __init__(self, options: dict = None):
        """Initialize with HA Supervisor options as read-only base.

        Args:
            options: Dict from /data/options.json (HA Supervisor managed)
        """
        self._options = options or {}
        self._settings = deepcopy(DEFAULTS)
        self._load()

    def _load(self):
        """Load settings from disk, merging with defaults."""
        if not os.path.exists(SETTINGS_PATH):
            logger.info("No settings.json found, using defaults")
            return

        try:
            with open(SETTINGS_PATH) as f:
                saved = json.load(f)
            # Deep merge saved settings into defaults
            self._deep_merge(self._settings, saved)
            logger.info(f"Settings loaded from {SETTINGS_PATH}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load settings: {e}")

    def _save(self):
        """Persist current settings to disk."""
        try:
            with open(SETTINGS_PATH, 'w') as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
            logger.debug("Settings saved")
        except IOError as e:
            logger.error(f"Failed to save settings: {e}")

    @staticmethod
    def _deep_merge(base: dict, override: dict):
        """Recursively merge override into base (mutates base)."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                SettingsManager._deep_merge(base[key], value)
            else:
                base[key] = value

    # ── Read API ──────────────────────────────────────────────────

    def get_all(self) -> dict:
        """Return all settings (merged defaults + saved)."""
        return deepcopy(self._settings)

    def get_category(self, category: str) -> Optional[dict]:
        """Return settings for a specific category."""
        if category in self._settings:
            return deepcopy(self._settings[category])
        return None

    def get(self, category: str, key: str, default: Any = None) -> Any:
        """Get a single setting value."""
        cat = self._settings.get(category, {})
        return cat.get(key, default)

    def get_option(self, key: str, default: Any = None) -> Any:
        """Get a value from HA Supervisor options (read-only)."""
        return self._options.get(key, default)

    # ── Write API ─────────────────────────────────────────────────

    def set_category(self, category: str, data: dict):
        """Replace entire category settings and persist."""
        self._settings[category] = data
        self._save()
        logger.info(f"Settings category '{category}' updated")

    def update_category(self, category: str, updates: dict):
        """Merge updates into a category and persist."""
        if category not in self._settings:
            self._settings[category] = {}
        self._deep_merge(self._settings[category], updates)
        self._save()
        logger.info(f"Settings category '{category}' patched")

    def set_value(self, category: str, key: str, value: Any):
        """Set a single value in a category and persist."""
        if category not in self._settings:
            self._settings[category] = {}
        self._settings[category][key] = value
        self._save()

    def delete_key(self, category: str, key: str) -> bool:
        """Delete a key from a category. Returns True if existed."""
        cat = self._settings.get(category, {})
        if key in cat:
            del cat[key]
            self._save()
            return True
        return False

    def reset_category(self, category: str):
        """Reset a category to defaults."""
        if category in DEFAULTS:
            self._settings[category] = deepcopy(DEFAULTS[category])
            self._save()
            logger.info(f"Settings category '{category}' reset to defaults")

    # ── Convenience getters ───────────────────────────────────────

    def get_hustilstand_config(self) -> dict:
        """Get hustilstand sensor configuration."""
        return self.get_category('hustilstand') or DEFAULTS['hustilstand']

    def get_zone_activity_map(self) -> dict:
        """Get zone -> activity mapping for person inference."""
        return self.get('activity', 'zone_map', {})

    def get_person_config(self, slug: str) -> dict:
        """Get config overrides for a specific person."""
        return self._settings.get('persons', {}).get(slug, {})

    def get_room_config(self, area_id: str) -> dict:
        """Get config overrides for a specific room."""
        return self._settings.get('rooms', {}).get(area_id, {})

    def get_entity_selections(self, category: str = None) -> dict | list:
        """Get entity selections — all categories or a specific one."""
        selections = self._settings.get('entity_selections', {})
        if category:
            return selections.get(category, [])
        return selections

    def set_entity_selections(self, category: str, entity_ids: list):
        """Set selected entity_ids for a category and persist."""
        if 'entity_selections' not in self._settings:
            self._settings['entity_selections'] = {}
        self._settings['entity_selections'][category] = entity_ids
        self._save()
        logger.info(f"Entity selections '{category}' updated: {len(entity_ids)} entities")

    def get_dismissed_suggestions(self) -> list:
        """Get list of dismissed entity suggestion IDs."""
        return self._settings.get('entity_suggestions_dismissed', [])

    def dismiss_suggestion(self, entity_id: str):
        """Add entity_id to dismissed suggestions list."""
        dismissed = self._settings.get('entity_suggestions_dismissed', [])
        if entity_id not in dismissed:
            dismissed.append(entity_id)
            self._settings['entity_suggestions_dismissed'] = dismissed
            self._save()
