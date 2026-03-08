"""Entity category definitions for the entity-picker UI."""

import re

# Each category defines:
#   key: settings.json key in entity_selections
#   label: Danish display name
#   icon: MDI icon name
#   domains: list of entity domain prefixes to match
#   device_classes: list of device_class values to match (optional)
#   pattern: regex pattern for entity_id matching (optional)
#   exclude_pattern: regex pattern to exclude (optional)

CATEGORIES = [
    {
        'key': 'persons',
        'label': 'Person',
        'icon': 'mdi:account',
        'domains': ['person'],
        'device_classes': [],
        'pattern': None,
        'exclude_pattern': None,
    },
    {
        'key': 'presence',
        'label': 'Tilstedeværelse/Bevægelse',
        'icon': 'mdi:motion-sensor',
        'domains': ['binary_sensor'],
        'device_classes': ['occupancy', 'motion', 'presence'],
        'pattern': None,
        'exclude_pattern': None,
    },
    {
        'key': 'ble_tracking',
        'label': 'BLE Tracking',
        'icon': 'mdi:bluetooth',
        'domains': ['sensor'],
        'device_classes': [],
        'pattern': r'bermuda.*distance|bermuda.*area',
        'exclude_pattern': None,
    },
    {
        'key': 'lights',
        'label': 'Lys',
        'icon': 'mdi:lightbulb',
        'domains': ['light'],
        'device_classes': [],
        'pattern': None,
        'exclude_pattern': None,
    },
    {
        'key': 'climate',
        'label': 'Klima',
        'icon': 'mdi:thermometer',
        'domains': ['climate', 'sensor'],
        'device_classes': ['temperature', 'humidity'],
        'pattern': None,
        'exclude_pattern': None,
    },
    {
        'key': 'media',
        'label': 'Medier',
        'icon': 'mdi:television',
        'domains': ['media_player'],
        'device_classes': [],
        'pattern': None,
        'exclude_pattern': r'_group$',
    },
    {
        'key': 'energy',
        'label': 'Energi',
        'icon': 'mdi:flash',
        'domains': ['sensor'],
        'device_classes': ['power', 'energy', 'voltage', 'current'],
        'pattern': r'strom_priser',
        'exclude_pattern': None,
    },
    {
        'key': 'appliances',
        'label': 'Apparater',
        'icon': 'mdi:power-plug',
        'domains': ['switch', 'sensor'],
        'device_classes': ['power'],
        'pattern': None,
        'exclude_pattern': None,
    },
    {
        'key': 'security',
        'label': 'Sikkerhed',
        'icon': 'mdi:shield-home',
        'domains': ['alarm_control_panel', 'lock', 'binary_sensor'],
        'device_classes': ['door', 'window', 'opening'],
        'pattern': None,
        'exclude_pattern': None,
    },
    {
        'key': 'calendars',
        'label': 'Kalendere',
        'icon': 'mdi:calendar',
        'domains': ['calendar'],
        'device_classes': [],
        'pattern': None,
        'exclude_pattern': None,
    },
    {
        'key': 'cameras',
        'label': 'Kameraer',
        'icon': 'mdi:cctv',
        'domains': ['camera'],
        'device_classes': [],
        'pattern': None,
        'exclude_pattern': None,
    },
]

CATEGORY_MAP = {c['key']: c for c in CATEGORIES}


def get_category_keys() -> list[str]:
    """Return list of category keys in display order."""
    return [c['key'] for c in CATEGORIES]


def matches_category(entity_id: str, device_class: str | None,
                     category: dict) -> bool:
    """Check if an entity matches a category's filters.

    For sensor domain with device_classes defined:
      entity must have matching device_class.
    For other domains:
      domain match is sufficient (unless pattern/exclude specified).
    """
    domain = entity_id.split('.')[0]

    # Check domain match
    if domain not in category['domains']:
        # Check pattern match (e.g. bermuda or strom_priser)
        if category.get('pattern'):
            if re.search(category['pattern'], entity_id, re.IGNORECASE):
                return True
        return False

    # If category has device_classes, entity must match one
    if category['device_classes']:
        if domain in ('sensor', 'binary_sensor'):
            if device_class not in category['device_classes']:
                # Still allow pattern-matched entities
                if category.get('pattern') and re.search(
                    category['pattern'], entity_id, re.IGNORECASE
                ):
                    return True
                return False

    # Check exclude pattern
    if category.get('exclude_pattern'):
        if re.search(category['exclude_pattern'], entity_id, re.IGNORECASE):
            return False

    return True
