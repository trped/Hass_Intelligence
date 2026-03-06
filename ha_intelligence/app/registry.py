"""Entity and Device Registry loader for HA Intelligence.

Fetches entity and device registries from HA WebSocket API at startup,
builds entity_id -> area_id mappings, and stays updated via events.
"""

import asyncio
import json
import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Domain relevance for filtering events
DOMAIN_RELEVANCE = {
    # High - always track, critical for intelligence
    'binary_sensor': 'high',
    'person': 'high',
    'device_tracker': 'high',
    'light': 'high',
    'switch': 'high',
    'fan': 'high',
    'cover': 'high',
    'climate': 'high',
    'media_player': 'high',
    'lock': 'high',
    'alarm_control_panel': 'high',
    'input_boolean': 'high',
    'input_select': 'high',
    'input_number': 'high',
    'vacuum': 'high',
    'water_heater': 'high',

    # Medium - track but may aggregate later
    'sensor': 'medium',
    'weather': 'medium',
    'automation': 'medium',
    'script': 'medium',
    'scene': 'medium',
    'timer': 'medium',
    'counter': 'medium',
    'calendar': 'medium',
    'todo': 'medium',
    'input_datetime': 'medium',
    'input_text': 'medium',
    'group': 'medium',
    'remote': 'medium',

    # Low - skip by default (config/system entities)
    'number': 'low',
    'text': 'low',
    'select': 'low',
    'button': 'low',
    'update': 'low',
    'persistent_notification': 'low',
    'tts': 'low',
    'stt': 'low',
    'image': 'low',
    'camera': 'low',
    'conversation': 'low',
    'event': 'low',
}

# Event types to subscribe to
SUBSCRIBED_EVENT_TYPES = [
    'state_changed',
    'automation_triggered',
    'call_service',
    'entity_registry_updated',
    'device_registry_updated',
]


class Registry:
    """Loads and maintains entity and device registry mappings."""

    def __init__(self, db):
        self.db = db
        self.ha_url = os.environ.get('SUPERVISOR_API', 'http://supervisor/core')
        self.token = os.environ.get('SUPERVISOR_TOKEN', '')
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
        }

        # In-memory mappings
        self._entities = {}   # entity_id -> {area_id, device_id, device_class, platform, ...}
        self._devices = {}    # device_id -> {area_id, manufacturer, model, name, ...}
        self._area_map = {}   # entity_id -> resolved area_id

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def device_count(self) -> int:
        return len(self._devices)

    @property
    def mapped_count(self) -> int:
        """Number of entities with a resolved area_id."""
        return len([v for v in self._area_map.values() if v])

    def get_area_id(self, entity_id: str) -> Optional[str]:
        """Get resolved area_id for an entity (entity -> device fallback)."""
        return self._area_map.get(entity_id)

    def get_entity_info(self, entity_id: str) -> dict:
        """Get full registry info for an entity."""
        return self._entities.get(entity_id, {})

    def get_device_info(self, device_id: str) -> dict:
        """Get full registry info for a device."""
        return self._devices.get(device_id, {})

    def get_entities_in_area(self, area_id: str) -> list:
        """Get all entity_ids assigned to an area."""
        return [eid for eid, aid in self._area_map.items() if aid == area_id]

    def get_relevance(self, entity_id: str) -> str:
        """Get relevance level for an entity based on domain."""
        domain = entity_id.split('.')[0] if '.' in entity_id else 'unknown'
        return DOMAIN_RELEVANCE.get(domain, 'low')

    async def load_all(self):
        """Fetch entity and device registries from HA via WebSocket."""
        logger.info("Loading entity and device registries...")
        ws_url = self.ha_url.replace('http', 'ws') + '/websocket'

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    # Authenticate
                    msg = await ws.receive_json()
                    if msg.get('type') != 'auth_required':
                        logger.error(f"WS: unexpected initial message: {msg}")
                        return
                    await ws.send_json({
                        'type': 'auth',
                        'access_token': self.token
                    })
                    msg = await ws.receive_json()
                    if msg.get('type') != 'auth_ok':
                        logger.error(f"WS: auth failed: {msg}")
                        return

                    # Fetch entity registry
                    await ws.send_json({
                        'id': 1,
                        'type': 'config/entity_registry/list'
                    })
                    msg = await ws.receive_json()
                    if msg.get('success'):
                        self._parse_entity_registry(msg.get('result', []))
                    else:
                        logger.error(f"Entity registry fetch failed: {msg}")

                    # Fetch device registry
                    await ws.send_json({
                        'id': 2,
                        'type': 'config/device_registry/list'
                    })
                    msg = await ws.receive_json()
                    if msg.get('success'):
                        self._parse_device_registry(msg.get('result', []))
                    else:
                        logger.error(f"Device registry fetch failed: {msg}")

        except Exception as e:
            logger.error(f"Registry load error: {e}")

        self._build_area_map()
        self._update_db()
        logger.info(
            f"Registry loaded: {len(self._entities)} entities, "
            f"{len(self._devices)} devices, "
            f"{self.mapped_count} with area_id"
        )

    def _parse_entity_registry(self, entities: list):
        """Parse entity registry response."""
        for ent in entities:
            eid = ent.get('entity_id', '')
            if not eid:
                continue
            self._entities[eid] = {
                'area_id': ent.get('area_id'),
                'device_id': ent.get('device_id'),
                'device_class': (
                    ent.get('device_class')
                    or ent.get('original_device_class')
                ),
                'platform': ent.get('platform'),
                'labels': ent.get('labels', []),
                'disabled_by': ent.get('disabled_by'),
                'hidden_by': ent.get('hidden_by'),
                'name': ent.get('name') or ent.get('original_name'),
            }
        logger.info(f"Entity registry: {len(self._entities)} entities")

    def _parse_device_registry(self, devices: list):
        """Parse device registry response."""
        for dev in devices:
            did = dev.get('id', '')
            if not did:
                continue
            self._devices[did] = {
                'area_id': dev.get('area_id'),
                'manufacturer': dev.get('manufacturer'),
                'model': dev.get('model'),
                'name': dev.get('name_by_user') or dev.get('name'),
                'labels': dev.get('labels', []),
                'disabled_by': dev.get('disabled_by'),
            }
        logger.info(f"Device registry: {len(self._devices)} devices")

    def _build_area_map(self):
        """Build entity_id -> area_id mapping with device fallback."""
        self._area_map = {}
        for entity_id, info in self._entities.items():
            # Entity's own area_id takes priority
            area_id = info.get('area_id')
            if not area_id and info.get('device_id'):
                # Fallback to device's area_id
                device = self._devices.get(info['device_id'], {})
                area_id = device.get('area_id')
            if area_id:
                self._area_map[entity_id] = area_id

    def _update_db(self):
        """Update discovered_entities with registry data."""
        batch = []
        for entity_id, info in self._entities.items():
            area_id = self._area_map.get(entity_id)
            domain = entity_id.split('.')[0] if '.' in entity_id else 'unknown'
            batch.append({
                'entity_id': entity_id,
                'domain': domain,
                'area_id': area_id,
                'friendly_name': info.get('name'),
                'device_class': info.get('device_class'),
                'platform': info.get('platform'),
                'device_id': info.get('device_id'),
            })

        # Bulk upsert in chunks
        for i in range(0, len(batch), 100):
            chunk = batch[i:i + 100]
            self.db.upsert_entities_bulk(chunk)

    def on_entity_registry_updated(self, data: dict):
        """Handle entity_registry_updated event."""
        action = data.get('action', '')
        entity_id = data.get('entity_id', '')

        if action == 'remove' and entity_id in self._entities:
            del self._entities[entity_id]
            self._area_map.pop(entity_id, None)
            logger.debug(f"Entity removed from registry: {entity_id}")
            return

        if action in ('create', 'update') and entity_id:
            changes = data.get('changes', {})
            if entity_id in self._entities:
                self._entities[entity_id].update(changes)
            else:
                self._entities[entity_id] = changes

            # Rebuild area for this entity
            info = self._entities.get(entity_id, {})
            area_id = info.get('area_id')
            if not area_id and info.get('device_id'):
                device = self._devices.get(info['device_id'], {})
                area_id = device.get('area_id')
            if area_id:
                self._area_map[entity_id] = area_id
            else:
                self._area_map.pop(entity_id, None)

            logger.debug(f"Entity registry updated: {entity_id} -> area={area_id}")

    def on_device_registry_updated(self, data: dict):
        """Handle device_registry_updated event."""
        action = data.get('action', '')
        device_id = data.get('device_id', '')

        if not device_id:
            return

        if action == 'remove':
            self._devices.pop(device_id, None)
            return

        changes = data.get('changes', {})
        if device_id in self._devices:
            self._devices[device_id].update(changes)
        else:
            self._devices[device_id] = changes

        # Rebuild area map for all entities on this device
        new_area = self._devices.get(device_id, {}).get('area_id')
        for entity_id, info in self._entities.items():
            if info.get('device_id') == device_id and not info.get('area_id'):
                if new_area:
                    self._area_map[entity_id] = new_area
                else:
                    self._area_map.pop(entity_id, None)

        logger.debug(f"Device registry updated: {device_id} -> area={new_area}")
