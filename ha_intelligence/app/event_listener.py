"""Home Assistant Event Bus listener via WebSocket.

Subscribes to multiple event types and filters by domain relevance.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import aiohttp

from registry import DOMAIN_RELEVANCE, SUBSCRIBED_EVENT_TYPES

logger = logging.getLogger(__name__)


class EventListener:
    """Connects to HA WebSocket API and listens for multiple event types."""

    def __init__(self, db, registry=None, on_state_change=None):
        self.db = db
        self.registry = registry
        self.on_state_change = on_state_change
        self._running = False
        self._ws = None
        self._msg_id = 0
        self._event_counts = {}  # event_type -> count
        self._filtered_count = 0

        # Inside add-on: use Supervisor API
        self.ha_url = os.environ.get(
            'SUPERVISOR_API', 'http://supervisor/core'
        )
        self.token = os.environ.get('SUPERVISOR_TOKEN', '')
        self.ws_url = self.ha_url.replace('http', 'ws') + '/websocket'
        logger.info(f"WS URL: {self.ws_url}, token present: {bool(self.token)}")

    @property
    def event_count(self) -> int:
        return sum(self._event_counts.values())

    @property
    def event_counts_by_type(self) -> dict:
        return dict(self._event_counts)

    @property
    def filtered_count(self) -> int:
        return self._filtered_count

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def start(self):
        """Start listening. Reconnects on failure."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                logger.error(f"Event listener error: {e}")
                if self._running:
                    logger.info("Reconnecting in 10s...")
                    await asyncio.sleep(10)

    async def stop(self):
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _connect_and_listen(self):
        logger.info(f"Connecting to HA WebSocket at {self.ws_url}")

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.ws_url) as ws:
                self._ws = ws

                # Step 1: Wait for auth_required
                msg = await ws.receive_json()
                if msg.get('type') != 'auth_required':
                    logger.error(f"Unexpected initial message: {msg}")
                    return

                # Step 2: Authenticate
                await ws.send_json({
                    'type': 'auth',
                    'access_token': self.token
                })
                msg = await ws.receive_json()
                if msg.get('type') != 'auth_ok':
                    logger.error(f"Auth failed: {msg}")
                    await asyncio.sleep(30)
                    return
                logger.info("WebSocket authenticated successfully")

                # Step 3: Subscribe to multiple event types
                for event_type in SUBSCRIBED_EVENT_TYPES:
                    sub_id = self._next_id()
                    await ws.send_json({
                        'id': sub_id,
                        'type': 'subscribe_events',
                        'event_type': event_type
                    })
                    msg = await ws.receive_json()
                    if msg.get('success'):
                        logger.info(f"Subscribed to {event_type}")
                    else:
                        logger.warning(f"Failed to subscribe to {event_type}: {msg}")

                # Step 4: Listen for events
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get('type') == 'event':
                            await self._route_event(data['event'])
                    elif msg.type in (aiohttp.WSMsgType.ERROR,
                                      aiohttp.WSMsgType.CLOSED):
                        logger.warning("WebSocket closed")
                        break

    async def _route_event(self, event: dict):
        """Route event to the appropriate handler based on event_type."""
        event_type = event.get('event_type', '')

        if event_type == 'state_changed':
            await self._handle_state_changed(event)
        elif event_type == 'automation_triggered':
            await self._handle_automation_triggered(event)
        elif event_type == 'call_service':
            await self._handle_call_service(event)
        elif event_type == 'entity_registry_updated':
            self._handle_registry_event(event, 'entity')
        elif event_type == 'device_registry_updated':
            self._handle_registry_event(event, 'device')

    async def _handle_state_changed(self, event: dict):
        """Process a state_changed event with relevance filtering."""
        try:
            event_data = event.get('data', {})
            entity_id = event_data.get('entity_id', '')
            new_state_obj = event_data.get('new_state', {})
            old_state_obj = event_data.get('old_state', {})

            if not entity_id or not new_state_obj:
                return

            # Filter by domain relevance
            relevance = self._get_relevance(entity_id)
            if relevance == 'low':
                self._filtered_count += 1
                return

            new_state = new_state_obj.get('state', '')
            old_state = old_state_obj.get('state', '') if old_state_obj else ''
            attributes = new_state_obj.get('attributes', {})
            timestamp = new_state_obj.get('last_changed',
                                          datetime.now(timezone.utc).isoformat())

            # Skip unavailable/unknown transitions
            if new_state in ('unavailable', 'unknown'):
                return

            # For medium relevance, skip unchanged states (attribute-only updates)
            if relevance == 'medium' and old_state == new_state:
                self._filtered_count += 1
                return

            # Store event
            self.db.insert_event(entity_id, old_state, new_state,
                                 attributes, timestamp, 'state_changed')
            self._event_counts['state_changed'] = (
                self._event_counts.get('state_changed', 0) + 1
            )

            # Track entity discovery with registry data
            domain = entity_id.split('.')[0] if '.' in entity_id else 'unknown'
            friendly_name = attributes.get('friendly_name')
            area_id = None
            device_class = None
            platform = None
            device_id = None

            if self.registry:
                area_id = self.registry.get_area_id(entity_id)
                info = self.registry.get_entity_info(entity_id)
                device_class = info.get('device_class')
                platform = info.get('platform')
                device_id = info.get('device_id')

            self.db.upsert_entity(
                entity_id, domain, area_id=area_id,
                friendly_name=friendly_name, device_class=device_class,
                platform=platform, device_id=device_id
            )

            # Callback for real-time processing
            if self.on_state_change:
                await self.on_state_change(
                    entity_id, old_state, new_state, attributes
                )

        except Exception as e:
            logger.error(f"Error handling state_changed: {e}")

    async def _handle_automation_triggered(self, event: dict):
        """Process automation_triggered event."""
        try:
            data = event.get('data', {})
            entity_id = data.get('entity_id', '')
            name = data.get('name', '')
            source = data.get('source', '')

            if not entity_id:
                return

            self.db.insert_event(
                entity_id=entity_id,
                old_state='',
                new_state='triggered',
                attributes={'name': name, 'source': source},
                event_type='automation_triggered'
            )
            self._event_counts['automation_triggered'] = (
                self._event_counts.get('automation_triggered', 0) + 1
            )
        except Exception as e:
            logger.error(f"Error handling automation_triggered: {e}")

    async def _handle_call_service(self, event: dict):
        """Process call_service event (only track relevant domains)."""
        try:
            data = event.get('data', {})
            domain = data.get('domain', '')
            service = data.get('service', '')
            service_data = data.get('service_data', {})

            # Only track service calls to relevant domains
            relevance = DOMAIN_RELEVANCE.get(domain, 'low')
            if relevance == 'low':
                self._filtered_count += 1
                return

            entity_id = service_data.get('entity_id', f'{domain}.{service}')
            # Handle list of entity_ids
            if isinstance(entity_id, list):
                entity_id = entity_id[0] if entity_id else f'{domain}.{service}'

            self.db.insert_event(
                entity_id=entity_id,
                old_state='',
                new_state=service,
                attributes={'domain': domain, 'service': service},
                event_type='call_service'
            )
            self._event_counts['call_service'] = (
                self._event_counts.get('call_service', 0) + 1
            )
        except Exception as e:
            logger.error(f"Error handling call_service: {e}")

    def _handle_registry_event(self, event: dict, registry_type: str):
        """Forward registry events to Registry for live updates."""
        if not self.registry:
            return
        try:
            data = event.get('data', {})
            if registry_type == 'entity':
                self.registry.on_entity_registry_updated(data)
            elif registry_type == 'device':
                self.registry.on_device_registry_updated(data)

            event_key = f'{registry_type}_registry_updated'
            self._event_counts[event_key] = (
                self._event_counts.get(event_key, 0) + 1
            )
        except Exception as e:
            logger.error(f"Error handling {registry_type}_registry_updated: {e}")

    def _get_relevance(self, entity_id: str) -> str:
        """Get relevance level for an entity."""
        if self.registry:
            return self.registry.get_relevance(entity_id)
        domain = entity_id.split('.')[0] if '.' in entity_id else 'unknown'
        return DOMAIN_RELEVANCE.get(domain, 'low')
