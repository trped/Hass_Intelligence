"""Home Assistant Event Bus listener via WebSocket."""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import aiohttp

logger = logging.getLogger(__name__)


class EventListener:
    """Connects to HA WebSocket API and listens for state_changed events."""

    def __init__(self, db, on_state_change=None):
        self.db = db
        self.on_state_change = on_state_change
        self._running = False
        self._ws = None
        self._msg_id = 0
        self._event_count = 0

        # Inside add-on: use Supervisor API
        self.ha_url = os.environ.get(
            'SUPERVISOR_API', 'http://supervisor/core'
        )
        self.token = os.environ.get('SUPERVISOR_TOKEN', '')
        self.ws_url = self.ha_url.replace('http', 'ws') + '/websocket'

    @property
    def event_count(self) -> int:
        return self._event_count

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
                    return
                logger.info("WebSocket authenticated successfully")

                # Step 3: Subscribe to state_changed events
                sub_id = self._next_id()
                await ws.send_json({
                    'id': sub_id,
                    'type': 'subscribe_events',
                    'event_type': 'state_changed'
                })
                msg = await ws.receive_json()
                if not msg.get('success'):
                    logger.error(f"Subscribe failed: {msg}")
                    return
                logger.info("Subscribed to state_changed events")

                # Step 4: Listen for events
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get('type') == 'event':
                            await self._handle_event(data['event'])
                    elif msg.type in (aiohttp.WSMsgType.ERROR,
                                      aiohttp.WSMsgType.CLOSED):
                        logger.warning("WebSocket closed")
                        break

    async def _handle_event(self, event: dict):
        """Process a state_changed event."""
        try:
            event_data = event.get('data', {})
            entity_id = event_data.get('entity_id', '')
            new_state_obj = event_data.get('new_state', {})
            old_state_obj = event_data.get('old_state', {})

            if not entity_id or not new_state_obj:
                return

            new_state = new_state_obj.get('state', '')
            old_state = old_state_obj.get('state', '') if old_state_obj else ''
            attributes = new_state_obj.get('attributes', {})
            timestamp = new_state_obj.get('last_changed',
                                          datetime.now(timezone.utc).isoformat())

            # Skip unavailable/unknown transitions
            if new_state in ('unavailable', 'unknown'):
                return

            # Store event
            self.db.insert_event(entity_id, old_state, new_state,
                                 attributes, timestamp)
            self._event_count += 1

            # Track entity discovery
            domain = entity_id.split('.')[0] if '.' in entity_id else 'unknown'
            friendly_name = attributes.get('friendly_name')
            self.db.upsert_entity(entity_id, domain,
                                  friendly_name=friendly_name)

            # Callback for real-time processing
            if self.on_state_change:
                await self.on_state_change(
                    entity_id, old_state, new_state, attributes
                )

        except Exception as e:
            logger.error(f"Error handling event: {e}")
