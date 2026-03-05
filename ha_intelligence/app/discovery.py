"""Discover areas (rooms) and persons from Home Assistant."""

import logging
import os
import re

import aiohttp

logger = logging.getLogger(__name__)


class Discovery:
    """Reads HA area registry and person entities on startup."""

    def __init__(self, db):
        self.db = db
        self.ha_url = os.environ.get(
            'SUPERVISOR_API', 'http://supervisor/core'
        )
        self.token = os.environ.get('SUPERVISOR_TOKEN', '')
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
        }

    def _slugify(self, name: str) -> str:
        """Convert name to slug: 'Alrum' -> 'alrum', 'Darwins Værelse' -> 'darwins_vaerelse'."""
        slug = name.lower().strip()
        # Danish characters
        slug = slug.replace('æ', 'ae').replace('ø', 'oe').replace('å', 'aa')
        slug = slug.replace('é', 'e').replace('ü', 'u')
        slug = re.sub(r'[^a-z0-9]+', '_', slug)
        slug = slug.strip('_')
        return slug

    async def discover_all(self):
        """Run full discovery of rooms and persons."""
        logger.info("Starting HA discovery...")
        await self._discover_areas()
        await self._discover_persons()
        logger.info("Discovery complete")

    async def _discover_areas(self):
        """Fetch areas from HA area registry via WebSocket-style REST."""
        try:
            async with aiohttp.ClientSession() as session:
                # Use the config API to list areas
                url = f"{self.ha_url}/api/config/area_registry/list"
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        areas = await resp.json()
                        for area in areas:
                            area_id = area.get('area_id', '')
                            name = area.get('name', '')
                            if area_id and name:
                                slug = self._slugify(name)
                                self.db.upsert_room(area_id, name, slug)
                                logger.info(f"Discovered room: {name} ({slug})")
                    else:
                        # Fallback: try WebSocket command
                        logger.warning(f"Area registry API returned {resp.status}, trying states fallback")
                        await self._discover_areas_from_states(session)
        except Exception as e:
            logger.error(f"Area discovery error: {e}")

    async def _discover_areas_from_states(self, session):
        """Fallback: discover areas from entity attributes."""
        url = f"{self.ha_url}/api/states"
        async with session.get(url, headers=self.headers) as resp:
            if resp.status == 200:
                states = await resp.json()
                seen_areas = set()
                for state in states:
                    area = state.get('attributes', {}).get('area_id')
                    if area and area not in seen_areas:
                        seen_areas.add(area)
                        slug = self._slugify(area)
                        self.db.upsert_room(area, area, slug)
                        logger.info(f"Discovered room (fallback): {area}")

    async def _discover_persons(self):
        """Fetch person entities from HA states."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ha_url}/api/states"
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to fetch states: {resp.status}")
                        return
                    states = await resp.json()

                    for state in states:
                        entity_id = state.get('entity_id', '')
                        if not entity_id.startswith('person.'):
                            continue

                        name = state.get('attributes', {}).get(
                            'friendly_name', entity_id.split('.')[1]
                        )
                        slug = self._slugify(name)
                        self.db.upsert_person(entity_id, name, slug)
                        logger.info(f"Discovered person: {name} ({slug})")

        except Exception as e:
            logger.error(f"Person discovery error: {e}")

    async def get_all_states(self) -> dict:
        """Fetch all current HA states. Returns dict of entity_id -> state obj."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ha_url}/api/states"
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        states = await resp.json()
                        return {s['entity_id']: s for s in states}
        except Exception as e:
            logger.error(f"Error fetching states: {e}")
        return {}
