"""Ingress web UI for HA Intelligence."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

logger = logging.getLogger(__name__)

# Ingress path from HA
INGRESS_PATH = os.environ.get('INGRESS_PATH', '')


def create_app(db, event_listener, mqtt_pub, registry=None, ml_engine=None) -> FastAPI:
    """Create FastAPI app with ingress support."""

    # Strip trailing slashes from ingress path to avoid double-slash issues
    root = INGRESS_PATH.rstrip('/')

    app = FastAPI(
        title="HA Intelligence",
        root_path=root,
    )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return get_dashboard_html()

    @app.get("//", response_class=HTMLResponse)
    async def index_double_slash():
        """Handle HA ingress double-slash redirect."""
        return get_dashboard_html()

    @app.get("/api/stats")
    async def stats():
        s = db.get_stats()
        s['event_listener_connected'] = event_listener.connected
        s['event_listener_count'] = event_listener.event_count
        s['event_counts_by_type'] = event_listener.event_counts_by_type
        s['filtered_count'] = event_listener.filtered_count
        s['mqtt_connected'] = mqtt_pub.connected
        if registry:
            s['registry_entities'] = registry.entity_count
            s['registry_devices'] = registry.device_count
            s['registry_mapped'] = registry.mapped_count
        return s

    @app.get("/api/rooms")
    async def rooms():
        return db.get_rooms()

    @app.get("/api/persons")
    async def persons():
        return db.get_persons()

    @app.get("/api/entities")
    async def entities(domain: str = None, limit: int = 50):
        ents = db.get_discovered_entities(domain)
        return ents[:limit]

    @app.get("/api/entities/mapped")
    async def entities_mapped():
        """Get entities with area_id assigned."""
        return db.get_entities_with_area()

    @app.get("/api/events/recent")
    async def recent_events(limit: int = 20):
        return db.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?",
            (limit,), fetch=True
        )

    @app.get("/api/events/types")
    async def event_type_stats():
        """Get event counts by type for last 24h."""
        return db.get_event_type_stats()

    @app.get("/api/registry/summary")
    async def registry_summary():
        """Get registry summary data."""
        if not registry:
            return {'error': 'Registry not loaded'}
        return {
            'entities': registry.entity_count,
            'devices': registry.device_count,
            'mapped': registry.mapped_count,
        }

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "version": "0.3.2",
            "ws_connected": event_listener.connected,
            "mqtt_connected": mqtt_pub.connected,
            "registry_loaded": registry is not None and registry.entity_count > 0,
            "ml_active": ml_engine is not None and ml_engine.get_stats().get('ml_active', False),
        }

    @app.get("/api/ml/stats")
    async def ml_stats():
        """Get ML engine statistics."""
        if not ml_engine:
            return {'error': 'ML engine not initialized', 'ml_active': False}
        stats = ml_engine.get_stats()
        return stats

    @app.get("/api/ml/models")
    async def ml_models():
        """Get per-model details."""
        if not ml_engine:
            return {'room_models': [], 'person_models': []}
        room_list = []
        for area_id, model in ml_engine.models.room_models.items():
            room_list.append({
                'area_id': area_id,
                'samples_seen': model.samples_seen,
                'active': model.samples_seen >= 50,
            })
        person_list = []
        for person_id, model in ml_engine.models.person_models.items():
            person_list.append({
                'person_id': person_id,
                'samples_seen': model.samples_seen,
                'active': model.samples_seen >= 50,
            })
        return {'room_models': room_list, 'person_models': person_list}

    @app.get("/api/predictions/recent")
    async def recent_predictions(limit: int = 20):
        """Get recent predictions from DB."""
        return db.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,), fetch=True
        )

    return app


def get_dashboard_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="da">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HA Intelligence</title>
<style>
  :root {
    --primary: #3B82F6;
    --bg: #0f172a;
    --card: #1e293b;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --green: #22c55e;
    --red: #ef4444;
    --orange: #f59e0b;
    --purple: #a855f7;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text);
    padding: 24px; min-height: 100vh;
  }
  h1 { font-size: 24px; margin-bottom: 24px; display: flex; align-items: center; gap: 10px; }
  h1 .icon { font-size: 28px; }
  h1 .version { font-size: 12px; color: var(--text-dim); background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 6px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card {
    background: var(--card); border-radius: 12px; padding: 20px;
    border: 1px solid rgba(255,255,255,0.05);
  }
  .card h2 { font-size: 14px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
  .stat { font-size: 36px; font-weight: 700; }
  .stat-label { font-size: 13px; color: var(--text-dim); margin-top: 4px; }
  .status-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 6px;
  }
  .status-dot.ok { background: var(--green); }
  .status-dot.err { background: var(--red); }
  .list { list-style: none; }
  .list li {
    padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex; justify-content: space-between; align-items: center;
  }
  .list li:last-child { border-bottom: none; }
  .badge {
    background: rgba(59,130,246,0.15); color: var(--primary);
    padding: 2px 8px; border-radius: 6px; font-size: 12px;
  }
  .badge.purple { background: rgba(168,85,247,0.15); color: var(--purple); }
  .badge.green { background: rgba(34,197,94,0.15); color: var(--green); }
  .badge.orange { background: rgba(245,158,11,0.15); color: var(--orange); }
  .events-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .events-table th { text-align: left; color: var(--text-dim); padding: 8px 12px; font-weight: 500; }
  .events-table td { padding: 8px 12px; border-top: 1px solid rgba(255,255,255,0.05); }
  .events-table .entity { color: var(--primary); }
  .events-table .state { font-family: monospace; }
  .refresh-btn {
    background: var(--primary); color: white; border: none; padding: 8px 16px;
    border-radius: 8px; cursor: pointer; font-size: 13px; float: right;
  }
  .refresh-btn:hover { opacity: 0.9; }
  .mini-stat { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; }
  .mini-stat .item { font-size: 13px; }
  .mini-stat .item .val { font-weight: 600; }
  .mini-stat .item .lbl { color: var(--text-dim); }
</style>
</head>
<body>

<h1>
  <span class="icon">&#129504;</span> HA Intelligence
  <span class="version">v0.3.2</span>
  <span style="flex:1"></span>
  <button class="refresh-btn" onclick="loadAll()">Opdater</button>
</h1>

<div class="grid">
  <div class="card">
    <h2>System Status</h2>
    <div id="system-status">Indlaeser...</div>
  </div>
  <div class="card">
    <h2>Registry</h2>
    <div id="registry-info">Indlaeser...</div>
  </div>
  <div class="card">
    <h2>Events (24t)</h2>
    <div class="stat" id="events-count">-</div>
    <div class="stat-label">events indsamlet</div>
    <div id="event-types" class="mini-stat" style="margin-top:12px"></div>
  </div>
  <div class="card">
    <h2>ML Status</h2>
    <div id="ml-info">Indlaeser...</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Rum</h2>
    <ul class="list" id="rooms-list"><li>Indlaeser...</li></ul>
  </div>
  <div class="card">
    <h2>Personer</h2>
    <ul class="list" id="persons-list"><li>Indlaeser...</li></ul>
  </div>
</div>

<div class="card" style="margin-top: 0;">
  <h2>Seneste Events</h2>
  <table class="events-table">
    <thead><tr><th>Tid</th><th>Type</th><th>Entity</th><th>Fra</th><th>Til</th></tr></thead>
    <tbody id="events-body"><tr><td colspan="5">Indlaeser...</td></tr></tbody>
  </table>
</div>

<script>
const BASE = window.location.pathname.replace(/\/+$/, '');

async function fetchJson(path) {
  const res = await fetch(BASE + path);
  return res.json();
}

async function loadStats() {
  try {
    const s = await fetchJson('/api/stats');
    document.getElementById('events-count').textContent = s.events_24h.toLocaleString('da-DK');

    const wsOk = s.event_listener_connected;
    const mqttOk = s.mqtt_connected;
    const regLoaded = s.registry_entities > 0;
    document.getElementById('system-status').innerHTML = `
      <p><span class="status-dot ${wsOk ? 'ok' : 'err'}"></span>Event Bus: ${wsOk ? 'Forbundet' : 'Afbrudt'}</p>
      <p style="margin-top:6px"><span class="status-dot ${mqttOk ? 'ok' : 'err'}"></span>MQTT: ${mqttOk ? 'Forbundet' : 'Afbrudt'}</p>
      <p style="margin-top:6px"><span class="status-dot ${regLoaded ? 'ok' : 'err'}"></span>Registry: ${regLoaded ? 'Loaded' : 'Mangler'}</p>
      <p style="margin-top:6px;color:var(--text-dim);font-size:12px">
        Events i session: ${s.event_listener_count.toLocaleString('da-DK')}
        &middot; Filtreret: ${(s.filtered_count||0).toLocaleString('da-DK')}
      </p>
    `;

    // Registry card
    if (s.registry_entities) {
      document.getElementById('registry-info').innerHTML = `
        <div class="stat">${s.registry_mapped.toLocaleString('da-DK')}</div>
        <div class="stat-label">entiteter med rum-mapping</div>
        <div class="mini-stat">
          <div class="item"><span class="val">${s.registry_entities.toLocaleString('da-DK')}</span> <span class="lbl">entiteter</span></div>
          <div class="item"><span class="val">${s.registry_devices.toLocaleString('da-DK')}</span> <span class="lbl">enheder</span></div>
          <div class="item"><span class="val">${s.entities_discovered.toLocaleString('da-DK')}</span> <span class="lbl">sporet</span></div>
        </div>
      `;
    }

    // Event type breakdown
    const types = s.event_counts_by_type || {};
    const typeHtml = Object.entries(types).map(([t, c]) =>
      `<div class="item"><span class="val">${c.toLocaleString('da-DK')}</span> <span class="lbl">${t.replace('_', ' ')}</span></div>`
    ).join('');
    document.getElementById('event-types').innerHTML = typeHtml;

  } catch(e) { console.error('Stats error:', e); }
}

async function loadRooms() {
  try {
    const rooms = await fetchJson('/api/rooms');
    const el = document.getElementById('rooms-list');
    if (!rooms.length) { el.innerHTML = '<li style="color:var(--text-dim)">Ingen rum opdaget endnu</li>'; return; }
    el.innerHTML = rooms.map(r =>
      `<li><span>${r.name}</span><span class="badge">${r.slug}</span></li>`
    ).join('');
  } catch(e) { console.error('Rooms error:', e); }
}

async function loadPersons() {
  try {
    const persons = await fetchJson('/api/persons');
    const el = document.getElementById('persons-list');
    if (!persons.length) { el.innerHTML = '<li style="color:var(--text-dim)">Ingen personer opdaget endnu</li>'; return; }
    el.innerHTML = persons.map(p =>
      `<li><span>${p.name}</span><span class="badge">${p.entity_id}</span></li>`
    ).join('');
  } catch(e) { console.error('Persons error:', e); }
}

async function loadEvents() {
  try {
    const events = await fetchJson('/api/events/recent?limit=15');
    const el = document.getElementById('events-body');
    if (!events.length) { el.innerHTML = '<tr><td colspan="5" style="color:var(--text-dim)">Ingen events endnu</td></tr>'; return; }
    el.innerHTML = events.map(e => {
      const t = new Date(e.recorded_at).toLocaleTimeString('da-DK');
      const etype = (e.event_type || 'state_changed').replace('_', ' ');
      const badgeClass = e.event_type === 'automation_triggered' ? 'purple'
        : e.event_type === 'call_service' ? 'orange' : '';
      return `<tr>
        <td>${t}</td>
        <td><span class="badge ${badgeClass}">${etype}</span></td>
        <td class="entity">${e.entity_id}</td>
        <td class="state">${e.old_state || '-'}</td>
        <td class="state">${e.new_state}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error('Events error:', e); }
}

async function loadML() {
  try {
    const ml = await fetchJson('/api/ml/stats');
    const models = await fetchJson('/api/ml/models');
    const el = document.getElementById('ml-info');
    const active = ml.ml_active;
    const statusBadge = active
      ? '<span class="badge green">Aktiv</span>'
      : '<span class="badge orange">Laerer</span>';
    el.innerHTML = `
      <p>${statusBadge} ${ml.total_samples || 0} samples (threshold: ${ml.ml_threshold || 50})</p>
      <div class="mini-stat" style="margin-top:8px">
        <div class="item"><span class="val">${ml.room_models || 0}</span> <span class="lbl">rum-modeller</span></div>
        <div class="item"><span class="val">${ml.person_models || 0}</span> <span class="lbl">person-modeller</span></div>
      </div>
      ${(models.room_models||[]).length ? '<p style="margin-top:8px;font-size:12px;color:var(--text-dim)">Rum: ' +
        models.room_models.map(m => m.area_id + ' (' + m.samples_seen + ')').join(', ') + '</p>' : ''}
      ${(models.person_models||[]).length ? '<p style="margin-top:4px;font-size:12px;color:var(--text-dim)">Person: ' +
        models.person_models.map(m => m.person_id.replace('person.','') + ' (' + m.samples_seen + ')').join(', ') + '</p>' : ''}
    `;
  } catch(e) { console.error('ML error:', e); }
}

function loadAll() {
  loadStats(); loadRooms(); loadPersons(); loadEvents(); loadML();
}

loadAll();
setInterval(loadAll, 15000);
</script>

</body>
</html>"""
