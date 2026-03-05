"""Ingress web UI for HA Intelligence."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

logger = logging.getLogger(__name__)

# Ingress path from HA
INGRESS_PATH = os.environ.get('INGRESS_PATH', '')


def create_app(db, event_listener, mqtt_pub) -> FastAPI:
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
        s['mqtt_connected'] = mqtt_pub.connected
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

    @app.get("/api/events/recent")
    async def recent_events(limit: int = 20):
        return db.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?",
            (limit,), fetch=True
        )

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "version": "0.1.9",
            "ws_connected": event_listener.connected,
            "mqtt_connected": mqtt_pub.connected,
        }

    return app


def get_dashboard_html() -> str:
    return """<!DOCTYPE html>
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
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text);
    padding: 24px; min-height: 100vh;
  }
  h1 { font-size: 24px; margin-bottom: 24px; display: flex; align-items: center; gap: 10px; }
  h1 .icon { font-size: 28px; }
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
  #status-text { font-size: 13px; color: var(--text-dim); }
</style>
</head>
<body>

<h1>
  <span class="icon">&#129504;</span> HA Intelligence
  <span style="flex:1"></span>
  <button class="refresh-btn" onclick="loadAll()">Opdater</button>
</h1>

<div class="grid">
  <div class="card">
    <h2>System Status</h2>
    <div id="system-status">Indlæser...</div>
  </div>
  <div class="card">
    <h2>Events (24t)</h2>
    <div class="stat" id="events-count">-</div>
    <div class="stat-label">state_changed events indsamlet</div>
  </div>
  <div class="card">
    <h2>Opdagede Entiteter</h2>
    <div class="stat" id="entities-count">-</div>
    <div class="stat-label">unikke sensorer sporet</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Rum</h2>
    <ul class="list" id="rooms-list"><li>Indlæser...</li></ul>
  </div>
  <div class="card">
    <h2>Personer</h2>
    <ul class="list" id="persons-list"><li>Indlæser...</li></ul>
  </div>
</div>

<div class="card" style="margin-top: 0;">
  <h2>Seneste Events</h2>
  <table class="events-table">
    <thead><tr><th>Tid</th><th>Entity</th><th>Fra</th><th>Til</th></tr></thead>
    <tbody id="events-body"><tr><td colspan="4">Indlæser...</td></tr></tbody>
  </table>
</div>

<script>
// Derive ingress base from current page URL (strip trailing slashes)
const BASE = window.location.pathname.replace(/\/+$/, '');

async function fetchJson(path) {
  const res = await fetch(BASE + path);
  return res.json();
}

async function loadStats() {
  try {
    const s = await fetchJson('/api/stats');
    document.getElementById('events-count').textContent = s.events_24h.toLocaleString('da-DK');
    document.getElementById('entities-count').textContent = s.entities_discovered.toLocaleString('da-DK');
    const wsOk = s.event_listener_connected;
    const mqttOk = s.mqtt_connected;
    document.getElementById('system-status').innerHTML = `
      <p><span class="status-dot ${wsOk ? 'ok' : 'err'}"></span>Event Bus: ${wsOk ? 'Forbundet' : 'Afbrudt'}</p>
      <p style="margin-top:6px"><span class="status-dot ${mqttOk ? 'ok' : 'err'}"></span>MQTT: ${mqttOk ? 'Forbundet' : 'Afbrudt'}</p>
      <p style="margin-top:6px"><span class="status-dot ok"></span>Events i session: ${s.event_listener_count.toLocaleString('da-DK')}</p>
      <p style="margin-top:6px;color:var(--text-dim);font-size:12px">Rum: ${s.rooms} &middot; Personer: ${s.persons} &middot; Total events: ${s.events_total.toLocaleString('da-DK')}</p>
    `;
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
    if (!events.length) { el.innerHTML = '<tr><td colspan="4" style="color:var(--text-dim)">Ingen events endnu</td></tr>'; return; }
    el.innerHTML = events.map(e => {
      const t = new Date(e.recorded_at).toLocaleTimeString('da-DK');
      return `<tr>
        <td>${t}</td>
        <td class="entity">${e.entity_id}</td>
        <td class="state">${e.old_state || '-'}</td>
        <td class="state">${e.new_state}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error('Events error:', e); }
}

function loadAll() {
  loadStats(); loadRooms(); loadPersons(); loadEvents();
}

loadAll();
setInterval(loadAll, 15000);
</script>

</body>
</html>"""
