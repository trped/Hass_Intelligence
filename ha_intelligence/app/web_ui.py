"""Ingress web UI for HA Intelligence."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

logger = logging.getLogger(__name__)

# Ingress path from HA
INGRESS_PATH = os.environ.get('INGRESS_PATH', '')


def create_app(db, event_listener, mqtt_pub, registry=None, ml_engine=None,
               notification_engine=None, feedback_engine=None,
               activity_engine=None) -> FastAPI:
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

    @app.get("/api/notifications")
    async def notifications():
        if notification_engine:
            return notification_engine.get_status()
        return {'active': False, 'sent_today': 0, 'history': []}

    @app.get("/api/rooms")
    async def rooms():
        return db.get_rooms(enabled_only=False)

    @app.get("/api/persons")
    async def persons():
        return db.get_persons(enabled_only=False)

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
            "version": "0.7.3",
            "ws_connected": event_listener.connected,
            "mqtt_connected": mqtt_pub.connected,
            "registry_loaded": registry is not None and registry.entity_count > 0,
            "ml_active": ml_engine is not None and ml_engine.get_stats().get('ml_active', False),
        }

    @app.get("/api/ml/stats")
    async def ml_stats():
        """Get ML engine statistics including prediction accuracy."""
        if not ml_engine:
            return {'error': 'ML engine not initialized', 'ml_active': False}
        stats = ml_engine.get_stats()
        # Add prediction accuracy from DB
        try:
            accuracy = db.get_prediction_accuracy()
            stats['accuracy'] = accuracy
        except Exception:
            stats['accuracy'] = {'total': 0, 'correct': 0, 'accuracy_pct': 0.0}
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
        markov_list = []
        for person_id, model in ml_engine.models.markov_models.items():
            markov_list.append({
                'person_id': person_id,
                'total_transitions': model.total_transitions,
                'rooms_tracked': len(model.transitions),
            })
        anomaly_list = []
        for area_id, model in ml_engine.models.anomaly_models.items():
            stats = model.get_stats()
            anomaly_list.append({
                'area_id': area_id,
                'samples_seen': model.samples_seen,
                'anomalies_detected': model.anomalies_detected,
                'ready': stats.get('ready', False),
            })
        return {
            'room_models': room_list,
            'person_models': person_list,
            'markov_models': markov_list,
            'anomaly_models': anomaly_list,
        }

    @app.get("/api/ml/markov/{person_id}")
    async def ml_markov(person_id: str):
        """Get Markov movement predictions for a person."""
        if not ml_engine:
            return {'error': 'ML engine not initialized'}
        entity = f'person.{person_id}' if not person_id.startswith('person.') else person_id
        markov = ml_engine.models.markov_models.get(entity)
        if not markov:
            return {'error': f'No Markov model for {entity}', 'predictions': []}
        # Get current room from ML engine
        best_room = ml_engine.get_person_best_room(entity)
        current_room = best_room['room'] if best_room else None
        predictions = []
        if current_room:
            result = ml_engine.predict_next_room(entity, current_room)
            if result:
                predictions = [{'room': r, 'probability': round(p, 3)} for r, p in result]
        return {
            'person_id': entity,
            'current_room': current_room,
            'total_transitions': markov.total_transitions,
            'predictions': predictions,
        }

    @app.get("/api/ml/anomaly/{area_id}")
    async def ml_anomaly(area_id: str):
        """Get anomaly detection stats for a room."""
        if not ml_engine:
            return {'error': 'ML engine not initialized'}
        score_info = ml_engine.get_anomaly_score(area_id)
        if not score_info:
            return {'error': f'No anomaly model for {area_id}'}
        return {'area_id': area_id, **score_info}

    @app.get("/api/ml/batch")
    async def ml_batch():
        """Get batch training status and results."""
        if not ml_engine or not ml_engine.models.batch_trainer:
            return {'error': 'Batch trainer not available'}
        return ml_engine.models.batch_trainer.get_stats()

    @app.get("/api/predictions/recent")
    async def recent_predictions(limit: int = 20):
        """Get recent predictions from DB."""
        return db.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,), fetch=True
        )

    @app.get("/api/priors")
    async def priors_overview():
        """Get prior calculation status and targets."""
        if not ml_engine:
            return {'error': 'ML engine not initialized', 'targets': []}
        targets = ml_engine.priors.get_all_targets()
        return {
            'targets': targets,
            'total_targets': len(targets),
            'last_run': (
                ml_engine.priors._last_run.isoformat()
                if ml_engine.priors._last_run else None
            ),
        }

    @app.get("/api/priors/heatmap")
    async def priors_heatmap(target_type: str = 'room',
                              target_id: str = '',
                              state: str = 'occupied'):
        """Get 24x7 heatmap data for a target/state."""
        if not ml_engine:
            return {'error': 'ML engine not initialized', 'heatmap': []}
        if not target_id:
            return {'error': 'target_id required', 'heatmap': []}
        heatmap = ml_engine.priors.get_heatmap(target_type, target_id, state)
        return {
            'target_type': target_type,
            'target_id': target_id,
            'state': state,
            'heatmap': heatmap,
        }

    @app.get("/api/priors/current")
    async def priors_current(target_type: str = 'room',
                              target_id: str = ''):
        """Get current prior probability for a target."""
        if not ml_engine:
            return {'error': 'ML engine not initialized'}
        if not target_id:
            return {'error': 'target_id required'}
        return ml_engine.priors.get_prior(target_type, target_id)

    # --- Toggle endpoints ---

    @app.post("/api/rooms/{slug}/toggle")
    async def toggle_room(slug: str):
        rooms = db.get_rooms(enabled_only=False)
        room = next((r for r in rooms if r['slug'] == slug), None)
        if not room:
            return {"error": "Room not found"}
        new_state = not bool(room.get('enabled', 1))
        db.set_room_enabled(slug, new_state)
        if not new_state:
            mqtt_pub.remove_sensor(f"room_{slug}")
        return {"slug": slug, "enabled": new_state}

    @app.post("/api/persons/{slug}/toggle")
    async def toggle_person(slug: str):
        persons = db.get_persons(enabled_only=False)
        person = next((p for p in persons if p['slug'] == slug), None)
        if not person:
            return {"error": "Person not found"}
        new_state = not bool(person.get('enabled', 1))
        db.set_person_enabled(slug, new_state)
        if not new_state:
            mqtt_pub.remove_sensor(f"person_{slug}")
            mqtt_pub.remove_sensor(f"activity_{slug}")
        return {"slug": slug, "enabled": new_state}

    # --- Feedback endpoints ---

    @app.get("/api/feedback/pending")
    async def get_pending_feedback():
        return db.get_pending_questions()

    @app.post("/api/feedback/{question_id}")
    async def answer_feedback(question_id: int, request: Request):
        body = await request.json()
        answer = body.get('answer', '')
        if not answer:
            return {"error": "Missing answer"}
        if feedback_engine:
            feedback_engine._process_answer(question_id, answer)
        return {"ok": True, "question_id": question_id, "answer": answer}

    @app.get("/api/feedback/stats")
    async def get_feedback_stats():
        if feedback_engine:
            return feedback_engine.get_status()
        return {"active": False}

    # --- Activity endpoints ---

    @app.get("/api/activities/current")
    async def get_current_activities():
        if activity_engine:
            return activity_engine.get_current_activities()
        return {}

    @app.get("/api/activities/learned")
    async def get_learned_activities():
        return db.get_learned_activities()

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
  .toggle-row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.05); }
  .toggle-row:last-child { border-bottom:none; }
  .toggle-switch { position:relative; width:44px; height:24px; }
  .toggle-switch input { opacity:0; width:0; height:0; }
  .toggle-slider { position:absolute; cursor:pointer; top:0; left:0; right:0; bottom:0; background:#475569; border-radius:24px; transition:.3s; }
  .toggle-slider:before { position:absolute; content:""; height:18px; width:18px; left:3px; bottom:3px; background:white; border-radius:50%; transition:.3s; }
  .toggle-switch input:checked + .toggle-slider { background:var(--primary); }
  .toggle-switch input:checked + .toggle-slider:before { transform:translateX(20px); }
  .feedback-q { background:rgba(59,130,246,0.1); border-radius:8px; padding:12px; margin:8px 0; }
  .feedback-q button { margin:4px 4px 0 0; padding:4px 12px; border:1px solid var(--primary); background:transparent; color:var(--primary); border-radius:6px; cursor:pointer; }
  .feedback-q button:hover { background:var(--primary); color:white; }
</style>
</head>
<body>

<h1>
  <span class="icon">&#129504;</span> HA Intelligence
  <span class="version">v0.8.0</span>
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
  <div class="card">
    <h2>State Priors</h2>
    <div id="priors-info">Indlaeser...</div>
  </div>
  <div class="card">
    <h2>Notifikationer</h2>
    <div id="notify-info">Indlaeser...</div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Feedback System</h2>
    <div id="feedback-status">Indlaeser...</div>
    <h3 style="font-size:13px;color:var(--text-dim);margin-top:12px;margin-bottom:8px">Ventende</h3>
    <ul class="list" id="feedback-pending"><li>Indlaeser...</li></ul>
  </div>
  <div class="card">
    <h2>Aktiviteter</h2>
    <ul class="list" id="activities-current"><li>Indlaeser...</li></ul>
  </div>
  <div class="card">
    <h2>Rum</h2>
    <ul class="list" id="rooms-admin"><li>Indlaeser...</li></ul>
  </div>
  <div class="card">
    <h2>Personer</h2>
    <ul class="list" id="persons-admin"><li>Indlaeser...</li></ul>
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
    const el = document.getElementById('rooms-admin');
    if (!rooms.length) { el.innerHTML = '<li style="color:var(--text-dim)">Ingen rum opdaget endnu</li>'; return; }
    el.innerHTML = rooms.map(r => `
        <li class="toggle-row">
            <span>${r.name} <span class="badge">${r.slug}</span></span>
            <label class="toggle-switch">
                <input type="checkbox" ${r.enabled ? 'checked' : ''}
                    onchange="toggleRoom('${r.slug}')">
                <span class="toggle-slider"></span>
            </label>
        </li>`).join('');
  } catch(e) { console.error('Rooms error:', e); }
}

async function loadPersons() {
  try {
    const persons = await fetchJson('/api/persons');
    const el = document.getElementById('persons-admin');
    if (!persons.length) { el.innerHTML = '<li style="color:var(--text-dim)">Ingen personer opdaget endnu</li>'; return; }
    el.innerHTML = persons.map(p => `
        <li class="toggle-row">
            <span>${p.name} <span class="badge">${p.slug}</span></span>
            <label class="toggle-switch">
                <input type="checkbox" ${p.enabled ? 'checked' : ''}
                    onchange="togglePerson('${p.slug}')">
                <span class="toggle-slider"></span>
            </label>
        </li>`).join('');
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
    const acc = ml.accuracy || {};
    const accPct = ((acc.accuracy || 0) * 100).toFixed(1);
    const accTotal = acc.total || 0;
    const accCorrect = acc.correct || 0;
    const accBadge = accTotal > 0
      ? `<span class="badge ${accPct >= 70 ? 'green' : accPct >= 40 ? 'orange' : ''}">${accPct}%</span>`
      : '<span class="badge">-</span>';
    const markovCount = ml.markov_models || 0;
    const anomalyCount = ml.anomaly_models || 0;
    const totalTransitions = ml.total_transitions || 0;
    const totalAnomalies = ml.total_anomalies || 0;
    const batch = ml.batch || {};
    const batchTrained = batch.last_trained ? new Date(batch.last_trained).toLocaleString('da-DK') : null;
    el.innerHTML = `
      <p>${statusBadge} ${ml.total_samples || 0} samples (threshold: ${ml.ml_threshold || 50})</p>
      <div class="mini-stat" style="margin-top:8px">
        <div class="item"><span class="val">${ml.room_models || 0}</span> <span class="lbl">rum</span></div>
        <div class="item"><span class="val">${ml.person_models || 0}</span> <span class="lbl">person</span></div>
        <div class="item"><span class="val">${markovCount}</span> <span class="lbl">markov</span></div>
        <div class="item"><span class="val">${anomalyCount}</span> <span class="lbl">anomaly</span></div>
      </div>
      <p style="margin-top:10px">Accuracy: ${accBadge} <span style="font-size:12px;color:var(--text-dim)">${accCorrect}/${accTotal} predictions</span></p>
      <div class="mini-stat" style="margin-top:6px">
        <div class="item"><span class="val">${totalTransitions}</span> <span class="lbl">transitions</span></div>
        <div class="item"><span class="val">${totalAnomalies}</span> <span class="lbl">anomalier</span></div>
      </div>
      ${batchTrained ? '<p style="margin-top:6px;font-size:11px;color:var(--text-dim)">Batch: ' + batchTrained + ' (' + (batch.room_models||0) + ' rum, ' + (batch.person_models||0) + ' person)</p>' : ''}
      ${(models.room_models||[]).length ? '<p style="margin-top:8px;font-size:12px;color:var(--text-dim)">Rum: ' +
        models.room_models.map(m => m.area_id + ' (' + m.samples_seen + ')').join(', ') + '</p>' : ''}
      ${(models.person_models||[]).length ? '<p style="margin-top:4px;font-size:12px;color:var(--text-dim)">Person: ' +
        models.person_models.map(m => m.person_id.replace('person.','') + ' (' + m.samples_seen + ')').join(', ') + '</p>' : ''}
      ${(models.markov_models||[]).length ? '<p style="margin-top:4px;font-size:12px;color:var(--text-dim)">Markov: ' +
        models.markov_models.map(m => m.person_id.replace('person.','') + ' (' + m.total_transitions + ' trans)').join(', ') + '</p>' : ''}
      ${(models.anomaly_models||[]).length ? '<p style="margin-top:4px;font-size:12px;color:var(--text-dim)">Anomaly: ' +
        models.anomaly_models.map(m => m.area_id + ' (' + m.samples_seen + (m.anomalies_detected ? ', ' + m.anomalies_detected + ' anom' : '') + ')').join(', ') + '</p>' : ''}
    `;
  } catch(e) { console.error('ML error:', e); }
}

async function loadPriors() {
  try {
    const p = await fetchJson('/api/priors');
    const el = document.getElementById('priors-info');
    const lastRun = p.last_run ? new Date(p.last_run).toLocaleString('da-DK') : 'Aldrig';
    const targets = p.targets || [];
    const rooms = targets.filter(t => t.target_type === 'room');
    const persons = targets.filter(t => t.target_type === 'person');
    el.innerHTML = `
      <p><span class="badge ${targets.length > 0 ? 'green' : 'orange'}">${targets.length} targets</span></p>
      <div class="mini-stat" style="margin-top:8px">
        <div class="item"><span class="val">${rooms.length}</span> <span class="lbl">rum</span></div>
        <div class="item"><span class="val">${persons.length}</span> <span class="lbl">personer</span></div>
      </div>
      <p style="margin-top:8px;font-size:12px;color:var(--text-dim)">Sidste beregning: ${lastRun}</p>
      <p style="margin-top:4px;font-size:11px;color:var(--text-dim)">Koerer kl. 03:00 UTC</p>
    `;
  } catch(e) { console.error('Priors error:', e); }
}

async function loadNotifications() {
  try {
    const n = await fetchJson('/api/notifications');
    const el = document.getElementById('notify-info');
    const statusBadge = n.active
      ? (n.is_quiet
        ? '<span class="badge orange">Stille timer</span>'
        : n.can_send
          ? '<span class="badge green">Aktiv</span>'
          : '<span class="badge orange">Cooldown</span>')
      : '<span class="badge">Deaktiveret</span>';
    const history = n.history || [];
    const historyHtml = history.length
      ? '<ul class="list" style="margin-top:10px">' + history.slice().reverse().map(h => {
          const ts = new Date(h[0]).toLocaleTimeString('da-DK');
          const typeBadge = h[1] === 'anomaly' ? 'badge purple'
            : h[1] === 'prediction' ? 'badge green'
            : h[1] === 'low_confidence' ? 'badge orange'
            : 'badge';
          return `<li><span style="font-size:12px">${ts} <span class="${typeBadge}">${h[1]}</span></span><span style="font-size:11px;color:var(--text-dim);max-width:60%;text-align:right">${h[2].substring(0, 80)}</span></li>`;
        }).join('') + '</ul>'
      : '<p style="margin-top:10px;font-size:12px;color:var(--text-dim)">Ingen notifikationer endnu</p>';
    el.innerHTML = `
      <p>${statusBadge}</p>
      <div class="mini-stat" style="margin-top:8px">
        <div class="item"><span class="val">${n.sent_today || 0}</span> <span class="lbl">sendt i dag</span></div>
        <div class="item"><span class="val">${n.max_daily || 0}</span> <span class="lbl">maks dagligt</span></div>
        <div class="item"><span class="val">${n.cooldown_min || 0}m</span> <span class="lbl">cooldown</span></div>
      </div>
      <p style="margin-top:6px;font-size:11px;color:var(--text-dim)">Stille timer: ${n.quiet_hours || '-'}</p>
      ${historyHtml}
    `;
  } catch(e) { console.error('Notifications error:', e); }
}

async function toggleRoom(slug) {
    await fetch(BASE + '/api/rooms/' + slug + '/toggle', {method:'POST'});
    loadRooms();
}
async function togglePerson(slug) {
    await fetch(BASE + '/api/persons/' + slug + '/toggle', {method:'POST'});
    loadPersons();
}

async function loadFeedback() {
    try {
        const [statsRes, pendingRes] = await Promise.all([
            fetchJson('/api/feedback/stats'),
            fetchJson('/api/feedback/pending')
        ]);
        const stats = statsRes;
        const pending = pendingRes;

        document.getElementById('feedback-status').innerHTML = `
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;text-align:center">
                <div><div style="font-size:1.5em;font-weight:bold">${stats.mode || '-'}</div><small>Tilstand</small></div>
                <div><div style="font-size:1.5em;font-weight:bold">${stats.pending_questions || 0}</div><small>Ventende</small></div>
                <div><div style="font-size:1.5em;font-weight:bold">${stats.answered_today || 0}</div><small>Besvaret i dag</small></div>
            </div>`;

        const el = document.getElementById('feedback-pending');
        if (!pending.length) {
            el.innerHTML = '<li style="color:var(--text-dim)">Ingen ventende</li>';
            return;
        }
        el.innerHTML = pending.slice(0, 5).map(q => {
            const opts = JSON.parse(q.options || '[]');
            return `<li class="feedback-q">
                <div>${q.question_text}</div>
                <div>${opts.map(o => `<button onclick="answerFeedback(${q.id},'${o}')">${o}</button>`).join('')}</div>
            </li>`;
        }).join('');
    } catch(e) { console.error('loadFeedback error', e); }
}

async function answerFeedback(id, answer) {
    await fetch(BASE + '/api/feedback/' + id, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({answer})
    });
    loadFeedback();
}

async function loadActivities() {
    try {
        const acts = await fetchJson('/api/activities/current');
        const el = document.getElementById('activities-current');
        const entries = Object.entries(acts);
        if (!entries.length) {
            el.innerHTML = '<li style="color:var(--text-dim)">Ingen aktiviteter</li>';
            return;
        }
        el.innerHTML = entries.map(([slug, a]) => `
            <li class="toggle-row">
                <span>${slug}</span>
                <span class="badge">${a.activity} (${Math.round(a.confidence*100)}%)</span>
            </li>`).join('');
    } catch(e) { console.error('loadActivities error', e); }
}

function loadAll() {
  loadStats(); loadRooms(); loadPersons(); loadEvents(); loadML(); loadPriors(); loadNotifications(); loadFeedback(); loadActivities();
}

loadAll();
setInterval(loadAll, 15000);
</script>

</body>
</html>"""
