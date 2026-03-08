"""Ingress web UI for HA Intelligence."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from entity_categories import CATEGORIES, CATEGORY_MAP, matches_category, get_category_keys

logger = logging.getLogger(__name__)

# Ingress path from HA
INGRESS_PATH = os.environ.get('INGRESS_PATH', '')


def create_app(db, event_listener, mqtt_pub, registry=None, ml_engine=None,
               notification_engine=None, feedback_engine=None,
               activity_engine=None, sensor_engine=None,
               settings_manager=None) -> FastAPI:
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
            "version": "1.0.2",
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

    @app.get("/api/insights")
    async def get_insights():
        if sensor_engine:
            return sensor_engine._insights_cache
        return {'persons': {}, 'rooms': {}}

    # ── Settings API ──────────────────────────────────────────────

    @app.get("/api/settings")
    async def get_settings():
        """Get all settings."""
        if not settings_manager:
            return {'error': 'Settings not initialized'}
        return settings_manager.get_all()

    @app.get("/api/settings/{category}")
    async def get_settings_category(category: str):
        """Get settings for a specific category."""
        if not settings_manager:
            return {'error': 'Settings not initialized'}
        data = settings_manager.get_category(category)
        if data is None:
            return JSONResponse({'error': f'Unknown category: {category}'}, status_code=404)
        return data

    @app.post("/api/settings/{category}")
    async def update_settings_category(category: str, request: Request):
        """Update settings for a category (merge)."""
        if not settings_manager:
            return JSONResponse({'error': 'Settings not initialized'}, status_code=500)
        try:
            body = await request.json()
            settings_manager.update_category(category, body)
            return {'ok': True, 'category': category}
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=400)

    @app.put("/api/settings/{category}")
    async def replace_settings_category(category: str, request: Request):
        """Replace entire category settings."""
        if not settings_manager:
            return JSONResponse({'error': 'Settings not initialized'}, status_code=500)
        try:
            body = await request.json()
            settings_manager.set_category(category, body)
            return {'ok': True, 'category': category}
        except Exception as e:
            return JSONResponse({'error': str(e)}, status_code=400)

    @app.post("/api/settings/{category}/reset")
    async def reset_settings_category(category: str):
        """Reset a category to defaults."""
        if not settings_manager:
            return JSONResponse({'error': 'Settings not initialized'}, status_code=500)
        settings_manager.reset_category(category)
        return {'ok': True, 'category': category, 'reset': True}

    @app.get("/api/options")
    async def get_options():
        """Get HA Supervisor options (read-only)."""
        if not settings_manager:
            return {'error': 'Settings not initialized'}
        return {
            'mqtt_host': settings_manager.get_option('mqtt_host'),
            'mqtt_port': settings_manager.get_option('mqtt_port'),
            'mqtt_user': settings_manager.get_option('mqtt_user'),
            'feedback_active': settings_manager.get_option('feedback_active'),
            'feedback_max_daily': settings_manager.get_option('feedback_max_daily'),
            'feedback_cooldown_min': settings_manager.get_option('feedback_cooldown_min'),
            'feedback_quiet_start': settings_manager.get_option('feedback_quiet_start'),
            'feedback_quiet_end': settings_manager.get_option('feedback_quiet_end'),
            'log_level': settings_manager.get_option('log_level'),
        }

    # ── Entity Picker API ──────────────────────────────────────

    @app.get("/api/entity-picker/categories")
    async def entity_picker_categories():
        """Return all 11 categories with metadata and selection counts."""
        selections = settings_manager.get_entity_selections() if settings_manager else {}
        result = []
        for cat in CATEGORIES:
            selected = selections.get(cat['key'], [])
            result.append({
                'key': cat['key'],
                'label': cat['label'],
                'icon': cat['icon'],
                'selected_count': len(selected),
            })
        return {'categories': result}

    @app.get("/api/entity-picker/{category}/entities")
    async def entity_picker_entities(category: str):
        """Return matching entities for a category with current state."""
        if category not in CATEGORY_MAP:
            return JSONResponse({'error': f'Unknown category: {category}'}, 404)

        cat_def = CATEGORY_MAP[category]
        selected = settings_manager.get_entity_selections(category) if settings_manager else []
        dismissed = settings_manager.get_dismissed_suggestions() if settings_manager else []

        # Get all entities from registry
        entities_by_area = {}  # area_name -> [entity_info]
        if registry:
            for eid, info in registry._entities.items():
                device_class = info.get('device_class') or info.get('original_device_class', '')
                if not matches_category(eid, device_class, cat_def):
                    continue

                area_id = registry.get_area_id(eid) or 'unassigned'
                area_name = registry.get_area_name(area_id) or 'Ingen rum'

                entity_info = {
                    'entity_id': eid,
                    'friendly_name': info.get('name') or info.get('original_name') or eid,
                    'device_class': device_class or None,
                    'area_id': area_id,
                    'area_name': area_name,
                    'selected': eid in selected,
                    'suggested': eid not in selected and eid not in dismissed,
                    'state': None,
                    'state_color': 'gray',
                }
                entities_by_area.setdefault(area_name, []).append(entity_info)

        # Sort areas and entities
        sorted_areas = []
        for area_name in sorted(entities_by_area.keys()):
            ents = sorted(entities_by_area[area_name], key=lambda e: e['friendly_name'])
            sorted_areas.append({
                'area_name': area_name,
                'entities': ents,
                'count': len(ents),
            })

        return {
            'category': category,
            'label': cat_def['label'],
            'areas': sorted_areas,
            'total': sum(a['count'] for a in sorted_areas),
            'selected_count': len(selected),
        }

    @app.post("/api/entity-picker/{category}/select")
    async def entity_picker_select(category: str, request: Request):
        """Save selected entity_ids for a category."""
        if category not in CATEGORY_MAP:
            return JSONResponse({'error': f'Unknown category: {category}'}, 404)
        body = await request.json()
        entity_ids = body.get('entity_ids', [])
        if not isinstance(entity_ids, list):
            return JSONResponse({'error': 'entity_ids must be a list'}, 400)
        if settings_manager:
            settings_manager.set_entity_selections(category, entity_ids)
        return {'status': 'ok', 'category': category, 'count': len(entity_ids)}

    @app.get("/api/entity-picker/suggestions")
    async def entity_picker_suggestions():
        """Return auto-suggest results across all categories."""
        if not registry or not settings_manager:
            return {'suggestions': {}, 'total': 0}

        selections = settings_manager.get_entity_selections()
        dismissed = settings_manager.get_dismissed_suggestions()
        suggestions = {}

        for cat in CATEGORIES:
            cat_selected = selections.get(cat['key'], [])
            cat_suggestions = []
            for eid, info in registry._entities.items():
                dc = info.get('device_class') or info.get('original_device_class', '')
                if not matches_category(eid, dc, cat):
                    continue
                if eid in cat_selected or eid in dismissed:
                    continue
                cat_suggestions.append({
                    'entity_id': eid,
                    'friendly_name': info.get('name') or info.get('original_name') or eid,
                    'device_class': dc or None,
                    'area_id': registry.get_area_id(eid) or 'unassigned',
                })
            if cat_suggestions:
                suggestions[cat['key']] = cat_suggestions

        total = sum(len(v) for v in suggestions.values())
        return {'suggestions': suggestions, 'total': total}

    @app.post("/api/entity-picker/suggestions/dismiss")
    async def entity_picker_dismiss(request: Request):
        """Dismiss a suggestion so it won't appear again."""
        body = await request.json()
        entity_id = body.get('entity_id', '')
        if not entity_id:
            return JSONResponse({'error': 'entity_id required'}, 400)
        if settings_manager:
            settings_manager.dismiss_suggestion(entity_id)
        return {'status': 'ok', 'dismissed': entity_id}

    @app.get("/api/entity-picker/stats")
    async def entity_picker_stats():
        """Return stability stats for entities (unavailable %)."""
        if not settings_manager:
            return {'stats': {}}

        selections = settings_manager.get_entity_selections()
        all_selected = []
        for ents in selections.values():
            all_selected.extend(ents)

        stats = {}
        for eid in all_selected:
            # TODO: Calculate unavailable % from DB history
            stats[eid] = {'unavailable_pct': 0, 'unstable': False}

        return {
            'stats': stats,
            'total_selected': len(all_selected),
        }

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
  /* Tab navigation */
  .tab-bar { display:flex; gap:4px; margin-bottom:24px; }
  .tab-btn { padding:8px 20px; border-radius:8px; border:none; background:transparent; color:var(--text-dim); cursor:pointer; font-size:14px; font-weight:500; transition:.2s; }
  .tab-btn:hover { background:rgba(255,255,255,0.05); }
  .tab-btn.active { background:var(--primary); color:white; }
  .tab-content { display:none; }
  .tab-content.active { display:block; }
  /* Insight cards */
  .insight-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:16px; margin-bottom:24px; }
  .insight-card { background:var(--card); border-radius:12px; padding:20px; border:1px solid rgba(255,255,255,0.05); }
  .insight-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.08); }
  .insight-header h3 { font-size:16px; display:flex; align-items:center; gap:8px; }
  .insight-section { margin-bottom:14px; }
  .insight-section h4 { font-size:11px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }
  .insight-row { display:flex; gap:12px; flex-wrap:wrap; font-size:13px; margin:3px 0; }
  .insight-row .key { color:var(--text-dim); min-width:90px; }
  .insight-row .val { font-weight:500; }
  .evidence-item { padding:4px 8px; background:rgba(255,255,255,0.03); border-radius:4px; margin:2px 0; font-size:12px; display:flex; align-items:center; gap:6px; }
  .evidence-dot { width:6px; height:6px; border-radius:50%; background:var(--green); flex-shrink:0; }
  .confidence-bar { height:4px; border-radius:2px; background:rgba(255,255,255,0.1); margin-top:4px; width:100%; }
  .confidence-fill { height:100%; border-radius:2px; transition:width .3s; }
  .decision-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .decision-box { background:rgba(255,255,255,0.03); border-radius:6px; padding:8px 10px; font-size:12px; }
  .decision-box .label { color:var(--text-dim); margin-bottom:2px; }
  .decision-box .value { font-weight:600; }
  .state-badge { padding:2px 10px; border-radius:6px; font-size:12px; font-weight:600; }
  .state-badge.active { background:rgba(34,197,94,0.15); color:var(--green); }
  .state-badge.idle { background:rgba(245,158,11,0.15); color:var(--orange); }
  .state-badge.sleeping { background:rgba(168,85,247,0.15); color:var(--purple); }
  .state-badge.away { background:rgba(148,163,184,0.15); color:var(--text-dim); }
  .state-badge.occupied { background:rgba(34,197,94,0.15); color:var(--green); }
  .state-badge.empty { background:rgba(239,68,68,0.15); color:var(--red); }
  .insight-empty { color:var(--text-dim); font-size:14px; text-align:center; padding:40px 0; }

  /* Settings tab */
  .settings-layout { display:flex; gap:16px; min-height:400px; }
  .settings-nav { width:180px; flex-shrink:0; display:flex; flex-direction:column; gap:4px; }
  .settings-nav button { background:var(--card); border:1px solid var(--border); color:var(--text); padding:10px 14px; border-radius:8px; cursor:pointer; text-align:left; font-size:13px; transition:all 0.2s; }
  .settings-nav button:hover { border-color:var(--accent); }
  .settings-nav button.active { background:var(--accent); color:#fff; border-color:var(--accent); }
  .settings-page { flex:1; display:none; }
  .settings-page.active { display:block; }
  .settings-card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:20px; margin-bottom:16px; }
  .settings-card h3 { margin:0 0 12px; font-size:15px; }
  .settings-field { margin-bottom:14px; }
  .settings-field label { display:block; font-size:12px; color:var(--text-dim); margin-bottom:4px; }
  .settings-field input, .settings-field select { width:100%; background:var(--bg); border:1px solid var(--border); color:var(--text); padding:8px 10px; border-radius:6px; font-size:13px; box-sizing:border-box; }
  .settings-field input:focus, .settings-field select:focus { border-color:var(--accent); outline:none; }
  .settings-field .hint { font-size:11px; color:var(--text-dim); margin-top:2px; }
  .settings-actions { display:flex; gap:8px; margin-top:16px; }
  .settings-actions button { padding:8px 16px; border-radius:6px; font-size:13px; cursor:pointer; border:none; }
  .btn-save { background:var(--accent); color:#fff; }
  .btn-save:hover { opacity:0.9; }
  .btn-reset { background:var(--card); color:var(--text-dim); border:1px solid var(--border) !important; }
  .btn-reset:hover { border-color:var(--red) !important; color:var(--red); }
  .btn-add { background:rgba(34,197,94,0.15); color:var(--green); border:1px solid var(--green) !important; font-size:12px; padding:6px 12px; }
  .btn-delete { background:rgba(239,68,68,0.1); color:var(--red); border:1px solid transparent !important; font-size:11px; padding:4px 8px; cursor:pointer; border-radius:4px; }
  .btn-delete:hover { border-color:var(--red) !important; }
  .zone-row { display:flex; gap:8px; align-items:center; margin-bottom:8px; padding:8px; background:var(--bg); border-radius:6px; }
  .zone-row input { flex:1; }
  .settings-toast { position:fixed; bottom:20px; right:20px; background:var(--green); color:#fff; padding:10px 20px; border-radius:8px; font-size:13px; z-index:100; display:none; }
  @media(max-width:640px) { .settings-layout { flex-direction:column; } .settings-nav { width:100%; flex-direction:row; overflow-x:auto; } }

  /* Entity picker */
  .entity-picker-layout { display:flex; gap:0; min-height:500px; }
  .entity-picker-sidebar { width:180px; border-right:1px solid var(--border); padding:12px; flex-shrink:0; }
  .entity-picker-main { flex:1; padding:16px; overflow-y:auto; }
  .ep-cat-btn { display:flex; align-items:center; gap:8px; width:100%; padding:8px 12px; border:none; background:transparent; cursor:pointer; border-radius:6px; font-size:13px; text-align:left; color:var(--text); }
  .ep-cat-btn:hover { background:rgba(255,255,255,0.05); }
  .ep-cat-btn.active { background:var(--accent); color:white; }
  .ep-cat-btn .ep-count { margin-left:auto; font-size:11px; opacity:0.7; }
  .ep-area-group { margin-bottom:12px; }
  .ep-area-header { display:flex; align-items:center; gap:8px; padding:8px; cursor:pointer; font-weight:600; border-radius:6px; }
  .ep-area-header:hover { background:rgba(255,255,255,0.05); }
  .ep-entity-row { display:flex; flex-direction:column; gap:2px; padding:8px 12px; border:1px solid var(--border); border-radius:6px; margin:4px 0; }
  .ep-entity-row:hover { border-color:var(--accent); }
  .ep-entity-line1 { display:flex; align-items:center; gap:8px; }
  .ep-entity-line1 input[type=checkbox] { flex-shrink:0; }
  .ep-entity-name { font-weight:500; flex:1; }
  .ep-badge { font-size:11px; padding:1px 6px; border-radius:10px; }
  .ep-badge-suggested { background:#fef3c7; color:#92400e; }
  .ep-entity-id { font-size:11px; color:var(--text-dim); padding-left:26px; }
  .ep-entity-meta { font-size:11px; color:var(--text-dim); padding-left:26px; display:flex; gap:12px; }
  .ep-state-on { color:var(--green); }
  .ep-state-off { color:var(--text-dim); }
  .ep-sidebar-stats { margin-top:16px; padding-top:12px; border-top:1px solid var(--border); font-size:12px; color:var(--text-dim); }
  .ep-actions { display:flex; gap:8px; margin-top:16px; padding-top:12px; border-top:1px solid var(--border); }
  @media(max-width:640px) { .entity-picker-layout { flex-direction:column; } .entity-picker-sidebar { width:100%; border-right:none; border-bottom:1px solid var(--border); } }
</style>
</head>
<body>

<h1>
  <span class="icon">&#129504;</span> HA Intelligence
  <span class="version">v1.0.2</span>
  <span style="flex:1"></span>
  <button class="refresh-btn" onclick="loadAll()">Opdater</button>
</h1>

<div class="tab-bar">
  <button class="tab-btn active" data-tab="system" onclick="switchTab('system')">System</button>
  <button class="tab-btn" data-tab="insights" onclick="switchTab('insights')">Indsigt</button>
  <button class="tab-btn" data-tab="settings" onclick="switchTab('settings');loadSettings()">Indstillinger</button>
</div>

<div id="tab-system" class="tab-content active">

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

</div><!-- /tab-system -->

<div id="tab-insights" class="tab-content">
  <div id="hustilstand-card" class="card" style="margin-bottom:16px;display:none">
    <div style="display:flex;align-items:center;justify-content:space-between">
      <h2 style="margin:0">&#127968; Hustilstand</h2>
      <span id="hust-state-badge" class="state-badge"></span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-top:12px">
      <div class="stat-box"><div class="stat-value" id="hust-persons-home">-</div><div class="stat-label">Hjemme</div></div>
      <div class="stat-box"><div class="stat-value" id="hust-rooms-active">-</div><div class="stat-label">Aktive rum</div></div>
      <div class="stat-box"><div class="stat-value" id="hust-source">-</div><div class="stat-label">Kilde</div></div>
    </div>
    <div id="hust-persons-list" style="margin-top:8px;font-size:12px;color:var(--text-dim)"></div>
  </div>
  <div class="insight-grid" id="person-insights"></div>
  <div class="insight-grid" id="room-insights"></div>
  <div id="insights-empty" class="insight-empty" style="display:none">Venter paa data... Indsigt vises naar foerste publish-cyklus er koert.</div>
</div><!-- /tab-insights -->

<div id="tab-settings" class="tab-content">
<div class="settings-layout">
  <div class="settings-nav">
    <button class="active" data-page="s-hustilstand" onclick="switchSettingsPage('s-hustilstand')">Hustilstand</button>
    <button data-page="s-zones" onclick="switchSettingsPage('s-zones')">Zoner</button>
    <button data-page="s-ml" onclick="switchSettingsPage('s-ml')">ML Model</button>
    <button data-page="s-system" onclick="switchSettingsPage('s-system')">System</button>
    <button data-page="s-entities" onclick="switchSettingsPage('s-entities')">Entities</button>
  </div>
  <div style="flex:1">

    <!-- Hustilstand -->
    <div id="s-hustilstand" class="settings-page active">
      <div class="settings-card">
        <h3>Hustilstand Sensor</h3>
        <p style="font-size:12px;color:var(--text-dim);margin-bottom:14px">Synkroniser sensor.hai_hustilstand med en HA input_select</p>
        <div class="settings-field">
          <label>Aktiveret</label>
          <label class="toggle-switch" style="margin-top:4px">
            <input type="checkbox" id="hust-enabled">
            <span class="toggle-slider"></span>
          </label>
        </div>
        <div class="settings-field">
          <label>Entity ID</label>
          <input type="text" id="hust-entity" placeholder="input_select.hus_tilstand">
          <div class="hint">HA entity der spoeres (input_select)</div>
        </div>
        <div class="settings-card" style="margin-top:12px;padding:14px;background:var(--bg)">
          <h3 style="font-size:13px">State Map</h3>
          <p style="font-size:11px;color:var(--text-dim);margin-bottom:10px">HA state &rarr; HAI state</p>
          <div id="hust-statemap"></div>
        </div>
        <div class="settings-actions">
          <button class="btn-save" onclick="saveHustilstand()">Gem</button>
          <button class="btn-reset" onclick="resetCategory('hustilstand')">Nulstil</button>
        </div>
      </div>
    </div>

    <!-- Zones -->
    <div id="s-zones" class="settings-page">
      <div class="settings-card">
        <h3>Zone &rarr; Aktivitet</h3>
        <p style="font-size:12px;color:var(--text-dim);margin-bottom:14px">Map HA-zoner til aktiviteter for personer der er ude</p>
        <div id="zone-map-list"></div>
        <button class="btn-add" onclick="addZoneRow()" style="margin-top:8px">+ Tilfoej zone</button>
        <div class="settings-actions">
          <button class="btn-save" onclick="saveZones()">Gem</button>
          <button class="btn-reset" onclick="resetCategory('activity')">Nulstil</button>
        </div>
      </div>
    </div>

    <!-- ML -->
    <div id="s-ml" class="settings-page">
      <div class="settings-card">
        <h3>ML Parametre</h3>
        <div class="settings-field">
          <label>Threshold (min. samples foer ML bruges)</label>
          <input type="number" id="ml-threshold" min="10" max="500" step="10">
        </div>
        <div class="settings-field">
          <label>ML Weight (andel af ML i endelig confidence)</label>
          <input type="number" id="ml-weight" min="0" max="1" step="0.05">
        </div>
        <div class="settings-field">
          <label>Prior Weight (andel af prior i endelig confidence)</label>
          <input type="number" id="ml-prior-weight" min="0" max="1" step="0.05">
          <div class="hint">ML Weight + Prior Weight boer vaere 1.0</div>
        </div>
        <div class="settings-actions">
          <button class="btn-save" onclick="saveML()">Gem</button>
          <button class="btn-reset" onclick="resetCategory('ml')">Nulstil</button>
        </div>
      </div>
    </div>

    <!-- System -->
    <div id="s-system" class="settings-page">
      <div class="settings-card">
        <h3>System</h3>
        <div class="settings-field">
          <label>Publish interval (sekunder)</label>
          <input type="number" id="sys-interval" min="10" max="600" step="5">
          <div class="hint">Hvor ofte sensorer opdateres via MQTT</div>
        </div>
        <div class="settings-actions">
          <button class="btn-save" onclick="saveSystem()">Gem</button>
          <button class="btn-reset" onclick="resetCategory('system')">Nulstil</button>
        </div>
      </div>
    </div>

    <!-- Entities -->
    <div id="s-entities" class="settings-page">
      <div class="entity-picker-layout">
        <div class="entity-picker-sidebar">
          <h3 style="margin:0 0 12px;font-size:14px;">Kategorier</h3>
          <div id="ep-category-list"></div>
          <div class="ep-sidebar-stats">
            <div>Valgte: <span id="ep-total-selected">0</span></div>
            <div>Foreslaaede: <span id="ep-total-suggestions">0</span></div>
          </div>
        </div>
        <div class="entity-picker-main">
          <div id="ep-category-header"></div>
          <input type="text" id="ep-search" placeholder="Soeg entities..."
                 class="settings-field" style="margin-bottom:12px;"
                 oninput="filterEntities(this.value)">
          <div id="ep-entity-list"></div>
          <div class="ep-actions">
            <button class="btn-save" onclick="saveEntitySelections()">Gem valg</button>
            <button class="btn-reset" onclick="resetEntitySelections()">Nulstil</button>
          </div>
        </div>
      </div>
    </div>

  </div>
</div>
<div class="settings-toast" id="settings-toast">Gemt!</div>
</div><!-- /tab-settings -->

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

// ── Tab switching ──────────────────────────────────
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
  document.getElementById(`tab-${tabId}`).classList.add('active');
  localStorage.setItem('hai_tab', tabId);
}

// Restore saved tab
const savedTab = localStorage.getItem('hai_tab');
if (savedTab) switchTab(savedTab);

// ── Insight rendering ──────────────────────────────
function confColor(c) {
  if (c >= 0.8) return 'var(--green)';
  if (c >= 0.5) return 'var(--orange)';
  return 'var(--red)';
}
function pct(v) { return Math.round((v || 0) * 100) + '%'; }

function renderPersonInsights(persons) {
  const el = document.getElementById('person-insights');
  const keys = Object.keys(persons);
  if (!keys.length) { el.innerHTML = ''; return; }
  el.innerHTML = keys.map(slug => {
    const p = persons[slug];
    const stateClass = p.state || 'away';
    const room = p.room || 'ukendt';
    const roomSrc = p.room_source || '-';
    const roomConf = p.room_confidence || 0;
    const ble = p.ble_distance != null ? p.ble_distance + 'm' : '-';
    const mins = p.minutes_in_room || 0;
    const act = p.inferred_activity || '-';
    const actConf = p.activity_confidence || 0;
    const actSrc = p.activity_source || '-';
    const actZone = p.activity_zone || '-';
    const actCands = (p.activity_candidates || []).join(', ') || '-';
    const actDevs = p.activity_devices ? Object.entries(p.activity_devices).map(([k,v]) => k+': '+v).join(', ') : '-';
    const ruleS = p.rule_state || '-';
    const ruleC = p.rule_confidence || 0;
    const mlS = p.ml_state || '-';
    const mlC = p.ml_confidence || 0;
    const priorS = p.prior_state || '-';
    const priorP = p.prior_probability || 0;
    const nextRoom = p.predicted_next_room || '-';
    const nextP = p.next_room_probability || 0;
    return `<div class="insight-card">
      <div class="insight-header">
        <h3>&#128100; ${p.name}</h3>
        <span class="state-badge ${stateClass}">${p.state}</span>
      </div>
      <div class="insight-section">
        <h4>Placering</h4>
        <div class="insight-row"><span class="key">Rum:</span><span class="val">${room}</span></div>
        <div class="insight-row"><span class="key">Kilde:</span><span class="val">${roomSrc}</span><span class="key" style="margin-left:8px">Confidence:</span><span class="val">${pct(roomConf)}</span></div>
        <div class="confidence-bar"><div class="confidence-fill" style="width:${roomConf*100}%;background:${confColor(roomConf)}"></div></div>
        <div class="insight-row"><span class="key">BLE afstand:</span><span class="val">${ble}</span><span class="key" style="margin-left:8px">I rummet:</span><span class="val">${mins} min</span></div>
      </div>
      ${act !== '-' ? `<div class="insight-section">
        <h4>Aktivitet</h4>
        <div class="insight-row"><span class="key">Aktivitet:</span><span class="val">${act}</span></div>
        <div class="insight-row"><span class="key">Kilde:</span><span class="val">${actSrc}</span><span class="key" style="margin-left:8px">Confidence:</span><span class="val">${pct(actConf)}</span></div>
        <div class="confidence-bar"><div class="confidence-fill" style="width:${actConf*100}%;background:${confColor(actConf)}"></div></div>
        <div class="insight-row"><span class="key">Zone:</span><span class="val">${actZone}</span></div>
        <div class="insight-row"><span class="key">Enheder:</span><span class="val">${actDevs}</span></div>
        <div class="insight-row"><span class="key">Kandidater:</span><span class="val">${actCands}</span></div>
      </div>` : ''}
      <div class="insight-section">
        <h4>ML Beslutning</h4>
        <div class="decision-grid">
          <div class="decision-box"><div class="label">Regel</div><div class="value">${ruleS} (${pct(ruleC)})</div></div>
          <div class="decision-box"><div class="label">ML</div><div class="value">${mlS} (${pct(mlC)})</div></div>
          <div class="decision-box"><div class="label">Prior</div><div class="value">${priorS} (${pct(priorP)})</div></div>
          <div class="decision-box"><div class="label">Naeste rum</div><div class="value">${nextRoom} (${pct(nextP)})</div></div>
        </div>
      </div>
    </div>`;
  }).join('');
}

function renderRoomInsights(rooms) {
  const el = document.getElementById('room-insights');
  const keys = Object.keys(rooms);
  if (!keys.length) { el.innerHTML = ''; return; }
  el.innerHTML = keys.map(slug => {
    const r = rooms[slug];
    const stateClass = r.state === 'occupied' ? 'occupied' : (r.state === 'empty' ? 'empty' : 'active');
    const motionTotal = r.motion_sensors || 0;
    const motionActive = r.active_sensors || 0;
    const lastOcc = r.last_occupied || '-';
    const conf = r.confidence || 0;
    const src = r.source || '-';
    const evidSrcs = r.evidence_sources || [];
    const evidCount = r.evidence_count || 0;
    const evidDetailRaw = r.evidence_detail || {};
    const evidDetail = Array.isArray(evidDetailRaw) ? evidDetailRaw : Object.entries(evidDetailRaw).map(([k,v]) => `${k}: ${v}`);
    const ruleS = r.rule_state || '-';
    const ruleC = r.rule_confidence || 0;
    const mlS = r.ml_state || '-';
    const mlC = r.ml_confidence || 0;
    const priorS = r.prior_state || '-';
    const priorP = r.prior_probability || 0;
    const anomScore = r.anomaly_score != null ? r.anomaly_score.toFixed(2) : '-';
    const anomReady = r.anomaly_ready || false;
    return `<div class="insight-card">
      <div class="insight-header">
        <h3>&#127968; ${r.name}</h3>
        <span class="state-badge ${stateClass}">${r.state}</span>
      </div>
      <div class="insight-section">
        <h4>Sensorer</h4>
        <div class="insight-row"><span class="key">Motion:</span><span class="val">${motionActive} aktive / ${motionTotal} total</span></div>
        <div class="insight-row"><span class="key">Sidst optaget:</span><span class="val">${lastOcc}</span></div>
        <div class="insight-row"><span class="key">Confidence:</span><span class="val">${pct(conf)}</span><span class="key" style="margin-left:8px">Kilde:</span><span class="val">${src}</span></div>
        <div class="confidence-bar"><div class="confidence-fill" style="width:${conf*100}%;background:${confColor(conf)}"></div></div>
      </div>
      ${evidCount > 0 ? `<div class="insight-section">
        <h4>Evidence (${evidCount} kilder)</h4>
        ${evidDetail.map(e => `<div class="evidence-item"><span class="evidence-dot"></span>${e}</div>`).join('')}
        ${Object.keys(r.evidence_entities || {}).length ? `<div style="margin-top:6px;font-size:11px;color:var(--text-dim)">${Object.entries(r.evidence_entities).map(([eid,name]) => name || eid).join(', ')}</div>` : ''}
      </div>` : ''}
      <div class="insight-section">
        <h4>ML Beslutning</h4>
        <div class="decision-grid">
          <div class="decision-box"><div class="label">Regel</div><div class="value">${ruleS} (${pct(ruleC)})</div></div>
          <div class="decision-box"><div class="label">ML</div><div class="value">${mlS} (${pct(mlC)})</div></div>
          <div class="decision-box"><div class="label">Prior</div><div class="value">${priorS} (${pct(priorP)})</div></div>
          <div class="decision-box"><div class="label">Anomaly</div><div class="value">${anomScore} ${anomReady ? '&#9989;' : '&#9203;'}</div></div>
        </div>
      </div>
    </div>`;
  }).join('');
}

function renderHustilstand(ht) {
  const card = document.getElementById('hustilstand-card');
  if (!ht || !ht.state) { card.style.display = 'none'; return; }
  card.style.display = 'block';
  const badge = document.getElementById('hust-state-badge');
  badge.textContent = ht.state;
  badge.className = 'state-badge ' + (ht.state === 'hjemme' ? 'active' : ht.state === 'nat' ? 'empty' : 'occupied');
  document.getElementById('hust-persons-home').textContent = (ht.persons_home || 0) + '/' + (ht.persons_total || 0);
  document.getElementById('hust-rooms-active').textContent = ht.rooms_active || 0;
  document.getElementById('hust-source').textContent = ht.source === 'ha_entity' ? 'HA' : 'lokal';
  const pList = ht.persons_at_home || [];
  document.getElementById('hust-persons-list').textContent = pList.length ? 'Hjemme: ' + pList.join(', ') : '';
}

async function loadInsights() {
  try {
    const data = await fetchJson('/api/insights');
    const hasData = Object.keys(data.persons || {}).length > 0 || Object.keys(data.rooms || {}).length > 0;
    document.getElementById('insights-empty').style.display = hasData ? 'none' : 'block';
    renderHustilstand(data.hustilstand || {});
    renderPersonInsights(data.persons || {});
    renderRoomInsights(data.rooms || {});
  } catch(e) { console.error('loadInsights error', e); }
}

// ── Settings tab ──────────────────────────────────
function switchSettingsPage(pageId) {
  document.querySelectorAll('.settings-nav button').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.settings-page').forEach(p => p.classList.remove('active'));
  document.querySelector(`.settings-nav [data-page="${pageId}"]`).classList.add('active');
  document.getElementById(pageId).classList.add('active');
  if (pageId === 's-entities') {
    loadEntityPickerCategories();
    loadEntityPickerCategory(epCurrentCategory);
  }
}

// ── Entity Picker ──────────────────────────────
let epCurrentCategory = 'persons';
let epCategoryData = {};
let epSelections = {};

async function loadEntityPickerCategories() {
  try {
    const data = await fetchJson('/api/entity-picker/categories');
    const list = document.getElementById('ep-category-list');
    if (!list) return;
    list.innerHTML = '';
    let totalSelected = 0;
    data.categories.forEach(cat => {
      totalSelected += cat.selected_count;
      const btn = document.createElement('button');
      btn.className = 'ep-cat-btn' + (cat.key === epCurrentCategory ? ' active' : '');
      btn.innerHTML = '<span>' + cat.label + '</span><span class="ep-count">' + cat.selected_count + '</span>';
      btn.onclick = () => loadEntityPickerCategory(cat.key);
      list.appendChild(btn);
    });
    document.getElementById('ep-total-selected').textContent = totalSelected;

    const sugData = await fetchJson('/api/entity-picker/suggestions');
    document.getElementById('ep-total-suggestions').textContent = sugData.total;
  } catch(e) { console.error('Failed to load categories:', e); }
}

async function loadEntityPickerCategory(category) {
  epCurrentCategory = category;
  document.querySelectorAll('.ep-cat-btn').forEach(btn => btn.classList.remove('active'));
  const btns = document.querySelectorAll('.ep-cat-btn');
  btns.forEach(btn => {
    if (btn.onclick && btn.textContent.trim()) {
      // Re-highlight active
    }
  });
  await loadEntityPickerCategories();

  try {
    const data = await fetchJson('/api/entity-picker/' + category + '/entities');
    epCategoryData[category] = data;

    if (!epSelections[category]) {
      epSelections[category] = new Set();
      data.areas.forEach(area => {
        area.entities.forEach(e => {
          if (e.selected) epSelections[category].add(e.entity_id);
        });
      });
    }
    renderEntityList(data);
  } catch(e) { console.error('Failed to load entities:', e); }
}

function renderEntityList(data) {
  const container = document.getElementById('ep-entity-list');
  const header = document.getElementById('ep-category-header');
  if (!container) return;

  header.innerHTML = '<h3 style="margin:0 0 8px">' + data.label + ' (' + data.total + ' entities)</h3>';
  container.innerHTML = '';

  data.areas.forEach(area => {
    const group = document.createElement('div');
    group.className = 'ep-area-group';

    const areaHeader = document.createElement('div');
    areaHeader.className = 'ep-area-header';
    areaHeader.innerHTML = '<span>&#9660;</span> <span>' + area.area_name + '</span> <span style="opacity:0.5">(' + area.count + ')</span>';
    areaHeader.onclick = () => {
      const content = group.querySelector('.ep-area-content');
      const arrow = areaHeader.querySelector('span');
      if (content.style.display === 'none') {
        content.style.display = 'block'; arrow.textContent = '\u25BC';
      } else {
        content.style.display = 'none'; arrow.textContent = '\u25B6';
      }
    };
    group.appendChild(areaHeader);

    const content = document.createElement('div');
    content.className = 'ep-area-content';
    area.entities.forEach(e => {
      const row = document.createElement('div');
      row.className = 'ep-entity-row';
      row.dataset.entityId = e.entity_id;

      const isSelected = epSelections[epCurrentCategory] && epSelections[epCurrentCategory].has(e.entity_id);
      const stateClass = e.state === 'on' ? 'ep-state-on' : 'ep-state-off';
      let badges = '';
      if (e.suggested) badges += '<span class="ep-badge ep-badge-suggested">Foreslaaet</span>';

      row.innerHTML =
        '<div class="ep-entity-line1">' +
          '<input type="checkbox" ' + (isSelected ? 'checked' : '') +
          ' onchange="toggleEntity(\'' + e.entity_id.replace(/'/g, "\\'") + '\', this.checked)">' +
          '<span class="ep-entity-name">' + (e.friendly_name || e.entity_id) + '</span>' +
          badges +
        '</div>' +
        '<div class="ep-entity-id">' + e.entity_id + '</div>' +
        '<div class="ep-entity-meta">' +
          (e.device_class ? '<span>' + e.device_class + '</span>' : '') +
          '<span class="' + stateClass + '">State: ' + (e.state || '?') + '</span>' +
        '</div>';
      content.appendChild(row);
    });
    group.appendChild(content);
    container.appendChild(group);
  });
}

function toggleEntity(entityId, checked) {
  if (!epSelections[epCurrentCategory]) {
    epSelections[epCurrentCategory] = new Set();
  }
  if (checked) {
    epSelections[epCurrentCategory].add(entityId);
  } else {
    epSelections[epCurrentCategory].delete(entityId);
  }
}

async function saveEntitySelections() {
  const category = epCurrentCategory;
  const entityIds = [...(epSelections[category] || [])];
  try {
    const resp = await fetch(BASE + '/api/entity-picker/' + category + '/select', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({entity_ids: entityIds}),
    });
    if (resp.ok) {
      showToast(entityIds.length + ' entities gemt for ' + category);
      await loadEntityPickerCategories();
    } else {
      showToast('Fejl ved gem');
    }
  } catch(e) { showToast('Fejl ved gem'); }
}

async function resetEntitySelections() {
  delete epSelections[epCurrentCategory];
  await loadEntityPickerCategory(epCurrentCategory);
}

function filterEntities(query) {
  const q = query.toLowerCase();
  document.querySelectorAll('.ep-entity-row').forEach(row => {
    const text = row.textContent.toLowerCase();
    row.style.display = text.includes(q) ? '' : 'none';
  });
}

function showToast(msg) {
  const t = document.getElementById('settings-toast');
  t.textContent = msg || 'Gemt!';
  t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 2000);
}

async function loadSettings() {
  try {
    const all = await fetchJson('/api/settings');
    // Hustilstand
    const h = all.hustilstand || {};
    document.getElementById('hust-enabled').checked = h.enabled !== false;
    document.getElementById('hust-entity').value = h.entity_id || 'input_select.hus_tilstand';
    renderStateMap(h.state_map || {});
    // Zones
    renderZoneMap((all.activity || {}).zone_map || {});
    // ML
    const ml = all.ml || {};
    document.getElementById('ml-threshold').value = ml.threshold || 50;
    document.getElementById('ml-weight').value = ml.ml_weight || 0.7;
    document.getElementById('ml-prior-weight').value = ml.prior_weight || 0.3;
    // System
    const sys = all.system || {};
    document.getElementById('sys-interval').value = sys.publish_interval || 60;
  } catch(e) { console.error('loadSettings error', e); }
}

function renderStateMap(map) {
  const el = document.getElementById('hust-statemap');
  const entries = Object.entries(map);
  if (!entries.length) { el.innerHTML = '<p style="color:var(--text-dim);font-size:12px">Ingen state map</p>'; return; }
  el.innerHTML = entries.map(([k, v]) =>
    `<div class="zone-row">
      <input type="text" value="${k}" data-orig="${k}" class="smap-key" placeholder="HA state">
      <span style="color:var(--text-dim)">&rarr;</span>
      <input type="text" value="${v}" class="smap-val" placeholder="HAI state">
    </div>`
  ).join('');
}

function renderZoneMap(map) {
  const el = document.getElementById('zone-map-list');
  const entries = Object.entries(map);
  if (!entries.length) { el.innerHTML = ''; return; }
  el.innerHTML = entries.map(([zone, cfg]) => {
    const act = typeof cfg === 'string' ? cfg : (cfg.activity || '');
    const name = typeof cfg === 'object' ? (cfg.name || '') : '';
    return `<div class="zone-row">
      <input type="text" value="${zone}" class="zmap-zone" placeholder="zone.xxx">
      <span style="color:var(--text-dim)">&rarr;</span>
      <input type="text" value="${act}" class="zmap-act" placeholder="aktivitet">
      <input type="text" value="${name}" class="zmap-name" placeholder="navn" style="max-width:100px">
      <button class="btn-delete" onclick="this.parentElement.remove()">&#10005;</button>
    </div>`;
  }).join('');
}

function addZoneRow(zone, activity, name) {
  const el = document.getElementById('zone-map-list');
  const row = document.createElement('div');
  row.className = 'zone-row';
  row.innerHTML = `
    <input type="text" value="${zone||''}" class="zmap-zone" placeholder="zone.xxx">
    <span style="color:var(--text-dim)">&rarr;</span>
    <input type="text" value="${activity||''}" class="zmap-act" placeholder="aktivitet">
    <input type="text" value="${name||''}" class="zmap-name" placeholder="navn" style="max-width:100px">
    <button class="btn-delete" onclick="this.parentElement.remove()">&#10005;</button>`;
  el.appendChild(row);
}

async function saveHustilstand() {
  const data = {
    enabled: document.getElementById('hust-enabled').checked,
    entity_id: document.getElementById('hust-entity').value.trim(),
    state_map: {}
  };
  document.querySelectorAll('#hust-statemap .zone-row').forEach(row => {
    const k = row.querySelector('.smap-key').value.trim();
    const v = row.querySelector('.smap-val').value.trim();
    if (k && v) data.state_map[k] = v;
  });
  await fetch(BASE + '/api/settings/hustilstand', {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  showToast('Hustilstand gemt!');
}

async function saveZones() {
  const zone_map = {};
  document.querySelectorAll('#zone-map-list .zone-row').forEach(row => {
    const zone = row.querySelector('.zmap-zone').value.trim();
    const act = row.querySelector('.zmap-act').value.trim();
    const name = row.querySelector('.zmap-name').value.trim();
    if (zone && act) {
      zone_map[zone] = name ? {activity: act, name: name} : act;
    }
  });
  await fetch(BASE + '/api/settings/activity', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({zone_map: zone_map})});
  showToast('Zoner gemt!');
}

async function saveML() {
  const data = {
    threshold: parseInt(document.getElementById('ml-threshold').value) || 50,
    ml_weight: parseFloat(document.getElementById('ml-weight').value) || 0.7,
    prior_weight: parseFloat(document.getElementById('ml-prior-weight').value) || 0.3,
  };
  await fetch(BASE + '/api/settings/ml', {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  showToast('ML parametre gemt!');
}

async function saveSystem() {
  const data = {
    publish_interval: parseInt(document.getElementById('sys-interval').value) || 60,
  };
  await fetch(BASE + '/api/settings/system', {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  showToast('System gemt!');
}

async function resetCategory(cat) {
  if (!confirm('Nulstil ' + cat + ' til standardvaerdier?')) return;
  await fetch(BASE + '/api/settings/' + cat + '/reset', {method:'POST'});
  showToast(cat + ' nulstillet!');
  loadSettings();
}

function loadAll() {
  loadStats(); loadRooms(); loadPersons(); loadEvents(); loadML(); loadPriors(); loadNotifications(); loadFeedback(); loadActivities(); loadInsights();
}

loadAll();
setInterval(loadAll, 15000);
</script>

</body>
</html>"""
