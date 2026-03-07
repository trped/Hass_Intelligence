# HAI v0.7.0 Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all broken/stalled ML systems and wire remaining features to bring HAI from v0.6.0 to v0.7.0.

**Architecture:** Most code already exists but has integration bugs. Phase 4c (batch training) just needs scikit-learn added to requirements.txt. Phase 3 (priors) needs debugging. Notifications need wiring into main.py. Frontend gets floorplan + chat.

**Tech Stack:** Python 3, River 0.23.0, scikit-learn (NEW), FastAPI, MQTT, SQLite, Docker (Alpine)

**Repo:** `C:\Users\peder\OneDrive\Dokumenter\claude code\Hass_Intelligence\`
**Add-on path:** `ha_intelligence/`
**App code:** `ha_intelligence/app/`

---

## Task 1: Add scikit-learn to requirements.txt (Fixes Phase 4c)

**Why:** `batch_trainer.py` is fully implemented but `from sklearn.ensemble import GradientBoostingClassifier` fails silently because scikit-learn isn't installed. This single-line fix enables the entire batch training pipeline.

**Files:**
- Modify: `ha_intelligence/requirements.txt`

**Step 1: Add scikit-learn dependency**

In `ha_intelligence/requirements.txt`, add `scikit-learn` after `river==0.23.0`:

```
scikit-learn>=1.3.0
```

**Step 2: Verify Dockerfile handles build deps**

The Dockerfile already has `gcc musl-dev g++ libffi-dev` which scikit-learn needs. No Dockerfile changes required.

**Step 3: Commit**

```bash
git add ha_intelligence/requirements.txt
git commit -m "fix: add scikit-learn to requirements (enables batch training)"
```

---

## Task 2: Debug Phase 3 — Priors Never Running

**Why:** `priors_calculated: 0` and `priors_last_run: null` despite code being present and asyncio task being registered. Need to find why.

**Files:**
- Modify: `ha_intelligence/app/priors.py` (lines 63-68, 180-239)
- Read: `ha_intelligence/app/database.py` (observations table)

**Step 1: Add diagnostic logging to nightly_job**

In `ha_intelligence/app/priors.py`, the `nightly_job()` method (line ~195) has a sleep loop. Add logging BEFORE and AFTER `calculate_all_priors()`:

```python
async def nightly_job(self):
    """Run nightly prior calculation at 03:00 UTC."""
    logger.info("Prior nightly job started — waiting for 03:00 UTC")
    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        logger.info(f"Priors: next run at {target.isoformat()}, waiting {wait_seconds/3600:.1f}h")
        await asyncio.sleep(wait_seconds)

        try:
            logger.info("Priors: starting nightly calculation...")
            result = self.calculate_all_priors()
            logger.info(f"Priors: calculation done — {result}")
        except Exception as e:
            logger.error(f"Priors: nightly calculation FAILED: {e}", exc_info=True)
```

**Step 2: Fix SQL double-percent escaping**

In `ha_intelligence/app/priors.py`, lines 63-68 likely have `%%H` and `%%w` for strftime. Python's `str.format()` or f-strings don't need double-percent, but `cursor.execute()` with `%s` placeholders does. Check if the query uses parameterized queries (which DO need `%%`) or f-strings (which DON'T).

Look at the actual query. If it uses `cursor.execute(query)` without params, change `%%H` → `%H` and `%%w` → `%w`.

If it uses `cursor.execute(query, params)`, keep `%%H` and `%%w`.

**Step 3: Lower MIN_DAYS_FOR_PRIORS for testing**

Temporarily change `MIN_DAYS_FOR_PRIORS = 7` to `MIN_DAYS_FOR_PRIORS = 1` in `priors.py` to verify the calculation works. Revert to 3 (not 7) once confirmed working.

**Step 4: Add observation count check**

Add a method to check if there's enough data before the nightly job waits:

```python
def has_enough_data(self) -> bool:
    """Check if we have enough observations for prior calculation."""
    count = self.db.get_observation_count()
    days = count / (24 * 4)  # rough estimate: ~4 obs/hour
    logger.info(f"Priors: {count} observations (~{days:.1f} days)")
    return days >= self.min_days
```

**Step 5: Commit**

```bash
git add ha_intelligence/app/priors.py
git commit -m "fix: add diagnostic logging to priors, fix SQL escaping"
```

---

## Task 3: Verify Markov Chain Integration (Phase 4a)

**Why:** `ml_markov_models: 0` in sensor stats but code is fully integrated. The stat shows `len(self.markov_models)` which is 0 if no person has changed rooms since the addon started. This might just be a timing issue — but we need to verify.

**Files:**
- Read: `ha_intelligence/app/ml_engine.py:55-96`
- Read: `ha_intelligence/app/models/model_manager.py:50-54`
- Potentially modify: `ha_intelligence/app/ml_engine.py`

**Step 1: Add debug logging to _record_markov_transition**

In `ha_intelligence/app/ml_engine.py`, line 88-96:

```python
def _record_markov_transition(self, person_entity: str,
                               from_room: str, to_room: str):
    """Record a room transition for Markov prediction."""
    try:
        hour = datetime.now(timezone.utc).hour
        markov = self.models.get_or_create_markov_model(person_entity)
        markov.record_transition(from_room, to_room, hour)
        logger.info(
            f"Markov transition: {person_entity} "
            f"{from_room} → {to_room} (hour={hour}, "
            f"total={markov.total_transitions})"
        )
    except Exception as e:
        logger.error(f"Markov transition error: {e}", exc_info=True)
```

Note: Change `logger.debug` to `logger.info` for the error too, and add `exc_info=True`.

**Step 2: Verify update_person_room is called**

In `ml_engine.py`, `update_person_room()` (line ~45) is what triggers Markov recording. Verify this is called from `main.py` when BLE room changes happen.

Search for calls to `ml_engine.update_person_room` in `main.py`. If it's called, transitions should be recording. If `ml_markov_models` is still 0, it means:
- No room transitions happened yet (all persons stayed in same room)
- OR `update_person_room` isn't being called

**Step 3: Commit**

```bash
git add ha_intelligence/app/ml_engine.py
git commit -m "fix: add Markov transition logging for debugging"
```

---

## Task 4: Wire NotificationEngine into main.py (Phase 6c)

**Why:** `NotificationEngine` in `notifications.py` is complete with rate limiting, quiet hours, anomaly/prediction/low-confidence notifications. But it's NEVER instantiated in `main.py`.

**Files:**
- Modify: `ha_intelligence/app/main.py`
- Read: `ha_intelligence/app/notifications.py`

**Step 1: Import NotificationEngine in main.py**

At the top imports section of `main.py`, add:

```python
from notifications import NotificationEngine
```

**Step 2: Instantiate in SensorEngine.__init__**

In the `SensorEngine.__init__()` method, after ML engine init, add:

```python
# Notification engine
self.notifications = NotificationEngine(
    options=self.options,
    mqtt_publisher=self.mqtt,
)
```

**Step 3: Add periodic notification check to main loop**

In the main processing loop (wherever the periodic tasks run), add notification checks:

```python
# Check for anomalies and low confidence — runs every 5 minutes
if self.notifications and self.ml_engine:
    self.notifications.check_anomalies(self.ml_engine)
    self.notifications.check_low_confidence(self._room_states)
```

This should be called from a periodic task, not on every event.

**Step 4: Add notification status to system sensor**

In the system sensor publish method, add notification stats:

```python
# In the attributes dict for hai_system sensor:
'notifications_sent_today': self.notifications._sent_today if self.notifications else 0,
'notifications_active': self.notifications.active if self.notifications else False,
```

**Step 5: Add notification API endpoint to web_ui**

In `web_ui.py`, add an endpoint:

```python
@app.get("/api/notifications")
async def get_notifications():
    if not engine or not engine.notifications:
        return {"error": "Notifications not available"}
    return engine.notifications.get_status()
```

**Step 6: Commit**

```bash
git add ha_intelligence/app/main.py ha_intelligence/app/web_ui.py
git commit -m "feat: wire NotificationEngine into main loop and web UI"
```

---

## Task 5: Version Bump to v0.7.0

**Why:** All ML systems are now functional. Bump version to reflect the milestone.

**Files:**
- Modify: `ha_intelligence/config.yaml` (line 2: version)
- Modify: `ha_intelligence/Dockerfile` (line 29: version label)
- Modify: `ha_intelligence/app/main.py` (VERSION constant)

**Step 1: Bump all version strings**

Change `"0.6.0"` to `"0.7.0"` in all three files.

**Step 2: Commit and push**

```bash
git add ha_intelligence/config.yaml ha_intelligence/Dockerfile ha_intelligence/app/main.py
git commit -m "release: v0.7.0 — fix ML pipeline, add notifications, add scikit-learn"
git push
```

**Step 3: Deploy via HA**

Use HA MCP:
1. `homeassistant.update_entity` on `update.ha_intelligence_opdatering`
2. Wait 10 sec
3. `update.install` on `update.ha_intelligence_opdatering`
4. Wait 60 sec
5. Verify via `ha_get_state` on `sensor.hai_system`

---

## Task 6: Verify All ML Systems After Deploy

**Why:** Need to confirm everything works after deployment.

**Steps:**

1. Check `sensor.hai_system` attributes:
   - `ml_markov_models` should increase as people move
   - `batch_room_models` should be 0 until 04:00 UTC
   - `priors_calculated` should be 0 until 03:00 UTC (or next nightly run)

2. Check add-on logs for:
   - "Markov transition:" log lines (confirms Task 3)
   - "Prior nightly job started" (confirms Task 2)
   - "scikit-learn not installed" should NOT appear (confirms Task 1)

3. Check notification status via `/api/notifications`

---

## Task 7: Frontend — Notification Dashboard Card (Phase 6c UI)

**Why:** Notifications need a visible section in the web UI dashboard.

**Files:**
- Modify: `ha_intelligence/app/web_ui.py`

**Step 1: Add notifications section to dashboard HTML**

In the dashboard HTML template in `web_ui.py`, add a Notifications card:

```html
<div class="card">
    <h2>🔔 Notifikationer</h2>
    <div id="notifications-status">Indlæser...</div>
</div>
```

**Step 2: Add JavaScript fetch for notifications**

```javascript
async function loadNotifications() {
    const resp = await fetch('/api/notifications');
    const data = await resp.json();
    const el = document.getElementById('notifications-status');
    el.innerHTML = `
        <p>Status: ${data.active ? 'Aktiv' : 'Inaktiv'}</p>
        <p>Sendt i dag: ${data.sent_today} / ${data.max_daily}</p>
        <p>Stille timer: ${data.quiet_hours} ${data.is_quiet ? '(aktiv nu)' : ''}</p>
        <p>Kan sende: ${data.can_send ? 'Ja' : 'Nej'}</p>
        <h3>Seneste</h3>
        <ul>${(data.history || []).map(h => `<li>${h[0].slice(0,16)} [${h[1]}] ${h[2]}</li>`).join('')}</ul>
    `;
}
```

**Step 3: Commit**

```bash
git add ha_intelligence/app/web_ui.py
git commit -m "feat: add notification dashboard card"
```

---

## Future Tasks (Post v0.7.0 — NOT in this plan)

### Phase 6a: Floorplan Editor
- SVG upload + drag-drop room placement
- Requires significant frontend work (canvas/SVG manipulation)
- Lower priority — the system works without visual floorplan

### Phase 6b: Chat Interface
- Haiku-powered Q&A about household state
- Depends on Haiku being enabled (currently disabled by default)
- Moderate effort — needs WebSocket for real-time chat

### Enhanced Anomaly Notifications
- Push notifications to HA mobile app (not just persistent_notification)
- Requires `notify.mobile_app_*` service integration

---

## Execution Order Summary

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 1 | Add scikit-learn | 1 min | Enables entire batch training |
| 2 | Debug priors | 15 min | Fixes Bayesian prior integration |
| 3 | Verify Markov | 10 min | Confirms movement prediction |
| 4 | Wire notifications | 20 min | Connects anomaly alerting |
| 5 | Version bump | 5 min | Release v0.7.0 |
| 6 | Verify deploy | 10 min | Confirm all systems |
| 7 | Notification UI | 15 min | Dashboard visibility |

**Total estimated effort: ~75 minutes**
