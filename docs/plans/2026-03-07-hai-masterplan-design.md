# HAI Masterplan Design — Fra v0.3.4 til Fuld Vision

**Dato:** 2026-03-07
**Status:** Godkendt
**Baseline:** v0.3.4 (68% accuracy, GaussianNB, publish-time feedback)

## Udgangspunkt

HAI v0.3.4 har:
- Event pipeline fra HA WebSocket
- River GaussianNB online ML for rum (occupied/empty) og person (active/away)
- Publish-time feedback loop med ~68% accuracy
- 5 MQTT sensorer: room, person, system, time_context, household
- Feature extraction med rig kontekst (lys, media, klima) — men ML bruger kun motion count + tid
- Model persistence via joblib
- Web UI med basic dashboard

HAI v0.3.4 mangler:
- Person-rum lokalisering (altid "unknown")
- BLE/Bermuda integration
- EPL zone data som ML features
- TV/lys/CO2 sensor evidence i ML
- State priors (nattelig beregning)
- Evidence weights (konfigurerbar)
- Markov Chain bevægelsesprediction
- Isolation Forest anomalidetektion
- Claude Haiku integration
- Floorplan editor
- Chat interface
- Actionable notifications
- scikit-learn batch training

## Arkitektur-tilgang

**Valgt:** Inkrementel forbedring med River online ML som rygrad + scikit-learn batch som supplement i Phase 4.

**Begrundelse:**
- Lavest risiko — hvert phase efterlader system kørende
- `features.py` har allerede rig kontekst-extraction der bare skal wires til ML
- River GaussianNB er testet og fungerer
- scikit-learn tilføjes som nattelig batch retraining når historisk data er akkumuleret

## Bermuda BLE Setup (Eksisterende)

| Person | Area Sensor | Distance Sensor | Proxies |
|--------|-------------|-----------------|---------|
| Troels | `sensor.bermuda_..._area_last_seen` | `sensor.bermuda_..._distance` | 6 EPL |
| Hanne | `sensor.hanne_iphone_area` | `sensor.hanne_iphone_distance` | 6 EPL |
| Darwin | `sensor.darwin_iphone_area` | `sensor.darwin_iphone_distance` | 6 EPL |
| Maria | `sensor.maria_iphone_area` | `sensor.maria_iphone_distance` | 6 EPL |

Bermuda area sensorer har `area_id` attribut der mapper direkte til HA areas.

## Phase 1: Person-Room Tracking (v0.4.0)

**Mål:** `sensor.hai_person_troels` viser `room: "Køkken"` med confidence

**Ændringer:**

`config.yaml` — Nye options:
```yaml
bermuda_sensors:
  person.troels: sensor.bermuda_aff9c3e51e7943268af745c50577d7a0_100_40004_area_last_seen
  person.hanne: sensor.hanne_iphone_area
  person.darwin: sensor.darwin_iphone_area
  person.maria: sensor.maria_iphone_area
```

`app/main.py`:
- Nyt `_person_rooms` dict: `{person_entity: {area_id, area_name, confidence, source, distance, updated_at}}`
- `on_state_change()`: Fang Bermuda area sensor ændringer → opdater `_person_rooms`
- `_publish_persons()`: Tilføj room attributter fra `_person_rooms`
- Motion fallback: Hvis BLE unavailable, inferér fra rum med seneste motion + person home

`app/features.py`:
- `extract_person_features()`: Tilføj `ble_area_known`, `ble_distance`, `ble_rssi`, `minutes_in_current_room`

`app/mqtt_publisher.py`:
- `publish_person()`: Nye attributter: `room`, `room_id`, `room_confidence`, `room_source`, `ble_distance`

**Verifikation:**
- `sensor.hai_person_hanne` har `room: "Køkken"` når Bermuda siger kokken
- Room skifter inden 60s når person bevæger sig
- Fallback til "unknown" når BLE unavailable

## Phase 2: Rich Evidence Integration (v0.4.1)

**Mål:** ML bruger alle tilgængelige sensor-data, ikke kun motion count + tid

**Nøgle-indsigt:** `features.py._get_room_context()` udtrækker allerede 20+ features (lys, media, klima, sensorer, kontakter) — men `ml_engine._train_room()` bygger et minimalt room_state dict og ignorerer konteksten.

**Ændringer:**

`app/ml_engine.py`:
- `_train_room()`: Brug fuldt room_state med alle entities fra `_context_states`
- `on_state_change()`: Alle entity state changes (ikke kun motion/person) opdaterer allerede `_context_states` — men nu bruger ML dem aktivt

`app/features.py`:
- Tilføj EPL zone features: `epl_zone_1_occupied`..`epl_zone_4_occupied`, `epl_target_count`, `epl_assumed_present`
- Tilføj TV power: `tv_power_watts` (via power sensor)
- Tilføj CO2: `co2_ppm` (hvis tilgængelig i rummet)

`app/main.py`:
- Ny `evidence_weights` dict i system_config:
  ```json
  {"motion": 1.0, "ble": 0.9, "epl_zone": 0.8, "media": 0.7, "light": 0.5, "climate": 0.3}
  ```
- `_publish_rooms()`: Nye attributter: `evidence_sources`, `active_evidence_count`, `evidence_detail`

**Verifikation:**
- ML feature vector vokser fra ~7 til ~30+ features
- Accuracy stiger (forventet: 75-85%)
- Room sensor viser `evidence_sources: ["motion", "light", "media"]`

## Phase 3: State Priors & Nattelig Beregning (v0.4.2)

**Mål:** P(occupied | hour=14, weekday=3) = 0.72

**Nye filer:**

`app/priors.py` — `PriorCalculator`:
- `calculate_all_priors()`: Aggregér observations: GROUP BY target_type, target_id, hour, weekday → beregn frekvens
- `nightly_job()`: Async task kl. 03:00 → kald `calculate_all_priors()` → gem via `db.upsert_state_prior()`
- Minimum 7 dages data før priors bruges

**Ændringer:**

`app/main.py`:
- Tilføj nattelig prior task til main() event loop
- `_publish_rooms()`: Inkluder prior i confidence beregning

`app/ml_engine.py`:
- `predict_room()`: Kombinér ML + prior:
  ```
  final = 0.7 * ml_confidence + 0.3 * prior_probability
  ```

`app/web_ui.py`:
- Nyt endpoint: `/api/priors` med time×weekday heatmap data

**Verifikation:**
- `state_priors` tabel populeres efter 7+ dages drift
- Prior-vægtede predictions vises som attribut på room sensors
- Web UI viser prior heatmap

## Phase 4: Avancerede ML Modeller (v0.5.0)

**Mål:** Bevægelsesprediction, anomalidetektion, batch retraining

### 4a: Markov Chain Movement Prediction

`app/models/markov_chain.py` — `MovementPredictor`:
- Transition matrix: P(next_room | current_room, time_bucket)
- Time buckets: morgen/dag/aften/nat (4 perioder)
- Online update ved hvert room-skift (fra Phase 1 data)
- `predict_next(person, current_room, time_bucket)` → [(room, probability), ...]
- Persistence: JSON (simpel dict-serialisering)

Nye attributter på `sensor.hai_person_*`:
- `predicted_next_room`: "Soveværelset"
- `predicted_next_probability`: 0.65

### 4b: Anomaly Detection

`app/models/anomaly_model.py` — `AnomalyDetector`:
- River `HalfSpaceTrees` (online Isolation Forest variant)
- Trænes på room features (samme som GaussianNB)
- Anomaly score: 0.0 (normal) → 1.0 (anomali)
- Threshold: 0.7 for alert

Nye attributter på `sensor.hai_system`:
- `anomalies_detected`: 0
- `latest_anomaly`: "Usædvanlig aktivitet i køkkenet kl. 03:14"

### 4c: scikit-learn Batch Retraining

`app/models/batch_trainer.py` — `BatchTrainer`:
- `GradientBoostingClassifier` fra scikit-learn
- Nattelig retraining kl. 04:00 (efter priors kl. 03:00)
- Input: Alle observations fra seneste 14 dage
- Model comparison: sammenlign online vs. batch accuracy på seneste 24h
- Vinder bruges som primary, taber som fallback

`requirements.txt`:
- Tilføj `scikit-learn==1.4.2`

`Dockerfile`:
- scikit-learn har C dependencies → evt. tilføj build deps

**Verifikation:**
- Markov predictions vises på person sensorer
- Anomaly score vises på system sensor
- Batch model træner natligt og accuracy sammenlignes

## Phase 5: Claude Haiku Integration (v0.6.0)

**Mål:** Naturlig sproglig beskrivelse af husstandens tilstand

`app/haiku_engine.py` — `HaikuEngine`:
- Claude API kald med struktureret prompt
- Input: Alle sensor states + ML predictions + priors + person rooms
- Output: Kort naturlig beskrivelse
  - "Hanne laver aftensmad i køkkenet. Darwin er på sit værelse. Troels er ikke hjemme endnu."
- Rate limiting: Max 1 kald per 5 minutter
- Change detection: Kun kald API ved signifikante state ændringer
- Fallback: Template-baseret beskrivelse hvis API fejler/mangler

`config.yaml`:
- Ny option: `anthropic_api_key` (valgfri)

`app/mqtt_publisher.py`:
- Ny: `publish_summary()` → `sensor.hai_summary`

`app/main.py`:
- Ny publish task: Haiku summary hvert 5. minut (kun hvis API key konfigureret)

**Verifikation:**
- `sensor.hai_summary` viser naturlig sproglig beskrivelse
- Opdateres ved signifikante state changes
- Graceful degradation uden API key

## Phase 6: Frontend & Notifikationer (v0.7.0)

**Mål:** Interaktivt dashboard med floorplan, chat, smart notifications

### 6a: Floorplan Editor
- Upload SVG/billede af plantegning
- Drag-drop sensorer og rum-zoner
- Real-time overlay med occupancy status
- Endpoints: `/api/floorplan/config`, `/api/floorplan/status`

### 6b: Chat Interface
- Simpelt Q&A interface i web UI
- Bruger Haiku til at besvare spørgsmål
- "Hvem er hjemme?" → struktureret svar fra ML states
- "Hvornår kom Darwin hjem?" → query observations/events

### 6c: Actionable Notifications
- Anomali-notifikationer via MQTT → HA persistent_notification
- Prediction-baserede notifikationer ("Hanne plejer at komme hjem om 15 min")
- Konfigurerbar via web UI (hvilke notifications, thresholds)

**Verifikation:**
- Web UI har floorplan med live status
- Chat besvarer simple spørgsmål
- HA modtager notifications ved anomalier

## Filstruktur efter alle phases

```
app/
  main.py              [ÆNDRET] Phase 1-6
  database.py          [UÆNDRET] Allerede forberedt med alle tabeller
  features.py          [ÆNDRET] Phase 1-2 (BLE + EPL features)
  ml_engine.py         [ÆNDRET] Phase 1-4 (person room, evidence, priors, batch)
  mqtt_publisher.py    [ÆNDRET] Phase 1-6 (nye attributter + sensorer)
  web_ui.py            [ÆNDRET] Phase 3-6 (priors, floorplan, chat)
  priors.py            [NY] Phase 3
  haiku_engine.py      [NY] Phase 5
  models/
    __init__.py        [UÆNDRET]
    room_model.py      [UÆNDRET]
    person_model.py    [UÆNDRET]
    model_manager.py   [ÆNDRET] Phase 4 (nye model typer)
    markov_chain.py    [NY] Phase 4a
    anomaly_model.py   [NY] Phase 4b
    batch_trainer.py   [NY] Phase 4c
  registry.py          [UÆNDRET]
  event_listener.py    [UÆNDRET]
  discovery.py         [UÆNDRET]
Dockerfile             [ÆNDRET] Phase 4 (scikit-learn deps)
requirements.txt       [ÆNDRET] Phase 4-5 (scikit-learn, anthropic)
config.yaml            [ÆNDRET] Alle phases (version bumps + nye options)
```

## Tidsestimat

| Phase | Version | Estimat | Kumulativ |
|-------|---------|---------|-----------|
| Phase 1: Person-Room BLE | v0.4.0 | ~2 dage | 2 dage |
| Phase 2: Evidence Integration | v0.4.1 | ~1 dag | 3 dage |
| Phase 3: State Priors | v0.4.2 | ~1 dag | 4 dage |
| Phase 4: Advanced ML | v0.5.0 | ~3 dage | 7 dage |
| Phase 5: Haiku | v0.6.0 | ~2 dage | 9 dage |
| Phase 6: Frontend | v0.7.0 | ~3 dage | 12 dage |
