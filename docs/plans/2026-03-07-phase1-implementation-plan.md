# Phase 1 Implementation Plan: Person-Room BLE Tracking (v0.4.0)

**Dato:** 2026-03-07
**Baseline:** v0.3.4
**Target:** v0.4.0
**Estimat:** ~2 dage

## Oversigt

Tilføj person-rum lokalisering via Bermuda BLE integration. Hver persons
`sensor.hai_person_*` sensor skal vise hvilket rum personen er i, med
confidence og source. Motion fallback når BLE er utilgængelig.

## Bermuda Entity Mapping

| Person | Area Sensor | Type |
|--------|-------------|------|
| Troels | `sensor.bermuda_aff9c3e51e7943268af745c50577d7a0_100_40004_area_last_seen` | Bermuda |
| Hanne | `sensor.hanne_iphone_area` | Bermuda |
| Darwin | `sensor.darwin_iphone_area` | Bermuda |
| Maria | `sensor.maria_iphone_area` | Bermuda |

Bermuda area sensorer har `area_id` attribut der mapper direkte til HA areas.

## Fil-ændringer

### 1. `config.yaml` — Ny bermuda_sensors option

Tilføj `bermuda_sensors` som JSON-string option der mapper person entity → bermuda sensor.

```yaml
options:
  bermuda_sensors: '{"person.troels":"sensor.bermuda_aff9c3e51e7943268af745c50577d7a0_100_40004_area_last_seen","person.hanne":"sensor.hanne_iphone_area","person.darwin":"sensor.darwin_iphone_area","person.maria":"sensor.maria_iphone_area"}'

schema:
  bermuda_sensors: str?
```

### 2. `app/main.py` — BLE Person-Room Tracking

**Nye data-strukturer (linje ~78):**
```python
self._person_rooms = {}      # person_entity -> {area_id, area_name, confidence, source, distance, updated_at}
self._bermuda_to_person = {} # bermuda_sensor_entity -> person_entity
```

**Nye metoder:**
- `_init_bermuda_mapping(options)` — parse bermuda_sensors JSON fra options, byg reverse mapping
- `_update_person_room(person_entity, area_id, area_name, confidence, source, distance)` — opdater `_person_rooms`
- `_infer_room_from_motion(person_entity)` — motion fallback: find rum med seneste motion + person home

**Ændringer i `on_state_change()`:**
- Fang Bermuda area sensor state changes (sensor domain, entity i `_bermuda_to_person`)
- Extract `area_id` fra state/attributes → kald `_update_person_room()`

**Ændringer i `_publish_persons()`:**
- Hent room data fra `_person_rooms` for hver person
- Tilføj attributter: `room`, `room_id`, `room_confidence`, `room_source`, `ble_distance`
- Motion fallback: Hvis BLE unavailable/unknown, inferér fra rum med seneste motion

**Ændringer i `_publish_system()`:**
- Bump version string til "0.4.0"

**Ændringer i `main()`:**
- Parse bermuda_sensors fra options og send til SensorEngine

### 3. `app/features.py` — BLE Person Features

**Ændringer i `extract_person_features()`:**
Tilføj nye features fra person_state dict:
- `ble_area_known`: 1 hvis BLE area er known, 0 ellers
- `ble_distance`: distance i meter (0.0 hvis unknown)
- `minutes_in_current_room`: minutter i nuværende rum (0 hvis unknown)

### 4. `app/mqtt_publisher.py` — Fix stale version

- Opdater `sw_version` fra "0.3.2" til dynamisk (eller "0.4.0")

### 5. `app/ml_engine.py` — BLE training data

**Ændringer i `on_state_change()`:**
- Bermuda sensor state changes → opdater person context med BLE room data
- Gør BLE room tilgængelig for person feature extraction

**Ændringer i `_train_person()`:**
- Inkluder BLE room info i person_state for feature extraction

## Verifikation

1. `sensor.hai_person_hanne` har `room: "Køkken"` når Bermuda siger kokken
2. Room skifter inden 60s når person bevæger sig
3. Fallback til "unknown" når BLE unavailable
4. ML features inkluderer `ble_area_known`, `ble_distance`
5. System sensor viser version "0.4.0"

## Implementeringsrækkefølge

1. config.yaml (bermuda_sensors option)
2. main.py (BLE tracking + room publishing)
3. features.py (BLE features)
4. mqtt_publisher.py (version fix)
5. ml_engine.py (BLE i training)
6. Version bump (config.yaml, main.py)
