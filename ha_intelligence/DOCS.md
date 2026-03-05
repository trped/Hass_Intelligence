# HA Intelligence

Et selv-lærende husstandsintelligens-system til Home Assistant.

## Hvad gør det?

HA Intelligence lytter på **alle** state-ændringer i dit Home Assistant system og lærer automatisk mønstre over tid. Det publicerer virtuelle sensorer der beskriver:

- **Rum-tilstand** (`sensor.hai_room_*`) — Er rummet optaget? Hvem er der?
- **Person-tilstand** (`sensor.hai_person_*`) — Er personen hjemme? Aktiv? Sover?
- **System-status** (`sensor.hai_system`) — Systemets sundhed og statistik

## Installation

1. Tilføj dette repository som custom add-on repository i Home Assistant
2. Installer "HA Intelligence" fra add-on store
3. Konfigurér MQTT-forbindelse (typisk `core-mosquitto`)
4. Start add-on'et

## Konfiguration

| Option | Standard | Beskrivelse |
|--------|----------|-------------|
| `mqtt_host` | `core-mosquitto` | MQTT broker hostname |
| `mqtt_port` | `1883` | MQTT broker port |
| `mqtt_user` | *(tom)* | MQTT brugernavn |
| `mqtt_password` | *(tom)* | MQTT password |
| `haiku_api_key` | *(tom)* | Anthropic API key til Claude Haiku |
| `haiku_enabled` | `false` | Aktivér AI-assistance |
| `feedback_active` | `true` | Send feedback-notifikationer |
| `feedback_max_daily` | `3` | Max feedback-spørgsmål per dag |
| `log_level` | `info` | Log niveau (debug/info/warning/error) |

## Sensorer

### Rum-sensorer
`sensor.hai_room_[slug]`

States: `occupied`, `empty`, `active`, `quiet`

Attributter:
- `motion_sensors` — antal bevægelsessensorer i rummet
- `active_sensors` — antal aktive sensorer
- `last_occupied` — seneste bevægelse
- `confidence` — systemets sikkerhed (0.0-1.0)

### Person-sensorer
`sensor.hai_person_[slug]`

States: `active`, `idle`, `sleeping`, `away`

Attributter:
- `home` — er personen hjemme
- `location` — nuværende lokation
- `room` — hvilket rum (når ML er aktiv)
- `confidence` — systemets sikkerhed (0.0-1.0)

## Roadmap

- **v0.1** — Event indsamling, sensor discovery, basale sensorer
- **v0.2** — Online ML (River), mønstergenkendelse
- **v0.3** — Claude Haiku integration, aktiv feedback
- **v0.4** — Anomali-detektion, energi-optimering
