#!/usr/bin/with-contenv bashio

bashio::log.info "Starting HA Intelligence v0.1.6..."

# Export options as environment variables
export MQTT_HOST="$(bashio::config 'mqtt_host')"
export MQTT_PORT="$(bashio::config 'mqtt_port')"
export MQTT_USER="$(bashio::config 'mqtt_user')"
export MQTT_PASSWORD="$(bashio::config 'mqtt_password')"
export LOG_LEVEL="$(bashio::config 'log_level')"
export HAIKU_API_KEY="$(bashio::config 'haiku_api_key')"
export HAIKU_ENABLED="$(bashio::config 'haiku_enabled')"
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"

bashio::log.info "MQTT: ${MQTT_HOST}:${MQTT_PORT}"
bashio::log.info "Log level: ${LOG_LEVEL}"

exec python3 /app/main.py
