ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install system dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3-dev \
    mosquitto-clients

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy application
COPY rootfs /
COPY app /app

# Set working directory
WORKDIR /app

# Labels
LABEL \
    io.hass.name="HA Intelligence" \
    io.hass.description="Self-learning household intelligence system" \
    io.hass.type="addon" \
    io.hass.version="0.1.0"

CMD [ "/run.sh" ]
