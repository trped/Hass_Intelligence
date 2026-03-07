ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install build dependencies for River (C extensions), compile, then clean up
COPY requirements.txt /tmp/
RUN apk add --no-cache --virtual .build-deps \
        gcc musl-dev g++ libffi-dev \
    && pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && apk del .build-deps

# Create model persistence directory
RUN mkdir -p /data/models

# Copy run script
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# Copy application
COPY app /app

# Set working directory
WORKDIR /app

# Labels
LABEL \
    io.hass.name="HA Intelligence" \
    io.hass.description="Self-learning household intelligence system" \
    io.hass.type="addon" \
    io.hass.version="0.5.1"

CMD ["python3", "/app/main.py"]
