ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install scikit-learn from Alpine repos (pre-compiled, avoids 10+ min build)
# Then install remaining Python deps via pip
COPY requirements.txt /tmp/
RUN apk add --no-cache py3-scikit-learn \
    && apk add --no-cache --virtual .build-deps \
        gcc musl-dev g++ libffi-dev \
    && pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt \
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
    io.hass.version="0.5.0"

CMD ["python3", "/app/main.py"]
