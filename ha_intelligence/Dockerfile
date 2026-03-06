ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

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
    io.hass.version="0.2.0"

CMD ["python3", "/app/main.py"]
