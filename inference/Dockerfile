# LogiEdge Inference Service — Dockerfile (Task D2)
# Base image: python:3.11-slim as specified in the assignment
# pip install layers BEFORE model COPY for OTA layer-cache efficiency

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies FIRST (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference service code (changes less often than model)
COPY inference_service.py .

# Copy model file LAST (changes most often — separate layer)
# This ensures that when only the model changes, only this layer rebuilds
COPY model.tflite .

# Environment variables for runtime configuration
# MODEL_PATH can be overridden to switch model variants without rebuild
ENV MODEL_PATH=/app/model.tflite
ENV TRUCK_ID=TRUCK_01
ENV MQTT_BROKER=host.docker.internal
ENV MQTT_PORT=1883
ENV OPS_MQTT_BROKER=
ENV OPS_MQTT_PORT=1883
ENV ALERT_DB_PATH=/opt/logibridge/alerts.db

# Run the inference service
CMD ["python", "inference_service.py"]
