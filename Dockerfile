# Demo / evaluation image (light dependency set: no torch, no audio hardware; for the real camera, YOLO, Piper and Vosk install on your machine, see docs/SETUP_LOCAL.md). Containers usually have no camera or microphone, so the
# default command runs the synthetic demo. UNTESTED here (no Docker daemon in the
# authoring sandbox): build it once and report any failure.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 VISIONASSIST_HOME=/app
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libgl1 libglib2.0-0 libportaudio2 alsa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-ci.txt pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -r requirements-ci.txt && pip install -e . --no-deps
COPY . .

EXPOSE 8000
# Dashboard on all interfaces; set WEB_TOKEN if the port is reachable by others.
CMD ["visionassist", "run", "--ui", "web", "--demo", "--no-browser", "--host", "0.0.0.0"]
