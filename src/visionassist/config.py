"""Central configuration.

Every setting has a sensible default and can be overridden from the
environment or a ``.env`` file (see ``.env.example``). Modules read values as
``config.NAME`` at call time, so tests can monkeypatch them.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


def _str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


# ---- Paths (relative to VISIONASSIST_HOME, default: current directory) ----
HOME_DIR = Path(_str("VISIONASSIST_HOME", str(Path.cwd()))).resolve()
VOICE_DIR = HOME_DIR / "voices"
MODEL_DIR = HOME_DIR / "models"
OUTPUT_DIR = HOME_DIR / "output"
LOG_DIR = HOME_DIR / "logs"

VOICE_MODEL = VOICE_DIR / "en_US-lessac-medium.onnx"
HINDI_VOICE_MODEL = VOICE_DIR / "hi_IN-priyamvada-medium.onnx"
YOLO_MODEL = MODEL_DIR / "yolov8n.pt"
LOG_LEVEL = _str("LOG_LEVEL", "INFO")

# ---- Camera / detection ----
CAMERA_INDEX = _int("CAMERA_INDEX", 0)
CAMERA_WIDTH = _int("CAMERA_WIDTH", 640)
CAMERA_HEIGHT = _int("CAMERA_HEIGHT", 480)
CONF_THRESH = _float("CONF_THRESH", 0.60)
INFER_EVERY = _int("INFER_EVERY", 2)          # run YOLO every N frames
YOLO_IMGSZ = _int("YOLO_IMGSZ", 320)          # 640 = more accurate, slower

# ---- Alert stability (see detector.py) ----
NAVIGATION_ONLY = _bool("NAVIGATION_ONLY", True)
MIN_SPEAK_CONF = _float("MIN_SPEAK_CONF", 0.70)
EMA_ALPHA = _float("EMA_ALPHA", 0.4)
STABLE_FRAMES = _int("STABLE_FRAMES", 3)
MIN_SPEAK_AREA = _float("MIN_SPEAK_AREA", 0.02)
MIN_SPEAK_AREA_SMALL = _float("MIN_SPEAK_AREA_SMALL", 0.008)
REANNOUNCE_AFTER_S = _float("REANNOUNCE_AFTER_S", 2.0)

# ---- Speech-to-text (Vosk) ----
# Default is the small Hindi model, so commands are spoken in Hindi.
VOSK_MODEL_PATH = _str("VOSK_MODEL", str(MODEL_DIR / "vosk-model-small-hi-0.22"))
_mic = _str("VOICE_MIC_DEVICE", "")
VOICE_MIC_DEVICE = int(_mic) if _mic.isdigit() else None
VOICE_MIC_NAMES = [
    n.strip() for n in _str("VOICE_MIC_NAMES", "OMEN Cam,Microphone Array").split(",")
    if n.strip()
]
MIN_CMD_CONF = _float("MIN_CMD_CONF", 0.5)
VOICE_ENABLED = _bool("VOICE_ENABLED", True)

# ---- Language / speech ----
# Hindi only: everything the user hears or reads is Hindi. English remains
# only in logs and developer tooling.
HINDI_ONLY = _bool("HINDI_ONLY", True)
USE_HINDI = True
USE_HINDI_ALERTS = True
VLM_REPLY_IN_HINDI = True
USE_SONIFY_FOR_VLM = _bool("USE_SONIFY_FOR_VLM", False)
HINDI_EMOTION = _str("HINDI_EMOTION", "WARM")
OCR_EMOTION = _str("OCR_EMOTION", "NEUTRAL")
USE_EMOTION_TAGS = _bool("USE_EMOTION_TAGS", True)
OCR_MAX_CHARS = _int("OCR_MAX_CHARS", 300)
TTS_ENABLED = _bool("TTS_ENABLED", True)

# ---- Scene description ----
SCENE_SAME_MAX_AGE_S = _float("SCENE_SAME_MAX_AGE_S", 30.0)
SCENE_PIXEL_THRESH = _float("SCENE_PIXEL_THRESH", 25.0)
# A free VLM must never imply "it is safe", so "path clear" is off by default.
ANNOUNCE_PATH_CLEAR = _bool("ANNOUNCE_PATH_CLEAR", False)

# ---- API keys ----
OCR_API_KEY = _str("OCR_API_KEY", "")
OPENROUTER_API_KEY = _str("OPENROUTER_API_KEY", "")
CARTESIA_API_KEY = _str("CARTESIA_API_KEY", "")   # optional

# ---- Privacy: LOCAL_ONLY=true never sends a frame or text off the device ----
# OCR uses local Tesseract, cloud scene description and translation are disabled.
LOCAL_ONLY = _bool("LOCAL_ONLY", False)

# ---- OCR engine: "ocrspace" (network, default) or "tesseract" (local binary) ----
OCR_ENGINE = _str("OCR_ENGINE", "ocrspace").lower()
TESSERACT_CMD = _str("TESSERACT_CMD", "tesseract")
TESSERACT_LANG = _str("TESSERACT_LANG", "eng")
TESSERACT_PSM = _int("TESSERACT_PSM", 6)

# ---- API endpoints / timeouts ----
OCR_URL = "https://api.ocr.space/parse/image"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OCR_TIMEOUT = _float("OCR_TIMEOUT", 8)
VLM_TIMEOUT = _float("VLM_TIMEOUT", 10)
# Comma-separated OpenRouter model ids, in order of preference (optional).
VLM_MODELS = [m.strip() for m in _str("VLM_MODELS", "").split(",") if m.strip()]
# If the preferred VLM has not answered after this long, also ask the next one.
HEDGE_DELAY_S = _float("HEDGE_DELAY_S", 5.0)

# ---- Web dashboard ----
WEB_HOST = _str("WEB_HOST", "127.0.0.1")
WEB_PORT = _int("WEB_PORT", 8000)
WEB_TOKEN = _str("WEB_TOKEN", "")   # if set, every request needs ?token=...
STREAM_FPS = _int("STREAM_FPS", 15)
JPEG_QUALITY = _int("JPEG_QUALITY", 80)
