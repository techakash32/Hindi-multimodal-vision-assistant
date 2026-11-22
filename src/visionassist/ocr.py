"""OCR of a camera frame through the OCR.space API."""
from __future__ import annotations

import csv
import io
import logging
import shutil
import subprocess
from typing import Optional

import cv2
import requests

from . import config as cfg
from .net import SESSION
from .resilience import CircuitBreaker

log = logging.getLogger(__name__)


# After 3 network/HTTP failures, stop calling OCR for 60 s.
_ocr_breaker = CircuitBreaker("OCR", failure_threshold=3, cooldown_sec=60)


class OcrText(str):
    """OCR output. Behaves exactly like the plain string every caller already
    handles, and additionally carries the engine's own confidence (0..1) when
    the engine reports one. OCR.space does not, so its ``confidence`` is None
    and the pipeline falls back to a text-plausibility proxy."""

    confidence: Optional[float] = None
    engine: str = ""

    def __new__(cls, text: str, confidence: Optional[float] = None, engine: str = ""):
        obj = super().__new__(cls, text)
        obj.confidence = confidence
        obj.engine = engine
        return obj


def tesseract_available() -> bool:
    return shutil.which(cfg.TESSERACT_CMD) is not None


def read_text_tesseract(frame):
    """Local OCR through the ``tesseract`` binary (no network, no Python
    package). Returns ``OcrText`` whose ``confidence`` is the mean word
    confidence reported by Tesseract, scaled to 0..1."""
    if frame is None or frame.size == 0:
        return "OCR error: Empty image."
    if not tesseract_available():
        return "OCR error: tesseract is not installed."
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    if max(h, w) < 1000:      # Tesseract likes ~30 px glyphs
        k = 1000 / max(h, w)
        gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        return "OCR error: Could not encode image."
    try:
        proc = subprocess.run(
            [cfg.TESSERACT_CMD, "stdin", "stdout", "--psm", str(cfg.TESSERACT_PSM), "-l", cfg.TESSERACT_LANG, "tsv"],
            input=buf.tobytes(), capture_output=True, timeout=cfg.OCR_TIMEOUT * 4, check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return f"OCR error: {exc}"
    words, confs = [], []
    reader = csv.DictReader(io.StringIO(proc.stdout.decode("utf-8", "replace")), delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        word = (row.get("text") or "").strip()
        try:
            conf = float(row.get("conf", "-1"))
        except ValueError:
            continue
        if word and conf >= 0:
            words.append(word)
            confs.append(conf / 100.0)
    if not words:
        return "No text found. Try pointing closer."
    return OcrText(" ".join(words), sum(confs) / len(confs), "tesseract")


def read_text_from_frame(frame):
    """Crop, resize, preprocess and extract text using OCR.space (default)
    or the local Tesseract engine (``OCR_ENGINE=tesseract``)."""

    if cfg.OCR_ENGINE == "tesseract" or cfg.LOCAL_ONLY:
        return read_text_tesseract(frame)

    if frame is None or frame.size == 0:
        return "OCR error: Empty image."

    if not _ocr_breaker.allow():
        # Keep the "OCR error" prefix: main.py uses it to pick the message.
        return "OCR error: service paused after repeated failures."

    # Center-crop 70% to focus on text in the center.
    h, w = frame.shape[:2]
    crop_w = max(1, int(w * 0.7))
    crop_h = max(1, int(h * 0.7))

    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    frame = frame[y1:y1 + crop_h, x1:x1 + crop_w]

    # Keep OCR image dimensions within a practical range.
    h, w = frame.shape[:2]
    longest = max(h, w)

    if longest > 1400:
        scale = 1400 / longest
        frame = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    elif longest < 1000:
        scale = 1000 / longest
        frame = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    # Grayscale and contrast enhancement.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)

    # Adaptive thresholding improves text contrast.
    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    success, buffer = cv2.imencode(".png", thresholded)

    if not success:
        return "OCR error: Could not encode image."

    files = {
        "file": ("frame.png", buffer.tobytes(), "image/png")
    }

    data = {
        "apikey": cfg.OCR_API_KEY,
        "language": "eng",
        "OCREngine": 2,
        "scale": True,
        "isOverlayRequired": False,
    }

    try:
        response = SESSION.post(
            cfg.OCR_URL,
            files=files,
            data=data,
            timeout=cfg.OCR_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()

        # The service answered; processing errors are not an outage.
        _ocr_breaker.record_success()

        if result.get("IsErroredOnProcessing"):
            errors = result.get(
                "ErrorMessage", ["Unknown OCR error"]
            )

            if isinstance(errors, list):
                errors = "; ".join(map(str, errors))

            return f"OCR error: {errors}"

        parsed = result.get("ParsedResults") or []

        text = " ".join(
            item.get("ParsedText", "")
            for item in parsed
        ).strip()

        return OcrText(text, None, "ocr.space") if text else "No text found. Try pointing closer."

    except (requests.RequestException, ValueError) as exc:
        _ocr_breaker.record_failure()
        return f"OCR error: {exc}"


def breaker_state() -> str:
    return "open" if _ocr_breaker.is_open else "closed"
