"""Cheap image-quality and text-plausibility measures used as confidence
proxies where an engine does not report its own confidence (OCR.space and
the free VLMs do not)."""
from __future__ import annotations

import re

import cv2
import numpy as np

# Ramps chosen from the score distribution of eval/fixtures (synthetic, so they
# are a starting point, not a universal truth): a blurred frame has a Laplacian
# variance of 1-5, an intact one 28 or more; a frame with text hidden in the
# background has a grey std-dev of 2-5, a dark room 7 or more.
SHARP_BAD, SHARP_GOOD = 3.0, 25.0         # variance of the Laplacian at 320 px
CONTRAST_BAD, CONTRAST_GOOD = 3.0, 8.0    # grey-level std-dev
BRIGHT_BAD, BRIGHT_GOOD = 12.0, 30.0      # mean grey level


def _ramp(x: float, bad: float, good: float) -> float:
    return float(min(1.0, max(0.0, (x - bad) / (good - bad))))


def image_quality(frame) -> dict:
    """Return sharpness / contrast / brightness and a 0..1 ``confidence``.

    The confidence is the *weakest* of the three, because one failure (a blur,
    a dark room) is enough to make OCR and scene description unreliable.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return {"sharpness": 0.0, "contrast": 0.0, "brightness": 0.0, "confidence": 0.0}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    s = 320.0 / max(h, w)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    contrast = float(np.std(small))
    bright = float(np.mean(small))
    conf = min(_ramp(sharp, SHARP_BAD, SHARP_GOOD),
               _ramp(contrast, CONTRAST_BAD, CONTRAST_GOOD),
               _ramp(bright, BRIGHT_BAD, BRIGHT_GOOD))
    return {"sharpness": sharp, "contrast": contrast, "brightness": bright, "confidence": conf}


_WORDY = re.compile(r"[A-Za-z0-9ऀ-ॿ\s.,:;!?%/()'\-+₹$]")


def text_plausibility(text: str) -> float:
    """0..1: share of characters that look like real text (letters, digits,
    Devanagari, common punctuation), discounted for very short output. OCR
    garbage ("|~^#@") scores low."""
    t = (text or "").strip()
    if not t:
        return 0.0
    ok = sum(1 for c in t if _WORDY.match(c)) / len(t)
    length = min(1.0, len(t) / 6.0)
    return ok * (0.6 + 0.4 * length)
