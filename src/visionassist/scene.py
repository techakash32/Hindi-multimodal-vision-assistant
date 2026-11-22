"""Scene-change logic. Pure functions, no network, no hardware."""
from __future__ import annotations

import cv2
import numpy as np


def frame_signature(frame):
    """Create a lightweight signature for frame similarity."""
    small = cv2.resize(frame, (16, 16))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.astype("float32").flatten()


def signatures_similar(a, b, thresh=8.0):
    """Check whether two frame signatures are sufficiently similar."""
    if a is None or b is None:
        return False

    return float(np.mean(np.abs(a - b))) < thresh



# ==================================================
# "SAME SCENE?" CHECK (uses YOLO results, no network)
# ==================================================

def detection_signature(current, frame_shape):
    """
    Coarse, hashable summary of what YOLO sees: a set of
    (class, left/centre/right, far/mid/near). Coarse on purpose so that a
    little hand shake does not look like a new scene.
    `current` = {cls: (conf, xyxy, xc, area_ratio)} from Detector.detect().
    """
    _h, w = frame_shape[:2]
    sig = set()

    for cls_name, (_conf, _xyxy, xc, area) in (current or {}).items():
        zone = "L" if xc < w * 0.33 else "R" if xc > w * 0.66 else "C"
        dist = "near" if area > 0.15 else "mid" if area > 0.05 else "far"
        sig.add((cls_name, zone, dist))

    return frozenset(sig)


def scene_meaningfully_changed(prev_sig, cur_sig):
    """Returns (changed, added, removed). Sets of (class, zone, dist)."""
    if prev_sig is None:
        return True, set(cur_sig or ()), set()

    added = set(cur_sig) - set(prev_sig)
    removed = set(prev_sig) - set(cur_sig)
    return bool(added or removed), added, removed


def scene_is_same(prev, cur_det_sig, cur_pix, now,
                  max_age_s, pixel_thresh):
    """
    True only if ALL of these hold, so a blind user is never told "same"
    when it may not be:
      - a previous description exists and is younger than max_age_s
      - YOLO sees the same objects in the same zones/distances
      - the picture itself still looks alike (an empty wall has no
        detections, so detections alone cannot prove "same")
    `prev` = {"det": frozenset, "pix": array, "ts": float} or None.
    """
    if not prev:
        return False

    if now - prev["ts"] >= max_age_s:
        return False

    changed, _a, _r = scene_meaningfully_changed(prev["det"], cur_det_sig)
    if changed:
        return False

    return signatures_similar(prev["pix"], cur_pix, thresh=pixel_thresh)
