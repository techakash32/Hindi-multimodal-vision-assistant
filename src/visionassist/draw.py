"""Drawing helpers shared by the OpenCV window and the web stream."""
from __future__ import annotations

from typing import Dict, Mapping

import cv2
import numpy as np

# Black and white only. Line weight encodes proximity: near = thick, mid = medium, far = thin.
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
THICKNESS = {"near": 5, "mid": 3, "far": 1}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def proximity(area_ratio: float) -> str:
    return "near" if area_ratio > 0.15 else "mid" if area_ratio > 0.05 else "far"


def zone(xc: float, width: int) -> str:
    return "L" if xc < width * 0.33 else "R" if xc > width * 0.66 else "C"


def draw_detections(frame: np.ndarray, detections: Mapping, labels: bool = False) -> np.ndarray:
    """Draw boxes; line weight shows proximity. Mutates `frame`.

    Labels are off by default: OpenCV cannot render Devanagari, so Hindi labels
    are drawn by the web dashboard instead.
    """
    w = frame.shape[1]
    for name, (conf, xyxy, xc, area) in (detections or {}).items():
        x1, y1, x2, y2 = xyxy
        prox = proximity(area)
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), BLACK, THICKNESS[prox] + 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, THICKNESS[prox], cv2.LINE_AA)

        if not labels:
            continue
        label = f"{name} {int(conf * 100)}% {zone(xc, w)}"
        (tw, th), base = cv2.getTextSize(label, FONT, 0.5, 1)
        ty = max(y1, th + base + 6)
        cv2.rectangle(frame, (x1, ty - th - base - 6), (x1 + tw + 10, ty), WHITE, -1)
        cv2.putText(frame, label, (x1 + 5, ty - base - 2), FONT, 0.5, BLACK, 1, cv2.LINE_AA)
    return frame


def _panel(frame, x1, y1, x2, y2, alpha=0.62):
    roi = frame[y1:y2, x1:x2]
    if roi.size:
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def draw_hud(frame: np.ndarray, snap: Dict) -> np.ndarray:
    """OpenCV window HUD: status bar, zone guides, last spoken line, key help."""
    h, w = frame.shape[:2]

    # zone guides (left / centre / right thirds)
    guides = frame.copy()
    for x in (int(w * 0.33), int(w * 0.66)):
        cv2.line(guides, (x, 36), (x, h - 32), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(guides, 0.3, frame, 0.7, 0, dst=frame)

    # top bar: busy dot, fps. No words: OpenCV cannot draw Hindi.
    _panel(frame, 0, 0, w, 34)
    cv2.circle(frame, (18, 17), 6, WHITE, -1 if snap.get("busy") else 1, cv2.LINE_AA)
    right = f"{snap.get('fps', 0):.0f} fps   {snap.get('infer_ms', 0):.0f} ms"
    (tw, _), _ = cv2.getTextSize(right, FONT, 0.5, 1)
    cv2.putText(frame, right, (w - tw - 12, 23), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    if snap.get("muted"):   # speaker with a slash
        cv2.line(frame, (w // 2 - 10, 24), (w // 2 + 10, 10), WHITE, 2, cv2.LINE_AA)
        cv2.circle(frame, (w // 2, 17), 11, WHITE, 1, cv2.LINE_AA)

    # bottom: key hints (key names only)
    _panel(frame, 0, h - 30, w, h)
    cv2.putText(frame, "Space   D   C   R   M   Q", (12, h - 10), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return frame
