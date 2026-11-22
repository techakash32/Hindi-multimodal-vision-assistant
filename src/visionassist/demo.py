"""Demo mode: runs the full engine with no camera, models or API keys.

The "YOLO model" here is a colour-blob finder on the synthetic feed, so the
real Detector smoothing / announcement logic still runs end to end. OCR and
scene description return canned text after a short delay. Everything is
labelled as demo in the UI.
"""
from __future__ import annotations

import time

import numpy as np

from .detector import Detector

PERSON_BGR = (150, 120, 90)


class _T:
    """Tiny tensor-like: supports [0] and float()/int()/map(int, ...)."""

    def __init__(self, v):
        self.v = v

    def __getitem__(self, i):
        return self.v

    def __iter__(self):
        return iter(self.v if hasattr(self.v, "__iter__") else [self.v])


class _Box:
    def __init__(self, conf, cls, xyxy):
        self.conf, self.cls, self.xyxy = _T(conf), _T(cls), _T(xyxy)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class BlobModel:
    names = {0: "person", 1: "chair"}

    def _find(self, frame, bgr, tol=6):
        diff = np.abs(frame.astype(np.int16) - np.array(bgr, dtype=np.int16)).sum(axis=2)
        ys, xs = np.nonzero(diff < tol)
        if xs.size < 400:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def __call__(self, frame, conf=0.0, imgsz=0, verbose=False):
        boxes = []
        person = self._find(frame, PERSON_BGR)
        if person:
            x1, y1, x2, y2 = person
            boxes.append(_Box(0.91, 0, (x1, y1 - 52, x2, y2)))
        from .sources import CHAIR_BGR
        chair = self._find(frame, CHAIR_BGR)
        if chair:
            boxes.append(_Box(0.84, 1, chair))
        return [_Result(boxes)]


def make_demo_detector() -> Detector:
    return Detector(model=BlobModel())


def demo_ocr(frame) -> str:
    time.sleep(1.2)
    return "दवाई की दुकान सुबह 9 बजे खुलती है। कृपया बाईं ओर कतार में लगें।"


def demo_vlm(frame, dets, shape) -> dict:
    time.sleep(1.8)
    from .hindi_text import CLASS_HI
    names = "، ".join(CLASS_HI.get(n, n) for n in dets) if dets else "कोई ख़ास चीज़ नहीं"
    return {
        "summary": f"कमरे में सामने {names} है।".replace("،", ","),
        "obstacles": ["सामने एक व्यक्ति"] if "person" in dets else [],
        "path_status": "obstacle_ahead" if "person" in dets else "unclear",
        "structured": True,
    }
