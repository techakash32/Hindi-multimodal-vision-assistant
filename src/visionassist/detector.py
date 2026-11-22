"""YOLOv8 object detection with temporal smoothing, so the user only hears
stable, relevant, newly appearing objects."""
from __future__ import annotations

import logging
import time
from collections import deque

from . import config as cfg

log = logging.getLogger(__name__)


# COCO classes that matter for navigation. Everything else (kite, tie,
# frisbee, sports ball...) is ignored to avoid alert fatigue.
NAVIGATION_CLASSES = {
    # people, vehicles, traffic
    "person", "bicycle", "motorcycle", "car", "bus", "train", "truck",
    "traffic light", "fire hydrant", "stop sign", "bench",
    # animals (unpredictable; cows and elephants are common on Indian roads)
    "dog", "cat", "bird", "cow", "horse", "sheep", "elephant",
    # furniture / collision risks
    "chair", "couch", "bed", "dining table", "toilet", "potted plant",
    # carried items
    "backpack", "umbrella", "handbag", "suitcase",
    # reachable objects
    "bottle", "cup", "bowl", "fork", "knife", "spoon", "laptop",
    "keyboard", "mouse", "cell phone", "remote", "book",
    # fixtures
    "sink", "refrigerator", "oven", "microwave",
}

# Small handheld things never fill 2% of the frame; allow a lower area.
SMALL_OBJECTS = {
    "bottle", "cup", "bowl", "fork", "knife", "spoon",
    "cell phone", "remote", "mouse", "book", "keyboard",
}

PRIORITY = {
    "person": 0, "bicycle": 0, "car": 0, "motorcycle": 0, "bus": 0,
    "truck": 0, "train": 0, "dog": 0, "cow": 0, "horse": 0, "elephant": 0,
    "chair": 1, "couch": 1, "dining table": 1, "bench": 1, "bed": 1,
    "toilet": 1, "table": 1, "door": 1,
    "laptop": 2, "cup": 2, "bottle": 2, "book": 2,
}


class Detector:
    def __init__(self, model_path=None, model=None):
        """`model` lets tests (and the demo) inject any YOLO-compatible object."""
        if model is None:
            from ultralytics import YOLO  # lazy: heavy import

            model = YOLO(str(model_path or cfg.YOLO_MODEL))
        self.model = model

        # Temporal consistency state (per class)
        self._hist = {}        # cls -> deque of recent booleans
        self._ema = {}         # cls -> smoothed confidence
        self._last_seen = {}   # cls -> timestamp
        self._present = set()  # classes announced and still in view

        # Log throttling: last confidence printed per class
        self._last_print = {}

    def _log_changes(self, current):
        """Print a [RAW] line only when a class appears or its
        confidence moves by more than 0.15 (keeps the log readable)."""
        for name, info in current.items():
            conf = info[0]
            prev = self._last_print.get(name)

            if prev is None or abs(prev - conf) > 0.15:
                log.debug("raw %s conf=%.2f area=%.3f", name, conf, info[3])
                self._last_print[name] = conf

        for name in list(self._last_print):
            if name not in current:
                del self._last_print[name]

    def _update_state(self, current, now):
        for cls in self._hist:
            if cls not in current:
                self._hist[cls].append(False)

        for cls, info in current.items():
            conf = info[0]

            hist = self._hist.setdefault(cls, deque(maxlen=cfg.STABLE_FRAMES))
            hist.append(True)

            if cls in self._ema:
                self._ema[cls] = (
                    cfg.EMA_ALPHA * conf + (1 - cfg.EMA_ALPHA) * self._ema[cls]
                )
            else:
                self._ema[cls] = conf

            self._last_seen[cls] = now

    def _forget_absent(self, now):
        """Forget classes gone for cfg.REANNOUNCE_AFTER_S so they can be
        announced again when they come back."""
        for cls in list(self._last_seen):
            if now - self._last_seen[cls] > cfg.REANNOUNCE_AFTER_S:
                del self._last_seen[cls]
                self._hist.pop(cls, None)
                self._ema.pop(cls, None)
                self._present.discard(cls)

    def detect_still(self, frame):
        """One-shot detection of a single image: no smoothing, no history.
        Returns ``{class: (conf, xyxy, xc, area_ratio)}``, the best box per
        class. Used by the pipeline for still images and evaluation."""
        if frame is None or frame.size == 0:
            return {}

        h, w = frame.shape[:2]

        result = self.model(
            frame, conf=cfg.CONF_THRESH, imgsz=cfg.YOLO_IMGSZ, verbose=False
        )[0]

        current = {}

        for box in result.boxes:
            conf = float(box.conf[0])

            # Safety net in case the model ignores the conf argument
            if conf < cfg.CONF_THRESH:
                continue

            cls_name = self.model.names[int(box.cls[0])]

            if cfg.NAVIGATION_ONLY and cls_name not in NAVIGATION_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = max(0, x2 - x1) * max(0, y2 - y1)
            area_ratio = area / max(1, h * w)
            xc = (x1 + x2) / 2

            # Keep the highest-confidence detection per class.
            if (
                cls_name not in current
                or conf > current[cls_name][0]
            ):
                current[cls_name] = (
                    conf, (x1, y1, x2, y2), xc, area_ratio
                )

        return current

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return {}, []

        current = self.detect_still(frame)

        self._log_changes(current)

        now = time.time()
        self._update_state(current, now)

        newly = []

        for cls, info in current.items():
            hist = self._hist.get(cls, ())
            min_area = (
                cfg.MIN_SPEAK_AREA_SMALL if cls in SMALL_OBJECTS
                else cfg.MIN_SPEAK_AREA
            )

            if (
                cls not in self._present
                and info[3] >= min_area                       # big enough
                and self._ema.get(cls, 0.0) >= cfg.MIN_SPEAK_CONF  # steadily confident
                and len(hist) == cfg.STABLE_FRAMES                # enough history
                and all(hist)                                 # consecutive frames
            ):
                newly.append((cls, *info))
                self._present.add(cls)

        self._forget_absent(now)

        return current, newly

    @staticmethod
    def describe(obj, frame_shape):
        name, conf, _xyxy, xc, area = obj
        h, w = frame_shape[:2]

        direction = (
            "on your left" if xc < w * 0.33
            else "on your right" if xc > w * 0.66
            else "ahead"
        )

        proximity = (
            "very close" if area > 0.15
            else "nearby" if area > 0.05
            else ""
        )

        parts = [name]
        if proximity:
            parts.append(proximity)
        parts.append(direction)

        return ", ".join(parts)

    @staticmethod
    def prioritize(newly, top_n=2):
        return sorted(
            newly,
            key=lambda obj: (
                PRIORITY.get(obj[0], 5),
                -obj[4],
            ),
        )[:top_n]
