"""Frame sources: a real webcam and a synthetic one for demos and tests."""
from __future__ import annotations

import logging
import math
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger(__name__)


CHAIR_BGR = (80, 110, 60)


class CameraSource:
    """OpenCV webcam. Reopenable, so the engine can recover from unplugging."""

    def __init__(self, index: Optional[int] = None):
        self.index = cfg.CAMERA_INDEX if index is None else index
        self._cap: Optional[cv2.VideoCapture] = None
        self.open()

    def open(self) -> bool:
        self.release()
        # DirectShow starts much faster on Windows; default elsewhere.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
        self._cap = cap
        return cap.isOpened()

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticSource:
    """Generated frames with moving blocks, so the UI can be demonstrated and
    tested on a machine without a camera. Clearly labelled DEMO."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 15.0):
        self.w, self.h, self.fps = width, height, fps
        self._t0 = time.time()
        self._last = 0.0

    def is_open(self) -> bool:
        return True

    def open(self) -> bool:
        return True

    def release(self) -> None:
        pass

    def read(self) -> Tuple[bool, np.ndarray]:
        wait = self._last + 1.0 / self.fps - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        t = self._last - self._t0

        y = np.linspace(38, 62, self.h, dtype=np.uint8)[:, None]
        frame = np.dstack([y + 14, y + 6, y]).repeat(self.w, axis=1).astype(np.uint8)
        cv2.line(frame, (0, int(self.h * 0.72)), (self.w, int(self.h * 0.72)), (90, 80, 70), 2)
        cx = int(self.w * (0.5 + 0.38 * math.sin(t * 0.5)))
        cv2.rectangle(frame, (cx - 55, int(self.h * 0.3)), (cx + 55, int(self.h * 0.72)), (150, 120, 90), -1)
        cv2.circle(frame, (cx, int(self.h * 0.24)), 26, (170, 150, 130), -1)
        if 8 < (t % 24) < 18:   # a chair drifts into view now and then
            cv2.rectangle(frame, (int(self.w * 0.74), int(self.h * 0.5)),
                          (int(self.w * 0.92), int(self.h * 0.72)), CHAIR_BGR, -1)
        return True, frame
