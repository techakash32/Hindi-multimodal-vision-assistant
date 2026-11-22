"""OpenCV window frontend (default). Keyboard shortcuts mirror the dashboard."""
from __future__ import annotations

import logging
import time

import cv2

from .draw import draw_hud

log = logging.getLogger(__name__)

WINDOW = "VisionAssist"
KEYMAP = {
    ord(" "): "read",
    ord("r"): "repeat", ord("R"): "repeat",
    ord("d"): "describe", ord("D"): "describe",
    ord("c"): "changed", ord("C"): "changed",
    ord("m"): "mute", ord("M"): "mute",
    ord("q"): "quit", ord("Q"): "quit", 27: "quit",
}


def run_window(engine) -> None:
    """Blocks until the engine quits. Must run on the main thread."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    snap, snap_t = {}, 0.0
    try:
        while not engine.quit_event.is_set():
            frame = engine.annotated_frame()
            if frame is None:
                cv2.waitKey(30)
                continue
            if time.time() - snap_t > 0.2:
                snap, snap_t = engine.snapshot(), time.time()
            cv2.imshow(WINDOW, draw_hud(frame, snap))

            key = cv2.waitKey(15) & 0xFF
            cmd = KEYMAP.get(key)
            if cmd:
                engine.dispatch(cmd, "keyboard")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                engine.dispatch("quit", "window")
    except KeyboardInterrupt:
        log.info("Ctrl+C")
    finally:
        cv2.destroyAllWindows()
