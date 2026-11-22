"""Event metrics: append-only CSV (same format as v0) plus in-memory
latency statistics for the dashboard."""
from __future__ import annotations

import csv
import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)


def percentile(values, p: float) -> float:
    """Linear-interpolated percentile (p in 0..100). Empty input -> 0.0."""
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


class Metrics:
    def __init__(self, path: Optional[Path] = None, window: int = 200):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = defaultdict(int)
        self._lat: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._csv_ready = False

    def _ensure_csv(self) -> bool:
        if self.path is None:
            return False
        if self._csv_ready:
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                with open(self.path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(["ts", "event", "detail", "latency_ms"])
            self._csv_ready = True
        except OSError as exc:
            log.warning("metrics CSV disabled: %s", exc)
            self.path = None
        return self._csv_ready

    def record(self, event: str, detail: str = "", latency_ms=None) -> None:
        with self._lock:
            self._counts[event] += 1
            if latency_ms not in (None, ""):
                self._lat[event].append(float(latency_ms))
            if self._ensure_csv():
                try:
                    with open(self.path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(
                            [time.time(), event, detail, "" if latency_ms is None else latency_ms]
                        )
                except OSError as exc:
                    log.warning("metrics write failed: %s", exc)

    def summary(self) -> Dict[str, Dict]:
        with self._lock:
            out = {}
            for ev, n in self._counts.items():
                lat = list(self._lat.get(ev, ()))
                out[ev] = {
                    "count": n,
                    "p50_ms": round(percentile(lat, 50)) if lat else None,
                    "p95_ms": round(percentile(lat, 95)) if lat else None,
                    "last_ms": round(lat[-1]) if lat else None,
                }
            return out
