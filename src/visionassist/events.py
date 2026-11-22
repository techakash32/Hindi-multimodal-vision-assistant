"""Thread-safe ring buffer of user-visible events (feeds the UIs)."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, List


class EventBus:
    """Keeps the most recent events; each gets an increasing integer id so
    a UI can ask for "everything after id N"."""

    def __init__(self, maxlen: int = 200):
        self._events: Deque[Dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._next_id = 1
        self._sinks: list = []

    def add_sink(self, fn) -> None:
        """Call ``fn(event_dict)`` for every published event (e.g. a session recorder)."""
        self._sinks.append(fn)

    def publish(self, kind: str, text: str, level: str = "info") -> int:
        with self._lock:
            eid = self._next_id
            self._next_id += 1
            self._events.append(
                {"id": eid, "ts": time.time(), "kind": kind,
                 "level": level, "text": str(text)}
            )
            ev = self._events[-1]
        for fn in self._sinks:
            try:
                fn(ev)
            except Exception:       # a broken sink must never break the app
                pass
        return eid

    def since(self, last_id: int = 0, limit: int = 100) -> List[Dict]:
        with self._lock:
            out = [e for e in self._events if e["id"] > last_id]
        return out[-limit:]

    def recent(self, n: int = 5, kinds: tuple = ()) -> List[Dict]:
        with self._lock:
            items = [e for e in self._events if not kinds or e["kind"] in kinds]
        return items[-n:]
