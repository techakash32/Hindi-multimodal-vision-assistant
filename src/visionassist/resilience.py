# resilience.py
import logging
import threading
import time

log = logging.getLogger(__name__)


class CircuitBreaker:
    """Skip a failing provider for a cooldown instead of retrying it
    on every request."""

    def __init__(self, name, failure_threshold=3, cooldown_sec=60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec

        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0
        self.total_calls = 0
        self.total_failures = 0

    def allow(self) -> bool:
        with self._lock:
            if self._open_until == 0.0:
                return True

            if time.time() >= self._open_until:
                self._open_until = 0.0
                self._failures = 0
                log.info("circuit %s re-enabled", self.name)
                return True

            return False

    def record_success(self):
        with self._lock:
            self.total_calls += 1
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self):
        with self._lock:
            self.total_calls += 1
            self.total_failures += 1
            self._failures += 1

            if (
                self._failures >= self.failure_threshold
                and self._open_until == 0.0
            ):
                self._open_until = time.time() + self.cooldown_sec
                log.warning("circuit %s OPEN after %d failures (cooldown %.0fs)", self.name, self._failures, self.cooldown_sec)

    def trip(self, cooldown_sec=None, reason=""):
        """Open immediately (e.g. HTTP 404 gone, 429 rate limited)."""
        with self._lock:
            cooldown = (
                cooldown_sec if cooldown_sec is not None
                else self.cooldown_sec
            )
            self.total_calls += 1
            self.total_failures += 1
            self._open_until = time.time() + cooldown

            log.warning("circuit %s OPEN (%s, %.0fs)", self.name, reason, cooldown)

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open_until != 0.0 and time.time() < self._open_until

    def stats(self) -> str:
        state = "OPEN" if self._open_until else "CLOSED"
        return (
            f"{self.name}: {self.total_calls} calls, "
            f"{self.total_failures} failures, {state}"
        )
