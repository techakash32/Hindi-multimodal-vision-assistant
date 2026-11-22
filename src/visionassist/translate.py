"""English -> Hindi translation (cached, with retry)."""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

from . import config as cfg

log = logging.getLogger(__name__)

_CACHE: "OrderedDict[str, str]" = OrderedDict()
_CACHE_MAX = 256

# Google's free endpoint allows about 5 requests per second. Space calls out, and
# after a rate-limit reply stop calling for a while instead of retrying and
# sleeping (the user would hear nothing for seconds, then still get an error).
MIN_INTERVAL_S = 0.3
RATE_LIMIT_COOLDOWN_S = 45.0
_RATE_MARKERS = ("too many requests", "429", "rate limit")
_lock = threading.Lock()
_last_call = 0.0
_cooldown_until = 0.0


def has_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text or "")


def _translate_once(text: str) -> str:
    from deep_translator import GoogleTranslator  # lazy: optional at import time
    return GoogleTranslator(source="en", target="hi").translate(text)


def to_hindi(text: str, retries: int = 3) -> str:
    """Translate English to Hindi. Cached; retries on failure; returns the
    original text if everything fails (callers then fall back to English)."""
    if not text or len(text.strip()) < 2 or has_devanagari(text):
        return text
    if cfg.LOCAL_ONLY:                 # translation is a network call
        return text

    key = text.strip().lower()[:200]
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    global _last_call, _cooldown_until
    if time.monotonic() < _cooldown_until:
        log.info("translation paused (rate limited); using English text")
        return text

    for attempt in range(retries):
        try:
            with _lock:
                wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
                if wait > 0:
                    time.sleep(wait)
                _last_call = time.monotonic()
            result = _translate_once(text)
            if result:
                _CACHE[key] = result
                if len(_CACHE) > _CACHE_MAX:
                    _CACHE.popitem(last=False)
                return result
        except ImportError as exc:  # package missing: retrying cannot help
            log.warning("translation unavailable: %s", exc)
            return text
        except Exception as exc:  # network, rate limit
            if any(m in str(exc).lower() for m in _RATE_MARKERS):
                _cooldown_until = time.monotonic() + RATE_LIMIT_COOLDOWN_S
                log.warning("translation rate limited; pausing for %.0fs", RATE_LIMIT_COOLDOWN_S)
                break
            log.warning("translate attempt %d failed: %s", attempt + 1, exc)
            time.sleep(0.6 * (attempt + 1))

    log.warning("translation failed; using English text")
    return text
