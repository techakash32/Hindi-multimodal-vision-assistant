"""Shared HTTP session (connection reuse) for OCR and VLM calls."""
from __future__ import annotations

import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# Only *connection* errors are retried; a request that was already sent is
# never re-sent (it could double-spend free-tier quota).
SESSION = requests.Session()
SESSION.mount(
    "https://",
    HTTPAdapter(
        pool_connections=4,
        pool_maxsize=8,
        max_retries=Retry(total=2, connect=2, read=0, status=0, redirect=0),
    ),
)


def warm_connection() -> None:
    """Open TLS connections ahead of time (uses no API quota)."""
    for url in ("https://openrouter.ai", "https://api.ocr.space"):
        try:
            SESSION.head(url, timeout=4)
        except Exception:
            pass  # warm-up only; real calls handle failures
    log.info("connections warmed")
