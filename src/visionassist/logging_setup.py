"""Logging: readable console output plus a rotating file in ``logs/``."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from . import config

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str | None = None, to_file: bool = True) -> None:
    """Configure the root logger once. Safe to call repeatedly."""
    root = logging.getLogger()
    if getattr(root, "_visionassist_configured", False):
        return
    root.setLevel(getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
    root.addHandler(console)

    if to_file:
        try:
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                config.LOG_DIR / "visionassist.log",
                maxBytes=1_000_000, backupCount=3, encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter(_FMT))
            root.addHandler(fh)
        except OSError as exc:  # read-only disk etc.: console only
            root.warning("file logging disabled: %s", exc)

    # Third-party chatter is not useful to the user.
    for noisy in ("urllib3", "werkzeug", "PIL", "matplotlib", "waitress"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._visionassist_configured = True  # type: ignore[attr-defined]
