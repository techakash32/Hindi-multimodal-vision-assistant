"""Startup health checks.

Anything the offline safety loop needs (detector, English voice) is
*required*. Everything else only warns, because the assistant must keep
working without internet, voice commands or Hindi.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from . import config as cfg

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    status: str          # "ok" | "warn" | "fail"
    detail: str = ""
    required: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def run_checks(demo: bool = False) -> List[Check]:
    """Return one Check per capability. Never raises."""
    if demo:
        return [Check("Demo mode", "ok", "synthetic camera and detector; no models needed")]

    checks: List[Check] = []

    def file_check(name, path, required, hint):
        p = Path(path)
        if p.exists():
            checks.append(Check(name, "ok", str(p.name), required))
        else:
            checks.append(Check(name, "fail" if required else "warn",
                                f"missing: {p} — {hint}", required))

    file_check("YOLO model", cfg.YOLO_MODEL, True, "run: visionassist download-models")
    if cfg.HINDI_ONLY:
        file_check("Hindi voice", cfg.HINDI_VOICE_MODEL, True, "run: visionassist download-models --only hi")
        file_check("Hindi voice config", str(cfg.HINDI_VOICE_MODEL) + ".json", True,
                   "run: visionassist download-models --only hi")
    else:
        file_check("English voice", cfg.VOICE_MODEL, True, "run: visionassist download-models")
        file_check("English voice config", str(cfg.VOICE_MODEL) + ".json", True,
                   "run: visionassist download-models")

    if cfg.VOICE_ENABLED:
        file_check("Voice commands (Vosk)", cfg.VOSK_MODEL_PATH, False,
                   "voice commands disabled; run: visionassist download-models")
    if cfg.USE_HINDI and not cfg.HINDI_ONLY:
        file_check("Hindi voice", cfg.HINDI_VOICE_MODEL, False, "alerts will be in English")

    checks.append(
        Check("OCR.space key", "ok" if cfg.OCR_API_KEY else "warn",
              "set" if cfg.OCR_API_KEY else "OCR_API_KEY not set — reading text disabled")
    )
    checks.append(
        Check("OpenRouter key", "ok" if cfg.OPENROUTER_API_KEY else "warn",
              "set" if cfg.OPENROUTER_API_KEY else "OPENROUTER_API_KEY not set — scene description disabled")
    )
    return checks


def run_health_check(demo: bool = False) -> bool:
    """Log the result of every check. True if the safety loop can run."""
    checks = run_checks(demo)
    for c in checks:
        if c.status == "warn":
            log.warning("health: %s — %s", c.name, c.detail)
        elif c.status == "fail":
            log.error("health: %s — %s", c.name, c.detail)
    ok = not any(c.status == "fail" for c in checks)
    log.info("health check %s", "passed" if ok else "FAILED")
    return ok
