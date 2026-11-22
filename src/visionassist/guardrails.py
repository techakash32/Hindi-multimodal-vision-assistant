"""Abstention guardrails: decide whether the assistant should say "I'm not
sure" instead of answering.

``should_abstain`` is a pure function of the *percepts* gathered so far (image
quality, OCR text and confidence, detections, the router's verdict) and a
threshold config. The pipeline calls it before the VLM stage, so an abstention
costs no VLM latency or tokens, and again after the VLM answered.

A percept whose confidence is ``None`` was not measured. It never triggers a
confidence-based abstention (we only abstain on evidence), but a missing
modality or a router conflict always does.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable, Optional

# Reason code -> key in strings.MESSAGES / MESSAGES_DISPLAY.
REASON_MESSAGE = {
    "ocr_low": "abstain_ocr",
    "vlm_low": "abstain_vlm",
    "missing_modality": "abstain_missing",
    "conflict": "abstain_conflict",
}

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "abstention.json"


@dataclass
class Percept:
    modality: str                         # "image" | "ocr" | "detection" | "vlm" | "router"
    confidence: Optional[float] = None    # 0..1, None = not measured
    text: str = ""
    available: bool = True                # False: the input never arrived (no camera...)
    conflict: bool = False                # router: request is ambiguous / contradictory
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"modality": self.modality, "confidence": self.confidence,
                "available": self.available, "conflict": self.conflict}

    @staticmethod
    def from_dict(d: dict) -> "Percept":
        return Percept(d["modality"], d.get("confidence"), available=d.get("available", True),
                       conflict=d.get("conflict", False))


@dataclass
class AbstentionConfig:
    ocr_tau: float = 0.60
    vlm_tau: float = 0.50
    aggregate_tau_low: float = 0.45
    aggregate_tau_high: float = 0.70
    enabled: bool = True

    @classmethod
    def load(cls, path: Optional[os.PathLike] = None) -> "AbstentionConfig":
        """config/abstention.json, then env overrides (ABSTAIN_OCR_TAU, ...)."""
        data: dict = {}
        for p in ([Path(path)] if path else [Path.cwd() / "config" / "abstention.json", CONFIG_PATH]):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                break
            except (OSError, ValueError):
                continue
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in names}
        for name in names - {"enabled"}:
            raw = os.getenv("ABSTAIN_" + name.upper())
            if raw:
                try:
                    kwargs[name] = float(raw)
                except ValueError:
                    pass
        if os.getenv("ABSTAIN_ENABLED", "").strip().lower() in {"0", "false", "no", "off"}:
            kwargs["enabled"] = False
        return cls(**kwargs)

    @classmethod
    def disabled(cls) -> "AbstentionConfig":
        return cls(enabled=False)

    def thresholds(self) -> dict:
        d = asdict(self)
        d.pop("enabled")
        return d


def _measured(percepts: dict[str, Percept]) -> list[Percept]:
    return [p for p in percepts.values() if p.available and p.confidence is not None
            and p.modality != "router"]


def should_abstain(percepts: dict[str, Percept], cfg: AbstentionConfig) -> tuple[bool, Optional[str]]:
    """Return ``(abstain, reason)``. Rules, in order:

    1. a required modality is unavailable        -> ``missing_modality``
    2. the router flagged the request as unclear  -> ``conflict``
    3. OCR confidence below ``ocr_tau``           -> ``ocr_low``
    4. VLM / image confidence below ``vlm_tau``   -> ``vlm_low``
    5. mean confidence below ``aggregate_tau_low`` -> ``ocr_low`` or ``vlm_low``
       (whichever modality is weakest)

    ``Percept.conflict`` is set by whoever saw the contradiction: the router
    (unclear request) or the pipeline (the VLM says "nothing ahead" while the
    detector sees a person close in front).
    """
    if not cfg.enabled:
        return False, None
    if any(not p.available for p in percepts.values()):
        return True, "missing_modality"
    if any(p.conflict for p in percepts.values()):
        return True, "conflict"

    ocr = percepts.get("ocr")
    if ocr and ocr.confidence is not None and ocr.confidence < cfg.ocr_tau:
        return True, "ocr_low"
    for name in ("vlm", "image"):
        p = percepts.get(name)
        if p and p.confidence is not None and p.confidence < cfg.vlm_tau:
            return True, "vlm_low"

    measured = _measured(percepts)
    if measured:
        mean = sum(p.confidence for p in measured) / len(measured)
        if mean < cfg.aggregate_tau_low:
            weakest = min(measured, key=lambda p: p.confidence)
            return True, "ocr_low" if weakest.modality == "ocr" else "vlm_low"
    return False, None


def is_hedged(percepts: dict[str, Percept], cfg: AbstentionConfig) -> bool:
    """Answer, but flag it as uncertain: mean confidence between the two
    aggregate thresholds."""
    if not cfg.enabled:
        return False
    measured = _measured(percepts)
    if not measured:
        return False
    mean = sum(p.confidence for p in measured) / len(measured)
    return cfg.aggregate_tau_low <= mean < cfg.aggregate_tau_high


def staged_decision(phases: Iterable[dict[str, Percept]], cfg: AbstentionConfig) -> tuple[bool, Optional[str], int]:
    """Check each phase in order (the pipeline's before-VLM and after-VLM
    gates). Returns ``(abstain, reason, phase_index)``. The calibration script
    replays recorded runs through this, so thresholds are tuned on exactly the
    logic the app runs."""
    last = -1
    for i, percepts in enumerate(phases):
        last = i
        stop, reason = should_abstain(percepts, cfg)
        if stop:
            return True, reason, i
    return False, None, last
