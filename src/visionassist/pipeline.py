"""One request through the perception stack: the single entry point shared by
the app (``Assistant``) and the evaluation harness (``eval/run_eval.py``).

    run_once(PipelineInput(frame=..., text=...), "Ye dawa ka naam kya hai?", ...)

Stages, in order (each is timed with ``time.perf_counter``):

    router   -> which modality does the request need? (or: unclear -> conflict)
    quality  -> cheap image-quality proxy (sharpness / contrast / brightness)
    gate 1   -> ``guardrails.should_abstain`` on router + image quality.
                An abstention here costs no OCR / detection / VLM call at all.
    ocr | detect + vlm -> the perception call that the route asked for
    gate 2   -> ``should_abstain`` again, now with OCR / VLM confidence
    answer   -> Devanagari text for the voice, Hinglish text for the screen

The function never speaks and never touches a window or the network by
itself: OCR / VLM / detector / translator are injected, so the app passes the
live ones and tests pass fakes. Nothing in here is specific to evaluation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import config as cfg
from .draw import proximity, zone
from .guardrails import (
    REASON_MESSAGE,
    AbstentionConfig,
    Percept,
    is_hedged,
    should_abstain,
)
from .hinglish import describe_hinglish, to_hinglish
from .quality import image_quality, text_plausibility
from .router import ENGINE_COMMAND, Route, resolve_command, route
from .strings import MESSAGES, MESSAGES_DISPLAY, PATH, PATH_DISPLAY
from .translate import has_devanagari, to_hindi


def stage_budgets_ms() -> dict[str, float]:
    """Per-stage latency budgets. OCR / VLM follow the request timeouts."""
    return {"router": 50.0, "quality": 250.0, "detect": 1500.0,
            "ocr": cfg.OCR_TIMEOUT * 1000.0, "vlm": cfg.VLM_TIMEOUT * 1000.0}


@dataclass
class PipelineInput:
    frame: Any = None                 # BGR ndarray, or None: no camera frame
    text: Optional[str] = None        # typed / transcribed command, if any
    audio: Any = None                 # reserved: raw audio is decoded by voice.py
    detections: Optional[dict] = None # live detections from the capture loop, if any


@dataclass
class Trace:
    stage: str
    ms: float
    confidence: Optional[float] = None
    over_budget: bool = False
    detail: dict = field(default_factory=dict)


@dataclass
class Utterance:
    """One thing to say. ``kind`` lets the app pick an emotion for the voice."""
    text: str                          # Devanagari, for the voice
    display: str                       # Hinglish, for the screen
    kind: str = "answer"               # answer | path | obstacle | summary | message


@dataclass
class PipelineResult:
    request: str
    route: Route
    status: str = "answer"             # answer | abstain | no_text | error | control
    intent: Optional[str] = None       # read | describe | change | repeat | stop
    engine_command: Optional[str] = None
    modalities: list[str] = field(default_factory=list)   # stages actually invoked
    traces: list[Trace] = field(default_factory=list)
    percepts: dict[str, Percept] = field(default_factory=dict)
    phases: list[dict[str, Percept]] = field(default_factory=list)  # one entry per gate
    abstained: bool = False
    reason: Optional[str] = None
    hedged: bool = False
    utterances: list[Utterance] = field(default_factory=list)
    raw: dict = field(default_factory=dict)   # raw_text, vlm, detections ...
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def answer_text(self) -> str:
        return " ".join(u.text for u in self.utterances)

    @property
    def display_text(self) -> str:
        return " | ".join(u.display for u in self.utterances)

    @property
    def confidences(self) -> dict[str, Optional[float]]:
        return {k: (None if p.confidence is None else round(p.confidence, 3))
                for k, p in self.percepts.items() if p.modality != "router"}

    def trace(self, stage: str) -> Optional[Trace]:
        return next((t for t in self.traces if t.stage == stage), None)

    def over_budget(self) -> list[str]:
        return [t.stage for t in self.traces if t.over_budget]

    def to_dict(self) -> dict:
        return {
            "status": self.status, "intent": self.intent, "modalities": self.modalities,
            "abstained": self.abstained, "reason": self.reason, "hedged": self.hedged,
            "confidence": self.confidences, "answer_text": self.answer_text,
            "display_text": self.display_text, "latency_ms": round(self.latency_ms, 2),
            "stage_ms": {t.stage: round(t.ms, 2) for t in self.traces},
            "over_budget": self.over_budget(), "route": {"intent": self.route.intent,
            "modality": self.route.modality, "conflict": self.route.conflict, "why": self.route.why},
            "phases": [{k: p.to_dict() for k, p in ph.items()} for ph in self.phases],
            "raw": self.raw, "error": self.error,
        }


# ----------------------------------------------------------------- helpers
def _hindi_or_none(text: str) -> Optional[str]:
    """Hindi version of ``text`` (translated if needed) or None if it failed."""
    if has_devanagari(text):
        return text
    hi = to_hindi(text)
    return hi if has_devanagari(hi) else None


def _message(key: str, kind: str = "message") -> Utterance:
    return Utterance(MESSAGES[key], MESSAGES_DISPLAY[key], kind)


def _timed(result: PipelineResult, stage: str, fn: Callable, budgets: dict[str, float]):
    t0 = time.perf_counter()
    try:
        return fn()
    finally:
        ms = (time.perf_counter() - t0) * 1000.0
        result.traces.append(Trace(stage, ms, over_budget=ms > budgets.get(stage, float("inf"))))


def _set_conf(result: PipelineResult, stage: str, conf: Optional[float]) -> None:
    t = result.trace(stage)
    if t is not None:
        t.confidence = conf


def _abstain(result: PipelineResult, reason: str) -> PipelineResult:
    result.status, result.abstained, result.reason = "abstain", True, reason
    result.utterances = [_message(REASON_MESSAGE[reason])]
    return result


def _gate(result: PipelineResult, acfg: AbstentionConfig) -> bool:
    """Record this phase's percepts and apply the guardrail. True = abstained."""
    result.phases.append(dict(result.percepts))
    stop, reason = should_abstain(result.percepts, acfg)
    if not stop and any(not p.available for p in result.percepts.values()):
        stop, reason = True, "missing_modality"     # an input that never arrived cannot be answered, guardrails off or on
    if stop:
        _abstain(result, reason or "vlm_low")
    return stop


def _finish(result: PipelineResult, t_start: float) -> PipelineResult:
    result.latency_ms = (time.perf_counter() - t_start) * 1000.0
    return result


# ------------------------------------------------------------------ stages
def _ocr_stage(result, frame, ocr_fn, budgets, translate_fn):
    out = _timed(result, "ocr", lambda: ocr_fn(frame), budgets)
    text = str(out)
    result.modalities.append("ocr")
    result.raw["raw_text"] = text
    if text.startswith("No text found"):
        result.raw["raw_text"] = ""
        _set_conf(result, "ocr", 0.0)
        result.percepts["ocr"] = Percept("ocr", 0.0, text="", detail={"source": "no_text"})
        result.phases.append(dict(result.percepts))
        result.status, result.abstained, result.reason = "no_text", True, "ocr_low"
        result.utterances = [_message("no_text")]
        return False
    if text.startswith("OCR error"):
        result.status, result.error = "error", text
        result.utterances = [_message("ocr_error")]
        return False

    engine_conf = getattr(out, "confidence", None)
    plaus = text_plausibility(text)
    conf = plaus if engine_conf is None else min(engine_conf, 1.0)
    quality = result.percepts["ocr"].confidence
    result.percepts["ocr"] = Percept("ocr", min(conf, quality if quality is not None else 1.0), text=text,
                                     detail={"engine_confidence": engine_conf, "plausibility": plaus,
                                             "image_quality": quality})
    _set_conf(result, "ocr", result.percepts["ocr"].confidence)
    return True


def _detections(result, frame, inp, detector, budgets) -> dict:
    if inp.detections is not None:
        return dict(inp.detections)
    if detector is None:
        return {}
    still = getattr(detector, "detect_still", None)
    if still is None:
        return {}
    return _timed(result, "detect", lambda: still(frame), budgets)


def _conflicting(dets: dict, frame_w: int, path_status: str) -> bool:
    """The VLM claims nothing is ahead while the detector sees something big
    in the middle of the frame."""
    if path_status != "none_seen":
        return False
    return any(zone(xc, frame_w) == "C" and proximity(area) in ("near", "mid")
               for (_conf, _xyxy, xc, area) in dets.values())


def _vlm_stage(result, frame, dets, vlm_fn, budgets, translate_fn):
    shape = frame.shape
    data = _timed(result, "vlm", lambda: vlm_fn(frame, dets, shape), budgets)
    result.modalities.append("vlm")
    result.raw["vlm"] = {k: data.get(k) for k in ("summary", "obstacles", "path_status", "confidence")}
    conf = data.get("confidence")
    result.percepts["vlm"] = Percept("vlm", conf, text=data.get("summary") or "",
                                     detail={"self_reported": conf is not None})
    _set_conf(result, "vlm", conf)
    if _conflicting(dets, shape[1], data.get("path_status")):
        result.percepts["router"] = Percept("router", conflict=True,
                                            detail={"why": "VLM says path clear, detector sees an object ahead"})
    return data


def _describe_utterances(result, data, dets, translate_fn) -> bool:
    obstacles, path, summary = data.get("obstacles") or [], data.get("path_status"), data.get("summary")
    out: list[Utterance] = []
    if obstacles:
        hi = translate_fn(", ".join(obstacles))
        if hi is None:
            return False
        out.append(Utterance(hi, to_hinglish(hi), "obstacle"))
    if path in ("obstacle_ahead", "unclear") or (path == "none_seen" and cfg.ANNOUNCE_PATH_CLEAR):
        out.insert(0, Utterance(PATH[path], PATH_DISPLAY[path], "path"))
    if summary:
        hi = translate_fn(summary)
        if hi is None:
            return False
        out.append(Utterance(hi, to_hinglish(hi), "summary"))
    result.utterances = out or ([Utterance(PATH[path], PATH_DISPLAY[path], "path")] if path in PATH else [])
    return True


def _hedge(result: PipelineResult) -> None:
    """Prefix the first spoken / shown sentence with "shayad" (maybe)."""
    first = next((u for u in result.utterances if u.kind in ("answer", "summary", "obstacle")), None)
    if first is not None:
        first.text = f"{MESSAGES['hedge']} {first.text}"
        first.display = f"{MESSAGES_DISPLAY['hedge']} {first.display}"
    result.hedged = True


# -------------------------------------------------------------- entry point
def run_once(
    inp: PipelineInput,
    request: str,
    *,
    detector=None,
    ocr_fn: Optional[Callable] = None,
    vlm_fn: Optional[Callable] = None,
    translate_fn: Callable[[str], Optional[str]] = _hindi_or_none,
    abstention: Optional[AbstentionConfig] = None,
    budgets: Optional[dict[str, float]] = None,
) -> PipelineResult:
    """Run one request and return everything that happened (see module doc)."""
    t_start = time.perf_counter()
    acfg = abstention if abstention is not None else AbstentionConfig.load()
    budgets = budgets or stage_budgets_ms()
    if ocr_fn is None:
        from .ocr import read_text_from_frame as ocr_fn
    if vlm_fn is None:
        from .vlm import describe_scene_structured as vlm_fn

    result = PipelineResult(request=request, route=Route("unclear", "none"))

    # -- router -----------------------------------------------------------
    command_only = inp.text is not None and inp.frame is None
    rt = _timed(result, "router", lambda: route(request), budgets)
    result.route = rt
    cmd = resolve_command(request)
    if command_only:
        # A bare transcript (voice command): resolve the intent, no perception.
        result.modalities.append("asr")
        result.intent = cmd
        result.engine_command = ENGINE_COMMAND.get(cmd) if cmd else None
        result.status = "control" if cmd else "abstain"
        if cmd is None:
            result.abstained, result.reason = True, "conflict"
            result.utterances = [_message("abstain_conflict")]
        return _finish(result, t_start)
    result.intent = rt.intent if rt.intent != "unclear" else None
    result.engine_command = ENGINE_COMMAND.get(rt.intent)
    if rt.modality == "asr":         # change / repeat / stop typed with no frame needed
        result.modalities.append("asr")
        result.status = "control"
        return _finish(result, t_start)

    # -- gate 1: router verdict + camera + image quality --------------------
    frame = inp.frame
    target = "ocr" if rt.modality == "ocr" else "image"
    result.percepts["router"] = Percept("router", conflict=rt.conflict, detail={"why": rt.why})
    if frame is None:
        result.percepts[target] = Percept(target, available=False)
    else:
        q = _timed(result, "quality", lambda: image_quality(frame), budgets)
        _set_conf(result, "quality", q["confidence"])
        result.percepts[target] = Percept(target, q["confidence"], detail={"source": "image_quality", **q})
        result.raw["image_quality"] = {k: round(v, 3) for k, v in q.items()}
    if _gate(result, acfg):
        return _finish(result, t_start)

    # -- perception ----------------------------------------------------------
    if rt.modality == "ocr":
        if not _ocr_stage(result, frame, ocr_fn, budgets, translate_fn):
            return _finish(result, t_start)
        if _gate(result, acfg):
            return _finish(result, t_start)
        hi = translate_fn(result.raw["raw_text"][:cfg.OCR_MAX_CHARS])
        if hi is None:
            result.status, result.error = "error", "translation failed"
            result.utterances = [_message("translate_error")]
            return _finish(result, t_start)
        result.utterances = [Utterance(hi, to_hinglish(hi), "answer")]
    else:   # describe
        dets = _detections(result, frame, inp, detector, budgets)
        if detector is not None or inp.detections is not None:
            result.modalities.append("detect")
        result.raw["detections"] = {n: {"conf": round(c, 3), "zone": zone(xc, frame.shape[1]),
                                        "prox": proximity(a)} for n, (c, _b, xc, a) in dets.items()}
        data = _vlm_stage(result, frame, dets, vlm_fn, budgets, translate_fn)
        if _gate(result, acfg):
            return _finish(result, t_start)
        if not _describe_utterances(result, data, dets, translate_fn):
            result.status, result.error = "error", "translation failed"
            result.utterances = [_message("translate_error")]
            return _finish(result, t_start)
        shape = frame.shape
        result.raw["detections_spoken"] = [describe_hinglish((n, *info), shape) for n, info in dets.items()]

    if is_hedged(result.percepts, acfg):
        _hedge(result)
    return _finish(result, t_start)
