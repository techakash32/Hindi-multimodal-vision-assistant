"""Perception backends the evaluation can run against.

The harness always calls ``visionassist.pipeline.run_once``; a backend only
decides which OCR / detector / VLM callables are injected.

    auto  (default)  OCR: local Tesseract if installed, else the scripted fake.
                     Detector: the demo colour-blob model (NOT YOLO).
                     VLM: a deterministic template over the detections (NOT a
                     language model). Needs no network and no API key.
    fake             Everything scripted from the scenario file. This is a
                     plumbing test: it knows the answers, so its accuracy
                     numbers say nothing about the models. Used by the
                     end-to-end abstention test, which needs to run anywhere.
    live             Real OCR.space + OpenRouter VLM (+ YOLO if available).
                     Needs OCR_API_KEY / OPENROUTER_API_KEY and network.

``describe()`` returns the strings that go in the report header, so a report
always says which parts were real.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2

from visionassist import config as cfg
from visionassist.demo import make_demo_detector
from visionassist.hindi_text import CLASS_HI
from visionassist.ocr import OcrText, read_text_tesseract, tesseract_available

ROOT = Path(__file__).resolve().parent.parent


def frame_key(frame) -> str:
    return hashlib.sha1(frame.tobytes()).hexdigest()


@dataclass
class Backends:
    name: str
    detector: object
    ocr_fn: Callable
    vlm_fn: Callable
    notes: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return dict(self.notes)


# ----------------------------------------------------------------- fakes
class FakeOCR:
    """Returns the scenario's ground truth with a high confidence, except for
    the scenarios whose point is a bad read: those return garbage with a low
    confidence. Keyed by image content, so it works without knowing the id."""

    GARBAGE = OcrText("Hwgoogs jowe, re1e,", 0.17, "fake")

    def __init__(self, scenarios_path: Path):
        self.table: dict[str, OcrText] = {}
        for line in Path(scenarios_path).read_text(encoding="utf-8").splitlines():
            sc = json.loads(line)
            img = (sc["input"] or {}).get("image")
            if not img or not (ROOT / img).exists():
                continue
            frame = cv2.imread(str(ROOT / img))
            if sc["category"] == "negative_abstain" and sc["expected_modality"] == "ocr":
                # a bad read (upside-down text, a line hidden by a bar): garbage, low confidence
                self.table[frame_key(frame)] = self.GARBAGE
            elif sc.get("ground_truth") and sc["expected_modality"] == "ocr":
                self.table[frame_key(frame)] = OcrText(sc["ground_truth"], 0.95, "fake")

    def __call__(self, frame):
        return self.table.get(frame_key(frame), "No text found. Try pointing closer.")


def _side_hi(zone: str) -> str:
    return {"L": "बाईं ओर", "R": "दाईं ओर", "C": "सामने"}[zone]


class TemplateVLM:
    """Deterministic stand-in for a VLM: one Devanagari sentence built from the
    detections it is given. It adds nothing the detector did not see, so a
    result here measures the detector plus the pipeline, not a language model.
    Confidence is the weakest detection confidence (0.6 when nothing was
    detected, because 'no known objects' is itself not certain)."""

    def __call__(self, frame, dets, shape):
        from visionassist.draw import proximity, zone
        w = shape[1]
        if not dets:
            return {"summary": "कोई ख़ास चीज़ नहीं दिखी।", "obstacles": [], "path_status": "unclear",
                    "structured": True, "confidence": 0.6}
        items = sorted(dets.items(), key=lambda kv: -kv[1][3])
        parts, obstacles = [], []
        for name, (_c, _b, xc, area) in items:
            z, hi = zone(xc, w), CLASS_HI.get(name, name)
            parts.append(f"{_side_hi(z)} एक {hi} है")
            if z == "C" and proximity(area) in ("near", "mid"):
                obstacles.append(f"सामने एक {hi}")
        summary = " और ".join(parts) + "।"
        if len(items) > 1:
            summary = "; ".join(parts) + "।"
        return {"summary": summary, "obstacles": obstacles[:3],
                "path_status": "obstacle_ahead" if obstacles else "unclear", "structured": True,
                "confidence": round(min(c for _n, (c, *_r) in items), 3)}


def build(name: str, scenarios_path: Path) -> Backends:
    name = name.lower()
    detector = make_demo_detector()
    notes = {"detector": "demo colour-blob model (not YOLO)",
             "vlm": "deterministic template over detections (not a language model)",
             "asr": "not exercised (typed / simulated transcripts; no audio, no Vosk)"}
    if name == "fake":
        notes["ocr"] = "scripted fake (knows the ground truth; plumbing test only)"
        return Backends("fake", detector, FakeOCR(scenarios_path), TemplateVLM(), notes)
    if name == "auto":
        if tesseract_available():
            import subprocess
            ver = subprocess.run([cfg.TESSERACT_CMD, "--version"], capture_output=True, text=True).stdout.split("\n")[0]
            notes["ocr"] = f"{ver} (local, real; psm {cfg.TESSERACT_PSM}, lang {cfg.TESSERACT_LANG})"
            return Backends("auto", detector, read_text_tesseract, TemplateVLM(), notes)
        notes["ocr"] = "scripted fake (tesseract not installed)"
        return Backends("auto", detector, FakeOCR(scenarios_path), TemplateVLM(), notes)
    if name == "live":
        from visionassist.ocr import read_text_from_frame
        from visionassist.vlm import describe_scene_structured
        try:
            from visionassist.detector import Detector
            detector = Detector()
            notes["detector"] = f"YOLO {cfg.YOLO_MODEL.name}"
        except Exception as exc:          # ultralytics / weights missing
            notes["detector"] += f" (YOLO unavailable: {type(exc).__name__})"
        notes["ocr"] = f"engine={cfg.OCR_ENGINE}"
        models = cfg.VLM_MODELS or ["built-in free OpenRouter list"]
        notes["vlm"] = "OpenRouter: " + ", ".join(models)
        return Backends("live", detector, read_text_from_frame, describe_scene_structured, notes)
    raise ValueError(f"unknown backend {name!r} (auto | fake | live)")


def identity_translate(text: str) -> Optional[str]:
    """Translation is a network call (Google via deep-translator), so the
    harness does not exercise it: the text passes through unchanged."""
    return text
