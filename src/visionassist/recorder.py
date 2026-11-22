"""Session recording and replay.

``visionassist run --record`` writes a folder::

    logs/sessions/20261121-163000/
        session.jsonl        one JSON object per line
        frames/0001.jpg      the frame behind every OCR / scene request

Line types
    {"type": "session", ...}   version, start time, demo flag
    {"type": "event", ...}     everything the UI shows: commands, speech, OCR text, errors
    {"type": "request", ...}   one ``run_once`` call: request, frame file, detections, full result

``visionassist replay <folder>`` feeds every recorded request back through
``pipeline.run_once`` and reports which answers changed. Frames are stored as
JPEG (quality 90), so a replay sees a slightly compressed copy of what the
camera produced: small OCR differences can come from that, not from a code change.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2

from . import __version__


def _jsonable(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


class SessionRecorder:
    def __init__(self, directory: Path, demo: bool = False):
        self.dir = Path(directory)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "session.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._n = 0
        self._write({"type": "session", "t": time.time(), "version": __version__, "demo": demo})

    def _write(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False, default=_jsonable)
        with self._lock:
            if self._fh and not self._fh.closed:
                self._fh.write(line + "\n")
                self._fh.flush()

    def event(self, ev: dict) -> None:
        """EventBus sink."""
        self._write({"type": "event", **ev})

    def request(self, request: str, frame, detections: Optional[dict], result) -> None:
        with self._lock:
            self._n += 1
            n = self._n
        name = f"frames/{n:04d}.jpg"
        if frame is not None:
            cv2.imwrite(str(self.dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self._write({
            "type": "request", "t": time.time(), "request": request,
            "frame": name if frame is not None else None,
            "detections": detections or {}, "result": result.to_dict(),
            "raw_text": str(result.raw.get("raw_text", ""))[:2000] if isinstance(result.raw, dict) else "",
        })

    def close(self) -> None:
        with self._lock:
            if self._fh and not self._fh.closed:
                self._fh.close()


def session_file(path: Path) -> Path:
    path = Path(path)
    return path / "session.jsonl" if path.is_dir() else path


def load_session(path: Path) -> list[dict]:
    rows = []
    for line in session_file(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@dataclass
class ReplayRow:
    index: int
    request: str
    same: bool
    changes: list[str]
    recorded: dict
    replayed: dict


def _restore_detections(raw: dict) -> dict:
    return {k: (v[0], tuple(v[1]), v[2], v[3]) for k, v in (raw or {}).items()}


def replay(path: Path, *, ocr_fn: Optional[Callable] = None, vlm_fn: Optional[Callable] = None,
           detector=None, abstention=None, translate_fn: Optional[Callable] = None) -> list[ReplayRow]:
    """Run every recorded request again and compare with what happened live."""
    from .pipeline import PipelineInput, run_once

    root = session_file(path).parent
    out: list[ReplayRow] = []
    kw = {k: v for k, v in dict(ocr_fn=ocr_fn, vlm_fn=vlm_fn, detector=detector, abstention=abstention,
                                translate_fn=translate_fn).items() if v is not None}
    for i, rec in enumerate(r for r in load_session(path) if r["type"] == "request"):
        frame = cv2.imread(str(root / rec["frame"])) if rec.get("frame") else None
        res = run_once(PipelineInput(frame=frame, detections=_restore_detections(rec.get("detections"))),
                       rec["request"], **kw)
        was, now = rec["result"], res.to_dict()
        changes = [f"{k}: {was.get(k)!r} -> {now.get(k)!r}"
                   for k in ("status", "abstained", "modalities", "answer_text") if was.get(k) != now.get(k)]
        out.append(ReplayRow(i, rec["request"], not changes, changes, was, now))
    return out


def export_scenarios(path: Path, out: Path) -> int:
    """Turn recorded requests into eval-scenario skeletons. Ground truth is left
    null on purpose: a person has to label each one before it can be scored."""
    root = session_file(path).parent
    lines = []
    for i, rec in enumerate(r for r in load_session(path) if r["type"] == "request"):
        lines.append(json.dumps({
            "id": f"recorded_{i + 1:03d}", "category": "recorded",
            "input": {"image": str(root / rec["frame"]) if rec.get("frame") else None, "text": None, "audio": None},
            "user_request": rec["request"], "expected_modality": "ocr" if rec["request"] == "read" else "vlm",
            "ground_truth": None, "expected_behavior": "answer", "difficulty": "unknown",
            "notes": "recorded live; label ground_truth / expected_behavior before scoring",
            "recorded_answer": rec["result"].get("display_text"),
        }, ensure_ascii=False))
    Path(out).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)
