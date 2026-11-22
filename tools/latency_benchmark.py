"""Per-stage and end-to-end latency benchmark.

    python tools/latency_benchmark.py                 # whatever can run here
    python tools/latency_benchmark.py --ocr-engine ocrspace --detector yolo

Each stage runs in isolation on a fixed workload:
    ASR 5 clips | OCR 10 images | detection 10 frames | VLM 5 prompts |
    end-to-end 20 scenarios from eval/scenarios.jsonl (pipeline.run_once)
"Orchestration overhead" is end-to-end time minus the time spent inside the
OCR / detector / VLM calls, i.e. what the pipeline itself costs.

Method
  * time.perf_counter(); one warm-up call per stage is made and DISCARDED (the
    first call loads models, opens connections and fills caches, so it is not
    representative). Warm-up is not in the CSV.
  * p50 / p95 use linear interpolation. With n = 5 or 10, p95 is close to the
    maximum, so read it as "worst case seen".
  * A stage that cannot run here prints "SKIPPED: <reason>" and is written as
    n/a. Nothing is estimated or filled in.

Output: logs/latency_report.csv (stage, run_idx, latency_ms) and
reports/latency_summary.md.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import cv2  # noqa: E402
from eval import backends as be  # noqa: E402
from eval.report import host_description, percentile  # noqa: E402
from eval.run_eval import load_frame, load_scenarios  # noqa: E402

from visionassist import config as cfg  # noqa: E402
from visionassist.guardrails import AbstentionConfig  # noqa: E402
from visionassist.pipeline import PipelineInput, run_once  # noqa: E402

STAGES = ["ASR", "OCR", "Detection", "VLM", "Orchestration overhead", "End-to-end"]


def timed(fn, n_warm=1):
    """Run fn once to warm up (discarded); then yield (index, ms) per call."""
    def run(items):
        for it in items[:1] * n_warm:
            fn(it)
        out = []
        for it in items:
            t0 = time.perf_counter()
            fn(it)
            out.append((time.perf_counter() - t0) * 1000.0)
        return out
    return run


def bench_asr(clips_dir: Path):
    wavs = sorted(clips_dir.glob("*.wav"))[:5]
    if len(wavs) < 5:
        return None, f"needs 5 .wav clips in {clips_dir}/ (found {len(wavs)}); none ship with the repo"
    try:
        import json
        import wave

        from vosk import KaldiRecognizer, Model
    except ImportError:
        return None, "vosk is not installed"
    model_dir = Path(cfg.VOSK_MODEL_PATH)
    if not model_dir.exists():
        return None, f"Vosk model not found at {model_dir} (run `visionassist download-models`)"
    model = Model(str(model_dir))
    from visionassist.voice import GRAMMAR

    def recognise(path):
        with wave.open(str(path), "rb") as w:
            rec = KaldiRecognizer(model, w.getframerate(), GRAMMAR)
            while True:
                data = w.readframes(4000)
                if not data:
                    break
                rec.AcceptWaveform(data)
            return json.loads(rec.FinalResult())
    return timed(recognise)(wavs), None


def bench_ocr(engine: str, images: list):
    from visionassist import ocr
    if engine == "auto":
        engine = "ocrspace" if cfg.OCR_API_KEY else "tesseract"
    if engine == "tesseract":
        if not ocr.tesseract_available():
            return None, "tesseract binary not found", engine
        fn = ocr.read_text_tesseract
    else:
        if not cfg.OCR_API_KEY:
            return None, "OCR_API_KEY not set (OCR.space needs network and a key)", engine
        cfg.OCR_ENGINE = "ocrspace"       # read_text_from_frame checks this at call time
        fn = ocr.read_text_from_frame
    return timed(fn)(images), None, engine


def bench_detection(kind: str, frames: list):
    if kind in ("auto", "yolo"):
        try:
            from visionassist.detector import Detector
            det = Detector()
            label = f"YOLO {cfg.YOLO_MODEL.name}"
        except Exception as exc:
            if kind == "yolo":
                return None, f"YOLO unavailable ({type(exc).__name__}: {exc})", ""
            return None, f"YOLO unavailable ({type(exc).__name__}); the demo blob model is not timed because its latency says nothing about YOLO", ""
        return timed(det.detect_still)(frames), None, label
    from visionassist.demo import make_demo_detector
    return timed(make_demo_detector().detect_still)(frames), None, "demo blob model (NOT YOLO)"


def bench_vlm(frames: list):
    if not cfg.OPENROUTER_API_KEY:
        return None, "OPENROUTER_API_KEY not set (the VLM is a network call)"
    from visionassist.demo import make_demo_detector
    from visionassist.vlm import describe_scene_structured
    det = make_demo_detector()

    def call(frame):
        describe_scene_structured(frame, det.detect_still(frame), frame.shape)
    return timed(call)(frames), None


def bench_pipeline(scenarios: list, backend: be.Backends):
    acfg = AbstentionConfig.load()
    rows = []                                  # (e2e_ms, inner_ms)
    warm = scenarios[0]
    for phase in ("warm", "timed"):
        for sc in ([warm] if phase == "warm" else scenarios):
            frame = load_frame(sc["input"].get("image"))
            inp = PipelineInput(frame=frame, text=sc["input"].get("text") if frame is None else None)
            request = inp.text if inp.text else sc["user_request"]
            t0 = time.perf_counter()
            res = run_once(inp, request, detector=backend.detector, ocr_fn=backend.ocr_fn,
                           vlm_fn=backend.vlm_fn, translate_fn=be.identity_translate, abstention=acfg)
            e2e = (time.perf_counter() - t0) * 1000.0
            if phase == "timed":
                inner = sum(t.ms for t in res.traces if t.stage in ("ocr", "detect", "vlm"))
                rows.append((e2e, inner))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenarios", default="eval/scenarios.jsonl")
    ap.add_argument("--ocr-engine", default="auto", choices=["auto", "tesseract", "ocrspace"])
    ap.add_argument("--detector", default="auto", choices=["auto", "yolo", "demo"],
                    help="auto: YOLO if installed, otherwise skip. demo: time the colour-blob stand-in (labelled).")
    ap.add_argument("--audio-dir", default="eval/audio")
    ap.add_argument("--csv", default="logs/latency_report.csv")
    ap.add_argument("--summary", default="reports/latency_summary.md")
    args = ap.parse_args(argv)

    scen = load_scenarios(ROOT / args.scenarios)
    img_scen = [s for s in scen if s["input"].get("image") and (ROOT / s["input"]["image"]).exists()]
    ocr_imgs = [load_frame(s["input"]["image"]) for s in img_scen if s["category"].startswith("ocr_")][:10]
    scene_frames = [load_frame(s["input"]["image"]) for s in img_scen if s["category"] == "scene_description"]
    det_frames = (scene_frames * 2)[:10]
    vlm_frames = scene_frames[:5]
    e2e_scen = scen[::4][:20]

    results: dict[str, list[float] | None] = {}
    notes: dict[str, str] = {}

    results["ASR"], why = bench_asr(ROOT / args.audio_dir)
    notes["ASR"] = why or "Vosk, restricted grammar, 5 clips"
    ocr_res, why, engine = bench_ocr(args.ocr_engine, ocr_imgs)
    results["OCR"], notes["OCR"] = ocr_res, why or f"{engine}, 10 synthetic text images"
    det_res, why, label = bench_detection(args.detector, det_frames)
    results["Detection"], notes["Detection"] = det_res, why or f"{label}, 10 scene frames (640x480)"
    try:
        results["VLM"], why = bench_vlm(vlm_frames)
    except Exception as exc:        # rate limit, timeout, provider down: report it, do not crash
        results["VLM"], why = None, f"VLM calls failed ({type(exc).__name__}: {str(exc)[:120]}). Free models are often rate limited: retry later or set VLM_MODELS"
    notes["VLM"] = why or "OpenRouter free models, 5 scene images"

    backend = be.build("auto", ROOT / args.scenarios)
    rows = bench_pipeline(e2e_scen, backend)
    results["End-to-end"] = [r[0] for r in rows]
    results["Orchestration overhead"] = [max(0.0, r[0] - r[1]) for r in rows]
    notes["End-to-end"] = (f"run_once on every 4th scenario (20, a mix: some abstain before any model call); OCR: {backend.notes['ocr'].split(' (')[0]}; "
                           f"detector: {backend.notes['detector']}; VLM: {backend.notes['vlm']}")
    notes["Orchestration overhead"] = "end-to-end minus time inside OCR / detector / VLM calls"

    for stage in STAGES:
        if results[stage] is None:
            print(f"{stage}: SKIPPED: {notes[stage]}")
        else:
            v = results[stage]
            print(f"{stage}: n={len(v)} p50={percentile(v, 50):.1f} ms p95={percentile(v, 95):.1f} ms")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "run_idx", "latency_ms"])
        for stage in STAGES:
            if results[stage] is None:
                w.writerow([stage, "n/a", "n/a"])
            else:
                for i, ms in enumerate(results[stage]):
                    w.writerow([stage, i, f"{ms:.3f}"])

    lines = ["| Stage | n | p50 (ms) | p95 (ms) |", "|---|---|---|---|"]
    for stage in STAGES:
        v = results[stage]
        lines.append(f"| {stage} | n/a | n/a | n/a |" if v is None else
                     f"| {stage} | {len(v)} | {percentile(v, 50):.1f} | {percentile(v, 95):.1f} |")
    lines += ["", f"Measured on {host_description()} with Python {sys.version.split()[0]}, OpenCV {cv2.__version__}; "
                  "model versions per stage are listed below. Run `python tools/latency_benchmark.py` to reproduce "
                  "on your hardware.", "",
              "Stage details (one warm-up call per stage is made and discarded; p95 on n=5 or 10 is near the maximum):", ""]
    for stage in STAGES:
        lines.append(f"- **{stage}:** " + (f"SKIPPED: {notes[stage]}" if results[stage] is None else notes[stage]))
    skipped = [s for s in STAGES if results[s] is None]
    if skipped:
        lines += ["", "Skipped stages are not estimated. To fill them in on your own machine: "
                      "set `OPENROUTER_API_KEY` (VLM), install `ultralytics` and run "
                      "`visionassist download-models` (detection), put five Hindi command clips in `eval/audio/` and "
                      "download the Vosk model (ASR)."]
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.csv} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
