"""Render reports/eval_report.md from the raw records."""
from __future__ import annotations

import datetime as dt
import platform
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .classify_failure import BUCKETS
from .scoring import ABSTAIN_CATEGORIES, OCR_CATEGORIES


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def host_description() -> str:
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    import os
    gpu = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()
    return f"{cpu or 'unknown CPU'} ({os.cpu_count()} cores), GPU: {gpu[0] if gpu else 'none'}, {platform.platform()}"


def collect_meta(backend, acfg, judge_name: str) -> dict:
    import cv2
    commit = _run(["git", "rev-parse", "--short", "HEAD"]) or "n/a (not a git checkout)"
    return {
        "date": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M UTC%z"),
        "commit": commit, "host": host_description(),
        "python": platform.python_version(), "opencv": cv2.__version__,
        "backend": backend.name, "models": backend.describe(), "judge": judge_name,
        "abstention": {**acfg.thresholds(), "enabled": acfg.enabled},
    }


# ------------------------------------------------------------------- stats
def pct(x: float) -> str:
    return f"{x:.2f}"


def prf(counts: Counter) -> tuple[str, str, str]:
    tp, fp, fn = counts["correct"], counts["incorrect"], counts["abstained"]
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p is not None and r is not None and p + r else None
    fmt = lambda v: "n/a" if v is None else pct(v)  # noqa: E731
    return fmt(p), fmt(r), fmt(f)


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def modality_rows(scen: dict, recs: list[dict]) -> list[tuple]:
    groups = {
        "OCR: medicine labels": [r for r in recs if r["category"] == "ocr_medicine"],
        "OCR: signs": [r for r in recs if r["category"] == "ocr_sign"],
        "OCR: menus": [r for r in recs if r["category"] == "ocr_menu"],
        "OCR: all": [r for r in recs if r["category"] in OCR_CATEGORIES],
        "Scene description (VLM)": [r for r in recs if r["category"] == "scene_description"],
        "Voice command (intent)": [r for r in recs if r["category"] == "voice_command"],
    }
    rows = []
    for name, rs in groups.items():
        c = Counter(r["scores"]["verdict"] for r in rs)
        p, r_, f = prf(c)
        n = len(rs)
        notes = [f"acc {c['correct']}/{n}"]
        if c["partial"]:
            notes.append(f"{c['partial']} partial")
        if c["abstained"]:
            notes.append(f"{c['abstained']} abstained")
        if name.startswith("OCR"):
            ran = [r["scores"] for r in rs if "cer" in r["scores"]]
            if ran:
                notes.append(f"CER {pct(statistics.mean(s['cer'] for s in ran))}, "
                             f"WER {pct(statistics.mean(s['wer'] for s in ran))} over {len(ran)} runs that reached OCR")
            notes.append(f"exact match {pct(c['correct'] / n)}")
        rows.append((name, n, p, r_, f, "; ".join(notes)))
    return rows


# ------------------------------------------------------------------ render
LIMITATIONS = """\
- **Synthetic images only.** Every fixture is rendered with PIL / OpenCV (flat text on a background, coloured blocks for
  "people" and "chairs"). There are no photographs, no real handwriting, no motion blur from a moving camera, and no
  Devanagari text. Numbers here say little about performance on real street, shop or pharmacy photos.
- **No live camera.** Frames come from files; camera loss, auto-exposure and focus behaviour are untested here.
- **ASR is not exercised.** Voice scenarios feed typed or simulated transcripts to the intent router. Real Vosk
  recognition of Hindi (accuracy, noise, accents, the grammar restriction) was not run, so `asr_substitution` here
  only covers two hand-written mis-transcriptions.
- **Piper (text to speech) is not exercised.** The voice text is checked as Devanagari strings, never as audio.
- **Translation is a pass-through.** Hindi translation of OCR text goes through the network (deep-translator), so the
  harness does not call it; OCR text is shown as read.
- **Detector and VLM are stand-ins unless the header says otherwise.** The colour-blob demo detector knows two classes
  and keeps one box per class (so counting scenarios and the dog/bus scenarios fail by construction), and the template
  "VLM" only restates detections. Real YOLO and real OpenRouter VLMs were not run; scene numbers measure the
  pipeline around them, not the models.
- **OCR.space was not run.** If the header shows Tesseract, that is the OCR engine measured; OCR.space (the app's
  default engine) has no confidence output, so the app uses an image-quality and text-plausibility proxy for it.
- **Abstention thresholds were tuned on the 40 even-indexed (dev) scenarios.** The 10/10 above covers all 80 and so
  includes the dev half; the number to quote is the holdout result in `reports/calibration.md`. Holdout has only about
  5 negatives, so it is a weak estimate. Other pipeline constants (image-quality gates in `quality.py`) were chosen by
  looking at these same fixtures.
- **The keyword judge is crude** (keyword overlap, side and count contradiction checks). It is deterministic, not
  accurate; an LLM judge (`--judge llm`) was not run here.
- **Small n.** 8 to 20 scenarios per row; one scenario moves a metric by 5 to 12 points. No confidence intervals.
- **Latency numbers are from this machine only** and, unless the latency table says otherwise, exclude network
  stages (OCR.space, VLM) that dominate in the real app.
"""


def render(meta: dict, scen_list: list[dict], recs: list[dict], latency_md: Path) -> str:
    scen = {s["id"]: s for s in scen_list}
    L: list[str] = []
    L += ["# VisionAssist evaluation report", "",
          f"- **Date:** {meta['date']}", f"- **Commit:** {meta['commit']}",
          f"- **Host:** {meta['host']}", f"- **Python / OpenCV:** {meta['python']} / {meta['opencv']}",
          f"- **Backend:** `{meta['backend']}`",
          *[f"- **{k.upper()}:** {v}" for k, v in meta["models"].items()],
          f"- **Judge:** {meta['judge']}",
          f"- **Abstention thresholds:** `{meta['abstention']}`", "",
          f"> {len(recs)} scenarios. Read the Known limitations section before quoting any number: "
          "the inputs are synthetic and, depending on the backend above, parts of the stack are stand-ins.", ""]

    L += ["## Results by modality", "",
          "| modality | n | precision | recall | F1 | notes |", "|---|---|---|---|---|---|"]
    for name, n, p, r, f, notes in modality_rows(scen, recs):
        L.append(f"| {name} | {n} | {p} | {r} | {f} | {notes} |")
    L += ["",
          "Per scenario, *correct* is a true positive and *incorrect* (an answer that is wrong) a false positive. "
          "*Abstained* on an answerable request is a false negative, and *partial* (scene descriptions only) is "
          "counted in neither. precision = TP/(TP+FP), recall = TP/(TP+FN); accuracy in the notes is TP/n. "
          "OCR is correct only on an exact match after lower-casing and removing punctuation (CER = 0).", ""]

    # ---- Part 3
    neg = [r for r in recs if r["category"] == "negative_abstain"]
    neg_ok = sum(1 for r in neg if r["abstained"])
    L += ["## Abstention", "", f"**Abstention accuracy: {neg_ok}/{len(neg)}**", "",
          "| category | n | abstention_accuracy |", "|---|---|---|"]
    for cat in ABSTAIN_CATEGORIES:
        rs = [r for r in recs if r["category"] == cat]
        ok = sum(1 for r in rs if r["abstained"])
        L.append(f"| {cat} | {len(rs)} | {pct(ok / len(rs)) if rs else 'n/a'} ({ok}/{len(rs)}) |")
    pos = [r for r in recs if r["expected_behavior"] == "answer"]
    wrong_abstain = [r["id"] for r in pos if r["abstained"]]
    reasons = [r for r in recs if r["expected_behavior"] == "abstain" and r["abstained"]]
    rm = sum(1 for r in reasons if r["scores"].get("reason_match"))
    L += ["",
          f"Answerable scenarios that were wrongly refused: {len(wrong_abstain)}/{len(pos)}"
          + (f" ({', '.join(wrong_abstain)})" if wrong_abstain else "") + ". "
          f"Abstentions whose reason matched the scenario's expected reason: {rm}/{len(reasons)}.", ""]

    # ---- Part 2
    first: dict[str, str] = {}
    counts: Counter = Counter()
    for r in recs:
        if r["failure_bucket"]:
            counts[r["failure_bucket"]] += 1
            first.setdefault(r["failure_bucket"], r["id"])
    L += ["## Failure taxonomy", "", "| Bucket | Count | Example ID |", "|---|---|---|"]
    for b in BUCKETS:
        L.append(f"| {b} | {counts[b]} | {first.get(b, '—')} |")
    total = sum(counts.values())
    L += ["", f"{total} of {len(recs)} scenarios were not fully correct; each has exactly one bucket "
              f"(`unclassified` = {counts['unclassified']}).", ""]
    if total:
        L += ["<details><summary>Every failure</summary>", "", "| id | bucket | request | what happened |", "|---|---|---|---|"]
        for r in recs:
            if r["failure_bucket"]:
                s = scen[r["id"]]
                got = (r["raw_output"].get("raw_text") or r["display_text"] or r["reason"] or "")[:50].replace("|", "/")
                L.append(f"| {r['id']} | {r['failure_bucket']} | {s['user_request']} | "
                         f"{'abstained' if r['abstained'] else 'answered'}: {got} |")
        L += ["", "</details>", ""]

    # ---- Part 4
    L += ["## Latency", ""]
    if latency_md.exists():
        L += [latency_md.read_text(encoding="utf-8").strip(), ""]
    else:
        L += ["`reports/latency_summary.md` not found (run `python tools/latency_benchmark.py`); "
              "timings recorded during this evaluation run:", "",
              "| Stage | n | p50 (ms) | p95 (ms) |", "|---|---|---|---|"]
        per = defaultdict(list)
        for r in recs:
            for k, v in r["stage_ms"].items():
                per[k].append(v)
        for k, v in per.items():
            L.append(f"| {k} | {len(v)} | {percentile(v, 50):.1f} | {percentile(v, 95):.1f} |")
        e2e = [r["latency_ms"] for r in recs]
        L.append(f"| End-to-end | {len(e2e)} | {percentile(e2e, 50):.1f} | {percentile(e2e, 95):.1f} |")
        L.append("")

    L += ["## Known limitations", "", LIMITATIONS]
    return "\n".join(L)
