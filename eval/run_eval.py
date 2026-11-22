"""Run the benchmark through ``visionassist.pipeline.run_once``.

    python -m eval.run_eval --scenarios eval/scenarios.jsonl --out reports/eval_raw.json

Writes reports/eval_raw.json (machine-readable, one record per scenario) and
reports/eval_report.md. See eval/backends.py for what "auto" runs against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from visionassist.guardrails import AbstentionConfig
from visionassist.pipeline import PipelineInput, run_once

from . import backends as be
from .classify_failure import classify, is_failure
from .scoring import LLMJudge, keyword_judge, openrouter_call, score

ROOT = Path(__file__).resolve().parent.parent


def load_scenarios(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario ids")
    return rows


def load_frame(rel: str | None):
    """Image for a scenario, or None when there is none (a deliberate 'no
    camera' scenario) or the fixture file is missing."""
    if not rel:
        return None
    p = ROOT / rel
    return cv2.imread(str(p)) if p.exists() else None


def run_scenario(sc: dict, backend: be.Backends, acfg: AbstentionConfig) -> dict:
    inp = sc["input"]
    frame = load_frame(inp.get("image"))
    fixture_missing = bool(inp.get("image")) and frame is None
    request = inp["text"] if (inp.get("text") and frame is None) else sc["user_request"]
    res = run_once(PipelineInput(frame=frame, text=inp.get("text") if frame is None else None, audio=inp.get("audio")),
                   request, detector=backend.detector, ocr_fn=backend.ocr_fn, vlm_fn=backend.vlm_fn,
                   translate_fn=be.identity_translate, abstention=acfg)
    d = res.to_dict()
    raw = dict(res.raw)
    raw["intent"] = res.intent
    return {
        "id": sc["id"], "category": sc["category"], "expected_modality": sc["expected_modality"],
        "expected_behavior": sc["expected_behavior"],
        "actual_modalities": res.modalities, "raw_output": raw, "latency_ms": d["latency_ms"],
        "confidence": d["confidence"], "answer_text": d["answer_text"], "display_text": d["display_text"],
        "abstained": res.abstained, "reason": res.reason, "status": res.status, "intent": res.intent,
        "route": d["route"], "stage_ms": d["stage_ms"], "over_budget": d["over_budget"],
        "phases": d["phases"], "hedged": res.hedged, "fixture_missing": fixture_missing,
    }


def run_all(scenarios: list[dict], backend: be.Backends, acfg: AbstentionConfig, judge=None) -> list[dict]:
    judge = judge or keyword_judge
    out = []
    for sc in scenarios:
        rec = run_scenario(sc, backend, acfg)
        rec["scores"] = score(sc, rec, judge)
        rec["failure_bucket"] = classify(sc, rec) if is_failure(sc, rec) else None
        out.append(rec)
    return out


def make_judge(kind: str, cache: Path):
    if kind == "llm":
        from visionassist import config as cfg
        if cfg.OPENROUTER_API_KEY:
            model = (cfg.VLM_MODELS or ["google/gemma-4-26b-a4b-it:free"])[0]
            return LLMJudge(cache, openrouter_call(model), model), f"LLM judge ({model}), keyword fallback"
        print("SKIPPED: --judge llm needs OPENROUTER_API_KEY; using the keyword-overlap judge", file=sys.stderr)
    return LLMJudge(cache), "keyword-overlap judge (deterministic)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.run_eval", description=__doc__.splitlines()[0])
    ap.add_argument("--scenarios", default="eval/scenarios.jsonl")
    ap.add_argument("--out", default="reports/eval_raw.json")
    ap.add_argument("--report", default="reports/eval_report.md")
    ap.add_argument("--backend", default="auto", choices=["auto", "fake", "live"])
    ap.add_argument("--judge", default="keyword", choices=["keyword", "llm"])
    ap.add_argument("--abstention-config", default=None, help="default: config/abstention.json")
    ap.add_argument("--no-abstain", action="store_true", help="disable guardrails (records every stage)")
    ap.add_argument("--latency-summary", default="reports/latency_summary.md")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    from . import report

    scen = load_scenarios(Path(args.scenarios))
    if args.limit:
        scen = scen[:args.limit]
    backend = be.build(args.backend, Path(args.scenarios))
    acfg = AbstentionConfig.disabled() if args.no_abstain else AbstentionConfig.load(args.abstention_config)
    judge, judge_name = make_judge(args.judge, Path(args.out).parent / "vlm_judge_cache.json")
    recs = run_all(scen, backend, acfg, judge)
    if hasattr(judge, "save"):
        judge.save()

    meta = report.collect_meta(backend, acfg, judge_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "records": recs}, indent=1, ensure_ascii=False), encoding="utf-8")
    md = report.render(meta, scen, recs, Path(args.latency_summary))
    Path(args.report).write_text(md, encoding="utf-8")

    unclassified = sum(1 for r in recs if r["failure_bucket"] == "unclassified")
    fails = sum(1 for r in recs if r["failure_bucket"])
    print(f"{len(recs)} scenarios, {fails} not fully correct, {unclassified} unclassified -> {args.report}")
    return 1 if unclassified else 0


if __name__ == "__main__":
    raise SystemExit(main())
