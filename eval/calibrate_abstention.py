"""Pick abstention thresholds from the benchmark.

    python -m eval.calibrate_abstention            # sweep, print, write config/abstention.json

How it works
  1. Run every scenario once with the guardrails *disabled*, so every stage
     executes and the pipeline records the percepts (confidences) at each gate.
  2. Replay those recorded percepts through ``guardrails.staged_decision`` for
     each candidate (ocr_tau, vlm_tau, aggregate_tau_low). It is the same code
     the app runs, so the thresholds are tuned on the real decision logic.
  3. Objective: maximise abstention accuracy on the ``negative_abstain``
     scenarios, subject to the *selective accuracy* on answerable scenarios
     (correct answers / answers given) being at least the no-guardrail
     baseline and coverage (share still answered) staying above
     ``--min-coverage`` of baseline coverage. Without the coverage floor,
     "refuse everything" would trivially satisfy both.
  4. Many settings tie. The one closest to the centre of the optimal plateau is
     chosen (not the edge), because the edge sits right on a fixture.
  5. aggregate_tau_high (when an answer is hedged with "shayad") is the lowest
     value above aggregate_tau_low from which answers are at least
     ``--hedge-target`` accurate.

Split: thresholds are fitted on the even-indexed half (dev) only; the odd-indexed
half (holdout) is scored once and reported as the headline number.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from visionassist.guardrails import AbstentionConfig, Percept, _measured, staged_decision

from . import backends as be
from .run_eval import load_scenarios, run_all

GRID = [round(0.20 + 0.05 * i, 2) for i in range(0, 15)]       # 0.20 .. 0.90


def replay(rec: dict, cfg: AbstentionConfig) -> bool:
    """Would the guardrails abstain on this recorded run under ``cfg``?"""
    if rec["status"] == "no_text":                              # OCR found nothing: no threshold involved
        return True
    phases = [{k: Percept.from_dict(v) for k, v in ph.items()} for ph in rec["phases"]]
    return staged_decision(phases, cfg)[0] if phases else rec["abstained"]


def evaluate(rows: list[tuple[dict, dict]], cfg: AbstentionConfig) -> dict:
    neg = [(s, r) for s, r in rows if s["category"] == "negative_abstain"]
    pos = [(s, r) for s, r in rows if s["expected_behavior"] == "answer" and s["category"] != "voice_command"]
    neg_ok = sum(1 for _s, r in neg if replay(r, cfg))
    answered = [(s, r) for s, r in pos if not replay(r, cfg)]
    correct = sum(1 for _s, r in answered if r["scores"]["verdict"] == "correct")
    return {"neg_ok": neg_ok, "neg_n": len(neg), "answered": len(answered), "pos_n": len(pos),
            "correct": correct, "sel_acc": correct / len(answered) if answered else 0.0,
            "coverage": len(answered) / len(pos) if pos else 0.0}


def sweep(rows, baseline, min_coverage):
    results = []
    for o, v, a in itertools.product(GRID, GRID, GRID):
        cfg = AbstentionConfig(ocr_tau=o, vlm_tau=v, aggregate_tau_low=a, aggregate_tau_high=1.0)
        m = evaluate(rows, cfg)
        ok = m["sel_acc"] >= baseline["sel_acc"] - 1e-9 and m["coverage"] >= baseline["coverage"] * min_coverage
        results.append(({"ocr_tau": o, "vlm_tau": v, "aggregate_tau_low": a}, m, ok))
    return results


def choose(results):
    feasible = [(t, m) for t, m, ok in results if ok]
    if not feasible:
        return None, []
    best = max(m["neg_ok"] for _t, m in feasible)
    plateau = [(t, m) for t, m in feasible if m["neg_ok"] == best]
    centre = {k: statistics.mean(t[k] for t, _m in plateau) for k in plateau[0][0]}
    pick = min(plateau, key=lambda tm: sum((tm[0][k] - centre[k]) ** 2 for k in centre))
    return pick, plateau


def pick_high(rows, low_cfg: AbstentionConfig, target: float) -> float:
    pts = []
    for s, r in rows:
        if s["expected_behavior"] != "answer" or s["category"] == "voice_command" or replay(r, low_cfg):
            continue
        ph = [{k: Percept.from_dict(v) for k, v in ph.items()} for ph in r["phases"]][-1]
        meas = _measured(ph)
        if meas:
            pts.append((sum(p.confidence for p in meas) / len(meas), r["scores"]["verdict"] == "correct"))
    for t in [round(low_cfg.aggregate_tau_low + 0.05 * i, 2) for i in range(1, 12)]:
        above = [ok for c, ok in pts if c >= t]
        if len(above) >= 5 and sum(above) / len(above) >= target:
            return t
    return 0.70


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.calibrate_abstention", description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenarios", default="eval/scenarios.jsonl")
    ap.add_argument("--backend", default="auto", choices=["auto", "fake", "live"])
    ap.add_argument("--out", default="config/abstention.json")
    ap.add_argument("--report", default="reports/calibration.md")
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--hedge-target", type=float, default=0.95)
    ap.add_argument("--dry-run", action="store_true", help="print only; do not write the config")
    args = ap.parse_args(argv)

    scen = load_scenarios(Path(args.scenarios))
    backend = be.build(args.backend, Path(args.scenarios))
    recs = run_all(scen, backend, AbstentionConfig.disabled())
    rows = list(zip(scen, recs, strict=True))
    # Fixed split: even-indexed scenarios are "dev" (tune here), odd-indexed are
    # "holdout" (report only). The file is grouped by category and alternates
    # within each, so both halves hold about 40 scenarios of every kind.
    dev, held = rows[::2], rows[1::2]

    base_cfg = AbstentionConfig(ocr_tau=0.0, vlm_tau=0.0, aggregate_tau_low=0.0)
    baseline = evaluate(dev, base_cfg)
    allres = sweep(dev, baseline, args.min_coverage)
    pick, plateau = choose(allres)
    if pick is None:
        print("no threshold setting satisfies the constraints; config left unchanged")
        return 1
    chosen, m = pick
    cfg = AbstentionConfig(**chosen, aggregate_tau_high=1.0)
    chosen["aggregate_tau_high"] = pick_high(dev, cfg, args.hedge_target)

    full_cfg = AbstentionConfig(**chosen)
    hm, hb = evaluate(held, full_cfg), evaluate(held, base_cfg)
    held_line = (f"holdout ({len(held)} scenarios, never used for tuning): abstention {hm['neg_ok']}/{hm['neg_n']}, "
                 f"selective accuracy {hm['sel_acc']:.2f} (no-guardrail baseline {hb['sel_acc']:.2f}), "
                 f"coverage {hm['coverage']:.2f} (baseline {hb['coverage']:.2f})")

    lines = [
        "# Abstention calibration", "",
        f"Backend `{backend.name}`; tuned on {len(dev)} dev scenarios, reported on {len(held)} holdout; grid {GRID[0]}..{GRID[-1]} step 0.05 on three thresholds.", "",
        f"Baseline (no guardrails): selective accuracy {baseline['sel_acc']:.2f}, coverage {baseline['coverage']:.2f} "
        f"({baseline['answered']}/{baseline['pos_n']} answerable scenarios answered; "
        f"negatives refused anyway {baseline['neg_ok']}/{baseline['neg_n']}).", "",
        f"Feasible optimum: {m['neg_ok']}/{m['neg_n']} negatives refused; plateau of {len(plateau)} settings; "
        f"centre pick {chosen} -> selective accuracy {m['sel_acc']:.2f}, coverage {m['coverage']:.2f}.", "",
        "## Sweep: best result per ocr_tau (other thresholds free, constraints applied)", "",
        "| ocr_tau | best neg abstained | sel. acc | coverage |", "|---|---|---|---|",
    ]
    for o in GRID:
        cand = [(mm, ok) for t, mm, ok in allres if t["ocr_tau"] == o and ok]
        if cand:
            mm = max((c[0] for c in cand), key=lambda x: (x["neg_ok"], x["coverage"]))
            lines.append(f"| {o:.2f} | {mm['neg_ok']}/{mm['neg_n']} | {mm['sel_acc']:.2f} | {mm['coverage']:.2f} |")
        else:
            lines.append(f"| {o:.2f} | infeasible | | |")
    lines += ["", "## Holdout result (the number to quote)", "", held_line, "",
              "Thresholds were fitted on the dev half only. The sweep above is dev data. Holdout n is small (about 5 negatives), so one scenario moves abstention by 20 points."]
    text = "\n".join(lines)
    print(text)
    if not args.dry_run:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n", encoding="utf-8")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
        print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
