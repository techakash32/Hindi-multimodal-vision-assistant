"""Stratified k-fold cross-validation of the abstention thresholds.

    python -m eval.kfold_abstention            # k=5, writes reports/kfold_abstention.md

A single 40/40 split leaves only ~5 negative scenarios in the holdout, so one
scenario moves the result by 20 points. Here every scenario is held out exactly
once: for each fold the thresholds are fitted on the other folds (same sweep and
selection rule as ``calibrate_abstention``) and scored on the held fold. Folds are
stratified by category (round-robin within each category, deterministic), so each
holds a share of the negatives.

Reported: pooled out-of-fold abstention on negatives (every negative is scored by
thresholds that never saw it), and mean and standard deviation across folds for
selective accuracy and coverage, next to the no-guardrail baseline on the same fold.
Folds are small (16 scenarios), so the spread is wide; that is the honest answer.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from visionassist.guardrails import AbstentionConfig

from . import backends as be
from .calibrate_abstention import choose, evaluate, pick_high, sweep
from .run_eval import load_scenarios, run_all


def stratified_folds(rows, k: int) -> list[list]:
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row[0]["category"]].append(row)
    folds = [[] for _ in range(k)]
    i = 0
    for cat in sorted(by_cat):
        for row in by_cat[cat]:
            folds[i % k].append(row)
            i += 1
    return folds


def cross_validate(rows, k: int = 5, min_coverage: float = 0.90, hedge_target: float = 0.95) -> dict:
    folds = stratified_folds(rows, k)
    base_cfg = AbstentionConfig(ocr_tau=0.0, vlm_tau=0.0, aggregate_tau_low=0.0)
    per_fold, neg_ok, neg_n = [], 0, 0
    for i, held in enumerate(folds):
        train = [r for j, f in enumerate(folds) if j != i for r in f]
        pick, _ = choose(sweep(train, evaluate(train, base_cfg), min_coverage))
        if pick is None:
            per_fold.append({"fold": i, "infeasible": True})
            continue
        chosen = dict(pick[0])
        chosen["aggregate_tau_high"] = pick_high(train, AbstentionConfig(**chosen, aggregate_tau_high=1.0), hedge_target)
        m, b = evaluate(held, AbstentionConfig(**chosen)), evaluate(held, base_cfg)
        neg_ok += m["neg_ok"]
        neg_n += m["neg_n"]
        per_fold.append({"fold": i, "thresholds": chosen, "n": len(held), "neg": f"{m['neg_ok']}/{m['neg_n']}",
                         "sel_acc": m["sel_acc"], "coverage": m["coverage"],
                         "base_sel_acc": b["sel_acc"], "base_coverage": b["coverage"]})
    ok = [f for f in per_fold if "infeasible" not in f]

    def ms(key):
        v = [f[key] for f in ok]
        return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)

    return {"k": k, "folds": per_fold, "neg_ok": neg_ok, "neg_n": neg_n, "sel_acc": ms("sel_acc"),
            "coverage": ms("coverage"), "base_sel_acc": ms("base_sel_acc"), "base_coverage": ms("base_coverage")}


def render(res: dict, backend_name: str, n: int) -> str:
    f = lambda t: f"{t[0]:.2f} ± {t[1]:.2f}"  # noqa: E731
    lines = [
        "# Abstention: stratified k-fold cross-validation", "",
        f"Backend `{backend_name}`; {n} scenarios; k = {res['k']}; thresholds fitted on k-1 folds, scored on the held fold.", "",
        f"**Pooled out-of-fold abstention on negatives: {res['neg_ok']}/{res['neg_n']}**", "",
        "| metric | with guardrails (mean ± sd over folds) | no guardrails (same folds) |", "|---|---|---|",
        f"| selective accuracy | {f(res['sel_acc'])} | {f(res['base_sel_acc'])} |",
        f"| coverage | {f(res['coverage'])} | {f(res['base_coverage'])} |", "",
        "| fold | n | negatives abstained | selective acc | coverage | fitted thresholds |", "|---|---|---|---|---|---|",
    ]
    for r in res["folds"]:
        if "infeasible" in r:
            lines.append(f"| {r['fold'] + 1} | | infeasible | | | |")
        else:
            t = r["thresholds"]
            lines.append(f"| {r['fold'] + 1} | {r['n']} | {r['neg']} | {r['sel_acc']:.2f} | {r['coverage']:.2f} | "
                         f"ocr {t['ocr_tau']}, vlm {t['vlm_tau']}, low {t['aggregate_tau_low']}, high {t['aggregate_tau_high']} |")
    lines += ["", "Each fold holds only about 2 negatives and 16 scenarios, so per-fold numbers are noisy; the pooled count "
              "and the spread are the honest summary. Inputs are synthetic (see the evaluation report's limitations)."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.kfold_abstention")
    ap.add_argument("--scenarios", default="eval/scenarios.jsonl")
    ap.add_argument("--backend", default="auto", choices=["auto", "fake", "live"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--report", default="reports/kfold_abstention.md")
    args = ap.parse_args(argv)
    scen = load_scenarios(Path(args.scenarios))
    backend = be.build(args.backend, Path(args.scenarios))
    rows = list(zip(scen, run_all(scen, backend, AbstentionConfig.disabled()), strict=True))
    res = cross_validate(rows, args.k)
    text = render(res, backend.name, len(rows))
    print(text)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
