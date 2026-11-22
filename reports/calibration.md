# Abstention calibration

Backend `auto`; tuned on 40 dev scenarios, reported on 40 holdout; grid 0.2..0.9 step 0.05 on three thresholds.

Baseline (no guardrails): selective accuracy 0.88, coverage 1.00 (26/26 answerable scenarios answered; negatives refused anyway 2/5).

Feasible optimum: 5/5 negatives refused; plateau of 2296 settings; centre pick {'ocr_tau': 0.5, 'vlm_tau': 0.5, 'aggregate_tau_low': 0.5, 'aggregate_tau_high': 0.7} -> selective accuracy 0.88, coverage 1.00.

## Sweep: best result per ocr_tau (other thresholds free, constraints applied)

| ocr_tau | best neg abstained | sel. acc | coverage |
|---|---|---|---|
| 0.20 | 5/5 | 0.88 | 1.00 |
| 0.25 | 5/5 | 0.88 | 1.00 |
| 0.30 | 5/5 | 0.88 | 1.00 |
| 0.35 | 5/5 | 0.88 | 1.00 |
| 0.40 | 5/5 | 0.88 | 1.00 |
| 0.45 | 5/5 | 0.88 | 1.00 |
| 0.50 | 5/5 | 0.88 | 1.00 |
| 0.55 | 5/5 | 0.88 | 1.00 |
| 0.60 | 5/5 | 0.88 | 1.00 |
| 0.65 | 5/5 | 0.88 | 1.00 |
| 0.70 | 5/5 | 0.88 | 1.00 |
| 0.75 | 5/5 | 0.88 | 1.00 |
| 0.80 | 5/5 | 0.88 | 1.00 |
| 0.85 | 5/5 | 0.88 | 1.00 |
| 0.90 | infeasible | | |

## Holdout result (the number to quote)

holdout (40 scenarios, never used for tuning): abstention 5/5, selective accuracy 0.85 (no-guardrail baseline 0.85), coverage 1.00 (baseline 1.00)

Thresholds were fitted on the dev half only. The sweep above is dev data. Holdout n is small (about 5 negatives), so one scenario moves abstention by 20 points.
