# Abstention: stratified k-fold cross-validation

Backend `auto`; 80 scenarios; k = 5; thresholds fitted on k-1 folds, scored on the held fold.

**Pooled out-of-fold abstention on negatives: 10/10**

| metric | with guardrails (mean ± sd over folds) | no guardrails (same folds) |
|---|---|---|
| selective accuracy | 0.77 ± 0.13 | 0.72 ± 0.09 |
| coverage | 0.93 ± 0.08 | 0.98 ± 0.04 |

| fold | n | negatives abstained | selective acc | coverage | fitted thresholds |
|---|---|---|---|---|---|
| 1 | 16 | 2/2 | 0.78 | 0.82 | ocr 0.5, vlm 0.5, low 0.5, high 0.7 |
| 2 | 16 | 2/2 | 0.90 | 0.91 | ocr 0.5, vlm 0.4, low 0.45, high 0.7 |
| 3 | 16 | 2/2 | 0.70 | 1.00 | ocr 0.5, vlm 0.4, low 0.45, high 0.7 |
| 4 | 16 | 2/2 | 0.60 | 1.00 | ocr 0.55, vlm 0.4, low 0.45, high 0.7 |
| 5 | 16 | 2/2 | 0.89 | 0.90 | ocr 0.5, vlm 0.5, low 0.45, high 0.7 |

Each fold holds only about 2 negatives and 16 scenarios, so per-fold numbers are noisy; the pooled count and the spread are the honest summary. Inputs are synthetic (see the evaluation report's limitations).
