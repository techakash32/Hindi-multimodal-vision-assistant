# Changelog

## Unreleased (after 1.0.0)
- Session recording (`run --record`) and `visionassist replay` through `pipeline.run_once`.
- Hinglish transliterator rewritten around a 200+ pair golden file: flap consonants (ड़ ढ़), right-to-left schwa deletion, long-vowel and final-vowel rules, m/n before labials, `-ein`/`-yein` endings, loanword dictionary. Fixed real bugs the golden file found (`parho`, `thora`, `Prvesh`, `huee`).
- Synthetic adversarial pack (20 cases) with a with/without-guardrails comparison.
- Stratified k-fold cross-validation of abstention thresholds.
- Architecture decision records in `docs/adr/`.

## 1.0.0, second pass
- Abstention thresholds tuned on a dev half and reported on a holdout half; `LOCAL_ONLY` privacy mode; SHA256 pinning for model downloads; Dockerfile; accessibility basics; local setup guide.

## 1.0.0, first pass: evaluation
- Evaluation harness (80 scenarios), failure taxonomy, guardrails, latency benchmark.

## 1.0.0, base
- Hindi-only speech, Hinglish screen text, monochrome dashboard with radar view.
