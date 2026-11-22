# ADR 0004: Abstain before calling the VLM

- Status: accepted
- Date: 2026-11-21

## Context

A wrong confident answer is worse than "I'm not sure" for a blind user, and VLM calls cost latency and free-tier quota. Free VLMs hallucinate.

## Decision

`guardrails.should_abstain` runs on router and image-quality percepts before any model call (gate 1), and again with OCR/VLM confidence (gate 2). On abstention the user gets a natural clarification in Devanagari (voice) and Hinglish (screen), not an error. Thresholds live in `config/abstention.json`, tuned by `eval/calibrate_abstention.py` on a dev half and reported on a holdout half; `eval/kfold_abstention.py` adds k-fold mean and spread.

## Consequences

Some answerable requests get refused (6 of 60 in the synthetic set). OCR.space reports no confidence, so a text-plausibility and image-quality proxy is used for it; Tesseract's own confidence is used when available. Thresholds fitted on synthetic images may not transfer to real photographs.
