# ADR 0003: One `run_once` entry point for the app and the evaluation

- Status: accepted
- Date: 2026-11-21

## Context

An evaluation that exercises different code from the app proves nothing about the app. The original engine mixed OCR, VLM, speech and UI state in one class.

## Decision

`pipeline.run_once(PipelineInput, request, ocr_fn=, vlm_fn=, detector=, translate_fn=, abstention=)` does routing, quality gating, OCR or detection plus VLM, abstention and answer rendering, and returns a `PipelineResult` with traces and percepts. It never speaks and never touches the network by itself: backends are injected. `Assistant` and `eval/run_eval.py` both call it, and `visionassist replay` reuses it.

## Consequences

Tests and eval can run with fakes and no network. The function has many injected parameters. Anything not in `run_once` (speech, busy handling, camera loop) is not covered by the eval, which the report says.
