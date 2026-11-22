# ADR 0008: Synthetic images for the first evaluation

- Status: accepted
- Date: 2026-11-21

## Context

No real photographs with trustworthy labels were available when the harness was built, and ground truth for a made-up label is worthless.

## Decision

Fixtures are rendered with PIL and OpenCV from a fixed seed (`eval/generate_dataset.py`), so the ground truth is exactly what was drawn. A synthetic adversarial pack applies blur, gamma, occlusion and noise to the same images. Cases that cannot be generated honestly (real handwriting, Devanagari photos, audio) are listed in `eval/fixtures/README.md` instead of faked.

## Consequences

The numbers test the pipeline logic, guardrails and OCR on clean rendered text. They say little about real-world accuracy, and every report carries that caveat. A hand-labelled real-photo set is the biggest remaining gap.
