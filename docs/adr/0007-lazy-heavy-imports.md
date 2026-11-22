# ADR 0007: Heavy dependencies are imported lazily

- Status: accepted
- Date: 2026-11-21

## Context

Ultralytics (PyTorch), Piper, Vosk and sounddevice are large and need hardware. CI and reviewers should be able to run the tests and the demo without them.

## Decision

These imports happen at the point of use. A synthetic camera, colour-blob detector and canned OCR/VLM (`--demo`) replace them. `requirements-ci.txt` has the light set.

## Consequences

Missing-dependency errors show up at first use, so `visionassist check` verifies them up front. The demo detector is not YOLO, and reports say so in their header.
