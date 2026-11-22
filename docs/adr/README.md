# Architecture decision records

Short records of decisions that are not obvious from the code. Each lists its trade-offs.

- [0001: Hinglish on screen, Devanagari for the voice](0001-hinglish-on-screen-devanagari-audio.md)
- [0002: Strict black and white dashboard](0002-monochrome-ui.md)
- [0003: One `run_once` entry point for the app and the evaluation](0003-one-pipeline-entry-point.md)
- [0004: Abstain before calling the VLM](0004-abstain-before-the-vlm.md)
- [0005: Tesseract as a local OCR path, and a LOCAL_ONLY mode](0005-local-ocr-fallback-and-local-only.md)
- [0006: Circuit breakers and hedged VLM requests](0006-resilience-breakers-and-hedging.md)
- [0007: Heavy dependencies are imported lazily](0007-lazy-heavy-imports.md)
- [0008: Synthetic images for the first evaluation](0008-synthetic-evaluation-fixtures.md)
