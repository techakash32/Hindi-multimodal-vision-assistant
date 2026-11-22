| Stage | n | p50 (ms) | p95 (ms) |
|---|---|---|---|
| ASR | n/a | n/a | n/a |
| OCR | 10 | 104.5 | 131.1 |
| Detection | n/a | n/a | n/a |
| VLM | n/a | n/a | n/a |
| Orchestration overhead | 20 | 1.0 | 2.3 |
| End-to-end | 20 | 11.1 | 192.5 |

Measured on Intel(R) Xeon(R) Processor @ 2.10GHz (2 cores), GPU: none, Linux-6.18.44-fc-v726-x86_64-with-glibc2.39 with Python 3.13.18, OpenCV 5.0.0; model versions per stage are listed below. Run `python tools/latency_benchmark.py` to reproduce on your hardware.

Stage details (one warm-up call per stage is made and discarded; p95 on n=5 or 10 is near the maximum):

- **ASR:** SKIPPED: needs 5 .wav clips in /home/claude/visionassist/eval/audio/ (found 0); none ship with the repo
- **OCR:** tesseract, 10 synthetic text images
- **Detection:** SKIPPED: YOLO unavailable (ModuleNotFoundError); the demo blob model is not timed because its latency says nothing about YOLO
- **VLM:** SKIPPED: OPENROUTER_API_KEY not set (the VLM is a network call)
- **Orchestration overhead:** end-to-end minus time inside OCR / detector / VLM calls
- **End-to-end:** run_once on every 4th scenario (20, a mix: some abstain before any model call); OCR: tesseract 5.3.4; detector: demo colour-blob model (not YOLO); VLM: deterministic template over detections (not a language model)

Skipped stages are not estimated. To fill them in on your own machine: set `OPENROUTER_API_KEY` (VLM), install `ultralytics` and run `visionassist download-models` (detection), put five Hindi command clips in `eval/audio/` and download the Vosk model (ASR).
