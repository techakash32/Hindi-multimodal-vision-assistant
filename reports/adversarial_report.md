# VisionAssist evaluation report

- **Date:** 2026-11-21 17:03 UTC+0000
- **Commit:** n/a (not a git checkout)
- **Host:** Intel(R) Xeon(R) Processor @ 2.10GHz (2 cores), GPU: none, Linux-6.18.44-fc-v726-x86_64-with-glibc2.39
- **Python / OpenCV:** 3.13.18 / 5.0.0
- **Backend:** `auto`
- **DETECTOR:** demo colour-blob model (not YOLO)
- **VLM:** deterministic template over detections (not a language model)
- **ASR:** not exercised (typed / simulated transcripts; no audio, no Vosk)
- **OCR:** tesseract 5.3.4 (local, real; psm 6, lang eng)
- **Judge:** keyword-overlap judge (deterministic)
- **Abstention thresholds:** `{'ocr_tau': 0.5, 'vlm_tau': 0.4, 'aggregate_tau_low': 0.45, 'aggregate_tau_high': 0.7, 'enabled': True}`

> 20 scenarios. Read the Known limitations section before quoting any number: the inputs are synthetic and, depending on the backend above, parts of the stack are stand-ins.

## Results by modality

| modality | n | precision | recall | F1 | notes |
|---|---|---|---|---|---|
| OCR: medicine labels | 2 | 0.50 | 1.00 | 0.67 | acc 1/2; CER 0.09, WER 0.25 over 2 runs that reached OCR; exact match 0.50 |
| OCR: signs | 6 | 1.00 | 1.00 | 1.00 | acc 6/6; CER 0.00, WER 0.00 over 6 runs that reached OCR; exact match 1.00 |
| OCR: menus | 2 | 1.00 | 1.00 | 1.00 | acc 2/2; CER 0.00, WER 0.00 over 2 runs that reached OCR; exact match 1.00 |
| OCR: all | 10 | 0.90 | 1.00 | 0.95 | acc 9/10; CER 0.02, WER 0.05 over 10 runs that reached OCR; exact match 0.90 |
| Scene description (VLM) | 0 | n/a | n/a | n/a | acc 0/0 |
| Voice command (intent) | 0 | n/a | n/a | n/a | acc 0/0 |

Per scenario, *correct* is a true positive and *incorrect* (an answer that is wrong) a false positive. *Abstained* on an answerable request is a false negative, and *partial* (scene descriptions only) is counted in neither. precision = TP/(TP+FP), recall = TP/(TP+FN); accuracy in the notes is TP/n. OCR is correct only on an exact match after lower-casing and removing punctuation (CER = 0).

## Abstention

**Abstention accuracy: 10/10**

| category | n | abstention_accuracy |
|---|---|---|
| ambiguous_question | 0 | n/a (0/0) |
| negative_abstain | 10 | 1.00 (10/10) |

Answerable scenarios that were wrongly refused: 0/10. Abstentions whose reason matched the scenario's expected reason: 0/10.

## Failure taxonomy

| Bucket | Count | Example ID |
|---|---|---|
| detection_miss | 0 | — |
| ocr_detection_fail | 0 | — |
| ocr_recognition_error | 1 | adv_005 |
| vlm_hallucination | 0 | — |
| asr_substitution | 0 | — |
| router_misroute | 0 | — |
| latency_timeout | 0 | — |
| failure_to_abstain | 0 | — |
| incorrect_abstention | 0 | — |
| unclassified | 0 | — |

1 of 20 scenarios were not fully correct; each has exactly one bucket (`unclassified` = 0).

<details><summary>Every failure</summary>

| id | bucket | request | what happened |
|---|---|---|---|
| adv_005 | ocr_recognition_error | Ye dawa ka naam kya hai? | answered: —_ Paracetamol 500mg |

</details>

## Latency

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

## Known limitations

- **Synthetic images only.** Every fixture is rendered with PIL / OpenCV (flat text on a background, coloured blocks for
  "people" and "chairs"). There are no photographs, no real handwriting, no motion blur from a moving camera, and no
  Devanagari text. Numbers here say little about performance on real street, shop or pharmacy photos.
- **No live camera.** Frames come from files; camera loss, auto-exposure and focus behaviour are untested here.
- **ASR is not exercised.** Voice scenarios feed typed or simulated transcripts to the intent router. Real Vosk
  recognition of Hindi (accuracy, noise, accents, the grammar restriction) was not run, so `asr_substitution` here
  only covers two hand-written mis-transcriptions.
- **Piper (text to speech) is not exercised.** The voice text is checked as Devanagari strings, never as audio.
- **Translation is a pass-through.** Hindi translation of OCR text goes through the network (deep-translator), so the
  harness does not call it; OCR text is shown as read.
- **Detector and VLM are stand-ins unless the header says otherwise.** The colour-blob demo detector knows two classes
  and keeps one box per class (so counting scenarios and the dog/bus scenarios fail by construction), and the template
  "VLM" only restates detections. Real YOLO and real OpenRouter VLMs were not run; scene numbers measure the
  pipeline around them, not the models.
- **OCR.space was not run.** If the header shows Tesseract, that is the OCR engine measured; OCR.space (the app's
  default engine) has no confidence output, so the app uses an image-quality and text-plausibility proxy for it.
- **Abstention thresholds were tuned on the 40 even-indexed (dev) scenarios.** The 10/10 above covers all 80 and so
  includes the dev half; the number to quote is the holdout result in `reports/calibration.md`. Holdout has only about
  5 negatives, so it is a weak estimate. Other pipeline constants (image-quality gates in `quality.py`) were chosen by
  looking at these same fixtures.
- **The keyword judge is crude** (keyword overlap, side and count contradiction checks). It is deterministic, not
  accurate; an LLM judge (`--judge llm`) was not run here.
- **Small n.** 8 to 20 scenarios per row; one scenario moves a metric by 5 to 12 points. No confidence intervals.
- **Latency numbers are from this machine only** and, unless the latency table says otherwise, exclude network
  stages (OCR.space, VLM) that dominate in the real app.
