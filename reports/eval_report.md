# VisionAssist evaluation report

- **Date:** 2026-11-22 12:53 UTC+0530
- **Commit:** n/a (not a git checkout)
- **Host:** Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (12 cores), GPU: NVIDIA GeForce RTX 2050, Windows-10-10.0.26300-SP0
- **Python / OpenCV:** 3.11.9 / 5.1.0
- **Backend:** `auto`
- **DETECTOR:** demo colour-blob model (not YOLO)
- **VLM:** deterministic template over detections (not a language model)
- **ASR:** not exercised (typed / simulated transcripts; no audio, no Vosk)
- **OCR:** scripted fake (tesseract not installed)
- **Judge:** keyword-overlap judge (deterministic)
- **Abstention thresholds:** `{'ocr_tau': 0.5, 'vlm_tau': 0.4, 'aggregate_tau_low': 0.45, 'aggregate_tau_high': 0.7, 'enabled': True}`

> 80 scenarios. Read the Known limitations section before quoting any number: the inputs are synthetic and, depending on the backend above, parts of the stack are stand-ins.

## Results by modality

| modality | n | precision | recall | F1 | notes |
|---|---|---|---|---|---|
| OCR: medicine labels | 12 | 1.00 | 1.00 | 1.00 | acc 12/12; CER 0.00, WER 0.00 over 12 runs that reached OCR; exact match 1.00 |
| OCR: signs | 12 | 1.00 | 1.00 | 1.00 | acc 12/12; CER 0.00, WER 0.00 over 12 runs that reached OCR; exact match 1.00 |
| OCR: menus | 8 | 1.00 | 1.00 | 1.00 | acc 8/8; CER 0.00, WER 0.00 over 8 runs that reached OCR; exact match 1.00 |
| OCR: all | 32 | 1.00 | 1.00 | 1.00 | acc 32/32; CER 0.00, WER 0.00 over 32 runs that reached OCR; exact match 1.00 |
| Scene description (VLM) | 20 | 0.81 | 1.00 | 0.90 | acc 13/20; 4 partial |
| Voice command (intent) | 8 | 1.00 | 0.75 | 0.86 | acc 6/8; 2 abstained |

Per scenario, *correct* is a true positive and *incorrect* (an answer that is wrong) a false positive. *Abstained* on an answerable request is a false negative, and *partial* (scene descriptions only) is counted in neither. precision = TP/(TP+FP), recall = TP/(TP+FN); accuracy in the notes is TP/n. OCR is correct only on an exact match after lower-casing and removing punctuation (CER = 0).

## Abstention

**Abstention accuracy: 10/10**

| category | n | abstention_accuracy |
|---|---|---|
| ambiguous_question | 10 | 1.00 (10/10) |
| negative_abstain | 10 | 1.00 (10/10) |

Answerable scenarios that were wrongly refused: 2/60 (voice_command_007, voice_command_008). Abstentions whose reason matched the scenario's expected reason: 20/20.

## Failure taxonomy

| Bucket | Count | Example ID |
|---|---|---|
| detection_miss | 5 | scene_description_011 |
| ocr_detection_fail | 0 | — |
| ocr_recognition_error | 0 | — |
| vlm_hallucination | 2 | scene_description_007 |
| asr_substitution | 2 | voice_command_007 |
| router_misroute | 0 | — |
| latency_timeout | 0 | — |
| failure_to_abstain | 0 | — |
| incorrect_abstention | 0 | — |
| unclassified | 0 | — |

9 of 80 scenarios were not fully correct; each has exactly one bucket (`unclassified` = 0).

<details><summary>Every failure</summary>

| id | bucket | request | what happened |
|---|---|---|---|
| scene_description_007 | vlm_hallucination | Aage koi rukaawat hai? | answered: Saamne rukaawat hai / Saamne ek person / Saamne ek |
| scene_description_011 | detection_miss | Kitne log hain? | answered: Saamne rukaawat hai / Saamne ek person / Saamne ek |
| scene_description_012 | detection_miss | Kitni chair hain? | answered: Saamne rukaawat hai / Saamne ek chair / Saamne ek  |
| scene_description_016 | vlm_hallucination | Kaun zyada paas hai, person ya chair? | answered: Saamne rukaawat hai / Saamne ek person / Saamne ek |
| scene_description_018 | detection_miss | Saamne kutta hai kya? | answered: Raasta saaf nahi dikh raha / Koi khaas cheez nahi  |
| scene_description_019 | detection_miss | Saamne bus hai kya? | answered: Raasta saaf nahi dikh raha / Koi khaas cheez nahi  |
| scene_description_020 | detection_miss | Mere aaspaas kya hai? | answered: Saamne rukaawat hai / Saamne ek person / Saamne ek |
| voice_command_007 | asr_substitution | padho | abstained: Jaankari mil nahi rahi, thoda aur batao. |
| voice_command_008 | asr_substitution | dohrao | abstained: Jaankari mil nahi rahi, thoda aur batao. |

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
