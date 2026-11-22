# VisionAssist

**Offline-first assistive vision for blind and low-vision users.** A webcam watches the scene; the assistant speaks short, prioritised alerts in Hindi ("व्यक्ति, बहुत पास, सामने"), reads printed text aloud, and describes the whole scene on request. Safety alerts never depend on the internet.

![Dashboard](docs/dashboard-light.png)

> The dashboard above is running in `--demo` mode (synthetic camera feed, canned OCR and scene text). With a real camera it shows your video and real results.

## What it does

| Need | How | Works offline |
| --- | --- | --- |
| Know what is nearby | YOLOv8n detects objects every other frame; alerts say *what*, *how close* and *which side* | Yes |
| Not be nagged | Alerts need a stable, confident, large-enough detection, and each object is announced once until it leaves view | Yes |
| Hear and read it in Hindi only | Offline Hindi Piper voice; every spoken and on-screen message is Hindi. Text from signs is translated to Hindi | Yes (translating English signs needs internet) |
| Read a label or sign | OCR.space on the centre of the frame (**Space**) | No |
| Understand the scene | Free OpenRouter vision models with JSON output, hedged requests and circuit breakers (**D**) | No |
| Know what changed | Compares objects now versus the last description (**C**) | Yes |
| Hear it again | **R** repeats the last message | Yes |
| Hands-free | Offline Hindi voice commands: "Text padhein", "Scene batayein", "Kya badla", "Dobara bolein", "Awaaz band karein" | Yes |

## Architecture

```
                 ┌──────────────── Assistant engine (engine.py) ────────────────┐
 Camera ───────► │ capture loop ─► YOLOv8n ─► stability filter ─► alert ─► TTS  │ ─► Piper (offline)
 (or demo feed)  │      │                                                        │
                 │      └─► latest frame ──► worker thread ─► OCR / VLM ─► TTS   │ ─► OpenRouter, OCR.space
                 └───────┬──────────────────────────────┬────────────────────────┘
                         │ snapshot() / jpeg()          │ dispatch(cmd)
              ┌──────────▼───────────┐       ┌──────────▼───────────┐
              │ OpenCV window + HUD  │       │ Flask web dashboard  │   keyboard · voice · buttons
              └──────────────────────┘       └──────────────────────┘
```

Design decisions worth knowing:

- **Fast loop and reasoning tier are separate.** Detection and speech run locally. Network calls run on one worker thread, so a slow API never freezes the camera.
- **Graceful degradation.** Every failure produces a spoken fallback and detection continues. Failing providers are benched by circuit breakers (404 for an hour, 429 and repeated errors for a minute).
- **Latency hedging.** If the preferred vision model has not answered after 5 s, the next one is also asked and the first answer wins.
- **Honest about safety.** The scene prompt forbids "safe to pass" claims; "path clear" is never announced unless you opt in.
- **Newest alert wins.** If speech backs up, the *oldest* queued utterance is dropped, not the newest.
- **One engine, many frontends.** The engine has no UI code, so the window, the dashboard and the tests all drive it the same way.

## Quick start

Full step-by-step setup for Windows, macOS and Linux (including real-hardware benchmarking and Docker): [docs/SETUP_LOCAL.md](docs/SETUP_LOCAL.md).

```bash
# 1. Get the code and create an environment (Python 3.10 to 3.12)
cd visionassist
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Installgit 
pip install -r requirements.txt
pip install -e .                  # adds the `visionassist` command

# 3. Download the models (YOLO ~6 MB, Hindi voice ~60 MB, Hindi Vosk model ~40 MB)
visionassist download-models

# 4. Add your free API keys (only needed for Read text and Describe scene)
cp .env.example .env              # Windows: copy .env.example .env
#    then edit .env: OCR_API_KEY=...  OPENROUTER_API_KEY=...

# 5. Check everything is in place, then run
visionassist check
visionassist run                  # opens the Hindi dashboard in your browser
```

**Try it with no camera, models or keys:**

```bash
visionassist run --ui web --demo      # open http://127.0.0.1:8000
```

**Evaluate and benchmark (no camera, models or keys needed):**

```bash
pip install -r requirements-dev.txt
python -m eval.generate_dataset                   # rebuild the synthetic fixtures (already committed)
python -m eval.run_eval --scenarios eval/scenarios.jsonl --out reports/eval_raw.json
python -m eval.calibrate_abstention               # re-fit config/abstention.json
python tools/latency_benchmark.py                 # writes logs/latency_report.csv, reports/latency_summary.md
```

## All commands

| Command | Purpose |
| --- | --- |
| `visionassist run` | Camera + Hindi browser dashboard at `http://127.0.0.1:8000` (opens automatically) |
| `visionassist run --no-browser` | Same, without opening a browser tab |
| `visionassist run --ui window` | OpenCV window (boxes only; OpenCV cannot draw Hindi text) |
| `visionassist run --ui both` | Window and dashboard together |
| `visionassist run --ui headless` | No UI; voice and speech only (for a wearable or Raspberry Pi) |
| `visionassist run --demo` | Synthetic camera and detector; no hardware, models or keys |
| `visionassist run --camera 1` | Use a different camera |
| `visionassist run --mute` | Show speech on screen without playing audio |
| `visionassist run --no-voice` | Disable microphone commands |
| `visionassist run --port 9000 --host 0.0.0.0` | Dashboard on another port or the network (set `WEB_TOKEN` first) |
| `visionassist check` | Verify models, voices and keys; exit code 1 if the safety loop cannot run |
| `visionassist download-models [--only yolo,hi,vosk-hi,en,vosk] [--force]` | Fetch models (default: yolo, hi, vosk-hi) |
| `visionassist mics` | List microphones and the order they will be tried |
| `python -m visionassist ...` | Same as `visionassist ...` |
| `make test` / `make lint` | Run tests with coverage / lint |
| `python tools/latency_benchmark.py --vlm 3 --ocr 3` | Measure FPS and latency on your hardware |

## Controls

| Key / voice | Action |
| --- | --- |
| **Space** / "टेक्स्ट पढ़ें" (Text padhein) | Read text in front of the camera |
| **D** / "सीन बताएं" (Scene batayein) | Describe the scene (skipped if nothing changed in 30 s) |
| **C** / "क्या बदला" (Kya badla) | What changed since the last description (offline, instant) |
| **R** / "दोबारा बोलें" (Dobara bolein) | Say the last message again |
| **M** | Mute or unmute sound (text still appears on screen) |
| **Q**, **Esc** / "आवाज़ बंद करें" (Awaaz band karein) | Quit the application. By voice the whole phrase is needed; "band" alone does nothing. The dashboard button of that name only mutes. |

Click the OpenCV window once so it receives keystrokes.

## Hindi-only mode

Everything the user hears or reads is Hindi: spoken alerts, status messages, scene descriptions (the vision model is asked to answer in Hindi; English answers are translated), text read from signs, the dashboard labels, the activity feed and voice commands. All user-facing text lives in [`strings.py`](src/visionassist/strings.py). English remains only in logs, CLI help and docs. Details:

- If the Hindi voice is missing the assistant stays silent and reports the problem instead of speaking English; `visionassist check` fails until you run `visionassist download-models --only hi`.
- If a translation fails the user hears a Hindi "translation unavailable" message, never English.
- OpenCV cannot draw Devanagari, so object names are shown by the dashboard; the optional window shows boxes only.
- Voice commands use the small Vosk Hindi model with a restricted word list (with spelling variants). Recognition quality depends on your microphone and accent, so tune `COMMANDS` in [`voice.py`](src/visionassist/voice.py) after trying it.
- Set `HINDI_ONLY=false` to bring back the English voice path (it is not the default).

## Web dashboard

Live annotated video, a left/ahead/right strip showing where each object is (solid chip = very close, outlined = nearby, dashed = far), the last spoken message in large type, command buttons, activity feed, response-time table and a system panel. Hindi labels, strictly black and white with a serif headline style, built with plain HTML/CSS/JS and system fonts, so it works with no internet. Light and dark themes follow your system, controls have keyboard shortcuts and focus outlines, and the speech card is an `aria-live` region.

Security: it binds to `127.0.0.1` by default, sends a strict Content-Security-Policy, accepts commands only as JSON, and refuses remote `quit`. If you expose it on a network, set `WEB_TOKEN` in `.env`; it is accepted once as `?token=...` and then kept in an HttpOnly cookie. The camera feed is sensitive, so do not expose it without a token.

## Configuration

Everything is configurable through `.env` or environment variables; see [`.env.example`](.env.example) and [`src/visionassist/config.py`](src/visionassist/config.py). Common ones: `CAMERA_INDEX`, `USE_HINDI`, `YOLO_IMGSZ` (320 fast, 640 accurate), `VLM_MODELS` (free models come and go; override the list without code changes, and expect occasional 429 rate limits or timeouts on free tiers), `VOSK_MODEL` (an Indian-English model recognises Indian accents much better), `VOICE_MIC_NAMES`.

Files are looked up under `VISIONASSIST_HOME` (default: the folder you run from): `models/`, `voices/`, `logs/`.

## Evaluation

`eval/scenarios.jsonl` holds 80 scenarios (OCR medicine 12, OCR sign 12, OCR menu 8, scene description 20, ambiguous question 10, negative/abstain 10, voice command 8). Fixtures in `eval/fixtures/` are **synthetic** (rendered text and coloured blocks), see `eval/fixtures/README.md`. `eval/run_eval.py` runs each scenario through `visionassist.pipeline.run_once`, the same entry point the app uses, then scores OCR (CER, WER, exact match), scenes (judge), voice intent and abstention, and assigns every failure exactly one bucket (`eval/classify_failure.py`; decision order documented at the top of that file).

Latest run (this sandbox: Tesseract OCR, demo colour-blob detector, template stand-in for the VLM, typed transcripts; full report in `reports/eval_report.md`):

| modality | n | precision | recall | F1 | notes |
|---|---|---|---|---|---|
| OCR: medicine labels | 12 | 0.80 | 0.80 | 0.80 | acc 8/12; 2 abstained; CER 0.15, WER 0.24 over 12 runs that reached OCR; exact match 0.67 |
| OCR: signs | 12 | 1.00 | 0.92 | 0.96 | acc 11/12; 1 abstained; CER 0.08, WER 0.08 over 12 runs that reached OCR; exact match 0.92 |
| OCR: menus | 8 | 0.71 | 0.83 | 0.77 | acc 5/8; 1 abstained; CER 0.09, WER 0.17 over 8 runs that reached OCR; exact match 0.62 |
| OCR: all | 32 | 0.86 | 0.86 | 0.86 | acc 24/32; 4 abstained; CER 0.11, WER 0.16 over 32 runs that reached OCR; exact match 0.75 |
| Scene description (VLM) | 20 | 0.81 | 1.00 | 0.90 | acc 13/20; 4 partial |
| Voice command (intent) | 8 | 1.00 | 0.75 | 0.86 | acc 6/8; 2 abstained |

**Abstention accuracy: 10/10** on the 10 `negative_abstain` scenarios (5/5 on the holdout half that was never used for tuning), with 6 of 60 answerable scenarios wrongly refused. Thresholds in `config/abstention.json` are tuned on 40 dev scenarios only; the quotable number is the 40-scenario holdout in `reports/calibration.md` (small n: about 5 negatives). Guardrails (`src/visionassist/guardrails.py`) refuse before any VLM call and answer with a clarification in Devanagari (voice) and Hinglish (screen).

Failure taxonomy of the 17 scenarios not fully correct: detection_miss 5, ocr_recognition_error 4, incorrect_abstention 3, vlm_hallucination 2, asr_substitution 2, ocr_detection_fail 1, unclassified 0. Do not quote these as real-world accuracy: see Known limitations.

### Adversarial pack and cross-validation

- **Adversarial pack** (`eval/scenarios_adversarial.jsonl`, 20 cases, synthetic): motion blur, low light, noise, JPEG blocking, rotation and occlusion applied to clean fixtures. Ten are mild (text still readable, so the right behaviour is to answer) and ten are severe (right behaviour is to abstain). With guardrails: 19/20 as intended, all 10 severe cases abstained, and the one miss is a stray character from a corner patch on a medicine label. With guardrails switched off the same set gives 7 failures (6 `failure_to_abstain`), see `reports/adversarial_noguardrails.md`, so the guardrails are doing real work. Caveats: three of the ten severe images (adv_013, adv_016, adv_018) were still read correctly by Tesseract, so those abstentions were conservative rather than necessary; the severity labels are my own and the pack is easy for large clean text.
- **k-fold abstention** (`python -m eval.kfold_abstention`, `reports/kfold_abstention.md`): 5 stratified folds, thresholds fitted on four and scored on the fifth. Pooled out-of-fold abstention on negatives is 10/10; selective accuracy is 0.77 ± 0.13 against 0.72 ± 0.09 without guardrails, and coverage 0.93 ± 0.08. Folds hold 16 scenarios, so the spread is wide.

## Session replay

```bash
visionassist run --record                      # writes logs/sessions/<time>/session.jsonl and frames/
visionassist replay logs/sessions/<time>       # re-runs every recorded request through pipeline.run_once and lists what changed
visionassist replay logs/sessions/<time> --demo
visionassist replay logs/sessions/<time> --export-scenarios my_cases.jsonl   # unlabelled eval scenarios to hand-label
```
Frames are stored as JPEG, so a replay can differ slightly from the live run in OCR text without any code change.

## Performance

| Stage | n | p50 (ms) | p95 (ms) |
|---|---|---|---|
| ASR | n/a | n/a | n/a |
| OCR | 10 | 112.9 | 137.9 |
| Detection | n/a | n/a | n/a |
| VLM | n/a | n/a | n/a |
| Orchestration overhead | 20 | 1.0 | 1.9 |
| End-to-end | 20 | 9.2 | 121.8 |

Measured on this sandbox (2-core Xeon, no GPU, Python 3.13). ASR, detection (YOLO) and VLM were **not run here**: they show `n/a`, not estimates. Stage notes are in `reports/latency_summary.md`. Run `python tools/latency_benchmark.py` to reproduce on your hardware.

TODO (run on your machine): put five Hindi `.wav` clips in `eval/audio/`, install `ultralytics` and run `visionassist download-models`, set `OPENROUTER_API_KEY`, then rerun the benchmark and commit `reports/latency_summary.md` and refresh this table.

## Privacy

Set `LOCAL_ONLY=true` and no frame or text leaves the device: OCR runs on local Tesseract, cloud scene description and translation are switched off, and the dashboard shows "frame is device se bahar nahi jaate". In the default mode, OCR.space and OpenRouter receive camera frames when you ask to read text or describe a scene; the dashboard says so.

## Graceful degradation

| What fails | What the user gets | Source |
|---|---|---|
| Camera lost mid-session | Spoken "camera kaam nahi kar raha", automatic reconnect attempts, spoken confirmation when frames return | `engine._loop` |
| OCR call fails or times out | Spoken OCR error; after 3 failures OCR is paused for 60 s (circuit breaker) instead of hanging every request | `ocr.py`, `resilience.py` |
| Scene description fails | Spoken scene error; several models are raced and each has its own breaker, a slow one is skipped next time | `vlm.py` |
| Translation fails | Original text is shown; spoken output falls back to the Devanagari tables | `translate.py` |
| Low confidence input | The VLM is not called; a clarification question is spoken and shown (abstention) | `guardrails.py` |
| Speech queue backs up | Oldest alert is dropped, newest kept | `tts.py` |
| Hindi voice missing | `visionassist check` fails; in Hindi-only mode nothing is spoken in English, text stays on screen | `tts.py`, `health.py` |
| Vosk model or microphone missing | Voice commands off; keys and dashboard buttons keep working | `voice.py` |
| Dashboard connection lost | Offline banner (screen-reader alert) and automatic retry | `web/static/app.js` |

These are covered by unit tests with fakes; none was tested against a real unplugged camera or a dead network.

## Accessibility of the dashboard

Skip link, `aria-live` region for the last spoken sentence, `role="alert"` for connection loss, `<noscript>` fallback, visible focus, keyboard shortcuts, pure black-on-white (21:1 contrast, checked by a test that rejects any other colour). **Not yet tested with a real screen reader (NVDA / VoiceOver)**; for a blind user the `--ui headless` voice mode is the primary interface.

See also [docs/adr/](docs/adr/README.md) for the architecture decision records, and `tests/golden/hinglish.tsv` for the 200+ hand-written Devanagari-to-Hinglish pairs that pin the screen-text transliterator.

## Project layout

```
src/visionassist/
  engine.py        camera loop, commands, OCR/VLM workers (no UI code)
  detector.py      YOLO + temporal smoothing and announce rules
  scene.py         "same scene?" logic (pure functions)
  vlm.py ocr.py    network tiers with circuit breakers and hedging
  tts.py voice.py  Piper speech out, Vosk commands in
  ui_cv.py draw.py OpenCV window and HUD
  web/             Flask dashboard (templates, static)
  strings.py       every user-facing string (Devanagari voice, Hinglish screen)
  pipeline.py      run_once(): the single entry point used by the app and by eval
  guardrails.py    should_abstain(): refuse before the VLM when unsure
  hinglish.py      Devanagari to Roman screen text
  recorder.py      session recording and replay
  cli.py           `visionassist` command
tests/             unit tests, all run without a camera, models or network
eval/              80-scenario benchmark, runner, failure classifier, calibration
config/            abstention.json (calibrated thresholds)
reports/           eval and latency reports
tools/             manual hardware checks and latency benchmark
```

## Testing

```bash
pip install -r requirements-dev.txt     # or requirements-ci.txt for a light set
pytest --cov=visionassist
```

Tests cover detection stability rules, scene-change logic, VLM parsing / racing / circuit breakers, OCR failure handling, translation fallback, the engine (busy handling, camera loss and recovery, error reporting), the web API (validation, auth, headers) and the downloader (atomic writes, zip-slip protection). The ML models, audio hardware and live APIs are replaced with fakes, so CI needs no torch or GPU. GitHub Actions runs lint and tests on Python 3.10 to 3.12.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Cannot open camera` | Try `--camera 1`; close Teams, Zoom or other apps using the camera |
| Window ignores keys | Click the window first |
| `[FAIL] YOLO model` / voice missing | `visionassist download-models` |
| Voice commands never trigger | `visionassist mics`; set `VOICE_MIC_NAMES` or `VOICE_MIC_DEVICE`; try an Indian-English Vosk model |
| No sound on Linux | Install `alsa-utils` (`aplay`), `pulseaudio-utils` or `ffmpeg` |
| Piper fails to load a voice | Re-download with `--force`; check `piper-tts` version |
| "Scene description unavailable" | Add `OPENROUTER_API_KEY`; free models may be rate limited, wait a minute |
| No sound at all | Hindi voice missing: `visionassist download-models --only hi`, then `visionassist check` |
| Voice commands not understood | Speak close to the mic; edit the word list in `voice.py`; run `visionassist mics` |

## Known limitations

- This is an **assistive aid, not a safety device**. It can miss or misjudge objects, and free vision models can be wrong. Do not rely on it as a replacement for a cane, guide dog or sighted help.
- Detection uses COCO classes; steps, kerbs, holes and glass doors are not detected.
- "Near / nearby / far" is estimated from box size, not measured depth.
- **Evaluation caveats:** synthetic images only; no live camera, real Vosk ASR, real Piper audio, OCR.space, YOLO or LLM VLM was exercised in the reported numbers; thresholds are tuned on the scoring set; n is small (8 to 20 per row). Full list in `reports/eval_report.md`.
- Hindi voice-command accuracy has not been measured on real speech here. Scene and text results come from free online services with rate limits.

## License

MIT. See [LICENSE](LICENSE).