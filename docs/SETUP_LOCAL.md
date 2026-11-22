# Set up VisionAssist on your own machine

Needs Python 3.10 to 3.12, a webcam and a microphone (the demo needs neither), and about 1 GB of disk for models.

## 1. Install

**Windows (PowerShell)**
```powershell
cd Hindi-multimodal-vision-assistant
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**macOS / Linux**
```bash
cd Hindi-multimodal-vision-assistant
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
# Linux audio and OCR helpers (Debian/Ubuntu):
sudo apt install tesseract-ocr libportaudio2 alsa-utils
```
macOS: `brew install tesseract portaudio`.

## 2. Models and keys

```bash
visionassist download-models          # YOLO, Hindi Piper voice, Hindi Vosk model
cp .env.example .env                  # Windows: copy .env.example .env
```
Edit `.env`:
```
OCR_API_KEY=...            # free key from ocr.space (skip if you use Tesseract)
OPENROUTER_API_KEY=...     # free key from openrouter.ai (scene description)
```
Fully offline instead? Set `LOCAL_ONLY=true`. OCR then uses Tesseract, scene description is disabled, and no frame leaves the device. The dashboard shows a privacy badge.

## 3. Check and run

```bash
visionassist check                    # every line should say OK; fix FAIL lines first
visionassist run                      # camera + dashboard, opens http://127.0.0.1:8000
visionassist run --ui headless        # voice only, no screen (the mode a blind user actually uses)
visionassist run --camera 1           # another camera
visionassist mics                     # list microphones if voice commands do nothing
visionassist run --demo               # no camera, models or keys
```
Keys: Space reads text, D describes the scene, C reports changes, R repeats, M mutes, Q quits (same as the Hindi commands in the README).

## 4. Run the tests, evaluation and benchmark

```bash
pip install -r requirements-dev.txt
pytest                                # 170+ tests, no camera or network needed
ruff check .
python -m eval.run_eval --scenarios eval/scenarios.jsonl --out reports/eval_raw.json
python -m eval.calibrate_abstention   # tunes on 40 dev scenarios, reports 40 holdout
python tools/check_urls.py            # are the download URLs alive? add --record to pin SHA256
python tools/latency_benchmark.py     # writes logs/latency_report.csv and reports/latency_summary.md
```

### Getting real numbers (replaces the n/a rows)
1. Put five short Hindi command clips (16 kHz mono `.wav`) in `eval/audio/` and run `visionassist download-models --only vosk-hi`.
2. `pip install ultralytics` (already in `requirements.txt`) and `visionassist download-models --only yolo`.
3. Set `OPENROUTER_API_KEY` in `.env`.
4. `python tools/latency_benchmark.py`, then paste `reports/latency_summary.md` into the README Performance section and add your CPU and webcam model to the caption.

### Record and replay a real session
```bash
visionassist run --record                  # use it as normal, press Q when done
visionassist replay logs/sessions/<newest folder>
```

### Other evaluation commands
```bash
python -m eval.generate_adversarial        # rebuild the 20 adversarial cases
python -m eval.run_eval --scenarios eval/scenarios_adversarial.jsonl --out reports/adversarial_raw.json --report reports/adversarial_report.md
python -m eval.kfold_abstention            # 5-fold cross-validation of the thresholds
```

## 5. Docker (demo only)

```bash
docker build -t visionassist .
docker run --rm -p 8000:8000 visionassist     # http://localhost:8000
```
The container has no camera or microphone, so it runs the synthetic demo. This image has not been built by the author yet.

## Troubleshooting
See the table in the README. Most problems are a missing model (`visionassist download-models`), the camera in use by another app (`--camera 1`), or the wrong microphone (`visionassist mics`).
