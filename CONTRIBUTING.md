# Contributing to VisionAssist

Thanks for helping. This project is most useful when blind and low-vision users, Hindi speakers and accessibility practitioners shape it.

## Good first contributions

- Try the app with a real screen reader (NVDA, VoiceOver, TalkBack) and open an issue describing what breaks.
- Record short Hindi voice-command clips and share how well recognition works on your microphone and accent.
- Add a language: user-facing text lives in `src/visionassist/strings.py`, and Piper and Vosk have models for other languages.
- Benchmark on a Raspberry Pi or other edge device with `python tools/latency_benchmark.py` and share the numbers.
- Improve docs and translations.

## Development setup

```bash
git clone https://github.com/techakash32/Hindi-multimodal-vision-assistant.git
cd Hindi-multimodal-vision-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
make test
make lint
```

Tests run without a camera, models or network. Please keep it that way: use fakes for hardware and live APIs.

## Pull requests

- Keep changes focused and add or update tests.
- Run `make test` and `make lint` first.
- Every user-facing string goes in `strings.py`, in Hindi.
- Do not report synthetic-benchmark numbers as real-world accuracy.

## Safety

VisionAssist is an assistive aid, not a safety device. Changes must not add "path is safe" style claims.
