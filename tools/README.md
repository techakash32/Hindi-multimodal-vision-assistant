# Hardware check scripts

Manual scripts that need a real camera, microphone or API keys. They are **not** part of the test suite.

| Script | What it does |
| --- | --- |
| `latency_benchmark.py` | Measures detection FPS, speech synthesis and (optionally) API latency; writes `logs/latency_report.csv`. Run `python tools/latency_benchmark.py --vlm 3 --ocr 3`. |
| `mic_check.py` | Finds which microphone really works. Also: `visionassist mics`. |
| `vosk_check.py` | Tests offline speech recognition with the Vosk model. |
| `ocr_check.py` | Press `O` to read text from the camera view with OCR.space. |
| `vlm_check.py` | Takes one picture and asks the scene-description model about it. |
