.PHONY: install install-dev models check run demo web test lint fmt clean eval calibrate bench

PY ?= python

install:           ## install runtime dependencies
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

install-dev:       ## install dev tools too
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

models:            ## download YOLO, Piper voices, Vosk model
	visionassist download-models

check:             ## verify models, voices, API keys
	visionassist check

run:               ## OpenCV window
	visionassist run

web:               ## browser dashboard at http://127.0.0.1:8000
	visionassist run --ui web

demo:              ## no camera/models/keys needed
	visionassist run --ui web --demo

test:
	$(PY) -m pytest --cov=visionassist --cov-report=term-missing

adversarial:       ## adversarial pack
	$(PY) -m eval.generate_adversarial && $(PY) -m eval.run_eval --scenarios eval/scenarios_adversarial.jsonl --out reports/adversarial_raw.json --report reports/adversarial_report.md

kfold:             ## k-fold abstention cross-validation
	$(PY) -m eval.kfold_abstention

docker:            ## build and run the demo in a container
	docker build -t visionassist . && docker run --rm -p 8000:8000 visionassist

lint:
	ruff check .

fmt:
	ruff check --fix .

eval:              ## run the 80-scenario benchmark -> reports/eval_report.md
	$(PY) -m eval.run_eval --scenarios eval/scenarios.jsonl --out reports/eval_raw.json

calibrate:         ## re-pick abstention thresholds -> config/abstention.json
	$(PY) -m eval.calibrate_abstention

bench:             ## per-stage latency -> logs/latency_report.csv, reports/latency_summary.md
	$(PY) tools/latency_benchmark.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage build dist src/*.egg-info