# test_vosk.py — standalone Vosk + microphone diagnostic
#
# Usage:
#   python test_vosk.py            -> uses the default input device
#   python test_vosk.py 1          -> uses input device index 1

import array
import json
import queue
import sys
import time

import sounddevice as sd
import vosk

MODEL_PATH = "models/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000

# ---- 1. List only devices that can record ----
print("Input devices (index: name):")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"  {i}: {d['name']}  "
              f"(default rate {d['default_samplerate']:.0f} Hz)")

try:
    print(f"\nDefault input device: {sd.default.device[0]}")
except Exception:
    pass

device = int(sys.argv[1]) if len(sys.argv) > 1 else None
print(f"Using device: {device if device is not None else 'default'}\n")

# ---- 2. Load model ----
model = vosk.Model(MODEL_PATH)
rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
q = queue.Queue()


def cb(indata, frames, t, status):
    if status:
        print(f"[STATUS] {status}", flush=True)
    q.put(bytes(indata))


def peak_level(raw: bytes) -> int:
    """Peak amplitude of an int16 buffer (0 - 32767)."""
    samples = array.array("h", raw)
    return max(abs(s) for s in samples) if samples else 0


last_partial = ""
last_meter = 0.0
heard_anything = False
start = time.time()

with sd.RawInputStream(
    samplerate=SAMPLE_RATE,
    blocksize=8000,
    dtype="int16",
    channels=1,
    device=device,
    callback=cb,
):
    print("Say 'read', 'describe', 'quit'... (Ctrl+C to stop)\n")

    try:
        while True:
            data = q.get()

            # Level meter once a second: tells you if the mic hears anything
            if time.time() - last_meter > 1.0:
                level = peak_level(data)
                bar = "#" * min(40, level // 400)
                print(f"[MIC LEVEL] {level:5d} {bar}", flush=True)
                last_meter = time.time()

            if rec.AcceptWaveform(data):
                txt = json.loads(rec.Result()).get("text", "")
                last_partial = ""
                if txt:
                    heard_anything = True
                    print(f"HEARD: {txt!r}", flush=True)
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial and partial != last_partial:
                    heard_anything = True
                    print(f"  partial: {partial!r}", flush=True)
                    last_partial = partial

            if not heard_anything and time.time() - start > 15:
                print("[HINT] 15s with no recognised speech. Check the "
                      "MIC LEVEL lines: if they stay near 0, the wrong "
                      "device is selected or the mic is muted.",
                      flush=True)
                start = time.time()

    except KeyboardInterrupt:
        print("\nStopped.")
