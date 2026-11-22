# tools/mic_check.py — find which microphone really works.
# Run it, then keep talking for the ~2 seconds each device is tested.

import time

import numpy as np
import sounddevice as sd

from visionassist.mic_utils import (
    candidate_devices,
    describe_devices,
    first_working_settings,
    open_stream,
)

print("=== All input devices ===")
print(describe_devices())

print("\n=== Testing candidates (speak now!) ===")

results = []

for idx in candidate_devices():
    name = sd.query_devices(idx)["name"]
    settings = first_working_settings(idx)

    if settings is None:
        print(f"[{idx}] {name}: no usable rate/channel combo")
        continue

    rate, ch = settings
    peak = {"v": 0}

    def on_audio(raw, peak=peak):
        arr = np.frombuffer(raw, dtype=np.int16)
        if arr.size:
            peak["v"] = max(peak["v"], int(np.abs(arr.astype(np.int32)).max()))

    try:
        stream = open_stream(idx, rate, ch, on_audio)
        time.sleep(2.0)
        stream.stop()
        stream.close()
    except Exception as e:
        print(f"[{idx}] {name}: open failed — {e}")
        continue

    print(f"[{idx}] {name}: rate={rate} ch={ch} peak level={peak['v']}")
    results.append((peak["v"], idx, name, rate, ch))

print()

if results:
    results.sort(reverse=True)
    level, idx, name, rate, ch = results[0]
    print(f"Loudest working device: [{idx}] {name} (peak {level})")
    print("Levels near 0 = silent / wrong device; hundreds-thousands = good.")
    print(f"To force it:  $env:VOICE_MIC_DEVICE = {idx}")
else:
    print("No device opened. Check Windows Settings -> Privacy -> "
          "Microphone -> 'Let desktop apps access your microphone'.")
