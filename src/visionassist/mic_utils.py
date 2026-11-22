from __future__ import annotations

# mic_utils.py
# Microphone discovery that survives Windows re-numbering devices.
import logging

import numpy as np
import sounddevice as sd

from . import config as cfg

log = logging.getLogger(__name__)

TARGET_RATE = 16000   # what Vosk wants

# Loopback / virtual / output-ish devices that must never be used as a mic.
BAD_NAME_KEYWORDS = (
    "stereo mix", "pc speaker", "primary sound capture",
    "microsoft sound mapper", "hands-free", "loopback", "what u hear",
)

# WDM-KS devices often refuse to open from sounddevice.
BAD_HOSTAPIS = ("wdm-ks",)

# Prefer APIs that resample for us (MME, DirectSound) over WASAPI.
HOSTAPI_ORDER = ("mme", "directsound", "wasapi")

RATES = (16000, 44100, 48000)
CHANNELS = (1, 2)


def _api_name(dev) -> str:
    try:
        return sd.query_hostapis(dev["hostapi"])["name"].lower()
    except Exception:
        return ""


def describe_devices() -> str:
    """Human-readable list of every input device."""
    lines = []

    for idx, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            lines.append(
                f"  [{idx}] {d['name']} | {_api_name(d)} | "
                f"ch={d['max_input_channels']} "
                f"rate={int(d['default_samplerate'])}"
            )

    return "\n".join(lines)


def candidate_devices():
    """Input device indices, best first, with loopback devices removed."""
    devices = sd.query_devices()
    scored = []

    for idx, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue

        name = d["name"].lower()
        api = _api_name(d)

        if any(k in name for k in BAD_NAME_KEYWORDS):
            continue
        if any(k in api for k in BAD_HOSTAPIS):
            continue

        pref = next(
            (i for i, n in enumerate(cfg.VOICE_MIC_NAMES) if n.lower() in name),
            len(cfg.VOICE_MIC_NAMES),
        )
        api_rank = next(
            (i for i, a in enumerate(HOSTAPI_ORDER) if a in api),
            len(HOSTAPI_ORDER),
        )

        scored.append((pref, api_rank, idx))

    order = [idx for _, _, idx in sorted(scored)]

    # An explicit index from cfg.VOICE_MIC_DEVICE is tried first.
    if (
        cfg.VOICE_MIC_DEVICE is not None
        and 0 <= cfg.VOICE_MIC_DEVICE < len(devices)
        and devices[cfg.VOICE_MIC_DEVICE]["max_input_channels"] > 0
    ):
        order = [cfg.VOICE_MIC_DEVICE] + [i for i in order if i != cfg.VOICE_MIC_DEVICE]

    return order


def first_working_settings(idx):
    """(rate, channels) the device accepts, or None."""
    for rate in RATES:
        for ch in CHANNELS:
            try:
                sd.check_input_settings(
                    device=idx, channels=ch, samplerate=rate, dtype="int16"
                )
                return rate, ch
            except Exception:
                continue

    return None


def to_16k_mono(raw: bytes, channels: int, rate: int) -> bytes:
    """Convert device audio to 16 kHz mono int16 for Vosk."""
    if channels == 1 and rate == TARGET_RATE:
        return raw

    arr = np.frombuffer(raw, dtype=np.int16)

    if channels > 1:
        usable = len(arr) // channels * channels
        arr = arr[:usable].reshape(-1, channels).mean(axis=1)
    else:
        arr = arr.astype(np.float32)

    if rate != TARGET_RATE and arr.size:
        # Crude low-pass (moving average) before decimating, to limit aliasing.
        win = max(1, int(round(rate / TARGET_RATE)))
        if win > 1:
            arr = np.convolve(arr, np.ones(win, dtype=np.float32) / win,
                              mode="same")

        n_out = int(arr.size * TARGET_RATE / rate)
        if n_out == 0:
            return b""

        arr = np.interp(
            np.linspace(0, arr.size - 1, n_out),
            np.arange(arr.size),
            arr,
        )

    return np.clip(arr, -32768, 32767).astype(np.int16).tobytes()


def open_stream(idx, rate, channels, on_audio):
    """Start a stream; on_audio receives 16 kHz mono int16 bytes."""

    def _cb(indata, frames, time_info, status):
        on_audio(to_16k_mono(bytes(indata), channels, rate))

    stream = sd.RawInputStream(
        samplerate=rate,
        blocksize=int(rate * 0.25),
        dtype="int16",
        channels=channels,
        device=idx,
        callback=_cb,
    )
    stream.start()
    return stream


def open_first_working(on_audio):
    """
    Try every candidate microphone until one really opens.
    Returns (stream, info) or (None, None).
    """
    errors = []

    for idx in candidate_devices():
        settings = first_working_settings(idx)

        if settings is None:
            errors.append(f"[{idx}] accepts no tested rate/channel combo")
            continue

        rate, ch = settings

        try:
            stream = open_stream(idx, rate, ch, on_audio)
            name = sd.query_devices(idx)["name"]

            return stream, {
                "device": idx,
                "name": name,
                "channels": ch,
                "rate": rate,
            }

        except Exception as e:
            errors.append(f"[{idx}] open failed (rate={rate}, ch={ch}): {e}")

    log.error("no microphone could be opened:\n  %s", "\n  ".join(errors))
    log.info("input devices:\n%s", describe_devices())

    return None, None
