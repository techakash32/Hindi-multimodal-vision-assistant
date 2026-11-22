"""Speech output.

Offline Piper voices (English + Hindi) play through a single worker thread
so utterances never overlap. An optional emotional engine (``snfy``) can
voice scene descriptions. When muted (demo / web-only), text is still
published to listeners so the UI can display what *would* be spoken.
"""
from __future__ import annotations

import io
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable, Optional

from . import config as cfg

log = logging.getLogger(__name__)

# ---- Module state ----
_voices = {"en": None, "hi": None}
_ready = {"en": threading.Event(), "hi": threading.Event()}
_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_queue: "queue.Queue" = queue.Queue(maxsize=5)
_init_started = False
_init_lock = threading.Lock()

_speaking = threading.Event()      # True while audio plays
_playback_lock = threading.Lock()  # Piper and Sonify never overlap
_muted = not cfg.TTS_ENABLED
_listener: Optional[Callable[[str, str, Optional[str]], None]] = None
_last_spoken: Optional[tuple] = None   # (text, lang, fallback, display)

_sonify = None
_sonify_tried = False

_TAG_RE = re.compile(r"\[(?:happy|sad|angry|warm|neutral)\]\s*", re.I)


def _has_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def _clean(text: str) -> str:
    """Remove emotion tags; Piper would read them aloud."""
    return _TAG_RE.sub("", text).strip()


def is_speaking() -> bool:
    return _speaking.is_set()


def reset_speaking() -> None:
    """Force-clear the speaking flag (used by the voice watchdog)."""
    _speaking.clear()


def set_muted(value: bool) -> None:
    global _muted
    _muted = bool(value)


def is_muted() -> bool:
    return _muted


def set_speech_listener(fn: Optional[Callable[[str, str, Optional[str]], None]]) -> None:
    """Call ``fn(text, lang, display)`` for every utterance (spoken or muted).
    ``display`` is the on-screen (Hinglish) version, if the caller gave one."""
    global _listener
    _listener = fn


def _notify(text: str, lang: str, display: Optional[str] = None) -> None:
    if _listener is not None:
        try:
            _listener(text, lang, display)
        except Exception:  # a UI bug must never break speech
            log.exception("speech listener failed")


def status() -> dict:
    """Which voices are loaded (for the health panel)."""
    return {
        "english_piper": _voices["en"] is not None,
        "hindi_piper": _voices["hi"] is not None,
        "emotional": _sonify is not None,
        "muted": _muted,
    }


def _get_sonify():
    """Optional emotional TTS, loaded on first use."""
    global _sonify, _sonify_tried
    if not _sonify_tried:
        _sonify_tried = True
        try:
            import snfy  # type: ignore

            _sonify = snfy.SonifyEngine()
        except Exception as exc:
            log.info("emotional TTS unavailable: %s", exc)
    return _sonify


# ==================================================
# VOICE LOADING (both voices, once, in the background)
# ==================================================

def _load_voices() -> None:
    try:
        from piper import PiperVoice

        if cfg.HINDI_ONLY:
            _ready["en"].set()          # English voice is never needed
        else:
            try:
                log.info("loading English Piper voice")
                _voices["en"] = PiperVoice.load(str(cfg.VOICE_MODEL))
            except Exception as exc:
                log.error("English voice failed: %s", exc)
            finally:
                _ready["en"].set()

        try:
            if cfg.USE_HINDI and Path(cfg.HINDI_VOICE_MODEL).exists():
                log.info("loading Hindi Piper voice")
                _voices["hi"] = PiperVoice.load(str(cfg.HINDI_VOICE_MODEL))
            elif cfg.HINDI_ONLY:
                log.error("Hindi voice not found at %s; run: visionassist download-models --only hi",
                          cfg.HINDI_VOICE_MODEL)
            elif cfg.USE_HINDI:
                log.warning("Hindi voice not found at %s; alerts use English",
                            cfg.HINDI_VOICE_MODEL)
        except Exception as exc:
            log.error("Hindi voice failed: %s", exc)
    except ImportError as exc:
        log.error("piper-tts is not installed: %s", exc)
    finally:
        _ready["en"].set()
        _ready["hi"].set()
        log.info("tts status: %s", status())


def init_tts() -> None:
    """Start loading voices now (non-blocking). No-op when muted."""
    global _init_started
    if _muted:
        return
    with _init_lock:
        if not _init_started:
            _init_started = True
            threading.Thread(target=_load_voices, daemon=True).start()


def hindi_voice_available() -> bool:
    return cfg.USE_HINDI and Path(cfg.HINDI_VOICE_MODEL).exists()


# ==================================================
# PLAYBACK
# ==================================================

def _linux_player(path: str) -> list:
    for exe, args in (("aplay", ["-q"]), ("paplay", []), ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"])):
        if shutil.which(exe):
            return [exe, *args, path]
    raise RuntimeError("no audio player found (install alsa-utils, pulseaudio-utils or ffmpeg)")


def _play_wav_bytes(wav_bytes: bytes) -> None:
    """Play WAV audio; blocks until playback finishes."""
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = f.name
    try:
        cmd = ["afplay", path] if sys.platform == "darwin" else _linux_player(path)
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _pick_voice(text: str, lang: str, fallback: Optional[str]):
    """Return (voice, text) to synthesize, falling back to English."""
    if lang == "hi":
        _ready["hi"].wait(timeout=30)
        if _voices["hi"] is not None:
            return _voices["hi"], _clean(text)
        if cfg.HINDI_ONLY:
            return None, None       # never fall back to English
        # English Piper cannot read Devanagari.
        text, lang = fallback, "en"
        if not text:
            return None, None

    _ready["en"].wait(timeout=30)
    return _voices["en"], _clean(text)


def _worker() -> None:
    """Synthesize queued text and play it, one utterance at a time."""
    while True:
        item = _queue.get()
        try:
            if item is None:
                return
            text, lang, fallback = item
            voice, text = _pick_voice(text, lang, fallback)
            if voice is None or not text:
                log.error("no voice available for %r", text)
                continue

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                voice.synthesize_wav(text, wav)

            with _playback_lock:
                _speaking.set()
                try:
                    _play_wav_bytes(buf.getvalue())
                finally:
                    _speaking.clear()
        except Exception as exc:
            log.error("TTS failure: %s", exc)
            _speaking.clear()
        finally:
            _queue.task_done()


def speak(text: str, lang: str = "en", fallback_text: Optional[str] = None,
          record: bool = True, display: Optional[str] = None) -> None:
    """Queue speech. Never blocks the camera loop.

    ``lang="hi"`` uses the Hindi voice; if it is missing, or the text has no
    Devanagari, ``fallback_text`` (English) is spoken instead. If the queue
    is full the *oldest* item is dropped: a fresh safety alert matters more
    than a stale one.
    """
    global _worker_thread, _last_spoken

    if not text or not text.strip():
        return
    text = text.strip()

    if cfg.HINDI_ONLY:
        # Everything the user hears is Hindi. Callers translate first; text
        # that is still not Hindi is dropped rather than spoken in English.
        fallback_text = None
        if not _has_devanagari(_clean(text)):
            log.warning("not speaking non-Hindi text: %r", text[:60])
            return
        lang = "hi"
    elif lang == "hi" and not _has_devanagari(text):
        lang = "en"
    elif lang == "en" and _has_devanagari(text) and fallback_text:
        text = fallback_text

    if record:
        _last_spoken = (text, lang, fallback_text, display)
    _notify(_clean(text), lang, display)

    if _muted:
        return

    init_tts()
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_worker, daemon=True)
            _worker_thread.start()

    try:
        _queue.put_nowait((text, lang, fallback_text))
    except queue.Full:
        try:
            _queue.get_nowait()
            _queue.task_done()
            _queue.put_nowait((text, lang, fallback_text))
            log.warning("speech queue full; dropped the oldest utterance")
        except (queue.Empty, queue.Full):
            log.warning("speech queue full; utterance skipped")


def repeat_last() -> bool:
    """Say the last utterance again. Returns False if there is none."""
    if _last_spoken is None:
        return False
    text, lang, fallback, display = _last_spoken
    speak(text, lang=lang, fallback_text=fallback, record=False, display=display)
    return True


def speak_hindi_emotional(text: str, emotion: str = "WARM",
                          fallback_text: Optional[str] = None, display: Optional[str] = None) -> None:
    """Blocking. For VLM/OCR output only, never safety alerts.

    Fallback chain: emotional engine -> offline Hindi Piper -> English Piper.
    """
    if not text or not text.strip():
        return
    plain = _clean(text)
    engine = None if _muted else _get_sonify()

    if engine is None:
        speak(plain, lang="hi", fallback_text=fallback_text, display=display)
        return

    import snfy  # type: ignore

    output_path = None
    try:
        emo = getattr(snfy.Emotion, emotion.upper(), snfy.Emotion.NEUTRAL)
        kwargs = {"emotion": emo}
        lang_enum = getattr(snfy, "Language", None)
        if lang_enum is not None and hasattr(lang_enum, "HI_IN"):
            kwargs["language"] = lang_enum.HI_IN

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)
        engine.synthesize(text.strip(), output_path=str(output_path), **kwargs)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("emotional TTS produced no audio")

        _notify(plain, "hi", display)
        with _playback_lock:
            _speaking.set()
            try:
                _play_wav_bytes(output_path.read_bytes())
            finally:
                _speaking.clear()
    except Exception as exc:
        log.warning("emotional TTS failed (%s); falling back to Piper", exc)
        speak(plain, lang="hi", fallback_text=fallback_text, display=display)
    finally:
        if output_path is not None:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
