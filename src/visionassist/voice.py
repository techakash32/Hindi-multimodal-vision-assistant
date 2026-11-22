# voice_input.py

import difflib
import importlib.util
import json
import logging
import queue
import threading
import time
import unicodedata

import numpy as np

from . import config as cfg
from .tts import is_speaking, reset_speaking

log = logging.getLogger(__name__)

VOICE_AVAILABLE = all(
    importlib.util.find_spec(m) is not None for m in ("vosk", "sounddevice")
)


# Hindi command words (Vosk Hindi model) with common spelling variants. English
# words are kept only so a user who installs an English model can still use it.
COMMANDS = {
    # पढ़ें: read text
    "पढ़ो": "read", "पढो": "read", "पढ़ें": "read", "पढ़": "read", "पढ़ना": "read",
    "पढ़िए": "read", "पढ़िये": "read", "पढ़ दो": "read",
    # वर्णन: describe the scene
    "बताओ": "describe", "बताइए": "describe", "वर्णन": "describe", "देखो": "describe",
    "बताइये": "describe", "बताएं": "describe", "देखिए": "describe", "देखें": "describe",
    # क्या बदला: what changed
    "बदला": "changed", "बदलाव": "changed", "बदल": "changed", "बदली": "changed",
    # दोहराओ: repeat
    "दोहराओ": "repeat", "दोहराइए": "repeat", "दोहराइये": "repeat", "दोहराएं": "repeat",
    "दोबारा": "repeat", "फिर": "repeat", "फिरसे": "repeat",
    # बंद: quit
    "बंद": "quit", "बन्द": "quit", "बाहर": "quit", "समाप्त": "quit",
    "खत्म": "quit", "रुको": "quit",
    # English fallbacks (English Vosk model only)
    "read": "read", "describe": "describe", "changed": "changed",
    "repeat": "repeat", "quit": "quit", "exit": "quit",
}



def normalize(word: str) -> str:
    """Fold spelling variants that Vosk and a keyboard disagree on: nukta
    (ढ़ as one character or two), chandrabindu vs anusvara, joiners."""
    w = unicodedata.normalize("NFD", word or "").strip().lower()
    for ch in ("\u093c", "\u200d", "\u200c"):
        w = w.replace(ch, "")
    return w.replace("\u0901", "\u0902")


# Live microphone commands: exactly the five phrases printed on the dashboard
# buttons (Text padhein, Scene batayein, Kya badla, Dobara bolein, Awaaz band
# karein). The short forms only exist because the small Hindi model may not know
# the English loanwords ("text", "scene") and then drops them. "Awaaz band karein"
# CLOSES THE APPLICATION, so it must be said in full: "band" alone does nothing.
PHRASES = {
    "read": ["टेक्स्ट पढ़ें", "टेक्स्ट पढ़ो", "पढ़ें"],
    "describe": ["सीन बताएं", "सीन बताइए", "सीन बताइये", "बताएं"],
    "changed": ["क्या बदला", "बदला"],
    "repeat": ["दोबारा बोलें", "दोबारा बोलो", "दोबारा"],
    "quit": ["आवाज़ बंद करें", "आवाज़ बंद करो"],
}
MATCH_CUTOFF = 0.82          # difflib ratio on the whole utterance
QUIT_CUTOFF = 0.9            # closing the app needs a clearer match


def _norm_phrases():
    return [(normalize(p), cmd) for cmd, ps in PHRASES.items() for p in ps]


_PHRASE_NORM = _norm_phrases()


def match_phrase(text: str):
    """Command for an utterance, tolerant of spelling variants and small slips; None if no phrase is close."""
    t = " ".join(normalize(text).split())
    if not t or t == "[unk]":
        return None
    best, best_r = None, 0.0
    for phrase, cmd in _PHRASE_NORM:
        r = difflib.SequenceMatcher(None, t, phrase).ratio()
        if r > best_r:
            best, best_r = cmd, r
    if best is None or best_r < (QUIT_CUTOFF if best == "quit" else MATCH_CUTOFF):
        return None
    return best


# Restricting Vosk to these words is the biggest accuracy gain: it can no
# longer mishear a command as a random phrase.
# Vosk may know a word with or without the nukta, so offer both spellings.
_G = []
for _p in (p for ps in PHRASES.values() for p in ps):
    for _v in (_p, _p.replace("\u093c", "")):
        if _v not in _G:
            _G.append(_v)
GRAMMAR = json.dumps(_G + ["[unk]"], ensure_ascii=False)

# If is_speaking() stays True longer than this, assume the flag is stuck.
STUCK_TIMEOUT_S = 30.0

# Ignore the same command repeated within this window.
DEBOUNCE_S = 1.5

# Warn if the mic stays this quiet for the first SILENCE_CHECK_S seconds.
SILENCE_CHECK_S = 10.0
SILENCE_PEAK = 50

DEBUG_PARTIAL = False   # True = also print partial results


def start_listening(on_command):
    """Spawn a daemon thread that calls on_command for voice commands."""

    if not VOICE_AVAILABLE:
        log.warning("vosk/sounddevice missing; voice commands disabled")
        return False

    import vosk

    from .mic_utils import open_first_working

    try:
        model = vosk.Model(cfg.VOSK_MODEL_PATH)
    except Exception as e:
        log.warning("Vosk model not found (%s); voice commands disabled", e)
        return False

    try:
        rec = vosk.KaldiRecognizer(model, 16000, GRAMMAR)
        log.info("command grammar enabled")
    except Exception as e:
        log.info(f"grammar unsupported ({e}); using open vocabulary")
        rec = vosk.KaldiRecognizer(model, 16000)

    rec.SetWords(True)   # per-word confidence

    audio_q = queue.Queue(maxsize=20)
    level = {"peak": 0}

    # ---- Stuck-speaking watchdog ----
    gate_lock = threading.Lock()
    gate = {"since": None}

    def _speech_active() -> bool:
        """True while TTS is really playing; resets a stuck flag."""
        with gate_lock:
            if not is_speaking():
                gate["since"] = None
                return False

            now = time.monotonic()

            if gate["since"] is None:
                gate["since"] = now
                return True

            if now - gate["since"] > STUCK_TIMEOUT_S:
                log.info(f"is_speaking() stuck > {STUCK_TIMEOUT_S:.0f}s "
                    "— resetting")
                reset_speaking()
                gate["since"] = None
                return False

            return True

    def on_audio(raw: bytes):
        """Called with 16 kHz mono int16 audio from the mic stream."""
        if raw:
            arr = np.frombuffer(raw, dtype=np.int16)
            if arr.size:
                level["peak"] = max(
                    level["peak"], int(np.abs(arr.astype(np.int32)).max())
                )

        # Do not enqueue microphone audio while TTS is playing.
        if _speech_active():
            return

        try:
            audio_q.put_nowait(raw)
        except queue.Full:
            pass

    def _drain():
        try:
            while True:
                audio_q.get_nowait()
        except queue.Empty:
            pass

    def _extract_command(result):
        """Return the command for the recognised phrase, or None. Word
        confidence (when Vosk reports it) must average at least MIN_CMD_CONF."""
        text = result.get("text", "")
        cmd = match_phrase(text)
        if cmd is None:
            return None
        confs = [w.get("conf", 1.0) for w in result.get("result") or []]
        if confs and sum(confs) / len(confs) < cfg.MIN_CMD_CONF:
            log.info("rejected %r (conf %.2f)", text, sum(confs) / len(confs))
            return None
        return cmd

    def _loop():
        last_partial = ""
        last_cmd, last_cmd_t = None, 0.0
        was_speaking = False
        warned_silent = False

        # Windows can renumber devices; find a working real mic by name.
        stream, info = open_first_working(on_audio)

        if stream is None:
            log.info("voice disabled")
            return

        log.info(f"using [{info['device']}] {info['name']} "
            f"(channels={info['channels']}, rate={info['rate']})")

        started = time.monotonic()

        try:
            log.info("listening — say 'read', 'describe', 'quit'")

            while True:
                try:
                    data = audio_q.get(timeout=1.0)
                except queue.Empty:
                    data = None

                if (
                    not warned_silent
                    and time.monotonic() - started > SILENCE_CHECK_S
                    and level["peak"] < SILENCE_PEAK
                ):
                    warned_silent = True
                    log.warning("microphone is silent. Wrong device or muted? Run: visionassist mics, "
                                "pick another device (VOICE_MIC_NAMES in .env), and on Windows check "
                                "Settings > Privacy > Microphone is allowed for desktop apps.")

                if data is None:
                    continue

                if _speech_active():
                    was_speaking = True
                    continue

                # TTS just finished: drop its tail so the recognizer
                # doesn't hear the assistant's own voice.
                if was_speaking:
                    was_speaking = False
                    rec.Reset()
                    _drain()
                    continue

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    txt = result.get("text", "").strip()
                    last_partial = ""

                    if not txt:
                        continue

                    # Pure noise / unknown speech: nothing to report.
                    if txt == "[unk]":
                        continue

                    log.info(f"heard raw: {txt!r}")

                    cmd = _extract_command(result)

                    if cmd is None:
                        continue

                    now = time.monotonic()

                    if cmd == last_cmd and now - last_cmd_t < DEBOUNCE_S:
                        continue

                    last_cmd, last_cmd_t = cmd, now
                    log.info("command: %s (%s)", cmd, txt)
                    on_command(cmd)

                elif DEBUG_PARTIAL:
                    partial = (
                        json.loads(rec.PartialResult())
                        .get("partial", "")
                        .strip()
                    )

                    if partial and partial != last_partial:
                        log.info(f"partial: {partial!r}")
                        last_partial = partial

        except Exception as e:
            log.info(f"error: {e} — voice disabled")

        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    threading.Thread(target=_loop, daemon=True, name="voice").start()
    return True
