"""The assistant engine: camera loop, detection, voice/keyboard commands and
the OCR / scene-description workers.

The engine knows nothing about windows or web pages. Frontends (OpenCV
window, web dashboard, headless) read ``snapshot()`` / ``jpeg()`` and send
commands through ``dispatch()``.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

import cv2

from . import config as cfg
from . import tts
from .draw import draw_detections, proximity, zone
from .events import EventBus
from .hindi_text import CLASS_HI, describe_hindi
from .hinglish import CLASS_HING, describe_hinglish, to_hinglish
from .metrics import Metrics
from .pipeline import PipelineInput, run_once
from .scene import (
    detection_signature,
    frame_signature,
    scene_is_same,
    scene_meaningfully_changed,
)
from .strings import (
    BUSY,
    COMMAND_NAMES,
    EVENTS,
    MESSAGES,
    MESSAGES_DISPLAY,
    PATH_DISPLAY,
    SOURCES,
)
from .translate import has_devanagari, to_hindi

log = logging.getLogger(__name__)

COMMANDS = ("read", "describe", "changed", "repeat", "mute", "quit")

def emotion_tag(text: str) -> str:
    """Inline emotion tag chosen from keywords (English or Hindi)."""
    low = text.lower()
    if any(k in low for k in ("clear", "open", "साफ", "खुला", "खाली")):
        return "[happy] "
    if any(k in low for k in ("obstacle", "blocked", "बाधा", "रुकावट", "बंद")):
        return "[warm] "
    return ""


def spoken_item(cls_name: str, z: str, dist: str) -> str:
    side = {"L": "बाईं ओर", "R": "दाईं ओर", "C": "सामने"}[z]
    near = {"near": "बहुत पास ", "mid": "पास ", "far": ""}[dist]
    return f"{CLASS_HI.get(cls_name, cls_name)} {near}{side}"


def _default_ocr(frame):
    from .ocr import read_text_from_frame
    return read_text_from_frame(frame)


def _default_vlm(frame, dets, shape):
    from .vlm import describe_scene_structured
    return describe_scene_structured(frame, dets, shape)


class Assistant:
    def __init__(
        self,
        detector,
        source,
        *,
        metrics: Optional[Metrics] = None,
        events: Optional[EventBus] = None,
        ocr_fn: Callable = _default_ocr,
        vlm_fn: Callable = _default_vlm,
        demo: bool = False,
        voice: bool = True,
    ):
        self.detector = detector
        self.source = source
        self.metrics = metrics or Metrics(None)
        self.events = events or EventBus()
        self.ocr_fn, self.vlm_fn = ocr_fn, vlm_fn
        self.demo = demo
        self.voice = voice and cfg.VOICE_ENABLED and not demo

        self.quit_event = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="api")
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._raw = None            # latest un-annotated frame
        self._annotated = None      # latest frame with boxes
        self._frame_id = 0
        self._jpeg_cache = (-1, b"")
        self._dets: Dict = {}
        self._shape = (cfg.CAMERA_HEIGHT, cfg.CAMERA_WIDTH, 3)

        self.recorder = None
        self._busy: Optional[str] = None
        self._last_busy_msg = 0.0
        self._last_scene = None     # {'det','pix','ts'} of the last description

        self._fps = 0.0
        self._infer_ms = 0.0
        self._camera_ok = True
        self._started = time.time()
        self._last_text = ""
        self._last_display = ""
        self._last_lang = "en"
        self._voice_active = False

    def attach_recorder(self, recorder) -> None:
        self.recorder = recorder
        self.events.add_sink(recorder.event)

    def _record(self, request, frame, dets, res) -> None:
        if self.recorder is not None:
            try:
                self.recorder.request(request, frame, dets, res)
            except Exception:
                log.exception("session recording failed")

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        tts.set_speech_listener(self._on_speech)
        self.events.publish("system", EVENTS["starting_demo"] if self.demo else EVENTS["starting"])

        if not self.demo:
            tts.init_tts()
            from .net import warm_connection
            threading.Thread(target=warm_connection, daemon=True, name="warmup").start()

        if self.voice:
            from .voice import start_listening
            self._voice_active = bool(start_listening(lambda c: self.dispatch(c, "voice")))
            self.events.publish(
                "system",
                EVENTS["voice_on"] if self._voice_active else EVENTS["voice_off"],
                "info" if self._voice_active else "warn",
            )

        self._thread = threading.Thread(target=self._loop, daemon=True, name="capture")
        self._thread.start()
        self._say("ready")

    def stop(self) -> None:
        self.quit_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._pool.shutdown(wait=False, cancel_futures=True)
        try:
            self.source.release()
        finally:
            tts.set_speech_listener(None)
            if self.recorder is not None:
                self.recorder.close()

    # --------------------------------------------------------------- speech
    def _on_speech(self, text: str, lang: str, display=None) -> None:
        self._last_text, self._last_lang = text, lang
        self._last_display = display or to_hinglish(text)
        self.events.publish("speech", self._last_display)

    def _hindi(self, text: str):
        """Hindi version of `text` (translated if needed), or None if translation failed."""
        if has_devanagari(text):
            return text
        hi = to_hindi(text)
        return hi if has_devanagari(hi) else None

    def _fail_translation(self) -> None:
        self.events.publish("error", MESSAGES_DISPLAY["translate_error"], "error")
        tts.speak(MESSAGES["translate_error"], lang="hi", display=MESSAGES_DISPLAY["translate_error"])

    def _say(self, key: str) -> None:
        tts.speak(MESSAGES[key], lang="hi", display=MESSAGES_DISPLAY[key])

    def _speak_reasoning(self, text: str, emotion: str, use_tags: bool = False):
        """Speak VLM/OCR output (never safety alerts) in Hindi. Returns the
        Hinglish text shown on screen, or None if translation failed."""
        hindi = self._hindi(text)
        if hindi is None:
            self._fail_translation()
            return None
        display = to_hinglish(hindi)
        if cfg.USE_SONIFY_FOR_VLM:
            tag = emotion_tag(text) if (use_tags and cfg.USE_EMOTION_TAGS) else ""
            tts.speak_hindi_emotional(tag + hindi, emotion=emotion, display=display)
        else:
            tts.speak(hindi, lang="hi", display=display)
        return display

    # -------------------------------------------------------------- frames
    def get_frame(self):
        with self._lock:
            return None if self._raw is None else self._raw.copy()

    def get_detections(self) -> Dict:
        with self._lock:
            return dict(self._dets)

    def jpeg(self) -> bytes:
        """Latest annotated frame as JPEG (encoded once per frame)."""
        with self._lock:
            fid, frame = self._frame_id, self._annotated
            cached_id, cached = self._jpeg_cache
        if frame is None:
            return b""
        if cached_id == fid:
            return cached
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY])
        data = buf.tobytes() if ok else b""
        with self._lock:
            self._jpeg_cache = (fid, data)
        return data

    def annotated_frame(self):
        with self._lock:
            return None if self._annotated is None else self._annotated.copy()

    @property
    def frame_id(self) -> int:
        return self._frame_id

    # --------------------------------------------------------------- state
    def snapshot(self) -> Dict:
        with self._lock:
            dets = dict(self._dets)
            h, w = self._shape[:2]
        items = []
        for name, (conf, xyxy, xc, area) in sorted(dets.items(), key=lambda kv: -kv[1][3]):
            items.append({
                "name": name,
                "name_hi": CLASS_HI.get(name, name),
                "label": CLASS_HING.get(name, name),
                "conf": round(conf, 2),
                "zone": zone(xc, w),
                "prox": proximity(area),
                "box": [round(xyxy[0] / w, 4), round(xyxy[1] / h, 4),
                        round(xyxy[2] / w, 4), round(xyxy[3] / h, 4)],
            })
        from . import ocr, vlm
        return {
            "demo": self.demo,
            "camera_ok": self._camera_ok,
            "busy": self._busy,
            "fps": round(self._fps, 1),
            "infer_ms": round(self._infer_ms, 1),
            "uptime_s": int(time.time() - self._started),
            "muted": tts.is_muted(),
            "voice": self._voice_active,
            "last_text": self._last_text,
            "last_display": self._last_display,
            "last_lang": self._last_lang,
            "detections": items,
            "metrics": self.metrics.summary(),
            "services": {
                "tts": tts.status(),
                "ocr_circuit": ocr.breaker_state(),
                "vlm_circuits": vlm.breaker_states(),
            },
            "quit": self.quit_event.is_set(),
        }

    # ------------------------------------------------------------- commands
    def _begin(self, label: str) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = label
            return True

    def _end(self) -> None:
        with self._lock:
            self._busy = None

    def dispatch(self, cmd: str, source: str = "ui") -> bool:
        """Handle a command from keyboard, voice or web. Returns True if it
        was accepted."""
        if cmd not in COMMANDS:
            return False
        self.events.publish("command", f"{COMMAND_NAMES[cmd]} ({SOURCES.get(source, source)})")

        if cmd == "quit":
            self.quit_event.set()
            return True
        if cmd == "mute":
            tts.set_muted(not tts.is_muted())
            tts.init_tts()
            self.events.publish("system", EVENTS["muted"] if tts.is_muted() else EVENTS["unmuted"])
            return True
        if cmd == "repeat":
            if not tts.repeat_last():
                self.events.publish("system", EVENTS["nothing_to_repeat"])
            return True
        if cmd == "changed":          # offline and instant: never blocked
            self._run_changed()
            return True

        if self._busy:
            self.events.publish("system", f"{EVENTS['skipped_busy']}: {self._busy}", "warn")
            if time.time() - self._last_busy_msg > 4.0:
                self._last_busy_msg = time.time()
                self._say("busy")
            return False

        if cmd == "read":
            if not self._begin(BUSY["read"]):
                return False
            self._pool.submit(self._run_ocr)
            return True

        # describe
        frame = self.get_frame()
        if frame is None:
            return False
        dets = self.get_detections()
        cur_sig = detection_signature(dets, frame.shape)
        pix = frame_signature(frame)
        if scene_is_same(self._last_scene, cur_sig, pix, time.time(),
                         cfg.SCENE_SAME_MAX_AGE_S, cfg.SCENE_PIXEL_THRESH):
            self.events.publish("system", EVENTS["scene_same"])
            self._say("scene_unchanged")
            return True
        if not self._begin(BUSY["describe"]):
            return False
        self._pool.submit(self._run_vlm, cur_sig, pix, dets, frame.shape)
        return True

    # -------------------------------------------------------------- workers
    def _speak_utterance(self, u, emotion: str, use_tags: bool = False) -> None:
        """Speak one pipeline utterance (Devanagari voice, Hinglish screen)."""
        if cfg.USE_SONIFY_FOR_VLM and u.kind in ("answer", "obstacle", "summary"):
            tag = emotion_tag(u.text) if (use_tags and cfg.USE_EMOTION_TAGS) else ""
            tts.speak_hindi_emotional(tag + u.text, emotion=emotion, display=u.display)
        else:
            tts.speak(u.text, lang="hi", display=u.display)

    def _run_ocr(self) -> None:
        frame = self.get_frame()
        if frame is None:
            self._end()
            return
        try:
            res = run_once(PipelineInput(frame=frame), "read", ocr_fn=self.ocr_fn,
                           vlm_fn=self.vlm_fn, translate_fn=self._hindi)
            self._record("read", frame, None, res)
            t = res.trace("ocr")
            ms = int(t.ms) if t else int(res.latency_ms)
            if res.status == "error":
                if res.error and res.error.startswith("OCR error"):
                    self.metrics.record("ocr", res.error[:60], ms)
                    self.events.publish("error", MESSAGES_DISPLAY["ocr_error"], "error")
                else:   # translation failed
                    self.events.publish("error", MESSAGES_DISPLAY["translate_error"], "error")
                self._speak_utterance(res.utterances[0], cfg.OCR_EMOTION)
                return
            self.metrics.record("ocr", (res.raw.get("raw_text") or "")[:60], ms)
            if res.abstained:       # no text found, or the guardrails said "not sure"
                u = res.utterances[0]
                self.events.publish("ocr", u.display)
                tts.speak(u.text, lang="hi", display=u.display)
                if res.status == "abstain":
                    self.metrics.record("abstain", res.reason or "", ms)
                return
            for u in res.utterances:
                self._speak_utterance(u, cfg.OCR_EMOTION)
            self.events.publish("ocr", (res.display_text or EVENTS["empty"])[:200])
        except Exception as exc:
            log.exception("OCR failed")
            self.metrics.record("ocr_error", str(exc)[:80], 0)
            self.events.publish("error", EVENTS["ocr_failed"], "error")
            self._say("ocr_error")
        finally:
            self._end()

    def _run_vlm(self, cur_sig, pix, dets, shape) -> None:
        frame = self.get_frame()
        if frame is None:
            self._end()
            return
        start = time.time()
        try:
            res = run_once(PipelineInput(frame=frame, detections=dets), "describe",
                           ocr_fn=self.ocr_fn, vlm_fn=self.vlm_fn, translate_fn=self._hindi)
            self._record("describe", frame, dets, res)
            ms = int((time.time() - start) * 1000)
            if res.status == "error":
                self.events.publish("error", MESSAGES_DISPLAY["translate_error"], "error")
                self._speak_utterance(res.utterances[0], cfg.HINDI_EMOTION)
                return
            if res.abstained:       # not sure: ask for a retry instead of guessing
                u = res.utterances[0]
                self.metrics.record("abstain", res.reason or "", ms)
                self.events.publish("vlm", u.display)
                tts.speak(u.text, lang="hi", display=u.display)
                return
            self.metrics.record("vlm", (res.raw.get("vlm", {}).get("summary") or "")[:60], ms)

            # Remember the scene only after the description succeeded.
            self._last_scene = {"det": cur_sig, "pix": pix, "ts": time.time()}

            # Most important first; each piece is its own utterance.
            for u in res.utterances:
                self._speak_utterance(u, cfg.HINDI_EMOTION, use_tags=True)
            fallback = PATH_DISPLAY.get((res.raw.get("vlm") or {}).get("path_status"), "")
            self.events.publish("vlm", " | ".join(u.display for u in res.utterances) or fallback)
        except Exception as exc:
            if isinstance(exc, RuntimeError):      # providers down / rate limited: expected
                log.warning("scene description failed: %s", exc)
            else:
                log.exception("scene description failed")
            self.metrics.record("vlm_error", str(exc)[:80], int((time.time() - start) * 1000))
            self.events.publish("error", EVENTS["vlm_failed"], "error")
            self._say("vlm_error")
        finally:
            self._end()

    def _run_changed(self) -> None:
        """Offline: compare YOLO objects now vs. at the last description."""
        if self._last_scene is None:
            self._say("nothing_to_compare")
            return
        frame = self.get_frame()
        if frame is None:
            return
        cur = detection_signature(self.get_detections(), frame.shape)
        changed, added, removed = scene_meaningfully_changed(self._last_scene["det"], cur)
        if not changed:
            same = scene_is_same(self._last_scene, cur, frame_signature(frame),
                                 time.time(), 1e9, cfg.SCENE_PIXEL_THRESH)
            self._say("no_change" if same else "view_changed")
            return
        parts = []
        if added:
            parts.append(f"{MESSAGES['added']}: " + ", ".join(spoken_item(*a) for a in sorted(added)[:3]))
        if removed:
            parts.append(f"{MESSAGES['removed']}: " + ", ".join(spoken_item(*r) for r in sorted(removed)[:3]))
        shown = [f"{MESSAGES_DISPLAY['added']}: " + ", ".join(
                     CLASS_HING.get(a[0], a[0]) for a in sorted(added)[:3])] if added else []
        if removed:
            shown.append(f"{MESSAGES_DISPLAY['removed']}: " + ", ".join(
                CLASS_HING.get(r[0], r[0]) for r in sorted(removed)[:3]))
        tts.speak("। ".join(parts) + "।", lang="hi", display=". ".join(shown) + ".")

    # ------------------------------------------------------------ main loop
    def _loop(self) -> None:
        frame_idx = read_failures = reconnects = 0
        fps_t, fps_n = time.time(), 0
        try:
            while not self.quit_event.is_set():
                ok, frame = self.source.read()
                if not ok or frame is None:
                    self._camera_ok = False
                    read_failures += 1
                    if read_failures % 10 == 0:
                        reconnects += 1
                        if reconnects > 3:
                            log.error("camera could not be recovered")
                            self.events.publish("error", EVENTS["camera_dead"], "error")
                            self._say("camera_error")
                            time.sleep(3)   # let the message play
                            break
                        log.warning("camera lost; reconnect attempt %d/3", reconnects)
                        self.events.publish("system", f"{EVENTS['camera_lost']} ({reconnects}/3)", "warn")
                        self.source.release()
                        time.sleep(1.0)
                        self.source.open()
                    time.sleep(0.05)
                    continue

                if not self._camera_ok:
                    self.events.publish("system", EVENTS["camera_back"])
                self._camera_ok = True
                read_failures = reconnects = 0

                if frame_idx % cfg.INFER_EVERY == 0:
                    t0 = time.time()
                    try:
                        current, newly = self.detector.detect(frame)
                    except Exception:
                        log.exception("detection failed")
                        current, newly = {}, []
                    ms = (time.time() - t0) * 1000
                    self._infer_ms = ms if not self._infer_ms else 0.8 * self._infer_ms + 0.2 * ms
                    with self._lock:
                        self._dets = dict(current)
                    if newly:
                        self._announce(newly, frame.shape)

                with self._lock:
                    dets = dict(self._dets)
                annotated = draw_detections(frame.copy(), dets, labels=False)
                with self._lock:
                    self._raw, self._annotated = frame, annotated
                    self._shape = frame.shape
                    self._frame_id += 1

                frame_idx += 1
                fps_n += 1
                now = time.time()
                if now - fps_t >= 1.0:
                    self._fps = fps_n / (now - fps_t)
                    fps_t, fps_n = now, 0
        except Exception:
            log.exception("capture loop crashed")
            self.events.publish("error", EVENTS["loop_crashed"], "error")
        finally:
            self.quit_event.set()

    def _announce(self, newly, shape) -> None:
        top = self.detector.prioritize(newly, top_n=2)
        msg_hi = "; ".join(describe_hindi(o, shape) for o in top)
        msg_show = "; ".join(describe_hinglish(o, shape) for o in top)
        self.metrics.record("detect", msg_show[:80])
        self.events.publish("detect", msg_show)
        tts.speak(msg_hi, lang="hi", display=msg_show)
