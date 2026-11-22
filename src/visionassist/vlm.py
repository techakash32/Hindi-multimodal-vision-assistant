"""Scene description through free OpenRouter vision-language models.

Several free models are tried with circuit breakers and hedged requests, so a
dead or slow model never blocks the user for the full timeout.
"""
from __future__ import annotations

import base64
import json
import logging
import queue
import re
import threading
import time

import cv2
import requests

from . import config as cfg
from .net import SESSION
from .resilience import CircuitBreaker
from .scene import frame_signature, signatures_similar

log = logging.getLogger(__name__)

VLM_CACHE_TTL = 10.0
_vlm_cache = {"sig": None, "result": None, "ts": 0.0}


PROMPT_EN = (
    "Describe this scene in one short sentence for a blind person. "
    "Prioritize nearby obstacles and important points of interest. "
    "Do not claim an obstacle is safe to pass."
)

PROMPT_HI = (
    "इस सीन को एक छोटे वाक्य में आम बोलचाल की हिंदी में बताएँ (जैसे टेबल, चेयर, डोर जैसे आम अंग्रेज़ी शब्द देवनागरी में लिख सकते हैं), "
    "जो दृष्टिबाधित व्यक्ति के लिए हो। "
    "पास की बाधाओं और महत्वपूर्ण वस्तुओं पर ध्यान दें। "
    "सुरक्षित होने का दावा न करें।"
)

# Free models come and go. Order = preference. Each has a circuit breaker:
# 404 (gone) benches it for an hour, 429 (rate limited) for a minute, and
# repeated other errors for a minute, so we stop wasting seconds on it.
MODELS_TO_TRY = [
    ("nvidia/nemotron-3.5-super-vl:free", {
        "max_tokens": 500,
        "reasoning": {"effort": "none"},   # force reasoning OFF
    }),
    ("google/gemma-4-26b-a4b-it:free", {"max_tokens": 200}),
    # allenai/molmo-2-8b:free was removed: OpenRouter answered 404 (model gone).
    # Free models come and go; set VLM_MODELS in .env to your own working list.
]

def _models():
    """Configured models (VLM_MODELS env) or the built-in preference list."""
    if cfg.VLM_MODELS:
        return [(m, {"max_tokens": 300}) for m in cfg.VLM_MODELS]
    return MODELS_TO_TRY


def breaker_states():
    """{model_id: 'open'|'closed'} for the dashboard."""
    return {m[0]: ("open" if _breaker(m[0]).is_open else "closed") for m in _models()}


_vlm_breakers = {}

# A model that answers correctly but slowly is asked LAST for a while (not
# removed), so one 18-second answer doesn't repeat on the next press.
SLOW_ANSWER_S = 8.0
SLOW_PENALTY_S = 120.0
_slow_until = {}   # model_id -> time until which it is asked last


def _breaker(model_id):
    return _vlm_breakers.setdefault(
        model_id,
        CircuitBreaker(model_id, failure_threshold=2, cooldown_sec=60),
    )


def _request_model(model_id, overrides, b64, prompt, headers):
    """One VLM request. Updates that model's circuit breaker.
    Returns the description text, or raises."""
    cb = _breaker(model_id)
    t_start = time.time()

    try:
        payload = {
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
        }
        payload.update(overrides)

        response = SESSION.post(
            cfg.OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=(5, cfg.VLM_TIMEOUT),   # (connect, read)
        )
        response.raise_for_status()
        result = response.json()

        choices = result.get("choices") or []
        if not choices:
            raise ValueError("No choices returned by the model.")

        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )

        if not isinstance(content, str) or not content.strip():
            fr = choice.get("finish_reason")
            raise ValueError(f"Empty content (finish_reason={fr})")

        cb.record_success()

        elapsed = time.time() - t_start
        if elapsed > SLOW_ANSWER_S:
            _slow_until[model_id] = time.time() + SLOW_PENALTY_S
            log.info(f"{model_id} was slow ({elapsed:.0f}s) — asked last "
                f"for the next {SLOW_PENALTY_S:.0f}s")

        return content.strip()

    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else None

        if code == 404:
            cb.trip(3600, reason="404 model gone")
        elif code == 429:
            cb.trip(60, reason="429 rate limited")
        else:
            cb.record_failure()

        log.warning(f"{model_id} failed: {exc}")
        raise

    except (requests.RequestException, ValueError, KeyError) as exc:
        cb.record_failure()
        log.warning(f"{model_id} failed: {exc}")
        raise


def _race_models(candidates, b64, prompt, headers):
    """
    Ask the preferred model first. If it has not answered within
    cfg.HEDGE_DELAY_S, ask the next entry too and use whichever answers first.
    Same models, same prompt and image, so the answer quality is unchanged;
    only a slow or hung request no longer costs the full timeout.

    If the same model appears twice (only one model is available), the
    second entry is a repeat request: provider latency varies a lot per
    request, so a repeat often returns in a second or two. A model that
    FAILED is never asked again here (a 429 would just repeat).
    """
    results = queue.Queue()
    pending = list(candidates)   # entries not started yet
    running = 0
    failed_ids = set()
    last_err = None
    t0 = time.time()

    def _worker(model_id, overrides):
        try:
            text = _request_model(model_id, overrides, b64, prompt, headers)
            results.put((model_id, text, None))
        except Exception as exc:
            results.put((model_id, None, exc))

    def _start_next():
        """Start the next entry whose model has not failed. Returns its
        model id, or None if there is nothing left to start."""
        nonlocal running

        while pending:
            model_id, overrides = pending.pop(0)

            if model_id in failed_ids:
                continue

            running += 1
            threading.Thread(
                target=_worker, args=(model_id, overrides), daemon=True
            ).start()

            return model_id

        return None

    _start_next()

    while True:
        can_hedge = bool(pending)

        # Once everything is running, no request can outlive its timeouts.
        wait = cfg.HEDGE_DELAY_S if can_hedge else cfg.VLM_TIMEOUT + 7

        try:
            model_id, text, err = results.get(timeout=wait)

        except queue.Empty:
            if not can_hedge:
                raise RuntimeError("All VLM models timed out.") from None

            started_id = _start_next()

            if started_id is not None:
                log.info(f"no answer after {cfg.HEDGE_DELAY_S:.0f}s — "
                    f"also asking {started_id}")

            continue

        running -= 1

        if text is not None:
            ms = int((time.time() - t0) * 1000)
            log.info(f"{model_id} answered in {ms}ms")
            return text

        failed_ids.add(model_id)
        last_err = err

        # Failed fast: move to the next model at once.
        if _start_next() is None and running == 0:
            raise RuntimeError(
                f"All VLM models failed. Last error: {last_err}"
            )


def _ask_vlm(frame, prompt):
    """Shared by describe_scene variants: pick models, encode, race."""
    if cfg.LOCAL_ONLY:
        raise RuntimeError("Cloud scene description is disabled (LOCAL_ONLY).")
    candidates = [m for m in _models() if _breaker(m[0]).allow()]

    if not candidates:
        raise RuntimeError("All VLM models are temporarily unavailable.")

    # Stable sort: recently slow models move to the back, order otherwise kept.
    now_s = time.time()
    candidates.sort(key=lambda m: _slow_until.get(m[0], 0.0) > now_s)

    # Only one model available: allow a repeat request to it if it is slow
    # (see _race_models). Costs one extra free-tier call, only when slow.
    if len(candidates) == 1:
        candidates = candidates * 2

    # Downscale
    h, w = frame.shape[:2]
    if max(h, w) > 512:
        s = 512 / max(h, w)
        frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Could not encode image for vision analysis.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {cfg.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    return _race_models(candidates, b64, prompt, headers)


def describe_scene(frame):
    """Describe the scene (Hindi if configured) using free VLM models."""

    if frame is None or frame.size == 0:
        raise ValueError("Cannot describe an empty image.")

    # Cache check
    sig = frame_signature(frame)
    now = time.time()
    if (_vlm_cache["sig"] is not None
            and now - _vlm_cache["ts"] < VLM_CACHE_TTL
            and signatures_similar(_vlm_cache["sig"], sig)):
        log.info("cache hit — API skipped")
        return _vlm_cache["result"]

    prompt = PROMPT_HI if (cfg.USE_HINDI and cfg.VLM_REPLY_IN_HINDI) else PROMPT_EN
    text = _ask_vlm(frame, prompt)

    _vlm_cache["sig"] = sig
    _vlm_cache["result"] = text
    _vlm_cache["ts"] = time.time()

    return text


# ==================================================
# STRUCTURED DESCRIPTION (YOLO hints + JSON answer)
# ==================================================

PATH_TOKENS = ("obstacle_ahead", "unclear", "none_seen")


def build_vlm_prompt(detections, frame_shape):
    """
    Prompt with YOLO hints and a fixed JSON answer format.
    Deliberately NO left/right "advice" field and no "path is safe" claim:
    a free model must not tell a blind person where to walk.
    """

    _h, w = frame_shape[:2]
    hints = []

    for cls_name, (_conf, _xyxy, xc, area) in (detections or {}).items():
        side = ("left" if xc < w * 0.33 else
                "right" if xc > w * 0.66 else "centre")
        dist = ("very close" if area > 0.15 else
                "close" if area > 0.05 else "far")
        hints.append(f"{cls_name} ({side}, {dist})")

    hint_line = (
        "A simple detector thinks it sees: " + ", ".join(hints) +
        ". It may be wrong; trust the image over this list."
        if hints else
        "A simple detector found no known objects; look at the image yourself."
    )

    lang = ("everyday spoken Hindi written in Devanagari, as people really talk (Hinglish style: "
            "common English words such as table, chair, door, room, bottle, phone may be written "
            "in Devanagari instead of formal Hindi)"
            if (cfg.USE_HINDI and cfg.VLM_REPLY_IN_HINDI) else "English")

    return (
        "You help a blind person understand a camera picture.\n"
        f"{hint_line}\n"
        "Reply with ONLY a JSON object, no other text, with these keys:\n"
        f'"summary": one short sentence in {lang};\n'
        f'"obstacles": list of up to 3 short {lang} phrases for things that '
        "could block walking, nearest first (empty list if none);\n"
        '"path_status": exactly one of "obstacle_ahead", "unclear", '
        '"none_seen";\n'
        '"confidence": a number from 0.0 to 1.0, your honest estimate that the summary is correct '
        "(use below 0.5 if the picture is blurry, dark or hidden).\n"
        "Do not claim anything is safe to pass. If unsure, use \"unclear\"."
    )


def parse_vlm_json(text):
    """
    Parse the model's reply. Returns dict with summary (str),
    obstacles (list[str]), path_status (one of PATH_TOKENS), structured
    (bool) and, if the model reported one, confidence (float 0..1). Never raises; if the reply is not JSON, the whole
    text becomes the summary.
    """
    out = {"summary": "", "obstacles": [], "path_status": "unclear",
           "structured": False}

    if not isinstance(text, str):
        return out

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                summary = data.get("summary")
                obs = data.get("obstacles")
                path = data.get("path_status")

                out["summary"] = summary.strip() if isinstance(summary, str) else ""
                out["obstacles"] = [
                    o.strip() for o in obs
                    if isinstance(o, str) and o.strip()
                ][:3] if isinstance(obs, list) else []
                out["path_status"] = path if path in PATH_TOKENS else "unclear"
                out["structured"] = bool(out["summary"] or out["obstacles"])
                conf = data.get("confidence")
                if isinstance(conf, (int, float)) and not isinstance(conf, bool) and 0.0 <= conf <= 1.0:
                    out["confidence"] = float(conf)   # only present when the model gave one
        except ValueError:
            pass

    if not out["structured"]:
        out["summary"] = text.strip()

    return out


def describe_scene_structured(frame, detections, frame_shape):
    """Describe with YOLO context. Returns parse_vlm_json() dict."""
    if frame is None or frame.size == 0:
        raise ValueError("Cannot describe an empty image.")

    prompt = build_vlm_prompt(detections, frame_shape)
    return parse_vlm_json(_ask_vlm(frame, prompt))
