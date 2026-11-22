"""Decide what a request is asking for.

Two entry points:

* ``resolve_command(text)``: the voice / keyboard channel. One command word
  ("padho", "पढ़ो", "dohrao") -> ``read | describe | change | repeat | stop``.
  It reuses the Hindi command table of ``voice.py`` (the words Vosk is
  restricted to) and adds Roman spellings, because people also type Hinglish.
* ``route(request)``: a free-form question ("Ye dawa ka naam kya hai?") ->
  which modality should answer it, or a *conflict* when it is unclear.

The router is deliberately small and rule-based: a word list, not a model. It
is wrong on phrasings it has never seen, and eval/ measures exactly that. The
important property is the failure mode: when it is not sure, it says
``conflict`` and the guardrails turn that into a clarifying question instead
of guessing a modality.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from .voice import COMMANDS as _HINDI_COMMANDS

# voice.py / engine.py name the intents "changed" and "quit"; the evaluation
# (and this module's public API) uses "change" and "stop".
_INTENT = {"changed": "change", "quit": "stop"}

COMMAND_WORDS: dict[str, str] = {}
for _w, _c in _HINDI_COMMANDS.items():
    COMMAND_WORDS[unicodedata.normalize("NFC", _w)] = _INTENT.get(_c, _c)
for _intent, _words in {
    "read": "padho padhiye padhein padh paddho",
    "describe": "batao bataiye bataao varnan dekho",
    "change": "badla badlaav badlav badal badli",
    "repeat": "dohrao dohraiye dobara dubara phir",
    "stop": "band bahar samapt",
}.items():
    for _w in _words.split():
        COMMAND_WORDS[_w] = _intent

# Engine command (engine.COMMANDS) for each routed intent.
ENGINE_COMMAND = {"read": "read", "describe": "describe", "change": "changed",
                  "repeat": "repeat", "stop": "quit"}

# Strong cues name the action itself; weak cues only hint at it, and only
# count when no strong cue is present.
STRONG_READ = set("padho padh padhna padhiye read likha likhi likhe likhe".split())
WEAK_READ = set("""naam name dawa dawai dose warning menu sign board number rate price label text
                   dukaan special platform""".split())
STRONG_DESCRIBE = set("""saamne samne aaspaas aas-paas aage rukaawat rukavat raaste rasta taraf
                         describe scene dikh dikha dikhai""".split())
WEAK_DESCRIBE = set("kaun kitne kitni zyada log chair person kutta dog".split())
# A free model must never certify safety, so these are never answered.
SAFETY = set("safe surakshit".split())

_TOKEN = re.compile(r"[\wऀ-ॿ-]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFC", (text or "").lower()))


def resolve_command(text: Optional[str]) -> Optional[str]:
    """Intent of a one-word command, or None if ``text`` is not one."""
    toks = tokens(text or "")
    if not toks:
        return None
    intents = {COMMAND_WORDS.get(t) for t in toks}
    return intents.pop() if len(intents) == 1 and None not in intents else None


@dataclass
class Route:
    intent: str               # read | describe | change | repeat | stop | unclear
    modality: str             # ocr | vlm | asr | none
    conflict: bool = False
    why: str = ""


def route(request: str) -> Route:
    toks = set(tokens(request))
    if SAFETY & toks:
        return Route("unclear", "none", True, "safety question")

    sr, sd = bool(STRONG_READ & toks), bool(STRONG_DESCRIBE & toks)
    wr, wd = bool(WEAK_READ & toks), bool(WEAK_DESCRIBE & toks)
    if sr and sd:
        return Route("unclear", "none", True, "read and describe in one request")
    if sr:
        return Route("read", "ocr")
    if sd:
        return Route("describe", "vlm")
    if wr and not wd:
        return Route("read", "ocr")
    if wd and not wr:
        return Route("describe", "vlm")
    cmd = resolve_command(request)
    if cmd in ("change", "repeat", "stop"):
        return Route(cmd, "asr")
    if cmd == "describe":
        return Route("describe", "vlm")
    return Route("unclear", "none", True, "no usable target")
