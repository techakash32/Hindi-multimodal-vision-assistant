"""Metrics: CER / WER, a judge for scene descriptions, per-scenario verdicts.

CER / WER use ``jiwer`` when it is installed (pinned in requirements-dev.txt)
and an equivalent built-in Levenshtein implementation otherwise, so the
harness also runs where jiwer cannot be installed. A test checks they agree.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Callable, Optional, Sequence

try:                                    # pragma: no cover - depends on the environment
    import jiwer as _jiwer
except ImportError:                     # pragma: no cover
    _jiwer = None

_PUNCT = re.compile(r"[.,;:!?'\"“”‘’()\[\]]")


def normalize(text: Optional[str]) -> str:
    """casefold, drop sentence punctuation, collapse whitespace. Digits, units
    and letters are kept: '500mg' vs '500 mg' still counts as an error."""
    t = unicodedata.normalize("NFC", text or "").casefold()
    return re.sub(r"\s+", " ", _PUNCT.sub("", t)).strip()


def _edit_distance(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    if _jiwer is not None:
        return float(_jiwer.cer(ref, hyp)) if hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    if _jiwer is not None:
        return float(_jiwer.wer(" ".join(ref), " ".join(hyp))) if hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


# ------------------------------------------------------------ judging
JUDGE_LABELS = ("correct", "partially_correct", "incorrect")

_STOP = set("hai hain ek ki ka ke ko se mein me par pe aur ya kya yeh ye woh vo aap".split())
_SYN = {"kursi": "chair", "kursiyan": "chair", "vyakti": "person", "log": "person", "aadmi": "person",
        "insaan": "person", "kutta": "dog", "kutte": "dog", "cheyar": "chair", "kuta": "dog",
        "baayi": "baayin", "bayin": "baayin", "baayen": "baayin", "daayi": "daayin", "dayin": "daayin",
        "samne": "saamne", "aamne": "saamne", "paas": "paas", "persons": "person", "chairs": "chair",
        "ek": "ek"}
_OBJECTS = {"person", "chair", "dog", "bus"}
_SIDES = {"baayin", "daayin", "saamne"}
_COUNTS = {"do", "teen", "char", "paanch", "chhe", "saat"}


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").casefold())
    return [_SYN.get(t, t) for t in toks if t not in _STOP]


def keyword_judge(question: str, answer: str, truth: str) -> str:
    """Deterministic fallback judge. A description is

    * ``incorrect``  if it contradicts the ground truth (other side, other
      count) or covers under half of its key words (objects, sides, counts,
      distance words, negations);
    * ``partially_correct`` if it covers 50-99% of the key words, or covers all
      of them but also names an object the ground truth does not contain
      (a likely hallucination);
    * ``correct`` otherwise.
    It is crude by design and deliberately not tuned to pass.
    """
    t, a = _tokens(truth), set(_tokens(answer))
    if not t or not a:
        return "incorrect"
    key = [w for w in t if w in _OBJECTS | _SIDES | _COUNTS | {"paas", "bahut", "zyada", "nahi", "rukaawat"}
           or w.isdigit()]
    if not key:
        key = t
    t_sides, a_sides = {w for w in t if w in _SIDES}, {w for w in a if w in _SIDES}
    if t_sides and a_sides and not (t_sides & a_sides):
        return "incorrect"
    t_cnt, a_cnt = {w for w in t if w in _COUNTS}, {w for w in a if w in _COUNTS}
    if t_cnt and a_cnt and t_cnt != a_cnt:
        return "incorrect"
    recall = sum(1 for w in set(key) if w in a) / len(set(key))
    if recall < 0.5:
        return "incorrect"
    extra_objects = {w for w in a if w in _OBJECTS} - {w for w in t if w in _OBJECTS}
    if recall < 1.0 or extra_objects:
        return "partially_correct"
    return "correct"


class LLMJudge:
    """LLM-as-judge with an on-disk cache (reports/vlm_judge_cache.json), so a
    rerun costs nothing. ``call`` takes the prompt and returns the model text;
    the default goes to the same OpenRouter endpoint the app uses. With no API
    key, or when the call fails, it falls back to ``keyword_judge``."""

    PROMPT = (
        "You grade a scene description for a blind user.\n"
        "Question: {q}\nModel answer: {a}\nGround truth: {t}\n"
        "Reply with exactly one word: correct, partially_correct or incorrect. "
        "Use correct only if every object, side and count in the ground truth is stated correctly "
        "and nothing is invented."
    )

    def __init__(self, cache_path: Path, call: Optional[Callable[[str], str]] = None, model: str = ""):
        self.path = Path(cache_path)
        self.call, self.model = call, model
        self.cache: dict[str, str] = {}
        try:
            self.cache = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        self.calls = self.hits = self.fallbacks = 0

    def _key(self, q, a, t) -> str:
        return hashlib.sha1(json.dumps([self.model, q, a, t], ensure_ascii=False).encode()).hexdigest()

    def __call__(self, question: str, answer: str, truth: str) -> str:
        key = self._key(question, answer, truth)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if self.call is not None:
            try:
                reply = self.call(self.PROMPT.format(q=question, a=answer, t=truth)).strip().lower()
                label = next((lab for lab in ("partially_correct", "incorrect", "correct") if lab in reply), None)
                if label:
                    self.calls += 1
                    self.cache[key] = label
                    return label
            except Exception:
                pass
            self.fallbacks += 1
        return keyword_judge(question, answer, truth)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def openrouter_call(model: str, timeout: float = 30.0) -> Callable[[str], str]:
    """Text-only chat call to OpenRouter using the app's key and session."""
    from visionassist import config as cfg
    from visionassist.net import SESSION

    def call(prompt: str) -> str:
        r = SESSION.post(cfg.OPENROUTER_URL, timeout=timeout,
                         headers={"Authorization": f"Bearer {cfg.OPENROUTER_API_KEY}"},
                         json={"model": model, "max_tokens": 10, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    return call


# ------------------------------------------------------------ verdicts
OCR_CATEGORIES = ("ocr_medicine", "ocr_sign", "ocr_menu")
ABSTAIN_CATEGORIES = ("ambiguous_question", "negative_abstain")


def score(scenario: dict, rec: dict, judge: Callable[[str, str, str], str]) -> dict:
    """Verdict for one run.

    ``verdict`` is one of correct | partial | incorrect | abstained, where
    "abstained" on an answerable scenario means no answer was given (a false
    negative) and, for an abstain-expected scenario, ``correct`` means the
    assistant abstained and ``incorrect`` that it answered confidently.
    """
    cat, expected = scenario["category"], scenario["expected_behavior"]
    out: dict = {}
    if expected == "abstain":
        out["verdict"] = "correct" if rec["abstained"] else "incorrect"
        out["reason_match"] = (rec.get("reason") == scenario.get("expected_reason")) if rec["abstained"] else None
        return out
    if rec["abstained"]:
        out["verdict"] = "abstained"
        if cat in OCR_CATEGORIES and "ocr" in rec.get("actual_modalities", []):
            text = (rec.get("raw_output") or {}).get("raw_text", "")   # OCR ran, then we held back
            out["cer"], out["wer"] = cer(scenario["ground_truth"], text), wer(scenario["ground_truth"], text)
        return out
    if cat in OCR_CATEGORIES:
        text = (rec.get("raw_output") or {}).get("raw_text", "")
        out["cer"], out["wer"] = cer(scenario["ground_truth"], text), wer(scenario["ground_truth"], text)
        out["verdict"] = "correct" if out["cer"] == 0 else "incorrect"
    elif cat == "scene_description":
        label = judge(scenario["user_request"], rec.get("display_text", ""), scenario["ground_truth"])
        out["judge"] = label
        out["verdict"] = {"correct": "correct", "partially_correct": "partial"}.get(label, "incorrect")
    elif cat == "voice_command":
        out["verdict"] = "correct" if rec.get("intent") == scenario["ground_truth"] else "incorrect"
    else:
        out["verdict"] = "incorrect"
    return out
