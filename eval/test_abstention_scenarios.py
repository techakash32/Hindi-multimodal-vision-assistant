"""End to end: the ten negative_abstain scenarios through the real pipeline.

Uses the ``fake`` backend (scripted OCR, demo detector, template VLM), so it
needs no network and no Tesseract. The fake OCR returns garbage with low
confidence for the "upside-down text" scenario; the other nine abstain on image
quality, a missing frame or an unclear request, before any model is called.
"""
from pathlib import Path

import pytest

from eval import backends as be
from eval.run_eval import load_scenarios, run_scenario
from visionassist.guardrails import REASON_MESSAGE, AbstentionConfig
from visionassist.strings import MESSAGES, MESSAGES_DISPLAY

SCEN = Path(__file__).resolve().parent / "scenarios.jsonl"
NEGATIVES = [s for s in load_scenarios(SCEN) if s["category"] == "negative_abstain"]


@pytest.fixture(scope="module")
def backend():
    return be.build("fake", SCEN)


def test_there_are_ten_negative_scenarios():
    assert len(NEGATIVES) == 10 and all(s["expected_behavior"] == "abstain" for s in NEGATIVES)


@pytest.mark.parametrize("sc", NEGATIVES, ids=lambda s: s["id"])
def test_negative_scenario_abstains_with_the_expected_clarification(sc, backend):
    rec = run_scenario(sc, backend, AbstentionConfig.load())
    assert rec["abstained"] is True, rec["display_text"]
    assert rec["reason"] == sc["expected_reason"]
    key = REASON_MESSAGE[sc["expected_reason"]]
    assert rec["answer_text"] == MESSAGES[key]            # Devanagari, for the voice
    assert rec["display_text"] == MESSAGES_DISPLAY[key]   # Hinglish, for the screen
    assert rec["raw_output"].get("vlm") is None           # no confident answer was produced


def test_abstaining_before_a_model_call_saves_that_call(backend):
    """Blurry, dark and missing-frame scenarios must not reach OCR / VLM at all."""
    cfg = AbstentionConfig.load()
    for sc in NEGATIVES:
        if sc["id"] in {f"negative_abstain_00{i}" for i in (1, 2, 4, 5, 6, 7, 8, 9)}:
            rec = run_scenario(sc, backend, cfg)
            assert rec["actual_modalities"] == [], sc["id"]


def test_the_same_scenarios_are_answered_when_guardrails_are_off(backend):
    """Proof that the abstentions come from the guardrails and not from the fakes."""
    off = AbstentionConfig.disabled()
    answered = [run_scenario(sc, backend, off) for sc in NEGATIVES
                if sc["input"]["image"] and sc["id"] != "negative_abstain_003"]
    assert any(not r["abstained"] for r in answered)
