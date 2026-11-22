import json

import pytest

from visionassist.guardrails import (
    REASON_MESSAGE,
    AbstentionConfig,
    Percept,
    is_hedged,
    should_abstain,
    staged_decision,
)
from visionassist.strings import MESSAGES, MESSAGES_DISPLAY

CFG = AbstentionConfig(ocr_tau=0.60, vlm_tau=0.50, aggregate_tau_low=0.45, aggregate_tau_high=0.70)


def P(name, conf, **kw):
    return {name: Percept(name, conf, **kw)}


# ---- confidence thresholds -------------------------------------------------
def test_abstains_on_low_ocr_confidence():
    assert should_abstain(P("ocr", 0.30), CFG) == (True, "ocr_low")


def test_answers_on_high_ocr_confidence():
    assert should_abstain(P("ocr", 0.95), CFG) == (False, None)


def test_abstains_on_low_vlm_confidence():
    assert should_abstain(P("vlm", 0.20), CFG) == (True, "vlm_low")


def test_answers_on_high_vlm_confidence():
    assert should_abstain(P("vlm", 0.90), CFG) == (False, None)


@pytest.mark.parametrize("name,tau,reason", [("ocr", 0.60, "ocr_low"), ("vlm", 0.50, "vlm_low")])
def test_threshold_boundary_is_inclusive_for_answering(name, tau, reason):
    """conf == tau answers; just below abstains."""
    assert should_abstain(P(name, tau), CFG)[0] is False
    assert should_abstain(P(name, tau - 1e-6), CFG) == (True, reason)


def test_image_confidence_uses_the_vlm_threshold():
    assert should_abstain(P("image", 0.49), CFG) == (True, "vlm_low")
    assert should_abstain(P("image", 0.50), CFG)[0] is False


def test_aggregate_low_threshold_catches_two_mediocre_percepts():
    # each above its own tau, but the mean (0.44) is under aggregate_tau_low
    percepts = {**P("ocr", 0.61), **P("vlm", 0.27 + 0.23)}
    assert should_abstain(percepts, CFG)[0] is False            # mean 0.555
    cfg = AbstentionConfig(ocr_tau=0.5, vlm_tau=0.4, aggregate_tau_low=0.70, aggregate_tau_high=0.8)
    assert should_abstain({**P("ocr", 0.55), **P("vlm", 0.50)}, cfg) == (True, "vlm_low")


def test_unmeasured_confidence_never_abstains():
    assert should_abstain(P("vlm", None), CFG) == (False, None)
    assert should_abstain({}, CFG) == (False, None)


# ---- missing modality / conflict -------------------------------------------
def test_missing_modality_always_abstains():
    assert should_abstain(P("ocr", None, available=False), CFG) == (True, "missing_modality")
    assert should_abstain({**P("ocr", 0.99), **P("image", None, available=False)}, CFG) == (True, "missing_modality")


def test_conflict_always_abstains_even_with_high_confidence():
    percepts = {**P("ocr", 0.99), **P("router", None, conflict=True)}
    assert should_abstain(percepts, CFG) == (True, "conflict")


def test_missing_modality_beats_low_confidence():
    percepts = {**P("ocr", 0.1), **P("image", None, available=False)}
    assert should_abstain(percepts, CFG) == (True, "missing_modality")


# ---- config ---------------------------------------------------------------
def test_disabled_config_never_abstains():
    assert should_abstain(P("ocr", 0.0, available=False), AbstentionConfig.disabled()) == (False, None)


def test_config_loads_json_and_env_overrides(tmp_path, monkeypatch):
    f = tmp_path / "a.json"
    f.write_text(json.dumps({"ocr_tau": 0.62, "vlm_tau": 0.55, "unknown": 1}))
    cfg = AbstentionConfig.load(f)
    assert (cfg.ocr_tau, cfg.vlm_tau) == (0.62, 0.55)
    monkeypatch.setenv("ABSTAIN_OCR_TAU", "0.9")
    assert AbstentionConfig.load(f).ocr_tau == 0.9
    monkeypatch.setenv("ABSTAIN_ENABLED", "off")
    assert AbstentionConfig.load(f).enabled is False


def test_config_survives_a_missing_file(tmp_path):
    assert AbstentionConfig.load(tmp_path / "nope.json").ocr_tau == AbstentionConfig().ocr_tau


def test_shipped_config_is_valid():
    cfg = AbstentionConfig.load()
    assert 0 < cfg.aggregate_tau_low < cfg.aggregate_tau_high <= 1
    assert 0 < cfg.ocr_tau <= 1 and 0 < cfg.vlm_tau <= 1


# ---- hedging, staging -------------------------------------------------------
def test_hedged_only_between_the_aggregate_thresholds():
    assert is_hedged({**P("ocr", 0.65), **P("vlm", 0.55)}, CFG) is True       # mean 0.60
    assert is_hedged({**P("ocr", 0.95), **P("vlm", 0.90)}, CFG) is False
    assert is_hedged(P("ocr", None), CFG) is False


def test_staged_decision_stops_at_the_first_failing_phase():
    phases = [P("image", 0.9), {**P("image", 0.9), **P("vlm", 0.1)}]
    assert staged_decision(phases, CFG) == (True, "vlm_low", 1)
    assert staged_decision([P("image", 0.9)], CFG) == (False, None, 0)


# ---- user-facing strings ------------------------------------------------------
def test_every_reason_has_a_devanagari_voice_line_and_a_roman_screen_line():
    spoken = {"abstain_ocr": "पक्का नहीं हूँ — कैमरा थोड़ा पास करो?",
              "abstain_vlm": "मुझे ठीक से दिख नहीं रहा, दोबारा कोशिश करें?",
              "abstain_missing": "कैमरा उपलब्ध नहीं है।",
              "abstain_conflict": "जानकारी मिल नहीं रही, थोड़ा और बताओ।"}
    shown = {"abstain_ocr": "Pakka nahi hoon — camera thoda paas karo?",
             "abstain_vlm": "Mujhe theek se dikh nahi raha, dobara koshish karein?",
             "abstain_missing": "Camera available nahi hai.",
             "abstain_conflict": "Jaankari mil nahi rahi, thoda aur batao."}
    assert set(REASON_MESSAGE.values()) == set(spoken)
    for key in spoken:
        assert MESSAGES[key] == spoken[key]
        assert MESSAGES_DISPLAY[key] == shown[key]
