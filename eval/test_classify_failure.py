"""Every taxonomy bucket must be reachable from a constructed result."""
import pytest

from eval.classify_failure import BUCKETS, classify, is_failure


def sc(category="ocr_sign", modality="ocr", truth="EXIT", behavior="answer", **kw):
    return {"id": f"{category}_x", "category": category, "expected_modality": modality,
            "ground_truth": truth, "expected_behavior": behavior, "user_request": kw.pop("user_request", "padho"),
            "input": {"image": "x.jpg", "text": kw.pop("text", None), "audio": None}, **kw}


def res(**kw):
    base = {"abstained": False, "reason": None, "route": {"modality": "ocr"}, "actual_modalities": ["ocr"],
            "intent": "read", "over_budget": [], "raw_output": {"raw_text": "EXIT"}, "scores": {}}
    base.update(kw)
    return base


def test_detection_miss():
    s = sc("scene_description", "vlm", "Saamne ek person hai.", expected_objects=["person"])
    r = res(route={"modality": "vlm"}, actual_modalities=["detect", "vlm"],
            raw_output={"detections": {}, "vlm": {"summary": "kuch nahi"}}, scores={"verdict": "incorrect"})
    assert classify(s, r) == "detection_miss"


def test_detection_miss_when_count_exceeds_one_box_per_class():
    s = sc("scene_description", "vlm", "Do person hain.", expected_objects=["person"], expected_counts={"person": 2})
    r = res(route={"modality": "vlm"}, raw_output={"detections": {"person": {}}}, scores={"verdict": "incorrect"})
    assert classify(s, r) == "detection_miss"


def test_ocr_detection_fail():
    assert classify(sc(), res(raw_output={"raw_text": ""}, abstained=True, reason="ocr_low")) == "ocr_detection_fail"
    assert classify(sc(), res(raw_output={"raw_text": ""})) == "ocr_detection_fail"


def test_ocr_recognition_error():
    assert classify(sc(), res(raw_output={"raw_text": "EXlT"})) == "ocr_recognition_error"


def test_vlm_hallucination():
    s = sc("scene_description", "vlm", "Saamne ek person hai.", expected_objects=["person"])
    r = res(route={"modality": "vlm"}, raw_output={"detections": {"person": {}}, "vlm": {"summary": "ek dog hai"}},
            scores={"verdict": "incorrect"})
    assert classify(s, r) == "vlm_hallucination"


def test_asr_substitution():
    s = sc("voice_command", "asr", "read", user_request="padho", text="पड़ो", simulated_asr=True)
    assert classify(s, res(intent=None, route={"modality": "none"})) == "asr_substitution"


def test_router_misroute_for_a_typed_command():
    s = sc("voice_command", "asr", "describe", user_request="batao", text="batao")
    assert classify(s, res(intent="read")) == "router_misroute"


def test_router_misroute_for_a_free_question():
    assert classify(sc(), res(route={"modality": "vlm"}, actual_modalities=["vlm"])) == "router_misroute"


def test_latency_timeout_outranks_other_causes():
    assert classify(sc(), res(over_budget=["ocr"], raw_output={"raw_text": "EXlT"})) == "latency_timeout"


def test_failure_to_abstain():
    s = sc("negative_abstain", "ocr", None, behavior="abstain")
    assert classify(s, res(abstained=False)) == "failure_to_abstain"


def test_incorrect_abstention():
    r = res(abstained=True, reason="ocr_low", actual_modalities=[], route={"modality": "ocr"}, raw_output={})
    assert classify(sc(), r) == "incorrect_abstention"


def test_router_unclear_on_an_answerable_request_is_an_incorrect_abstention_not_a_misroute():
    r = res(abstained=True, reason="conflict", actual_modalities=[], route={"modality": "none"}, raw_output={})
    assert classify(sc(), r) == "incorrect_abstention"


def test_unclassified_is_reachable_but_only_for_results_that_are_not_failures():
    assert classify(sc(), res()) == "unclassified"      # a correct run has no bucket
    assert not is_failure(sc(), res(scores={"verdict": "correct"}))


def test_every_bucket_is_covered_by_this_file():
    covered = {"detection_miss", "ocr_detection_fail", "ocr_recognition_error", "vlm_hallucination",
               "asr_substitution", "router_misroute", "latency_timeout", "failure_to_abstain",
               "incorrect_abstention", "unclassified"}
    assert covered == set(BUCKETS)


@pytest.mark.parametrize("behavior,abstained,expected", [("abstain", True, False), ("abstain", False, True)])
def test_is_failure_for_abstention_scenarios(behavior, abstained, expected):
    assert is_failure(sc(behavior=behavior), res(abstained=abstained)) is expected
