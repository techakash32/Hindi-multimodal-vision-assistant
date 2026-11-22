import cv2
import numpy as np

from visionassist.guardrails import AbstentionConfig
from visionassist.ocr import OcrText
from visionassist.pipeline import PipelineInput, run_once
from visionassist.strings import MESSAGES, MESSAGES_DISPLAY

CFG = AbstentionConfig()
IDENT = lambda t: t  # noqa: E731  (no translation: tests stay offline)


def sharp_frame():
    f = np.full((480, 640, 3), 235, np.uint8)
    cv2.putText(f, "EXIT 21", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 4, (20, 20, 20), 8)
    return f


def blurred_frame():
    return cv2.GaussianBlur(sharp_frame(), (0, 0), 12)


class Spy:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def __call__(self, *a):
        self.calls += 1
        return self.result


def vlm_data(conf=0.9, path="unclear"):
    return {"summary": "saamne ek chair hai", "obstacles": [], "path_status": path, "confidence": conf}


def go(frame, request, ocr=None, vlm=None, dets=None, cfg=CFG, text=None):
    return run_once(PipelineInput(frame=frame, text=text, detections=dets), request,
                    ocr_fn=ocr or Spy(OcrText("EXIT 21", 0.95)), vlm_fn=vlm or Spy(vlm_data()),
                    translate_fn=IDENT, abstention=cfg)


def test_read_answers_and_reports_per_stage_confidence():
    r = go(sharp_frame(), "padho")
    assert not r.abstained and r.status == "answer"
    assert r.modalities == ["ocr"] and r.raw["raw_text"] == "EXIT 21"
    assert r.confidences["ocr"] > 0.6
    assert {t.stage for t in r.traces} >= {"router", "quality", "ocr"}
    assert r.latency_ms >= sum(t.ms for t in r.traces) - 1e-6


def test_blurry_frame_abstains_before_ocr_is_called():
    ocr = Spy(OcrText("EXIT 21", 0.95))
    r = go(blurred_frame(), "padho", ocr=ocr)
    assert r.abstained and r.reason == "ocr_low" and ocr.calls == 0
    assert r.modalities == []
    assert r.answer_text == MESSAGES["abstain_ocr"] and r.display_text == MESSAGES_DISPLAY["abstain_ocr"]


def test_blurry_frame_abstains_before_vlm_is_called():
    vlm = Spy(vlm_data())
    r = go(blurred_frame(), "saamne kya hai", vlm=vlm)
    assert r.abstained and r.reason == "vlm_low" and vlm.calls == 0


def test_low_ocr_engine_confidence_abstains_after_ocr():
    r = go(sharp_frame(), "padho", ocr=Spy(OcrText("Hwgoogs jowe", 0.17)))
    assert r.abstained and r.reason == "ocr_low" and r.modalities == ["ocr"]


def test_low_vlm_self_reported_confidence_abstains():
    r = go(sharp_frame(), "saamne kya hai", vlm=Spy(vlm_data(conf=0.2)))
    assert r.abstained and r.reason == "vlm_low" and r.modalities[-1] == "vlm"


def test_missing_camera_abstains_without_any_call():
    ocr = Spy("x")
    r = go(None, "padho", ocr=ocr)
    assert r.abstained and r.reason == "missing_modality" and ocr.calls == 0
    assert r.answer_text == "कैमरा उपलब्ध नहीं है।"


def test_unclear_request_abstains_with_a_clarifying_question():
    ocr, vlm = Spy("x"), Spy(vlm_data())
    r = go(sharp_frame(), "Is it safe?", ocr=ocr, vlm=vlm)
    assert r.abstained and r.reason == "conflict" and ocr.calls == vlm.calls == 0
    assert r.display_text == "Jaankari mil nahi rahi, thoda aur batao."


def test_no_text_is_an_abstention_with_the_existing_message():
    r = go(sharp_frame(), "padho", ocr=Spy("No text found. Try pointing closer."))
    assert r.abstained and r.status == "no_text" and r.answer_text == MESSAGES["no_text"]


def test_ocr_error_is_not_an_abstention():
    r = go(sharp_frame(), "padho", ocr=Spy("OCR error: timeout"))
    assert r.status == "error" and not r.abstained and r.answer_text == MESSAGES["ocr_error"]


def test_vlm_claiming_clear_path_conflicts_with_a_close_detection():
    near_person = {"person": (0.9, (200, 50, 440, 470), 320.0, 0.30)}
    r = go(sharp_frame(), "saamne kya hai", vlm=Spy(vlm_data(path="none_seen")), dets=near_person)
    assert r.abstained and r.reason == "conflict"


def test_medium_confidence_answer_is_hedged():
    cfg = AbstentionConfig(ocr_tau=0.3, vlm_tau=0.3, aggregate_tau_low=0.3, aggregate_tau_high=0.95)
    r = go(sharp_frame(), "padho", ocr=Spy(OcrText("EXIT 21", 0.80)), cfg=cfg)
    assert r.hedged and r.answer_text.startswith(MESSAGES["hedge"])


def test_abstention_can_be_disabled():
    r = go(blurred_frame(), "padho", cfg=AbstentionConfig.disabled())
    assert not r.abstained and r.modalities == ["ocr"]


def test_voice_commands_resolve_without_perception():
    ocr = Spy("x")
    for word, intent in [("padho", "read"), ("बंद", "stop"), ("dohrao", "repeat"), ("badla", "change")]:
        r = go(None, word, ocr=ocr, text=word)
        assert (r.intent, r.status, r.modalities) == (intent, "control", ["asr"])
    assert ocr.calls == 0


def test_unrecognised_transcript_asks_for_clarification():
    r = go(None, "पड़ो", text="पड़ो")
    assert r.abstained and r.intent is None and r.reason == "conflict"


def test_result_is_json_serialisable():
    import json
    json.dumps(go(sharp_frame(), "padho").to_dict(), ensure_ascii=False)
