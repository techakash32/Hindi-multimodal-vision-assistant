"""Failure taxonomy: every scenario that is not fully correct gets exactly one
bucket.

``classify(scenario, result)`` reads only structured fields that the pipeline
already records (see ``run_eval.run_scenario``):

    result["abstained"], result["reason"]
    result["route"]["modality"]        what the router chose (ocr | vlm | asr | none)
    result["actual_modalities"]        stages that really ran
    result["intent"]                   resolved voice intent
    result["over_budget"]              stages that exceeded their latency budget
    result["raw_output"]["raw_text"]   raw OCR text
    result["raw_output"]["detections"] {class: {...}} the detector's boxes
    result["raw_output"]["vlm"]        the VLM / template reply
    result["scores"]["judge"]          correct | partially_correct | incorrect

Decision order (the first rule that matches wins):

  Expected "abstain"
    1. the assistant answered                              -> failure_to_abstain

  Expected "answer"
    2. a stage exceeded its latency budget                  -> latency_timeout
    3. voice scenario with the wrong / no intent            -> asr_substitution
       (the transcript differs from what was said)             or router_misroute
    4. the router picked a modality other than the          -> router_misroute
       expected one (a router "unclear" is not a misroute:
       it is an abstention, rule 5)
    5. the assistant abstained
         OCR ran and found no text at all                   -> ocr_detection_fail
         otherwise                                          -> incorrect_abstention
    6. OCR scenario
         OCR returned no text                               -> ocr_detection_fail
         text found but CER > 0                             -> ocr_recognition_error
    7. scene scenario
         an expected object (or enough instances of it)
         is missing from the detections                     -> detection_miss
         objects were detected but the description is
         wrong or invents something                         -> vlm_hallucination
    8. anything else                                        -> unclassified
       (must stay at zero; extend the rules instead of ignoring it)

``is_failure`` says whether a scenario needs a bucket at all.
"""
from __future__ import annotations

from .scoring import OCR_CATEGORIES, cer

BUCKETS = (
    "detection_miss", "ocr_detection_fail", "ocr_recognition_error", "vlm_hallucination",
    "asr_substitution", "router_misroute", "latency_timeout", "failure_to_abstain",
    "incorrect_abstention", "unclassified",
)


def is_failure(scenario: dict, result: dict) -> bool:
    """True when the run is not fully correct."""
    if scenario["expected_behavior"] == "abstain":
        return not result.get("abstained", False)
    verdict = (result.get("scores") or {}).get("verdict")
    if verdict is not None:
        return verdict != "correct"
    return True


def _expected_objects(scenario: dict) -> dict[str, int]:
    need = {name: 1 for name in scenario.get("expected_objects") or []}
    need.update(scenario.get("expected_counts") or {})
    return need


def classify(scenario: dict, result: dict) -> str:
    modality = scenario["expected_modality"]

    # -- expected abstention ------------------------------------------------
    if scenario["expected_behavior"] == "abstain":
        return "unclassified" if result.get("abstained") else "failure_to_abstain"

    # -- expected answer -----------------------------------------------------
    if result.get("over_budget"):
        return "latency_timeout"

    route_modality = (result.get("route") or {}).get("modality", "none")
    if scenario["category"] == "voice_command" or modality == "asr":
        if result.get("intent") != scenario.get("ground_truth"):
            transcript_differs = (scenario["input"].get("text") or "") != scenario["user_request"]
            return "asr_substitution" if transcript_differs or scenario.get("simulated_asr") else "router_misroute"
        return "unclassified"

    if route_modality not in ("none", modality):
        return "router_misroute"

    raw = result.get("raw_output") or {}
    raw_text = (raw.get("raw_text") or "").strip()
    ocr_ran = "ocr" in result.get("actual_modalities", [])

    if result.get("abstained"):
        return "ocr_detection_fail" if ocr_ran and not raw_text else "incorrect_abstention"

    if scenario["category"] in OCR_CATEGORIES:
        if not raw_text:
            return "ocr_detection_fail"
        if cer(scenario["ground_truth"], raw_text) > 0:
            return "ocr_recognition_error"
        return "unclassified"

    if scenario["category"] == "scene_description":
        found = raw.get("detections") or {}
        for name, count in _expected_objects(scenario).items():
            # the detector keeps one box per class, so a count above 1 cannot be met
            if name not in found or count > 1:
                return "detection_miss"
        return "vlm_hallucination"

    return "unclassified"
