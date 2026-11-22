import json
from pathlib import Path

import pytest

from eval import scoring as sc
from eval.run_eval import load_scenarios

SCEN = Path(__file__).resolve().parent / "scenarios.jsonl"


def test_cer_wer_basics():
    assert sc.cer("Paracetamol 500mg", "paracetamol 500mg.") == 0.0       # case and punctuation ignored
    assert sc.cer("abcd", "abxd") == pytest.approx(0.25)
    assert sc.cer("abc", "") == 1.0 and sc.cer("", "") == 0.0
    assert sc.wer("take one tablet", "take two tablet") == pytest.approx(1 / 3)
    assert sc.wer("a b", "") == 1.0


def test_builtin_edit_distance_matches_jiwer_when_installed():
    jiwer = pytest.importorskip("jiwer")
    for ref, hyp in [("metformin 500mg", "metformin s00m9"), ("keep left", "keep lef"), ("a b c d", "a c d e")]:
        assert sc.cer(ref, hyp) == pytest.approx(jiwer.cer(ref, hyp))
        assert sc.wer(ref, hyp) == pytest.approx(jiwer.wer(ref, hyp))


def test_keyword_judge_labels():
    q, truth = "Mere saamne kya hai?", "Baayin taraf ek person hai."
    assert sc.keyword_judge(q, "Baayin taraf ek person hai", truth) == "correct"
    assert sc.keyword_judge(q, "Daayin taraf ek person hai", truth) == "incorrect"        # wrong side
    assert sc.keyword_judge(q, "Saamne ek chair hai", truth) == "incorrect"
    assert sc.keyword_judge(q, "Baayin taraf ek person aur ek dog hai", truth) == "partially_correct"  # invented
    assert sc.keyword_judge("Kitne log hain?", "Ek person hai", "Do person hain.") == "partially_correct"
    assert sc.keyword_judge("Kitne log hain?", "Teen person hain", "Do person hain.") == "incorrect"   # wrong count
    assert sc.keyword_judge(q, "", truth) == "incorrect"


def test_llm_judge_caches_and_falls_back(tmp_path):
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return "partially_correct."

    j = sc.LLMJudge(tmp_path / "cache.json", fake, "m")
    assert j("q", "a", "t") == "partially_correct" and j("q", "a", "t") == "partially_correct"
    assert len(calls) == 1 and j.hits == 1
    j.save()
    again = sc.LLMJudge(tmp_path / "cache.json", lambda p: 1 / 0, "m")      # would crash if it were called
    assert again("q", "a", "t") == "partially_correct"
    broken = sc.LLMJudge(tmp_path / "other.json", lambda p: 1 / 0, "m")
    assert broken("Mere saamne kya hai?", "Saamne ek person hai", "Saamne ek person hai.") == "correct"
    assert broken.fallbacks == 1


def test_score_verdicts():
    s = {"category": "ocr_sign", "expected_behavior": "answer", "ground_truth": "EXIT", "user_request": "x"}
    ok = {"abstained": False, "raw_output": {"raw_text": "exit"}}
    bad = {"abstained": False, "raw_output": {"raw_text": "EXlT"}}
    held = {"abstained": True, "actual_modalities": [], "raw_output": {}}
    assert sc.score(s, ok, sc.keyword_judge)["verdict"] == "correct"
    assert sc.score(s, bad, sc.keyword_judge)["verdict"] == "incorrect"
    assert sc.score(s, held, sc.keyword_judge)["verdict"] == "abstained"
    n = {"category": "negative_abstain", "expected_behavior": "abstain", "expected_reason": "ocr_low"}
    assert sc.score(n, {"abstained": True, "reason": "ocr_low"}, sc.keyword_judge) == {"verdict": "correct", "reason_match": True}
    assert sc.score(n, {"abstained": False}, sc.keyword_judge)["verdict"] == "incorrect"


# ---------------------------------------------------------------- dataset
def test_dataset_has_the_required_shape():
    rows = load_scenarios(SCEN)
    counts = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
        assert {"id", "category", "input", "user_request", "expected_modality", "ground_truth",
                "expected_behavior", "difficulty", "notes"} <= set(r)
        assert set(r["input"]) == {"image", "text", "audio"}
        assert r["expected_behavior"] in ("answer", "abstain")
        if r["category"] == "negative_abstain":
            assert r["expected_behavior"] == "abstain"
        img = r["input"]["image"]
        if img:
            assert img.startswith("eval/fixtures/") and (SCEN.parent.parent / img).exists(), r["id"]
    assert counts == {"ocr_medicine": 12, "ocr_sign": 12, "ocr_menu": 8, "scene_description": 20,
                      "ambiguous_question": 10, "negative_abstain": 10, "voice_command": 8}


def test_no_ground_truth_for_abstain_scenarios():
    for r in load_scenarios(SCEN):
        if r["expected_behavior"] == "abstain":
            assert r["ground_truth"] is None, r["id"]


def test_raw_report_roundtrips_as_json(tmp_path):
    from eval import run_eval
    out = tmp_path / "raw.json"
    rc = run_eval.main(["--scenarios", str(SCEN), "--backend", "fake", "--out", str(out),
                        "--report", str(tmp_path / "r.md"), "--latency-summary", str(tmp_path / "none.md")])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["records"]) == 80
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    for needle in ("| modality | n | precision | recall | F1 | notes |", "**Abstention accuracy: ", "| Bucket | Count | Example ID |",
                   "| Stage | n | p50 (ms) | p95 (ms) |", "## Known limitations"):
        assert needle in md
    assert all(r["failure_bucket"] != "unclassified" for r in data["records"])
