import json
from pathlib import Path

from . import generate_adversarial as adv
from .kfold_abstention import render, stratified_folds

ROOT = Path(__file__).resolve().parent.parent


def test_adversarial_pack_shape():
    rows = adv.build()
    assert len(rows) == 20 and len({r["id"] for r in rows}) == 20
    assert sum(r["severity"] == "mild" for r in rows) == 10
    for r in rows:
        severe = r["severity"] == "severe"
        assert (r["expected_behavior"] == "abstain") == severe
        assert (r["ground_truth"] is None) == severe          # never invent an answer for an abstain case
        assert (ROOT / r["input"]["image"]).exists()


def test_adversarial_file_matches_generator():
    on_disk = [json.loads(x) for x in (Path(__file__).parent / "scenarios_adversarial.jsonl").read_text().splitlines()]
    assert on_disk == adv.build()


def test_transforms_really_change_the_image():
    import cv2
    import numpy as np
    for r in adv.build():
        base = cv2.imread(str(ROOT / next(
            json.loads(x)["input"]["image"] for x in (Path(__file__).parent / "scenarios.jsonl").read_text().splitlines()
            if json.loads(x)["id"] == r["base_id"])))
        img = cv2.imread(str(ROOT / r["input"]["image"]))
        assert img.shape == base.shape and np.abs(img.astype(int) - base.astype(int)).mean() > 0.5, r["id"]


def test_folds_partition_every_row_once_and_spread_negatives():
    rows = [({"category": "negative_abstain" if i < 10 else "ocr_sign", "id": i}, {}) for i in range(40)]
    folds = stratified_folds(rows, 5)
    assert sorted(s["id"] for f in folds for s, _ in f) == list(range(40))
    assert [sum(s["category"] == "negative_abstain" for s, _ in f) for f in folds] == [2] * 5


def test_render_mentions_pooled_result():
    res = {"k": 2, "folds": [{"fold": 0, "infeasible": True}], "neg_ok": 3, "neg_n": 4, "sel_acc": (0.5, 0.1),
           "coverage": (0.9, 0.0), "base_sel_acc": (0.4, 0.1), "base_coverage": (1.0, 0.0)}
    assert "3/4" in render(res, "fake", 8) and "infeasible" in render(res, "fake", 8)
