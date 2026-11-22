"""Synthetic adversarial pack: 20 harder variants of clean fixtures.

    python -m eval.generate_adversarial     # writes eval/fixtures/adv_*.jpg and eval/scenarios_adversarial.jsonl
    python -m eval.run_eval --scenarios eval/scenarios_adversarial.jsonl \
        --out reports/adversarial_raw.json --report reports/adversarial_report.md

Each case applies one synthetic degradation (motion blur with a line kernel, gamma / brightness
loss, sensor noise, JPEG blocking, rotation, occlusion) to a clean image whose text is known.
Ten are *mild*: the text is still readable, so the right behaviour is to answer with the original
text. Ten are *severe*: the text is gone or unreliable, so the right behaviour is to abstain. This
is the same degradation-severity method used on real data, but the images are still synthetic, and
the quality gates in ``quality.py`` were tuned on the base fixtures, so this set is only partly
out-of-sample for them. Severity is a judgement call; results are reported as they come, and no
case is moved between answer and abstain after seeing the outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import generate_dataset as g

ROOT = Path(__file__).resolve().parent
BASE_IDS = ["ocr_sign_001", "ocr_sign_002", "ocr_sign_003", "ocr_sign_004", "ocr_sign_005", "ocr_sign_006",
            "ocr_medicine_001", "ocr_medicine_002", "ocr_menu_001", "ocr_menu_002"]


def motion_blur(a, length, angle=0.0):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, m, (length, length))
    return cv2.filter2D(a, -1, k / max(k.sum(), 1e-6))


def jpeg(a, q):
    ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def bar(a, x1, y1, x2, y2):
    return g.occlude(a, x1, y1, x2, y2)


# (name, severity, base index, transform, description)
CASES = [
    ("motion_blur", "mild", 0, lambda a: motion_blur(a, 7), "horizontal motion blur, 7 px line"),
    ("motion_blur", "mild", 1, lambda a: motion_blur(a, 9, 15), "motion blur, 9 px line at 15 degrees"),
    ("low_light", "mild", 2, lambda a: g.darken(a, 0.55), "brightness x0.55"),
    ("low_light_noise", "mild", 3, lambda a: g.noise(g.darken(a, 0.45), 6, 1), "brightness x0.45 plus sensor noise 6"),
    ("occlusion_corner", "mild", 6, lambda a: bar(a, g.W - 70, 0, g.W, 40), "black patch in a corner, text untouched"),
    ("jpeg", "mild", 4, lambda a: jpeg(a, 12), "JPEG quality 12"),
    ("rotation", "mild", 5, lambda a: g.rotate(a, 5, (0, 0, 0)), "rotated 5 degrees"),
    ("noise", "mild", 7, lambda a: g.noise(a, 14, 2), "sensor noise sigma 14"),
    ("motion_blur", "mild", 8, lambda a: motion_blur(a, 7), "motion blur 7 px on a menu"),
    ("low_light", "mild", 9, lambda a: g.darken(a, 0.6), "brightness x0.6 on a menu"),
    ("motion_blur", "severe", 0, lambda a: motion_blur(a, 61), "motion blur 61 px: text smeared"),
    ("motion_blur", "severe", 1, lambda a: motion_blur(a, 81, 30), "motion blur 81 px at 30 degrees"),
    ("low_light", "severe", 2, lambda a: g.darken(a, 0.06), "brightness x0.06: almost black"),
    ("low_light_noise", "severe", 3, lambda a: g.noise(g.darken(a, 0.1), 20, 3), "brightness x0.1 plus heavy noise"),
    ("occlusion_text", "severe", 6, lambda a: bar(a, 0, 100, g.W, 380), "black band across all the text"),
    ("low_contrast", "severe", 4, lambda a: g.low_contrast(a, 0.06), "contrast reduced to 6 percent"),
    ("defocus", "severe", 5, lambda a: g.blur(a, 12), "gaussian blur sigma 12"),
    ("low_light", "severe", 7, lambda a: g.darken(a, 0.08), "brightness x0.08"),
    ("occlusion_text", "severe", 8, lambda a: bar(a, 0, 90, g.W, 400), "black band across the menu text"),
    ("blur_noise", "severe", 9, lambda a: g.noise(motion_blur(a, 71), 18, 4), "motion blur 71 px plus noise"),
]


def build() -> list[dict]:
    base = {r["id"]: r for r in (json.loads(x) for x in (ROOT / "scenarios.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    out = []
    for i, (name, sev, bi, fn, desc) in enumerate(CASES, 1):
        b = base[BASE_IDS[bi]]
        img = cv2.imread(str(ROOT.parent / b["input"]["image"]))
        path = g.save(fn(img), f"adv_{i:03d}.jpg")
        severe = sev == "severe"
        out.append({
            "id": f"adv_{i:03d}", "category": "negative_abstain" if severe else b["category"],
            "input": {"image": path, "text": None, "audio": None},
            "user_request": b["user_request"], "expected_modality": "ocr",
            "ground_truth": None if severe else b["ground_truth"],
            "expected_behavior": "abstain" if severe else "answer",
            "difficulty": "hard" if severe else "medium", "notes": f"{name} ({sev}): {desc}",
            "transform": name, "severity": sev, "base_id": b["id"],
        })
    return out


def main() -> None:
    rows = build()
    (ROOT / "scenarios_adversarial.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} adversarial scenarios and fixtures")


if __name__ == "__main__":
    main()
