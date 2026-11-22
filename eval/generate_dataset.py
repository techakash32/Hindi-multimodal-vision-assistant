"""Generate eval/scenarios.jsonl and the synthetic fixtures in eval/fixtures/.

    python -m eval.generate_dataset

Everything is rendered with PIL / OpenCV from a fixed seed, so the dataset is
reproducible and the ground truth is exact: it is the text that was drawn. The
images are *synthetic* (flat renders, not photographs); see
eval/fixtures/README.md for what that does and does not tell you.

For OCR scenarios ``ground_truth`` is the full text visible in the image, so CER
and WER are well defined. Occlusion scenarios list only the text left visible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FIX = ROOT / "fixtures"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
W, H = 640, 480
PERSON_BGR, CHAIR_BGR = (150, 120, 90), (80, 110, 60)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size)


SANS, BOLD, SERIF, SERIF_B, MONO = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf",
                                    "DejaVuSerif-Bold.ttf", "DejaVuSansMono.ttf")


# ------------------------------------------------------------ drawing
def canvas(bg=(245, 245, 240)) -> Image.Image:
    return Image.new("RGB", (W, H), bg)


def text_block(img, lines, face=SANS, size=44, fill=(20, 20, 20), top=None, left=40, gap=1.5):
    d = ImageDraw.Draw(img)
    f = font(face, size)
    total = len(lines) * size * gap
    y = (H - total) / 2 if top is None else top
    for line in lines:
        d.text((left, y), line, font=f, fill=fill)
        y += size * gap
    return img


def centered(img, lines, face=BOLD, size=52, fill=(255, 255, 255)):
    d = ImageDraw.Draw(img)
    f = font(face, size)
    y = (H - len(lines) * size * 1.4) / 2
    for line in lines:
        w = d.textlength(line, font=f)
        d.text(((W - w) / 2, y), line, font=f, fill=fill)
        y += size * 1.4
    return img


def handwritten(img, lines, size=46, fill=(30, 30, 90), seed=0):
    """Per-character jitter in position and rotation: a stand-in for
    handwriting (no handwriting font is available offline)."""
    rng = random.Random(seed)
    f = font("DejaVuSerif-Italic.ttf", size)
    y = (H - len(lines) * size * 1.6) / 2
    for line in lines:
        x = 40
        for ch in line:
            tile = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((size // 2, size // 3), ch, font=f, fill=fill + (255,))
            tile = tile.rotate(rng.uniform(-9, 9), resample=Image.BICUBIC)
            img.paste(tile, (int(x - size // 2), int(y + rng.uniform(-5, 5) - size // 3)), tile)
            x += ImageDraw.Draw(img).textlength(ch, font=f) + rng.uniform(0, 3)
        y += size * 1.6
    return img


def to_cv(img) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def blur(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), sigma)


def low_contrast(a, keep=0.25, centre=200):
    return np.clip(centre + (a.astype(np.float32) - centre) * keep, 0, 255).astype(np.uint8)


def darken(a, factor):
    return np.clip(a.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def noise(a, sigma, seed):
    n = np.random.default_rng(seed).normal(0, sigma, a.shape)
    return np.clip(a.astype(np.float32) + n, 0, 255).astype(np.uint8)


def rotate(a, deg, bg):
    m = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    return cv2.warpAffine(a, m, (W, H), borderValue=bg)


def occlude(a, x1, y1, x2, y2):
    a = a.copy()
    cv2.rectangle(a, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return a


def save(a, name: str) -> str:
    FIX.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIX / name), a, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return f"eval/fixtures/{name}"


# ------------------------------------------------------------ scenes
def scene(items, seed=0, dark=1.0):
    """Dark gradient room with solid-colour 'person' / 'chair' / 'dog' blocks.
    These colours are what the demo detector (visionassist.demo.BlobModel)
    recognises. They are NOT photographs and say nothing about real YOLO."""
    y = np.linspace(38, 62, H, dtype=np.uint8)[:, None]
    f = np.dstack([y + 14, y + 6, y]).repeat(W, axis=1).astype(np.uint8)
    cv2.line(f, (0, int(H * 0.72)), (W, int(H * 0.72)), (90, 80, 70), 2)
    for kind, cx, scale in items:
        cx = int(W * cx)
        if kind == "person":
            hw, top, bot = int(55 * scale), int(H * 0.3), int(H * 0.72)
            top = bot - int((bot - top) * scale)
            cv2.rectangle(f, (cx - hw, top), (cx + hw, bot), PERSON_BGR, -1)
            cv2.circle(f, (cx, top - 26), int(26 * scale), (170, 150, 130), -1)
        elif kind == "chair":
            hw = int(60 * scale)
            cv2.rectangle(f, (cx - hw, int(H * 0.72 - 130 * scale)), (cx + hw, int(H * 0.72)), CHAIR_BGR, -1)
        elif kind == "dog":      # brown ellipse the toy detector has no class for
            cv2.ellipse(f, (cx, int(H * 0.66)), (int(70 * scale), int(34 * scale)), 0, 0, 360, (40, 90, 140), -1)
        elif kind == "bus":
            cv2.rectangle(f, (cx - 150, 200), (cx + 150, 340), (30, 60, 170), -1)
    f = noise(f, 3, seed)
    return darken(f, dark) if dark != 1.0 else f


# ------------------------------------------------------------ scenarios
def build() -> list[dict]:
    rows: list[dict] = []
    counters: dict[str, int] = {}

    def add(cat, prefix, image, request, modality, truth, behavior="answer", diff="easy", notes="",
            text=None, **extra):
        counters[cat] = counters.get(cat, 0) + 1
        n = counters[cat]
        img_path = save(image, f"{prefix}_{n:03d}.jpg") if image is not None else None
        rows.append({
            "id": f"{cat}_{n:03d}", "category": cat,
            "input": {"image": img_path, "text": text, "audio": None},
            "user_request": request, "expected_modality": modality, "ground_truth": truth,
            "expected_behavior": behavior, "difficulty": diff, "notes": notes, **extra,
        })

    paper = (240, 238, 228)

    # ---- ocr_medicine (12)
    def med(lines, **kw):
        return text_block(canvas(paper), lines, **kw)

    add("ocr_medicine", "medicine", to_cv(med(["Paracetamol 500mg"], face=BOLD, size=50)),
        "Ye dawa ka naam kya hai?", "ocr", "Paracetamol 500mg", notes="clean label")
    add("ocr_medicine", "medicine", to_cv(med(["Azithromycin 250mg"], face=BOLD, size=48)),
        "Dawa ka naam batao", "ocr", "Azithromycin 250mg", notes="clean label")
    add("ocr_medicine", "medicine", to_cv(med(["Take 1 tablet", "twice daily after meals"], size=40)),
        "Isme dose kitni leni hai? padho", "ocr", "Take 1 tablet twice daily after meals", notes="dosage")
    add("ocr_medicine", "medicine", to_cv(med(["Store below 25C.", "Keep away from children."], size=40)),
        "Isme warning kya likhi hai?", "ocr", "Store below 25C. Keep away from children.", notes="warning text")
    add("ocr_medicine", "medicine", low_contrast(to_cv(med(["Cetirizine 10mg"], face=BOLD, size=50, fill=(90, 90, 90))), 0.5),
        "Ye dawa ka naam kya hai?", "ocr", "Cetirizine 10mg", diff="medium", notes="reduced contrast")
    add("ocr_medicine", "medicine", rotate(to_cv(med(["Metformin 500mg"], face=BOLD, size=52)), 10, (228, 238, 240)),
        "Dawa ka naam padho", "ocr", "Metformin 500mg", diff="medium", notes="rotated 10 degrees")
    add("ocr_medicine", "medicine", rotate(to_cv(med(["Amoxicillin 500mg"], face=BOLD, size=50)), 20, (228, 238, 240)),
        "Dawa ka naam padho", "ocr", "Amoxicillin 500mg", diff="hard", notes="rotated 20 degrees")
    add("ocr_medicine", "medicine", occlude(to_cv(med(["Ibuprofen 400mg", "Expiry 08/2027"], face=BOLD, size=48)), 0, 290, 640, 480),
        "Ye dawa ka naam kya hai?", "ocr", "Ibuprofen 400mg", diff="medium",
        notes="expiry line hidden under a black bar; ground truth lists only visible text")
    add("ocr_medicine", "medicine", noise(low_contrast(to_cv(med(["Omeprazole 20mg"], face=BOLD, size=48, fill=(70, 70, 70))), 0.45), 6, 1),
        "Ye dawa ka naam kya hai?", "ocr", "Omeprazole 20mg", diff="hard", notes="low contrast plus sensor noise")
    add("ocr_medicine", "medicine", to_cv(med(["Dolo 650", "Tablets IP"], face=MONO, size=46)),
        "Is dawa ka naam padho", "ocr", "Dolo 650 Tablets IP", notes="monospace label")
    add("ocr_medicine", "medicine", to_cv(med(["Vitamin D3 60000 IU"], face=SANS, size=24, top=230)),
        "Isme kya likha hai? padho", "ocr", "Vitamin D3 60000 IU", diff="hard", notes="small print")
    add("ocr_medicine", "medicine", to_cv(med(["Crocin Advance 500mg"], face=SERIF_B, size=46)),
        "Dawa ka naam batao", "ocr", "Crocin Advance 500mg", notes="serif label")

    # ---- ocr_sign (12)
    def sign(lines, bg=(40, 110, 60), fill=(255, 255, 255), size=56, face=BOLD):
        return centered(canvas(bg), lines, face=face, size=size, fill=fill)

    add("ocr_sign", "sign", to_cv(sign(["MG ROAD"])), "Is board par kya likha hai?", "ocr", "MG ROAD", notes="street sign")
    add("ocr_sign", "sign", to_cv(sign(["NO", "PARKING"], bg=(30, 40, 170))), "Sign padho", "ocr", "NO PARKING", notes="regulatory sign")
    add("ocr_sign", "sign", to_cv(sign(["Sharma Medical Store"], bg=(235, 235, 235), fill=(160, 20, 20), size=44)),
        "Ye dukaan ka naam kya hai?", "ocr", "Sharma Medical Store", notes="shop name")
    add("ocr_sign", "sign", to_cv(sign(["Bus 512", "to Andheri"], bg=(250, 200, 40), fill=(20, 20, 20))),
        "Ye kaunsi bus hai? padho", "ocr", "Bus 512 to Andheri", notes="bus number")
    add("ocr_sign", "sign", to_cv(sign(["EXIT"], bg=(20, 120, 40), size=90)), "Board par kya likha hai?", "ocr", "EXIT", notes="exit sign")
    add("ocr_sign", "sign", to_cv(sign(["Platform 3"], bg=(30, 50, 120), size=64)), "Platform number padho", "ocr", "Platform 3", notes="station sign")
    add("ocr_sign", "sign", blur(to_cv(sign(["STATE BANK OF INDIA"], bg=(30, 70, 160), size=44)), 2.2),
        "Is board par kya likha hai?", "ocr", "STATE BANK OF INDIA", diff="hard", notes="blurry (sigma 2.2)")
    add("ocr_sign", "sign", noise(darken(to_cv(sign(["HOTEL SUNRISE"], bg=(60, 60, 70), fill=(235, 220, 160), size=54)), 0.55), 8, 2),
        "Hotel ka naam padho", "ocr", "HOTEL SUNRISE", diff="hard", notes="night-time: dark and noisy")
    add("ocr_sign", "sign", noise(darken(to_cv(sign(["Bus 21A"], bg=(40, 40, 50), fill=(240, 240, 240), size=80)), 0.6), 7, 3),
        "Bus number kya hai?", "ocr", "Bus 21A", diff="hard", notes="night-time bus stop")
    add("ocr_sign", "sign", to_cv(sign(["Pharmacy", "24 Hours"], bg=(30, 130, 80), size=56)),
        "Dukaan ka board padho", "ocr", "Pharmacy 24 Hours", notes="shop board")
    add("ocr_sign", "sign", rotate(to_cv(sign(["Keep Left"], bg=(30, 60, 150), size=70)), 5, (30, 60, 150)),
        "Sign padho", "ocr", "Keep Left", diff="medium", notes="slight tilt")
    add("ocr_sign", "sign", to_cv(sign(["Gandhi Chowk", "200 m"], bg=(40, 110, 60), size=54)),
        "Is board par kya likha hai?", "ocr", "Gandhi Chowk 200 m", notes="distance marker")

    # ---- ocr_menu (8)
    def menu(lines, **kw):
        return text_block(canvas((250, 244, 230)), lines, face=SERIF, size=38, **kw)

    add("ocr_menu", "menu", to_cv(menu(["Paneer Tikka 220", "Dal Makhani 180", "Butter Naan 40"])),
        "Menu padho", "ocr", "Paneer Tikka 220 Dal Makhani 180 Butter Naan 40", notes="dish names and prices")
    add("ocr_menu", "menu", to_cv(menu(["Masala Dosa 90", "Idli Sambar 60"])),
        "Menu mein kya hai? padho", "ocr", "Masala Dosa 90 Idli Sambar 60", notes="south Indian items")
    add("ocr_menu", "menu", to_cv(handwritten(canvas((250, 244, 230)), ["Chai 15", "Samosa 20"], seed=4)),
        "Ye menu padho", "ocr", "Chai 15 Samosa 20", diff="hard", notes="handwriting-style (per-letter jitter), simulated")
    two = canvas((250, 244, 230))
    d = ImageDraw.Draw(two)
    f = font(SERIF, 30)
    for i, (a, b) in enumerate([("Veg Biryani", "Chicken Biryani"), ("Plain Dosa", "Egg Curry")]):
        d.text((30, 140 + i * 90), a, font=f, fill=(20, 20, 20))
        d.text((350, 140 + i * 90), b, font=f, fill=(20, 20, 20))
    add("ocr_menu", "menu", to_cv(two), "Menu padho", "ocr", "Veg Biryani Plain Dosa Chicken Biryani Egg Curry",
        diff="hard", notes="two columns; ground truth is column order (left column, then right)")
    add("ocr_menu", "menu", low_contrast(to_cv(menu(["Pav Bhaji 120", "Lassi 50"])), 0.4),
        "Menu padho", "ocr", "Pav Bhaji 120 Lassi 50", diff="medium", notes="faded print")
    add("ocr_menu", "menu", to_cv(menu(["Chole Bhature 110", "Rajma Chawal 100", "Gulab Jamun 40"])),
        "Isme kya kya milta hai? padho", "ocr", "Chole Bhature 110 Rajma Chawal 100 Gulab Jamun 40", notes="three items")
    add("ocr_menu", "menu", to_cv(handwritten(canvas((250, 244, 230)), ["Special: Kadhi 80"], seed=9)),
        "Aaj ka special padho", "ocr", "Special: Kadhi 80", diff="hard", notes="handwriting-style, simulated")
    add("ocr_menu", "menu", rotate(to_cv(menu(["Thali Rs 150"])), 7, (250, 244, 230)),
        "Thali ka rate padho", "ocr", "Thali Rs 150", diff="medium", notes="rotated 7 degrees")

    # ---- scene_description (20)
    S = scene
    sc = [
        (S([("person", 0.5, 1)], 1), "Mere saamne kya hai?", "Saamne ek person hai.", ["person"], {}, "easy", "person ahead"),
        (S([("person", 0.2, 1)], 2), "Mere saamne kya hai?", "Baayin taraf ek person hai.", ["person"], {}, "easy", "person on the left"),
        (S([("person", 0.8, 1)], 3), "Mere saamne kya hai?", "Daayin taraf ek person hai.", ["person"], {}, "easy", "person on the right"),
        (S([("chair", 0.5, 1)], 4), "Saamne kya hai? batao", "Saamne ek chair hai.", ["chair"], {}, "easy", "chair ahead"),
        (S([("chair", 0.82, 1)], 5), "Saamne kya hai? batao", "Daayin taraf ek chair hai.", ["chair"], {}, "easy", "chair on the right"),
        (S([("person", 0.25, 1), ("chair", 0.75, 1)], 6), "Describe the scene", "Baayin taraf person aur daayin taraf chair hai.", ["person", "chair"], {}, "medium", "two classes"),
        (S([("person", 0.5, 1.45)], 7), "Aage koi rukaawat hai?", "Saamne ek person hai, bahut paas.", ["person"], {}, "easy", "hazard: large person close ahead"),
        (S([("chair", 0.45, 1.5)], 8), "Aage koi rukaawat hai?", "Saamne ek chair hai, rukaawat hai.", ["chair"], {}, "medium", "hazard: chair near centre"),
        (S([], 9), "Aage koi rukaawat hai?", "Koi khaas cheez nahi dikhi.", [], {}, "medium", "hazard question on an empty room; the app must not claim it is safe"),
        (S([("person", 0.3, 1.2), ("chair", 0.7, 1)], 10), "Raaste mein kya kya hai?", "Baayin taraf person aur daayin taraf chair hai.", ["person", "chair"], {}, "medium", "hazards on both sides"),
        (S([("person", 0.3, 1), ("person", 0.7, 1)], 11), "Kitne log hain?", "Do person hain.", ["person"], {"person": 2}, "hard", "count: two people (the detector keeps one box per class)"),
        (S([("chair", 0.25, 1), ("chair", 0.5, 1), ("chair", 0.78, 1)], 12), "Kitni chair hain?", "Teen chair hain.", ["chair"], {"chair": 3}, "hard", "count: three chairs"),
        (S([("person", 0.5, 1)], 13), "Kitne log hain?", "Ek person hai.", ["person"], {"person": 1}, "easy", "count: one person"),
        (S([("chair", 0.2, 1)], 14), "Chair kis taraf hai?", "Chair baayin taraf hai.", ["chair"], {}, "easy", "spatial: chair left"),
        (S([("person", 0.85, 1)], 15), "Person kis taraf hai?", "Person daayin taraf hai.", ["person"], {}, "easy", "spatial: person right"),
        (S([("person", 0.5, 1.4), ("chair", 0.8, 0.8)], 16), "Kaun zyada paas hai, person ya chair?", "Person zyada paas hai.", ["person", "chair"], {}, "hard", "spatial: nearer object"),
        (S([("chair", 0.3, 1), ("person", 0.7, 1)], 17), "Chair ke daayin taraf kya hai?", "Chair ke daayin taraf person hai.", ["person", "chair"], {}, "hard", "spatial: relation between objects"),
        (S([("dog", 0.5, 1)], 18), "Saamne kutta hai kya?", "Saamne ek dog hai.", ["dog"], {}, "medium", "dog: no class for it in the demo detector"),
        (S([("bus", 0.5, 1)], 19), "Saamne bus hai kya?", "Saamne ek bus hai.", ["bus"], {}, "medium", "bus: no class for it in the demo detector"),
        (S([("person", 0.5, 1), ("dog", 0.2, 1)], 20), "Mere aaspaas kya hai?", "Saamne person aur baayin taraf dog hai.", ["person", "dog"], {}, "hard", "mixed: one detectable, one not"),
    ]
    for img, req, gt, objs, counts, diff, notes in sc:
        add("scene_description", "scene", img, req, "vlm", gt, diff=diff, notes=notes,
            expected_objects=objs, expected_counts=counts)

    # ---- ambiguous_question (10): the right behaviour is to ask, not guess
    amb = [
        ("What's that?", scene([("person", 0.5, 1)], 31)),
        ("Yeh kya hai?", to_cv(med(["Paracetamol 500mg"], face=BOLD, size=50))),
        ("Is it safe?", scene([("chair", 0.5, 1)], 32)),
        ("Kya yeh safe hai?", to_cv(med(["Cetirizine 10mg"], face=BOLD, size=50))),
        ("Is it safe to cross?", scene([("person", 0.4, 1)], 33)),
        ("Ye theek hai?", to_cv(sign(["EXIT"], size=90))),
        ("What is this thing?", scene([("chair", 0.3, 1)], 34)),
        ("Kya ye sahi hai?", to_cv(med(["Azithromycin 250mg"], face=BOLD, size=48))),
        ("Woh kya hai udhar?", scene([("person", 0.8, 1)], 35)),
        ("Can I eat this?", to_cv(menu(["Paneer Tikka 220", "Dal Makhani 180"]))),
    ]
    for req, img in amb:
        add("ambiguous_question", "ambiguous", img, req, "none", None, behavior="abstain",
            diff="medium", notes="deictic or safety question with no usable target", expected_reason="conflict")

    # ---- negative_abstain (10)
    neg = [
        (blur(to_cv(med(["Paracetamol 500mg"], face=BOLD, size=50)), 9), "Ye dawa ka naam kya hai?", "ocr_low", "heavy blur (sigma 9)"),
        (low_contrast(to_cv(sign(["MG ROAD"], bg=(200, 200, 200), fill=(170, 170, 170))), 0.4), "Is board par kya likha hai?", "ocr_low", "text almost the same grey as the board"),
        (occlude(to_cv(med(["Metformin 500mg"], face=BOLD, size=50)), 0, 150, 640, 330), "Ye dawa ka naam kya hai?", "ocr_low", "the whole text line is under a black bar"),
        (darken(scene([("person", 0.5, 1)], 41), 0.25), "Mere saamne kya hai?", "vlm_low", "almost black frame"),
        (blur(scene([("person", 0.5, 1), ("chair", 0.8, 1)], 42), 14), "Mere saamne kya hai?", "vlm_low", "heavy blur on a scene"),
        (None, "Padho", "missing_modality", "no camera frame at all"),
        (None, "Mere saamne kya hai?", "missing_modality", "no camera frame at all"),
        (scene([("chair", 0.5, 1)], 43), "Padho aur batao saamne kya hai?", "conflict", "asks for reading and describing at once"),
        (to_cv(med(["Ye text kya hai?"], size=44)), "Is board par kya likha hai aur saamne kaun hai?", "conflict", "read and describe cues together"),
        (cv2.rotate(to_cv(med(["Paracetamol 500mg"], face=BOLD, size=50)), cv2.ROTATE_180), "Ye dawa ka naam kya hai?", "ocr_low", "sharp image but text upside down"),
    ]
    for img, req, reason, notes in neg:
        mod = "none" if reason in ("conflict",) else ("ocr" if "padho" in req.lower() or "naam" in req or "likha" in req else "vlm")
        add("negative_abstain", "negative", img, req, mod, None, behavior="abstain", diff="hard",
            notes=notes, expected_reason=reason)

    # ---- voice_command (8): input.text is the ASR hypothesis (no audio is processed)
    voice = [
        ("padho", "padho", "read", False, "typed Hinglish command"),
        ("batao", "batao", "describe", False, "typed Hinglish command"),
        ("badla", "badla", "change", False, "typed Hinglish command"),
        ("dohrao", "dohrao", "repeat", False, "typed Hinglish command"),
        ("पढ़ो", "पढ़ो", "read", False, "Devanagari, as the Hindi Vosk model emits it"),
        ("बंद", "बंद", "stop", False, "Devanagari quit word"),
        ("padho", "पड़ो", "read", True, "SIMULATED mis-recognition: the hypothesis is not a command word"),
        ("dohrao", "दोहरा", "repeat", True, "SIMULATED mis-recognition: truncated word"),
    ]
    for said, heard, intent, sim, notes in voice:
        add("voice_command", "voice", None, said, "asr", intent, diff="hard" if sim else "easy",
            notes=notes, text=heard, simulated_asr=sim)

    return rows


def main() -> None:
    rows = build()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    want = {"ocr_medicine": 12, "ocr_sign": 12, "ocr_menu": 8, "scene_description": 20,
            "ambiguous_question": 10, "negative_abstain": 10, "voice_command": 8}
    assert counts == want and len(rows) == 80, counts
    with open(ROOT / "scenarios.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} scenarios and {len(list(FIX.glob('*.jpg')))} fixtures")


if __name__ == "__main__":
    main()
