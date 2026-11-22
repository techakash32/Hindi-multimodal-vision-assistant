"""Hinglish for the screen: Hindi written in Roman letters, with everyday
English words kept as English ("person, bahut paas, saamne").

Spoken audio stays Devanagari Hindi, because the Hindi voice cannot read
Roman text. Fixed messages have hand-written Hinglish (see strings.py);
this module converts *dynamic* Hindi (scene descriptions, text read from
signs) with a rule-based transliterator.
"""
from __future__ import annotations

import re

from .hindi_text import CLASS_HI

# How each detected object is named on screen. Common English words stay English.
CLASS_HING = {
    "person": "person", "bicycle": "cycle", "car": "car", "motorcycle": "bike", "airplane": "plane",
    "bus": "bus", "train": "train", "truck": "truck", "boat": "boat", "traffic light": "traffic light",
    "fire hydrant": "fire hydrant", "stop sign": "stop sign", "parking meter": "parking meter",
    "bench": "bench", "bird": "chidiya", "cat": "billi", "dog": "kutta", "horse": "ghoda",
    "sheep": "bhed", "cow": "gaay", "elephant": "haathi", "bear": "bhaalu", "zebra": "zebra",
    "giraffe": "giraffe", "backpack": "bag", "umbrella": "chhaata", "handbag": "handbag",
    "tie": "tie", "suitcase": "suitcase", "frisbee": "frisbee", "skis": "skis",
    "snowboard": "snowboard", "sports ball": "gend", "kite": "patang", "baseball bat": "bat",
    "baseball glove": "glove", "skateboard": "skateboard", "surfboard": "surfboard",
    "tennis racket": "racket", "bottle": "bottle", "wine glass": "glass", "cup": "cup",
    "fork": "kaanta", "knife": "chaaku", "spoon": "chammach", "bowl": "katora", "banana": "kela",
    "apple": "seb", "sandwich": "sandwich", "orange": "santra", "broccoli": "broccoli",
    "carrot": "gaajar", "hot dog": "hot dog", "pizza": "pizza", "donut": "donut", "cake": "cake",
    "chair": "chair", "couch": "sofa", "potted plant": "plant", "bed": "bed",
    "dining table": "table", "table": "table", "door": "darwaza", "toilet": "toilet", "tv": "TV",
    "laptop": "laptop", "mouse": "mouse", "remote": "remote", "keyboard": "keyboard",
    "cell phone": "phone", "microwave": "microwave", "oven": "oven", "toaster": "toaster",
    "sink": "sink", "refrigerator": "fridge", "book": "kitaab", "clock": "ghadi", "vase": "vase",
    "scissors": "kainchi", "teddy bear": "teddy bear", "hair drier": "hair dryer",
    "toothbrush": "toothbrush",
}

# Hindi word -> Hinglish, for text that arrives in Devanagari.
_FIXED = {
    "बाईं ओर": "baayin taraf", "दाईं ओर": "daayin taraf", "बाएँ": "baayein", "दाएँ": "daayein",
    "है": "hai", "हैं": "hain", "में": "mein", "नहीं": "nahi", "को": "ko", "से": "se", "का": "ka",
    "की": "ki", "के": "ke", "और": "aur", "यह": "yeh", "वह": "woh", "कि": "ki", "तो": "to",
    "भी": "bhi", "पर": "par", "ओर": "or", "एक": "ek", "कोई": "koi", "कृपया": "please",
    "आपके": "aapke", "आप": "aap", "सामने": "saamne", "दाईं": "daayin", "बाईं": "baayin",
    "बहुत": "bahut", "पास": "paas", "दूर": "door", "अभी": "abhi", "दरवाज़ा": "darwaza",
    "रुकावट": "rukaawat", "रास्ता": "raasta", "साफ़": "saaf", "दिख": "dikh", "रहा": "raha",
    "रही": "rahi", "रहे": "rahe", "हो": "ho", "था": "tha", "थी": "thi", "थे": "the", "ही": "hi",
    "या": "ya", "लगें": "lagein", "कतार": "qatar", "दुकान": "dukaan", "सुबह": "subah",
    "खुलती": "khulti", "बजे": "baje", "कमरे": "kamre", "कमरा": "kamra", "ओर्": "or",
    "तैयार": "taiyaar", "प्रतीक्षा": "intezaar", "बदलाव": "badlaav", "बदला": "badla",
    "नया": "naya", "हट": "hat", "गया": "gaya", "गई": "gayi", "दृश्य": "scene", "टेक्स्ट": "text",
    "कैमरा": "camera", "सीन": "scene", "डिस्क्रिप्शन": "description", "असिस्टेंट": "assistant",
    "सहायक": "assistant", "लिखावट": "text", "व्यक्ति": "person", "ट्राई": "try", "पॉसिबल": "possible",
    "प्लीज़": "please", "ट्रांसलेशन": "translation", "डिस्क्राइब": "describe", "ऊपर": "upar",
    "पासवर्ड": "password", "इमरजेंसी": "emergency", "गेट": "gate", "स्कूल": "school",
    "हॉस्पिटल": "hospital", "स्टेशन": "station", "टिकट": "ticket", "एंट्री": "entry", "एग्ज़िट": "exit",
    "वैसा": "waisa", "दवाई": "dawaai", "दवाखाना": "dawaakhaana", "पहली": "pehli", "पहले": "pehle",
    "नंबर": "number", "बैंक": "bank", "लिफ्ट": "lift", "पार्किंग": "parking", "लाइन": "line",
    "कॉफ़ी": "coffee", "कॉफी": "coffee", "डॉक्टर": "doctor", "स्टॉक": "stock", "सेल": "sale",
    "स्टोर": "store", "स्पेशल": "special", "एक्सपायरी": "expiry", "रेट": "rate", "बटर": "butter",
    "सलाद": "salad", "रायता": "raita", "मीटर": "meter", "मिनट": "minute", "मिलीग्राम": "milligram",
    "शौचालय": "toilet", "मेज़": "table", "बिस्तर": "bed", "घी": "ghee", "बेंच": "bench",
    "साइकिल": "cycle", "बस": "bus", "ट्रेन": "train", "बैग": "bag", "फ़ोन": "phone", "लाइट": "light",
    "रुपये": "rupaye", "रेलवे": "railway", "प्लेटफ़ॉर्म": "platform", "दोपहर": "dopahar", "पहचान": "pehchaan", "फर्श": "farsh",
    "नीचे": "neeche", "साथ": "saath", "लिए": "liye", "बारे": "baare", "बाद": "baad",
}

_DEV = "ऀ-ॿ"
_PHRASES = {}
for _k, _v in CLASS_HI.items():
    _PHRASES[_v] = CLASS_HING.get(_k, _k)
_PHRASES.update(_FIXED)
_PHRASE_RE = re.compile(
    r"(?<![" + _DEV + r"])(" + "|".join(re.escape(p) for p in sorted(_PHRASES, key=len, reverse=True)) + r")(?![" + _DEV + r"])"
)

_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
    "ञ": "ny", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
    "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
    "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
    "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "d", "ढ़": "dh", "फ़": "f",
}
_NUKTA = {"क": "क़", "ख": "ख़", "ग": "ग़", "ज": "ज़", "ड": "ड़", "ढ": "ढ़", "फ": "फ़"}
_INDEP = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e",
          "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
_MATRA = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai",
          "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e"}
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
HALANT, NUKTA, ANUSVARA, CANDRA, VISARGA = "्", "़", "ं", "ँ", "ः"


def _units(word: str):
    """Split a Devanagari word into [kind, roman, vowel] units.
    kind: 'V' independent vowel, 'C' consonant. vowel: None = inherent a,
    '' = halant (none), otherwise the matra's Roman form."""
    units, i = [], 0
    while i < len(word):
        ch = word[i]
        if ch in _INDEP:
            units.append(["V", _INDEP[ch], ""])
        elif ch in _CONS:
            base = ch
            if i + 1 < len(word) and word[i + 1] == NUKTA and base in _NUKTA:
                base, i = _NUKTA[base], i + 1
            units.append(["C", _CONS[base], None])
        elif ch in _MATRA and units and units[-1][0] == "C":
            units[-1][2] = _MATRA[ch]
        elif ch == HALANT and units and units[-1][0] == "C":
            units[-1][2] = ""
        elif ch in (ANUSVARA, CANDRA) and units:
            units[-1].append("n")
        elif ch == VISARGA and units:
            units[-1].append("h")
        i += 1
    return units


_LABIAL = set("पफबभम")


def _pronounced(unit, keep: bool) -> bool:
    """Does this syllable carry a sounded vowel? (A halant syllable does not.)"""
    if unit[0] == "V" or unit[2]:
        return True
    return unit[2] is None and keep


def _translit_word(word: str) -> str:
    """Devanagari word -> Roman. Schwa deletion follows the usual Hindi rule: a
    consonant's inherent 'a' is dropped when the syllable before it is
    pronounced and the syllable after it is pronounced too (kamra, not kamara);
    the final consonant never keeps it (kitaab, not kitaaba)."""
    units = _units(word)
    n = len(units)
    keep = [True] * n                       # does unit i pronounce its vowel?
    for i in range(n - 1, -1, -1):          # right to left, so "next" is already decided
        u = units[i]
        if u[0] != "C" or u[2] is not None:
            continue
        nxt_pron = i + 1 < n and _pronounced(units[i + 1], keep[i + 1])
        prev_pron = i > 0 and _pronounced(units[i - 1], keep[i - 1])
        if i == n - 1 and len(u) == 3:
            keep[i] = False
        elif i > 0 and prev_pron and nxt_pron:
            keep[i] = False
    out = []
    for idx, u in enumerate(units):
        kind, roman, vowel = u[0], u[1], u[2]
        tail = "".join(u[3:])
        last = idx == n - 1
        if tail == "n" and idx + 1 < n and units[idx + 1][0] == "C" and _unit_src(units[idx + 1]) in _LABIAL:
            tail = "m"
        if kind == "V":
            if last and idx > 0 and roman in ("aa", "ee"):   # hua / hui / rasoi
                roman = {"aa": "a", "ee": "i"}[roman]
            if last and roman == "e" and tail == "n":    # hilaayein, not hilaaen
                prev = "".join(out)
                roman, tail = ("yein" if prev and prev[-1] in "aeiou" else "ein"), ""
            out.append(roman + tail)
            continue
        if vowel is None:
            vowel_out = "a" if keep[idx] else ""
        elif vowel == "":
            vowel_out = ""
        else:
            vowel_out = vowel
        if last and vowel_out == "aa":                  # kamraa -> kamra
            vowel_out = "a"
        elif last and vowel_out == "ee":                # paani, not paanee
            vowel_out = "i"
        elif last and vowel_out == "oo" and idx > 0:    # chaalu, not chaaloo
            vowel_out = "u"
        if last and tail == "n":                        # karein, hain
            vowel_out = {"e": "ei", "ai": "ai"}.get(vowel_out, vowel_out)
        out.append(roman + vowel_out + tail)
    word_out = "".join(out)
    word_out = word_out.replace("chchh", "chh").replace("jny", "gy")
    if word_out.endswith("ie"):                         # bataaie -> bataaiye
        word_out = word_out[:-1] + "ye"
    return word_out


def _unit_src(unit) -> str:
    """The Devanagari letter a consonant unit came from (for the m/n choice)."""
    for dev, rom in _CONS.items():
        if rom == unit[1] and dev in _LABIAL:
            return dev
    return ""


def _capitalise(text: str) -> str:
    """Capital letter at the start and after each full stop."""
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def to_hinglish(text: str) -> str:
    """Hindi (Devanagari) -> natural Hinglish. Latin text and numbers are kept."""
    if not text:
        return text
    text = text.translate(_DIGITS).replace("।", ".").replace("॥", ".")
    text = _PHRASE_RE.sub(lambda m: _PHRASES[m.group(1)], text)
    text = re.sub("[" + _DEV + "]+", lambda m: _translit_word(m.group(0)), text)
    text = re.sub(r"\s+([.,;!?])", r"\1", text).strip()
    return _capitalise(text)


def describe_hinglish(obj, frame_shape) -> str:
    """Hinglish version of an alert: 'person, bahut paas, saamne'."""
    name, _conf, _xyxy, xc, area = obj
    w = frame_shape[1]
    direction = "aapke baayin taraf" if xc < w * 0.33 else "aapke daayin taraf" if xc > w * 0.66 else "saamne"
    prox = "bahut paas" if area > 0.15 else "paas" if area > 0.05 else ""
    return _capitalise(", ".join(p for p in (CLASS_HING.get(name, name), prox, direction) if p))
