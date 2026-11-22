import json
import re

from visionassist import config as cfg
from visionassist import downloads, health, strings, voice
from visionassist.engine import COMMANDS
from visionassist.hindi_text import CLASS_HI
from visionassist.hinglish import CLASS_HING, describe_hinglish, to_hinglish

LATIN = re.compile("[A-Za-z]")
DEVANAGARI = re.compile("[\u0900-\u097f]")


def _clean(text):
    return re.sub(r"\{[a-z]+\}", "", text)


def test_spoken_strings_are_devanagari_without_latin_letters():
    for name, table in {"MESSAGES": strings.MESSAGES, "PATH": strings.PATH}.items():
        for key, text in table.items():
            assert DEVANAGARI.search(text) and not LATIN.search(text), (name, key, text)


def test_screen_strings_are_roman_hinglish_never_devanagari():
    tables = {"MESSAGES_DISPLAY": strings.MESSAGES_DISPLAY, "PATH_DISPLAY": strings.PATH_DISPLAY,
              "BUSY": strings.BUSY, "EVENTS": strings.EVENTS, "COMMAND_NAMES": strings.COMMAND_NAMES,
              "SOURCES": strings.SOURCES, "UI": strings.UI}
    for name, table in tables.items():
        for key, text in table.items():
            assert text.strip() and not DEVANAGARI.search(_clean(text)), (name, key, text)


def test_every_spoken_message_has_a_screen_version():
    assert set(strings.MESSAGES) == set(strings.MESSAGES_DISPLAY)
    assert set(strings.PATH) == set(strings.PATH_DISPLAY)


def test_every_command_has_a_screen_name():
    assert set(strings.COMMAND_NAMES) == set(COMMANDS)


def test_transliteration_samples():
    cases = {
        "कुर्सी": "Chair",                                       # known object: kept as English
        "आपके": "Aapke", "सामने": "Saamne", "बहुत पास": "Bahut paas", "नहीं": "Nahi",
        "दरवाज़ा खुला है": "Darwaza khula hai", "बच्चे खेल रहे हैं": "Bachche khel rahe hain",
        "दवाई": "Dawaai",
        "Wi-Fi का पासवर्ड ७८९ है": "Wi-Fi ka password 789 hai",
        "कमरे में सामने व्यक्ति, कुर्सी है।": "Kamre mein saamne person, chair hai.",
        "सब ठीक है। बाईं ओर दुकान है।": "Sab theek hai. Baayin taraf dukaan hai.",
    }
    for src, want in cases.items():
        assert to_hinglish(src) == want, (src, to_hinglish(src))


def test_transliteration_keeps_latin_and_handles_empty():
    assert to_hinglish("") == "" and to_hinglish("EXIT 5") == "EXIT 5"


def test_no_devanagari_survives_transliteration():
    for text in ("सीढ़ियाँ ऊपर जा रही हैं", "ट्रैफ़िक लाइट लाल है", "कृपया शांति बनाए रखें", "अस्पताल का रास्ता"):
        out = to_hinglish(text)
        assert out and not DEVANAGARI.search(out), (text, out)


def test_every_object_class_has_a_hinglish_name():
    assert set(CLASS_HI) <= set(CLASS_HING)


def test_describe_hinglish():
    shape = (480, 640, 3)
    assert describe_hinglish(("person", .9, None, 320, .2), shape) == "Person, bahut paas, saamne"
    assert describe_hinglish(("dog", .9, None, 50, .01), shape) == "Kutta, aapke baayin taraf"
    assert describe_hinglish(("car", .9, None, 600, .08), shape) == "Car, paas, aapke daayin taraf"


def test_voice_commands_cover_every_spoken_action_in_hindi():
    targets = set(voice.COMMANDS.values())
    assert targets == {"read", "describe", "changed", "repeat", "quit"}
    for cmd in targets:
        assert any(DEVANAGARI.search(w) for w, c in voice.COMMANDS.items() if c == cmd), cmd
    assert targets <= set(COMMANDS)


def test_grammar_is_valid_json_with_real_devanagari():
    words = json.loads(voice.GRAMMAR)
    assert "आवाज़ बंद करें" in words and "आवाज बंद करें" in words and "[unk]" in words and "\\u" not in voice.GRAMMAR


def test_health_requires_hindi_voice_not_english(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "YOLO_MODEL", tmp_path / "y.pt")
    monkeypatch.setattr(cfg, "HINDI_VOICE_MODEL", tmp_path / "hi.onnx")
    monkeypatch.setattr(cfg, "VOICE_MODEL", tmp_path / "en.onnx")
    checks = {c.name: c for c in health.run_checks()}
    assert checks["Hindi voice"].status == "fail" and checks["Hindi voice"].required
    assert "English voice" not in checks


def test_default_downloads_are_hindi():
    groups = downloads.assets()
    assert all(k in groups for k in ["yolo", "hi", "vosk-hi", "en", "vosk"])
    assert "vosk-model-small-hi" in groups["vosk-hi"][0].url
    assert cfg.VOSK_MODEL_PATH.endswith("vosk-model-small-hi-0.22")
