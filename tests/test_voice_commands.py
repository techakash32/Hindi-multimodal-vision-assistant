import json
import threading
import time
import unicodedata

import pytest

from visionassist import cli, voice

# The five phrases on the dashboard buttons, as a user would say them.
FIVE = {
    "read": ["टेक्स्ट पढ़ें", "पढ़ें"],
    "describe": ["सीन बताएं", "बताएं"],
    "changed": ["क्या बदला", "बदला"],
    "repeat": ["दोबारा बोलें", "दोबारा"],
    "quit": ["आवाज़ बंद करें", "आवाज बंद करें", "आवाज़ बंद करो"],
}


@pytest.mark.parametrize("cmd,phrases", FIVE.items())
def test_each_button_phrase_maps_to_its_command(cmd, phrases):
    for p in phrases:
        assert voice.match_phrase(p) == cmd, p


def test_nukta_spelling_variants_are_equivalent():
    composed = "आवाज़"                    # आवाज़ with single-character ज़
    assert voice.match_phrase(composed + " बंद करें") == "quit"
    assert voice.match_phrase(unicodedata.normalize("NFD", composed) + " बंद करें") == "quit"
    assert voice.match_phrase("आवाज बंद करें") == "quit"


def test_closing_the_app_needs_the_whole_phrase():
    for w in ("बंद", "बंद करें", "आवाज़", "बाहर", "समाप्त", "quit", "exit"):
        assert voice.match_phrase(w) != "quit", w


def test_other_words_and_old_commands_do_nothing():
    for w in ("नमस्ते", "[unk]", "", "धन्यवाद", "कल", "बाहर", "वर्णन", "देखो", "दोहराओ", "पढ़ो", "read"):
        assert voice.match_phrase(w) is None, w


def test_small_slip_still_matches():
    assert voice.match_phrase("क्या बदल") == "changed"
    assert voice.match_phrase("टेक्स्ट पढ़े") == "read"


def test_grammar_offers_only_the_five_phrases_plus_unk():
    words = json.loads(voice.GRAMMAR)
    assert "[unk]" in words and "आवाज़ बंद करें" in words and "बंद" not in words
    assert set(voice.PHRASES) == {"read", "describe", "changed", "repeat", "quit"}


def test_voice_quit_ends_web_mode(monkeypatch):
    """Regression: saying the quit phrase set quit_event but the web server kept the program alive."""
    class Eng:
        quit_event = threading.Event()
        stopped = False
        def start(self): pass
        def stop(self): self.stopped = True

    eng = Eng()
    monkeypatch.setattr(cli, "build_assistant", lambda args: eng)
    import visionassist.web as web
    monkeypatch.setattr(web, "create_app", lambda *a, **k: object())
    monkeypatch.setattr(web, "serve", lambda *a: time.sleep(60))     # blocks like waitress
    args = cli.parser().parse_args(["run", "--ui", "web", "--no-browser"])
    threading.Timer(0.3, eng.quit_event.set).start()
    t0 = time.time()
    assert cli.cmd_run(args) == 0
    assert time.time() - t0 < 5 and eng.stopped
