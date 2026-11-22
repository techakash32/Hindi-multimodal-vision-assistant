import numpy as np
import pytest

from visionassist import translate, tts


@pytest.fixture(autouse=True)
def silent_tts():
    """Never touch audio hardware in tests."""
    tts.set_muted(True)
    tts.set_speech_listener(None)
    yield
    tts.set_muted(True)
    tts.set_speech_listener(None)


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def fake_translator(monkeypatch):
    """Never hit the network: 'translate' by prefixing a Hindi word."""
    translate._CACHE.clear()
    monkeypatch.setattr(translate, "_cooldown_until", 0.0)
    monkeypatch.setattr(translate, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(translate, "_translate_once", lambda text: "अनुवाद: " + text)
