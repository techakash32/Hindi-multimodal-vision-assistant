import threading
from pathlib import Path

import pytest

from visionassist import config, translate, tts
from visionassist.events import EventBus
from visionassist.metrics import Metrics, percentile
from visionassist.resilience import CircuitBreaker


def test_percentile():
    assert percentile([], 50) == 0.0
    assert percentile([10], 95) == 10
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 100], 95) == pytest.approx(95)


def test_metrics_summary_and_csv(tmp_path):
    m = Metrics(tmp_path / "m.csv")
    for ms in (100, 200, 300):
        m.record("ocr", "text", ms)
    m.record("detect", "person")
    s = m.summary()
    assert s["ocr"]["count"] == 3 and s["ocr"]["p50_ms"] == 200 and s["detect"]["p50_ms"] is None
    rows = (tmp_path / "m.csv").read_text().splitlines()
    assert rows[0] == "ts,event,detail,latency_ms" and len(rows) == 5


def test_metrics_without_path_is_memory_only():
    m = Metrics(None)
    m.record("x", "", 5)
    assert m.summary()["x"]["last_ms"] == 5


def test_metrics_unwritable_path_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    m = Metrics(blocker / "sub" / "m.csv")
    m.record("x", "", 1)
    assert m.summary()["x"]["count"] == 1


def test_event_bus_ids_and_since():
    bus = EventBus(maxlen=3)
    ids = [bus.publish("system", f"e{i}") for i in range(5)]
    assert ids == [1, 2, 3, 4, 5]
    assert [e["text"] for e in bus.since(0)] == ["e2", "e3", "e4"]
    assert [e["text"] for e in bus.since(4)] == ["e4"]


def test_event_bus_thread_safety():
    bus = EventBus(maxlen=1000)
    ts = [threading.Thread(target=lambda: [bus.publish("k", "x") for _ in range(100)]) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len({e["id"] for e in bus.since(0, limit=1000)}) == 800


def test_circuit_breaker_cycle(monkeypatch):
    import visionassist.resilience as r
    now = [1000.0]
    monkeypatch.setattr(r.time, "time", lambda: now[0])
    cb = CircuitBreaker("t", failure_threshold=2, cooldown_sec=10)
    cb.record_failure()
    assert cb.allow()
    cb.record_failure()
    assert not cb.allow()
    now[0] += 11
    assert cb.allow()
    cb.record_success()
    assert cb.total_failures == 2


def test_translate_caches_and_falls_back(monkeypatch):
    translate._CACHE.clear()
    calls = []
    monkeypatch.setattr(translate, "_translate_once", lambda t: calls.append(t) or "नमस्ते")
    assert translate.to_hindi("hello there") == "नमस्ते"
    assert translate.to_hindi("Hello there") == "नमस्ते" and len(calls) == 1

    def boom(t):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(translate, "_translate_once", boom)
    monkeypatch.setattr(translate.time, "sleep", lambda s: None)
    assert translate.to_hindi("something new") == "something new"


def test_translate_skips_hindi_and_tiny_text():
    assert translate.to_hindi("कुर्सी") == "कुर्सी" and translate.to_hindi("a") == "a"


def test_config_env_parsing(monkeypatch):
    monkeypatch.setenv("X_INT", "abc")
    monkeypatch.setenv("X_BOOL", "Yes")
    monkeypatch.setenv("X_FLOAT", "2.5")
    assert config._int("X_INT", 7) == 7
    assert config._bool("X_BOOL", False) is True and config._bool("X_MISSING", True) is True
    assert config._float("X_FLOAT", 1) == 2.5


def test_muted_speech_notifies_and_repeats():
    heard = []
    tts.set_speech_listener(lambda text, lang, display=None: heard.append((text, lang)))
    tts.speak("[warm] कुर्सी सामने", lang="hi")
    assert tts.repeat_last()
    assert heard == [("कुर्सी सामने", "hi")] * 2


def test_display_text_travels_with_speech_and_repeat():
    seen = []
    tts.set_speech_listener(lambda text, lang, display=None: seen.append((text, display)))
    tts.speak("कुर्सी सामने", lang="hi", display="Chair saamne")
    assert tts.repeat_last()
    assert seen == [("कुर्सी सामने", "Chair saamne")] * 2


def test_hindi_only_never_speaks_english():
    heard = []
    tts.set_speech_listener(lambda text, lang, display=None: heard.append((text, lang)))
    tts.speak("plain english", lang="en")
    tts.speak("plain english", lang="hi", fallback_text="plain english")
    assert heard == []


def test_english_fallback_still_available_when_hindi_only_is_off(monkeypatch):
    monkeypatch.setattr(config, "HINDI_ONLY", False)
    heard = []
    tts.set_speech_listener(lambda text, lang, display=None: heard.append((text, lang)))
    tts.speak("plain english", lang="hi")
    assert heard == [("plain english", "en")]


def test_missing_hindi_voice_means_silence_not_english(monkeypatch):
    monkeypatch.setitem(tts._voices, "hi", None)
    monkeypatch.setattr(tts._ready["hi"], "wait", lambda timeout=None: True)
    assert tts._pick_voice("कुर्सी", "hi", "chair") == (None, None)


def test_speech_queue_drops_oldest_not_newest(monkeypatch):
    import queue
    tts.set_muted(False)
    monkeypatch.setattr(tts, "init_tts", lambda: None)
    monkeypatch.setattr(tts, "_worker", lambda: threading.Event().wait(5))   # worker never drains
    q = queue.Queue(maxsize=2)
    monkeypatch.setattr(tts, "_queue", q)
    monkeypatch.setattr(tts, "_worker_thread", None)
    for word in ("एक", "दो", "तीन"):
        tts.speak(word)
    assert [q.get_nowait()[0] for _ in range(2)] == ["दो", "तीन"]


# ---- privacy mode, checksums, dashboard accessibility ----
def test_local_only_blocks_cloud_calls(monkeypatch):
    import numpy as np

    from visionassist import config, translate, vlm
    monkeypatch.setattr(config, "LOCAL_ONLY", True)
    assert translate.to_hindi("Hello there") == "Hello there"
    with pytest.raises(RuntimeError, match="LOCAL_ONLY"):
        vlm._ask_vlm(np.zeros((8, 8, 3), np.uint8), "p")


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    from visionassist import downloads
    asset = downloads.Asset("x", "http://example.invalid/f.bin", tmp_path / "f.bin")

    class R:
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield b"data"

    monkeypatch.setattr(downloads, "known_checksums", lambda: {asset.url: "0" * 64})
    with pytest.raises(IOError, match="checksum mismatch"):
        downloads.fetch(asset, get=lambda *a, **k: R())
    assert not (tmp_path / "f.bin").exists() and not (tmp_path / "f.bin.part").exists()


def test_dashboard_accessibility_basics():
    from visionassist.strings import UI
    html = (Path(downloads_root()) / "web/templates/index.html").read_text(encoding="utf-8")
    assert "<noscript>" in html and 'lang="hi-Latn"' in html
    assert 'aria-live="polite"' in html and 'role="alert"' in html and 'class="skip"' in html
    assert "privacy_local" in html and UI["privacy_local"] and UI["privacy_cloud"]


def _lum(hexcol):
    h = hexcol.lstrip("#")
    h = "".join(c * 2 for c in h) if len(h) == 3 else h
    ch = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    ch = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def test_every_css_colour_pair_meets_wcag_aaa():
    """The theme is pure black and white, so any grey would be a regression."""
    import re
    css = (Path(downloads_root()) / "web/static/app.css").read_text(encoding="utf-8")
    colours = {c.lower() for c in re.findall(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b", css)}
    colours = {"#" + c for c in colours}
    assert colours <= {"#fff", "#000", "#ffffff", "#000000"}, f"unexpected colours: {colours}"
    assert (_lum("#fff") + 0.05) / (_lum("#000") + 0.05) >= 7
    rules = "\n".join(ln for ln in css.splitlines() if not ln.lstrip().startswith("@keyframes"))
    assert "opacity" not in rules          # no faded text: contrast stays 21:1


def downloads_root():
    import visionassist
    return Path(visionassist.__file__).parent


def test_translate_rate_limit_pauses_instead_of_retrying(monkeypatch):
    calls = []

    def limited(text):
        calls.append(text)
        raise RuntimeError("You made too many requests to the server")
    monkeypatch.setattr(translate, "_translate_once", limited)
    assert translate.to_hindi("first sentence") == "first sentence"
    assert translate.to_hindi("second sentence") == "second sentence"
    assert calls == ["first sentence"]          # no retries, and no call during the cooldown
