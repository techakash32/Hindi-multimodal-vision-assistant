import time

import pytest

from visionassist.demo import make_demo_detector
from visionassist.engine import Assistant
from visionassist.sources import SyntheticSource
from visionassist.strings import EVENTS, MESSAGES, MESSAGES_DISPLAY
from visionassist.web import create_app


def wait_for(cond, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def engine():
    vlm_calls = []

    def vlm(frame, dets, shape):
        vlm_calls.append(1)
        return {"summary": "A hall.", "obstacles": ["door"], "path_status": "obstacle_ahead", "structured": True}

    a = Assistant(make_demo_detector(), SyntheticSource(fps=60), ocr_fn=lambda f: "EXIT", vlm_fn=vlm, demo=True)
    a.vlm_calls = vlm_calls
    a.start()
    assert wait_for(lambda: a.frame_id > 5)
    yield a
    a.stop()


def test_frames_flow_and_jpeg(engine):
    assert engine.jpeg()[:2] == b"\xff\xd8"
    assert engine.snapshot()["camera_ok"]


def test_demo_detector_announces_person(engine):
    assert wait_for(lambda: any(e["kind"] == "detect" for e in engine.events.since(0)))
    assert "person" in {d["name"] for d in engine.snapshot()["detections"]}


def test_read_command_speaks_result_and_records_latency(engine):
    assert engine.dispatch("read")
    assert wait_for(lambda: any(e["kind"] == "ocr" for e in engine.events.since(0)))
    assert wait_for(lambda: engine.snapshot()["busy"] is None)
    assert engine.snapshot()["metrics"]["ocr"]["count"] == 1
    assert engine.snapshot()["last_text"] == "अनुवाद: EXIT"      # translated before speaking


def test_second_request_while_busy_is_rejected(monkeypatch):
    a = Assistant(make_demo_detector(), SyntheticSource(fps=60), ocr_fn=lambda f: (time.sleep(0.5), "slow")[1], demo=True)
    a.start()
    try:
        assert wait_for(lambda: a.frame_id > 3)
        assert a.dispatch("read") and not a.dispatch("read")
        assert wait_for(lambda: a.snapshot()["busy"] is None)
    finally:
        a.stop()


def test_describe_then_unchanged_skips_network(engine):
    engine.dispatch("describe")
    assert wait_for(lambda: len(engine.vlm_calls) == 1 and engine.snapshot()["busy"] is None)
    engine.dispatch("describe")
    time.sleep(0.3)
    assert len(engine.vlm_calls) == 1


def test_changed_without_description_prompts_user(engine):
    engine.dispatch("changed")
    assert wait_for(lambda: engine.snapshot()["last_text"] == MESSAGES["nothing_to_compare"])


def test_unknown_command_rejected(engine):
    assert engine.dispatch("explode") is False


def test_ocr_failure_is_reported_not_raised():
    def boom(frame):
        raise RuntimeError("api down")
    a = Assistant(make_demo_detector(), SyntheticSource(fps=60), ocr_fn=boom, demo=True)
    a.start()
    try:
        assert wait_for(lambda: a.frame_id > 3)
        a.dispatch("read")
        assert wait_for(lambda: any(e["kind"] == "error" for e in a.events.since(0)))
        assert wait_for(lambda: a.snapshot()["busy"] is None)
        assert a.snapshot()["metrics"]["ocr_error"]["count"] == 1
    finally:
        a.stop()


def test_camera_loss_recovers_then_gives_up():
    class Flaky:
        def __init__(self):
            self.n, self.opens = 0, 0

        def read(self):
            self.n += 1
            return False, None

        def open(self):
            self.opens += 1
            return False

        def release(self):
            pass
    src = Flaky()
    a = Assistant(make_demo_detector(), src, demo=True)
    a.start()
    try:
        assert wait_for(lambda: a.quit_event.is_set(), timeout=15)
        assert src.opens == 3 and any(EVENTS["camera_dead"] in e["text"] for e in a.events.since(0))
    finally:
        a.stop()


# ---------------- web ----------------

def test_web_api(engine):
    c = create_app(engine).test_client()
    assert c.get("/healthz").json["ok"]
    state = c.get("/api/state").json
    assert state["demo"] and "read" in state["commands"] and "quit" not in state["commands"]
    assert c.get("/snapshot.jpg").mimetype == "image/jpeg"
    r = c.get("/")
    assert r.status_code == 200 and "Drishti Sahayak" in r.get_data(as_text=True)
    assert "script-src 'self'" in r.headers["Content-Security-Policy"]


def test_web_command_validation(engine):
    c = create_app(engine).test_client()
    assert c.post("/api/command", data={"cmd": "read"}).status_code == 415        # not JSON
    assert c.post("/api/command", json={"cmd": "quit"}).status_code == 400       # never remote quit
    assert c.post("/api/command", json={"cmd": "nope"}).status_code == 400
    assert c.post("/api/command", json={"cmd": "repeat"}).json == {"accepted": True}
    assert not engine.quit_event.is_set()


def test_web_events_since(engine):
    c = create_app(engine).test_client()
    ev = c.get("/api/events?since=0").json["events"]
    assert ev and c.get(f"/api/events?since={ev[-1]['id']}").json["events"] == []


def test_web_token_required(engine):
    c = create_app(engine, token="s3cret").test_client()
    assert c.get("/api/state").status_code == 401
    assert c.get("/api/state?token=wrong").status_code == 401
    assert c.get("/healthz").status_code == 200
    assert c.get("/?token=s3cret").status_code == 200
    assert c.get("/api/state").status_code == 200       # cookie now set


DEVANAGARI = __import__("re").compile("[\u0900-\u097f]")


def test_screen_text_is_hinglish_and_audio_text_is_hindi():
    from visionassist.demo import demo_ocr, demo_vlm
    a = Assistant(make_demo_detector(), SyntheticSource(fps=60), ocr_fn=demo_ocr, vlm_fn=demo_vlm, demo=True)
    a.start()
    try:
        assert wait_for(lambda: a.frame_id > 5)
        a.dispatch("read")
        assert wait_for(lambda: a.snapshot()["busy"] is None and any(e["kind"] == "ocr" for e in a.events.since(0)), 8)
        a.dispatch("describe")
        assert wait_for(lambda: a.snapshot()["busy"] is None and any(e["kind"] == "vlm" for e in a.events.since(0)), 8)
        a.dispatch("mute")
        a.dispatch("changed")
        texts = [e["text"] for e in a.events.since(0)]
        assert len(texts) >= 6
        assert not [t for t in texts if DEVANAGARI.search(t)], texts       # screen: Roman only
        snap = a.snapshot()
        assert DEVANAGARI.search(snap["last_text"]) and not DEVANAGARI.search(snap["last_display"])
        assert snap["busy"] is None
    finally:
        a.stop()


def test_translation_failure_is_spoken_in_hindi_and_reported(monkeypatch):
    from visionassist import translate
    monkeypatch.setattr(translate, "_translate_once", lambda t: t)    # "translation" returns English
    a = Assistant(make_demo_detector(), SyntheticSource(fps=60), ocr_fn=lambda f: "EXIT", demo=True)
    a.start()
    try:
        assert wait_for(lambda: a.frame_id > 3)
        a.dispatch("read")
        assert wait_for(lambda: any(e["kind"] == "error" for e in a.events.since(0)))
        # Check the event log, not last_text: a detection alert from the capture
        # loop can overwrite last_text between the two polls.
        assert wait_for(lambda: any(e["kind"] == "speech" and e["text"] == MESSAGES_DISPLAY["translate_error"]
                                    for e in a.events.since(0)))
    finally:
        a.stop()


def test_snapshot_has_normalised_boxes_for_labels(engine):
    assert wait_for(lambda: engine.snapshot()["detections"])
    d = engine.snapshot()["detections"][0]
    x1, y1, x2, y2 = d["box"]
    assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and d["name_hi"] and d["label"].isascii()


def test_dashboard_is_hinglish(engine):
    html = create_app(engine).test_client().get("/").get_data(as_text=True)
    assert "Text padhein" in html and "Scene batayein" in html and "Aas-paas ka view" in html
    assert "Drishti Sahayak" in html and 'id="radar"' in html
    for formal in ("लिखावट पढ़ें", "Read text", "Describe scene", "Mute sound"):
        assert formal not in html
