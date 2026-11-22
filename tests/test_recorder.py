import json

import numpy as np

from visionassist import cli
from visionassist.demo import demo_ocr, demo_vlm, make_demo_detector
from visionassist.engine import Assistant
from visionassist.events import EventBus
from visionassist.metrics import Metrics
from visionassist.recorder import SessionRecorder, export_scenarios, load_session, replay
from visionassist.sources import SyntheticSource


def make_engine(tmp_path, ocr_fn=demo_ocr):
    eng = Assistant(make_demo_detector(), SyntheticSource(), ocr_fn=ocr_fn, vlm_fn=demo_vlm,
                    metrics=Metrics(None), demo=True, voice=False)
    eng.attach_recorder(SessionRecorder(tmp_path / "s", demo=True))
    return eng


def record_one(eng, kind):
    import time
    eng.start()
    try:
        time.sleep(0.6)                       # let the capture loop produce a frame
        eng.dispatch(kind, "test")
        for _ in range(60):
            time.sleep(0.1)
            if any(r["type"] == "request" for r in load_session(eng.recorder.path)):
                break
    finally:
        eng.stop()


def test_event_bus_sink_receives_events_and_survives_a_broken_sink():
    bus, seen = EventBus(), []
    bus.add_sink(lambda ev: 1 / 0)
    bus.add_sink(seen.append)
    bus.publish("system", "hello")
    assert seen and seen[0]["text"] == "hello"


def test_session_records_requests_and_replays_identically(tmp_path):
    eng = make_engine(tmp_path)
    record_one(eng, "read")
    rows = load_session(tmp_path / "s")
    kinds = {r["type"] for r in rows}
    assert {"session", "event", "request"} <= kinds
    req = next(r for r in rows if r["type"] == "request")
    assert req["request"] == "read" and (tmp_path / "s" / req["frame"]).exists()
    out = replay(tmp_path / "s", ocr_fn=demo_ocr, vlm_fn=demo_vlm)
    assert out and all(r.same for r in out), [r.changes for r in out]


def test_replay_reports_a_changed_answer(tmp_path):
    eng = make_engine(tmp_path)
    record_one(eng, "read")
    from visionassist.ocr import OcrText
    rows = replay(tmp_path / "s", ocr_fn=lambda f: OcrText("completely different words here", 0.95, "fake"),
                  vlm_fn=demo_vlm)
    assert any(not r.same for r in rows)
    assert any("answer_text" in c for r in rows for c in r.changes)


def test_export_scenarios_leaves_ground_truth_unlabelled(tmp_path):
    rec = SessionRecorder(tmp_path / "s")
    from visionassist.pipeline import PipelineInput, run_once
    frame = np.full((120, 160, 3), 255, np.uint8)
    res = run_once(PipelineInput(frame=frame), "read", ocr_fn=demo_ocr, vlm_fn=demo_vlm)
    rec.request("read", frame, None, res)
    rec.close()
    n = export_scenarios(tmp_path / "s", tmp_path / "out.jsonl")
    row = json.loads((tmp_path / "out.jsonl").read_text().splitlines()[0])
    assert n == 1 and row["ground_truth"] is None and row["category"] == "recorded"


def test_cli_replay_command(tmp_path, capsys):
    eng = make_engine(tmp_path)
    record_one(eng, "read")
    assert cli.main(["replay", str(tmp_path / "s"), "--demo"]) == 0
    assert "requests replayed, 0 changed" in capsys.readouterr().out
