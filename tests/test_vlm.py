import pytest
import requests

from visionassist import config as cfg
from visionassist import vlm
from visionassist.resilience import CircuitBreaker


def test_parse_valid_json():
    out = vlm.parse_vlm_json('{"summary": "A hall", "obstacles": ["door", " "], "path_status": "unclear"}')
    assert out == {"summary": "A hall", "obstacles": ["door"], "path_status": "unclear", "structured": True}


def test_parse_json_inside_prose_and_fences():
    out = vlm.parse_vlm_json('Sure!\n```json\n{"summary":"x","obstacles":[],"path_status":"none_seen"}\n```')
    assert out["structured"] and out["path_status"] == "none_seen"


def test_parse_invalid_path_becomes_unclear_and_caps_obstacles():
    out = vlm.parse_vlm_json('{"summary":"s","obstacles":["a","b","c","d"],"path_status":"totally safe"}')
    assert out["path_status"] == "unclear" and out["obstacles"] == ["a", "b", "c"]


@pytest.mark.parametrize("text", ["just a sentence", "{broken", "", None, 42])
def test_parse_never_raises(text):
    out = vlm.parse_vlm_json(text)
    assert out["path_status"] == "unclear" and out["obstacles"] == []
    assert out["summary"] == (text.strip() if isinstance(text, str) else "")


def test_prompt_contains_hints_and_never_claims_safety():
    dets = {"person": (0.9, (0, 0, 1, 1), 320, 0.2)}
    p = vlm.build_vlm_prompt(dets, (480, 640, 3))
    assert "person (centre, very close)" in p and "Do not claim anything is safe" in p
    assert "no known objects" in vlm.build_vlm_prompt({}, (480, 640, 3))


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code, self._p = status, payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._p


@pytest.fixture(autouse=True)
def fresh_breakers(monkeypatch):
    monkeypatch.setattr(vlm, "_vlm_breakers", {})
    monkeypatch.setattr(vlm, "_slow_until", {})


def ok(text):
    return FakeResp(200, {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})


def test_request_model_success(monkeypatch):
    monkeypatch.setattr(vlm.SESSION, "post", lambda *a, **k: ok(" hello "))
    assert vlm._request_model("m", {}, "b64", "p", {}) == "hello"


def test_404_trips_breaker_for_an_hour(monkeypatch):
    monkeypatch.setattr(vlm.SESSION, "post", lambda *a, **k: FakeResp(404))
    with pytest.raises(requests.HTTPError):
        vlm._request_model("gone", {}, "b", "p", {})
    assert not vlm._breaker("gone").allow()


def test_empty_content_is_an_error(monkeypatch):
    monkeypatch.setattr(vlm.SESSION, "post", lambda *a, **k: ok("  "))
    with pytest.raises(ValueError):
        vlm._request_model("m", {}, "b", "p", {})


def test_race_falls_through_failed_model(monkeypatch):
    def fake(model_id, *a):
        if model_id == "bad":
            raise RuntimeError("boom")
        return "from good"
    monkeypatch.setattr(vlm, "_request_model", fake)
    assert vlm._race_models([("bad", {}), ("good", {})], "b", "p", {}) == "from good"


def test_race_all_fail(monkeypatch):
    def fake(*a):
        raise RuntimeError("nope")
    monkeypatch.setattr(vlm, "_request_model", fake)
    with pytest.raises(RuntimeError, match="All VLM models failed"):
        vlm._race_models([("a", {}), ("b", {})], "b", "p", {})


def test_race_hedges_when_first_is_slow(monkeypatch):
    import time
    monkeypatch.setattr(cfg, "HEDGE_DELAY_S", 0.1)

    def fake(model_id, *a):
        if model_id == "slow":
            time.sleep(1.0)
            return "slow answer"
        return "fast answer"
    monkeypatch.setattr(vlm, "_request_model", fake)
    assert vlm._race_models([("slow", {}), ("fast", {})], "b", "p", {}) == "fast answer"


def test_describe_structured_end_to_end(monkeypatch, frame):
    monkeypatch.setattr(vlm, "_ask_vlm", lambda f, p: '{"summary":"s","obstacles":["x"],"path_status":"obstacle_ahead"}')
    out = vlm.describe_scene_structured(frame, {}, frame.shape)
    assert out["obstacles"] == ["x"] and out["path_status"] == "obstacle_ahead"


def test_describe_rejects_empty_image():
    import numpy as np
    with pytest.raises(ValueError):
        vlm.describe_scene_structured(np.zeros((0, 0, 3), dtype="uint8"), {}, (0, 0, 3))


def test_configured_models_override(monkeypatch):
    monkeypatch.setattr(cfg, "VLM_MODELS", ["a/b:free"])
    assert [m[0] for m in vlm._models()] == ["a/b:free"]


def test_breaker_open_state_visible():
    cb = CircuitBreaker("x", failure_threshold=1, cooldown_sec=60)
    cb.record_failure()
    assert cb.is_open and not cb.allow()
