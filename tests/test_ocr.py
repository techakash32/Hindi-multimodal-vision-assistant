import numpy as np
import pytest
import requests

from visionassist import config as cfg
from visionassist import ocr


class Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError("bad")

    def json(self):
        return self.payload


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(ocr, "_ocr_breaker", ocr.CircuitBreaker("OCR", 3, 60))
    monkeypatch.setattr(cfg, "OCR_API_KEY", "test-key")


def test_returns_text(monkeypatch, frame):
    monkeypatch.setattr(ocr.SESSION, "post", lambda *a, **k: Resp({"ParsedResults": [{"ParsedText": "EXIT "}, {"ParsedText": "door"}]}))
    assert ocr.read_text_from_frame(frame) == "EXIT  door"


def test_no_text(monkeypatch, frame):
    monkeypatch.setattr(ocr.SESSION, "post", lambda *a, **k: Resp({"ParsedResults": [{"ParsedText": ""}]}))
    assert ocr.read_text_from_frame(frame).startswith("No text found")


def test_service_error_message(monkeypatch, frame):
    monkeypatch.setattr(ocr.SESSION, "post", lambda *a, **k: Resp({"IsErroredOnProcessing": True, "ErrorMessage": ["Bad key"]}))
    assert ocr.read_text_from_frame(frame) == "OCR error: Bad key"


def test_empty_frame():
    assert ocr.read_text_from_frame(np.zeros((0, 0, 3), dtype="uint8")).startswith("OCR error")


def test_breaker_opens_after_three_network_failures(monkeypatch, frame):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("down")
    monkeypatch.setattr(ocr.SESSION, "post", boom)
    for _ in range(5):
        assert ocr.read_text_from_frame(frame).startswith("OCR error")
    assert len(calls) == 3 and ocr.breaker_state() == "open"


def test_sends_api_key_but_never_logs_it(monkeypatch, frame, caplog):
    seen = {}
    monkeypatch.setattr(ocr.SESSION, "post", lambda url, **k: seen.update(k) or Resp({"ParsedResults": []}))
    ocr.read_text_from_frame(frame)
    assert seen["data"]["apikey"] == "test-key"
    assert "test-key" not in caplog.text
