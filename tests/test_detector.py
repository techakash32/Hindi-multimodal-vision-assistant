import time

import numpy as np
import pytest

from visionassist import config as cfg
from visionassist.detector import NAVIGATION_CLASSES, Detector
from visionassist.hindi_text import CLASS_HI, describe_hindi


class T:
    def __init__(self, v):
        self.v = v

    def __getitem__(self, i):
        return self.v


class Box:
    def __init__(self, conf, cls, xyxy):
        self.conf, self.cls, self.xyxy = T(conf), T(cls), T(xyxy)


class FakeModel:
    names = {0: "person", 1: "kite", 2: "cup"}

    def __init__(self):
        self.boxes = []

    def __call__(self, frame, **kw):
        return [type("R", (), {"boxes": self.boxes})()]


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
BIG = (200, 100, 440, 460)     # ~19% of frame, centre


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(cfg, "REANNOUNCE_AFTER_S", 0.1)
    m = FakeModel()
    return Detector(model=m), m


def run(det, n=1):
    last = None
    for _ in range(n):
        last = det.detect(FRAME)
    return last


def test_announces_only_after_stable_frames(setup):
    det, m = setup
    m.boxes = [Box(0.9, 0, BIG)]
    assert run(det, 1)[1] == [] and run(det, 1)[1] == []     # frames 1-2: not yet
    _cur, newly = det.detect(FRAME)                           # frame 3
    assert [o[0] for o in newly] == ["person"]
    assert det.detect(FRAME)[1] == []                         # not repeated


def test_reannounced_after_absence(setup):
    det, m = setup
    m.boxes = [Box(0.9, 0, BIG)]
    run(det, 3)
    m.boxes = []
    det.detect(FRAME)
    time.sleep(0.15)
    det.detect(FRAME)
    m.boxes = [Box(0.9, 0, BIG)]
    newly = run(det, 3)[1]
    assert [o[0] for o in newly] == ["person"]


def test_flicker_is_not_announced(setup):
    det, m = setup
    for i in range(6):
        m.boxes = [Box(0.9, 0, BIG)] if i % 2 == 0 else []
        assert det.detect(FRAME)[1] == []


def test_low_confidence_and_irrelevant_classes_ignored(setup):
    det, m = setup
    m.boxes = [Box(0.3, 0, BIG), Box(0.95, 1, BIG)]    # weak person, kite
    cur, newly = run(det, 4)
    assert cur == {} and newly == []


def test_tiny_objects_are_not_announced(setup):
    det, m = setup
    m.boxes = [Box(0.95, 0, (10, 10, 30, 30))]
    assert run(det, 4)[1] == []


def test_small_handheld_objects_use_lower_area_threshold(setup):
    det, m = setup
    m.boxes = [Box(0.95, 2, (300, 300, 360, 360))]     # cup, ~1.2%
    assert [o[0] for o in run(det, 3)[1]] == ["cup"]


def test_highest_confidence_per_class_kept(setup):
    det, m = setup
    m.boxes = [Box(0.8, 0, BIG), Box(0.95, 0, (0, 0, 300, 300))]
    cur, _ = det.detect(FRAME)
    assert cur["person"][0] == 0.95


def test_empty_frame_is_safe(setup):
    det, _ = setup
    assert det.detect(None) == ({}, [])
    assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == ({}, [])


def test_describe_direction_and_proximity():
    shape = (480, 640, 3)
    assert Detector.describe(("person", .9, None, 320, .2), shape) == "person, very close, ahead"
    assert Detector.describe(("cup", .9, None, 50, .01), shape) == "cup, on your left"
    assert Detector.describe(("car", .9, None, 600, .08), shape) == "car, nearby, on your right"
    assert describe_hindi(("person", .9, None, 320, .2), shape) == "व्यक्ति, बहुत पास, सामने"


def test_prioritize_hazards_first_then_size():
    newly = [("cup", .9, 0, 0, .5), ("person", .9, 0, 0, .1), ("car", .9, 0, 0, .3)]
    assert [o[0] for o in Detector.prioritize(newly, top_n=2)] == ["car", "person"]


def test_every_navigation_class_has_hindi_name():
    assert not [c for c in NAVIGATION_CLASSES if c not in CLASS_HI]
