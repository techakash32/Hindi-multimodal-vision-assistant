import numpy as np

from visionassist.scene import (
    detection_signature,
    frame_signature,
    scene_is_same,
    scene_meaningfully_changed,
    signatures_similar,
)

SHAPE = (480, 640, 3)


def det(xc, area):
    return (0.9, (0, 0, 10, 10), xc, area)


def test_signature_zones_and_distance():
    sig = detection_signature({"person": det(50, 0.2), "chair": det(320, 0.08), "car": det(600, 0.01)}, SHAPE)
    assert sig == {("person", "L", "near"), ("chair", "C", "mid"), ("car", "R", "far")}


def test_signature_empty_and_none():
    assert detection_signature({}, SHAPE) == frozenset()
    assert detection_signature(None, SHAPE) == frozenset()


def test_changed_reports_added_removed():
    prev = frozenset({("person", "C", "near")})
    cur = frozenset({("chair", "L", "far")})
    changed, added, removed = scene_meaningfully_changed(prev, cur)
    assert changed and added == {("chair", "L", "far")} and removed == {("person", "C", "near")}


def test_no_previous_counts_as_changed():
    assert scene_meaningfully_changed(None, frozenset({("a", "C", "far")}))[0]


def test_same_scene_requires_all_conditions(frame):
    sig, pix = frozenset(), frame_signature(frame)
    prev = {"det": sig, "pix": pix, "ts": 100.0}
    assert scene_is_same(prev, sig, pix, 110.0, 30.0, 25.0)
    assert not scene_is_same(prev, sig, pix, 131.0, 30.0, 25.0)            # too old
    assert not scene_is_same(None, sig, pix, 110.0, 30.0, 25.0)            # nothing before
    assert not scene_is_same(prev, frozenset({("x", "C", "far")}), pix, 110.0, 30.0, 25.0)
    other = frame_signature(np.zeros_like(frame))
    assert not scene_is_same(prev, sig, other, 110.0, 30.0, 25.0)          # picture changed


def test_signatures_similar_handles_none():
    assert not signatures_similar(None, np.zeros(3))
