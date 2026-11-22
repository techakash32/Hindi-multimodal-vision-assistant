"""Golden-file test for the Devanagari -> Hinglish transliterator.

Every line of tests/golden/hinglish.tsv is a pair written by hand. A failing
pair is either a transliterator bug or a wrong expectation: decide which, then fix
that side. Do not regenerate the file from the program's output."""
from pathlib import Path

import pytest

from visionassist.hinglish import to_hinglish

GOLDEN = Path(__file__).parent / "golden" / "hinglish.tsv"


def load():
    rows = []
    for n, line in enumerate(GOLDEN.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() and not line.startswith("#"):
            src, want = line.split("\t")
            rows.append(pytest.param(src, want, id=f"{n}:{src[:24]}"))
    return rows


def test_golden_file_is_big_enough():
    assert len(load()) >= 200


@pytest.mark.parametrize("src,want", load())
def test_hinglish_golden(src, want):
    assert to_hinglish(src) == want


def test_output_is_roman_only():
    for p in load():
        assert not any("ऀ" <= c <= "ॿ" for c in to_hinglish(p.values[0]))
