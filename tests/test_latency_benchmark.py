import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_tool():
    spec = importlib.util.spec_from_file_location("latency_benchmark", ROOT / "tools" / "latency_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_skipped_stages_are_written_as_na_and_never_estimated(tmp_path, monkeypatch, capsys):
    mod = load_tool()
    monkeypatch.setattr(mod.cfg, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(mod.cfg, "OCR_API_KEY", "")
    monkeypatch.setattr(mod.cfg, "OCR_ENGINE", "ocrspace")
    monkeypatch.setattr(mod, "bench_detection", lambda *a: (None, "no YOLO here", ""))
    out_csv, out_md = tmp_path / "l.csv", tmp_path / "s.md"
    assert mod.main(["--csv", str(out_csv), "--summary", str(out_md), "--audio-dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "ASR: SKIPPED:" in printed and "VLM: SKIPPED: OPENROUTER_API_KEY" in printed
    rows = list(csv.DictReader(out_csv.open()))
    assert list(rows[0]) == ["stage", "run_idx", "latency_ms"]
    by_stage = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(r["latency_ms"])
    assert by_stage["ASR"] == ["n/a"] and by_stage["VLM"] == ["n/a"] and by_stage["Detection"] == ["n/a"]
    assert len(by_stage["End-to-end"]) == 20 and len(by_stage["Orchestration overhead"]) == 20
    md = out_md.read_text()
    assert "| Stage | n | p50 (ms) | p95 (ms) |" in md and "| ASR | n/a | n/a | n/a |" in md
    assert "| End-to-end | 20 |" in md


def test_vlm_failure_is_reported_as_skipped_not_a_crash(tmp_path, monkeypatch, capsys):
    mod = load_tool()
    monkeypatch.setattr(mod.cfg, "OCR_ENGINE", "ocrspace")
    monkeypatch.setattr(mod, "bench_detection", lambda *a: (None, "no YOLO here", ""))

    def boom(frames):
        raise RuntimeError("All VLM models failed. Last error: Read timed out")
    monkeypatch.setattr(mod, "bench_vlm", boom)
    out_csv, out_md = tmp_path / "l.csv", tmp_path / "s.md"
    assert mod.main(["--csv", str(out_csv), "--summary", str(out_md), "--audio-dir", str(tmp_path)]) == 0
    assert "VLM: SKIPPED: VLM calls failed" in capsys.readouterr().out
    assert "| VLM | n/a | n/a | n/a |" in out_md.read_text()
