import io
import zipfile

import pytest

from visionassist import cli, downloads
from visionassist import config as cfg


def test_default_command_is_run():
    p = cli.parser()
    a = p.parse_args(["run", "--demo", "--ui", "web", "--port", "9000"])
    assert a.demo and a.ui == "web" and a.port == 9000 and a.func is cli.cmd_run


def test_bare_flags_imply_run():
    assert cli.normalize_argv([]) == ["run"]
    assert cli.normalize_argv(["--demo"]) == ["run", "--demo"]
    assert cli.normalize_argv(["check"]) == ["check"]
    assert cli.normalize_argv(["--version"]) == ["--version"]


def test_check_exit_code(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cfg, "YOLO_MODEL", tmp_path / "none.pt")
    monkeypatch.setattr(cfg, "VOICE_MODEL", tmp_path / "none.onnx")
    assert cli.cmd_check(None) == 1
    assert "[FAIL] YOLO model" in capsys.readouterr().out


class FakeResponse:
    def __init__(self, data, declared=None):
        self.data, self.headers = data, {"content-length": str(len(data) if declared is None else declared)}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data


def test_fetch_writes_atomically_and_skips_existing(tmp_path):
    a = downloads.Asset("x", "http://h/f.bin", tmp_path / "d" / "f.bin")
    calls = []
    get = lambda *args, **kw: calls.append(1) or FakeResponse(b"hello")  # noqa: E731
    assert downloads.fetch(a, get=get).read_bytes() == b"hello"
    downloads.fetch(a, get=get)
    assert len(calls) == 1 and not list((tmp_path / "d").glob("*.part"))


def test_incomplete_download_is_rejected(tmp_path):
    a = downloads.Asset("x", "http://h/f.bin", tmp_path / "f.bin")
    with pytest.raises(IOError):
        downloads.fetch(a, get=lambda *a, **k: FakeResponse(b"abc", declared=10))
    assert not (tmp_path / "f.bin").exists() and not (tmp_path / "f.bin.part").exists()


def make_zip(names):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in names:
            z.writestr(n, "x")
    return buf.getvalue()


def test_zip_extracts_and_cleans_up(tmp_path):
    a = downloads.Asset("vosk", "http://h/m.zip", tmp_path / "m.zip", unzip_to=tmp_path)
    out = downloads.fetch(a, get=lambda *a, **k: FakeResponse(make_zip(["m/a.txt"])))
    assert (tmp_path / "m" / "a.txt").exists() and not (tmp_path / "m.zip").exists() and out == tmp_path / "m"


def test_zip_slip_blocked(tmp_path):
    a = downloads.Asset("vosk", "http://h/m.zip", tmp_path / "sub" / "m.zip", unzip_to=tmp_path / "sub")
    with pytest.raises(ValueError, match="unsafe"):
        downloads.fetch(a, get=lambda *a, **k: FakeResponse(make_zip(["../evil.txt"])))
    assert not (tmp_path / "evil.txt").exists()


def test_unknown_asset_group():
    with pytest.raises(ValueError):
        downloads.download(["nope"])
