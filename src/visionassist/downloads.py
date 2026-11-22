"""Download the models VisionAssist needs (``visionassist download-models``)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

from . import config as cfg

log = logging.getLogger(__name__)

HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


CHECKSUM_FILE = Path(__file__).resolve().parents[2] / "config" / "checksums.json"


def known_checksums() -> Dict[str, str]:
    """url -> sha256 pins from config/checksums.json (empty until you record them
    with ``python tools/check_urls.py --record``). Pinned files are verified."""
    try:
        return json.loads(CHECKSUM_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Asset:
    key: str
    url: str
    dest: Path
    unzip_to: Optional[Path] = None   # extract archive here (then delete it)


def assets() -> Dict[str, List[Asset]]:
    v = cfg.VOICE_DIR
    return {
        "yolo": [Asset("yolo", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
                       cfg.MODEL_DIR / "yolov8n.pt")],
        "en": [
            Asset("en", f"{HF}/en/en_US/lessac/medium/en_US-lessac-medium.onnx", v / "en_US-lessac-medium.onnx"),
            Asset("en", f"{HF}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json", v / "en_US-lessac-medium.onnx.json"),
        ],
        "hi": [
            Asset("hi", f"{HF}/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx", v / "hi_IN-priyamvada-medium.onnx"),
            Asset("hi", f"{HF}/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx.json", v / "hi_IN-priyamvada-medium.onnx.json"),
        ],
        "vosk-hi": [Asset("vosk-hi", "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
                          cfg.MODEL_DIR / "vosk-model-small-hi-0.22.zip", unzip_to=cfg.MODEL_DIR)],
        "vosk": [Asset("vosk", "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
                       cfg.MODEL_DIR / "vosk-model-small-en-us-0.15.zip", unzip_to=cfg.MODEL_DIR)],
    }


def _safe_extract(archive: Path, target: Path) -> None:
    """Extract a zip, refusing members that would escape `target`."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not (target / member).resolve().is_relative_to(target):
                raise ValueError(f"unsafe path in archive: {member}")
        zf.extractall(target)


def fetch(asset: Asset, get: Callable = requests.get, force: bool = False) -> Path:
    """Download one asset atomically. Skips files that already exist."""
    final = asset.dest
    marker = (asset.unzip_to / final.stem) if asset.unzip_to else final
    if marker.exists() and not force:
        log.info("already present: %s", marker)
        return marker

    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(final.suffix + ".part")
    log.info("downloading %s", asset.url)
    with get(asset.url, stream=True, timeout=(10, 60), allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
        if total and done != total:
            tmp.unlink(missing_ok=True)
            raise IOError(f"incomplete download ({done}/{total} bytes): {asset.url}")
    digest = sha256_of(tmp)
    pinned = known_checksums().get(asset.url)
    if pinned and pinned != digest:
        tmp.unlink(missing_ok=True)
        raise IOError(f"checksum mismatch for {asset.url}: expected {pinned[:12]}…, got {digest[:12]}…")
    log.info("sha256 %s %s%s", digest, final.name, "" if pinned else " (not pinned)")
    os.replace(tmp, final)

    if asset.unzip_to:
        _safe_extract(final, asset.unzip_to)
        final.unlink()
        return marker
    return final


def download(only: Optional[List[str]] = None, force: bool = False, get: Callable = requests.get) -> bool:
    """Download the selected asset groups. Returns True if all succeeded."""
    groups = assets()
    default = ["yolo", "hi", "vosk-hi"] if cfg.HINDI_ONLY else ["yolo", "en", "hi", "vosk"]
    wanted = only or default
    unknown = [k for k in wanted if k not in groups]
    if unknown:
        raise ValueError(f"unknown asset(s): {', '.join(unknown)}; choose from {', '.join(groups)}")
    ok = True
    for key in wanted:
        for asset in groups[key]:
            try:
                fetch(asset, get=get, force=force)
            except Exception as exc:
                ok = False
                log.error("failed %s: %s", asset.url, exc)
    return ok
