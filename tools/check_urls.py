"""Check that every model download URL still works, and optionally pin checksums.

    python tools/check_urls.py            # HEAD each URL, print status and size
    python tools/check_urls.py --record   # download each file, write config/checksums.json

Needs network access, so it is run by hand, not in CI. After --record, commit
config/checksums.json: ``visionassist download-models`` then rejects any file
whose SHA256 differs.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import requests

from visionassist.downloads import CHECKSUM_FILE, assets, known_checksums, sha256_of


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="download files and pin their SHA256")
    args = ap.parse_args()
    pins, bad = known_checksums(), 0
    for group, items in assets().items():
        for a in items:
            try:
                r = requests.head(a.url, allow_redirects=True, timeout=20)
                size = int(r.headers.get("content-length", 0))
                ok = r.status_code == 200
                print(f"{'OK  ' if ok else 'FAIL'} {r.status_code} {size / 1e6:8.1f} MB  {group:8s} {a.url}")
                bad += not ok
                if ok and args.record:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        with requests.get(a.url, stream=True, timeout=(10, 120)) as g:
                            for chunk in g.iter_content(1 << 20):
                                tmp.write(chunk)
                    pins[a.url] = sha256_of(Path(tmp.name))
                    Path(tmp.name).unlink()
            except requests.RequestException as exc:
                print(f"FAIL ---  {group:8s} {a.url}  ({exc})")
                bad += 1
    if args.record and not bad:
        CHECKSUM_FILE.parent.mkdir(exist_ok=True)
        CHECKSUM_FILE.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"pinned {len(pins)} checksums in {CHECKSUM_FILE}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
