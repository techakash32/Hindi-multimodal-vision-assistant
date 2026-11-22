"""Command line interface: ``visionassist`` or ``python -m visionassist``."""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

from . import __version__
from . import config as cfg
from .logging_setup import setup_logging

log = logging.getLogger("visionassist")


def build_assistant(args):
    """Create the engine for real hardware or for demo mode."""
    from . import tts
    from .engine import Assistant
    from .metrics import Metrics

    if args.mute or args.demo:
        tts.set_muted(True)

    if args.demo:
        from .demo import demo_ocr, demo_vlm, make_demo_detector
        from .sources import SyntheticSource
        return Assistant(make_demo_detector(), SyntheticSource(), ocr_fn=demo_ocr, vlm_fn=demo_vlm,
                         metrics=Metrics(None), demo=True)

    from .detector import Detector
    from .health import run_health_check
    from .sources import CameraSource

    if not run_health_check():
        log.error("Cannot start: run `visionassist download-models`, then `visionassist check`.")
        return None

    source = CameraSource(args.camera)
    if not source.is_open():
        log.error("Cannot open camera %s. Try --camera 1, or close apps that use the camera.", source.index)
        return None
    return Assistant(Detector(), source, metrics=Metrics(cfg.LOG_DIR / "metrics.csv"),
                     voice=not args.no_voice)


def cmd_run(args) -> int:
    engine = build_assistant(args)
    if engine is None:
        return 1
    if args.record:
        import time as _t

        from .recorder import SessionRecorder
        folder = Path(args.record) if args.record != "auto" else cfg.LOG_DIR / "sessions" / _t.strftime("%Y%m%d-%H%M%S")
        engine.attach_recorder(SessionRecorder(folder, demo=args.demo))
        log.info("recording session to %s", folder)
    engine.start()
    ui = args.ui
    web_thread = None

    try:
        if ui in ("web", "both"):
            from .web import create_app, serve
            app = create_app(engine, token=cfg.WEB_TOKEN)
            if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
                import webbrowser
                threading.Timer(1.5, webbrowser.open, args=(f"http://{args.host}:{args.port}/",)).start()
            web_thread = threading.Thread(target=serve, args=(app, args.host, args.port), daemon=True)
            web_thread.start()
            if ui == "web":
                # Wait here so a spoken "band" (quit) ends the program. The server
                # thread alone would block forever. A short timeout keeps Ctrl+C responsive on Windows.
                while not engine.quit_event.wait(0.5):
                    pass
        if ui in ("window", "both"):
            from .ui_cv import run_window
            run_window(engine)
        elif ui == "headless":
            engine.quit_event.wait()
    except KeyboardInterrupt:
        log.info("shutting down (Ctrl+C)")
    finally:
        engine.stop()
    return 0


def cmd_check(args) -> int:
    from .health import run_checks
    checks = run_checks(demo=False)
    icon = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}
    for c in checks:
        print(f"[{icon[c.status]}] {c.name}: {c.detail}")
    failed = any(c.status == "fail" for c in checks)
    print("\nReady to run." if not failed else "\nFix the FAIL items first (try: visionassist download-models).")
    return 1 if failed else 0


def cmd_download(args) -> int:
    from .downloads import download
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    return 0 if download(only, force=args.force) else 1


def cmd_mics(args) -> int:
    try:
        from .mic_utils import candidate_devices, describe_devices
    except ImportError:
        print("sounddevice is not installed: pip install -r requirements.txt")
        return 1
    print("Input devices:\n" + describe_devices())
    print("\nOrder VisionAssist will try:", candidate_devices())
    return 0


def cmd_replay(args) -> int:
    from .recorder import export_scenarios, replay
    kw = {}
    if args.demo:
        from .demo import demo_ocr, demo_vlm, make_demo_detector
        kw = dict(ocr_fn=demo_ocr, vlm_fn=demo_vlm, detector=make_demo_detector())
    if args.no_abstain:
        from .guardrails import AbstentionConfig
        kw["abstention"] = AbstentionConfig.disabled()
    if args.export_scenarios:
        n = export_scenarios(Path(args.session), Path(args.export_scenarios))
        print(f"wrote {n} unlabelled scenarios to {args.export_scenarios}")
        return 0
    rows = replay(Path(args.session), **kw)
    for r in rows:
        print(f"{'same   ' if r.same else 'CHANGED'} #{r.index + 1} {r.request}")
        for c in r.changes:
            print(f"    {c}")
    changed = sum(not r.same for r in rows)
    print(f"\n{len(rows)} requests replayed, {changed} changed")
    return 1 if changed else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="visionassist", description="Assistive vision for blind and low-vision users.")
    p.add_argument("--version", action="version", version=f"visionassist {__version__}")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="start the assistant (default; opens the dashboard)")
    r.add_argument("--ui", choices=["web", "window", "both", "headless"], default="web",
                   help="web = Hindi browser dashboard (default); window = OpenCV window (no Hindi text); headless = speech only")
    r.add_argument("--no-browser", action="store_true", help="do not open the dashboard in a browser automatically")
    r.add_argument("--demo", action="store_true", help="synthetic camera and detector; needs no models, keys or camera")
    r.add_argument("--camera", type=int, default=None, help="camera index (default: CAMERA_INDEX or 0)")
    r.add_argument("--host", default=cfg.WEB_HOST, help="dashboard host (default 127.0.0.1)")
    r.add_argument("--port", type=int, default=cfg.WEB_PORT, help="dashboard port (default 8000)")
    r.add_argument("--mute", action="store_true", help="show speech on screen but play no audio")
    r.add_argument("--no-voice", action="store_true", help="disable microphone voice commands")
    r.add_argument("--record", nargs="?", const="auto", default=None, metavar="DIR",
                   help="record events, requests and frames (default folder: logs/sessions/<time>) for `visionassist replay`")
    r.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING (default: LOG_LEVEL or INFO)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("check", help="verify models, voices and API keys")
    c.set_defaults(func=cmd_check)

    d = sub.add_parser("download-models", help="download YOLO, Piper voices and the Vosk model")
    d.add_argument("--only", help="comma list of: yolo,hi,vosk-hi,en,vosk (default: yolo,hi,vosk-hi)")
    d.add_argument("--force", action="store_true", help="re-download existing files")
    d.set_defaults(func=cmd_download)

    rp = sub.add_parser("replay", help="replay a recorded session through the pipeline and show what changed")
    rp.add_argument("session", help="session folder or session.jsonl written by `run --record`")
    rp.add_argument("--demo", action="store_true", help="use the offline demo OCR / VLM / detector")
    rp.add_argument("--no-abstain", action="store_true", help="disable guardrails during replay")
    rp.add_argument("--export-scenarios", metavar="FILE", help="write the requests as unlabelled eval scenarios instead of replaying")
    rp.set_defaults(func=cmd_replay)

    m = sub.add_parser("mics", help="list microphones")
    m.set_defaults(func=cmd_mics)
    return p


def normalize_argv(argv):
    """`visionassist --demo` means `visionassist run --demo`."""
    argv = list(argv)
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--version")):
        argv.insert(0, "run")
    return argv


def main(argv=None) -> int:
    argv = normalize_argv(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    setup_logging(getattr(args, "log_level", None))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
