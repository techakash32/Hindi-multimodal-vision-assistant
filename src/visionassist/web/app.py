"""Flask dashboard: live annotated video (MJPEG), controls, speech and
event feed, latency stats and system health.

Bound to localhost by default. Set WEB_TOKEN to require a token when you
expose it on a network (it is accepted once as ?token=... and then kept in
an HttpOnly cookie).
"""
from __future__ import annotations

import hmac
import logging
import time

from flask import Flask, Response, abort, jsonify, render_template, request

from .. import __version__
from .. import config as cfg
from ..engine import COMMANDS
from ..health import run_checks
from ..strings import TITLE, UI

log = logging.getLogger(__name__)

WEB_COMMANDS = tuple(c for c in COMMANDS if c != "quit")   # never stop it remotely


def create_app(engine, token: str = "") -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["JSON_AS_ASCII"] = False
    health = [c.as_dict() for c in run_checks(demo=engine.demo)]

    @app.before_request
    def _auth():
        if not token or request.path == "/healthz":
            return None
        supplied = request.args.get("token") or request.cookies.get("va_token", "")
        if not hmac.compare_digest(supplied, token):
            abort(401)
        return None

    @app.after_request
    def _headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.path.startswith("/api/") or request.path == "/video":
            resp.headers["Cache-Control"] = "no-store"
        if token and request.args.get("token") and request.path == "/":
            resp.set_cookie("va_token", token, httponly=True, samesite="Strict")
        return resp

    @app.get("/")
    def index():
        return render_template("index.html", version=__version__, demo=engine.demo, s=UI, title=TITLE,
                               local_only=cfg.LOCAL_ONLY)

    @app.get("/healthz")
    def healthz():
        return jsonify(ok=not engine.quit_event.is_set(), camera=engine._camera_ok,
                       frames=engine.frame_id, version=__version__)

    @app.get("/api/state")
    def state():
        data = engine.snapshot()
        data["health"] = health
        data["commands"] = list(WEB_COMMANDS)
        return jsonify(data)

    @app.get("/api/events")
    def events():
        since = request.args.get("since", 0, type=int)
        return jsonify(events=engine.events.since(since))

    @app.post("/api/command")
    def command():
        # JSON-only: a cross-site form cannot send this without a CORS preflight.
        if not request.is_json:
            abort(415)
        cmd = (request.get_json(silent=True) or {}).get("cmd", "")
        if cmd not in WEB_COMMANDS:
            return jsonify(error="unknown command"), 400
        return jsonify(accepted=engine.dispatch(cmd, "web"))

    @app.get("/video")
    def video():
        def gen():
            last, interval = -1, 1.0 / max(1, cfg.STREAM_FPS)
            while not engine.quit_event.is_set():
                fid = engine.frame_id
                if fid != last:
                    data = engine.jpeg()
                    if data:
                        last = fid
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                               b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                time.sleep(interval)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot.jpg")
    def snapshot_jpg():
        data = engine.jpeg()
        if not data:
            abort(503)
        return Response(data, mimetype="image/jpeg")

    return app


def serve(app: Flask, host: str, port: int) -> None:
    """Run with waitress when installed (production), else Werkzeug."""
    if host not in ("127.0.0.1", "localhost", "::1") and not cfg.WEB_TOKEN:
        log.warning("dashboard is exposed on %s without WEB_TOKEN; anyone on the "
                    "network can see the camera. Set WEB_TOKEN in .env.", host)
    try:
        from waitress import serve as wserve
        log.info("dashboard on http://%s:%d (waitress)", host, port)
        wserve(app, host=host, port=port, threads=8, ident="visionassist")
    except ImportError:
        log.info("dashboard on http://%s:%d (werkzeug; pip install waitress for production)", host, port)
        app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
