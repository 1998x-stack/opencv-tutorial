"""Offline, dependency-light JSON HTTP server and static learning UI."""
from __future__ import annotations

import base64
import binascii
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from .algorithms import make_sample
from .curriculum import get_curriculum
from .pipeline import ALGORITHMS, run_pipeline

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
MAX_REQUEST = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 8_000_000


def encode_image(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def decode_image(payload: str) -> np.ndarray:
    if not isinstance(payload, str) or len(payload) > MAX_REQUEST * 2:
        raise ValueError("Invalid image payload")
    if payload.startswith("data:"):
        if ";base64," not in payload:
            raise ValueError("Expected base64 image data URL")
        payload = payload.split(";base64,", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image") from exc
    if len(raw) > MAX_REQUEST:
        raise ValueError("Image exceeds upload limit")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Invalid image or image exceeds 8 megapixels")
    return image


def process_payload(data: dict) -> dict:
    image = decode_image(data["image"])
    result = run_pipeline(image, algorithm=str(data.get("algorithm", "gaussian")),
                          kernel=int(data.get("kernel", 5)), sigma=float(data.get("sigma", 1.2)),
                          threshold=int(data.get("threshold", 120)))
    return {"images": {name: encode_image(stage) for name, stage in result.stages.items()},
            "metrics": result.metrics}


class LabHandler(BaseHTTPRequestHandler):
    def _json(self, status: int, data: dict | list) -> None:
        content = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            return self._json(200, {"status": "ok", "opencv": cv2.__version__})
        if path == "/api/catalog":
            return self._json(200, ALGORITHMS)
        if path == "/api/curriculum":
            return self._json(200, get_curriculum())
        if path == "/api/sample":
            return self._json(200, {"image": encode_image(make_sample())})
        static_files = {"/": ("index.html", "text/html; charset=utf-8"),
                        "/style.css": ("style.css", "text/css; charset=utf-8"),
                        "/app.js": ("app.js", "text/javascript; charset=utf-8")}
        if path not in static_files:
            return self._json(404, {"error": "Not found"})
        filename, mime = static_files[path]
        content = (WEB_DIR / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/process":
            return self._json(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 1 or length > MAX_REQUEST:
                return self._json(413, {"error": "Request exceeds 8 MB or is empty"})
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                return self._json(415, {"error": "Expected application/json"})
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("Expected a JSON object")
            output = process_payload(request)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, OverflowError) as exc:
            return self._json(400, {"error": str(exc)[:300]})
        return self._json(200, output)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), LabHandler)
    print(f"OpenCV Lab at http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
