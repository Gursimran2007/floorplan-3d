"""
Pure-stdlib HTTP server tying the pipeline together (only dep beyond stdlib is
opencv/numpy, used by detect.py). No web framework.

    /opt/anaconda3/bin/python server.py      # serves the viewer on :8000

Routes:
  GET  /                -> the three.js viewer (static/index.html)
  GET  /api/demo        -> {image: <data-url>} of the bundled sample plan
  POST /api/model       -> body {image:<data-url>}; runs detect + geometry,
                           returns the 3D model JSON the viewer renders.

The detector is the swappable AI slice: replace detect.detect() with a learned
model and nothing else here changes.
"""
import base64
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from detect import detect, DetectionError
from geometry import build

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")


def _decode_data_url(data_url):
    """data:image/png;base64,XXXX -> raw bytes."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


def _pdf_first_page_to_png(raw):
    """Rasterize page 1 of a PDF to PNG bytes at ~2x so thin CAD lines survive."""
    import fitz   # PyMuPDF
    doc = fitz.open(stream=raw, filetype="pdf")
    if doc.page_count == 0:
        raise DetectionError("the PDF has no pages.")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return pix.tobytes("png")


def model_from_image_bytes(raw):
    # PDF? (magic bytes) -> rasterize the first page. PNG/JPG/etc. go straight
    # to OpenCV's imread, which decodes them all.
    if raw[:5] == b"%PDF-":
        raw = _pdf_first_page_to_png(raw)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(raw)
        path = f.name
    try:
        det = detect(path)
        return build(det)
    finally:
        os.unlink(path)


def _demo_image():
    """Generate a fresh sample plan on the fly and return it as a data URL."""
    import io
    from synth import generate
    img, _ = generate(seed=7)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/demo":
            self._send(200, {"image": _demo_image()})
        elif path == "/api/cubicasa":
            # precomputed model from the pretrained CubiCasa5K detector (demo)
            p = os.path.join(STATIC, "_cubicasa_model.json")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    self._send(200, f.read(), "application/json")
            else:
                self._send(404, {"error": "no precomputed model"})
        elif path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/model":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
            raw = _decode_data_url(payload["image"])
            model = model_from_image_bytes(raw)
            self._send(200, model)
        except DetectionError as e:
            self._send(422, {"error": str(e)})
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(400, {"error": str(e)})


def main():
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"floorplan-3d serving on http://localhost:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
