"""Local, read-only demo UI. Run: python demo_ui.py --port 7861."""
from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from demo_backend import DemoBackend, ROOT


def handler_for(backend: DemoBackend):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path: Path, *, video: bool = False) -> None:
            size = path.stat().st_size
            start, end = 0, size - 1
            byte_range = self.headers.get("Range") if video else None
            if byte_range:
                try:
                    unit, span = byte_range.split("=", 1)
                    left, right = span.split("-", 1)
                    if unit != "bytes":
                        raise ValueError
                    if left:
                        start = int(left)
                        end = min(int(right), size - 1) if right else size - 1
                    else:
                        suffix = int(right)
                        start = max(0, size - suffix)
                    if start < 0 or start > end or start >= size:
                        raise ValueError
                except (ValueError, IndexError):
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            self.send_response(206 if byte_range else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes" if video else "none")
            if byte_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            if url.path == "/":
                return self.send_file(ROOT / "demo_ui.html")
            if url.path == "/api/videos":
                return self.send_json({"videos": backend.videos()})
            if url.path == "/api/search":
                params = parse_qs(url.query)
                query = params.get("q", [""])[0]
                experiment_id = params.get("experiment_id", [None])[0]
                try:
                    return self.send_json(backend.search(query, experiment_id))
                except (ValueError, FileNotFoundError, ImportError) as exc:
                    return self.send_json({"error": str(exc)}, 400)
                except Exception as exc:
                    return self.send_json({"error": f"RAG unavailable: {exc}"}, 503)
            if url.path.startswith("/api/video/"):
                demo_id = unquote(url.path.removeprefix("/api/video/"))
                path = backend.video_path(demo_id)
                if path:
                    return self.send_file(path, video=True)
            self.send_error(404, "Not found")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepared-video and RAG demo UI")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT.parent / "chemistry-dataset")
    parser.add_argument("--manifest", type=Path, default=ROOT / "demo_video_cache.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    backend = DemoBackend(args.dataset_dir, args.manifest)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(backend))
    print(f"Demo UI: http://{args.host}:{args.port}", flush=True)
    print(f"Dataset: {backend.dataset_dir}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
