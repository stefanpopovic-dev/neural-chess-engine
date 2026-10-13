#!/usr/bin/env python
"""Run the web demo locally: serves public/ and the /api/move function.

Uses the same request handling as the Vercel deployment (api/move.py).

Example:
    python scripts/serve_demo.py              # then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("api_move", ROOT / "api" / "move.py")
assert _spec and _spec.loader
api_move = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api_move)


class DemoHandler(SimpleHTTPRequestHandler):
    """Static files from public/, plus the API route."""

    # Explicit types: on Windows the registry can map .js to text/plain, which breaks ES modules.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".css": "text/css",
        ".svg": "image/svg+xml",
        ".html": "text/html",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(ROOT / "public"), **kwargs)

    def _is_api(self) -> bool:
        return self.path.split("?", 1)[0] == "/api/move"

    def do_POST(self) -> None:  # noqa: N802
        if self._is_api():
            api_move.handle_post(self)
        else:
            self.send_error(404)

    def do_GET(self) -> None:  # noqa: N802
        if self._is_api():
            api_move.handler.do_GET(self)
        else:
            super().do_GET()

    def end_headers(self) -> None:
        if not self._is_api():
            self.send_header("Cache-Control", "no-cache")  # always serve edited files while developing
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo running at http://{args.host}:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
