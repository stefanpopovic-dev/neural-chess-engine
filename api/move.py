"""Vercel serverless function: the engine's reply in a game.

POST /api/move with JSON
    {"fen": <start FEN, optional>, "moves": ["e2e4", ...], "engine": "nn" | "material", "time": 1.0}
returns the engine's move for the position reached by playing ``moves`` from
``fen``, plus its search statistics. The full move list (rather than just the
current FEN) lets the search detect repetitions.

The network runs through NumpyNNEvaluator, so the function needs only numpy
and python-chess (PyTorch is far too large for a serverless bundle).
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

from chess_engine.evaluators import CachedEvaluator, MaterialEvaluator, NumpyNNEvaluator  # noqa: E402
from chess_engine.search import Searcher  # noqa: E402
from config import SearchConfig  # noqa: E402

WEIGHTS = ROOT / "models" / "nn_eval.npz"
ENGINES = ("nn", "material")
MIN_TIME, MAX_TIME, DEFAULT_TIME = 0.2, 3.0, 1.0
MAX_MOVES = 1000
MAX_BODY_BYTES = 64 * 1024
CACHE_SIZE = 200_000

_searchers: dict[str, Searcher] = {}
_lock = threading.Lock()  # one search at a time per instance; Searcher is not thread-safe


def _searcher(engine: str) -> Searcher:
    """Searchers are built on first use and kept warm. The cache persists across
    requests: static evaluations depend only on the position, so sharing them
    between games is safe."""
    if engine not in _searchers:
        evaluator = CachedEvaluator(NumpyNNEvaluator(WEIGHTS), CACHE_SIZE) if engine == "nn" else MaterialEvaluator()
        _searchers[engine] = Searcher(evaluator, SearchConfig())
    return _searchers[engine]


def game_status(board: chess.Board) -> dict[str, str] | None:
    """Result and reason if the game is over (threefold and fifty-move draws claimed automatically)."""
    outcome = board.outcome()
    if outcome is not None:
        return {"result": outcome.result(), "reason": outcome.termination.name.lower()}
    if board.is_repetition(3):
        return {"result": "1/2-1/2", "reason": "threefold_repetition"}
    if board.halfmove_clock >= 100:
        return {"result": "1/2-1/2", "reason": "fifty_moves"}
    return None


def _error(message: str) -> tuple[int, dict[str, Any]]:
    return 400, {"error": message}


def handle_request(payload: Any) -> tuple[int, dict[str, Any]]:
    """Validate a request, search the position and return (HTTP status, JSON body)."""
    if not isinstance(payload, dict):
        return _error("body must be a JSON object")
    try:
        board = chess.Board(payload.get("fen") or chess.STARTING_FEN)
    except (ValueError, TypeError):
        return _error("invalid FEN")
    moves = payload.get("moves", [])
    if not isinstance(moves, list) or len(moves) > MAX_MOVES:
        return _error(f"moves must be a list of at most {MAX_MOVES} UCI strings")
    for index, uci in enumerate(moves):
        try:
            move = chess.Move.from_uci(str(uci))
        except ValueError:
            return _error(f"move {index + 1} ({uci!r}) is not valid UCI")
        if move not in board.legal_moves:
            return _error(f"move {index + 1} ({uci}) is illegal")
        board.push(move)

    engine = payload.get("engine", "nn")
    if engine not in ENGINES:
        return _error(f"engine must be one of {', '.join(ENGINES)}")
    try:
        time_limit = min(MAX_TIME, max(MIN_TIME, float(payload.get("time", DEFAULT_TIME))))
    except (TypeError, ValueError):
        return _error("time must be a number of seconds")

    status = game_status(board)
    if status is not None:
        return 200, {"move": None, "game_over": status}

    with _lock:
        result = _searcher(engine).search(board, time_limit=time_limit)

    move = result.best_move
    white_sign = 1 if board.turn == chess.WHITE else -1
    response = {
        "move": move.uci(),
        "san": board.san(move),
        "score": result.score,  # centipawns for the engine (the side to move)
        "score_white": result.score * white_sign,
        "mate_in": result.mate_in,  # moves; positive means the engine mates
        "depth": result.depth,
        "nodes": result.nodes,
        "nps": round(result.nps),
        "time": round(result.time, 3),
        "pv": board.variation_san(result.pv) if result.pv else "",
        "engine": engine,
    }
    board.push(move)
    response["fen"] = board.fen()
    response["game_over"] = game_status(board)
    return 200, response


def _send(handler: BaseHTTPRequestHandler, status: int, body: dict[str, Any]) -> None:
    data = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(data)


def handle_post(handler: BaseHTTPRequestHandler) -> None:
    """Read the JSON body of a POST and answer it (shared with the local dev server)."""
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        length = 0
    if length > MAX_BODY_BYTES:
        _send(handler, 413, {"error": "request too large"})
        return
    try:
        payload = json.loads(handler.rfile.read(length) or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        _send(handler, 400, {"error": "body must be JSON"})
        return
    status, body = handle_request(payload)
    _send(handler, status, body)


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel requires this name
    def do_POST(self) -> None:  # noqa: N802
        handle_post(self)

    def do_GET(self) -> None:  # noqa: N802
        _send(self, 200, {"status": "ok", "engines": list(ENGINES), "time_range": [MIN_TIME, MAX_TIME]})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
