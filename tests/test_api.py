"""Web demo function (api/move.py): request validation and engine replies."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import chess
import pytest

_spec = importlib.util.spec_from_file_location("api_move", Path(__file__).resolve().parent.parent / "api" / "move.py")
assert _spec and _spec.loader
api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(api)


@pytest.mark.parametrize("engine", ["nn", "material"])
def test_returns_a_legal_reply(engine: str) -> None:
    status, body = api.handle_request({"moves": ["e2e4"], "engine": engine, "time": 0.2})
    assert status == 200
    board = chess.Board()
    board.push_uci("e2e4")
    assert chess.Move.from_uci(body["move"]) in board.legal_moves
    assert body["san"] == board.san(chess.Move.from_uci(body["move"]))
    assert body["score_white"] == -body["score"]  # Black (the engine) is to move
    assert body["depth"] >= 1 and body["nodes"] > 0
    board.push_uci(body["move"])
    assert body["fen"] == board.fen()
    assert body["game_over"] is None


def test_engine_finds_mate_and_reports_game_over() -> None:
    status, body = api.handle_request({"fen": "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "time": 0.5})
    assert status == 200
    assert body["san"] == "Rd8#" and body["mate_in"] == 1
    assert body["game_over"] == {"result": "1-0", "reason": "checkmate"}


def test_finished_game_returns_no_move() -> None:
    status, body = api.handle_request({"moves": ["f2f3", "e7e5", "g2g4", "d8h4"]})
    assert status == 200
    assert body == {"move": None, "game_over": {"result": "0-1", "reason": "checkmate"}}


def test_threefold_repetition_is_game_over() -> None:
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2
    status, body = api.handle_request({"moves": shuffle})
    assert status == 200
    assert body["game_over"]["reason"] == "threefold_repetition"


@pytest.mark.parametrize(
    "payload, message",
    [
        (["e2e4"], "JSON object"),
        ({"fen": "not a fen"}, "invalid FEN"),
        ({"moves": "e2e4"}, "list"),
        ({"moves": ["e2e5"]}, "illegal"),
        ({"moves": ["zz"]}, "not valid UCI"),
        ({"engine": "stockfish"}, "engine must be"),
        ({"time": "fast"}, "number of seconds"),
        ({"moves": ["e2e4"] * (api.MAX_MOVES + 1)}, "at most"),
    ],
)
def test_rejects_bad_requests(payload: object, message: str) -> None:
    status, body = api.handle_request(payload)
    assert status == 400
    assert message in body["error"]


def test_think_time_is_clamped() -> None:
    status, body = api.handle_request({"moves": ["e2e4"], "engine": "material", "time": 60})
    assert status == 200
    assert body["time"] <= api.MAX_TIME + 0.2
