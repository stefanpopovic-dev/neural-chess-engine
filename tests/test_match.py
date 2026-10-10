"""Match script tests: openings, scheduling, adjudication, Elo maths, a real game."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import chess
import chess.pgn
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import match  # noqa: E402
from chess_engine.openings import OPENINGS  # noqa: E402


def test_openings_are_legal_and_distinct() -> None:
    positions = set()
    for opening in OPENINGS:
        board = opening.board()  # raises on an illegal SAN move
        assert not board.is_game_over()
        positions.add(board.epd())
    assert len(OPENINGS) == 50
    assert len(positions) == len(OPENINGS)


def test_schedule_swaps_colours_per_opening() -> None:
    specs = match.schedule(6, "a", "b")
    assert [s.opening for s in specs] == [0, 0, 1, 1, 2, 2]
    assert [(s.white, s.black) for s in specs[:2]] == [("a", "b"), ("b", "a")]
    assert [s.a_is_white for s in specs] == [True, False] * 3


def test_schedule_wraps_openings() -> None:
    specs = match.schedule(len(OPENINGS) * 2 + 2, "a", "b")
    assert specs[-1].opening == 0


@pytest.mark.parametrize(
    "setup, expected",
    [
        (["f3", "e5", "g4", "Qh4#"], ("checkmate", "0-1")),
        (["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"], ("threefold_repetition", "1/2-1/2")),
        (["e4"], None),
    ],
)
def test_adjudicate(setup: list[str], expected: tuple[str, str] | None) -> None:
    board = chess.Board()
    for san in setup:
        board.push_san(san)
    assert match.adjudicate(board, max_plies=200) == expected


def test_adjudicate_move_cap_and_fifty_moves() -> None:
    board = chess.Board()
    board.push_san("e4")
    assert match.adjudicate(board, max_plies=1) == ("move_cap", "1/2-1/2")
    assert match.adjudicate(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 100 90"), 500) == ("fifty_moves", "1/2-1/2")
    assert match.adjudicate(chess.Board("8/8/4k3/8/8/3K4/8/8 w - - 0 1"), 500)[0] == "insufficient_material"


def test_elo_estimate() -> None:
    even = match.elo_estimate(10, 10, 10)
    assert even["score"] == 0.5 and even["elo"] == 0.0 and even["los"] == 0.5
    assert even["margin"] > 0
    strong = match.elo_estimate(30, 10, 10)
    assert strong["score"] == pytest.approx(0.7)
    assert strong["elo"] == pytest.approx(-400 * __import__("math").log10(1 / 0.7 - 1))
    assert strong["elo_low"] < strong["elo"] < strong["elo_high"]
    assert strong["los"] > 0.99
    perfect = match.elo_estimate(8, 0, 0)  # 100% is clamped to stay finite
    assert perfect["elo"] < float("inf") and perfect["elo"] > 300
    assert match.elo_estimate(0, 0, 0)["games"] == 0


def test_more_games_shrink_the_margin() -> None:
    assert match.elo_estimate(40, 40, 20)["margin"] < match.elo_estimate(4, 4, 2)["margin"]


def test_score_for_a() -> None:
    assert match.score_for_a({"result": "1-0", "a_is_white": True}) == 1.0
    assert match.score_for_a({"result": "1-0", "a_is_white": False}) == 0.0
    assert match.score_for_a({"result": "0-1", "a_is_white": False}) == 1.0
    assert match.score_for_a({"result": "1/2-1/2", "a_is_white": True}) == 0.5


def test_play_game_produces_valid_record() -> None:
    spec = match.schedule(1, "material", "material")[0]
    record = match.play_game(spec, time_limit=0.01, max_plies=24, early_stop=1.0)
    assert record["termination"] in ("move_cap", "checkmate", "stalemate", "threefold_repetition")
    assert record["plies"] <= 24
    game = chess.pgn.read_game(io.StringIO(record["pgn"]))
    assert game is not None
    assert game.headers["Result"] == record["result"]
    assert game.headers["Opening"] == OPENINGS[0].name
    assert len(list(game.mainline_moves())) == record["plies"]
    assert record["stats"]["white"]["moves"] > 0


def test_summary_and_resume_records(tmp_path: Path) -> None:
    records = [
        {"game_id": 0, "result": "1-0", "a_is_white": True, "termination": "checkmate", "plies": 40,
         "seconds": 4.0, "pgn": "[Round \"1\"]\n\n1. e4 1-0",
         "stats": {c: {"depth": 4, "nps": 1000.0, "evals_per_move": 10.0, "network_calls_per_move": 5.0,
                       "cache_hit_rate": None} for c in ("white", "black")}},
        {"game_id": 1, "result": "1-0", "a_is_white": False, "termination": "move_cap", "plies": 200,
         "seconds": 20.0, "pgn": "[Round \"2\"]\n\n1. d4 1-0",
         "stats": {c: {"depth": 6, "nps": 3000.0, "evals_per_move": 30.0, "network_calls_per_move": 0.0,
                       "cache_hit_rate": 0.5} for c in ("white", "black")}},
    ]
    summary = match.summarize(records, "a", "b")
    assert (summary["wins"], summary["draws"], summary["losses"]) == (1, 0, 1)
    assert summary["a_as_white"] == {"wins": 1, "draws": 0, "losses": 0}
    assert summary["terminations"] == {"checkmate": 1, "move_cap": 1}
    assert "Elo diff" in match.summary_markdown(summary, {"time_limit": 0.1, "max_plies": 200, "workers": 1,
                                                          "openings": 50})

    path = tmp_path / "results.jsonl"
    import json

    path.write_text(json.dumps(records[0]) + "\n" + json.dumps(records[1])[:30])  # torn last line
    assert [r["game_id"] for r in match.load_records(path)] == [0]
    pgn = tmp_path / "games.pgn"
    match.rewrite_pgn(pgn, records[:1])
    assert pgn.read_text().count("[Round") == 1


def test_duration_estimate() -> None:
    expected, upper = match.estimate_duration(10, 1.0, 200, 2, [])
    assert 0 < expected < upper
    assert match.format_duration(30) == "1 min"
    assert match.format_duration(3700) == "1h 02m"
