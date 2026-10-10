"""Evaluator tests: perspective, symmetry, checkmate and every draw rule."""

from __future__ import annotations

import chess
import pytest

from chess_engine.evaluators import (
    DRAW_SCORE,
    MATE_SCORE,
    MaterialEvaluator,
    is_mate_score,
    terminal_score,
)
from config import MaterialConfig


@pytest.fixture
def ev() -> MaterialEvaluator:
    return MaterialEvaluator()


def test_start_position_is_balanced(ev: MaterialEvaluator) -> None:
    assert ev.evaluate(chess.Board()) == 0


def test_side_to_move_perspective(ev: MaterialEvaluator) -> None:
    # White has an extra knight (pawns keep it from being an insufficient-material draw).
    white_to_move = chess.Board("4k3/4p3/8/8/8/8/4P3/1N2K3 w - - 0 1")
    black_to_move = chess.Board("4k3/4p3/8/8/8/8/4P3/1N2K3 b - - 0 1")
    assert ev.evaluate(white_to_move) == 320
    assert ev.evaluate(black_to_move) == -320


def test_mirrored_position_scores_the_same(ev: MaterialEvaluator) -> None:
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    board.remove_piece_at(chess.F6)  # make it unbalanced
    assert ev.evaluate(board) == ev.evaluate(board.mirror()) == 320


def test_custom_piece_values() -> None:
    ev = MaterialEvaluator(MaterialConfig(pawn=1, knight=3, bishop=3, rook=5, queen=9))
    assert ev.evaluate(chess.Board("4k3/8/8/8/8/8/PPP5/R3K3 w - - 0 1")) == 8


def test_checkmate_scored_as_loss_for_side_to_move(ev: MaterialEvaluator) -> None:
    board = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        board.push_san(san)
    score = ev.evaluate(board)
    assert is_mate_score(score) and score < 0
    assert score == -(MATE_SCORE - board.ply())


def test_faster_mates_score_better_for_the_winner(ev: MaterialEvaluator) -> None:
    mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 {}"
    early = ev.evaluate(chess.Board(mate.format(3)))  # mated at ply 4
    late = ev.evaluate(chess.Board(mate.format(10)))  # same mate at ply 18
    # The mated side (to move) prefers the later mate; the winner the earlier one.
    assert early < late < 0


def test_stalemate_is_draw(ev: MaterialEvaluator) -> None:
    board = chess.Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert ev.evaluate(board) == DRAW_SCORE


@pytest.mark.parametrize(
    "fen",
    [
        "8/8/4k3/8/8/3K4/8/8 w - - 0 1",  # K vs K
        "8/8/4k3/8/8/3KB3/8/8 w - - 0 1",  # K+B vs K
        "8/8/4k3/8/8/3KN3/8/8 b - - 0 1",  # K+N vs K
    ],
)
def test_insufficient_material_is_draw(ev: MaterialEvaluator, fen: str) -> None:
    assert ev.evaluate(chess.Board(fen)) == DRAW_SCORE


def test_fifty_move_rule_is_draw(ev: MaterialEvaluator) -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 100 80")
    assert ev.evaluate(board) == DRAW_SCORE
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 99 80")
    assert ev.evaluate(board) == 500


def test_checkmate_beats_fifty_move_rule(ev: MaterialEvaluator) -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 99 80")
    board.push_san("Rd8#")
    assert board.halfmove_clock == 100
    assert ev.evaluate(board) < 0 and is_mate_score(ev.evaluate(board))


def test_threefold_repetition_is_draw(ev: MaterialEvaluator) -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")  # white is a rook up
    shuffle = ("Ra2", "Ke7", "Ra1", "Ke8")
    for san in shuffle:
        board.push_san(san)
    assert ev.evaluate(board) == 500  # second occurrence: not yet a draw
    for san in shuffle:
        board.push_san(san)
    assert board.is_repetition(3)
    assert ev.evaluate(board) == DRAW_SCORE


def test_terminal_score_none_for_normal_position() -> None:
    assert terminal_score(chess.Board()) is None


def test_counters(ev: MaterialEvaluator) -> None:
    ev.evaluate(chess.Board())
    ev.evaluate(chess.Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1"))
    assert ev.stats() == {"calls": 2, "terminal": 1, "network_calls": 0}
    ev.reset_stats()
    assert ev.calls == 0
