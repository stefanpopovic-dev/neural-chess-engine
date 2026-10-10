"""Search tests: forced mates, legality, time control, completed-depth semantics."""

from __future__ import annotations

import random
import time

import chess
import pytest

from chess_engine.evaluators import MaterialEvaluator
from chess_engine.positions import BENCHMARK_POSITIONS
from chess_engine.search import Searcher
from config import SearchConfig


def make_searcher(**overrides) -> Searcher:
    return Searcher(MaterialEvaluator(), SearchConfig(**overrides))


MATE_IN_ONE = [
    ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", {"d1d8"}),  # back-rank mate
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", {"h5f7"}),  # scholar's mate
    ("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2", {"d8h4"}),  # fool's mate
    ("7k/8/6K1/8/8/8/8/R7 w - - 0 1", {"a1a8"}),  # K+R vs K
]

MATE_IN_TWO = [
    ("kbK5/pp6/1P6/8/8/8/8/R7 w - - 0 1", {"a1a6"}),  # 1.Ra6! bxa6 2.b7#
    ("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1", {"d5f6"}),  # 1.Nf6+ gxf6 2.Bxf7#
    ("r1b2k1r/ppppq3/5N1p/4P2Q/4PP2/1B6/PP5P/n2K2R1 w - - 1 1", {"h5h6"}),  # 1.Qxh6+ Rxh6 2.Rg8#
]


def test_finds_mate_in_three() -> None:
    board = chess.Board("2r3k1/p4p2/3Rp2p/1p2P1pK/8/1P4P1/P3Q2P/1q6 b - - 0 1")
    result = make_searcher().search(board, time_limit=10.0)
    assert result.mate_in == 3


@pytest.mark.parametrize("fen, solutions", MATE_IN_ONE)
def test_finds_mate_in_one(fen: str, solutions: set[str]) -> None:
    result = make_searcher().search(chess.Board(fen), time_limit=2.0)
    assert result.best_move.uci() in solutions
    assert result.mate_in == 1


@pytest.mark.parametrize("fen, solutions", MATE_IN_TWO)
def test_finds_mate_in_two(fen: str, solutions: set[str]) -> None:
    result = make_searcher().search(chess.Board(fen), time_limit=5.0)
    assert result.best_move.uci() in solutions
    assert result.mate_in == 2


def test_mated_side_reports_negative_mate() -> None:
    # Black to move; white threatens Qh7# / Qg7# no matter what (mate in 1 for white after any reply).
    board = chess.Board("6k1/5p1p/5PpQ/8/8/8/8/6K1 b - - 0 1")
    result = make_searcher().search(board, time_limit=2.0)
    assert result.mate_in is not None and result.mate_in < 0


def test_prefers_faster_mate() -> None:
    # Ra8# and Qd8# mate at once; dozens of other moves mate in two.
    board = chess.Board("7k/8/6K1/8/8/8/8/R2Q4 w - - 0 1")
    result = make_searcher().search(board, time_limit=2.0)
    assert result.mate_in == 1
    board.push(result.best_move)
    assert board.is_checkmate()


def random_positions(count: int, seed: int = 1) -> list[chess.Board]:
    """Positions reached by random play, skipping finished games."""
    rng = random.Random(seed)
    boards = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(rng.randint(0, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            boards.append(board)
    return boards


def test_never_returns_illegal_move() -> None:
    searcher = make_searcher()
    boards = random_positions(40) + [chess.Board(p.fen) for p in BENCHMARK_POSITIONS]
    for board in boards:
        fen_before = board.fen()
        result = searcher.search(board, time_limit=0.05)
        assert result.best_move in board.legal_moves, board.fen()
        assert board.fen() == fen_before  # caller's board untouched


def test_tiny_time_limit_still_returns_legal_move() -> None:
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    result = make_searcher().search(board, time_limit=0.0001)
    assert result.depth >= 1
    assert result.best_move in board.legal_moves


@pytest.mark.parametrize("limit", [0.2, 0.5, 1.0])
def test_respects_time_limit(limit: float) -> None:
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    start = time.perf_counter()
    result = make_searcher().search(board, time_limit=limit)
    elapsed = time.perf_counter() - start
    assert elapsed < limit + 0.1, f"took {elapsed:.3f}s for a {limit}s limit"
    assert result.depth >= 2


def test_returns_move_from_last_completed_depth() -> None:
    board = chess.Board(BENCHMARK_POSITIONS[4].fen)
    result = make_searcher().search(board, time_limit=0.5)
    last = result.iterations[-1]
    assert result.depth == last.depth == len(result.iterations)
    assert result.best_move == last.move
    assert result.score == last.score
    assert result.pv[0] == result.best_move


def test_node_limited_search_is_deterministic() -> None:
    board = chess.Board(BENCHMARK_POSITIONS[5].fen)
    a = make_searcher().search(board, time_limit=60, max_nodes=20_000)
    b = make_searcher().search(board, time_limit=60, max_nodes=20_000)
    assert (a.best_move, a.score, a.depth, a.nodes) == (b.best_move, b.score, b.depth, b.nodes)


def test_counts_nodes_and_evaluations() -> None:
    evaluator = MaterialEvaluator()
    result = Searcher(evaluator).search(chess.Board(), time_limit=10, max_depth=3)
    assert result.depth == 3
    assert result.nodes > result.qnodes > 0
    assert 0 < result.evals <= result.nodes
    assert evaluator.calls == result.evals
    assert result.network_calls == 0


def test_max_depth_and_no_quiescence() -> None:
    result = make_searcher(quiescence=False).search(chess.Board(), time_limit=10, max_depth=2)
    assert result.depth == 2
    assert result.qnodes == 0
    # Depth 1: 20 leaves. Depth 2: the first move needs all 20 replies; every
    # other move is refuted (score >= beta) by its first reply under alpha-beta.
    assert result.evals == 20 + 20 + 19


def test_wins_hanging_queen() -> None:
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p1q1/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 1")
    result = make_searcher().search(board, time_limit=1.0)
    assert result.best_move == chess.Move.from_uci("c1g5")


def test_avoids_stalemate_when_winning() -> None:
    # Qb6?? / Qc7?? stalemate; any sane move keeps the win.
    board = chess.Board("k7/8/1K6/8/8/8/8/2Q5 w - - 0 1")
    result = make_searcher().search(board, time_limit=1.0)
    board.push(result.best_move)
    assert not board.is_stalemate()


def test_repetition_scored_as_draw_when_losing() -> None:
    # Black (a queen down) can repeat an earlier position with ...Kh8.
    board = chess.Board("6k1/8/8/8/8/8/8/1Q4K1 w - - 0 1")
    for san in ("Kh1", "Kh8", "Kg1", "Kg8", "Kh1"):
        board.push_san(san)
    result = make_searcher().search(board, time_limit=1.0)
    assert result.best_move == chess.Move.from_uci("g8h8")
    assert result.score == 0

    # Without the history the same position is simply lost.
    fresh = chess.Board(board.fen())
    assert make_searcher().search(fresh, time_limit=1.0).score < -500


def test_no_legal_moves_returns_none() -> None:
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")  # fool's mate
    result = make_searcher().search(board, time_limit=0.1)
    assert result.best_move is None
    assert result.score < 0 and result.depth == 0
