"""CachedEvaluator and Zobrist hashing tests."""

from __future__ import annotations

import random

import chess
import chess.polyglot

from chess_engine.evaluators import CachedEvaluator, MaterialEvaluator, is_mate_score
from chess_engine.search import Searcher
from chess_engine.zobrist import zobrist_hash


class CountingEvaluator(MaterialEvaluator):
    """Material evaluator that counts static evaluations (stands in for a network)."""

    def __init__(self) -> None:
        super().__init__()
        self.static_calls = 0

    def static_eval(self, board: chess.Board) -> int:
        self.static_calls += 1
        return super().static_eval(board)

    def network_calls(self) -> int:
        return self.static_calls


def random_boards(count: int, seed: int = 3) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    for _ in range(count):
        board = chess.Board()
        for _ in range(rng.randint(0, 160)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        boards.append(board)
    return boards


def test_fast_zobrist_matches_polyglot() -> None:
    boards = random_boards(500)
    # Add en-passant and castling edge cases.
    boards.append(chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"))
    boards.append(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b Qk - 0 1"))
    for board in boards:
        assert zobrist_hash(board) == chess.polyglot.zobrist_hash(board), board.fen()


def test_cache_returns_same_scores_and_counts_hits() -> None:
    inner = CountingEvaluator()
    cached = CachedEvaluator(inner, max_size=1000)
    boards = random_boards(50)
    first = [cached.evaluate(b) for b in boards]
    second = [cached.evaluate(b) for b in boards]
    assert first == second == [MaterialEvaluator().evaluate(b) for b in boards]
    non_terminal = sum(1 for b in boards if not b.is_game_over(claim_draw=True))
    assert cached.misses == inner.static_calls
    assert cached.hits >= non_terminal  # the whole second pass is served from the cache
    assert 0.0 < cached.hit_rate < 1.0
    assert cached.network_calls() == inner.static_calls


def test_lru_eviction_keeps_recent_entries() -> None:
    inner = CountingEvaluator()
    cached = CachedEvaluator(inner, max_size=2)
    a, b, c = chess.Board(), chess.Board(), chess.Board()
    b.push_san("e4")
    c.push_san("d4")
    cached.evaluate(a)
    cached.evaluate(b)
    cached.evaluate(a)  # a becomes most recently used
    cached.evaluate(c)  # evicts b, the least recently used
    assert len(cached) == 2 and cached.evictions == 1
    calls = inner.static_calls
    cached.evaluate(a)
    assert inner.static_calls == calls  # still cached
    cached.evaluate(b)
    assert inner.static_calls == calls + 1  # was evicted


def test_rule_based_scores_are_not_cached() -> None:
    # The same placement is a draw after a threefold repetition but not before it.
    cached = CachedEvaluator(CountingEvaluator(), max_size=100)
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert cached.evaluate(board) == 500
    for san in ("Ra2", "Ke7", "Ra1", "Ke8") * 2:
        board.push_san(san)
    assert cached.evaluate(board) == 0  # threefold repetition
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert is_mate_score(cached.evaluate(mate))
    assert len(cached) == 1


def test_new_game_clears_cache_but_stats_persist_until_reset() -> None:
    cached = CachedEvaluator(CountingEvaluator(), max_size=100)
    cached.evaluate(chess.Board())
    cached.evaluate(chess.Board())
    cached.new_game()
    assert len(cached) == 0
    stats = cached.stats()
    assert stats["cache_hits"] == 1 and stats["cache_misses"] == 1
    cached.reset_stats()
    assert cached.stats()["cache_hits"] == 0


def test_cache_persists_across_moves_and_cuts_static_calls() -> None:
    board = chess.Board("r1bq1rk1/2pnbppp/p2p1n2/1p2p3/3PP3/1BP2N1P/PP3PP1/RNBQR1K1 w - - 1 11")
    plain, cached_inner = CountingEvaluator(), CountingEvaluator()
    cached = CachedEvaluator(cached_inner, max_size=1_000_000)
    s_plain, s_cached = Searcher(plain), Searcher(cached)
    for _ in range(3):  # three consecutive moves, same node budget for both
        r1 = s_plain.search(board, time_limit=60, max_depth=4)
        r2 = s_cached.search(board, time_limit=60, max_depth=4)
        assert (r1.best_move, r1.score, r1.nodes) == (r2.best_move, r2.score, r2.nodes)  # cache is transparent
        board.push(r1.best_move)
    assert cached_inner.static_calls < plain.static_calls
    assert cached.hit_rate > 0.1
