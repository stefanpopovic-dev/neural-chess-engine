"""Benchmark positions must be valid and cover every phase."""

from __future__ import annotations

import chess

from chess_engine.positions import BENCHMARK_POSITIONS, PHASES


def test_benchmark_positions_valid() -> None:
    names = set()
    for pos in BENCHMARK_POSITIONS:
        board = chess.Board(pos.fen)
        assert board.is_valid(), pos.name
        assert not board.is_game_over(), pos.name
        assert pos.phase in PHASES
        names.add(pos.name)
    assert len(names) == len(BENCHMARK_POSITIONS)
    assert {p.phase for p in BENCHMARK_POSITIONS} == set(PHASES)
