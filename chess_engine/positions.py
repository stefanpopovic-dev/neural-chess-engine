"""Fixed test positions for the depth benchmark (and, later, match openings)."""

from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class TestPosition:
    name: str
    phase: str  # "opening", "middlegame" or "endgame"
    fen: str


BENCHMARK_POSITIONS: tuple[TestPosition, ...] = (
    # Openings: the start position and popular early tabiyas.
    TestPosition("start", "opening", chess.STARTING_FEN),
    TestPosition("ruy_lopez", "opening", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    TestPosition("najdorf", "opening", "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    TestPosition("qgd", "opening", "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4"),
    # Middlegames reached from main-line theory.
    TestPosition("ruy_lopez_closed", "middlegame", "r1bq1rk1/2pnbppp/p2p1n2/1p2p3/3PP3/1BP2N1P/PP3PP1/RNBQR1K1 w - - 1 11"),
    TestPosition("qgd_orthodox", "middlegame", "r1bq1rk1/pp1nbppp/2p1p3/3n2B1/2BP4/2N1PN2/PP3PPP/2RQK2R w K - 1 10"),
    TestPosition("najdorf_english_attack", "middlegame", "r2q1rk1/3nbppp/p2pbn2/1p2p3/4P1P1/1NN1BP2/PPPQ3P/2KR1B1R w - - 0 12"),
    TestPosition("kings_indian", "middlegame", "r1bq1rk1/pppnn1bp/3p4/3Pp1p1/2P1Pp2/2N2P2/PP2BBPP/R2QNRK1 w - - 0 13"),
    # Endgames: rook, pawn and a classic deep-search test (Fine #70).
    TestPosition("lucena", "endgame", "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1"),
    TestPosition("rook_ending", "endgame", "8/5pk1/6p1/8/3R4/6P1/r4PK1/8 w - - 0 1"),
    TestPosition("pawn_race", "endgame", "8/pp3k2/8/2P5/1P3K2/8/8/8 w - - 0 1"),
    TestPosition("fine_70", "endgame", "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1"),
)

PHASES = ("opening", "middlegame", "endgame")
