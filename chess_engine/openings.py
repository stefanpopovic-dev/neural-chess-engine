"""Opening book for self-play matches: 50 short main-line openings.

Each opening is played twice in a match, once with each engine as White, so
neither engine profits from a favourable opening.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class Opening:
    name: str
    moves: str  # space-separated SAN from the initial position

    def board(self) -> chess.Board:
        board = chess.Board()
        for san in self.moves.split():
            board.push_san(san)
        return board


OPENINGS: tuple[Opening, ...] = (
    Opening("Ruy Lopez, Morphy Defence", "e4 e5 Nf3 Nc6 Bb5 a6"),
    Opening("Ruy Lopez, Berlin Defence", "e4 e5 Nf3 Nc6 Bb5 Nf6"),
    Opening("Italian Game, Giuoco Piano", "e4 e5 Nf3 Nc6 Bc4 Bc5"),
    Opening("Two Knights Defence", "e4 e5 Nf3 Nc6 Bc4 Nf6"),
    Opening("Scotch Game", "e4 e5 Nf3 Nc6 d4 exd4"),
    Opening("Four Knights Game", "e4 e5 Nf3 Nc6 Nc3 Nf6"),
    Opening("Petrov Defence", "e4 e5 Nf3 Nf6 Nxe5 d6"),
    Opening("Philidor Defence", "e4 e5 Nf3 d6 d4 Nf6"),
    Opening("Vienna Game", "e4 e5 Nc3 Nf6 f4 d5"),
    Opening("King's Gambit Accepted", "e4 e5 f4 exf4 Nf3 g5"),
    Opening("Bishop's Opening", "e4 e5 Bc4 Nf6 d3 c6"),
    Opening("Sicilian, Najdorf", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"),
    Opening("Sicilian, Dragon", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6"),
    Opening("Sicilian, Classical", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6"),
    Opening("Sicilian, Scheveningen", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6"),
    Opening("Sicilian, Taimanov", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6"),
    Opening("Sicilian, Sveshnikov", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5"),
    Opening("Sicilian, Alapin", "e4 c5 c3 Nf6 e5 Nd5"),
    Opening("Sicilian, Closed", "e4 c5 Nc3 Nc6 g3 g6"),
    Opening("Sicilian, Rossolimo", "e4 c5 Nf3 Nc6 Bb5 g6"),
    Opening("French, Winawer", "e4 e6 d4 d5 Nc3 Bb4"),
    Opening("French, Classical", "e4 e6 d4 d5 Nc3 Nf6"),
    Opening("French, Advance", "e4 e6 d4 d5 e5 c5"),
    Opening("French, Tarrasch", "e4 e6 d4 d5 Nd2 c5"),
    Opening("Caro-Kann, Classical", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5"),
    Opening("Caro-Kann, Advance", "e4 c6 d4 d5 e5 Bf5"),
    Opening("Pirc Defence", "e4 d6 d4 Nf6 Nc3 g6"),
    Opening("Modern Defence", "e4 g6 d4 Bg7 Nc3 d6"),
    Opening("Scandinavian Defence", "e4 d5 exd5 Qxd5 Nc3 Qa5"),
    Opening("Alekhine Defence", "e4 Nf6 e5 Nd5 d4 d6"),
    Opening("Queen's Gambit Declined", "d4 d5 c4 e6 Nc3 Nf6"),
    Opening("QGD, Exchange Variation", "d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5"),
    Opening("Queen's Gambit Accepted", "d4 d5 c4 dxc4 Nf3 Nf6"),
    Opening("Slav Defence", "d4 d5 c4 c6 Nf3 Nf6"),
    Opening("Semi-Slav Defence", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6"),
    Opening("Nimzo-Indian Defence", "d4 Nf6 c4 e6 Nc3 Bb4"),
    Opening("Queen's Indian Defence", "d4 Nf6 c4 e6 Nf3 b6"),
    Opening("Bogo-Indian Defence", "d4 Nf6 c4 e6 Nf3 Bb4+"),
    Opening("King's Indian Defence", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6"),
    Opening("Grunfeld Defence", "d4 Nf6 c4 g6 Nc3 d5"),
    Opening("Modern Benoni", "d4 Nf6 c4 c5 d5 e6"),
    Opening("Benko Gambit", "d4 Nf6 c4 c5 d5 b5"),
    Opening("Dutch Defence", "d4 f5 g3 Nf6 Bg2 e6"),
    Opening("London System", "d4 d5 Bf4 Nf6 e3 c5"),
    Opening("Trompowsky Attack", "d4 Nf6 Bg5 Ne4 Bf4 c5"),
    Opening("Catalan Opening", "d4 Nf6 c4 e6 g3 d5"),
    Opening("English, Symmetrical", "c4 c5 Nc3 Nc6 g3 g6"),
    Opening("English, Reversed Sicilian", "c4 e5 Nc3 Nf6 Nf3 Nc6"),
    Opening("Reti Opening", "Nf3 d5 c4 e6 g3 Nf6"),
    Opening("Bird Opening", "f4 d5 Nf3 Nf6 e3 g6"),
)
