"""Board -> feature vector for the neural-network evaluator.

Features are encoded from the side to move's perspective: the side to move's
pieces come first, and when black is to move the board is flipped vertically so
"our" pawns always advance up the board. A position and its colour-mirrored
twin therefore encode identically (apart from the side-to-move bit), which
halves what the network has to learn.

Layout (781 features):
  [0, 768)   one-hot piece-square: 12 planes x 64 squares,
             plane = owner * 6 + (piece_type - 1)  (owner 0 = side to move, 1 = opponent),
             square index a1=0 ... h8=63 after the perspective flip
  768        side to move is white (1) or black (0)
  769..772   castling rights: ours kingside, ours queenside, theirs kingside, theirs queenside
  773..780   en-passant file, one-hot, set only when an en-passant capture is legal

Positions are stored compactly as 98 bytes: the 12 plane bitboards in
little-endian order (96 bytes) followed by the 13 extra bits. Unpacking those
bytes bit-by-bit (little-endian bit order) yields exactly the feature vector.
"""

from __future__ import annotations

from collections.abc import Iterable

import chess
import numpy as np

NUM_PIECE_SQUARE = 768
STM_INDEX = 768
CASTLING_INDEX = 769
EP_INDEX = 773
NUM_FEATURES = 781
PACKED_BYTES = (NUM_FEATURES + 7) // 8  # 98
ENCODING_VERSION = 1

_PIECE_ATTRS = ("pawns", "knights", "bishops", "rooks", "queens", "kings")


def _planes_and_extras(board: chess.Board) -> tuple[list[int], int]:
    """The 12 perspective bitboards and the 13 extra feature bits (as an int)."""
    us = board.turn
    ours = board.occupied_co[us]
    theirs = board.occupied_co[not us]
    pieces = [board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings]
    planes = [bb & ours for bb in pieces] + [bb & theirs for bb in pieces]
    if us == chess.BLACK:
        planes = [chess.flip_vertical(bb) for bb in planes]

    extras = 1 if us == chess.WHITE else 0
    them = not us
    if board.has_kingside_castling_rights(us):
        extras |= 1 << 1
    if board.has_queenside_castling_rights(us):
        extras |= 1 << 2
    if board.has_kingside_castling_rights(them):
        extras |= 1 << 3
    if board.has_queenside_castling_rights(them):
        extras |= 1 << 4
    if board.ep_square is not None and board.has_legal_en_passant():
        extras |= 1 << (5 + chess.square_file(board.ep_square))
    return planes, extras


def encode_packed(board: chess.Board) -> np.ndarray:
    """Compact 98-byte encoding of ``board`` (see module docstring)."""
    planes, extras = _planes_and_extras(board)
    out = np.empty(PACKED_BYTES, dtype=np.uint8)
    out[:96] = np.array(planes, dtype="<u8").view(np.uint8)
    out[96] = extras & 0xFF
    out[97] = extras >> 8
    return out


def unpack(packed: np.ndarray) -> np.ndarray:
    """(N, 98) packed bytes -> (N, 781) uint8 features. Also accepts a single row."""
    bits = np.unpackbits(packed, axis=-1, bitorder="little")
    return bits[..., :NUM_FEATURES]


def encode_board(board: chess.Board, out: np.ndarray | None = None) -> np.ndarray:
    """Float32 feature vector of length 781, written into ``out`` if given."""
    if out is None:
        out = np.empty(NUM_FEATURES, dtype=np.float32)
    planes, extras = _planes_and_extras(board)
    out[:NUM_PIECE_SQUARE] = np.unpackbits(np.array(planes, dtype="<u8").view(np.uint8), bitorder="little")
    out[NUM_PIECE_SQUARE:] = [(extras >> i) & 1 for i in range(NUM_FEATURES - NUM_PIECE_SQUARE)]
    return out


def encode_fens_packed(fens: Iterable[str]) -> np.ndarray:
    """Packed encodings for many FENs: (N, 98) uint8."""
    rows = [encode_packed(chess.Board(fen)) for fen in fens]
    return np.stack(rows) if rows else np.empty((0, PACKED_BYTES), dtype=np.uint8)
