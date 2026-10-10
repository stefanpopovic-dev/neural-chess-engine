"""Fast Polyglot Zobrist hashing.

Produces exactly the same 64-bit keys as ``chess.polyglot.zobrist_hash`` for
positions reached by legal play, but about twice as fast: it walks the six
piece bitboards directly instead of calling ``piece_type_at`` for every
occupied square, and reads castling rights from the raw rook-square bitmask
(which python-chess keeps clean during normal play).
"""

from __future__ import annotations

import chess
import chess.polyglot

_ARRAY = chess.polyglot.POLYGLOT_RANDOM_ARRAY
_CASTLING = (
    (chess.BB_H1, _ARRAY[768]),
    (chess.BB_A1, _ARRAY[769]),
    (chess.BB_H8, _ARRAY[770]),
    (chess.BB_A8, _ARRAY[771]),
)
_TURN = _ARRAY[780]
_NOT_FILE_A = ~chess.BB_FILE_A & chess.BB_ALL
_NOT_FILE_H = ~chess.BB_FILE_H & chess.BB_ALL


def zobrist_hash(board: chess.Board) -> int:
    """Polyglot Zobrist key of ``board`` (pieces, castling, en passant, turn)."""
    arr = _ARRAY
    white = board.occupied_co[chess.WHITE]
    h = 0
    # Polyglot piece index = 2 * (piece_type - 1) + (1 if white else 0); 64 squares each.
    base = 0
    for bb in (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings):
        x = bb & ~white
        while x:
            lsb = x & -x
            h ^= arr[base + lsb.bit_length() - 1]
            x ^= lsb
        x = bb & white
        while x:
            lsb = x & -x
            h ^= arr[base + 63 + lsb.bit_length()]  # base + 64 + square
            x ^= lsb
        base += 128

    rights = board.castling_rights
    if rights:
        for mask, value in _CASTLING:
            if rights & mask:
                h ^= value

    ep = board.ep_square
    if ep is not None:
        # Polyglot hashes the en-passant file only if a pawn could capture (pseudo-legally).
        behind = chess.BB_SQUARES[ep] >> 8 if board.turn == chess.WHITE else chess.BB_SQUARES[ep] << 8
        attackers = ((behind << 1) & _NOT_FILE_A) | ((behind >> 1) & _NOT_FILE_H)
        if attackers & board.pawns & board.occupied_co[board.turn]:
            h ^= arr[772 + chess.square_file(ep)]

    if board.turn == chess.WHITE:
        h ^= _TURN
    return h
