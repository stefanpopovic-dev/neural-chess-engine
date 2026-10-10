"""Feature encoding tests."""

from __future__ import annotations

import chess
import numpy as np

from chess_engine.encoding import (
    CASTLING_INDEX,
    EP_INDEX,
    NUM_FEATURES,
    STM_INDEX,
    encode_board,
    encode_fens_packed,
    encode_packed,
    unpack,
)

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def plane(features: np.ndarray, index: int) -> np.ndarray:
    return features[index * 64 : (index + 1) * 64]


def test_start_position() -> None:
    f = encode_board(chess.Board())
    assert f.shape == (NUM_FEATURES,) and f.dtype == np.float32
    assert f[:768].sum() == 32
    assert f[STM_INDEX] == 1
    assert f[CASTLING_INDEX : CASTLING_INDEX + 4].tolist() == [1, 1, 1, 1]
    assert f[EP_INDEX:].sum() == 0
    assert np.flatnonzero(plane(f, 0)).tolist() == list(range(8, 16))  # our pawns on rank 2
    assert np.flatnonzero(plane(f, 6)).tolist() == list(range(48, 56))  # their pawns on rank 7


def test_black_to_move_is_flipped() -> None:
    board = chess.Board()
    board.push_san("e4")
    f = encode_board(board)
    assert f[STM_INDEX] == 0
    # Black's pawns now appear on "rank 2"; White's e4 pawn appears on e5 (square 36).
    assert np.flatnonzero(plane(f, 0)).tolist() == list(range(8, 16))
    assert chess.E5 in np.flatnonzero(plane(f, 6)).tolist()


def test_mirror_invariance() -> None:
    for fen in (KIWIPETE, "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"):
        board = chess.Board(fen)
        a, b = encode_board(board), encode_board(board.mirror())
        diff = np.flatnonzero(a != b).tolist()
        assert diff == [STM_INDEX]


def test_castling_rights_from_perspective() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1")  # white K-side, black Q-side
    f = encode_board(board)
    # Black to move: ours = black (queenside only), theirs = white (kingside only).
    assert f[CASTLING_INDEX : CASTLING_INDEX + 4].tolist() == [0, 1, 1, 0]


def test_en_passant_only_when_legal() -> None:
    board = chess.Board()
    for san in ("e4", "a6", "e5", "d5"):
        board.push_san(san)
    f = encode_board(board)
    assert f[EP_INDEX:].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]  # d-file
    board = chess.Board()
    board.push_san("e4")  # ep square set, but no black pawn can capture
    assert encode_board(board)[EP_INDEX:].sum() == 0


def test_packed_round_trip() -> None:
    fens = [chess.STARTING_FEN, KIWIPETE, "8/8/4k3/8/2pP4/8/8/4K3 b - d3 0 1"]
    packed = encode_fens_packed(fens)
    assert packed.shape == (3, 98) and packed.dtype == np.uint8
    for fen, row in zip(fens, unpack(packed)):
        assert np.array_equal(row, encode_board(chess.Board(fen)).astype(np.uint8))
    assert np.array_equal(unpack(encode_packed(chess.Board(KIWIPETE))), encode_board(chess.Board(KIWIPETE)))


def test_encode_into_buffer() -> None:
    buf = np.full(NUM_FEATURES, 7, dtype=np.float32)
    out = encode_board(chess.Board(), buf)
    assert out is buf and set(np.unique(buf)) <= {0.0, 1.0}
