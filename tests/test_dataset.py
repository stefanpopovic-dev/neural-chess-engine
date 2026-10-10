"""Dataset builder tests: score conversion and resumable PGN sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import chess.engine
import pytest
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_dataset  # noqa: E402
from config import DatasetConfig  # noqa: E402

GAME_MOVES = [
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7 c4 c6 cxb5 axb5 Nc3 Bb7",
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 Nbd7 Rc1 c6 Bd3 dxc4 Bxc4 Nd5 Bxe7 Qxe7 O-O Nxc3 Rxc3 e5",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 O-O O-O-O Nbd7 g4 b5 g5 b4",
    "f3 e5 g4 Qh4#",  # too short to yield positions after the skipped plies
]


def write_pgn_zst(path: Path, games: list[str]) -> None:
    text = ""
    for i, moves in enumerate(games):
        board = chess.Board()
        numbered = []
        for j, san in enumerate(moves.split()):
            if j % 2 == 0:
                numbered.append(f"{j // 2 + 1}.")
            numbered.append(san)
            board.push_san(san)
        result = board.result(claim_draw=True)
        text += f'[Event "Test {i}"]\n[Result "{result}"]\n\n{" ".join(numbered)} {result}\n\n'
    path.write_bytes(zstandard.ZstdCompressor().compress(text.encode()))


def test_convert_score_perspective_mate_and_clip() -> None:
    pov = chess.engine.PovScore(chess.engine.Cp(150), chess.WHITE)
    assert build_dataset.convert_score(pov, chess.BLACK, 10_000, 2000) == (-150, "cp:-150")
    mate = chess.engine.PovScore(chess.engine.Mate(3), chess.BLACK)
    assert build_dataset.convert_score(mate, chess.BLACK, 10_000, 2000) == (2000, "mate:3")
    assert build_dataset.convert_score(mate, chess.WHITE, 10_000, 2000) == (-2000, "mate:-3")
    big = chess.engine.PovScore(chess.engine.Cp(-5000), chess.WHITE)
    assert build_dataset.convert_score(big, chess.WHITE, 10_000, 2000)[0] == -2000


def test_position_key_ignores_move_counters() -> None:
    assert build_dataset.position_key(chess.STARTING_FEN) == build_dataset.position_key(
        chess.STARTING_FEN.replace(" 0 1", " 7 30")
    )


def make_cfg(tmp_path: Path, **overrides) -> DatasetConfig:
    return DatasetConfig(
        positions_file=str(tmp_path / "positions.txt"),
        labels_file=str(tmp_path / "labels.tsv"),
        dataset_file=str(tmp_path / "dataset.npz"),
        **overrides,
    )


def test_sampling_filters_and_resumes(tmp_path: Path) -> None:
    pgn = tmp_path / "games.pgn.zst"
    write_pgn_zst(pgn, GAME_MOVES)
    cfg = make_cfg(tmp_path, skip_plies=10, max_per_game=3, skip_captures=True)

    path = build_dataset.stage_sample(pgn, cfg, target=4)
    first = path.read_text().splitlines()
    assert len(first) == 4

    # Resume with a larger target: earlier positions are kept, nothing is duplicated.
    build_dataset.stage_sample(pgn, cfg, target=100)
    fens = path.read_text().splitlines()
    assert fens[:4] == first
    assert len(fens) == len({build_dataset.position_key(f) for f in fens})
    assert len(fens) <= 3 * 3  # three usable games, at most three positions each

    for fen in fens:
        board = chess.Board(fen)
        assert board.ply() >= 10
        assert not board.is_check()


def test_partial_line_is_truncated(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_bytes(b"a\nb\nhalf")
    assert build_dataset._read_lines_truncating(path) == ["a", "b"]
    assert path.read_bytes() == b"a\nb\n"


def test_pack_stage(tmp_path: Path) -> None:
    from config import Config

    cfg = Config(dataset=make_cfg(tmp_path))
    labels = tmp_path / "labels.tsv"
    labels.write_text(f"{chess.STARTING_FEN}\t25\tcp:25\n8/8/4k3/8/8/3KQ3/8/8 w - - 0 1\t2000\tmate:5\n")
    out = build_dataset.stage_pack(cfg)
    import numpy as np

    with np.load(out) as data:
        assert data["features"].shape == (2, 98)
        assert data["cp"].tolist() == [25, 2000]


@pytest.mark.parametrize("value", ["", "missing.exe"])
def test_label_stage_requires_stockfish(tmp_path: Path, value: str) -> None:
    from config import Config, ConfigError, StockfishConfig

    cfg = Config(dataset=make_cfg(tmp_path), stockfish=StockfishConfig(path=str(tmp_path / value) if value else ""))
    with pytest.raises(ConfigError):
        build_dataset.stage_label(cfg, workers=1, depth=1, chunk_size=1)
