"""Model, checkpoint and NNEvaluator tests (using a small untrained network)."""

from __future__ import annotations

from pathlib import Path

import chess
import pytest
import torch

from chess_engine.encoding import NUM_FEATURES
from chess_engine.evaluators import NNEvaluator, is_mate_score
from chess_engine.model import EvalMLP, load_checkpoint, save_checkpoint
from chess_engine.search import Searcher


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    path = tmp_path / "net.pt"
    save_checkpoint(path, EvalMLP(hidden=[32, 8]), cp_scale=173.7, metrics={"val_loss": 0.1})
    return path


def test_forward_shape() -> None:
    model = EvalMLP(hidden=[16])
    assert model(torch.zeros(5, NUM_FEATURES)).shape == (5,)


def test_checkpoint_round_trip(checkpoint: Path) -> None:
    model, meta = load_checkpoint(checkpoint)
    assert meta["cp_scale"] == pytest.approx(173.7)
    assert meta["model"]["hidden"] == [32, 8]
    assert meta["metrics"]["val_loss"] == 0.1
    assert not model.training


def test_missing_checkpoint_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_checkpoint(tmp_path / "missing.pt")


def test_wrong_encoding_version_rejected(tmp_path: Path) -> None:
    path = tmp_path / "old.pt"
    save_checkpoint(path, EvalMLP(hidden=[8]), cp_scale=100.0)
    ckpt = torch.load(path, weights_only=True)
    ckpt["encoding_version"] = -1
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="encoding version"):
        load_checkpoint(path)


def test_nn_evaluator_matches_model(checkpoint: Path) -> None:
    ev = NNEvaluator(checkpoint, max_cp=2000)
    board = chess.Board()
    model, meta = load_checkpoint(checkpoint)
    from chess_engine.encoding import encode_board

    expected = model(torch.from_numpy(encode_board(board)).unsqueeze(0)).item() * meta["cp_scale"]
    assert ev.evaluate(board) == max(-2000, min(2000, round(expected)))
    assert ev.network_calls() == 1


def test_nn_evaluator_clips_and_counts(checkpoint: Path) -> None:
    ev = NNEvaluator(checkpoint, max_cp=1)
    assert abs(ev.evaluate(chess.Board())) <= 1
    # Terminal positions are scored by the rules without a forward pass.
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert is_mate_score(ev.evaluate(mated))
    assert ev.network_calls() == 1
    assert ev.stats()["calls"] == 2


def test_search_with_nn_returns_legal_move(checkpoint: Path) -> None:
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    result = Searcher(NNEvaluator(checkpoint)).search(board, time_limit=0.2)
    assert result.best_move in board.legal_moves
    assert result.network_calls > 0
