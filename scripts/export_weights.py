#!/usr/bin/env python
"""Export the trained PyTorch checkpoint to a numpy .npz file.

NumpyNNEvaluator loads this file, so the network can run without PyTorch,
for example inside the size-limited serverless function of the web demo.
The export is checked against the PyTorch model on random positions.

Example:
    python scripts/export_weights.py          # models/nn_eval.pt -> models/nn_eval.npz
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
import numpy as np  # noqa: E402
from torch import nn  # noqa: E402

from chess_engine.encoding import ENCODING_VERSION  # noqa: E402
from chess_engine.evaluators import NNEvaluator, NumpyNNEvaluator  # noqa: E402
from chess_engine.model import load_checkpoint  # noqa: E402
from config import get_config, resolve_path  # noqa: E402


def export(checkpoint: Path, out: Path) -> None:
    """Write each Linear layer's weight (out x in) and bias as w{i} / b{i}."""
    model, meta = load_checkpoint(checkpoint)
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    arrays = {}
    for i, layer in enumerate(linears):
        arrays[f"w{i}"] = layer.weight.detach().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().numpy().astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(
        tmp,
        num_layers=np.int32(len(linears)),
        cp_scale=np.float64(meta["cp_scale"]),
        encoding_version=np.int32(ENCODING_VERSION),
        **arrays,
    )
    os.replace(tmp, out)


def random_positions(count: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(rng.randint(0, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            boards.append(board)
    return boards


def main() -> int:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=cfg.nn.checkpoint)
    parser.add_argument("--out", default=cfg.nn.weights)
    parser.add_argument("--check", type=int, default=500, help="random positions to compare (0 = skip)")
    args = parser.parse_args()

    checkpoint, out = resolve_path(args.checkpoint), resolve_path(args.out)
    try:
        export(checkpoint, out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"exported {checkpoint} -> {out} ({out.stat().st_size / 1024:.0f} KB)")

    if args.check:
        torch_eval, numpy_eval = NNEvaluator(checkpoint), NumpyNNEvaluator(out)
        diffs = [abs(torch_eval.static_eval(b) - numpy_eval.static_eval(b)) for b in random_positions(args.check)]
        print(f"checked {len(diffs)} random positions: max difference {max(diffs)} cp")
        if max(diffs) > 1:
            print("error: numpy and PyTorch evaluations disagree", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
