"""PyTorch evaluation network.

A small multilayer perceptron mapping the 781 encoded features to a single
logit of the side to move's win probability. Centipawns are recovered as
``cp = logit * cp_scale`` because training targets are ``sigmoid(cp / cp_scale)``.
With the default hidden sizes (256, 32) the network has ~209k parameters,
small enough for single-position CPU inference inside the search.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from chess_engine.encoding import ENCODING_VERSION, NUM_FEATURES


class EvalMLP(nn.Module):
    """features -> hidden ReLU layers -> win-probability logit."""

    def __init__(self, input_size: int = NUM_FEATURES, hidden: Sequence[int] = (256, 32)) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden = list(hidden)
        layers: list[nn.Module] = []
        prev = input_size
        for width in self.hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def save_checkpoint(
    path: str | os.PathLike[str],
    model: EvalMLP,
    cp_scale: float,
    metrics: dict[str, Any] | None = None,
    train_config: dict[str, Any] | None = None,
) -> None:
    """Save weights plus everything needed to rebuild and interpret the network."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model": {"input_size": model.input_size, "hidden": list(model.hidden)},
        "cp_scale": float(cp_scale),
        "encoding_version": ENCODING_VERSION,
        "metrics": metrics or {},
        "train_config": train_config or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | os.PathLike[str]) -> tuple[EvalMLP, dict[str, Any]]:
    """Load a checkpoint saved by :func:`save_checkpoint`; returns (model in eval mode, metadata)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No trained network at '{path}'. Train one with: python scripts/train.py")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("encoding_version") != ENCODING_VERSION:
        raise ValueError(
            f"Checkpoint '{path}' uses encoding version {checkpoint.get('encoding_version')}, "
            f"but the code expects {ENCODING_VERSION}; retrain the network."
        )
    model = EvalMLP(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    meta = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return model, meta
