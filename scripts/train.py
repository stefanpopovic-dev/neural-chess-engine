#!/usr/bin/env python
"""Train the evaluation MLP on Stockfish-labelled positions.

Target: the side to move's win probability ``p = sigmoid(cp / cp_scale)``
(cp_scale = 400 / ln 10, the Elo logistic curve). The network outputs a logit
and is trained with mean squared error between ``sigmoid(logit)`` and ``p``,
which down-weights the exact size of already-decisive evaluations.

Train/validation split, AdamW, learning-rate halving on plateaus, early
stopping on validation loss. Outputs:
  models/nn_eval.pt           best checkpoint (by validation loss)
  results/train_log.csv       per-epoch losses
  results/train_loss.png      loss curves
  results/train_metrics.json  final metrics and baselines

Example:
    python scripts/train.py
    python scripts/train.py --limit 50000 --epochs 3   # quick smoke run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from chess_engine.encoding import ENCODING_VERSION, NUM_PIECE_SQUARE, unpack  # noqa: E402
from chess_engine.model import EvalMLP, save_checkpoint  # noqa: E402
from chess_engine.reporting import write_json  # noqa: E402
from config import RESULTS_DIR, get_config, resolve_path  # noqa: E402


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; build it with scripts/build_dataset.py")
    with np.load(path) as data:
        if int(data["encoding_version"]) != ENCODING_VERSION:
            raise ValueError(f"{path} was encoded with version {int(data['encoding_version'])}, "
                             f"expected {ENCODING_VERSION}; rerun build_dataset.py --stage pack")
        meta = json.loads(str(data["meta"])) if "meta" in data else {}
        return data["features"], data["cp"].astype(np.float32), meta


def material_cp(packed: np.ndarray, values: tuple[int, ...]) -> np.ndarray:
    """Side-to-move material balance computed straight from the packed piece planes."""
    bits = unpack(packed)[:, :NUM_PIECE_SQUARE].reshape(-1, 12, 64).sum(axis=2, dtype=np.int32)
    weights = np.array(list(values) + [0] + [-v for v in values] + [0], dtype=np.int32)  # kings count 0
    return (bits * weights).sum(axis=1).astype(np.float32)


def batches(indices: np.ndarray, size: int) -> list[np.ndarray]:
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def to_tensor(packed: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(unpack(packed)).to(device=device, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model: EvalMLP,
    packed: np.ndarray,
    cp: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    cp_scale: float,
    clip_cp: float,
    device: torch.device,
) -> dict[str, float]:
    """Validation loss (MSE on win probability) and mean absolute error in centipawns."""
    model.eval()
    sq_err = abs_cp = 0.0
    for idx in batches(indices, 8192):
        logits = model(to_tensor(packed[idx], device)).cpu().numpy()
        sq_err += float(((1.0 / (1.0 + np.exp(-logits)) - target[idx]) ** 2).sum())
        pred_cp = np.clip(logits * cp_scale, -clip_cp, clip_cp)
        abs_cp += float(np.abs(pred_cp - cp[idx]).sum())
    return {"loss": sq_err / len(indices), "mae_cp": abs_cp / len(indices)}


def plot_losses(log: list[dict[str, float]], baselines: dict[str, float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [row["epoch"] for row in log]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    ax.plot(epochs, [row["train_loss"] for row in log], label="train", color="#2563eb", linewidth=2)
    ax.plot(epochs, [row["val_loss"] for row in log], label="validation", color="#ea580c", linewidth=2)
    ax.axhline(baselines["material_val_loss"], color="#6b7280", linestyle="--", linewidth=1,
               label="material count (validation)")
    best = min(log, key=lambda r: r["val_loss"])
    ax.scatter([best["epoch"]], [best["val_loss"]], color="#ea580c", zorder=3)
    ax.annotate(f"best {best['val_loss']:.4f}", (best["epoch"], best["val_loss"]),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE on win probability")
    ax.set_title("Evaluation network training")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    cfg = get_config()
    t, n = cfg.train, cfg.nn
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=cfg.dataset.dataset_file)
    parser.add_argument("--out", default=n.checkpoint, help="checkpoint path")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--hidden", type=int, nargs="+", default=n.hidden)
    parser.add_argument("--batch-size", type=int, default=t.batch_size)
    parser.add_argument("--lr", type=float, default=t.lr)
    parser.add_argument("--weight-decay", type=float, default=t.weight_decay)
    parser.add_argument("--epochs", type=int, default=t.max_epochs)
    parser.add_argument("--patience", type=int, default=t.patience)
    parser.add_argument("--val-fraction", type=float, default=t.val_fraction)
    parser.add_argument("--seed", type=int, default=t.seed)
    parser.add_argument("--device", default=t.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="CPU threads for training")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N positions (0 = all)")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          (args.device if args.device != "auto" else "cpu"))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    try:
        packed, cp, meta = load_dataset(resolve_path(args.dataset))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.limit:
        packed, cp = packed[: args.limit], cp[: args.limit]
    clip_cp = float(meta.get("clip_cp", cfg.dataset.clip_cp))
    cp_scale = n.cp_scale
    target = (1.0 / (1.0 + np.exp(-cp / cp_scale))).astype(np.float32)

    perm = rng.permutation(len(cp))
    n_val = max(1, int(len(cp) * args.val_fraction))
    val_idx, train_idx = np.sort(perm[:n_val]), perm[n_val:]
    print(f"dataset: {len(cp):,} positions ({len(train_idx):,} train / {len(val_idx):,} val), device {device}")

    # Reference points: a material count and a constant predictor on the same validation set.
    m = cfg.material
    mat = material_cp(packed[val_idx], (m.pawn, m.knight, m.bishop, m.rook, m.queen))
    mat_p = 1.0 / (1.0 + np.exp(-mat / cp_scale))
    baselines = {
        "material_val_loss": float(((mat_p - target[val_idx]) ** 2).mean()),
        "material_val_mae_cp": float(np.abs(np.clip(mat, -clip_cp, clip_cp) - cp[val_idx]).mean()),
        "constant_val_loss": float(target[val_idx].var()),
    }
    print(f"baselines: material loss {baselines['material_val_loss']:.4f} "
          f"(MAE {baselines['material_val_mae_cp']:.0f} cp), constant loss {baselines['constant_val_loss']:.4f}")

    model = EvalMLP(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=t.lr_patience)
    loss_fn = nn.MSELoss()
    target_t = torch.from_numpy(target)

    results_dir = resolve_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = resolve_path(args.out)
    log: list[dict[str, float]] = []
    best_loss, best_epoch, stale = math.inf, 0, 0
    train_config = {**vars(args), "cp_scale": cp_scale, "clip_cp": clip_cp, "dataset_meta": meta}
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total, count = 0.0, 0
        for idx in batches(rng.permutation(train_idx), args.batch_size):
            x = to_tensor(packed[idx], device)
            y = target_t[idx].to(device)
            loss = loss_fn(torch.sigmoid(model(x)), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            count += len(idx)
        val = evaluate(model, packed, cp, target, val_idx, cp_scale, clip_cp, device)
        scheduler.step(val["loss"])
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "val_loss": val["loss"],
            "val_mae_cp": val["mae_cp"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        log.append(row)
        improved = val["loss"] < best_loss - 1e-6
        if improved:
            best_loss, best_epoch, stale = val["loss"], epoch, 0
            save_checkpoint(out_path, model, cp_scale, metrics={**row, **baselines}, train_config=train_config)
        else:
            stale += 1
        print(f"epoch {epoch:3d}  train {row['train_loss']:.5f}  val {row['val_loss']:.5f}  "
              f"MAE {row['val_mae_cp']:6.1f} cp  lr {row['lr']:.1e}  {row['seconds']:.0f}s"
              f"{'  *' if improved else ''}", flush=True)
        with (results_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(log)
        if stale >= args.patience:
            print(f"early stopping: no improvement for {args.patience} epochs")
            break

    best = log[best_epoch - 1]
    metrics = {
        "positions": len(cp),
        "train_positions": len(train_idx),
        "val_positions": len(val_idx),
        "parameters": n_params,
        "hidden": args.hidden,
        "epochs_run": len(log),
        "best_epoch": best_epoch,
        "best_val_loss": best["val_loss"],
        "best_val_mae_cp": best["val_mae_cp"],
        "train_loss_at_best": best["train_loss"],
        **baselines,
        "training_minutes": (time.perf_counter() - start) / 60,
        "checkpoint": str(out_path.relative_to(resolve_path("."))) if out_path.is_relative_to(resolve_path(".")) else str(out_path),
        "dataset_meta": meta,
    }
    write_json(results_dir / "train_metrics.json", metrics)
    plot_losses(log, baselines, results_dir / "train_loss.png")
    print(f"best epoch {best_epoch}: val loss {best['val_loss']:.5f}, MAE {best['val_mae_cp']:.1f} cp "
          f"(material baseline {baselines['material_val_loss']:.5f}) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
