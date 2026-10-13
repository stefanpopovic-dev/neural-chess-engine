"""Central configuration for the chess engine project.

Every tunable parameter has a default here. A user-local ``config.toml`` in the
project root (copy ``config.example.toml``) can override any of them, section by
section, and the scripts' command-line flags override both. The Stockfish binary
path is the one setting with no usable default: it must be set in ``config.toml``.

Set the environment variable ``CHESS_ENGINE_CONFIG`` to load a different file.
"""

from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DEFAULT_CONFIG_FILE = ROOT / "config.toml"
EXAMPLE_CONFIG_FILE = ROOT / "config.example.toml"


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or invalid."""


@dataclass
class SearchConfig:
    """Alpha-beta search parameters."""

    time_limit: float = 1.0  # seconds per move
    max_depth: int = 64  # hard cap on iterative-deepening depth
    quiescence: bool = True  # resolve captures at the horizon
    qsearch_max_depth: int = 8  # max plies of quiescence below the horizon
    check_interval: int = 16  # nodes between clock checks (power of two)
    # Stop iterative deepening early once this fraction of the time budget is
    # used, because the next depth almost certainly cannot finish. 1.0 = always
    # use the full budget (comparable per-move metrics); lower values save time.
    early_stop_fraction: float = 1.0


@dataclass
class MaterialConfig:
    """Hand-tuned piece values in centipawns (the classic Simplified Evaluation values)."""

    pawn: int = 100
    knight: int = 320
    bishop: int = 330
    rook: int = 500
    queen: int = 900


@dataclass
class NNConfig:
    """Neural-network evaluator: architecture, output scaling and inference."""

    checkpoint: str = "models/nn_eval.pt"
    hidden: list[int] = field(default_factory=lambda: [256, 32])
    # Centipawn <-> win-probability scale: p = sigmoid(cp / cp_scale).
    # 400 / ln(10) makes this the Elo logistic curve, p = 1 / (1 + 10^(-cp/400)).
    cp_scale: float = 400.0 / math.log(10.0)
    max_cp: int = 2000  # clip network output to +/- this many centipawns
    torch_threads: int = 1  # single-position inference is fastest single-threaded


@dataclass
class CacheConfig:
    """Evaluation cache (Zobrist-keyed LRU)."""

    max_size: int = 500_000  # entries; ~100 bytes each


@dataclass
class StockfishConfig:
    """Stockfish labelling settings. ``path`` must be set in config.toml."""

    path: str = ""
    depth: int = 10  # fixed search depth per label
    threads: int = 1  # threads per Stockfish process
    hash_mb: int = 16  # hash per Stockfish process
    workers: int = max(1, (os.cpu_count() or 2) - 1)  # parallel Stockfish processes


@dataclass
class DatasetConfig:
    """Position sampling from Lichess games and label post-processing."""

    target_positions: int = 1_000_000
    skip_plies: int = 10  # ignore the first N plies (book-like openings)
    max_per_game: int = 6  # at most this many positions from one game
    skip_captures: bool = True  # skip positions where the game move was a capture (non-quiet)
    clip_cp: int = 2000  # clip labels to +/- clip_cp
    mate_cp: int = 10_000  # value assigned to mate scores before clipping
    seed: int = 0
    positions_file: str = "data/processed/positions.txt"
    labels_file: str = "data/processed/labels.tsv"
    dataset_file: str = "data/processed/dataset.npz"


@dataclass
class TrainConfig:
    """Network training hyper-parameters."""

    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-6
    max_epochs: int = 60
    patience: int = 6  # early stopping: epochs without validation improvement
    lr_patience: int = 2  # epochs without improvement before halving the LR
    val_fraction: float = 0.05
    seed: int = 0
    device: str = "auto"  # "auto", "cpu" or "cuda"


@dataclass
class BenchmarkConfig:
    """Depth benchmark settings."""

    time_limit: float = 1.0
    moves_per_position: int = 3  # consecutive self-play moves per test position


@dataclass
class MatchConfig:
    """Self-play match settings."""

    games: int = 100
    time_limit: float = 0.5
    max_plies: int = 200  # move cap; games reaching it are scored as draws
    workers: int = 1  # parallel games
    engine_a: str = "nn-cache"
    engine_b: str = "material"
    out_dir: str = "results/match"


@dataclass
class Config:
    """All configuration sections."""

    search: SearchConfig = field(default_factory=SearchConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    stockfish: StockfishConfig = field(default_factory=StockfishConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    match: MatchConfig = field(default_factory=MatchConfig)


def _apply_section(section: Any, values: dict[str, Any], name: str) -> None:
    """Override dataclass fields of ``section`` with ``values`` (type-checked loosely)."""
    known = {f.name: f for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            raise ConfigError(f"Unknown setting [{name}].{key}; valid keys: {', '.join(sorted(known))}")
        current = getattr(section, key)
        if isinstance(current, bool) and not isinstance(value, bool):
            raise ConfigError(f"[{name}].{key} must be true or false")
        if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        setattr(section, key, value)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load defaults, then overrides from ``path`` / $CHESS_ENGINE_CONFIG / ./config.toml."""
    cfg = Config()
    explicit = path is not None or bool(os.environ.get("CHESS_ENGINE_CONFIG"))
    if path is None:
        path = os.environ.get("CHESS_ENGINE_CONFIG") or DEFAULT_CONFIG_FILE
    path = Path(path)
    if explicit and not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    if path.is_file():
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Could not parse {path}: {exc}") from exc
        for name, values in data.items():
            if not hasattr(cfg, name) or not isinstance(values, dict):
                raise ConfigError(f"Unknown config section [{name}] in {path}")
            _apply_section(getattr(cfg, name), values, name)
    return cfg


_CONFIG: Config | None = None


def get_config() -> Config:
    """Process-wide cached configuration."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def resolve_path(p: str | os.PathLike[str]) -> Path:
    """Resolve a config path; relative paths are relative to the project root."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else ROOT / path


def stockfish_path(cfg: Config | None = None) -> Path:
    """Return the configured Stockfish binary, or raise ConfigError with a clear fix."""
    cfg = cfg or get_config()
    raw = cfg.stockfish.path.strip()
    if not raw:
        raise ConfigError(
            "Stockfish path is not configured.\n"
            f"  1. Copy {EXAMPLE_CONFIG_FILE.name} to {DEFAULT_CONFIG_FILE.name} in {ROOT}\n"
            "  2. Set the binary location, e.g.\n"
            "       [stockfish]\n"
            '       path = "tools/stockfish/stockfish.exe"   # or /usr/local/bin/stockfish\n'
            "  Download Stockfish from https://stockfishchess.org/download/"
        )
    path = resolve_path(raw)
    if not path.is_file():
        raise ConfigError(
            f"Stockfish binary not found at '{path}' (from [stockfish].path = '{raw}' in "
            f"{DEFAULT_CONFIG_FILE.name}). Fix the path or install Stockfish."
        )
    return path
