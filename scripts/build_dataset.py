#!/usr/bin/env python
"""Build a Stockfish-labelled training set from a Lichess games database.

Stages (each resumable: rerun the same command after an interruption):

  sample  Stream the .pgn.zst file (decompressed on the fly, never to disk),
          sample up to N positions per game after skipping the opening plies,
          positions in check and (by default) positions whose game move was a
          capture, deduplicate, and stop at --target unique positions.
          -> data/processed/positions.txt
  label   Evaluate every position with Stockfish at a fixed depth in parallel
          worker processes. Scores are from the side to move's perspective,
          mates become +/-mate_cp, and everything is clipped to +/-clip_cp.
          -> data/processed/labels.tsv   (fen, clipped cp, raw score)
  pack    Encode labelled positions into compact model-ready features.
          -> data/processed/dataset.npz

Example:
    python scripts/build_dataset.py data/raw/lichess_db_standard_rated_2014-01.pgn.zst --target 1000000

The Stockfish binary is read from [stockfish].path in config.toml.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
import chess.engine  # noqa: E402
import chess.pgn  # noqa: E402
import numpy as np  # noqa: E402
import zstandard  # noqa: E402
from tqdm import tqdm  # noqa: E402

from chess_engine.encoding import ENCODING_VERSION, PACKED_BYTES, encode_packed  # noqa: E402
from config import RESULTS_DIR, ConfigError, DatasetConfig, get_config, resolve_path, stockfish_path  # noqa: E402

FLUSH_EVERY_GAMES = 2000


# --------------------------------------------------------------------- sampling


def position_key(fen: str) -> str:
    """Deduplication key: the FEN without the move counters."""
    return fen.rsplit(" ", 2)[0]


class PositionSampler(chess.pgn.BaseVisitor[list[str]]):
    """PGN visitor that reservoir-samples eligible positions from one game.

    It never builds a game tree, which makes it several times faster than
    ``chess.pgn.read_game`` with the default builder.
    """

    def __init__(self, cfg: DatasetConfig, rng: random.Random) -> None:
        self.cfg = cfg
        self.rng = rng
        self.samples: list[str] = []
        self.eligible = 0
        self.ply = 0
        self.skip = False

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        # Only standard chess from the initial position.
        if tagname == "Variant" and tagvalue.lower() not in ("standard", "chess"):
            self.skip = True
        elif tagname == "FEN":
            self.skip = True

    def end_headers(self) -> chess.pgn.SkipType | None:
        return chess.pgn.SKIP if self.skip else None

    def begin_variation(self) -> chess.pgn.SkipType | None:
        return chess.pgn.SKIP  # main line only

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        ply = self.ply
        self.ply += 1
        if ply < self.cfg.skip_plies or board.is_check():
            return
        if self.cfg.skip_captures and board.is_capture(move):
            return
        # Reservoir sampling keeps a uniform sample of size max_per_game, and only
        # materialises FEN strings for positions that enter the reservoir.
        self.eligible += 1
        k = self.cfg.max_per_game
        if len(self.samples) < k:
            self.samples.append(board.fen())
        else:
            j = self.rng.randrange(self.eligible)
            if j < k:
                self.samples[j] = board.fen()

    def handle_error(self, error: Exception) -> None:
        self.skip = True  # drop malformed games instead of aborting the stream

    def result(self) -> list[str]:
        return [] if self.skip else self.samples


def _read_lines_truncating(path: Path) -> list[str]:
    """Read complete lines; drop a partially written last line left by a crash."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        with path.open("r+b") as fh:
            fh.truncate(cut)
        data = data[:cut]
    return data.decode("utf-8").splitlines()


def stage_sample(pgn_path: Path, cfg: DatasetConfig, target: int) -> Path:
    positions_path = resolve_path(cfg.positions_file)
    state_path = positions_path.with_suffix(".state.json")
    positions_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_lines_truncating(positions_path)
    seen = {position_key(f) for f in existing}
    count = len(existing)
    params = {
        "source": pgn_path.name,
        "skip_plies": cfg.skip_plies,
        "max_per_game": cfg.max_per_game,
        "skip_captures": cfg.skip_captures,
        "seed": cfg.seed,
    }
    state: dict[str, Any] = {"games": 0, "params": params}
    if state_path.exists() and existing:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("params") != params:
            print(f"warning: sampling parameters changed since {positions_path.name} was started "
                  f"({state.get('params')} -> {params}); continuing anyway")
            state["params"] = params

    if count >= target:
        print(f"[sample] {positions_path} already has {count:,} positions (target {target:,}); skipping")
        return positions_path

    print(f"[sample] streaming {pgn_path} (resuming after {state['games']:,} games, {count:,} positions)")
    games = state["games"]

    def save_state() -> None:
        state["games"] = games
        state["positions"] = count
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, state_path)

    with pgn_path.open("rb") as raw, positions_path.open("a", encoding="utf-8", newline="\n") as out:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        for _ in tqdm(range(games), desc="skipping processed games", unit="game", disable=games == 0):
            if not chess.pgn.skip_game(text):
                break

        bar = tqdm(total=target, initial=count, desc="sampling", unit="pos", smoothing=0.05)
        try:
            while count < target:
                rng = random.Random(cfg.seed * 1_000_003 + games)  # per-game seed: resume-stable
                fens = chess.pgn.read_game(text, Visitor=lambda: PositionSampler(cfg, rng))
                if fens is None:
                    print(f"\n[sample] end of file after {games:,} games")
                    break
                games += 1
                for fen in fens:
                    key = position_key(fen)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(fen + "\n")
                    count += 1
                    bar.update(1)
                    if count >= target:
                        break
                if games % FLUSH_EVERY_GAMES == 0:
                    out.flush()
                    save_state()
                    bar.set_postfix(games=f"{games:,}")
        finally:
            out.flush()
            save_state()
            bar.close()

    print(f"[sample] {count:,} unique positions from {games:,} games -> {positions_path}")
    if count < target:
        print(f"warning: the database ran out before reaching the target of {target:,}")
    return positions_path


# --------------------------------------------------------------------- labelling

_ENGINE: chess.engine.SimpleEngine | None = None
_ENGINE_ARGS: tuple[str, int, int] | None = None
_LIMIT: chess.engine.Limit | None = None


def _start_engine() -> None:
    global _ENGINE
    assert _ENGINE_ARGS is not None
    path, threads, hash_mb = _ENGINE_ARGS
    _ENGINE = chess.engine.SimpleEngine.popen_uci(path)
    _ENGINE.configure({"Threads": threads, "Hash": hash_mb})


def _init_worker(path: str, depth: int, threads: int, hash_mb: int) -> None:
    """Pool initializer: each worker process owns one Stockfish instance."""
    global _ENGINE_ARGS, _LIMIT
    _ENGINE_ARGS = (path, threads, hash_mb)
    _LIMIT = chess.engine.Limit(depth=depth)
    _start_engine()


def convert_score(score: chess.engine.PovScore, turn: chess.Color, mate_cp: int, clip_cp: int) -> tuple[int, str]:
    """Side-to-move centipawns (mates -> +/-mate_cp, then clipped) and the raw score string."""
    pov = score.pov(turn)
    mate = pov.mate()
    raw = f"mate:{mate}" if mate is not None else f"cp:{pov.score()}"
    cp = pov.score(mate_score=mate_cp)
    assert cp is not None
    return max(-clip_cp, min(clip_cp, cp)), raw


def _label_chunk(job: tuple[list[str], int, int]) -> list[tuple[str, int, str]]:
    fens, mate_cp, clip_cp = job
    results = []
    for fen in fens:
        board = chess.Board(fen)
        for attempt in range(2):
            try:
                assert _ENGINE is not None
                info = _ENGINE.analyse(board, _LIMIT)
                break
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
                if attempt:
                    raise
                _start_engine()  # restart a crashed engine once
        cp, raw = convert_score(info["score"], board.turn, mate_cp, clip_cp)
        results.append((fen, cp, raw))
    return results


def stage_label(cfg_all: Any, workers: int, depth: int, chunk_size: int) -> Path:
    cfg: DatasetConfig = cfg_all.dataset
    sf = stockfish_path(cfg_all)
    positions_path = resolve_path(cfg.positions_file)
    labels_path = resolve_path(cfg.labels_file)
    if not positions_path.exists():
        raise FileNotFoundError(f"{positions_path} not found; run the sample stage first")

    fens = _read_lines_truncating(positions_path)
    done = {line.split("\t", 1)[0] for line in _read_lines_truncating(labels_path)}
    todo = [f for f in fens if f not in done]
    print(f"[label] {len(fens):,} positions, {len(done):,} already labelled, {len(todo):,} to go "
          f"(Stockfish depth {depth}, {workers} workers)")
    if not todo:
        return labels_path

    meta_path = labels_path.with_suffix(".meta.json")
    meta = {"stockfish": str(sf.name), "depth": depth, "mate_cp": cfg.mate_cp, "clip_cp": cfg.clip_cp}
    if meta_path.exists() and done:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != meta:
            print(f"warning: labelling settings changed ({old} -> {meta}); labels will be mixed")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    jobs = [(todo[i : i + chunk_size], cfg.mate_cp, cfg.clip_cp) for i in range(0, len(todo), chunk_size)]
    ctx = mp.get_context("spawn")  # same behaviour on Windows, macOS and Linux
    pool = ctx.Pool(
        workers,
        initializer=_init_worker,
        initargs=(str(sf), depth, cfg_all.stockfish.threads, cfg_all.stockfish.hash_mb),
    )
    start = time.perf_counter()
    labelled = 0
    try:
        with labels_path.open("a", encoding="utf-8", newline="\n") as out, tqdm(
            total=len(todo), desc="labelling", unit="pos", smoothing=0.05
        ) as bar:
            for results in pool.imap_unordered(_label_chunk, jobs):
                out.writelines(f"{fen}\t{cp}\t{raw}\n" for fen, cp, raw in results)
                out.flush()
                labelled += len(results)
                bar.update(len(results))
    finally:
        # Always terminate rather than close(): python-chess runs each engine on a
        # non-daemon thread, so a worker exiting normally would wait for it forever.
        # Each Stockfish process exits by itself once its stdin pipe closes.
        pool.terminate()
        pool.join()
    elapsed = time.perf_counter() - start
    print(f"[label] labelled {labelled:,} positions in {elapsed / 60:.1f} min "
          f"({labelled / max(elapsed, 1e-9):.0f} pos/s) -> {labels_path}")
    return labels_path


# --------------------------------------------------------------------- packing


def stage_pack(cfg_all: Any, summary_path: Path | None = None) -> Path:
    """Encode labels into dataset.npz; optionally write summary statistics as JSON."""
    cfg: DatasetConfig = cfg_all.dataset
    labels_path = resolve_path(cfg.labels_file)
    dataset_path = resolve_path(cfg.dataset_file)
    lines = _read_lines_truncating(labels_path)
    if not lines:
        raise FileNotFoundError(f"{labels_path} is empty or missing; run the label stage first")

    n = len(lines)
    features = np.empty((n, PACKED_BYTES), dtype=np.uint8)
    cp = np.empty(n, dtype=np.int16)
    mates = 0
    for i, line in enumerate(tqdm(lines, desc="encoding", unit="pos", smoothing=0.05)):
        fen, score, raw = line.split("\t")
        features[i] = encode_packed(chess.Board(fen))
        cp[i] = int(score)
        mates += raw.startswith("mate")

    meta_path = labels_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    tmp = dataset_path.with_name(dataset_path.stem + ".tmp.npz")
    np.savez(
        tmp,
        features=features,
        cp=cp,
        encoding_version=np.int32(ENCODING_VERSION),
        meta=np.array(json.dumps(meta)),
    )
    os.replace(tmp, dataset_path)
    print(f"[pack] {n:,} positions -> {dataset_path} ({dataset_path.stat().st_size / 1e6:.0f} MB)")
    print(f"[pack] label stats: mean {cp.mean():.1f} cp, std {cp.std():.1f} cp, "
          f"|cp| >= {cfg.clip_cp}: {(np.abs(cp) >= cfg.clip_cp).mean():.1%}")
    if summary_path is not None:
        pieces = np.unpackbits(features[:, :96], axis=1).sum(axis=1)
        state_path = resolve_path(cfg.positions_file).with_suffix(".state.json")
        sampling = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        abs_cp = np.abs(cp.astype(np.int32))
        summary = {
            "positions": n,
            "sampling": sampling,
            "labelling": meta,
            "label_mean_cp": float(cp.mean()),
            "label_std_cp": float(cp.std()),
            "label_abs_quantiles_cp": {f"p{q}": float(np.percentile(abs_cp, q)) for q in (25, 50, 75, 90, 99)},
            "mate_scores": mates,
            "clipped_fraction": float((abs_cp >= cfg.clip_cp).mean()),
            "pieces_on_board": {
                "26-32": int((pieces >= 26).sum()),
                "13-25": int(((pieces >= 13) & (pieces <= 25)).sum()),
                "2-12": int((pieces <= 12).sum()),
            },
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[pack] summary -> {summary_path}")
    return dataset_path


# --------------------------------------------------------------------- main


def main() -> int:
    cfg = get_config()
    d, s = cfg.dataset, cfg.stockfish
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pgn", nargs="?", help="Lichess .pgn.zst database (needed for the sample stage)")
    parser.add_argument("--stage", choices=["all", "sample", "label", "pack"], default="all")
    parser.add_argument("--target", type=int, default=d.target_positions, help="unique positions to sample")
    parser.add_argument("--skip-plies", type=int, default=d.skip_plies)
    parser.add_argument("--max-per-game", type=int, default=d.max_per_game)
    parser.add_argument("--skip-captures", action=argparse.BooleanOptionalAction, default=d.skip_captures,
                        help="skip positions where the move played was a capture")
    parser.add_argument("--depth", type=int, default=s.depth, help="Stockfish search depth per label")
    parser.add_argument("--workers", type=int, default=s.workers, help="parallel Stockfish processes")
    parser.add_argument("--chunk-size", type=int, default=128, help="positions per worker task")
    parser.add_argument("--seed", type=int, default=d.seed)
    args = parser.parse_args()

    d.skip_plies, d.max_per_game, d.skip_captures, d.seed = args.skip_plies, args.max_per_game, args.skip_captures, args.seed

    try:
        if args.stage in ("all", "sample"):
            if not args.pgn:
                parser.error("the sample stage needs the path to a .pgn.zst file")
            pgn_path = Path(args.pgn)
            if not pgn_path.is_file():
                parser.error(f"PGN database not found: {pgn_path}")
            stage_sample(pgn_path, d, args.target)
        if args.stage in ("all", "label"):
            stage_label(cfg, args.workers, args.depth, args.chunk_size)
        if args.stage in ("all", "pack"):
            stage_pack(cfg, RESULTS_DIR / "dataset_summary.json")
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; rerun the same command to resume", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
