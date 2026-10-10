#!/usr/bin/env python
"""Depth benchmark: how deep does the engine search at a fixed time per move?

For every test position (opening, middlegame, endgame) the engine plays a few
consecutive moves against itself at a fixed time per move. Each search records
the completed depth, nodes per second, leaf evaluations and network calls. The
evaluator state (e.g. an evaluation cache) persists across the moves played from
one position, exactly as it would within a game, and is reset between positions.

Examples:
    python scripts/depth_benchmark.py                          # material baseline
    python scripts/depth_benchmark.py --evaluators nn
    python scripts/depth_benchmark.py --evaluators material nn nn-cache   # comparison table
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402

from chess_engine.evaluators import EVALUATOR_NAMES, build_evaluator  # noqa: E402
from chess_engine.positions import BENCHMARK_POSITIONS, PHASES, TestPosition  # noqa: E402
from chess_engine.reporting import markdown_table, write_json  # noqa: E402
from chess_engine.search import Searcher  # noqa: E402
from config import RESULTS_DIR, Config, ConfigError, get_config, resolve_path  # noqa: E402


def run_benchmark(
    name: str,
    cfg: Config,
    time_limit: float,
    moves_per_position: int,
    positions: list[TestPosition],
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one evaluator configuration over all positions; return records and summary."""
    evaluator = build_evaluator(name, cfg)
    # Always spend the full budget so per-move counts are comparable across configs.
    searcher = Searcher(evaluator, dataclasses.replace(cfg.search, early_stop_fraction=1.0))
    records: list[dict[str, Any]] = []

    for pos in positions:
        evaluator.new_game()
        board = chess.Board(pos.fen)
        for move_index in range(moves_per_position):
            if board.is_game_over(claim_draw=True):
                break
            before = evaluator.stats()
            result = searcher.search(board, time_limit=time_limit)
            after = evaluator.stats()
            hits = after.get("cache_hits", 0) - before.get("cache_hits", 0)
            misses = after.get("cache_misses", 0) - before.get("cache_misses", 0)
            record = {
                "position": pos.name,
                "phase": pos.phase,
                "move_index": move_index,
                "fen": board.fen(),
                "move": result.best_move.uci() if result.best_move else None,
                "score": result.score,
                "depth": result.depth,
                "nodes": result.nodes,
                "qnodes": result.qnodes,
                "evals": result.evals,
                "network_calls": result.network_calls,
                "cache_hits": hits,
                "cache_misses": misses,
                "time": round(result.time, 4),
                "nps": round(result.nps, 1),
            }
            records.append(record)
            if verbose:
                cache = f" hit={hits / (hits + misses):5.1%}" if hits + misses else ""
                print(
                    f"  {name:9s} {pos.name:24s} #{move_index} depth={result.depth:2d} "
                    f"nodes={result.nodes:7d} nps={result.nps:8.0f} evals={result.evals:6d} "
                    f"net={result.network_calls:6d}{cache} move={record['move']}",
                    flush=True,
                )
            board.push(result.best_move)

    return {
        "config": name,
        "time_limit": time_limit,
        "moves_per_position": moves_per_position,
        "search": dataclasses.asdict(cfg.search),
        "summary": summarize(records),
        "records": records,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-move records into per-phase and overall statistics."""
    def agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        total_time = sum(r["time"] for r in rows)
        hits = sum(r["cache_hits"] for r in rows)
        lookups = hits + sum(r["cache_misses"] for r in rows)
        return {
            "moves": len(rows),
            "depth_mean": statistics.fmean(r["depth"] for r in rows),
            "depth_min": min(r["depth"] for r in rows),
            "depth_max": max(r["depth"] for r in rows),
            "nps": sum(r["nodes"] for r in rows) / total_time if total_time else 0.0,
            "nodes_per_move": statistics.fmean(r["nodes"] for r in rows),
            "evals_per_move": statistics.fmean(r["evals"] for r in rows),
            "network_calls_per_move": statistics.fmean(r["network_calls"] for r in rows),
            "cache_hit_rate": hits / lookups if lookups else None,
            "time_per_move": total_time / len(rows),
        }

    summary = {"all": agg(records)}
    for phase in PHASES:
        summary[phase] = agg([r for r in records if r["phase"] == phase])
    return summary


def comparison_rows(runs: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "config",
        "depth (all)",
        "depth open/mid/end",
        "nodes/sec",
        "evals/move",
        "net calls/move",
        "cache hit rate",
    ]
    rows = []
    for run in runs:
        s = run["summary"]
        phases = "/".join(f"{s[p]['depth_mean']:.1f}" if s[p] else "-" for p in PHASES)
        hit = s["all"]["cache_hit_rate"]
        rows.append(
            [
                run["config"],
                round(s["all"]["depth_mean"], 2),
                phases,
                round(s["all"]["nps"]),
                round(s["all"]["evals_per_move"]),
                round(s["all"]["network_calls_per_move"]),
                f"{hit:.1%}" if hit is not None else "-",
            ]
        )
    return headers, rows


def main() -> int:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluators", nargs="+", default=["material"], choices=EVALUATOR_NAMES)
    parser.add_argument("--time", type=float, default=cfg.benchmark.time_limit, help="seconds per move")
    parser.add_argument("--moves", type=int, default=cfg.benchmark.moves_per_position, help="moves per position")
    parser.add_argument("--phases", nargs="+", default=list(PHASES), choices=PHASES)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR), help="where to write depth_*.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    positions = [p for p in BENCHMARK_POSITIONS if p.phase in args.phases]
    out_dir = resolve_path(args.out_dir)
    n_moves = len(positions) * args.moves
    print(
        f"Depth benchmark: {len(positions)} positions x {args.moves} moves at {args.time:g}s/move "
        f"(~{n_moves * args.time * len(args.evaluators) / 60:.1f} min)"
    )

    runs = []
    for name in args.evaluators:
        print(f"\n== {name} ==")
        try:
            run = run_benchmark(name, cfg, args.time, args.moves, positions, verbose=not args.quiet)
        except (ConfigError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        write_json(out_dir / f"depth_{name}.json", run)
        runs.append(run)

    headers, rows = comparison_rows(runs)
    table = markdown_table(headers, rows)
    print("\n" + table)
    if len(runs) > 1:
        note = (
            f"Depth benchmark at {args.time:g}s/move, {len(positions)} positions x {args.moves} moves "
            f"(search: quiescence={cfg.search.quiescence}).\n\n"
        )
        (out_dir / "depth_comparison.md").write_text(note + table + "\n", encoding="utf-8")
        write_json(out_dir / "depth_comparison.json", {"headers": headers, "rows": rows})
        print(f"\nSaved comparison to {out_dir / 'depth_comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
