#!/usr/bin/env python
"""Profile a search to find where the time goes.

For each evaluator the same position is searched twice with the same budget:

1. Wall-clock attribution with negligible overhead: the evaluator's
   ``evaluate`` and its innermost ``static_eval`` (the network forward pass plus
   encoding, for the NN) are wrapped with ``perf_counter`` timers.
2. cProfile, for a function-level breakdown. cProfile inflates the cost of
   every Python call, so its percentages are indicative only.

Outputs results/profile_<evaluator>.txt (full report) and
results/profile_summary.md (one comparison table).

Example:
    python scripts/profile_search.py --evaluators material nn nn-cache --time 5
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402

from chess_engine.evaluators import EVALUATOR_NAMES, Evaluator, build_evaluator  # noqa: E402
from chess_engine.positions import BENCHMARK_POSITIONS  # noqa: E402
from chess_engine.reporting import markdown_table, write_json  # noqa: E402
from chess_engine.search import Searcher  # noqa: E402
from config import RESULTS_DIR, ConfigError, get_config, resolve_path  # noqa: E402

# Functions whose cumulative cProfile time is reported (matched by function name).
TRACKED = {
    "evaluate": "Evaluator.evaluate (rules + static eval)",
    "terminal_score": "  rules check (mate/draw detection)",
    "zobrist_hash": "  cache key (Zobrist hash)",
    "encode_board": "  board -> features encoding",
    "_call_impl": "  PyTorch forward pass",
    "generate_legal_moves": "move generation (legal moves)",
    "generate_legal_captures": "move generation (captures)",
    "push": "Board.push",
    "pop": "Board.pop",
    "_order_moves": "move ordering",
    "_is_rule_draw": "draw detection in tree",
}


def _timed(fn: Callable[[chess.Board], int], bucket: dict[str, float], key: str) -> Callable[[chess.Board], int]:
    perf = time.perf_counter

    def wrapper(board: chess.Board) -> int:
        start = perf()
        value = fn(board)
        bucket[key] += perf() - start
        return value

    return wrapper


def instrument(evaluator: Evaluator) -> dict[str, float]:
    """Wrap evaluate() and the innermost static_eval() with timers (instance-level patches)."""
    timers = {"evaluate": 0.0, "cache_layer": 0.0, "static": 0.0}
    evaluator.evaluate = _timed(evaluator.evaluate, timers, "evaluate")  # type: ignore[method-assign]
    inner = getattr(evaluator, "inner", None)
    if inner is not None:
        evaluator.static_eval = _timed(evaluator.static_eval, timers, "cache_layer")  # type: ignore[method-assign]
        inner.static_eval = _timed(inner.static_eval, timers, "static")  # type: ignore[method-assign]
    else:
        evaluator.static_eval = _timed(evaluator.static_eval, timers, "static")  # type: ignore[method-assign]
    return timers


def wall_clock(name: str, board: chess.Board, seconds: float) -> dict[str, Any]:
    cfg = get_config()
    evaluator = build_evaluator(name, cfg)
    timers = instrument(evaluator)
    searcher = Searcher(evaluator, cfg.search)
    result = searcher.search(board, time_limit=seconds)
    total = result.time
    static_calls = result.network_calls if result.network_calls else result.evals - evaluator.terminal
    row = {
        "evaluator": name,
        "search_seconds": total,
        "depth": result.depth,
        "nodes": result.nodes,
        "nps": result.nps,
        "evals": result.evals,
        "network_calls": result.network_calls,
        "eval_seconds": timers["evaluate"],
        "eval_share": timers["evaluate"] / total,
        "static_seconds": timers["static"],
        "static_share": timers["static"] / total,
        "cache_layer_share": timers["cache_layer"] / total if timers["cache_layer"] else None,
        "us_per_eval": 1e6 * timers["evaluate"] / max(result.evals, 1),
        "us_per_static_call": 1e6 * timers["static"] / max(static_calls, 1),
        "us_per_node_excluding_eval": 1e6 * (total - timers["evaluate"]) / max(result.nodes, 1),
        "nps_if_eval_free": result.nodes / max(total - timers["evaluate"], 1e-9),
    }
    if hasattr(evaluator, "hit_rate"):
        row["cache_hit_rate"] = evaluator.hit_rate
    return row


def profiled(name: str, board: chess.Board, seconds: float, top: int) -> tuple[str, dict[str, float]]:
    cfg = get_config()
    evaluator = build_evaluator(name, cfg)
    searcher = Searcher(evaluator, cfg.search)
    profiler = cProfile.Profile()
    result = profiler.runcall(searcher.search, board, seconds)
    stats = pstats.Stats(profiler)
    total = stats.total_tt  # type: ignore[attr-defined]

    shares: dict[str, float] = {}
    for (filename, _line, func), (_cc, _nc, _tt, ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
        if func in TRACKED and ("chess" in filename or "torch" in filename):
            if func == "evaluate" and "evaluators" not in filename:
                continue
            if func in ("push", "pop") and "chess" not in filename:
                continue
            shares[func] = max(shares.get(func, 0.0), ct / total)

    buf = io.StringIO()
    buf.write(f"cProfile of a {seconds:g}s search ({name}): depth {result.depth}, {result.nodes:,} nodes, "
              f"{result.evals:,} evals, {result.network_calls:,} network calls, profiled total {total:.2f}s\n\n")
    buf.write("Cumulative share of profiled time (inclusive of callees):\n")
    for func, label in TRACKED.items():
        if func in shares:
            buf.write(f"  {label:45s} {shares[func]:6.1%}\n")
    for sort in ("tottime", "cumulative"):
        buf.write(f"\n--- top {top} functions by {sort} ---\n")
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(sort).print_stats(top)
        text = stream.getvalue()
        buf.write(text[text.find("   ncalls"):] if "   ncalls" in text else text)
    return buf.getvalue(), shares


def main() -> int:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluators", nargs="+", default=list(EVALUATOR_NAMES), choices=EVALUATOR_NAMES)
    parser.add_argument("--position", default="ruy_lopez_closed",
                        help="benchmark position name: " + ", ".join(p.name for p in BENCHMARK_POSITIONS))
    parser.add_argument("--time", type=float, default=5.0, help="seconds per search")
    parser.add_argument("--top", type=int, default=25, help="functions listed per cProfile table")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    by_name = {p.name: p for p in BENCHMARK_POSITIONS}
    if args.position not in by_name:
        parser.error(f"unknown position {args.position}")
    board = chess.Board(by_name[args.position].fen)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.evaluators:
        print(f"== {name}: {args.time:g}s search on {args.position}")
        try:
            row = wall_clock(name, board, args.time)
            report, shares = profiled(name, board, args.time, args.top)
        except (ConfigError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        row["cprofile_shares"] = shares
        rows.append(row)
        header = (
            f"Wall-clock attribution (no profiler): {row['nodes']:,} nodes in {row['search_seconds']:.2f}s "
            f"= {row['nps']:,.0f} nodes/s, depth {row['depth']}\n"
            f"  time in evaluate():           {row['eval_share']:6.1%}  ({row['us_per_eval']:.1f} us per call)\n"
            f"  time in innermost static_eval {row['static_share']:6.1%}  ({row['us_per_static_call']:.1f} us per call)\n"
            f"  rest of the search:           {1 - row['eval_share']:6.1%}  "
            f"({row['us_per_node_excluding_eval']:.1f} us per node)\n"
            f"  nodes/s if evaluation were free: {row['nps_if_eval_free']:,.0f}\n\n"
        )
        (out_dir / f"profile_{name}.txt").write_text(header + report, encoding="utf-8")
        print(header.rstrip())

    headers = ["evaluator", "nodes/sec", "depth", "evals", "net calls", "time in evaluate()",
               "time in static eval / network", "us per eval", "nodes/sec if eval free"]
    table_rows = [
        [r["evaluator"], round(r["nps"]), r["depth"], r["evals"], r["network_calls"], f"{r['eval_share']:.1%}",
         f"{r['static_share']:.1%}", round(r["us_per_eval"], 1), round(r["nps_if_eval_free"])]
        for r in rows
    ]
    table = markdown_table(headers, table_rows)
    note = f"Search profile: {args.time:g}s on `{args.position}`, wall-clock timers around the evaluator.\n\n"
    (out_dir / "profile_summary.md").write_text(note + table + "\n", encoding="utf-8")
    write_json(out_dir / "profile_summary.json", rows)
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
