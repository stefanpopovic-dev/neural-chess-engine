#!/usr/bin/env python
"""Self-play benchmark: the NN + cache engine against the material baseline.

Games start from a book of 50 openings; each opening is played twice with the
colours swapped. Both engines get the same fixed time per move. A game ends by
the rules of chess as implemented by python-chess (checkmate, stalemate,
insufficient material, the 75-move and fivefold-repetition rules, with
threefold repetition and the fifty-move rule claimed automatically) or at a
move cap, which is scored as a draw.

Each finished game is appended to results.jsonl and games.pgn in the output
directory, so an interrupted run resumes when restarted with the same settings
(raise --games to extend a finished run).

Examples:
    python scripts/match.py --quick                  # 4 short games: checks the pipeline
    python scripts/match.py                          # full match with config defaults
    python scripts/match.py --games 100 --time 0.5 --workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

from chess_engine.evaluators import EVALUATOR_NAMES, build_evaluator  # noqa: E402
from chess_engine.openings import OPENINGS  # noqa: E402
from chess_engine.reporting import markdown_table, write_json  # noqa: E402
from chess_engine.search import SearchResult, Searcher  # noqa: E402
from config import ConfigError, get_config, resolve_path  # noqa: E402

QUICK = {"games": 4, "time": 0.05, "max_plies": 60, "out_dir": "results/match_quick"}


@dataclass(frozen=True)
class GameSpec:
    game_id: int
    opening: int  # index into OPENINGS
    white: str
    black: str
    a_is_white: bool  # whether engine A has the white pieces


def schedule(games: int, engine_a: str, engine_b: str, n_openings: int = len(OPENINGS)) -> list[GameSpec]:
    """Game g uses opening g // 2; even games give engine A white, odd games black."""
    specs = []
    for g in range(games):
        a_white = g % 2 == 0
        white, black = (engine_a, engine_b) if a_white else (engine_b, engine_a)
        specs.append(GameSpec(g, (g // 2) % n_openings, white, black, a_white))
    return specs


def adjudicate(board: chess.Board, max_plies: int) -> tuple[str, str] | None:
    """(termination, result) if the game is over, else None."""
    outcome = board.outcome()  # checkmate, stalemate, insufficient material, 75 moves, fivefold
    if outcome is not None:
        return outcome.termination.name.lower(), outcome.result()
    if board.halfmove_clock >= 100:
        return "fifty_moves", "1/2-1/2"
    if board.is_repetition(3):
        return "threefold_repetition", "1/2-1/2"
    if board.ply() >= max_plies:
        return "move_cap", "1/2-1/2"
    return None


def format_score(result: SearchResult) -> str:
    mate = result.mate_in
    if mate is not None:
        return f"#{mate}"
    return f"{result.score / 100:+.2f}"


# ------------------------------------------------------------------ game play

_SEARCHERS: dict[tuple[str, bool], Searcher] = {}  # per process: (engine, colour) -> searcher


def _init_worker() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _searcher(name: str, color: chess.Color) -> Searcher:
    # One instance per colour, so two copies of the same engine never share a cache.
    key = (name, color)
    if key not in _SEARCHERS:
        cfg = get_config()
        _SEARCHERS[key] = Searcher(build_evaluator(name, cfg), cfg.search)
    return _SEARCHERS[key]


def play_game(spec: GameSpec, time_limit: float, max_plies: int, early_stop: float) -> dict[str, Any]:
    """Play one game and return its record (including the PGN text)."""
    cfg = get_config()
    opening = OPENINGS[spec.opening]
    board = opening.board()
    book_plies = board.ply()
    searchers = {chess.WHITE: _searcher(spec.white, chess.WHITE), chess.BLACK: _searcher(spec.black, chess.BLACK)}
    totals: dict[chess.Color, Counter[str]] = {}
    for color, searcher in searchers.items():
        searcher.config = dataclasses.replace(cfg.search, early_stop_fraction=early_stop)
        searcher.evaluator.new_game()  # fresh cache per game; it then persists across moves
        searcher.evaluator.reset_stats()
        totals[color] = Counter()

    comments: list[str] = []
    start = time.perf_counter()
    while (ending := adjudicate(board, max_plies)) is None:
        searcher = searchers[board.turn]
        result = searcher.search(board, time_limit=time_limit)
        t = totals[board.turn]
        t["moves"] += 1
        t["depth"] += result.depth
        t["nodes"] += result.nodes
        t["evals"] += result.evals
        t["network_calls"] += result.network_calls
        t["time"] += result.time
        comments.append(f"d{result.depth} {format_score(result)}")
        board.push(result.best_move)
    termination, outcome = ending

    def engine_stats(color: chess.Color) -> dict[str, Any]:
        t = totals[color]
        moves = max(t["moves"], 1)
        stats = searchers[color].evaluator.stats()
        lookups = stats.get("cache_hits", 0) + stats.get("cache_misses", 0)
        return {
            "engine": spec.white if color == chess.WHITE else spec.black,
            "moves": t["moves"],
            "depth": t["depth"] / moves,
            "nps": t["nodes"] / t["time"] if t["time"] else 0.0,
            "evals_per_move": t["evals"] / moves,
            "network_calls_per_move": t["network_calls"] / moves,
            "cache_hit_rate": stats.get("cache_hits", 0) / lookups if lookups else None,
        }

    game = chess.pgn.Game.from_board(board)
    game.headers.update(
        {
            "Event": f"{spec.white if spec.a_is_white else spec.black} vs "
                     f"{spec.black if spec.a_is_white else spec.white}",
            "Site": "local",
            "Date": time.strftime("%Y.%m.%d"),
            "Round": str(spec.game_id + 1),
            "White": spec.white,
            "Black": spec.black,
            "Result": outcome,
            "Opening": opening.name,
            "TimeControl": f"{time_limit:g}s/move",
            "Termination": termination,
            "PlyCount": str(board.ply()),
        }
    )
    node = game
    for ply, _move in enumerate(board.move_stack):
        node = node.next()  # type: ignore[assignment]
        if ply >= book_plies:
            node.comment = comments[ply - book_plies]
        elif ply == book_plies - 1:
            node.comment = "book"

    return {
        "game_id": spec.game_id,
        "opening": opening.name,
        "white": spec.white,
        "black": spec.black,
        "a_is_white": spec.a_is_white,
        "result": outcome,
        "termination": termination,
        "plies": board.ply(),
        "seconds": round(time.perf_counter() - start, 2),
        "stats": {"white": engine_stats(chess.WHITE), "black": engine_stats(chess.BLACK)},
        "pgn": str(game),
    }


def _play_game_job(args: tuple[GameSpec, float, int, float]) -> dict[str, Any]:
    return play_game(*args)


# ------------------------------------------------------------------ statistics


def elo_from_score(p: float) -> float:
    return -400.0 * math.log10(1.0 / p - 1.0) + 0.0  # + 0.0 turns -0.0 into 0.0


def elo_estimate(wins: int, draws: int, losses: int, z: float = 1.96) -> dict[str, float]:
    """Elo difference with a confidence margin from the per-game score variance.

    The score's standard error is sd(per-game score) / sqrt(n); the score
    interval p +/- z*se is mapped through the logistic Elo curve. Scores of 0 or
    100% are clamped by half a game so the estimate stays finite.
    """
    n = wins + draws + losses
    if n == 0:
        return {"games": 0, "score": float("nan"), "elo": float("nan"), "margin": float("nan"),
                "elo_low": float("nan"), "elo_high": float("nan"), "los": float("nan")}
    p = (wins + 0.5 * draws) / n
    var = (wins * (1 - p) ** 2 + draws * (0.5 - p) ** 2 + losses * p**2) / n
    se = math.sqrt(var / n)
    eps = 0.5 / n
    clamp = lambda x: min(max(x, eps), 1 - eps)  # noqa: E731
    low, high = elo_from_score(clamp(p - z * se)), elo_from_score(clamp(p + z * se))
    decisive = wins + losses
    los = 0.5 * (1 + math.erf((wins - losses) / math.sqrt(2 * decisive))) if decisive else 0.5
    return {
        "games": n,
        "score": p,
        "elo": elo_from_score(clamp(p)),
        "margin": (high - low) / 2,
        "elo_low": low,
        "elo_high": high,
        "los": los,
    }


def score_for_a(record: dict[str, Any]) -> float:
    result = record["result"]
    if result == "1/2-1/2":
        return 0.5
    white_won = result == "1-0"
    return 1.0 if white_won == record["a_is_white"] else 0.0


def summarize(records: list[dict[str, Any]], engine_a: str, engine_b: str) -> dict[str, Any]:
    scores = [score_for_a(r) for r in records]
    wins, draws, losses = scores.count(1.0), scores.count(0.5), scores.count(0.0)

    def by_colour(a_white: bool) -> dict[str, int]:
        s = [score_for_a(r) for r in records if r["a_is_white"] == a_white]
        return {"wins": s.count(1.0), "draws": s.count(0.5), "losses": s.count(0.0)}

    def engine_average(role_a: bool, key: str) -> float | None:
        values = []
        for r in records:
            colour = "white" if r["a_is_white"] == role_a else "black"
            value = r["stats"][colour][key]
            if value is not None:
                values.append(value)
        return statistics.fmean(values) if values else None

    engines = {}
    for role, name in (("a", engine_a), ("b", engine_b)):
        role_a = role == "a"
        engines[role] = {
            "name": name,
            **{k: engine_average(role_a, k)
               for k in ("depth", "nps", "evals_per_move", "network_calls_per_move", "cache_hit_rate")},
        }
    return {
        "engine_a": engine_a,
        "engine_b": engine_b,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        **elo_estimate(wins, draws, losses),
        "a_as_white": by_colour(True),
        "a_as_black": by_colour(False),
        "terminations": dict(Counter(r["termination"] for r in records).most_common()),
        "avg_plies": statistics.fmean(r["plies"] for r in records) if records else 0,
        "engines": engines,
    }


def summary_markdown(summary: dict[str, Any], settings: dict[str, Any]) -> str:
    a, b = summary["engine_a"], summary["engine_b"]
    lines = [
        f"# Match: {a} vs {b}",
        "",
        f"{summary['games']} games, {settings['time_limit']:g}s per move, move cap {settings['max_plies']} plies "
        f"(scored as a draw), {settings['workers']} parallel game(s), {settings['openings']} openings x 2 colours.",
        "",
        markdown_table(
            ["engine A", "W", "D", "L", "score", "Elo diff (A - B)", "95% CI", "LOS"],
            [[a, summary["wins"], summary["draws"], summary["losses"], f"{summary['score']:.1%}",
              f"{summary['elo']:+.0f} +/- {summary['margin']:.0f}",
              f"[{summary['elo_low']:+.0f}, {summary['elo_high']:+.0f}]", f"{summary['los']:.1%}"]],
        ),
        "",
        f"As White: {summary['a_as_white']}; as Black: {summary['a_as_black']}.",
        f"Terminations: {summary['terminations']}; average game length {summary['avg_plies']:.0f} plies.",
        "",
        markdown_table(
            ["engine", "avg depth", "nodes/sec", "evals/move", "net calls/move", "cache hit rate"],
            [[e["name"], round(e["depth"] or 0, 2), round(e["nps"] or 0), round(e["evals_per_move"] or 0),
              round(e["network_calls_per_move"] or 0),
              f"{e['cache_hit_rate']:.1%}" if e["cache_hit_rate"] is not None else "-"]
             for e in summary["engines"].values()],
        ),
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ run management


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a partially written last line from an interrupted run
    return records


def rewrite_pgn(path: Path, records: list[dict[str, Any]]) -> None:
    """games.pgn is derived from results.jsonl; rebuilding drops games without a result record."""
    text = "".join(r["pgn"] + "\n\n" for r in sorted(records, key=lambda r: r["game_id"]))
    path.write_text(text, encoding="utf-8")


def format_duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{max(minutes, 1)} min"


def estimate_duration(
    remaining: int, time_limit: float, max_plies: int, workers: int, records: list[dict[str, Any]]
) -> tuple[float, float]:
    """(expected, upper bound) wall-clock seconds for the remaining games."""
    book = statistics.fmean(o.board().ply() for o in OPENINGS)
    if records:
        plies = statistics.fmean(r["plies"] for r in records)
        per_ply = statistics.fmean(r["seconds"] / max(r["plies"] - book, 1) for r in records)
    else:
        plies = min(max_plies, 120)  # typical length of games between these engines
        per_ply = time_limit * 1.03  # search plus move overhead
    rounds = math.ceil(remaining / max(workers, 1))
    return rounds * (plies - book) * per_ply, rounds * (max_plies - book) * per_ply


def main() -> int:
    cfg = get_config()
    m = cfg.match
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=None, help=f"number of games (default {m.games})")
    parser.add_argument("--time", type=float, default=None, help=f"seconds per move (default {m.time_limit})")
    parser.add_argument("--max-plies", type=int, default=None, help=f"move cap in plies (default {m.max_plies})")
    parser.add_argument("--workers", type=int, default=m.workers, help="games played in parallel")
    parser.add_argument("--engine-a", default=m.engine_a, choices=EVALUATOR_NAMES)
    parser.add_argument("--engine-b", default=m.engine_b, choices=EVALUATOR_NAMES)
    parser.add_argument("--early-stop", type=float, default=cfg.search.early_stop_fraction,
                        help="stop deepening after this fraction of the move time (1.0 = use it all)")
    parser.add_argument("--out-dir", default=None, help=f"results directory (default {m.out_dir})")
    parser.add_argument("--quick", action="store_true",
                        help=f"smoke test: {QUICK['games']} games at {QUICK['time']}s/move, "
                             f"{QUICK['max_plies']}-ply cap, in {QUICK['out_dir']}")
    args = parser.parse_args()

    defaults = QUICK if args.quick else {"games": m.games, "time": m.time_limit, "max_plies": m.max_plies,
                                         "out_dir": m.out_dir}
    games = args.games if args.games is not None else defaults["games"]
    time_limit = args.time if args.time is not None else defaults["time"]
    max_plies = args.max_plies if args.max_plies is not None else defaults["max_plies"]
    out_dir = resolve_path(args.out_dir or defaults["out_dir"])
    workers = max(1, args.workers)

    settings = {
        "engine_a": args.engine_a,
        "engine_b": args.engine_b,
        "time_limit": time_limit,
        "max_plies": max_plies,
        "early_stop": args.early_stop,
        "openings": len(OPENINGS),
        "nn_checkpoint": cfg.nn.checkpoint,
        "search": dataclasses.asdict(cfg.search) | {"early_stop_fraction": args.early_stop},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    settings_path = out_dir / "settings.json"
    results_path = out_dir / "results.jsonl"
    pgn_path = out_dir / "games.pgn"

    records = load_records(results_path)
    if records and settings_path.exists():
        previous = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous != settings:
            changed = sorted(k for k in settings if previous.get(k) != settings[k])
            print(f"error: {out_dir} holds a run with different settings ({', '.join(changed)}). "
                  f"Use the same settings to resume, or choose another --out-dir.", file=sys.stderr)
            return 1
    write_json(settings_path, settings)
    # Rewrite results/PGN from the valid records (drops a torn last line or orphaned PGN).
    results_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    rewrite_pgn(pgn_path, records)

    done = {r["game_id"] for r in records}
    pending = [s for s in schedule(games, args.engine_a, args.engine_b) if s.game_id not in done]

    # Fail fast (e.g. missing checkpoint) before starting worker processes.
    try:
        for name in {args.engine_a, args.engine_b}:
            build_evaluator(name, cfg)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Match {args.engine_a} vs {args.engine_b}: {games} games at {time_limit:g}s/move, "
          f"cap {max_plies} plies, {workers} worker(s) -> {out_dir}")
    if records:
        print(f"Resuming: {len(records)} games already played, {len(pending)} to go")
    if workers > 1:
        cores = os.cpu_count() or 1
        print(f"WARNING: {workers} games run in parallel and share this machine's {cores} CPU cores "
              f"(plus memory bandwidth and turbo headroom). Each engine searches fewer nodes per move than "
              f"in a serial run, which lowers its depth and can change the result.")
        if workers > cores:
            print("WARNING: more workers than CPU cores; engines will be time-sliced.")
    expected, upper = estimate_duration(len(pending), time_limit, max_plies, workers, records)
    if pending:
        print(f"Estimated duration: ~{format_duration(expected)} "
              f"(at most {format_duration(upper)} if every game reaches the move cap)")

    jobs = [(spec, time_limit, max_plies, args.early_stop) for spec in pending]
    started = time.perf_counter()
    try:
        with results_path.open("a", encoding="utf-8") as results_fh, pgn_path.open("a", encoding="utf-8") as pgn_fh:

            def record_game(record: dict[str, Any]) -> None:
                record["workers"] = workers
                results_fh.write(json.dumps(record) + "\n")
                results_fh.flush()
                pgn_fh.write(record["pgn"] + "\n\n")
                pgn_fh.flush()
                records.append(record)
                s = summarize(records, args.engine_a, args.engine_b)
                a_colour = "W" if record["a_is_white"] else "B"
                print(f"[{len(records):3d}/{games}] {record['opening'][:28]:28s} {args.engine_a}({a_colour}) "
                      f"{record['result']:7s} {record['termination']:20s} {record['plies']:3d} plies  "
                      f"| {args.engine_a} {s['wins'] + 0.5 * s['draws']:.1f}/{s['games']} ({s['score']:.1%})",
                      flush=True)

            if workers == 1:
                _init_worker()
                for job in jobs:
                    record_game(_play_game_job(job))
            else:
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker) as pool:
                    futures: list[Future[dict[str, Any]]] = [pool.submit(_play_game_job, job) for job in jobs]
                    try:
                        for future in as_completed(futures):
                            record_game(future.result())
                    except BaseException:
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(records)} games; rerun the same command to resume.", file=sys.stderr)
        return 130

    rewrite_pgn(pgn_path, records)  # parallel games finish out of order; store them by round
    summary = summarize(records, args.engine_a, args.engine_b)
    summary["settings"] = settings | {"workers": workers}
    summary["elapsed_minutes_this_run"] = (time.perf_counter() - started) / 60
    write_json(out_dir / "summary.json", summary)
    text = summary_markdown(summary, settings | {"workers": workers})
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
