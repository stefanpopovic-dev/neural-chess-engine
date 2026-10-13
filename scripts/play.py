#!/usr/bin/env python
"""Play a game against the engine in the terminal.

Enter moves in standard notation (e4, Nf3, O-O, exd5, e8=Q) or UCI (e2e4).
Commands: moves (list legal moves), undo (take back your last move),
hint (ask the engine), flip (turn the board), resign, quit, help.

Examples:
    python scripts/play.py                         # you play White vs NN + cache, 1 s/move
    python scripts/play.py --color black --time 3
    python scripts/play.py --engine material
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

from chess_engine.evaluators import EVALUATOR_NAMES, build_evaluator  # noqa: E402
from chess_engine.search import SearchResult, Searcher  # noqa: E402
from config import ConfigError, get_config  # noqa: E402

HELP = """Moves: standard notation (e4, Nf3, O-O, exd5, e8=Q) or UCI (e2e4, e7e8q).
Commands: moves, undo, hint, flip, resign, quit, help"""


def render(board: chess.Board, white_bottom: bool) -> str:
    """ASCII board: uppercase = White, lowercase = Black, '.' = empty."""
    ranks = range(7, -1, -1) if white_bottom else range(8)
    files = range(8) if white_bottom else range(7, -1, -1)
    lines = []
    for rank in ranks:
        row = []
        for file in files:
            piece = board.piece_at(chess.square(file, rank))
            row.append(piece.symbol() if piece else ".")
        lines.append(f"  {rank + 1}  " + " ".join(row))
    lines.append("     " + " ".join("abcdefgh"[f] for f in files))
    return "\n".join(lines)


def game_over(board: chess.Board) -> str | None:
    """Result text if the game has ended (draws by repetition / fifty moves are claimed automatically)."""
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        reason = outcome.termination.name.replace("_", " ").lower()
        if outcome.winner is None:
            return f"Draw ({reason})"
        return f"{'White' if outcome.winner else 'Black'} wins ({reason})"
    if board.is_repetition(3):
        return "Draw (threefold repetition)"
    if board.halfmove_clock >= 100:
        return "Draw (fifty-move rule)"
    return None


def describe(result: SearchResult, side: str = "the engine") -> str:
    """Search summary; the score is from the point of view of ``side`` (the side to move)."""
    if result.mate_in is not None:
        winner = side if result.mate_in > 0 else ("you" if side == "the engine" else "the engine")
        score = f"mate in {abs(result.mate_in)} for {winner}"
    else:
        score = f"{result.score / 100:+.2f} for {side}"
    return f"depth {result.depth}, {score}, {result.nodes:,} nodes in {result.time:.1f}s"


def parse_move(board: chess.Board, text: str) -> chess.Move | None:
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(text)
        except ValueError:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", default="nn-cache", choices=EVALUATOR_NAMES, help="engine evaluator")
    parser.add_argument("--time", type=float, default=1.0, help="engine seconds per move")
    parser.add_argument("--color", default="white", choices=["white", "black", "random"], help="your colour")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="start from this position")
    args = parser.parse_args()

    cfg = get_config()
    try:
        evaluator = build_evaluator(args.engine, cfg)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    searcher = Searcher(evaluator, cfg.search)
    evaluator.new_game()

    try:
        board = chess.Board(args.fen)
    except ValueError as exc:
        print(f"error: bad FEN: {exc}", file=sys.stderr)
        return 1
    color = random.choice(["white", "black"]) if args.color == "random" else args.color
    human = chess.WHITE if color == "white" else chess.BLACK
    white_bottom = human == chess.WHITE
    print(f"You play {color.capitalize()} against the '{args.engine}' engine at {args.time:g}s per move.")
    print(HELP)

    result_text = None
    while True:
        print("\n" + render(board, white_bottom))
        result_text = game_over(board)
        if result_text:
            break

        if board.turn != human:
            print("\nEngine is thinking...", flush=True)
            result = searcher.search(board, time_limit=args.time)
            print(f"Engine plays {board.san(result.best_move)}   ({describe(result)})")
            board.push(result.best_move)
            continue

        try:
            text = input(f"\nYour move ({'White' if human else 'Black'}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            result_text = "Game abandoned"
            break
        command = text.lower()
        if not text:
            continue
        if command in ("quit", "exit", "q"):
            result_text = "Game abandoned"
            break
        if command == "help":
            print(HELP)
        elif command == "moves":
            print(" ".join(sorted(board.san(m) for m in board.legal_moves)))
        elif command == "flip":
            white_bottom = not white_bottom
        elif command == "hint":
            hint = searcher.search(board, time_limit=args.time)
            print(f"Hint: {board.san(hint.best_move)}   ({describe(hint, side='you')})")
        elif command == "undo":
            if len(board.move_stack) >= 2:
                board.pop()
                board.pop()
            else:
                print("Nothing to undo.")
        elif command == "resign":
            result_text = f"{'Black' if human else 'White'} wins (resignation)"
            break
        else:
            move = parse_move(board, text)
            if move is None or move not in board.legal_moves:
                print(f"'{text}' is not a legal move here. Type 'moves' to list them.")
                continue
            board.push(move)

    print(f"\n{result_text}.")
    if board.move_stack:
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = "You" if human == chess.WHITE else f"engine ({args.engine})"
        game.headers["Black"] = "You" if human == chess.BLACK else f"engine ({args.engine})"
        print("\nPGN:\n" + game.accept(chess.pgn.StringExporter(headers=False)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
