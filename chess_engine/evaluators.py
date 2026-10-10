"""Position evaluators.

Every evaluator implements one interface::

    evaluate(board) -> int   # centipawns from the side to move's perspective

``Evaluator.evaluate`` first applies the rules of chess (checkmate and every draw
condition) via :func:`terminal_score`, then defers to the subclass's
``static_eval`` for ordinary positions. Wrappers such as a cache only ever wrap
``static_eval``, because rule-based scores depend on move history (repetitions,
the fifty-move clock) and on the ply, which a position hash does not capture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import chess

from config import Config, MaterialConfig, get_config

MATE_SCORE = 100_000
# Any score at least this large in magnitude is a forced mate. Mate scores are
# MATE_SCORE minus the game ply, so this leaves room for 10,000 plies.
MATE_BOUND = MATE_SCORE - 10_000
DRAW_SCORE = 0


def mated_score(board: chess.Board) -> int:
    """Score for the side to move being checkmated in ``board``.

    Using the game ply makes faster mates score higher for the winner (and
    slower ones higher for the loser) without the evaluator knowing the search
    root: within one search the root ply is constant, so mate distance is
    ``MATE_SCORE - |score| - root_ply``.
    """
    return -(MATE_SCORE - board.ply())


def is_mate_score(score: float) -> bool:
    return abs(score) >= MATE_BOUND


def terminal_score(board: chess.Board) -> int | None:
    """Rules-based score if the game is over or a draw can be claimed, else None.

    Covers checkmate, stalemate, insufficient material, the fifty-move rule
    (claimable at 100 half-moves; the 75-move rule is implied) and threefold
    repetition (claimable; fivefold is implied).
    """
    # Checkmate and stalemate both mean "no legal moves"; any() stops at the first one.
    if not any(board.generate_legal_moves()):
        return mated_score(board) if board.is_check() else DRAW_SCORE
    if board.is_insufficient_material():
        return DRAW_SCORE
    if board.halfmove_clock >= 100:
        return DRAW_SCORE
    # A third occurrence needs at least 8 reversible plies; skip the history scan otherwise.
    if board.halfmove_clock >= 8 and board.is_repetition(3):
        return DRAW_SCORE
    return None


class Evaluator(ABC):
    """Base class: rules first, then a subclass-specific static evaluation."""

    name = "evaluator"

    def __init__(self) -> None:
        self.calls = 0  # evaluate() calls
        self.terminal = 0  # calls answered by the rules alone

    def evaluate(self, board: chess.Board) -> int:
        """Centipawn score of ``board`` from the side to move's perspective."""
        self.calls += 1
        score = terminal_score(board)
        if score is not None:
            self.terminal += 1
            return score
        return self.static_eval(board)

    @abstractmethod
    def static_eval(self, board: chess.Board) -> int:
        """Score of a non-terminal position, side to move's perspective."""

    def new_game(self) -> None:
        """Hook called before each new game (evaluators with state reset here)."""

    def network_calls(self) -> int:
        """Number of neural-network forward passes made (0 for non-NN evaluators)."""
        return 0

    def reset_stats(self) -> None:
        self.calls = 0
        self.terminal = 0

    def stats(self) -> dict[str, float]:
        return {"calls": self.calls, "terminal": self.terminal, "network_calls": self.network_calls()}


class MaterialEvaluator(Evaluator):
    """Hand-tuned material count: sum of piece values, side to move minus opponent."""

    name = "material"

    def __init__(self, values: MaterialConfig | None = None) -> None:
        super().__init__()
        v = values or MaterialConfig()
        self.values = (v.pawn, v.knight, v.bishop, v.rook, v.queen)

    def static_eval(self, board: chess.Board) -> int:
        white = board.occupied_co[chess.WHITE]
        black = board.occupied_co[chess.BLACK]
        p, n, b, r, q = self.values
        score = 0
        for bb, value in (
            (board.pawns, p),
            (board.knights, n),
            (board.bishops, b),
            (board.rooks, r),
            (board.queens, q),
        ):
            score += value * ((bb & white).bit_count() - (bb & black).bit_count())
        return score if board.turn == chess.WHITE else -score


EVALUATOR_NAMES = ("material",)


def build_evaluator(name: str, cfg: Config | None = None) -> Evaluator:
    """Construct an evaluator by its short name (see EVALUATOR_NAMES)."""
    cfg = cfg or get_config()
    if name == "material":
        return MaterialEvaluator(cfg.material)
    raise ValueError(f"Unknown evaluator '{name}'; choose from {', '.join(EVALUATOR_NAMES)}")
