"""Negamax alpha-beta search over python-chess move generation.

Features:
  * iterative deepening with a per-move time limit; the returned move always
    comes from the last *fully completed* depth (depth 1 is never interrupted,
    so a legal move is always available);
  * quiescence search over captures and queen promotions (with full evasion
    search when in check) to avoid the horizon effect;
  * move ordering: previous iteration's principal-variation move first, then
    captures by MVV-LVA (most valuable victim, least valuable attacker), queen
    promotions, checking moves, and finally quiet moves;
  * draw detection inside the tree (repetition, fifty-move rule, insufficient
    material) and mate-distance-aware scores;
  * node and leaf-evaluation counters for benchmarking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import chess

from chess_engine.evaluators import DRAW_SCORE, MATE_SCORE, Evaluator, is_mate_score, mated_score
from config import SearchConfig

INF = MATE_SCORE + 1
MAX_PLY = 128

# Ordering keys: higher is searched first.
_KEY_PV = 1_000_000
_KEY_CAPTURE = 100_000
_KEY_PROMOTION = 90_000
_KEY_CHECK = 50_000
_KEY_UNDERPROMOTION = -1

_BB_SQUARES = chess.BB_SQUARES


class SearchTimeout(Exception):
    """Raised inside the tree when the time or node budget is exhausted."""


@dataclass
class IterationInfo:
    """Result of one completed iterative-deepening iteration."""

    depth: int
    score: int
    move: chess.Move
    nodes: int
    evals: int
    elapsed: float


@dataclass
class SearchResult:
    """Outcome of :meth:`Searcher.search`."""

    best_move: chess.Move | None
    score: int  # centipawns, side to move at the root; mate scores are +/-(MATE_SCORE - ply)
    depth: int  # last fully completed depth
    nodes: int  # all nodes, including quiescence nodes
    qnodes: int  # quiescence nodes
    evals: int  # leaf evaluations requested from the evaluator
    network_calls: int  # forward passes the evaluator actually made (NN only)
    time: float  # seconds
    pv: list[chess.Move] = field(default_factory=list)
    iterations: list[IterationInfo] = field(default_factory=list)
    root_ply: int = 0

    @property
    def nps(self) -> float:
        return self.nodes / self.time if self.time > 0 else 0.0

    @property
    def mate_in(self) -> int | None:
        """Moves to mate (positive: side to move mates; negative: gets mated), else None."""
        if not is_mate_score(self.score):
            return None
        plies = MATE_SCORE - abs(self.score) - self.root_ply
        moves = (plies + 1) // 2
        return moves if self.score > 0 else -moves


def _position_key(board: chess.Board) -> tuple[int, ...]:
    """Cheap repetition key. Raw ep_square is fine: a position right after a double
    pawn push can never repeat an earlier one within the reversible-move window."""
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.turn,
        board.castling_rights,
        board.ep_square or -1,
    )


class Searcher:
    """Iterative-deepening negamax alpha-beta search driven by an :class:`Evaluator`."""

    def __init__(self, evaluator: Evaluator, config: SearchConfig | None = None) -> None:
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        if self.config.check_interval & (self.config.check_interval - 1):
            raise ValueError("check_interval must be a power of two")
        self._mask = self.config.check_interval - 1
        self.nodes = self.qnodes = self.evals = 0
        self._deadline = float("inf")
        self._node_limit: int | None = None
        self._can_stop = False
        self._keys: list[tuple[int, ...]] = []
        self._pv: list[list[chess.Move]] = [[] for _ in range(MAX_PLY + 1)]
        self._prev_pv: list[chess.Move] = []

    # ------------------------------------------------------------------ public

    def search(
        self,
        board: chess.Board,
        time_limit: float | None = None,
        max_depth: int | None = None,
        max_nodes: int | None = None,
    ) -> SearchResult:
        """Search ``board`` and return the best move of the last completed depth.

        ``time_limit`` (seconds) and ``max_depth`` default to the config values.
        ``max_nodes`` optionally bounds the search by node count instead, which
        makes results reproducible. The caller's board is never modified.
        """
        cfg = self.config
        time_limit = cfg.time_limit if time_limit is None else time_limit
        max_depth = cfg.max_depth if max_depth is None else max_depth
        start = time.perf_counter()
        self._deadline = start + time_limit
        self._node_limit = max_nodes
        self.nodes = self.qnodes = self.evals = 0
        net_before = self.evaluator.network_calls()

        board = board.copy()  # the tree search may be aborted mid-line by a timeout
        root_ply = board.ply()
        root_moves = list(board.generate_legal_moves())
        if not root_moves:
            score = mated_score(board) if board.is_check() else DRAW_SCORE
            return SearchResult(None, score, 0, 0, 0, 0, 0, time.perf_counter() - start, root_ply=root_ply)

        self._init_repetition_keys(board)
        root_moves = self._order_moves(board, root_moves, None)
        best_move, best_score, completed, pv = root_moves[0], 0, 0, []
        iterations: list[IterationInfo] = []

        for depth in range(1, max_depth + 1):
            self._can_stop = depth > 1  # depth 1 always completes so a move exists
            self._prev_pv = pv
            try:
                score, move, line = self._search_root(board, depth, root_moves)
            except SearchTimeout:
                break
            completed, best_move, best_score, pv = depth, move, score, line
            root_moves.remove(move)
            root_moves.insert(0, move)  # previous iteration's best move goes first
            elapsed = time.perf_counter() - start
            iterations.append(IterationInfo(depth, score, move, self.nodes, self.evals, elapsed))
            if is_mate_score(score) and MATE_SCORE - abs(score) - root_ply <= depth:
                break  # a proven mate within the searched depth cannot improve
            if elapsed >= time_limit * cfg.early_stop_fraction:
                break
            if max_nodes is not None and self.nodes >= max_nodes:
                break

        return SearchResult(
            best_move=best_move,
            score=best_score,
            depth=completed,
            nodes=self.nodes,
            qnodes=self.qnodes,
            evals=self.evals,
            network_calls=self.evaluator.network_calls() - net_before,
            time=time.perf_counter() - start,
            pv=pv,
            iterations=iterations,
            root_ply=root_ply,
        )

    # ------------------------------------------------------------- tree search

    def _search_root(
        self, board: chess.Board, depth: int, root_moves: list[chess.Move]
    ) -> tuple[int, chess.Move, list[chess.Move]]:
        alpha, beta = -INF, INF
        best_score, best_move = -INF, root_moves[0]
        pv: list[chess.Move] = []
        for move in root_moves:
            board.push(move)
            self._keys.append(_position_key(board))
            score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            self._keys.pop()
            board.pop()
            if score > best_score:
                best_score, best_move = score, move
                pv = [move] + self._pv[1]
            if score > alpha:
                alpha = score
        return best_score, best_move, pv

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._tick()
        self._pv[ply] = []

        if self._is_rule_draw(board):
            return DRAW_SCORE
        if depth <= 0 or ply >= MAX_PLY:
            if self.config.quiescence and ply < MAX_PLY:
                return self._quiesce(board, alpha, beta, ply, 0)
            return self._evaluate(board)

        moves = list(board.generate_legal_moves())
        if not moves:
            return mated_score(board) if board.is_check() else DRAW_SCORE

        pv_move = self._prev_pv[ply] if ply < len(self._prev_pv) else None
        best = -INF
        for move in self._order_moves(board, moves, pv_move):
            board.push(move)
            self._keys.append(_position_key(board))
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            self._keys.pop()
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    self._pv[ply] = [move] + self._pv[ply + 1]
                    if alpha >= beta:
                        break
        return best

    def _quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int, qdepth: int) -> int:
        """Search captures (and all evasions when in check) until the position is quiet."""
        self._tick()
        self.qnodes += 1
        self._pv[ply] = []
        at_limit = qdepth >= self.config.qsearch_max_depth or ply >= MAX_PLY

        if board.is_check():
            # No stand-pat in check: the side to move must answer it.
            if at_limit:
                return self._evaluate(board)
            moves = list(board.generate_legal_moves())
            if not moves:
                return mated_score(board)
            best = -INF
        else:
            stand_pat = self._evaluate(board)
            if at_limit or stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best = stand_pat
            moves = self._noisy_moves(board)

        for move in self._order_moves(board, moves, None, checks=False):
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, ply + 1, qdepth + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # ----------------------------------------------------------------- helpers

    def _tick(self) -> None:
        """Count a node and abort the iteration if the budget is exhausted."""
        self.nodes += 1
        if self.nodes & self._mask == 0 and self._can_stop:
            if time.perf_counter() >= self._deadline:
                raise SearchTimeout
            if self._node_limit is not None and self.nodes >= self._node_limit:
                raise SearchTimeout

    def _evaluate(self, board: chess.Board) -> int:
        self.evals += 1
        return self.evaluator.evaluate(board)

    def _init_repetition_keys(self, board: chess.Board) -> None:
        """Seed the repetition stack with game positions inside the reversible window."""
        history: list[tuple[int, ...]] = []
        replay = board.copy()
        for _ in range(min(board.halfmove_clock, len(replay.move_stack))):
            replay.pop()
            history.append(_position_key(replay))
        history.reverse()
        self._keys = history + [_position_key(board)]

    def _is_rule_draw(self, board: chess.Board) -> bool:
        """Draw by fifty-move rule, insufficient material or repetition.

        Inside the tree a single repetition counts as a draw (the standard engine
        convention): if repeating is best for one side, it can repeat again.
        """
        halfmoves = board.halfmove_clock
        if halfmoves >= 100:
            return not board.is_checkmate()  # mate on the 100th half-move still counts
        # Insufficient material is impossible while any pawn, rook or queen is left.
        if not (board.pawns | board.rooks | board.queens) and board.is_insufficient_material():
            return True
        if halfmoves >= 4:
            keys = self._keys
            n = len(keys) - 1
            key = keys[n]
            for i in range(n - 4, max(n - halfmoves, 0) - 1, -2):
                if keys[i] == key:
                    return True
        return False

    @staticmethod
    def _noisy_moves(board: chess.Board) -> list[chess.Move]:
        """Legal captures plus non-capturing queen promotions."""
        moves = list(board.generate_legal_captures())
        us = board.turn
        promoting = board.pawns & board.occupied_co[us] & (chess.BB_RANK_7 if us else chess.BB_RANK_2)
        if promoting:
            targets = (chess.BB_RANK_8 if us else chess.BB_RANK_1) & ~board.occupied
            moves.extend(m for m in board.generate_legal_moves(promoting, targets) if m.promotion == chess.QUEEN)
        return moves

    @staticmethod
    def _check_masks(board: chess.Board) -> tuple[list[int], int, int] | None:
        """Squares from which each piece type would give direct check, plus our
        pieces whose move would uncover a discovered check, and the enemy king square.

        A cheap approximation of ``board.gives_check`` (which pushes the move):
        it ignores castling checks and a slider vacating its own checking line.
        """
        them = not board.turn
        king = board.king(them)
        if king is None:
            return None
        occ = board.occupied
        rook_att = (
            chess.BB_RANK_ATTACKS[king][chess.BB_RANK_MASKS[king] & occ]
            | chess.BB_FILE_ATTACKS[king][chess.BB_FILE_MASKS[king] & occ]
        )
        bishop_att = chess.BB_DIAG_ATTACKS[king][chess.BB_DIAG_MASKS[king] & occ]
        direct = [
            0,
            chess.BB_PAWN_ATTACKS[them][king],
            chess.BB_KNIGHT_ATTACKS[king],
            bishop_att,
            rook_att,
            bishop_att | rook_att,
            0,
        ]
        ours = board.occupied_co[board.turn]
        snipers = ours & (
            (chess.BB_RANK_ATTACKS[king][0] | chess.BB_FILE_ATTACKS[king][0]) & (board.rooks | board.queens)
            | chess.BB_DIAG_ATTACKS[king][0] & (board.bishops | board.queens)
        )
        discovered = 0
        for sniper in chess.scan_reversed(snipers):
            blockers = chess.between(king, sniper) & occ
            if blockers and not blockers & (blockers - 1) and blockers & ours:
                discovered |= blockers
        return direct, discovered, king

    def _order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        pv_move: chess.Move | None,
        checks: bool = True,
    ) -> list[chess.Move]:
        """Sort moves: PV move, captures (MVV-LVA), queen promotions, checks, quiet."""
        if len(moves) < 2:
            return moves
        piece_type_at = board.piece_type_at
        them = board.occupied_co[not board.turn]
        ep_square = board.ep_square
        check_info = self._check_masks(board) if checks else None
        keyed: list[tuple[int, int, chess.Move]] = []
        for index, move in enumerate(moves):
            if move == pv_move:
                key = _KEY_PV
            else:
                to_sq = move.to_square
                attacker = piece_type_at(move.from_square) or 0
                promotion = move.promotion
                if _BB_SQUARES[to_sq] & them:
                    # MVV-LVA: piece types are ordered by value (PAWN=1 ... KING=6).
                    key = _KEY_CAPTURE + 10 * (piece_type_at(to_sq) or 0) - attacker
                    if promotion and promotion != chess.QUEEN:
                        key -= 1000
                elif attacker == chess.PAWN and to_sq == ep_square:
                    key = _KEY_CAPTURE + 10 * chess.PAWN - chess.PAWN  # en passant
                elif promotion:
                    key = _KEY_PROMOTION if promotion == chess.QUEEN else _KEY_UNDERPROMOTION
                elif check_info is not None:
                    direct, discovered, king = check_info
                    gives_check = bool(_BB_SQUARES[to_sq] & direct[attacker]) or (
                        bool(_BB_SQUARES[move.from_square] & discovered)
                        and not chess.ray(king, move.from_square) & _BB_SQUARES[to_sq]
                    )
                    key = _KEY_CHECK if gives_check else 0
                else:
                    key = 0
            keyed.append((-key, index, move))
        keyed.sort()
        return [move for _, _, move in keyed]
