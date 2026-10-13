// Play against the engine: the board, move input and calls to the /api/move function.
// chess.js handles the rules in the browser; the server only computes engine moves.
import { Chess } from "https://cdn.jsdelivr.net/npm/chess.js@1.4.0/+esm";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const FILES = "abcdefgh";
const PIECE_NAMES = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };
const ENGINE_NAMES = { nn: "Neural-network engine", material: "Material-count engine" };

const $ = (id) => document.getElementById(id);
const els = {
  board: $("board"),
  promotion: $("promotion"),
  status: $("status"),
  retry: $("retry"),
  evalFill: $("eval-fill"),
  evalText: $("eval-text"),
  depth: $("stat-depth"),
  nps: $("stat-nps"),
  nodes: $("stat-nodes"),
  pv: $("pv"),
  engine: $("engine"),
  think: $("think"),
  undo: $("undo"),
  flip: $("flip"),
  newWhite: $("new-white"),
  newBlack: $("new-black"),
  moves: $("moves"),
  movesEmpty: $("moves-empty"),
  playerTop: $("player-top"),
  playerBottom: $("player-bottom"),
};

const state = {
  game: new Chess(),
  startFen: START_FEN,
  human: "w",
  orientation: "w",
  selected: null, // square of the piece the player picked up
  targets: new Map(), // destination square -> legal moves (several when promoting)
  thinking: false,
  slow: false, // the engine is taking much longer than its think time (cold start)
  requestId: 0, // bumped whenever a pending engine reply should be ignored
  info: null, // last engine reply from the server
  infoEngineColor: null,
  error: null,
  drag: null,
  pendingPromotion: null,
};

// ------------------------------------------------------------------ board

function squaresInViewOrder() {
  const ranks = state.orientation === "w" ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const files = state.orientation === "w" ? [...FILES] : [...FILES].reverse();
  return ranks.flatMap((rank) => files.map((file) => file + rank));
}

function buildBoard() {
  els.board.replaceChildren();
  squaresInViewOrder().forEach((square, index) => {
    const div = document.createElement("div");
    const fileIndex = FILES.indexOf(square[0]);
    const rank = Number(square[1]);
    div.className = "sq " + ((fileIndex + rank) % 2 === 1 ? "dark" : "light"); // a1 is dark
    div.dataset.square = square;
    div.setAttribute("role", "gridcell");
    if (index % 8 === 0) div.append(coordinate("rank", square[1]));
    if (index >= 56) div.append(coordinate("file", square[0]));
    els.board.append(div);
  });
}

function coordinate(kind, text) {
  const span = document.createElement("span");
  span.className = `coord ${kind}`;
  span.textContent = text;
  return span;
}

function kingSquare(color) {
  for (const row of state.game.board()) {
    for (const cell of row) {
      if (cell && cell.type === "k" && cell.color === color) return cell.square;
    }
  }
  return null;
}

function humanCanMove() {
  return (
    !state.thinking &&
    !state.pendingPromotion &&
    !state.game.isGameOver() &&
    state.game.turn() === state.human
  );
}

function renderBoard() {
  const { game } = state;
  const history = game.history({ verbose: true });
  const last = history[history.length - 1];
  const checked = game.inCheck() ? kingSquare(game.turn()) : null;
  const canMove = humanCanMove();

  for (const div of els.board.children) {
    const square = div.dataset.square;
    const piece = game.get(square);
    let img = div.querySelector("img.piece");
    if (piece) {
      const src = `pieces/${piece.color}${piece.type.toUpperCase()}.svg`;
      if (!img) {
        img = document.createElement("img");
        img.className = "piece";
        img.draggable = false;
        div.append(img);
      }
      if (img.getAttribute("src") !== src) img.setAttribute("src", src);
      img.alt = `${piece.color === "w" ? "White" : "Black"} ${PIECE_NAMES[piece.type]}`;
      img.classList.toggle("dragging", Boolean(state.drag?.moved && state.drag.from === square));
    } else if (img) {
      img.remove();
    }
    const isTarget = state.targets.has(square);
    div.classList.toggle("last", Boolean(last) && (square === last.from || square === last.to));
    div.classList.toggle("selected", square === state.selected);
    div.classList.toggle("target", isTarget && !piece);
    div.classList.toggle("capture", isTarget && Boolean(piece));
    div.classList.toggle("check", square === checked);
    div.classList.toggle("movable", canMove && Boolean(piece) && piece.color === state.human);
    div.setAttribute("aria-label", piece ? `${square}, ${img.alt}` : square);
  }
}

// ------------------------------------------------------------------ input

function select(square) {
  state.selected = square;
  state.targets = new Map();
  for (const move of state.game.moves({ square, verbose: true })) {
    const list = state.targets.get(move.to) || [];
    list.push(move);
    state.targets.set(move.to, list);
  }
}

function clearSelection() {
  state.selected = null;
  state.targets = new Map();
}

function squareAt(x, y) {
  const el = document.elementFromPoint(x, y)?.closest(".sq");
  return el && els.board.contains(el) ? el.dataset.square : null;
}

function onPointerDown(event) {
  if (event.button !== 0 || !humanCanMove()) return;
  const square = event.target.closest(".sq")?.dataset.square;
  if (!square) return;
  event.preventDefault();

  // A click on a highlighted square completes a click-to-move.
  if (state.selected && state.targets.has(square)) {
    tryMove(state.selected, square);
    return;
  }
  const piece = state.game.get(square);
  if (!piece || piece.color !== state.human) {
    clearSelection();
    render();
    return;
  }
  const wasSelected = state.selected === square;
  select(square);
  state.drag = {
    from: square,
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    moved: false,
    wasSelected,
    ghost: null,
    size: 0,
  };
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  window.addEventListener("pointercancel", onPointerCancel);
  render();
}

function onPointerMove(event) {
  const drag = state.drag;
  if (!drag || event.pointerId !== drag.pointerId) return;
  if (!drag.moved) {
    if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 4) return;
    const img = els.board.querySelector(`[data-square="${drag.from}"] img.piece`);
    if (!img) return;
    drag.moved = true;
    drag.size = img.getBoundingClientRect().width;
    drag.ghost = img.cloneNode();
    drag.ghost.className = "ghost";
    drag.ghost.style.width = drag.ghost.style.height = `${drag.size}px`;
    document.body.append(drag.ghost);
    img.classList.add("dragging");
  }
  drag.ghost.style.transform = `translate(${event.clientX - drag.size / 2}px, ${event.clientY - drag.size / 2}px)`;
  const over = squareAt(event.clientX, event.clientY);
  for (const div of els.board.children) {
    div.classList.toggle("hover", div.dataset.square === over && state.targets.has(over));
  }
}

function endDrag() {
  window.removeEventListener("pointermove", onPointerMove);
  window.removeEventListener("pointerup", onPointerUp);
  window.removeEventListener("pointercancel", onPointerCancel);
  state.drag?.ghost?.remove();
  for (const div of els.board.children) div.classList.remove("hover");
  state.drag = null;
}

function onPointerUp(event) {
  const drag = state.drag;
  if (!drag || event.pointerId !== drag.pointerId) return;
  endDrag();
  if (!drag.moved) {
    // A plain click: keep the piece selected for click-to-move, or deselect it if it already was.
    if (drag.wasSelected) clearSelection();
    render();
    return;
  }
  const to = squareAt(event.clientX, event.clientY);
  if (to && state.targets.has(to)) tryMove(drag.from, to);
  else render(); // dropped somewhere illegal: the piece returns to its square
}

function onPointerCancel() {
  endDrag();
  render();
}

function tryMove(from, to) {
  const options = state.targets.get(to) || [];
  if (options.length === 0) return;
  if (options.length > 1) {
    showPromotion(from, to); // four options: the same pawn move with each promotion piece
    return;
  }
  commitMove({ from, to, promotion: options[0].promotion });
}

function commitMove(move) {
  try {
    state.game.move(move);
  } catch {
    return; // not legal (should not happen: moves come from the legal-move list)
  }
  clearSelection();
  state.error = null;
  render();
  if (!state.game.isGameOver()) requestEngineMove();
}

function showPromotion(from, to) {
  state.pendingPromotion = { from, to };
  const box = document.createElement("div");
  box.className = "promo-box";
  const label = document.createElement("span");
  label.textContent = "Promote to";
  box.append(label);
  for (const piece of ["q", "r", "b", "n"]) {
    const button = document.createElement("button");
    button.className = "promo-choice";
    button.setAttribute("aria-label", PIECE_NAMES[piece]);
    const img = document.createElement("img");
    img.src = `pieces/${state.human}${piece.toUpperCase()}.svg`;
    img.alt = "";
    button.append(img);
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const pending = state.pendingPromotion;
      hidePromotion();
      commitMove({ ...pending, promotion: piece });
    });
    box.append(button);
  }
  els.promotion.replaceChildren(box);
  els.promotion.hidden = false;
  render();
}

function hidePromotion() {
  state.pendingPromotion = null;
  els.promotion.hidden = true;
  els.promotion.replaceChildren();
}

// ------------------------------------------------------------------ engine

function thinkTime() {
  return Number(els.think.value) || 1;
}

async function requestEngineMove() {
  const id = ++state.requestId;
  state.thinking = true;
  state.slow = false;
  state.error = null;
  render();
  const slowTimer = setTimeout(() => {
    if (id === state.requestId && state.thinking) {
      state.slow = true;
      renderStatus();
    }
  }, (thinkTime() + 3) * 1000);

  try {
    const body = {
      fen: state.startFen,
      moves: state.game.history({ verbose: true }).map((m) => m.from + m.to + (m.promotion || "")),
      engine: els.engine.value,
      time: thinkTime(),
    };
    const response = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (id !== state.requestId) return; // the game changed while the engine was thinking
    if (!response.ok) throw new Error(data.error || `server error ${response.status}`);
    if (data.move) {
      const engineColor = state.game.turn();
      state.game.move({ from: data.move.slice(0, 2), to: data.move.slice(2, 4), promotion: data.move[4] });
      state.info = data;
      state.infoEngineColor = engineColor;
    }
  } catch (error) {
    if (id !== state.requestId) return;
    state.error = error instanceof TypeError ? "the engine could not be reached" : error.message;
  } finally {
    clearTimeout(slowTimer);
    if (id === state.requestId) {
      state.thinking = false;
      state.slow = false;
      render();
    }
  }
}

function cancelThinking() {
  state.requestId++;
  state.thinking = false;
  state.slow = false;
}

// ------------------------------------------------------------------ panel

function gameOverText() {
  const { game } = state;
  if (game.isCheckmate()) return game.turn() === state.human ? "Checkmate. The engine wins." : "Checkmate. You win!";
  if (game.isStalemate()) return "Draw by stalemate.";
  if (game.isInsufficientMaterial()) return "Draw: insufficient material.";
  if (game.isThreefoldRepetition()) return "Draw by threefold repetition.";
  if (game.isDrawByFiftyMoves()) return "Draw by the fifty-move rule.";
  return "Game over.";
}

function renderStatus() {
  const { game } = state;
  els.status.className = "status";
  els.retry.hidden = true;
  let spinner = false;
  let text;
  if (game.isGameOver()) {
    text = gameOverText();
    els.status.classList.add("over");
  } else if (state.error) {
    text = `Couldn't get the engine's move: ${state.error}.`;
    els.status.classList.add("error");
    els.retry.hidden = false;
  } else if (state.thinking) {
    spinner = true;
    text = state.slow ? "Waking the engine up. The first move after a break can take a few seconds…" : "Engine is thinking…";
  } else if (game.turn() === state.human) {
    text = game.inCheck() ? "Your move. You're in check!" : "Your move";
  } else {
    text = "Engine to move";
  }
  els.status.replaceChildren();
  if (spinner) {
    const s = document.createElement("span");
    s.className = "spinner";
    s.setAttribute("aria-hidden", "true");
    els.status.append(s);
  }
  els.status.append(text);
}

function renderInfo() {
  const info = state.info;
  if (!info) {
    els.evalFill.style.width = "50%";
    els.evalText.textContent = "–";
    els.depth.textContent = els.nps.textContent = els.nodes.textContent = els.pv.textContent = "–";
    return;
  }
  let text;
  let whiteShare;
  if (info.mate_in !== null && info.mate_in !== undefined) {
    const engineMates = info.mate_in > 0;
    const whiteMates = engineMates === (state.infoEngineColor === "w");
    text = `${whiteMates ? "" : "-"}M${Math.abs(info.mate_in)}`;
    whiteShare = whiteMates ? 1 : 0;
  } else {
    const cp = info.score_white;
    text = `${cp > 0 ? "+" : ""}${(cp / 100).toFixed(2)}`;
    whiteShare = 1 / (1 + Math.pow(10, -cp / 400)); // White's expected score
  }
  els.evalFill.style.width = `${(whiteShare * 100).toFixed(1)}%`;
  els.evalText.textContent = text;
  els.depth.textContent = String(info.depth);
  els.nps.textContent = Number(info.nps).toLocaleString();
  els.nodes.textContent = Number(info.nodes).toLocaleString();
  els.pv.textContent = info.pv || "–";
}

function renderMoves() {
  const sans = state.game.history();
  els.moves.replaceChildren();
  for (let i = 0; i < sans.length; i += 2) {
    const li = document.createElement("li");
    for (const san of sans.slice(i, i + 2)) {
      const span = document.createElement("span");
      span.textContent = san;
      li.append(span);
    }
    els.moves.append(li);
  }
  const spans = els.moves.querySelectorAll("span");
  spans[spans.length - 1]?.classList.add("current");
  els.moves.scrollTop = els.moves.scrollHeight;
  els.movesEmpty.hidden = sans.length > 0;
}

function playerLabel(color) {
  const div = document.createElement("div");
  div.style.display = "contents";
  const chip = document.createElement("span");
  chip.className = `chip ${color}`;
  const name = document.createElement("span");
  name.textContent = color === state.human ? "You" : ENGINE_NAMES[els.engine.value];
  const sub = document.createElement("span");
  sub.className = "sub";
  sub.textContent = color === "w" ? "White" : "Black";
  div.append(chip, name, sub);
  return div;
}

function renderPlayers() {
  const bottom = state.orientation;
  const top = bottom === "w" ? "b" : "w";
  els.playerTop.replaceChildren(playerLabel(top));
  els.playerBottom.replaceChildren(playerLabel(bottom));
}

function render() {
  renderBoard();
  renderStatus();
  renderInfo();
  renderMoves();
  renderPlayers();
  const humanMoves = state.game.history({ verbose: true }).some((m) => m.color === state.human);
  els.undo.disabled = !humanMoves || Boolean(state.pendingPromotion);
}

// ------------------------------------------------------------------ actions

function startPosition() {
  // A link like  #fen=8/P7/8/8/8/8/k7/4K3_w_-_-_0_1  starts from that position ("_" for spaces).
  const match = location.hash.match(/fen=([^&]+)/);
  if (!match) return START_FEN;
  const fen = decodeURIComponent(match[1]).replace(/_/g, " ");
  try {
    new Chess(fen); // throws on an invalid FEN
    return fen;
  } catch {
    return START_FEN;
  }
}

function newGame(color) {
  cancelThinking();
  hidePromotion();
  state.startFen = startPosition();
  state.game.load(state.startFen);
  state.human = color;
  state.orientation = color;
  state.info = null;
  state.error = null;
  clearSelection();
  buildBoard();
  render();
  if (state.game.turn() !== state.human && !state.game.isGameOver()) requestEngineMove();
}

function undo() {
  // Take back the player's last move (and the engine's reply to it, if any).
  const history = state.game.history({ verbose: true });
  let lastHuman = -1;
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].color === state.human) {
      lastHuman = i;
      break;
    }
  }
  if (lastHuman < 0) return;
  cancelThinking();
  hidePromotion();
  while (state.game.history().length > lastHuman) state.game.undo();
  clearSelection();
  state.info = null;
  state.error = null;
  render();
}

function flip() {
  state.orientation = state.orientation === "w" ? "b" : "w";
  buildBoard();
  render();
}

els.board.addEventListener("pointerdown", onPointerDown);
els.promotion.addEventListener("click", (event) => {
  if (event.target === els.promotion) {
    hidePromotion(); // clicking outside the choices cancels the promotion
    clearSelection();
    render();
  }
});
els.newWhite.addEventListener("click", () => newGame("w"));
els.newBlack.addEventListener("click", () => newGame("b"));
els.undo.addEventListener("click", undo);
els.flip.addEventListener("click", flip);
els.retry.addEventListener("click", () => requestEngineMove());
els.engine.addEventListener("change", renderPlayers);

newGame(startPosition().split(" ")[1] === "b" ? "b" : "w");
