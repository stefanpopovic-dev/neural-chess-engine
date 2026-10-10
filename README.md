# Python chess engine: material vs. neural-network evaluation

An alpha-beta chess engine built on python-chess move generation, with two
interchangeable evaluators: a hand-tuned material count (the baseline) and a
small PyTorch network trained on one million Stockfish-labelled positions from
Lichess games.

## Layout

```
chess_engine/
  search.py       negamax alpha-beta, iterative deepening, time control, quiescence, move ordering
  evaluators.py   Evaluator interface, MaterialEvaluator, NNEvaluator, CachedEvaluator
  encoding.py     board -> 781 side-to-move-relative features (and a 98-byte packed form)
  model.py        PyTorch MLP and checkpoint save/load
  zobrist.py      fast Polyglot Zobrist hashing (cache keys)
  positions.py    fixed benchmark positions (4 opening, 4 middlegame, 4 endgame)
  reporting.py    Markdown/JSON helpers
scripts/
  build_dataset.py   Lichess .pgn.zst -> sampled positions -> Stockfish labels -> packed dataset
  train.py           train the network, save the best checkpoint, log and plot losses
  depth_benchmark.py depth / nodes per second / evaluations per move at a fixed time per move
  profile_search.py  where does search time go? (wall-clock timers + cProfile)
data/       raw and processed data (gitignored)
models/     trained checkpoints
results/    metrics and plots
tests/      pytest suite
config.py   every default; overridden by config.toml, then by command-line flags
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows; `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
cp config.example.toml config.toml  # then set [stockfish].path
pytest
```

## Phase 1: search core and material baseline

```bash
python scripts/depth_benchmark.py --evaluators material --time 1.0
```

## Phase 2: dataset and neural-network evaluator

```bash
# ~4 min sampling + ~27 min labelling (7 workers, depth 10) + ~1 min packing on an 8-core CPU
python scripts/build_dataset.py data/raw/lichess_db_standard_rated_2014-01.pgn.zst --target 1000000
python scripts/train.py
```

The database comes from https://database.lichess.org/ (`lichess_db_standard_rated_2014-01.pgn.zst`, 111 MB).
It is streamed through `zstandard` and never decompressed to disk. Every stage
resumes when rerun.

## Phase 3: NN in the search, profiling, evaluation cache

```bash
python scripts/depth_benchmark.py --evaluators material nn nn-cache   # -> results/depth_comparison.md
python scripts/profile_search.py --evaluators material nn nn-cache    # -> results/profile_*.txt
```

The network is about 3x slower per node than the material count. Profiling
shows 65-75% of NN search time is spent inside the evaluator (about 50 us per
network call). The Zobrist-keyed LRU cache, which persists across moves within
a game, lifts nodes per second by ~57% at a 60% hit rate.
