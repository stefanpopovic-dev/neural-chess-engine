# Python chess engine: material vs. neural-network evaluation

An alpha-beta chess engine built on python-chess move generation, with a
hand-tuned material evaluator as the baseline.

## Layout

```
chess_engine/
  search.py       negamax alpha-beta, iterative deepening, quiescence, move ordering
  evaluators.py   Evaluator interface, MaterialEvaluator
  positions.py    fixed benchmark positions
  reporting.py    Markdown/JSON helpers
scripts/
  depth_benchmark.py
tests/
config.py         defaults for every parameter; overridden by config.toml
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
cp config.example.toml config.toml   # then edit [stockfish].path
pytest
```

## Phase 1: search core and material baseline

```bash
python scripts/depth_benchmark.py --evaluators material --time 1.0
```

Results are written to `results/depth_material.json`.
