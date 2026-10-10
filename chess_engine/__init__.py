"""Alpha-beta chess engine with material and neural-network evaluators."""

import sys
from pathlib import Path

# Make the project-root ``config`` module importable when the package is used
# from scripts, tests or an interactive session started elsewhere.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
