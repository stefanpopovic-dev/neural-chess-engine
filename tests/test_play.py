"""Smoke test for the interactive play script (scripted stdin)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "play.py"


def run(stdin: str, *args: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--engine", "material", "--time", "0.05", *args],
        input=stdin, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_moves_illegal_input_and_undo() -> None:
    out = run("e4\nxx\nundo\nquit\n")
    assert out.count("Engine plays") == 1
    assert "'xx' is not a legal move" in out
    assert "Game abandoned" in out


def test_engine_delivers_mate_as_white() -> None:
    out = run("quit\n", "--color", "black", "--fen", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    assert "Engine plays Rd8#" in out
    assert "White wins (checkmate)" in out
