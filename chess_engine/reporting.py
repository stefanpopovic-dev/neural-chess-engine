"""Small helpers for writing results files and Markdown tables."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured Markdown table with right-aligned numeric columns."""
    cells = [[str(h) for h in headers]] + [[_fmt(v) for v in row] for row in rows]
    widths = [max(3, max(len(r[i]) for r in cells)) for i in range(len(headers))]
    numeric = [
        bool(rows) and all(isinstance(row[i], (int, float)) and not isinstance(row[i], bool) for row in rows)
        for i in range(len(headers))
    ]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(v.rjust(w) if numeric[i] else v.ljust(w) for i, (v, w) in enumerate(zip(values, widths))) + " |"

    sep = "| " + " | ".join(("-" * (w - 1) + ":") if numeric[i] else "-" * w for i, w in enumerate(widths)) + " |"
    return "\n".join([line(cells[0]), sep] + [line(r) for r in cells[1:]])


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    """Write JSON atomically (write to a temp file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
