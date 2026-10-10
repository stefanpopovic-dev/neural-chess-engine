"""Configuration loading and the Stockfish path check."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import ConfigError, load_config, stockfish_path


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHESS_ENGINE_CONFIG", raising=False)
    cfg = load_config(write(tmp_path, ""))
    assert cfg.search.time_limit == 1.0
    assert cfg.material.queen == 900


def test_overrides(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, "[search]\ntime_limit = 2\nquiescence = false\n[material]\nqueen = 950\n"))
    assert cfg.search.time_limit == 2.0 and isinstance(cfg.search.time_limit, float)
    assert cfg.search.quiescence is False
    assert cfg.material.queen == 950


def test_unknown_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown setting"):
        load_config(write(tmp_path, "[search]\ntime_limt = 2\n"))


def test_unknown_section_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown config section"):
        load_config(write(tmp_path, "[serch]\ntime_limit = 2\n"))


def test_missing_explicit_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_stockfish_path_not_configured(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, ""))
    with pytest.raises(ConfigError, match="Stockfish path is not configured"):
        stockfish_path(cfg)


def test_stockfish_binary_missing(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, f'[stockfish]\npath = "{(tmp_path / "missing.exe").as_posix()}"\n'))
    with pytest.raises(ConfigError, match="Stockfish binary not found"):
        stockfish_path(cfg)


def test_stockfish_binary_found(tmp_path: Path) -> None:
    fake = tmp_path / "stockfish"
    fake.write_text("")
    cfg = load_config(write(tmp_path, f'[stockfish]\npath = "{fake.as_posix()}"\n'))
    assert stockfish_path(cfg) == fake
