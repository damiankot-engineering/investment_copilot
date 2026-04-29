"""Tests for ``investment_copilot.config``."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from investment_copilot.config import AppConfig, ConfigError, load_config


def _write(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_load_minimal_config_with_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-test-123")
    cfg_path = _write(
        tmp_path / "config.yaml",
        """
        llm:
          api_key: ${GROQ_API_KEY}
        """,
    )

    cfg = load_config(cfg_path, env_file=None)

    assert isinstance(cfg, AppConfig)
    assert cfg.llm.api_key == "sk-test-123"
    # Defaults applied
    assert cfg.providers.market_data == "stooq"
    assert cfg.backtest.benchmark == "wig20"
    assert cfg.llm.language == "pl"


def test_env_var_default_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg_path = _write(
        tmp_path / "config.yaml",
        """
        llm:
          api_key: ${GROQ_API_KEY:-fallback-key}
        """,
    )

    cfg = load_config(cfg_path, env_file=None)
    assert cfg.llm.api_key == "fallback-key"


def test_missing_required_env_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg_path = _write(
        tmp_path / "config.yaml",
        """
        llm:
          api_key: ${GROQ_API_KEY}
        """,
    )

    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        load_config(cfg_path, env_file=None)


def test_unknown_top_level_key_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg_path = _write(
        tmp_path / "config.yaml",
        """
        llm:
          api_key: ${GROQ_API_KEY}
        nonsense_section:
          foo: bar
        """,
    )

    with pytest.raises(ConfigError):
        load_config(cfg_path, env_file=None)


def test_ma_crossover_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg_path = _write(
        tmp_path / "config.yaml",
        """
        llm:
          api_key: ${GROQ_API_KEY}
        strategies:
          ma_crossover:
            fast: 200
            slow: 50
        """,
    )

    with pytest.raises(ConfigError, match="slow window"):
        load_config(cfg_path, env_file=None)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does-not-exist.yaml", env_file=None)


def test_example_config_loads(monkeypatch):
    """The shipped example config must validate."""
    monkeypatch.setenv("GROQ_API_KEY", "sk-example")
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "config.example.yaml", env_file=None)
    assert cfg.llm.model_analysis == "llama-3.3-70b-versatile"
    assert cfg.providers.news == ["stooq", "rss"]
