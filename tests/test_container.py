"""Tests for the ServiceContainer factory."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from investment_copilot.config import load_config
from investment_copilot.infrastructure.providers import (
    RSSProvider,
    StooqNewsProvider,
    StooqProvider,
    build_market_provider,
    build_news_providers,
)
from investment_copilot.services import (
    BacktestService,
    DataService,
    PortfolioService,
    ServiceContainer,
    build_container,
)


def _write_config(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# --- Provider builder tests -------------------------------------------------


def test_build_market_provider_default_is_stooq(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = load_config(
        _write_config(tmp_path, "llm:\n  api_key: ${GROQ_API_KEY}\n"),
        env_file=None,
    )
    provider = build_market_provider(cfg.providers)
    assert isinstance(provider, StooqProvider)
    assert provider.name == "stooq"


def test_build_news_providers_default_stooq_and_rss(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = load_config(
        _write_config(tmp_path, "llm:\n  api_key: ${GROQ_API_KEY}\n"),
        env_file=None,
    )
    providers = build_news_providers(cfg.providers)
    types = [type(p) for p in providers]
    assert types == [StooqNewsProvider, RSSProvider]


def test_build_news_providers_skips_unimplemented(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = load_config(
        _write_config(
            tmp_path,
            """
            providers:
              news: [newsapi, stooq]
            llm:
              api_key: ${GROQ_API_KEY}
            """,
        ),
        env_file=None,
    )
    providers = build_news_providers(cfg.providers)
    assert [p.name for p in providers] == ["stooq"]


def test_build_news_providers_rss_skipped_when_feeds_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = load_config(
        _write_config(
            tmp_path,
            """
            providers:
              news: [rss]
              rss_feeds: []
            llm:
              api_key: ${GROQ_API_KEY}
            """,
        ),
        env_file=None,
    )
    providers = build_news_providers(cfg.providers)
    assert providers == []


# --- ServiceContainer tests -------------------------------------------------


def test_build_container_wires_everything(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = load_config(
        _write_config(
            tmp_path,
            f"""
            providers:
              news: [rss]
            storage:
              sqlite_path: {tmp_path / "cache.db"}
              parquet_dir: {tmp_path / "ohlcv"}
            llm:
              api_key: ${{GROQ_API_KEY}}
            """,
        ),
        env_file=None,
    )

    container = build_container(cfg)

    assert isinstance(container, ServiceContainer)
    assert isinstance(container.data_service, DataService)
    assert isinstance(container.portfolio_service, PortfolioService)
    assert isinstance(container.backtest_service, BacktestService)

    # PortfolioService received the same DataService instance
    assert container.portfolio_service._data is container.data_service
    # BacktestService likewise
    assert container.backtest_service._data is container.data_service

    # LLM client is wired
    from investment_copilot.infrastructure.llm import GroqClient

    assert isinstance(container.llm_client, GroqClient)
    assert container.llm_client.name == "groq"

    # Config carried through
    assert container.config is cfg
    assert container.config.storage.sqlite_path == tmp_path / "cache.db"


def test_container_is_immutable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = load_config(
        _write_config(tmp_path, "llm:\n  api_key: ${GROQ_API_KEY}\n"),
        env_file=None,
    )
    container = build_container(cfg)
    with pytest.raises(Exception):
        container.config = cfg  # type: ignore[misc]


def test_build_container_creates_storage_directories(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "x")
    nested = tmp_path / "deep" / "nest"
    cfg = load_config(
        _write_config(
            tmp_path,
            f"""
            storage:
              sqlite_path: {nested / "cache.db"}
              parquet_dir: {nested / "ohlcv"}
            llm:
              api_key: ${{GROQ_API_KEY}}
            """,
        ),
        env_file=None,
    )
    build_container(cfg)
    assert nested.is_dir()
    assert (nested / "ohlcv").is_dir()
