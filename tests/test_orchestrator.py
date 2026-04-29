"""Tests for the Orchestrator pipelines."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from investment_copilot.config.schema import (
    AppConfig,
    BacktestConfig,
    LLMConfig,
    StrategiesConfig,
)
from investment_copilot.domain.backtest import BacktestError
from investment_copilot.domain.portfolio import Holding, Portfolio
from investment_copilot.domain.prompts import (
    HoldingComment,
    PortfolioAnalysis,
    RiskAlert,
    RiskAlerts,
)
from investment_copilot.infrastructure.llm import LLMError
from investment_copilot.infrastructure.storage import ParquetCache, SQLiteStore
from investment_copilot.orchestrator import Orchestrator
from investment_copilot.services.backtest_service import BacktestService
from investment_copilot.services.container import ServiceContainer
from investment_copilot.services.copilot_service import CopilotService
from investment_copilot.services.data_service import DataService
from investment_copilot.services.portfolio_service import PortfolioService
from investment_copilot.services.report_service import ReportService


# --- Stubs -----------------------------------------------------------------


class FakeMarket:
    name = "fake-market"

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.fail_for: set[str] = set()

    def _frame(self, n: int = 30) -> pd.DataFrame:
        idx = pd.date_range("2023-06-01", periods=n, freq="B")
        c = np.linspace(100, 130, n)
        return pd.DataFrame(
            {
                "open": c,
                "high": c * 1.01,
                "low": c * 0.99,
                "close": c,
                "volume": np.full(n, 1000.0),
            },
            index=idx,
        )

    def fetch_ohlcv(self, ticker: str, start: date, end: date | None = None) -> pd.DataFrame:
        self.calls.append(("ohlcv", ticker))
        if ticker in self.fail_for:
            from investment_copilot.infrastructure.providers.base import ProviderError

            raise ProviderError(f"fail {ticker}")
        return self._frame()

    def fetch_benchmark(self, name: str, start: date, end: date | None = None) -> pd.DataFrame:
        self.calls.append(("bench", name))
        return self._frame()


class FakeNewsProvider:
    name = "fake-news"

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def fetch_news(self, since, *, ticker=None, keywords=None):
        self.calls.append({"ticker": ticker, "keywords": keywords})
        return []


class FakeLLM:
    name = "fake-llm"

    def __init__(self) -> None:
        self.return_for_schema: dict[type, Any] = {}
        self.fail_for_schema: set[type] = set()

    def complete_structured(self, *, response_schema, **_kw):
        if response_schema in self.fail_for_schema:
            raise LLMError("LLM down")
        return self.return_for_schema[response_schema]

    def complete_text(self, **_kw):
        return ""


# --- Container builder for tests -------------------------------------------


def _make_container(tmp_path: Path) -> tuple[ServiceContainer, FakeMarket, FakeNewsProvider, FakeLLM]:
    market = FakeMarket()
    news_provider = FakeNewsProvider()
    sqlite = SQLiteStore(tmp_path / "cache.db")
    parquet = ParquetCache(tmp_path / "ohlcv")
    llm = FakeLLM()

    data = DataService(
        market_provider=market,
        news_providers=[news_provider],
        sqlite_store=sqlite,
        parquet_cache=parquet,
    )
    portfolio = PortfolioService(data_service=data)
    bt_cfg = BacktestConfig(
        start_date=date(2023, 6, 1), initial_capital=10_000.0, benchmark="wig20"
    )
    backtest = BacktestService(
        data_service=data,
        backtest_config=bt_cfg,
        strategies_config=StrategiesConfig(),
    )
    copilot = CopilotService(
        llm_client=llm, data_service=data, llm_config=LLMConfig(api_key="x")
    )
    cfg = AppConfig(llm=LLMConfig(api_key="x"), backtest=bt_cfg)

    container = ServiceContainer(
        config=cfg,
        sqlite_store=sqlite,
        parquet_cache=parquet,
        market_provider=market,
        news_providers=[news_provider],
        llm_client=llm,
        data_service=data,
        portfolio_service=portfolio,
        backtest_service=backtest,
        copilot_service=copilot,
    )
    return container, market, news_provider, llm


def _portfolio() -> Portfolio:
    return Portfolio(
        holdings=[
            Holding(
                ticker="PKN",
                shares=10,
                entry_price=100,
                entry_date=date(2023, 1, 2),
                thesis="t",
                keywords=["Orlen"],
            ),
            Holding(
                ticker="CDR",
                shares=5,
                entry_price=100,
                entry_date=date(2023, 1, 2),
                thesis="t",
            ),
        ]
    )


# --- update_data ----------------------------------------------------------


def test_update_data_pipeline_happy_path(tmp_path) -> None:
    container, market, news_provider, _ = _make_container(tmp_path)
    orch = Orchestrator(container, reports_dir=tmp_path / "reports")

    report = orch.update_data(_portfolio())

    # OHLCV refreshed for both holdings + benchmark
    assert set(report.ohlcv_updated) == {"pkn.pl", "cdr.pl"}
    assert report.benchmark_symbol == "^wig20"
    assert report.benchmark_rows > 0
    # News provider was called once per ticker (orchestrator passes keywords map)
    tickers_called = [c["ticker"] for c in news_provider.calls]
    assert set(tickers_called) == {"pkn.pl", "cdr.pl"}


def test_update_data_records_per_ticker_failures(tmp_path) -> None:
    container, market, _, _ = _make_container(tmp_path)
    market.fail_for = {"cdr.pl"}
    orch = Orchestrator(container, reports_dir=tmp_path / "reports")

    report = orch.update_data(_portfolio())

    # PKN succeeded, CDR didn't make it into ohlcv_updated
    assert "pkn.pl" in report.ohlcv_updated
    assert "cdr.pl" not in report.ohlcv_updated


# --- run_analysis ---------------------------------------------------------


def test_run_analysis_pipeline_happy_path(tmp_path) -> None:
    container, _, _, llm = _make_container(tmp_path)
    llm.return_for_schema[PortfolioAnalysis] = PortfolioAnalysis(
        summary="ok",
        holdings_comments=[
            HoldingComment(ticker="pkn.pl", comment="ok", recommendation="trzymaj"),
        ],
        diversification_notes="ok",
        confidence=7,
    )
    llm.return_for_schema[RiskAlerts] = RiskAlerts(overview="ok", alerts=[])

    bundle = Orchestrator(container, reports_dir=tmp_path).run_analysis(_portfolio())

    assert bundle.analysis is not None
    assert bundle.risks is not None
    assert bundle.warnings == []


def test_run_analysis_captures_llm_failure_in_warnings(tmp_path) -> None:
    container, _, _, llm = _make_container(tmp_path)
    llm.fail_for_schema = {PortfolioAnalysis, RiskAlerts}
    bundle = Orchestrator(container, reports_dir=tmp_path).run_analysis(_portfolio())
    assert bundle.analysis is None
    assert bundle.risks is None
    assert len(bundle.warnings) == 2


def test_run_analysis_skips_risks_when_disabled(tmp_path) -> None:
    container, _, _, llm = _make_container(tmp_path)
    llm.return_for_schema[PortfolioAnalysis] = PortfolioAnalysis(
        summary="ok",
        holdings_comments=[
            HoldingComment(ticker="pkn.pl", comment="ok", recommendation="trzymaj"),
        ],
        diversification_notes="ok",
        confidence=7,
    )
    bundle = Orchestrator(container, reports_dir=tmp_path).run_analysis(
        _portfolio(), include_risks=False
    )
    assert bundle.risks is None


# --- backtest -------------------------------------------------------------


def test_backtest_pipeline_propagates_errors(tmp_path) -> None:
    container, _, _, _ = _make_container(tmp_path)
    # No data ever loaded into cache -> BacktestError must propagate
    with pytest.raises(BacktestError):
        Orchestrator(container, reports_dir=tmp_path).backtest(
            _portfolio(), strategy_name="ma_crossover"
        )


# --- generate_report ------------------------------------------------------


def test_generate_report_full_path_writes_file(tmp_path) -> None:
    container, _, _, llm = _make_container(tmp_path)
    llm.return_for_schema[PortfolioAnalysis] = PortfolioAnalysis(
        summary="ok",
        holdings_comments=[
            HoldingComment(ticker="pkn.pl", comment="ok", recommendation="trzymaj"),
        ],
        diversification_notes="ok",
        confidence=7,
    )
    llm.return_for_schema[RiskAlerts] = RiskAlerts(overview="ok", alerts=[])
    orch = Orchestrator(container, reports_dir=tmp_path / "reports")

    # Refresh data first so backtest has something to work with
    orch.update_data(_portfolio())

    full = orch.generate_report(
        _portfolio(), strategy_name="ma_crossover", filename="run.md"
    )

    assert full.report_path.is_file()
    assert full.report_path.name == "run.md"
    body = full.report_path.read_text(encoding="utf-8")
    assert "# Raport portfela" in body
    assert "## Backtest" in body
    assert "## Analiza (AI)" in body
    assert "## Ryzyka (AI)" in body
    assert full.warnings == []


def test_generate_report_degrades_gracefully_when_llm_fails(tmp_path) -> None:
    container, _, _, llm = _make_container(tmp_path)
    llm.fail_for_schema = {PortfolioAnalysis, RiskAlerts}

    orch = Orchestrator(container, reports_dir=tmp_path / "reports")
    orch.update_data(_portfolio())

    full = orch.generate_report(_portfolio(), strategy_name="ma_crossover")

    assert full.analysis is None
    assert full.risks is None
    assert len(full.warnings) == 2
    body = full.report_path.read_text(encoding="utf-8")
    assert "## Portfel" in body
    assert "## Backtest" in body
    assert "## Analiza (AI)" not in body
    assert "## Ostrzeżenia" in body


def test_generate_report_skips_backtest_when_no_strategy(tmp_path) -> None:
    container, _, _, llm = _make_container(tmp_path)
    llm.return_for_schema[PortfolioAnalysis] = PortfolioAnalysis(
        summary="ok",
        holdings_comments=[
            HoldingComment(ticker="pkn.pl", comment="ok", recommendation="trzymaj"),
        ],
        diversification_notes="ok",
        confidence=7,
    )
    llm.return_for_schema[RiskAlerts] = RiskAlerts(overview="ok", alerts=[])

    orch = Orchestrator(container, reports_dir=tmp_path / "reports")
    full = orch.generate_report(_portfolio())  # no strategy_name
    assert full.backtest is None
    body = full.report_path.read_text(encoding="utf-8")
    assert "## Backtest" not in body


def test_generate_report_records_backtest_failure_as_warning(tmp_path) -> None:
    """No cached data + strategy requested -> backtest skipped, report still written."""
    container, _, _, llm = _make_container(tmp_path)
    llm.return_for_schema[PortfolioAnalysis] = PortfolioAnalysis(
        summary="ok",
        holdings_comments=[
            HoldingComment(ticker="pkn.pl", comment="ok", recommendation="trzymaj"),
        ],
        diversification_notes="ok",
        confidence=7,
    )
    llm.return_for_schema[RiskAlerts] = RiskAlerts(overview="ok", alerts=[])

    orch = Orchestrator(container, reports_dir=tmp_path / "reports")
    full = orch.generate_report(_portfolio(), strategy_name="ma_crossover")

    assert full.backtest is None
    assert any("Backtest skipped" in w for w in full.warnings)
