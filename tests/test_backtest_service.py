"""Tests for BacktestService — composition with synthetic OHLCV."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from investment_copilot.config.schema import (
    BacktestConfig,
    MACrossoverParams,
    MomentumParams,
    StrategiesConfig,
)
from investment_copilot.domain.backtest.engine import BacktestError
from investment_copilot.domain.portfolio import Holding, Portfolio
from investment_copilot.services.backtest_service import BacktestService


class FakeData:
    """Minimal DataService surface used by BacktestService."""

    def __init__(self, panel: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None = None):
        self._panel = panel
        self._benchmark = benchmark

    def load_ohlcv(self, ticker: str, *, start=None, end=None) -> pd.DataFrame:
        df = self._panel.get(ticker, pd.DataFrame())
        if df.empty:
            return df
        out = df
        if start is not None:
            out = out[out.index >= pd.Timestamp(start)]
        if end is not None:
            out = out[out.index <= pd.Timestamp(end)]
        return out

    def load_benchmark(self, name: str, *, start=None, end=None) -> pd.DataFrame:
        if self._benchmark is None or self._benchmark.empty:
            return pd.DataFrame()
        out = self._benchmark
        if start is not None:
            out = out[out.index >= pd.Timestamp(start)]
        if end is not None:
            out = out[out.index <= pd.Timestamp(end)]
        return out


def _ohlcv_from_close(close: np.ndarray, start: str = "2023-01-02") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(close), freq="B")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        },
        index=idx,
    )


def _holding(ticker: str) -> Holding:
    return Holding(
        ticker=ticker,
        shares=10,
        entry_price=100.0,
        entry_date=date(2023, 1, 2),
        thesis="t",
    )


def _service(data: FakeData) -> BacktestService:
    return BacktestService(
        data_service=data,
        backtest_config=BacktestConfig(
            start_date=date(2023, 6, 1),  # well inside the synthetic data
            initial_capital=10_000.0,
            benchmark="wig20",
        ),
        strategies_config=StrategiesConfig(
            ma_crossover=MACrossoverParams(fast=5, slow=10),
            momentum=MomentumParams(lookback=20, threshold=0.0),
        ),
    )


# --- Smoke run -------------------------------------------------------------


def test_run_ma_crossover_in_uptrend_makes_money() -> None:
    n = 300  # ~1.2 years business days starting 2023-01-02
    panel = {
        "a.pl": _ohlcv_from_close(np.linspace(100, 200, n)),
        "b.pl": _ohlcv_from_close(np.linspace(50, 110, n)),
    }
    bench = _ohlcv_from_close(np.linspace(2000, 2400, n))
    portfolio = Portfolio(holdings=[_holding("a.pl"), _holding("b.pl")])

    svc = _service(FakeData(panel, bench))
    result = svc.run(portfolio, strategy_name="ma_crossover")

    assert result.strategy_name == "ma_crossover"
    assert result.strategy_params == {"fast": 5, "slow": 10}
    assert result.tickers_used == ["a.pl", "b.pl"]
    assert result.missing_tickers == []
    assert result.final_value > result.initial_capital
    assert result.metrics.total_return > 0
    assert result.benchmark_symbol == "^wig20"
    assert result.benchmark_metrics is not None
    assert len(result.equity_curve) > 0
    assert len(result.benchmark_equity_curve) > 0


def test_run_records_missing_tickers() -> None:
    n = 200
    panel = {"a.pl": _ohlcv_from_close(np.linspace(100, 200, n))}
    portfolio = Portfolio(holdings=[_holding("a.pl"), _holding("b.pl")])

    svc = _service(FakeData(panel, None))
    result = svc.run(portfolio, strategy_name="momentum", include_benchmark=False)

    assert result.missing_tickers == ["b.pl"]
    assert result.tickers_used == ["a.pl"]
    assert result.benchmark_symbol is None
    assert result.benchmark_metrics is None


def test_run_raises_when_no_data() -> None:
    portfolio = Portfolio(holdings=[_holding("a.pl")])
    svc = _service(FakeData({}, None))
    with pytest.raises(BacktestError, match="No OHLCV"):
        svc.run(portfolio, strategy_name="ma_crossover")


def test_run_handles_missing_benchmark_gracefully() -> None:
    n = 200
    panel = {"a.pl": _ohlcv_from_close(np.linspace(100, 200, n))}
    portfolio = Portfolio(holdings=[_holding("a.pl")])
    svc = _service(FakeData(panel, None))  # no benchmark
    result = svc.run(portfolio, strategy_name="ma_crossover")
    assert result.benchmark_symbol is None
    assert result.benchmark_metrics is None


def test_result_is_pydantic_serializable() -> None:
    n = 200
    panel = {"a.pl": _ohlcv_from_close(np.linspace(100, 200, n))}
    portfolio = Portfolio(holdings=[_holding("a.pl")])
    svc = _service(FakeData(panel, None))
    result = svc.run(portfolio, strategy_name="ma_crossover")
    # If model_dump_json works without raising, the API surface is JSON-ready.
    payload = result.model_dump_json()
    assert isinstance(payload, str)
    assert "equity_curve" in payload
