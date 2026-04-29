"""Tests for the portfolio-level simulator."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from investment_copilot.domain.backtest.engine import (
    BacktestError,
    benchmark_buy_and_hold,
    simulate_portfolio,
)


def _ohlcv_from_close(close: np.ndarray, start: str = "2024-01-02") -> pd.DataFrame:
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


# --- Engine ----------------------------------------------------------------


def test_always_long_single_ticker_replicates_buy_and_hold() -> None:
    close = np.linspace(100, 200, 50)
    panel = {"x.pl": _ohlcv_from_close(close)}
    sigs = {"x.pl": pd.Series(1, index=panel["x.pl"].index, dtype=int)}

    run = simulate_portfolio(
        panel, sigs, initial_capital=10_000.0, start=date(2024, 1, 2)
    )
    # Final equity = 10_000 * (200/100) = 20_000
    assert float(run.equity_curve.iloc[-1]) == pytest.approx(20_000.0, rel=1e-9)
    # First point exactly equals initial capital
    assert float(run.equity_curve.iloc[0]) == pytest.approx(10_000.0)


def test_always_flat_keeps_equity_constant() -> None:
    close = np.linspace(100, 200, 50)
    panel = {"x.pl": _ohlcv_from_close(close)}
    sigs = {"x.pl": pd.Series(0, index=panel["x.pl"].index, dtype=int)}

    run = simulate_portfolio(
        panel, sigs, initial_capital=10_000.0, start=date(2024, 1, 2)
    )
    assert (run.equity_curve == 10_000.0).all()


def test_equal_weight_two_tickers_both_long() -> None:
    # Two tickers, both equally long the whole time. Expected portfolio
    # daily return = mean of the two daily returns (50/50 weight).
    n = 30
    a = np.linspace(100, 130, n)
    b = np.linspace(50, 75, n)
    panel = {
        "a.pl": _ohlcv_from_close(a),
        "b.pl": _ohlcv_from_close(b),
    }
    idx = panel["a.pl"].index
    sigs = {
        "a.pl": pd.Series(1, index=idx, dtype=int),
        "b.pl": pd.Series(1, index=idx, dtype=int),
    }
    run = simulate_portfolio(
        panel, sigs, initial_capital=10_000.0, start=date(2024, 1, 2)
    )

    # Hand-compute expected equity: 10000 * cumprod(1 + 0.5*ra + 0.5*rb)
    ra = pd.Series(a).pct_change().fillna(0.0).values
    rb = pd.Series(b).pct_change().fillna(0.0).values
    expected = 10_000.0 * np.cumprod(1 + 0.5 * ra + 0.5 * rb)
    expected[0] = 10_000.0
    np.testing.assert_allclose(run.equity_curve.values, expected, rtol=1e-9)


def test_weights_renormalize_when_one_ticker_flat() -> None:
    # A is flat the whole time, B is long. Portfolio should track B exactly.
    n = 20
    a = np.full(n, 100.0)  # flat price too
    b = np.linspace(100, 200, n)
    panel = {"a.pl": _ohlcv_from_close(a), "b.pl": _ohlcv_from_close(b)}
    idx = panel["a.pl"].index
    sigs = {
        "a.pl": pd.Series(0, index=idx, dtype=int),
        "b.pl": pd.Series(1, index=idx, dtype=int),
    }
    run = simulate_portfolio(panel, sigs, initial_capital=10_000.0, start=date(2024, 1, 2))
    assert float(run.equity_curve.iloc[-1]) == pytest.approx(20_000.0, rel=1e-9)


def test_engine_raises_when_no_data() -> None:
    with pytest.raises(BacktestError):
        simulate_portfolio({}, {}, initial_capital=10_000.0, start=date(2024, 1, 2))


def test_engine_raises_when_window_outside_data() -> None:
    panel = {"x.pl": _ohlcv_from_close(np.linspace(100, 110, 20))}
    sigs = {"x.pl": pd.Series(1, index=panel["x.pl"].index, dtype=int)}
    with pytest.raises(BacktestError, match="No price data"):
        simulate_portfolio(
            panel, sigs, initial_capital=10_000.0, start=date(2030, 1, 1)
        )


def test_engine_rejects_non_positive_capital() -> None:
    panel = {"x.pl": _ohlcv_from_close(np.linspace(100, 110, 5))}
    sigs = {"x.pl": pd.Series(1, index=panel["x.pl"].index, dtype=int)}
    with pytest.raises(BacktestError):
        simulate_portfolio(panel, sigs, initial_capital=0, start=date(2024, 1, 2))


# --- Benchmark -------------------------------------------------------------


def test_benchmark_buy_and_hold() -> None:
    close = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.date_range("2024-01-02", periods=3, freq="B"),
    )
    eq = benchmark_buy_and_hold(
        close, initial_capital=10_000.0, start=date(2024, 1, 2)
    )
    assert float(eq.iloc[0]) == pytest.approx(10_000.0)
    assert float(eq.iloc[-1]) == pytest.approx(12_100.0)


def test_benchmark_empty_raises() -> None:
    with pytest.raises(BacktestError):
        benchmark_buy_and_hold(
            pd.Series(dtype=float), initial_capital=10_000.0, start=date(2024, 1, 2)
        )
