"""Tests for performance metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from investment_copilot.domain.backtest.metrics import (
    annualized_return,
    annualized_volatility,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)


def _equity(values: list[float], start: str = "2024-01-02") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"))


# --- total_return ----------------------------------------------------------


def test_total_return_simple() -> None:
    eq = _equity([100, 110, 121])
    assert total_return(eq) == pytest.approx(0.21)


def test_total_return_loss() -> None:
    eq = _equity([100, 50])
    assert total_return(eq) == pytest.approx(-0.5)


def test_total_return_short_series() -> None:
    assert total_return(pd.Series(dtype=float)) == 0.0
    assert total_return(_equity([100])) == 0.0


# --- annualized_return -----------------------------------------------------


def test_annualized_return_one_year_doubles() -> None:
    # 252 daily steps = 1 year; doubled
    eq = _equity([100.0] + [200.0] * 252)
    assert annualized_return(eq, trading_days_per_year=252) == pytest.approx(1.0, rel=1e-3)


def test_annualized_return_two_years_4x() -> None:
    eq = _equity([100.0] + [400.0] * (252 * 2))
    # (4)^(1/2) - 1 = 1.0
    assert annualized_return(eq, trading_days_per_year=252) == pytest.approx(1.0, rel=1e-3)


# --- volatility / sharpe ---------------------------------------------------


def test_annualized_volatility_constant_returns_zero() -> None:
    eq = _equity(np.linspace(100, 110, 50).tolist())
    rets = eq.pct_change().dropna()
    # Linear equity has slightly varying returns; use truly constant returns:
    rets_const = pd.Series([0.001] * 100)
    assert annualized_volatility(rets_const) == pytest.approx(0.0, abs=1e-12)
    assert annualized_volatility(rets) > 0


def test_sharpe_zero_for_zero_returns() -> None:
    rets = pd.Series([0.0] * 50)
    assert sharpe_ratio(rets) == 0.0


def test_sharpe_positive_for_positive_drift() -> None:
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert sharpe_ratio(rets) > 0


# --- max_drawdown ----------------------------------------------------------


def test_max_drawdown_known_case() -> None:
    eq = _equity([100, 120, 90, 80, 110])  # peak 120 -> trough 80 = -33.33%
    dd, dur = max_drawdown(eq)
    assert dd == pytest.approx(-1 / 3, abs=1e-9)
    # Peak at index 1, trough at index 3 -> 2 business days; expressed in calendar days
    assert dur is not None and dur >= 1


def test_max_drawdown_monotonic_zero() -> None:
    eq = _equity([100, 110, 120, 130])
    dd, dur = max_drawdown(eq)
    assert dd == 0.0


# --- win_rate --------------------------------------------------------------


def test_win_rate() -> None:
    rets = pd.Series([0.01, -0.01, 0.02, 0.0, -0.005])
    assert win_rate(rets) == pytest.approx(2 / 5)
    assert win_rate(pd.Series(dtype=float)) == 0.0


# --- compute_metrics aggregate ---------------------------------------------


def test_compute_metrics_returns_strategy_metrics_object() -> None:
    eq = _equity([100, 105, 102, 108, 110])
    m = compute_metrics(eq)
    assert m.n_observations == 5
    assert m.total_return == pytest.approx(0.10)
    assert m.max_drawdown <= 0
    assert math.isfinite(m.sharpe_ratio)
