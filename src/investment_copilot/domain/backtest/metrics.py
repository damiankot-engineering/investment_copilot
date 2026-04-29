"""Performance metrics. Pure functions over pandas series.

Conventions
-----------
* ``equity`` is a date-indexed :class:`pandas.Series` of portfolio value.
* ``returns`` is a date-indexed :class:`pandas.Series` of daily returns
  (already chained, i.e. ``r = equity.pct_change()`` if computing from equity).
* All annualizations use ``trading_days_per_year`` (default 252).
* Standard deviations use ``ddof=0`` (population) to match common quant
  practice for backtests; with hundreds of observations the difference is
  negligible.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from investment_copilot.domain.backtest.results import StrategyMetrics


# --- Atomic metrics ---------------------------------------------------------


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    first = float(equity.iloc[0])
    if first == 0:
        return 0.0
    return float(equity.iloc[-1]) / first - 1.0


def annualized_return(
    equity: pd.Series,
    *,
    trading_days_per_year: int = 252,
) -> float:
    n = len(equity) - 1
    if n <= 0:
        return 0.0
    first = float(equity.iloc[0])
    last = float(equity.iloc[-1])
    if first <= 0 or last <= 0:
        return 0.0
    years = n / trading_days_per_year
    if years <= 0:
        return 0.0
    return (last / first) ** (1.0 / years) - 1.0


def annualized_volatility(
    returns: pd.Series,
    *,
    trading_days_per_year: int = 252,
) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=0) * math.sqrt(trading_days_per_year))


def sharpe_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> float:
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / trading_days_per_year
    excess = returns - daily_rf
    sd = float(excess.std(ddof=0))
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * math.sqrt(trading_days_per_year))


def max_drawdown(equity: pd.Series) -> tuple[float, int | None]:
    """Return ``(max_drawdown, duration_days)``.

    * ``max_drawdown`` is a non-positive float (``-0.25`` means -25%).
    * ``duration_days`` is the number of calendar days from the prior peak
      to the trough that produced the max drawdown. ``None`` if the series
      is empty or never recovers.
    """
    if equity.empty:
        return 0.0, None
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    min_val = float(dd.min())
    if not np.isfinite(min_val):
        return 0.0, None
    trough_idx = dd.idxmin()
    peak_idx = equity.loc[:trough_idx].idxmax()
    try:
        duration = int((trough_idx - peak_idx).days)
    except (AttributeError, TypeError):
        duration = None
    return min_val, duration


def win_rate(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum()) / float(len(returns))


# --- Aggregate -------------------------------------------------------------


def compute_metrics(
    equity: pd.Series,
    *,
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> StrategyMetrics:
    """Compute all metrics from a single equity curve."""
    returns = equity.pct_change().dropna()
    dd, dd_days = max_drawdown(equity)
    return StrategyMetrics(
        total_return=total_return(equity),
        annualized_return=annualized_return(
            equity, trading_days_per_year=trading_days_per_year
        ),
        annualized_volatility=annualized_volatility(
            returns, trading_days_per_year=trading_days_per_year
        ),
        sharpe_ratio=sharpe_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            trading_days_per_year=trading_days_per_year,
        ),
        max_drawdown=dd,
        max_drawdown_duration_days=dd_days,
        win_rate=win_rate(returns),
        n_observations=int(len(equity)),
    )
