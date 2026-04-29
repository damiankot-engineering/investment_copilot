"""Pydantic result models for backtests.

These are the shape returned by :class:`BacktestService.run` and what a
future ``/backtest`` HTTP endpoint will serialize to JSON. They are frozen
to prevent downstream layers from mutating computed results.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class EquityPoint(BaseModel):
    """A single (date, value) point on an equity curve."""

    model_config = ConfigDict(frozen=True)

    date: date
    value: float


class StrategyMetrics(BaseModel):
    """Standard performance metrics."""

    model_config = ConfigDict(frozen=True)

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float                       # negative, e.g. -0.25 for -25%
    max_drawdown_duration_days: int | None
    win_rate: float
    n_observations: int


class BacktestResult(BaseModel):
    """Output of a single backtest run."""

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    strategy_params: dict[str, float | int | str]
    start_date: date
    end_date: date
    initial_capital: float
    final_value: float
    equity_curve: list[EquityPoint]
    metrics: StrategyMetrics

    benchmark_symbol: str | None = None
    benchmark_equity_curve: list[EquityPoint] = Field(default_factory=list)
    benchmark_metrics: StrategyMetrics | None = None

    missing_tickers: list[str] = Field(default_factory=list)
    tickers_used: list[str] = Field(default_factory=list)

    generated_at: datetime


# --- Conversion helpers -----------------------------------------------------


def equity_points(series: pd.Series) -> list[EquityPoint]:
    """Convert a date-indexed equity series to a list of :class:`EquityPoint`."""
    return [
        EquityPoint(date=_to_date(idx), value=float(val))
        for idx, val in series.items()
    ]


def equity_series(points: Iterable[EquityPoint]) -> pd.Series:
    """Inverse of :func:`equity_points` — useful in reports."""
    points = list(points)
    if not points:
        return pd.Series(dtype=float)
    s = pd.Series(
        [p.value for p in points],
        index=pd.DatetimeIndex([pd.Timestamp(p.date) for p in points]),
    )
    s.index.name = "date"
    return s


def _to_date(x) -> date:
    if hasattr(x, "date") and callable(x.date):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(x).date()
