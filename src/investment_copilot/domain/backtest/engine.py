"""Portfolio-level backtest simulator.

Math
----
Let ``S[t][k] in {0, 1}`` be the signal for ticker ``k`` on day ``t``
(strategies are responsible for lagging this so it uses no future info).
Let ``has_data[t][k]`` be 1 if ticker ``k`` has a valid close on day ``t``,
else 0. The *effective* signal is ``S' = S * has_data``.

The active count is ``n[t] = sum_k S'[t][k]``. Equal-weighting:

    w[t][k] = S'[t][k] / n[t]      if n[t] > 0
    w[t][k] = 0                    otherwise

Daily portfolio return:

    r_p[t] = sum_k ( w[t][k] * r[t][k] )

where ``r[t][k] = close[t][k] / close[t-1][k] - 1`` (NaN when data is
missing; treated as 0). Equity:

    E[t] = E[0] * cumprod(1 + r_p)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EngineRun:
    """Internal output of :func:`simulate_portfolio`. Service-layer wraps it."""

    equity_curve: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    tickers_used: list[str]


class BacktestError(RuntimeError):
    """Raised by the engine when inputs cannot be simulated."""


def simulate_portfolio(
    panel: Mapping[str, pd.DataFrame],
    signals: Mapping[str, pd.Series],
    *,
    initial_capital: float,
    start: date,
    end: date | None = None,
) -> EngineRun:
    """Run a portfolio-level backtest over the provided data.

    Parameters
    ----------
    panel:
        ``{ticker: ohlcv_dataframe}``. Frames must have a 'close' column
        and a :class:`~pandas.DatetimeIndex`. Tickers absent from this map
        are skipped.
    signals:
        ``{ticker: signal_series}`` of dtype int in {0, 1}, indexed like
        the corresponding panel frame. Missing tickers default to 0.
    initial_capital:
        Starting equity, in the portfolio's base currency.
    start, end:
        Date window for the equity curve (inclusive). ``end=None`` runs to
        the latest available data.
    """
    if initial_capital <= 0:
        raise BacktestError("initial_capital must be > 0")

    tickers = [t for t, df in panel.items() if df is not None and not df.empty]
    if not tickers:
        raise BacktestError("simulate_portfolio: no tickers with data")

    closes = pd.DataFrame({t: panel[t]["close"].astype(float) for t in tickers})
    closes = closes.sort_index()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else closes.index.max()
    closes = closes.loc[(closes.index >= start_ts) & (closes.index <= end_ts)]

    if closes.empty:
        raise BacktestError(
            f"No price data in window [{start_ts.date()}, {end_ts.date()}]"
        )

    sigs = pd.DataFrame(
        {t: signals.get(t, pd.Series(dtype=int)) for t in tickers}
    )
    sigs = sigs.reindex(closes.index).fillna(0).astype(int)

    # A ticker can only contribute on days where it has a valid close
    has_data = closes.notna().astype(int)
    sigs_eff = sigs * has_data

    # Equal-weight allocation across the active set
    active_count = sigs_eff.sum(axis=1)
    safe_count = active_count.where(active_count > 0, np.nan)
    weights = sigs_eff.div(safe_count, axis=0).fillna(0.0)

    daily_returns_per_ticker = closes.pct_change().fillna(0.0)
    portfolio_returns = (weights * daily_returns_per_ticker).sum(axis=1).fillna(0.0)
    portfolio_returns.name = "portfolio_return"

    equity = float(initial_capital) * (1.0 + portfolio_returns).cumprod()
    equity.iloc[0] = float(initial_capital)  # ensure exact start
    equity.name = "equity"

    return EngineRun(
        equity_curve=equity,
        daily_returns=portfolio_returns,
        weights=weights,
        tickers_used=tickers,
    )


def benchmark_buy_and_hold(
    benchmark_close: pd.Series,
    *,
    initial_capital: float,
    start: date,
    end: date | None = None,
) -> pd.Series:
    """Buy-and-hold equity curve on a benchmark price series."""
    if benchmark_close.empty:
        raise BacktestError("benchmark series is empty")
    s = benchmark_close.astype(float).sort_index()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else s.index.max()
    s = s.loc[(s.index >= start_ts) & (s.index <= end_ts)]
    if s.empty:
        raise BacktestError(
            f"No benchmark data in window [{start_ts.date()}, {end_ts.date()}]"
        )
    base = float(s.iloc[0])
    if base <= 0:
        raise BacktestError("benchmark first value is non-positive")
    equity = initial_capital * s / base
    equity.name = "benchmark_equity"
    return equity
