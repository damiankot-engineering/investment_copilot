"""Pure helpers for the Streamlit GUI.

Anything that can be unit-tested without spinning up Streamlit lives here.
The :mod:`streamlit_app` module imports from this and stays focused on
layout/rendering.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from investment_copilot.domain.backtest import BacktestResult, EquityPoint
from investment_copilot.domain.portfolio import (
    Holding,
    HoldingStatus,
    Portfolio,
    PortfolioStatus,
)


# --- Holdings table for display --------------------------------------------


def holdings_dataframe(
    portfolio: Portfolio,
    status: PortfolioStatus,
) -> pd.DataFrame:
    """Build a tidy DataFrame for the portfolio holdings table.

    Includes both the input fields (shares, entry) and the computed status
    fields (last price, PnL). Missing-data rows render with NaN, which
    Streamlit displays as empty cells.
    """
    by_ticker = {h.ticker: h for h in portfolio.holdings}
    rows: list[dict] = []
    for s in status.holdings:
        holding = by_ticker.get(s.ticker)
        rows.append(
            {
                "Ticker": s.ticker,
                "Name": (holding.name if holding and holding.name else None),
                "Shares": s.shares,
                "Entry": s.entry_price,
                "Last": s.last_price,
                "Last date": s.last_price_date,
                "Cost basis": s.cost_basis,
                "Value": s.market_value,
                "PnL": s.unrealized_pnl,
                "PnL%": s.unrealized_pnl_pct,
            }
        )
    df = pd.DataFrame(rows)
    return df


def equity_curves_dataframe(result: BacktestResult) -> pd.DataFrame:
    """Combine strategy + (optional) benchmark equity curves into one frame.

    Index is the date; columns are 'Strategy' and (when present) 'Benchmark'.
    Designed for direct consumption by Plotly or ``st.line_chart``.
    """
    strat = _equity_to_series(result.equity_curve, name="Strategy")
    if not result.benchmark_equity_curve:
        return strat.to_frame()
    bench = _equity_to_series(
        result.benchmark_equity_curve,
        name=f"Benchmark ({result.benchmark_symbol})",
    )
    return pd.concat([strat, bench], axis=1)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Return drawdown as a fraction (e.g. -0.15 for -15%)."""
    if equity.empty:
        return equity
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _equity_to_series(points: Iterable[EquityPoint], *, name: str) -> pd.Series:
    points = list(points)
    if not points:
        return pd.Series(dtype=float, name=name)
    s = pd.Series(
        [p.value for p in points],
        index=pd.DatetimeIndex([pd.Timestamp(p.date) for p in points]),
        name=name,
    )
    s.index.name = "date"
    return s


# --- Reports listing -------------------------------------------------------


def list_reports(directory: Path | str) -> list[Path]:
    """Return Markdown reports in ``directory``, newest first by mtime."""
    d = Path(directory)
    if not d.is_dir():
        return []
    files = [p for p in d.iterdir() if p.is_file() and p.suffix == ".md"]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


# --- Formatters ------------------------------------------------------------


def format_pct(value: float | None, *, signed: bool = True) -> str:
    if value is None:
        return "—"
    sign = "+" if signed else ""
    return f"{value * 100:{sign}.2f}%"


def format_money(value: float | None, *, currency: str = "PLN") -> str:
    if value is None:
        return "—"
    return f"{value:,.2f} {currency}"


def format_money_signed(value: float | None, *, currency: str = "PLN") -> str:
    if value is None:
        return "—"
    return f"{value:+,.2f} {currency}"
