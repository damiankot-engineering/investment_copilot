"""Tests for ``investment_copilot.gui`` helper functions."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from investment_copilot.domain.backtest import (
    BacktestResult,
    EquityPoint,
    StrategyMetrics,
)
from investment_copilot.domain.portfolio import (
    Holding,
    HoldingStatus,
    Portfolio,
    PortfolioStatus,
)
from investment_copilot.gui import (
    drawdown_series,
    equity_curves_dataframe,
    format_money,
    format_money_signed,
    format_pct,
    holdings_dataframe,
    list_reports,
)


# --- Fixtures --------------------------------------------------------------


def _portfolio() -> Portfolio:
    return Portfolio(
        holdings=[
            Holding(
                ticker="PKN",
                name="PKN Orlen",
                shares=100,
                entry_price=65.0,
                entry_date=date(2023, 4, 12),
                thesis="t",
            ),
            Holding(
                ticker="CDR",
                shares=25,
                entry_price=140.0,
                entry_date=date(2024, 1, 8),
                thesis="t",
            ),
        ]
    )


def _status() -> PortfolioStatus:
    pkn = HoldingStatus(
        ticker="pkn.pl",
        name="PKN Orlen",
        shares=100,
        entry_price=65.0,
        entry_date=date(2023, 4, 12),
        cost_basis=6500.0,
        last_price=70.0,
        last_price_date=date(2024, 4, 1),
        market_value=7000.0,
        unrealized_pnl=500.0,
        unrealized_pnl_pct=500 / 6500,
    )
    cdr = HoldingStatus(
        ticker="cdr.pl",
        name=None,
        shares=25,
        entry_price=140.0,
        entry_date=date(2024, 1, 8),
        cost_basis=3500.0,
    )
    return PortfolioStatus(
        base_currency="PLN",
        as_of=datetime(2024, 4, 1, tzinfo=timezone.utc),
        holdings=[pkn, cdr],
        total_cost_basis=10_000.0,
        priced_cost_basis=6500.0,
        total_market_value=7000.0,
        total_unrealized_pnl=500.0,
        total_unrealized_pnl_pct=500 / 6500,
        missing_data=["cdr.pl"],
    )


def _backtest_with_benchmark() -> BacktestResult:
    metrics = StrategyMetrics(
        total_return=0.25,
        annualized_return=0.12,
        annualized_volatility=0.18,
        sharpe_ratio=0.67,
        max_drawdown=-0.15,
        max_drawdown_duration_days=42,
        win_rate=0.55,
        n_observations=3,
    )
    bench = StrategyMetrics(
        total_return=0.10,
        annualized_return=0.05,
        annualized_volatility=0.20,
        sharpe_ratio=0.25,
        max_drawdown=-0.20,
        max_drawdown_duration_days=60,
        win_rate=0.52,
        n_observations=3,
    )
    return BacktestResult(
        strategy_name="ma_crossover",
        strategy_params={"fast": 50, "slow": 200},
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 4),
        initial_capital=100_000.0,
        final_value=125_000.0,
        equity_curve=[
            EquityPoint(date=date(2024, 1, 2), value=100_000.0),
            EquityPoint(date=date(2024, 1, 3), value=110_000.0),
            EquityPoint(date=date(2024, 1, 4), value=125_000.0),
        ],
        metrics=metrics,
        benchmark_symbol="^wig20",
        benchmark_equity_curve=[
            EquityPoint(date=date(2024, 1, 2), value=100_000.0),
            EquityPoint(date=date(2024, 1, 3), value=105_000.0),
            EquityPoint(date=date(2024, 1, 4), value=110_000.0),
        ],
        benchmark_metrics=bench,
        generated_at=datetime(2024, 4, 1, tzinfo=timezone.utc),
    )


# --- holdings_dataframe ----------------------------------------------------


def test_holdings_dataframe_shape() -> None:
    df = holdings_dataframe(_portfolio(), _status())
    assert list(df.columns) == [
        "Ticker", "Name", "Shares", "Entry", "Last", "Last date",
        "Cost basis", "Value", "PnL", "PnL%",
    ]
    assert len(df) == 2


def test_holdings_dataframe_priced_row() -> None:
    df = holdings_dataframe(_portfolio(), _status())
    pkn = df[df["Ticker"] == "pkn.pl"].iloc[0]
    assert pkn["Name"] == "PKN Orlen"
    assert pkn["Last"] == 70.0
    assert pkn["Value"] == 7000.0
    assert pkn["PnL"] == 500.0
    assert pkn["PnL%"] == pytest.approx(500 / 6500)


def test_holdings_dataframe_unpriced_row_has_nans() -> None:
    df = holdings_dataframe(_portfolio(), _status())
    cdr = df[df["Ticker"] == "cdr.pl"].iloc[0]
    # pandas converts None to NaN in object columns containing strings;
    # both are valid "missing" representations for our purposes.
    assert pd.isna(cdr["Name"])
    assert pd.isna(cdr["Last"])
    assert pd.isna(cdr["Value"])
    assert pd.isna(cdr["PnL"])


# --- equity_curves_dataframe ----------------------------------------------


def test_equity_curves_dataframe_with_benchmark() -> None:
    df = equity_curves_dataframe(_backtest_with_benchmark())
    assert list(df.columns) == ["Strategy", "Benchmark (^wig20)"]
    assert len(df) == 3
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df["Strategy"].iloc[-1] == 125_000.0
    assert df["Benchmark (^wig20)"].iloc[-1] == 110_000.0


def test_equity_curves_dataframe_without_benchmark() -> None:
    bt = _backtest_with_benchmark()
    bt = bt.model_copy(update={"benchmark_equity_curve": [], "benchmark_symbol": None})
    df = equity_curves_dataframe(bt)
    assert list(df.columns) == ["Strategy"]


# --- drawdown_series ------------------------------------------------------


def test_drawdown_series_known_case() -> None:
    s = pd.Series(
        [100, 120, 90, 80, 110],
        index=pd.date_range("2024-01-02", periods=5, freq="B"),
    )
    dd = drawdown_series(s)
    # peak 120 -> trough 80 = -33.33%
    assert dd.min() == pytest.approx(-1 / 3, abs=1e-9)
    # First point has 0 drawdown
    assert dd.iloc[0] == 0.0


def test_drawdown_series_empty() -> None:
    assert drawdown_series(pd.Series(dtype=float)).empty


# --- list_reports ---------------------------------------------------------


def test_list_reports_orders_newest_first(tmp_path: Path) -> None:
    import time

    a = tmp_path / "a.md"; a.write_text("a")
    time.sleep(0.05)  # ensure different mtime
    b = tmp_path / "b.md"; b.write_text("b")
    time.sleep(0.05)
    c = tmp_path / "c.md"; c.write_text("c")

    files = list_reports(tmp_path)
    assert [p.name for p in files] == ["c.md", "b.md", "a.md"]


def test_list_reports_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "report.md").write_text("md")
    (tmp_path / "notes.txt").write_text("txt")
    (tmp_path / "data.json").write_text("{}")

    files = list_reports(tmp_path)
    assert {p.name for p in files} == {"report.md"}


def test_list_reports_missing_directory(tmp_path: Path) -> None:
    assert list_reports(tmp_path / "does-not-exist") == []


# --- formatters ----------------------------------------------------------


def test_format_pct() -> None:
    assert format_pct(0.1234) == "+12.34%"
    assert format_pct(-0.05) == "-5.00%"
    assert format_pct(0.1234, signed=False) == "12.34%"
    assert format_pct(None) == "—"


def test_format_money() -> None:
    assert format_money(1234.5, currency="PLN") == "1,234.50 PLN"
    assert format_money(None) == "—"


def test_format_money_signed() -> None:
    assert format_money_signed(1234.5, currency="PLN") == "+1,234.50 PLN"
    assert format_money_signed(-1234.5, currency="PLN") == "-1,234.50 PLN"
    assert format_money_signed(None) == "—"
