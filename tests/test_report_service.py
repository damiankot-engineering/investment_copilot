"""Tests for ReportService."""

from __future__ import annotations

from datetime import date, datetime, timezone

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
from investment_copilot.domain.prompts import (
    HoldingComment,
    PortfolioAnalysis,
    RiskAlert,
    RiskAlerts,
)
from investment_copilot.services.report_service import ReportService


# --- Fixtures --------------------------------------------------------------


def _portfolio() -> Portfolio:
    return Portfolio(
        holdings=[
            Holding(
                ticker="PKN",
                name="PKN Orlen",
                shares=100,
                entry_price=65,
                entry_date=date(2023, 4, 12),
                thesis="thesis",
            ),
            Holding(
                ticker="CDR",
                shares=25,
                entry_price=140,
                entry_date=date(2024, 1, 8),
                thesis="thesis",
            ),
        ]
    )


def _status(*, missing: list[str] | None = None) -> PortfolioStatus:
    pkn = HoldingStatus(
        ticker="pkn.pl",
        name="PKN Orlen",
        shares=100,
        entry_price=65,
        entry_date=date(2023, 4, 12),
        cost_basis=6500,
        last_price=70,
        last_price_date=date(2024, 4, 1),
        market_value=7000,
        unrealized_pnl=500,
        unrealized_pnl_pct=500 / 6500,
    )
    cdr = HoldingStatus(
        ticker="cdr.pl",
        name=None,
        shares=25,
        entry_price=140,
        entry_date=date(2024, 1, 8),
        cost_basis=3500,
    )
    return PortfolioStatus(
        base_currency="PLN",
        as_of=datetime(2024, 4, 1, 12, 0, tzinfo=timezone.utc),
        holdings=[pkn, cdr],
        total_cost_basis=10_000.0,
        priced_cost_basis=6500.0,
        total_market_value=7000.0,
        total_unrealized_pnl=500.0,
        total_unrealized_pnl_pct=500 / 6500,
        missing_data=["cdr.pl"] if missing is None else missing,
    )


def _backtest() -> BacktestResult:
    metrics = StrategyMetrics(
        total_return=0.25,
        annualized_return=0.12,
        annualized_volatility=0.18,
        sharpe_ratio=0.67,
        max_drawdown=-0.15,
        max_drawdown_duration_days=42,
        win_rate=0.55,
        n_observations=500,
    )
    bench = StrategyMetrics(
        total_return=0.10,
        annualized_return=0.05,
        annualized_volatility=0.20,
        sharpe_ratio=0.25,
        max_drawdown=-0.20,
        max_drawdown_duration_days=60,
        win_rate=0.52,
        n_observations=500,
    )
    return BacktestResult(
        strategy_name="ma_crossover",
        strategy_params={"fast": 50, "slow": 200},
        start_date=date(2022, 1, 3),
        end_date=date(2024, 4, 1),
        initial_capital=100_000.0,
        final_value=125_000.0,
        equity_curve=[EquityPoint(date=date(2022, 1, 3), value=100_000.0)],
        metrics=metrics,
        benchmark_symbol="^wig20",
        benchmark_metrics=bench,
        generated_at=datetime(2024, 4, 1, tzinfo=timezone.utc),
    )


def _analysis() -> PortfolioAnalysis:
    return PortfolioAnalysis(
        summary="Krótkie podsumowanie portfela.",
        holdings_comments=[
            HoldingComment(ticker="pkn.pl", comment="OK", recommendation="trzymaj"),
            HoldingComment(ticker="cdr.pl", comment="brak danych", recommendation="obserwuj"),
        ],
        diversification_notes="Skupienie na GPW.",
        confidence=7,
    )


def _risks(empty: bool = False) -> RiskAlerts:
    if empty:
        return RiskAlerts(overview="Brak istotnych ryzyk.", alerts=[])
    return RiskAlerts(
        overview="Krótki przegląd ryzyka.",
        alerts=[
            RiskAlert(
                ticker="pkn.pl",
                severity="wysokie",
                title="Spadek cen ropy",
                description="Wpływa na rafinerie.",
                suggested_action="Monitoruj raporty kwartalne.",
            ),
            RiskAlert(
                ticker=None,
                severity="średnie",
                title="Koncentracja sektorowa",
                description="Wszystko na GPW.",
                suggested_action="Rozważ dywersyfikację.",
            ),
        ],
    )


# --- Pure rendering tests --------------------------------------------------


def test_render_minimum_sections() -> None:
    svc = ReportService(output_dir="/tmp/unused")
    body = svc.render(portfolio=_portfolio(), status=_status())
    assert "# Raport portfela" in body
    assert "## Portfel" in body
    assert "Łączna wartość rynkowa" in body
    assert "PKN Orlen" in body
    assert "Brak danych rynkowych" in body  # because missing_data is set
    assert "## Backtest" not in body
    assert "## Analiza (AI)" not in body


def test_render_all_sections() -> None:
    svc = ReportService(output_dir="/tmp/unused")
    body = svc.render(
        portfolio=_portfolio(),
        status=_status(),
        backtest=_backtest(),
        analysis=_analysis(),
        risks=_risks(),
        warnings=["something happened"],
    )
    assert "## Backtest" in body
    assert "ma_crossover" in body
    assert "Benchmark" in body or "^wig20" in body
    assert "## Analiza (AI)" in body
    assert "Pewność analizy" in body
    assert "## Ryzyka (AI)" in body
    assert "🔴" in body  # severity marker for "wysokie"
    assert "🟡" in body  # severity marker for "średnie"
    assert "## Ostrzeżenia" in body
    assert "something happened" in body


def test_render_empty_risks_list() -> None:
    svc = ReportService(output_dir="/tmp/unused")
    body = svc.render(portfolio=_portfolio(), status=_status(), risks=_risks(empty=True))
    assert "brak istotnych ryzyk" in body


def test_render_no_missing_data_omits_warning() -> None:
    svc = ReportService(output_dir="/tmp/unused")
    body = svc.render(portfolio=_portfolio(), status=_status(missing=[]))
    assert "Brak danych rynkowych" not in body


def test_render_handles_pipe_in_name() -> None:
    p = Portfolio(
        holdings=[
            Holding(
                ticker="PKN",
                name="Bad|Name",
                shares=10,
                entry_price=65,
                entry_date=date(2023, 4, 12),
                thesis="t",
            )
        ]
    )
    s = PortfolioStatus(
        base_currency="PLN",
        as_of=datetime(2024, 4, 1, tzinfo=timezone.utc),
        holdings=[
            HoldingStatus(
                ticker="pkn.pl",
                name="Bad|Name",
                shares=10,
                entry_price=65,
                entry_date=date(2023, 4, 12),
                cost_basis=650,
                last_price=70,
                last_price_date=date(2024, 4, 1),
                market_value=700,
                unrealized_pnl=50,
                unrealized_pnl_pct=50 / 650,
            )
        ],
        total_cost_basis=650,
        priced_cost_basis=650,
        total_market_value=700,
        total_unrealized_pnl=50,
        total_unrealized_pnl_pct=50 / 650,
    )
    body = ReportService(output_dir="/tmp/unused").render(portfolio=p, status=s)
    # Pipe must not break the markdown table
    assert "Bad|Name" not in body
    assert "Bad/Name" in body


# --- Disk write -----------------------------------------------------------


def test_write_creates_file_and_directory(tmp_path) -> None:
    svc = ReportService(output_dir=tmp_path / "reports")
    path = svc.write(portfolio=_portfolio(), status=_status())
    assert path.is_file()
    assert path.parent == tmp_path / "reports"
    assert path.read_text(encoding="utf-8").startswith("# Raport portfela")


def test_write_explicit_filename(tmp_path) -> None:
    svc = ReportService(output_dir=tmp_path)
    path = svc.write(
        portfolio=_portfolio(), status=_status(), filename="custom.md"
    )
    assert path.name == "custom.md"


def test_write_default_filename_is_timestamped(tmp_path) -> None:
    svc = ReportService(output_dir=tmp_path)
    path = svc.write(portfolio=_portfolio(), status=_status())
    assert path.name.startswith("report_")
    assert path.name.endswith(".md")
