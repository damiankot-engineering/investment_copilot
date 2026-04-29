"""Tests for the prompt-context builder."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from investment_copilot.domain.backtest import (
    BacktestResult,
    EquityPoint,
    StrategyMetrics,
)
from investment_copilot.domain.models import NewsItem
from investment_copilot.domain.portfolio import (
    Holding,
    HoldingStatus,
    Portfolio,
    PortfolioStatus,
)
from investment_copilot.domain.prompts.context import (
    build_portfolio_context,
    build_thesis_context,
    render_backtest,
    render_holdings_table,
    render_news,
    render_status,
)


def _holding(ticker: str, **over) -> Holding:
    base = dict(
        ticker=ticker,
        shares=10,
        entry_price=100.0,
        entry_date=date(2023, 1, 2),
        thesis="long thesis goes here",
    )
    base.update(over)
    return Holding(**base)


def _holding_status(ticker: str, *, last_price: float | None = 110.0) -> HoldingStatus:
    return HoldingStatus(
        ticker=ticker,
        name=None,
        shares=10,
        entry_price=100.0,
        entry_date=date(2023, 1, 2),
        cost_basis=1000.0,
        last_price=last_price,
        last_price_date=date(2024, 4, 1) if last_price else None,
        market_value=last_price * 10 if last_price else None,
        unrealized_pnl=(last_price - 100.0) * 10 if last_price else None,
        unrealized_pnl_pct=(last_price - 100.0) / 100.0 if last_price else None,
    )


# --- holdings table --------------------------------------------------------


def test_render_holdings_table_includes_each_ticker() -> None:
    p = Portfolio(holdings=[_holding("PKN"), _holding("CDR", name="CD Projekt")])
    out = render_holdings_table(p)
    assert "pkn.pl" in out
    assert "cdr.pl" in out
    assert "CD Projekt" in out
    assert "## Holdings" in out


def test_render_holdings_table_empty() -> None:
    p = Portfolio(holdings=[])
    out = render_holdings_table(p)
    assert "brak pozycji" in out


def test_render_holdings_escapes_pipes() -> None:
    p = Portfolio(holdings=[_holding("PKN", name="Ev|il")])
    out = render_holdings_table(p)
    # Pipe in name must be replaced so the markdown table doesn't break
    assert "Ev|il" not in out
    assert "Ev/il" in out


# --- status ---------------------------------------------------------------


def test_render_status_includes_totals_and_rows() -> None:
    status = PortfolioStatus(
        base_currency="PLN",
        as_of=datetime(2024, 4, 1, 12, 0, tzinfo=timezone.utc),
        holdings=[_holding_status("pkn.pl"), _holding_status("cdr.pl", last_price=None)],
        total_cost_basis=2000.0,
        priced_cost_basis=1000.0,
        total_market_value=1100.0,
        total_unrealized_pnl=100.0,
        total_unrealized_pnl_pct=0.10,
        missing_data=["cdr.pl"],
    )
    out = render_status(status)
    assert "Total cost basis" in out
    assert "Missing market data" in out and "cdr.pl" in out
    assert "+10.00%" in out  # total PnL%
    assert "1,100.00" in out  # total market value


# --- backtest -------------------------------------------------------------


def _backtest_result(*, with_benchmark: bool = True) -> BacktestResult:
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
    bench_metrics = (
        StrategyMetrics(
            total_return=0.10,
            annualized_return=0.05,
            annualized_volatility=0.20,
            sharpe_ratio=0.25,
            max_drawdown=-0.20,
            max_drawdown_duration_days=60,
            win_rate=0.52,
            n_observations=500,
        )
        if with_benchmark
        else None
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
        benchmark_symbol="^wig20" if with_benchmark else None,
        benchmark_equity_curve=[],
        benchmark_metrics=bench_metrics,
        missing_tickers=[],
        tickers_used=["pkn.pl"],
        generated_at=datetime(2024, 4, 1, tzinfo=timezone.utc),
    )


def test_render_backtest_with_benchmark() -> None:
    out = render_backtest(_backtest_result(with_benchmark=True))
    assert "ma_crossover" in out
    assert "^wig20" in out
    assert "Sharpe" in out
    assert "Benchmark" in out


def test_render_backtest_without_benchmark() -> None:
    out = render_backtest(_backtest_result(with_benchmark=False))
    assert "Benchmark" not in out


def test_render_backtest_none() -> None:
    assert "brak" in render_backtest(None)


# --- news -----------------------------------------------------------------


def _news(title: str, *, ticker: str | None = "pkn.pl", days_ago: int = 1) -> NewsItem:
    return NewsItem(
        ticker=ticker,
        source="rss:test",
        title=title,
        url=f"https://example.com/{title}".replace(" ", "-"),
        published_at=datetime(2024, 4, 1, tzinfo=timezone.utc),
    )


def test_render_news_orders_recent_first() -> None:
    items = [
        NewsItem(
            ticker="pkn.pl",
            source="rss:test",
            title=f"news-{i}",
            url=f"https://example.com/{i}",
            published_at=datetime(2024, 4, i + 1, tzinfo=timezone.utc),
        )
        for i in range(5)
    ]
    out = render_news(items)
    # Newest (i=4 -> 2024-04-05) should appear before oldest (2024-04-01)
    assert out.index("news-4") < out.index("news-0")


def test_render_news_per_ticker_limit() -> None:
    items = [
        NewsItem(
            ticker="pkn.pl",
            source="rss:test",
            title=f"pkn-{i}",
            url=f"https://example.com/pkn/{i}",
            published_at=datetime(2024, 4, i + 1, tzinfo=timezone.utc),
        )
        for i in range(8)
    ] + [
        NewsItem(
            ticker="cdr.pl",
            source="rss:test",
            title="cdr-1",
            url="https://example.com/cdr/1",
            published_at=datetime(2024, 4, 1, tzinfo=timezone.utc),
        )
    ]
    out = render_news(items, per_ticker_limit=3)
    pkn_count = sum(1 for line in out.splitlines() if "pkn-" in line)
    assert pkn_count == 3
    assert "cdr-1" in out


def test_render_news_empty() -> None:
    assert "brak" in render_news([])


# --- end-to-end builders --------------------------------------------------


def test_build_portfolio_context_combines_all_blocks() -> None:
    p = Portfolio(holdings=[_holding("PKN")])
    s = PortfolioStatus(
        base_currency="PLN",
        as_of=datetime(2024, 4, 1, tzinfo=timezone.utc),
        holdings=[_holding_status("pkn.pl")],
        total_cost_basis=1000.0,
        priced_cost_basis=1000.0,
        total_market_value=1100.0,
        total_unrealized_pnl=100.0,
        total_unrealized_pnl_pct=0.10,
        missing_data=[],
    )
    ctx = build_portfolio_context(
        p, s, backtest=_backtest_result(), news=[_news("Headline 1")]
    )
    assert "## Holdings" in ctx
    assert "## Current status" in ctx
    assert "## Backtest" in ctx
    assert "## Recent news" in ctx
    assert "Headline 1" in ctx


def test_build_thesis_context_for_known_ticker() -> None:
    p = Portfolio(holdings=[_holding("PKN", thesis="A" * 50)])
    s = PortfolioStatus(
        base_currency="PLN",
        as_of=datetime(2024, 4, 1, tzinfo=timezone.utc),
        holdings=[_holding_status("pkn.pl")],
        total_cost_basis=1000.0,
        priced_cost_basis=1000.0,
        total_market_value=1100.0,
        total_unrealized_pnl=100.0,
        total_unrealized_pnl_pct=0.10,
    )
    ctx = build_thesis_context(p, s, ticker="pkn.pl")
    assert "## Theses" in ctx
    assert "## Status: pkn.pl" in ctx


def test_build_thesis_context_unknown_ticker_raises() -> None:
    p = Portfolio(holdings=[_holding("PKN")])
    s = PortfolioStatus(
        base_currency="PLN",
        as_of=datetime(2024, 4, 1, tzinfo=timezone.utc),
        holdings=[],
        total_cost_basis=0,
        priced_cost_basis=0,
        total_market_value=0,
        total_unrealized_pnl=0,
        total_unrealized_pnl_pct=0,
    )
    with pytest.raises(ValueError, match="not found"):
        build_thesis_context(p, s, ticker="xyz.pl")
