"""Build compact Markdown context blocks from domain objects.

Pure functions over the typed domain models. No I/O, no LLM calls.
Output is deterministic Markdown, which makes prompt content trivially
unit-testable.

The blocks are designed to be:
* Token-economical (fixed-width, no decorative whitespace).
* Self-describing (each block has a header so the model can reference
  it).
* Length-bounded (truncation rules are explicit constants).
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence

from investment_copilot.domain.backtest import BacktestResult
from investment_copilot.domain.models import NewsItem
from investment_copilot.domain.portfolio import (
    Holding,
    HoldingStatus,
    Portfolio,
    PortfolioStatus,
)

# --- Length / size limits ---------------------------------------------------

MAX_THESIS_CHARS: int = 1500
MAX_NEWS_PER_TICKER: int = 5
MAX_NEWS_TOTAL: int = 30
MAX_NEWS_TITLE_CHARS: int = 200


# --- Holdings table ---------------------------------------------------------


def render_holdings_table(portfolio: Portfolio) -> str:
    if not portfolio.holdings:
        return "## Holdings\n_(brak pozycji)_"
    lines = [
        "## Holdings",
        "| Ticker | Name | Shares | Entry | Entry date | Cost basis | Thesis (skrót) |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for h in portfolio.holdings:
        lines.append(
            f"| {h.ticker} | {_safe(h.name) or '—'} | {h.shares:g} | "
            f"{h.entry_price:.2f} | {h.entry_date.isoformat()} | "
            f"{h.cost_basis:.2f} | {_one_liner(h.thesis, 120)} |"
        )
    return "\n".join(lines)


def render_full_theses(portfolio: Portfolio, *, only_ticker: str | None = None) -> str:
    """Render full theses (truncated). For per-ticker thesis updates."""
    items: Iterable[Holding] = portfolio.holdings
    if only_ticker is not None:
        items = [h for h in portfolio.holdings if h.ticker == only_ticker]
    body: list[str] = ["## Theses"]
    for h in items:
        thesis = h.thesis.strip()
        if len(thesis) > MAX_THESIS_CHARS:
            thesis = thesis[:MAX_THESIS_CHARS] + "…"
        body.append(f"### {h.ticker}{f' ({h.name})' if h.name else ''}\n{thesis}")
    return "\n\n".join(body)


# --- Status snapshot --------------------------------------------------------


def render_status(status: PortfolioStatus) -> str:
    lines = [
        "## Current status",
        f"As of: {status.as_of.strftime('%Y-%m-%d %H:%M UTC')}  "
        f"Base currency: {status.base_currency}",
        f"Total cost basis (all):       {status.total_cost_basis:>14,.2f}",
        f"Priced cost basis:            {status.priced_cost_basis:>14,.2f}",
        f"Total market value:           {status.total_market_value:>14,.2f}",
        f"Unrealized PnL:               {status.total_unrealized_pnl:>+14,.2f}  "
        f"({status.total_unrealized_pnl_pct * 100:+.2f}%)",
    ]
    if status.missing_data:
        lines.append(
            "Missing market data for: " + ", ".join(status.missing_data)
        )
    lines.append("")
    lines.append(
        "| Ticker | Last | Date | Value | Unrealized PnL | PnL% |"
    )
    lines.append("|---|---:|---|---:|---:|---:|")
    for s in status.holdings:
        lines.append(_status_row(s))
    return "\n".join(lines)


def _status_row(s: HoldingStatus) -> str:
    if not s.has_price:
        return f"| {s.ticker} | — | — | — | — | — |"
    return (
        f"| {s.ticker} | {s.last_price:.2f} | "
        f"{s.last_price_date.isoformat() if s.last_price_date else '—'} | "
        f"{s.market_value:.2f} | {s.unrealized_pnl:+.2f} | "
        f"{s.unrealized_pnl_pct * 100:+.2f}% |"
    )


# --- Backtest summary -------------------------------------------------------


def render_backtest(result: BacktestResult | None) -> str:
    if result is None:
        return "## Backtest\n_(brak — uruchom `backtest` aby dodać kontekst)_"
    m = result.metrics
    bm_section = ""
    if result.benchmark_metrics is not None:
        bm = result.benchmark_metrics
        bm_section = (
            "\n\n**Benchmark (buy & hold)**: "
            f"{result.benchmark_symbol} | "
            f"total {bm.total_return * 100:+.2f}% | "
            f"ann. {bm.annualized_return * 100:+.2f}% | "
            f"vol {bm.annualized_volatility * 100:.2f}% | "
            f"Sharpe {bm.sharpe_ratio:.2f} | "
            f"max DD {bm.max_drawdown * 100:+.2f}%"
        )
    return (
        "## Backtest\n"
        f"Strategy: **{result.strategy_name}** params={result.strategy_params}  "
        f"window {result.start_date.isoformat()} → {result.end_date.isoformat()}  "
        f"capital {result.initial_capital:,.0f} → {result.final_value:,.2f}\n\n"
        f"**Metrics**: total {m.total_return * 100:+.2f}% | "
        f"ann. {m.annualized_return * 100:+.2f}% | "
        f"vol {m.annualized_volatility * 100:.2f}% | "
        f"Sharpe {m.sharpe_ratio:.2f} | "
        f"max DD {m.max_drawdown * 100:+.2f}% "
        f"({m.max_drawdown_duration_days}d) | "
        f"win rate {m.win_rate * 100:.1f}% | "
        f"obs {m.n_observations}"
        f"{bm_section}"
    )


# --- News -------------------------------------------------------------------


def render_news(
    news: Sequence[NewsItem],
    *,
    per_ticker_limit: int = MAX_NEWS_PER_TICKER,
    total_limit: int = MAX_NEWS_TOTAL,
) -> str:
    if not news:
        return "## Recent news\n_(brak ostatnich wiadomości w cache)_"

    sorted_news = sorted(news, key=lambda n: n.published_at, reverse=True)
    if per_ticker_limit:
        kept: list[NewsItem] = []
        seen: dict[str | None, int] = {}
        for n in sorted_news:
            key = n.ticker
            count = seen.get(key, 0)
            if count >= per_ticker_limit:
                continue
            kept.append(n)
            seen[key] = count + 1
            if len(kept) >= total_limit:
                break
        sorted_news = kept
    else:
        sorted_news = sorted_news[:total_limit]

    lines = ["## Recent news"]
    for n in sorted_news:
        title = _one_liner(n.title, MAX_NEWS_TITLE_CHARS)
        when = n.published_at.strftime("%Y-%m-%d")
        ticker = f"[{n.ticker}] " if n.ticker else ""
        lines.append(f"- {when} {ticker}{title}  ({n.source})")
    return "\n".join(lines)


# --- Top-level assemblers --------------------------------------------------


def build_portfolio_context(
    portfolio: Portfolio,
    status: PortfolioStatus,
    *,
    backtest: BacktestResult | None = None,
    news: Sequence[NewsItem] = (),
) -> str:
    return "\n\n".join(
        [
            render_holdings_table(portfolio),
            render_status(status),
            render_backtest(backtest),
            render_news(news),
        ]
    )


def build_thesis_context(
    portfolio: Portfolio,
    status: PortfolioStatus,
    *,
    ticker: str,
    news: Sequence[NewsItem] = (),
) -> str:
    holding = portfolio.find(ticker)
    if holding is None:
        raise ValueError(f"Ticker {ticker} not found in portfolio")
    holding_status = next(
        (s for s in status.holdings if s.ticker == holding.ticker), None
    )
    pieces = [render_full_theses(portfolio, only_ticker=holding.ticker)]
    if holding_status:
        pieces.append(_render_single_holding_status(holding_status))
    pieces.append(render_news(news))
    return "\n\n".join(pieces)


def _render_single_holding_status(s: HoldingStatus) -> str:
    if not s.has_price:
        return (
            f"## Status: {s.ticker}\n"
            "_(brak danych rynkowych w cache — uruchom `update-data`)_"
        )
    return (
        f"## Status: {s.ticker}\n"
        f"shares {s.shares:g} | entry {s.entry_price:.2f} "
        f"({s.entry_date.isoformat()}) | cost {s.cost_basis:.2f}\n"
        f"last {s.last_price:.2f} on "
        f"{s.last_price_date.isoformat() if s.last_price_date else '—'} | "
        f"value {s.market_value:.2f} | "
        f"PnL {s.unrealized_pnl:+.2f} ({s.unrealized_pnl_pct * 100:+.2f}%)"
    )


# --- Helpers ----------------------------------------------------------------


def _safe(s: str | None) -> str | None:
    if s is None:
        return None
    return s.replace("|", "/").replace("\n", " ").strip() or None


def _one_liner(s: str, max_chars: int) -> str:
    cleaned = " ".join(s.split())
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 1].rstrip() + "…"
    return cleaned
