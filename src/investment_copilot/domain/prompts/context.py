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

from investment_copilot.domain.analysis_metrics import (
    CitationRegistry,
    PortfolioMetrics,
    build_metric_keys,
)
from investment_copilot.domain.backtest import BacktestResult
from investment_copilot.domain.fundamentals import (
    FundamentalsSnapshot,
    MonitoringSnapshot,
    is_earnings_related,
)
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
        f"Total market value:  {status.total_market_value:>14,.2f}",
        f"Unrealized PnL:      {status.total_unrealized_pnl:>+14,.2f}  "
        f"({status.total_unrealized_pnl_pct * 100:+.2f}%)",
    ]
    if status.missing_data:
        lines.append(
            "Missing market data for: " + ", ".join(status.missing_data)
        )
    lines.append("")
    lines.append(
        "| Ticker | Last | Value | Waga portfela | PnL% |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    total_mv = status.total_market_value or 0.0
    for s in status.holdings:
        if not s.has_price:
            lines.append(f"| {s.ticker} | — | — | — | — |")
            continue
        weight = (s.market_value / total_mv * 100) if total_mv > 0 else 0.0
        lines.append(
            f"| {s.ticker} | {s.last_price:.2f} | "
            f"{s.market_value:,.2f} | **{weight:.1f}%** | "
            f"{s.unrealized_pnl_pct * 100:+.2f}% |"
        )
    return "\n".join(lines)


def _status_row(s: HoldingStatus) -> str:
    """Legacy single-row formatter (kept for any external callers)."""
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


def _pick_news_for_context(
    news: Sequence[NewsItem],
    *,
    per_ticker_limit: int,
    total_limit: int,
) -> list[NewsItem]:
    sorted_news = sorted(news, key=lambda n: n.published_at, reverse=True)
    if not per_ticker_limit:
        return sorted_news[:total_limit]
    kept: list[NewsItem] = []
    seen: dict[str | None, int] = {}
    for n in sorted_news:
        count = seen.get(n.ticker, 0)
        if count >= per_ticker_limit:
            continue
        kept.append(n)
        seen[n.ticker] = count + 1
        if len(kept) >= total_limit:
            break
    return kept


def render_news(
    news: Sequence[NewsItem],
    *,
    per_ticker_limit: int = MAX_NEWS_PER_TICKER,
    total_limit: int = MAX_NEWS_TOTAL,
) -> str:
    """Render recent news with stable `[news:N]` IDs the LLM can cite."""
    picked = _pick_news_for_context(
        news, per_ticker_limit=per_ticker_limit, total_limit=total_limit
    )
    if not picked:
        return "## Recent news\n_(brak ostatnich wiadomości w cache)_"

    lines = [
        "## Recent news",
        "_Cytuj te pozycje jako `news:N` w polu `citations` (np. `news:3`)._",
    ]
    for idx, n in enumerate(picked, start=1):
        title = _one_liner(n.title, MAX_NEWS_TITLE_CHARS)
        when = n.published_at.strftime("%Y-%m-%d")
        ticker = f"[{n.ticker}] " if n.ticker else ""
        lines.append(f"- **news:{idx}** {when} {ticker}{title}  ({n.source})")
    return "\n".join(lines)


def news_ids_for(
    news: Sequence[NewsItem],
    *,
    per_ticker_limit: int = MAX_NEWS_PER_TICKER,
    total_limit: int = MAX_NEWS_TOTAL,
) -> set[str]:
    """The set of valid `news:N` IDs given the same picking rules as `render_news`."""
    picked = _pick_news_for_context(
        news, per_ticker_limit=per_ticker_limit, total_limit=total_limit
    )
    return {f"news:{i}" for i in range(1, len(picked) + 1)}


# --- Quantitative metrics block --------------------------------------------


def _fmt_pct(v: float | None, *, signed: bool = False, decimals: int = 2) -> str:
    if v is None:
        return "—"
    sign = "+" if signed and v > 0 else ""
    return f"{sign}{v:.{decimals}f}%"


def _fmt_num(v: float | None, *, decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def render_quant_metrics(metrics: PortfolioMetrics | None) -> str:
    """Render the quantitative metrics block — the LLM should NEVER recompute these.

    Includes a stable citation-key convention (e.g. ``metric:pkn.pl.ret_30d_pct``,
    ``metric:portfolio.hhi``) so the model can ground claims in specific numbers.
    """
    if metrics is None:
        return "## Quant metrics\n_(brak — wymaga danych OHLCV w cache)_"

    bench = metrics.benchmark_symbol or "—"
    hhi_label = (
        "wysoka" if metrics.hhi is not None and metrics.hhi > 2500
        else "umiarkowana" if metrics.hhi is not None and metrics.hhi > 1500
        else "niska" if metrics.hhi is not None
        else "—"
    )
    lines = [
        "## Quant metrics — PRE-COMPUTED, cytuj wprost",
        "_Cytuj te wartości jako `metric:KEY` w polu `citations` "
        "(np. `metric:portfolio.hhi`, `metric:pkn.pl.ret_30d_pct`). "
        "NIE rób własnych obliczeń — wszystkie liczby poniżej są autorytatywne._",
        "",
        "### Portfolio-level",
        f"- **portfolio.hhi**: {_fmt_num(metrics.hhi, decimals=0)}  (koncentracja: {hhi_label})",
        f"- **portfolio.top3_weight_pct**: {_fmt_pct(metrics.top3_weight_pct, decimals=1)}",
        f"- **portfolio.largest_position_weight_pct**: "
        f"{_fmt_pct(metrics.largest_position_weight_pct, decimals=1)} "
        f"({metrics.largest_position_ticker or '—'})",
        f"- Benchmark dla bety: **{bench}**",
        "",
        "### Per-holding",
        "| Ticker | Waga % | Ret 30d | Ret 90d | Ret 252d | Od 52w high | Od 52w low | Vol ann. | Beta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in metrics.holdings:
        lines.append(
            "| "
            + " | ".join(
                [
                    h.ticker,
                    _fmt_pct(h.weight_pct, decimals=1),
                    _fmt_pct(h.ret_30d_pct, signed=True, decimals=1),
                    _fmt_pct(h.ret_90d_pct, signed=True, decimals=1),
                    _fmt_pct(h.ret_252d_pct, signed=True, decimals=1),
                    _fmt_pct(h.distance_from_52w_high_pct, signed=True, decimals=1),
                    _fmt_pct(h.distance_from_52w_low_pct, signed=True, decimals=1),
                    _fmt_pct(h.ann_volatility_pct, decimals=1),
                    _fmt_num(h.beta_vs_benchmark, decimals=2),
                ]
            )
            + " |"
        )

    if metrics.top_correlations:
        lines.append("")
        lines.append("### Top correlations (daily log returns)")
        for c in metrics.top_correlations:
            a, b = sorted([c.ticker_a, c.ticker_b])
            lines.append(
                f"- **corr.{a}.{b}**: {c.correlation:+.2f}"
            )
    return "\n".join(lines)


def render_history(reports: Sequence[tuple[str, str]]) -> str:
    """Render previously generated Markdown reports as RAG context.

    ``reports`` is an iterable of ``(label, body)`` pairs already truncated
    to a reasonable length by the loader. The label is used as the
    citation key (e.g. ``previous_report:weekly_2026-05-07``).
    """
    if not reports:
        return ""
    lines = [
        "## Previous reports (RAG)",
        "_Cytuj jako `previous_report:LABEL`. Uwzględnij zmiany sytuacji od "
        "tych raportów. NIE kopiuj wniosków słowo w słowo._",
    ]
    for label, body in reports:
        lines.append("")
        lines.append(f"### previous_report:{label}")
        lines.append(body.strip())
    return "\n".join(lines)


# --- Top-level assemblers --------------------------------------------------


def build_portfolio_context(
    portfolio: Portfolio,
    status: PortfolioStatus,
    *,
    backtest: BacktestResult | None = None,
    news: Sequence[NewsItem] = (),
    metrics: PortfolioMetrics | None = None,
    history: Sequence[tuple[str, str]] = (),
) -> str:
    pieces = [
        render_holdings_table(portfolio),
        render_status(status),
        render_quant_metrics(metrics),
        render_backtest(backtest),
        render_news(news),
    ]
    history_block = render_history(history)
    if history_block:
        pieces.append(history_block)
    return "\n\n".join(pieces)


def build_citation_registry(
    *,
    news: Sequence[NewsItem] = (),
    metrics: PortfolioMetrics | None = None,
    fundamentals: Sequence[FundamentalsSnapshot] = (),
    history: Sequence[tuple[str, str]] = (),
) -> CitationRegistry:
    """Build the set of valid citation targets matching the prompt context."""
    news_ids = news_ids_for(news)
    metric_keys = build_metric_keys(metrics) if metrics is not None else set()
    fund_keys: set[str] = set()
    for f in fundamentals:
        for field in (
            "last_price", "market_cap", "pe_ratio", "pbv_ratio", "eps",
            "dividend_yield", "week52_high", "week52_low",
            "revenue_yoy_pct", "ebitda_yoy_pct", "net_profit_yoy_pct",
            "latest_quarter_label", "next_report_estimated_date",
        ):
            fund_keys.add(f"{f.ticker.lower()}.{field}")
    report_keys = {f"previous_report:{label}" for label, _ in history}
    return CitationRegistry(
        news_ids=news_ids,
        metric_keys=metric_keys,
        fundamentals_keys=fund_keys,
        report_keys=report_keys,
    )


def render_fundamentals(snapshots: Sequence[FundamentalsSnapshot]) -> str:
    """Render the latest fundamentals panel for the prompt context.

    Output has TWO sections:

    * Compact ratios table (Mcap, P/E, P/BV, EPS, 52w, source)
    * Per-ticker BiznesRadar narrative block — pre-computed YoY % changes,
      sector, last/next report dates, ready-to-quote bullets. This is what
      the LLM should base on (no need to invent numbers).
    """
    if not snapshots:
        return "## Fundamentals\n_(brak danych — provider nie zwrócił snapshotów)_"

    lines = [
        "## Fundamentals — wskaźniki rynkowe",
        "| Ticker | Nazwa | Cena | Mcap (PLN) | P/E | P/BV | EPS | 52w (low–high) | Źródło |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for s in snapshots:
        mcap = f"{s.market_cap:,.0f}" if s.market_cap is not None else "—"
        price = f"{s.last_price:.2f}" if s.last_price is not None else "—"
        pe = f"{s.pe_ratio:.2f}" if s.pe_ratio is not None else "—"
        pbv = f"{s.pbv_ratio:.2f}" if s.pbv_ratio is not None else "—"
        eps = f"{s.eps:.2f}" if s.eps is not None else "—"
        if s.week52_low is not None and s.week52_high is not None:
            band = f"{s.week52_low:.2f}–{s.week52_high:.2f}"
        else:
            band = "—"
        name = (s.name or "—").replace("|", "/")
        lines.append(
            f"| {s.ticker} | {name} | {price} | {mcap} | {pe} | {pbv} | "
            f"{eps} | {band} | **{s.source}** |"
        )

    # BR-rich narrative section — only printed for tickers with rich data
    rich = [s for s in snapshots if s.source == "biznesradar"]
    if rich:
        lines.append("")
        lines.append("## Fundamentals — narracja BiznesRadar (cytuj wartości WPROST)")
        for s in rich:
            lines.append("")
            sector = s.sector or "—"
            lq = s.latest_quarter_label or "—"
            last_rep = s.last_report_date.isoformat() if s.last_report_date else "—"
            next_rep = (
                s.next_report_estimated_date.isoformat()
                if s.next_report_estimated_date else "—"
            )
            lines.append(
                f"### {s.ticker} — {s.name or '—'} (sektor: {sector})"
            )
            lines.append(
                f"- Ostatni raport: **{lq}** (publikacja {last_rep}) · "
                f"Szacowany następny raport: **{next_rep}** (orientacyjnie)"
            )
            yoy_parts = []
            if s.revenue_yoy_pct is not None:
                yoy_parts.append(f"przychody {s.revenue_yoy_pct:+.2f}% r/r")
            if s.ebitda_yoy_pct is not None:
                yoy_parts.append(f"EBITDA {s.ebitda_yoy_pct:+.2f}% r/r")
            if s.net_profit_yoy_pct is not None:
                yoy_parts.append(f"zysk netto {s.net_profit_yoy_pct:+.2f}% r/r")
            if yoy_parts:
                lines.append(f"- YoY: {' · '.join(yoy_parts)}")
            for bullet in s.latest_summary[:6]:
                lines.append(f"- {bullet}")

    return "\n".join(lines)


def render_news_with_espi_flag(
    news: Sequence[NewsItem],
    *,
    per_ticker_limit: int = MAX_NEWS_PER_TICKER,
    total_limit: int = MAX_NEWS_TOTAL,
) -> str:
    """Like :func:`render_news`, but flags ESPI/earnings titles with [ESPI]."""
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

    lines = ["## Recent news (z flagą ESPI/earnings)"]
    for n in sorted_news:
        title = _one_liner(n.title, MAX_NEWS_TITLE_CHARS)
        when = n.published_at.strftime("%Y-%m-%d")
        ticker = f"[{n.ticker}] " if n.ticker else ""
        flag = "[ESPI/earnings] " if is_earnings_related(n.title) else ""
        lines.append(f"- {when} {ticker}{flag}{title}  ({n.source})")
    return "\n".join(lines)


def render_previous_snapshot(prev: MonitoringSnapshot | None) -> str:
    """Render the previous monitoring snapshot for diff context.

    Includes BOTH the previous fundamentals (so the LLM can compute deltas)
    AND the previous LLM-generated narrative per ticker (so the LLM can
    carry forward the analysis when fresh data is sparse).
    """
    if prev is None:
        return (
            "## Previous snapshot\n"
            "_(brak — to pierwszy raport monitorujący. Pole 'change_*' "
            "ustaw na null. Dla każdej pozycji rozwiń pełną tezę z sekcji "
            "'Theses (full)' — opisz model biznesowy, znane historyczne "
            "wyniki widoczne w newsach, kontekst branżowy.)_"
        )
    lines = [
        "## Previous snapshot",
        f"_Poprzedni raport: {prev.generated_at.strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "### Fundamentals z poprzedniego raportu",
        "| Ticker | Cena | Mcap | P/E | EPS | Stopa dyw. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for f in prev.fundamentals:
        price = f"{f.last_price:.2f}" if f.last_price is not None else "—"
        mcap = f"{f.market_cap:,.0f}" if f.market_cap is not None else "—"
        pe = f"{f.pe_ratio:.2f}" if f.pe_ratio is not None else "—"
        eps = f"{f.eps:.2f}" if f.eps is not None else "—"
        dy = f"{f.dividend_yield * 100:.2f}%" if f.dividend_yield is not None else "—"
        lines.append(
            f"| {f.ticker} | {price} | {mcap} | {pe} | {eps} | {dy} |"
        )

    # Previous LLM-generated narrative (the actual prior report content).
    if prev.report:
        lines.append("")
        lines.append("### Treść poprzedniego raportu (do bazowania)")
        lines.append(
            "_Gdy nie ma świeżych danych, OPRZYJ się na poniższych "
            "narracjach z poprzedniego raportu — przekopiuj je do nowego "
            "raportu z minimalnymi aktualizacjami. Zaktualizuj tylko gdy "
            "pojawiły się nowe news/fundamentals._"
        )
        for co in prev.report.get("companies", []):
            lines.append("")
            lines.append(f"#### {co.get('ticker', '?')} — {co.get('name', '?')}")
            lines.append(f"- headline: {co.get('headline', '')}")
            lines.append(f"- last_reading_label: {co.get('last_reading_label', '')}")
            lines.append(f"- vs_expectations: {co.get('vs_expectations', '')}")
            lines.append(f"- next_report_label: {co.get('next_report_label', '')}")
            lines.append(f"- key_question: {co.get('key_question', '')}")
            lines.append(f"- thesis_status: {co.get('thesis_status', '')}")
            lines.append(f"- signal: {co.get('signal', '')} / {co.get('signal_title', '')}")
            lines.append(f"- recommendation: {co.get('recommendation', '')}")
            lines.append("")
            lines.append("**last_results_summary (poprzedni):**")
            lines.append(co.get("last_results_summary", ""))
            lines.append("")
            lines.append("**next_catalyst_focus (poprzedni):**")
            lines.append(co.get("next_catalyst_focus", ""))
        if prev.report.get("calendar"):
            lines.append("")
            lines.append("**Poprzedni kalendarz katalizatorów:**")
            for c in prev.report["calendar"]:
                t = c.get("ticker") or "—"
                lines.append(
                    f"- {c.get('date_label', '')} [{t}] "
                    f"{c.get('title', '')}: {c.get('description', '')}"
                )
    return "\n".join(lines)


def build_monitoring_context(
    portfolio: Portfolio,
    status: PortfolioStatus,
    *,
    fundamentals: Sequence[FundamentalsSnapshot] = (),
    news: Sequence[NewsItem] = (),
    previous_snapshot: MonitoringSnapshot | None = None,
) -> str:
    """Assemble the full prompt context for the monitoring report.

    Skips :func:`render_holdings_table` (truncated theses) since we render
    the full theses below — saves ~1 KB of redundant prompt tokens.
    """
    return "\n\n".join(
        [
            render_full_theses(portfolio),
            render_status(status),
            render_fundamentals(fundamentals),
            render_news_with_espi_flag(news),
            render_previous_snapshot(previous_snapshot),
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
