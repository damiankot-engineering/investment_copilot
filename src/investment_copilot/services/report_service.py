"""Report service.

Renders Markdown reports from the typed outputs of other services. The
rendering itself is a set of pure functions; the service layer is a thin
filesystem adapter on top.

Reports are written to a configurable directory; the service guarantees
the directory exists. The filename includes a timestamp so reports never
overwrite one another.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from investment_copilot.domain.backtest import BacktestResult
from investment_copilot.domain.portfolio import Portfolio, PortfolioStatus
from investment_copilot.domain.prompts import PortfolioAnalysis, RiskAlerts

logger = logging.getLogger(__name__)


class ReportService:
    """Renders and persists Markdown reports."""

    def __init__(self, *, output_dir: Path | str = "reports") -> None:
        self.output_dir = Path(output_dir)

    # -- Public --------------------------------------------------------------

    def render(
        self,
        *,
        portfolio: Portfolio,
        status: PortfolioStatus,
        backtest: BacktestResult | None = None,
        analysis: PortfolioAnalysis | None = None,
        risks: RiskAlerts | None = None,
        warnings: list[str] | None = None,
    ) -> str:
        """Render the Markdown report as a string (no I/O)."""
        sections: list[str] = [
            _render_header(status),
            _render_portfolio_section(portfolio, status),
        ]
        if backtest is not None:
            sections.append(_render_backtest_section(backtest))
        if analysis is not None:
            sections.append(_render_analysis_section(analysis))
        if risks is not None:
            sections.append(_render_risks_section(risks))
        if warnings:
            sections.append(_render_warnings_section(warnings))
        sections.append(_render_footer())
        return "\n\n".join(sections) + "\n"

    def write(
        self,
        *,
        portfolio: Portfolio,
        status: PortfolioStatus,
        backtest: BacktestResult | None = None,
        analysis: PortfolioAnalysis | None = None,
        risks: RiskAlerts | None = None,
        warnings: list[str] | None = None,
        filename: str | None = None,
    ) -> Path:
        """Render and write the report to disk; return the path written."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        body = self.render(
            portfolio=portfolio,
            status=status,
            backtest=backtest,
            analysis=analysis,
            risks=risks,
            warnings=warnings,
        )
        name = filename or _default_filename(status.as_of)
        path = self.output_dir / name
        path.write_text(body, encoding="utf-8")
        logger.info("Report written: %s", path)
        return path


# --- Pure rendering helpers -------------------------------------------------


def _render_header(status: PortfolioStatus) -> str:
    return (
        f"# Raport portfela\n\n"
        f"_Wygenerowano: {status.as_of.strftime('%Y-%m-%d %H:%M UTC')}_  \n"
        f"_Waluta bazowa: {status.base_currency}_"
    )


def _render_portfolio_section(portfolio: Portfolio, status: PortfolioStatus) -> str:
    lines = ["## Portfel"]
    lines.append(
        f"Łączna wartość rynkowa: **{status.total_market_value:,.2f} "
        f"{status.base_currency}**  \n"
        f"Łączny koszt nabycia (wszystkie pozycje): "
        f"{status.total_cost_basis:,.2f}  \n"
        f"Niezrealizowany PnL (wyceniane): "
        f"{status.total_unrealized_pnl:+,.2f} "
        f"({status.total_unrealized_pnl_pct * 100:+.2f}%)"
    )
    if status.missing_data:
        lines.append(
            "\n> ⚠️  Brak danych rynkowych dla: "
            + ", ".join(f"`{t}`" for t in status.missing_data)
        )

    lines.append("")
    lines.append(
        "| Ticker | Nazwa | Akcje | Cena wejścia | Ostatnia | Wartość | PnL | PnL% |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    by_ticker = {h.ticker: h for h in portfolio.holdings}
    for s in status.holdings:
        h = by_ticker.get(s.ticker)
        name = (h.name if h and h.name else "—").replace("|", "/")
        if s.has_price:
            lines.append(
                f"| `{s.ticker}` | {name} | {s.shares:g} | "
                f"{s.entry_price:.2f} | {s.last_price:.2f} | "
                f"{s.market_value:,.2f} | {s.unrealized_pnl:+,.2f} | "
                f"{s.unrealized_pnl_pct * 100:+.2f}% |"
            )
        else:
            lines.append(
                f"| `{s.ticker}` | {name} | {s.shares:g} | "
                f"{s.entry_price:.2f} | — | — | — | — |"
            )
    return "\n".join(lines)


def _render_backtest_section(result: BacktestResult) -> str:
    m = result.metrics
    lines = [
        "## Backtest",
        f"**Strategia:** `{result.strategy_name}` "
        f"(parametry: {result.strategy_params})  \n"
        f"**Okno:** {result.start_date.isoformat()} → "
        f"{result.end_date.isoformat()}  \n"
        f"**Kapitał:** {result.initial_capital:,.0f} → "
        f"**{result.final_value:,.2f}**",
        "",
        "| Metryka | Strategia |"
        + (" Benchmark |" if result.benchmark_metrics else ""),
        "|---|---:|" + ("---:|" if result.benchmark_metrics else ""),
        _metric_row("Total return", m.total_return, result.benchmark_metrics, "total_return", pct=True),
        _metric_row("Annualized return", m.annualized_return, result.benchmark_metrics, "annualized_return", pct=True),
        _metric_row("Annualized volatility", m.annualized_volatility, result.benchmark_metrics, "annualized_volatility", pct=True),
        _metric_row("Sharpe", m.sharpe_ratio, result.benchmark_metrics, "sharpe_ratio"),
        _metric_row("Max drawdown", m.max_drawdown, result.benchmark_metrics, "max_drawdown", pct=True),
        _metric_row("Win rate", m.win_rate, result.benchmark_metrics, "win_rate", pct=True),
    ]
    if result.benchmark_symbol:
        lines.append(f"\n_Benchmark: `{result.benchmark_symbol}` (buy & hold)._")
    if result.missing_tickers:
        lines.append(
            "\n> ⚠️  Pominięto (brak danych): "
            + ", ".join(f"`{t}`" for t in result.missing_tickers)
        )
    return "\n".join(lines)


def _metric_row(label, strat_val, bench, attr, *, pct: bool = False) -> str:
    s = _fmt(strat_val, pct=pct)
    if bench is None:
        return f"| {label} | {s} |"
    b = _fmt(getattr(bench, attr), pct=pct)
    return f"| {label} | {s} | {b} |"


def _fmt(v: float | None, *, pct: bool) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:+.2f}%" if v != 0 else "0.00%"
    return f"{v:.2f}"


def _render_analysis_section(a: PortfolioAnalysis) -> str:
    lines = [
        "## Analiza (AI)",
        a.summary,
        "",
        "### Komentarze do pozycji",
    ]
    for c in a.holdings_comments:
        lines.append(f"- **`{c.ticker}`** _(rekomendacja: {c.recommendation})_  \n  {c.comment}")
    lines.append("")
    lines.append("### Dywersyfikacja")
    lines.append(a.diversification_notes)
    lines.append("")
    lines.append(f"_Pewność analizy: **{a.confidence}/10**._")
    return "\n".join(lines)


def _render_risks_section(r: RiskAlerts) -> str:
    lines = ["## Ryzyka (AI)", r.overview, ""]
    if not r.alerts:
        lines.append("_(brak istotnych ryzyk)_")
        return "\n".join(lines)

    severity_marker = {"niskie": "🟢", "średnie": "🟡", "wysokie": "🔴"}
    for alert in r.alerts:
        marker = severity_marker.get(alert.severity, "•")
        target = f"`{alert.ticker}`" if alert.ticker else "_(portfelowe)_"
        lines.append(
            f"### {marker} {alert.title}  \n"
            f"_Cel: {target} · Istotność: **{alert.severity}**_\n\n"
            f"{alert.description}\n\n"
            f"**Sugerowane działanie:** {alert.suggested_action}"
        )
    return "\n\n".join(lines)


def _render_warnings_section(warnings: list[str]) -> str:
    body = "\n".join(f"- {w}" for w in warnings)
    return f"## Ostrzeżenia\n{body}"


def _render_footer() -> str:
    return (
        "---\n"
        "_Raport wygenerowany przez Investment Copilot. "
        "Materiał decyzyjny — nie stanowi rekomendacji inwestycyjnej._"
    )


def _default_filename(as_of: datetime) -> str:
    ts = as_of.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"report_{ts}.md"
