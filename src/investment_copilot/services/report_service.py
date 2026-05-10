"""Report service.

Renders Markdown reports from the typed outputs of other services. The
rendering itself is a set of pure functions; the service layer is a thin
filesystem adapter on top.

Reports are written to a configurable directory; the service guarantees
the directory exists. The filename includes a timestamp so reports never
overwrite one another.
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from pathlib import Path

from investment_copilot.domain.backtest import BacktestResult
from investment_copilot.domain.portfolio import Portfolio, PortfolioStatus
from investment_copilot.domain.prompts import (
    MonitoringCalendarEntry,
    MonitoringCompany,
    MonitoringMetric,
    MonitoringReport,
    PortfolioAnalysis,
    RiskAlerts,
)

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

    def render_monitoring_html(
        self,
        report: MonitoringReport,
        *,
        generated_at: datetime,
        portfolio: Portfolio,
        had_previous_snapshot: bool,
    ) -> str:
        """Render the monitoring report as a self-contained HTML document."""
        return _render_monitoring_html(
            report,
            generated_at=generated_at,
            portfolio=portfolio,
            had_previous_snapshot=had_previous_snapshot,
        )

    def write_monitoring(
        self,
        report: MonitoringReport,
        *,
        generated_at: datetime,
        portfolio: Portfolio,
        had_previous_snapshot: bool,
        output_dir: Path | str | None = None,
        filename: str | None = None,
    ) -> Path:
        """Render and write the monitoring report HTML to disk."""
        out_dir = Path(output_dir) if output_dir else self.output_dir / "monitoring"
        out_dir.mkdir(parents=True, exist_ok=True)
        body = self.render_monitoring_html(
            report,
            generated_at=generated_at,
            portfolio=portfolio,
            had_previous_snapshot=had_previous_snapshot,
        )
        ts = generated_at.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = filename or f"monitoring_{ts}.html"
        path = out_dir / name
        path.write_text(body, encoding="utf-8")
        logger.info("Monitoring report written: %s", path)
        return path

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


# --- Monitoring HTML rendering ---------------------------------------------


_MONITORING_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: linear-gradient(160deg,#050810 0%,#080d1a 60%,#050810 100%); font-family:'Courier New',monospace; color:#c8d0e0; min-height:100vh; font-size:13px; }
.page-header { padding:24px 28px; border-bottom:1px solid rgba(255,255,255,0.07); background:rgba(8,12,24,0.95); }
.page-title { font-family:Georgia,serif; font-size:22px; font-weight:900; color:#f1f5f9; letter-spacing:-.02em; margin-bottom:6px; }
.page-sub { font-size:10px; color:#475569; letter-spacing:.12em; }
.date-badge { display:inline-block; padding:3px 10px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.09); border-radius:4px; font-size:9px; color:#64748b; margin-top:8px; }
.summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; padding:20px 28px; background:rgba(6,10,20,0.6); border-bottom:1px solid rgba(255,255,255,0.06); }
.sum-card { border-radius:8px; padding:14px 16px; }
.sum-ticker { font-size:11px; font-weight:700; letter-spacing:.1em; margin-bottom:4px; }
.sum-score { font-size:22px; font-weight:900; margin:6px 0; }
.sum-label { font-size:9px; color:#475569; letter-spacing:.09em; }
.sum-tag { display:inline-block; padding:2px 8px; border-radius:3px; font-size:8.5px; font-weight:700; letter-spacing:.1em; margin-top:6px; }
.main { max-width:1100px; margin:0 auto; padding:28px; }
.synthesis-box { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:10px; padding:18px 22px; margin-bottom:32px; }
.synthesis-label { color:#94a3b8; font-size:9px; font-weight:700; letter-spacing:.14em; margin-bottom:10px; }
.synthesis-body { font-size:12px; line-height:1.9; color:#e2e8f0; }
.company-section { margin-bottom:48px; }
.co-header { display:flex; align-items:center; gap:14px; margin-bottom:18px; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.06); flex-wrap:wrap; }
.co-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.co-block { flex:1; min-width:200px; }
.co-name { font-size:16px; font-weight:900; font-family:Georgia,serif; }
.co-ticker { font-size:11px; color:#475569; }
.co-badge { padding:3px 10px; border-radius:4px; font-size:9px; font-weight:700; letter-spacing:.1em; }
.metrics-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin-bottom:16px; }
.metric-box { border-radius:7px; padding:12px 14px; }
.metric-label { font-size:9px; color:#475569; letter-spacing:.1em; margin-bottom:4px; }
.metric-val { font-size:17px; font-weight:700; color:#f1f5f9; }
.metric-sub { font-size:10px; margin-top:3px; }
.pos { color:#34d399; } .neg { color:#f87171; } .warn { color:#fbbf24; } .neu { color:#94a3b8; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px; }
.analysis-box { border-radius:8px; padding:16px 18px; }
.box-title { font-size:9px; font-weight:700; letter-spacing:.12em; margin-bottom:9px; }
.box-body { font-size:11.5px; line-height:1.85; color:#7a8aaa; }
.box-body strong { color:#e2e8f0; }
.signal-bar { border-radius:7px; padding:14px 18px; margin-bottom:14px; display:flex; align-items:flex-start; gap:14px; }
.signal-icon { font-size:20px; flex-shrink:0; line-height:1.3; }
.signal-title { font-size:12px; font-weight:700; margin-bottom:4px; }
.signal-body { font-size:11px; line-height:1.75; color:#64748b; }
.signal-body strong { color:#cbd5e1; }
.change-row { margin-top:10px; padding:8px 12px; border-radius:6px; background:rgba(255,255,255,0.018); border:1px solid rgba(255,255,255,0.05); display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.change-tag { display:inline-block; padding:2px 8px; border-radius:3px; font-size:8.5px; font-weight:700; letter-spacing:.08em; }
.change-tag.akceleracja { background:rgba(16,185,129,0.12); color:#6ee7b7; border:1px solid rgba(16,185,129,0.3); }
.change-tag.rozczarowanie { background:rgba(239,68,68,0.12); color:#fca5a5; border:1px solid rgba(239,68,68,0.3); }
.change-tag.stabilizacja { background:rgba(245,158,11,0.12); color:#fde047; border:1px solid rgba(245,158,11,0.3); }
.change-tag.brak-zmian { background:rgba(148,163,184,0.12); color:#cbd5e1; border:1px solid rgba(148,163,184,0.25); }
.change-narrative { font-size:11px; color:#94a3b8; font-style:italic; flex:1; min-width:200px; }
.outlook-table { width:100%; border-collapse:collapse; font-size:11px; margin-bottom:28px; }
.outlook-table th { padding:10px 14px; text-align:left; font-size:9px; font-weight:700; letter-spacing:.09em; border-bottom:2px solid rgba(255,255,255,0.08); color:#64748b; }
.outlook-table td { padding:12px 14px; border-bottom:1px solid rgba(255,255,255,0.04); vertical-align:top; line-height:1.65; color:#7a8aaa; }
.outlook-table td:first-child { font-weight:700; }
.outlook-table tr:nth-child(even) td { background:rgba(255,255,255,0.014); }
.outlook-table .rec-pill { padding:2px 8px; border-radius:3px; font-size:9px; font-weight:700; }
.section-title { color:#94a3b8; font-size:9px; font-weight:700; letter-spacing:.14em; margin-bottom:14px; }
.calendar-box { background:rgba(255,255,255,0.015); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:16px 22px; margin-bottom:16px; }
.calendar-grid { display:grid; grid-template-columns:110px 1fr; gap:10px 16px; align-items:start; }
.cal-date { font-size:10px; font-weight:700; padding-top:2px; }
.cal-date.high { color:#f87171; }
.cal-date.medium { color:#fbbf24; }
.cal-date.low { color:#6b7a99; }
.cal-body { font-size:11.5px; color:#94a3b8; line-height:1.65; }
.cal-body strong { color:#e2e8f0; }
.cal-ticker { font-size:10px; color:#475569; font-weight:700; margin-right:6px; }
.footer { padding:14px 28px; border-top:1px solid rgba(255,255,255,0.05); }
.footer p { font-size:9px; color:#1e293b; line-height:1.6; }
@media(max-width:760px){ .two-col, .summary-grid, .metrics-row { grid-template-columns:1fr 1fr !important; } }
@media(max-width:480px){ .summary-grid, .metrics-row, .two-col { grid-template-columns:1fr !important; } }
"""


# Per-position color theme — cycled by index. Each tuple defines the
# accent color (hex), translucent background, and translucent border in
# the same family. Inspired by the user-provided template.
_PALETTE: tuple[tuple[str, str, str, str], ...] = (
    ("#22d3ee", "rgba(6,182,212,0.07)", "rgba(6,182,212,0.2)", "rgba(6,182,212,0.05)"),
    ("#fbbf24", "rgba(245,158,11,0.07)", "rgba(245,158,11,0.2)", "rgba(245,158,11,0.05)"),
    ("#a855f7", "rgba(168,85,247,0.07)", "rgba(168,85,247,0.2)", "rgba(168,85,247,0.05)"),
    ("#34d399", "rgba(16,185,129,0.07)", "rgba(16,185,129,0.2)", "rgba(16,185,129,0.05)"),
    ("#60a5fa", "rgba(96,165,250,0.07)", "rgba(96,165,250,0.2)", "rgba(96,165,250,0.05)"),
    ("#f472b6", "rgba(244,114,182,0.07)", "rgba(244,114,182,0.2)", "rgba(244,114,182,0.05)"),
)


_TONE_TO_CLASS = {
    "positive": "pos",
    "negative": "neg",
    "warning": "warn",
    "neutral": "neu",
}


_SIGNAL_TO_ICON = {
    "bullish": "🟢",
    "bearish": "🔴",
    "neutral": "🟡",
}


_SIGNAL_TO_BG = {
    "bullish": ("rgba(16,185,129,0.04)", "rgba(16,185,129,0.2)", "#34d399"),
    "bearish": ("rgba(239,68,68,0.04)", "rgba(239,68,68,0.2)", "#f87171"),
    "neutral": ("rgba(245,158,11,0.04)", "rgba(245,158,11,0.2)", "#fbbf24"),
}


_REC_PILL_COLORS = {
    "trzymaj": ("rgba(96,165,250,0.12)", "#60a5fa"),
    "zwiększ": ("rgba(16,185,129,0.12)", "#34d399"),
    "zmniejsz": ("rgba(245,158,11,0.12)", "#fbbf24"),
    "obserwuj": ("rgba(245,158,11,0.12)", "#fbbf24"),
    "zamknij": ("rgba(239,68,68,0.12)", "#f87171"),
}


def _esc(s: str | None) -> str:
    if s is None:
        return ""
    return html.escape(str(s), quote=True)


def _palette_for(idx: int) -> tuple[str, str, str, str]:
    return _PALETTE[idx % len(_PALETTE)]


def _render_metric(m: MonitoringMetric, bg: str) -> str:
    cls = _TONE_TO_CLASS.get(m.tone, "neu")
    detail = (
        f'<div class="metric-sub {cls}">{_esc(m.detail)}</div>' if m.detail else ""
    )
    return (
        f'<div class="metric-box" style="background:{bg};border:1px solid rgba(255,255,255,0.08)">'
        f'<div class="metric-label">{_esc(m.label)}</div>'
        f'<div class="metric-val">{_esc(m.value)}</div>'
        f"{detail}"
        "</div>"
    )


def _render_change_row(co: MonitoringCompany) -> str:
    if not (co.change_direction or co.change_narrative):
        return ""
    parts: list[str] = []
    if co.change_direction:
        cls = co.change_direction.replace(" ", "-")
        parts.append(
            f'<span class="change-tag {cls}">{_esc(co.change_direction.upper())}</span>'
        )
    if co.change_narrative:
        parts.append(f'<span class="change-narrative">{_esc(co.change_narrative)}</span>')
    return f'<div class="change-row">{"".join(parts)}</div>'


def _render_company(co: MonitoringCompany, idx: int) -> str:
    accent, bg_card, border_card, bg_metric = _palette_for(idx)
    metrics_html = "".join(_render_metric(m, bg_metric) for m in co.metrics)
    sig = co.signal if co.signal in _SIGNAL_TO_BG else "neutral"
    sig_bg, sig_border, sig_color = _SIGNAL_TO_BG[sig]
    icon = _SIGNAL_TO_ICON.get(sig, "🟡")

    return (
        '<div class="company-section">'
        '<div class="co-header">'
        f'<div class="co-dot" style="background:{accent}"></div>'
        '<div class="co-block">'
        f'<div class="co-name" style="color:{accent}">{_esc(co.name)}</div>'
        f'<div class="co-ticker">GPW: {_esc(co.ticker.split(".")[0].upper())} · '
        f'{_esc(co.ticker)}</div>'
        "</div>"
        f'<div class="co-badge" style="margin-left:auto;background:{bg_card};'
        f'border:1px solid {border_card};color:{accent}">'
        f'{_esc(co.headline)}</div>'
        "</div>"
        f'<div class="metrics-row">{metrics_html}</div>'
        '<div class="two-col">'
        f'<div class="analysis-box" style="background:{bg_metric};'
        f'border:1px solid {border_card}">'
        f'<div class="box-title" style="color:{accent}">📊 CO MÓWI RAPORT</div>'
        f'<div class="box-body">{_esc(co.last_results_summary)}</div>'
        "</div>"
        '<div class="analysis-box" style="background:rgba(255,255,255,0.02);'
        'border:1px solid rgba(255,255,255,0.07)">'
        '<div class="box-title" style="color:#94a3b8">🎯 NA CO CZEKAMY</div>'
        f'<div class="box-body">{_esc(co.next_catalyst_focus)}</div>'
        "</div>"
        "</div>"
        f'<div class="signal-bar" style="background:{sig_bg};border:1px solid {sig_border}">'
        f'<div class="signal-icon">{icon}</div>'
        "<div>"
        f'<div class="signal-title" style="color:{sig_color}">{_esc(co.signal_title)}</div>'
        f'<div class="signal-body">{_esc(co.signal_body)}</div>'
        "</div>"
        "</div>"
        f"{_render_change_row(co)}"
        "</div>"
    )


def _render_summary_grid(companies: list[MonitoringCompany]) -> str:
    cards: list[str] = []
    for idx, co in enumerate(companies):
        accent, bg_card, border_card, _ = _palette_for(idx)
        # Headline metric — first one, per prompt contract.
        headline_metric = co.metrics[0] if co.metrics else None
        if headline_metric:
            score = headline_metric.value
            score_label = headline_metric.label
            score_color = {
                "positive": "#34d399",
                "negative": "#f87171",
                "warning": "#fbbf24",
            }.get(headline_metric.tone, accent)
        else:
            score = co.recommendation.upper()
            score_label = "REKOMENDACJA"
            score_color = accent

        ticker_short = co.ticker.split(".")[0].upper()
        cards.append(
            f'<div class="sum-card" style="background:{bg_card};border:1px solid {border_card}">'
            f'<div class="sum-ticker" style="color:{accent}">{_esc(ticker_short)} · {_esc(co.name)}</div>'
            f'<div class="sum-score" style="color:{score_color}">{_esc(score)}</div>'
            f'<div class="sum-label">{_esc(score_label)}</div>'
            f'<div class="sum-tag" style="background:{bg_card};color:{accent};border:1px solid {border_card}">'
            f'{_esc(co.next_report_label)}</div>'
            "</div>"
        )
    return f'<div class="summary-grid">{"".join(cards)}</div>'


def _render_outlook_table(companies: list[MonitoringCompany]) -> str:
    rows: list[str] = []
    for idx, co in enumerate(companies):
        accent, _, _, _ = _palette_for(idx)
        rec = co.recommendation
        rec_bg, rec_color = _REC_PILL_COLORS.get(rec, ("rgba(148,163,184,0.12)", "#cbd5e1"))
        ticker_short = co.ticker.split(".")[0].upper()
        rows.append(
            "<tr>"
            f'<td style="color:{accent}">{_esc(ticker_short)}</td>'
            f"<td>{_esc(co.last_reading_label)}</td>"
            f"<td>{_esc(co.vs_expectations)}</td>"
            f"<td>{_esc(co.next_report_label)}</td>"
            f"<td>{_esc(co.key_question)}</td>"
            f"<td>{_esc(co.thesis_status.upper())}</td>"
            f'<td><span class="rec-pill" style="background:{rec_bg};color:{rec_color}">'
            f'{_esc(rec.upper())}</span></td>'
            "</tr>"
        )
    return (
        '<div style="margin-bottom:28px">'
        '<div class="section-title">📊 ZBIORCZA OCENA PORTFELA</div>'
        '<div style="overflow-x:auto"><table class="outlook-table">'
        "<thead><tr>"
        "<th>SPÓŁKA</th><th>OSTATNI ODCZYT</th><th>vs OCZEKIWANIA</th>"
        "<th>NASTĘPNY RAPORT</th><th>KLUCZOWE PYTANIE</th>"
        "<th>STATUS TEZY</th><th>REKOMENDACJA</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div></div>"
    )


def _render_calendar(entries: list[MonitoringCalendarEntry]) -> str:
    if not entries:
        return ""
    rows: list[str] = []
    for e in entries:
        importance = e.importance if e.importance in {"high", "medium", "low"} else "medium"
        ticker_html = (
            f'<span class="cal-ticker">[{_esc(e.ticker)}]</span>'
            if e.ticker else ""
        )
        rows.append(
            f'<span class="cal-date {importance}">{_esc(e.date_label)}</span>'
            f'<span class="cal-body">{ticker_html}'
            f"<strong>{_esc(e.title)}</strong> — {_esc(e.description)}</span>"
        )
    return (
        '<div class="calendar-box">'
        '<div class="section-title">📅 KALENDARZ KATALIZATORÓW — NAJBLIŻSZE TYGODNIE</div>'
        f'<div class="calendar-grid">{"".join(rows)}</div>'
        "</div>"
    )


def _format_date_pl(d: datetime) -> str:
    months = [
        "STYCZNIA", "LUTEGO", "MARCA", "KWIETNIA", "MAJA", "CZERWCA",
        "LIPCA", "SIERPNIA", "WRZEŚNIA", "PAŹDZIERNIKA", "LISTOPADA", "GRUDNIA",
    ]
    return f"{d.day} {months[d.month - 1]} {d.year}"


def _render_monitoring_html(
    report: MonitoringReport,
    *,
    generated_at: datetime,
    portfolio: Portfolio,
    had_previous_snapshot: bool,
) -> str:
    ts_utc = generated_at.astimezone(timezone.utc)
    diff_note = (
        "Z PORÓWNANIEM DO POPRZEDNIEGO RAPORTU"
        if had_previous_snapshot
        else "PIERWSZY RAPORT — BRAK PORÓWNANIA"
    )
    sub = report.subtitle or "BUY-SIDE EQUITY REVIEW · PORTFEL GPW"
    n_pos = len(portfolio.holdings)
    date_pl = _format_date_pl(ts_utc).upper()

    companies_html = "".join(
        _render_company(co, idx) for idx, co in enumerate(report.companies)
    )

    return (
        "<!DOCTYPE html>\n"
        '<html lang="pl"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(report.title)}</title>"
        f"<style>{_MONITORING_CSS}</style>"
        "</head><body>"
        # Page header
        '<div class="page-header">'
        f'<div class="page-title">📋 {_esc(report.title)}</div>'
        f'<div class="page-sub">{_esc(sub)} &nbsp;·&nbsp; {n_pos} POZYCJI &nbsp;·&nbsp; {diff_note}</div>'
        f'<div class="date-badge">DATA ANALIZY: {_esc(date_pl)} &nbsp;|&nbsp; '
        f"PEWNOŚĆ: {report.confidence}/10 &nbsp;|&nbsp; "
        f"WYGENEROWANO: {ts_utc.strftime('%H:%M')} UTC</div>"
        "</div>"
        # Summary grid
        f"{_render_summary_grid(report.companies)}"
        # Main content
        '<div class="main">'
        # Synthesis
        '<div class="synthesis-box">'
        '<div class="synthesis-label">💭 SYNTEZA — CO MÓWIĄ NAJNOWSZE DANE</div>'
        f'<div class="synthesis-body">{_esc(report.synthesis)}</div>'
        "</div>"
        # Per-company sections
        f"{companies_html}"
        # Outlook table
        f"{_render_outlook_table(report.companies)}"
        # Calendar
        f"{_render_calendar(report.calendar)}"
        "</div>"
        # Footer
        '<div class="footer"><p>'
        "ZASTRZEŻENIE: Analiza wyłącznie w celach edukacyjnych. Nie stanowi "
        "rekomendacji inwestycyjnej ani doradztwa finansowego. Dane: lokalny "
        "cache OHLCV (Stooq), komunikaty ESPI/EBI via Stooq, RSS portali "
        "finansowych, narracja LLM (Groq). Inwestowanie wiąże się z ryzykiem "
        "utraty kapitału."
        "</p></div>"
        "</body></html>"
    )
