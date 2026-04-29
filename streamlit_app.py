"""Streamlit GUI for Investment Copilot.

Run with:
    streamlit run streamlit_app.py

Pure addition over the existing architecture. Calls the same orchestrator
pipelines as the CLI; never bypasses services.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from investment_copilot import __version__
from investment_copilot.config import ConfigError, load_config
from investment_copilot.domain.backtest import BacktestError, BacktestResult
from investment_copilot.domain.portfolio import Portfolio, PortfolioStatus
from investment_copilot.domain.prompts import PortfolioAnalysis, RiskAlerts
from investment_copilot.domain.strategies import KNOWN_STRATEGIES
from investment_copilot.gui import (
    drawdown_series,
    equity_curves_dataframe,
    format_money,
    format_money_signed,
    format_pct,
    holdings_dataframe,
    list_reports,
)
from investment_copilot.infrastructure.llm import LLMError
from investment_copilot.infrastructure.logging import configure_logging
from investment_copilot.orchestrator import Orchestrator
from investment_copilot.services import (
    PortfolioError,
    ServiceContainer,
    build_container,
    load_portfolio,
)

logger = logging.getLogger(__name__)

# --- Page config -----------------------------------------------------------

st.set_page_config(
    page_title="Investment Copilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Bootstrap (cached) ----------------------------------------------------


@st.cache_resource(show_spinner=False)
def _bootstrap(config_path: str, portfolio_path: str | None) -> tuple[
    ServiceContainer, Portfolio, Orchestrator
]:
    """Load config, build the container, load portfolio. Cached per (config, portfolio) pair."""
    cfg = load_config(config_path)
    configure_logging(cfg.logging.level)
    container = build_container(cfg)
    pf_path = portfolio_path or str(cfg.portfolio.path)
    portfolio = load_portfolio(pf_path)
    orchestrator = Orchestrator(container)
    return container, portfolio, orchestrator


def _safe_bootstrap(config_path: str, portfolio_path: str | None):
    try:
        return _bootstrap(config_path, portfolio_path)
    except ConfigError as exc:
        st.error(f"❌ Config error: {exc}")
        st.stop()
    except PortfolioError as exc:
        st.error(f"❌ Portfolio error: {exc}")
        st.stop()
    except Exception as exc:  # pragma: no cover
        st.error(f"❌ Failed to bootstrap: {exc}")
        st.stop()


# --- Sidebar ---------------------------------------------------------------


def _sidebar() -> tuple[str, str | None]:
    st.sidebar.title("📈 Investment Copilot")
    st.sidebar.caption(f"v{__version__}")

    config_path = st.sidebar.text_input(
        "Config path",
        value=os.environ.get("COPILOT_CONFIG", "config.yaml"),
        help="Path to config.yaml",
    )
    portfolio_override = st.sidebar.text_input(
        "Portfolio path (optional)",
        value=os.environ.get("COPILOT_PORTFOLIO", ""),
        help="Override the portfolio path from config.yaml. Leave empty to use the default.",
    )
    portfolio_override = portfolio_override.strip() or None

    if st.sidebar.button("🔄 Reload config", use_container_width=True):
        _bootstrap.clear()
        st.session_state.clear()
        st.rerun()

    return config_path, portfolio_override


# --- Session helpers -------------------------------------------------------


def _get_state(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def _set_state(key: str, value: Any) -> None:
    st.session_state[key] = value


# --- View: Portfolio -------------------------------------------------------


def _view_portfolio(portfolio: Portfolio, orchestrator: Orchestrator) -> None:
    st.header("Portfolio")

    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("🔄 Update data", use_container_width=True):
            with st.spinner("Refreshing OHLCV + news…"):
                try:
                    report = orchestrator.update_data(portfolio)
                    _set_state("last_refresh_report", report)
                    _set_state("last_status", None)  # invalidate cached status
                    st.success(
                        f"OHLCV updated: {len(report.ohlcv_updated)} · "
                        f"News inserted: {report.news_inserted} · "
                        f"Benchmark: {report.benchmark_symbol or '—'}"
                    )
                except Exception as exc:
                    st.error(f"Refresh failed: {exc}")

    with cols[1]:
        if st.button("📊 Compute status", use_container_width=True):
            _set_state("last_status", None)  # force recompute below

    # Compute (or reuse) current status
    status = _get_state("last_status")
    if status is None:
        with st.spinner("Computing status…"):
            status = orchestrator._container.portfolio_service.current_status(portfolio)
        _set_state("last_status", status)

    _render_status_panel(portfolio, status)
    _render_refresh_warnings()


def _render_status_panel(portfolio: Portfolio, status: PortfolioStatus) -> None:
    base = status.base_currency
    pnl_color = "green" if status.total_unrealized_pnl >= 0 else "red"

    cols = st.columns(4)
    cols[0].metric(
        "Market value",
        format_money(status.total_market_value, currency=base),
    )
    cols[1].metric(
        "Cost basis (priced)",
        format_money(status.priced_cost_basis, currency=base),
    )
    cols[2].metric(
        "Unrealized PnL",
        format_money_signed(status.total_unrealized_pnl, currency=base),
        format_pct(status.total_unrealized_pnl_pct),
    )
    cols[3].metric(
        "Holdings",
        f"{len(status.holdings)} ({len(status.missing_data)} missing data)",
    )

    if status.missing_data:
        st.warning(
            "Missing market data for: " + ", ".join(f"`{t}`" for t in status.missing_data)
            + "\n\nRun **Update data** to fetch."
        )

    df = holdings_dataframe(portfolio, status)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Shares": st.column_config.NumberColumn(format="%.0f"),
            "Entry": st.column_config.NumberColumn(format="%.2f"),
            "Last": st.column_config.NumberColumn(format="%.2f"),
            "Last date": st.column_config.DateColumn(),
            "Cost basis": st.column_config.NumberColumn(format="%.2f"),
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "PnL": st.column_config.NumberColumn(format="%+.2f"),
            "PnL%": st.column_config.NumberColumn(format="%+.2%"),
        },
    )

    # Investment theses (collapsed by default)
    with st.expander("Investment theses", expanded=False):
        for h in portfolio.holdings:
            label = f"`{h.ticker}`" + (f" — {h.name}" if h.name else "")
            st.markdown(f"**{label}**")
            st.markdown(h.thesis)
            st.divider()


def _render_refresh_warnings() -> None:
    report = _get_state("last_refresh_report")
    if report is None:
        return
    if report.ohlcv_failed:
        st.warning("OHLCV failed for: " + ", ".join(report.ohlcv_failed.keys()))
    if report.news_failed:
        for line in report.news_failed:
            st.warning(line)


# --- View: Backtest --------------------------------------------------------


def _view_backtest(portfolio: Portfolio, orchestrator: Orchestrator) -> None:
    st.header("Backtest")

    cols = st.columns([2, 1, 1])
    strategy = cols[0].selectbox(
        "Strategy", options=list(KNOWN_STRATEGIES), index=0
    )
    include_bm = cols[1].checkbox("Include benchmark", value=True)
    run = cols[2].button("▶ Run backtest", use_container_width=True)

    if run:
        with st.spinner(f"Running {strategy}…"):
            try:
                result = orchestrator.backtest(
                    portfolio,
                    strategy_name=strategy,
                    include_benchmark=include_bm,
                )
                _set_state("last_backtest", result)
            except BacktestError as exc:
                st.error(f"Backtest failed: {exc}")
                return

    result: BacktestResult | None = _get_state("last_backtest")
    if result is None:
        st.info(
            "Click **Run backtest** to start. "
            "Make sure data is fresh — use the Portfolio tab's **Update data** first."
        )
        return

    _render_backtest_panel(result)


def _render_backtest_panel(result: BacktestResult) -> None:
    m = result.metrics
    bm = result.benchmark_metrics

    # Summary line
    st.markdown(
        f"**Strategy:** `{result.strategy_name}` &nbsp;·&nbsp; "
        f"params: `{result.strategy_params}` &nbsp;·&nbsp; "
        f"window: {result.start_date} → {result.end_date}"
    )

    # Key metrics
    cols = st.columns(5)
    cols[0].metric("Final value", format_money(result.final_value, currency=""))
    cols[1].metric("Total return", format_pct(m.total_return))
    cols[2].metric("Annualized", format_pct(m.annualized_return))
    cols[3].metric("Sharpe", f"{m.sharpe_ratio:.2f}")
    cols[4].metric("Max drawdown", format_pct(m.max_drawdown))

    if result.missing_tickers:
        st.warning(
            "Missing in cache: " + ", ".join(f"`{t}`" for t in result.missing_tickers)
        )

    # Equity curve chart
    df = equity_curves_dataframe(result)
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], mode="lines", name=col)
        )
    fig.update_layout(
        title="Equity curve",
        xaxis_title="Date",
        yaxis_title=f"Equity ({result.initial_capital:,.0f} initial)",
        height=420,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown
    if "Strategy" in df.columns:
        dd = drawdown_series(df["Strategy"])
        dd_fig = go.Figure()
        dd_fig.add_trace(
            go.Scatter(
                x=dd.index, y=dd * 100,
                fill="tozeroy", mode="lines",
                line=dict(color="#d62728"),
                name="Drawdown",
            )
        )
        dd_fig.update_layout(
            title="Drawdown (%)",
            xaxis_title="Date",
            yaxis_title="%",
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(dd_fig, use_container_width=True)

    # Detailed metrics table
    with st.expander("Detailed metrics", expanded=False):
        rows = [
            ("Total return", m.total_return, getattr(bm, "total_return", None), True),
            ("Annualized return", m.annualized_return, getattr(bm, "annualized_return", None), True),
            ("Annualized volatility", m.annualized_volatility, getattr(bm, "annualized_volatility", None), True),
            ("Sharpe", m.sharpe_ratio, getattr(bm, "sharpe_ratio", None), False),
            ("Max drawdown", m.max_drawdown, getattr(bm, "max_drawdown", None), True),
            ("Max DD duration (days)", m.max_drawdown_duration_days, None, False),
            ("Win rate", m.win_rate, getattr(bm, "win_rate", None), True),
            ("Observations", m.n_observations, getattr(bm, "n_observations", None), False),
        ]

        def _fmt(v, is_pct):
            if v is None:
                return "—"
            if is_pct:
                return format_pct(v)
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)

        table = pd.DataFrame(
            [
                {
                    "Metric": label,
                    "Strategy": _fmt(s, p),
                    f"Benchmark ({result.benchmark_symbol or '—'})": _fmt(b, p),
                }
                for label, s, b, p in rows
            ]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)


# --- View: AI Analysis -----------------------------------------------------


def _view_analysis(portfolio: Portfolio, orchestrator: Orchestrator) -> None:
    st.header("AI Analysis")
    st.caption(
        "Polish-language analyses powered by Groq. "
        "Make sure `GROQ_API_KEY` is set and outbound network is available."
    )

    cols = st.columns([1, 1, 1, 2])
    include_risks = cols[0].checkbox("Include risk alerts", value=True)
    news_days = cols[1].number_input(
        "News window (days)", min_value=0, max_value=120, value=14, step=1,
        key="analysis_news_days",
    )
    use_backtest = cols[2].checkbox(
        "Include last backtest as context",
        value=_get_state("last_backtest") is not None,
        disabled=_get_state("last_backtest") is None,
    )
    run = cols[3].button("✨ Run analysis", use_container_width=True)

    if run:
        backtest_for_context = _get_state("last_backtest") if use_backtest else None
        with st.spinner("Asking Groq…"):
            bundle = orchestrator.run_analysis(
                portfolio,
                include_risks=include_risks,
                backtest_for_context=backtest_for_context,
                news_days_back=int(news_days),
            )
            _set_state("last_analysis_bundle", bundle)

    bundle = _get_state("last_analysis_bundle")
    if bundle is None:
        st.info("Click **Run analysis** to start.")
        return

    if bundle.warnings:
        for w in bundle.warnings:
            st.warning(w)

    if bundle.analysis is not None:
        _render_analysis_panel(bundle.analysis)
    if bundle.risks is not None:
        _render_risks_panel(bundle.risks)


def _render_analysis_panel(a: PortfolioAnalysis) -> None:
    st.subheader("📋 Analiza portfela")
    st.markdown(a.summary)

    st.markdown("**Komentarze do pozycji:**")
    for c in a.holdings_comments:
        st.markdown(
            f"- `{c.ticker}` _(rekomendacja: **{c.recommendation}**)_  \n"
            f"  {c.comment}"
        )

    st.markdown("**Dywersyfikacja:**")
    st.markdown(a.diversification_notes)

    st.caption(f"Pewność analizy: **{a.confidence}/10**")


_SEVERITY_BADGE = {
    "wysokie": "🔴",
    "średnie": "🟡",
    "niskie": "🟢",
}


def _render_risks_panel(r: RiskAlerts) -> None:
    st.subheader("⚠️ Ryzyka")
    st.markdown(r.overview)

    if not r.alerts:
        st.info("(brak istotnych ryzyk)")
        return

    for alert in r.alerts:
        marker = _SEVERITY_BADGE.get(alert.severity, "•")
        target = f"`{alert.ticker}`" if alert.ticker else "_(portfelowe)_"
        with st.expander(
            f"{marker} **{alert.title}** — {target} · _{alert.severity}_",
            expanded=(alert.severity == "wysokie"),
        ):
            st.markdown(alert.description)
            st.markdown(f"**Sugerowane działanie:** {alert.suggested_action}")


# --- View: Reports ---------------------------------------------------------


def _view_reports(
    container: ServiceContainer,
    portfolio: Portfolio,
    orchestrator: Orchestrator,
) -> None:
    st.header("Reports")

    cols = st.columns([2, 1, 1, 2])
    strategy_choice = cols[0].selectbox(
        "Strategy for the embedded backtest",
        options=["(none)", *KNOWN_STRATEGIES],
        index=1,
    )
    news_days = cols[1].number_input(
        "News window (days)", min_value=0, max_value=120, value=14, step=1,
        key="reports_news_days",
    )
    filename = cols[2].text_input("Filename (optional)", value="")
    generate = cols[3].button("📄 Generate report", use_container_width=True)

    if generate:
        with st.spinner("Generating Markdown report…"):
            try:
                full = orchestrator.generate_report(
                    portfolio,
                    strategy_name=(
                        strategy_choice if strategy_choice != "(none)" else None
                    ),
                    news_days_back=int(news_days),
                    filename=filename or None,
                )
            except (LLMError, OSError) as exc:
                st.error(f"Report generation failed: {exc}")
                return
            for w in full.warnings:
                st.warning(w)
            st.success(f"Report written: `{full.report_path}`")

    st.subheader("Existing reports")
    reports_dir = Path("reports")
    files = list_reports(reports_dir)
    if not files:
        st.info(f"No reports yet in `{reports_dir.resolve()}`. Generate one above.")
        return

    selected = st.selectbox(
        "Select a report",
        options=files,
        format_func=lambda p: f"{p.name} ({_human_size(p.stat().st_size)}, "
        f"{datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M})",
    )
    if selected is not None:
        body = selected.read_text(encoding="utf-8")
        col_a, col_b = st.columns([3, 1])
        with col_b:
            st.download_button(
                "⬇️ Download",
                data=body,
                file_name=selected.name,
                mime="text/markdown",
                use_container_width=True,
            )
        with col_a:
            st.caption(f"Path: `{selected}`")
        st.markdown(body)


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.0f} GB"


# --- Main ------------------------------------------------------------------


def main() -> None:
    config_path, portfolio_override = _sidebar()
    container, portfolio, orchestrator = _safe_bootstrap(config_path, portfolio_override)

    st.sidebar.divider()
    st.sidebar.markdown("**Loaded:**")
    st.sidebar.caption(f"Portfolio: {len(portfolio.holdings)} holding(s)")
    st.sidebar.caption(f"Base currency: {portfolio.base_currency}")
    st.sidebar.caption(f"Benchmark: {container.config.backtest.benchmark}")
    st.sidebar.caption(f"LLM: {container.config.llm.provider} ({container.config.llm.model_analysis})")

    tabs = st.tabs(["📊 Portfolio", "📈 Backtest", "✨ AI Analysis", "📄 Reports"])
    with tabs[0]:
        _view_portfolio(portfolio, orchestrator)
    with tabs[1]:
        _view_backtest(portfolio, orchestrator)
    with tabs[2]:
        _view_analysis(portfolio, orchestrator)
    with tabs[3]:
        _view_reports(container, portfolio, orchestrator)

    st.sidebar.divider()
    st.sidebar.caption(
        "Investment Copilot — research material, not financial advice."
    )


if __name__ == "__main__":
    main()
