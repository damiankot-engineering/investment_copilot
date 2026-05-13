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
    list_monitoring_reports,
    list_reports,
)
from investment_copilot.infrastructure.llm import LLMError
from investment_copilot.infrastructure.logging import configure_logging
from investment_copilot.orchestrator import Orchestrator
from investment_copilot.domain.portfolio import Holding
from investment_copilot.services import (
    PortfolioError,
    ServiceContainer,
    build_container,
    load_portfolio,
    save_portfolio,
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
    ServiceContainer, Portfolio, Orchestrator, str
]:
    """Load config, build the container, load portfolio. Cached per (config, portfolio) pair."""
    cfg = load_config(config_path)
    configure_logging(cfg.logging.level)
    container = build_container(cfg)
    pf_path = portfolio_path or str(cfg.portfolio.path)
    portfolio = load_portfolio(pf_path)
    orchestrator = Orchestrator(container)
    return container, portfolio, orchestrator, pf_path


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

    if st.sidebar.button("🔄 Reload config", width='stretch'):
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


def _view_portfolio(
    portfolio: Portfolio,
    orchestrator: Orchestrator,
    pf_path: str,
) -> None:
    st.header("Portfolio")

    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("🔄 Update data", width='stretch'):
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
        if st.button("📊 Compute status", width='stretch'):
            _set_state("last_status", None)  # force recompute below

    # Compute (or reuse) current status
    status = _get_state("last_status")
    if status is None:
        with st.spinner("Computing status…"):
            status = orchestrator._container.portfolio_service.current_status(portfolio)
        _set_state("last_status", status)

    _render_status_panel(portfolio, status)
    _render_refresh_warnings()
    _render_portfolio_editor(portfolio, pf_path)


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
        width='stretch',
        hide_index=True,
        column_config={
            "Shares": st.column_config.NumberColumn(format="%.0f"),
            "Entry": st.column_config.NumberColumn(format="%.2f"),
            "Last": st.column_config.NumberColumn(format="%.2f"),
            "Last date": st.column_config.DateColumn(),
            "Cost basis": st.column_config.NumberColumn(format="%.2f"),
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "Weight": st.column_config.NumberColumn(
                format="%.2f%%",
                help="Udział pozycji w wartości rynkowej portfela",
            ),
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


def _portfolio_to_editor_df(portfolio: Portfolio) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": h.ticker,
                "name": h.name or "",
                "shares": float(h.shares),
                "entry_price": float(h.entry_price),
                "entry_date": h.entry_date,
                "thesis": h.thesis,
                "keywords": ", ".join(h.keywords),
            }
            for h in portfolio.holdings
        ]
    )


def _editor_row_to_holding(row: dict) -> Holding:
    raw_keywords = row.get("keywords") or ""
    keywords = [k.strip() for k in str(raw_keywords).split(",") if k.strip()]
    name = (row.get("name") or "").strip() or None
    return Holding(
        ticker=str(row["ticker"]).strip(),
        shares=float(row["shares"]),
        entry_price=float(row["entry_price"]),
        entry_date=row["entry_date"],
        thesis=str(row["thesis"]),
        name=name,
        keywords=keywords,
    )


def _render_portfolio_editor(portfolio: Portfolio, pf_path: str) -> None:
    with st.expander("Edit portfolio", expanded=False):
        st.caption(
            f"Plik: `{pf_path}` · zmiany trafią do YAML po kliknięciu **Zapisz**. "
            "Poprzednia wersja zostanie skopiowana do `<plik>.bak`."
        )

        edited = st.data_editor(
            _portfolio_to_editor_df(portfolio),
            num_rows="dynamic",
            width="stretch",
            key="portfolio_editor",
            column_config={
                "ticker": st.column_config.TextColumn(
                    "Ticker", required=True,
                    help="np. PKN, PKN.WA lub pkn.pl — zostanie znormalizowany.",
                ),
                "name": st.column_config.TextColumn("Name"),
                "shares": st.column_config.NumberColumn(
                    "Shares", min_value=0.0, step=1.0, required=True, format="%.4f",
                ),
                "entry_price": st.column_config.NumberColumn(
                    "Entry price", min_value=0.0, step=0.01, required=True, format="%.4f",
                ),
                "entry_date": st.column_config.DateColumn(
                    "Entry date", required=True,
                ),
                "thesis": st.column_config.TextColumn(
                    "Thesis", required=True,
                    help="Krótka teza inwestycyjna (markdown).",
                ),
                "keywords": st.column_config.TextColumn(
                    "Keywords",
                    help="Słowa kluczowe oddzielone przecinkami (puste → ticker).",
                ),
            },
        )

        cols = st.columns([1, 1, 4])
        save_clicked = cols[0].button("💾 Zapisz do YAML", type="primary", width="stretch")
        if cols[1].button("↩️ Odrzuć zmiany", width="stretch"):
            st.session_state.pop("portfolio_editor", None)
            st.rerun()

        if not save_clicked:
            return

        try:
            holdings: list[Holding] = []
            for idx, row in edited.iterrows():
                row_dict = row.to_dict()
                if not row_dict.get("ticker") or pd.isna(row_dict.get("shares")):
                    continue
                try:
                    holdings.append(_editor_row_to_holding(row_dict))
                except Exception as exc:
                    st.error(f"Wiersz {int(idx) + 1}: {exc}")
                    return

            new_portfolio = Portfolio(
                base_currency=portfolio.base_currency,
                holdings=holdings,
            )
        except Exception as exc:
            st.error(f"❌ Walidacja portfolio: {exc}")
            return

        try:
            written = save_portfolio(new_portfolio, pf_path)
        except PortfolioError as exc:
            st.error(f"❌ Zapis nie powiódł się: {exc}")
            return

        st.success(f"✅ Zapisano `{written}` (backup: `{written}.bak`).")
        _bootstrap.clear()
        _set_state("last_status", None)
        _set_state("last_refresh_report", None)
        st.rerun()


def _render_refresh_warnings() -> None:
    report = _get_state("last_refresh_report")
    if report is None:
        return
    if report.ohlcv_failed:
        st.warning("OHLCV failed for: " + ", ".join(report.ohlcv_failed.keys()))
    if report.news_failed:
        for line in report.news_failed:
            st.warning(line)


# --- Strategy descriptions ------------------------------------------------


_STRATEGY_DESCRIPTIONS = {
    "ma_crossover": (
        "**Moving Average Crossover**  \n"
        "Kupuje akcje gdy krótka średnia krocząca (SMA) przechodzi powyżej długiej średniej — "
        "sygnał trendu wzrostowego (złoty krzyż). Sprzedaje przy przecięciu w dół.  \n"
        "_Trend-following strategy. Skuteczna w rynkach z wyraźnym trendem._"
    ),
    "momentum": (
        "**Time-Series Momentum**  \n"
        "Kupuje akcje, które miały dodatni zwrot w ostatnich N dni (lookback period). "
        "Każda akcja oceniana niezależnie na podstawie swojego historycznego zwrotu.  \n"
        "_Momentum strategy. Polega na tym, że trendy się utrzymują w krótkim terminie._"
    ),
    "buy_and_hold": (
        "**Buy & Hold**  \n"
        "Kupuje wszystkie akcje portfela na pierwszym dniu i trzyma je przez cały okres backtestu. "
        "Brak rebalansowania ani sprzedaży — idealna strategia bazowa.  \n"
        "_Passive buy-and-hold baseline. Punkt odniesienia dla aktywnych strategii._"
    ),
}


# --- View: Backtest --------------------------------------------------------


def _view_backtest(portfolio: Portfolio, orchestrator: Orchestrator) -> None:
    st.header("Backtest")

    cols = st.columns([2, 1, 1, 1])
    strategy = cols[0].selectbox(
        "Strategy", 
        options=list(KNOWN_STRATEGIES), 
        index=2,
        help="Wybierz strategię tradingową do przetestowania"
    )
    start_date = cols[1].date_input(
        "Start date",
        value=orchestrator._container.config.backtest.start_date,
        help="Backtest period start date"
    )
    include_bm = cols[2].checkbox("Include benchmark", value=True)
    run = cols[3].button("▶ Run backtest", width='stretch')

    # Display strategy description
    if strategy in _STRATEGY_DESCRIPTIONS:
        st.info(_STRATEGY_DESCRIPTIONS[strategy])

    # Display current strategy parameters
    strat_cfg = orchestrator._container.config.strategies
    if strategy == "ma_crossover":
        st.caption(
            f"📊 Parameters: fast SMA = {strat_cfg.ma_crossover.fast} dni, "
            f"slow SMA = {strat_cfg.ma_crossover.slow} dni"
        )
    elif strategy == "momentum":
        st.caption(
            f"📊 Parameters: lookback = {strat_cfg.momentum.lookback} dni, "
            f"threshold = {strat_cfg.momentum.threshold:.1%}"
        )
    elif strategy == "buy_and_hold":
        st.caption("📊 Parameters: brak (zawsze trzymaj wszystkie pozycje)")

    if run:
        with st.spinner(f"Running {strategy}…"):
            try:
                result = orchestrator.backtest(
                    portfolio,
                    strategy_name=strategy,
                    start=start_date,
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
    # Convert to percentage returns: (value - initial_capital) / initial_capital * 100
    df_pct = ((df - result.initial_capital) / result.initial_capital * 100)
    
    fig = go.Figure()
    
    for col in df_pct.columns:
        fig.add_trace(
            go.Scatter(x=df_pct.index, y=df_pct[col], mode="lines", name=col, connectgaps=True)
        )
    fig.update_layout(
        title="Equity curve (% return)",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        height=420,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width='stretch')

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
        st.plotly_chart(dd_fig, width='stretch')

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
        st.dataframe(table, width='stretch', hide_index=True)


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
    run = cols[3].button("✨ Run analysis", width='stretch')

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
    generate = cols[3].button("📄 Generate report", width='stretch')

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
                width='stretch',
            )
        with col_a:
            st.caption(f"Path: `{selected}`")
        st.markdown(body)


# --- View: Monitoring -----------------------------------------------------


def _view_monitoring(
    container: ServiceContainer,
    portfolio: Portfolio,
    orchestrator: Orchestrator,
) -> None:
    st.header("📋 Monitoring portfela")
    st.caption(
        "Cykliczny raport monitorujący: fundamentals (Stooq) + ESPI/news + "
        "porównanie z poprzednim raportem. Wynik to plik HTML w "
        "`reports/monitoring/`."
    )

    monitoring_dir = Path("reports") / "monitoring"

    cols = st.columns([1, 1, 2])
    news_days = cols[0].number_input(
        "News window (days)",
        min_value=7,
        max_value=120,
        value=30,
        step=1,
        key="monitoring_news_days",
        help="Window for ESPI/news context fed to the LLM.",
    )
    filename = cols[1].text_input(
        "Filename (optional)",
        value="",
        key="monitoring_filename",
    )
    generate = cols[2].button(
        "📋 Generuj raport monitorujący",
        width='stretch',
        type="primary",
    )

    prev = container.monitoring_service.load_latest_snapshot()
    if prev is not None:
        st.caption(
            f"📎 Poprzedni snapshot: "
            f"{prev.generated_at.strftime('%Y-%m-%d %H:%M UTC')} — "
            f"raport będzie zawierał porównanie."
        )
    else:
        st.caption("📎 Brak poprzedniego snapshotu — to będzie pierwszy raport.")

    if generate:
        with st.spinner("Pobieram fundamentals + news, pytam LLM…"):
            try:
                result = orchestrator.generate_monitoring_report(
                    portfolio,
                    news_days_back=int(news_days),
                    filename=filename or None,
                )
            except LLMError as exc:
                st.error(f"❌ LLM failed: {exc}")
                return
            except Exception as exc:  # pragma: no cover
                st.error(f"❌ Monitoring run failed: {exc}")
                return

            _set_state("last_monitoring_result", result)
            for w in result.warnings:
                st.warning(w)
            st.success(
                f"✅ Raport zapisany: `{result.html_path.name}` · "
                f"snapshot: `{result.snapshot_path.name}` · "
                f"{'z porównaniem' if result.had_previous_snapshot else 'pierwszy raport'}"
            )

    st.subheader("Istniejące raporty")
    files = list_monitoring_reports(monitoring_dir)
    if not files:
        st.info(f"Brak raportów w `{monitoring_dir.resolve()}`. Wygeneruj nowy powyżej.")
        return

    # Clear-all action with confirm checkbox
    clear_cols = st.columns([1, 3])
    with clear_cols[0]:
        confirm_clear = st.checkbox(
            "Potwierdź czyszczenie",
            key="monitoring_confirm_clear",
            help="Zaznacz aby odblokować przycisk 'Wyczyść wszystkie raporty'.",
        )
    with clear_cols[1]:
        if st.button(
            "🧹 Wyczyść wszystkie raporty (HTML + snapshoty)",
            disabled=not confirm_clear,
            width='stretch',
        ):
            removed = _delete_all_monitoring(monitoring_dir)
            st.success(f"Usunięto {removed} plik(ów).")
            st.session_state["monitoring_confirm_clear"] = False
            st.rerun()

    st.divider()

    # Per-report list with delete buttons
    st.markdown("**Raporty (najnowsze pierwsze):**")
    for path in files:
        cols = st.columns([5, 1])
        size = _human_size(path.stat().st_size)
        when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        with cols[0]:
            if st.button(
                f"📄 {path.name} · {size} · {when}",
                key=f"select_{path.name}",
                width='stretch',
            ):
                _set_state("monitoring_selected_path", str(path))
        with cols[1]:
            if st.button(
                "🗑️",
                key=f"del_{path.name}",
                help=f"Usuń {path.name} (i powiązany snapshot)",
                width='stretch',
            ):
                _delete_monitoring_pair(path)
                # If currently selected, clear selection
                if _get_state("monitoring_selected_path") == str(path):
                    _set_state("monitoring_selected_path", None)
                st.rerun()

    selected_path_str = _get_state("monitoring_selected_path")
    selected = Path(selected_path_str) if selected_path_str else files[0]
    if not selected.exists():
        return

    st.divider()
    body = selected.read_text(encoding="utf-8")
    col_a, col_b = st.columns([3, 1])
    with col_b:
        st.download_button(
            "⬇️ Download HTML",
            data=body,
            file_name=selected.name,
            mime="text/html",
            width='stretch',
        )
    with col_a:
        st.caption(f"Podgląd: `{selected.name}`")
    st.iframe(body, height=900)


def _delete_monitoring_pair(report_path: Path) -> None:
    """Delete an HTML report + its sibling snapshot JSON (matched by timestamp)."""
    try:
        report_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to delete %s: %s", report_path, exc)
        return
    # Snapshot filename: snapshot_<TS>.json next to monitoring/snapshots/
    if report_path.name.startswith("monitoring_") and report_path.name.endswith(".html"):
        ts = report_path.stem.removeprefix("monitoring_")
        snap = report_path.parent / "snapshots" / f"snapshot_{ts}.json"
        try:
            snap.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover
            logger.warning("Failed to delete snapshot %s: %s", snap, exc)


def _delete_all_monitoring(monitoring_dir: Path) -> int:
    """Delete every HTML report and snapshot under ``monitoring_dir``."""
    if not monitoring_dir.is_dir():
        return 0
    count = 0
    for p in monitoring_dir.iterdir():
        if p.is_file() and p.suffix == ".html":
            try:
                p.unlink()
                count += 1
            except OSError as exc:  # pragma: no cover
                logger.warning("Failed to delete %s: %s", p, exc)
    snaps = monitoring_dir / "snapshots"
    if snaps.is_dir():
        for p in snaps.iterdir():
            if p.is_file() and p.suffix == ".json":
                try:
                    p.unlink()
                    count += 1
                except OSError as exc:  # pragma: no cover
                    logger.warning("Failed to delete %s: %s", p, exc)
    return count


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.0f} GB"


# --- Main ------------------------------------------------------------------


def main() -> None:
    config_path, portfolio_override = _sidebar()
    container, portfolio, orchestrator, pf_path = _safe_bootstrap(config_path, portfolio_override)

    st.sidebar.divider()
    st.sidebar.markdown("**Loaded:**")
    st.sidebar.caption(f"Portfolio: {len(portfolio.holdings)} holding(s)")
    st.sidebar.caption(f"Base currency: {portfolio.base_currency}")
    st.sidebar.caption(f"Benchmark: {container.config.backtest.benchmark}")
    st.sidebar.caption(f"LLM: {container.config.llm.provider} ({container.config.llm.model_analysis})")

    tabs = st.tabs([
        "📊 Portfolio",
        "📈 Backtest",
        "✨ AI Analysis",
        "📋 Monitoring",
        "📄 Reports",
    ])
    with tabs[0]:
        _view_portfolio(portfolio, orchestrator, pf_path)
    with tabs[1]:
        _view_backtest(portfolio, orchestrator)
    with tabs[2]:
        _view_analysis(portfolio, orchestrator)
    with tabs[3]:
        _view_monitoring(container, portfolio, orchestrator)
    with tabs[4]:
        _view_reports(container, portfolio, orchestrator)

    st.sidebar.divider()
    st.sidebar.caption(
        "Investment Copilot — research material, not financial advice."
    )


if __name__ == "__main__":
    main()
