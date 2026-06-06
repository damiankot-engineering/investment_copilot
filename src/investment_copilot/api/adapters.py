"""Conversions between domain models and frontend wire DTOs."""

from __future__ import annotations

from datetime import date

from investment_copilot.api.schemas import (
    AnalysisBundleDTO,
    BacktestMetricsDTO,
    BacktestResultDTO,
    CalendarBundleDTO,
    CalendarEventDTO,
    CitationDTO,
    CorrelationPairDTO,
    DrawdownPoint,
    EquityCurvePoint,
    HoldingDTO,
    HoldingMetricsDTO,
    HoldingStatusDTO,
    MonitoringItemDTO,
    PortfolioAnalysisDTO,
    PortfolioDTO,
    PortfolioMetricsDTO,
    PortfolioStatusDTO,
    RiskAlertDTO,
    ThesisUpdateDTO,
    TransactionDTO,
    WatchlistDTO,
    WatchlistItemDTO,
    WatchlistItemStatusDTO,
    WatchlistStatusDTO,
)
from investment_copilot.domain.analysis_metrics import PortfolioMetrics
from investment_copilot.domain.calendar import CalendarBundle, CalendarEvent
from investment_copilot.domain.prompts import Citation
from investment_copilot.domain.watchlist import Watchlist, WatchlistItem
from investment_copilot.services.watchlist_service import (
    WatchlistItemStatus,
    WatchlistStatus,
)
from investment_copilot.domain.backtest import BacktestResult, StrategyMetrics
from investment_copilot.domain.portfolio import (
    Holding,
    HoldingStatus,
    Portfolio,
    PortfolioStatus,
)
from investment_copilot.domain.prompts import (
    MonitoringReport,
    PortfolioAnalysis,
    RiskAlerts,
)
from investment_copilot.services.pipeline_results import AnalysisBundle


# --- Tickers ----------------------------------------------------------------


def display_ticker(canonical: str) -> str:
    """`pkn.pl` -> `PKN`. Best-effort; preserves stems without a dot."""
    stem = canonical.split(".")[0]
    return stem.upper()


# --- Portfolio --------------------------------------------------------------


def holding_to_dto(h: Holding) -> HoldingDTO:
    return HoldingDTO(
        ticker=h.ticker,
        display_ticker=display_ticker(h.ticker),
        name=h.name,
        shares=h.shares,
        entry_price=h.avg_entry_price,
        entry_date=h.first_entry_date,
        thesis=h.thesis,
        keywords=list(h.keywords),
        target_weight=h.target_weight,
        transactions=[
            TransactionDTO(
                date=tx.date,
                action=tx.action,
                shares=tx.shares,
                price_per_share=tx.price_per_share,
                fees=tx.fees,
                note=tx.note,
            )
            for tx in h.transactions
        ],
        realized_pnl=h.realized_pnl,
    )


def holding_status_to_dto(s: HoldingStatus) -> HoldingStatusDTO:
    return HoldingStatusDTO(
        ticker=s.ticker,
        display_ticker=display_ticker(s.ticker),
        name=s.name,
        shares=s.shares,
        entry_price=s.entry_price,
        entry_date=s.entry_date,
        thesis="",  # not present on HoldingStatus; UI does not need it here
        keywords=[],
        transactions=[],  # not loaded here; portfolio endpoint surfaces them
        realized_pnl=s.realized_pnl,
        n_transactions=s.n_transactions,
        last_price=s.last_price,
        last_price_date=s.last_price_date,
        value=s.market_value,
        pnl=s.unrealized_pnl,
        # Domain stores as fraction (0.012 = 1.2%); UI renders as percent.
        pnl_pct=(s.unrealized_pnl_pct * 100.0) if s.unrealized_pnl_pct is not None else None,
    )


def portfolio_to_dto(p: Portfolio) -> PortfolioDTO:
    return PortfolioDTO(
        name=p.name,
        base_currency=p.base_currency,
        holdings=[holding_to_dto(h) for h in p.holdings],
    )


def portfolio_status_to_dto(
    status: PortfolioStatus,
    *,
    portfolio: Portfolio | None = None,
) -> PortfolioStatusDTO:
    """Build the wire-format status; enrich with thesis/keywords if a
    matching :class:`Portfolio` is provided (the status itself drops them).
    """
    thesis_by_ticker: dict[str, tuple[str, list[str], list[TransactionDTO]]] = {}
    if portfolio is not None:
        thesis_by_ticker = {
            h.ticker: (
                h.thesis,
                list(h.keywords),
                [
                    TransactionDTO(
                        date=tx.date,
                        action=tx.action,
                        shares=tx.shares,
                        price_per_share=tx.price_per_share,
                        fees=tx.fees,
                        note=tx.note,
                    )
                    for tx in h.transactions
                ],
            )
            for h in portfolio.holdings
        }

    holdings: list[HoldingStatusDTO] = []
    for s in status.holdings:
        dto = holding_status_to_dto(s)
        if s.ticker in thesis_by_ticker:
            thesis, keywords, transactions = thesis_by_ticker[s.ticker]
            dto = dto.model_copy(update={
                "thesis": thesis, "keywords": keywords,
                "transactions": transactions,
            })
        holdings.append(dto)

    return PortfolioStatusDTO(
        base_currency=status.base_currency,
        holdings=holdings,
        total_value=status.total_market_value,
        total_pnl=status.total_unrealized_pnl,
        total_pnl_pct=status.total_unrealized_pnl_pct * 100.0,
        total_cost_basis=status.total_cost_basis,
        total_realized_pnl=status.total_realized_pnl,
        missing_data=list(status.missing_data),
        as_of=status.as_of,
    )


# --- Backtest ---------------------------------------------------------------


def _metrics_to_dto(m: StrategyMetrics) -> BacktestMetricsDTO:
    return BacktestMetricsDTO(
        total_return=m.total_return,
        annualized_return=m.annualized_return,
        volatility=m.annualized_volatility,
        sharpe=m.sharpe_ratio,
        max_drawdown=m.max_drawdown,
        max_drawdown_duration_days=m.max_drawdown_duration_days,
        win_rate=m.win_rate,
        n_observations=m.n_observations,
    )


def backtest_to_dto(r: BacktestResult) -> BacktestResultDTO:
    """Flatten portfolio + benchmark curves into one percent-return series.

    Both portfolio and benchmark are normalized to their own starting value,
    so the chart shows cumulative return (start = 0%). The drawdown series
    is also in percent off the running portfolio peak.
    """
    portfolio_base = r.equity_curve[0].value if r.equity_curve else r.initial_capital
    bench_base = (
        r.benchmark_equity_curve[0].value if r.benchmark_equity_curve else None
    )

    def pct(value: float, base: float) -> float:
        return ((value / base) - 1.0) * 100.0 if base > 0 else 0.0

    bench_by_date = {
        p.date: pct(p.value, bench_base) if bench_base is not None else None
        for p in r.benchmark_equity_curve
    }
    curve = [
        EquityCurvePoint(
            date=p.date,
            portfolio=pct(p.value, portfolio_base),
            benchmark=bench_by_date.get(p.date),
        )
        for p in r.equity_curve
    ]

    drawdown: list[DrawdownPoint] = []
    peak = float("-inf")
    for p in r.equity_curve:
        if p.value > peak:
            peak = p.value
        dd = ((p.value / peak) - 1.0) * 100.0 if peak > 0 else 0.0
        drawdown.append(DrawdownPoint(date=p.date, drawdown=dd))

    return BacktestResultDTO(
        strategy=r.strategy_name,
        strategy_params=dict(r.strategy_params),
        start_date=r.start_date,
        end_date=r.end_date,
        initial_capital=r.initial_capital,
        final_value=r.final_value,
        equity_curve=curve,
        drawdown=drawdown,
        metrics=_metrics_to_dto(r.metrics),
        benchmark_symbol=r.benchmark_symbol,
        benchmark_metrics=(
            _metrics_to_dto(r.benchmark_metrics) if r.benchmark_metrics else None
        ),
        missing_tickers=list(r.missing_tickers),
        tickers_used=list(r.tickers_used),
        generated_at=r.generated_at,
    )


# --- AI analysis ------------------------------------------------------------


_SEVERITY_PL_TO_EN = {"niskie": "low", "średnie": "medium", "wysokie": "high"}


def _citations_to_dto(citations: list[Citation]) -> list[CitationDTO]:
    return [
        CitationDTO(source_type=c.source_type, reference=c.reference)
        for c in citations
    ]


def portfolio_analysis_to_dto(a: PortfolioAnalysis) -> PortfolioAnalysisDTO:
    summary_md = a.summary
    if a.diversification_notes:
        summary_md = (
            f"{summary_md}\n\n### Dywersyfikacja\n{a.diversification_notes}"
        )
    thesis_updates = [
        ThesisUpdateDTO(
            ticker=display_ticker(c.ticker),
            assessment=f"{c.comment}\n\nRekomendacja: **{c.recommendation}**",
            citations=_citations_to_dto(c.citations),
        )
        for c in a.holdings_comments
    ]
    return PortfolioAnalysisDTO(
        summary_md=summary_md,
        thesis_updates=thesis_updates,
        confidence=a.confidence,
    )


def risk_alerts_to_dtos(r: RiskAlerts) -> tuple[str, list[RiskAlertDTO]]:
    alerts = [
        RiskAlertDTO(
            severity=_SEVERITY_PL_TO_EN.get(a.severity, "medium"),  # type: ignore[arg-type]
            title=a.title,
            description=a.description,
            ticker=display_ticker(a.ticker) if a.ticker else None,
            suggested_action=a.suggested_action,
            citations=_citations_to_dto(a.citations),
        )
        for a in r.alerts
    ]
    return r.overview, alerts


# --- Watchlist --------------------------------------------------------------


def watchlist_item_to_dto(it: WatchlistItem) -> WatchlistItemDTO:
    return WatchlistItemDTO(
        ticker=it.ticker,
        display_ticker=display_ticker(it.ticker),
        name=it.name,
        added_date=it.added_date,
        target_buy_price=it.target_buy_price,
        notes=it.notes,
        keywords=list(it.keywords),
    )


def watchlist_to_dto(wl: Watchlist) -> WatchlistDTO:
    return WatchlistDTO(items=[watchlist_item_to_dto(it) for it in wl.items])


def watchlist_item_status_to_dto(s: WatchlistItemStatus) -> WatchlistItemStatusDTO:
    return WatchlistItemStatusDTO(
        ticker=s.ticker,
        display_ticker=display_ticker(s.ticker),
        name=s.name,
        added_date=s.added_date,
        target_buy_price=s.target_buy_price,
        notes=s.notes,
        keywords=list(s.keywords),
        last_price=s.last_price,
        last_price_date=s.last_price_date,
        distance_to_target_pct=s.distance_to_target_pct,
        alert=s.alert,
        news_count_30d=s.news_count_30d,
    )


def watchlist_status_to_dto(status: WatchlistStatus) -> WatchlistStatusDTO:
    return WatchlistStatusDTO(
        as_of=status.as_of,
        items=[watchlist_item_status_to_dto(it) for it in status.items],
        missing_data=list(status.missing_data),
    )


# --- Calendar ---------------------------------------------------------------


def calendar_event_to_dto(e: CalendarEvent) -> CalendarEventDTO:
    days_until = (e.event_date - date.today()).days if e.event_date else None
    return CalendarEventDTO(
        ticker=e.ticker,
        display_ticker=display_ticker(e.ticker),
        name=e.name,
        kind=e.kind,
        event_date=e.event_date,
        label=e.label,
        description=e.description,
        importance=e.importance,
        amount_pln=e.amount_pln,
        days_until=days_until,
    )


def calendar_bundle_to_dto(bundle: CalendarBundle) -> CalendarBundleDTO:
    return CalendarBundleDTO(
        events=[calendar_event_to_dto(e) for e in bundle.events],
        snapshot_age_days=bundle.snapshot_age_days,
        warnings=list(bundle.warnings),
    )


def portfolio_metrics_to_dto(m: PortfolioMetrics) -> PortfolioMetricsDTO:
    return PortfolioMetricsDTO(
        n_holdings=m.n_holdings,
        n_priced=m.n_priced,
        hhi=m.hhi,
        top3_weight_pct=m.top3_weight_pct,
        largest_position_ticker=m.largest_position_ticker,
        largest_position_display_ticker=(
            display_ticker(m.largest_position_ticker)
            if m.largest_position_ticker else None
        ),
        largest_position_weight_pct=m.largest_position_weight_pct,
        benchmark_symbol=m.benchmark_symbol,
        holdings=[
            HoldingMetricsDTO(
                ticker=h.ticker,
                display_ticker=display_ticker(h.ticker),
                weight_pct=h.weight_pct,
                ret_30d_pct=h.ret_30d_pct,
                ret_90d_pct=h.ret_90d_pct,
                ret_252d_pct=h.ret_252d_pct,
                distance_from_52w_high_pct=h.distance_from_52w_high_pct,
                distance_from_52w_low_pct=h.distance_from_52w_low_pct,
                ann_volatility_pct=h.ann_volatility_pct,
                beta_vs_benchmark=h.beta_vs_benchmark,
            )
            for h in m.holdings
        ],
        top_correlations=[
            CorrelationPairDTO(
                ticker_a=c.ticker_a,
                ticker_b=c.ticker_b,
                display_a=display_ticker(c.ticker_a),
                display_b=display_ticker(c.ticker_b),
                correlation=c.correlation,
            )
            for c in m.top_correlations
        ],
    )


def analysis_bundle_to_dto(
    bundle: AnalysisBundle,
    *,
    portfolio: Portfolio | None = None,
    metrics: PortfolioMetrics | None = None,
) -> AnalysisBundleDTO:
    overview: str | None = None
    alerts: list[RiskAlertDTO] = []
    if bundle.risks is not None:
        overview, alerts = risk_alerts_to_dtos(bundle.risks)

    return AnalysisBundleDTO(
        status=portfolio_status_to_dto(bundle.status, portfolio=portfolio),
        analysis=(
            portfolio_analysis_to_dto(bundle.analysis)
            if bundle.analysis is not None
            else None
        ),
        risk_overview=overview,
        alerts=alerts,
        metrics=portfolio_metrics_to_dto(metrics) if metrics is not None else None,
        warnings=list(bundle.warnings),
        generated_at=bundle.generated_at,
    )


# --- Monitoring -------------------------------------------------------------


def _signal_to_status(signal: str) -> str:
    return {
        "bullish": "on_track",
        "neutral": "watch",
        "bearish": "at_risk",
    }.get(signal, "watch")


def monitoring_report_to_items(report: MonitoringReport) -> list[MonitoringItemDTO]:
    """Flatten companies → frontend item list."""
    return [
        MonitoringItemDTO(
            ticker=display_ticker(c.ticker),
            status=_signal_to_status(c.signal),  # type: ignore[arg-type]
            rationale=c.signal_body or c.headline,
        )
        for c in report.companies
    ]
