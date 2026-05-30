"""Wire-format DTOs for the web frontend.

These intentionally diverge from the internal domain models so the
frontend can stay on simple, conventional names (`total_value`, `pnl`,
`pnl_pct`, English severity tags, etc.). Adapters in
:mod:`investment_copilot.api.adapters` build these from domain objects.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from investment_copilot.domain.prompts import MonitoringReport


# --- Portfolio --------------------------------------------------------------


class TransactionDTO(BaseModel):
    """A single BUY or SELL row inside HoldingDTO.transactions."""

    date: date
    action: Literal["BUY", "SELL"]
    shares: float
    price_per_share: float
    fees: float = 0.0
    note: str = ""


class HoldingDTO(BaseModel):
    """A single holding as the frontend sees it (no derived pricing fields).

    ``entry_price`` and ``entry_date`` remain on the wire for backwards
    compatibility — they now reflect the FIFO-derived **average cost** and
    **first BUY date**. The authoritative source of truth is
    ``transactions``; the frontend should treat the legacy fields as
    convenience read-only when no transaction-aware UI is wired yet.
    """

    ticker: str = Field(description="Canonical Stooq form, e.g. 'pkn.pl'.")
    display_ticker: str = Field(
        default="",
        description=(
            "Short display form, e.g. 'PKN'. Set by the backend on GET; "
            "ignored on PUT (the backend re-derives it from `ticker`)."
        ),
    )
    name: str | None = None
    shares: float = 0.0
    entry_price: float = Field(default=0.0, description="Avg cost per active share (FIFO).")
    entry_date: date = Field(
        default_factory=date.today,
        description=(
            "Date of the earliest BUY. Defaulted when the client sends "
            "only `transactions`; the backend re-derives it on save."
        ),
    )
    thesis: str
    keywords: list[str] = Field(default_factory=list)
    transactions: list[TransactionDTO] = Field(
        default_factory=list,
        description=(
            "Chronological transactions. On PUT the backend re-derives "
            "shares/entry_price/entry_date from this list and ignores the "
            "legacy scalar fields."
        ),
    )
    realized_pnl: float = 0.0


class HoldingStatusDTO(HoldingDTO):
    """Holding + computed pricing. Mirrors the frontend mock shape."""

    n_transactions: int = 0
    last_price: float | None = None
    last_price_date: date | None = None
    value: float | None = None
    pnl: float | None = None
    pnl_pct: float | None = None


class PortfolioDTO(BaseModel):
    """A portfolio as the frontend sees it (for GET/PUT /api/portfolio)."""

    base_currency: str = "PLN"
    holdings: list[HoldingDTO]


class PortfolioStatusDTO(BaseModel):
    """Aggregate portfolio status. Mirrors MOCK_PORTFOLIO."""

    base_currency: str
    holdings: list[HoldingStatusDTO]
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    total_cost_basis: float
    total_realized_pnl: float = 0.0
    missing_data: list[str] = Field(default_factory=list)
    as_of: datetime


# --- Data update ------------------------------------------------------------


class DataUpdateResult(BaseModel):
    ohlcv_updated: dict[str, int]
    ohlcv_failed: dict[str, str]
    benchmark_symbol: str | None
    benchmark_rows: int
    news_inserted: int
    news_failed: list[str]


# --- Backtest ---------------------------------------------------------------


class BacktestMetricsDTO(BaseModel):
    total_return: float
    annualized_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    max_drawdown_duration_days: int | None
    win_rate: float
    n_observations: int


class EquityCurvePoint(BaseModel):
    """Combined point — portfolio and benchmark on the same date."""

    date: date
    portfolio: float
    benchmark: float | None = None


class DrawdownPoint(BaseModel):
    date: date
    drawdown: float  # percent, negative or zero


class BacktestResultDTO(BaseModel):
    strategy: str
    strategy_params: dict[str, float | int | str]
    start_date: date
    end_date: date
    initial_capital: float
    final_value: float
    equity_curve: list[EquityCurvePoint]
    drawdown: list[DrawdownPoint]
    metrics: BacktestMetricsDTO
    benchmark_symbol: str | None = None
    benchmark_metrics: BacktestMetricsDTO | None = None
    missing_tickers: list[str] = Field(default_factory=list)
    tickers_used: list[str] = Field(default_factory=list)
    generated_at: datetime


# --- AI analysis ------------------------------------------------------------


class CitationDTO(BaseModel):
    """Grounding reference attached to a claim in the LLM output."""

    source_type: Literal["news", "metric", "fundamentals", "previous_report"]
    reference: str


class ThesisUpdateDTO(BaseModel):
    ticker: str
    assessment: str
    citations: list[CitationDTO] = Field(default_factory=list)


class PortfolioAnalysisDTO(BaseModel):
    """Frontend-friendly view of PortfolioAnalysis."""

    model_config = ConfigDict(extra="forbid")

    summary_md: str
    thesis_updates: list[ThesisUpdateDTO]
    confidence: int


SeverityEN = Literal["low", "medium", "high"]


class RiskAlertDTO(BaseModel):
    severity: SeverityEN
    title: str
    description: str
    ticker: str | None = None
    suggested_action: str | None = None
    citations: list[CitationDTO] = Field(default_factory=list)


class HoldingMetricsDTO(BaseModel):
    """Per-holding quantitative metrics (mirrors PortfolioMetrics.holdings)."""

    ticker: str
    display_ticker: str
    weight_pct: float | None = None
    ret_30d_pct: float | None = None
    ret_90d_pct: float | None = None
    ret_252d_pct: float | None = None
    distance_from_52w_high_pct: float | None = None
    distance_from_52w_low_pct: float | None = None
    ann_volatility_pct: float | None = None
    beta_vs_benchmark: float | None = None


class CorrelationPairDTO(BaseModel):
    ticker_a: str
    ticker_b: str
    display_a: str
    display_b: str
    correlation: float


class PortfolioMetricsDTO(BaseModel):
    """Quant metrics for the analysis tab — what the LLM was citing."""

    n_holdings: int
    n_priced: int
    hhi: float | None = None
    top3_weight_pct: float | None = None
    largest_position_ticker: str | None = None
    largest_position_display_ticker: str | None = None
    largest_position_weight_pct: float | None = None
    benchmark_symbol: str | None = None
    holdings: list[HoldingMetricsDTO] = Field(default_factory=list)
    top_correlations: list[CorrelationPairDTO] = Field(default_factory=list)


class AnalysisBundleDTO(BaseModel):
    status: PortfolioStatusDTO
    analysis: PortfolioAnalysisDTO | None = None
    risk_overview: str | None = None
    alerts: list[RiskAlertDTO] = Field(default_factory=list)
    metrics: PortfolioMetricsDTO | None = None
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime


# --- Reports ----------------------------------------------------------------


class ReportFileDTO(BaseModel):
    name: str
    mtime: datetime
    size_bytes: int


class ReportContentDTO(BaseModel):
    name: str
    mtime: datetime
    size_bytes: int
    content_md: str


class GenerateReportRequest(BaseModel):
    strategy: str | None = "ma_crossover"
    news_days_back: int = 14
    filename: str | None = None


class GenerateReportResponse(BaseModel):
    report: ReportFileDTO
    warnings: list[str] = Field(default_factory=list)


# --- Monitoring -------------------------------------------------------------

MonitoringStatusEN = Literal["on_track", "watch", "at_risk"]


class MonitoringItemDTO(BaseModel):
    """Flattened per-holding monitoring status (matches the existing UI)."""

    ticker: str
    status: MonitoringStatusEN
    rationale: str


class MonitoringSnapshotDTO(BaseModel):
    """Wire payload for the monitoring tab."""

    generated_at: datetime
    items: list[MonitoringItemDTO]
    reports: list[ReportFileDTO]
    had_previous_snapshot: bool
    report: MonitoringReport | None = None
    warnings: list[str] = Field(default_factory=list)


class RunMonitoringRequest(BaseModel):
    news_days_back: int = 30


# --- Misc -------------------------------------------------------------------


# --- Watchlist --------------------------------------------------------------


class WatchlistItemDTO(BaseModel):
    """A watchlist entry as the frontend sees it."""

    ticker: str
    display_ticker: str = Field(
        default="",
        description=(
            "Short display form, e.g. 'CDR'. Set by the backend on GET; "
            "ignored on PUT (the backend re-derives it from `ticker`)."
        ),
    )
    name: str | None = None
    added_date: date
    target_buy_price: float | None = None
    notes: str = ""
    keywords: list[str] = Field(default_factory=list)


class WatchlistDTO(BaseModel):
    items: list[WatchlistItemDTO]


class WatchlistItemStatusDTO(WatchlistItemDTO):
    """A watchlist item enriched with the latest cached price + alert flag."""

    last_price: float | None = None
    last_price_date: date | None = None
    distance_to_target_pct: float | None = None
    alert: bool = False
    news_count_30d: int = 0


class WatchlistStatusDTO(BaseModel):
    as_of: datetime
    items: list[WatchlistItemStatusDTO]
    missing_data: list[str] = Field(default_factory=list)


# --- Calendar ---------------------------------------------------------------


class CalendarEventDTO(BaseModel):
    ticker: str
    display_ticker: str
    name: str | None = None
    kind: Literal["report", "dividend", "agm", "espi", "dividend_record", "dividend_payment"]
    event_date: date | None = None
    label: str
    description: str = ""
    importance: Literal["high", "medium", "low"] = "medium"
    amount_pln: float | None = None
    days_until: int | None = Field(
        default=None,
        description="Positive when in the future; null when no date.",
    )


class CalendarBundleDTO(BaseModel):
    events: list[CalendarEventDTO]
    snapshot_age_days: int | None = None
    warnings: list[str] = Field(default_factory=list)


class HealthDTO(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class StrategyInfoDTO(BaseModel):
    value: str
    label: str


class BenchmarkInfoDTO(BaseModel):
    value: str
    label: str


class AppConfigDTO(BaseModel):
    """Subset of AppConfig safe to expose to the frontend."""

    benchmark: str
    benchmark_label: str
    backtest_start_date: date
    backtest_end_date: date | None
    available_benchmarks: list[BenchmarkInfoDTO]
