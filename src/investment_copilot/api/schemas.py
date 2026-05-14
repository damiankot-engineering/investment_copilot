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


class HoldingDTO(BaseModel):
    """A single holding as the frontend sees it (no derived pricing fields)."""

    ticker: str = Field(description="Canonical Stooq form, e.g. 'pkn.pl'.")
    display_ticker: str = Field(
        default="",
        description=(
            "Short display form, e.g. 'PKN'. Set by the backend on GET; "
            "ignored on PUT (the backend re-derives it from `ticker`)."
        ),
    )
    name: str | None = None
    shares: float
    entry_price: float
    entry_date: date
    thesis: str
    keywords: list[str] = Field(default_factory=list)


class HoldingStatusDTO(HoldingDTO):
    """Holding + computed pricing. Mirrors the frontend mock shape."""

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


class ThesisUpdateDTO(BaseModel):
    ticker: str
    assessment: str


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


class AnalysisBundleDTO(BaseModel):
    status: PortfolioStatusDTO
    analysis: PortfolioAnalysisDTO | None = None
    risk_overview: str | None = None
    alerts: list[RiskAlertDTO] = Field(default_factory=list)
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
