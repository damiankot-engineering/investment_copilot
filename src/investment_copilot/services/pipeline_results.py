"""Typed result models returned by orchestrator pipelines.

Each pipeline returns a small frozen container so the CLI (Step 10) and
the future FastAPI app can render or serialize results without touching
service internals.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from investment_copilot.domain.backtest import BacktestResult
from investment_copilot.domain.portfolio import PortfolioStatus
from investment_copilot.domain.prompts import PortfolioAnalysis, RiskAlerts


class AnalysisBundle(BaseModel):
    """Output of the ``run_analysis`` pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    status: PortfolioStatus
    analysis: PortfolioAnalysis | None = None
    risks: RiskAlerts | None = None
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime


class FullReport(BaseModel):
    """Output of the ``generate_report`` pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    status: PortfolioStatus
    backtest: BacktestResult | None = None
    analysis: PortfolioAnalysis | None = None
    risks: RiskAlerts | None = None
    report_path: Path
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime
