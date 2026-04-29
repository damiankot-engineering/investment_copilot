"""Prompt templates, structured-output schemas, and context builders."""

from investment_copilot.domain.prompts.context import (
    MAX_NEWS_PER_TICKER,
    MAX_NEWS_TOTAL,
    MAX_THESIS_CHARS,
    build_portfolio_context,
    build_thesis_context,
    render_backtest,
    render_full_theses,
    render_holdings_table,
    render_news,
    render_status,
)
from investment_copilot.domain.prompts.schemas import (
    HoldingComment,
    PortfolioAnalysis,
    RecommendationAction,
    RiskAlert,
    RiskAlerts,
    SeverityLevel,
    ThesisUpdate,
)
from investment_copilot.domain.prompts.templates import (
    PORTFOLIO_SYSTEM,
    PORTFOLIO_USER_TEMPLATE,
    RISK_SYSTEM,
    RISK_USER_TEMPLATE,
    THESIS_SYSTEM,
    THESIS_USER_TEMPLATE,
)

__all__ = [
    "HoldingComment",
    "MAX_NEWS_PER_TICKER",
    "MAX_NEWS_TOTAL",
    "MAX_THESIS_CHARS",
    "PORTFOLIO_SYSTEM",
    "PORTFOLIO_USER_TEMPLATE",
    "PortfolioAnalysis",
    "RISK_SYSTEM",
    "RISK_USER_TEMPLATE",
    "RecommendationAction",
    "RiskAlert",
    "RiskAlerts",
    "SeverityLevel",
    "THESIS_SYSTEM",
    "THESIS_USER_TEMPLATE",
    "ThesisUpdate",
    "build_portfolio_context",
    "build_thesis_context",
    "render_backtest",
    "render_full_theses",
    "render_holdings_table",
    "render_news",
    "render_status",
]
