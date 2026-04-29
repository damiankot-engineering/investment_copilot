"""Copilot service.

Wires together:

* the user's :class:`~investment_copilot.domain.portfolio.Portfolio` and
  computed :class:`~investment_copilot.domain.portfolio.PortfolioStatus`,
* an optional :class:`~investment_copilot.domain.backtest.BacktestResult`,
* recent news from
  :class:`~investment_copilot.services.data_service.DataService`,
* an :class:`~investment_copilot.infrastructure.llm.base.LLMClient`,

into typed Polish analyses (:class:`PortfolioAnalysis`,
:class:`RiskAlerts`, :class:`ThesisUpdate`).

The service is read-only: it never persists outputs, never refreshes data,
and never mutates the portfolio. Persisting analyses is the orchestrator
or report layer's responsibility (Step 9 / Step 11).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Sequence

from investment_copilot.config.schema import LLMConfig
from investment_copilot.domain.backtest import BacktestResult
from investment_copilot.domain.models import NewsItem, normalize_ticker
from investment_copilot.domain.portfolio import Portfolio, PortfolioStatus
from investment_copilot.domain.prompts import (
    PORTFOLIO_SYSTEM,
    PORTFOLIO_USER_TEMPLATE,
    PortfolioAnalysis,
    RISK_SYSTEM,
    RISK_USER_TEMPLATE,
    RiskAlerts,
    THESIS_SYSTEM,
    THESIS_USER_TEMPLATE,
    ThesisUpdate,
    build_portfolio_context,
    build_thesis_context,
)
from investment_copilot.infrastructure.llm import LLMClient
from investment_copilot.services.data_service import DataService

logger = logging.getLogger(__name__)


DEFAULT_NEWS_DAYS_BACK: int = 14


class CopilotService:
    """Generates Polish analyses backed by a swappable LLM client."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        data_service: DataService,
        llm_config: LLMConfig,
    ) -> None:
        self._llm = llm_client
        self._data = data_service
        self._llm_cfg = llm_config

    # -- Public API ---------------------------------------------------------

    def analyze_portfolio(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        backtest: BacktestResult | None = None,
        news_days_back: int = DEFAULT_NEWS_DAYS_BACK,
    ) -> PortfolioAnalysis:
        """Holistic portfolio analysis."""
        news = self._collect_news(portfolio, days_back=news_days_back)
        context = build_portfolio_context(
            portfolio, status, backtest=backtest, news=news
        )
        user_prompt = PORTFOLIO_USER_TEMPLATE.format(context=context)
        return self._llm.complete_structured(
            system_prompt=PORTFOLIO_SYSTEM,
            user_prompt=user_prompt,
            response_schema=PortfolioAnalysis,
            model=self._llm_cfg.model_analysis,
            temperature=self._llm_cfg.temperature,
            max_tokens=self._llm_cfg.max_tokens,
        )

    def detect_risks(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        backtest: BacktestResult | None = None,
        news_days_back: int = DEFAULT_NEWS_DAYS_BACK,
    ) -> RiskAlerts:
        """Identify risk signals from current data."""
        news = self._collect_news(portfolio, days_back=news_days_back)
        context = build_portfolio_context(
            portfolio, status, backtest=backtest, news=news
        )
        user_prompt = RISK_USER_TEMPLATE.format(context=context)
        return self._llm.complete_structured(
            system_prompt=RISK_SYSTEM,
            user_prompt=user_prompt,
            response_schema=RiskAlerts,
            model=self._llm_cfg.model_analysis,
            temperature=self._llm_cfg.temperature,
            max_tokens=self._llm_cfg.max_tokens,
        )

    def update_thesis(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        ticker: str,
        news_days_back: int = DEFAULT_NEWS_DAYS_BACK,
    ) -> ThesisUpdate:
        """Reassess the investment thesis for a single position."""
        norm = normalize_ticker(ticker)
        if portfolio.find(norm) is None:
            raise ValueError(f"Ticker {norm} is not in the portfolio")

        since = datetime.now(timezone.utc) - timedelta(days=news_days_back)
        news = self._data.load_news(ticker=norm, since=since)
        context = build_thesis_context(portfolio, status, ticker=norm, news=news)
        user_prompt = THESIS_USER_TEMPLATE.format(ticker=norm, context=context)

        return self._llm.complete_structured(
            system_prompt=THESIS_SYSTEM,
            user_prompt=user_prompt,
            response_schema=ThesisUpdate,
            model=self._llm_cfg.model_analysis,
            temperature=self._llm_cfg.temperature,
            max_tokens=self._llm_cfg.max_tokens,
        )

    # -- Internal -----------------------------------------------------------

    def _collect_news(
        self,
        portfolio: Portfolio,
        *,
        days_back: int,
    ) -> Sequence[NewsItem]:
        """Pull recent news per ticker + a slice of general (untagged) news."""
        since = datetime.now(timezone.utc) - timedelta(days=max(0, days_back))
        items: list[NewsItem] = []
        for h in portfolio.holdings:
            items.extend(self._data.load_news(ticker=h.ticker, since=since))
        # General (no ticker) news, capped — useful for macro context.
        general = self._data.load_news(ticker=None, since=since, limit=20)
        items.extend(g for g in general if g.ticker is None)
        return items
