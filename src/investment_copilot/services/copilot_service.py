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
from pathlib import Path
from typing import Sequence

import pandas as pd

from investment_copilot.config.schema import LLMConfig
from investment_copilot.domain.analysis_metrics import (
    CitationRegistry,
    PortfolioMetrics,
    compute_portfolio_metrics,
    filter_unknown_citations,
)
from investment_copilot.domain.backtest import BacktestResult
from investment_copilot.domain.fundamentals import (
    FundamentalsSnapshot,
    MonitoringSnapshot,
)
from investment_copilot.domain.models import NewsItem, normalize_ticker, resolve_benchmark
from investment_copilot.domain.portfolio import Portfolio, PortfolioStatus
from investment_copilot.domain.prompts import (
    MONITORING_SYSTEM,
    MONITORING_USER_TEMPLATE,
    MonitoringReport,
    PORTFOLIO_SYSTEM,
    PORTFOLIO_USER_TEMPLATE,
    PortfolioAnalysis,
    RISK_SYSTEM,
    RISK_USER_TEMPLATE,
    RiskAlerts,
    THESIS_SYSTEM,
    THESIS_USER_TEMPLATE,
    ThesisUpdate,
    build_citation_registry,
    build_monitoring_context,
    build_portfolio_context,
    build_thesis_context,
)
from investment_copilot.infrastructure.llm import LLMClient
from investment_copilot.services.analysis_history import load_recent_reports
from investment_copilot.services.data_service import DataService

logger = logging.getLogger(__name__)


DEFAULT_NEWS_DAYS_BACK: int = 14
DEFAULT_REPORTS_DIR: Path = Path("reports")
DEFAULT_HISTORY_COUNT: int = 2


class CopilotService:
    """Generates Polish analyses backed by a swappable LLM client."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        data_service: DataService,
        llm_config: LLMConfig,
        reports_dir: Path | str = DEFAULT_REPORTS_DIR,
        history_count: int = DEFAULT_HISTORY_COUNT,
    ) -> None:
        self._llm = llm_client
        self._data = data_service
        self._llm_cfg = llm_config
        self._reports_dir = Path(reports_dir)
        self._history_count = history_count

    # -- Public API ---------------------------------------------------------

    def analyze_portfolio(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        backtest: BacktestResult | None = None,
        news_days_back: int = DEFAULT_NEWS_DAYS_BACK,
    ) -> PortfolioAnalysis:
        """Holistic portfolio analysis with quant grounding + RAG + citations."""
        news = self._collect_news(portfolio, days_back=news_days_back)
        metrics = self._compute_metrics(portfolio, status, backtest=backtest)
        history = load_recent_reports(self._reports_dir, n=self._history_count)
        registry = build_citation_registry(
            news=news, metrics=metrics, history=history
        )
        context = build_portfolio_context(
            portfolio, status,
            backtest=backtest, news=news, metrics=metrics, history=history,
        )
        user_prompt = PORTFOLIO_USER_TEMPLATE.format(context=context)
        result = self._llm.complete_structured(
            system_prompt=PORTFOLIO_SYSTEM,
            user_prompt=user_prompt,
            response_schema=PortfolioAnalysis,
            model=self._llm_cfg.model_analysis,
            temperature=self._llm_cfg.temperature,
            max_tokens=self._llm_cfg.max_tokens,
        )
        return _validate_portfolio_citations(result, registry)

    def detect_risks(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        backtest: BacktestResult | None = None,
        news_days_back: int = DEFAULT_NEWS_DAYS_BACK,
    ) -> RiskAlerts:
        """Identify risk signals from current data with citation grounding."""
        news = self._collect_news(portfolio, days_back=news_days_back)
        metrics = self._compute_metrics(portfolio, status, backtest=backtest)
        history = load_recent_reports(self._reports_dir, n=self._history_count)
        registry = build_citation_registry(
            news=news, metrics=metrics, history=history
        )
        context = build_portfolio_context(
            portfolio, status,
            backtest=backtest, news=news, metrics=metrics, history=history,
        )
        user_prompt = RISK_USER_TEMPLATE.format(context=context)
        result = self._llm.complete_structured(
            system_prompt=RISK_SYSTEM,
            user_prompt=user_prompt,
            response_schema=RiskAlerts,
            model=self._llm_cfg.model_analysis,
            temperature=self._llm_cfg.temperature,
            max_tokens=self._llm_cfg.max_tokens,
        )
        return _validate_risk_citations(result, registry)

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

    def generate_monitoring(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        fundamentals: Sequence[FundamentalsSnapshot] = (),
        news: Sequence[NewsItem] = (),
        previous_snapshot: MonitoringSnapshot | None = None,
    ) -> MonitoringReport:
        """Generate the monitoring report (HTML-source structured output)."""
        context = build_monitoring_context(
            portfolio,
            status,
            fundamentals=fundamentals,
            news=news,
            previous_snapshot=previous_snapshot,
        )
        user_prompt = MONITORING_USER_TEMPLATE.format(context=context)
        return self._llm.complete_structured(
            system_prompt=MONITORING_SYSTEM,
            user_prompt=user_prompt,
            response_schema=MonitoringReport,
            model=self._llm_cfg.model_analysis,
            temperature=self._llm_cfg.temperature,
            # 4000 tokens response — needed for richer per-company narratives
            # + portfolio_structure section. Total request stays ~11.8k <
            # 12k TPM cap thanks to lean prompt + selectively optional
            # schema fields (defaults instead of required min_length).
            max_tokens=4000,
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

    def _compute_metrics(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        backtest: BacktestResult | None = None,
    ) -> PortfolioMetrics | None:
        """Build the quant-metrics block from cached OHLCV + benchmark.

        Returns ``None`` if no OHLCV data is available for any holding
        (the block then renders as the explicit "brak danych" placeholder).
        """
        panel: dict[str, pd.DataFrame] = {}
        for h in portfolio.holdings:
            df = self._data.load_ohlcv(h.ticker)
            if df is not None and not df.empty:
                panel[h.ticker] = df
        if not panel:
            return None

        benchmark_close: pd.Series | None = None
        benchmark_symbol: str | None = None
        # Prefer the symbol used by the backtest (if a result was passed);
        # otherwise no benchmark is wired here. Beta then stays None.
        if backtest is not None and backtest.benchmark_symbol:
            benchmark_symbol = backtest.benchmark_symbol
            try:
                bdf = self._data.load_benchmark(benchmark_symbol)
            except Exception:  # noqa: BLE001
                bdf = None
            if bdf is not None and not bdf.empty and "close" in bdf.columns:
                benchmark_close = bdf["close"]

        try:
            return compute_portfolio_metrics(
                portfolio,
                status,
                ohlcv_panel=panel,
                benchmark_close=benchmark_close,
                benchmark_symbol=benchmark_symbol,
            )
        except Exception as exc:  # noqa: BLE001 - never let metrics break the run
            logger.warning("compute_portfolio_metrics failed: %s", exc)
            return None


# --- Citation validation (lenient: log + strip unknown) ---------------------


def _validate_portfolio_citations(
    analysis: PortfolioAnalysis, registry: CitationRegistry
) -> PortfolioAnalysis:
    cleaned_comments = []
    dropped: list[str] = []
    for c in analysis.holdings_comments:
        valid, drop = filter_unknown_citations(c.citations, registry)
        dropped.extend(f"{d.source_type}:{d.reference}" for d in drop)
        cleaned_comments.append(c.model_copy(update={"citations": valid}))
    if dropped:
        logger.warning(
            "Dropped %d unknown citations from portfolio analysis: %s",
            len(dropped),
            ", ".join(dropped[:10]),
        )
    return analysis.model_copy(update={"holdings_comments": cleaned_comments})


def _validate_risk_citations(
    risks: RiskAlerts, registry: CitationRegistry
) -> RiskAlerts:
    cleaned_alerts = []
    dropped: list[str] = []
    for a in risks.alerts:
        valid, drop = filter_unknown_citations(a.citations, registry)
        dropped.extend(f"{d.source_type}:{d.reference}" for d in drop)
        cleaned_alerts.append(a.model_copy(update={"citations": valid}))
    if dropped:
        logger.warning(
            "Dropped %d unknown citations from risk alerts: %s",
            len(dropped),
            ", ".join(dropped[:10]),
        )
    return risks.model_copy(update={"alerts": cleaned_alerts})
