"""Orchestrator — composes services into end-to-end pipelines.

The Orchestrator is the only place that knows the *order of operations*
for a given user-facing action. Services know their single domain;
pipelines know how the domains chain together.

Pipelines
---------
* :meth:`Orchestrator.update_data` — refresh OHLCV + news + benchmark
* :meth:`Orchestrator.run_analysis` — current status + AI analysis + risks
* :meth:`Orchestrator.backtest` — run a strategy on the portfolio
* :meth:`Orchestrator.generate_report` — full run; calls the others and
  writes a Markdown report to disk

Errors that aren't fatal to the pipeline (e.g. an LLM failure during a
report run) are captured into ``warnings`` lists on the result, so
downstream sections still render. Errors that *are* fatal (e.g. no data
in cache during backtest) propagate.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

from investment_copilot.domain.backtest import BacktestError, BacktestResult
from investment_copilot.domain.portfolio import Portfolio
from investment_copilot.infrastructure.llm import LLMError
from investment_copilot.services.data_service import RefreshReport
from investment_copilot.services.pipeline_results import AnalysisBundle, FullReport
from investment_copilot.services.container import ServiceContainer
from investment_copilot.services.report_service import ReportService

logger = logging.getLogger(__name__)


class Orchestrator:
    """Composes services into named pipelines."""

    def __init__(
        self,
        container: ServiceContainer,
        *,
        report_service: ReportService | None = None,
        reports_dir: Path | str = "reports",
    ) -> None:
        self._container = container
        self._reports = report_service or ReportService(output_dir=reports_dir)

    # -- Pipeline 1: update-data --------------------------------------------

    def update_data(
        self,
        portfolio: Portfolio,
        *,
        start: date | None = None,
        end: date | None = None,
        news_days_back: int = 14,
    ) -> RefreshReport:
        """Refresh OHLCV (per holding + benchmark) and news caches."""
        cfg = self._container.config
        data = self._container.data_service
        start_date = start or cfg.backtest.start_date
        report = RefreshReport()

        # OHLCV per ticker (DataService.refresh_ohlcv handles per-ticker errors)
        try:
            updated = data.refresh_ohlcv(
                portfolio.tickers, start=start_date, end=end
            )
            report.ohlcv_updated.update(updated)
        except Exception as exc:  # provider-level catastrophic failure
            logger.exception("update_data: OHLCV refresh aborted")
            for t in portfolio.tickers:
                report.ohlcv_failed.setdefault(t, str(exc))

        # Benchmark
        try:
            symbol, rows = data.refresh_benchmark(
                cfg.backtest.benchmark, start=start_date, end=end
            )
            report.benchmark_symbol = symbol
            report.benchmark_rows = rows
        except Exception as exc:
            logger.warning("update_data: benchmark refresh failed: %s", exc)
            report.news_failed.append(f"benchmark: {exc}")

        # News
        try:
            since = datetime.now(timezone.utc).replace(
                tzinfo=timezone.utc
            )
            from datetime import timedelta

            since = since - timedelta(days=max(0, news_days_back))
            keywords = self._container.portfolio_service.keywords_map(portfolio)
            inserted = data.refresh_news(since, keywords_by_ticker=keywords)
            report.news_inserted = inserted
        except Exception as exc:
            logger.warning("update_data: news refresh failed: %s", exc)
            report.news_failed.append(f"news: {exc}")

        return report

    # -- Pipeline 2: run-analysis ------------------------------------------

    def run_analysis(
        self,
        portfolio: Portfolio,
        *,
        include_risks: bool = True,
        backtest_for_context: BacktestResult | None = None,
        news_days_back: int = 14,
    ) -> AnalysisBundle:
        """Compute current status and run AI analysis (and optional risks)."""
        warnings: list[str] = []
        status = self._container.portfolio_service.current_status(portfolio)

        analysis = None
        try:
            analysis = self._container.copilot_service.analyze_portfolio(
                portfolio,
                status,
                backtest=backtest_for_context,
                news_days_back=news_days_back,
            )
        except LLMError as exc:
            logger.warning("Portfolio analysis failed: %s", exc)
            warnings.append(f"Portfolio analysis failed: {exc}")

        risks = None
        if include_risks:
            try:
                risks = self._container.copilot_service.detect_risks(
                    portfolio,
                    status,
                    backtest=backtest_for_context,
                    news_days_back=news_days_back,
                )
            except LLMError as exc:
                logger.warning("Risk analysis failed: %s", exc)
                warnings.append(f"Risk analysis failed: {exc}")

        return AnalysisBundle(
            status=status,
            analysis=analysis,
            risks=risks,
            warnings=warnings,
            generated_at=datetime.now(timezone.utc),
        )

    # -- Pipeline 3: backtest ----------------------------------------------

    def backtest(
        self,
        portfolio: Portfolio,
        *,
        strategy_name: str,
        start: date | None = None,
        end: date | None = None,
        include_benchmark: bool = True,
    ) -> BacktestResult:
        """Run a backtest. Errors propagate — there's nothing to gracefully degrade."""
        return self._container.backtest_service.run(
            portfolio,
            strategy_name=strategy_name,
            start=start,
            end=end,
            include_benchmark=include_benchmark,
        )

    # -- Pipeline 4: generate-report ---------------------------------------

    def generate_report(
        self,
        portfolio: Portfolio,
        *,
        strategy_name: str | None = None,
        news_days_back: int = 14,
        filename: str | None = None,
    ) -> FullReport:
        """Full pipeline: status + backtest + AI + write Markdown to disk."""
        warnings: list[str] = []

        # 1. Status (must succeed; if this fails the rest is meaningless)
        status = self._container.portfolio_service.current_status(portfolio)

        # 2. Backtest (best effort)
        backtest_result: BacktestResult | None = None
        if strategy_name:
            try:
                backtest_result = self.backtest(
                    portfolio, strategy_name=strategy_name
                )
            except BacktestError as exc:
                logger.warning("Backtest skipped: %s", exc)
                warnings.append(f"Backtest skipped: {exc}")

        # 3. AI analysis + risks (best effort)
        analysis = None
        try:
            analysis = self._container.copilot_service.analyze_portfolio(
                portfolio,
                status,
                backtest=backtest_result,
                news_days_back=news_days_back,
            )
        except LLMError as exc:
            logger.warning("Portfolio analysis failed: %s", exc)
            warnings.append(f"Portfolio analysis failed: {exc}")

        risks = None
        try:
            risks = self._container.copilot_service.detect_risks(
                portfolio,
                status,
                backtest=backtest_result,
                news_days_back=news_days_back,
            )
        except LLMError as exc:
            logger.warning("Risk analysis failed: %s", exc)
            warnings.append(f"Risk analysis failed: {exc}")

        # 4. Persist Markdown
        report_path = self._reports.write(
            portfolio=portfolio,
            status=status,
            backtest=backtest_result,
            analysis=analysis,
            risks=risks,
            warnings=warnings,
            filename=filename,
        )

        return FullReport(
            status=status,
            backtest=backtest_result,
            analysis=analysis,
            risks=risks,
            report_path=report_path,
            warnings=warnings,
            generated_at=datetime.now(timezone.utc),
        )
