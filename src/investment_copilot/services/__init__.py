"""Application services. The seam between domain/infrastructure and entrypoints."""

from investment_copilot.services.backtest_service import BacktestService
from investment_copilot.services.container import ServiceContainer, build_container
from investment_copilot.services.company_report_service import CompanyReportService
from investment_copilot.services.copilot_service import CopilotService
from investment_copilot.services.data_service import DataService, RefreshReport
from investment_copilot.services.monitoring_service import MonitoringService
from investment_copilot.services.pipeline_results import (
    AnalysisBundle,
    FullReport,
    MonitoringRunResult,
)
from investment_copilot.services.portfolio_service import (
    PortfolioError,
    PortfolioService,
    load_portfolio,
    save_portfolio,
)
from investment_copilot.services.report_service import ReportService
from investment_copilot.services.watchlist_service import (
    WatchlistError,
    WatchlistService,
    WatchlistStatus,
    load_watchlist,
    save_watchlist,
)

__all__ = [
    "AnalysisBundle",
    "BacktestService",
    "CompanyReportService",
    "CopilotService",
    "DataService",
    "FullReport",
    "MonitoringRunResult",
    "MonitoringService",
    "PortfolioError",
    "PortfolioService",
    "RefreshReport",
    "ReportService",
    "ServiceContainer",
    "WatchlistError",
    "WatchlistService",
    "WatchlistStatus",
    "build_container",
    "load_portfolio",
    "load_watchlist",
    "save_portfolio",
    "save_watchlist",
]
