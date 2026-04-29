"""Service container — the single place that wires everything from config.

The container is a frozen dataclass holding constructed services and their
shared infrastructure (storage, providers). Both the CLI and the future
FastAPI app build the application by calling :func:`build_container` once
at startup; nothing else constructs services directly.

Why a dataclass and not a DI framework
--------------------------------------
* The graph is small and shallow; explicit beats clever here.
* Tests can swap any field via ``dataclasses.replace`` or by constructing
  a custom container directly.
* In FastAPI, a single ``Depends(get_container)`` provider gives every
  route the same wired services without re-instantiating providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from investment_copilot.config.schema import AppConfig
from investment_copilot.infrastructure.llm import LLMClient, build_llm_client
from investment_copilot.infrastructure.providers import (
    MarketDataProvider,
    NewsProvider,
    build_market_provider,
    build_news_providers,
)
from investment_copilot.infrastructure.storage import ParquetCache, SQLiteStore
from investment_copilot.services.backtest_service import BacktestService
from investment_copilot.services.copilot_service import CopilotService
from investment_copilot.services.data_service import DataService
from investment_copilot.services.portfolio_service import PortfolioService


@dataclass(slots=True, frozen=True)
class ServiceContainer:
    """Holds all wired services and their shared infrastructure."""

    config: AppConfig

    # Infrastructure (exposed for advanced use; most callers use services)
    sqlite_store: SQLiteStore
    parquet_cache: ParquetCache
    market_provider: MarketDataProvider
    news_providers: Sequence[NewsProvider]
    llm_client: LLMClient

    # Services
    data_service: DataService
    portfolio_service: PortfolioService
    backtest_service: BacktestService
    copilot_service: CopilotService


def build_container(config: AppConfig) -> ServiceContainer:
    """Construct all services for ``config``.

    Storage directories are created on first use by ``SQLiteStore`` and
    ``ParquetCache`` themselves, so this function has no side effects on
    the filesystem beyond what those constructors do.
    """
    sqlite_store = SQLiteStore(config.storage.sqlite_path)
    parquet_cache = ParquetCache(config.storage.parquet_dir)

    market_provider = build_market_provider(config.providers)
    news_providers = build_news_providers(config.providers)
    llm_client = build_llm_client(config.llm)

    data_service = DataService(
        market_provider=market_provider,
        news_providers=news_providers,
        sqlite_store=sqlite_store,
        parquet_cache=parquet_cache,
    )
    portfolio_service = PortfolioService(data_service=data_service)
    backtest_service = BacktestService(
        data_service=data_service,
        backtest_config=config.backtest,
        strategies_config=config.strategies,
    )
    copilot_service = CopilotService(
        llm_client=llm_client,
        data_service=data_service,
        llm_config=config.llm,
    )

    return ServiceContainer(
        config=config,
        sqlite_store=sqlite_store,
        parquet_cache=parquet_cache,
        market_provider=market_provider,
        news_providers=news_providers,
        llm_client=llm_client,
        data_service=data_service,
        portfolio_service=portfolio_service,
        backtest_service=backtest_service,
        copilot_service=copilot_service,
    )
