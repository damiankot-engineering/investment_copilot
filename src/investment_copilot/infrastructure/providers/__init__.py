"""Concrete data provider adapters (Stooq, RSS, ...)."""

from investment_copilot.infrastructure.providers.base import (
    MarketDataProvider,
    NewsProvider,
    ProviderError,
)
from investment_copilot.infrastructure.providers.factory import (
    build_market_provider,
    build_news_providers,
)
from investment_copilot.infrastructure.providers.rss import RSSProvider
from investment_copilot.infrastructure.providers.stooq import StooqProvider
from investment_copilot.infrastructure.providers.stooq_news import StooqNewsProvider

__all__ = [
    "MarketDataProvider",
    "NewsProvider",
    "ProviderError",
    "RSSProvider",
    "StooqNewsProvider",
    "StooqProvider",
    "build_market_provider",
    "build_news_providers",
]
