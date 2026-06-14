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
from investment_copilot.infrastructure.providers.biznesradar import (
    BiznesRadarProvider,
)
from investment_copilot.infrastructure.providers.stooq import StooqProvider
from investment_copilot.infrastructure.providers.stooq_fundamentals import (
    StooqFundamentalsProvider,
)
from investment_copilot.infrastructure.providers.stooq_news import StooqNewsProvider
from investment_copilot.infrastructure.providers.yahoo import YahooProvider

__all__ = [
    "BiznesRadarProvider",
    "MarketDataProvider",
    "NewsProvider",
    "ProviderError",
    "RSSProvider",
    "StooqFundamentalsProvider",
    "StooqNewsProvider",
    "StooqProvider",
    "YahooProvider",
    "build_market_provider",
    "build_news_providers",
]
