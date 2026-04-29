"""Build concrete provider instances from :class:`ProvidersConfig`.

This is the single place that knows the mapping ``config name -> class``.
Adding a new provider means: implement the adapter, add a literal to the
config schema, add a branch here.
"""

from __future__ import annotations

import logging
from typing import Sequence

from investment_copilot.config.schema import ProvidersConfig
from investment_copilot.infrastructure.providers.base import (
    MarketDataProvider,
    NewsProvider,
)
from investment_copilot.infrastructure.providers.rss import RSSProvider
from investment_copilot.infrastructure.providers.stooq import StooqProvider
from investment_copilot.infrastructure.providers.stooq_news import StooqNewsProvider

logger = logging.getLogger(__name__)


def build_market_provider(cfg: ProvidersConfig) -> MarketDataProvider:
    """Construct the market-data provider selected by config."""
    if cfg.market_data == "stooq":
        return StooqProvider()
    raise ValueError(f"Unsupported market_data provider: {cfg.market_data!r}")


def build_news_providers(cfg: ProvidersConfig) -> Sequence[NewsProvider]:
    """Construct the ordered list of news providers selected by config.

    Unsupported names are skipped with a warning rather than raising — the
    pipeline can still produce useful output if e.g. NewsAPI key is missing.
    """
    providers: list[NewsProvider] = []
    for name in cfg.news:
        if name == "stooq":
            providers.append(StooqNewsProvider())
        elif name == "rss":
            if not cfg.rss_feeds:
                logger.warning("RSS provider requested but no feeds configured")
                continue
            providers.append(RSSProvider(cfg.rss_feeds))
        elif name == "newsapi":
            logger.warning(
                "NewsAPI provider is not implemented in v1; skipping. "
                "Remove it from `providers.news` to silence this warning."
            )
            continue
        else:  # pragma: no cover - schema literal forbids unknown names
            logger.warning("Unknown news provider %r; skipping", name)
    return providers
