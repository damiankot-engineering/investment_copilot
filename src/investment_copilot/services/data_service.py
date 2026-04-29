"""High-level data orchestration.

The :class:`DataService` is the only thing in the application that knows
*both* about providers and about storage. Higher layers (PortfolioService,
BacktestService, CopilotService) talk to ``DataService`` only.

This is the seam where future API endpoints will plug in: every method
takes a typed input and returns a typed output, free of CLI/HTTP concerns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Mapping, Sequence

import pandas as pd

from investment_copilot.domain.models import NewsItem, normalize_ticker, resolve_benchmark
from investment_copilot.infrastructure.providers.base import (
    MarketDataProvider,
    NewsProvider,
    ProviderError,
)
from investment_copilot.infrastructure.storage import ParquetCache, SQLiteStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RefreshReport:
    """Summary of a refresh run, returned to callers (CLI, future API).

    Populated by the orchestrator from the return values of individual
    DataService calls. Lives in the service module historically; the
    orchestrator now does the assembly.
    """

    ohlcv_updated: dict[str, int] = field(default_factory=dict)   # symbol -> rows merged
    ohlcv_failed: dict[str, str] = field(default_factory=dict)    # symbol -> error msg
    benchmark_symbol: str | None = None
    benchmark_rows: int = 0
    news_inserted: int = 0
    news_failed: list[str] = field(default_factory=list)


class DataService:
    """Orchestrates providers and storage. Stateless except for its dependencies."""

    def __init__(
        self,
        *,
        market_provider: MarketDataProvider,
        news_providers: Sequence[NewsProvider],
        sqlite_store: SQLiteStore,
        parquet_cache: ParquetCache,
    ) -> None:
        self._market = market_provider
        self._news_providers = list(news_providers)
        self._sqlite = sqlite_store
        self._parquet = parquet_cache

    # -- OHLCV --------------------------------------------------------------

    def refresh_ohlcv(
        self,
        tickers: Sequence[str],
        *,
        start: date,
        end: date | None = None,
    ) -> dict[str, int]:
        """Fetch OHLCV for each ticker and upsert into the parquet cache.

        Returns a map ``symbol -> rows_after_merge`` for successfully fetched
        symbols. Failures are logged and recorded but don't abort the run.
        """
        results: dict[str, int] = {}
        for ticker in tickers:
            symbol = normalize_ticker(ticker)
            try:
                df = self._market.fetch_ohlcv(symbol, start=start, end=end)
            except ProviderError as exc:
                logger.warning("OHLCV fetch failed for %s: %s", symbol, exc)
                continue
            merged = self._parquet.upsert(symbol, df)
            self._sqlite.set_ohlcv_meta(
                symbol,
                last_fetched_at=datetime.now(timezone.utc),
                earliest_date=str(merged.index.min().date()) if not merged.empty else None,
                latest_date=str(merged.index.max().date()) if not merged.empty else None,
            )
            results[symbol] = len(merged)
            logger.info("OHLCV refreshed: %s (%d rows)", symbol, len(merged))
        return results

    def refresh_benchmark(
        self,
        benchmark: str,
        *,
        start: date,
        end: date | None = None,
    ) -> tuple[str, int]:
        """Fetch and cache benchmark OHLCV. Returns ``(symbol, rows)``."""
        symbol = resolve_benchmark(benchmark)
        df = self._market.fetch_benchmark(benchmark, start=start, end=end)
        merged = self._parquet.upsert(symbol, df)
        self._sqlite.set_ohlcv_meta(
            symbol,
            last_fetched_at=datetime.now(timezone.utc),
            earliest_date=str(merged.index.min().date()) if not merged.empty else None,
            latest_date=str(merged.index.max().date()) if not merged.empty else None,
        )
        return symbol, len(merged)

    def load_ohlcv(
        self,
        ticker: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        return self._parquet.load(normalize_ticker(ticker), start=start, end=end)

    def load_benchmark(
        self,
        benchmark: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        return self._parquet.load(resolve_benchmark(benchmark), start=start, end=end)

    # -- News ---------------------------------------------------------------

    def refresh_news(
        self,
        since: datetime,
        *,
        keywords_by_ticker: Mapping[str, Sequence[str]] | None = None,
    ) -> int:
        """Fetch news from all configured providers and persist to SQLite.

        For each ticker in ``keywords_by_ticker``, every provider is queried
        with that ticker + keywords. If ``keywords_by_ticker`` is ``None`` or
        empty, providers are queried for general news (no ticker association).

        Returns the total number of inserted (newly seen) news items.
        """
        total_inserted = 0
        targets: list[tuple[str | None, list[str]]] = (
            [(normalize_ticker(t), list(kws)) for t, kws in keywords_by_ticker.items()]
            if keywords_by_ticker
            else [(None, [])]
        )

        for ticker, kws in targets:
            collected: list[NewsItem] = []
            for provider in self._news_providers:
                try:
                    items = provider.fetch_news(since, ticker=ticker, keywords=kws)
                except Exception as exc:  # provider must not abort the loop
                    logger.warning(
                        "News provider %s failed for %s: %s",
                        provider.name,
                        ticker,
                        exc,
                    )
                    continue
                # Stamp ticker on items that lack one (general feeds).
                if ticker is not None:
                    items = [
                        i if i.ticker else i.model_copy(update={"ticker": ticker})
                        for i in items
                    ]
                collected.extend(items)
            inserted = self._sqlite.upsert_news(collected)
            total_inserted += inserted
            logger.info(
                "News refresh: ticker=%s collected=%d inserted=%d",
                ticker,
                len(collected),
                inserted,
            )
        return total_inserted

    def load_news(
        self,
        *,
        ticker: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[NewsItem]:
        norm = normalize_ticker(ticker) if ticker else None
        return self._sqlite.load_news(ticker=norm, since=since, limit=limit)
