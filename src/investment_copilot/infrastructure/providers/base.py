"""Protocols (interfaces) for the data layer's provider adapters.

Concrete providers (Stooq, RSS, NewsAPI, …) live alongside this module and
are wired in at runtime by the data service. The rest of the application
depends only on these Protocols, never on a concrete class.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, runtime_checkable

import pandas as pd

from investment_copilot.domain.models import NewsItem


class ProviderError(RuntimeError):
    """Raised when a provider cannot satisfy a request.

    Providers should raise this for transient *and* terminal failures; the
    service layer decides whether to retry, fall back, or surface to the user.
    """


@runtime_checkable
class MarketDataProvider(Protocol):
    """Provides daily OHLCV data for tickers and benchmarks."""

    name: str

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Return an OHLCV frame for ``ticker`` between ``start`` and ``end``.

        The returned frame must satisfy
        :func:`investment_copilot.domain.models.validate_ohlcv_frame`.
        """
        ...

    def fetch_benchmark(
        self,
        benchmark: str,
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Like :meth:`fetch_ohlcv` but for indices (e.g. ``"wig20"``)."""
        ...


@runtime_checkable
class NewsProvider(Protocol):
    """Provides news items, optionally filtered."""

    name: str

    def fetch_news(
        self,
        since: datetime,
        *,
        ticker: str | None = None,
        keywords: list[str] | None = None,
    ) -> list[NewsItem]:
        """Return news items published at or after ``since``.

        Providers that support per-ticker queries should use ``ticker``;
        otherwise they fetch general news and filter by ``keywords``.
        Providers that cannot honor the filter must filter client-side.
        """
        ...
