"""Tests for DataService — composition of providers + storage with fakes."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from investment_copilot.domain.models import NewsItem
from investment_copilot.infrastructure.providers.base import ProviderError
from investment_copilot.infrastructure.storage import ParquetCache, SQLiteStore
from investment_copilot.services.data_service import DataService


# --- Fakes ------------------------------------------------------------------


class FakeMarketProvider:
    name = "fake-market"

    def __init__(self) -> None:
        self.calls: list[tuple[str, date, date | None]] = []
        self.fail_for: set[str] = set()

    def _make(self, periods: int = 5) -> pd.DataFrame:
        idx = pd.date_range("2024-01-02", periods=periods, freq="B")
        return pd.DataFrame(
            {
                "open": np.arange(periods, dtype=float) + 100,
                "high": np.arange(periods, dtype=float) + 101,
                "low": np.arange(periods, dtype=float) + 99,
                "close": np.arange(periods, dtype=float) + 100.5,
                "volume": np.full(periods, 1000.0),
            },
            index=idx,
        )

    def fetch_ohlcv(self, ticker: str, start: date, end: date | None = None) -> pd.DataFrame:
        self.calls.append((ticker, start, end))
        if ticker in self.fail_for:
            raise ProviderError(f"fail {ticker}")
        return self._make()

    def fetch_benchmark(
        self, benchmark: str, start: date, end: date | None = None
    ) -> pd.DataFrame:
        return self._make()


class FakeNewsProvider:
    name = "fake-news"

    def __init__(self, items: list[NewsItem] | None = None) -> None:
        self.items = items or []
        self.calls: list[dict] = []

    def fetch_news(
        self,
        since: datetime,
        *,
        ticker: str | None = None,
        keywords: list[str] | None = None,
    ) -> list[NewsItem]:
        self.calls.append({"since": since, "ticker": ticker, "keywords": keywords})
        return list(self.items)


# --- Tests ------------------------------------------------------------------


def _make_service(tmp_path, news_items: list[NewsItem] | None = None) -> tuple[
    DataService, FakeMarketProvider, FakeNewsProvider
]:
    market = FakeMarketProvider()
    news = FakeNewsProvider(news_items or [])
    svc = DataService(
        market_provider=market,
        news_providers=[news],
        sqlite_store=SQLiteStore(tmp_path / "cache.db"),
        parquet_cache=ParquetCache(tmp_path / "ohlcv"),
    )
    return svc, market, news


def test_refresh_ohlcv_normalizes_and_caches(tmp_path) -> None:
    svc, market, _ = _make_service(tmp_path)

    result = svc.refresh_ohlcv(["PKN", "CDR.WA"], start=date(2024, 1, 1))

    assert result == {"pkn.pl": 5, "cdr.pl": 5}
    # Tickers were normalized before being passed to the provider
    assert {c[0] for c in market.calls} == {"pkn.pl", "cdr.pl"}

    # And they're loadable from cache
    df = svc.load_ohlcv("pkn", start=date(2024, 1, 1))
    assert len(df) == 5


def test_refresh_ohlcv_skips_failures(tmp_path) -> None:
    svc, market, _ = _make_service(tmp_path)
    market.fail_for = {"cdr.pl"}

    result = svc.refresh_ohlcv(["PKN", "CDR"], start=date(2024, 1, 1))

    assert result == {"pkn.pl": 5}  # cdr skipped, pkn succeeded


def test_refresh_benchmark(tmp_path) -> None:
    svc, _, _ = _make_service(tmp_path)
    symbol, rows = svc.refresh_benchmark("wig20", start=date(2024, 1, 1))
    assert symbol == "^wig20"
    assert rows == 5
    assert not svc.load_benchmark("wig20").empty


def test_refresh_news_persists_and_stamps_ticker(tmp_path) -> None:
    items = [
        NewsItem(
            ticker=None,  # provider didn't tag with ticker
            source="rss:test",
            title="Orlen ogłasza wyniki",
            url="https://example.com/1",
            published_at=datetime(2024, 4, 1, tzinfo=timezone.utc),
        )
    ]
    svc, _, news = _make_service(tmp_path, items)

    inserted = svc.refresh_news(
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        keywords_by_ticker={"PKN": ["Orlen"]},
    )

    assert inserted == 1
    # Provider was queried with the normalized ticker + keywords
    assert news.calls[0]["ticker"] == "pkn.pl"
    assert news.calls[0]["keywords"] == ["Orlen"]
    # And the loaded news now has the ticker stamped on it
    loaded = svc.load_news(ticker="pkn")
    assert len(loaded) == 1
    assert loaded[0].ticker == "pkn.pl"


def test_refresh_news_provider_exception_is_isolated(tmp_path) -> None:
    class BoomNews:
        name = "boom"

        def fetch_news(self, *args, **kwargs):
            raise RuntimeError("boom")

    market = FakeMarketProvider()
    boom = BoomNews()
    good = FakeNewsProvider(
        [
            NewsItem(
                ticker="pkn.pl",
                source="rss:good",
                title="ok",
                url="https://example.com/ok",
                published_at=datetime(2024, 4, 1, tzinfo=timezone.utc),
            )
        ]
    )
    svc = DataService(
        market_provider=market,
        news_providers=[boom, good],  # boom first; good must still run
        sqlite_store=SQLiteStore(tmp_path / "cache.db"),
        parquet_cache=ParquetCache(tmp_path / "ohlcv"),
    )

    inserted = svc.refresh_news(
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        keywords_by_ticker={"PKN": ["ok"]},
    )
    assert inserted == 1
