"""Tests for SQLite news store and Parquet OHLCV cache."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from investment_copilot.domain.models import NewsItem
from investment_copilot.infrastructure.storage import ParquetCache, SQLiteStore


# --- SQLite -----------------------------------------------------------------


def _news(url: str, ticker: str = "pkn.pl", *, when: datetime | None = None) -> NewsItem:
    return NewsItem(
        ticker=ticker,
        source="rss:test",
        title=f"news for {ticker} @ {url}",
        url=url,
        published_at=when or datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
    )


def test_sqlite_upsert_and_load(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "cache.db")

    inserted = store.upsert_news([_news("https://x/1"), _news("https://x/2")])
    assert inserted == 2

    # Re-insert one duplicate -> ignored
    inserted2 = store.upsert_news([_news("https://x/1"), _news("https://x/3")])
    assert inserted2 == 1

    rows = store.load_news()
    assert {r.url for r in rows} == {"https://x/1", "https://x/2", "https://x/3"}


def test_sqlite_load_filters(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "cache.db")
    store.upsert_news(
        [
            _news("https://a", ticker="pkn.pl", when=datetime(2024, 1, 1, tzinfo=timezone.utc)),
            _news("https://b", ticker="cdr.pl", when=datetime(2024, 2, 1, tzinfo=timezone.utc)),
            _news("https://c", ticker="pkn.pl", when=datetime(2024, 3, 1, tzinfo=timezone.utc)),
        ]
    )

    pkn = store.load_news(ticker="pkn.pl")
    assert {r.url for r in pkn} == {"https://a", "https://c"}

    recent = store.load_news(since=datetime(2024, 2, 15, tzinfo=timezone.utc))
    assert {r.url for r in recent} == {"https://c"}

    limited = store.load_news(limit=1)
    assert len(limited) == 1


def test_sqlite_ohlcv_meta(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "cache.db")
    assert store.get_ohlcv_meta("pkn.pl") is None

    store.set_ohlcv_meta(
        "pkn.pl",
        last_fetched_at=datetime(2024, 4, 1, tzinfo=timezone.utc),
        earliest_date="2020-01-02",
        latest_date="2024-04-01",
    )
    meta = store.get_ohlcv_meta("pkn.pl")
    assert meta is not None
    assert meta["earliest_date"] == "2020-01-02"


# --- Parquet ----------------------------------------------------------------


def _ohlcv(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 110, periods),
            "high": np.linspace(101, 111, periods),
            "low": np.linspace(99, 109, periods),
            "close": np.linspace(100.5, 110.5, periods),
            "volume": np.full(periods, 1000.0),
        },
        index=idx,
    )


def test_parquet_upsert_creates_file(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    df = _ohlcv("2024-01-02", 5)

    merged = cache.upsert("pkn.pl", df)

    assert cache.has("pkn.pl")
    assert len(merged) == 5


def test_parquet_upsert_merges_and_dedupes(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    cache.upsert("pkn.pl", _ohlcv("2024-01-02", 5))

    overlapping = _ohlcv("2024-01-04", 5)  # 3 overlapping bdays + 2 new
    merged = cache.upsert("pkn.pl", overlapping)

    # Business days from 2024-01-02 ({2,3,4,5,8}) ∪ from 2024-01-04 ({4,5,8,9,10}) = 7
    assert len(merged) == 7
    assert merged.index.is_unique
    assert merged.index.is_monotonic_increasing


def test_parquet_load_with_date_range(tmp_path) -> None:
    from datetime import date

    cache = ParquetCache(tmp_path)
    cache.upsert("pkn.pl", _ohlcv("2024-01-02", 10))

    sliced = cache.load("pkn.pl", start=date(2024, 1, 5), end=date(2024, 1, 10))
    assert sliced.index.min() >= pd.Timestamp("2024-01-05")
    assert sliced.index.max() <= pd.Timestamp("2024-01-10")


def test_parquet_load_missing_returns_empty(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    assert cache.load("nope.pl").empty


def test_parquet_safe_filename_for_index(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    cache.upsert("^wig20", _ohlcv("2024-01-02", 3))
    # File must exist with a sanitized name (no '^').
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    assert "^" not in files[0].name


def test_parquet_rejects_invalid_frame(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    bad = _ohlcv("2024-01-02", 3).drop(columns=["volume"])
    with pytest.raises(ValueError):
        cache.upsert("pkn.pl", bad)
