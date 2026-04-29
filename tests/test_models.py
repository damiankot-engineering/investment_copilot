"""Tests for ``investment_copilot.domain.models``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investment_copilot.domain.models import (
    OHLCV_COLUMNS,
    NewsItem,
    normalize_ticker,
    resolve_benchmark,
    validate_ohlcv_frame,
)


# --- normalize_ticker -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("PKN", "pkn.pl"),
        ("pkn", "pkn.pl"),
        ("PKN.WA", "pkn.pl"),
        ("PKN.PL", "pkn.pl"),
        (" pkn.pl ", "pkn.pl"),
        ("CDR.WA", "cdr.pl"),
        ("^WIG20", "^wig20"),
        ("^wig20", "^wig20"),
    ],
)
def test_normalize_ticker(raw: str, expected: str) -> None:
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", ".pl", ".wa"])
def test_normalize_ticker_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_ticker(bad)


def test_normalize_ticker_type_error() -> None:
    with pytest.raises(TypeError):
        normalize_ticker(123)  # type: ignore[arg-type]


# --- resolve_benchmark -----------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("wig20", "^wig20"),
        ("WIG20", "^wig20"),
        (" mwig40 ", "^mwig40"),
        ("^anything", "^anything"),
    ],
)
def test_resolve_benchmark(name: str, expected: str) -> None:
    assert resolve_benchmark(name) == expected


def test_resolve_benchmark_unknown() -> None:
    with pytest.raises(ValueError):
        resolve_benchmark("not-a-real-index")


# --- validate_ohlcv_frame ---------------------------------------------------


def _frame(rows: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "open": np.arange(rows, dtype=float),
            "high": np.arange(rows, dtype=float) + 1,
            "low": np.arange(rows, dtype=float) - 1,
            "close": np.arange(rows, dtype=float) + 0.5,
            "volume": np.arange(rows, dtype=float) * 100,
        },
        index=idx,
    )


def test_validate_ohlcv_happy_path() -> None:
    out = validate_ohlcv_frame(_frame())
    assert list(out.columns) == list(OHLCV_COLUMNS)
    assert out.index.name == "date"
    assert out.index.is_monotonic_increasing


def test_validate_ohlcv_dedupes_index() -> None:
    df = _frame(3)
    df = pd.concat([df, df.iloc[[0]]])  # duplicate the first row
    out = validate_ohlcv_frame(df)
    assert out.index.is_unique


def test_validate_ohlcv_uppercase_columns() -> None:
    df = _frame()
    df.columns = [c.upper() for c in df.columns]
    out = validate_ohlcv_frame(df)
    assert list(out.columns) == list(OHLCV_COLUMNS)


def test_validate_ohlcv_missing_columns() -> None:
    df = _frame().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_ohlcv_frame(df)


def test_validate_ohlcv_requires_datetime_index() -> None:
    df = _frame().reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_ohlcv_frame(df)


def test_validate_ohlcv_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_ohlcv_frame(pd.DataFrame())


# --- NewsItem ---------------------------------------------------------------


def test_newsitem_is_frozen() -> None:
    from datetime import datetime

    n = NewsItem(
        ticker="pkn.pl",
        source="rss:test",
        title="hello",
        url="https://example.com/a",
        published_at=datetime(2024, 1, 1),
    )
    with pytest.raises(Exception):  # pydantic ValidationError on assignment
        n.title = "changed"  # type: ignore[misc]
