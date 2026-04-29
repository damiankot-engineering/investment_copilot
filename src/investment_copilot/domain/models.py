"""Domain models and primitives used across the application.

This module defines the *lingua franca* of the system: the types that cross
layer boundaries. Anything passed between providers, storage, services, and
the CLI/API lives here.

OHLCV data is intentionally **not** a Pydantic model — wrapping every bar
would make backtests slow. Instead, the contract is a ``pandas.DataFrame``
with a known column set, validated via :func:`validate_ohlcv_frame`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# --- OHLCV contract ----------------------------------------------------------

#: Required columns for an OHLCV frame, in canonical order.
OHLCV_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")


def validate_ohlcv_frame(df: pd.DataFrame, *, symbol: str = "<unknown>") -> pd.DataFrame:
    """Validate and normalize an OHLCV DataFrame.

    Returns a copy with:
    * exactly the columns in :data:`OHLCV_COLUMNS`
    * a sorted, unique :class:`~pandas.DatetimeIndex` named ``"date"``
    * numeric dtypes
    """
    if df is None or df.empty:
        raise ValueError(f"OHLCV frame for {symbol} is empty")

    cols = {c.lower() for c in df.columns}
    missing = set(OHLCV_COLUMNS) - cols
    if missing:
        raise ValueError(f"OHLCV frame for {symbol} missing columns: {sorted(missing)}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"OHLCV frame for {symbol} must have a DatetimeIndex")

    # Lowercase column names to be safe and pick canonical order.
    out = df.rename(columns={c: c.lower() for c in df.columns})[list(OHLCV_COLUMNS)].copy()
    out.index.name = "date"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    for col in OHLCV_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# --- Ticker / benchmark normalization ---------------------------------------

#: Friendly benchmark name -> Stooq symbol.
BENCHMARK_SYMBOLS: Final[dict[str, str]] = {
    "wig": "^wig",
    "wig20": "^wig20",
    "mwig40": "^mwig40",
    "swig80": "^swig80",
    "wig30": "^wig30",
}


def normalize_ticker(ticker: str) -> str:
    """Normalize a ticker string to Stooq convention.

    Examples
    --------
    >>> normalize_ticker("PKN")
    'pkn.pl'
    >>> normalize_ticker("PKN.WA")
    'pkn.pl'
    >>> normalize_ticker(" pkn.pl ")
    'pkn.pl'
    >>> normalize_ticker("^WIG20")
    '^wig20'
    """
    if not isinstance(ticker, str):
        raise TypeError(f"ticker must be str, got {type(ticker).__name__}")

    t = ticker.strip().lower()
    if not t:
        raise ValueError("Empty ticker")

    if t.startswith("^"):
        return t

    for suffix in (".wa", ".pl"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            break

    if not t:
        raise ValueError(f"Ticker '{ticker}' has no symbol body")

    return f"{t}.pl"


def resolve_benchmark(name: str) -> str:
    """Resolve a friendly benchmark name (e.g. ``wig20``) to a Stooq symbol."""
    n = name.strip().lower()
    if n in BENCHMARK_SYMBOLS:
        return BENCHMARK_SYMBOLS[n]
    if n.startswith("^"):
        return n
    raise ValueError(f"Unknown benchmark: {name!r}")


# --- News -------------------------------------------------------------------


class NewsItem(BaseModel):
    """A single news article, source-agnostic."""

    model_config = ConfigDict(frozen=True)

    ticker: str | None = Field(
        default=None,
        description="Normalized ticker the item is associated with, if any.",
    )
    source: str = Field(description="Provider or feed name, e.g. 'rss:bankier'.")
    title: str
    url: str
    published_at: datetime
    summary: str | None = None
