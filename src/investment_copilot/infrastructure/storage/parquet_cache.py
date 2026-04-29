"""Parquet-based OHLCV cache.

One file per symbol. ``upsert`` merges new rows with existing data,
deduplicates on date (last write wins), and sorts the index. Whole-file
rewrites are fine at our scale (~5k rows per symbol for 20 years daily).
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd

from investment_copilot.domain.models import validate_ohlcv_frame

logger = logging.getLogger(__name__)


_FILENAME_SAFE = re.compile(r"[^a-z0-9]+")


def _safe_filename(symbol: str) -> str:
    """Map a Stooq symbol (``pkn.pl``, ``^wig20``) to a safe filename stem."""
    s = symbol.lower().replace("^", "idx_")
    s = _FILENAME_SAFE.sub("_", s).strip("_")
    return s or "unknown"


class ParquetCache:
    """OHLCV cache backed by per-symbol parquet files."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # -- Paths ---------------------------------------------------------------

    def path_for(self, symbol: str) -> Path:
        return self.base_dir / f"{_safe_filename(symbol)}.parquet"

    def has(self, symbol: str) -> bool:
        return self.path_for(symbol).is_file()

    # -- Read / write --------------------------------------------------------

    def upsert(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Merge ``df`` into the cache for ``symbol`` and return the merged frame."""
        new = validate_ohlcv_frame(df, symbol=symbol)
        path = self.path_for(symbol)
        if path.is_file():
            existing = pd.read_parquet(path)
            merged = pd.concat([existing, new])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = new
        merged.to_parquet(path)
        return merged

    def load(
        self,
        symbol: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        path = self.path_for(symbol)
        if not path.is_file():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def delete(self, symbol: str) -> bool:
        path = self.path_for(symbol)
        if path.is_file():
            path.unlink()
            return True
        return False
