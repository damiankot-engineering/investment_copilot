"""Watchlist service: YAML load/save + enriched per-item status.

Mirrors the load/save pattern from :mod:`portfolio_service` (encoding
detection, ``.bak`` rotation, friendly errors). The enrich method joins
each watchlist item with the latest OHLCV close from the cache so the
UI can show a price + distance from the target_buy_price without
hitting any provider.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from investment_copilot.domain.watchlist import Watchlist, WatchlistItem
from investment_copilot.services.data_service import DataService

logger = logging.getLogger(__name__)


class WatchlistError(RuntimeError):
    """Raised when a watchlist file cannot be loaded or validated."""


# --- YAML loading -----------------------------------------------------------


def load_watchlist(path: Path | str) -> Watchlist:
    """Load and validate a watchlist YAML file.

    Returns an empty :class:`Watchlist` if the file does not exist —
    watchlists are optional, unlike portfolios. UTF-8 / UTF-16 / CP1250
    are all detected via the shared encoding helper.
    """
    from investment_copilot.config.encoding import (
        FileEncodingError,
        detect_encoding_label,
        read_text_robust,
    )

    p = Path(path)
    if not p.is_file():
        return Watchlist(items=[])

    try:
        text = read_text_robust(p)
    except FileEncodingError as exc:
        raise WatchlistError(str(exc)) from exc

    label = detect_encoding_label(p)
    if label not in {"UTF-8", "empty"}:
        logger.warning(
            "%s was decoded as %s. Re-save it as UTF-8 (without BOM).",
            p,
            label,
        )

    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise WatchlistError(f"Invalid YAML in {p}: {exc}") from exc

    if not isinstance(raw, dict):
        raise WatchlistError(
            f"Watchlist root must be a mapping, got {type(raw).__name__}"
        )

    try:
        return Watchlist.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError -> WatchlistError
        raise WatchlistError(f"Invalid watchlist: {exc}") from exc


# --- YAML saving ------------------------------------------------------------


def save_watchlist(watchlist: Watchlist, path: Path | str) -> Path:
    """Persist ``watchlist`` to ``path`` as UTF-8 YAML, with a ``.bak`` backup."""
    p = Path(path)

    if p.exists():
        backup = p.with_suffix(p.suffix + ".bak")
        try:
            backup.write_bytes(p.read_bytes())
        except OSError as exc:
            raise WatchlistError(f"Failed to write backup {backup}: {exc}") from exc

    data = watchlist.model_dump(mode="json", exclude_none=False)
    for it in data.get("items", []):
        if not it.get("name"):
            it.pop("name", None)
        if not it.get("keywords"):
            it.pop("keywords", None)
        if not it.get("notes"):
            it.pop("notes", None)
        if it.get("target_buy_price") is None:
            it.pop("target_buy_price", None)

    try:
        text = yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    except yaml.YAMLError as exc:
        raise WatchlistError(f"Failed to serialize watchlist: {exc}") from exc

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# --- Status enrichment ------------------------------------------------------


class WatchlistItemStatus(BaseModel):
    """A watchlist item joined with the latest cached price."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    name: str | None
    added_date: date
    target_buy_price: float | None
    notes: str
    keywords: list[str]

    last_price: float | None = None
    last_price_date: date | None = None
    # Negative when current price is below target (i.e. "alert" should fire);
    # positive when still above. Null if either side missing.
    distance_to_target_pct: float | None = None
    alert: bool = False
    news_count_30d: int = Field(
        default=0,
        description="Number of cached news items for this ticker in the last 30 days.",
    )


class WatchlistStatus(BaseModel):
    """Aggregate view of the watchlist."""

    model_config = ConfigDict(frozen=True)

    as_of: datetime
    items: list[WatchlistItemStatus]
    missing_data: list[str] = Field(default_factory=list)


class WatchlistService:
    """Enrich a Watchlist with current pricing from the parquet cache."""

    def __init__(self, *, data_service: DataService) -> None:
        self._data = data_service

    @staticmethod
    def keywords_map(watchlist: Watchlist) -> dict[str, list[str]]:
        """``ticker -> news-match terms`` mapping, same shape as PortfolioService.

        Uses :attr:`WatchlistItem.news_identifiers` (ticker stem + brand name)
        so news is matched on company identity, not broad sector themes.
        """
        return {it.ticker: it.news_identifiers for it in watchlist.items}

    def current_status(self, watchlist: Watchlist) -> WatchlistStatus:
        as_of = datetime.now(timezone.utc)
        news_since = as_of - timedelta(days=30)
        missing: list[str] = []
        items: list[WatchlistItemStatus] = []
        for it in watchlist.items:
            df = self._data.load_ohlcv(it.ticker)
            last_price: float | None = None
            last_date: date | None = None
            if df is not None and not df.empty and "close" in df.columns:
                last_price, last_date = _last_close(df)
            else:
                missing.append(it.ticker)

            distance: float | None = None
            alert = False
            if last_price is not None and it.target_buy_price is not None:
                distance = (last_price / it.target_buy_price - 1.0) * 100.0
                alert = last_price <= it.target_buy_price

            try:
                news_count = len(
                    self._data.load_news(ticker=it.ticker, since=news_since)
                )
            except Exception:  # noqa: BLE001
                news_count = 0

            items.append(
                WatchlistItemStatus(
                    ticker=it.ticker,
                    name=it.name,
                    added_date=it.added_date,
                    target_buy_price=it.target_buy_price,
                    notes=it.notes,
                    keywords=list(it.keywords),
                    last_price=last_price,
                    last_price_date=last_date,
                    distance_to_target_pct=distance,
                    alert=alert,
                    news_count_30d=news_count,
                )
            )

        return WatchlistStatus(as_of=as_of, items=items, missing_data=missing)


# --- helpers ----------------------------------------------------------------


def _last_close(df: pd.DataFrame) -> tuple[float, date | None]:
    last_row = df.iloc[-1]
    last_close = float(last_row["close"])
    idx = df.index[-1]
    if hasattr(idx, "date") and callable(idx.date):
        last_date = idx.date()
    elif isinstance(idx, date):
        last_date = idx
    else:
        try:
            last_date = pd.Timestamp(idx).date()
        except Exception:  # noqa: BLE001
            last_date = None
    return last_close, last_date


__all__ = [
    "WatchlistError",
    "WatchlistItemStatus",
    "WatchlistService",
    "WatchlistStatus",
    "load_watchlist",
    "save_watchlist",
]
