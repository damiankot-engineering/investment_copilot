"""Watchlist domain models.

Tickers the user is researching but does NOT own. Stored in
``watchlist.yaml`` next to ``portfolio.yaml``; loaded/saved by
:mod:`investment_copilot.services.watchlist_service`.

Mirrors the same validation philosophy as :class:`Portfolio` — strict
schema (``extra="forbid"``), normalized tickers, no duplicates.
"""

from __future__ import annotations

from datetime import date
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from investment_copilot.domain.models import normalize_ticker
from investment_copilot.domain.news_match import derive_news_identifiers


class WatchlistItem(BaseModel):
    """A single ticker on the watchlist."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Ticker; normalized to Stooq form on load.")
    name: str | None = Field(
        default=None,
        description="Optional display name (e.g. 'CCC').",
    )
    added_date: date = Field(
        description="When the ticker was added to the watchlist.",
    )
    target_buy_price: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional alert price — UI flags the item when the current "
            "price drops to or below this level."
        ),
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Free-form research notes (Polish or English).",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Optional news-filter keywords. Defaults to the ticker stem "
            "(e.g. 'CCC' for 'ccc.pl') if empty — same convention as Holding."
        ),
    )

    # -- Validators ---------------------------------------------------------

    @field_validator("ticker")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return normalize_ticker(v)

    @field_validator("added_date")
    @classmethod
    def _not_in_future(cls, v: date) -> date:
        if v > date.today():
            raise ValueError(f"added_date {v} is in the future")
        return v

    @field_validator("keywords")
    @classmethod
    def _strip_keywords(cls, v: list[str]) -> list[str]:
        return [k.strip() for k in v if k and k.strip()]

    @property
    def effective_keywords(self) -> list[str]:
        if self.keywords:
            return list(self.keywords)
        return [self.ticker.split(".")[0].upper()]

    @property
    def news_identifiers(self) -> list[str]:
        """Company-identifying terms for relevance-filtering news.

        Mirrors :attr:`Holding.news_identifiers` — ticker stem + brand name,
        excluding broad thematic keywords. See
        :mod:`investment_copilot.domain.news_match`.
        """
        return derive_news_identifiers(self.ticker, self.name, self.keywords)


class Watchlist(BaseModel):
    """A list of tickers the user is monitoring but does not own."""

    model_config = ConfigDict(extra="forbid")

    items: list[WatchlistItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_duplicate_tickers(self) -> Watchlist:
        seen: set[str] = set()
        for it in self.items:
            if it.ticker in seen:
                raise ValueError(f"Duplicate ticker in watchlist: {it.ticker}")
            seen.add(it.ticker)
        return self

    @property
    def tickers(self) -> list[str]:
        return [it.ticker for it in self.items]

    def find(self, ticker: str) -> WatchlistItem | None:
        norm = normalize_ticker(ticker)
        return next((it for it in self.items if it.ticker == norm), None)


WATCHLIST_SCHEMA_VERSION: Final[str] = "1.0"


__all__ = [
    "WATCHLIST_SCHEMA_VERSION",
    "Watchlist",
    "WatchlistItem",
]
