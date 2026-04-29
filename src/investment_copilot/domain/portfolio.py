"""Portfolio domain models.

Inputs (from ``portfolio.yaml``):
    * :class:`Holding` — a single position
    * :class:`Portfolio` — the user's tracked positions

Computed outputs (returned by services):
    * :class:`HoldingStatus` — a :class:`Holding` enriched with current price / PnL
    * :class:`PortfolioStatus` — aggregate view of the portfolio

Inputs are validated strictly: ``extra="forbid"`` rejects unknown YAML keys,
and a model validator forbids duplicate tickers. Ticker normalization
happens at field-validation time, so users may write ``PKN``, ``PKN.WA``,
or ``pkn.pl`` interchangeably.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from investment_copilot.domain.models import normalize_ticker


# --- Inputs (YAML) ----------------------------------------------------------


class Holding(BaseModel):
    """A single tracked position."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Ticker; normalized to Stooq form on load.")
    shares: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    entry_date: date
    thesis: str = Field(min_length=1)
    name: str | None = Field(
        default=None,
        description="Optional display name (e.g. 'PKN Orlen').",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Optional keywords for news filtering. If empty, defaults to the "
            "ticker stem (e.g. 'PKN'). Recommended: provide brand-style "
            "keywords matching how the company appears in headlines."
        ),
    )

    # -- Validators ---------------------------------------------------------

    @field_validator("ticker")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return normalize_ticker(v)

    @field_validator("entry_date")
    @classmethod
    def _not_in_future(cls, v: date) -> date:
        if v > date.today():
            raise ValueError(f"entry_date {v} is in the future")
        return v

    @field_validator("keywords")
    @classmethod
    def _strip_keywords(cls, v: list[str]) -> list[str]:
        return [k.strip() for k in v if k and k.strip()]

    # -- Derived properties -------------------------------------------------

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price

    @property
    def effective_keywords(self) -> list[str]:
        """Keywords used for news filtering.

        Returns user-provided keywords if any; otherwise falls back to the
        ticker stem in upper case (e.g. ``pkn.pl`` -> ``["PKN"]``).
        """
        if self.keywords:
            return list(self.keywords)
        return [self.ticker.split(".")[0].upper()]


class Portfolio(BaseModel):
    """The user's portfolio: base currency + a list of holdings."""

    model_config = ConfigDict(extra="forbid")

    base_currency: str = Field(default="PLN", min_length=3, max_length=3)
    holdings: list[Holding] = Field(default_factory=list)

    @field_validator("base_currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _no_duplicate_tickers(self) -> Portfolio:
        seen: set[str] = set()
        for h in self.holdings:
            if h.ticker in seen:
                raise ValueError(f"Duplicate ticker in portfolio: {h.ticker}")
            seen.add(h.ticker)
        return self

    # -- Convenience --------------------------------------------------------

    @property
    def tickers(self) -> list[str]:
        return [h.ticker for h in self.holdings]

    def find(self, ticker: str) -> Holding | None:
        norm = normalize_ticker(ticker)
        return next((h for h in self.holdings if h.ticker == norm), None)


# --- Computed outputs -------------------------------------------------------


class HoldingStatus(BaseModel):
    """A holding with current pricing data attached.

    All price-derived fields are optional because cached OHLCV may be missing
    (e.g. brand-new ticker, fetch failure, stale cache cleared).
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    name: str | None
    shares: float
    entry_price: float
    entry_date: date
    cost_basis: float

    last_price: float | None = None
    last_price_date: date | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None

    @property
    def has_price(self) -> bool:
        return self.last_price is not None


class PortfolioStatus(BaseModel):
    """Aggregate, computed view of a portfolio at a point in time.

    Notes on totals
    ---------------
    * ``total_cost_basis`` covers **all** holdings.
    * ``total_market_value`` and ``total_unrealized_pnl`` aggregate **only**
      holdings that have cached pricing data. The ``missing_data`` list
      identifies holdings excluded from the market-value totals so the
      caller can decide how to present the gap.
    """

    model_config = ConfigDict(frozen=True)

    base_currency: str
    as_of: datetime
    holdings: list[HoldingStatus]
    total_cost_basis: float
    priced_cost_basis: float
    total_market_value: float
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    missing_data: list[str] = Field(default_factory=list)


PORTFOLIO_SCHEMA_VERSION: Final[str] = "1.0"
