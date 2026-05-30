"""Portfolio domain models.

Inputs (from ``portfolio.yaml``):
    * :class:`Transaction` — a single BUY or SELL event
    * :class:`Holding` — a tracked position; cumulative state of its transactions
    * :class:`Portfolio` — the user's tracked positions

Computed outputs (returned by services):
    * :class:`HoldingStatus` — a :class:`Holding` enriched with current price / PnL
    * :class:`PortfolioStatus` — aggregate view of the portfolio

Inputs are validated strictly: ``extra="forbid"`` rejects unknown YAML keys,
and a model validator forbids duplicate tickers. Ticker normalization
happens at field-validation time, so users may write ``PKN``, ``PKN.WA``,
or ``pkn.pl`` interchangeably.

Cost basis follows **FIFO** (compatible with Polish PIT reporting): when
SELL transactions consume shares, they remove from the oldest still-active
BUY lot first. ``Holding.realized_pnl`` exposes the cumulative gain or
loss from those closed lots; ``Holding.cost_basis`` is the basis of what's
still held today.

Legacy single-entry portfolio.yaml files (the pre-transactions schema with
``entry_price``/``shares``/``entry_date``) are auto-migrated on load to a
one-BUY transaction list — the first ``save_portfolio`` will then rewrite
the file in the new format.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from investment_copilot.domain.models import normalize_ticker


# --- Inputs (YAML) ----------------------------------------------------------


TransactionAction = Literal["BUY", "SELL"]


class Transaction(BaseModel):
    """A single BUY or SELL event for one holding.

    All transactions for a holding live in ``Holding.transactions``; the
    holding's current shares, cost basis and realized PnL are all derived
    from walking that list chronologically through a FIFO matcher.
    """

    model_config = ConfigDict(extra="forbid")

    date: date
    action: TransactionAction
    shares: float = Field(gt=0)
    price_per_share: float = Field(gt=0)
    fees: float = Field(default=0.0, ge=0)
    note: str = Field(default="", max_length=200)

    @field_validator("date")
    @classmethod
    def _not_in_future(cls, v: date) -> date:
        if v > date.today():
            raise ValueError(f"transaction date {v} is in the future")
        return v


class _Lot(BaseModel):
    """Internal FIFO accounting unit — one active BUY remainder."""

    model_config = ConfigDict(frozen=False)

    shares: float
    price_per_share: float
    acquired_on: date


def _fifo_walk(transactions: list[Transaction]) -> tuple[list[_Lot], float]:
    """Walk transactions chronologically, return ``(active_lots, realized_pnl)``.

    SELL transactions are matched against the oldest active BUY lots
    (FIFO). Lot fragments — when a SELL only partially consumes a lot —
    leave the remaining shares with their original cost basis. Sale fees
    reduce the realized gain; purchase fees are amortized into the lot's
    cost basis (so future PnL on those shares already accounts for them).

    Raises ``ValueError`` if a SELL would push the position below zero —
    user error caught early at YAML load time.
    """
    sorted_txs = sorted(transactions, key=lambda t: t.date)
    lots: list[_Lot] = []
    realized = 0.0
    for tx in sorted_txs:
        if tx.action == "BUY":
            per_share_basis = tx.price_per_share + (
                tx.fees / tx.shares if tx.shares > 0 else 0.0
            )
            lots.append(_Lot(
                shares=tx.shares,
                price_per_share=per_share_basis,
                acquired_on=tx.date,
            ))
            continue
        # SELL
        remaining = tx.shares
        while remaining > 1e-9 and lots:
            lot = lots[0]
            take = min(lot.shares, remaining)
            realized += (tx.price_per_share - lot.price_per_share) * take
            lot.shares -= take
            remaining -= take
            if lot.shares <= 1e-9:
                lots.pop(0)
        realized -= tx.fees
        if remaining > 1e-9:
            raise ValueError(
                f"SELL on {tx.date} of {tx.shares} shares exceeds available "
                f"holdings (short {remaining:g} shares)."
            )
    return lots, realized


class Holding(BaseModel):
    """A single tracked position — current state derived from transactions."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Ticker; normalized to Stooq form on load.")
    thesis: str = Field(min_length=1)
    transactions: list[Transaction] = Field(
        min_length=1,
        description=(
            "Chronological list of BUY/SELL transactions. Order is not "
            "required in the YAML — FIFO matching sorts by date internally."
        ),
    )
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

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data: Any) -> Any:
        """Convert pre-transactions YAML (entry_price + shares + entry_date)
        into a single BUY transaction so old portfolio.yaml files keep working
        without a separate migration step."""
        if not isinstance(data, dict):
            return data
        if data.get("transactions"):
            return data
        legacy_keys = {"entry_price", "shares", "entry_date"}
        if not legacy_keys.issubset(data.keys()):
            return data
        migrated = dict(data)
        entry_date = migrated.pop("entry_date")
        shares = migrated.pop("shares")
        entry_price = migrated.pop("entry_price")
        migrated["transactions"] = [{
            "date": entry_date,
            "action": "BUY",
            "shares": shares,
            "price_per_share": entry_price,
        }]
        return migrated

    @field_validator("ticker")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return normalize_ticker(v)

    @field_validator("keywords")
    @classmethod
    def _strip_keywords(cls, v: list[str]) -> list[str]:
        return [k.strip() for k in v if k and k.strip()]

    @model_validator(mode="after")
    def _validate_fifo(self) -> Holding:
        # Run the FIFO walk once at load time so SELL-overdraft errors
        # surface as clean Pydantic validation messages.
        _fifo_walk(self.transactions)
        return self

    # -- Derived properties -------------------------------------------------

    @property
    def shares(self) -> float:
        """Net shares currently held (sum BUYs minus SELLs)."""
        return sum(
            tx.shares if tx.action == "BUY" else -tx.shares
            for tx in self.transactions
        )

    @property
    def cost_basis(self) -> float:
        """Total cost basis of the lots still active (per FIFO)."""
        lots, _ = _fifo_walk(self.transactions)
        return sum(lot.shares * lot.price_per_share for lot in lots)

    @property
    def avg_entry_price(self) -> float:
        """Weighted-average price of the lots still active. ``0`` if flat."""
        s = self.shares
        return self.cost_basis / s if s > 0 else 0.0

    @property
    def first_entry_date(self) -> date:
        """Date of the earliest BUY transaction."""
        buys = [tx.date for tx in self.transactions if tx.action == "BUY"]
        return min(buys) if buys else self.transactions[0].date

    @property
    def realized_pnl(self) -> float:
        """Cumulative gain/loss from closed lots (SELL transactions)."""
        _, realized = _fifo_walk(self.transactions)
        return realized

    def position_at(self, target: date) -> tuple[float, float]:
        """Return ``(shares_held, fifo_cost_basis)`` as of ``target``.

        Walks transactions up to and including ``target`` and applies the
        FIFO matcher to the active lots. Used by the backtest to fix the
        starting position when the user asks for a window that begins
        after some BUYs (and possibly SELLs) have already happened.
        """
        txs_up_to = [tx for tx in self.transactions if tx.date <= target]
        if not txs_up_to:
            return 0.0, 0.0
        lots, _ = _fifo_walk(txs_up_to)
        shares = sum(lot.shares for lot in lots)
        basis = sum(lot.shares * lot.price_per_share for lot in lots)
        return shares, basis

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
    entry_price: float = Field(
        description="Backwards-compatible avg cost per active share (FIFO).",
    )
    entry_date: date = Field(
        description="Backwards-compatible first BUY date.",
    )
    cost_basis: float
    realized_pnl: float = 0.0
    n_transactions: int = 0

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
    total_realized_pnl: float = 0.0
    missing_data: list[str] = Field(default_factory=list)


PORTFOLIO_SCHEMA_VERSION: Final[str] = "2.0"
