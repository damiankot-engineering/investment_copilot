"""Portfolio service: YAML loading and current-status (PnL) computation.

Read-only by design: the service does not refresh OHLCV data — that is
:class:`~investment_copilot.services.data_service.DataService`'s job. The
orchestrator wires the two together for end-to-end runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

from investment_copilot.domain.portfolio import (
    Holding,
    HoldingStatus,
    Portfolio,
    PortfolioStatus,
)
from investment_copilot.services.data_service import DataService

logger = logging.getLogger(__name__)


class PortfolioError(RuntimeError):
    """Raised when a portfolio file cannot be loaded or validated."""


# --- YAML loading -----------------------------------------------------------


def load_portfolio(path: Path | str) -> Portfolio:
    """Load and validate a portfolio YAML file.

    Tickers are normalized to Stooq form (``PKN``, ``PKN.WA`` -> ``pkn.pl``)
    by the :class:`Holding` validator. Unknown YAML keys are rejected.

    The file is read with encoding detection (UTF-8, UTF-8 with BOM,
    UTF-16 LE/BE, CP1250) so files saved by Notepad in non-UTF-8
    encodings still load.
    """
    from investment_copilot.config.encoding import (
        FileEncodingError,
        detect_encoding_label,
        read_text_robust,
    )

    p = Path(path)
    if not p.is_file():
        raise PortfolioError(f"Portfolio file not found: {p}")

    try:
        text = read_text_robust(p)
    except FileEncodingError as exc:
        raise PortfolioError(str(exc)) from exc

    label = detect_encoding_label(p)
    if label not in {"UTF-8", "empty"}:
        logger.warning(
            "%s was decoded as %s. Re-save it as UTF-8 (without BOM) "
            "to silence this warning.",
            p,
            label,
        )

    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise PortfolioError(f"Invalid YAML in {p}: {exc}") from exc

    if not isinstance(raw, dict):
        raise PortfolioError(
            f"Portfolio root must be a mapping, got {type(raw).__name__}"
        )

    try:
        return Portfolio.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError -> PortfolioError
        raise PortfolioError(f"Invalid portfolio: {exc}") from exc


# --- Service ---------------------------------------------------------------


class PortfolioService:
    """Operate on :class:`Portfolio` objects against cached market data."""

    def __init__(self, data_service: DataService) -> None:
        self._data = data_service

    # -- Pure / no-data methods ---------------------------------------------

    @staticmethod
    def keywords_map(portfolio: Portfolio) -> dict[str, list[str]]:
        """Build the ``ticker -> keywords`` mapping for news refresh."""
        return {h.ticker: h.effective_keywords for h in portfolio.holdings}

    @staticmethod
    def total_cost_basis(portfolio: Portfolio) -> float:
        return sum(h.cost_basis for h in portfolio.holdings)

    # -- Data-dependent methods ---------------------------------------------

    def current_status(
        self,
        portfolio: Portfolio,
        *,
        as_of: datetime | None = None,
    ) -> PortfolioStatus:
        """Compute current PnL using the latest cached close per holding.

        For holdings without cached data, the corresponding
        :class:`HoldingStatus` has price fields set to ``None`` and the ticker
        is recorded in :attr:`PortfolioStatus.missing_data`.
        """
        as_of = as_of or datetime.now(timezone.utc)

        statuses: list[HoldingStatus] = []
        missing: list[str] = []

        for h in portfolio.holdings:
            statuses.append(self._holding_status(h, missing))

        total_cost = sum(h.cost_basis for h in portfolio.holdings)
        priced = [s for s in statuses if s.has_price]
        priced_cost = sum(s.cost_basis for s in priced)
        total_value = sum(s.market_value or 0.0 for s in priced)
        total_pnl = total_value - priced_cost
        total_pnl_pct = (total_pnl / priced_cost) if priced_cost > 0 else 0.0

        return PortfolioStatus(
            base_currency=portfolio.base_currency,
            as_of=as_of,
            holdings=statuses,
            total_cost_basis=total_cost,
            priced_cost_basis=priced_cost,
            total_market_value=total_value,
            total_unrealized_pnl=total_pnl,
            total_unrealized_pnl_pct=total_pnl_pct,
            missing_data=missing,
        )

    # -- Internal -----------------------------------------------------------

    def _holding_status(
        self,
        h: Holding,
        missing: list[str],
    ) -> HoldingStatus:
        df = self._data.load_ohlcv(h.ticker)
        if df is None or df.empty:
            missing.append(h.ticker)
            return HoldingStatus(
                ticker=h.ticker,
                name=h.name,
                shares=h.shares,
                entry_price=h.entry_price,
                entry_date=h.entry_date,
                cost_basis=h.cost_basis,
            )

        last_price, last_date = _last_close(df)
        market_value = h.shares * last_price
        pnl = market_value - h.cost_basis
        pnl_pct = pnl / h.cost_basis if h.cost_basis > 0 else 0.0

        return HoldingStatus(
            ticker=h.ticker,
            name=h.name,
            shares=h.shares,
            entry_price=h.entry_price,
            entry_date=h.entry_date,
            cost_basis=h.cost_basis,
            last_price=last_price,
            last_price_date=last_date,
            market_value=market_value,
            unrealized_pnl=pnl,
            unrealized_pnl_pct=pnl_pct,
        )


# --- helpers ----------------------------------------------------------------


def _last_close(df: pd.DataFrame):
    last_idx = df.index[-1]
    last_close = float(df["close"].iloc[-1])
    last_date = last_idx.date() if hasattr(last_idx, "date") else last_idx
    return last_close, last_date
