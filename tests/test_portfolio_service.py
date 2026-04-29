"""Tests for ``PortfolioService.current_status``."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from investment_copilot.domain.portfolio import Holding, Portfolio
from investment_copilot.services.portfolio_service import PortfolioService


# --- Fake DataService ------------------------------------------------------


class FakeData:
    """Minimal stand-in: only the methods PortfolioService needs."""

    def __init__(self, prices: dict[str, float | None]) -> None:
        # ticker -> last close price (or None to simulate missing data)
        self._prices = prices

    def load_ohlcv(self, ticker: str, *, start=None, end=None) -> pd.DataFrame:
        price = self._prices.get(ticker)
        if price is None:
            return pd.DataFrame()
        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        return pd.DataFrame(
            {
                "open": [price - 1, price - 0.5, price],
                "high": [price, price, price + 0.5],
                "low": [price - 1.5, price - 1, price - 0.5],
                "close": [price - 0.5, price - 0.2, price],
                "volume": np.full(3, 1000.0),
            },
            index=idx,
        )


def _holding(ticker: str, shares: float, entry_price: float) -> Holding:
    return Holding(
        ticker=ticker,
        shares=shares,
        entry_price=entry_price,
        entry_date=date(2023, 1, 2),
        thesis="t",
    )


# --- Tests -----------------------------------------------------------------


def test_current_status_all_holdings_priced() -> None:
    portfolio = Portfolio(
        holdings=[
            _holding("PKN", shares=100, entry_price=65.0),   # cost 6500, mv 70*100=7000
            _holding("CDR", shares=10, entry_price=200.0),   # cost 2000, mv 180*10=1800
        ]
    )
    svc = PortfolioService(FakeData({"pkn.pl": 70.0, "cdr.pl": 180.0}))

    status = svc.current_status(portfolio)

    assert status.missing_data == []
    assert status.total_cost_basis == pytest.approx(6500 + 2000)
    assert status.priced_cost_basis == pytest.approx(8500)
    assert status.total_market_value == pytest.approx(7000 + 1800)
    assert status.total_unrealized_pnl == pytest.approx(8800 - 8500)
    assert status.total_unrealized_pnl_pct == pytest.approx(300 / 8500)

    by_ticker = {s.ticker: s for s in status.holdings}
    assert by_ticker["pkn.pl"].unrealized_pnl == pytest.approx(500)
    assert by_ticker["pkn.pl"].unrealized_pnl_pct == pytest.approx(500 / 6500)
    assert by_ticker["cdr.pl"].unrealized_pnl == pytest.approx(-200)


def test_current_status_handles_missing_data() -> None:
    portfolio = Portfolio(
        holdings=[
            _holding("PKN", shares=100, entry_price=65.0),
            _holding("CDR", shares=10, entry_price=200.0),  # no data
        ]
    )
    svc = PortfolioService(FakeData({"pkn.pl": 70.0, "cdr.pl": None}))

    status = svc.current_status(portfolio)

    assert status.missing_data == ["cdr.pl"]
    # All-holdings cost basis still reflects everything
    assert status.total_cost_basis == pytest.approx(6500 + 2000)
    # Priced totals exclude CDR
    assert status.priced_cost_basis == pytest.approx(6500)
    assert status.total_market_value == pytest.approx(7000)
    assert status.total_unrealized_pnl == pytest.approx(500)

    cdr_status = next(s for s in status.holdings if s.ticker == "cdr.pl")
    assert not cdr_status.has_price
    assert cdr_status.market_value is None
    assert cdr_status.unrealized_pnl is None


def test_current_status_empty_portfolio() -> None:
    svc = PortfolioService(FakeData({}))
    status = svc.current_status(Portfolio(holdings=[]))
    assert status.holdings == []
    assert status.total_cost_basis == 0
    assert status.total_market_value == 0
    assert status.total_unrealized_pnl == 0
    assert status.total_unrealized_pnl_pct == 0


def test_keywords_map() -> None:
    portfolio = Portfolio(
        holdings=[
            Holding(
                ticker="PKN",
                shares=10,
                entry_price=65,
                entry_date=date(2023, 1, 1),
                thesis="t",
                keywords=["Orlen", "PKN"],
            ),
            Holding(
                ticker="CDR",
                shares=10,
                entry_price=100,
                entry_date=date(2023, 1, 1),
                thesis="t",
            ),  # no keywords -> defaults to ["CDR"]
        ]
    )
    km = PortfolioService.keywords_map(portfolio)
    assert km == {"pkn.pl": ["Orlen", "PKN"], "cdr.pl": ["CDR"]}


def test_total_cost_basis_static() -> None:
    portfolio = Portfolio(
        holdings=[
            _holding("PKN", shares=100, entry_price=65.0),
            _holding("CDR", shares=10, entry_price=200.0),
        ]
    )
    assert PortfolioService.total_cost_basis(portfolio) == pytest.approx(8500)


def test_last_price_date_uses_index() -> None:
    portfolio = Portfolio(holdings=[_holding("PKN", 100, 65.0)])
    svc = PortfolioService(FakeData({"pkn.pl": 70.0}))
    status = svc.current_status(portfolio)
    s = status.holdings[0]
    assert s.last_price == pytest.approx(70.0)
    # Fake data spans 3 business days starting 2024-01-02 -> last is 2024-01-04
    assert s.last_price_date == date(2024, 1, 4)
