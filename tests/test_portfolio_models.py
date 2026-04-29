"""Tests for ``investment_copilot.domain.portfolio`` (input models)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from investment_copilot.domain.portfolio import Holding, Portfolio


# --- Holding ----------------------------------------------------------------


def _holding_kwargs(**over):
    base = dict(
        ticker="PKN",
        shares=100,
        entry_price=65.0,
        entry_date=date(2023, 1, 2),
        thesis="energy champion",
    )
    base.update(over)
    return base


def test_holding_normalizes_ticker() -> None:
    h = Holding(**_holding_kwargs(ticker="PKN.WA"))
    assert h.ticker == "pkn.pl"


def test_holding_cost_basis() -> None:
    h = Holding(**_holding_kwargs(shares=10, entry_price=50.0))
    assert h.cost_basis == 500.0


def test_holding_effective_keywords_default() -> None:
    h = Holding(**_holding_kwargs(ticker="cdr.pl"))
    assert h.effective_keywords == ["CDR"]


def test_holding_effective_keywords_user_provided_wins() -> None:
    h = Holding(**_holding_kwargs(keywords=["Orlen", "PKN"]))
    assert h.effective_keywords == ["Orlen", "PKN"]


def test_holding_strips_blank_keywords() -> None:
    h = Holding(**_holding_kwargs(keywords=["  Orlen  ", "", " "]))
    assert h.effective_keywords == ["Orlen"]


@pytest.mark.parametrize("bad_field,bad_value", [("shares", 0), ("shares", -1), ("entry_price", 0)])
def test_holding_rejects_non_positive(bad_field: str, bad_value) -> None:
    with pytest.raises(ValidationError):
        Holding(**_holding_kwargs(**{bad_field: bad_value}))


def test_holding_rejects_future_entry_date() -> None:
    future = date.today() + timedelta(days=30)
    with pytest.raises(ValidationError, match="future"):
        Holding(**_holding_kwargs(entry_date=future))


def test_holding_rejects_empty_thesis() -> None:
    with pytest.raises(ValidationError):
        Holding(**_holding_kwargs(thesis=""))


def test_holding_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        Holding(**_holding_kwargs(unknown_key="oops"))


# --- Portfolio --------------------------------------------------------------


def test_portfolio_empty_is_valid() -> None:
    p = Portfolio(holdings=[])
    assert p.tickers == []
    assert p.base_currency == "PLN"


def test_portfolio_uppercases_currency() -> None:
    p = Portfolio(base_currency="pln", holdings=[])
    assert p.base_currency == "PLN"


def test_portfolio_rejects_duplicate_tickers() -> None:
    with pytest.raises(ValidationError, match="Duplicate ticker"):
        Portfolio(
            holdings=[
                Holding(**_holding_kwargs(ticker="PKN")),
                Holding(**_holding_kwargs(ticker="PKN.WA")),  # normalizes to same
            ]
        )


def test_portfolio_find() -> None:
    p = Portfolio(
        holdings=[
            Holding(**_holding_kwargs(ticker="PKN")),
            Holding(**_holding_kwargs(ticker="CDR")),
        ]
    )
    assert p.find("pkn.pl") is not None
    assert p.find("PKN") is not None  # normalization on lookup
    assert p.find("XYZ") is None


def test_portfolio_tickers_normalized() -> None:
    p = Portfolio(
        holdings=[
            Holding(**_holding_kwargs(ticker="PKN.WA")),
            Holding(**_holding_kwargs(ticker="cdr")),
        ]
    )
    assert p.tickers == ["pkn.pl", "cdr.pl"]
