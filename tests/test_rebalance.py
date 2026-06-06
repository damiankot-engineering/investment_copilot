"""Tests for the rebalancing engine + service."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from investment_copilot.config.schema import RebalanceConfig
from investment_copilot.domain.portfolio import (
    Holding,
    HoldingStatus,
    Portfolio,
    PortfolioStatus,
    Transaction,
)
from investment_copilot.domain.rebalance import (
    RebalanceConstraints,
    compute_rebalance,
)
from investment_copilot.services.rebalance_service import RebalanceService


def _holding(ticker: str, shares: float, buy_price: float, target_weight=None) -> Holding:
    return Holding(
        ticker=ticker,
        thesis="t",
        target_weight=target_weight,
        transactions=[Transaction(
            date=date(2024, 1, 2), action="BUY", shares=shares, price_per_share=buy_price,
        )],
    )


def _status(portfolio: Portfolio, prices: dict[str, float]) -> PortfolioStatus:
    rows: list[HoldingStatus] = []
    total_mv = 0.0
    total_cb = 0.0
    for h in portfolio.holdings:
        price = prices.get(h.ticker)
        mv = (h.shares * price) if price is not None else None
        cb = h.cost_basis
        total_cb += cb
        if mv is not None:
            total_mv += mv
        rows.append(HoldingStatus(
            ticker=h.ticker, name=h.name, shares=h.shares,
            entry_price=h.avg_entry_price, entry_date=h.first_entry_date,
            cost_basis=cb, last_price=price, last_price_date=date.today() if price else None,
            market_value=mv,
            unrealized_pnl=(mv - cb if mv is not None else None),
            unrealized_pnl_pct=((mv - cb) / cb if (mv is not None and cb > 0) else None),
        ))
    return PortfolioStatus(
        base_currency=portfolio.base_currency,
        as_of=datetime.now(timezone.utc),
        holdings=rows,
        total_cost_basis=total_cb,
        priced_cost_basis=total_cb,
        total_market_value=total_mv,
        total_unrealized_pnl=total_mv - total_cb,
        total_unrealized_pnl_pct=((total_mv - total_cb) / total_cb if total_cb > 0 else 0.0),
    )


def _trade(plan, ticker):
    return next((t for t in plan.trades if t.ticker == ticker), None)


# --- equal-weight + self-financing ------------------------------------------


def test_equal_weight_fallback_self_financing() -> None:
    pf = Portfolio(holdings=[_holding("a.pl", 10, 100), _holding("b.pl", 10, 100)])
    status = _status(pf, {"a.pl": 140, "b.pl": 60})  # 1400 vs 600, total 2000
    plan = compute_rebalance(pf, status)

    a, b = _trade(plan, "a.pl"), _trade(plan, "b.pl")
    assert a.action == "SELL" and a.shares == 3   # 400/140 -> round 3
    assert b.action == "BUY" and b.shares == 7    # 400/60  -> round 7
    # self-financing: sells ≈ buys (rounding residual small)
    assert abs(plan.residual_cash) < 140
    assert plan.total_market_value == 2000


def test_manual_target_weight() -> None:
    pf = Portfolio(holdings=[
        _holding("a.pl", 10, 100, target_weight=0.7),
        _holding("b.pl", 10, 100, target_weight=0.3),
    ])
    status = _status(pf, {"a.pl": 100, "b.pl": 100})  # 1000/1000, total 2000
    plan = compute_rebalance(pf, status)
    a, b = _trade(plan, "a.pl"), _trade(plan, "b.pl")
    assert a.action == "BUY" and a.shares == 4    # to 1400 -> +400/100
    assert b.action == "SELL" and b.shares == 4   # to 600  -> -400/100


def test_targets_override_beats_yaml() -> None:
    pf = Portfolio(holdings=[
        _holding("a.pl", 10, 100, target_weight=0.7),
        _holding("b.pl", 10, 100, target_weight=0.3),
    ])
    status = _status(pf, {"a.pl": 100, "b.pl": 100})
    # override → equal weight (fractions)
    plan = compute_rebalance(pf, status, targets={"a.pl": 0.5, "b.pl": 0.5})
    assert _trade(plan, "a.pl") is None and _trade(plan, "b.pl") is None  # already 50/50


# --- filters ----------------------------------------------------------------


def test_drift_band_skips_near_target() -> None:
    pf = Portfolio(holdings=[_holding("a.pl", 10, 100), _holding("b.pl", 10, 100)])
    status = _status(pf, {"a.pl": 104, "b.pl": 96})  # 52% / 48%, drift 2pp < 5
    plan = compute_rebalance(pf, status)
    assert plan.trades == []


def test_min_trade_value_skips_tiny() -> None:
    pf = Portfolio(holdings=[_holding("a.pl", 10, 100), _holding("b.pl", 10, 100)])
    status = _status(pf, {"a.pl": 112, "b.pl": 88})  # 56/44, drift 6pp > band
    plan = compute_rebalance(pf, status)  # delta ≈ 120 < 200 min_trade_value
    assert plan.trades == []


def test_no_round_keeps_fractional() -> None:
    pf = Portfolio(holdings=[_holding("a.pl", 10, 100), _holding("b.pl", 10, 100)])
    status = _status(pf, {"a.pl": 140, "b.pl": 60})
    plan = compute_rebalance(
        pf, status, constraints=RebalanceConstraints(round_to_whole_shares=False),
    )
    a = _trade(plan, "a.pl")
    assert a.shares != round(a.shares)  # fractional kept


# --- tax preview ------------------------------------------------------------


def test_tax_preview_standard_vs_ike() -> None:
    pf = Portfolio(holdings=[_holding("a.pl", 10, 100), _holding("b.pl", 10, 100)])
    status = _status(pf, {"a.pl": 140, "b.pl": 60})
    # SELL 3 of a.pl @140, basis 100 → gain (140-100)*3 = 120 → tax 22.8
    std = compute_rebalance(pf, status, tax_exempt=False)
    assert _trade(std, "a.pl").realized_pnl == pytest.approx(120.0)
    assert _trade(std, "a.pl").est_tax == pytest.approx(120.0 * 0.19)
    assert std.est_total_tax == pytest.approx(22.8)

    ike = compute_rebalance(pf, status, tax_exempt=True)
    assert _trade(ike, "a.pl").est_tax == 0.0
    assert ike.est_total_tax == 0.0


def test_account_type_ike_is_exempt_via_service() -> None:
    pf = Portfolio(account_type="ike", holdings=[
        _holding("a.pl", 10, 100), _holding("b.pl", 10, 100),
    ])
    status = _status(pf, {"a.pl": 140, "b.pl": 60})
    svc = RebalanceService(RebalanceConfig())
    plan = svc.plan(pf, status)
    assert plan.tax_exempt is True and plan.est_total_tax == 0.0


# --- apply ------------------------------------------------------------------


def test_apply_appends_transactions_and_revalidates() -> None:
    pf = Portfolio(holdings=[_holding("a.pl", 10, 100), _holding("b.pl", 10, 100)])
    status = _status(pf, {"a.pl": 140, "b.pl": 60})
    svc = RebalanceService(RebalanceConfig())
    plan = svc.plan(pf, status)
    updated = svc.apply(pf, plan)

    # a.pl sold 3 → 7 shares; b.pl bought 7 → 17 shares
    assert updated.find("a.pl").shares == 7
    assert updated.find("b.pl").shares == 17
    # the new transactions are tagged
    a_txs = updated.find("a.pl").transactions
    assert a_txs[-1].action == "SELL" and a_txs[-1].note.startswith("rebalance")


def test_unpriced_holding_warns_and_skips() -> None:
    pf = Portfolio(holdings=[_holding("a.pl", 10, 100), _holding("b.pl", 10, 100)])
    status = _status(pf, {"a.pl": 140})  # b.pl unpriced
    plan = compute_rebalance(pf, status)
    assert any("b.pl" in w for w in plan.warnings)
    assert all(t.ticker != "b.pl" for t in plan.trades)
