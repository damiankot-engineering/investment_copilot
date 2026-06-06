"""Rebalancing engine (pure — no I/O).

Given a portfolio's current market values and a target allocation, compute the
BUY/SELL trades that move current weights toward the target, **self-financing
within the current total market value** (sells fund buys; whole-share rounding
leaves a small residual cash). Realism knobs: whole-share rounding, a drift
band (no-trade zone), a min-trade filter, and a FIFO realized-gain + Polish PIT
(19%) tax preview that is zero for tax-exempt accounts (IKE/IKZE).

Target resolution (handled by the caller / :func:`compute_rebalance`):
``targets`` override (fractions 0–1) → per-holding ``target_weight`` → equal
weight. Weights are normalized over the **priced** holdings.

All money is in the portfolio's base currency; weights here are fractions
internally and surfaced as percent (0–100) on the models.
"""

from __future__ import annotations

from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from investment_copilot.domain.portfolio import (
    Portfolio,
    PortfolioStatus,
    preview_realized_pnl,
)

PIT_RATE: float = 0.19  # Polish capital-gains tax (Belka), standard accounts


class RebalanceConstraints(BaseModel):
    """Realism filters applied when turning weight gaps into trades."""

    model_config = ConfigDict(frozen=True)

    drift_band_pct: float = Field(
        default=5.0, ge=0.0,
        description="Skip a holding whose |current − target| weight gap (pp) is below this.",
    )
    min_trade_value: float = Field(
        default=200.0, ge=0.0,
        description="Skip a trade whose value is below this (base currency).",
    )
    round_to_whole_shares: bool = Field(
        default=True,
        description="Round share deltas to whole shares (GPW trades whole units).",
    )


class RebalanceTrade(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    name: str | None = None
    action: Literal["BUY", "SELL"]
    shares: float
    est_price: float
    est_value: float  # positive magnitude = shares * est_price
    current_weight_pct: float
    target_weight_pct: float
    drift_pct: float  # target − current (pp); +ve = underweight → buy
    realized_pnl: float | None = None  # SELL only (FIFO)
    est_tax: float = 0.0


class RebalancePosition(BaseModel):
    """Per-holding current vs target weight (for the UI bars — all priced)."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    name: str | None = None
    market_value: float
    current_weight_pct: float
    target_weight_pct: float


class RebalancePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    account_type: str
    tax_exempt: bool
    total_market_value: float
    positions: list[RebalancePosition] = Field(default_factory=list)
    trades: list[RebalanceTrade] = Field(default_factory=list)
    turnover_pct: float = 0.0
    est_total_tax: float = 0.0
    residual_cash: float = 0.0  # sells − buys after rounding (+ = cash left over)
    warnings: list[str] = Field(default_factory=list)


def _resolve_targets(
    portfolio: Portfolio,
    priced: list[str],
    override: Mapping[str, float] | None,
    warnings: list[str],
) -> dict[str, float]:
    """Return normalized target fractions over ``priced`` tickers."""
    raw: dict[str, float | None] = {}
    for t in priced:
        if override is not None and t in override:
            raw[t] = float(override[t])
        else:
            h = portfolio.find(t)
            raw[t] = h.target_weight if h is not None else None

    provided = [v for v in raw.values() if v is not None]
    if not provided:
        # Equal-weight fallback.
        w = 1.0 / len(priced)
        return {t: w for t in priced}

    total = sum(provided)
    if total <= 0:
        warnings.append("Sumaryczne wagi docelowe ≤ 0 — użyto equal-weight.")
        w = 1.0 / len(priced)
        return {t: w for t in priced}

    if abs(total - 1.0) > 0.005:
        warnings.append(
            f"Wagi docelowe sumują się do {total * 100:.1f}% — znormalizowano do 100%."
        )
    return {t: (raw[t] or 0.0) / total for t in priced}


def compute_rebalance(
    portfolio: Portfolio,
    status: PortfolioStatus,
    *,
    targets: Mapping[str, float] | None = None,
    constraints: RebalanceConstraints | None = None,
    tax_exempt: bool = False,
) -> RebalancePlan:
    """Compute a self-financing rebalance plan. ``targets`` are fractions (0–1)."""
    c = constraints or RebalanceConstraints()
    warnings: list[str] = []

    # Priced universe: holdings with a market value and a positive price.
    priced_status = [
        s for s in status.holdings
        if s.market_value is not None and s.last_price and s.last_price > 0
    ]
    unpriced = [s.ticker for s in status.holdings if s not in priced_status]
    if unpriced:
        warnings.append(
            "Pominięto pozycje bez wyceny: " + ", ".join(sorted(unpriced))
        )
    if not priced_status:
        return RebalancePlan(
            account_type=portfolio.account_type,
            tax_exempt=tax_exempt,
            total_market_value=0.0,
            warnings=warnings or ["Brak wycenionych pozycji do rebalansu."],
        )

    priced = [s.ticker for s in priced_status]
    total = float(sum(s.market_value for s in priced_status))
    target_w = _resolve_targets(portfolio, priced, targets, warnings)

    positions: list[RebalancePosition] = []
    trades: list[RebalanceTrade] = []
    sum_buys = 0.0
    sum_sells = 0.0
    est_total_tax = 0.0

    for s in priced_status:
        holding = portfolio.find(s.ticker)
        cur_val = float(s.market_value)
        price = float(s.last_price)
        cur_w = cur_val / total
        tgt_w = target_w[s.ticker]
        positions.append(RebalancePosition(
            ticker=s.ticker,
            name=holding.name if holding else None,
            market_value=cur_val,
            current_weight_pct=cur_w * 100.0,
            target_weight_pct=tgt_w * 100.0,
        ))

        drift_pct = (tgt_w - cur_w) * 100.0
        # Drift band: leave near-target holdings alone.
        if abs(drift_pct) < c.drift_band_pct:
            continue

        delta_value = tgt_w * total - cur_val
        delta_shares = delta_value / price
        if c.round_to_whole_shares:
            delta_shares = float(round(delta_shares))
        # Never sell more than held.
        if delta_shares < 0 and abs(delta_shares) > s.shares:
            delta_shares = -float(s.shares)
        if delta_shares == 0:
            continue

        exec_value = abs(delta_shares) * price
        if exec_value < c.min_trade_value:
            continue

        action = "BUY" if delta_shares > 0 else "SELL"
        realized = None
        est_tax = 0.0
        if action == "SELL":
            realized = preview_realized_pnl(holding, abs(delta_shares), price)
            est_tax = 0.0 if tax_exempt else max(0.0, realized) * PIT_RATE
            sum_sells += exec_value
        else:
            sum_buys += exec_value
        est_total_tax += est_tax

        trades.append(RebalanceTrade(
            ticker=s.ticker,
            name=holding.name if holding else None,
            action=action,
            shares=abs(delta_shares),
            est_price=price,
            est_value=exec_value,
            current_weight_pct=cur_w * 100.0,
            target_weight_pct=tgt_w * 100.0,
            drift_pct=drift_pct,
            realized_pnl=realized,
            est_tax=est_tax,
        ))

    trades.sort(key=lambda t: abs(t.drift_pct), reverse=True)
    gross = sum_buys + sum_sells
    residual = sum_sells - sum_buys
    if residual < -1e-6:
        warnings.append(
            f"Zakupy przewyższają sprzedaże o {-residual:,.0f} {portfolio.base_currency} "
            "(rebalans wymaga dopłaty po zaokrągleniach)."
        )

    return RebalancePlan(
        account_type=portfolio.account_type,
        tax_exempt=tax_exempt,
        total_market_value=total,
        positions=positions,
        trades=trades,
        turnover_pct=(gross / total / 2.0 * 100.0) if total > 0 else 0.0,
        est_total_tax=est_total_tax,
        residual_cash=residual,
        warnings=warnings,
    )
