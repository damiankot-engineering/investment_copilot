"""Rebalancing service — thin wrapper over the pure engine + an apply step.

``plan`` computes a :class:`RebalancePlan` (read-only). ``apply`` materializes
a plan into the portfolio by appending one BUY/SELL :class:`Transaction` per
trade and returning a re-validated :class:`Portfolio` (the caller persists it
with ``save_portfolio``). Default constraints come from config; a request can
override them.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Mapping

from investment_copilot.config.schema import RebalanceConfig
from investment_copilot.domain.portfolio import Portfolio, PortfolioStatus
from investment_copilot.domain.rebalance import (
    RebalanceConstraints,
    RebalancePlan,
    compute_rebalance,
)


class RebalanceService:
    """Compute and apply rebalancing plans for a single portfolio."""

    def __init__(self, rebalance_config: RebalanceConfig) -> None:
        self._cfg = rebalance_config

    def default_constraints(self) -> RebalanceConstraints:
        return RebalanceConstraints(
            drift_band_pct=self._cfg.drift_band_pct,
            min_trade_value=self._cfg.min_trade_value,
            round_to_whole_shares=self._cfg.round_to_whole_shares,
        )

    def plan(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        targets: Mapping[str, float] | None = None,
        constraints: RebalanceConstraints | None = None,
    ) -> RebalancePlan:
        """Read-only plan. ``targets`` are fractions (0–1); tax follows the
        portfolio's ``account_type`` (IKE/IKZE → exempt)."""
        return compute_rebalance(
            portfolio,
            status,
            targets=targets,
            constraints=constraints or self.default_constraints(),
            tax_exempt=portfolio.is_tax_exempt,
        )

    def apply(self, portfolio: Portfolio, plan: RebalancePlan) -> Portfolio:
        """Append the plan's trades as transactions; return a re-validated
        Portfolio (FIFO/no-duplicate validators re-run on construction)."""
        note = f"rebalance {date.today().isoformat()}"
        by_ticker: dict[str, list] = defaultdict(list)
        for tr in plan.trades:
            by_ticker[tr.ticker].append(tr)

        data = portfolio.model_dump()
        for h in data["holdings"]:
            for tr in by_ticker.get(h["ticker"], []):
                h["transactions"].append({
                    "date": date.today().isoformat(),
                    "action": tr.action,
                    "shares": tr.shares,
                    "price_per_share": tr.est_price,
                    "fees": 0.0,
                    "note": note,
                })
        return Portfolio.model_validate(data)
