"""Buy & Hold strategy.

Buys all portfolio positions on the first day and holds them indefinitely.
No rebalancing or selling; equal-weight allocation across all holdings.

Signal is 1 (long) from the first available data day onwards.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


class BuyAndHold:
    """Buy once on day 1, hold forever. Simple passive baseline strategy."""

    name: str = "buy_and_hold"

    @property
    def warmup_days(self) -> int:
        return 0  # No warmup needed; signal is immediate

    def generate_signals(
        self,
        panel: Mapping[str, pd.DataFrame],
    ) -> dict[str, pd.Series]:
        signals: dict[str, pd.Series] = {}
        for ticker, df in panel.items():
            if df.empty or "close" not in df.columns:
                signals[ticker] = pd.Series(dtype=int, name=ticker)
                continue
            # Always long: signal = 1 for all days
            sig = pd.Series(1, index=df.index, dtype=int, name=ticker)
            signals[ticker] = sig
        return signals
