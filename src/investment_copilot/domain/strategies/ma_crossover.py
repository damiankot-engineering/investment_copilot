"""Moving-average crossover strategy.

Per ticker:
    raw[t]    = 1 if SMA(close, fast)[t] > SMA(close, slow)[t] else 0
    signal[t] = raw[t-1]      # one-day lag → no lookahead

Signals are 0 during the warmup period when either MA is undefined.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


class MACrossover:
    """Long when fast SMA > slow SMA (golden cross), flat otherwise."""

    name: str = "ma_crossover"

    def __init__(self, fast: int, slow: int) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("fast and slow must be >= 1")
        if slow <= fast:
            raise ValueError(f"slow ({slow}) must be greater than fast ({fast})")
        self.fast = fast
        self.slow = slow

    @property
    def warmup_days(self) -> int:
        return self.slow

    def generate_signals(
        self,
        panel: Mapping[str, pd.DataFrame],
    ) -> dict[str, pd.Series]:
        signals: dict[str, pd.Series] = {}
        for ticker, df in panel.items():
            if df.empty or "close" not in df.columns:
                signals[ticker] = pd.Series(dtype=int, name=ticker)
                continue
            close = df["close"].astype(float)
            fast_ma = close.rolling(self.fast, min_periods=self.fast).mean()
            slow_ma = close.rolling(self.slow, min_periods=self.slow).mean()
            raw = (fast_ma > slow_ma).astype(int)
            # One-day lag: signal[t] uses MA at t-1
            sig = raw.shift(1).fillna(0).astype(int)
            sig.name = ticker
            signals[ticker] = sig
        return signals
