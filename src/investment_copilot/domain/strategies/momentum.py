"""Time-series momentum (TSMOM) strategy.

Per ticker, independently:
    ret[t]    = close[t] / close[t - lookback] - 1
    raw[t]    = 1 if ret[t] > threshold else 0
    signal[t] = raw[t-1]      # one-day lag → no lookahead

This is *time-series* momentum, not cross-sectional. Each ticker's signal
is decided on its own trailing return; the engine then equal-weights the
active set across all tickers each day.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


class Momentum:
    """Long if trailing ``lookback``-day return exceeds ``threshold``."""

    name: str = "momentum"

    def __init__(self, lookback: int, threshold: float = 0.0) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = lookback
        self.threshold = threshold

    @property
    def warmup_days(self) -> int:
        return self.lookback + 1  # +1 for the pct_change

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
            trailing_ret = close.pct_change(self.lookback)
            raw = (trailing_ret > self.threshold).astype(int)
            sig = raw.shift(1).fillna(0).astype(int)
            sig.name = ticker
            signals[ticker] = sig
        return signals
