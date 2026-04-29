"""Strategy interface.

A :class:`Strategy` consumes a *panel* of OHLCV data (one frame per ticker)
and returns per-ticker signal series. Signals are state-based:

* ``1`` → desired target state is "long that ticker"
* ``0`` → desired target state is "flat"

Strategies are expected to internally ``shift(1)`` so the signal for day
``t`` uses information available no later than the close of day ``t-1`` —
this keeps backtests free of lookahead bias by construction.

The :attr:`warmup_days` property tells the service how much extra history
to load before the requested backtest start.
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Strategy(Protocol):
    """Per-day target-state signal generator."""

    name: str
    warmup_days: int

    def generate_signals(
        self,
        panel: Mapping[str, pd.DataFrame],
    ) -> dict[str, pd.Series]:
        """Return ``{ticker: signal_series}`` of dtype ``int`` in {0, 1}."""
        ...
