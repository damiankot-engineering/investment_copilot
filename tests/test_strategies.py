"""Tests for strategy signal generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investment_copilot.domain.strategies import MACrossover, Momentum, make_strategy
from investment_copilot.config.schema import (
    MACrossoverParams,
    MomentumParams,
    StrategiesConfig,
)


def _ohlcv_from_close(close: np.ndarray, start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(close), freq="B")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        },
        index=idx,
    )


# --- MACrossover ------------------------------------------------------------


def test_ma_crossover_validation() -> None:
    with pytest.raises(ValueError):
        MACrossover(fast=10, slow=10)
    with pytest.raises(ValueError):
        MACrossover(fast=10, slow=5)
    with pytest.raises(ValueError):
        MACrossover(fast=0, slow=5)


def test_ma_crossover_warmup() -> None:
    s = MACrossover(fast=5, slow=20)
    assert s.warmup_days == 20


def test_ma_crossover_zeros_during_warmup() -> None:
    close = np.linspace(100, 110, 30)
    panel = {"x.pl": _ohlcv_from_close(close)}
    sigs = MACrossover(fast=5, slow=20).generate_signals(panel)
    s = sigs["x.pl"]
    # Before slow MA is defined (index 19) AND lagged by 1, signal is 0.
    assert (s.iloc[:20] == 0).all()


def test_ma_crossover_uptrend_eventually_long() -> None:
    close = np.linspace(50, 200, 100)  # strict uptrend
    panel = {"x.pl": _ohlcv_from_close(close)}
    sigs = MACrossover(fast=10, slow=30).generate_signals(panel)
    s = sigs["x.pl"]
    # Late in the series, fast > slow, so signal == 1.
    assert s.iloc[-1] == 1
    assert s.dtype.kind in ("i", "u")


def test_ma_crossover_no_lookahead() -> None:
    """Today's signal must depend only on info up to yesterday's close."""
    close = np.concatenate([np.linspace(100, 90, 40), np.linspace(90, 200, 40)])
    panel = {"x.pl": _ohlcv_from_close(close)}
    sigs = MACrossover(fast=5, slow=20).generate_signals(panel)
    s = sigs["x.pl"]
    # Find the first day where fast MA > slow MA (the "raw" cross)
    fast_ma = pd.Series(close).rolling(5).mean()
    slow_ma = pd.Series(close).rolling(20).mean()
    raw = (fast_ma > slow_ma).astype(int).values
    # Signal must equal raw shifted by 1 (with leading 0)
    expected_first_long = int(np.argmax(raw == 1))
    assert s.iloc[expected_first_long] == 0  # not yet — it's lagged
    assert s.iloc[expected_first_long + 1] == 1


def test_ma_crossover_handles_empty_frame() -> None:
    sigs = MACrossover(fast=5, slow=20).generate_signals({"x.pl": pd.DataFrame()})
    assert sigs["x.pl"].empty


# --- Momentum --------------------------------------------------------------


def test_momentum_validation() -> None:
    with pytest.raises(ValueError):
        Momentum(lookback=0)


def test_momentum_warmup() -> None:
    assert Momentum(lookback=126).warmup_days == 127


def test_momentum_uptrend_long() -> None:
    close = np.linspace(50, 200, 200)
    panel = {"x.pl": _ohlcv_from_close(close)}
    sigs = Momentum(lookback=126, threshold=0.0).generate_signals(panel)
    s = sigs["x.pl"]
    # After the lookback + lag, momentum should be persistently long
    assert s.iloc[-1] == 1
    assert (s.iloc[-50:] == 1).mean() > 0.8


def test_momentum_downtrend_flat() -> None:
    close = np.linspace(200, 50, 200)
    panel = {"x.pl": _ohlcv_from_close(close)}
    sigs = Momentum(lookback=60, threshold=0.0).generate_signals(panel)
    s = sigs["x.pl"]
    assert s.iloc[-1] == 0
    assert (s.iloc[-50:] == 0).mean() > 0.8


def test_momentum_threshold_gates_signal() -> None:
    # Modest uptrend (~10% over 60 days), threshold demands 50%.
    close = np.linspace(100, 110, 200)
    panel = {"x.pl": _ohlcv_from_close(close)}
    sigs = Momentum(lookback=60, threshold=0.5).generate_signals(panel)
    assert (sigs["x.pl"] == 0).all()


# --- Factory ---------------------------------------------------------------


def test_make_strategy_factory() -> None:
    cfg = StrategiesConfig(
        ma_crossover=MACrossoverParams(fast=10, slow=30),
        momentum=MomentumParams(lookback=60, threshold=0.05),
    )
    a = make_strategy("ma_crossover", cfg)
    assert isinstance(a, MACrossover) and a.fast == 10 and a.slow == 30

    b = make_strategy("MOMENTUM", cfg)
    assert isinstance(b, Momentum) and b.lookback == 60 and b.threshold == 0.05

    with pytest.raises(ValueError):
        make_strategy("unknown", cfg)
