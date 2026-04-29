"""Trading strategies — pure signal generators."""

from __future__ import annotations

from investment_copilot.config.schema import StrategiesConfig
from investment_copilot.domain.strategies.base import Strategy
from investment_copilot.domain.strategies.ma_crossover import MACrossover
from investment_copilot.domain.strategies.momentum import Momentum

#: Names exposed via the CLI / API.
KNOWN_STRATEGIES: tuple[str, ...] = ("ma_crossover", "momentum")


def make_strategy(name: str, config: StrategiesConfig) -> Strategy:
    """Construct a :class:`Strategy` from its name and the global config."""
    n = name.strip().lower()
    if n == "ma_crossover":
        return MACrossover(fast=config.ma_crossover.fast, slow=config.ma_crossover.slow)
    if n == "momentum":
        return Momentum(
            lookback=config.momentum.lookback,
            threshold=config.momentum.threshold,
        )
    raise ValueError(
        f"Unknown strategy: {name!r}. Known: {', '.join(KNOWN_STRATEGIES)}"
    )


__all__ = ["KNOWN_STRATEGIES", "MACrossover", "Momentum", "Strategy", "make_strategy"]
