"""Backtest engine, metrics, and result models."""

from investment_copilot.domain.backtest.engine import (
    BacktestError,
    EngineRun,
    benchmark_buy_and_hold,
    simulate_portfolio,
)
from investment_copilot.domain.backtest.metrics import (
    annualized_return,
    annualized_volatility,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)
from investment_copilot.domain.backtest.results import (
    BacktestResult,
    EquityPoint,
    StrategyMetrics,
    equity_points,
    equity_series,
)

__all__ = [
    "BacktestError",
    "BacktestResult",
    "EngineRun",
    "EquityPoint",
    "StrategyMetrics",
    "annualized_return",
    "annualized_volatility",
    "benchmark_buy_and_hold",
    "compute_metrics",
    "equity_points",
    "equity_series",
    "max_drawdown",
    "sharpe_ratio",
    "simulate_portfolio",
    "total_return",
    "win_rate",
]
