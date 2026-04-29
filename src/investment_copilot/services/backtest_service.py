"""Backtest service.

Wires together:

* the user's :class:`~investment_copilot.domain.portfolio.Portfolio`,
* cached OHLCV from
  :class:`~investment_copilot.services.data_service.DataService`,
* a :class:`~investment_copilot.domain.strategies.base.Strategy`,
* the simulator and metrics from
  :mod:`~investment_copilot.domain.backtest`,

into a single typed :class:`~investment_copilot.domain.backtest.BacktestResult`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Mapping

import pandas as pd

from investment_copilot.config.schema import BacktestConfig, StrategiesConfig
from investment_copilot.domain.backtest import (
    BacktestError,
    BacktestResult,
    benchmark_buy_and_hold,
    compute_metrics,
    equity_points,
    simulate_portfolio,
)
from investment_copilot.domain.portfolio import Portfolio
from investment_copilot.domain.strategies import Strategy, make_strategy
from investment_copilot.services.data_service import DataService

logger = logging.getLogger(__name__)


class BacktestService:
    """Run portfolio-level backtests over a configured time window."""

    def __init__(
        self,
        *,
        data_service: DataService,
        backtest_config: BacktestConfig,
        strategies_config: StrategiesConfig,
    ) -> None:
        self._data = data_service
        self._bt_cfg = backtest_config
        self._strat_cfg = strategies_config

    # -- Public API ---------------------------------------------------------

    def run(
        self,
        portfolio: Portfolio,
        *,
        strategy_name: str,
        start: date | None = None,
        end: date | None = None,
        include_benchmark: bool = True,
    ) -> BacktestResult:
        """Run ``strategy_name`` on ``portfolio`` and return a typed result."""
        strategy = make_strategy(strategy_name, self._strat_cfg)
        start_date = start or self._bt_cfg.start_date
        end_date = end or self._bt_cfg.end_date

        panel = self._load_panel(
            tickers=portfolio.tickers,
            start=start_date,
            end=end_date,
            warmup_days=strategy.warmup_days,
        )
        if not panel:
            raise BacktestError(
                "No OHLCV data available for any portfolio ticker; "
                "run `update-data` first."
            )

        missing = sorted(set(portfolio.tickers) - set(panel.keys()))
        signals = strategy.generate_signals(panel)

        run = simulate_portfolio(
            panel,
            signals,
            initial_capital=self._bt_cfg.initial_capital,
            start=start_date,
            end=end_date,
        )

        metrics = compute_metrics(
            run.equity_curve,
            trading_days_per_year=self._bt_cfg.trading_days_per_year,
        )

        bench_symbol: str | None = None
        bench_curve_points = []
        bench_metrics = None
        if include_benchmark:
            bench_symbol, bench_equity = self._load_benchmark(start_date, end_date)
            if bench_equity is not None and not bench_equity.empty:
                bench_curve_points = equity_points(bench_equity)
                bench_metrics = compute_metrics(
                    bench_equity,
                    trading_days_per_year=self._bt_cfg.trading_days_per_year,
                )
            else:
                logger.warning(
                    "Benchmark %s has no data in window; skipping",
                    self._bt_cfg.benchmark,
                )
                bench_symbol = None

        equity_curve_points = equity_points(run.equity_curve)
        actual_start = equity_curve_points[0].date if equity_curve_points else start_date
        actual_end = equity_curve_points[-1].date if equity_curve_points else (
            end_date or start_date
        )

        return BacktestResult(
            strategy_name=strategy.name,
            strategy_params=self._strategy_params(strategy),
            start_date=actual_start,
            end_date=actual_end,
            initial_capital=self._bt_cfg.initial_capital,
            final_value=float(run.equity_curve.iloc[-1]),
            equity_curve=equity_curve_points,
            metrics=metrics,
            benchmark_symbol=bench_symbol,
            benchmark_equity_curve=bench_curve_points,
            benchmark_metrics=bench_metrics,
            missing_tickers=missing,
            tickers_used=run.tickers_used,
            generated_at=datetime.now(timezone.utc),
        )

    # -- Internal -----------------------------------------------------------

    def _load_panel(
        self,
        *,
        tickers: list[str],
        start: date,
        end: date | None,
        warmup_days: int,
    ) -> dict[str, pd.DataFrame]:
        # Pad start by ~2x warmup for safety against weekends/holidays.
        pad_days = max(warmup_days * 2, 5)
        load_start = start - timedelta(days=pad_days + warmup_days)
        panel: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self._data.load_ohlcv(ticker, start=load_start, end=end)
            if df is None or df.empty:
                logger.warning("Backtest: no cached data for %s, skipping", ticker)
                continue
            panel[ticker] = df
        return panel

    def _load_benchmark(
        self,
        start: date,
        end: date | None,
    ) -> tuple[str | None, pd.Series | None]:
        symbol = self._bt_cfg.benchmark
        df = self._data.load_benchmark(symbol, start=start, end=end)
        if df is None or df.empty:
            return None, None
        try:
            equity = benchmark_buy_and_hold(
                df["close"],
                initial_capital=self._bt_cfg.initial_capital,
                start=start,
                end=end,
            )
        except BacktestError:
            return None, None
        # Resolve the actual cached symbol for the result
        from investment_copilot.domain.models import resolve_benchmark

        return resolve_benchmark(symbol), equity

    @staticmethod
    def _strategy_params(strategy: Strategy) -> dict[str, float | int | str]:
        # Surface only public, primitive attributes for the result payload.
        out: dict[str, float | int | str] = {}
        for attr in ("fast", "slow", "lookback", "threshold"):
            if hasattr(strategy, attr):
                out[attr] = getattr(strategy, attr)
        return out
