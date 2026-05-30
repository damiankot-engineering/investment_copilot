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
        benchmark: str | None = None,
    ) -> BacktestResult:
        """Run ``strategy_name`` on ``portfolio`` and return a typed result."""
        strategy = make_strategy(strategy_name, self._strat_cfg)
        start_date = start or self._bt_cfg.start_date
        end_date = end or self._bt_cfg.end_date

        # Initial capital = FIFO cost basis of each holding at the LATER of
        # the user's backtest start and the holding's first BUY. This handles
        # both cases cleanly:
        #
        # * Backtest 2020 with positions acquired 2024 → each holding's
        #   contribution kicks in from its own first BUY date, so the user
        #   sees realistic 'as if I had held this' returns instead of a
        #   zero-capital error.
        # * Backtest 2024 with positions acquired 2022 + 2023 SELLs → uses
        #   FIFO basis of what's still active by 2024.
        #
        # Transactions DURING the window are not yet simulated (a v2
        # extension would flow cash on each BUY/SELL).
        initial_capital = 0.0
        for holding in portfolio.holdings:
            anchor = max(start_date, holding.first_entry_date)
            initial_capital += holding.position_at(anchor)[1]

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
            initial_capital=initial_capital,
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
            bench_symbol, bench_equity = self._load_benchmark(
                start_date, end_date, initial_capital, benchmark_override=benchmark
            )
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
            initial_capital=initial_capital,
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
        initial_capital: float,
        *,
        benchmark_override: str | None = None,
    ) -> tuple[str | None, pd.Series | None]:
        symbol = benchmark_override or self._bt_cfg.benchmark
        # Auto-extend the benchmark cache backwards if the user requested a
        # start date earlier than what we have on disk. Without this, picking
        # e.g. 2025-12-01 in the UI silently chops the benchmark line at
        # whatever the cache's earliest date happens to be.
        self._ensure_benchmark_covers(symbol, start, end)
        df = self._data.load_benchmark(symbol, start=start, end=end)
        if df is None or df.empty:
            return None, None
        try:
            equity = benchmark_buy_and_hold(
                df["close"],
                initial_capital=initial_capital,
                start=start,
                end=end,
            )
        except BacktestError:
            return None, None
        # Resolve the actual cached symbol for the result
        from investment_copilot.domain.models import resolve_benchmark

        return resolve_benchmark(symbol), equity

    def _ensure_benchmark_covers(
        self, symbol: str, start: date, end: date | None
    ) -> None:
        """Refresh the benchmark cache backwards if it doesn't cover ``start``.

        Best-effort — if the refresh fails (network down, provider rate
        limited), fall through to whatever is on disk. Adds a small
        padding so we have a price ON the start date (else the first
        comparison point sits later than the portfolio's).
        """
        try:
            cached = self._data.load_benchmark(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Benchmark cache probe failed for %s: %s", symbol, exc)
            cached = None
        if cached is None or cached.empty:
            need_refresh = True
        else:
            earliest = cached.index.min().date()
            need_refresh = earliest > start
        if not need_refresh:
            return
        load_start = start - timedelta(days=5)
        logger.info(
            "Backtest: extending benchmark %s cache backwards to %s",
            symbol, load_start,
        )
        try:
            self._data.refresh_benchmark(symbol, start=load_start, end=end)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Benchmark backfill for %s failed (%s) — using cached range only.",
                symbol, exc,
            )

    @staticmethod
    def _strategy_params(strategy: Strategy) -> dict[str, float | int | str]:
        # Surface only public, primitive attributes for the result payload.
        out: dict[str, float | int | str] = {}
        for attr in ("fast", "slow", "lookback", "threshold"):
            if hasattr(strategy, attr):
                out[attr] = getattr(strategy, attr)
        return out
