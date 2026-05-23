"""Quantitative metrics injected into LLM prompts.

The CopilotService runs these BEFORE calling the LLM and renders the
results as a Markdown block. Goal: keep the LLM in the role of
*interpreter*, not *calculator*. Anything that can be computed deterministically
(HHI, beta, returns, correlations) is computed here and quoted to the model.

All computations are pure (`pandas`/`numpy`/`math` only — no providers,
no I/O). The orchestrator/service is responsible for loading the OHLCV
panel and feeding it in.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from investment_copilot.domain.portfolio import Portfolio, PortfolioStatus

logger = logging.getLogger(__name__)


# --- Output models ----------------------------------------------------------


class HoldingMetrics(BaseModel):
    """Computed metrics for a single holding."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    weight_pct: float | None = Field(
        default=None,
        description="Share of total market value, in percent (0–100).",
    )
    ret_30d_pct: float | None = None
    ret_90d_pct: float | None = None
    ret_252d_pct: float | None = None
    distance_from_52w_high_pct: float | None = Field(
        default=None,
        description="Negative — how far below the trailing 52-week high (percent).",
    )
    distance_from_52w_low_pct: float | None = Field(
        default=None,
        description="Positive — how far above the trailing 52-week low (percent).",
    )
    ann_volatility_pct: float | None = None
    beta_vs_benchmark: float | None = None


class CorrelationPair(BaseModel):
    model_config = ConfigDict(frozen=True)
    ticker_a: str
    ticker_b: str
    correlation: float


class PortfolioMetrics(BaseModel):
    """Computed metrics for the portfolio as a whole."""

    model_config = ConfigDict(frozen=True)

    n_holdings: int
    n_priced: int
    hhi: float | None = Field(
        default=None,
        description="Herfindahl-Hirschman Index of weights (0–10000). >2500 = high.",
    )
    top3_weight_pct: float | None = None
    largest_position_ticker: str | None = None
    largest_position_weight_pct: float | None = None
    benchmark_symbol: str | None = None
    holdings: list[HoldingMetrics] = Field(default_factory=list)
    top_correlations: list[CorrelationPair] = Field(
        default_factory=list,
        description="Up to 5 strongest absolute pairwise correlations.",
    )


# --- Core computations ------------------------------------------------------


def _pct_return(series: pd.Series, lookback_days: int) -> float | None:
    """Trailing return over the last ``lookback_days`` trading sessions, in percent."""
    s = series.dropna()
    if len(s) < lookback_days + 1:
        return None
    start = float(s.iloc[-lookback_days - 1])
    end = float(s.iloc[-1])
    if start <= 0:
        return None
    return (end / start - 1.0) * 100.0


def _distance_from_extrema(
    series: pd.Series, *, today: date | None = None, lookback_days: int = 252
) -> tuple[float | None, float | None]:
    """Return ``(distance_from_high, distance_from_low)`` both in percent.

    Distance from high is **negative** when price is below the trailing
    high (the typical case); distance from low is **positive** when price
    is above the trailing low.
    """
    s = series.dropna().tail(lookback_days)
    if s.empty:
        return None, None
    last = float(s.iloc[-1])
    hi = float(s.max())
    lo = float(s.min())
    high_dist = (last / hi - 1.0) * 100.0 if hi > 0 else None
    low_dist = (last / lo - 1.0) * 100.0 if lo > 0 else None
    return high_dist, low_dist


def _ann_volatility(series: pd.Series, *, trading_days: int = 252) -> float | None:
    """Annualized stdev of daily log returns, percent."""
    s = series.dropna()
    if len(s) < 20:
        return None
    rets = np.log(s / s.shift(1)).dropna()
    if rets.empty:
        return None
    return float(rets.std() * math.sqrt(trading_days) * 100.0)


def _beta(asset: pd.Series, benchmark: pd.Series) -> float | None:
    """OLS beta of asset returns vs benchmark returns. Daily, aligned on dates."""
    aligned = pd.concat(
        [asset.rename("a"), benchmark.rename("b")], axis=1, join="inner"
    ).dropna()
    if len(aligned) < 30:
        return None
    a_ret = np.log(aligned["a"] / aligned["a"].shift(1)).dropna()
    b_ret = np.log(aligned["b"] / aligned["b"].shift(1)).dropna()
    pair = pd.concat([a_ret, b_ret], axis=1, join="inner").dropna()
    if len(pair) < 30:
        return None
    cov = float(pair.iloc[:, 0].cov(pair.iloc[:, 1]))
    var = float(pair.iloc[:, 1].var())
    if var <= 0:
        return None
    return cov / var


def _pairwise_correlations(
    panel: Mapping[str, pd.Series], top_n: int = 5
) -> list[CorrelationPair]:
    """Top-N absolute pairwise correlations of daily log returns."""
    if len(panel) < 2:
        return []
    returns: dict[str, pd.Series] = {}
    for ticker, series in panel.items():
        s = series.dropna()
        if len(s) < 30:
            continue
        returns[ticker] = np.log(s / s.shift(1)).dropna()
    if len(returns) < 2:
        return []
    df = pd.concat(returns, axis=1, join="inner").dropna()
    if len(df) < 30:
        return []
    corr = df.corr()
    pairs: list[CorrelationPair] = []
    seen: set[frozenset[str]] = set()
    for t1 in corr.index:
        for t2 in corr.columns:
            if t1 == t2:
                continue
            key = frozenset((t1, t2))
            if key in seen:
                continue
            seen.add(key)
            value = float(corr.loc[t1, t2])
            if math.isnan(value):
                continue
            pairs.append(CorrelationPair(ticker_a=t1, ticker_b=t2, correlation=value))
    pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
    return pairs[:top_n]


# --- Top-level assembler ----------------------------------------------------


def compute_portfolio_metrics(
    portfolio: Portfolio,
    status: PortfolioStatus,
    *,
    ohlcv_panel: Mapping[str, pd.DataFrame],
    benchmark_close: pd.Series | None = None,
    benchmark_symbol: str | None = None,
) -> PortfolioMetrics:
    """Compute the full set of quantitative metrics for the prompt context.

    Parameters
    ----------
    ohlcv_panel:
        Map of ``ticker -> OHLCV DataFrame`` (with at least a ``close``
        column, DatetimeIndex). Tickers without data are silently skipped
        in per-holding metrics.
    benchmark_close:
        Closing-price series of the benchmark, used to compute beta.
        Pass ``None`` to skip beta.
    """
    closes_by_ticker: dict[str, pd.Series] = {
        t: df["close"]
        for t, df in ohlcv_panel.items()
        if isinstance(df, pd.DataFrame) and "close" in df.columns and not df.empty
    }

    total_mv = status.total_market_value or 0.0

    holding_metrics: list[HoldingMetrics] = []
    weights: list[tuple[str, float]] = []
    for s in status.holdings:
        weight = (
            (s.market_value / total_mv * 100.0)
            if total_mv > 0 and s.market_value is not None
            else None
        )
        if weight is not None:
            weights.append((s.ticker, weight))

        close = closes_by_ticker.get(s.ticker)
        if close is None or close.empty:
            holding_metrics.append(
                HoldingMetrics(ticker=s.ticker, weight_pct=weight)
            )
            continue

        hi_dist, lo_dist = _distance_from_extrema(close)
        holding_metrics.append(
            HoldingMetrics(
                ticker=s.ticker,
                weight_pct=weight,
                ret_30d_pct=_pct_return(close, 30),
                ret_90d_pct=_pct_return(close, 90),
                ret_252d_pct=_pct_return(close, 252),
                distance_from_52w_high_pct=hi_dist,
                distance_from_52w_low_pct=lo_dist,
                ann_volatility_pct=_ann_volatility(close),
                beta_vs_benchmark=(
                    _beta(close, benchmark_close)
                    if benchmark_close is not None and not benchmark_close.empty
                    else None
                ),
            )
        )

    hhi: float | None = None
    top3_weight: float | None = None
    largest: tuple[str, float] | None = None
    if weights:
        hhi = float(sum(w * w for _, w in weights))  # Σ(w%)² in (0, 10000)
        sorted_w = sorted(weights, key=lambda x: x[1], reverse=True)
        top3_weight = float(sum(w for _, w in sorted_w[:3]))
        largest = sorted_w[0]

    correlations = _pairwise_correlations(closes_by_ticker)

    return PortfolioMetrics(
        n_holdings=len(portfolio.holdings),
        n_priced=sum(1 for s in status.holdings if s.has_price),
        hhi=hhi,
        top3_weight_pct=top3_weight,
        largest_position_ticker=largest[0] if largest else None,
        largest_position_weight_pct=largest[1] if largest else None,
        benchmark_symbol=benchmark_symbol,
        holdings=holding_metrics,
        top_correlations=correlations,
    )


# --- Reference registry for citation validation -----------------------------


class CitationRegistry(BaseModel):
    """Set of valid citation targets the LLM can reference.

    Built from the same data we feed into the prompt, so the LLM
    citing `news:5` or `metric:pkn.pl.ret_30d_pct` can be verified
    deterministically.
    """

    model_config = ConfigDict(frozen=True)

    news_ids: set[str] = Field(default_factory=set)
    metric_keys: set[str] = Field(default_factory=set)
    fundamentals_keys: set[str] = Field(default_factory=set)
    report_keys: set[str] = Field(default_factory=set)

    def is_known(self, source_type: str, reference: str) -> bool:
        # Lenient: accept either bare key (`pkn.pl.ret_30d_pct`) or the
        # display form with redundant prefix (`metric:pkn.pl.ret_30d_pct`).
        # Models trained on similar formats often produce both.
        raw = (reference or "").strip().lower()
        ref = raw.split(":", 1)[1] if ":" in raw and raw.startswith(
            (f"{source_type}:", "metric:", "news:", "fundamentals:", "previous_report:")
        ) else raw
        match source_type:
            case "news":
                return raw in self.news_ids or f"news:{ref}" in self.news_ids
            case "metric":
                return ref in self.metric_keys
            case "fundamentals":
                return ref in self.fundamentals_keys
            case "previous_report":
                return raw in self.report_keys or f"previous_report:{ref}" in self.report_keys
            case _:
                return False


def build_metric_keys(metrics: PortfolioMetrics) -> set[str]:
    """All `ticker.field_name` citation keys derivable from `metrics`."""
    keys: set[str] = {
        "portfolio.hhi",
        "portfolio.top3_weight_pct",
        "portfolio.largest_position_weight_pct",
    }
    interesting = (
        "weight_pct",
        "ret_30d_pct",
        "ret_90d_pct",
        "ret_252d_pct",
        "distance_from_52w_high_pct",
        "distance_from_52w_low_pct",
        "ann_volatility_pct",
        "beta_vs_benchmark",
    )
    for h in metrics.holdings:
        for field in interesting:
            keys.add(f"{h.ticker.lower()}.{field}")
    for c in metrics.top_correlations:
        a, b = sorted([c.ticker_a.lower(), c.ticker_b.lower()])
        keys.add(f"corr.{a}.{b}")
    return keys


def filter_unknown_citations(
    citations: Sequence["Citation"],  # noqa: F821 - forward ref for type-hint only
    registry: CitationRegistry,
) -> tuple[list, list]:
    """Split citations into ``(valid, dropped)`` based on the registry."""
    valid: list = []
    dropped: list = []
    for c in citations:
        if registry.is_known(getattr(c, "source_type", ""), getattr(c, "reference", "")):
            valid.append(c)
        else:
            dropped.append(c)
    return valid, dropped


__all__ = [
    "CitationRegistry",
    "CorrelationPair",
    "HoldingMetrics",
    "PortfolioMetrics",
    "build_metric_keys",
    "compute_portfolio_metrics",
    "filter_unknown_citations",
]
