"""Pydantic schema for application configuration.

Every section of ``config.yaml`` maps to a model below. The top-level
:class:`AppConfig` is what the rest of the application receives; nothing else
should ever read the raw YAML directly.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, field_validator


# --- Providers ---------------------------------------------------------------


MarketDataProviderName = Literal["stooq"]
NewsProviderName = Literal["stooq", "rss", "newsapi"]
FundamentalsProviderName = Literal["alpha_vantage", "none"]


class ProvidersConfig(BaseModel):
    """Selects concrete data providers used by the data layer."""

    market_data: MarketDataProviderName = "stooq"
    news: list[NewsProviderName] = Field(default_factory=lambda: ["stooq", "rss"])
    fundamentals: FundamentalsProviderName = "none"

    rss_feeds: list[str] = Field(
        default_factory=lambda: [
            "https://www.bankier.pl/rss/wiadomosci.xml",
            "https://www.money.pl/rss/",
        ],
        description="RSS feeds polled by the RSS news provider.",
    )

    newsapi_api_key: str | None = None
    alpha_vantage_api_key: str | None = None


# --- Storage -----------------------------------------------------------------


class StorageConfig(BaseModel):
    """Local persistence layer (SQLite metadata + parquet OHLCV cache)."""

    sqlite_path: Path = Path("data/cache.db")
    parquet_dir: Path = Path("data/ohlcv")

    @field_validator("sqlite_path", "parquet_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(v).expanduser()


# --- Portfolio reference -----------------------------------------------------


class PortfolioRefConfig(BaseModel):
    """Where the user's portfolio YAML lives."""

    path: Path = Path("portfolio.yaml")

    @field_validator("path")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(v).expanduser()


# --- Strategies --------------------------------------------------------------


class MACrossoverParams(BaseModel):
    fast: PositiveInt = 50
    slow: PositiveInt = 200

    @field_validator("slow")
    @classmethod
    def _slow_gt_fast(cls, v: int, info) -> int:  # type: ignore[no-untyped-def]
        fast = info.data.get("fast")
        if fast is not None and v <= fast:
            raise ValueError("slow window must be greater than fast window")
        return v


class MomentumParams(BaseModel):
    """Time-series momentum (TSMOM): long when trailing return > threshold."""

    lookback: PositiveInt = 126   # ~6 months of trading days
    threshold: float = 0.0        # minimum trailing return to trigger long


class StrategiesConfig(BaseModel):
    ma_crossover: MACrossoverParams = MACrossoverParams()
    momentum: MomentumParams = MomentumParams()


# --- Backtest ----------------------------------------------------------------


class BacktestConfig(BaseModel):
    start_date: date = date(2020, 1, 1)
    end_date: date | None = None  # None means "up to latest available"
    benchmark: str = "wig20"
    initial_capital: PositiveFloat = 100_000.0
    trading_days_per_year: PositiveInt = 252


# --- LLM ---------------------------------------------------------------------


class LLMConfig(BaseModel):
    provider: Literal["groq"] = "groq"
    api_key: str  # required; must resolve from env
    model_analysis: str = "llama-3.3-70b-versatile"
    model_summary: str = "llama-3.1-8b-instant"
    language: Literal["pl", "en"] = "pl"
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: PositiveInt = 2048
    request_timeout_s: PositiveInt = 60


# --- Logging -----------------------------------------------------------------


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# --- Top-level ---------------------------------------------------------------


class AppConfig(BaseModel):
    """The single, validated configuration object passed around the app."""

    model_config = {"extra": "forbid"}  # typos in config.yaml become errors

    providers: ProvidersConfig = ProvidersConfig()
    storage: StorageConfig = StorageConfig()
    portfolio: PortfolioRefConfig = PortfolioRefConfig()
    strategies: StrategiesConfig = StrategiesConfig()
    backtest: BacktestConfig = BacktestConfig()
    llm: LLMConfig
    logging: LoggingConfig = LoggingConfig()
