"""Domain models for company fundamentals + monitoring snapshots.

These are the cross-layer types used by the new monitoring pipeline.
Everything is best-effort: a missing source field becomes ``None`` so a
downstream renderer / LLM can degrade gracefully instead of crashing.

Snapshots are persisted as JSON sidecars next to the rendered HTML reports
so the next run can diff against the previous state.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field


SNAPSHOT_SCHEMA_VERSION: Final[str] = "1.0"


class FundamentalsSnapshot(BaseModel):
    """Snapshot of a company's headline fundamentals at a point in time.

    The ``source`` field documents WHICH data provider supplied this
    snapshot. Downstream rendering shows a badge so the user knows the
    data quality:

    * ``biznesradar`` — primary, rich (YoY %, sector, latest narrative)
    * ``stooq`` — best-effort scrape of Stooq snapshot panel
    * ``ohlcv_cache`` — derived from local OHLCV parquet (price/52w only)
    * ``empty`` — no data found anywhere
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    name: str | None = None

    # --- Price + market data (from any source) -----------------------------
    last_price: float | None = None
    market_cap: float | None = Field(
        default=None,
        description="Market capitalization in PLN (raw number, not 'mld').",
    )
    enterprise_value: float | None = None
    pe_ratio: float | None = None
    pbv_ratio: float | None = None
    eps: float | None = None
    dividend_yield: float | None = Field(
        default=None,
        description="Dividend yield as a fraction (0.025 == 2.5%).",
    )
    week52_high: float | None = None
    week52_low: float | None = None

    # --- BiznesRadar-rich fields (None when source is stooq/ohlcv_cache) ---
    sector: str | None = Field(
        default=None,
        description="Branża/sektor wg BiznesRadar, np. 'Sieci handlowe'.",
    )
    latest_quarter_label: str | None = Field(
        default=None,
        description="Etykieta ostatniego raportowanego okresu, np. '2025/Q4' lub '2025 (gru 25)'.",
    )
    last_report_date: date | None = Field(
        default=None,
        description="Data publikacji ostatniego dostępnego raportu kwartalnego/rocznego.",
    )
    next_report_estimated_date: date | None = Field(
        default=None,
        description=(
            "Szacowana data następnego raportu — ekstrapolowana z last_report_date "
            "+ ~90 dni (lub +365 dla rocznych). Tylko orientacyjna."
        ),
    )
    revenue_yoy_pct: float | None = Field(
        default=None,
        description="Procentowa zmiana przychodów r/r (np. 14.9 dla +14.9%).",
    )
    ebitda_yoy_pct: float | None = None
    net_profit_yoy_pct: float | None = None
    latest_summary: list[str] = Field(
        default_factory=list,
        description=(
            "Lista pre-computed narracyjnych podsumowań z BR, gotowych do "
            "zacytowania przez LLM, np. ['Przychody: wzrost o 14.9% r/r', "
            "'Zysk netto: wzrost o 3.5% r/r']."
        ),
    )

    # --- Source attribution ------------------------------------------------
    source: str = "stooq"
    fetched_at: datetime
    source_url: str | None = None


class EarningsCalendarEntry(BaseModel):
    """Upcoming earnings / report publication date."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    report_type: str = Field(
        description="Free-form label, e.g. 'Q1 2026', 'FY2025 wstępne'."
    )
    expected_date: date
    source: str = "llm"


class TickerNewsRef(BaseModel):
    """Compact news reference stored in the monitoring snapshot for diffing."""

    model_config = ConfigDict(frozen=True)

    title: str
    published_at: datetime
    source: str
    url: str | None = None


class MonitoringSnapshot(BaseModel):
    """The full monitoring snapshot persisted to disk as JSON.

    A new monitoring run reads the previous snapshot from disk to give the
    LLM both "what changed since last time" (data deltas) AND the previous
    report's per-ticker narrative — so when fresh data is sparse the LLM
    can carry forward the prior analysis instead of producing empty
    sections.

    The :attr:`report` field is forward-referenced as ``Any`` here to avoid
    a circular import with :mod:`domain.prompts.schemas`; it is the
    :class:`MonitoringReport` produced by the LLM in this run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    generated_at: datetime
    fundamentals: list[FundamentalsSnapshot] = Field(default_factory=list)
    news_by_ticker: dict[str, list[TickerNewsRef]] = Field(default_factory=dict)
    # Stored as a plain dict (model_dump output) to avoid a circular import.
    # The monitoring service is responsible for shaping/validating it.
    report: dict | None = Field(
        default=None,
        description=(
            "The full MonitoringReport produced by the LLM in this run, "
            "as a model_dump(). Used as fallback context for the next run."
        ),
    )

    def find_fundamentals(self, ticker: str) -> FundamentalsSnapshot | None:
        return next((f for f in self.fundamentals if f.ticker == ticker), None)


# --- ESPI / earnings news filtering ----------------------------------------

#: Polish keywords typical of ESPI/earnings announcements. Matched against
#: lowercased news titles. Hits surface to the LLM as "reporting-related"
#: signal so it can lean on them when scoring vs. expectations.
EARNINGS_KEYWORDS: tuple[str, ...] = (
    "raport okresowy",
    "raport kwartalny",
    "raport roczny",
    "raport polroczny",
    "raport półroczny",
    "skonsolidowany raport",
    "wstępne wyniki",
    "wstepne wyniki",
    "wyniki finansowe",
    "wyniki kwartalne",
    "wyniki roczne",
    "publikacja raportu",
    "kalendarz raportów",
    "kalendarz raportow",
    "harmonogram publikacji",
    "espi",
    "ebi",
    "rekomendacja",
    "dywidenda",
    "rada nadzorcza",
    "zwz",
    "wza",
    "przychody",
    "ebitda",
    "zysk netto",
)


def is_earnings_related(title: str) -> bool:
    """Return ``True`` if the news title looks ESPI/earnings-related."""
    if not title:
        return False
    needle = title.lower()
    return any(kw in needle for kw in EARNINGS_KEYWORDS)
