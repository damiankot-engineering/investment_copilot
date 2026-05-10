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
    """Snapshot of a company's headline fundamentals at a point in time."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    name: str | None = None
    last_price: float | None = None
    market_cap: float | None = Field(
        default=None,
        description="Market capitalization in PLN (raw number, not 'mld').",
    )
    pe_ratio: float | None = None
    pbv_ratio: float | None = None
    eps: float | None = None
    dividend_yield: float | None = Field(
        default=None,
        description="Dividend yield as a fraction (0.025 == 2.5%).",
    )
    week52_high: float | None = None
    week52_low: float | None = None
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
