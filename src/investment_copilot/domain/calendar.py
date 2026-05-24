"""Forward-looking calendar of corporate events for the portfolio.

Two kinds of events in v1:

* ``report`` — earnings report dates extracted from the latest monitoring
  snapshot (BR ``next_report_estimated_date``). Has a concrete date.
* ``dividend`` — annual cash dividend *estimate* derived from the trailing
  ``dividend_yield`` and current market value. **Has no date in v1** — Stooq
  / BR scraping for actual ex-dividend dates is a separate iteration. The
  UI shows these as "Roczna szacowana wypłata" without a calendar slot.

Future kinds (placeholder slots in the type Literal): ``agm``, ``espi``,
``dividend_record``, ``dividend_payment``.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


CalendarEventKind = Literal[
    "report",
    "dividend",
    "agm",                  # walne zgromadzenie (planned, future)
    "espi",                 # ESPI announcement (future scrapers)
    "dividend_record",      # dzień prawa do dywidendy (future)
    "dividend_payment",     # data wypłaty (future)
]

EventImportance = Literal["high", "medium", "low"]


class CalendarEvent(BaseModel):
    """A single upcoming event tied to a portfolio holding."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(description="Canonical Stooq form, e.g. 'pkn.pl'.")
    name: str | None = None
    kind: CalendarEventKind
    event_date: date | None = Field(
        default=None,
        description="The scheduled date. None for date-less estimates (e.g. trailing dividends).",
    )
    label: str = Field(
        description="Short user-facing label, e.g. 'Raport Q1 2026', 'Dywidenda (estymat. roczna)'.",
    )
    description: str = Field(
        default="",
        description="Optional one-liner with extra context (yield %, amount, etc.).",
    )
    importance: EventImportance = "medium"
    # Approximate cash amount for ``dividend`` events (PLN), else None.
    amount_pln: float | None = None


class CalendarBundle(BaseModel):
    """Top-level calendar payload."""

    model_config = ConfigDict(frozen=True)

    events: list[CalendarEvent] = Field(default_factory=list)
    snapshot_age_days: int | None = Field(
        default=None,
        description=(
            "Age (days) of the monitoring snapshot the report dates came "
            "from. None when no snapshot exists yet."
        ),
    )
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "CalendarBundle",
    "CalendarEvent",
    "CalendarEventKind",
    "EventImportance",
]
