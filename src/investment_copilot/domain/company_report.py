"""Per-company report schema.

This is the simplified version of the ``cyberfolks_raport`` template.
Field names match the JSON schema the HTML template reads via
``window.reportData`` so a ``CompanyReport.model_dump()`` can be injected
straight into the standalone HTML for download.

Sections we DON'T have data for (and therefore drop from the template):

* ``profile`` (ISIN, headquarters, CEO/CFO, employees, segments, reach)
* ``annual_table_*`` (3-year revenue/EBITDA history — BR only gives YoY
  for the latest period)
* ``strategy_events`` (M&A milestones — would need editorial input)
* ``company_suffix`` (we don't split "X S.A." into stem + suffix)

All deterministic fields come from BiznesRadar fundamentals, OHLCV cache,
and the user's portfolio. The narrative trio (``tldr`` / ``strengths`` /
``risks``) is produced by a SINGLE LLM call with must-cite grounding.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Trend = Literal["pos", "neg", "neu"]


class Kpi(BaseModel):
    """One cell in the 4×2 KPI grid."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=40)
    value: str = Field(min_length=1, max_length=40)
    delta: str = Field(default="", max_length=40)
    trend: Trend = "neu"


class LabelValue(BaseModel):
    """A label/value row used by the market section."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=60)
    value: str = Field(min_length=1, max_length=120)


class BulletWithCitation(BaseModel):
    """A single bullet in strengths/risks, grounded by one or more sources.

    Each entry in ``citations`` is one of:
    * ``metric:<key>`` — e.g. ``metric:revenue_yoy_pct``
    * ``news:<N>`` — N-th item in the rendered news block
    * ``fundamentals:<field>`` — e.g. ``fundamentals:pe_ratio``
    * ``thesis`` — the user's thesis text
    * ``portfolio:<field>`` — e.g. ``portfolio:unrealized_pnl_pct``

    Unknown citations are stripped at validation time by the service; a
    bullet with no surviving citation is dropped entirely.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=10, max_length=240)
    citations: list[str] = Field(min_length=1, max_length=4)


class CalendarItem(BaseModel):
    """One entry in the investor calendar.

    Mirrors the ``highlighted_item`` shape the HTML template expects:
    ``highlight`` is the date (or short label), ``text`` is the event
    description.
    """

    model_config = ConfigDict(extra="forbid")

    highlight: str = Field(min_length=1, max_length=40)
    text: str = Field(min_length=1, max_length=200)


Sentiment = Literal["positive", "negative", "neutral"]


class NewsHeadline(BaseModel):
    """One recent news/ESPI item shown in the factsheet news section.

    ``sentiment`` is ``None`` until the per-ticker AI report runs (and the
    batch sentiment LLM call classifies + caches it per URL). The factsheet
    renders unrated items without a coloured dot.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=240)
    date: str = Field(min_length=1, max_length=20)
    source: str = Field(default="", max_length=40)
    url: str | None = Field(default=None, max_length=400)
    sentiment: Sentiment | None = None


class CompanyReport(BaseModel):
    """The single-company snapshot report.

    Serializes to JSON whose keys are 1:1 with the
    ``cyberfolks_dane.json`` template, so the front-end (and the
    standalone HTML export) can render it without any field remapping.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Header ------------------------------------------------------------
    eyebrow: str = Field(default="Monitoring portfela · Snapshot", max_length=80)
    company_name: str = Field(min_length=1, max_length=120)
    ticker: str = Field(min_length=1, max_length=40)
    report_date: str = Field(min_length=1, max_length=20)
    # Optional pass-throughs from upstream data (rendered when present)
    sector: str | None = Field(default=None, max_length=120)

    # --- Streszczenie ------------------------------------------------------
    tldr: str = Field(min_length=20, max_length=600)

    # --- Zmiana vs poprzedni raport (LLM, gdy jest prior) ------------------
    change_since_last: str | None = Field(default=None, max_length=400)

    # --- KPI grid ----------------------------------------------------------
    kpi_section_title: str = Field(min_length=1, max_length=80)
    kpis: list[Kpi] = Field(min_length=4, max_length=8)

    # --- Dane rynkowe ------------------------------------------------------
    market: list[LabelValue] = Field(default_factory=list, max_length=12)

    # --- Newsy (Top 3, z sentymentem gdy sklasyfikowane) ------------------
    news: list[NewsHeadline] = Field(default_factory=list, max_length=5)

    # --- Mocne / Ryzyka ----------------------------------------------------
    strengths: list[BulletWithCitation] = Field(default_factory=list, max_length=6)
    risks: list[BulletWithCitation] = Field(default_factory=list, max_length=6)

    # --- Kalendarz ---------------------------------------------------------
    calendar: list[CalendarItem] = Field(default_factory=list, max_length=8)

    # --- Stopka ------------------------------------------------------------
    sources: str = Field(
        default=(
            "Źródła: BiznesRadar (fundamentals), Stooq (OHLCV), "
            "ESPI cache, portfolio.yaml. "
            "TL;DR/mocne/ryzyka generowane przez LLM z grounding "
            "na powyższych danych."
        ),
        max_length=600,
    )
    disclaimer: str = Field(
        default=(
            "Niniejszy materiał ma charakter wyłącznie informacyjny i nie "
            "stanowi rekomendacji inwestycyjnej w rozumieniu Rozporządzenia MAR."
        ),
        max_length=400,
    )
    # Metadata not rendered by the HTML template but useful for callers
    confidence: int = Field(default=5, ge=1, le=10)
    generated_at_iso: str | None = None
    warnings: list[str] = Field(default_factory=list)
