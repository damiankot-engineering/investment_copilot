"""Build the forward-looking calendar from existing data.

No new scrapers in v1. Sources:

* :class:`MonitoringSnapshot` (latest persisted JSON) → report dates +
  trailing dividend yield per holding.
* :class:`PortfolioStatus` → market value to convert yield into a PLN
  estimate.
* :class:`Portfolio` → human-readable name for each ticker.

Dividend events have ``event_date=None`` because v1 has no source for
actual ex-dividend / payment dates. They're surfaced as date-less yearly
estimates so the UI can show *some* income context, with a clear caveat.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from investment_copilot.domain.calendar import CalendarBundle, CalendarEvent
from investment_copilot.domain.portfolio import Portfolio, PortfolioStatus
from investment_copilot.services.monitoring_service import MonitoringService

logger = logging.getLogger(__name__)


def _importance_for_report(d: date, *, today: date | None = None) -> str:
    today = today or date.today()
    delta = (d - today).days
    if delta <= 14:
        return "high"
    if delta <= 60:
        return "medium"
    return "low"


class CalendarService:
    """Aggregate report + dividend events for the portfolio."""

    def __init__(self, *, monitoring_service: MonitoringService) -> None:
        self._monitoring = monitoring_service

    def build(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        today: date | None = None,
    ) -> CalendarBundle:
        today = today or date.today()
        warnings: list[str] = []
        events: list[CalendarEvent] = []

        snapshot = self._monitoring.load_latest_snapshot()
        snapshot_age: int | None = None
        if snapshot is None:
            warnings.append(
                "Brak monitoring snapshot — uruchom 'Monitoring → Uruchom snapshot' "
                "aby pobrać daty raportów."
            )
        else:
            generated_at = snapshot.generated_at
            if isinstance(generated_at, datetime):
                # Normalize to date for "age" math.
                gen_date = generated_at.astimezone(timezone.utc).date()
                snapshot_age = (today - gen_date).days

        # Map ticker -> name for nicer labels
        name_by_ticker = {h.ticker: h.name for h in portfolio.holdings}
        # Map ticker -> market value for dividend $ estimate
        mv_by_ticker = {
            s.ticker: s.market_value
            for s in status.holdings
            if s.market_value is not None
        }

        # Report events from BR fundamentals snapshots.
        if snapshot is not None:
            for f in snapshot.fundamentals:
                if f.next_report_estimated_date and f.next_report_estimated_date >= today:
                    period = f.latest_quarter_label or ""
                    label = f"Raport {period}".strip() or "Raport okresowy"
                    desc_parts = []
                    if f.last_report_date:
                        desc_parts.append(f"Ostatni: {f.last_report_date.isoformat()}")
                    if f.revenue_yoy_pct is not None:
                        desc_parts.append(f"Przychody {f.revenue_yoy_pct:+.1f}% r/r")
                    events.append(
                        CalendarEvent(
                            ticker=f.ticker,
                            name=name_by_ticker.get(f.ticker) or f.name,
                            kind="report",
                            event_date=f.next_report_estimated_date,
                            label=label,
                            description=" · ".join(desc_parts),
                            importance=_importance_for_report(
                                f.next_report_estimated_date, today=today
                            ),  # type: ignore[arg-type]
                        )
                    )

        # Dividend estimates (date-less; trailing yield × current MV).
        if snapshot is not None:
            for f in snapshot.fundamentals:
                if f.dividend_yield is None or f.dividend_yield <= 0:
                    continue
                mv = mv_by_ticker.get(f.ticker)
                if mv is None:
                    continue
                est_annual = f.dividend_yield * mv
                if est_annual <= 0:
                    continue
                events.append(
                    CalendarEvent(
                        ticker=f.ticker,
                        name=name_by_ticker.get(f.ticker) or f.name,
                        kind="dividend",
                        event_date=None,
                        label="Dywidenda (rocznie, est.)",
                        description=(
                            f"Yield {f.dividend_yield * 100:.2f}% × wartość "
                            f"~{mv:,.0f} PLN ≈ {est_annual:,.0f} PLN/rok"
                        ),
                        importance="low",
                        amount_pln=est_annual,
                    )
                )

        # Sort: dated events first (ascending date), then date-less.
        events.sort(
            key=lambda e: (
                0 if e.event_date is not None else 1,
                e.event_date or date.max,
                e.ticker,
            )
        )

        return CalendarBundle(
            events=events,
            snapshot_age_days=snapshot_age,
            warnings=warnings,
        )


__all__ = ["CalendarService"]
