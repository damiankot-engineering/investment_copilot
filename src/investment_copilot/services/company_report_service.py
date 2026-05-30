"""Per-company report service.

Produces a :class:`CompanyReport` for a single ticker by combining:

* **Deterministic data** (BiznesRadar fundamentals, OHLCV cache, news cache,
  portfolio status) → header, KPI grid, market section, calendar, footer.
* **A single LLM call** that emits the narrative trio (TL;DR + strengths +
  risks) constrained by a must-cite schema. Citations are verified
  Python-side and unknown ones are dropped before the report is returned.

This service intentionally REPLACES the legacy ``MonitoringService``
all-in-one pipeline. The legacy service stays in the codebase until the
frontend stops calling its endpoints.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

from investment_copilot.config.schema import LLMConfig
from investment_copilot.domain.company_report import (
    BulletWithCitation,
    CalendarItem,
    CompanyReport,
    Kpi,
    LabelValue,
)
from investment_copilot.domain.fundamentals import DividendEvent, FundamentalsSnapshot
from investment_copilot.domain.models import NewsItem, normalize_ticker
from investment_copilot.domain.portfolio import (
    Holding,
    HoldingStatus,
    Portfolio,
    PortfolioStatus,
)
from investment_copilot.domain.prompts import (
    COMPANY_NARRATIVE_SYSTEM,
    COMPANY_NARRATIVE_USER_TEMPLATE,
    CompanyNarrative,
)
from investment_copilot.infrastructure.llm import LLMClient
from investment_copilot.infrastructure.providers.base import ProviderError
from investment_copilot.infrastructure.providers.biznesradar import BiznesRadarProvider
from investment_copilot.infrastructure.storage import SQLiteStore
from investment_copilot.services.data_service import DataService
from investment_copilot.services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


BR_CACHE_TTL: timedelta = timedelta(hours=24)
DIVIDENDS_CACHE_TTL: timedelta = timedelta(days=7)
NEWS_DAYS_BACK: int = 30
NEWS_TOP_N: int = 3
WEEK52_TRADING_DAYS: int = 252
FIVE_YEAR_TRADING_DAYS: int = 252 * 5
REPORT_CACHE_KEY_PREFIX: str = "company_report:"
DIVIDENDS_CACHE_KEY_PREFIX: str = "biznesradar:dividends:"


class CompanyReportService:
    """Builds per-company snapshot reports (deterministic + LLM narrative)."""

    def __init__(
        self,
        *,
        data_service: DataService,
        portfolio_service: PortfolioService,
        sqlite_store: SQLiteStore,
        llm_client: LLMClient,
        llm_config: LLMConfig,
        biznesradar_provider: BiznesRadarProvider | None = None,
    ) -> None:
        self._data = data_service
        self._portfolio_svc = portfolio_service
        self._sqlite = sqlite_store
        self._llm = llm_client
        self._llm_cfg = llm_config
        self._br = biznesradar_provider or BiznesRadarProvider()

    # ---------------------------------------------------------------- Public

    def build_factsheet(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        ticker: str,
    ) -> CompanyReport:
        """Return a CompanyReport with deterministic sections filled.

        TL;DR / strengths / risks are populated with placeholders so the
        UI can render the card immediately. Call :meth:`generate_report`
        to add the LLM-generated narrative.
        """
        holding, holding_status = self._lookup(portfolio, status, ticker)
        fundamentals = self._fetch_fundamentals_cached(holding.ticker)
        fundamentals = self._overlay_ohlcv(holding.ticker, fundamentals)
        fundamentals = self._overlay_dividend_yield(holding.ticker, fundamentals)
        news = self._load_news(holding.ticker)
        return self._build_report(
            holding=holding,
            holding_status=holding_status,
            fundamentals=fundamentals,
            news=news,
            narrative=None,
        )

    def generate_report(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        ticker: str,
    ) -> CompanyReport:
        """Build factsheet + run the LLM narrative call + persist."""
        holding, holding_status = self._lookup(portfolio, status, ticker)
        fundamentals = self._fetch_fundamentals_cached(holding.ticker)
        fundamentals = self._overlay_ohlcv(holding.ticker, fundamentals)
        fundamentals = self._overlay_dividend_yield(holding.ticker, fundamentals)
        news = self._load_news(holding.ticker)

        warnings: list[str] = []
        narrative: CompanyNarrative | None
        try:
            narrative, dropped = self._call_llm_narrative(
                holding=holding,
                holding_status=holding_status,
                fundamentals=fundamentals,
                news=news,
            )
            if dropped:
                warnings.append(
                    f"Odrzucone cytowania bez źródła: {', '.join(dropped[:5])}"
                )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("LLM narrative failed for %s: %s", holding.ticker, exc)
            warnings.append(f"LLM nie wygenerował narracji: {exc}")
            narrative = None

        report = self._build_report(
            holding=holding,
            holding_status=holding_status,
            fundamentals=fundamentals,
            news=news,
            narrative=narrative,
        )
        report = report.model_copy(update={"warnings": warnings})
        self._save_to_cache(holding.ticker, report)
        return report

    def get_cached_report(self, ticker: str) -> CompanyReport | None:
        """Return the most recently saved CompanyReport for a ticker, or None."""
        cache_key = REPORT_CACHE_KEY_PREFIX + normalize_ticker(ticker)
        payload = self._sqlite.cache_get(cache_key, max_age=timedelta(days=365))
        if payload is None:
            return None
        try:
            return CompanyReport.model_validate_json(payload)
        except ValueError as exc:
            logger.warning("Cached report for %s invalid: %s", ticker, exc)
            return None

    def list_upcoming_reports(
        self, portfolio: Portfolio
    ) -> list[CalendarItem]:
        """Calendar across the whole portfolio (sorted, closest first).

        Mixes three kinds of events into one chronologically-sorted list:

        * Estimated next-report dates from BR (24h cache).
        * Upcoming dividend D-day (record date) from BR ``/dywidenda``
          (7-day cache) — last day a position must be held to qualify.
        * Upcoming dividend payment date — informational, when the cash
          hits the brokerage account.
        """
        items: list[tuple[date, CalendarItem]] = []
        for h in portfolio.holdings:
            f = self._fetch_fundamentals_cached(h.ticker)
            if (
                f is not None
                and f.next_report_estimated_date is not None
                and f.next_report_estimated_date >= date.today()
            ):
                days = (f.next_report_estimated_date - date.today()).days
                items.append((
                    f.next_report_estimated_date,
                    CalendarItem(
                        highlight=f.next_report_estimated_date.isoformat(),
                        text=(
                            f"{h.ticker.upper()} · raport "
                            f"({self._format_countdown(days)})"
                        ),
                    ),
                ))
            # Upcoming dividend events for this ticker
            for ev in self._fetch_dividends_cached(h.ticker):
                if not ev.is_upcoming:
                    continue
                amount = (
                    f" · {ev.amount_per_share:.2f} PLN/akcję"
                    if ev.amount_per_share is not None else ""
                )
                if ev.record_date and ev.record_date >= date.today():
                    days = (ev.record_date - date.today()).days
                    items.append((
                        ev.record_date,
                        CalendarItem(
                            highlight=ev.record_date.isoformat(),
                            text=(
                                f"{h.ticker.upper()} · D-day dywidendy"
                                f"{amount} ({self._format_countdown(days)})"
                            ),
                        ),
                    ))
                if ev.payment_date and ev.payment_date >= date.today():
                    days = (ev.payment_date - date.today()).days
                    items.append((
                        ev.payment_date,
                        CalendarItem(
                            highlight=ev.payment_date.isoformat(),
                            text=(
                                f"{h.ticker.upper()} · wypłata dywidendy"
                                f"{amount} ({self._format_countdown(days)})"
                            ),
                        ),
                    ))
        items.sort(key=lambda t: t[0])
        return [it for _, it in items]

    def _overlay_dividend_yield(
        self,
        ticker: str,
        fundamentals: FundamentalsSnapshot | None,
    ) -> FundamentalsSnapshot | None:
        """Compute a trailing dividend yield when BR doesn't provide one.

        BiznesRadar's "stopa dywidendy" column is JS-rendered (empty in the
        static HTML we scrape), so ``fundamentals.dividend_yield`` is almost
        always ``None``. We derive it instead as ``latest_annual_dividend /
        current_price``: take the most recent fiscal year's total dividend
        per share from the ``/dywidenda`` table (covers both upcoming
        "uchwalona" and paid years) over the latest close. Stored as a
        fraction (0.0126 == 1.26%) to match the field's convention.
        """
        if fundamentals is None or fundamentals.dividend_yield is not None:
            return fundamentals
        price = fundamentals.last_price
        if price is None or price <= 0:
            return fundamentals
        events = self._fetch_dividends_cached(ticker)
        with_amount = [e for e in events if e.amount_per_share is not None]
        if not with_amount:
            return fundamentals
        latest = max(with_amount, key=lambda e: e.fiscal_year)
        if latest.amount_per_share is None or latest.amount_per_share <= 0:
            return fundamentals
        yld = latest.amount_per_share / price
        return fundamentals.model_copy(update={"dividend_yield": yld})

    # --------------------------------------------------------------- Internal

    def _lookup(
        self, portfolio: Portfolio, status: PortfolioStatus, ticker: str,
    ) -> tuple[Holding, HoldingStatus | None]:
        norm = normalize_ticker(ticker)
        holding = portfolio.find(norm)
        if holding is None:
            raise ValueError(f"Ticker {ticker!r} not found in portfolio")
        hs = next(
            (s for s in status.holdings if s.ticker.lower() == norm.lower()), None
        )
        return holding, hs

    def _fetch_fundamentals_cached(
        self, ticker: str
    ) -> FundamentalsSnapshot | None:
        cache_key = f"biznesradar:fundamentals:{normalize_ticker(ticker)}"
        cached = self._sqlite.cache_get(cache_key, max_age=BR_CACHE_TTL)
        if cached:
            try:
                return FundamentalsSnapshot.model_validate_json(cached)
            except ValueError as exc:
                logger.warning("Cached BR for %s invalid: %s", ticker, exc)
        try:
            snap = self._br.fetch_fundamentals(ticker)
        except ProviderError as exc:
            logger.info("BR fundamentals fetch failed for %s: %s", ticker, exc)
            return None
        try:
            self._sqlite.cache_set(cache_key, snap.model_dump_json())
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to cache BR for %s: %s", ticker, exc)
        return snap

    def _fetch_dividends_cached(self, ticker: str) -> list[DividendEvent]:
        """Fetch BR dividend events for ``ticker`` with 7-day SQLite cache.

        Returns ``[]`` on any failure (no fundamentals page, network down,
        parse error) — dividend data is nice-to-have, never load-bearing.
        """
        norm = normalize_ticker(ticker)
        cache_key = DIVIDENDS_CACHE_KEY_PREFIX + norm
        cached = self._sqlite.cache_get(cache_key, max_age=DIVIDENDS_CACHE_TTL)
        if cached is not None:
            try:
                payload = json.loads(cached)
                return [DividendEvent.model_validate(p) for p in payload]
            except (ValueError, KeyError) as exc:
                logger.warning("Cached BR dividends invalid for %s: %s", ticker, exc)

        events = self._br.fetch_dividends(ticker)
        try:
            self._sqlite.cache_set(
                cache_key,
                json.dumps([e.model_dump(mode="json") for e in events]),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to cache BR dividends for %s: %s", ticker, exc)
        return events

    def _overlay_ohlcv(
        self,
        ticker: str,
        fundamentals: FundamentalsSnapshot | None,
    ) -> FundamentalsSnapshot | None:
        """Fill BR's missing price-derived fields from local OHLCV cache.

        BR scraper intentionally skips ``last_price``/``week52_high``/
        ``week52_low`` (it focuses on YoY narrative). Stooq OHLCV cache
        has them — derive and overlay so the factsheet KPI grid + market
        section don't show empty cells where we *do* have the data.

        If BR returned ``None`` (no fundamentals at all, e.g. ETF), build
        a minimal snapshot from OHLCV alone so we still get cena+52w.
        """
        try:
            df = self._data.load_ohlcv(ticker)
        except Exception:  # noqa: BLE001
            return fundamentals
        if df is None or df.empty or "close" not in df.columns:
            return fundamentals

        last_close: float | None = None
        try:
            last_close = float(df["close"].iloc[-1])
        except (TypeError, ValueError):
            pass

        recent = df.tail(WEEK52_TRADING_DAYS)
        w52_high: float | None = None
        w52_low: float | None = None
        try:
            if "high" in recent.columns:
                w52_high = float(recent["high"].max())
            if "low" in recent.columns:
                w52_low = float(recent["low"].min())
        except (TypeError, ValueError):
            pass

        if fundamentals is None:
            return FundamentalsSnapshot(
                ticker=normalize_ticker(ticker),
                last_price=last_close,
                week52_high=w52_high,
                week52_low=w52_low,
                source="ohlcv_cache",
                fetched_at=datetime.now(timezone.utc),
            )

        # Pydantic v2 frozen model — model_copy with updates
        updates: dict = {}
        if fundamentals.last_price is None and last_close is not None:
            updates["last_price"] = last_close
        if fundamentals.week52_high is None and w52_high is not None:
            updates["week52_high"] = w52_high
        if fundamentals.week52_low is None and w52_low is not None:
            updates["week52_low"] = w52_low
        if not updates:
            return fundamentals
        return fundamentals.model_copy(update=updates)

    def _load_news(self, ticker: str) -> list[NewsItem]:
        since = datetime.now(timezone.utc) - timedelta(days=NEWS_DAYS_BACK)
        items = self._data.load_news(ticker=ticker, since=since)
        # Sort desc by date so news:1 is freshest
        items.sort(key=lambda n: n.published_at, reverse=True)
        return items[:NEWS_TOP_N]

    # ------------------------------------------------------- Report assembly

    def _build_report(
        self,
        *,
        holding: Holding,
        holding_status: HoldingStatus | None,
        fundamentals: FundamentalsSnapshot | None,
        news: Sequence[NewsItem],
        narrative: CompanyNarrative | None,
    ) -> CompanyReport:
        # KPI grid (8 cells: 3 BR YoY + 5 price/portfolio/ratio)
        kpis = self._build_kpis(fundamentals, holding_status)
        market = self._build_market(fundamentals, holding, holding_status)
        calendar = self._build_calendar(fundamentals)
        period_label = (
            fundamentals.latest_quarter_label
            if fundamentals and fundamentals.latest_quarter_label
            else "Ostatni okres"
        )

        if narrative is None:
            tldr = (
                "Brak analizy AI dla tej spółki. Kliknij 'Generuj raport AI' "
                "aby wygenerować streszczenie z grounding na powyższych danych."
            )
            strengths = []
            risks = []
            confidence = 0
        else:
            tldr = narrative.tldr
            strengths = [
                BulletWithCitation(text=b.text, citations=list(b.citations))
                for b in narrative.strengths
            ]
            risks = [
                BulletWithCitation(text=b.text, citations=list(b.citations))
                for b in narrative.risks
            ]
            confidence = narrative.confidence

        return CompanyReport(
            company_name=holding.name or holding.ticker.upper(),
            ticker=holding.ticker,
            report_date=date.today().isoformat(),
            sector=fundamentals.sector if fundamentals else None,
            tldr=tldr,
            kpi_section_title=f"Kluczowe wskaźniki · {period_label}",
            kpis=kpis,
            market=market,
            strengths=strengths,
            risks=risks,
            calendar=calendar,
            confidence=max(1, confidence) if confidence else 1,
            generated_at_iso=datetime.now(timezone.utc).isoformat(),
        )

    def _build_kpis(
        self,
        f: FundamentalsSnapshot | None,
        hs: HoldingStatus | None,
    ) -> list[Kpi]:
        """Build the 4×2 KPI grid — must return min. 4 entries."""
        kpis: list[Kpi] = []

        # 1-3: BR YoY (skip with placeholder text "—" if missing)
        kpis.append(_yoy_kpi("Przychody YoY", f.revenue_yoy_pct if f else None))
        kpis.append(_yoy_kpi("EBITDA YoY", f.ebitda_yoy_pct if f else None))
        kpis.append(_yoy_kpi("Zysk netto YoY", f.net_profit_yoy_pct if f else None))

        # 4: Cena (z OHLCV/BR)
        last_price = (hs.last_price if hs and hs.has_price else None) or (
            f.last_price if f else None
        )
        kpis.append(Kpi(
            label="Cena",
            value=_fmt_price(last_price),
            delta=(
                hs.last_price_date.isoformat() if hs and hs.last_price_date else ""
            ),
            trend="neu",
        ))

        # 5: PnL pozycji
        if hs and hs.has_price:
            pnl_pct = hs.unrealized_pnl_pct * 100
            kpis.append(Kpi(
                label="PnL pozycji",
                value=f"{pnl_pct:+.2f}%",
                delta=f"{hs.unrealized_pnl:+,.0f} PLN",
                trend="pos" if pnl_pct > 0 else ("neg" if pnl_pct < 0 else "neu"),
            ))
        else:
            kpis.append(Kpi(label="PnL pozycji", value="—", trend="neu"))

        # 6: Dystans od 52w high
        high = f.week52_high if f else None
        if high and last_price:
            dist = (last_price / high - 1.0) * 100
            kpis.append(Kpi(
                label="Od 52w high",
                value=f"{dist:+.1f}%",
                delta=f"high {high:.2f}",
                trend="neg" if dist < -10 else "neu",
            ))
        else:
            kpis.append(Kpi(label="Od 52w high", value="—", trend="neu"))

        # 7: Dywidenda (yield)
        if f and f.dividend_yield is not None:
            dy = f.dividend_yield * 100
            kpis.append(Kpi(
                label="Stopa dywidendy",
                value=f"{dy:.2f}%",
                trend="pos" if dy > 3 else "neu",
            ))
        else:
            kpis.append(Kpi(label="Stopa dywidendy", value="—", trend="neu"))

        # 8: P/E
        if f and f.pe_ratio is not None:
            kpis.append(Kpi(
                label="P/E",
                value=f"{f.pe_ratio:.1f}",
                trend="pos" if 5 <= f.pe_ratio <= 18 else "neu",
            ))
        else:
            kpis.append(Kpi(label="P/E", value="—", trend="neu"))

        return kpis

    def _build_market(
        self,
        f: FundamentalsSnapshot | None,
        h: Holding,
        hs: HoldingStatus | None,
    ) -> list[LabelValue]:
        items: list[LabelValue] = []
        # Cena (with date)
        last_price = (hs.last_price if hs and hs.has_price else None) or (
            f.last_price if f else None
        )
        price_date = (
            hs.last_price_date.isoformat() if hs and hs.last_price_date else "—"
        )
        if last_price is not None:
            items.append(LabelValue(
                label=f"Kurs ({price_date})",
                value=f"{last_price:.2f} PLN",
            ))
        # Kapitalizacja
        if f and f.market_cap:
            items.append(LabelValue(
                label="Kapitalizacja",
                value=_fmt_mcap(f.market_cap),
            ))
        # 52w range
        if f and f.week52_high and f.week52_low:
            items.append(LabelValue(
                label="52W zakres",
                value=f"{f.week52_low:.2f} – {f.week52_high:.2f}",
            ))
        # P/BV
        if f and f.pbv_ratio:
            items.append(LabelValue(
                label="P/BV",
                value=f"{f.pbv_ratio:.2f}",
            ))
        # EPS
        if f and f.eps:
            items.append(LabelValue(
                label="EPS",
                value=f"{f.eps:.2f} PLN",
            ))
        # Sektor
        if f and f.sector:
            items.append(LabelValue(label="Sektor", value=f.sector))
        # Liczba akcji w portfelu
        n_tx = len(h.transactions)
        tx_suffix = f" · {n_tx} transakcji" if n_tx > 1 else ""
        items.append(LabelValue(
            label="W portfelu",
            value=(
                f"{h.shares:g} szt. · avg {h.avg_entry_price:.2f} "
                f"(od {h.first_entry_date.isoformat()}){tx_suffix}"
            ),
        ))
        # PnL absolute
        if hs and hs.has_price:
            items.append(LabelValue(
                label="Wartość pozycji / PnL",
                value=(
                    f"{hs.market_value:,.2f} PLN  "
                    f"({hs.unrealized_pnl:+,.0f} PLN, "
                    f"{hs.unrealized_pnl_pct * 100:+.2f}%)"
                ),
            ))
        # 5-letnia stopa zwrotu (z OHLCV jeśli mamy dość historii)
        five_y = self._compute_5y_return(h.ticker)
        if five_y is not None:
            items.append(LabelValue(
                label="5-letni zwrot",
                value=f"{five_y:+.1f}%",
            ))
        return items

    def _compute_5y_return(self, ticker: str) -> float | None:
        try:
            df = self._data.load_ohlcv(ticker)
        except Exception:  # noqa: BLE001
            return None
        if df is None or df.empty or "close" not in df.columns:
            return None
        if len(df) < FIVE_YEAR_TRADING_DAYS // 2:
            return None  # not enough history to be meaningful
        tail = df.tail(FIVE_YEAR_TRADING_DAYS)
        try:
            first = float(tail["close"].iloc[0])
            last = float(tail["close"].iloc[-1])
        except (TypeError, ValueError):
            return None
        if first <= 0:
            return None
        return (last / first - 1.0) * 100

    def _build_calendar(
        self, f: FundamentalsSnapshot | None
    ) -> list[CalendarItem]:
        items: list[CalendarItem] = []
        if f and f.next_report_estimated_date:
            days = (f.next_report_estimated_date - date.today()).days
            label = self._format_countdown(days)
            items.append(CalendarItem(
                highlight=f.next_report_estimated_date.isoformat(),
                text=(
                    "Szacowana publikacja następnego raportu kwartalnego/rocznego "
                    f"({label}; orientacyjna data ekstrapolowana z BR)."
                ),
            ))
        if f and f.last_report_date:
            items.append(CalendarItem(
                highlight=f.last_report_date.isoformat(),
                text=(
                    "Ostatni raportowany okres: "
                    f"{f.latest_quarter_label or '—'} (publikacja)."
                ),
            ))
        return items

    @staticmethod
    def _format_countdown(days: int) -> str:
        if days < 0:
            return f"{-days} dni temu"
        if days == 0:
            return "dzisiaj"
        if days == 1:
            return "jutro"
        return f"za {days} dni"

    # ------------------------------------------------------------------- LLM

    def _call_llm_narrative(
        self,
        *,
        holding: Holding,
        holding_status: HoldingStatus | None,
        fundamentals: FundamentalsSnapshot | None,
        news: Sequence[NewsItem],
    ) -> tuple[CompanyNarrative, list[str]]:
        """Run the narrative LLM call and validate citations.

        Returns ``(narrative, dropped_citations)``. Bullets whose citation
        key isn't in the registry are removed from the narrative.
        """
        registry, context_md = _render_narrative_context(
            holding=holding,
            holding_status=holding_status,
            fundamentals=fundamentals,
            news=news,
        )
        user_prompt = COMPANY_NARRATIVE_USER_TEMPLATE.format(
            ticker=holding.ticker, context=context_md,
        )
        result: CompanyNarrative = self._llm.complete_structured(
            system_prompt=COMPANY_NARRATIVE_SYSTEM,
            user_prompt=user_prompt,
            response_schema=CompanyNarrative,
            model=self._llm_cfg.model_analysis,
            temperature=self._llm_cfg.temperature,
            max_tokens=1500,
        )
        # Per-bullet: keep only citations present in the registry. A bullet
        # that loses ALL its citations is dropped (unsupported claim);
        # bullets that keep at least one survive with the valid subset.
        dropped: list[str] = []

        def _filter(bullets):
            kept = []
            for b in bullets:
                valid = [c for c in b.citations if c in registry]
                invalid = [c for c in b.citations if c not in registry]
                dropped.extend(invalid)
                if valid:
                    kept.append(b.model_copy(update={"citations": valid}))
            return kept

        kept_strengths = _filter(result.strengths)
        kept_risks = _filter(result.risks)
        # Backfill if validation killed everything (LLM still ran — keep its
        # output rather than render an empty section).
        cleaned = result.model_copy(update={
            "strengths": kept_strengths or result.strengths,
            "risks": kept_risks or result.risks,
        })
        return cleaned, dropped

    # ----------------------------------------------------------------- Cache

    def _save_to_cache(self, ticker: str, report: CompanyReport) -> None:
        cache_key = REPORT_CACHE_KEY_PREFIX + normalize_ticker(ticker)
        try:
            self._sqlite.cache_set(cache_key, report.model_dump_json())
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to cache report for %s: %s", ticker, exc)


# --- Helpers --------------------------------------------------------------


def _yoy_kpi(label: str, pct: float | None) -> Kpi:
    if pct is None:
        return Kpi(label=label, value="—", trend="neu")
    trend = "pos" if pct > 0 else ("neg" if pct < 0 else "neu")
    return Kpi(label=label, value=f"{pct:+.1f}%", delta="r/r", trend=trend)


def _fmt_price(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p:.2f} PLN"


def _fmt_mcap(mcap: float) -> str:
    """Format market cap as 'X,XX mld zł' or 'X,X mln zł'."""
    if mcap >= 1e9:
        return f"{mcap / 1e9:,.2f} mld PLN".replace(",", " ").replace(".", ",")
    if mcap >= 1e6:
        return f"{mcap / 1e6:,.1f} mln PLN".replace(",", " ").replace(".", ",")
    return f"{mcap:,.0f} PLN".replace(",", " ")


def _render_narrative_context(
    *,
    holding: Holding,
    holding_status: HoldingStatus | None,
    fundamentals: FundamentalsSnapshot | None,
    news: Sequence[NewsItem],
) -> tuple[set[str], str]:
    """Render the per-ticker context as Markdown + the set of valid citation keys.

    The LLM sees the Markdown; the service uses the citation set to strip
    unverified references after the call returns.
    """
    registry: set[str] = {"thesis"}
    lines: list[str] = [f"# {holding.ticker} ({holding.name or '—'})", ""]

    # Teza inwestycyjna
    lines.append("## Teza inwestycyjna (źródło: portfolio.yaml)")
    lines.append(holding.thesis.strip())
    lines.append("")

    # Dostępne źródła
    lines.append("## Dostępne źródła (cytuj WPROST przez `citation`)")
    src_lines: list[str] = []

    # Status pozycji
    if holding_status and holding_status.has_price:
        registry.add("portfolio:unrealized_pnl_pct")
        registry.add("portfolio:market_value")
        src_lines.append(
            f"- `portfolio:unrealized_pnl_pct` = "
            f"{holding_status.unrealized_pnl_pct * 100:+.2f}%  "
            f"(`portfolio:market_value` = {holding_status.market_value:,.2f} PLN)"
        )

    # BR fundamentals
    if fundamentals is not None:
        if fundamentals.last_price is not None:
            registry.add("fundamentals:last_price")
            band = ""
            if fundamentals.week52_low is not None and fundamentals.week52_high is not None:
                band = (
                    f"  (52t: {fundamentals.week52_low:.2f}–"
                    f"{fundamentals.week52_high:.2f})"
                )
            src_lines.append(
                f"- `fundamentals:last_price` = "
                f"{fundamentals.last_price:.2f} PLN{band}"
            )
        if fundamentals.revenue_yoy_pct is not None:
            registry.add("metric:revenue_yoy_pct")
            src_lines.append(
                f"- `metric:revenue_yoy_pct` = "
                f"{fundamentals.revenue_yoy_pct:+.2f}%  "
                f"(okres: {fundamentals.latest_quarter_label or '—'})"
            )
        if fundamentals.ebitda_yoy_pct is not None:
            registry.add("metric:ebitda_yoy_pct")
            src_lines.append(
                f"- `metric:ebitda_yoy_pct` = {fundamentals.ebitda_yoy_pct:+.2f}%"
            )
        if fundamentals.net_profit_yoy_pct is not None:
            registry.add("metric:net_profit_yoy_pct")
            src_lines.append(
                f"- `metric:net_profit_yoy_pct` = "
                f"{fundamentals.net_profit_yoy_pct:+.2f}%"
            )
        if fundamentals.pe_ratio is not None:
            registry.add("fundamentals:pe_ratio")
            src_lines.append(
                f"- `fundamentals:pe_ratio` = {fundamentals.pe_ratio:.2f}"
            )
        if fundamentals.pbv_ratio is not None:
            registry.add("fundamentals:pbv_ratio")
            src_lines.append(
                f"- `fundamentals:pbv_ratio` = {fundamentals.pbv_ratio:.2f}"
            )
        if fundamentals.dividend_yield is not None:
            registry.add("fundamentals:dividend_yield")
            src_lines.append(
                f"- `fundamentals:dividend_yield` = "
                f"{fundamentals.dividend_yield * 100:.2f}%"
            )
        if fundamentals.week52_high is not None and fundamentals.last_price:
            registry.add("metric:distance_from_52w_high_pct")
            dist = (fundamentals.last_price / fundamentals.week52_high - 1) * 100
            src_lines.append(
                f"- `metric:distance_from_52w_high_pct` = {dist:+.2f}%"
            )
        if fundamentals.sector:
            registry.add("fundamentals:sector")
            src_lines.append(f"- `fundamentals:sector` = {fundamentals.sector}")
        if fundamentals.next_report_estimated_date:
            registry.add("fundamentals:next_report_estimated_date")
            days = (
                fundamentals.next_report_estimated_date - date.today()
            ).days
            src_lines.append(
                f"- `fundamentals:next_report_estimated_date` = "
                f"{fundamentals.next_report_estimated_date.isoformat()} "
                f"(za {days} dni)"
            )

    # News
    for idx, n in enumerate(news, start=1):
        key = f"news:{idx}"
        registry.add(key)
        when = n.published_at.strftime("%Y-%m-%d")
        src_lines.append(f"- `{key}` ({when}, {n.source}): {n.title}")

    if not src_lines:
        lines.append("_(brak metryk i newsów — bazuj wyłącznie na tezie)_")
    else:
        lines.extend(src_lines)

    lines.append("")
    lines.append("## Dozwolone wartości `citation`")
    lines.append("Tylko klucze wymienione powyżej. Inne zostaną odrzucone.")

    return registry, "\n".join(lines)
