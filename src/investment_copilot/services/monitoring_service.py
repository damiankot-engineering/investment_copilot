"""Monitoring service.

Orchestrates the per-portfolio monitoring pipeline:

* fetches headline fundamentals (best-effort) for each holding,
* pulls recent news (incl. ESPI flagging),
* loads the previous monitoring snapshot from disk so the LLM can
  describe what changed,
* asks the copilot LLM for a structured :class:`MonitoringReport`,
* persists the new snapshot as JSON for the next run's diff.

The HTML rendering itself lives in :mod:`report_service`. This service is
only responsible for the data + LLM + snapshot persistence.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from investment_copilot.domain.fundamentals import (
    FundamentalsSnapshot,
    MonitoringSnapshot,
    TickerNewsRef,
)
from investment_copilot.domain.models import NewsItem, normalize_ticker
from investment_copilot.domain.portfolio import Portfolio, PortfolioStatus
from investment_copilot.domain.prompts import MonitoringReport
from investment_copilot.infrastructure.providers.base import ProviderError
from investment_copilot.infrastructure.providers.biznesradar import (
    BiznesRadarProvider,
)
from investment_copilot.infrastructure.providers.stooq_fundamentals import (
    StooqFundamentalsProvider,
)
from investment_copilot.infrastructure.storage import SQLiteStore
from investment_copilot.services.copilot_service import CopilotService
from investment_copilot.services.data_service import DataService
from investment_copilot.services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


DEFAULT_NEWS_DAYS_BACK: int = 30
SNAPSHOTS_DIRNAME: str = "snapshots"
MAX_NEWS_REFS_PER_TICKER: int = 8
LATEST_FALLBACK_LIMIT: int = 5  # if window is empty, take this many newest
WEEK52_TRADING_DAYS: int = 252
FUNDAMENTALS_CACHE_TTL: timedelta = timedelta(hours=24)
ESPI_CACHE_TTL: timedelta = timedelta(hours=24)


class MonitoringService:
    """Builds + persists monitoring reports for the portfolio."""

    def __init__(
        self,
        *,
        copilot_service: CopilotService,
        data_service: DataService,
        portfolio_service: PortfolioService,
        sqlite_store: SQLiteStore,
        biznesradar_provider: BiznesRadarProvider | None = None,
        fundamentals_provider: StooqFundamentalsProvider | None = None,
        snapshots_dir: Path | str = "reports/monitoring/snapshots",
    ) -> None:
        self._copilot = copilot_service
        self._data = data_service
        self._portfolio_svc = portfolio_service
        self._sqlite = sqlite_store
        self._biznesradar = biznesradar_provider or BiznesRadarProvider()
        self._fundamentals = fundamentals_provider or StooqFundamentalsProvider()
        self.snapshots_dir = Path(snapshots_dir)

    # -- Public --------------------------------------------------------------

    def generate(
        self,
        portfolio: Portfolio,
        status: PortfolioStatus,
        *,
        news_days_back: int = DEFAULT_NEWS_DAYS_BACK,
        refresh_news: bool = True,
    ) -> tuple[MonitoringReport, MonitoringSnapshot, list[str]]:
        """Run the full pipeline.

        Returns ``(report, snapshot, warnings)``. ``snapshot`` represents
        the data that fed the LLM; persist it via :meth:`save_snapshot`.
        ``warnings`` captures non-fatal hiccups (e.g. fundamentals fetch
        failures) so the caller can surface them in the UI.

        When ``refresh_news`` is ``True`` (default) the service triggers
        a per-ticker news refresh from configured providers BEFORE reading
        from cache. This ensures the LLM always sees the latest available
        ESPI/news items even if the user hasn't run "Update data" recently.
        """
        warnings: list[str] = []

        if refresh_news:
            self._refresh_news_per_ticker(portfolio, days_back=news_days_back, warnings=warnings)

        fundamentals = self._collect_fundamentals(portfolio, warnings)
        news = self._collect_news(portfolio, days_back=news_days_back, warnings=warnings)
        previous = self.load_latest_snapshot()

        report = self._copilot.generate_monitoring(
            portfolio,
            status,
            fundamentals=fundamentals,
            news=news,
            previous_snapshot=previous,
        )

        snapshot = MonitoringSnapshot(
            generated_at=datetime.now(timezone.utc),
            fundamentals=fundamentals,
            news_by_ticker=self._news_to_refs(news),
            report=report.model_dump(mode="json"),
        )
        return report, snapshot, warnings

    def save_snapshot(self, snapshot: MonitoringSnapshot) -> Path:
        """Persist a snapshot as JSON. Returns the path written."""
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        ts = snapshot.generated_at.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.snapshots_dir / f"snapshot_{ts}.json"
        path.write_text(
            snapshot.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Monitoring snapshot saved: %s", path)
        return path

    def load_latest_snapshot(self) -> MonitoringSnapshot | None:
        """Load the most recent persisted snapshot, or ``None`` if none."""
        if not self.snapshots_dir.is_dir():
            return None
        files = sorted(
            (p for p in self.snapshots_dir.iterdir() if p.suffix == ".json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return None
        try:
            data = json.loads(files[0].read_text(encoding="utf-8"))
            return MonitoringSnapshot.model_validate(data)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load previous snapshot %s: %s", files[0], exc)
            return None

    # -- Internal ------------------------------------------------------------

    def _collect_fundamentals(
        self,
        portfolio: Portfolio,
        warnings: list[str],
    ) -> list[FundamentalsSnapshot]:
        """Build per-ticker fundamentals snapshots.

        Strategy (per ticker):

        1. **BiznesRadar (primary)** — rich data with sector, YoY %s,
           last report date, latest narrative bullets. Cached 24h in SQLite.
        2. **Stooq fundamentals (fallback)** — best-effort HTML scrape.
           Usually all-None since Stooq moved to JS rendering.
        3. **OHLCV cache (last resort)** — provides last_price + 52w range
           derived from local parquet.

        The :attr:`FundamentalsSnapshot.source` field records which tier
        won. OHLCV-derived price/52w is ALWAYS merged in last (most reliable
        local source for those specific fields).
        """
        out: list[FundamentalsSnapshot] = []
        for h in portfolio.holdings:
            primary = self._fetch_biznesradar_cached(h.ticker, warnings)

            if primary is None:
                # Fall back to Stooq HTML scraper
                try:
                    primary = self._fundamentals.fetch_snapshot(h.ticker)
                except ProviderError as exc:
                    logger.info(
                        "Stooq fundamentals fallback failed for %s: %s",
                        h.ticker, exc,
                    )

            ohlcv_fields = self._fundamentals_from_ohlcv(h.ticker, holding_name=h.name)

            if primary is None:
                # Last resort — OHLCV-only snapshot, or fully empty
                if ohlcv_fields is None:
                    warnings.append(
                        f"Brak danych fundamentals i OHLCV dla {h.ticker} — "
                        f"uruchom 'Update data' aby zasilić cache."
                    )
                    out.append(
                        FundamentalsSnapshot(
                            ticker=normalize_ticker(h.ticker),
                            name=h.name,
                            source="empty",
                            fetched_at=datetime.now(timezone.utc),
                        )
                    )
                    continue
                out.append(ohlcv_fields)
                continue

            # Merge: primary keeps its rich fields; OHLCV fills price/52w when
            # primary doesn't have them (BR intentionally skips last_price).
            merged = primary.model_copy(
                update={
                    "name": primary.name or h.name
                        or (ohlcv_fields.name if ohlcv_fields else None),
                    "last_price": primary.last_price
                    if primary.last_price is not None
                    else (ohlcv_fields.last_price if ohlcv_fields else None),
                    "week52_high": primary.week52_high
                    if primary.week52_high is not None
                    else (ohlcv_fields.week52_high if ohlcv_fields else None),
                    "week52_low": primary.week52_low
                    if primary.week52_low is not None
                    else (ohlcv_fields.week52_low if ohlcv_fields else None),
                }
            )
            out.append(merged)
        return out

    def _fetch_biznesradar_cached(
        self, ticker: str, warnings: list[str]
    ) -> FundamentalsSnapshot | None:
        """Fetch BR fundamentals, with 24h SQLite cache. Returns None on failure."""
        cache_key = f"biznesradar:fundamentals:{normalize_ticker(ticker)}"
        cached_json = self._sqlite.cache_get(cache_key, max_age=FUNDAMENTALS_CACHE_TTL)
        if cached_json:
            try:
                return FundamentalsSnapshot.model_validate_json(cached_json)
            except ValueError as exc:
                logger.warning("Cached BR fundamentals invalid for %s: %s", ticker, exc)

        try:
            snap = self._biznesradar.fetch_fundamentals(ticker)
        except ProviderError as exc:
            logger.info("BR fundamentals failed for %s: %s", ticker, exc)
            warnings.append(
                f"BiznesRadar niedostępny dla {ticker} ({exc}) — używam fallbacku."
            )
            return None

        # Persist for the next 24h
        try:
            self._sqlite.cache_set(cache_key, snap.model_dump_json())
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to cache BR for %s: %s", ticker, exc)

        return snap

    def _fundamentals_from_ohlcv(
        self, ticker: str, *, holding_name: str | None
    ) -> FundamentalsSnapshot | None:
        """Derive a fundamentals snapshot from the local parquet OHLCV cache.

        Returns ``None`` if the cache has no data for the ticker.
        """
        symbol = normalize_ticker(ticker)
        try:
            df = self._data.load_ohlcv(symbol)
        except FileNotFoundError:
            return None
        except Exception as exc:  # pragma: no cover
            logger.warning("OHLCV load failed for %s: %s", symbol, exc)
            return None
        if df is None or df.empty:
            return None

        last_close = float(df["close"].iloc[-1]) if "close" in df.columns else None
        # 52-week range from the last ~252 trading days.
        recent = df.tail(WEEK52_TRADING_DAYS)
        try:
            high = float(recent["high"].max()) if "high" in recent.columns else None
            low = float(recent["low"].min()) if "low" in recent.columns else None
        except (TypeError, ValueError):
            high = None
            low = None

        return FundamentalsSnapshot(
            ticker=symbol,
            name=holding_name,
            last_price=last_close,
            week52_high=high if pd.notna(high) else None,
            week52_low=low if pd.notna(low) else None,
            source="ohlcv_cache",
            fetched_at=datetime.now(timezone.utc),
        )

    def _refresh_news_per_ticker(
        self,
        portfolio: Portfolio,
        *,
        days_back: int,
        warnings: list[str],
    ) -> None:
        """Force a news refresh from configured providers before reading cache.

        Also pulls BiznesRadar ESPI announcements per ticker (cached 24h)
        and inserts them into the news cache alongside Stooq + RSS items.
        """
        since = datetime.now(timezone.utc) - timedelta(days=max(0, days_back))
        try:
            keywords = self._portfolio_svc.keywords_map(portfolio)
            inserted = self._data.refresh_news(since, keywords_by_ticker=keywords)
            logger.info("Monitoring: news refresh inserted %d items", inserted)
        except Exception as exc:
            logger.warning("Monitoring news refresh failed: %s", exc)
            warnings.append(f"News refresh failed: {exc}")

        # Pull BR ESPI per-ticker (independent of Stooq/RSS path)
        br_inserted = 0
        for h in portfolio.holdings:
            try:
                items = self._fetch_br_espi_cached(h.ticker, since=since)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("BR ESPI failed for %s: %s", h.ticker, exc)
                continue
            if items:
                br_inserted += self._sqlite.upsert_news(items)
        if br_inserted:
            logger.info("Monitoring: BR ESPI inserted %d items", br_inserted)

    def _fetch_br_espi_cached(
        self, ticker: str, *, since: datetime
    ) -> list[NewsItem]:
        """Fetch BR ESPI for a ticker with 24h cache."""
        cache_key = f"biznesradar:espi:{normalize_ticker(ticker)}"
        cached_json = self._sqlite.cache_get(cache_key, max_age=ESPI_CACHE_TTL)
        if cached_json is not None:
            try:
                payload = json.loads(cached_json)
                items = [NewsItem.model_validate(p) for p in payload]
                # Filter by since here (cache stores full set; consumer may
                # have a different window each call).
                since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                return [i for i in items if i.published_at >= since_aware]
            except (ValueError, KeyError) as exc:
                logger.warning("Cached BR ESPI invalid for %s: %s", ticker, exc)

        try:
            items = self._biznesradar.fetch_espi(ticker, since=since)
        except ProviderError as exc:
            logger.info("BR ESPI fetch failed for %s: %s", ticker, exc)
            return []

        try:
            self._sqlite.cache_set(
                cache_key,
                json.dumps([i.model_dump(mode="json") for i in items]),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to cache BR ESPI for %s: %s", ticker, exc)
        return items

    def _collect_news(
        self,
        portfolio: Portfolio,
        *,
        days_back: int,
        warnings: list[str],
    ) -> list[NewsItem]:
        """Load news from cache; for tickers with nothing in window, take the
        latest available items regardless of date so the LLM never has to
        report on a position with zero context."""
        since = datetime.now(timezone.utc) - timedelta(days=max(0, days_back))
        items: list[NewsItem] = []
        empty_tickers: list[str] = []

        for h in portfolio.holdings:
            ticker_items = self._data.load_news(ticker=h.ticker, since=since)
            if not ticker_items:
                # Fall back to the most recent items regardless of window.
                fallback = self._data.load_news(
                    ticker=h.ticker, since=None, limit=LATEST_FALLBACK_LIMIT
                )
                if fallback:
                    items.extend(fallback)
                else:
                    empty_tickers.append(h.ticker)
            else:
                items.extend(ticker_items)

        general = self._data.load_news(ticker=None, since=since, limit=20)
        items.extend(g for g in general if g.ticker is None)

        if empty_tickers:
            warnings.append(
                "Brak jakichkolwiek wiadomości dla: "
                + ", ".join(empty_tickers)
                + " — LLM oprze analizę tylko na tezie i ogólnych newsach."
            )
        return items

    @staticmethod
    def _news_to_refs(
        news: Sequence[NewsItem],
    ) -> dict[str, list[TickerNewsRef]]:
        bucket: dict[str, list[TickerNewsRef]] = {}
        sorted_news = sorted(news, key=lambda n: n.published_at, reverse=True)
        for n in sorted_news:
            key = n.ticker or "_general"
            current = bucket.setdefault(key, [])
            if len(current) >= MAX_NEWS_REFS_PER_TICKER:
                continue
            current.append(
                TickerNewsRef(
                    title=n.title,
                    published_at=n.published_at,
                    source=n.source,
                    url=n.url,
                )
            )
        return bucket
