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
from investment_copilot.infrastructure.providers.stooq_fundamentals import (
    StooqFundamentalsProvider,
)
from investment_copilot.services.copilot_service import CopilotService
from investment_copilot.services.data_service import DataService
from investment_copilot.services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


DEFAULT_NEWS_DAYS_BACK: int = 30
SNAPSHOTS_DIRNAME: str = "snapshots"
MAX_NEWS_REFS_PER_TICKER: int = 8
LATEST_FALLBACK_LIMIT: int = 5  # if window is empty, take this many newest
WEEK52_TRADING_DAYS: int = 252


class MonitoringService:
    """Builds + persists monitoring reports for the portfolio."""

    def __init__(
        self,
        *,
        copilot_service: CopilotService,
        data_service: DataService,
        portfolio_service: PortfolioService,
        fundamentals_provider: StooqFundamentalsProvider | None = None,
        snapshots_dir: Path | str = "reports/monitoring/snapshots",
    ) -> None:
        self._copilot = copilot_service
        self._data = data_service
        self._portfolio_svc = portfolio_service
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

        Strategy: try the HTML scraper (best-effort), then ALWAYS augment
        with values derived from the local OHLCV cache (last_price,
        52-week range). Stooq's snapshot panel is now JS-rendered so the
        scraper typically returns all-None — the OHLCV-derived fields are
        the reliable backbone that lets the LLM ground its analysis.
        """
        out: list[FundamentalsSnapshot] = []
        for h in portfolio.holdings:
            scraped: FundamentalsSnapshot | None = None
            try:
                scraped = self._fundamentals.fetch_snapshot(h.ticker)
            except ProviderError as exc:
                logger.info("Fundamentals scrape failed for %s: %s", h.ticker, exc)

            ohlcv_fields = self._fundamentals_from_ohlcv(h.ticker, holding_name=h.name)

            if scraped is None:
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

            # Merge: scraped fields take priority where present, OHLCV fills gaps.
            merged = scraped.model_copy(
                update={
                    "name": scraped.name or h.name or (ohlcv_fields.name if ohlcv_fields else None),
                    "last_price": scraped.last_price
                    if scraped.last_price is not None
                    else (ohlcv_fields.last_price if ohlcv_fields else None),
                    "week52_high": scraped.week52_high
                    if scraped.week52_high is not None
                    else (ohlcv_fields.week52_high if ohlcv_fields else None),
                    "week52_low": scraped.week52_low
                    if scraped.week52_low is not None
                    else (ohlcv_fields.week52_low if ohlcv_fields else None),
                }
            )
            out.append(merged)
        return out

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
        """Force a news refresh from configured providers before reading cache."""
        since = datetime.now(timezone.utc) - timedelta(days=max(0, days_back))
        try:
            keywords = self._portfolio_svc.keywords_map(portfolio)
            inserted = self._data.refresh_news(since, keywords_by_ticker=keywords)
            logger.info("Monitoring: news refresh inserted %d items", inserted)
        except Exception as exc:
            logger.warning("Monitoring news refresh failed: %s", exc)
            warnings.append(f"News refresh failed: {exc}")

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
