"""Stooq news adapter (best-effort).

Stooq does not expose a clean per-ticker news API. This provider attempts to
parse the headlines section of Stooq's per-symbol HTML page using a small
set of regex patterns. If Stooq's markup changes, the provider returns an
empty list and logs a warning rather than crashing the pipeline.

The architecture is the value here: swap in a richer scraper or replace this
class entirely without touching the rest of the system.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import requests

from investment_copilot.domain.models import NewsItem, normalize_ticker
from investment_copilot.infrastructure.providers.base import ProviderError

logger = logging.getLogger(__name__)


class StooqNewsProvider:
    """Best-effort scraper of Stooq per-symbol news headlines."""

    name: str = "stooq"

    BASE_URL: str = "https://stooq.com/q/m/"  # message/news block per symbol
    DEFAULT_TIMEOUT: float = 30.0

    # Conservative pattern: rows like
    #   <tr ...><td ...>YYYY-MM-DD HH:MM</td><td ...><a href="URL">TITLE</a>...
    _ROW_RE = re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}).*?"
        r'href="(?P<url>https?://[^"]+)"[^>]*>(?P<title>[^<]{3,300})</a>',
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.setdefault(
            "User-Agent", "investment-copilot/0.1 (+https://stooq.com)"
        )

    def fetch_news(
        self,
        since: datetime,
        *,
        ticker: str | None = None,
        keywords: list[str] | None = None,  # noqa: ARG002 - not used by Stooq
    ) -> list[NewsItem]:
        if ticker is None:
            # Stooq's news block is per-symbol; nothing useful to do without one.
            return []

        symbol = normalize_ticker(ticker)
        try:
            resp = self._session.get(
                self.BASE_URL, params={"s": symbol}, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Stooq news fetch failed for %s: %s", symbol, exc)
            return []

        items: list[NewsItem] = []
        for m in self._ROW_RE.finditer(resp.text):
            try:
                published = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
            since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
            if published < since_aware:
                continue
            title = re.sub(r"\s+", " ", m.group("title")).strip()
            items.append(
                NewsItem(
                    ticker=symbol,
                    source=f"stooq:{symbol}",
                    title=title,
                    url=m.group("url"),
                    published_at=published,
                )
            )

        if not items:
            logger.debug("Stooq news: no items parsed for %s", symbol)
        return items

    # Mirror StooqProvider's surface for symmetry; not part of the Protocol.
    def __repr__(self) -> str:  # pragma: no cover
        return f"StooqNewsProvider(timeout={self.timeout})"


__all__ = ["StooqNewsProvider", "ProviderError"]
