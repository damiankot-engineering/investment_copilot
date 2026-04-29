"""RSS news adapter built on :mod:`feedparser`.

Configured with a list of feed URLs. Per call, fetches each feed, applies
``since`` and (optional) keyword filtering, and returns a flat
:class:`~investment_copilot.domain.models.NewsItem` list.

Keyword matching is case-insensitive substring matching on title + summary.
That's deliberately simple: GPW news is short-form and headlines reliably
contain the company name. Fancier matching (NER, fuzzy, ticker symbols) is
a future enhancement, not a v1 problem.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Iterable

import feedparser

from investment_copilot.domain.models import NewsItem

logger = logging.getLogger(__name__)


class RSSProvider:
    """News via configured RSS feeds (Bankier, Money.pl, Parkiet, ...)."""

    name: str = "rss"

    def __init__(
        self,
        feeds: Iterable[str],
        *,
        max_items_per_feed: int = 200,
    ) -> None:
        self.feeds: list[str] = list(feeds)
        self.max_items_per_feed = max_items_per_feed

    def fetch_news(
        self,
        since: datetime,
        *,
        ticker: str | None = None,
        keywords: list[str] | None = None,
    ) -> list[NewsItem]:
        if not self.feeds:
            return []

        since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        kw_lower = [k.lower() for k in (keywords or []) if k]

        results: list[NewsItem] = []
        for url in self.feeds:
            try:
                parsed = feedparser.parse(url)
            except Exception as exc:  # feedparser rarely raises, but be safe
                logger.warning("RSS parse failed for %s: %s", url, exc)
                continue

            feed_title = (parsed.feed.get("title") if parsed.feed else None) or url
            entries = list(parsed.entries[: self.max_items_per_feed])

            for entry in entries:
                published = _entry_published(entry)
                if published is None or published < since_aware:
                    continue

                title = (entry.get("title") or "").strip()
                summary = (entry.get("summary") or "").strip() or None
                link = (entry.get("link") or "").strip()
                if not title or not link:
                    continue

                if kw_lower and not _matches_keywords(title, summary, kw_lower):
                    continue

                results.append(
                    NewsItem(
                        ticker=ticker,
                        source=f"rss:{feed_title}",
                        title=title,
                        url=link,
                        published_at=published,
                        summary=summary,
                    )
                )

        # Newest first, deduped by URL.
        results.sort(key=lambda n: n.published_at, reverse=True)
        seen: set[str] = set()
        deduped: list[NewsItem] = []
        for item in results:
            if item.url in seen:
                continue
            seen.add(item.url)
            deduped.append(item)
        return deduped


# --- helpers ----------------------------------------------------------------


def _entry_published(entry: object) -> datetime | None:
    """Best-effort extraction of a UTC-aware datetime from a feedparser entry."""
    parsed = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if parsed is None and isinstance(entry, dict):
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    try:
        return datetime.fromtimestamp(time.mktime(parsed), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _matches_keywords(
    title: str,
    summary: str | None,
    keywords_lower: list[str],
) -> bool:
    haystack = (title + " " + (summary or "")).lower()
    return any(kw in haystack for kw in keywords_lower)
