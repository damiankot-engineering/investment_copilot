"""Tests for the RSS news provider."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from investment_copilot.infrastructure.providers.rss import RSSProvider


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>PKN Orlen ogłasza wyniki za Q4</title>
      <link>https://example.com/news/1</link>
      <description>Zysk netto wzrósł o 12%.</description>
      <pubDate>Tue, 02 Jan 2024 09:00:00 +0000</pubDate>
    </item>
    <item>
      <title>CD Projekt zapowiada nową grę</title>
      <link>https://example.com/news/2</link>
      <description>Premiera planowana na 2026.</description>
      <pubDate>Wed, 03 Jan 2024 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Stara wiadomość bez znaczenia</title>
      <link>https://example.com/news/3</link>
      <description>Coś zupełnie innego.</description>
      <pubDate>Mon, 01 Jan 2020 09:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""


def test_rss_returns_recent_items_and_filters_keywords() -> None:
    import feedparser

    parsed = feedparser.parse(SAMPLE_RSS)
    with patch("investment_copilot.infrastructure.providers.rss.feedparser.parse",
               return_value=parsed):
        provider = RSSProvider(["http://example.com/feed.xml"])

        items = provider.fetch_news(
            since=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ticker="pkn.pl",
            keywords=["Orlen"],
        )

    assert len(items) == 1
    assert items[0].title.startswith("PKN Orlen")
    assert items[0].ticker == "pkn.pl"
    assert items[0].source.startswith("rss:")


def test_rss_filters_by_since() -> None:
    import feedparser

    parsed = feedparser.parse(SAMPLE_RSS)
    with patch("investment_copilot.infrastructure.providers.rss.feedparser.parse",
               return_value=parsed):
        provider = RSSProvider(["http://example.com/feed.xml"])

        items = provider.fetch_news(since=datetime(2024, 1, 3, tzinfo=timezone.utc))

    # Only the Jan 3 item passes the date cut
    assert len(items) == 1
    assert "CD Projekt" in items[0].title


def test_rss_no_keywords_returns_all_after_since() -> None:
    import feedparser

    parsed = feedparser.parse(SAMPLE_RSS)
    with patch("investment_copilot.infrastructure.providers.rss.feedparser.parse",
               return_value=parsed):
        provider = RSSProvider(["http://example.com/feed.xml"])

        items = provider.fetch_news(since=datetime(2023, 1, 1, tzinfo=timezone.utc))

    assert len(items) == 2  # the two recent ones


def test_rss_dedupes_by_url() -> None:
    import feedparser

    parsed = feedparser.parse(SAMPLE_RSS)
    with patch("investment_copilot.infrastructure.providers.rss.feedparser.parse",
               return_value=parsed):
        provider = RSSProvider(
            ["http://example.com/feed-a.xml", "http://example.com/feed-b.xml"]
        )

        items = provider.fetch_news(since=datetime(2023, 1, 1, tzinfo=timezone.utc))

    urls = [i.url for i in items]
    assert len(urls) == len(set(urls))


def test_rss_no_feeds_returns_empty() -> None:
    provider = RSSProvider([])
    items = provider.fetch_news(since=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert items == []


def test_rss_handles_parser_exception() -> None:
    with patch("investment_copilot.infrastructure.providers.rss.feedparser.parse",
               side_effect=RuntimeError("boom")):
        provider = RSSProvider(["http://example.com/feed.xml"])
        items = provider.fetch_news(since=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert items == []
