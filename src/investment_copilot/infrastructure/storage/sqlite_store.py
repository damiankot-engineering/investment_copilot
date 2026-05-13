"""SQLite store for news items and OHLCV fetch metadata.

We deliberately use the stdlib ``sqlite3`` module (no SQLAlchemy) — the
schema is small, the queries are trivial, and avoiding the dependency keeps
the install footprint slim.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

from investment_copilot.domain.models import NewsItem

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT,
    source        TEXT NOT NULL,
    title         TEXT NOT NULL,
    url           TEXT NOT NULL UNIQUE,
    published_at  TEXT NOT NULL,           -- ISO8601 UTC
    summary       TEXT,
    fetched_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_news_ticker      ON news(ticker);
CREATE INDEX IF NOT EXISTS idx_news_published   ON news(published_at);

CREATE TABLE IF NOT EXISTS ohlcv_meta (
    ticker          TEXT PRIMARY KEY,
    last_fetched_at TEXT NOT NULL,
    earliest_date   TEXT,
    latest_date     TEXT
);

CREATE TABLE IF NOT EXISTS kv_cache (
    cache_key     TEXT PRIMARY KEY,
    payload_json  TEXT NOT NULL,
    fetched_at    TEXT NOT NULL              -- ISO8601 UTC
);
"""


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(s: str) -> datetime:
    # Python 3.11 fromisoformat handles the ISO output we produce.
    return datetime.fromisoformat(s)


class SQLiteStore:
    """Thin wrapper around a SQLite database file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # -- Connections ---------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- News ---------------------------------------------------------------

    def upsert_news(self, items: Iterable[NewsItem]) -> int:
        """Insert items, ignoring rows whose URL already exists. Returns count inserted."""
        rows = [
            (
                item.ticker,
                item.source,
                item.title,
                item.url,
                _to_iso(item.published_at),
                item.summary,
            )
            for item in items
        ]
        if not rows:
            return 0
        with self._connect() as conn:
            cur = conn.executemany(
                """
                INSERT OR IGNORE INTO news
                    (ticker, source, title, url, published_at, summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return cur.rowcount or 0

    def load_news(
        self,
        *,
        ticker: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[NewsItem]:
        clauses: list[str] = []
        params: list[object] = []
        if ticker is not None:
            clauses.append("ticker = ?")
            params.append(ticker)
        if since is not None:
            clauses.append("published_at >= ?")
            params.append(_to_iso(since))

        sql = "SELECT ticker, source, title, url, published_at, summary FROM news"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY published_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            NewsItem(
                ticker=r["ticker"],
                source=r["source"],
                title=r["title"],
                url=r["url"],
                published_at=_from_iso(r["published_at"]),
                summary=r["summary"],
            )
            for r in rows
        ]

    # -- OHLCV metadata -----------------------------------------------------

    def set_ohlcv_meta(
        self,
        ticker: str,
        *,
        last_fetched_at: datetime,
        earliest_date: str | None,
        latest_date: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ohlcv_meta (ticker, last_fetched_at, earliest_date, latest_date)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    last_fetched_at = excluded.last_fetched_at,
                    earliest_date   = COALESCE(excluded.earliest_date, ohlcv_meta.earliest_date),
                    latest_date     = COALESCE(excluded.latest_date,   ohlcv_meta.latest_date)
                """,
                (ticker, _to_iso(last_fetched_at), earliest_date, latest_date),
            )

    def get_ohlcv_meta(self, ticker: str) -> dict[str, object] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ticker, last_fetched_at, earliest_date, latest_date "
                "FROM ohlcv_meta WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        return dict(row) if row else None

    # -- Generic KV cache (used for slow external scrapes) ------------------

    def cache_set(self, key: str, payload_json: str) -> None:
        """Store ``payload_json`` under ``key``, overwriting any prior entry."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kv_cache (cache_key, payload_json, fetched_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    fetched_at = excluded.fetched_at
                """,
                (key, payload_json, _to_iso(datetime.now(timezone.utc))),
            )

    def cache_get(self, key: str, *, max_age: timedelta) -> str | None:
        """Return the cached payload for ``key`` if fresher than ``max_age``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, fetched_at FROM kv_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        try:
            fetched_at = _from_iso(row["fetched_at"])
        except (TypeError, ValueError):
            return None
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - fetched_at
        if age > max_age:
            return None
        return row["payload_json"]
