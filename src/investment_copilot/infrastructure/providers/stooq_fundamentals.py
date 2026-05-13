"""Stooq fundamentals adapter (best-effort).

Scrapes the public per-symbol page (``stooq.pl/q/?s=<symbol>``) for the
small snapshot panel: last price, market cap, P/E, P/BV, EPS, dividend
yield, 52-week range. There is no public free fundamentals API for GPW;
this scraper is the most stable option among the free sources.

Behaviour mirrors :class:`StooqNewsProvider`: best-effort. If Stooq's
markup changes or the page can't be fetched, return a snapshot with
``None`` fields rather than crashing the pipeline. The downstream LLM
is given whatever data is available.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import requests

from investment_copilot.domain.fundamentals import FundamentalsSnapshot
from investment_copilot.domain.models import normalize_ticker
from investment_copilot.infrastructure.providers.base import ProviderError

logger = logging.getLogger(__name__)


_NUM = r"-?\d[\d\s]*[.,]?\d*"


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    cleaned = raw.replace("\xa0", "").replace(" ", "").replace(",", ".")
    if not cleaned or cleaned in {"-", "—"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _market_cap_to_pln(raw: str | None) -> float | None:
    """Parse strings like '6.4 mld', '841 mln', '12.3 tys' to PLN units."""
    if raw is None:
        return None
    txt = raw.lower().replace("\xa0", " ").strip()
    m = re.match(r"(?P<num>" + _NUM + r")\s*(?P<unit>mld|mln|tys|bn|m|k)?", txt)
    if not m:
        return _to_float(txt)
    value = _to_float(m.group("num"))
    if value is None:
        return None
    unit = (m.group("unit") or "").strip()
    multiplier = {
        "mld": 1_000_000_000,
        "bn": 1_000_000_000,
        "mln": 1_000_000,
        "m": 1_000_000,
        "tys": 1_000,
        "k": 1_000,
    }.get(unit, 1)
    return value * multiplier


def _pct_to_fraction(raw: str | None) -> float | None:
    v = _to_float(raw)
    if v is None:
        return None
    # Stooq prints stopa dyw. as percentage (e.g. "2.50"); divide by 100.
    return v / 100.0


class StooqFundamentalsProvider:
    """Best-effort scraper of Stooq's per-symbol snapshot panel."""

    name: str = "stooq_fundamentals"

    BASE_URL: str = "https://stooq.pl/q/"
    DEFAULT_TIMEOUT: float = 30.0

    # Stooq's snapshot panel renders rows like:
    #   <td>Kapitalizacja</td><td>6.4 mld</td>
    #   <td>P/E</td><td>24.8</td>
    # We extract by label rather than positional index because the page
    # rearranges fields between equities/indices/funds.
    _LABEL_PATTERNS: dict[str, re.Pattern[str]] = {
        "name": re.compile(
            r"<title>\s*([^<>]+?)\s*[\|\-–]\s*Stooq", re.IGNORECASE
        ),
        "last_price": re.compile(
            r"(?:Ostatnio|Kurs)[^<]*</td>\s*<td[^>]*>\s*<[^>]*>?\s*("
            + _NUM + r")",
            re.IGNORECASE,
        ),
        "market_cap": re.compile(
            r"Kapitalizacja[^<]*</td>\s*<td[^>]*>\s*([^<]+)",
            re.IGNORECASE,
        ),
        "pe_ratio": re.compile(
            r"\bP/E\b[^<]*</td>\s*<td[^>]*>\s*([^<]+)", re.IGNORECASE
        ),
        "pbv_ratio": re.compile(
            r"\bC/WK\b[^<]*</td>\s*<td[^>]*>\s*([^<]+)", re.IGNORECASE
        ),
        "eps": re.compile(
            r"\bEPS\b[^<]*</td>\s*<td[^>]*>\s*([^<]+)", re.IGNORECASE
        ),
        "dividend_yield": re.compile(
            r"Stopa\s*dyw[^<]*</td>\s*<td[^>]*>\s*([^<]+)", re.IGNORECASE
        ),
        "week52_high": re.compile(
            r"Max\s*\(?\s*52\s*[Tt]?\s*\)?[^<]*</td>\s*<td[^>]*>\s*("
            + _NUM + r")",
            re.IGNORECASE,
        ),
        "week52_low": re.compile(
            r"Min\s*\(?\s*52\s*[Tt]?\s*\)?[^<]*</td>\s*<td[^>]*>\s*("
            + _NUM + r")",
            re.IGNORECASE,
        ),
    }

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

    # -- Public API ----------------------------------------------------------

    def fetch_snapshot(self, ticker: str) -> FundamentalsSnapshot:
        """Return a snapshot for ``ticker``.

        Always returns a snapshot — fields that couldn't be parsed are
        ``None``. Raises :class:`ProviderError` only on a hard HTTP failure.
        """
        symbol = normalize_ticker(ticker)
        url = f"{self.BASE_URL}?s={self._to_stooq_symbol(symbol)}"

        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Stooq fundamentals fetch failed for %s: %s", symbol, exc)
            raise ProviderError(
                f"Stooq fundamentals HTTP error for {symbol}: {exc}"
            ) from exc

        html = resp.text or ""
        return self._parse(symbol, html, source_url=url)

    # -- Internal ------------------------------------------------------------

    @staticmethod
    def _to_stooq_symbol(symbol: str) -> str:
        if symbol.startswith("^"):
            return symbol[1:]
        if symbol.endswith(".pl"):
            return symbol[:-3]
        return symbol

    def _parse(
        self, symbol: str, html: str, *, source_url: str
    ) -> FundamentalsSnapshot:
        def grab(field: str) -> str | None:
            pat = self._LABEL_PATTERNS[field]
            m = pat.search(html)
            return m.group(1).strip() if m else None

        name_raw = grab("name")
        # Stooq titles often look like "DNP - Dino Polska SA - Stooq".
        # Strip ticker prefix when present.
        name: str | None = None
        if name_raw:
            cleaned = re.sub(
                r"^\s*[A-Z0-9.^]+\s*[-–]\s*", "", name_raw
            ).strip()
            name = cleaned or name_raw

        snapshot = FundamentalsSnapshot(
            ticker=symbol,
            name=name,
            last_price=_to_float(grab("last_price")),
            market_cap=_market_cap_to_pln(grab("market_cap")),
            pe_ratio=_to_float(grab("pe_ratio")),
            pbv_ratio=_to_float(grab("pbv_ratio")),
            eps=_to_float(grab("eps")),
            dividend_yield=_pct_to_fraction(grab("dividend_yield")),
            week52_high=_to_float(grab("week52_high")),
            week52_low=_to_float(grab("week52_low")),
            source="stooq",
            fetched_at=datetime.now(timezone.utc),
            source_url=source_url,
        )

        # If literally everything is None, the page likely didn't render the
        # snapshot panel (rate limit / JS rendering / unsupported instrument).
        # Raise so the caller can fall back to a different source instead of
        # passing a hollow snapshot with a misleading "source=stooq" tag.
        if all(
            getattr(snapshot, f) is None
            for f in (
                "last_price",
                "market_cap",
                "pe_ratio",
                "eps",
                "week52_high",
            )
        ):
            raise ProviderError(
                f"Stooq returned an empty snapshot panel for {symbol} "
                "(likely JS-rendered or unsupported)"
            )

        return snapshot

    def __repr__(self) -> str:  # pragma: no cover
        return f"StooqFundamentalsProvider(timeout={self.timeout})"


__all__ = ["StooqFundamentalsProvider"]
