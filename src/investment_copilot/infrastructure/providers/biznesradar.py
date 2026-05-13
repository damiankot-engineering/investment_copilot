"""BiznesRadar.pl adapter — primary fundamentals + ESPI source.

BiznesRadar publishes server-rendered HTML with clean tables, so it's the
most stable free source for GPW fundamentals + recent ESPI/EBI messages.
The page structure is:

* ``/raporty-finansowe-rachunek-zyskow-i-strat/<TICKER>,Q,1``
  — quarterly P&L (table[2]: 19 rows × 40+ quarters), latest narrative
  bullets (table[0]: "Przychody: wzrost o X% r/r"), market data
  (table[3]: kapitalizacja, EV, branża).
* ``/wskazniki-wartosci-rynkowej/<TICKER>``
  — valuation ratios (P/E, P/BV, Graham, EPS).
* ``/komunikaty-espi/<TICKER>``
  — list of ESPI/EBI announcements.

Behaviour mirrors the other providers: best-effort. Network errors are
logged + a ``ProviderError`` is raised so the caller can fall back to a
secondary source (Stooq, OHLCV cache).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from io import StringIO

import pandas as pd
import requests

from investment_copilot.domain.fundamentals import FundamentalsSnapshot
from investment_copilot.domain.models import NewsItem, normalize_ticker
from investment_copilot.infrastructure.providers.base import ProviderError

logger = logging.getLogger(__name__)


_BASE = "https://www.biznesradar.pl"
_DEFAULT_TIMEOUT = 30.0

# Regex for the narrative-summary cells: "Przychody...: wzrost o 14.90% r/r"
_NARRATIVE_PCT_RE = re.compile(
    r"(wzrost|spadek)\s*o\s*([\d.,]+)\s*%\s*r/r", re.IGNORECASE
)

# Cell format in quarterly tables: "33 634 155r/r +14.90%~branża +11.04%k/k +3.52%~branża +3.38%"
_VALUE_AT_START_RE = re.compile(r"^\s*([-\d\s]+(?:[.,]\d+)?)")
_YOY_PCT_IN_CELL_RE = re.compile(r"r/r\s*([+-]?[\d.,]+)\s*%")


def _to_float(raw) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if pd.notna(raw) else None
    s = str(raw).replace("\xa0", "").replace(" ", "").replace(",", ".")
    if not s or s in {"-", "—", "nan", "NaN", "None"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_pct_narrative(text: str) -> float | None:
    """Extract a YoY % from cells like 'Przychody: wzrost o 14.90% r/r'."""
    m = _NARRATIVE_PCT_RE.search(text)
    if not m:
        return None
    direction, value = m.group(1), m.group(2)
    val = _to_float(value)
    if val is None:
        return None
    return -val if direction.lower() == "spadek" else val


def _parse_first_number(cell) -> float | None:
    """First numeric chunk in a quarterly-cell string."""
    if pd.isna(cell):
        return None
    s = str(cell)
    m = _VALUE_AT_START_RE.match(s)
    if not m:
        return None
    return _to_float(m.group(1))


def _ticker_to_br_symbol(ticker: str) -> str:
    """Normalize 'dnp.pl' → 'DNP' for BiznesRadar URLs."""
    norm = normalize_ticker(ticker)
    body = norm[1:] if norm.startswith("^") else norm
    if body.endswith(".pl"):
        body = body[:-3]
    return body.upper()


class BiznesRadarProvider:
    """Best-effort fundamentals + ESPI scraper for BiznesRadar.pl."""

    name: str = "biznesradar"

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.setdefault(
            "User-Agent",
            "Mozilla/5.0 (compatible; investment-copilot/0.1)",
        )

    # -- Public API ----------------------------------------------------------

    def fetch_fundamentals(self, ticker: str) -> FundamentalsSnapshot:
        """Fetch the rich fundamentals snapshot for ``ticker``.

        Combines two pages — quarterly P&L and valuation indicators — into
        a single :class:`FundamentalsSnapshot`. Raises :class:`ProviderError`
        on HTTP failure or when the page is unparsable.
        """
        symbol = _ticker_to_br_symbol(ticker)
        norm_ticker = normalize_ticker(ticker)

        pnl_url = f"{_BASE}/raporty-finansowe-rachunek-zyskow-i-strat/{symbol},Q,1"
        ind_url = f"{_BASE}/wskazniki-wartosci-rynkowej/{symbol}"

        pnl_html = self._fetch_html(pnl_url, what="quarterly P&L")
        ind_html = self._fetch_html(ind_url, what="indicators")

        pnl_data = self._parse_pnl_page(pnl_html)
        ind_data = self._parse_indicators_page(ind_html)

        # If we got nothing meaningful — no sector, no YoY, no market cap —
        # BR doesn't really cover this ticker (e.g. ETFs / funds without
        # quarterly P&L). Signal "not found" so the caller can fall back.
        has_anything_rich = any(
            pnl_data.get(k) is not None
            for k in (
                "sector",
                "market_cap",
                "revenue_yoy_pct",
                "ebitda_yoy_pct",
                "net_profit_yoy_pct",
                "last_report_date",
            )
        )
        if not has_anything_rich and not ind_data:
            raise ProviderError(
                f"BiznesRadar has no fundamentals coverage for {symbol} "
                f"(likely an ETF / fund / new listing)"
            )

        return FundamentalsSnapshot(
            ticker=norm_ticker,
            name=pnl_data.get("name"),
            # last_price intentionally left None — OHLCV cache is authoritative;
            # BR's "Kurs" field has inconsistent units across tickers.
            market_cap=pnl_data.get("market_cap"),
            enterprise_value=pnl_data.get("enterprise_value"),
            pe_ratio=ind_data.get("pe_ratio"),
            pbv_ratio=ind_data.get("pbv_ratio"),
            eps=ind_data.get("eps"),
            dividend_yield=ind_data.get("dividend_yield"),
            sector=pnl_data.get("sector"),
            latest_quarter_label=pnl_data.get("latest_quarter_label"),
            last_report_date=pnl_data.get("last_report_date"),
            next_report_estimated_date=pnl_data.get("next_report_estimated_date"),
            revenue_yoy_pct=pnl_data.get("revenue_yoy_pct"),
            ebitda_yoy_pct=pnl_data.get("ebitda_yoy_pct"),
            net_profit_yoy_pct=pnl_data.get("net_profit_yoy_pct"),
            latest_summary=pnl_data.get("latest_summary", []),
            source="biznesradar",
            fetched_at=datetime.now(timezone.utc),
            source_url=pnl_url,
        )

    def fetch_espi(
        self,
        ticker: str,  # noqa: ARG002 - kept for protocol stability
        *,
        since: datetime,  # noqa: ARG002
        limit: int = 30,  # noqa: ARG002
    ) -> list[NewsItem]:
        """Return an empty list — BR doesn't expose a free per-ticker ESPI feed.

        The ``/komunikaty-espi/<TICKER>`` URL 404s and ``/wiadomosci/<TICKER>``
        serves generic site-wide news. Per-ticker ESPI is paywalled behind
        BR Premium. Kept on the class for NewsProvider protocol stability;
        always returns an empty list.
        """
        return []

    # -- NewsProvider protocol surface (so BR can plug into news pipeline) --

    def fetch_news(
        self,
        since: datetime,
        *,
        ticker: str | None = None,
        keywords: list[str] | None = None,  # noqa: ARG002 - BR ignores keywords
    ) -> list[NewsItem]:
        if ticker is None:
            return []  # BR is per-symbol
        try:
            return self.fetch_espi(ticker, since=since)
        except ProviderError as exc:
            logger.warning("BR ESPI fetch failed for %s: %s", ticker, exc)
            return []

    # -- Internal ------------------------------------------------------------

    def _fetch_html(self, url: str, *, what: str) -> str:
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"BR {what} HTTP error: {exc}") from exc
        return resp.text or ""

    def _parse_pnl_page(self, html: str) -> dict:
        """Parse the quarterly P&L page.

        Three relevant tables (positions can shift between tickers, so we
        look up by content rather than index):

        * narrative summary list (1 col) — "Przychody...: wzrost o X% r/r"
        * wide quarterly table (~40+ cols) — rows = line items, cols = Qx/YYYY
        * 2-col market data table — Kapitalizacja, EV, Branża, ISIN
        """
        result: dict = {"latest_summary": []}
        if not html:
            return result

        try:
            tables = pd.read_html(StringIO(html), encoding="utf-8")
        except ValueError:
            return result

        # 1. Narrative summary — look for 1-col tables with "Przychody/wzrost/spadek"
        for t in tables:
            if t.shape[1] != 1:
                continue
            first_cells = t.iloc[:, 0].astype(str)
            if first_cells.str.contains(r"wzrost o|spadek o", regex=True).any():
                bullets = [c for c in first_cells.tolist() if "r/r" in c]
                result["latest_summary"].extend(bullets)
                for bullet in bullets:
                    pct = _parse_pct_narrative(bullet)
                    if pct is None:
                        continue
                    if "Przychody" in bullet:
                        result.setdefault("revenue_yoy_pct", pct)
                    elif "EBITDA" in bullet:
                        result.setdefault("ebitda_yoy_pct", pct)
                    elif "Zysk netto" in bullet:
                        result.setdefault("net_profit_yoy_pct", pct)

        # 2. Wide quarterly table — look for tables with many columns AND
        #    "Przychody ze sprzedaży" in column 0
        wide = self._find_quarterly_table(tables)
        if wide is not None:
            self._populate_from_quarterly(wide, result)

        # 3. Market data — look for 2-col tables with Kapitalizacja/Branża
        for t in tables:
            if t.shape[1] != 2:
                continue
            kv = self._table_to_kv(t)
            if "Kapitalizacja" in kv:
                result["market_cap"] = _market_cap_raw(kv.get("Kapitalizacja"))
            if "Enterprise Value" in kv:
                result["enterprise_value"] = _market_cap_raw(kv.get("Enterprise Value"))
            if "Branża" in kv:
                result["sector"] = kv["Branża"].strip() or None
            if "Nazwa" in kv:
                result["name"] = kv["Nazwa"].strip() or None

        return result

    @staticmethod
    def _find_quarterly_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
        for t in tables:
            if t.shape[1] < 8:
                continue
            first_col = t.iloc[:, 0].astype(str)
            if first_col.str.contains(r"Przychody ze sprzeda", regex=True).any():
                return t
        return None

    @staticmethod
    def _populate_from_quarterly(wide: pd.DataFrame, result: dict) -> None:
        """Pull last quarter label + publication date + estimate next date."""
        # The first data row is "Data publikacji" — use the rightmost non-NaN
        # cell to find the most recent publication date.
        cols = list(wide.columns)

        pub_row = wide[wide.iloc[:, 0].astype(str).str.contains("Data publikacji", na=False)]
        if pub_row.empty:
            return
        pub_series = pub_row.iloc[0]

        last_date_obj: date | None = None
        last_col_idx: int | None = None
        # iterate right-to-left to find the latest non-NaN publication date
        for idx in range(len(cols) - 1, -1, -1):
            val = pub_series.iloc[idx]
            if pd.isna(val):
                continue
            try:
                last_date_obj = datetime.strptime(str(val), "%Y-%m-%d").date()
                last_col_idx = idx
                break
            except ValueError:
                continue

        if last_date_obj:
            result["last_report_date"] = last_date_obj
            # Estimate next report at +~91 days (quarterly cadence).
            result["next_report_estimated_date"] = last_date_obj + timedelta(days=91)

        if last_col_idx is not None and last_col_idx < len(cols):
            # Column label looks like "2025/Q4  (gru 25)" or "2025  (gru 25)".
            label_raw = str(cols[last_col_idx])
            label = re.sub(r"\s+", " ", label_raw).strip()
            result["latest_quarter_label"] = label

            # If narratives weren't populated from summary tables, try to
            # extract YoY % from the wide table's last column.
            for row_label, key in (
                ("Przychody ze sprzeda", "revenue_yoy_pct"),
                ("EBITDA", "ebitda_yoy_pct"),
                ("Zysk netto", "net_profit_yoy_pct"),
            ):
                if key in result:
                    continue
                mask = wide.iloc[:, 0].astype(str).str.contains(row_label, na=False)
                if not mask.any():
                    continue
                cell = wide.loc[mask].iloc[0, last_col_idx]
                if pd.isna(cell):
                    continue
                m = _YOY_PCT_IN_CELL_RE.search(str(cell))
                if m:
                    result[key] = _to_float(m.group(1))

    def _parse_indicators_page(self, html: str) -> dict:
        result: dict = {}
        if not html:
            return result
        try:
            tables = pd.read_html(StringIO(html), encoding="utf-8")
        except ValueError:
            return result

        wide = self._find_quarterly_table(tables) if tables else None
        # Indicators page's wide table has many columns labelled by quarter.
        # Look for tables with 'Kurs' / 'C/Z' / 'C/WK' in first column.
        for t in tables:
            if t.shape[1] < 8:
                continue
            first_col = t.iloc[:, 0].astype(str)
            if not first_col.str.contains(r"Kurs|C/Z|C/WK|Wskaźnik", regex=True, na=False).any():
                continue
            wide = t
            break

        if wide is not None:
            cols = list(wide.columns)
            # Pick the right-most non-empty column as "latest".
            for idx in range(len(cols) - 1, 0, -1):
                if wide.iloc[:, idx].notna().any():
                    last_idx = idx
                    break
            else:
                last_idx = None

            if last_idx is not None:
                # NOTE: BR's "Kurs" field is unreliable (mixed units across
                # tickers — some report groszy, some PLN). We rely on the
                # local OHLCV cache for last_price and skip BR's value.
                for row_label, key in (
                    (r"^C/Z$|^Cena / Zysk$", "pe_ratio"),
                    (r"Cena / Wartość księgowa$|^C/WK$", "pbv_ratio"),
                    ("Zysk na akcję", "eps"),
                    ("Stopa dywidendy", "dividend_yield"),
                ):
                    mask = wide.iloc[:, 0].astype(str).str.contains(row_label, regex=True, na=False)
                    if not mask.any():
                        continue
                    cell = wide.loc[mask].iloc[0, last_idx]
                    val = _parse_first_number(cell)
                    if val is None:
                        continue
                    # Stopa dywidendy is published as % already; convert to fraction.
                    if key == "dividend_yield":
                        val = val / 100.0
                    result[key] = val

        return result

    @staticmethod
    def _table_to_kv(t: pd.DataFrame) -> dict[str, str]:
        out: dict[str, str] = {}
        for _, row in t.iterrows():
            k = str(row.iloc[0]).strip().rstrip(":")
            v = str(row.iloc[1]).strip()
            if k and v and v not in {"nan", "None"}:
                out[k] = v
        return out

    def _parse_espi(
        self,
        html: str,
        *,
        ticker: str,
        since: datetime,
        limit: int,
        source_url: str,
    ) -> list[NewsItem]:
        """Extract ESPI/EBI announcement rows. BR uses a list with date + title links."""
        if not html:
            return []
        # BR ESPI page has anchor tags around announcement titles. We parse
        # by regex rather than HTML parser to keep dependencies light.
        # Pattern: "<a href="/komunikat/..." title="..." class="...">TITLE</a>"
        # near a date string like "2026-05-09" or "9 maja 2026".
        items: list[NewsItem] = []
        since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)

        # Try a structured pattern first.
        row_re = re.compile(
            r'(?P<date>\d{4}-\d{2}-\d{2})[^<]{0,40}<a\s+href="(?P<url>/komunikat/[^"]+)"[^>]*>(?P<title>[^<]{4,300})</a>',
            re.IGNORECASE | re.DOTALL,
        )
        for m in row_re.finditer(html):
            try:
                published = datetime.strptime(m.group("date"), "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
            if published < since_aware:
                continue
            title = re.sub(r"\s+", " ", m.group("title")).strip()
            url = m.group("url")
            if url.startswith("/"):
                url = _BASE + url
            items.append(
                NewsItem(
                    ticker=ticker,
                    source="biznesradar:espi",
                    title=title,
                    url=url,
                    published_at=published,
                )
            )
            if len(items) >= limit:
                break

        if not items:
            logger.debug("BR ESPI: no items parsed for %s", ticker)
        return items

    def __repr__(self) -> str:  # pragma: no cover
        return f"BiznesRadarProvider(timeout={self.timeout})"


def _market_cap_raw(raw: str | None) -> float | None:
    """Parse '28 392 384 000' style cells into a raw float."""
    if raw is None:
        return None
    cleaned = str(raw).replace("\xa0", "").replace(" ", "").replace(",", ".")
    if not cleaned or cleaned in {"-", "—", "nan", "None"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


__all__ = ["BiznesRadarProvider"]
