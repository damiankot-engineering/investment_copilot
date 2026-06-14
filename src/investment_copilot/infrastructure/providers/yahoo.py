"""Yahoo Finance adapter for daily OHLCV.

Yahoo's public chart endpoint serves daily bars as JSON, with no API key. GPW
equities are quoted on the Warsaw exchange with a ``.WA`` suffix
(e.g. ``CDR.WA`` = CD Projekt), priced in PLN::

    https://query1.finance.yahoo.com/v8/finance/chart/CDR.WA?interval=1d&period1=...&period2=...

Yahoo does **not** carry daily history for the bare GPW indices (WIG20, etc.),
so benchmarks are mapped to the matching GPW-listed Beta ETF total-return
tracker (see :data:`_BENCHMARK_PROXY`).

Internal symbols arrive already normalized (``pkn.pl``, ``^wig20``) by
:mod:`investment_copilot.domain.models`; this adapter maps them to Yahoo's form.

Yahoo runs **TLS-fingerprint bot detection**: a plain ``requests`` client is
served HTTP 429 after a few calls, regardless of IP rate. We use ``curl_cffi``
impersonating Chrome so requests carry a real browser's TLS fingerprint, which
Yahoo accepts. We additionally seed session cookies once, throttle the per-ticker
fan-out, retry 429/5xx with backoff, and alternate the two query hosts.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
from curl_cffi import requests as cffi_requests
from curl_cffi.requests.exceptions import RequestException as HttpRequestError

from investment_copilot.domain.models import (
    OHLCV_COLUMNS,
    normalize_ticker,
    resolve_benchmark,
    validate_ohlcv_frame,
)
from investment_copilot.infrastructure.providers.base import ProviderError

logger = logging.getLogger(__name__)


#: Internal index symbol -> GPW-listed total-return ETF that Yahoo carries with
#: full daily history. Yahoo has no usable series for the bare price indices.
_BENCHMARK_PROXY: dict[str, str] = {
    "^wig20": "ETFBW20TR.WA",
    "^mwig40": "ETFBM40TR.WA",
    "^swig80": "ETFBS80TR.WA",
}

#: Internal exchange suffix (Stooq style) -> Yahoo exchange suffix. An empty
#: value means Yahoo uses no suffix (US listings). GPW (``.pl``) maps to Warsaw.
_YAHOO_EXCHANGE_SUFFIX: dict[str, str] = {
    "pl": "WA",  # GPW (Warsaw)
    "uk": "L",   # London Stock Exchange
    "de": "DE",  # Xetra / Frankfurt
    "fr": "PA",  # Euronext Paris
    "nl": "AS",  # Euronext Amsterdam
    "us": "",    # US listings carry no Yahoo suffix
}

#: HTTP statuses worth retrying (rate limit + transient server errors).
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class _NoDataError(Exception):
    """Internal signal: Yahoo returned no rows for a symbol."""


class YahooProvider:
    """Daily OHLCV via Yahoo Finance's public chart endpoint."""

    name: str = "yahoo"

    HOSTS: tuple[str, ...] = (
        "https://query1.finance.yahoo.com",
        "https://query2.finance.yahoo.com",
    )
    CHART_PATH: str = "/v8/finance/chart/"
    DEFAULT_TIMEOUT: float = 30.0
    MAX_RETRIES: int = 5
    BACKOFF_BASE: float = 0.6
    MAX_BACKOFF: float = 8.0
    #: Minimum gap between outgoing requests. A portfolio refresh fans out one
    #: request per ticker; spacing them keeps a burst from earning an IP block.
    MIN_REQUEST_INTERVAL: float = 0.4
    #: Browser profile curl_cffi impersonates so requests carry a real Chrome
    #: TLS fingerprint (the thing Yahoo's bot detection actually checks).
    IMPERSONATE: str = "chrome"

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        session: Any | None = None,
        max_retries: int = MAX_RETRIES,
        backoff_base: float = BACKOFF_BASE,
    ) -> None:
        self.timeout = timeout
        self._max_retries = max(1, max_retries)
        self._backoff_base = backoff_base
        # curl_cffi's session sends a genuine browser TLS fingerprint; a plain
        # requests session gets 429'd by Yahoo's bot detection. Tests inject a
        # stub session instead.
        self._session = session or cffi_requests.Session(impersonate=self.IMPERSONATE)
        # No manual User-Agent/Accept: impersonate="chrome" installs a complete,
        # self-consistent browser header set; overriding it defeats the point.
        self._warmed = False
        self._last_request_ts = 0.0

    # -- Public API ----------------------------------------------------------

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame:
        symbol = normalize_ticker(ticker)
        return self._fetch(symbol, start, end)

    def fetch_benchmark(
        self,
        benchmark: str,
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame:
        symbol = resolve_benchmark(benchmark)
        return self._fetch(symbol, start, end)

    # -- Internal ------------------------------------------------------------

    @staticmethod
    def _to_yahoo_symbol(symbol: str) -> str:
        """Map an internal symbol to Yahoo's convention.

        - Indices (``^wig20``): resolve to the GPW Beta ETF proxy.
        - Equities by exchange suffix: ``pkn.pl`` -> ``PKN.WA`` (Warsaw),
          ``cspx.uk`` -> ``CSPX.L`` (London), etc. (see
          :data:`_YAHOO_EXCHANGE_SUFFIX`).
        - Unknown / already-Yahoo suffixes pass through uppercased.
        """
        if symbol.startswith("^"):
            proxy = _BENCHMARK_PROXY.get(symbol.lower())
            if proxy is None:
                supported = ", ".join(sorted(k[1:] for k in _BENCHMARK_PROXY))
                raise ProviderError(
                    f"Yahoo has no daily history for index {symbol!r}; "
                    f"supported benchmarks: {supported}."
                )
            return proxy

        body, dot, suffix = symbol.lower().rpartition(".")
        if dot and suffix in _YAHOO_EXCHANGE_SUFFIX:
            yahoo_suffix = _YAHOO_EXCHANGE_SUFFIX[suffix]
            return f"{body.upper()}.{yahoo_suffix}" if yahoo_suffix else body.upper()
        # No known exchange suffix (e.g. an already-Yahoo ".WA" or a bare
        # symbol): hand it to Yahoo uppercased and let it 404 if unknown.
        return symbol.upper()

    def _ensure_warm(self) -> None:
        """Seed Yahoo consent/session cookies once; best-effort.

        An anonymous chart request is far more likely to be rate-limited (429)
        than one carrying the cookies Yahoo's homepage sets.
        """
        if self._warmed:
            return
        self._warmed = True
        try:
            self._session.get("https://finance.yahoo.com", timeout=self.timeout)
        except HttpRequestError:
            pass  # cookies are an optimization, not a requirement

    def _fetch(self, symbol: str, start: date, end: date | None) -> pd.DataFrame:
        effective_end = end or date.today()
        yahoo_s = self._to_yahoo_symbol(symbol)

        # Yahoo's period2 is exclusive of the bar's epoch; add a day so the end
        # date itself is included, then trim the returned frame to the window.
        period1 = int(
            datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()
        )
        period2 = int(
            (
                datetime(
                    effective_end.year,
                    effective_end.month,
                    effective_end.day,
                    tzinfo=timezone.utc,
                )
                + timedelta(days=1)
            ).timestamp()
        )
        params = {
            "interval": "1d",
            "period1": str(period1),
            "period2": str(period2),
            "includePrePost": "false",
        }

        payload = self._get_json(yahoo_s, params, symbol)
        try:
            df = self._parse_chart(payload, symbol)
        except _NoDataError as exc:
            raise ProviderError(str(exc)) from exc

        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(effective_end))]
        if df.empty:
            raise ProviderError(
                f"Yahoo returned no rows in [{start}, {effective_end}] for {symbol}"
            )
        return df

    def _get_json(
        self, yahoo_s: str, params: dict[str, str], symbol: str
    ) -> dict[str, Any]:
        """GET the chart JSON, retrying rate-limits/5xx across both hosts."""
        self._ensure_warm()
        last_status: int | None = None

        for attempt in range(self._max_retries):
            host = self.HOSTS[attempt % len(self.HOSTS)]
            url = f"{host}{self.CHART_PATH}{yahoo_s}"
            logger.debug("Yahoo GET %s params=%s", url, params)
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except HttpRequestError as exc:
                raise ProviderError(f"Yahoo HTTP error for {symbol}: {exc}") from exc

            status = resp.status_code
            if status in _RETRY_STATUSES:
                last_status = status
                self._sleep_before_retry(resp, attempt)
                continue

            # Yahoo returns a JSON error body even on 404 (unknown symbol), so
            # parse on any non-retryable status and let _parse_chart classify.
            try:
                return resp.json()
            except ValueError as exc:
                # 200-but-not-JSON is unexpected; treat like a transient blip.
                last_status = status
                if attempt + 1 < self._max_retries:
                    self._sleep_before_retry(resp, attempt)
                    continue
                raise ProviderError(
                    f"Yahoo returned a non-JSON response for {symbol} (HTTP {status})"
                ) from exc

        raise ProviderError(
            f"Yahoo rate-limited or unavailable for {symbol} "
            f"(last HTTP {last_status}); try again shortly."
        )

    def _throttle(self) -> None:
        """Space outgoing requests by at least ``MIN_REQUEST_INTERVAL`` seconds.

        Shared one provider instance across a refresh, so this paces the whole
        per-ticker fan-out. The first request never waits.
        """
        if self.MIN_REQUEST_INTERVAL <= 0 or self._last_request_ts == 0.0:
            self._last_request_ts = time.monotonic()
            return
        wait = self.MIN_REQUEST_INTERVAL - (time.monotonic() - self._last_request_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    def _sleep_before_retry(self, resp: Any, attempt: int) -> None:
        """Back off before retrying: honor ``Retry-After``, else exponential."""
        retry_after = resp.headers.get("Retry-After", "")
        if retry_after.strip().isdigit():
            delay = float(retry_after.strip())
        else:
            delay = self._backoff_base * (2**attempt)
        time.sleep(min(delay, self.MAX_BACKOFF))

    @staticmethod
    def _parse_chart(payload: dict[str, Any], symbol: str) -> pd.DataFrame:
        chart = payload.get("chart") or {}
        error = chart.get("error")
        if error:
            code = error.get("code", "error")
            desc = error.get("description", "")
            # "Not Found" / "No data found" are no-data, not a transport failure.
            if "not found" in f"{code} {desc}".lower():
                raise _NoDataError(f"Yahoo has no data for {symbol} ({code})")
            raise ProviderError(f"Yahoo error for {symbol}: {code} {desc}".strip())

        results = chart.get("result")
        if not results:
            raise _NoDataError(f"Yahoo returned no result for {symbol}")

        result = results[0]
        timestamps = result.get("timestamp")
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        if not timestamps or not quote:
            raise _NoDataError(f"Yahoo returned no rows for {symbol}")

        # Epoch seconds (UTC) -> Warsaw trading date (naive midnight), matching
        # the date semantics the rest of the pipeline expects.
        idx = (
            pd.to_datetime(timestamps, unit="s", utc=True)
            .tz_convert("Europe/Warsaw")
            .normalize()
            .tz_localize(None)
        )
        df = pd.DataFrame(
            {
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            },
            index=idx,
        )
        df.index.name = "date"

        for col in OHLCV_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Yahoo emits null rows for non-trading gaps; drop any without a close.
        df = df.dropna(subset=["close"])
        if df.empty:
            raise _NoDataError(f"Yahoo returned only empty rows for {symbol}")

        return validate_ohlcv_frame(df, symbol=symbol)
