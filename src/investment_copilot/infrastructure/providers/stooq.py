"""Stooq.com adapter for daily OHLCV.

Stooq requires an API key obtained via captcha at::

    https://stooq.pl/q/d/?s=<symbol>&get_apikey

Set ``STOOQ_API_KEY`` in the environment / ``.env`` file. URL format::

    https://stooq.com/q/d/l/?s=<symbol>&d1=YYYYMMDD&d2=YYYYMMDD&i=d&apikey=<key>

Indices use a ``^`` prefix (e.g. ``^wig20``); equities use lowercase suffix
notation (e.g. ``pkn.pl``). Symbol normalization is handled by
:mod:`investment_copilot.domain.models`.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from io import StringIO

import pandas as pd
import requests

from investment_copilot.domain.models import (
    OHLCV_COLUMNS,
    normalize_ticker,
    resolve_benchmark,
    validate_ohlcv_frame,
)
from investment_copilot.infrastructure.providers.base import ProviderError

logger = logging.getLogger(__name__)


class StooqProvider:
    """Daily OHLCV via Stooq's CSV endpoint."""

    name: str = "stooq"

    BASE_URL: str = "https://stooq.pl/q/d/l/"
    DEFAULT_TIMEOUT: float = 30.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self._api_key = api_key or os.environ.get("STOOQ_API_KEY") or ""
        self._session = session or requests.Session()
        self._session.headers.setdefault(
            "User-Agent", "investment-copilot/0.1 (+https://stooq.com)"
        )

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
    def _to_stooq_symbol(symbol: str) -> str:
        """Convert internal symbol to the form stooq.pl's authenticated API expects.

        The authenticated endpoint differs from the old unauthenticated one:
        - Equities: strip the `.pl` suffix (pkn.pl → pkn)
        - Indices:  strip the `^` prefix  (^wig20 → wig20)
        """
        if symbol.startswith("^"):
            return symbol[1:]
        if symbol.endswith(".pl"):
            return symbol[:-3]
        return symbol

    def _fetch(self, symbol: str, start: date, end: date | None) -> pd.DataFrame:
        # Stooq's authenticated endpoint requires d1 and d2 together; d1 alone
        # triggers a server-side DB error. Default d2 to today when not given.
        effective_end = end or date.today()
        params: dict[str, str] = {
            "s": self._to_stooq_symbol(symbol),
            "d1": start.strftime("%Y%m%d"),
            "d2": effective_end.strftime("%Y%m%d"),
            "i": "d",
        }
        if self._api_key:
            params["apikey"] = self._api_key

        logger.debug("Stooq GET %s params=%s", self.BASE_URL, params)
        try:
            resp = self._session.get(self.BASE_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"Stooq HTTP error for {symbol}: {exc}") from exc

        text = resp.text or ""
        stripped = text.lstrip()

        if not stripped:
            raise ProviderError(f"Stooq returned empty body for {symbol}")

        lower = stripped.lower()

        # Reject any response that is not a valid CSV.
        # Stooq returns various non-CSV bodies depending on the error:
        #   - API key instruction page (English or Polish)
        #   - HTML error page
        #   - PHP warning/error text
        #   - "No data" / "Brak danych" plaintext
        _API_KEY_MSG = (
            f"Stooq requires an API key for {symbol}. "
            "Obtain one at https://stooq.com/q/d/?s=pkn.pl&get_apikey "
            "and set STOOQ_API_KEY in your .env file."
        )
        if lower.startswith("uzyskaj apikey") or lower.startswith("get your apikey"):
            raise ProviderError(_API_KEY_MSG)

        first_line = stripped.splitlines()[0]
        first_lower = first_line.lower()

        if first_lower.startswith("<"):
            raise ProviderError(f"Stooq returned an HTML error page for {symbol}")
        if first_lower.startswith("no data") or first_lower.startswith("brak"):
            raise ProviderError(f"Stooq returned no data for {symbol}")
        # PHP warnings/errors emitted when the API key is missing or invalid.
        if first_lower.startswith("warning:") or first_lower.startswith("error:"):
            snippet = stripped[:300].replace("\n", " ")
            raise ProviderError(
                f"Stooq returned a server-side error for {symbol} "
                f"(missing or invalid STOOQ_API_KEY?): {snippet!r}"
            )
        # Final catch-all: a valid CSV header must contain commas.
        if "," not in first_line:
            snippet = stripped[:200].replace("\n", " ")
            raise ProviderError(
                f"Stooq returned an unexpected non-CSV response for {symbol}: {snippet!r}"
            )

        try:
            df = pd.read_csv(StringIO(text))
        except Exception as exc:
            raise ProviderError(f"Stooq CSV parse error for {symbol}: {exc}") from exc

        if df.empty:
            raise ProviderError(f"Stooq returned empty CSV for {symbol}")

        # stooq.pl returns Polish column headers; map them to the English names
        # the rest of the pipeline expects.
        _PL_TO_EN = {
            "data": "date",
            "otwarcie": "open",
            "najwyzszy": "high",
            "najnizszy": "low",
            "zamkniecie": "close",
            "wolumen": "volume",
        }
        df.columns = [_PL_TO_EN.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
        if "date" not in df.columns:
            raise ProviderError(
                f"Stooq CSV missing 'date' column for {symbol}: {list(df.columns)}"
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")

        # Indices may not include volume.
        if "volume" not in df.columns:
            df["volume"] = 0.0

        for col in OHLCV_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = validate_ohlcv_frame(df, symbol=symbol).dropna(subset=["close"])

        if df.empty:
            raise ProviderError(f"Stooq returned only invalid rows for {symbol}")

        return df
