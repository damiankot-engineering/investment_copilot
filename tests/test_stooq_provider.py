"""Tests for the Stooq OHLCV provider, with a stubbed HTTP session."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from investment_copilot.infrastructure.providers.base import ProviderError
from investment_copilot.infrastructure.providers.stooq import StooqProvider


CSV_OK = (
    "Date,Open,High,Low,Close,Volume\n"
    "2024-01-02,77.50,78.50,77.10,78.20,1234567\n"
    "2024-01-03,78.10,79.00,77.80,78.90,2345678\n"
    "2024-01-04,78.90,79.50,78.30,79.10,1111111\n"
)

CSV_INDEX_NO_VOLUME = (
    "Date,Open,High,Low,Close\n"
    "2024-01-02,2150.0,2160.0,2140.0,2155.0\n"
    "2024-01-03,2155.0,2170.0,2150.0,2165.0\n"
)


def _fake_session(text: str, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    session = MagicMock()
    session.get = MagicMock(return_value=response)
    session.headers = {}
    return session


def test_fetch_ohlcv_happy_path() -> None:
    session = _fake_session(CSV_OK)
    provider = StooqProvider(session=session)

    df = provider.fetch_ohlcv("PKN", start=date(2024, 1, 1), end=date(2024, 1, 31))

    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[-1] == pytest.approx(79.10)
    # URL params include normalized symbol
    call = session.get.call_args
    assert call.kwargs["params"]["s"] == "pkn.pl"
    assert call.kwargs["params"]["d1"] == "20240101"
    assert call.kwargs["params"]["d2"] == "20240131"
    assert call.kwargs["params"]["i"] == "d"


def test_fetch_benchmark_resolves_symbol() -> None:
    session = _fake_session(CSV_INDEX_NO_VOLUME)
    provider = StooqProvider(session=session)

    df = provider.fetch_benchmark("wig20", start=date(2024, 1, 1))

    # Volume defaulted to 0 since indices don't supply it
    assert (df["volume"] == 0.0).all()
    assert session.get.call_args.kwargs["params"]["s"] == "^wig20"


def test_fetch_no_data_response_raises() -> None:
    provider = StooqProvider(session=_fake_session("No data"))
    with pytest.raises(ProviderError, match="no data"):
        provider.fetch_ohlcv("XYZ", start=date(2024, 1, 1))


def test_fetch_html_error_raises() -> None:
    provider = StooqProvider(session=_fake_session("<html>oops</html>"))
    with pytest.raises(ProviderError):
        provider.fetch_ohlcv("XYZ", start=date(2024, 1, 1))


def test_fetch_empty_body_raises() -> None:
    provider = StooqProvider(session=_fake_session(""))
    with pytest.raises(ProviderError, match="empty"):
        provider.fetch_ohlcv("XYZ", start=date(2024, 1, 1))


def test_fetch_http_error_raises() -> None:
    import requests as _requests

    response = MagicMock()
    response.raise_for_status.side_effect = _requests.HTTPError("503")
    session = MagicMock()
    session.get.return_value = response
    session.headers = {}

    provider = StooqProvider(session=session)
    with pytest.raises(ProviderError):
        provider.fetch_ohlcv("PKN", start=date(2024, 1, 1))
