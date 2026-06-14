"""Tests for the Yahoo Finance OHLCV provider, with a stubbed HTTP session."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from investment_copilot.infrastructure.providers.base import ProviderError
from investment_copilot.infrastructure.providers.yahoo import YahooProvider


def _epoch(d: date) -> int:
    """Midday-UTC epoch for a date — safely inside the Warsaw trading day."""
    return int(datetime(d.year, d.month, d.day, 12, 0, tzinfo=timezone.utc).timestamp())


def _chart_payload(dates: list[date], *, with_volume: bool = True) -> dict:
    n = len(dates)
    quote = {
        "open": [77.5 + i for i in range(n)],
        "high": [78.5 + i for i in range(n)],
        "low": [77.1 + i for i in range(n)],
        "close": [78.2 + i for i in range(n)],
        "volume": [1_000_000 + i for i in range(n)] if with_volume else [None] * n,
    }
    return {
        "chart": {
            "result": [
                {
                    "meta": {"currency": "PLN", "instrumentType": "EQUITY"},
                    "timestamp": [_epoch(d) for d in dates],
                    "indicators": {"quote": [quote]},
                }
            ],
            "error": None,
        }
    }


def _fake_session(payload: dict, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.json = MagicMock(return_value=payload)
    response.status_code = status_code
    session = MagicMock()
    session.get = MagicMock(return_value=response)
    session.headers = {}
    return session


def test_fetch_ohlcv_happy_path() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    session = _fake_session(_chart_payload(dates))
    provider = YahooProvider(session=session)

    df = provider.fetch_ohlcv("PKN", start=date(2024, 1, 1), end=date(2024, 1, 31))

    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[-1] == pytest.approx(80.2)
    assert [d.date() for d in df.index] == dates
    # PKN -> Warsaw-listed PKN.WA; daily interval.
    call = session.get.call_args
    assert call.args[0].endswith("/PKN.WA")
    assert call.kwargs["params"]["interval"] == "1d"


def test_fetch_benchmark_maps_index_to_etf_proxy() -> None:
    session = _fake_session(_chart_payload([date(2024, 1, 2), date(2024, 1, 3)]))
    provider = YahooProvider(session=session)

    provider.fetch_benchmark("wig20", start=date(2024, 1, 1))

    # Yahoo has no WIG20 index history -> the Beta ETF WIG20TR proxy is used.
    assert session.get.call_args.args[0].endswith("/ETFBW20TR.WA")


def test_foreign_suffix_maps_to_yahoo_exchange() -> None:
    session = _fake_session(_chart_payload([date(2024, 1, 2), date(2024, 1, 3)]))
    provider = YahooProvider(session=session)

    # A London-listed benchmark (Stooq's "cspx.uk") -> Yahoo's "CSPX.L".
    provider.fetch_benchmark("cspx.uk", start=date(2024, 1, 1))

    assert session.get.call_args.args[0].endswith("/CSPX.L")


def test_fetch_window_is_trimmed() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 10), date(2024, 1, 20)]
    session = _fake_session(_chart_payload(dates))
    provider = YahooProvider(session=session)

    df = provider.fetch_ohlcv("PKN", start=date(2024, 1, 5), end=date(2024, 1, 15))

    assert [d.date() for d in df.index] == [date(2024, 1, 10)]


def test_fetch_no_data_error_raises() -> None:
    payload = {
        "chart": {
            "result": None,
            "error": {"code": "Not Found", "description": "No data found"},
        }
    }
    provider = YahooProvider(session=_fake_session(payload))
    with pytest.raises(ProviderError, match="no data"):
        provider.fetch_ohlcv("XYZ", start=date(2024, 1, 1))


def test_fetch_empty_result_raises() -> None:
    provider = YahooProvider(session=_fake_session({"chart": {"result": [], "error": None}}))
    with pytest.raises(ProviderError):
        provider.fetch_ohlcv("XYZ", start=date(2024, 1, 1))


def test_unsupported_benchmark_raises_actionable() -> None:
    provider = YahooProvider(session=_fake_session({}))
    # ^wig / ^wig30 have no GPW ETF tracker on Yahoo.
    with pytest.raises(ProviderError, match="supported benchmarks"):
        provider.fetch_benchmark("wig", start=date(2024, 1, 1))


def test_fetch_http_error_raises() -> None:
    from curl_cffi.requests.exceptions import RequestException

    session = MagicMock()
    session.get.side_effect = RequestException("boom")
    session.headers = {}

    provider = YahooProvider(session=session)
    with pytest.raises(ProviderError, match="HTTP error"):
        provider.fetch_ohlcv("PKN", start=date(2024, 1, 1))
