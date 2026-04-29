"""Tests for CopilotService — fully stubbed LLM and DataService."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from investment_copilot.config.schema import LLMConfig
from investment_copilot.domain.models import NewsItem
from investment_copilot.domain.portfolio import (
    Holding,
    HoldingStatus,
    Portfolio,
    PortfolioStatus,
)
from investment_copilot.domain.prompts import (
    HoldingComment,
    PortfolioAnalysis,
    RiskAlert,
    RiskAlerts,
    ThesisUpdate,
)
from investment_copilot.services.copilot_service import CopilotService


# --- Stubs -----------------------------------------------------------------


class FakeLLM:
    """Returns a preset BaseModel based on the requested response_schema."""

    name = "fake-llm"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.return_for_schema: dict[type, Any] = {}

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_schema": response_schema,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if response_schema in self.return_for_schema:
            return self.return_for_schema[response_schema]
        raise AssertionError(f"No stub configured for {response_schema}")

    def complete_text(self, **_kw: Any) -> str:  # not used here
        return ""


class FakeData:
    """Captures load_news calls; returns preset items."""

    def __init__(self, news_by_ticker: dict[str | None, list[NewsItem]] | None = None) -> None:
        self.news_by_ticker = news_by_ticker or {}
        self.calls: list[dict[str, Any]] = []

    def load_news(self, *, ticker: str | None = None, since=None, limit: int | None = None):
        self.calls.append({"ticker": ticker, "since": since, "limit": limit})
        return list(self.news_by_ticker.get(ticker, []))


# --- Fixtures --------------------------------------------------------------


def _portfolio() -> Portfolio:
    return Portfolio(
        holdings=[
            Holding(
                ticker="PKN",
                shares=10,
                entry_price=65,
                entry_date=date(2023, 1, 2),
                thesis="thesis pkn",
                keywords=["Orlen"],
            ),
            Holding(
                ticker="CDR",
                shares=5,
                entry_price=200,
                entry_date=date(2023, 1, 2),
                thesis="thesis cdr",
            ),
        ]
    )


def _status() -> PortfolioStatus:
    return PortfolioStatus(
        base_currency="PLN",
        as_of=datetime(2024, 4, 1, tzinfo=timezone.utc),
        holdings=[
            HoldingStatus(
                ticker="pkn.pl",
                name=None,
                shares=10,
                entry_price=65,
                entry_date=date(2023, 1, 2),
                cost_basis=650,
                last_price=70,
                last_price_date=date(2024, 4, 1),
                market_value=700,
                unrealized_pnl=50,
                unrealized_pnl_pct=50 / 650,
            ),
        ],
        total_cost_basis=1650.0,
        priced_cost_basis=650.0,
        total_market_value=700.0,
        total_unrealized_pnl=50.0,
        total_unrealized_pnl_pct=50 / 650,
        missing_data=["cdr.pl"],
    )


def _llm_config() -> LLMConfig:
    return LLMConfig(
        api_key="x",
        model_analysis="custom-analysis",
        model_summary="custom-summary",
        temperature=0.4,
        max_tokens=1234,
    )


def _service(llm: FakeLLM, data: FakeData) -> CopilotService:
    return CopilotService(llm_client=llm, data_service=data, llm_config=_llm_config())


# --- Tests: analyze_portfolio ---------------------------------------------


def test_analyze_portfolio_returns_typed_result_and_passes_polish_context() -> None:
    expected = PortfolioAnalysis(
        summary="Krótkie podsumowanie portfela.",
        holdings_comments=[
            HoldingComment(ticker="pkn.pl", comment="Pozycja stabilna.", recommendation="trzymaj"),
            HoldingComment(ticker="cdr.pl", comment="Brak danych.", recommendation="obserwuj"),
        ],
        diversification_notes="Koncentracja na GPW.",
        confidence=7,
    )
    llm = FakeLLM()
    llm.return_for_schema[PortfolioAnalysis] = expected
    data = FakeData(
        {
            "pkn.pl": [
                NewsItem(
                    ticker="pkn.pl",
                    source="rss:bankier",
                    title="Orlen ogłasza wyniki Q4",
                    url="https://example.com/n1",
                    published_at=datetime(2024, 3, 30, tzinfo=timezone.utc),
                )
            ],
            "cdr.pl": [],
            None: [],
        }
    )

    svc = _service(llm, data)
    result = svc.analyze_portfolio(_portfolio(), _status())

    assert result is expected
    call = llm.calls[0]
    assert call["response_schema"] is PortfolioAnalysis
    assert call["model"] == "custom-analysis"
    assert call["temperature"] == 0.4
    assert call["max_tokens"] == 1234

    # Polish system prompt with role
    assert "po polsku" in call["system_prompt"]
    assert "GPW" in call["system_prompt"]

    # User prompt embeds the structured context
    user = call["user_prompt"]
    assert "## Holdings" in user
    assert "## Current status" in user
    assert "## Recent news" in user
    assert "Orlen ogłasza wyniki" in user

    # News fetched once per ticker + once for general
    fetched_tickers = [c["ticker"] for c in data.calls]
    assert fetched_tickers == ["pkn.pl", "cdr.pl", None]


def test_analyze_portfolio_includes_backtest_when_provided() -> None:
    from datetime import date

    from investment_copilot.domain.backtest import (
        BacktestResult,
        EquityPoint,
        StrategyMetrics,
    )

    bt = BacktestResult(
        strategy_name="ma_crossover",
        strategy_params={"fast": 50, "slow": 200},
        start_date=date(2022, 1, 3),
        end_date=date(2024, 4, 1),
        initial_capital=100_000.0,
        final_value=125_000.0,
        equity_curve=[EquityPoint(date=date(2022, 1, 3), value=100_000.0)],
        metrics=StrategyMetrics(
            total_return=0.25,
            annualized_return=0.12,
            annualized_volatility=0.18,
            sharpe_ratio=0.67,
            max_drawdown=-0.15,
            max_drawdown_duration_days=42,
            win_rate=0.55,
            n_observations=500,
        ),
        generated_at=datetime(2024, 4, 1, tzinfo=timezone.utc),
    )
    llm = FakeLLM()
    llm.return_for_schema[PortfolioAnalysis] = PortfolioAnalysis(
        summary="x",
        holdings_comments=[
            HoldingComment(ticker="pkn.pl", comment="y", recommendation="trzymaj"),
        ],
        diversification_notes="z",
        confidence=5,
    )
    svc = _service(llm, FakeData())
    svc.analyze_portfolio(_portfolio(), _status(), backtest=bt)
    assert "ma_crossover" in llm.calls[0]["user_prompt"]


# --- Tests: detect_risks --------------------------------------------------


def test_detect_risks_uses_risk_schema_and_template() -> None:
    expected = RiskAlerts(
        overview="Krótki przegląd ryzyka.",
        alerts=[
            RiskAlert(
                ticker="pkn.pl",
                severity="średnie",
                title="Wahania cen ropy",
                description="Ceny ropy istotnie wpływają na wynik segmentu rafineryjnego.",
                suggested_action="Monitoruj raporty kwartalne.",
            )
        ],
    )
    llm = FakeLLM()
    llm.return_for_schema[RiskAlerts] = expected
    svc = _service(llm, FakeData())

    result = svc.detect_risks(_portfolio(), _status())

    assert result is expected
    assert llm.calls[0]["response_schema"] is RiskAlerts
    assert "ryzyka" in llm.calls[0]["system_prompt"].lower()


# --- Tests: update_thesis ------------------------------------------------


def test_update_thesis_returns_typed_result_and_only_loads_target_news() -> None:
    expected = ThesisUpdate(
        ticker="pkn.pl",
        thesis_status="potwierdzona",
        rationale="Wyniki Q4 wspierają tezę.",
        confidence=8,
    )
    llm = FakeLLM()
    llm.return_for_schema[ThesisUpdate] = expected
    data = FakeData({"pkn.pl": []})

    svc = _service(llm, data)
    result = svc.update_thesis(_portfolio(), _status(), ticker="PKN")

    assert result is expected
    # Only one news fetch — for the target ticker
    assert [c["ticker"] for c in data.calls] == ["pkn.pl"]
    # Template substituted the ticker
    assert "pkn.pl" in llm.calls[0]["user_prompt"]


def test_update_thesis_normalizes_ticker_input() -> None:
    expected = ThesisUpdate(
        ticker="pkn.pl",
        thesis_status="potwierdzona",
        rationale="r",
        confidence=5,
    )
    llm = FakeLLM()
    llm.return_for_schema[ThesisUpdate] = expected
    data = FakeData({"pkn.pl": []})
    svc = _service(llm, data)

    # Caller passes "PKN.WA" -> should normalize to pkn.pl
    svc.update_thesis(_portfolio(), _status(), ticker="PKN.WA")
    assert data.calls[0]["ticker"] == "pkn.pl"


def test_update_thesis_unknown_ticker_raises() -> None:
    svc = _service(FakeLLM(), FakeData())
    with pytest.raises(ValueError, match="not in the portfolio"):
        svc.update_thesis(_portfolio(), _status(), ticker="xyz.pl")
