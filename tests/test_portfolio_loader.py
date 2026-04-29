"""Tests for ``load_portfolio``."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from investment_copilot.services.portfolio_service import (
    PortfolioError,
    load_portfolio,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_load_minimal_portfolio(tmp_path) -> None:
    p = _write(
        tmp_path / "portfolio.yaml",
        """
        base_currency: PLN
        holdings:
          - ticker: PKN
            shares: 100
            entry_price: 65.40
            entry_date: 2023-04-12
            thesis: energy champion
        """,
    )

    portfolio = load_portfolio(p)

    assert portfolio.base_currency == "PLN"
    assert len(portfolio.holdings) == 1
    h = portfolio.holdings[0]
    assert h.ticker == "pkn.pl"  # normalized
    assert h.shares == 100
    assert h.thesis == "energy champion"


def test_load_portfolio_with_optional_fields(tmp_path) -> None:
    p = _write(
        tmp_path / "portfolio.yaml",
        """
        holdings:
          - ticker: cdr.pl
            name: CD Projekt
            shares: 25
            entry_price: 142.10
            entry_date: 2024-01-08
            keywords: [CD Projekt, CDR]
            thesis: long-cycle IP
        """,
    )

    portfolio = load_portfolio(p)
    h = portfolio.holdings[0]
    assert h.name == "CD Projekt"
    assert h.effective_keywords == ["CD Projekt", "CDR"]


def test_load_portfolio_missing_file(tmp_path) -> None:
    with pytest.raises(PortfolioError, match="not found"):
        load_portfolio(tmp_path / "nope.yaml")


def test_load_portfolio_invalid_yaml(tmp_path) -> None:
    p = _write(tmp_path / "p.yaml", "::: not yaml :::\n  - bad")
    with pytest.raises(PortfolioError):
        load_portfolio(p)


def test_load_portfolio_root_must_be_mapping(tmp_path) -> None:
    p = _write(tmp_path / "p.yaml", "- just_a_list\n- of_things")
    with pytest.raises(PortfolioError, match="mapping"):
        load_portfolio(p)


def test_load_portfolio_rejects_unknown_field(tmp_path) -> None:
    p = _write(
        tmp_path / "p.yaml",
        """
        holdings:
          - ticker: PKN
            shares: 10
            entry_price: 65
            entry_date: 2023-01-02
            thesis: ok
            mystery_field: oops
        """,
    )
    with pytest.raises(PortfolioError):
        load_portfolio(p)


def test_load_portfolio_rejects_duplicate_tickers(tmp_path) -> None:
    p = _write(
        tmp_path / "p.yaml",
        """
        holdings:
          - ticker: PKN
            shares: 10
            entry_price: 65
            entry_date: 2023-01-02
            thesis: a
          - ticker: pkn.pl
            shares: 5
            entry_price: 70
            entry_date: 2023-06-01
            thesis: b
        """,
    )
    with pytest.raises(PortfolioError, match="Duplicate"):
        load_portfolio(p)


def test_example_portfolio_loads() -> None:
    project_root = Path(__file__).resolve().parents[1]
    portfolio = load_portfolio(project_root / "portfolio.example.yaml")
    assert {h.ticker for h in portfolio.holdings} == {"pkn.pl", "cdr.pl", "pko.pl"}
    assert all(h.thesis for h in portfolio.holdings)
