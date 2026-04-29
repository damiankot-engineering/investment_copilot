"""Tests for the Typer CLI.

These tests exercise command parsing, exit codes, and rendering glue. They
patch the orchestrator wherever possible so no real services run.
"""

from __future__ import annotations

import textwrap
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from investment_copilot.cli import app
from investment_copilot.domain.backtest import (
    BacktestError,
    BacktestResult,
    EquityPoint,
    StrategyMetrics,
)
from investment_copilot.domain.portfolio import HoldingStatus, PortfolioStatus
from investment_copilot.domain.prompts import (
    HoldingComment,
    PortfolioAnalysis,
    RiskAlert,
    RiskAlerts,
)
from investment_copilot.services.data_service import RefreshReport
from investment_copilot.services.pipeline_results import AnalysisBundle, FullReport


runner = CliRunner()


# --- Fixtures: minimal config + portfolio on disk -------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    cfg_path = tmp_path / "config.yaml"
    portfolio_path = tmp_path / "portfolio.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            storage:
              sqlite_path: {tmp_path / "cache.db"}
              parquet_dir: {tmp_path / "ohlcv"}
            portfolio:
              path: {portfolio_path}
            backtest:
              start_date: 2023-01-02
              initial_capital: 10000
            llm:
              api_key: ${{GROQ_API_KEY}}
            """
        ),
        encoding="utf-8",
    )
    portfolio_path.write_text(
        textwrap.dedent(
            """
            holdings:
              - ticker: PKN
                shares: 10
                entry_price: 65
                entry_date: 2023-01-02
                thesis: ok
            """
        ),
        encoding="utf-8",
    )
    return {"config": cfg_path, "portfolio": portfolio_path, "tmp": tmp_path}


# --- Fixture results ------------------------------------------------------


def _refresh_report() -> RefreshReport:
    return RefreshReport(
        ohlcv_updated={"pkn.pl": 100},
        ohlcv_failed={},
        benchmark_symbol="^wig20",
        benchmark_rows=200,
        news_inserted=5,
    )


def _status() -> PortfolioStatus:
    h = HoldingStatus(
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
    )
    return PortfolioStatus(
        base_currency="PLN",
        as_of=datetime(2024, 4, 1, tzinfo=timezone.utc),
        holdings=[h],
        total_cost_basis=650,
        priced_cost_basis=650,
        total_market_value=700,
        total_unrealized_pnl=50,
        total_unrealized_pnl_pct=50 / 650,
    )


def _backtest_result() -> BacktestResult:
    metrics = StrategyMetrics(
        total_return=0.25,
        annualized_return=0.12,
        annualized_volatility=0.18,
        sharpe_ratio=0.67,
        max_drawdown=-0.15,
        max_drawdown_duration_days=42,
        win_rate=0.55,
        n_observations=300,
    )
    return BacktestResult(
        strategy_name="ma_crossover",
        strategy_params={"fast": 50, "slow": 200},
        start_date=date(2023, 1, 2),
        end_date=date(2024, 4, 1),
        initial_capital=10_000.0,
        final_value=12_500.0,
        equity_curve=[EquityPoint(date=date(2023, 1, 2), value=10_000.0)],
        metrics=metrics,
        benchmark_symbol="^wig20",
        benchmark_metrics=metrics,
        generated_at=datetime(2024, 4, 1, tzinfo=timezone.utc),
    )


def _analysis() -> PortfolioAnalysis:
    return PortfolioAnalysis(
        summary="ok",
        holdings_comments=[
            HoldingComment(ticker="pkn.pl", comment="ok", recommendation="trzymaj")
        ],
        diversification_notes="ok",
        confidence=7,
    )


def _risks() -> RiskAlerts:
    return RiskAlerts(
        overview="ok",
        alerts=[
            RiskAlert(
                ticker="pkn.pl",
                severity="wysokie",
                title="ryzyko",
                description="opis",
                suggested_action="działaj",
            )
        ],
    )


# --- Smoke / version -------------------------------------------------------


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "investment-copilot" in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # Typer prints help and exits 0 with no_args_is_help=True
    assert result.exit_code in (0, 2)
    assert "Investment Copilot" in result.stdout


# --- Bad config / missing portfolio ---------------------------------------


def test_missing_config_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["--config", str(tmp_path / "missing.yaml"), "version"]
    )
    # `version` does not need config, so it succeeds.
    assert result.exit_code == 0


def test_missing_config_for_real_command_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["--config", str(tmp_path / "missing.yaml"), "update-data"]
    )
    assert result.exit_code == 1
    assert "error" in result.stderr.lower()


def test_invalid_config_exits_1(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "x")
    bad = tmp_path / "config.yaml"
    bad.write_text("not_a_section: oops\nllm:\n  api_key: ${GROQ_API_KEY}\n")
    result = runner.invoke(app, ["--config", str(bad), "update-data"])
    assert result.exit_code == 1


def test_missing_portfolio_exits_1(cli_env, monkeypatch) -> None:
    cli_env["portfolio"].unlink()
    result = runner.invoke(app, ["--config", str(cli_env["config"]), "update-data"])
    assert result.exit_code == 1


# --- update-data ---------------------------------------------------------


def test_update_data_happy_path(cli_env) -> None:
    with patch(
        "investment_copilot.cli.Orchestrator.update_data",
        return_value=_refresh_report(),
    ):
        result = runner.invoke(
            app, ["--config", str(cli_env["config"]), "update-data"]
        )
    assert result.exit_code == 0
    out = result.stdout
    assert "Data refresh" in out
    assert "OHLCV updated" in out
    assert "^wig20" in out
    assert "5" in out  # news inserted


def test_update_data_provider_failure_exits_2(cli_env) -> None:
    with patch(
        "investment_copilot.cli.Orchestrator.update_data",
        side_effect=RuntimeError("provider down"),
    ):
        result = runner.invoke(
            app, ["--config", str(cli_env["config"]), "update-data"]
        )
    assert result.exit_code == 2
    assert "provider down" in result.stderr


# --- run-analysis --------------------------------------------------------


def test_run_analysis_happy_path(cli_env) -> None:
    bundle = AnalysisBundle(
        status=_status(),
        analysis=_analysis(),
        risks=_risks(),
        warnings=[],
        generated_at=datetime.now(timezone.utc),
    )
    with patch(
        "investment_copilot.cli.Orchestrator.run_analysis", return_value=bundle
    ):
        result = runner.invoke(
            app, ["--config", str(cli_env["config"]), "run-analysis"]
        )
    assert result.exit_code == 0
    assert "Portfolio status" in result.stdout
    assert "Analiza (AI)" in result.stdout
    assert "Ryzyka (AI)" in result.stdout


def test_run_analysis_no_risks_flag(cli_env) -> None:
    bundle = AnalysisBundle(
        status=_status(),
        analysis=_analysis(),
        risks=None,
        warnings=[],
        generated_at=datetime.now(timezone.utc),
    )
    with patch(
        "investment_copilot.cli.Orchestrator.run_analysis", return_value=bundle
    ) as run:
        result = runner.invoke(
            app,
            ["--config", str(cli_env["config"]), "run-analysis", "--no-risks"],
        )
    assert result.exit_code == 0
    # Verify --no-risks reached the orchestrator
    _, kwargs = run.call_args
    assert kwargs["include_risks"] is False
    assert "Ryzyka (AI)" not in result.stdout


def test_run_analysis_renders_warnings(cli_env) -> None:
    bundle = AnalysisBundle(
        status=_status(),
        analysis=None,
        risks=None,
        warnings=["LLM down"],
        generated_at=datetime.now(timezone.utc),
    )
    with patch(
        "investment_copilot.cli.Orchestrator.run_analysis", return_value=bundle
    ):
        result = runner.invoke(
            app, ["--config", str(cli_env["config"]), "run-analysis"]
        )
    assert result.exit_code == 0
    assert "LLM down" in result.stdout
    assert "Warnings" in result.stdout


# --- backtest -----------------------------------------------------------


def test_backtest_happy_path(cli_env) -> None:
    with patch(
        "investment_copilot.cli.Orchestrator.backtest", return_value=_backtest_result()
    ) as bt:
        result = runner.invoke(
            app,
            ["--config", str(cli_env["config"]), "backtest", "-s", "ma_crossover"],
        )
    assert result.exit_code == 0
    assert "ma_crossover" in result.stdout
    assert "Sharpe" in result.stdout
    assert "Benchmark" in result.stdout
    bt.assert_called_once()
    _, kwargs = bt.call_args
    assert kwargs["strategy_name"] == "ma_crossover"
    assert kwargs["include_benchmark"] is True


def test_backtest_unknown_strategy_exits_1(cli_env) -> None:
    result = runner.invoke(
        app,
        ["--config", str(cli_env["config"]), "backtest", "-s", "nope"],
    )
    assert result.exit_code == 1
    assert "unknown strategy" in result.stderr


def test_backtest_no_benchmark_flag(cli_env) -> None:
    res = _backtest_result()
    res = res.model_copy(update={"benchmark_metrics": None, "benchmark_symbol": None})
    with patch(
        "investment_copilot.cli.Orchestrator.backtest", return_value=res
    ) as bt:
        result = runner.invoke(
            app,
            [
                "--config", str(cli_env["config"]),
                "backtest", "-s", "momentum", "--no-benchmark",
            ],
        )
    assert result.exit_code == 0
    _, kwargs = bt.call_args
    assert kwargs["include_benchmark"] is False
    assert "Benchmark" not in result.stdout


def test_backtest_engine_error_exits_1(cli_env) -> None:
    with patch(
        "investment_copilot.cli.Orchestrator.backtest",
        side_effect=BacktestError("no data in cache"),
    ):
        result = runner.invoke(
            app,
            ["--config", str(cli_env["config"]), "backtest", "-s", "ma_crossover"],
        )
    assert result.exit_code == 1
    assert "no data in cache" in result.stderr


# --- generate-report ----------------------------------------------------


def test_generate_report_happy_path(cli_env, tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "out.md"
    full = FullReport(
        status=_status(),
        backtest=_backtest_result(),
        analysis=_analysis(),
        risks=_risks(),
        report_path=report_path,
        warnings=[],
        generated_at=datetime.now(timezone.utc),
    )
    with patch(
        "investment_copilot.cli.Orchestrator.generate_report", return_value=full
    ) as gr:
        result = runner.invoke(
            app,
            ["--config", str(cli_env["config"]), "generate-report"],
        )
    assert result.exit_code == 0
    assert "Report written" in result.stdout
    assert str(report_path) in result.stdout
    _, kwargs = gr.call_args
    assert kwargs["strategy_name"] == "ma_crossover"


def test_generate_report_skips_strategy_when_empty(cli_env) -> None:
    full = FullReport(
        status=_status(),
        backtest=None,
        analysis=_analysis(),
        risks=None,
        report_path=Path("/tmp/x.md"),
        warnings=[],
        generated_at=datetime.now(timezone.utc),
    )
    with patch(
        "investment_copilot.cli.Orchestrator.generate_report", return_value=full
    ) as gr:
        result = runner.invoke(
            app,
            ["--config", str(cli_env["config"]), "generate-report", "--strategy", ""],
        )
    assert result.exit_code == 0
    _, kwargs = gr.call_args
    assert kwargs["strategy_name"] is None


def test_portfolio_override_flag(cli_env, tmp_path: Path) -> None:
    alt = tmp_path / "alt_portfolio.yaml"
    alt.write_text(
        textwrap.dedent(
            """
            holdings:
              - ticker: CDR
                shares: 5
                entry_price: 100
                entry_date: 2023-01-02
                thesis: alt
            """
        )
    )
    captured: dict = {}

    def _fake_update(self, portfolio, **_):
        captured["tickers"] = portfolio.tickers
        return _refresh_report()

    with patch(
        "investment_copilot.cli.Orchestrator.update_data", new=_fake_update
    ):
        result = runner.invoke(
            app,
            [
                "--config", str(cli_env["config"]),
                "--portfolio", str(alt),
                "update-data",
            ],
        )
    assert result.exit_code == 0
    assert captured["tickers"] == ["cdr.pl"]


def test_unknown_strategy_in_generate_report_exits_1(cli_env) -> None:
    result = runner.invoke(
        app,
        [
            "--config", str(cli_env["config"]),
            "generate-report", "--strategy", "wat",
        ],
    )
    assert result.exit_code == 1
    assert "unknown strategy" in result.stderr


# --- init command --------------------------------------------------------


def test_init_creates_files_in_target_directory(tmp_path: Path) -> None:
    target = tmp_path / "fresh"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0

    expected = {"config.yaml", "portfolio.yaml", ".env", ".gitignore"}
    assert {p.name for p in target.iterdir()} == expected

    # Must be valid UTF-8 (no BOM) — the whole point of `init`
    for name in expected:
        raw = (target / name).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{name} has UTF-8 BOM"
        assert not raw.startswith(b"\xff\xfe"), f"{name} is UTF-16 LE"
        assert not raw.startswith(b"\xfe\xff"), f"{name} is UTF-16 BE"
        # Roundtrips cleanly through utf-8
        text = raw.decode("utf-8")
        assert text  # non-empty


def test_init_skips_existing_without_force(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("# existing", encoding="utf-8")
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    # Existing file untouched
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == "# existing"
    # Other files were created
    assert (tmp_path / "portfolio.yaml").exists()
    assert (tmp_path / ".env").exists()


def test_init_force_overwrites(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("# existing", encoding="utf-8")
    result = runner.invoke(app, ["init", str(tmp_path), "--force"])
    assert result.exit_code == 0
    new_content = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert new_content != "# existing"
    assert "Investment Copilot" in new_content


def test_init_creates_loadable_config(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: `init` followed by `load_config` must succeed."""
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0

    from investment_copilot.config import load_config

    cfg = load_config(tmp_path / "config.yaml", env_file=None)
    assert cfg.llm.api_key == "sk-test"
    assert cfg.providers.market_data == "stooq"
