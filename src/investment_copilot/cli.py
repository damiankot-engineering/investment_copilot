"""Command-line interface for Investment Copilot.

The CLI is a thin adapter over :class:`~investment_copilot.orchestrator.Orchestrator`.
Every command:

1. Loads config + portfolio (validated by Pydantic).
2. Builds the :class:`~investment_copilot.services.ServiceContainer`.
3. Calls a single orchestrator pipeline.
4. Renders the typed result via :mod:`rich`.

Exit codes
----------
* 0 — success
* 1 — user error (bad config, missing portfolio, invalid input)
* 2 — infrastructure failure (provider down, LLM unreachable, write error)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from investment_copilot import __version__
from investment_copilot.config import AppConfig, ConfigError, load_config
from investment_copilot.domain.backtest import BacktestError, BacktestResult
from investment_copilot.domain.portfolio import Portfolio, PortfolioStatus
from investment_copilot.domain.prompts import PortfolioAnalysis, RiskAlerts
from investment_copilot.domain.strategies import KNOWN_STRATEGIES
from investment_copilot.infrastructure.llm import LLMError
from investment_copilot.infrastructure.logging import configure_logging
from investment_copilot.orchestrator import Orchestrator
from investment_copilot.services import (
    PortfolioError,
    RefreshReport,
    ServiceContainer,
    build_container,
    load_portfolio,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="invcopilot",
    help="Investment Copilot — long-term GPW portfolio assistant.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)


# --- Global state via callback ---------------------------------------------


_GlobalCtx = dict[str, object]


def _get_ctx(ctx: typer.Context) -> _GlobalCtx:
    if ctx.obj is None:
        ctx.obj = {}
    return ctx.obj  # type: ignore[return-value]


@app.callback()
def main(
    ctx: typer.Context,
    config_path: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        "-c",
        help="Path to config.yaml.",
        envvar="COPILOT_CONFIG",
    ),
    portfolio_path: Optional[Path] = typer.Option(
        None,
        "--portfolio",
        "-p",
        help="Path to portfolio.yaml (overrides the value in config.yaml).",
        envvar="COPILOT_PORTFOLIO",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Override config.logging.level. One of DEBUG, INFO, WARNING, ERROR.",
    ),
) -> None:
    """Global options shared by every command."""
    state = _get_ctx(ctx)
    state["config_path"] = config_path
    state["portfolio_override"] = portfolio_path
    state["log_level_override"] = log_level


# --- Helpers --------------------------------------------------------------


def _console() -> Console:
    return Console()


def _err_console() -> Console:
    return Console(stderr=True)


def _die(message: str, *, code: int = 1) -> None:
    """Print an error to stderr and exit."""
    _err_console().print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(code=code)


def _bootstrap(ctx: typer.Context) -> tuple[AppConfig, Portfolio, ServiceContainer]:
    """Load config + portfolio, configure logging, build the container."""
    state = _get_ctx(ctx)
    config_path: Path = state["config_path"]  # type: ignore[assignment]

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        _die(str(exc), code=1)

    log_level = state.get("log_level_override") or cfg.logging.level
    configure_logging(str(log_level))

    portfolio_override = state.get("portfolio_override")
    portfolio_path = Path(portfolio_override) if portfolio_override else cfg.portfolio.path

    try:
        portfolio = load_portfolio(portfolio_path)
    except PortfolioError as exc:
        _die(str(exc), code=1)

    container = build_container(cfg)
    return cfg, portfolio, container


# --- Commands -------------------------------------------------------------


@app.command("update-data")
def update_data(
    ctx: typer.Context,
    news_days_back: int = typer.Option(
        14, "--news-days-back", "-n", min=0, help="How far back to fetch news (days)."
    ),
) -> None:
    """Refresh OHLCV (per holding + benchmark) and news caches."""
    cfg, portfolio, container = _bootstrap(ctx)
    orch = Orchestrator(container)

    console = _console()
    console.print(
        f"[bold]Refreshing data[/bold] for {len(portfolio.holdings)} holding(s) "
        f"+ benchmark [cyan]{cfg.backtest.benchmark}[/cyan] …"
    )
    try:
        report = orch.update_data(portfolio, news_days_back=news_days_back)
    except Exception as exc:  # provider-level catastrophic failure
        logger.exception("update-data failed")
        _die(f"data refresh aborted: {exc}", code=2)

    _render_refresh_report(console, report)


@app.command("run-analysis")
def run_analysis(
    ctx: typer.Context,
    no_risks: bool = typer.Option(
        False, "--no-risks", help="Skip the risk-alerts call."
    ),
    news_days_back: int = typer.Option(14, "--news-days-back", "-n", min=0),
) -> None:
    """Compute current PnL and run AI analysis (and optional risks)."""
    _, portfolio, container = _bootstrap(ctx)
    orch = Orchestrator(container)

    console = _console()
    bundle = orch.run_analysis(
        portfolio,
        include_risks=not no_risks,
        news_days_back=news_days_back,
    )

    _render_status(console, bundle.status, portfolio)
    if bundle.analysis is not None:
        _render_analysis(console, bundle.analysis)
    if bundle.risks is not None:
        _render_risks(console, bundle.risks)
    _render_warnings(console, bundle.warnings)


@app.command("backtest")
def backtest(
    ctx: typer.Context,
    strategy: str = typer.Option(
        "ma_crossover",
        "--strategy",
        "-s",
        help=f"Strategy name. One of: {', '.join(KNOWN_STRATEGIES)}.",
    ),
    no_benchmark: bool = typer.Option(
        False, "--no-benchmark", help="Skip benchmark comparison."
    ),
) -> None:
    """Run a strategy backtest over the portfolio."""
    if strategy not in KNOWN_STRATEGIES:
        _die(
            f"unknown strategy: {strategy!r}. Known: {', '.join(KNOWN_STRATEGIES)}",
            code=1,
        )

    _, portfolio, container = _bootstrap(ctx)
    orch = Orchestrator(container)

    console = _console()
    try:
        result = orch.backtest(
            portfolio,
            strategy_name=strategy,
            include_benchmark=not no_benchmark,
        )
    except BacktestError as exc:
        _die(str(exc), code=1)

    _render_backtest(console, result)


@app.command("generate-report")
def generate_report(
    ctx: typer.Context,
    strategy: Optional[str] = typer.Option(
        "ma_crossover",
        "--strategy",
        "-s",
        help="Strategy to backtest within the report. Pass empty to skip.",
    ),
    news_days_back: int = typer.Option(14, "--news-days-back", "-n", min=0),
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        "-o",
        help="Override the auto-generated report filename.",
    ),
) -> None:
    """Run the full pipeline and write a Markdown report."""
    if strategy is not None and strategy != "" and strategy not in KNOWN_STRATEGIES:
        _die(
            f"unknown strategy: {strategy!r}. Known: {', '.join(KNOWN_STRATEGIES)}",
            code=1,
        )

    _, portfolio, container = _bootstrap(ctx)
    orch = Orchestrator(container)

    console = _console()
    try:
        full = orch.generate_report(
            portfolio,
            strategy_name=strategy or None,
            news_days_back=news_days_back,
            filename=filename,
        )
    except (LLMError, OSError) as exc:
        _die(f"report generation failed: {exc}", code=2)

    _render_status(console, full.status, portfolio)
    if full.backtest is not None:
        _render_backtest(console, full.backtest)
    if full.analysis is not None:
        _render_analysis(console, full.analysis)
    if full.risks is not None:
        _render_risks(console, full.risks)
    _render_warnings(console, full.warnings)
    console.print()
    console.print(
        Panel.fit(
            Text(str(full.report_path), style="bold green"),
            title="Report written",
            border_style="green",
        )
    )


@app.command("version")
def version() -> None:
    """Print the installed version and exit."""
    typer.echo(f"investment-copilot {__version__}")


@app.command("init")
def init(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory to create config files in. Defaults to current directory.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing files.",
    ),
) -> None:
    """Create starter ``config.yaml``, ``portfolio.yaml``, and ``.env`` files.

    All files are written as UTF-8 *without* a BOM, so they load cleanly
    on Windows even when the user later edits them in Notepad.
    """
    target = directory.resolve()
    target.mkdir(parents=True, exist_ok=True)

    files = {
        "config.yaml": _STARTER_CONFIG,
        "portfolio.yaml": _STARTER_PORTFOLIO,
        ".env": _STARTER_ENV,
        ".gitignore": _STARTER_GITIGNORE,
    }

    console = _console()
    created: list[str] = []
    skipped: list[str] = []

    for name, content in files.items():
        path = target / name
        if path.exists() and not force:
            skipped.append(name)
            continue
        # newline="" + utf-8 (no BOM) is critical: this is the *fix* for the
        # Notepad UTF-16 problem.
        path.write_text(content, encoding="utf-8", newline="\n")
        created.append(name)

    if created:
        console.print(
            f"[green]Created in {target}:[/green] " + ", ".join(created)
        )
    if skipped:
        console.print(
            f"[yellow]Skipped (already exist, use --force to overwrite):[/yellow] "
            + ", ".join(skipped)
        )

    console.print()
    console.print(
        Panel.fit(
            "[bold]Next steps:[/bold]\n"
            "1. Edit [cyan].env[/cyan] and set your [bold]GROQ_API_KEY[/bold]\n"
            "2. Edit [cyan]portfolio.yaml[/cyan] with your real holdings\n"
            "3. Run [cyan]invcopilot update-data[/cyan] to fetch market data\n"
            "4. Run [cyan]invcopilot run-analysis[/cyan] to see the dashboard",
            title="Setup",
            border_style="cyan",
        )
    )


# --- Starter file contents (kept as Python strings to guarantee UTF-8 encoding) -


_STARTER_CONFIG = """\
# Investment Copilot — configuration.
# This file is UTF-8 (no BOM). When editing, save as UTF-8 (not UTF-16).

providers:
  market_data: stooq            # only "stooq" supported in v1
  news: [stooq, rss]
  fundamentals: none
  rss_feeds:
    - https://www.bankier.pl/rss/wiadomosci.xml
    - https://www.money.pl/rss/

storage:
  sqlite_path: data/cache.db
  parquet_dir: data/ohlcv

portfolio:
  path: portfolio.yaml

strategies:
  ma_crossover:
    fast: 50
    slow: 200
  momentum:
    lookback: 126
    threshold: 0.0

backtest:
  start_date: 2020-01-01
  benchmark: wig20
  initial_capital: 100000
  trading_days_per_year: 252

llm:
  provider: groq
  api_key: ${GROQ_API_KEY}
  model_analysis: llama-3.3-70b-versatile
  model_summary: llama-3.1-8b-instant
  language: pl
  temperature: 0.3
  max_tokens: 2048
  request_timeout_s: 60

logging:
  level: INFO
"""


_STARTER_PORTFOLIO = """\
# Investment Copilot — your portfolio.
# This file is UTF-8 (no BOM). When editing, save as UTF-8 (not UTF-16).
#
# Tickers use Stooq convention (lowercase + ".pl"). Variants like
# "PKN" or "PKN.WA" are normalized automatically.

base_currency: PLN

holdings:
  - ticker: pkn.pl
    name: PKN Orlen
    shares: 100
    entry_price: 65.40
    entry_date: 2023-04-12
    keywords: [Orlen, PKN]
    thesis: |
      Replace this with your real investment thesis.

  # Add more holdings below, or remove the example above and start from scratch.
"""


_STARTER_ENV = """\
# Investment Copilot — secrets.
# This file is UTF-8 (no BOM). DO NOT commit this file to git.

GROQ_API_KEY=your-groq-key-here

# Optional providers
# NEWSAPI_KEY=
# ALPHA_VANTAGE_KEY=
"""


_STARTER_GITIGNORE = """\
# Investment Copilot — gitignore.
.env
config.yaml
portfolio.yaml
data/
reports/
"""


# --- Renderers (Rich) -----------------------------------------------------


def _render_refresh_report(console: Console, r: RefreshReport) -> None:
    table = Table(title="Data refresh", show_header=True, header_style="bold cyan")
    table.add_column("What")
    table.add_column("Result", justify="right")
    table.add_row("OHLCV updated", str(len(r.ohlcv_updated)))
    table.add_row("OHLCV failed", str(len(r.ohlcv_failed)))
    table.add_row(
        "Benchmark",
        f"{r.benchmark_symbol or '—'} ({r.benchmark_rows} rows)",
    )
    table.add_row("News inserted", str(r.news_inserted))
    console.print(table)

    if r.ohlcv_updated:
        sub = Table(show_header=True, header_style="dim", title="OHLCV per ticker")
        sub.add_column("Ticker")
        sub.add_column("Rows", justify="right")
        for ticker, rows in sorted(r.ohlcv_updated.items()):
            sub.add_row(ticker, f"{rows:,}")
        console.print(sub)

    if r.news_failed:
        for line in r.news_failed:
            console.print(f"[yellow]warning:[/yellow] {line}")


def _render_status(
    console: Console,
    status: PortfolioStatus,
    portfolio: Portfolio,
) -> None:
    table = Table(
        title=f"Portfolio status ({status.as_of:%Y-%m-%d %H:%M UTC})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Ticker")
    table.add_column("Name")
    table.add_column("Shares", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Last", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("PnL%", justify="right")

    by_ticker = {h.ticker: h for h in portfolio.holdings}
    for s in status.holdings:
        h = by_ticker.get(s.ticker)
        name = (h.name if h and h.name else "—")
        if s.has_price:
            pnl_color = "green" if (s.unrealized_pnl or 0) >= 0 else "red"
            table.add_row(
                s.ticker,
                name,
                f"{s.shares:g}",
                f"{s.entry_price:.2f}",
                f"{s.last_price:.2f}",
                f"{s.market_value:,.2f}",
                f"[{pnl_color}]{s.unrealized_pnl:+,.2f}[/{pnl_color}]",
                f"[{pnl_color}]{s.unrealized_pnl_pct * 100:+.2f}%[/{pnl_color}]",
            )
        else:
            table.add_row(
                s.ticker, name, f"{s.shares:g}",
                f"{s.entry_price:.2f}", "—", "—", "—", "—",
            )
    console.print(table)

    pnl_color = "green" if status.total_unrealized_pnl >= 0 else "red"
    summary = (
        f"Market value: [bold]{status.total_market_value:,.2f} {status.base_currency}[/bold]   "
        f"PnL: [{pnl_color}]{status.total_unrealized_pnl:+,.2f}[/{pnl_color}] "
        f"([{pnl_color}]{status.total_unrealized_pnl_pct * 100:+.2f}%[/{pnl_color}])"
    )
    if status.missing_data:
        summary += f"\n[yellow]missing data:[/yellow] {', '.join(status.missing_data)}"
    console.print(Panel(summary, title="Totals", border_style="cyan"))


def _render_backtest(console: Console, r: BacktestResult) -> None:
    header = (
        f"strategy [cyan]{r.strategy_name}[/cyan] {r.strategy_params}\n"
        f"window {r.start_date} → {r.end_date}\n"
        f"capital {r.initial_capital:,.0f} → [bold]{r.final_value:,.2f}[/bold]"
    )
    console.print(Panel(header, title="Backtest", border_style="cyan"))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Strategy", justify="right")
    if r.benchmark_metrics is not None:
        table.add_column(f"Benchmark ({r.benchmark_symbol})", justify="right")

    rows: list[tuple[str, float, float | None, bool]] = [
        ("Total return", r.metrics.total_return,
            getattr(r.benchmark_metrics, "total_return", None), True),
        ("Annualized return", r.metrics.annualized_return,
            getattr(r.benchmark_metrics, "annualized_return", None), True),
        ("Annualized volatility", r.metrics.annualized_volatility,
            getattr(r.benchmark_metrics, "annualized_volatility", None), True),
        ("Sharpe", r.metrics.sharpe_ratio,
            getattr(r.benchmark_metrics, "sharpe_ratio", None), False),
        ("Max drawdown", r.metrics.max_drawdown,
            getattr(r.benchmark_metrics, "max_drawdown", None), True),
        ("Win rate", r.metrics.win_rate,
            getattr(r.benchmark_metrics, "win_rate", None), True),
    ]
    for label, strat_val, bench_val, pct in rows:
        s = f"{strat_val * 100:+.2f}%" if pct else f"{strat_val:.2f}"
        if r.benchmark_metrics is not None:
            b = f"{bench_val * 100:+.2f}%" if pct else f"{bench_val:.2f}"
            table.add_row(label, s, b)
        else:
            table.add_row(label, s)
    console.print(table)

    if r.missing_tickers:
        console.print(
            f"[yellow]missing in cache:[/yellow] {', '.join(r.missing_tickers)}"
        )


def _render_analysis(console: Console, a: PortfolioAnalysis) -> None:
    body = Text()
    body.append(a.summary + "\n\n")
    body.append("Komentarze:\n", style="bold")
    for c in a.holdings_comments:
        body.append(f"  • {c.ticker} ({c.recommendation}): ", style="cyan")
        body.append(c.comment + "\n")
    body.append("\nDywersyfikacja:\n", style="bold")
    body.append(a.diversification_notes + "\n")
    body.append(f"\nPewność: {a.confidence}/10", style="dim")
    console.print(Panel(body, title="Analiza (AI)", border_style="magenta"))


_SEVERITY_STYLE = {
    "wysokie": ("red", "🔴"),
    "średnie": ("yellow", "🟡"),
    "niskie": ("green", "🟢"),
}


def _render_risks(console: Console, r: RiskAlerts) -> None:
    if not r.alerts:
        console.print(
            Panel(r.overview + "\n\n[dim](brak istotnych ryzyk)[/dim]",
                  title="Ryzyka (AI)", border_style="yellow")
        )
        return

    body = Text()
    body.append(r.overview + "\n", style="dim")
    console.print(Panel(body, title="Ryzyka (AI)", border_style="yellow"))

    for alert in r.alerts:
        color, marker = _SEVERITY_STYLE.get(alert.severity, ("white", "•"))
        target = alert.ticker or "(portfelowe)"
        body = Text()
        body.append(f"{alert.description}\n\n")
        body.append("Działanie: ", style="bold")
        body.append(alert.suggested_action)
        console.print(
            Panel(
                body,
                title=f"{marker} {alert.title}  [dim]({target}, {alert.severity})[/dim]",
                border_style=color,
            )
        )


def _render_warnings(console: Console, warnings: list[str]) -> None:
    if not warnings:
        return
    body = Text()
    for w in warnings:
        body.append(f"• {w}\n", style="yellow")
    console.print(Panel(body, title="Warnings", border_style="yellow"))


# --- Module entry --------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    app()
