# Investment Copilot

A long-term investing companion for the **Polish stock market (GPW)**. Tracks a manually defined portfolio, fetches free market data, runs backtests, and produces structured Polish-language analyses powered by [Groq](https://groq.com).

> Investment Copilot is a **decision-support tool**, not a trading bot. Its outputs are research material, never financial advice.

---

## Overview

Investment Copilot helps an intermediate-level GPW investor maintain a disciplined process: track holdings against their original investment thesis, refresh market data and news on demand, run reproducible backtests, and ask an LLM to produce structured Polish summaries, risk alerts, and thesis assessments.

Designed to be:

- **Modular** — ports-and-adapters layout, every external dependency lives behind a `Protocol` and is swappable in one file.
- **Polish-market-first** — Stooq for OHLCV (full GPW history, free API key required), RSS feeds (Bankier, Money.pl) for news, WIG20 as the default benchmark.
- **API-ready** — every service takes Pydantic inputs and returns Pydantic outputs. The CLI is a thin adapter; a FastAPI app would be another.

---

## Features

- **Portfolio tracking** — define holdings in `portfolio.yaml` with shares, entry price, entry date, investment thesis, optional name and news keywords.
- **Free, fast data** — daily OHLCV from Stooq for any GPW ticker (free API key required, see [API keys](#api-keys)). Indices (`wig20`, `mwig40`, etc.) and equities are queried through the same adapter.
- **News aggregation** — multiple Polish RSS feeds plus best-effort scraping of Stooq's per-symbol news block. Persisted in SQLite, deduplicated by URL, queryable per ticker.
- **Backtesting** — portfolio-level simulator with two strategies (MA-crossover, time-series momentum). Equal-weight across active sleeves, daily checks, end-of-day fills, no leverage, no shorts, zero costs (v1). Equity curve and a full metrics suite (total / annualized return, volatility, Sharpe @ 252, max drawdown with duration, win rate). WIG20 buy-and-hold benchmark in the same window.
- **Groq copilot** — three structured analyses with Polish output: portfolio summary, risk alerts, thesis update. JSON-mode + Pydantic validation, automatic self-correction on schema violations, exponential backoff on transient errors, fast-fail on auth errors.
- **CLI** — Typer-based commands (`update-data`, `run-analysis`, `backtest`, `generate-report`) with Rich-rendered tables, severity-coloured risk panels, sensible exit codes, and graceful degradation when the LLM is unreachable.
- **Markdown reports** — generated to `reports/`, fully Polish, with portfolio table, backtest metrics vs benchmark, AI sections, and warnings.
- **Local persistence** — SQLite for news + metadata, parquet for OHLCV. Restart-safe, queryable, no ORM overhead.
- **API-first design** — `ServiceContainer` constructs every service from `AppConfig`; the same wiring will power a future FastAPI app via `Depends(get_container)`.

---

## Architecture

### Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  ENTRYPOINTS         CLI (Typer)         [future] FastAPI       │
│                          │                        │             │
│                          ▼                        ▼             │
├─────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION       Orchestrator (pipelines: update→analyze→   │
│                                    backtest→report)             │
├─────────────────────────────────────────────────────────────────┤
│  SERVICES            DataService  PortfolioService              │
│                      BacktestService  CopilotService            │
│                      ReportService                              │
├─────────────────────────────────────────────────────────────────┤
│  DOMAIN              Models (Pydantic)  Backtest engine         │
│                      Strategies  Metrics  Prompt templates      │
├─────────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE      Providers (Stooq, RSS)                     │
│                      LLM clients (Groq)                         │
│                      Storage (SQLite + parquet)                 │
│                      Config loader  Logger                      │
└─────────────────────────────────────────────────────────────────┘
```

Dependency rule: arrows point **downward only**. Domain knows nothing about infrastructure. Services depend on domain + infrastructure interfaces. Orchestration depends on services. Entrypoints depend on the orchestrator.

### Data flow — `generate-report`

```
CLI (Typer)
  └─► Orchestrator.generate_report()
        ├─► PortfolioService.current_status()        → PortfolioStatus
        ├─► BacktestService.run()                    → BacktestResult
        │     ├─► load OHLCV from parquet cache
        │     ├─► Strategy.generate_signals()
        │     ├─► simulate_portfolio()               → equity curve
        │     └─► compute_metrics()                  → Sharpe / DD / …
        ├─► CopilotService.analyze_portfolio()       → PortfolioAnalysis
        │     ├─► load news from SQLite
        │     ├─► build_portfolio_context()          → Markdown blocks
        │     ├─► render Polish prompt template
        │     └─► GroqClient.complete_structured()   → Pydantic
        ├─► CopilotService.detect_risks()            → RiskAlerts
        └─► ReportService.write()                    → reports/*.md
```

Every arrow is a typed call. Every box is independently testable. The orchestrator captures non-fatal errors (LLM timeouts, missing benchmark data, etc.) into a `warnings` list rather than aborting the run — so a partial result is always available.

### Project layout

```
investment-copilot/
├── pyproject.toml
├── config.example.yaml
├── portfolio.example.yaml
├── .env.example
├── data/                            # gitignored (cache + parquet)
├── reports/                         # gitignored (Markdown reports)
├── src/investment_copilot/
│   ├── cli.py                       # Typer app
│   ├── orchestrator.py              # named pipelines
│   ├── config/                      # AppConfig + loader
│   ├── domain/
│   │   ├── models.py                # core types (Ticker, NewsItem, ...)
│   │   ├── portfolio.py             # Holding, Portfolio, status models
│   │   ├── strategies/              # MACrossover, Momentum, factory
│   │   ├── backtest/                # engine, metrics, results
│   │   └── prompts/                 # context builder, schemas, templates
│   ├── infrastructure/
│   │   ├── providers/               # Stooq, RSS, factory
│   │   ├── llm/                     # GroqClient, factory, errors
│   │   ├── storage/                 # SQLite + parquet
│   │   └── logging.py
│   └── services/
│       ├── data_service.py
│       ├── portfolio_service.py
│       ├── backtest_service.py
│       ├── copilot_service.py
│       ├── report_service.py
│       ├── container.py             # ServiceContainer factory
│       └── pipeline_results.py
└── tests/                           # 208 tests
```

---

## Setup

### Prerequisites

- Python **3.11** or 3.12
- [`uv`](https://docs.astral.sh/uv/) (recommended) or plain `pip`
- A [Groq API key](https://console.groq.com/keys) — free tier is sufficient
- A [Stooq API key](https://stooq.pl/q/d/?s=pkn.pl&get_apikey) — free, obtained via one-time captcha

### Install

```bash
git clone <your-fork-url> investment-copilot
cd investment-copilot

# with uv (recommended)
uv sync                              # installs deps + dev extras into .venv
uv pip install -e .                  # editable install of the package

# or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

After install, the `invcopilot` command is available on `PATH`:

```bash
invcopilot version
# investment-copilot 0.1.0
```

### Streamlit GUI (optional)

A local web GUI is available as an optional install. It's a thin layer over the same orchestrator pipelines the CLI uses — pure addition, zero changes to the rest of the codebase.

```bash
# install with the GUI extra
uv pip install -e ".[gui]"

# run (opens http://localhost:8501)
streamlit run streamlit_app.py
```

The GUI gives you four tabs over the standard pipelines:

| Tab | What it does | Equivalent CLI command |
|---|---|---|
| 📊 **Portfolio** | Live PnL table, total metrics, **Update data** button | `update-data` + status |
| 📈 **Backtest** | Strategy picker, equity curve + drawdown charts, metrics vs benchmark | `backtest` |
| ✨ **AI Analysis** | Polish summary + risk alerts (with severity badges) from Groq | `run-analysis` |
| 📄 **Reports** | Generate Markdown reports and browse/download existing ones | `generate-report` |

Runs **fully offline** — it's just a local Python process. The same `config.yaml` and `portfolio.yaml` are used (configurable from the sidebar), and the same `GROQ_API_KEY` env var. State persists within a session, so navigating between tabs doesn't re-run pipelines.

### First-run configuration

The fastest path is the `init` command, which creates `config.yaml`, `portfolio.yaml`, `.env`, and `.gitignore` for you in **UTF-8 without BOM** — which matters on Windows because Notepad otherwise saves as UTF-16 LE and `python-dotenv` won't read it.

```bash
invcopilot init                  # creates files in the current directory
invcopilot init ~/my-portfolio   # or somewhere else
invcopilot init . --force        # overwrite existing files
```

Then edit `.env` to set `GROQ_API_KEY` and `portfolio.yaml` with your real holdings.

If you prefer to copy the example files manually:

```bash
cp config.example.yaml config.yaml
cp portfolio.example.yaml portfolio.yaml
cp .env.example .env

# edit .env and set GROQ_API_KEY and STOOQ_API_KEY
# edit portfolio.yaml with your real holdings
```

Both `config.yaml` and `portfolio.yaml` are gitignored by default so secrets and personal positions never end up in git.

### Encoding (Windows Notepad gotcha)

The loader transparently handles `.env`, `config.yaml`, and `portfolio.yaml` saved in any of: UTF-8, UTF-8 with BOM, UTF-16 LE (Notepad's "Unicode"), UTF-16 BE, and CP1250. Files in non-UTF-8 encodings will load fine but emit a warning suggesting you re-save them.

If you do want to fix it manually:

- **Notepad:** File → Save As → Encoding dropdown → **UTF-8** (not "UTF-8 with BOM" and not "Unicode").
- **VS Code:** click the encoding indicator in the bottom-right status bar → "Save with Encoding" → "UTF-8".
- **PowerShell:** `Get-Content .env | Set-Content -Encoding UTF8 .env` (this rewrites in UTF-8 without BOM in PowerShell 7+; on Windows PowerShell 5.1, use `Out-File` with `-Encoding utf8NoBOM`).

### Network notes

- **Stooq** requires a free API key (see [API keys](#api-keys) below). If you sit behind a strict outbound proxy, allow `stooq.com`.
- **Groq** is reached at `api.groq.com`.
- The Stooq adapter handles HTTP errors as `ProviderError`s — a single ticker failing never aborts the rest of the refresh.

### Run the tests

```bash
uv run pytest -q                     # 208 tests, ~2 seconds
```

---

## API keys

Investment Copilot resolves environment variables inside `config.yaml` using `${VAR}` and `${VAR:-default}` syntax. `.env` is loaded automatically at startup via `python-dotenv`.

| Variable | Required | Used by |
|---|---|---|
| `GROQ_API_KEY` | yes | `llm.api_key` (Groq calls) |
| `STOOQ_API_KEY` | yes | Stooq OHLCV CSV endpoint |
| `NEWSAPI_KEY` | no | optional NewsAPI provider (placeholder; not wired in v1) |
| `ALPHA_VANTAGE_KEY` | no | optional Alpha Vantage fundamentals (placeholder; v2) |

- **Groq key** — <https://console.groq.com/keys>. The free tier is generous enough for routine use.
- **Stooq key** — visit <https://stooq.pl/q/d/?s=pkn.pl&get_apikey>, complete the captcha, and copy the `apikey` value from the generated download URL. The key is free and does not expire.

If a referenced env var is missing **and** has no default, the CLI exits with a clear `error:` line and code `1` — the application never starts in a half-configured state.

---

## Example `config.yaml`

```yaml
# All sections are optional except `llm.api_key`. Defaults match v1 spec.

providers:
  market_data: stooq                  # only "stooq" supported in v1
  news:                               # ordered list; each provider is queried
    - stooq
    - rss
  fundamentals: none                  # "alpha_vantage" or "none"
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
    fast: 50                          # must be < slow
    slow: 200
  momentum:
    lookback: 126                     # ~6 months of trading days
    threshold: 0.0                    # min trailing return to go long

backtest:
  start_date: 2020-01-01
  # end_date: 2025-12-31              # optional; omit for "latest available"
  benchmark: wig20                    # wig20 / mwig40 / swig80 / wig / wig30
  initial_capital: 100000
  trading_days_per_year: 252

llm:
  provider: groq
  api_key: ${GROQ_API_KEY}            # required
  model_analysis: llama-3.3-70b-versatile
  model_summary: llama-3.1-8b-instant
  language: pl                        # output language for AI artifacts
  temperature: 0.3
  max_tokens: 2048
  request_timeout_s: 60

logging:
  level: INFO                         # DEBUG / INFO / WARNING / ERROR
```

Unknown top-level keys are rejected at load time (typo-safe). Cross-field rules are enforced too — e.g., `ma_crossover.slow` must be greater than `fast`.

---

## Example `portfolio.yaml`

```yaml
base_currency: PLN

# Tickers are normalized to Stooq form: "PKN", "PKN.WA", and "pkn.pl" are
# all equivalent and stored as "pkn.pl" internally.
#
# Optional per-holding fields:
#   name      — display name shown in reports.
#   keywords  — substrings used to filter RSS news. Defaults to the ticker
#               stem (e.g. "PKN") which often misses news that uses brand
#               names ("Orlen"). Set keywords explicitly for best matching.
#
# `entry_price` and `entry_date` drive live PnL tracking only. The
# backtester ignores them and runs strategies from `backtest.start_date`
# in config.yaml — otherwise results would be biased by your real entries.

holdings:
  - ticker: pkn.pl
    name: PKN Orlen
    shares: 100
    entry_price: 65.40
    entry_date: 2023-04-12
    keywords: [Orlen, PKN]
    thesis: |
      Integrated energy & refining champion. Dividend policy + Orlen-Lotos
      synergies. Risk: regulated prices, political exposure.

  - ticker: cdr.pl
    name: CD Projekt
    shares: 25
    entry_price: 142.10
    entry_date: 2024-01-08
    keywords: [CD Projekt, CDR, Cyberpunk, Witcher]
    thesis: |
      Long-cycle IP holder (Witcher, Cyberpunk). Pipeline visibility through
      the next flagship release. Risk: execution and release-window slippage.
```

Validation rules: positive shares and entry prices, no future entry dates, non-empty thesis, no duplicate tickers (after normalization), no unknown fields.

---

## CLI usage

```bash
invcopilot --help
```

### Global options

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--config`, `-c` | `COPILOT_CONFIG` | `config.yaml` | Path to config file. |
| `--portfolio`, `-p` | `COPILOT_PORTFOLIO` | _(value in `config.yaml`)_ | Override the portfolio path. Useful for multi-portfolio workflows. |
| `--log-level` | — | _(`logging.level` from config)_ | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

### Commands

#### `invcopilot update-data`

Refresh OHLCV (per holding + benchmark) and news caches.

```bash
invcopilot update-data
invcopilot update-data --news-days-back 30
```

| Flag | Default | Description |
|---|---|---|
| `--news-days-back`, `-n` | `14` | How far back to fetch news. |

#### `invcopilot run-analysis`

Compute current PnL and run AI analysis (and optionally risks).

```bash
invcopilot run-analysis
invcopilot run-analysis --no-risks
```

| Flag | Default | Description |
|---|---|---|
| `--no-risks` | false | Skip the risk-alerts call (faster, half the LLM cost). |
| `--news-days-back`, `-n` | `14` | News window for AI context. |

#### `invcopilot backtest`

Run a strategy backtest over the portfolio.

```bash
invcopilot backtest -s ma_crossover
invcopilot backtest -s momentum --no-benchmark
```

| Flag | Default | Description |
|---|---|---|
| `--strategy`, `-s` | `ma_crossover` | One of `ma_crossover`, `momentum`. |
| `--no-benchmark` | false | Skip WIG20 buy-and-hold benchmark column. |

#### `invcopilot generate-report`

Run the full pipeline (status + backtest + AI analysis + risks) and write a Markdown report to `reports/`.

```bash
invcopilot generate-report
invcopilot generate-report -s momentum -o monthly_review.md
invcopilot generate-report --strategy ""           # skip the backtest section
```

| Flag | Default | Description |
|---|---|---|
| `--strategy`, `-s` | `ma_crossover` | Strategy to backtest within the report. Pass `""` to skip. |
| `--news-days-back`, `-n` | `14` | News window for AI context. |
| `--filename`, `-o` | _(timestamped)_ | Override the auto-generated filename (`report_YYYYMMDD_HHMMSS.md`). |

#### `invcopilot version`

Print the installed version and exit.

#### `invcopilot init`

Create starter `config.yaml`, `portfolio.yaml`, `.env`, and `.gitignore` files. All written as **UTF-8 without BOM** so they load cleanly on Windows even after editing in Notepad.

```bash
invcopilot init                  # current directory
invcopilot init ~/portfolios/gpw # specific directory
invcopilot init . --force        # overwrite existing files
```

| Flag | Default | Description |
|---|---|---|
| `--force`, `-f` | false | Overwrite existing files. |

### Exit codes

| Code | Meaning | Examples |
|---|---|---|
| `0` | success | (also: pipeline succeeded with `warnings`) |
| `1` | user error | bad config, missing portfolio, unknown strategy, no cached data for backtest |
| `2` | infrastructure failure | provider down, LLM unreachable mid-pipeline, write error |

The CLI writes errors to **stderr** and the rendered report to **stdout** so it composes cleanly with shell pipelines and cron.

### Example session

```bash
# 1. First-run config
$ cp config.example.yaml config.yaml
$ cp portfolio.example.yaml portfolio.yaml

# 2. Pull the data
$ invcopilot update-data
Refreshing data for 3 holding(s) + benchmark wig20 …
                Data refresh
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ What           ┃              Result ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ OHLCV updated  │                   3 │
│ OHLCV failed   │                   0 │
│ Benchmark      │  ^wig20 (1620 rows) │
│ News inserted  │                  47 │
└────────────────┴─────────────────────┘

# 3. Look at current PnL with AI commentary
$ invcopilot run-analysis
                  Portfolio status (2026-04-29 14:00 UTC)
┏━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Ticker ┃ Name        ┃ Shares ┃ Entry ┃ Last ┃    Value ┃     PnL ┃   PnL% ┃
┡━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ pkn.pl │ PKN Orlen   │    100 │ 65.40 │ … etc.

# 4. Run a backtest
$ invcopilot backtest -s ma_crossover

# 5. Full Markdown report
$ invcopilot generate-report -o weekly.md
…
╭─ Report written ─╮
│ reports/weekly.md │
╰───────────────────╯

$ cat reports/weekly.md
# Raport portfela
…
```

---

## Future FastAPI notes

The architecture is FastAPI-ready by design. Adding HTTP routes is a one-file change because every service already accepts Pydantic inputs and returns Pydantic outputs.

### What's already in place

- **`ServiceContainer`** is the single wiring point. Both the CLI and a future `Depends(get_container)` will use it.
- **All pipeline outputs are Pydantic** (`AnalysisBundle`, `FullReport`, `BacktestResult`, etc.). They serialize to JSON via `.model_dump_json()` out of the box.
- **No global state.** Services are constructed fresh from config; tests prove they're independently mountable.
- **No `print` in services.** Everything goes through stdlib `logging`; integrating with `uvicorn`'s log config is a single line.

### Sketch of the future API layer

```python
# src/investment_copilot/api/main.py  (NOT in v1)
from fastapi import Depends, FastAPI
from investment_copilot.config import load_config
from investment_copilot.services import ServiceContainer, build_container, load_portfolio
from investment_copilot.orchestrator import Orchestrator
from investment_copilot.services.pipeline_results import AnalysisBundle, FullReport
from investment_copilot.domain.backtest import BacktestResult

app = FastAPI(title="Investment Copilot")

def get_container() -> ServiceContainer:
    return build_container(load_config("config.yaml"))

def get_orchestrator(c: ServiceContainer = Depends(get_container)) -> Orchestrator:
    return Orchestrator(c)

@app.post("/analysis", response_model=AnalysisBundle)
def run_analysis(orch: Orchestrator = Depends(get_orchestrator)):
    portfolio = load_portfolio(orch._container.config.portfolio.path)
    return orch.run_analysis(portfolio)

@app.post("/backtest", response_model=BacktestResult)
def backtest(strategy: str, orch: Orchestrator = Depends(get_orchestrator)):
    portfolio = load_portfolio(orch._container.config.portfolio.path)
    return orch.backtest(portfolio, strategy_name=strategy)
```

### What v1 deliberately doesn't have

- No auth (single-user assumed). Any production API would add an auth dependency.
- No async. The current LLM/HTTP calls are synchronous — for an API exposed to multiple clients, providers should be wrapped with `asyncio.to_thread` or replaced with async equivalents.
- No background scheduler. `update-data` is on-demand; an API deployment would likely want a periodic refresh job (cron, APScheduler, etc.).
- No persistence of analysis history. The copilot is read-only by design in v1.

These are easy to add when the time comes; none of them require restructuring.

---

## License

MIT.

## Disclaimer

Investment Copilot is a personal research tool. The Polish-language outputs it generates — portfolio summaries, risk alerts, thesis updates — are for informational purposes only. They do **not** constitute financial advice and must not be acted upon as such. Always do your own research and, if needed, consult a licensed advisor.
