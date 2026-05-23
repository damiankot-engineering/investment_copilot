"""FastAPI application for Investment Copilot.

Run locally:

    uv pip install -e ".[api]"
    uv run uvicorn investment_copilot.api.main:app --reload --port 8000

The static frontend at ``src/frontend/`` is mounted at the app root, so
opening ``http://localhost:8000`` loads the dashboard which then calls
``/api/...`` on the same origin (no CORS in production). CORS is allowed
in dev for ``http://localhost:5173`` and ``http://localhost:8000``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from investment_copilot import __version__
from investment_copilot.api import adapters
from investment_copilot.api.deps import (
    get_container,
    get_monitoring_dir,
    get_orchestrator,
    get_portfolio_path,
    get_reports_dir,
)
from investment_copilot.api.schemas import (
    AnalysisBundleDTO,
    AppConfigDTO,
    BacktestResultDTO,
    BenchmarkInfoDTO,
    DataUpdateResult,
    GenerateReportRequest,
    GenerateReportResponse,
    HealthDTO,
    MonitoringSnapshotDTO,
    PortfolioDTO,
    PortfolioStatusDTO,
    ReportContentDTO,
    ReportFileDTO,
    RunMonitoringRequest,
    StrategyInfoDTO,
)
from investment_copilot.domain.models import BENCHMARK_SYMBOLS, resolve_benchmark
from investment_copilot.domain.portfolio import Portfolio
from investment_copilot.domain.strategies import KNOWN_STRATEGIES
from investment_copilot.infrastructure.llm import LLMError
from investment_copilot.infrastructure.logging import configure_logging
from investment_copilot.orchestrator import Orchestrator
from investment_copilot.services import (
    PortfolioError,
    ServiceContainer,
    load_portfolio,
    save_portfolio,
)

logger = logging.getLogger(__name__)

# Resolved once at module-import time (used to mount static frontend).
# `src/investment_copilot/api/main.py` → up 3 → `src/`; then `frontend/`.
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _check_filename(name: str) -> None:
    """Reject anything that could escape the reports directory."""
    if not _SAFE_FILENAME.match(name) or ".." in name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid filename: {name!r}",
        )


def _stat_to_dto(p: Path) -> ReportFileDTO:
    st = p.stat()
    return ReportFileDTO(
        name=p.name,
        mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
        size_bytes=st.st_size,
    )


def create_app() -> FastAPI:
    configure_logging("INFO")
    app = FastAPI(
        title="Investment Copilot API",
        version=__version__,
        description=(
            "HTTP surface over the same ServiceContainer + Orchestrator the "
            "CLI uses. Single-user, no auth."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------- Health

    @app.get("/api/health", response_model=HealthDTO, tags=["meta"])
    async def health() -> HealthDTO:
        return HealthDTO(version=__version__)

    _BENCHMARK_LABELS = {
        "wig": "WIG",
        "wig20": "WIG20",
        "mwig40": "mWIG40",
        "swig80": "sWIG80",
        "wig30": "WIG30",
    }

    @app.get(
        "/api/config",
        response_model=AppConfigDTO,
        tags=["meta"],
    )
    async def get_app_config(
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> AppConfigDTO:
        bt = container.config.backtest
        available = [
            BenchmarkInfoDTO(value=v, label=_BENCHMARK_LABELS.get(v, v.upper()))
            for v in BENCHMARK_SYMBOLS.keys()
        ]
        # If the configured benchmark is a custom ticker (e.g. 'cspx.uk'),
        # surface it as a selectable option too.
        if bt.benchmark not in BENCHMARK_SYMBOLS:
            available.insert(
                0,
                BenchmarkInfoDTO(
                    value=bt.benchmark,
                    label=_BENCHMARK_LABELS.get(bt.benchmark, bt.benchmark.upper()),
                ),
            )
        return AppConfigDTO(
            benchmark=bt.benchmark,
            benchmark_label=_BENCHMARK_LABELS.get(bt.benchmark, bt.benchmark.upper()),
            backtest_start_date=bt.start_date,
            backtest_end_date=bt.end_date,
            available_benchmarks=available,
        )

    @app.get(
        "/api/strategies",
        response_model=list[StrategyInfoDTO],
        tags=["meta"],
    )
    async def strategies() -> list[StrategyInfoDTO]:
        labels = {
            "ma_crossover": "MA Crossover",
            "momentum": "Momentum",
            "buy_and_hold": "Buy & Hold",
        }
        return [
            StrategyInfoDTO(value=v, label=labels.get(v, v))
            for v in KNOWN_STRATEGIES
        ]

    # ------------------------------------------------------------- Portfolio

    @app.get(
        "/api/portfolio",
        response_model=PortfolioDTO,
        tags=["portfolio"],
    )
    async def get_portfolio(
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
    ) -> PortfolioDTO:
        try:
            portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        except PortfolioError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return adapters.portfolio_to_dto(portfolio)

    @app.put(
        "/api/portfolio",
        response_model=PortfolioDTO,
        tags=["portfolio"],
    )
    async def put_portfolio(
        payload: PortfolioDTO,
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> PortfolioDTO:
        # Round-trip through the domain model so all validators run
        # (ticker normalization, no duplicates, no future entry dates).
        raw = payload.model_dump(mode="json")
        for h in raw.get("holdings", []):
            h.pop("display_ticker", None)
        try:
            domain_portfolio = Portfolio.model_validate(raw)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        try:
            await asyncio.to_thread(save_portfolio, domain_portfolio, str(pf_path))
        except PortfolioError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _ = container  # ensure container builds before save (early failure)
        return adapters.portfolio_to_dto(domain_portfolio)

    @app.get(
        "/api/portfolio/status",
        response_model=PortfolioStatusDTO,
        tags=["portfolio"],
    )
    async def portfolio_status(
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> PortfolioStatusDTO:
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        status_obj = await asyncio.to_thread(
            container.portfolio_service.current_status, portfolio
        )
        return adapters.portfolio_status_to_dto(status_obj, portfolio=portfolio)

    # ------------------------------------------------------------------ Data

    @app.post(
        "/api/data/update",
        response_model=DataUpdateResult,
        tags=["data"],
    )
    async def update_data(
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        orch: Annotated[Orchestrator, Depends(get_orchestrator)],
        news_days_back: int = 14,
    ) -> DataUpdateResult:
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        report = await asyncio.to_thread(
            orch.update_data, portfolio, news_days_back=news_days_back
        )
        return DataUpdateResult(
            ohlcv_updated=dict(report.ohlcv_updated),
            ohlcv_failed=dict(report.ohlcv_failed),
            benchmark_symbol=report.benchmark_symbol,
            benchmark_rows=report.benchmark_rows,
            news_inserted=report.news_inserted,
            news_failed=list(report.news_failed),
        )

    # ---------------------------------------------------------------- Backtest

    @app.post(
        "/api/backtest",
        response_model=BacktestResultDTO,
        tags=["backtest"],
    )
    async def run_backtest(
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        orch: Annotated[Orchestrator, Depends(get_orchestrator)],
        strategy: str = "ma_crossover",
        start_date: date | None = None,
        end_date: date | None = None,
        include_benchmark: bool = True,
        benchmark: str | None = None,
    ) -> BacktestResultDTO:
        if strategy not in KNOWN_STRATEGIES:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy!r}")
        if benchmark is not None:
            try:
                resolve_benchmark(benchmark)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        try:
            result = await asyncio.to_thread(
                orch.backtest,
                portfolio,
                strategy_name=strategy,
                start=start_date,
                end=end_date,
                include_benchmark=include_benchmark,
                benchmark=benchmark,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Backtest failed: {exc}")
        return adapters.backtest_to_dto(result)

    # ---------------------------------------------------------------- Analysis

    @app.post(
        "/api/analysis",
        response_model=AnalysisBundleDTO,
        tags=["analysis"],
    )
    async def run_analysis(
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        orch: Annotated[Orchestrator, Depends(get_orchestrator)],
        container: Annotated[ServiceContainer, Depends(get_container)],
        include_risks: bool = True,
        news_days_back: int = 14,
    ) -> AnalysisBundleDTO:
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        try:
            bundle = await asyncio.to_thread(
                orch.run_analysis,
                portfolio,
                include_risks=include_risks,
                news_days_back=news_days_back,
            )
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

        # Also surface the same quant metrics the LLM saw, so the UI can
        # render them next to the citations the model produced.
        metrics = await asyncio.to_thread(
            _compute_metrics_for_status,
            portfolio,
            bundle.status,
            container,
        )
        return adapters.analysis_bundle_to_dto(
            bundle, portfolio=portfolio, metrics=metrics
        )

    # ----------------------------------------------------------------- Reports

    @app.get(
        "/api/reports",
        response_model=list[ReportFileDTO],
        tags=["reports"],
    )
    async def list_reports(
        reports_dir: Annotated[Path, Depends(get_reports_dir)],
    ) -> list[ReportFileDTO]:
        if not reports_dir.exists():
            return []
        files = sorted(
            (p for p in reports_dir.glob("*.md") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [_stat_to_dto(p) for p in files]

    @app.get(
        "/api/reports/{name}",
        response_model=ReportContentDTO,
        tags=["reports"],
    )
    async def get_report(
        name: str,
        reports_dir: Annotated[Path, Depends(get_reports_dir)],
    ) -> ReportContentDTO:
        _check_filename(name)
        path = reports_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Report not found: {name}")
        text = await asyncio.to_thread(path.read_text, "utf-8")
        st = path.stat()
        return ReportContentDTO(
            name=name,
            mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
            size_bytes=st.st_size,
            content_md=text,
        )

    @app.get("/api/reports/{name}/download", tags=["reports"])
    async def download_report(
        name: str,
        reports_dir: Annotated[Path, Depends(get_reports_dir)],
    ) -> FileResponse:
        _check_filename(name)
        path = reports_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Report not found: {name}")
        return FileResponse(
            path, media_type="text/markdown", filename=name
        )

    @app.post(
        "/api/reports",
        response_model=GenerateReportResponse,
        tags=["reports"],
    )
    async def generate_report(
        req: GenerateReportRequest,
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        orch: Annotated[Orchestrator, Depends(get_orchestrator)],
    ) -> GenerateReportResponse:
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        full = await asyncio.to_thread(
            orch.generate_report,
            portfolio,
            strategy_name=req.strategy or None,
            news_days_back=req.news_days_back,
            filename=req.filename,
        )
        return GenerateReportResponse(
            report=_stat_to_dto(full.report_path),
            warnings=list(full.warnings),
        )

    # -------------------------------------------------------------- Monitoring

    @app.post(
        "/api/monitoring",
        response_model=MonitoringSnapshotDTO,
        tags=["monitoring"],
    )
    async def run_monitoring(
        req: RunMonitoringRequest,
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        orch: Annotated[Orchestrator, Depends(get_orchestrator)],
        monitoring_dir: Annotated[Path, Depends(get_monitoring_dir)],
    ) -> MonitoringSnapshotDTO:
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        try:
            result = await asyncio.to_thread(
                orch.generate_monitoring_report,
                portfolio,
                news_days_back=req.news_days_back,
            )
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=f"LLM error: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface the cause to the user
            logger.exception("Monitoring pipeline failed")
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            )

        reports = (
            sorted(
                (p for p in monitoring_dir.glob("*.html") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if monitoring_dir.exists()
            else []
        )
        return MonitoringSnapshotDTO(
            generated_at=result.generated_at,
            items=adapters.monitoring_report_to_items(result.report),
            reports=[_stat_to_dto(p) for p in reports],
            had_previous_snapshot=result.had_previous_snapshot,
            report=result.report,
            warnings=list(result.warnings),
        )

    @app.get(
        "/api/monitoring/reports",
        response_model=list[ReportFileDTO],
        tags=["monitoring"],
    )
    async def list_monitoring_reports(
        monitoring_dir: Annotated[Path, Depends(get_monitoring_dir)],
    ) -> list[ReportFileDTO]:
        if not monitoring_dir.exists():
            return []
        files = sorted(
            (p for p in monitoring_dir.glob("*.html") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [_stat_to_dto(p) for p in files]

    @app.get(
        "/api/monitoring/reports/{name}",
        response_class=PlainTextResponse,
        tags=["monitoring"],
    )
    async def get_monitoring_report(
        name: str,
        monitoring_dir: Annotated[Path, Depends(get_monitoring_dir)],
    ) -> FileResponse:
        _check_filename(name)
        path = monitoring_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Report not found: {name}")
        return FileResponse(path, media_type="text/html")

    # ------------------------------------------------------------ Static frontend

    if _FRONTEND_DIR.is_dir():
        # Serve the in-repo CDN-React frontend at the root. `html=True`
        # makes `/` resolve to `index.html` automatically.
        app.mount(
            "/",
            StaticFiles(directory=str(_FRONTEND_DIR), html=True),
            name="frontend",
        )
    else:  # pragma: no cover
        logger.warning("Frontend directory not found at %s — API only.", _FRONTEND_DIR)

    return app


def _compute_metrics_for_status(
    portfolio: Portfolio,
    status,  # PortfolioStatus
    container: ServiceContainer,
):
    """Compute the same quant metrics the LLM saw, for the UI to render."""
    from investment_copilot.domain.analysis_metrics import compute_portfolio_metrics

    panel = {}
    for h in portfolio.holdings:
        df = container.data_service.load_ohlcv(h.ticker)
        if df is not None and not df.empty:
            panel[h.ticker] = df
    if not panel:
        return None

    benchmark_close = None
    benchmark_symbol = container.config.backtest.benchmark
    try:
        bdf = container.data_service.load_benchmark(benchmark_symbol)
        if bdf is not None and not bdf.empty and "close" in bdf.columns:
            benchmark_close = bdf["close"]
    except Exception:  # noqa: BLE001
        benchmark_close = None

    try:
        return compute_portfolio_metrics(
            portfolio,
            status,
            ohlcv_panel=panel,
            benchmark_close=benchmark_close,
            benchmark_symbol=benchmark_symbol if benchmark_close is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute metrics for UI failed: %s", exc)
        return None


app = create_app()
