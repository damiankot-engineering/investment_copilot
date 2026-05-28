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
import hashlib
import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from investment_copilot import __version__
from investment_copilot.api import adapters
from investment_copilot.api.deps import (
    get_container,
    get_monitoring_dir,
    get_orchestrator,
    get_portfolio_path,
    get_reports_dir,
    get_watchlist_path,
)
from investment_copilot.api.schemas import (
    AnalysisBundleDTO,
    AppConfigDTO,
    BacktestResultDTO,
    BenchmarkInfoDTO,
    CalendarBundleDTO,
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
    WatchlistDTO,
    WatchlistStatusDTO,
)
from investment_copilot.domain.company_report import CalendarItem, CompanyReport
from investment_copilot.domain.models import BENCHMARK_SYMBOLS, resolve_benchmark
from investment_copilot.domain.portfolio import Portfolio
from investment_copilot.domain.strategies import KNOWN_STRATEGIES
from investment_copilot.domain.watchlist import Watchlist
from investment_copilot.infrastructure.llm import LLMError
from investment_copilot.infrastructure.logging import configure_logging
from investment_copilot.orchestrator import Orchestrator
from investment_copilot.services import (
    PortfolioError,
    ServiceContainer,
    WatchlistError,
    load_portfolio,
    load_watchlist,
    save_portfolio,
    save_watchlist,
)

logger = logging.getLogger(__name__)

# Resolved once at module-import time (used to mount static frontend).
# `src/investment_copilot/api/main.py` → up 3 → `src/`; then `frontend/`.
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")

# Long TTL — analysis cache is invalidated manually (on update-data) rather
# than by time. 365 days makes it effectively persistent.
_ANALYSIS_CACHE_TTL = timedelta(days=365)


def _analysis_cache_key(portfolio: Portfolio) -> str:
    """Stable cache key tied to portfolio content.

    Any change to the portfolio YAML (ticker, shares, thesis, ...) yields a
    different hash, so a stale entry from a previous portfolio shape never
    surfaces. The first 16 hex chars are plenty of entropy for a single-user
    local cache and keep the key short.
    """
    payload = portfolio.model_dump_json()
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"analysis:{digest}"


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

    # ------------------------------------------------------------- Watchlist

    @app.get(
        "/api/watchlist",
        response_model=WatchlistDTO,
        tags=["watchlist"],
    )
    async def get_watchlist(
        wl_path: Annotated[Path, Depends(get_watchlist_path)],
    ) -> WatchlistDTO:
        try:
            wl = await asyncio.to_thread(load_watchlist, str(wl_path))
        except WatchlistError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return adapters.watchlist_to_dto(wl)

    @app.put(
        "/api/watchlist",
        response_model=WatchlistDTO,
        tags=["watchlist"],
    )
    async def put_watchlist(
        payload: WatchlistDTO,
        wl_path: Annotated[Path, Depends(get_watchlist_path)],
    ) -> WatchlistDTO:
        raw = payload.model_dump(mode="json")
        for it in raw.get("items", []):
            it.pop("display_ticker", None)
        try:
            domain_wl = Watchlist.model_validate(raw)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            await asyncio.to_thread(save_watchlist, domain_wl, str(wl_path))
        except WatchlistError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return adapters.watchlist_to_dto(domain_wl)

    @app.get(
        "/api/watchlist/status",
        response_model=WatchlistStatusDTO,
        tags=["watchlist"],
    )
    async def watchlist_status(
        wl_path: Annotated[Path, Depends(get_watchlist_path)],
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> WatchlistStatusDTO:
        wl = await asyncio.to_thread(load_watchlist, str(wl_path))
        status_obj = await asyncio.to_thread(
            container.watchlist_service.current_status, wl
        )
        return adapters.watchlist_status_to_dto(status_obj)

    # -------------------------------------------------------------- Calendar

    @app.get(
        "/api/calendar",
        response_model=CalendarBundleDTO,
        tags=["calendar"],
    )
    async def get_calendar(
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> CalendarBundleDTO:
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        status = await asyncio.to_thread(
            container.portfolio_service.current_status, portfolio
        )
        bundle = await asyncio.to_thread(
            container.calendar_service.build, portfolio, status
        )
        return adapters.calendar_bundle_to_dto(bundle)

    # ------------------------------------------------------------------ Data

    @app.post(
        "/api/data/update",
        response_model=DataUpdateResult,
        tags=["data"],
    )
    async def update_data(
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        wl_path: Annotated[Path, Depends(get_watchlist_path)],
        orch: Annotated[Orchestrator, Depends(get_orchestrator)],
        container: Annotated[ServiceContainer, Depends(get_container)],
        news_days_back: int = 14,
    ) -> DataUpdateResult:
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        # Watchlist is optional — load returns empty Watchlist if file missing.
        try:
            watchlist = await asyncio.to_thread(load_watchlist, str(wl_path))
        except WatchlistError:
            watchlist = None
        report = await asyncio.to_thread(
            orch.update_data,
            portfolio,
            news_days_back=news_days_back,
            watchlist=watchlist,
        )
        # Fresh OHLCV + news = stale cached analysis. Drop it so the next
        # GET /api/analysis/cached returns null and the UI prompts a regen.
        try:
            container.sqlite_store.cache_delete(_analysis_cache_key(portfolio))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to invalidate analysis cache: %s", exc)
        return DataUpdateResult(
            ohlcv_updated=dict(report.ohlcv_updated),
            ohlcv_failed=dict(report.ohlcv_failed),
            benchmark_symbol=report.benchmark_symbol,
            benchmark_rows=report.benchmark_rows,
            news_inserted=report.news_inserted,
            news_failed=list(report.news_failed),
        )

    @app.get(
        "/api/data/update/stream",
        tags=["data"],
    )
    async def update_data_stream(
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        wl_path: Annotated[Path, Depends(get_watchlist_path)],
        orch: Annotated[Orchestrator, Depends(get_orchestrator)],
        container: Annotated[ServiceContainer, Depends(get_container)],
        news_days_back: int = 14,
    ) -> StreamingResponse:
        """SSE variant of POST /api/data/update — emits per-ticker + per-stage progress.

        Same side effects as the POST variant (OHLCV / benchmark / news
        refresh, analysis-cache invalidation). EventSource-friendly: GET
        method, ``text/event-stream`` media, one JSON event per ``data:``
        line. The final ``done`` event carries the same shape as
        ``DataUpdateResult`` so the UI can update its summary chips.
        """
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        try:
            watchlist = await asyncio.to_thread(load_watchlist, str(wl_path))
        except WatchlistError:
            watchlist = None

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def emit(event: dict) -> None:
            # Called from the worker thread; hand off to the event loop.
            loop.call_soon_threadsafe(queue.put_nowait, event)

        async def runner() -> None:
            try:
                report = await asyncio.to_thread(
                    orch.update_data,
                    portfolio,
                    news_days_back=news_days_back,
                    watchlist=watchlist,
                    on_progress=emit,
                )
                # Invalidate analysis cache exactly like the POST endpoint does
                try:
                    container.sqlite_store.cache_delete(
                        _analysis_cache_key(portfolio)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to invalidate analysis cache: %s", exc
                    )
                emit({
                    "type": "done",
                    "ohlcv_updated": dict(report.ohlcv_updated),
                    "ohlcv_failed": dict(report.ohlcv_failed),
                    "benchmark_symbol": report.benchmark_symbol,
                    "benchmark_rows": report.benchmark_rows,
                    "news_inserted": report.news_inserted,
                    "news_failed": list(report.news_failed),
                })
            except Exception as exc:  # noqa: BLE001
                logger.exception("update_data/stream worker failed")
                emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                # Sentinel — tells the generator to close cleanly.
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.create_task(runner())

        async def event_stream():
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
            },
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
        dto = adapters.analysis_bundle_to_dto(
            bundle, portfolio=portfolio, metrics=metrics
        )
        # Persist the rendered DTO so the UI can show it on next page load
        # without spending tokens again. Invalidated by POST /api/data/update.
        try:
            container.sqlite_store.cache_set(
                _analysis_cache_key(portfolio),
                dto.model_dump_json(),
            )
        except Exception as exc:  # noqa: BLE001 - caching is best-effort
            logger.warning("Failed to cache analysis bundle: %s", exc)
        return dto

    @app.get(
        "/api/analysis/cached",
        response_model=AnalysisBundleDTO | None,
        tags=["analysis"],
    )
    async def get_cached_analysis(
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> AnalysisBundleDTO | None:
        """Return the most recently persisted analysis for the current portfolio.

        Cache hits when the portfolio YAML is unchanged AND no
        ``POST /api/data/update`` has run since. Returns ``null`` otherwise
        so the UI can prompt the user to click ``Regeneruj``.
        """
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        key = _analysis_cache_key(portfolio)
        payload = await asyncio.to_thread(
            container.sqlite_store.cache_get, key, max_age=_ANALYSIS_CACHE_TTL,
        )
        if payload is None:
            return None
        try:
            return AnalysisBundleDTO.model_validate_json(payload)
        except ValueError as exc:
            logger.warning("Cached analysis payload invalid: %s", exc)
            return None

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

    @app.delete(
        "/api/reports/{name}",
        status_code=204,
        tags=["reports"],
    )
    async def delete_report(
        name: str,
        reports_dir: Annotated[Path, Depends(get_reports_dir)],
    ) -> None:
        _check_filename(name)
        path = reports_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Report not found: {name}")
        try:
            await asyncio.to_thread(path.unlink)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete {name}: {exc}"
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

    @app.delete(
        "/api/monitoring/reports/{name}",
        status_code=204,
        tags=["monitoring"],
    )
    async def delete_monitoring_report(
        name: str,
        monitoring_dir: Annotated[Path, Depends(get_monitoring_dir)],
    ) -> None:
        """Delete a monitoring HTML report. Leaves snapshot JSONs intact —
        those drive the prev-report carry-over context for future runs."""
        _check_filename(name)
        path = monitoring_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Report not found: {name}")
        try:
            await asyncio.to_thread(path.unlink)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete {name}: {exc}"
            )

    # --------------------------------------- Per-company report (new monitoring)

    def _norm_ticker(t: str) -> str:
        # Allow Stooq-style with dot ("pkn.pl"), letters, digits, underscore, dash.
        if not re.match(r"^[A-Za-z0-9._-]+$", t) or ".." in t:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid ticker: {t!r}",
            )
        return t.lower()

    async def _load_pf_and_status(
        pf_path: Path, container: ServiceContainer,
    ) -> tuple[Portfolio, object]:
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        status_obj = await asyncio.to_thread(
            container.portfolio_service.current_status, portfolio
        )
        return portfolio, status_obj

    @app.get(
        "/api/companies/{ticker}/factsheet",
        response_model=CompanyReport,
        tags=["companies"],
    )
    async def get_company_factsheet(
        ticker: str,
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> CompanyReport:
        """Deterministic per-company snapshot (no LLM call).

        Fast (<1s typical) — fundamentals come from 24h BR cache, OHLCV
        and news from the local SQLite/parquet stores. tldr/strengths/
        risks are placeholder strings until ``POST /report`` runs the LLM.
        """
        norm = _norm_ticker(ticker)
        portfolio, status_obj = await _load_pf_and_status(pf_path, container)
        try:
            return await asyncio.to_thread(
                container.company_report_service.build_factsheet,
                portfolio, status_obj, ticker=norm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get(
        "/api/companies/{ticker}/report",
        response_model=CompanyReport | None,
        tags=["companies"],
    )
    async def get_cached_company_report(
        ticker: str,
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> CompanyReport | None:
        """Return the most recently generated AI report for a ticker (or null)."""
        norm = _norm_ticker(ticker)
        return await asyncio.to_thread(
            container.company_report_service.get_cached_report, norm,
        )

    @app.post(
        "/api/companies/{ticker}/report",
        response_model=CompanyReport,
        tags=["companies"],
    )
    async def generate_company_report(
        ticker: str,
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> CompanyReport:
        """Run the LLM narrative call and persist + return the full report."""
        norm = _norm_ticker(ticker)
        portfolio, status_obj = await _load_pf_and_status(pf_path, container)
        try:
            return await asyncio.to_thread(
                container.company_report_service.generate_report,
                portfolio, status_obj, ticker=norm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    @app.get(
        "/api/companies/{ticker}/report.html",
        response_class=HTMLResponse,
        tags=["companies"],
    )
    async def download_company_report_html(
        ticker: str,
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> HTMLResponse:
        """Standalone HTML download — uses the cached AI report when present,
        otherwise falls back to the deterministic factsheet (no LLM call)."""
        import json as _json

        norm = _norm_ticker(ticker)
        report = await asyncio.to_thread(
            container.company_report_service.get_cached_report, norm,
        )
        if report is None:
            portfolio, status_obj = await _load_pf_and_status(pf_path, container)
            try:
                report = await asyncio.to_thread(
                    container.company_report_service.build_factsheet,
                    portfolio, status_obj, ticker=norm,
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc))

        template_path = _TEMPLATES_DIR / "company_report.html"
        if not template_path.is_file():
            raise HTTPException(
                status_code=500,
                detail="Internal: company_report.html template missing.",
            )
        template = await asyncio.to_thread(template_path.read_text, "utf-8")
        payload_json = _json.dumps(
            report.model_dump(mode="json"), ensure_ascii=False
        )
        html = template.replace("__REPORT_DATA_JSON__", payload_json)
        return HTMLResponse(
            content=html,
            headers={
                "Content-Disposition": (
                    f'attachment; filename="report_{norm}_'
                    f'{report.report_date}.html"'
                ),
            },
        )

    @app.get(
        "/api/companies/upcoming",
        response_model=list[CalendarItem],
        tags=["companies"],
    )
    async def list_upcoming_reports(
        pf_path: Annotated[Path, Depends(get_portfolio_path)],
        container: Annotated[ServiceContainer, Depends(get_container)],
    ) -> list[CalendarItem]:
        """Calendar across the whole portfolio (sorted, closest first)."""
        portfolio = await asyncio.to_thread(load_portfolio, str(pf_path))
        return await asyncio.to_thread(
            container.company_report_service.list_upcoming_reports, portfolio,
        )

    # ------------------------------------------------------------ Static frontend

    if _FRONTEND_DIR.is_dir():
        # Serve the in-repo CDN-React frontend at the root. `html=True`
        # makes `/` resolve to `index.html` automatically. Cache-Control
        # is forced to "no-cache" because the JSX files are Babel-transpiled
        # in the browser and bumping versions per change would be painful —
        # this guarantees `uv run uvicorn ...` always serves fresh files.
        class _NoCacheStaticFiles(StaticFiles):
            async def get_response(self, path, scope):
                resp = await super().get_response(path, scope)
                resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
                return resp

        app.mount(
            "/",
            _NoCacheStaticFiles(directory=str(_FRONTEND_DIR), html=True),
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
