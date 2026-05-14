"""FastAPI dependencies.

Single source of truth for how the API constructs the service container,
loads the active portfolio path, and exposes an Orchestrator to routes.

Config / portfolio paths are resolved once at startup from env vars and
cached. Set ``COPILOT_CONFIG`` and (optionally) ``COPILOT_PORTFOLIO``
before launching uvicorn to point at a different config.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from investment_copilot.config import load_config
from investment_copilot.orchestrator import Orchestrator
from investment_copilot.services import ServiceContainer, build_container


def _config_path() -> str:
    return os.environ.get("COPILOT_CONFIG", "config.yaml")


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    """Build the ServiceContainer once per process."""
    return build_container(load_config(_config_path()))


def get_portfolio_path(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> Path:
    override = os.environ.get("COPILOT_PORTFOLIO")
    if override:
        return Path(override)
    return Path(container.config.portfolio.path)


def get_orchestrator(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> Orchestrator:
    return Orchestrator(container)


def get_reports_dir(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> Path:
    _ = container
    return Path("reports")


def get_monitoring_dir() -> Path:
    return Path("reports") / "monitoring"
