"""Centralized logging configuration.

Every entrypoint (CLI, future API) calls :func:`configure_logging` once at
startup. Uses :mod:`rich` for console output if available, plain stdlib
otherwise — no hard dependency switch.
"""

from __future__ import annotations

import logging
from typing import Final

_DEFAULT_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger. Idempotent — safe to call multiple times.

    The library code (services, providers) uses ``logging.getLogger(__name__)``
    everywhere. This function just decides how those records are rendered.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()

    # Remove handlers we previously installed; leave others alone.
    for h in list(root.handlers):
        if getattr(h, "_copilot_owned", False):
            root.removeHandler(h)

    handler: logging.Handler
    try:  # pragma: no cover - rich is optional cosmetic
        from rich.logging import RichHandler

        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            show_time=True,
            log_time_format="[%X]",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    except Exception:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DATE_FORMAT))

    handler._copilot_owned = True  # type: ignore[attr-defined]
    handler.setLevel(lvl)
    root.addHandler(handler)
    root.setLevel(lvl)

    # Quiet down noisy third-party libraries.
    for noisy in ("urllib3", "requests", "feedparser"):
        logging.getLogger(noisy).setLevel(max(lvl, logging.WARNING))
