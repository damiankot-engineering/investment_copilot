"""Tests for ``configure_logging``."""

from __future__ import annotations

import logging

from investment_copilot.infrastructure.logging import configure_logging


def test_configure_logging_sets_level() -> None:
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING

    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_is_idempotent() -> None:
    """Calling twice must not double-add handlers we own."""
    configure_logging("INFO")
    initial_owned = sum(
        1 for h in logging.getLogger().handlers if getattr(h, "_copilot_owned", False)
    )
    configure_logging("INFO")
    again_owned = sum(
        1 for h in logging.getLogger().handlers if getattr(h, "_copilot_owned", False)
    )
    assert initial_owned == again_owned == 1


def test_configure_logging_quiets_third_parties() -> None:
    configure_logging("DEBUG")
    # Even at DEBUG root, noisy libs are pinned at WARNING
    assert logging.getLogger("urllib3").level >= logging.WARNING
    assert logging.getLogger("requests").level >= logging.WARNING


def test_configure_logging_unknown_level_falls_back_to_info() -> None:
    configure_logging("NOT_A_LEVEL")
    assert logging.getLogger().level == logging.INFO
