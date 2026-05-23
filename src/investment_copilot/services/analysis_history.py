"""RAG-style loader for previously generated Markdown reports.

The CopilotService uses this to feed the LLM 1-3 recent reports as
context so it can frame current analysis as a *delta* over prior
assessments rather than starting from scratch each run.

Reports are picked from ``reports/*.md`` by mtime (newest first),
trimmed to the first few sections, and labeled with their filename
stem so the LLM can cite them as ``previous_report:LABEL``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# Headers that mark the substantive analysis sections worth keeping in
# RAG context. Anything below the next H1/H2 outside this list is dropped.
_KEPT_SECTION_KEYWORDS: tuple[str, ...] = (
    "podsumowanie",       # AI summary header
    "summary",
    "ryzyk",              # "Ryzyka" / "Ryzyko"
    "risk",
    "alert",
    "teza",
    "thesis",
    "ocena",
)

# Max characters per report after trimming. Each char ≈ 0.25 tokens.
MAX_REPORT_CHARS: int = 2000


def load_recent_reports(
    reports_dir: Path | str,
    *,
    n: int = 2,
    glob_pattern: str = "*.md",
) -> list[tuple[str, str]]:
    """Return the most recent N reports as ``(label, trimmed_body)`` tuples.

    The label is the filename stem (e.g. ``weekly_2026-05-07``). The body
    is reduced to the analysis sections only and capped at
    :data:`MAX_REPORT_CHARS`.

    Silently returns ``[]`` if ``reports_dir`` does not exist.
    """
    base = Path(reports_dir)
    if not base.is_dir():
        return []
    candidates = sorted(
        (p for p in base.glob(glob_pattern) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[tuple[str, str]] = []
    for path in candidates[:n]:
        try:
            text = path.read_text("utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - rare
            logger.warning("Could not read report %s: %s", path, exc)
            continue
        trimmed = _extract_kept_sections(text)
        if len(trimmed) > MAX_REPORT_CHARS:
            trimmed = trimmed[: MAX_REPORT_CHARS - 1].rstrip() + "…"
        if trimmed:
            out.append((path.stem, trimmed))
    return out


def _extract_kept_sections(markdown: str) -> str:
    """Keep only sections whose header matches one of the analysis keywords.

    Falls back to the full body if no matching header is found, so a
    free-form report still surfaces some signal.
    """
    lines = markdown.splitlines()
    kept: list[str] = []
    inside_kept = False
    for line in lines:
        if _is_section_header(line):
            inside_kept = _header_matches_keywords(line)
            if inside_kept:
                kept.append(line)
            continue
        if inside_kept:
            kept.append(line)
    if not kept:
        return markdown.strip()
    return "\n".join(kept).strip()


def _is_section_header(line: str) -> bool:
    return bool(re.match(r"^#{1,3}\s+", line))


def _header_matches_keywords(line: str) -> bool:
    lower = line.lower()
    return any(kw in lower for kw in _KEPT_SECTION_KEYWORDS)


def labels_of(reports: Iterable[tuple[str, str]]) -> set[str]:
    """Citation registry keys (e.g. ``{"previous_report:foo"}``) for the loaded reports."""
    return {f"previous_report:{label}" for label, _ in reports}


__all__ = [
    "MAX_REPORT_CHARS",
    "labels_of",
    "load_recent_reports",
]
