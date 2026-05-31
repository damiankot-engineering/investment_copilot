"""Company-identity matching for news relevance.

The problem this solves: a holding's ``keywords`` list mixes precise company
identifiers (ticker, brand name) with broad *thematic* terms (``akcje``,
``e-commerce``, ``hosting``, ``fintech``). Matching news on *any* of those as a
case-insensitive **substring** floods a company with off-topic sector news —
e.g. every GPW headline containing "akcje" gets stamped onto XTB.

The fix has two parts, both implemented here:

1. :func:`derive_news_identifiers` — reduce a holding to the terms that
   genuinely *identify* it: the ticker stem plus the brand phrase(s) from its
   display name (a parenthetical alias is kept too). Thematic keywords are
   intentionally dropped from news matching; they only ever described the
   investment thesis, not "this article is about this company".

2. :func:`compile_identity_matcher` — match those terms at **word
   boundaries** (not substrings), so a 3-letter ticker like ``DIG`` can't hit
   inside ``Digital`` and ``akcje`` can't hit inside a larger word.

A multi-word brand ("Digital Network") is matched as a contiguous phrase so a
single generic token ("Digital") never grants a match on its own.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from investment_copilot.domain.models import normalize_ticker

# Corporate-form / legal-suffix tokens that carry no brand signal. Compared
# lowercased with surrounding dots stripped (so "S.A." -> "sa").
_CORP_STOPWORDS: frozenset[str] = frozenset({
    "sa", "spolka", "spółka", "akcyjna", "fiz", "plc", "se", "asi",
    "nv", "ag", "inc", "group", "grupa", "holding", "holdings",
})

# Word-character class used for boundary lookarounds. ``\w`` under Unicode
# already covers Polish diacritics, but we spell it out to keep the boundary
# semantics explicit and independent of the ``re.ASCII`` flag.
_WORD = r"\w"


def _ticker_stem(ticker: str) -> str:
    """``"xtb.pl"`` -> ``"XTB"``; ``"etfbdivpl.pl"`` -> ``"ETFBDIVPL"``."""
    return normalize_ticker(ticker).split(".")[0].upper()


def _brand_terms_from_name(name: str) -> list[str]:
    """Extract distinctive brand term(s) from a display name.

    Keeps any parenthetical alias verbatim ("X-Trade Brokers"), strips the
    corporate form ("SA", "FIZ", ...) from the core, and returns the
    remaining leading phrase as one contiguous term.
    """
    terms: list[str] = []
    for alias in re.findall(r"\(([^)]+)\)", name):
        alias = alias.strip()
        if alias:
            terms.append(alias)
    core = re.sub(r"\([^)]*\)", " ", name)
    tokens = [t for t in re.split(r"\s+", core.strip()) if t]
    kept = [t for t in tokens if t.lower().strip(".") not in _CORP_STOPWORDS]
    if kept:
        terms.append(" ".join(kept))
    return terms


def _collapsed_variant(term: str) -> str | None:
    """Strip separators so "cyber_Folks" also matches "Cyberfolks".

    Returns ``None`` when collapsing changes nothing (already a plain token).
    """
    collapsed = re.sub(r"[^0-9A-Za-zÀ-ɏ]+", "", term)
    if collapsed and collapsed.lower() != term.lower():
        return collapsed
    return None


def derive_news_identifiers(
    ticker: str,
    name: str | None,
    keywords: Sequence[str] | None = None,
) -> list[str]:
    """Return the terms that identify *this* company in a news headline.

    Order: ticker stem, then brand phrase(s). When no display name is set we
    fall back to the user's single-token keywords as a best-effort identity
    (multi-word thematic phrases are still excluded). Separator-collapsed
    variants are appended so alternate stylings match. Result is
    case-insensitively de-duplicated; terms shorter than 2 chars are dropped.
    """
    out: list[str] = [_ticker_stem(ticker)]
    if name and name.strip():
        out.extend(_brand_terms_from_name(name))
    else:
        out.extend(k for k in (keywords or []) if k and " " not in k)

    variants = [v for t in out if (v := _collapsed_variant(t))]
    out.extend(variants)

    seen: set[str] = set()
    result: list[str] = []
    for t in out:
        key = t.lower()
        if len(t) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(t)
    return result


def compile_identity_matcher(terms: Iterable[str]) -> re.Pattern[str] | None:
    """Compile ``terms`` into one case-insensitive, word-boundary matcher.

    Returns ``None`` when there are no usable terms — callers treat that as
    "no filter" (match everything), matching the previous empty-keywords
    behaviour.
    """
    parts = [re.escape(t.strip()) for t in terms if t and t.strip()]
    if not parts:
        return None
    body = "|".join(parts)
    pattern = rf"(?<!{_WORD})(?:{body})(?!{_WORD})"
    return re.compile(pattern, re.IGNORECASE)


def matches_identity(text: str, matcher: re.Pattern[str] | None) -> bool:
    """True when ``matcher`` is ``None`` (no filter) or hits ``text``."""
    return matcher is None or matcher.search(text) is not None


__all__ = [
    "compile_identity_matcher",
    "derive_news_identifiers",
    "matches_identity",
]
