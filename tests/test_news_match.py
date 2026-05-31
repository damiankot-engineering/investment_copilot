"""Tests for company-identity news matching (relevance filtering)."""

from __future__ import annotations

from investment_copilot.domain.news_match import (
    compile_identity_matcher,
    derive_news_identifiers,
    matches_identity,
)


def _hits(terms, text) -> bool:
    return matches_identity(text, compile_identity_matcher(terms))


# --- derive_news_identifiers ------------------------------------------------


def test_identifiers_use_ticker_stem_and_brand() -> None:
    ids = derive_news_identifiers("xtb.pl", "XTB SA (X-Trade Brokers)", ["broker", "akcje"])
    # stem + parenthetical alias + brand phrase; thematic keywords excluded.
    assert "XTB" in ids
    assert "X-Trade Brokers" in ids
    assert "broker" not in ids
    assert "akcje" not in ids


def test_identifiers_strip_corporate_form() -> None:
    ids = derive_news_identifiers("snt.pl", "Synektik SA", ["MedTech"])
    assert "Synektik" in ids
    assert "SNT" in ids
    assert "MedTech" not in ids


def test_identifiers_multiword_brand_kept_as_phrase() -> None:
    ids = derive_news_identifiers("dig.pl", "Digital Network SA", ["DOOH"])
    assert "Digital Network" in ids
    # No bare "Digital" token that would match unrelated tech news.
    assert "Digital" not in ids


def test_identifiers_collapsed_variant_for_separators() -> None:
    ids = derive_news_identifiers("cbf.pl", "cyber_Folks SA", ["hosting"])
    assert "cyber_Folks" in ids
    assert "cyberFolks" in ids  # separator-collapsed alt styling
    assert "hosting" not in ids


def test_identifiers_fallback_to_single_token_keywords_without_name() -> None:
    ids = derive_news_identifiers("pkn.pl", None, ["Orlen", "rafineria ropy"])
    assert "PKN" in ids
    assert "Orlen" in ids
    # Multi-word thematic phrase is not treated as identity.
    assert "rafineria ropy" not in ids


# --- word-boundary matching -------------------------------------------------


def test_match_requires_word_boundary() -> None:
    # "DIG" must not hit inside "Digital".
    assert not _hits(["DIG"], "Cyfrowa transformacja i Digital marketing")
    assert _hits(["DIG"], "Spółka DIG ogłasza wyniki")


def test_broad_theme_word_does_not_match() -> None:
    ids = derive_news_identifiers("xtb.pl", "XTB SA (X-Trade Brokers)", ["akcje", "ETF"])
    assert not _hits(ids, "Akcje WIG20 rosną po danych z USA")
    assert _hits(ids, "XTB wypłaca rekordową dywidendę")


def test_multiword_brand_matches_only_as_phrase() -> None:
    ids = ["Digital Network"]
    assert _hits(ids, "Digital Network SA zwiększa przychody")
    assert not _hits(ids, "Transformacja digital w polskich firmach")


def test_collapsed_variant_matches_alternate_styling() -> None:
    ids = derive_news_identifiers("cbf.pl", "cyber_Folks SA", [])
    assert _hits(ids, "Cyberfolks przejmuje konkurenta")  # no underscore in headline
    assert _hits(ids, "Wyniki cyber_Folks za Q1")


def test_empty_terms_match_everything() -> None:
    # No identifiers -> no filter (matches previous empty-keywords behaviour).
    assert _hits([], "dowolny nagłówek")
    assert matches_identity("anything", None)


def test_match_is_case_insensitive() -> None:
    assert _hits(["Synektik"], "SYNEKTIK ogłasza nowy kontrakt")
    assert _hits(["XTB"], "notowania xtb w górę")
