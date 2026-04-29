"""Structured-output schemas for the copilot.

Field names are English (clean Python access, JSON-API stability). Field
``description``\\s are Polish and form the semantic contract the LLM reads
via the JSON Schema we embed in the system prompt. The model is therefore
free to follow Polish guidance while emitting English keys.

All field values are expected to be in Polish (per ``LLMConfig.language``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# --- Reusable parts ---------------------------------------------------------


SeverityLevel = Literal["niskie", "średnie", "wysokie"]
RecommendationAction = Literal["trzymaj", "zwiększ", "zmniejsz", "obserwuj", "zamknij"]


class HoldingComment(BaseModel):
    """Per-pozycja komentarz w analizie portfela."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Ticker w formacie Stooq, np. 'pkn.pl'.")
    comment: str = Field(
        min_length=1,
        max_length=600,
        description=(
            "Krótki komentarz do tej pozycji w języku polskim: bieżąca "
            "kondycja, zgodność z tezą inwestycyjną, niedawne wydarzenia."
        ),
    )
    recommendation: RecommendationAction = Field(
        description=(
            "Rekomendacja działania: 'trzymaj', 'zwiększ', 'zmniejsz', "
            "'obserwuj' (bez zmian, ale wymaga uwagi), 'zamknij'."
        ),
    )


# --- Portfolio analysis -----------------------------------------------------


class PortfolioAnalysis(BaseModel):
    """Holistyczna analiza portfela."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        min_length=1,
        max_length=1200,
        description=(
            "Zwięzłe podsumowanie stanu portfela w 3-6 zdaniach po polsku: "
            "łączny PnL, dywersyfikacja, główne motywy."
        ),
    )
    holdings_comments: list[HoldingComment] = Field(
        description="Komentarz dla każdej pozycji w portfelu."
    )
    diversification_notes: str = Field(
        min_length=1,
        max_length=600,
        description=(
            "Ocena dywersyfikacji portfela: koncentracja sektorowa, "
            "ekspozycja na pojedyncze ryzyka, korelacje."
        ),
    )
    confidence: int = Field(
        ge=1,
        le=10,
        description=(
            "Poziom pewności analizy (1-10). 10 oznacza pełen kontekst danych "
            "i jasne wnioski; 1 oznacza znaczne luki w danych."
        ),
    )


# --- Risk alerts ------------------------------------------------------------


class RiskAlert(BaseModel):
    """Pojedynczy sygnał ryzyka."""

    model_config = ConfigDict(extra="forbid")

    ticker: str | None = Field(
        default=None,
        description=(
            "Ticker w formacie Stooq, jeśli ryzyko dotyczy konkretnej pozycji. "
            "Pomiń (null), gdy ryzyko jest portfelowe lub makro."
        ),
    )
    severity: SeverityLevel = Field(
        description="Istotność: 'niskie', 'średnie', 'wysokie'."
    )
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(
        min_length=1,
        max_length=600,
        description="Wyjaśnienie ryzyka i jego potencjalnego wpływu, po polsku.",
    )
    suggested_action: str = Field(
        min_length=1,
        max_length=300,
        description="Konkretna sugestia działania monitorującego lub mitygującego.",
    )


class RiskAlerts(BaseModel):
    """Lista ryzyk wraz z krótkim wprowadzeniem."""

    model_config = ConfigDict(extra="forbid")

    overview: str = Field(
        min_length=1,
        max_length=600,
        description="Krótkie wprowadzenie kontekstu ryzyka (2-4 zdania), po polsku.",
    )
    alerts: list[RiskAlert] = Field(
        max_length=10,
        description=(
            "Lista do 10 najistotniejszych ryzyk uporządkowanych od "
            "najbardziej do najmniej istotnego."
        ),
    )


# --- Thesis update ----------------------------------------------------------


class ThesisUpdate(BaseModel):
    """Zaktualizowana ocena tezy inwestycyjnej dla pojedynczej pozycji."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Ticker w formacie Stooq, np. 'pkn.pl'.")
    thesis_status: Literal["potwierdzona", "osłabiona", "do rewizji", "wykonana"] = Field(
        description=(
            "Status tezy: 'potwierdzona' (dane wspierają tezę), 'osłabiona' "
            "(pojawiły się przeciwstawne sygnały), 'do rewizji' (kontekst "
            "się zmienił, teza wymaga przepisania), 'wykonana' (cel "
            "osiągnięty)."
        ),
    )
    rationale: str = Field(
        min_length=1,
        max_length=1500,
        description=(
            "Uzasadnienie statusu: jakie dane lub wydarzenia z ostatnich "
            "tygodni wspierają lub podważają pierwotną tezę."
        ),
    )
    suggested_thesis: str | None = Field(
        default=None,
        max_length=1500,
        description=(
            "Sugerowana nowa wersja tezy, jeśli status to 'do rewizji'. "
            "W innych przypadkach pomiń (null)."
        ),
    )
    confidence: int = Field(
        ge=1,
        le=10,
        description="Poziom pewności oceny (1-10).",
    )
