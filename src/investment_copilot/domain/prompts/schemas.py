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
ThesisStatus = Literal[
    "potwierdzona",
    "silnie potwierdzona",
    "w mocy",
    "pod testem",
    "osłabiona",
    "do rewizji",
    "wykonana",
]
SignalDirection = Literal["bullish", "neutral", "bearish"]
ChangeDirection = Literal["akceleracja", "stabilizacja", "rozczarowanie", "brak zmian"]


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


# --- Monitoring report ------------------------------------------------------


class MonitoringMetric(BaseModel):
    """Pojedyncza wyróżniona metryka pokazywana w sekcji per-spółka."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        min_length=1, max_length=80,
        description="Krótki nagłówek, np. 'PRZYCHODY Q1 2026', 'EBITDA FY2025'.",
    )
    value: str = Field(
        min_length=1, max_length=80,
        description="Wartość sformatowana, np. '42.9M PLN', '+71% r/r'.",
    )
    detail: str | None = Field(
        default=None, max_length=200,
        description="Opcjonalna druga linijka, np. 'marża 59.2%'.",
    )
    tone: Literal["positive", "negative", "warning", "neutral"] = Field(
        default="neutral",
        description="Wydźwięk: positive/negative/warning/neutral. Steruje kolorem.",
    )


class MonitoringCompany(BaseModel):
    """Sekcja per-spółka raportu monitorującego."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Ticker w formacie Stooq, np. 'dnp.pl'.")
    name: str = Field(description="Nazwa spółki.")
    headline: str = Field(
        min_length=1, max_length=200,
        description="Jednolinijkowa konkluzja, np. '+71% Q2 r/r — zaskoczenie'.",
    )
    metrics: list[MonitoringMetric] = Field(
        min_length=3, max_length=4,
        description=(
            "3-4 metryki (preferuj 4). Pierwsza = wynik ostatniego raportu, "
            "ostatnia = data najbliższego raportu lub 'TBA'."
        ),
    )
    last_reading_label: str = Field(
        min_length=1, max_length=80,
        description="Etykieta ostatniego odczytu do tabeli, np. 'RR 2025 + Q1'.",
    )
    vs_expectations: str = Field(
        min_length=1, max_length=120,
        description="vs oczekiwań do tabeli, np. '✅ Zgodny', '❌ -20% vs kons.'.",
    )
    next_report_label: str = Field(
        min_length=1, max_length=40,
        description="Etykieta najbliższego raportu, np. '29 MAJ 2026' lub 'TBA'.",
    )
    key_question: str = Field(
        min_length=1, max_length=160,
        description="Kluczowe pytanie do tabeli, np. 'Marże z BGMO, dywidenda'.",
    )
    last_results_summary: str = Field(
        min_length=80, max_length=1200,
        description=(
            "3-5 zdań: co pokazał ostatni raport / ESPI. Gdy brak świeżych "
            "danych — bazuj na poprzednim raporcie albo tezie. Konkretna "
            "treść; nie pisz 'brak danych'."
        ),
    )
    next_catalyst_focus: str = Field(
        min_length=80, max_length=1200,
        description=(
            "3-5 zdań: na co czekamy. Lista pytań (1)/(2)/(3). Bazuj na "
            "tezie/branży gdy brak ESPI."
        ),
    )
    thesis_status: ThesisStatus = Field(description="Status tezy inwestycyjnej.")
    signal: SignalDirection = Field(
        description="bullish / neutral / bearish (steruje kolorem signal-bara).",
    )
    signal_title: str = Field(
        min_length=1, max_length=160,
        description="Nagłówek pasa sygnałowego, np. 'TEZA NIENARUSZONA'.",
    )
    signal_body: str = Field(
        min_length=1, max_length=600,
        description="Uzasadnienie 2-3 zdania po polsku.",
    )
    recommendation: RecommendationAction = Field(
        description="trzymaj / zwiększ / zmniejsz / obserwuj / zamknij.",
    )
    change_narrative: str | None = Field(
        default=None, max_length=400,
        description=(
            "1-2 zdania o zmianie vs poprzedni raport. Null gdy brak "
            "poprzedniego raportu lub brak istotnej zmiany."
        ),
    )
    change_direction: ChangeDirection | None = Field(
        default=None,
        description=(
            "akceleracja / stabilizacja / rozczarowanie / brak zmian. "
            "Null gdy brak poprzedniego raportu."
        ),
    )


class MonitoringCalendarEntry(BaseModel):
    """Wiersz kalendarza katalizatorów w raporcie monitorującym."""

    model_config = ConfigDict(extra="forbid")

    date_label: str = Field(
        min_length=1, max_length=40,
        description="Data w formacie 'DD MMM YYYY', np. '14 MAJ 2026'.",
    )
    ticker: str | None = Field(
        default=None, description="Ticker lub null gdy ogólny katalizator.",
    )
    title: str = Field(
        min_length=1, max_length=200,
        description="Krótki tytuł, np. 'Dino Polska — Q1 2026'.",
    )
    description: str = Field(
        min_length=1, max_length=400,
        description="1-2 zdania dlaczego istotny.",
    )
    importance: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="high (czerwony) / medium (żółty) / low (szary). Steruje kolorem.",
    )


class MonitoringReport(BaseModel):
    """Raport monitorujący portfel — odpowiednik załączonego HTML wzoru."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=1, max_length=200,
        description="Tytuł, np. 'Przegląd Portfela — Q1 2026'.",
    )
    subtitle: str | None = Field(
        default=None, max_length=200, description="Opcjonalny podtytuł.",
    )
    synthesis: str = Field(
        min_length=1, max_length=1500,
        description=(
            "Synteza 4-6 zdań: najsilniejszy i najsłabszy sygnał, raporty "
            "na które czekamy."
        ),
    )
    companies: list[MonitoringCompany] = Field(
        description=(
            "Per pozycja DOKŁADNIE jeden wpis w kolejności z kontekstu."
        ),
    )
    calendar: list[MonitoringCalendarEntry] = Field(
        max_length=15,
        description="Do 15 katalizatorów chronologicznie.",
    )
    confidence: int = Field(
        ge=1, le=10,
        description="1-10 (10 = pełen kontekst, 4-6 = używasz poprzedniego raportu).",
    )
