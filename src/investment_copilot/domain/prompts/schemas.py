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


CitationSourceType = Literal["news", "metric", "fundamentals", "previous_report"]


class Citation(BaseModel):
    """A grounding citation — must reference something present in the prompt.

    Citations are verified Python-side after LLM completion; unknown
    references are dropped (lenient mode) so the model is encouraged
    to over-cite rather than fabricate.

    Examples
    --------
    ``Citation(source_type="news", reference="news:3")``
    ``Citation(source_type="metric", reference="metric:pkn.pl.ret_30d_pct")``
    ``Citation(source_type="fundamentals", reference="fundamentals:cdr.pl.revenue_yoy_pct")``
    ``Citation(source_type="previous_report", reference="previous_report:weekly_2026-05-07")``
    """

    model_config = ConfigDict(extra="forbid")

    source_type: CitationSourceType = Field(
        description=(
            "Rodzaj źródła: 'news' (id z sekcji Recent news), 'metric' "
            "(klucz z sekcji Quant metrics), 'fundamentals' (pole z sekcji "
            "Fundamentals), 'previous_report' (id z sekcji Previous reports)."
        ),
    )
    reference: str = Field(
        min_length=1, max_length=200,
        description=(
            "Identyfikator źródła BEZ duplikowania source_type. "
            "Dla source_type='news': 'news:3'. "
            "Dla source_type='metric': 'portfolio.hhi' albo 'pkn.pl.ret_30d_pct' "
            "(SAM klucz, bez prefiksu 'metric:'). "
            "Dla source_type='fundamentals': 'cdr.pl.revenue_yoy_pct'. "
            "Dla source_type='previous_report': 'previous_report:weekly_2026-05-07'."
        ),
    )


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
    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "Lista źródeł wspierających komentarz — przynajmniej 1 pozycja "
            "z sekcji Quant metrics LUB Recent news. Wartości nieobecne w "
            "kontekście są odrzucane przy walidacji."
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
    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "Lista źródeł uzasadniających istnienie tego ryzyka — co najmniej "
            "1 cytowanie z Quant metrics, Recent news lub Fundamentals."
        ),
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
    short_name: str | None = Field(
        default=None, max_length=40,
        description="Krótka etykieta do summary card, np. 'Synektik', 'Broker CFD'.",
    )
    header_meta: str | None = Field(
        default=None, max_length=200,
        description=(
            "Linijka pod nazwą w sekcji: cena/mcap/ATH, np. "
            "'~282 PLN · kap. ~2,4 mld · ATH 309,80'. Bazuj na "
            "'Fundamentals — wskaźniki rynkowe'."
        ),
    )
    headline: str = Field(
        min_length=1, max_length=200,
        description="Jednolinijkowa konkluzja, np. '+71% Q2 r/r — zaskoczenie'.",
    )
    summary_card_tag: str | None = Field(
        default=None, max_length=80,
        description=(
            "Krótki tag pod summary card pokazujący najbliższy event, np. "
            "'RAPORT 15 MAJ ‼️', 'PIERWSZA WYPŁATA 26 MAJ', 'AKCELERACJA ✅'."
        ),
    )
    metrics: list[MonitoringMetric] = Field(
        min_length=2, max_length=4,
        description=(
            "2-4 metryki (preferuj 4). Pierwsza = wynik ostatniego raportu, "
            "ostatnia = data najbliższego raportu lub 'TBA'."
        ),
    )
    last_reading_label: str = Field(
        default="", max_length=80,
        description="Etykieta ostatniego odczytu do tabeli, np. 'RR 2025 + Q1'.",
    )
    vs_expectations: str = Field(
        default="", max_length=120,
        description="vs oczekiwań do tabeli, np. '✅ Zgodny', '❌ -20% vs kons.'.",
    )
    next_report_label: str = Field(
        default="", max_length=40,
        description="Etykieta najbliższego raportu, np. '29 MAJ 2026' lub 'TBA'.",
    )
    key_question: str = Field(
        default="", max_length=160,
        description="Kluczowe pytanie do tabeli, np. 'Marże z BGMO, dywidenda'.",
    )
    last_results_summary: str = Field(
        min_length=1, max_length=1200,
        description=(
            "3-5 zdań: co pokazał ostatni raport. Cytuj BR YoY % wprost. "
            "Dla pozycji bez danych BR (ETFy) — 1-2 zdania o tezie wystarczą."
        ),
    )
    next_catalyst_focus: str = Field(
        min_length=1, max_length=1200,
        description="3-5 zdań: na co czekamy. Pytania kluczowe.",
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


class PortfolioWeightEntry(BaseModel):
    """Pojedynczy wiersz w sekcji 'Struktura portfela'."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Ticker w formacie Stooq.")
    weight_label: str = Field(
        min_length=1, max_length=20,
        description="Etykieta wagi, np. '50,0%' albo '14,0%'.",
    )
    role: str = Field(
        min_length=1, max_length=60,
        description=(
            "Krótki opis roli w portfelu, np. 'core dywidendowy', "
            "'medtech growth', 'broker CFD/cycl.'."
        ),
    )


class PortfolioStructure(BaseModel):
    """Sekcja '⚖️ STRUKTURA PORTFELA — RYZYKO & KONCENTRACJA'."""

    model_config = ConfigDict(extra="forbid")

    weights: list[PortfolioWeightEntry] = Field(
        description="Wagi pozycji w portfelu, posortowane malejąco.",
    )
    concentration_narrative: str = Field(
        min_length=1, max_length=2000,
        description=(
            "Analiza koncentracji 3-5 zdań: która pozycja dominuje, ekspozycja "
            "sektorowa, korelacje. Konkretna i merytoryczna."
        ),
    )


class MonitoringReport(BaseModel):
    """Raport monitorujący portfel — odpowiednik załączonego HTML wzoru."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=1, max_length=200,
        description="Tytuł, np. 'Przegląd Portfela — Analiza Raportów Q1 2026'.",
    )
    subtitle: str | None = Field(
        default=None, max_length=200,
        description=(
            "Podtytuł opisujący skład, np. 'PORTFEL GPW: ETF DYWIDENDOWY + "
            "4 SATELITY GROWTH'."
        ),
    )
    synthesis: str = Field(
        min_length=1, max_length=2000,
        description=(
            "Synteza 5-8 zdań: najsilniejszy + najsłabszy sygnał z BR, "
            "nadchodzące raporty z datami, kluczowe pytania do tezy. "
            "Konkretne tickery + liczby."
        ),
    )
    portfolio_structure: PortfolioStructure | None = Field(
        default=None,
        description=(
            "Opcjonalna sekcja '⚖️ Struktura portfela' z wagami i analizą "
            "koncentracji. Pomiń (null) tylko gdy nie da się jej zbudować."
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


# --- Single-company narrative (Faza A) -------------------------------------


class CompanyNarrativeBullet(BaseModel):
    """Bullet point with a citation key — used in CompanyReport strengths/risks."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=10,
        max_length=240,
        description="Treść mocnej strony / ryzyka po polsku, 1 zdanie.",
    )
    citation: str = Field(
        min_length=3,
        max_length=80,
        description=(
            "Klucz źródła z sekcji 'Dostępne źródła'. Dozwolone prefiksy: "
            "'metric:', 'news:', 'fundamentals:', 'portfolio:', 'thesis'. "
            "Bullety bez znanego źródła są odrzucane przy walidacji."
        ),
    )


class CompanyNarrative(BaseModel):
    """Narrative trio (TL;DR / strengths / risks) for a single company.

    Generated by ONE LLM call from a focused per-ticker context. The rest
    of the CompanyReport (header, KPI, market, calendar) is built
    deterministically from BR + OHLCV + portfolio.
    """

    model_config = ConfigDict(extra="forbid")

    tldr: str = Field(
        min_length=50,
        max_length=600,
        description=(
            "Streszczenie 2-3 zdania po polsku: kondycja spółki na tle "
            "tezy + ostatni okres + 1 kluczowy element. Min. 1 liczba "
            "cytowana wprost z 'Dostępne źródła'."
        ),
    )
    strengths: list[CompanyNarrativeBullet] = Field(
        min_length=2,
        max_length=5,
        description="3-5 mocnych stron, każda z citation kluczem.",
    )
    risks: list[CompanyNarrativeBullet] = Field(
        min_length=2,
        max_length=5,
        description="3-5 ryzyk, każde z citation kluczem.",
    )
    confidence: int = Field(
        ge=1,
        le=10,
        description="7-9 gdy są BR YoY + min. 2 świeże news; 4-6 gdy braki.",
    )
