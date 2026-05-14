/* Mock data for Investment Copilot.
   Typed in JSDoc but mirrors the spec's TS types. Trivially swappable for fetch(). */

/* ----------------------------- Holdings ----------------------------- */
const MOCK_HOLDINGS = [
  { ticker: 'PKN',  name: 'PKN Orlen',  shares: 120, entry_price:  62.40, entry_date: '2023-04-12',
    thesis: 'Konsolidacja sektora energetycznego po fuzji z Lotosem i PGNiG. Stabilne przepływy z petrochemii, ekspozycja na transformację energetyczną oraz polityka dywidendowa wspierają długoterminową tezę.',
    keywords: ['energia', 'dywidenda', 'transformacja', 'konsolidacja'] },
  { ticker: 'CDR',  name: 'CD Projekt', shares:  45, entry_price: 168.00, entry_date: '2023-01-30',
    thesis: 'Pipeline IP (Cyberpunk: Orion, Polaris) + monetyzacja istniejących franczyz. Wysoka marża operacyjna, ryzyko cyklu wydawniczego.',
    keywords: ['gaming', 'IP', 'globalna marka'] },
  { ticker: 'KGH',  name: 'KGHM',       shares:  60, entry_price: 124.50, entry_date: '2022-11-08',
    thesis: 'Cykl miedziowy + ekspozycja na elektryfikację. Ryzyko: podatek od wydobycia, koszty energii. Trzymamy do wyższego cyklu cen.',
    keywords: ['surowce', 'miedź', 'cykl', 'elektryfikacja'] },
  { ticker: 'LPP',  name: 'LPP',        shares:   8, entry_price: 12300.00, entry_date: '2023-09-22',
    thesis: 'Skala marek (Reserved, Sinsay) + ekspansja w Europie. Sinsay jako motor wzrostu w segmencie ultra fast fashion.',
    keywords: ['retail', 'fast fashion', 'ekspansja', 'Sinsay'] },
  { ticker: 'ALE',  name: 'Allegro',    shares: 320, entry_price:  29.80, entry_date: '2024-02-14',
    thesis: 'Marketplace #1 w CEE. Allegro Pay i monetyzacja reklam jako dźwignie marżowe. Konkurencja z Temu/Shein zarządzona.',
    keywords: ['e-commerce', 'platforma', 'reklamy', 'CEE'] },
];

// Last prices that produce a realistic mix of green/red
const LAST_PRICES = {
  PKN:  74.16,   // +18.8%
  CDR: 152.40,   // -9.3%
  KGH: 153.20,   // +23.0%
  LPP: 16850.00, // +37.0%
  ALE:  27.55,   // -7.5%
};

function buildPortfolioStatus() {
  const holdings = MOCK_HOLDINGS.map(h => {
    const last_price = LAST_PRICES[h.ticker];
    const value      = last_price * h.shares;
    const cost       = h.entry_price * h.shares;
    const pnl        = value - cost;
    const pnl_pct    = (value / cost - 1) * 100;
    return { ...h, last_price, value, pnl, pnl_pct };
  });
  const total_value   = holdings.reduce((a, x) => a + x.value, 0);
  const total_cost    = holdings.reduce((a, x) => a + x.entry_price * x.shares, 0);
  const total_pnl     = total_value - total_cost;
  const total_pnl_pct = (total_value / total_cost - 1) * 100;
  return {
    holdings,
    total_value,
    total_pnl,
    total_pnl_pct,
    as_of: '2026-05-14T09:42:18+02:00',
  };
}
const MOCK_PORTFOLIO = buildPortfolioStatus();

/* ----------------------------- Equity curve -----------------------------
   ~3 years of synthesized daily points. Portfolio drifts up, with 2022
   drawdown visible relative to WIG20 (which we model as a slower-grower). */
function buildEquityCurve() {
  // Deterministic pseudo-random
  let seed = 1337;
  const rnd = () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };
  const days = 3 * 252; // ~3 trading years
  const start = new Date('2023-04-01');
  const out = [];
  let pv = 100, bv = 100;
  for (let i = 0; i < days; i++) {
    const d = new Date(start);
    d.setDate(start.getDate() + i);
    // skip weekends
    if (d.getDay() === 0 || d.getDay() === 6) continue;

    const t = i / days;
    // Drawdown phase ~ days 150-250 (simulates a stress event)
    const drawdownBias = (i > 150 && i < 250) ? -0.0015 : 0;
    // Portfolio: slightly higher mean return, higher vol
    const pRet = 0.00065 + drawdownBias + (rnd() - 0.5) * 0.018;
    // Benchmark (WIG20): lower mean, lower vol
    const bRet = 0.00028 + drawdownBias * 0.7 + (rnd() - 0.5) * 0.012;
    pv *= 1 + pRet;
    bv *= 1 + bRet;
    out.push({
      date: d.toISOString().slice(0, 10),
      portfolio: +pv.toFixed(2),
      benchmark: +bv.toFixed(2),
    });
  }
  return out;
}
const MOCK_EQUITY = buildEquityCurve();

function computeMetricsFromCurve(curve, key) {
  const rets = [];
  for (let i = 1; i < curve.length; i++) {
    rets.push(curve[i][key] / curve[i - 1][key] - 1);
  }
  const total = curve[curve.length - 1][key] / curve[0][key] - 1;
  const years = curve.length / 252;
  const annualized = Math.pow(1 + total, 1 / years) - 1;
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / rets.length;
  const vol = Math.sqrt(variance) * Math.sqrt(252);
  const sharpe = (annualized - 0.045) / vol; // 4.5% risk-free
  // Max drawdown
  let peak = curve[0][key], maxDD = 0, ddStart = 0, ddEnd = 0, currentStart = 0;
  for (let i = 0; i < curve.length; i++) {
    const v = curve[i][key];
    if (v > peak) { peak = v; currentStart = i; }
    const dd = v / peak - 1;
    if (dd < maxDD) { maxDD = dd; ddStart = currentStart; ddEnd = i; }
  }
  const dd_days = ddEnd - ddStart;
  // win rate
  const wins = rets.filter(r => r > 0).length;
  return {
    total_return: total,
    annualized_return: annualized,
    volatility: vol,
    sharpe,
    max_drawdown: maxDD,
    max_drawdown_duration_days: dd_days,
    win_rate: wins / rets.length,
  };
}

const MOCK_BACKTEST = {
  strategy: 'ma_crossover',
  start_date: MOCK_EQUITY[0].date,
  end_date: MOCK_EQUITY[MOCK_EQUITY.length - 1].date,
  metrics: computeMetricsFromCurve(MOCK_EQUITY, 'portfolio'),
  benchmark_metrics: computeMetricsFromCurve(MOCK_EQUITY, 'benchmark'),
  equity_curve: MOCK_EQUITY,
};

// Pre-compute drawdown series for the chart
function buildDrawdownSeries(curve) {
  let peak = curve[0].portfolio;
  return curve.map(p => {
    if (p.portfolio > peak) peak = p.portfolio;
    return {
      date: p.date,
      drawdown: (p.portfolio / peak - 1) * 100,
    };
  });
}
const MOCK_DRAWDOWN = buildDrawdownSeries(MOCK_EQUITY);

/* ----------------------------- AI Analysis ----------------------------- */
const MOCK_ANALYSIS = {
  summary_md: `## Stan portfela — synteza

Portfel utrzymuje **dodatnią ekspozycję na cykl surowcowy** (KGHM) i **konsumencki** (LPP, Allegro), zrównoważoną przez stabilną pozycję energetyczną (PKN Orlen). Ekspozycja sektorowa pozostaje **niedywersyfikowana geograficznie** — 100% w PLN i WIG, co w średnim terminie tworzy ryzyko walutowe.

### Co działa
- **LPP**: Sinsay dostarcza wzrostu zgodnie z tezą; trzymać.
- **KGHM**: cykl miedzi zgodny z założeniami; pozycja powyżej średniego kosztu.
- **PKN Orlen**: stabilizuje portfel, dywidenda kompensuje wahania petrochemii.

### Co warto monitorować
- **CD Projekt**: pozycja pod wodą; ryzyko cyklu wydawniczego do najbliższego dużego release'u. Teza nieuszkodzona, ale wymaga cierpliwości.
- **Allegro**: presja konkurencyjna ze strony Temu i Shein. *Allegro Pay* powinien zrównoważyć — sprawdzaj kwartał do kwartału.

### Rekomendowane działania
1. Rozważ częściowy *rebalance* z LPP do mniej rozgrzanej pozycji (zysk +37% przekracza wagę docelową).
2. Dodaj jedną pozycję defensywną spoza WIG20 (np. infrastruktura, energetyka odnawialna).
3. Ustaw alerty cenowe na CDR (-15% od entry) i na ALE (-15% od entry) — to brama do ponownej oceny tezy.
`,
  thesis_updates: [
    { ticker: 'PKN', assessment: 'Teza zachowana. Marże petrochemii w trendzie, dywidenda potwierdzona.' },
    { ticker: 'CDR', assessment: 'Teza zachowana, ale pod presją. Cierpliwość do następnego dużego releasu.' },
    { ticker: 'KGH', assessment: 'Teza wzmocniona. Cykl miedziowy zgodny z założeniami.' },
    { ticker: 'LPP', assessment: 'Teza zrealizowana częściowo. Rozważ rebalans.' },
    { ticker: 'ALE', assessment: 'Teza zachowana. Monitoruj konkurencję z marketplace\u2019ami z Azji.' },
  ],
};

const MOCK_RISK_ALERTS = [
  { severity: 'high',   title: 'Koncentracja walutowa', description: 'Cały portfel w PLN. Rozważ ekspozycję na USD/EUR — choćby pośrednio przez spółkę z przychodami w walucie.', ticker: undefined },
  { severity: 'high',   title: 'Pozycja CDR pod presją', description: 'CD Projekt -9.3% od entry; brak katalizatora w najbliższym kwartale. Sprawdź wagę względem tezy.', ticker: 'CDR' },
  { severity: 'medium', title: 'Sektor surowcowy zbyt ciężki', description: 'KGHM stanowi >24% portfela. Cykl miedzi w fazie późnej.', ticker: 'KGH' },
  { severity: 'medium', title: 'LPP powyżej wagi docelowej', description: 'Pozycja +37% — przekracza założoną wagę 18% (obecnie ~22%). Rebalans warty rozważenia.', ticker: 'LPP' },
  { severity: 'low',    title: 'Brak ekspozycji defensywnej', description: 'Brak spółek typowo defensywnych (utilities, telco, consumer staples).', ticker: undefined },
  { severity: 'low',    title: 'Wysoki obrót w 30d',         description: '12 transakcji w ostatnich 30 dniach — zweryfikuj koszty transakcyjne.', ticker: undefined },
];

/* ----------------------------- Reports ----------------------------- */
const MOCK_REPORTS = [
  {
    name: 'portfolio_review_2026Q1.md',
    mtime: '2026-04-03T17:22:00+02:00',
    size_bytes: 14_320,
    content_md: `# Przegląd portfela — Q1 2026

## Wynik kwartału
- Portfel: **+8.4%**
- Benchmark (WIG20): **+3.1%**
- Alfa: **+5.3 pp**

## Najlepsi performerzy
1. **KGHM** (+14.2%) — wzrost cen miedzi
2. **LPP** (+11.0%) — wyniki Sinsay powyżej konsensusu

## Najgorsi performerzy
1. **CDR** (-6.8%) — przesunięcie premiery
2. **ALE** (-3.1%) — presja konkurencyjna

## Działania w kwartale
- Rebalans LPP: redukcja o 20% (zysk zrealizowany).
- Dodanie pozycji w *consumer staples* — w planie na Q2.

## Wnioski
Portfel zachowuje się zgodnie z tezą; ekspozycja sektorowa wymaga niewielkiej korekty.`,
  },
  {
    name: 'portfolio_review_2025Q4.md',
    mtime: '2026-01-08T11:05:00+01:00',
    size_bytes:  9_840,
    content_md: `# Przegląd portfela — Q4 2025

## Wynik kwartału
- Portfel: **+12.1%**
- Benchmark (WIG20): **+7.4%**

## Najlepsi performerzy
1. **PKN Orlen** (+9.8%)
2. **LPP** (+15.6%)

## Działania w kwartale
- Brak. Strategia *buy & hold*.

## Wnioski
Mocny kwartał. Cierpliwość w stosunku do CDR opłaca się — pierwsze sygnały odbicia.`,
  },
];

/* ----------------------------- Monitoring ----------------------------- */
const MOCK_MONITORING = {
  generated_at: '2026-05-14T08:30:11+02:00',
  items: [
    { ticker: 'PKN', status: 'on_track', rationale: 'Marże petrochemii w trendzie. Dywidenda potwierdzona. Brak istotnych news flow w 7 dni.' },
    { ticker: 'CDR', status: 'watch',     rationale: 'Brak katalizatora w bliskim terminie. Sentyment analityków stabilny, ale chłodny.' },
    { ticker: 'KGH', status: 'on_track', rationale: 'Cena miedzi $9,180/t, powyżej średniej 90d. Wynik kwartału zgodny z konsensusem.' },
    { ticker: 'LPP', status: 'on_track', rationale: 'Sinsay +28% r/r liczby sklepów. Marża brutto stabilna.' },
    { ticker: 'ALE', status: 'at_risk',  rationale: 'Udział Temu w PL +14% w 30d wg SimilarWeb. Allegro Pay rośnie, ale wolniej od oczekiwań.' },
  ],
  reports: [
    { name: 'monitoring_2026-05-14.html', mtime: '2026-05-14T08:30:11+02:00', size_bytes: 21_400 },
    { name: 'monitoring_2026-05-07.html', mtime: '2026-05-07T08:30:09+02:00', size_bytes: 20_120 },
    { name: 'monitoring_2026-04-30.html', mtime: '2026-04-30T08:30:14+02:00', size_bytes: 19_870 },
    { name: 'monitoring_2026-04-23.html', mtime: '2026-04-23T08:30:08+02:00', size_bytes: 19_310 },
  ],
};

Object.assign(window, {
  MOCK_HOLDINGS,
  MOCK_PORTFOLIO,
  MOCK_BACKTEST,
  MOCK_DRAWDOWN,
  MOCK_ANALYSIS,
  MOCK_RISK_ALERTS,
  MOCK_REPORTS,
  MOCK_MONITORING,
  buildPortfolioStatus,
});
