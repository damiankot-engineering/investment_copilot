/**
 * Investment Copilot — single-file React + TS dashboard for the Polish stock market (GPW).
 *
 * Drop into a Vite + React + Tailwind + shadcn/ui project. Mock data is inline at the
 * top of the file — swap `const MOCK_*` for `await fetch('/api/...')` to wire to a backend.
 *
 * External deps expected to be installed:
 *   - react, react-dom
 *   - tailwindcss (configured)
 *   - shadcn/ui components: card, button, badge, tabs, table, dialog, input, tooltip, skeleton, sonner, textarea, separator
 *   - recharts
 *   - framer-motion
 *   - lucide-react
 *
 * The file is intentionally self-contained — no separate hooks/components files. Split as needed.
 */

import * as React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LineChart, Line, AreaChart, Area, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, ReferenceLine,
  ResponsiveContainer,
} from 'recharts';
import {
  Wallet, BarChart3, Sparkles, FileText, Activity,
  RefreshCw, Edit3, Plus, Trash2, X, Download, Eye, Play,
  AlertTriangle, AlertCircle, Info, Check, Search, Settings, Bell,
  TrendingUp, TrendingDown, Layers, Target, Calendar, Clock,
  ChevronRight, Filter, Globe, ExternalLink, Zap, Gauge,
} from 'lucide-react';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';
import { Toaster, toast } from 'sonner';

/* ───────────────────────────────── Types (mirror backend) ──────────────────────────────── */

export type Holding = {
  ticker: string;
  name?: string;
  shares: number;
  entry_price: number;
  entry_date: string;
  thesis: string;
  keywords: string[];
};
export type HoldingStatus = Holding & {
  last_price: number;
  value: number;
  pnl: number;
  pnl_pct: number;
};
export type PortfolioStatus = {
  holdings: HoldingStatus[];
  total_value: number;
  total_pnl: number;
  total_pnl_pct: number;
  as_of: string;
};
export type BacktestMetrics = {
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe: number;
  max_drawdown: number;
  max_drawdown_duration_days: number;
  win_rate: number;
};
export type EquityPoint = { date: string; portfolio: number; benchmark: number };
export type BacktestResult = {
  strategy: string;
  start_date: string;
  end_date: string;
  metrics: BacktestMetrics;
  benchmark_metrics: BacktestMetrics;
  equity_curve: EquityPoint[];
};
export type RiskAlert = {
  severity: 'low' | 'medium' | 'high';
  title: string;
  description: string;
  ticker?: string;
};
export type PortfolioAnalysis = {
  summary_md: string;
  thesis_updates: { ticker: string; assessment: string }[];
};
export type ReportFile = { name: string; mtime: string; size_bytes: number; content_md?: string };
export type MonitoringItem = {
  ticker: string;
  status: 'on_track' | 'watch' | 'at_risk';
  rationale: string;
};
export type MonitoringReport = {
  generated_at: string;
  items: MonitoringItem[];
  reports: ReportFile[];
};

/* ──────────────────────────────────── Mock data ────────────────────────────────────────── */

const MOCK_HOLDINGS: Holding[] = [
  { ticker: 'PKN', name: 'PKN Orlen',  shares: 120, entry_price:  62.40, entry_date: '2023-04-12',
    thesis: 'Konsolidacja sektora energetycznego po fuzji z Lotosem i PGNiG. Stabilne przepływy z petrochemii, ekspozycja na transformację energetyczną oraz polityka dywidendowa wspierają długoterminową tezę.',
    keywords: ['energia', 'dywidenda', 'transformacja', 'konsolidacja'] },
  { ticker: 'CDR', name: 'CD Projekt', shares:  45, entry_price: 168.00, entry_date: '2023-01-30',
    thesis: 'Pipeline IP (Cyberpunk: Orion, Polaris) + monetyzacja istniejących franczyz. Wysoka marża operacyjna, ryzyko cyklu wydawniczego.',
    keywords: ['gaming', 'IP', 'globalna marka'] },
  { ticker: 'KGH', name: 'KGHM',       shares:  60, entry_price: 124.50, entry_date: '2022-11-08',
    thesis: 'Cykl miedziowy + ekspozycja na elektryfikację. Ryzyko: podatek od wydobycia, koszty energii. Trzymamy do wyższego cyklu cen.',
    keywords: ['surowce', 'miedź', 'cykl', 'elektryfikacja'] },
  { ticker: 'LPP', name: 'LPP',        shares:   8, entry_price: 12300.00, entry_date: '2023-09-22',
    thesis: 'Skala marek (Reserved, Sinsay) + ekspansja w Europie. Sinsay jako motor wzrostu w segmencie ultra fast fashion.',
    keywords: ['retail', 'fast fashion', 'ekspansja', 'Sinsay'] },
  { ticker: 'ALE', name: 'Allegro',    shares: 320, entry_price:  29.80, entry_date: '2024-02-14',
    thesis: 'Marketplace #1 w CEE. Allegro Pay i monetyzacja reklam jako dźwignie marżowe. Konkurencja z Temu/Shein zarządzona.',
    keywords: ['e-commerce', 'platforma', 'reklamy', 'CEE'] },
];

const LAST_PRICES: Record<string, number> = {
  PKN:  74.16, CDR: 152.40, KGH: 153.20, LPP: 16850.00, ALE: 27.55,
};

function computePortfolio(holdings: Holding[]): PortfolioStatus {
  const enriched: HoldingStatus[] = holdings.map(h => {
    const last_price = LAST_PRICES[h.ticker] ?? h.entry_price;
    const value = last_price * h.shares;
    const cost = h.entry_price * h.shares;
    return {
      ...h, last_price, value,
      pnl: value - cost,
      pnl_pct: cost ? (value / cost - 1) * 100 : 0,
    };
  });
  const total_value = enriched.reduce((a, x) => a + x.value, 0);
  const total_cost  = enriched.reduce((a, x) => a + x.entry_price * x.shares, 0);
  return {
    holdings: enriched,
    total_value,
    total_pnl: total_value - total_cost,
    total_pnl_pct: total_cost ? (total_value / total_cost - 1) * 100 : 0,
    as_of: new Date().toISOString(),
  };
}
const MOCK_PORTFOLIO: PortfolioStatus = computePortfolio(MOCK_HOLDINGS);

// Synthesize ~3 years of daily equity points (portfolio outperforming WIG20, visible drawdown)
function buildEquity(): EquityPoint[] {
  let seed = 1337;
  const rnd = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
  const days = 3 * 252;
  const start = new Date('2023-04-01');
  const out: EquityPoint[] = [];
  let pv = 100, bv = 100;
  for (let i = 0; i < days; i++) {
    const d = new Date(start);
    d.setDate(start.getDate() + i);
    if (d.getDay() === 0 || d.getDay() === 6) continue;
    const drawBias = (i > 150 && i < 250) ? -0.0015 : 0;
    pv *= 1 + 0.00065 + drawBias       + (rnd() - 0.5) * 0.018;
    bv *= 1 + 0.00028 + drawBias * 0.7 + (rnd() - 0.5) * 0.012;
    out.push({ date: d.toISOString().slice(0, 10), portfolio: +pv.toFixed(2), benchmark: +bv.toFixed(2) });
  }
  return out;
}
const MOCK_EQUITY: EquityPoint[] = buildEquity();

function metricsOf(curve: EquityPoint[], key: 'portfolio' | 'benchmark'): BacktestMetrics {
  const rets: number[] = [];
  for (let i = 1; i < curve.length; i++) rets.push(curve[i][key] / curve[i - 1][key] - 1);
  const total = curve[curve.length - 1][key] / curve[0][key] - 1;
  const years = curve.length / 252;
  const annualized = Math.pow(1 + total, 1 / years) - 1;
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / rets.length;
  const vol = Math.sqrt(variance) * Math.sqrt(252);
  const sharpe = (annualized - 0.045) / vol;
  let peak = curve[0][key], maxDD = 0, ddStart = 0, ddEnd = 0, currentStart = 0;
  for (let i = 0; i < curve.length; i++) {
    const v = curve[i][key];
    if (v > peak) { peak = v; currentStart = i; }
    const dd = v / peak - 1;
    if (dd < maxDD) { maxDD = dd; ddStart = currentStart; ddEnd = i; }
  }
  return {
    total_return: total,
    annualized_return: annualized,
    volatility: vol,
    sharpe,
    max_drawdown: maxDD,
    max_drawdown_duration_days: ddEnd - ddStart,
    win_rate: rets.filter(r => r > 0).length / rets.length,
  };
}
const MOCK_BACKTEST: BacktestResult = {
  strategy: 'ma_crossover',
  start_date: MOCK_EQUITY[0].date,
  end_date:   MOCK_EQUITY[MOCK_EQUITY.length - 1].date,
  metrics:           metricsOf(MOCK_EQUITY, 'portfolio'),
  benchmark_metrics: metricsOf(MOCK_EQUITY, 'benchmark'),
  equity_curve: MOCK_EQUITY,
};
function buildDrawdown(curve: EquityPoint[]) {
  let peak = curve[0].portfolio;
  return curve.map(p => {
    if (p.portfolio > peak) peak = p.portfolio;
    return { date: p.date, drawdown: (p.portfolio / peak - 1) * 100 };
  });
}
const MOCK_DRAWDOWN = buildDrawdown(MOCK_EQUITY);

const MOCK_ANALYSIS: PortfolioAnalysis = {
  summary_md: `## Stan portfela — synteza

Portfel utrzymuje **dodatnią ekspozycję na cykl surowcowy** (KGHM) i **konsumencki** (LPP, Allegro), zrównoważoną przez stabilną pozycję energetyczną (PKN Orlen). Ekspozycja sektorowa pozostaje **niedywersyfikowana geograficznie** — 100% w PLN i WIG, co w średnim terminie tworzy ryzyko walutowe.

### Co działa
- **LPP**: Sinsay dostarcza wzrostu zgodnie z tezą; trzymać.
- **KGHM**: cykl miedzi zgodny z założeniami; pozycja powyżej średniego kosztu.
- **PKN Orlen**: stabilizuje portfel, dywidenda kompensuje wahania petrochemii.

### Co warto monitorować
- **CD Projekt**: pozycja pod wodą; ryzyko cyklu wydawniczego do najbliższego dużego release'u.
- **Allegro**: presja konkurencyjna ze strony Temu i Shein.

### Rekomendowane działania
1. Częściowy rebalans z LPP do mniej rozgrzanej pozycji (zysk +37% przekracza wagę docelową).
2. Dodaj jedną pozycję defensywną spoza WIG20.
3. Ustaw alerty cenowe na CDR (-15% od entry) i ALE (-15% od entry).`,
  thesis_updates: [
    { ticker: 'PKN', assessment: 'Teza zachowana. Marże petrochemii w trendzie, dywidenda potwierdzona.' },
    { ticker: 'CDR', assessment: 'Teza zachowana, ale pod presją. Cierpliwość do następnego releasu.' },
    { ticker: 'KGH', assessment: 'Teza wzmocniona. Cykl miedziowy zgodny z założeniami.' },
    { ticker: 'LPP', assessment: 'Teza zrealizowana częściowo. Rozważ rebalans.' },
    { ticker: 'ALE', assessment: 'Teza zachowana. Monitoruj konkurencję z marketplace\u2019ami z Azji.' },
  ],
};

const MOCK_RISK_ALERTS: RiskAlert[] = [
  { severity: 'high',   title: 'Koncentracja walutowa', description: 'Cały portfel w PLN. Rozważ ekspozycję na USD/EUR.' },
  { severity: 'high',   title: 'Pozycja CDR pod presją', description: 'CD Projekt -9.3% od entry; brak katalizatora w najbliższym kwartale.', ticker: 'CDR' },
  { severity: 'medium', title: 'Sektor surowcowy zbyt ciężki', description: 'KGHM stanowi >24% portfela. Cykl miedzi w fazie późnej.', ticker: 'KGH' },
  { severity: 'medium', title: 'LPP powyżej wagi docelowej', description: 'Pozycja +37% — przekracza założoną wagę 18% (obecnie ~22%).', ticker: 'LPP' },
  { severity: 'low',    title: 'Brak ekspozycji defensywnej', description: 'Brak spółek typowo defensywnych (utilities, telco, consumer staples).' },
  { severity: 'low',    title: 'Wysoki obrót w 30d', description: '12 transakcji w ostatnich 30 dniach — zweryfikuj koszty transakcyjne.' },
];

const MOCK_REPORTS: ReportFile[] = [
  { name: 'portfolio_review_2026Q1.md', mtime: '2026-04-03T17:22:00+02:00', size_bytes: 14_320,
    content_md: `# Przegląd portfela — Q1 2026\n\n## Wynik kwartału\n- Portfel: **+8.4%**\n- Benchmark (WIG20): **+3.1%**\n- Alfa: **+5.3 pp**\n\n## Najlepsi performerzy\n1. **KGHM** (+14.2%)\n2. **LPP** (+11.0%)\n\n## Najgorsi performerzy\n1. **CDR** (-6.8%)\n2. **ALE** (-3.1%)` },
  { name: 'portfolio_review_2025Q4.md', mtime: '2026-01-08T11:05:00+01:00', size_bytes: 9_840,
    content_md: `# Przegląd portfela — Q4 2025\n\n## Wynik kwartału\n- Portfel: **+12.1%**\n- Benchmark (WIG20): **+7.4%**` },
];

const MOCK_MONITORING: MonitoringReport = {
  generated_at: '2026-05-14T08:30:11+02:00',
  items: [
    { ticker: 'PKN', status: 'on_track', rationale: 'Marże petrochemii w trendzie. Dywidenda potwierdzona.' },
    { ticker: 'CDR', status: 'watch',    rationale: 'Brak katalizatora w bliskim terminie. Sentyment analityków stabilny, ale chłodny.' },
    { ticker: 'KGH', status: 'on_track', rationale: 'Cena miedzi $9,180/t, powyżej średniej 90d.' },
    { ticker: 'LPP', status: 'on_track', rationale: 'Sinsay +28% r/r liczby sklepów. Marża brutto stabilna.' },
    { ticker: 'ALE', status: 'at_risk',  rationale: 'Udział Temu w PL +14% w 30d. Allegro Pay rośnie wolniej od oczekiwań.' },
  ],
  reports: [
    { name: 'monitoring_2026-05-14.html', mtime: '2026-05-14T08:30:11+02:00', size_bytes: 21_400 },
    { name: 'monitoring_2026-05-07.html', mtime: '2026-05-07T08:30:09+02:00', size_bytes: 20_120 },
    { name: 'monitoring_2026-04-30.html', mtime: '2026-04-30T08:30:14+02:00', size_bytes: 19_870 },
    { name: 'monitoring_2026-04-23.html', mtime: '2026-04-23T08:30:08+02:00', size_bytes: 19_310 },
  ],
};

/* ─────────────────────────────────── Formatters ────────────────────────────────────────── */

const fmtPLN = (n: number, opts: { decimals?: number; signed?: boolean } = {}) => {
  const { decimals = 0, signed = false } = opts;
  const sign = signed && n > 0 ? '+' : '';
  return sign + new Intl.NumberFormat('pl-PL', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).format(n) + ' zł';
};
const fmtPct = (n: number, opts: { decimals?: number; signed?: boolean } = {}) => {
  const { decimals = 2, signed = false } = opts;
  return (signed && n > 0 ? '+' : '') + n.toFixed(decimals) + '%';
};
const fmtNum = (n: number, decimals = 0) =>
  new Intl.NumberFormat('pl-PL', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).format(n);
const fmtBytes = (b: number) =>
  b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(2)} MB`;
const fmtRelTime = (iso: string) => {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60)    return 'przed chwilą';
  if (diff < 3600)  return `${Math.round(diff / 60)} min temu`;
  if (diff < 86400) return `${Math.round(diff / 3600)} godz. temu`;
  return `${Math.round(diff / 86400)} dni temu`;
};
const fmtDateTime = (iso: string) =>
  new Date(iso).toLocaleString('pl-PL', { dateStyle: 'medium', timeStyle: 'short' });
const fmtDate = (iso: string) =>
  new Date(iso).toLocaleDateString('pl-PL', { dateStyle: 'medium' });

/* ───────────────────────────────────── Helpers ─────────────────────────────────────────── */

function CountUp({ value, decimals = 0, duration = 1.1, format, prefix = '', suffix = '' }: {
  value: number; decimals?: number; duration?: number;
  format?: (v: number) => string; prefix?: string; suffix?: string;
}) {
  const ref = React.useRef<HTMLSpanElement>(null);
  const prev = React.useRef(0);
  React.useEffect(() => {
    const node = ref.current; if (!node) return;
    const from = prev.current, to = Number(value) || 0;
    const start = performance.now(); const ms = duration * 1000;
    let raf = 0;
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const eased = 1 - Math.pow(1 - t, 3);
      const v = from + (to - from) * eased;
      node.textContent = prefix + (format ? format(v) : v.toFixed(decimals)) + suffix;
      if (t < 1) raf = requestAnimationFrame(step); else prev.current = to;
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value, decimals, duration, format, prefix, suffix]);
  return <span ref={ref} className="tabular-nums">{prefix}0{suffix}</span>;
}

/** Tiny markdown renderer (h1/h2/h3, p, ul, **bold**, *em*, `code`). */
function Markdown({ children }: { children: string }) {
  const html = React.useMemo(() => {
    const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const inline = (s: string) =>
      esc(s)
        .replace(/`([^`]+)`/g, '<code class="rounded bg-white/5 px-1 py-0.5 text-[0.85em]">$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong class="text-foreground">$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em class="text-violet-300">$1</em>');
    let out = '', inList = false;
    const flush = () => { if (inList) { out += '</ul>'; inList = false; } };
    for (const raw of children.split('\n')) {
      const line = raw.trimEnd();
      if (/^###\s/.test(line)) { flush(); out += `<h3 class="mt-3 mb-1 text-[15px] font-semibold">${inline(line.replace(/^###\s/, ''))}</h3>`; continue; }
      if (/^##\s/.test(line))  { flush(); out += `<h2 class="mt-4 mb-1.5 text-base font-semibold">${inline(line.replace(/^##\s/, ''))}</h2>`; continue; }
      if (/^#\s/.test(line))   { flush(); out += `<h1 class="mb-2 text-lg font-semibold">${inline(line.replace(/^#\s/, ''))}</h1>`; continue; }
      if (/^\s*[-*]\s/.test(line))       { if (!inList) { out += '<ul class="ml-5 list-disc space-y-1">'; inList = true; } out += `<li>${inline(line.replace(/^\s*[-*]\s/, ''))}</li>`; continue; }
      if (/^\s*\d+\.\s/.test(line))      { if (!inList) { out += '<ul class="ml-5 list-disc space-y-1">'; inList = true; } out += `<li>${inline(line.replace(/^\s*\d+\.\s/, ''))}</li>`; continue; }
      if (line.trim() === '') { flush(); continue; }
      flush(); out += `<p class="my-2 leading-relaxed text-muted-foreground">${inline(line)}</p>`;
    }
    flush();
    return out;
  }, [children]);
  return <div className="text-[13.5px]" dangerouslySetInnerHTML={{ __html: html }} />;
}

/* ──────────────────────────────────── Tab views ────────────────────────────────────────── */

function KpiCard(props: {
  label: string; value: number; format?: (v: number) => string;
  decimals?: number; suffix?: string; sub?: React.ReactNode;
  hint?: string; accent: 'cyan' | 'violet' | 'green' | 'red'; delay?: number;
}) {
  const accentBar = {
    cyan:   'via-cyan-400/60',
    violet: 'via-violet-400/60',
    green:  'via-emerald-400/60',
    red:    'via-rose-400/60',
  }[props.accent];
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2 }}
      transition={{ delay: props.delay ?? 0, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <Card className="relative overflow-hidden border-white/5 bg-gradient-to-b from-white/[0.04] to-white/[0.01] backdrop-blur">
        <span className={`pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent ${accentBar} to-transparent`} />
        <CardContent className="relative px-5 py-4">
          <div className="flex items-center justify-between">
            <span className="text-[10.5px] uppercase tracking-[0.16em] text-muted-foreground">{props.label}</span>
            {props.hint && <span className="font-mono text-[10.5px] text-muted-foreground/70">{props.hint}</span>}
          </div>
          <div className="mt-3 text-[26px] font-semibold leading-none tracking-tight">
            <CountUp value={props.value} decimals={props.decimals ?? 0} format={props.format} suffix={props.suffix} />
          </div>
          {props.sub && <div className="mt-2 text-[12px] text-muted-foreground">{props.sub}</div>}
        </CardContent>
      </Card>
    </motion.div>
  );
}

function PnlBar({ pct, max }: { pct: number; max: number }) {
  const w = Math.min(Math.abs(pct) / max, 1) * 100;
  const positive = pct >= 0;
  return (
    <div className="relative h-1.5 w-20 overflow-hidden rounded-full bg-white/5">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${w}%` }}
        transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        className={`absolute top-0 h-full rounded-full ${
          positive
            ? 'left-1/2 bg-gradient-to-r from-emerald-500/40 to-emerald-400'
            : 'right-1/2 bg-gradient-to-l from-rose-500/40 to-rose-400'
        }`}
      />
      <div className="absolute inset-y-0 left-1/2 w-px bg-white/15" />
    </div>
  );
}

function EditPortfolioDialog({
  open, onOpenChange, holdings, onSave,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  holdings: HoldingStatus[];
  onSave: (rows: Holding[]) => void;
}) {
  const [rows, setRows] = React.useState<Holding[]>(() => holdings.map(h => ({ ...h })));
  React.useEffect(() => { if (open) setRows(holdings.map(h => ({ ...h }))); }, [open, holdings]);

  const update = (i: number, patch: Partial<Holding>) =>
    setRows(r => r.map((x, j) => j === i ? { ...x, ...patch } : x));
  const remove = (i: number) => setRows(r => r.filter((_, j) => j !== i));
  const add = () =>
    setRows(r => [...r, {
      ticker: '', name: '', shares: 0, entry_price: 0,
      entry_date: new Date().toISOString().slice(0, 10),
      thesis: '', keywords: [],
    }]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] max-w-4xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edytuj portfel</DialogTitle>
          <DialogDescription>Pozycje, tezy i słowa kluczowe</DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3 py-2">
          {rows.map((r, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03 }}
              className="rounded-xl border border-white/5 bg-white/[0.025] p-4"
            >
              <div className="grid grid-cols-12 gap-2">
                <div className="col-span-2">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">Ticker</label>
                  <Input value={r.ticker} onChange={e => update(i, { ticker: e.target.value.toUpperCase() })} placeholder="PKN" className="mt-1 font-mono" />
                </div>
                <div className="col-span-4">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">Nazwa</label>
                  <Input value={r.name ?? ''} onChange={e => update(i, { name: e.target.value })} placeholder="PKN Orlen" className="mt-1" />
                </div>
                <div className="col-span-2">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">Akcje</label>
                  <Input type="number" value={r.shares} onChange={e => update(i, { shares: Number(e.target.value) })} className="mt-1" />
                </div>
                <div className="col-span-2">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">Entry</label>
                  <Input type="number" step="0.01" value={r.entry_price} onChange={e => update(i, { entry_price: Number(e.target.value) })} className="mt-1" />
                </div>
                <div className="col-span-2">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">Data</label>
                  <Input type="date" value={r.entry_date} onChange={e => update(i, { entry_date: e.target.value })} className="mt-1" />
                </div>
              </div>

              <div className="mt-3">
                <label className="text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">Teza inwestycyjna</label>
                <Textarea
                  rows={2}
                  value={r.thesis}
                  onChange={e => update(i, { thesis: e.target.value })}
                  placeholder="Krótkie uzasadnienie…"
                  className="mt-1"
                />
              </div>

              <div className="mt-3 flex items-end justify-between gap-3">
                <div className="flex-1">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">Słowa kluczowe</label>
                  <KeywordsInput value={r.keywords} onChange={v => update(i, { keywords: v })} />
                </div>
                <Button variant="ghost" size="icon" onClick={() => remove(i)} title="Usuń pozycję">
                  <Trash2 className="size-4 text-rose-400" />
                </Button>
              </div>
            </motion.div>
          ))}

          <button
            type="button"
            onClick={add}
            className="flex h-12 items-center justify-center gap-2 rounded-xl border border-dashed border-white/15 text-[13px] text-muted-foreground transition hover:border-violet-400/50 hover:text-foreground"
          >
            <Plus className="size-4" /> Dodaj pozycję
          </button>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>Anuluj</Button>
          <Button onClick={() => onSave(rows)}><Check className="mr-1.5 size-4" /> Zapisz zmiany</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function KeywordsInput({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  const [draft, setDraft] = React.useState('');
  const commit = () => {
    const v = draft.trim();
    if (!v || value.includes(v)) return;
    onChange([...value, v]); setDraft('');
  };
  return (
    <div className="mt-1 flex flex-wrap items-center gap-1.5 rounded-md border border-white/10 bg-white/[0.03] p-1.5">
      {value.map((k, i) => (
        <span key={k + i} className="inline-flex items-center gap-1 rounded-md border border-violet-400/30 bg-violet-400/10 py-0.5 pl-2 pr-1 text-[11.5px] text-violet-300">
          {k}
          <button type="button" onClick={() => onChange(value.filter((_, j) => j !== i))} className="rounded p-0.5 hover:bg-white/10">
            <X className="size-3" />
          </button>
        </span>
      ))}
      <input
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); commit(); }
          if (e.key === 'Backspace' && !draft && value.length) onChange(value.slice(0, -1));
        }}
        onBlur={commit}
        placeholder={value.length ? '' : 'Dodaj słowo kluczowe…'}
        className="min-w-[140px] flex-1 bg-transparent px-1.5 text-[12.5px] outline-none placeholder:text-muted-foreground/60"
      />
    </div>
  );
}

function PortfolioView({ portfolio, onRefresh, refreshing, onSave }: {
  portfolio: PortfolioStatus;
  onRefresh: () => void;
  refreshing: boolean;
  onSave: (rows: Holding[]) => void;
}) {
  const [editOpen, setEditOpen] = React.useState(false);
  const maxAbs = Math.max(...portfolio.holdings.map(h => Math.abs(h.pnl_pct)), 5);

  const palette = ['#22d3ee', '#a78bfa', '#f0abfc', '#34d399', '#fbbf24'];
  const allocData = portfolio.holdings.map((h, i) => ({
    name: h.ticker, value: h.value, fill: palette[i % palette.length],
  }));

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between gap-4">
        <div>
          <div className="text-[10.5px] uppercase tracking-[0.16em] text-muted-foreground">Przegląd</div>
          <h2 className="mt-1 text-[20px] font-semibold tracking-tight">Portfel</h2>
          <p className="mt-1 text-[13px] text-muted-foreground">
            Aktualny stan pozycji, wycena rynkowa i wynik względem ceny wejścia.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => setEditOpen(true)}>
            <Edit3 className="mr-1.5 size-4" /> Edytuj portfel
          </Button>
          <Button onClick={onRefresh} disabled={refreshing}>
            <RefreshCw className={`mr-1.5 size-4 ${refreshing ? 'animate-spin' : ''}`} /> Odśwież dane rynkowe
          </Button>
        </div>
      </header>

      <AnimatePresence>
        {refreshing && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="relative h-1 overflow-hidden rounded-full bg-white/5"
          >
            <motion.div
              className="absolute top-0 bottom-0 w-1/3 bg-gradient-to-r from-cyan-400 to-violet-400"
              initial={{ x: '-100%' }} animate={{ x: '300%' }}
              transition={{ repeat: Infinity, duration: 1.4, ease: 'easeInOut' }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          label="Wartość portfela"
          value={portfolio.total_value}
          format={v => fmtPLN(v)} hint="PLN" accent="cyan"
          sub={`Suma wartości rynkowej ${portfolio.holdings.length} pozycji`}
          delay={0}
        />
        <KpiCard
          label="PnL łącznie"
          value={portfolio.total_pnl}
          format={v => fmtPLN(v, { signed: true })} hint="PLN"
          accent={portfolio.total_pnl >= 0 ? 'green' : 'red'}
          sub={<span className={portfolio.total_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
            {portfolio.total_pnl >= 0 ? '↑' : '↓'} względem kosztu wejścia
          </span>}
          delay={0.06}
        />
        <KpiCard
          label="PnL %"
          value={portfolio.total_pnl_pct} decimals={2}
          format={v => fmtPct(v, { signed: true, decimals: 2 })} hint="zwrot"
          accent={portfolio.total_pnl_pct >= 0 ? 'green' : 'red'}
          sub={<span className="text-muted-foreground">
            vs. WIG20 +{(portfolio.total_pnl_pct - 4.8).toFixed(2)} pp
          </span>}
          delay={0.12}
        />
        <KpiCard
          label="Ostatnia aktualizacja"
          value={0}
          format={() => fmtRelTime(portfolio.as_of)}
          hint="market data" accent="violet"
          sub={<span className="font-mono text-muted-foreground">{fmtDateTime(portfolio.as_of)}</span>}
          delay={0.18}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_360px]">
        <Card className="overflow-hidden border-white/5 bg-gradient-to-b from-white/[0.04] to-white/[0.01] backdrop-blur">
          <div className="flex items-center justify-between px-5 pb-3 pt-4">
            <div>
              <div className="text-[15px] font-medium tracking-tight">Pozycje</div>
              <div className="mt-0.5 text-[12px] text-muted-foreground">
                Stan na {fmtDateTime(portfolio.as_of)}
              </div>
            </div>
            <div className="flex items-center gap-2 text-[11.5px] text-muted-foreground">
              <Filter className="size-3" /> Wszystkie
            </div>
          </div>
          <Table>
            <TableHeader>
              <TableRow className="border-y border-white/5 hover:bg-transparent">
                <TableHead className="text-[10.5px] uppercase tracking-[0.14em]">Ticker</TableHead>
                <TableHead className="text-[10.5px] uppercase tracking-[0.14em]">Nazwa</TableHead>
                <TableHead className="text-right text-[10.5px] uppercase tracking-[0.14em]">Akcje</TableHead>
                <TableHead className="text-right text-[10.5px] uppercase tracking-[0.14em]">Entry</TableHead>
                <TableHead className="text-right text-[10.5px] uppercase tracking-[0.14em]">Ostatnia</TableHead>
                <TableHead className="text-right text-[10.5px] uppercase tracking-[0.14em]">Wartość</TableHead>
                <TableHead className="text-right text-[10.5px] uppercase tracking-[0.14em]">PnL</TableHead>
                <TableHead className="text-right text-[10.5px] uppercase tracking-[0.14em]">PnL %</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {portfolio.holdings.map((h, i) => {
                const pos = h.pnl >= 0;
                return (
                  <motion.tr
                    key={h.ticker}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.08 + i * 0.05 }}
                    whileHover={{ y: -1, backgroundColor: 'rgba(255,255,255,0.025)' }}
                    className="group border-b border-white/[0.03] last:border-0"
                  >
                    <TableCell>
                      <div className="flex items-center gap-2.5">
                        <div className="flex size-7 items-center justify-center rounded-md border border-white/5 bg-white/[0.04] font-mono text-[10.5px]">
                          {h.ticker.slice(0, 3)}
                        </div>
                        <span className="font-mono text-[12.5px] font-semibold">{h.ticker}</span>
                      </div>
                    </TableCell>
                    <TableCell className="text-[12.5px] text-foreground/85">{h.name}</TableCell>
                    <TableCell className="text-right font-mono text-[12.5px]">{fmtNum(h.shares)}</TableCell>
                    <TableCell className="text-right font-mono text-[12.5px] text-muted-foreground">{fmtPLN(h.entry_price, { decimals: 2 })}</TableCell>
                    <TableCell className="text-right font-mono text-[12.5px]">{fmtPLN(h.last_price, { decimals: 2 })}</TableCell>
                    <TableCell className="text-right font-mono text-[12.5px]">{fmtPLN(h.value)}</TableCell>
                    <TableCell>
                      <div className="flex items-center justify-end gap-2.5">
                        <PnlBar pct={h.pnl_pct} max={maxAbs} />
                        <span className={`font-mono text-[12.5px] ${pos ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {fmtPLN(h.pnl, { signed: true })}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className={`inline-flex items-center gap-1 font-mono text-[12.5px] font-semibold ${pos ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {pos ? <TrendingUp className="size-3" /> : <TrendingDown className="size-3" />}
                        {fmtPct(h.pnl_pct, { signed: true })}
                      </span>
                    </TableCell>
                  </motion.tr>
                );
              })}
            </TableBody>
          </Table>
        </Card>

        <Card className="border-white/5 bg-gradient-to-b from-white/[0.04] to-white/[0.01] backdrop-blur">
          <CardContent className="px-5 py-4">
            <div className="mb-2 flex items-center justify-between">
              <div>
                <div className="text-[10.5px] uppercase tracking-[0.16em] text-muted-foreground">Alokacja</div>
                <div className="mt-0.5 text-[15px] font-medium">Według pozycji</div>
              </div>
              <Badge variant="secondary"><Layers className="mr-1 size-3" />{portfolio.holdings.length} pozycji</Badge>
            </div>
            <div className="flex items-center gap-4">
              <div className="relative size-[150px] shrink-0">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={allocData} dataKey="value" innerRadius={48} outerRadius={70} paddingAngle={2} stroke="#0a0a0f" strokeWidth={2}>
                      {allocData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                    </Pie>
                    <RTooltip
                      contentStyle={{ background: 'rgba(15,15,22,0.95)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }}
                      formatter={(v: number) => fmtPLN(v)}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
                  <div className="text-[10.5px] uppercase tracking-[0.16em] text-muted-foreground">Razem</div>
                  <div className="mt-0.5 font-mono text-[13px] font-semibold">{fmtPLN(portfolio.total_value)}</div>
                </div>
              </div>
              <div className="grid min-w-0 flex-1 grid-cols-1 gap-1.5">
                {allocData.map(d => {
                  const pct = (d.value / portfolio.total_value) * 100;
                  return (
                    <div key={d.name} className="flex items-center gap-2 text-[12px]">
                      <span className="size-2 shrink-0 rounded-sm" style={{ background: d.fill }} />
                      <span className="w-9 font-mono">{d.name}</span>
                      <span className="flex-1 truncate text-muted-foreground">{fmtPLN(d.value)}</span>
                      <span className="font-mono tabular-nums">{pct.toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <EditPortfolioDialog
        open={editOpen}
        onOpenChange={setEditOpen}
        holdings={portfolio.holdings}
        onSave={(rows) => { onSave(rows); setEditOpen(false); toast.success('Portfel zapisany', { description: `${rows.length} pozycji zaktualizowanych.` }); }}
      />
    </div>
  );
}

function MetricCard(props: {
  label: string; value: number; format?: (v: number) => string;
  decimals?: number; hint?: string;
  accent: 'green' | 'red' | 'violet' | 'cyan' | 'amber'; delay: number;
}) {
  const c = {
    green: 'text-emerald-400', red: 'text-rose-400',
    violet: 'text-violet-300', cyan: 'text-cyan-300', amber: 'text-amber-300',
  }[props.accent];
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: props.delay, duration: 0.45 }}
      className="rounded-xl border border-white/5 bg-white/[0.025] px-4 py-3.5"
    >
      <div className="text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">{props.label}</div>
      <div className={`mt-1.5 font-mono text-[20px] font-semibold tracking-tight ${c}`}>
        <CountUp value={props.value} decimals={props.decimals ?? 2} format={props.format} />
      </div>
      {props.hint && <div className="mt-0.5 text-[11px] text-muted-foreground/70">{props.hint}</div>}
    </motion.div>
  );
}

function BacktestView() {
  const [strategy, setStrategy] = React.useState<'ma_crossover' | 'momentum'>('ma_crossover');
  const [running, setRunning] = React.useState(false);
  const [result, setResult] = React.useState<BacktestResult | null>(MOCK_BACKTEST);
  const [animKey, setAnimKey] = React.useState(0);

  const run = async () => {
    setRunning(true); setResult(null);
    await new Promise(r => setTimeout(r, 1400));
    setResult({ ...MOCK_BACKTEST, strategy });
    setAnimKey(k => k + 1);
    setRunning(false);
    toast.success('Backtest gotowy', { description: `Strategia ${strategy === 'ma_crossover' ? 'MA Crossover' : 'Momentum'}.` });
  };

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between gap-4">
        <div>
          <div className="text-[10.5px] uppercase tracking-[0.16em] text-muted-foreground">Symulacja historyczna</div>
          <h2 className="mt-1 text-[20px] font-semibold tracking-tight">Backtest strategii</h2>
          <p className="mt-1 text-[13px] text-muted-foreground">Porównaj wynik strategii względem benchmarku WIG20 w oknie 3 lat.</p>
        </div>
        <div className="flex items-center gap-3">
          <Tabs value={strategy} onValueChange={(v) => setStrategy(v as 'ma_crossover' | 'momentum')}>
            <TabsList>
              <TabsTrigger value="ma_crossover"><Gauge className="mr-1.5 size-3.5" /> MA Crossover</TabsTrigger>
              <TabsTrigger value="momentum"><Zap className="mr-1.5 size-3.5" /> Momentum</TabsTrigger>
            </TabsList>
          </Tabs>
          <Button onClick={run} disabled={running}>
            <Play className={`mr-1.5 size-4 ${running ? 'animate-pulse' : ''}`} /> Uruchom backtest
          </Button>
        </div>
      </header>

      <div className="-mt-2 flex items-center gap-2 text-[12px] text-muted-foreground">
        <Calendar className="size-3.5" />
        <span>
          Zakres:{' '}
          <span className="font-mono text-foreground/70">{result ? fmtDate(result.start_date) : '—'}</span>{' '}→{' '}
          <span className="font-mono text-foreground/70">{result ? fmtDate(result.end_date) : '—'}</span>{' '}·{' '}
          {result ? `${result.equity_curve.length} dni handlowych` : '—'}
        </span>
        <Badge variant="outline" className="ml-2">
          <Target className="mr-1 size-3" />
          {strategy === 'ma_crossover' ? 'Średnie kroczące 50/200' : 'Top-N momentum 12-1m'}
        </Badge>
      </div>

      <Card className="border-white/5 bg-gradient-to-b from-white/[0.04] to-white/[0.01] backdrop-blur">
        <CardContent className="px-5 py-4">
          <div className="mb-3 flex items-end justify-between">
            <div>
              <div className="text-[15px] font-medium">Krzywa kapitału</div>
              <div className="mt-0.5 text-[12px] text-muted-foreground">Portfel vs. WIG20 (start = 100)</div>
            </div>
            <div className="flex items-center gap-4 text-[12px]">
              <span className="inline-flex items-center gap-1.5">
                <span className="block h-1.5 w-3 rounded-sm bg-gradient-to-r from-cyan-400 to-violet-400" />Portfel
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="block h-1.5 w-3 rounded-sm bg-white/40" />WIG20
              </span>
            </div>
          </div>
          {running || !result ? (
            <Skeleton className="h-[320px] w-full" />
          ) : (
            <div className="h-[320px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={result.equity_curve} margin={{ top: 12, right: 20, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="lineP" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%"   stopColor="#67e8f9" />
                      <stop offset="100%" stopColor="#a78bfa" />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="2 4" vertical={false} stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="date" tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}
                    tickFormatter={(d: string) => d.slice(0, 7)}
                    interval={Math.floor(result.equity_curve.length / 8)}
                    tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} width={48}
                    tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}
                    tickFormatter={(v: number) => v.toFixed(0)}
                    domain={['dataMin - 5', 'dataMax + 5']} />
                  <RTooltip contentStyle={{ background: 'rgba(15,15,22,0.95)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12 }} />
                  <Line key={`b-${animKey}`} type="monotone" dataKey="benchmark" stroke="rgba(255,255,255,0.4)" strokeWidth={1.5} dot={false} animationDuration={1200} />
                  <Line key={`p-${animKey}`} type="monotone" dataKey="portfolio" stroke="url(#lineP)" strokeWidth={2.25} dot={false} animationDuration={1400} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border-white/5 bg-gradient-to-b from-white/[0.04] to-white/[0.01] backdrop-blur">
        <CardContent className="px-5 py-4">
          <div className="mb-2 flex items-end justify-between">
            <div>
              <div className="text-[15px] font-medium">Drawdown</div>
              <div className="mt-0.5 text-[12px] text-muted-foreground">Spadek od ostatniego szczytu (%)</div>
            </div>
            {result && (
              <Badge variant="destructive">
                <AlertTriangle className="mr-1 size-3" />
                Max DD {(result.metrics.max_drawdown * 100).toFixed(1)}% · {result.metrics.max_drawdown_duration_days} dni
              </Badge>
            )}
          </div>
          {running || !result ? <Skeleton className="h-[140px] w-full" /> : (
            <div className="h-[140px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={MOCK_DRAWDOWN} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%"   stopColor="#f87171" stopOpacity="0.55" />
                      <stop offset="100%" stopColor="#f87171" stopOpacity="0.02" />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="2 4" vertical={false} stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="date" tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}
                    tickFormatter={(d: string) => d.slice(0, 7)}
                    interval={Math.floor(MOCK_DRAWDOWN.length / 8)}
                    tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} width={48}
                    tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}
                    tickFormatter={(v: number) => `${v.toFixed(0)}%`} domain={['dataMin', 0]} />
                  <RTooltip contentStyle={{ background: 'rgba(15,15,22,0.95)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12 }} />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
                  <Area key={`d-${animKey}`} type="monotone" dataKey="drawdown" stroke="#f87171" strokeWidth={1.5} fill="url(#ddFill)" animationDuration={1200} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        {result ? (
          <>
            <MetricCard label="Zwrot łączny"  value={result.metrics.total_return * 100} decimals={1} format={v => fmtPct(v, { signed: true, decimals: 1 })} hint={`WIG20 ${fmtPct(result.benchmark_metrics.total_return * 100, { signed: true, decimals: 1 })}`} accent={result.metrics.total_return >= 0 ? 'green' : 'red'} delay={0.05} />
            <MetricCard label="Zwrot roczny"  value={result.metrics.annualized_return * 100} decimals={1} format={v => fmtPct(v, { signed: true, decimals: 1 })} hint="CAGR" accent="violet" delay={0.10} />
            <MetricCard label="Zmienność"     value={result.metrics.volatility * 100} decimals={1} format={v => fmtPct(v, { decimals: 1 })} hint="ann." accent="cyan"  delay={0.15} />
            <MetricCard label="Sharpe"        value={result.metrics.sharpe} hint="rf = 4.5%" accent={result.metrics.sharpe >= 1 ? 'green' : result.metrics.sharpe >= 0 ? 'amber' : 'red'} delay={0.20} />
            <MetricCard label="Max drawdown"  value={result.metrics.max_drawdown * 100} decimals={1} format={v => fmtPct(v, { decimals: 1 })} hint={`${result.metrics.max_drawdown_duration_days} dni`} accent="red" delay={0.25} />
            <MetricCard label="Win rate"      value={result.metrics.win_rate * 100} decimals={1} format={v => fmtPct(v, { decimals: 1 })} hint="dni dodatnie" accent="green" delay={0.30} />
          </>
        ) : (
          Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-[80px] w-full" />)
        )}
      </div>
    </div>
  );
}

function AnalysisView() {
  const [loading, setLoading] = React.useState(false);
  const [data, setData] = React.useState<{ summary: PortfolioAnalysis; alerts: RiskAlert[] } | null>({ summary: MOCK_ANALYSIS, alerts: MOCK_RISK_ALERTS });

  const run = async () => {
    setLoading(true); setData(null);
    await new Promise(r => setTimeout(r, 1500));
    setData({ summary: MOCK_ANALYSIS, alerts: MOCK_RISK_ALERTS });
    setLoading(false);
    toast.success('Analiza gotowa', { description: 'Wygenerowano podsumowanie i alerty.' });
  };
  const counts = data ? data.alerts.reduce((a, x) => ({ ...a, [x.severity]: (a[x.severity] ?? 0) + 1 }), {} as Record<string, number>) : {};

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between gap-4">
        <div>
          <div className="text-[10.5px] uppercase tracking-[0.16em] text-muted-foreground">LLM / Polski</div>
          <h2 className="mt-1 text-[20px] font-semibold tracking-tight">Analiza AI</h2>
          <p className="mt-1 text-[13px] text-muted-foreground">Podsumowanie portfela, ocena tez i alerty ryzyka generowane przez model językowy.</p>
        </div>
        <Button onClick={run} disabled={loading}>
          <Sparkles className={`mr-1.5 size-4 ${loading ? 'animate-pulse' : ''}`} /> Uruchom analizę
        </Button>
      </header>

      {data && (
        <div className="flex items-center gap-3 text-[12px]">
          <span className="text-muted-foreground">Alerty:</span>
          <Badge className="bg-rose-500/10 text-rose-300 hover:bg-rose-500/15">wysokich {counts.high ?? 0}</Badge>
          <Badge className="bg-amber-500/10 text-amber-300 hover:bg-amber-500/15">średnich {counts.medium ?? 0}</Badge>
          <Badge className="bg-blue-500/10 text-blue-300 hover:bg-blue-500/15">niskich {counts.low ?? 0}</Badge>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Card className="relative overflow-hidden border-white/5 bg-gradient-to-b from-white/[0.04] to-white/[0.01] backdrop-blur">
          <CardContent className="px-6 py-5">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <div className="text-[10.5px] uppercase tracking-[0.16em] text-muted-foreground">Podsumowanie portfela</div>
                <div className="mt-1 text-base font-semibold">Synteza — maj 2026</div>
              </div>
              <Badge variant="secondary"><Sparkles className="mr-1 size-3" /> AI</Badge>
            </div>
            {loading || !data ? (
              <div className="flex flex-col gap-2">
                {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-3 w-full" />)}
              </div>
            ) : (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
                <Markdown>{data.summary.summary_md}</Markdown>
              </motion.div>
            )}
          </CardContent>
        </Card>

        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div className="text-[15px] font-medium">Alerty ryzyka</div>
            <span className="text-[11.5px] text-muted-foreground">{data?.alerts.length ?? 0} pozycji</span>
          </div>
          <div className="flex flex-col gap-2.5">
            {loading || !data
              ? Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-[110px]" />)
              : data.alerts
                  .slice()
                  .sort((a, b) => ({ high: 0, medium: 1, low: 2 }[a.severity] - { high: 0, medium: 1, low: 2 }[b.severity]))
                  .map((a, i) => <AlertCard key={i} alert={a} index={i} />)}
          </div>
        </div>
      </div>

      {data && (
        <Card className="border-white/5 bg-gradient-to-b from-white/[0.04] to-white/[0.01] backdrop-blur">
          <CardContent className="px-5 py-4">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <div className="text-[15px] font-medium">Aktualizacje tez</div>
                <div className="mt-0.5 text-[12px] text-muted-foreground">Per ticker — krótka ocena</div>
              </div>
              <Badge variant="secondary"><Sparkles className="mr-1 size-3" /> AI</Badge>
            </div>
            <div className="flex flex-col gap-2">
              {data.summary.thesis_updates.map((t, i) => (
                <motion.div key={t.ticker}
                  initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}
                  className="flex items-start gap-3 border-b border-white/[0.04] py-2 last:border-0"
                >
                  <div className="mt-0.5 rounded border border-white/5 bg-white/[0.04] px-1.5 py-0.5 font-mono text-[11.5px] font-semibold">{t.ticker}</div>
                  <p className="flex-1 text-[12.5px] leading-relaxed text-muted-foreground">{t.assessment}</p>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function AlertCard({ alert, index }: { alert: RiskAlert; index: number }) {
  const cls = {
    high:   { bg: 'border-rose-500/20 bg-rose-500/[0.04]',     icon: 'bg-rose-500/15 text-rose-400',     badge: 'bg-rose-500/10 text-rose-300 border-rose-500/30',     I: AlertTriangle },
    medium: { bg: 'border-amber-500/20 bg-amber-500/[0.04]',   icon: 'bg-amber-500/15 text-amber-300',   badge: 'bg-amber-500/10 text-amber-300 border-amber-500/30',   I: AlertCircle },
    low:    { bg: 'border-blue-500/20 bg-blue-500/[0.04]',     icon: 'bg-blue-500/15 text-blue-300',     badge: 'bg-blue-500/10 text-blue-300 border-blue-500/30',     I: Info },
  }[alert.severity];
  const Ico = cls.I;
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.05 + index * 0.06, duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -2 }}
      className={`rounded-xl border ${cls.bg} p-4 backdrop-blur`}
    >
      <div className="flex items-start gap-3">
        <div className={`flex size-9 shrink-0 items-center justify-center rounded-lg ${cls.icon} ${alert.severity === 'high' ? 'animate-pulse' : ''}`}>
          <Ico className="size-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className={cls.badge}>
              {alert.severity === 'high' ? 'wysoki' : alert.severity === 'medium' ? 'średni' : 'niski'}
            </Badge>
            {alert.ticker && (
              <span className="rounded border border-white/5 bg-white/5 px-1.5 py-0.5 font-mono text-[11.5px] text-foreground/70">{alert.ticker}</span>
            )}
          </div>
          <div className="mt-2 text-[13.5px] font-semibold tracking-tight">{alert.title}</div>
          <p className="mt-1 text-[12.5px] leading-relaxed text-muted-foreground">{alert.description}</p>
        </div>
      </div>
    </motion.div>
  );
}

function ReportsView() {
  const [reports, setReports] = React.useState<ReportFile[]>(MOCK_REPORTS);
  const [generating, setGenerating] = React.useState(false);
  const [viewing, setViewing] = React.useState<ReportFile | null>(null);

  const generate = async () => {
    setGenerating(true);
    await new Promise(r => setTimeout(r, 1400));
    const now = new Date();
    const q = Math.floor(now.getMonth() / 3) + 1;
    const fresh: ReportFile = {
      name: `portfolio_review_${now.getFullYear()}Q${q}.md`,
      mtime: now.toISOString(),
      size_bytes: 15_240,
      content_md: `# Przegląd portfela — ${now.getFullYear()}Q${q}\n\n## Wynik\n- Portfel **+5.2%** vs WIG20 **+2.1%**`,
    };
    setReports(r => [fresh, ...r]); setGenerating(false);
    toast.success('Raport wygenerowany', { description: fresh.name });
  };

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between gap-4">
        <div>
          <div className="text-[10.5px] uppercase tracking-[0.16em] text-muted-foreground">Pliki</div>
          <h2 className="mt-1 text-[20px] font-semibold tracking-tight">Raporty</h2>
          <p className="mt-1 text-[13px] text-muted-foreground">Markdownowe przeglądy portfela. Generuj nowy lub przeglądaj historyczne.</p>
        </div>
        <Button onClick={generate} disabled={generating}>
          <FileText className={`mr-1.5 size-4 ${generating ? 'animate-pulse' : ''}`} /> Generuj raport
        </Button>
      </header>

      {generating && (
        <Card className="border-white/5 bg-gradient-to-b from-white/[0.04] to-white/[0.01]">
          <CardContent className="flex items-center gap-3 px-5 py-4">
            <div className="flex size-9 items-center justify-center rounded-lg bg-violet-500/15 text-violet-300">
              <Sparkles className="size-4" />
            </div>
            <div className="flex-1">
              <div className="text-[13px]">Generowanie raportu…</div>
              <div className="mt-2 h-1 overflow-hidden rounded-full bg-white/5">
                <motion.div className="h-full bg-gradient-to-r from-cyan-400 to-violet-400"
                  initial={{ width: 0 }} animate={{ width: '100%' }} transition={{ duration: 1.3 }} />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="flex flex-col gap-2.5">
        {reports.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-14 text-center">
            <div className="mb-3 text-[40px]">📄</div>
            <div className="text-[15px] font-medium">Brak raportów</div>
            <div className="mt-1.5 text-[12.5px] text-muted-foreground">Wygeneruj pierwszy raport, aby zobaczyć go tutaj.</div>
            <Button onClick={generate} className="mt-4">Generuj raport</Button>
          </div>
        ) : reports.map((r, i) => (
          <motion.div key={r.name}
            initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.04 + i * 0.04 }}
            whileHover={{ y: -1 }}
            className="group flex items-center gap-3 rounded-xl border border-white/5 bg-gradient-to-b from-white/[0.04] to-white/[0.01] px-4 py-3.5 backdrop-blur"
          >
            <div className="flex size-10 items-center justify-center rounded-lg border border-white/5 bg-gradient-to-br from-cyan-500/20 to-violet-500/20">
              <FileText className="size-4 text-violet-300" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <div className="truncate font-mono text-[13px] font-medium">{r.name}</div>
                <Badge variant="secondary">md</Badge>
              </div>
              <div className="mt-0.5 flex items-center gap-2 text-[11.5px] text-muted-foreground">
                <Clock className="size-3" /> {fmtRelTime(r.mtime)}
                <span className="opacity-30">·</span>
                <span className="font-mono">{fmtDateTime(r.mtime)}</span>
                <span className="opacity-30">·</span>
                <span className="font-mono">{fmtBytes(r.size_bytes)}</span>
              </div>
            </div>
            <div className="flex items-center gap-1.5 opacity-70 transition-opacity group-hover:opacity-100">
              <Button variant="ghost" size="sm" onClick={() => setViewing(r)}>
                <Eye className="mr-1.5 size-3.5" /> Podgląd
              </Button>
              <Button variant="ghost" size="sm">
                <Download className="mr-1.5 size-3.5" /> Pobierz
              </Button>
            </div>
          </motion.div>
        ))}
      </div>

      <Dialog open={!!viewing} onOpenChange={(v) => !v && setViewing(null)}>
        <DialogContent className="max-h-[88vh] max-w-3xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="font-mono">{viewing?.name}</DialogTitle>
            {viewing && (
              <DialogDescription>{fmtDateTime(viewing.mtime)} · {fmtBytes(viewing.size_bytes)}</DialogDescription>
            )}
          </DialogHeader>
          {viewing && <Markdown>{viewing.content_md ?? ''}</Markdown>}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setViewing(null)}>Zamknij</Button>
            <Button><Download className="mr-1.5 size-4" /> Pobierz .md</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function MonitoringView() {
  const [data, setData] = React.useState<MonitoringReport>(MOCK_MONITORING);
  const [running, setRunning] = React.useState(false);
  const counts = data.items.reduce((a, x) => ({ ...a, [x.status]: (a[x.status] ?? 0) + 1 }), {} as Record<string, number>);
  const run = async () => {
    setRunning(true);
    await new Promise(r => setTimeout(r, 1500));
    const now = new Date().toISOString();
    setData(d => ({
      ...d, generated_at: now,
      reports: [{ name: `monitoring_${now.slice(0, 10)}.html`, mtime: now, size_bytes: 21_640 }, ...d.reports.slice(0, 5)],
    }));
    setRunning(false);
    toast.success('Snapshot wykonany', { description: 'Monitoring tez zaktualizowany.' });
  };

  const STATUS = {
    on_track: { label: 'on track', cls: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300', I: Check, stripe: 'from-emerald-400 to-emerald-400/0' },
    watch:    { label: 'watch',    cls: 'border-amber-500/30 bg-amber-500/10 text-amber-300',     I: Eye,   stripe: 'from-amber-400 to-amber-400/0'   },
    at_risk:  { label: 'at risk',  cls: 'border-rose-500/30 bg-rose-500/10 text-rose-300',         I: AlertTriangle, stripe: 'from-rose-400 to-rose-400/0' },
  } as const;

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-end justify-between gap-4">
        <div>
          <div className="text-[10.5px] uppercase tracking-[0.16em] text-muted-foreground">Tezy w czasie</div>
          <h2 className="mt-1 text-[20px] font-semibold tracking-tight">Monitoring</h2>
          <p className="mt-1 text-[13px] text-muted-foreground">Cotygodniowy przegląd statusu tez dla każdej pozycji.</p>
        </div>
        <Button onClick={run} disabled={running}>
          <Activity className={`mr-1.5 size-4 ${running ? 'animate-pulse' : ''}`} /> Uruchom snapshot
        </Button>
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <Badge className="bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/15"><Check className="mr-1 size-3" />on track · {counts.on_track ?? 0}</Badge>
        <Badge className="bg-amber-500/10 text-amber-300 hover:bg-amber-500/15"><Eye className="mr-1 size-3" />watch · {counts.watch ?? 0}</Badge>
        <Badge className="bg-rose-500/10 text-rose-300 hover:bg-rose-500/15"><AlertTriangle className="mr-1 size-3" />at risk · {counts.at_risk ?? 0}</Badge>
        <span className="ml-2 inline-flex items-center gap-1.5 text-[12px] text-muted-foreground">
          <Clock className="size-3" /> ostatni snapshot {fmtRelTime(data.generated_at)} ·{' '}
          <span className="font-mono">{fmtDateTime(data.generated_at)}</span>
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.4fr_1fr]">
        <div className="flex flex-col gap-2.5">
          {data.items.map((it, i) => {
            const s = STATUS[it.status]; const I = s.I;
            return (
              <motion.div key={it.ticker}
                initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.05 + i * 0.05 }}
                whileHover={{ y: -2 }}
                className="relative overflow-hidden rounded-xl border border-white/5 bg-gradient-to-b from-white/[0.04] to-white/[0.01] px-4 py-4 backdrop-blur"
              >
                <div className={`absolute left-0 top-3 bottom-3 w-0.5 bg-gradient-to-b ${s.stripe}`} />
                <div className="flex items-start gap-3 pl-2">
                  <div className="flex size-10 shrink-0 items-center justify-center rounded-lg border border-white/5 bg-white/[0.04] font-mono text-[11px]">
                    {it.ticker}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="outline" className={`border ${s.cls} ${it.status === 'at_risk' ? 'animate-pulse' : ''}`}>
                        <I className="mr-1 size-3" /> {s.label}
                      </Badge>
                      <span className="text-[12px] text-muted-foreground">
                        teza: <span className="text-foreground/70">{it.ticker}</span>
                      </span>
                    </div>
                    <p className="mt-2 text-[12.5px] leading-relaxed text-muted-foreground">{it.rationale}</p>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>

        <Card className="border-white/5 bg-gradient-to-b from-white/[0.04] to-white/[0.01] backdrop-blur">
          <CardContent className="px-3.5 py-4">
            <div className="mb-2 flex items-center justify-between px-1.5">
              <div>
                <div className="text-[15px] font-medium">Historyczne raporty HTML</div>
                <div className="mt-0.5 text-[12px] text-muted-foreground">Cotygodniowe snapshoty</div>
              </div>
              <Badge variant="secondary">{data.reports.length}</Badge>
            </div>
            <div className="flex flex-col">
              {data.reports.map((r, i) => (
                <motion.div key={r.name}
                  initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.03 + i * 0.03 }}
                  className="group flex items-center gap-3 rounded-lg px-3.5 py-2.5 transition-colors hover:bg-white/[0.03]"
                >
                  <FileText className="size-3.5 text-muted-foreground/70" />
                  <span className="truncate font-mono text-[12.5px]">{r.name}</span>
                  <span className="ml-auto font-mono text-[11.5px] text-muted-foreground">{fmtRelTime(r.mtime)}</span>
                  <span className="font-mono text-[11.5px] text-muted-foreground">{fmtBytes(r.size_bytes)}</span>
                  <button className="rounded-md p-1 opacity-0 transition-opacity hover:bg-white/5 group-hover:opacity-100">
                    <ExternalLink className="size-3.5 text-muted-foreground" />
                  </button>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

/* ─────────────────────────────────── Sidebar / Shell ───────────────────────────────────── */

type TabId = 'portfolio' | 'backtest' | 'analysis' | 'reports' | 'monitoring';

const NAV: { id: TabId; label: string; Icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'portfolio',  label: 'Portfel',    Icon: Wallet },
  { id: 'backtest',   label: 'Backtest',   Icon: BarChart3 },
  { id: 'analysis',   label: 'Analiza AI', Icon: Sparkles },
  { id: 'reports',    label: 'Raporty',    Icon: FileText },
  { id: 'monitoring', label: 'Monitoring', Icon: Activity },
];

function Sidebar({ active, onChange, asOf }: { active: TabId; onChange: (v: TabId) => void; asOf: string }) {
  const ref = React.useRef<HTMLElement>(null);
  const blob1 = React.useRef<HTMLDivElement>(null);
  const blob2 = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    let raf = 0;
    const onMove = (e: MouseEvent) => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const r = ref.current?.getBoundingClientRect(); if (!r) return;
        const x = (e.clientX - (r.left + r.width / 2)) / 40;
        const y = (e.clientY - (r.top  + r.height / 2)) / 60;
        if (blob1.current) blob1.current.style.transform = `translate3d(${x}px, ${y}px, 0)`;
        if (blob2.current) blob2.current.style.transform = `translate3d(${-x * 1.2}px, ${-y * 1.2}px, 0)`;
      });
    };
    window.addEventListener('mousemove', onMove);
    return () => { window.removeEventListener('mousemove', onMove); cancelAnimationFrame(raf); };
  }, []);
  return (
    <aside ref={ref} className="relative flex h-full w-[244px] shrink-0 flex-col border-r border-white/5 bg-gradient-to-b from-white/[0.025] to-transparent px-3 py-4">
      <div ref={blob1} aria-hidden className="pointer-events-none absolute -left-10 -top-10 size-40 rounded-full opacity-40 blur-3xl"
        style={{ background: 'radial-gradient(circle, #22d3ee 0%, transparent 70%)', transition: 'transform 600ms cubic-bezier(0.22,1,0.36,1)' }} />
      <div ref={blob2} aria-hidden className="pointer-events-none absolute -left-12 bottom-12 size-44 rounded-full opacity-30 blur-3xl"
        style={{ background: 'radial-gradient(circle, #a78bfa 0%, transparent 70%)', transition: 'transform 700ms cubic-bezier(0.22,1,0.36,1)' }} />

      <div className="relative mb-6 flex items-center gap-2.5 px-2">
        <div className="relative flex size-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-violet-400 shadow-[0_0_40px_-8px_rgba(167,139,250,0.4)]">
          <Sparkles className="size-4 text-zinc-950" />
          <span className="absolute inset-0 rounded-xl ring-1 ring-white/20" />
        </div>
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-tight">
            Investment{' '}
            <span className="bg-gradient-to-r from-cyan-300 via-violet-300 to-fuchsia-300 bg-clip-text text-transparent">Copilot</span>
          </div>
          <div className="mt-0.5 text-[10.5px] uppercase tracking-[0.2em] text-muted-foreground">GPW · PL</div>
        </div>
      </div>

      <nav className="relative flex flex-col gap-0.5">
        {NAV.map(it => {
          const isActive = active === it.id;
          const I = it.Icon;
          return (
            <motion.button
              key={it.id}
              onClick={() => onChange(it.id)}
              whileHover={{ x: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              className={`group relative flex h-9 items-center gap-2.5 rounded-lg px-2.5 text-[13px] transition-colors ${
                isActive ? 'text-white' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {isActive && (
                <React.Fragment>
                  <motion.span layoutId="sb-bg"
                    className="absolute inset-0 rounded-lg border border-white/10 bg-gradient-to-r from-white/[0.08] to-white/[0.02]"
                    transition={{ type: 'spring', stiffness: 460, damping: 36 }}
                  />
                  <motion.span layoutId="sb-accent"
                    className="absolute bottom-1.5 left-0 top-1.5 w-0.5 rounded-full bg-gradient-to-b from-cyan-400 to-violet-400"
                  />
                </React.Fragment>
              )}
              <span className={`relative z-10 ${isActive ? 'text-violet-300' : ''}`}>
                <I className="size-[15px]" />
              </span>
              <span className="relative z-10 whitespace-nowrap font-medium">{it.label}</span>
              <span className="relative z-10 ml-auto opacity-0 transition-opacity group-hover:opacity-100">
                <ChevronRight className="size-3 text-muted-foreground/60" />
              </span>
            </motion.button>
          );
        })}
      </nav>

      <div className="flex-1" />

      <div className="relative">
        <div className="flex items-center gap-2.5 rounded-xl border border-white/5 bg-white/[0.025] px-3 py-2.5">
          <span className="relative size-2">
            <span className="absolute inset-0 animate-ping rounded-full bg-emerald-400 opacity-70" />
            <span className="relative block size-2 rounded-full bg-emerald-400" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-[11.5px] font-medium">Dane GPW · live</div>
            <div className="truncate font-mono text-[10.5px] text-muted-foreground">{fmtRelTime(asOf)}</div>
          </div>
          <Globe className="size-3.5 text-muted-foreground/60" />
        </div>
        <div className="mt-3 px-1 text-[10px] leading-relaxed text-muted-foreground/60">
          Narzędzie badawcze. Nie stanowi porady inwestycyjnej.
        </div>
      </div>
    </aside>
  );
}

/* ────────────────────────────────────── App ────────────────────────────────────────────── */

export default function InvestmentCopilot() {
  const [active, setActive] = React.useState<TabId>('portfolio');
  const [portfolio, setPortfolio] = React.useState<PortfolioStatus>(MOCK_PORTFOLIO);
  const [refreshing, setRefreshing] = React.useState(false);
  const [search, setSearch] = React.useState('');

  const refresh = async () => {
    setRefreshing(true);
    await new Promise(r => setTimeout(r, 1500));
    setPortfolio(p => {
      const enriched = p.holdings.map(h => {
        const jitter = (Math.random() - 0.5) * 0.02;
        const last_price = +(h.last_price * (1 + jitter)).toFixed(2);
        const value = last_price * h.shares;
        const cost = h.entry_price * h.shares;
        return { ...h, last_price, value, pnl: value - cost, pnl_pct: (value / cost - 1) * 100 };
      });
      const total_value = enriched.reduce((a, x) => a + x.value, 0);
      const total_cost  = enriched.reduce((a, x) => a + x.entry_price * x.shares, 0);
      return {
        ...p, holdings: enriched, total_value,
        total_pnl: total_value - total_cost,
        total_pnl_pct: (total_value / total_cost - 1) * 100,
        as_of: new Date().toISOString(),
      };
    });
    setRefreshing(false);
    toast.success('Dane zaktualizowane', { description: 'Świeże ceny rynkowe z GPW.' });
  };

  const onSavePortfolio = (rows: Holding[]) => setPortfolio(() => computePortfolio(rows));

  return (
    <TooltipProvider delayDuration={120}>
      <div className="relative flex h-screen w-screen overflow-hidden bg-[#070710] text-foreground">
        {/* aurora bg */}
        <div aria-hidden className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
          <div className="absolute -left-[15%] -top-[25%] size-[70vw] animate-[drift1_22s_ease-in-out_infinite_alternate] rounded-full opacity-35 blur-3xl"
            style={{ background: 'radial-gradient(circle, #22d3ee 0%, transparent 60%)' }} />
          <div className="absolute -bottom-[30%] -right-[20%] size-[70vw] animate-[drift2_28s_ease-in-out_infinite_alternate] rounded-full opacity-35 blur-3xl"
            style={{ background: 'radial-gradient(circle, #a78bfa 0%, transparent 60%)' }} />
        </div>

        <style>{`
          @keyframes drift1 { 0% { transform: translate(0,0) scale(1) } 100% { transform: translate(12vw,8vh) scale(1.15) } }
          @keyframes drift2 { 0% { transform: translate(0,0) scale(1) } 100% { transform: translate(-10vw,-6vh) scale(1.10) } }
        `}</style>

        <div className="relative z-10 flex h-full w-full">
          <Sidebar active={active} onChange={setActive} asOf={portfolio.as_of} />

          <main className="flex min-w-0 flex-1 flex-col">
            <div className="flex items-center justify-between gap-4 border-b border-white/5 bg-gradient-to-b from-white/[0.02] to-transparent px-6 py-3.5">
              <div className="relative max-w-md flex-1">
                <Search className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder="Szukaj pozycji, słowa kluczowego, raportu…"
                  className="h-9 pl-8"
                />
              </div>
              <div className="flex items-center gap-2">
                <div className="hidden items-center gap-2 text-[11.5px] text-muted-foreground md:flex">
                  <span className="relative size-1.5">
                    <span className="absolute inset-0 animate-ping rounded-full bg-emerald-400 opacity-70" />
                    <span className="relative block size-1.5 rounded-full bg-emerald-400" />
                  </span>
                  <span className="font-mono">GPW · live</span>
                  <span className="opacity-30">·</span>
                  <span>{fmtRelTime(portfolio.as_of)}</span>
                </div>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" className="relative">
                      <Bell className="size-4" />
                      <span className="absolute right-2 top-2 size-1.5 rounded-full bg-rose-400" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Powiadomienia</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon"><Settings className="size-4" /></Button>
                  </TooltipTrigger>
                  <TooltipContent>Ustawienia</TooltipContent>
                </Tooltip>
                <div className="flex size-9 items-center justify-center rounded-lg border border-white/10 bg-gradient-to-br from-cyan-500/30 to-violet-500/30 text-[11px] font-semibold">MK</div>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto">
              <div className="mx-auto max-w-[1480px] px-8 py-7">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={active}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
                  >
                    {active === 'portfolio'  && <PortfolioView portfolio={portfolio} onRefresh={refresh} refreshing={refreshing} onSave={onSavePortfolio} />}
                    {active === 'backtest'   && <BacktestView />}
                    {active === 'analysis'   && <AnalysisView />}
                    {active === 'reports'    && <ReportsView />}
                    {active === 'monitoring' && <MonitoringView />}
                  </motion.div>
                </AnimatePresence>
              </div>
              <div className="px-6 py-4 text-center text-[11px] text-muted-foreground/60">
                Investment Copilot is a research tool, not financial advice.
              </div>
            </div>
          </main>
        </div>

        <Toaster position="bottom-right" theme="dark" />
      </div>
    </TooltipProvider>
  );
}
