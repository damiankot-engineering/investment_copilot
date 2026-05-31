/* Backtest tab */

const { motion: BMot } = window.Motion;
const {
  AreaChart: RAreaChart, Area: RArea,
  LineChart: RLineChart, Line: RLine,
  XAxis: RXAxis, YAxis: RYAxis,
  CartesianGrid: RCartesianGrid, Tooltip: RTip2,
  ResponsiveContainer: RRC2, Legend: RLegend, ReferenceLine: RRefLine,
} = window.Recharts;

/* ---------- Strategy / picker ---------- */
const STRATEGIES = [
  { value: 'ma_crossover', label: 'MA Crossover', icon: 'pulse',  hint: 'Średnie kroczące 50/200' },
  { value: 'momentum',     label: 'Momentum',     icon: 'zap',    hint: 'Top-N momentum 12-1m' },
  { value: 'buy_and_hold', label: 'Buy & Hold',   icon: 'target', hint: 'Kup i trzymaj, bez sygnałów' },
];
// Strategy used for the auto-run on app start (BacktestConfig has no strategy
// field, so this is the frontend default — robust because Buy & Hold needs no
// long moving-average history).
const DEFAULT_STRATEGY = 'buy_and_hold';

/* ---------- Tooltips ---------- */
function EquityTooltip({ active, payload, label, benchmarkLabel = 'Benchmark' }) {
  if (!active || !payload?.length) return null;
  const p = payload.find(x => x.dataKey === 'portfolio')?.value;
  const b = payload.find(x => x.dataKey === 'benchmark')?.value;
  const fmt = v => (typeof v === 'number' ? (v >= 0 ? '+' : '') + v.toFixed(2) + '%' : '—');
  return (
    <div className="glass rounded-lg px-3 py-2 text-[11.5px] min-w-[200px]">
      <div className="mono text-white/60 mb-1.5">{label}</div>
      <div className="flex items-center justify-between gap-4">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-accent-violet" />
          <span className="text-white/85">Portfel</span>
        </span>
        <span className="mono text-white">{fmt(p)}</span>
      </div>
      <div className="flex items-center justify-between gap-4 mt-0.5">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-accent-cyan/70" />
          <span className="text-white/85">{benchmarkLabel}</span>
        </span>
        <span className="mono text-white">{fmt(b)}</span>
      </div>
    </div>
  );
}

function DrawdownTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass rounded-lg px-3 py-2 text-[11.5px]">
      <div className="mono text-white/60 mb-1">{label}</div>
      <div className="mono text-accent-red">{payload[0].value.toFixed(2)}%</div>
    </div>
  );
}

/* ---------- Metric Card (mini) ---------- */
function MetricCard({ label, value, hint, format, decimals = 2, signedSuffix, accent = 'default', delay = 0 }) {
  const accents = {
    default: '',
    green:   'text-accent-green',
    red:     'text-accent-red',
    violet:  'text-accent-violet',
    cyan:    'text-accent-cyan',
    amber:   'text-accent-amber',
  };
  return (
    <BMot.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.45 }}
      className="glass-soft rounded-xl px-4 py-3.5"
    >
      <div className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">{label}</div>
      <div className={`mt-1.5 text-[20px] font-semibold tracking-tight ${accents[accent]} mono`}>
        <CountUp value={value} decimals={decimals} format={format} />
        {signedSuffix && <span className="text-[12px] text-white/35 ml-1">{signedSuffix}</span>}
      </div>
      {hint && <div className="text-[11px] text-white/40 mt-0.5">{hint}</div>}
    </BMot.div>
  );
}

/* ---------- Equity chart ---------- */
function EquityCurveChart({ data, animateKey, benchmarkLabel }) {
  return (
    <div className="h-[320px] w-full">
      <RRC2 width="100%" height="100%">
        <RLineChart data={data} margin={{ top: 12, right: 20, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="lineP" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%"   stopColor="#67e8f9" />
              <stop offset="100%" stopColor="#a78bfa" />
            </linearGradient>
            <linearGradient id="lineB" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%"   stopColor="#94a3b8" stopOpacity="0.55" />
              <stop offset="100%" stopColor="#94a3b8" stopOpacity="0.55" />
            </linearGradient>
          </defs>
          <RCartesianGrid strokeDasharray="2 4" vertical={false} />
          <RXAxis
            dataKey="date"
            tickFormatter={d => d.slice(0, 7)}
            interval={Math.floor(data.length / 8)}
            tickLine={false} axisLine={false}
          />
          <RYAxis
            tickLine={false} axisLine={false} width={56}
            tickFormatter={v => (v >= 0 ? '+' : '') + v.toFixed(0) + '%'}
            domain={['dataMin - 2', 'dataMax + 2']}
          />
          <RRefLine y={0} stroke="rgba(255,255,255,0.18)" strokeDasharray="2 4" />
          <RTip2
            content={props => <EquityTooltip {...props} benchmarkLabel={benchmarkLabel} />}
            cursor={{ stroke: 'rgba(255,255,255,0.18)', strokeDasharray: '3 3' }}
          />
          <RLine
            type="monotone"
            dataKey="benchmark"
            stroke="url(#lineB)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive
            animationDuration={1200}
            animationEasing="ease-out"
            key={`b-${animateKey}`}
          />
          <RLine
            type="monotone"
            dataKey="portfolio"
            stroke="url(#lineP)"
            strokeWidth={2.25}
            dot={false}
            isAnimationActive
            animationDuration={1400}
            animationEasing="ease-out"
            key={`p-${animateKey}`}
          />
        </RLineChart>
      </RRC2>
    </div>
  );
}

/* ---------- Drawdown chart ---------- */
function DrawdownChart({ data, animateKey }) {
  return (
    <div className="h-[140px] w-full">
      <RRC2 width="100%" height="100%">
        <RAreaChart data={data} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#f87171" stopOpacity="0.55" />
              <stop offset="100%" stopColor="#f87171" stopOpacity="0.02" />
            </linearGradient>
          </defs>
          <RCartesianGrid strokeDasharray="2 4" vertical={false} />
          <RXAxis
            dataKey="date"
            tickFormatter={d => d.slice(0, 7)}
            interval={Math.floor(data.length / 8)}
            tickLine={false} axisLine={false}
          />
          <RYAxis
            tickLine={false} axisLine={false} width={48}
            tickFormatter={v => `${v.toFixed(0)}%`}
            domain={['dataMin', 0]}
          />
          <RTip2 content={<DrawdownTooltip />} cursor={{ stroke: 'rgba(248,113,113,0.4)', strokeDasharray: '3 3' }} />
          <RRefLine y={0} stroke="rgba(255,255,255,0.15)" />
          <RArea
            type="monotone"
            dataKey="drawdown"
            stroke="#f87171"
            strokeWidth={1.5}
            fill="url(#ddFill)"
            isAnimationActive
            animationDuration={1200}
            key={`d-${animateKey}`}
          />
        </RAreaChart>
      </RRC2>
    </div>
  );
}

/* ---------- Backtest Tab ---------- */
function BacktestTab() {
  const FALLBACK_BENCHMARKS = [
    { value: 'wig',    label: 'WIG'    },
    { value: 'wig20',  label: 'WIG20'  },
    { value: 'mwig40', label: 'mWIG40' },
    { value: 'swig80', label: 'sWIG80' },
    { value: 'wig30',  label: 'WIG30'  },
  ];
  const today = new Date().toISOString().slice(0, 10);
  const [strategy, setStrategy] = React.useState('buy_and_hold');
  const [benchmark, setBenchmark] = React.useState('wig20');
  const [availableBenchmarks, setAvailableBenchmarks] = React.useState(FALLBACK_BENCHMARKS);
  const [startDate, setStartDate] = React.useState('2026-01-01');
  const [endDate, setEndDate] = React.useState(today);
  const [running, setRunning] = React.useState(false);
  const [result, setResult] = React.useState(null);
  const [drawdown, setDrawdown] = React.useState([]);
  const [animateKey, setAnimateKey] = React.useState(0);
  const toast = useToast();

  // Core runner. `silent` suppresses both toasts (used by the auto-run so a
  // missing-data failure doesn't pop an error on every app start — the
  // EmptyState already guides the user to refresh data).
  const executeBacktest = async (
    { strategy: strat, benchmark: bm, startDate: sd, endDate: ed },
    { silent = false } = {},
  ) => {
    setRunning(true);
    setResult(null);
    try {
      const r = await window.API.runBacktest(strat, {
        startDate: sd || undefined,
        endDate: ed || undefined,
        benchmark: bm,
      });
      setResult(r);
      setDrawdown(r.drawdown || []);
      setAnimateKey(k => k + 1);
      if (!silent) {
        const label = STRATEGIES.find(s => s.value === strat)?.label || strat;
        toast.success('Backtest gotowy', `Strategia ${label}.`);
      }
    } catch (err) {
      console.error(err);
      if (!silent) toast.error('Backtest nie powiódł się', err.detail || err.message);
    } finally {
      setRunning(false);
    }
  };

  // Manual run (the "Uruchom backtest" button) — uses the current UI params.
  const runBacktest = () => executeBacktest({ strategy, benchmark, startDate, endDate });

  // On app start: load defaults from /api/config (best-effort; FALLBACK_BENCHMARKS
  // is the initial state so the dropdown never collapses if the call fails),
  // prefill the inputs, then auto-run the backtest with the default config.
  React.useEffect(() => {
    (async () => {
      let bm = benchmark;
      let sd = startDate;
      let ed = endDate;
      try {
        const cfg = await window.API.getConfig();
        if (cfg.benchmark) { bm = cfg.benchmark; setBenchmark(bm); }
        if (Array.isArray(cfg.available_benchmarks) && cfg.available_benchmarks.length > 0) {
          setAvailableBenchmarks(cfg.available_benchmarks);
        }
        if (cfg.backtest_start_date) { sd = cfg.backtest_start_date; setStartDate(sd); }
        // end_date null in config means "up to latest available" → empty input.
        if ('backtest_end_date' in cfg) { ed = cfg.backtest_end_date || ''; setEndDate(ed); }
      } catch (err) {
        console.error('Could not load /api/config:', err);
      }
      await executeBacktest(
        { strategy: DEFAULT_STRATEGY, benchmark: bm, startDate: sd, endDate: ed },
        { silent: true },
      );
    })();
  }, []);

  const benchmarkLabel = (availableBenchmarks.find(b => b.value === benchmark) || {}).label || benchmark.toUpperCase();

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader
        eyebrow="Symulacja historyczna"
        title="Backtest strategii"
        description={`Porównaj wynik strategii względem wybranego benchmarku (${benchmarkLabel}).`}
        right={
          <div className="flex items-center gap-3">
            <Tabs
              value={strategy}
              onChange={setStrategy}
              items={STRATEGIES.map(s => ({ value: s.value, label: s.label, icon: s.icon }))}
            />
            <Button variant="primary" icon="play" loading={running} onClick={runBacktest}>
              Uruchom backtest
            </Button>
          </div>
        }
      />

      {/* Parameters strip */}
      <Card className="px-5 py-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Benchmark</label>
            <select
              value={benchmark}
              onChange={e => setBenchmark(e.target.value)}
              className="mt-1.5 w-full h-9 px-3 rounded-lg bg-white/[0.03] border border-white/[0.08] hover:border-white/[0.15] text-[13px] text-white outline-none focus:border-accent-violet/50 mono"
            >
              {availableBenchmarks.map(b => (
                <option key={b.value} value={b.value} className="bg-ink-900">{b.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Data początkowa</label>
            <Input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="mt-1.5" />
          </div>
          <div>
            <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Data końcowa</label>
            <Input
              type="date"
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
              placeholder="(do najnowszych danych)"
              className="mt-1.5"
            />
            {!endDate && <div className="text-[11px] text-white/35 mt-1">pusto = do najnowszych danych</div>}
          </div>
          <div className="flex items-end">
            <div className="text-[11.5px] text-white/45">
              <div className="inline-flex items-center gap-1.5">
                <Icon name="target" size={12} className="text-accent-violet" />
                <span>{STRATEGIES.find(s => s.value === strategy)?.hint}</span>
              </div>
              {result && (
                <div className="mt-1 mono text-white/55">
                  {result.equity_curve.length} dni handlowych
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>

      {!result && !running ? (
        <EmptyState
          emoji="📈"
          title="Backtest jeszcze nie uruchomiony"
          description={
            <>
              Wybierz strategię, benchmark i zakres dat, a następnie kliknij <span className="text-white/80 font-medium">„Uruchom backtest”</span>,
              aby zobaczyć krzywą zwrotu, drawdown i metryki względem {benchmarkLabel}.
              <br />
              <span className="text-[11.5px] text-white/40">Wymagane: aktualne dane OHLCV — uruchom „Odśwież dane rynkowe” w zakładce Portfel, jeśli jeszcze tego nie zrobiłeś.</span>
            </>
          }
          cta={<Button variant="primary" icon="play" loading={running} onClick={runBacktest}>Uruchom backtest</Button>}
        />
      ) : (
        <>
          {/* Equity curve */}
          <Card className="px-5 pt-4 pb-3">
            <div className="flex items-end justify-between mb-3">
              <div>
                <div className="text-[15px] font-medium tracking-tight text-white">Krzywa zwrotu</div>
                <div className="text-[12px] text-white/45 mt-0.5">Portfel vs. {benchmarkLabel} (zwrot od startu, %)</div>
              </div>
              <div className="flex items-center gap-4 text-[12px]">
                <span className="inline-flex items-center gap-1.5"><span className="h-1.5 w-3 rounded-sm bg-gradient-to-r from-accent-cyan to-accent-violet" /><span className="text-white/75">Portfel</span></span>
                <span className="inline-flex items-center gap-1.5"><span className="h-1.5 w-3 rounded-sm bg-white/40" /><span className="text-white/75">{benchmarkLabel}</span></span>
              </div>
            </div>
            {running || !result ? (
              <Skeleton className="h-[320px] w-full" />
            ) : (
              <EquityCurveChart data={result.equity_curve} animateKey={animateKey} benchmarkLabel={benchmarkLabel} />
            )}
          </Card>

          {/* Drawdown */}
          <Card className="px-5 pt-4 pb-3">
            <div className="flex items-end justify-between mb-2">
              <div>
                <div className="text-[15px] font-medium tracking-tight text-white">Drawdown</div>
                <div className="text-[12px] text-white/45 mt-0.5">Spadek od ostatniego szczytu (%)</div>
              </div>
              {result && (
                <Badge variant="red" icon="alertTriangle">
                  Max DD {(result.metrics.max_drawdown * 100).toFixed(1)}% · {result.metrics.max_drawdown_duration_days} dni
                </Badge>
              )}
            </div>
            {running || !result ? (
              <Skeleton className="h-[140px] w-full" />
            ) : (
              <DrawdownChart data={drawdown} animateKey={animateKey} />
            )}
          </Card>

          {/* Metrics grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {result ? (
              <>
                <MetricCard
                  label="Zwrot łączny"
                  value={result.metrics.total_return * 100}
                  decimals={1}
                  format={v => fmtPct(v, { signed: true, decimals: 1 })}
                  hint={result.benchmark_metrics ? `${benchmarkLabel} ${fmtPct(result.benchmark_metrics.total_return * 100, { signed: true, decimals: 1 })}` : undefined}
                  accent={result.metrics.total_return >= 0 ? 'green' : 'red'}
                  delay={0.05}
                />
                <MetricCard
                  label="Zwrot roczny"
                  value={result.metrics.annualized_return * 100}
                  decimals={1}
                  format={v => fmtPct(v, { signed: true, decimals: 1 })}
                  hint="CAGR"
                  accent="violet"
                  delay={0.1}
                />
                <MetricCard
                  label="Zmienność"
                  value={result.metrics.volatility * 100}
                  decimals={1}
                  format={v => fmtPct(v, { decimals: 1 })}
                  hint="ann."
                  accent="cyan"
                  delay={0.15}
                />
                <MetricCard
                  label="Sharpe"
                  value={result.metrics.sharpe}
                  decimals={2}
                  hint="rf = 4.5%"
                  accent={result.metrics.sharpe >= 1 ? 'green' : result.metrics.sharpe >= 0 ? 'amber' : 'red'}
                  delay={0.2}
                />
                <MetricCard
                  label="Max drawdown"
                  value={result.metrics.max_drawdown * 100}
                  decimals={1}
                  format={v => fmtPct(v, { decimals: 1 })}
                  hint={`${result.metrics.max_drawdown_duration_days} dni`}
                  accent="red"
                  delay={0.25}
                />
                <MetricCard
                  label="Win rate"
                  value={result.metrics.win_rate * 100}
                  decimals={1}
                  format={v => fmtPct(v, { decimals: 1 })}
                  hint="dni dodatnie"
                  accent="green"
                  delay={0.3}
                />
              </>
            ) : (
              Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-[80px] w-full" />
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
}

window.BacktestTab = BacktestTab;
