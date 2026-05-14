/* Portfolio tab */

const { motion: PMot, AnimatePresence: PAP } = window.Motion;
const {
  PieChart: RPieChart, Pie: RPie, Cell: RCell,
  ResponsiveContainer: RRC, Tooltip: RTooltip,
} = window.Recharts;

/* ---------- KPI Card ---------- */
function KpiCard({ label, value, hint, accent = 'cyan', delay = 0, format, decimals = 0, prefix = '', suffix = '', sub }) {
  const accentBars = {
    cyan:   'from-accent-cyan/0    via-accent-cyan/60    to-accent-cyan/0',
    violet: 'from-accent-violet/0  via-accent-violet/60  to-accent-violet/0',
    green:  'from-accent-green/0   via-accent-green/60   to-accent-green/0',
    red:    'from-accent-red/0     via-accent-red/60     to-accent-red/0',
    amber:  'from-accent-amber/0   via-accent-amber/60   to-accent-amber/0',
  };
  return (
    <PMot.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -2 }}
      className="relative"
    >
      <Card className="relative px-5 pt-4 pb-4 overflow-hidden">
        {/* top accent line */}
        <div className={`absolute top-0 left-0 right-0 h-px bg-gradient-to-r ${accentBars[accent]}`} />
        {/* soft glow blob */}
        <div className={`absolute -top-8 -right-8 h-24 w-24 rounded-full blur-3xl opacity-30 pointer-events-none`}
             style={{ background: accent === 'green' ? '#34d399' : accent === 'red' ? '#f87171' : accent === 'violet' ? '#a78bfa' : '#22d3ee' }} />
        <div className="relative flex items-center justify-between mb-3">
          <span className="text-[10.5px] uppercase tracking-[0.16em] text-white/45">{label}</span>
          {hint && <span className="text-[10.5px] text-white/35 mono">{hint}</span>}
        </div>
        <div className="relative text-[26px] font-semibold tracking-tight text-white leading-none">
          <CountUp value={value} decimals={decimals} prefix={prefix} suffix={suffix} format={format} />
        </div>
        {sub && (
          <div className="relative mt-2 text-[12px] text-white/55">{sub}</div>
        )}
      </Card>
    </PMot.div>
  );
}

/* ---------- PnL Bar (inline mini-bar for the table column) ---------- */
function PnlBar({ pct, max }) {
  const w = Math.min(Math.abs(pct) / max, 1) * 100;
  const positive = pct >= 0;
  return (
    <div className="relative h-1.5 w-20 rounded-full bg-white/[0.04] overflow-hidden">
      <PMot.div
        initial={{ width: 0 }}
        animate={{ width: `${w}%` }}
        transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        className={`absolute top-0 ${positive ? 'left-1/2' : 'right-1/2'} h-full rounded-full ${
          positive
            ? 'bg-gradient-to-r from-accent-green/40 to-accent-green'
            : 'bg-gradient-to-l from-accent-red/40 to-accent-red'
        }`}
      />
      <div className="absolute inset-y-0 left-1/2 w-px bg-white/15" />
    </div>
  );
}

/* ---------- Allocation Donut ---------- */
const ALLOC_PALETTE = ['#22d3ee', '#a78bfa', '#f0abfc', '#34d399', '#fbbf24', '#60a5fa'];

function AllocationDonut({ holdings, totalValue }) {
  const data = holdings.map((h, i) => ({
    name: h.display_ticker || h.ticker,
    full_ticker: h.ticker,
    value: h.value,
    pct: h.value / totalValue * 100,
    fill: ALLOC_PALETTE[i % ALLOC_PALETTE.length],
  }));
  return (
    <Card className="relative px-5 pt-4 pb-5 h-full overflow-hidden">
      <div className="flex items-center justify-between mb-2">
        <div>
          <div className="text-[10.5px] uppercase tracking-[0.16em] text-white/45">Alokacja</div>
          <div className="text-[15px] font-medium text-white tracking-tight mt-0.5">Według pozycji</div>
        </div>
        <Badge variant="default" icon="layers">{holdings.length} pozycji</Badge>
      </div>
      <div className="flex items-center gap-4">
        <div className="relative h-[150px] w-[150px] shrink-0">
          <RRC width="100%" height="100%">
            <RPieChart>
              <RPie
                data={data}
                dataKey="value"
                innerRadius={48}
                outerRadius={70}
                paddingAngle={2}
                stroke="#0a0a0f"
                strokeWidth={2}
                animationDuration={900}
              >
                {data.map((d, i) => <RCell key={i} fill={d.fill} />)}
              </RPie>
              <RTooltip content={<AllocTooltip />} />
            </RPieChart>
          </RRC>
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <div className="text-[10.5px] uppercase tracking-[0.16em] text-white/40">Razem</div>
            <div className="text-[13px] font-semibold text-white mt-0.5 mono">{fmtPLN(totalValue)}</div>
          </div>
        </div>
        <div className="flex-1 grid grid-cols-1 gap-1.5 min-w-0">
          {data.map(d => {
            const longTicker = (d.name || '').length > 3;
            if (longTicker) {
              return (
                <div key={d.name} className="flex items-start gap-2 text-[12px] min-w-0">
                  <span className="h-2 w-2 rounded-sm shrink-0 mt-1" style={{ background: d.fill }} />
                  <div className="flex-1 min-w-0 leading-tight">
                    <div
                      className="mono text-white/85 truncate"
                      title={d.full_ticker}
                    >
                      {d.name}
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-white/40 truncate flex-1 min-w-0">{fmtPLN(d.value)}</span>
                      <span className="mono text-white/70 tabular-nums shrink-0">{d.pct.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              );
            }
            return (
              <div key={d.name} className="flex items-center gap-2 text-[12px] min-w-0">
                <span className="h-2 w-2 rounded-sm shrink-0" style={{ background: d.fill }} />
                <span className="mono text-white/85 shrink-0" title={d.full_ticker}>{d.name}</span>
                <span className="text-white/40 truncate flex-1 min-w-0">{fmtPLN(d.value)}</span>
                <span className="mono text-white/70 tabular-nums shrink-0">{d.pct.toFixed(1)}%</span>
              </div>
            );
          })}
        </div>
      </div>
    </Card>
  );
}
function AllocTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0];
  return (
    <div className="glass rounded-lg px-2.5 py-1.5 text-[11.5px]">
      <div className="mono text-white">{p.payload.name}</div>
      <div className="text-white/60">{fmtPLN(p.value)} · {p.payload.pct.toFixed(1)}%</div>
    </div>
  );
}

/* ---------- Holdings Table ---------- */
function HoldingsTable({ portfolio }) {
  const maxAbsPct = Math.max(...portfolio.holdings.map(h => Math.abs(h.pnl_pct)), 5);

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between px-5 pt-4 pb-3">
        <div>
          <div className="text-[15px] font-medium tracking-tight text-white">Pozycje</div>
          <div className="text-[12px] text-white/45 mt-0.5">
            Stan na {fmtDateTime(portfolio.as_of)}
          </div>
        </div>
        <div className="flex items-center gap-2 text-[11.5px] text-white/50">
          <Icon name="filter" size={12} />
          <span>Wszystkie</span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-[10.5px] uppercase tracking-[0.14em] text-white/40 border-y border-white/[0.05]">
              <th className="text-left  font-medium px-5 py-2.5">Ticker</th>
              <th className="text-left  font-medium px-3 py-2.5">Nazwa</th>
              <th className="text-right font-medium px-3 py-2.5">Akcje</th>
              <th className="text-right font-medium px-3 py-2.5">Entry</th>
              <th className="text-right font-medium px-3 py-2.5">Ostatnia</th>
              <th className="text-right font-medium px-3 py-2.5">Wartość</th>
              <th className="text-right font-medium px-3 py-2.5">PnL</th>
              <th className="text-right font-medium px-5 py-2.5">PnL %</th>
            </tr>
          </thead>
          <tbody>
            {portfolio.holdings.map((h, i) => {
              const positive = h.pnl >= 0;
              return (
                <PMot.tr
                  key={h.ticker}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.08 + i * 0.05 }}
                  whileHover={{ y: -1, backgroundColor: 'rgba(255,255,255,0.025)' }}
                  className="group border-b border-white/[0.03] last:border-0"
                >
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-2.5">
                      <div className="h-7 w-7 rounded-md bg-white/[0.04] border border-white/[0.06] flex items-center justify-center mono text-[10.5px] text-white/85">
                        {h.ticker.slice(0, 3)}
                      </div>
                      <span className="mono text-[12.5px] font-semibold text-white">{h.ticker}</span>
                    </div>
                  </td>
                  <td className="px-3 py-3.5 text-[12.5px] text-white/80">{h.name}</td>
                  <td className="px-3 py-3.5 text-right mono text-[12.5px] text-white/85">{fmtNum(h.shares)}</td>
                  <td className="px-3 py-3.5 text-right mono text-[12.5px] text-white/60">{fmtPLN(h.entry_price, { decimals: 2 })}</td>
                  <td className="px-3 py-3.5 text-right mono text-[12.5px] text-white">{fmtPLN(h.last_price, { decimals: 2 })}</td>
                  <td className="px-3 py-3.5 text-right mono text-[12.5px] text-white">{fmtPLN(h.value)}</td>
                  <td className="px-3 py-3.5 text-right">
                    <div className="flex items-center justify-end gap-2.5">
                      <PnlBar pct={h.pnl_pct} max={maxAbsPct} />
                      <span className={`mono text-[12.5px] font-medium ${positive ? 'text-accent-green' : 'text-accent-red'}`}>
                        {fmtPLN(h.pnl, { signed: true })}
                      </span>
                    </div>
                  </td>
                  <td className="px-5 py-3.5 text-right">
                    <span className={`mono text-[12.5px] font-semibold inline-flex items-center gap-1 ${positive ? 'text-accent-green' : 'text-accent-red'}`}>
                      <Icon name={positive ? 'trendingUp' : 'trendingDown'} size={11} />
                      {fmtPct(h.pnl_pct, { signed: true })}
                    </span>
                  </td>
                </PMot.tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/* ---------- Edit Portfolio Dialog ---------- */
function EditPortfolioDialog({ open, onClose, holdings, onSave }) {
  const [rows, setRows] = React.useState(() => holdings.map(h => ({ ...h })));
  React.useEffect(() => { if (open) setRows(holdings.map(h => ({ ...h }))); }, [open, holdings]);

  const update = (i, patch) => setRows(r => r.map((x, j) => j === i ? { ...x, ...patch } : x));
  const remove = (i) => setRows(r => r.filter((_, j) => j !== i));
  const add = () => setRows(r => [
    ...r,
    { ticker: '', name: '', shares: 0, entry_price: 0, entry_date: new Date().toISOString().slice(0, 10), thesis: '', keywords: [] },
  ]);

  return (
    <Dialog
      open={open} onClose={onClose}
      title="Edytuj portfel"
      subtitle="Pozycje, tezy i słowa kluczowe"
      width="max-w-4xl"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>Anuluj</Button>
          <Button variant="primary" icon="check" onClick={() => onSave(rows)}>Zapisz zmiany</Button>
        </>
      }
    >
      <div className="flex flex-col gap-3">
        {rows.map((r, i) => (
          <PMot.div
            key={i}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.03 }}
            className="glass-soft rounded-xl p-4"
          >
            <div className="flex items-start gap-3 mb-3">
              <div className="grid grid-cols-12 gap-2 flex-1">
                <div className="col-span-2">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Ticker</label>
                  <Input value={r.ticker} onChange={e => update(i, { ticker: e.target.value.toUpperCase() })} placeholder="PKN" className="mt-1" />
                </div>
                <div className="col-span-4">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Nazwa</label>
                  <Input value={r.name} onChange={e => update(i, { name: e.target.value })} placeholder="PKN Orlen" className="mt-1" />
                </div>
                <div className="col-span-2">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Akcje</label>
                  <Input type="number" value={r.shares} onChange={e => update(i, { shares: Number(e.target.value) })} className="mt-1" />
                </div>
                <div className="col-span-2">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Entry price</label>
                  <Input type="number" step="0.01" value={r.entry_price} onChange={e => update(i, { entry_price: Number(e.target.value) })} className="mt-1" />
                </div>
                <div className="col-span-2">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Entry date</label>
                  <Input type="date" value={r.entry_date} onChange={e => update(i, { entry_date: e.target.value })} className="mt-1" />
                </div>
              </div>
              <button
                onClick={() => remove(i)}
                className="mt-5 h-9 w-9 rounded-lg text-white/40 hover:text-accent-red hover:bg-accent-red/10 border border-white/[0.06] hover:border-accent-red/30 flex items-center justify-center transition-colors"
                title="Usuń pozycję"
              >
                <Icon name="trash" size={14} />
              </button>
            </div>

            <div className="mb-2">
              <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Teza inwestycyjna</label>
              <Textarea
                value={r.thesis}
                onChange={e => update(i, { thesis: e.target.value })}
                placeholder="Krótkie uzasadnienie, dlaczego trzymasz tę pozycję…"
                className="mt-1"
                rows={2}
              />
            </div>

            <div>
              <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Słowa kluczowe</label>
              <div className="mt-1">
                <KeywordsInput
                  value={r.keywords}
                  onChange={kw => update(i, { keywords: kw })}
                />
              </div>
            </div>
          </PMot.div>
        ))}

        <button
          onClick={add}
          className="group flex items-center justify-center gap-2 h-12 rounded-xl border border-dashed border-white/[0.12] hover:border-accent-violet/40 text-white/55 hover:text-white text-[13px] transition-colors"
        >
          <Icon name="plus" size={14} />
          Dodaj pozycję
        </button>
      </div>
    </Dialog>
  );
}

/* ---------- Update progress shimmer ---------- */
function UpdateProgressBar({ active }) {
  return (
    <PAP>
      {active && (
        <PMot.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="relative h-1 rounded-full overflow-hidden bg-white/[0.04]"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-accent-cyan to-accent-violet opacity-30" />
          <PMot.div
            className="absolute top-0 bottom-0 w-1/3 bg-gradient-to-r from-accent-cyan to-accent-violet"
            initial={{ x: '-100%' }}
            animate={{ x: '300%' }}
            transition={{ repeat: Infinity, duration: 1.4, ease: 'easeInOut' }}
          />
        </PMot.div>
      )}
    </PAP>
  );
}

/* ---------- Portfolio Tab ---------- */
function PortfolioTab({ portfolio, onUpdatePortfolio, onRefresh, refreshing, benchmarkLabel = 'WIG20' }) {
  const [editOpen, setEditOpen] = React.useState(false);
  const toast = useToast();

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader
        eyebrow="Przegląd"
        title="Portfel"
        description="Aktualny stan pozycji, wycena rynkowa i wynik względem ceny wejścia."
        right={
          <div className="flex items-center gap-2">
            <Button variant="outline" icon="edit" onClick={() => setEditOpen(true)}>
              Edytuj portfel
            </Button>
            <Button variant="primary" icon="refresh" loading={refreshing} onClick={onRefresh}>
              Odśwież dane rynkowe
            </Button>
          </div>
        }
      />

      <UpdateProgressBar active={refreshing} />

      {/* KPI row */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard
          label="Wartość portfela"
          value={portfolio.total_value}
          decimals={0}
          format={v => fmtPLN(v)}
          accent="cyan"
          hint="PLN"
          sub={<span>Suma wartości rynkowej {portfolio.holdings.length} pozycji</span>}
          delay={0.0}
        />
        <KpiCard
          label="PnL łącznie"
          value={portfolio.total_pnl}
          decimals={0}
          format={v => fmtPLN(v, { signed: true })}
          accent={portfolio.total_pnl >= 0 ? 'green' : 'red'}
          hint="PLN"
          sub={
            <span className={portfolio.total_pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}>
              {portfolio.total_pnl >= 0 ? '↑' : '↓'} względem kosztu wejścia
            </span>
          }
          delay={0.06}
        />
        <KpiCard
          label="PnL %"
          value={portfolio.total_pnl_pct}
          decimals={2}
          format={v => fmtPct(v, { signed: true, decimals: 2 })}
          accent={portfolio.total_pnl_pct >= 0 ? 'green' : 'red'}
          hint="zwrot"
          sub={<span className="text-white/45">vs. {benchmarkLabel} — uruchom Backtest, aby porównać</span>}
          delay={0.12}
        />
        <KpiCard
          label="Ostatnia aktualizacja"
          value={0}
          format={() => fmtRelTime(portfolio.as_of)}
          accent="violet"
          hint="market data"
          sub={<span className="mono text-white/55">{fmtDateTime(portfolio.as_of)}</span>}
          delay={0.18}
        />
      </div>

      {/* Holdings + allocation */}
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_360px] gap-4">
        <HoldingsTable portfolio={portfolio} />
        <AllocationDonut holdings={portfolio.holdings} totalValue={portfolio.total_value} />
      </div>

      <EditPortfolioDialog
        open={editOpen}
        onClose={() => setEditOpen(false)}
        holdings={portfolio.holdings}
        onSave={(rows) => {
          onUpdatePortfolio(rows);
          setEditOpen(false);
          toast.success('Portfel zapisany', `${rows.length} pozycji zaktualizowanych.`);
        }}
      />
    </div>
  );
}

window.PortfolioTab = PortfolioTab;
