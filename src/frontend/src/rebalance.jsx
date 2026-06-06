/* Rebalancing tab — current vs target weights, the resulting BUY/SELL plan
   (with FIFO tax preview), and an explicit "Zastosuj" that writes transactions.
   Portfolio-scoped automatically via ?portfolio= (api.jsx). */

const { motion: RbMot } = window.Motion;

function AccountBadge({ plan }) {
  if (!plan) return null;
  return plan.tax_exempt ? (
    <Badge variant="on_track" icon="check">{plan.account_type.toUpperCase()} — sprzedaże bez podatku</Badge>
  ) : (
    <Badge variant="default">Konto standardowe · PIT 19%</Badge>
  );
}

function WeightBar({ current, target }) {
  const cur = Math.max(0, Math.min(100, current));
  const tgt = Math.max(0, Math.min(100, target));
  return (
    <div className="relative h-2 w-full rounded-full bg-white/[0.06] overflow-hidden">
      <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-accent-cyan to-accent-violet rounded-full" style={{ width: `${cur}%` }} />
      <div className="absolute inset-y-0 w-0.5 bg-white/80" style={{ left: `calc(${tgt}% - 1px)` }} title={`cel ${tgt.toFixed(1)}%`} />
    </div>
  );
}

function RebalanceTab() {
  const [plan, setPlan] = React.useState(null);
  const [targets, setTargets] = React.useState({});      // ticker -> percent (string|number)
  const [drift, setDrift] = React.useState(5);
  const [minTrade, setMinTrade] = React.useState(200);
  const [loading, setLoading] = React.useState(true);
  const [computing, setComputing] = React.useState(false);
  const [applying, setApplying] = React.useState(false);
  const seeded = React.useRef(false);
  const toast = useToast();

  const buildBody = (withTargets) => {
    const body = { drift_band_pct: Number(drift), min_trade_value: Number(minTrade) };
    if (withTargets && Object.keys(targets).length) {
      body.targets = Object.fromEntries(
        Object.entries(targets).map(([k, v]) => [k, Number(v) || 0]),
      );
    }
    return body;
  };

  const compute = React.useCallback(async (withTargets) => {
    setComputing(true);
    try {
      const p = await window.API.rebalancePlan(buildBody(withTargets));
      setPlan(p);
      if (!seeded.current) {
        const seed = {};
        p.positions.forEach((pos) => { seed[pos.ticker] = Math.round(pos.target_weight_pct * 10) / 10; });
        setTargets(seed);
        seeded.current = true;
      }
    } catch (err) {
      console.error(err);
      toast.error('Nie udało się przeliczyć planu', err.detail || err.message);
    } finally {
      setComputing(false);
      setLoading(false);
    }
    // eslint-disable-next-line
  }, [drift, minTrade, targets, toast]);

  React.useEffect(() => { compute(false); /* on mount */ // eslint-disable-next-line
  }, []);

  const equalWeight = () => {
    if (!plan) return;
    const n = plan.positions.length || 1;
    const w = Math.round((100 / n) * 10) / 10;
    const t = {};
    plan.positions.forEach((p) => { t[p.ticker] = w; });
    setTargets(t);
  };

  const sumTargets = Object.values(targets).reduce((a, v) => a + (Number(v) || 0), 0);

  const onApply = async () => {
    if (!plan || !plan.trades.length) return;
    if (!window.confirm(
      `Zastosować ${plan.trades.length} transakcji do portfela?\n\n` +
      `Zostaną dopisane jako BUY/SELL (cena z cache, data dziś). ` +
      `Plik portfela dostanie kopię .bak.`
    )) return;
    setApplying(true);
    try {
      await window.API.rebalanceApply(buildBody(true));
      toast.success('Rebalans zastosowany', `${plan.trades.length} transakcji dopisanych do portfela.`);
      await compute(true);
    } catch (err) {
      toast.error('Zastosowanie nie powiodło się', err.detail || err.message);
    } finally {
      setApplying(false);
    }
  };

  const trendCls = (v) => (v > 0 ? 'text-accent-green' : v < 0 ? 'text-accent-red' : 'text-white/60');

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader
        eyebrow="Alokacja"
        title="Rebalancing"
        description="Doprowadź wagi do celu w obrębie bieżącej wartości (self-financing). Podgląd zleceń, podatku FIFO i obrotu — z opcją zapisu transakcji."
        right={
          <div className="flex items-center gap-2">
            <Button variant="outline" icon="refresh" loading={computing} onClick={() => compute(true)}>
              Przelicz plan
            </Button>
            <Button
              variant="primary"
              icon="check"
              loading={applying}
              onClick={onApply}
              disabled={!plan || !plan.trades.length}
            >
              Zastosuj
            </Button>
          </div>
        }
      />

      {loading ? (
        <Card><Skeleton className="h-[260px] w-full" /></Card>
      ) : !plan ? (
        <EmptyState emoji="⚖️" title="Brak planu" description="Nie udało się przeliczyć rebalansu." />
      ) : plan.positions.length === 0 ? (
        <EmptyState emoji="⚖️" title="Brak wycenionych pozycji" description="Dodaj pozycje i odśwież dane rynkowe, aby rebalansować." />
      ) : (
        <>
          {/* Constraints + account */}
          <Card className="px-5 py-4">
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Pasmo driftu (pp)</label>
                <Input type="number" step="0.5" value={drift} onChange={e => setDrift(e.target.value)} className="mt-1 w-28" />
              </div>
              <div>
                <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Min. zlecenie (PLN)</label>
                <Input type="number" step="50" value={minTrade} onChange={e => setMinTrade(e.target.value)} className="mt-1 w-32" />
              </div>
              <Button variant="ghost" icon="target" onClick={equalWeight}>Equal-weight</Button>
              <div className="ml-auto flex items-center gap-2">
                <AccountBadge plan={plan} />
              </div>
            </div>
          </Card>

          {/* Current vs target weights (editable) */}
          <Card className="px-5 pt-4 pb-3">
            <div className="flex items-center justify-between mb-3">
              <div className="text-[15px] font-medium tracking-tight text-white">Wagi: bieżąca → docelowa</div>
              <span className={`mono text-[12px] ${Math.abs(sumTargets - 100) < 0.5 ? 'text-white/45' : 'text-accent-amber'}`}>
                Σ celów: {sumTargets.toFixed(1)}%
              </span>
            </div>
            <div className="flex flex-col">
              {plan.positions.map((p) => (
                <div key={p.ticker} className="grid grid-cols-12 items-center gap-3 py-2 border-b border-dotted border-white/[0.06] last:border-0">
                  <div className="col-span-3 min-w-0">
                    <div className="mono text-[12.5px] text-white truncate">{p.ticker}</div>
                    <div className="text-[11px] text-white/40 truncate">{p.name || ''}</div>
                  </div>
                  <div className="col-span-5 flex items-center gap-3">
                    <WeightBar current={p.current_weight_pct} target={Number(targets[p.ticker]) || 0} />
                    <span className="mono text-[11.5px] text-white/55 w-12 text-right">{p.current_weight_pct.toFixed(1)}%</span>
                  </div>
                  <div className="col-span-2 flex items-center gap-1.5">
                    <Input
                      type="number" step="1"
                      value={targets[p.ticker] ?? ''}
                      onChange={e => setTargets(t => ({ ...t, [p.ticker]: e.target.value }))}
                      className="w-20 text-right"
                    />
                    <span className="text-[11px] text-white/35">%</span>
                  </div>
                  <div className="col-span-2 text-right mono text-[12px] text-white/55">
                    {fmtPLN(p.market_value, { decimals: 0 })}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Trades */}
          <Card className="px-5 pt-4 pb-3">
            <div className="text-[15px] font-medium tracking-tight text-white mb-3">
              Zlecenia <span className="text-white/40 text-[12px]">({plan.trades.length})</span>
            </div>
            {plan.trades.length === 0 ? (
              <div className="text-[12.5px] text-white/45 italic py-3">Brak zleceń — portfel jest w paśmie celu.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-[10.5px] uppercase tracking-[0.14em] text-white/40 border-y border-white/[0.05]">
                      <th className="text-left font-medium px-2 py-2">Ticker</th>
                      <th className="text-left font-medium px-2 py-2">Akcja</th>
                      <th className="text-right font-medium px-2 py-2">Sztuk</th>
                      <th className="text-right font-medium px-2 py-2">Cena</th>
                      <th className="text-right font-medium px-2 py-2">Wartość</th>
                      <th className="text-right font-medium px-2 py-2">Drift</th>
                      <th className="text-right font-medium px-2 py-2">Realized PnL</th>
                      <th className="text-right font-medium px-2 py-2">Podatek</th>
                    </tr>
                  </thead>
                  <tbody>
                    {plan.trades.map((t, i) => (
                      <tr key={i} className="border-b border-white/[0.03] last:border-0">
                        <td className="px-2 py-2 mono text-[12px] text-white">{t.ticker}</td>
                        <td className="px-2 py-2">
                          <Badge variant={t.action === 'BUY' ? 'on_track' : 'red'}>{t.action === 'BUY' ? 'KUP' : 'SPRZEDAJ'}</Badge>
                        </td>
                        <td className="px-2 py-2 text-right mono text-[12px] text-white/85">{t.shares}</td>
                        <td className="px-2 py-2 text-right mono text-[12px] text-white/60">{fmtPLN(t.est_price, { decimals: 2 })}</td>
                        <td className="px-2 py-2 text-right mono text-[12px] text-white">{fmtPLN(t.est_value, { decimals: 0 })}</td>
                        <td className={`px-2 py-2 text-right mono text-[12px] ${trendCls(t.drift_pct)}`}>{fmtPct(t.drift_pct, { signed: true, decimals: 1 })}</td>
                        <td className={`px-2 py-2 text-right mono text-[12px] ${t.realized_pnl == null ? 'text-white/30' : trendCls(t.realized_pnl)}`}>
                          {t.realized_pnl == null ? '—' : fmtPLN(t.realized_pnl, { decimals: 0 })}
                        </td>
                        <td className="px-2 py-2 text-right mono text-[12px] text-accent-amber">{t.est_tax ? fmtPLN(t.est_tax, { decimals: 0 }) : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {/* Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="glass-soft rounded-xl px-4 py-3.5">
              <div className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Obrót</div>
              <div className="mt-1 text-[18px] font-semibold mono text-white">{plan.turnover_pct.toFixed(1)}%</div>
            </div>
            <div className="glass-soft rounded-xl px-4 py-3.5">
              <div className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Szac. podatek</div>
              <div className="mt-1 text-[18px] font-semibold mono text-accent-amber">{fmtPLN(plan.est_total_tax, { decimals: 0 })}</div>
            </div>
            <div className="glass-soft rounded-xl px-4 py-3.5">
              <div className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Reszta gotówki</div>
              <div className={`mt-1 text-[18px] font-semibold mono ${plan.residual_cash < 0 ? 'text-accent-red' : 'text-white'}`}>{fmtPLN(plan.residual_cash, { decimals: 0 })}</div>
            </div>
            <div className="glass-soft rounded-xl px-4 py-3.5">
              <div className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Wartość portfela</div>
              <div className="mt-1 text-[18px] font-semibold mono text-white">{fmtPLN(plan.total_market_value, { decimals: 0 })}</div>
            </div>
          </div>

          {/* Warnings */}
          {plan.warnings?.length > 0 && (
            <div className="rounded-lg border border-accent-amber/25 bg-accent-amber/[0.04] px-3.5 py-2.5 flex flex-col gap-1">
              {plan.warnings.map((w, i) => (
                <div key={i} className="text-[12px] text-accent-amber/90 flex items-start gap-2">
                  <Icon name="alertTriangle" size={12} className="mt-0.5 shrink-0" />
                  <span>{w}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

window.RebalanceTab = RebalanceTab;
