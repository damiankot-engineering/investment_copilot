/* AI Analysis tab */

const { motion: AMot, AnimatePresence: AAP } = window.Motion;

function SeverityIcon({ severity }) {
  if (severity === 'high')   return <Icon name="alertTriangle" size={13} />;
  if (severity === 'medium') return <Icon name="alertCircle"   size={13} />;
  return <Icon name="info" size={13} />;
}

function ConfidenceBadge({ value }) {
  if (value == null) return null;
  const tier =
    value >= 8 ? { label: 'wysoka',    color: 'text-accent-green  bg-accent-green/10  border-accent-green/25'  } :
    value >= 6 ? { label: 'średnia',   color: 'text-accent-amber  bg-accent-amber/10  border-accent-amber/25'  } :
                 { label: 'niska',     color: 'text-accent-red    bg-accent-red/10    border-accent-red/25'    };
  return (
    <span
      title={`Pewność modelu: ${value}/10`}
      className={`inline-flex items-center gap-1.5 mono text-[10.5px] px-2 py-0.5 rounded border ${tier.color}`}
    >
      <Icon name="sparkles" size={10} />
      <span>conf {value}/10 · {tier.label}</span>
    </span>
  );
}

function CitationChips({ citations }) {
  if (!citations || citations.length === 0) return null;
  const chipColor = {
    news:            'text-accent-cyan   bg-accent-cyan/10   border-accent-cyan/25',
    metric:          'text-accent-violet bg-accent-violet/10 border-accent-violet/25',
    fundamentals:    'text-accent-green  bg-accent-green/10  border-accent-green/25',
    previous_report: 'text-accent-amber  bg-accent-amber/10  border-accent-amber/25',
  };
  return (
    <div className="flex flex-wrap items-center gap-1.5 mt-2">
      <span className="text-[10px] uppercase tracking-[0.14em] text-white/35 mr-1">Źródła:</span>
      {citations.map((c, idx) => (
        <span
          key={idx}
          title={`${c.source_type}: ${c.reference}`}
          className={`mono text-[10.5px] px-1.5 py-0.5 rounded border ${chipColor[c.source_type] || 'text-white/55 bg-white/[0.04] border-white/[0.08]'}`}
        >
          {c.reference}
        </span>
      ))}
    </div>
  );
}

function RiskAlertCard({ alert, i }) {
  const variantBg = {
    high:   'border-accent-red/20    bg-accent-red/[0.04]',
    medium: 'border-accent-amber/20  bg-accent-amber/[0.04]',
    low:    'border-accent-blue/20   bg-accent-blue/[0.04]',
  };
  const iconColor = {
    high:   'text-accent-red bg-accent-red/15',
    medium: 'text-accent-amber bg-accent-amber/15',
    low:    'text-accent-blue bg-accent-blue/15',
  };
  return (
    <AMot.div
      initial={{ opacity: 0, y: 10, scale: 0.985 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay: 0.05 + i * 0.06, duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -2 }}
      className={`glass rounded-xl p-4 border ${variantBg[alert.severity]} relative overflow-hidden`}
    >
      <div className="flex items-start gap-3">
        <div className={`h-9 w-9 shrink-0 rounded-lg flex items-center justify-center ${iconColor[alert.severity]} ${alert.severity === 'high' ? 'pulse-red' : ''}`}>
          <SeverityIcon severity={alert.severity} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant={alert.severity} pulse={alert.severity === 'high'}>
              {alert.severity === 'high' ? 'wysoki' : alert.severity === 'medium' ? 'średni' : 'niski'}
            </Badge>
            {alert.ticker && (
              <span className="mono text-[11.5px] text-white/65 bg-white/[0.05] border border-white/[0.06] px-1.5 py-0.5 rounded">
                {alert.ticker}
              </span>
            )}
          </div>
          <div className="text-[13.5px] font-semibold text-white mt-2 tracking-tight">{alert.title}</div>
          <p className="text-[12.5px] text-white/65 mt-1 leading-relaxed">{alert.description}</p>
          <CitationChips citations={alert.citations} />
        </div>
      </div>
    </AMot.div>
  );
}

function ThesisUpdatesList({ items }) {
  return (
    <Card className="px-5 pt-4 pb-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="text-[15px] font-medium tracking-tight text-white">Aktualizacje tez</div>
          <div className="text-[12px] text-white/45 mt-0.5">Per ticker — krótka ocena</div>
        </div>
        <Badge variant="violet" icon="sparkles">AI</Badge>
      </div>
      <div className="flex flex-col gap-2">
        {items.map((t, i) => (
          <AMot.div
            key={t.ticker}
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            className="flex items-start gap-3 py-2 border-b border-white/[0.04] last:border-0"
          >
            <div className="mono text-[11.5px] font-semibold text-white bg-white/[0.04] border border-white/[0.06] rounded px-1.5 py-0.5 mt-0.5">
              {t.ticker}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-[12.5px] text-white/70 leading-relaxed whitespace-pre-wrap">{t.assessment}</p>
              <CitationChips citations={t.citations} />
            </div>
          </AMot.div>
        ))}
      </div>
    </Card>
  );
}

function QuantMetricsSection({ metrics }) {
  if (!metrics) return null;
  const fmtPct1 = (v, signed = false) =>
    v == null ? '—' : (signed && v > 0 ? '+' : '') + v.toFixed(1) + '%';
  const fmtNum = (v, d = 2) => (v == null ? '—' : v.toFixed(d));

  const hhi = metrics.hhi;
  const hhiTier =
    hhi == null ? { label: '—', color: 'text-white/50' } :
    hhi > 2500   ? { label: 'wysoka',     color: 'text-accent-red'    } :
    hhi > 1500   ? { label: 'umiarkowana', color: 'text-accent-amber'  } :
                   { label: 'niska',      color: 'text-accent-green'  };

  return (
    <Card className="px-5 pt-4 pb-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="text-[10.5px] uppercase tracking-[0.16em] text-white/45">Metryki ilościowe</div>
          <div className="text-[15px] font-medium tracking-tight text-white mt-0.5">Pre-computed dla LLM</div>
        </div>
        <Badge variant="default">{metrics.n_priced}/{metrics.n_holdings} z danymi</Badge>
      </div>

      {/* Portfolio-level summary tiles */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5 mb-4">
        <div className="glass-soft rounded-lg px-3 py-2.5">
          <div className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">HHI</div>
          <div className={`mt-1 text-[16px] font-semibold mono ${hhiTier.color}`}>
            {hhi == null ? '—' : hhi.toFixed(0)}
          </div>
          <div className="text-[10.5px] text-white/40 mt-0.5">koncentracja {hhiTier.label}</div>
        </div>
        <div className="glass-soft rounded-lg px-3 py-2.5">
          <div className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Top 3 waga</div>
          <div className="mt-1 text-[16px] font-semibold mono text-white">{fmtPct1(metrics.top3_weight_pct)}</div>
          <div className="text-[10.5px] text-white/40 mt-0.5">suma 3 największych</div>
        </div>
        <div className="glass-soft rounded-lg px-3 py-2.5">
          <div className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Największa</div>
          <div className="mt-1 text-[16px] font-semibold mono text-white">
            {metrics.largest_position_display_ticker || '—'}
          </div>
          <div className="text-[10.5px] text-white/40 mt-0.5">
            {fmtPct1(metrics.largest_position_weight_pct)} portfela
          </div>
        </div>
        <div className="glass-soft rounded-lg px-3 py-2.5">
          <div className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Benchmark (β)</div>
          <div className="mt-1 text-[16px] font-semibold mono text-white">
            {metrics.benchmark_symbol ? metrics.benchmark_symbol.toUpperCase() : '—'}
          </div>
          <div className="text-[10.5px] text-white/40 mt-0.5">do liczenia bety</div>
        </div>
      </div>

      {/* Per-holding table */}
      <div className="overflow-x-auto -mx-5 px-5">
        <table className="w-full">
          <thead>
            <tr className="text-[10.5px] uppercase tracking-[0.14em] text-white/40 border-y border-white/[0.05]">
              <th className="text-left  font-medium px-2 py-2">Ticker</th>
              <th className="text-right font-medium px-2 py-2">Waga</th>
              <th className="text-right font-medium px-2 py-2">Ret 30d</th>
              <th className="text-right font-medium px-2 py-2">Ret 90d</th>
              <th className="text-right font-medium px-2 py-2">Ret 252d</th>
              <th className="text-right font-medium px-2 py-2">Od 52w high</th>
              <th className="text-right font-medium px-2 py-2">Vol ann.</th>
              <th className="text-right font-medium px-2 py-2">β</th>
            </tr>
          </thead>
          <tbody>
            {metrics.holdings.map((h) => {
              const ret30Pos = (h.ret_30d_pct ?? 0) >= 0;
              const ret252Pos = (h.ret_252d_pct ?? 0) >= 0;
              return (
                <tr key={h.ticker} className="border-b border-white/[0.03] last:border-0">
                  <td className="px-2 py-2 mono text-[12px] text-white">{h.display_ticker}</td>
                  <td className="px-2 py-2 text-right mono text-[12px] text-white/80">{fmtPct1(h.weight_pct)}</td>
                  <td className={`px-2 py-2 text-right mono text-[12px] ${ret30Pos ? 'text-accent-green' : 'text-accent-red'}`}>
                    {fmtPct1(h.ret_30d_pct, true)}
                  </td>
                  <td className="px-2 py-2 text-right mono text-[12px] text-white/70">{fmtPct1(h.ret_90d_pct, true)}</td>
                  <td className={`px-2 py-2 text-right mono text-[12px] ${ret252Pos ? 'text-accent-green' : 'text-accent-red'}`}>
                    {fmtPct1(h.ret_252d_pct, true)}
                  </td>
                  <td className="px-2 py-2 text-right mono text-[12px] text-accent-amber">
                    {fmtPct1(h.distance_from_52w_high_pct, true)}
                  </td>
                  <td className="px-2 py-2 text-right mono text-[12px] text-white/70">{fmtPct1(h.ann_volatility_pct)}</td>
                  <td className="px-2 py-2 text-right mono text-[12px] text-white/80">{fmtNum(h.beta_vs_benchmark, 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Correlations */}
      {metrics.top_correlations?.length > 0 && (
        <div className="mt-4">
          <div className="text-[10.5px] uppercase tracking-[0.14em] text-white/40 mb-2">Top korelacje (dzienne log-returny)</div>
          <div className="flex flex-wrap gap-2">
            {metrics.top_correlations.map((c, i) => {
              const strong = Math.abs(c.correlation) > 0.5;
              return (
                <span
                  key={i}
                  className={`mono text-[11.5px] px-2 py-1 rounded border ${
                    strong
                      ? 'text-accent-red bg-accent-red/10 border-accent-red/25'
                      : 'text-white/70 bg-white/[0.04] border-white/[0.08]'
                  }`}
                  title={`corr.${c.ticker_a}.${c.ticker_b}`}
                >
                  {c.display_a}–{c.display_b}: {(c.correlation >= 0 ? '+' : '') + c.correlation.toFixed(2)}
                </span>
              );
            })}
          </div>
        </div>
      )}
    </Card>
  );
}

function AnalysisTab() {
  const [loading, setLoading] = React.useState(false);
  const [data, setData] = React.useState(null);
  const [generatedAt, setGeneratedAt] = React.useState(null);
  const [fromCache, setFromCache] = React.useState(false);
  const [loadingCached, setLoadingCached] = React.useState(true);
  const toast = useToast();

  const ingestBundle = (bundle, { cached }) => {
    setData({
      summary: bundle.analysis,
      alerts: bundle.alerts || [],
      risk_overview: bundle.risk_overview,
      metrics: bundle.metrics,
      warnings: bundle.warnings || [],
    });
    setGeneratedAt(bundle.generated_at || null);
    setFromCache(cached);
  };

  // Try to show the previously persisted bundle without spending tokens.
  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cached = await window.API.getCachedAnalysis();
        if (cancelled) return;
        if (cached) ingestBundle(cached, { cached: true });
      } catch (err) {
        console.error('Cached analysis load failed:', err);
      } finally {
        if (!cancelled) setLoadingCached(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const runAnalysis = async () => {
    setLoading(true);
    setData(null);
    setGeneratedAt(null);
    setFromCache(false);
    try {
      const bundle = await window.API.runAnalysis();
      ingestBundle(bundle, { cached: false });
      if (bundle.analysis) {
        toast.success('Analiza gotowa', 'Wygenerowano podsumowanie i alerty.');
      } else if (bundle.warnings && bundle.warnings.length) {
        toast.error('Analiza częściowa', bundle.warnings[0]);
      }
    } catch (err) {
      console.error(err);
      toast.error('Analiza nie powiodła się', err.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const onRegenerate = async () => {
    if (!window.confirm('Wygenerować nową analizę? Cached zostanie nadpisany.')) return;
    await runAnalysis();
  };

  const severityCounts = data ? data.alerts.reduce((a, x) => ({ ...a, [x.severity]: (a[x.severity] || 0) + 1 }), {}) : {};

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader
        eyebrow="LLM / Polski"
        title="Analiza AI"
        description="Podsumowanie portfela, ocena tez i alerty ryzyka generowane przez model językowy."
        right={
          <div className="flex items-center gap-2">
            {data ? (
              <Button variant="primary" icon="refresh" loading={loading} onClick={onRegenerate}>
                Regeneruj analizę
              </Button>
            ) : (
              <Button variant="primary" icon="sparkles" loading={loading} onClick={runAnalysis}>
                Uruchom analizę
              </Button>
            )}
          </div>
        }
      />

      {fromCache && generatedAt && (
        <div className="flex items-center gap-2 -mt-2 text-[11.5px] text-white/55">
          <Icon name="clock" size={12} className="text-white/40" />
          <span>
            Wczytano z cache · wygenerowano {fmtRelTime(generatedAt)} ·{' '}
            <span className="mono text-white/70">{fmtDateTime(generatedAt)}</span>
          </span>
          <span className="text-white/30 ml-1">
            (cache jest invalidowany po Aktualizuj dane)
          </span>
        </div>
      )}

      {/* Severity strip */}
      {data && (
        <div className="flex items-center gap-3 text-[12px]">
          <span className="text-white/45">Alerty:</span>
          <Badge variant="high" pulse={(severityCounts.high || 0) > 0}>
            wysokich {severityCounts.high || 0}
          </Badge>
          <Badge variant="medium">średnich {severityCounts.medium || 0}</Badge>
          <Badge variant="low">niskich {severityCounts.low || 0}</Badge>
        </div>
      )}

      {(data || loading) && (
      <div className="grid grid-cols-1 xl:grid-cols-[1.4fr_1fr] gap-4">
        {/* Summary panel */}
        <Card className="px-6 pt-5 pb-5 relative overflow-hidden">
          <div className="absolute -top-12 -right-12 h-40 w-40 rounded-full blur-3xl opacity-25 pointer-events-none"
               style={{ background: 'radial-gradient(circle, #a78bfa 0%, transparent 70%)' }} />
          <div className="flex items-center justify-between mb-4 relative">
            <div>
              <div className="text-[10.5px] uppercase tracking-[0.16em] text-white/45">Podsumowanie portfela</div>
              <div className="text-[16px] font-semibold tracking-tight text-white mt-1">Synteza — maj 2026</div>
            </div>
            <div className="flex items-center gap-2">
              {data?.summary?.confidence != null && <ConfidenceBadge value={data.summary.confidence} />}
              <Badge variant="violet" icon="sparkles">claude · polski</Badge>
            </div>
          </div>
          {loading || !data ? (
            <div className="flex flex-col gap-2">
              <Skeleton className="h-5 w-3/4" />
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-3 w-11/12" />
              <Skeleton className="h-3 w-9/12" />
              <div className="h-3" />
              <Skeleton className="h-4 w-1/2" />
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-3 w-10/12" />
            </div>
          ) : data.summary ? (
            <AMot.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
            >
              <Markdown>{data.summary.summary_md}</Markdown>
            </AMot.div>
          ) : (
            <div className="text-[12.5px] text-white/55">
              Podsumowanie niedostępne (LLM error). {data.warnings?.[0] || ''}
            </div>
          )}
        </Card>

        {/* Alerts list */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div className="text-[15px] font-medium tracking-tight text-white">Alerty ryzyka</div>
            <span className="text-[11.5px] text-white/45">{data ? data.alerts.length : 0} pozycji</span>
          </div>
          {data?.risk_overview && (
            <div className="glass-soft rounded-lg px-3.5 py-2.5 text-[12.5px] text-white/70 leading-relaxed border-l-2 border-accent-violet/40">
              {data.risk_overview}
            </div>
          )}
          <div className="flex flex-col gap-2.5">
            {loading || !data ? (
              Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-[110px]" />)
            ) : (
              data.alerts
                .slice()
                .sort((a, b) => ({ high: 0, medium: 1, low: 2 }[a.severity] - { high: 0, medium: 1, low: 2 }[b.severity]))
                .map((a, i) => <RiskAlertCard key={i} alert={a} i={i} />)
            )}
          </div>
        </div>
      </div>
      )}

      {/* Quant metrics — what the LLM was citing */}
      {data && data.metrics && <QuantMetricsSection metrics={data.metrics} />}

      {/* Thesis updates */}
      {data && data.summary && <ThesisUpdatesList items={data.summary.thesis_updates} />}

      {!data && !loading && (
        <EmptyState
          emoji="🪐"
          title="Analiza AI jeszcze nie uruchomiona"
          description={
            <>
              Kliknij <span className="text-white/80 font-medium">„Uruchom analizę”</span>, aby model językowy wygenerował
              podsumowanie portfela, ocenę tez i alerty ryzyka po polsku.
              <br />
              <span className="text-[11.5px] text-white/40">Wymaga ustawionego <span className="mono">GROQ_API_KEY</span> oraz aktualnych danych w cache.</span>
            </>
          }
          cta={<Button variant="primary" icon="sparkles" onClick={runAnalysis}>Uruchom analizę</Button>}
        />
      )}
    </div>
  );
}

window.AnalysisTab = AnalysisTab;
