/* AI Analysis tab */

const { motion: AMot, AnimatePresence: AAP } = window.Motion;

function SeverityIcon({ severity }) {
  if (severity === 'high')   return <Icon name="alertTriangle" size={13} />;
  if (severity === 'medium') return <Icon name="alertCircle"   size={13} />;
  return <Icon name="info" size={13} />;
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
            <p className="text-[12.5px] text-white/70 leading-relaxed flex-1">{t.assessment}</p>
          </AMot.div>
        ))}
      </div>
    </Card>
  );
}

function AnalysisTab() {
  const [loading, setLoading] = React.useState(false);
  const [data, setData] = React.useState(null);
  const toast = useToast();

  const runAnalysis = async () => {
    setLoading(true);
    setData(null);
    try {
      const bundle = await window.API.runAnalysis();
      setData({
        summary: bundle.analysis,           // { summary_md, thesis_updates, confidence } or null
        alerts: bundle.alerts || [],
        risk_overview: bundle.risk_overview,
        warnings: bundle.warnings || [],
      });
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

  const severityCounts = data ? data.alerts.reduce((a, x) => ({ ...a, [x.severity]: (a[x.severity] || 0) + 1 }), {}) : {};

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader
        eyebrow="LLM / Polski"
        title="Analiza AI"
        description="Podsumowanie portfela, ocena tez i alerty ryzyka generowane przez model językowy."
        right={
          <div className="flex items-center gap-2">
            <Button variant="outline" icon="settings">Konfiguracja modelu</Button>
            <Button variant="primary" icon="sparkles" loading={loading} onClick={runAnalysis}>
              Uruchom analizę
            </Button>
          </div>
        }
      />

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
            <Badge variant="violet" icon="sparkles">claude · polski</Badge>
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
