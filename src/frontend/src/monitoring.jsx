/* Monitoring tab — per-company snapshots backed by /api/companies/*

   The legacy "Uruchom snapshot" bulk flow is gone. Each holding gets its
   own card; the deterministic factsheet (KPI grid + market data) loads
   instantly, and clicking "Generuj raport AI" runs one focused LLM call
   for that ticker (must-cite TL;DR + strengths + risks). Calendar across
   the whole portfolio is shown at the top. */

const { motion: MMot, AnimatePresence: MAP } = window.Motion;

const trendInline = (t) =>
  t === 'pos' ? 'text-accent-green'
  : t === 'neg' ? 'text-accent-red'
  : 'text-white/75';

// ---------------------------------------------------------- Upcoming card

function UpcomingReports({ items, loading }) {
  if (loading) {
    return (
      <Card className="px-4 py-3.5">
        <div className="flex items-center gap-3">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-4 w-40" />
        </div>
      </Card>
    );
  }
  if (!items?.length) {
    return (
      <Card className="px-4 py-3.5 text-[12.5px] text-white/55">
        <div className="flex items-center gap-2">
          <Icon name="calendar" size={13} className="text-white/40" />
          <span>Brak nadchodzących wydarzeń w cache BR. Odśwież dane.</span>
        </div>
      </Card>
    );
  }
  return (
    <Card className="px-4 py-3 overflow-hidden">
      <div className="flex items-center gap-2 mb-2.5 px-1">
        <Icon name="calendar" size={13} className="text-accent-violet" />
        <span className="text-[12.5px] font-medium text-white">Najbliższe wydarzenia</span>
        <span className="mono text-[11px] text-white/35 ml-1">{items.length}</span>
        <span className="text-[10.5px] text-white/35 ml-2">raporty kwartalne · dywidendy</span>
      </div>
      <div className="flex flex-col">
        {items.slice(0, 8).map((it, i) => (
          <div
            key={i}
            className="flex items-center gap-3 py-2 px-1 border-b border-dotted border-white/[0.05] last:border-b-0"
          >
            <span className="mono text-[11.5px] text-accent-cyan shrink-0 w-[88px]">
              {it.highlight}
            </span>
            <span className="text-[12.5px] text-white/75 flex-1 truncate">
              {it.text}
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}

// ------------------------------------------------------------ Company card

function CompanyCard({ holding, i, onExpand, expanded }) {
  const [factsheet, setFactsheet] = React.useState(null);
  const [aiReport, setAiReport] = React.useState(null);
  const [loadingFs, setLoadingFs] = React.useState(true);
  const [generating, setGenerating] = React.useState(false);
  const [err, setErr] = React.useState(null);
  const toast = useToast();

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadingFs(true);
      setErr(null);
      try {
        const [fs, cached] = await Promise.all([
          window.API.getCompanyFactsheet(holding.ticker),
          window.API.getCachedCompanyReport(holding.ticker).catch(() => null),
        ]);
        if (cancelled) return;
        setFactsheet(fs);
        if (cached) setAiReport(cached);
      } catch (e) {
        if (cancelled) return;
        setErr(e.detail || e.message);
      } finally {
        if (!cancelled) setLoadingFs(false);
      }
    })();
    return () => { cancelled = true; };
  }, [holding.ticker]);

  const onGenerate = async () => {
    setGenerating(true);
    try {
      const full = await window.API.generateCompanyReport(holding.ticker);
      setAiReport(full);
      toast.success('Raport AI wygenerowany', holding.ticker);
      if (full.warnings?.length) {
        toast.info('Ostrzeżenia', full.warnings[0]);
      }
    } catch (e) {
      toast.error('Generowanie raportu nie powiodło się', e.detail || e.message);
    } finally {
      setGenerating(false);
    }
  };

  const onRegenerate = async () => {
    if (!window.confirm('Wygenerować nowy raport AI? Poprzedni zostanie nadpisany.')) return;
    await onGenerate();
  };

  const report = aiReport || factsheet;
  const hasAi = !!aiReport;
  const previewKpis = report?.kpis?.slice(0, 4) || [];

  return (
    <MMot.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.04 + i * 0.04 }}
      className="glass rounded-2xl overflow-hidden"
    >
      <button
        type="button"
        onClick={onExpand}
        className="w-full flex items-start gap-4 p-4 text-left hover:bg-white/[0.02] transition-colors"
      >
        <div className="h-11 w-11 rounded-lg bg-white/[0.04] border border-white/[0.06] flex items-center justify-center mono text-[11px] text-white shrink-0">
          {holding.ticker.split('.')[0].slice(0, 6).toUpperCase()}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[14px] font-medium text-white truncate">
              {report?.company_name || holding.name || holding.ticker}
            </span>
            <span className="mono text-[11px] text-white/40">{holding.ticker}</span>
            {hasAi && (
              <Badge variant="default" icon="sparkles">
                AI · conf {report.confidence}/10
              </Badge>
            )}
            {!hasAi && !loadingFs && (
              <Badge variant="default">factsheet</Badge>
            )}
          </div>
          {loadingFs ? (
            <div className="flex gap-3 mt-2.5">
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-4 w-20" />
            </div>
          ) : (
            <div className="flex gap-4 mt-2 flex-wrap">
              {previewKpis.map((k, idx) => (
                <div key={idx} className="flex items-baseline gap-1.5">
                  <span className="mono text-[10px] text-white/40 uppercase tracking-wide">
                    {k.label}
                  </span>
                  <span className={`text-[13px] font-medium ${trendInline(k.trend)}`}>
                    {k.value}
                  </span>
                </div>
              ))}
            </div>
          )}
          {err && (
            <div className="text-[12px] text-accent-red mt-2 flex items-center gap-1.5">
              <Icon name="alertTriangle" size={12} />
              {err}
            </div>
          )}
        </div>
        <div className="shrink-0 flex items-center gap-1.5 mt-1">
          {!hasAi && !loadingFs && !err && (
            <span
              role="button"
              onClick={(e) => { e.stopPropagation(); onGenerate(); }}
              className={`px-3 h-8 rounded-md text-[12px] font-medium border inline-flex items-center gap-1.5 transition-colors ${
                generating
                  ? 'bg-accent-violet/10 text-accent-violet border-accent-violet/30 cursor-wait'
                  : 'bg-accent-violet/15 text-accent-violet hover:bg-accent-violet/25 border-accent-violet/30 cursor-pointer'
              }`}
            >
              {generating ? <Icon name="loader" size={12} className="animate-spin" /> : <Icon name="sparkles" size={12} />}
              {generating ? 'Generuję…' : 'Generuj AI'}
            </span>
          )}
          <Icon
            name="chevronRight"
            size={14}
            className={`text-white/40 transition-transform ${expanded ? 'rotate-90' : ''}`}
          />
        </div>
      </button>

      <MAP initial={false}>
        {expanded && report && !err && (
          <MMot.div
            key="body"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 pt-1 border-t border-white/[0.05]">
              <div className="flex items-center justify-end gap-2 mb-4 -mt-1">
                <a
                  href={window.API.companyReportHtmlUrl(holding.ticker)}
                  target="_blank"
                  rel="noopener"
                  className="inline-flex items-center gap-1.5 px-2.5 h-8 rounded-md text-[12px] text-white/75 hover:text-white border border-white/[0.08] hover:border-white/[0.16] transition-colors"
                  title="Otwórz standalone HTML raport"
                >
                  <Icon name="external" size={12} /> HTML
                </a>
                {hasAi ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    icon="refresh"
                    loading={generating}
                    onClick={onRegenerate}
                  >
                    Regeneruj AI
                  </Button>
                ) : (
                  <Button
                    variant="primary"
                    size="sm"
                    icon="sparkles"
                    loading={generating}
                    onClick={onGenerate}
                  >
                    Generuj raport AI
                  </Button>
                )}
              </div>

              <CompanyReport report={report} isFactsheet={!hasAi} />
            </div>
          </MMot.div>
        )}
      </MAP>
    </MMot.div>
  );
}

// ---------------------------------------------------------------- Main tab

function MonitoringTab() {
  const [portfolio, setPortfolio] = React.useState(null);
  const [upcoming, setUpcoming] = React.useState([]);
  const [loadingPf, setLoadingPf] = React.useState(true);
  const [loadingUp, setLoadingUp] = React.useState(true);
  const [expandedTicker, setExpandedTicker] = React.useState(null);
  const toast = useToast();

  React.useEffect(() => {
    (async () => {
      try {
        const pf = await window.API.getPortfolio();
        setPortfolio(pf);
      } catch (e) {
        toast.error('Nie można wczytać portfela', e.detail || e.message);
      } finally {
        setLoadingPf(false);
      }
    })();
  }, [toast]);

  React.useEffect(() => {
    (async () => {
      try {
        const u = await window.API.listUpcomingReports();
        setUpcoming(u);
      } catch (e) {
        console.error('Failed to load upcoming reports:', e);
      } finally {
        setLoadingUp(false);
      }
    })();
  }, []);

  const holdings = portfolio?.holdings || [];

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader
        eyebrow="Snapshot per spółka"
        title="Monitoring"
        description="Deterministyczne fact-sheety z BR + OHLCV. Pełna analiza AI on-demand per spółka, z must-cite walidacją cytowań."
      />

      <UpcomingReports items={upcoming} loading={loadingUp} />

      {loadingPf ? (
        <div className="flex flex-col gap-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="px-4 py-4">
              <div className="flex items-start gap-4">
                <Skeleton className="h-11 w-11 rounded-lg" />
                <div className="flex-1">
                  <Skeleton className="h-4 w-40 mb-2" />
                  <Skeleton className="h-3 w-3/4" />
                </div>
              </div>
            </Card>
          ))}
        </div>
      ) : holdings.length === 0 ? (
        <EmptyState
          emoji="📊"
          title="Brak pozycji w portfelu"
          description="Dodaj pozycje w zakładce Portfel aby uruchomić monitoring."
        />
      ) : (
        <div className="flex flex-col gap-3">
          {holdings.map((h, i) => (
            <CompanyCard
              key={h.ticker}
              holding={h}
              i={i}
              expanded={expandedTicker === h.ticker}
              onExpand={() =>
                setExpandedTicker((t) => (t === h.ticker ? null : h.ticker))
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}

window.MonitoringTab = MonitoringTab;
