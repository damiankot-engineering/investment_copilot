/* Monitoring tab */

const { motion: MMot } = window.Motion;

const STATUS_META = {
  on_track: { label: 'on track', variant: 'on_track', icon: 'check'         },
  watch:    { label: 'watch',    variant: 'watch',    icon: 'eye'           },
  at_risk:  { label: 'at risk',  variant: 'at_risk',  icon: 'alertTriangle' },
};

function MonitoringItemCard({ item, i }) {
  const meta = STATUS_META[item.status];
  const stripe = {
    on_track: 'from-accent-green to-accent-green/0',
    watch:    'from-accent-amber to-accent-amber/0',
    at_risk:  'from-accent-red   to-accent-red/0',
  }[item.status];
  return (
    <MMot.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.05 + i * 0.05 }}
      whileHover={{ y: -2 }}
      className="relative glass rounded-xl px-4 py-4 overflow-hidden"
    >
      <div className={`absolute left-0 top-3 bottom-3 w-0.5 bg-gradient-to-b ${stripe}`} />
      <div className="flex items-start gap-3 pl-2">
        <div className="h-10 w-10 rounded-lg bg-white/[0.04] border border-white/[0.06] flex items-center justify-center mono text-[11px] text-white shrink-0">
          {item.ticker}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant={meta.variant} pulse={item.status === 'at_risk'} icon={meta.icon}>
              {meta.label}
            </Badge>
            <span className="text-[12px] text-white/45">teza: <span className="text-white/75">{item.ticker}</span></span>
          </div>
          <p className="text-[12.5px] text-white/70 mt-2 leading-relaxed">{item.rationale}</p>
        </div>
      </div>
    </MMot.div>
  );
}

function HistoricalReportRow({ r, i }) {
  return (
    <MMot.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.03 + i * 0.03 }}
      className="flex items-center gap-3 px-3.5 py-2.5 rounded-lg hover:bg-white/[0.03] transition-colors group"
    >
      <Icon name="fileText" size={14} className="text-white/40" />
      <span className="mono text-[12.5px] text-white truncate">{r.name}</span>
      <span className="text-[11.5px] text-white/40 ml-auto mono">{fmtRelTime(r.mtime)}</span>
      <span className="text-[11.5px] text-white/40 mono">{fmtBytes(r.size_bytes)}</span>
      <a
        href={window.API.monitoringReportUrl(r.name)}
        target="_blank"
        rel="noopener"
        className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-md text-white/55 hover:text-white hover:bg-white/[0.06]"
        title="Otwórz raport HTML"
      >
        <Icon name="external" size={13} />
      </a>
    </MMot.div>
  );
}

function MonitoringTab() {
  const [data, setData] = React.useState({ generated_at: new Date().toISOString(), items: [], reports: [] });
  const [running, setRunning] = React.useState(false);
  const toast = useToast();

  const refreshReports = React.useCallback(async () => {
    try {
      const reports = await window.API.listMonitoringReports();
      setData(d => ({ ...d, reports }));
    } catch (err) {
      console.error(err);
    }
  }, []);

  React.useEffect(() => { refreshReports(); }, [refreshReports]);

  const counts = (data.items || []).reduce((a, x) => ({ ...a, [x.status]: (a[x.status] || 0) + 1 }), {});

  const run = async () => {
    setRunning(true);
    try {
      const result = await window.API.runMonitoring({ newsDaysBack: 30 });
      setData({
        generated_at: result.generated_at,
        items: result.items || [],
        reports: result.reports || [],
        report: result.report,
      });
      toast.success('Snapshot wykonany', `${(result.items || []).length} pozycji oceniono.`);
      if (result.warnings && result.warnings.length) {
        toast.info('Ostrzeżenia', result.warnings[0]);
      }
    } catch (err) {
      console.error(err);
      toast.error('Monitoring nie powiódł się', err.detail || err.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader
        eyebrow="Tezy w czasie"
        title="Monitoring"
        description="Cotygodniowy przegląd statusu tez dla każdej pozycji."
        right={
          <Button variant="primary" icon="activity" loading={running} onClick={run}>
            Uruchom snapshot
          </Button>
        }
      />

      {/* Summary row */}
      <div className="flex items-center gap-3 flex-wrap">
        <Badge variant="on_track" icon="check">on track · {counts.on_track || 0}</Badge>
        <Badge variant="watch" icon="eye">watch · {counts.watch || 0}</Badge>
        <Badge variant="at_risk" pulse={(counts.at_risk || 0) > 0} icon="alertTriangle">at risk · {counts.at_risk || 0}</Badge>
        <span className="text-[12px] text-white/45 ml-2 inline-flex items-center gap-1.5">
          <Icon name="clock" size={12} /> ostatni snapshot {fmtRelTime(data.generated_at)} · <span className="mono">{fmtDateTime(data.generated_at)}</span>
        </span>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1.4fr_1fr] gap-4">
        {/* Items */}
        <div className="flex flex-col gap-2.5">
          {data.items.map((it, i) => <MonitoringItemCard key={it.ticker} item={it} i={i} />)}
        </div>

        {/* Historical reports */}
        <Card className="px-3.5 pt-4 pb-3">
          <div className="flex items-center justify-between px-1.5 mb-2">
            <div>
              <div className="text-[15px] font-medium tracking-tight text-white">Historyczne raporty HTML</div>
              <div className="text-[12px] text-white/45 mt-0.5">Cotygodniowe snapshoty</div>
            </div>
            <Badge variant="default">{data.reports.length}</Badge>
          </div>
          <div className="flex flex-col">
            {data.reports.map((r, i) => <HistoricalReportRow key={r.name} r={r} i={i} />)}
          </div>
        </Card>
      </div>
    </div>
  );
}

window.MonitoringTab = MonitoringTab;
