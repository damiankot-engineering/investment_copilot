/* Calendar tab — upcoming reports + dividend estimates */

const { motion: CMot } = window.Motion;

const KIND_META = {
  report:    { icon: 'fileText',  label: 'Raport',      tone: 'cyan'   },
  dividend:  { icon: 'wallet',    label: 'Dywidenda',   tone: 'green'  },
  agm:       { icon: 'globe',     label: 'WZA',         tone: 'violet' },
  espi:      { icon: 'bell',      label: 'ESPI',        tone: 'amber'  },
  dividend_record:  { icon: 'calendar', label: 'Prawo do dywidendy', tone: 'green' },
  dividend_payment: { icon: 'calendar', label: 'Wypłata dywidendy',  tone: 'green' },
};

const IMPORTANCE_STRIPE = {
  high:   'bg-accent-red',
  medium: 'bg-accent-amber',
  low:    'bg-white/30',
};

function relDaysLabel(days) {
  if (days == null) return 'bez daty';
  if (days < 0) return `${Math.abs(days)} dni temu`;
  if (days === 0) return 'dzisiaj';
  if (days === 1) return 'jutro';
  if (days <= 7) return `za ${days} dni`;
  if (days <= 30) return `za ~${Math.round(days / 7)} tyg.`;
  return `za ~${Math.round(days / 30)} mies.`;
}

function CalendarEventCard({ event, i }) {
  const meta = KIND_META[event.kind] || KIND_META.report;
  const stripe = IMPORTANCE_STRIPE[event.importance] || 'bg-white/20';
  const toneColor = {
    cyan:   'text-accent-cyan   bg-accent-cyan/10',
    green:  'text-accent-green  bg-accent-green/10',
    violet: 'text-accent-violet bg-accent-violet/10',
    amber:  'text-accent-amber  bg-accent-amber/10',
  }[meta.tone];
  return (
    <CMot.div
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.04 + i * 0.04 }}
      whileHover={{ y: -1 }}
      className="relative glass rounded-xl px-4 py-3.5 overflow-hidden"
    >
      <div className={`absolute left-0 top-3 bottom-3 w-0.5 rounded-full ${stripe}`} />
      <div className="flex items-start gap-3 pl-2">
        <div className={`h-10 w-10 shrink-0 rounded-lg flex items-center justify-center ${toneColor}`}>
          <Icon name={meta.icon} size={15} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="mono text-[11.5px] font-semibold text-white bg-white/[0.04] border border-white/[0.06] px-1.5 py-0.5 rounded">
              {event.display_ticker}
            </span>
            {event.name && (
              <span className="text-[12.5px] text-white/65 truncate">{event.name}</span>
            )}
            <Badge variant="default">{meta.label}</Badge>
            {event.importance === 'high' && (
              <Badge variant="high" pulse>najbliższe</Badge>
            )}
          </div>
          <div className="text-[13.5px] font-medium text-white mt-1.5">{event.label}</div>
          {event.description && (
            <p className="text-[12.5px] text-white/55 mt-0.5 leading-relaxed">{event.description}</p>
          )}
        </div>
        <div className="text-right shrink-0 min-w-[100px]">
          {event.event_date ? (
            <>
              <div className="mono text-[12.5px] text-white">{fmtDate(event.event_date)}</div>
              <div className="text-[11px] text-white/45 mt-0.5">{relDaysLabel(event.days_until)}</div>
            </>
          ) : event.amount_pln != null ? (
            <>
              <div className="mono text-[12.5px] text-accent-green">~{fmtPLN(event.amount_pln, { decimals: 0 })}</div>
              <div className="text-[11px] text-white/45 mt-0.5">rocznie est.</div>
            </>
          ) : (
            <div className="text-[11px] text-white/35">TBA</div>
          )}
        </div>
      </div>
    </CMot.div>
  );
}

function CalendarTab() {
  const [data, setData] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const toast = useToast();

  const load = React.useCallback(async () => {
    try {
      const bundle = await window.API.getCalendar();
      setData(bundle);
    } catch (err) {
      console.error(err);
      toast.error('Nie można wczytać kalendarza', err.detail || err.message);
    } finally {
      setLoading(false);
    }
  }, [toast]);

  React.useEffect(() => { load(); }, [load]);

  const events = data?.events || [];
  const reports = events.filter(e => e.kind === 'report');
  const dividends = events.filter(e => e.kind === 'dividend');
  const highCount = events.filter(e => e.importance === 'high').length;
  const totalDividendsEst = dividends.reduce((a, e) => a + (e.amount_pln || 0), 0);

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader
        eyebrow="Forward-looking"
        title="Kalendarz"
        description="Raporty kwartalne z BiznesRadar + estymowane roczne dywidendy. Daty rzeczywistych wypłat dywidend wymagają osobnego scrapera (TODO)."
        right={
          <Button variant="outline" icon="refresh" onClick={load}>Odśwież</Button>
        }
      />

      {/* Summary strip */}
      <div className="flex items-center gap-3 flex-wrap text-[12px]">
        <Badge variant="default" icon="fileText">raporty · {reports.length}</Badge>
        <Badge variant="default" icon="wallet">dywidendy · {dividends.length}</Badge>
        {highCount > 0 && <Badge variant="high" pulse>{highCount} najbliższych (≤14 dni)</Badge>}
        {totalDividendsEst > 0 && (
          <span className="text-white/55 mono text-[12px]">
            ~{fmtPLN(totalDividendsEst, { decimals: 0 })} rocznie est.
          </span>
        )}
        <span className="text-white/35 ml-auto">
          {data?.snapshot_age_days != null
            ? `Dane z monitoring snapshot sprzed ${data.snapshot_age_days} dni`
            : 'Brak monitoring snapshot'}
        </span>
      </div>

      {/* Warnings (e.g. no snapshot yet) */}
      {data?.warnings?.length > 0 && (
        <div className="glass-soft rounded-lg px-4 py-3 border-l-2 border-accent-amber/50 text-[12.5px] text-white/75">
          {data.warnings.map((w, i) => <div key={i}>{w}</div>)}
        </div>
      )}

      {loading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-[80px] w-full" />)}
        </div>
      ) : events.length === 0 ? (
        <EmptyState
          emoji="📅"
          title="Brak nadchodzących wydarzeń"
          description={
            <>
              Aby zobaczyć daty raportów, uruchom <span className="text-white/80 font-medium">Monitoring → Uruchom snapshot</span>
              — to pobierze daty z BiznesRadar.
              <br />
              <span className="text-[11.5px] text-white/40">Daty wypłat dywidend nie są jeszcze obsługiwane (planowane).</span>
            </>
          }
        />
      ) : (
        <div className="flex flex-col gap-2.5">
          {events.map((e, i) => <CalendarEventCard key={`${e.ticker}-${e.kind}-${i}`} event={e} i={i} />)}
        </div>
      )}
    </div>
  );
}

window.CalendarTab = CalendarTab;
