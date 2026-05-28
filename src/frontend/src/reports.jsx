/* Reports tab */

const { motion: RMot } = window.Motion;

function ReportRow({ report, i, onView, onDelete }) {
  return (
    <RMot.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.04 + i * 0.04 }}
      whileHover={{ y: -1 }}
      className="glass rounded-xl px-4 py-3.5 flex items-center gap-3 group"
    >
      <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-accent-cyan/20 to-accent-violet/20 border border-white/[0.06] flex items-center justify-center">
        <Icon name="fileBarChart" size={16} className="text-accent-violet" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <div className="mono text-[13px] font-medium text-white truncate">{report.name}</div>
          <Badge variant="default">md</Badge>
        </div>
        <div className="text-[11.5px] text-white/45 mt-0.5 flex items-center gap-2">
          <span className="inline-flex items-center gap-1"><Icon name="clock" size={11} />{fmtRelTime(report.mtime)}</span>
          <span className="text-white/20">·</span>
          <span className="mono">{fmtDateTime(report.mtime)}</span>
          <span className="text-white/20">·</span>
          <span className="mono">{fmtBytes(report.size_bytes)}</span>
        </div>
      </div>
      <div className="flex items-center gap-1.5 opacity-70 group-hover:opacity-100 transition-opacity">
        <Button variant="ghost" size="sm" icon="eye" onClick={() => onView(report)}>Podgląd</Button>
        <Button variant="ghost" size="sm" icon="download">Pobierz</Button>
        <button
          onClick={() => onDelete(report)}
          title="Usuń raport"
          className="h-7 w-7 rounded-md text-white/45 hover:text-accent-red hover:bg-accent-red/10 border border-white/[0.06] hover:border-accent-red/25 flex items-center justify-center transition-colors"
        >
          <Icon name="trash" size={13} />
        </button>
      </div>
    </RMot.div>
  );
}

function ReportsTab() {
  const [reports, setReports] = React.useState([]);
  const [generating, setGenerating] = React.useState(false);
  const [viewing, setViewing] = React.useState(null);
  const [loadingContent, setLoadingContent] = React.useState(false);
  const toast = useToast();

  const refresh = React.useCallback(async () => {
    try {
      const list = await window.API.listReports();
      setReports(list);
    } catch (err) {
      console.error(err);
      toast.error('Nie można wczytać raportów', err.detail || err.message);
    }
  }, [toast]);

  React.useEffect(() => { refresh(); }, [refresh]);

  const generate = async () => {
    setGenerating(true);
    try {
      const resp = await window.API.generateReport({ strategy: 'ma_crossover' });
      await refresh();
      toast.success('Raport wygenerowany', resp.report.name);
      if (resp.warnings && resp.warnings.length) {
        toast.info('Ostrzeżenia', resp.warnings[0]);
      }
    } catch (err) {
      console.error(err);
      toast.error('Generowanie nie powiodło się', err.detail || err.message);
    } finally {
      setGenerating(false);
    }
  };

  const onDelete = async (report) => {
    if (!window.confirm(`Usunąć raport ${report.name}?\n\nTej operacji nie da się cofnąć.`)) return;
    try {
      await window.API.deleteReport(report.name);
      await refresh();
      toast.success('Raport usunięty', report.name);
    } catch (err) {
      console.error(err);
      toast.error('Usunięcie nie powiodło się', err.detail || err.message);
    }
  };

  const openReport = async (r) => {
    setViewing({ ...r, content_md: '' });
    setLoadingContent(true);
    try {
      const full = await window.API.getReport(r.name);
      setViewing(full);
    } catch (err) {
      toast.error('Nie można otworzyć raportu', err.detail || err.message);
      setViewing(null);
    } finally {
      setLoadingContent(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader
        eyebrow="Pliki"
        title="Raporty"
        description="Markdownowe przeglądy portfela. Generuj nowy lub przeglądaj historyczne."
        right={
          <Button variant="primary" icon="fileText" loading={generating} onClick={generate}>
            Generuj raport
          </Button>
        }
      />

      {generating && (
        <Card className="px-5 py-4 flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-accent-violet/15 text-accent-violet flex items-center justify-center">
            <Icon name="sparkles" size={14} />
          </div>
          <div className="flex-1">
            <div className="text-[13px] text-white">Generowanie raportu…</div>
            <div className="mt-2 h-1 rounded-full overflow-hidden bg-white/[0.05]">
              <RMot.div
                className="h-full bg-gradient-to-r from-accent-cyan to-accent-violet"
                initial={{ width: 0 }}
                animate={{ width: '100%' }}
                transition={{ duration: 1.3 }}
              />
            </div>
          </div>
        </Card>
      )}

      <div className="flex flex-col gap-2.5">
        {reports.length === 0 ? (
          <EmptyState
            emoji="📄"
            title="Brak raportów"
            description="Wygeneruj pierwszy raport, aby zobaczyć go tutaj."
            cta={<Button variant="primary" icon="fileText" onClick={generate}>Generuj raport</Button>}
          />
        ) : (
          reports.map((r, i) => (
            <ReportRow key={r.name} report={r} i={i} onView={openReport} onDelete={onDelete} />
          ))
        )}
      </div>

      <Dialog
        open={!!viewing}
        onClose={() => setViewing(null)}
        title={viewing?.name}
        subtitle={viewing ? `${fmtDateTime(viewing.mtime)} · ${fmtBytes(viewing.size_bytes)}` : ''}
        width="max-w-3xl"
        footer={
          <>
            <Button variant="ghost" onClick={() => setViewing(null)}>Zamknij</Button>
            {viewing && (
              <a
                href={window.API.downloadReportUrl(viewing.name)}
                download={viewing.name}
                className="inline-flex"
              >
                <Button variant="primary" icon="download">Pobierz .md</Button>
              </a>
            )}
          </>
        }
      >
        {viewing && (loadingContent || !viewing.content_md
          ? <div className="flex flex-col gap-2"><Skeleton className="h-4 w-1/2" /><Skeleton className="h-3 w-full" /><Skeleton className="h-3 w-11/12" /><Skeleton className="h-3 w-10/12" /></div>
          : <Markdown>{viewing.content_md}</Markdown>
        )}
      </Dialog>
    </div>
  );
}

window.ReportsTab = ReportsTab;
