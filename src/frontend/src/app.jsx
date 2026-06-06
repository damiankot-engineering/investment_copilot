/* App shell — composes sidebar + tab views, owns top-level state. */

const { motion: APMot, AnimatePresence: APAP } = window.Motion;

function TopBar({ asOf, onRefresh, refreshing, activePortfolioId, onSwitchPortfolio }) {
  const [q, setQ] = React.useState('');
  return (
    <div className="flex items-center justify-between gap-4 px-6 py-3.5 border-b border-white/[0.05] bg-gradient-to-b from-white/[0.02] to-transparent">
      <div className="flex items-center gap-3 flex-1 min-w-0">
        <PortfolioSwitcher activeId={activePortfolioId} onSwitch={onSwitchPortfolio} />
        <div className="relative flex-1 max-w-md">
          <Input
            icon="search"
            value={q}
            onChange={e => setQ(e.target.value)}
            placeholder="Szukaj pozycji, słowa kluczowego, raportu…"
            className=""
          />
        </div>
      </div>
      <div className="flex items-center gap-2">
        <div className="hidden md:flex items-center gap-2 text-[11.5px] text-white/45">
          <div className="relative h-1.5 w-1.5">
            <span className="absolute inset-0 rounded-full bg-accent-green animate-ping opacity-70" />
            <span className="relative block h-1.5 w-1.5 rounded-full bg-accent-green" />
          </div>
          <span className="mono">GPW · live</span>
          <span className="text-white/25">·</span>
          <span>{fmtRelTime(asOf)}</span>
        </div>
        <Tooltip label="Powiadomienia">
          <button className="h-9 w-9 rounded-lg text-white/55 hover:text-white border border-white/[0.06] hover:border-white/[0.12] flex items-center justify-center transition-colors relative">
            <Icon name="bell" size={14} />
            <span className="absolute top-2 right-2 h-1.5 w-1.5 rounded-full bg-accent-red" />
          </button>
        </Tooltip>
        <Tooltip label="Ustawienia">
          <button className="h-9 w-9 rounded-lg text-white/55 hover:text-white border border-white/[0.06] hover:border-white/[0.12] flex items-center justify-center transition-colors">
            <Icon name="settings" size={14} />
          </button>
        </Tooltip>
        <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-accent-cyan/30 to-accent-violet/30 border border-white/[0.08] flex items-center justify-center text-[11px] font-semibold text-white">
          MK
        </div>
      </div>
    </div>
  );
}

function Disclaimer() {
  return (
    <div className="text-center text-[11px] text-white/30 py-4 px-6">
      Investment Copilot is a research tool, not financial advice.
    </div>
  );
}

function App() {
  const [active, setActive] = React.useState('portfolio');
  // Multi-portfolio: the active id is restored from localStorage and pushed
  // into the API client (so the first status fetch is already scoped) before
  // any child effect runs.
  const [activePortfolioId, setActivePortfolioId] = React.useState(() => {
    let id = 'default';
    try { id = localStorage.getItem('activePortfolio') || 'default'; } catch (_) {}
    window.API.setActivePortfolio(id);
    return id;
  });
  const [portfolio, setPortfolio] = React.useState(MOCK_PORTFOLIO);
  const [loading, setLoading] = React.useState(true);
  const [refreshing, setRefreshing] = React.useState(false);
  const [refreshProgress, setRefreshProgress] = React.useState(null);
  const [appConfig, setAppConfig] = React.useState({ benchmark: 'wig20', benchmark_label: 'WIG20' });
  const toast = useToast();

  React.useEffect(() => {
    (async () => {
      try {
        const cfg = await window.API.getConfig();
        setAppConfig(cfg);
      } catch (err) {
        console.error('Could not load /api/config:', err);
      }
    })();
  }, []);

  const loadStatus = React.useCallback(async () => {
    try {
      const status = await window.API.getPortfolioStatus();
      setPortfolio(status);
    } catch (err) {
      console.error('Failed to load portfolio status:', err);
      toast.error('Nie można wczytać portfela', err.detail || err.message);
      setPortfolio(MOCK_PORTFOLIO); // fallback so the UI still renders
    } finally {
      setLoading(false);
    }
  }, [toast]);

  React.useEffect(() => { loadStatus(); }, [loadStatus]);

  const onSwitchPortfolio = (id) => {
    if (id === activePortfolioId) return;
    window.API.setActivePortfolio(id);
    try { localStorage.setItem('activePortfolio', id); } catch (_) {}
    setActivePortfolioId(id);   // re-keys the tab container → all tabs refetch
    setLoading(true);
    loadStatus();               // reload the Portfolio tab's status immediately
  };

  const onRefresh = async () => {
    setRefreshing(true);
    setRefreshProgress({ label: 'Łączenie…', pct: 0 });
    try {
      let ohlcvTotal = 0;
      let ohlcvDone = 0;
      const result = await window.API.streamUpdateData({
        newsDaysBack: 14,
        onEvent: (ev) => {
          if (ev.type === 'stage' && ev.status === 'start' && ev.name === 'ohlcv') {
            ohlcvTotal = ev.total || 0;
            setRefreshProgress({ label: 'Pobieranie OHLCV…', pct: 0 });
          } else if (ev.type === 'ohlcv_tick') {
            ohlcvDone = ev.idx;
            const pct = ohlcvTotal ? Math.round((ohlcvDone / ohlcvTotal) * 60) : 0; // OHLCV = 0-60%
            setRefreshProgress({
              label: `OHLCV ${ev.ticker} (${ev.idx}/${ev.total})${ev.status === 'failed' ? ' — błąd' : ''}`,
              pct,
            });
          } else if (ev.type === 'stage' && ev.name === 'benchmark') {
            setRefreshProgress({
              label: ev.status === 'start' ? 'Pobieranie benchmarka…' : `Benchmark: ${ev.rows} rzędów`,
              pct: ev.status === 'start' ? 65 : 75,
            });
          } else if (ev.type === 'stage' && ev.name === 'news') {
            setRefreshProgress({
              label: ev.status === 'start' ? 'Pobieranie newsów…' : `Newsy: ${ev.inserted} nowych`,
              pct: ev.status === 'start' ? 80 : 95,
            });
          }
        },
      });
      setRefreshProgress({ label: 'Gotowe', pct: 100 });
      const failed = Object.keys(result.ohlcv_failed || {}).length;
      await loadStatus();
      toast.success(
        'Dane zaktualizowane',
        `OHLCV: ${Object.keys(result.ohlcv_updated).length} · Newsy: ${result.news_inserted}` +
          (failed ? ` · błędów: ${failed}` : ''),
      );
    } catch (err) {
      toast.error('Aktualizacja nie powiodła się', err.detail || err.message);
    } finally {
      setRefreshing(false);
      setTimeout(() => setRefreshProgress(null), 1200);  // briefly show "Gotowe"
    }
  };

  const onUpdatePortfolio = async (rows) => {
    const payload = {
      base_currency: portfolio?.base_currency || 'PLN',
      holdings: rows.map(r => ({
        ticker: r.ticker,
        name: r.name || null,
        thesis: r.thesis || '',
        keywords: Array.isArray(r.keywords) ? r.keywords : [],
        transactions: (r.transactions || []).map(tx => ({
          date: tx.date,
          action: tx.action,
          shares: Number(tx.shares),
          price_per_share: Number(tx.price_per_share),
          fees: Number(tx.fees) || 0,
          note: tx.note || '',
        })),
      })),
    };
    try {
      await window.API.putPortfolio(payload);
      await loadStatus();
      toast.success('Portfel zapisany', `${rows.length} pozycji zaktualizowanych.`);
    } catch (err) {
      toast.error('Zapis portfela nie powiódł się', err.detail || err.message);
    }
  };

  // Render all tabs persistently and toggle visibility, so each tab keeps
  // its own state (backtest results, AI analysis, monitoring snapshot, etc.)
  // across tab switches. Only the entrance animation runs once per tab.
  // 'analysis' (<AnalysisTab/>) and 'reports' (<ReportsTab/>) are intentionally
  // not mounted — hidden from the UI per request. Their .jsx files + backends
  // stay in place; reports now live per feature rather than as one tab.
  const tabs = [
    { id: 'portfolio',  node: <PortfolioTab portfolio={portfolio} onUpdatePortfolio={onUpdatePortfolio} onRefresh={onRefresh} refreshing={refreshing} refreshProgress={refreshProgress} benchmarkLabel={appConfig.benchmark_label} /> },
    { id: 'watchlist',  node: <WatchlistTab /> },
    { id: 'backtest',   node: <BacktestTab /> },
    { id: 'rebalance',  node: <RebalanceTab /> },
    { id: 'monitoring', node: <MonitoringTab /> },
  ];

  return (
    <div className="flex h-screen w-screen">
      <Sidebar active={active} onChange={setActive} asOf={portfolio.as_of} />
      <main className="flex-1 flex flex-col min-w-0">
        <TopBar
          asOf={portfolio.as_of}
          onRefresh={onRefresh}
          refreshing={refreshing}
          activePortfolioId={activePortfolioId}
          onSwitchPortfolio={onSwitchPortfolio}
        />
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-[1480px] mx-auto px-8 py-7">
            {tabs.map(t => (
              <APMot.div
                key={`${t.id}:${activePortfolioId}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
                style={{ display: active === t.id ? 'block' : 'none' }}
              >
                {t.node}
              </APMot.div>
            ))}
          </div>
          <Disclaimer />
        </div>
      </main>
    </div>
  );
}

function Root() {
  return (
    <ToastProvider>
      <App />
    </ToastProvider>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<Root />);
