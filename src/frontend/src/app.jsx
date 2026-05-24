/* App shell — composes sidebar + tab views, owns top-level state. */

const { motion: APMot, AnimatePresence: APAP } = window.Motion;

function TopBar({ asOf, onRefresh, refreshing }) {
  const [q, setQ] = React.useState('');
  return (
    <div className="flex items-center justify-between gap-4 px-6 py-3.5 border-b border-white/[0.05] bg-gradient-to-b from-white/[0.02] to-transparent">
      <div className="flex items-center gap-3 flex-1 min-w-0">
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
  const [portfolio, setPortfolio] = React.useState(MOCK_PORTFOLIO);
  const [loading, setLoading] = React.useState(true);
  const [refreshing, setRefreshing] = React.useState(false);
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

  const onRefresh = async () => {
    setRefreshing(true);
    try {
      const result = await window.API.updateData(14);
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
    }
  };

  const onUpdatePortfolio = async (rows) => {
    const payload = {
      base_currency: portfolio?.base_currency || 'PLN',
      holdings: rows.map(r => ({
        ticker: r.ticker,
        name: r.name || null,
        shares: Number(r.shares),
        entry_price: Number(r.entry_price),
        entry_date: r.entry_date,
        thesis: r.thesis,
        keywords: Array.isArray(r.keywords) ? r.keywords : [],
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
  const tabs = [
    { id: 'portfolio',  node: <PortfolioTab portfolio={portfolio} onUpdatePortfolio={onUpdatePortfolio} onRefresh={onRefresh} refreshing={refreshing} benchmarkLabel={appConfig.benchmark_label} /> },
    { id: 'watchlist',  node: <WatchlistTab /> },
    { id: 'calendar',   node: <CalendarTab /> },
    { id: 'backtest',   node: <BacktestTab /> },
    { id: 'analysis',   node: <AnalysisTab /> },
    { id: 'reports',    node: <ReportsTab /> },
    { id: 'monitoring', node: <MonitoringTab /> },
  ];

  return (
    <div className="flex h-screen w-screen">
      <Sidebar active={active} onChange={setActive} asOf={portfolio.as_of} />
      <main className="flex-1 flex flex-col min-w-0">
        <TopBar asOf={portfolio.as_of} onRefresh={onRefresh} refreshing={refreshing} />
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-[1480px] mx-auto px-8 py-7">
            {tabs.map(t => (
              <APMot.div
                key={t.id}
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
