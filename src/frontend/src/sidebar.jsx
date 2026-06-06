/* Sidebar */

const SIDEBAR_NAV = [
  { id: 'portfolio',  label: 'Portfel',        icon: 'wallet',   hint: 'Pozycje i KPI' },
  { id: 'watchlist',  label: 'Watchlist',      icon: 'eye',      hint: 'Tickery do śledzenia + raporty' },
  { id: 'backtest',   label: 'Backtest',       icon: 'barChart', hint: 'Strategie i metryki' },
  { id: 'rebalance',  label: 'Rebalancing',    icon: 'target',   hint: 'Plan zleceń do alokacji docelowej' },
  // 'analysis' (Analiza AI) and 'reports' (concatenated Markdown reports) are
  // hidden from the UI — components + backends kept. Reports now live per
  // feature (company HTML report, backtest, …), not as one concatenated tab.
  { id: 'monitoring', label: 'Monitoring',     icon: 'activity', hint: 'Raporty per spółka + kalendarz' },
];

function Sidebar({ active, onChange, asOf }) {
  const { motion } = window.Motion;
  // Parallax via refs (no re-renders so framer-motion entry animations aren't cancelled)
  const ref = React.useRef(null);
  const blob1 = React.useRef(null);
  const blob2 = React.useRef(null);
  React.useEffect(() => {
    let raf;
    const onMove = e => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const rect = ref.current?.getBoundingClientRect();
        if (!rect) return;
        const x = (e.clientX - (rect.left + rect.width / 2)) / 40;
        const y = (e.clientY - (rect.top  + rect.height / 2)) / 60;
        if (blob1.current) blob1.current.style.transform = `translate3d(${x}px, ${y}px, 0)`;
        if (blob2.current) blob2.current.style.transform = `translate3d(${-x * 1.2}px, ${-y * 1.2}px, 0)`;
      });
    };
    window.addEventListener('mousemove', onMove);
    return () => { window.removeEventListener('mousemove', onMove); cancelAnimationFrame(raf); };
  }, []);

  return (
    <aside
      ref={ref}
      className="relative h-full w-[244px] shrink-0 flex flex-col px-3 py-4 border-r border-white/[0.05] bg-gradient-to-b from-white/[0.025] to-transparent"
    >
      {/* decorative gradient blob */}
      <div
        ref={blob1}
        aria-hidden
        className="absolute -top-10 -left-10 h-40 w-40 rounded-full blur-3xl opacity-40 pointer-events-none"
        style={{
          background: 'radial-gradient(circle, #22d3ee 0%, transparent 70%)',
          transition: 'transform 600ms cubic-bezier(0.22, 1, 0.36, 1)',
        }}
      />
      <div
        ref={blob2}
        aria-hidden
        className="absolute bottom-12 -left-12 h-44 w-44 rounded-full blur-3xl opacity-30 pointer-events-none"
        style={{
          background: 'radial-gradient(circle, #a78bfa 0%, transparent 70%)',
          transition: 'transform 700ms cubic-bezier(0.22, 1, 0.36, 1)',
        }}
      />

      {/* Logo */}
      <div className="relative flex items-center gap-2.5 px-2 mb-6">
        <div className="relative h-9 w-9 rounded-xl bg-gradient-to-br from-accent-cyan to-accent-violet flex items-center justify-center shadow-glow">
          <Icon name="copilot" size={18} className="text-ink-950" strokeWidth={2.2} />
          <span className="absolute inset-0 rounded-xl ring-1 ring-white/20"></span>
        </div>
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-tight text-white">
            Investment <span className="gradient-text">Copilot</span>
          </div>
          <div className="text-[10.5px] uppercase tracking-[0.2em] text-white/35 mt-0.5">GPW · PL</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="relative flex flex-col gap-0.5">
        {SIDEBAR_NAV.map((it, i) => {
          const isActive = active === it.id;
          return (
            <motion.button
              key={it.id}
              onClick={() => onChange(it.id)}
              whileHover={{ x: 1 }}
              transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              className={`relative group flex items-center gap-2.5 px-2.5 h-9 rounded-lg text-[13px] transition-colors ${
                isActive ? 'text-white' : 'text-white/55 hover:text-white/90'
              }`}
            >
              {isActive ? (
                <React.Fragment key="layout">
                  <motion.span
                    layoutId="sidebar-active"
                    className="absolute inset-0 rounded-lg bg-gradient-to-r from-white/[0.08] to-white/[0.02] border border-white/[0.08]"
                    transition={{ type: 'spring', stiffness: 460, damping: 36 }}
                  />
                  <motion.span
                    layoutId="sidebar-accent"
                    className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full bg-gradient-to-b from-accent-cyan to-accent-violet"
                  />
                </React.Fragment>
              ) : null}
              <span className={`relative z-10 ${isActive ? 'text-accent-violet' : ''}`}>
                <Icon name={it.icon} size={15} strokeWidth={1.8} />
              </span>
              <span className="relative z-10 font-medium whitespace-nowrap">{it.label}</span>
              <span className="relative z-10 ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
                <Icon name="chevronRight" size={12} className="text-white/40" />
              </span>
            </motion.button>
          );
        })}
      </nav>

      <div className="flex-1" />

      {/* Footer: status pill */}
      <div className="relative">
        <div className="glass-soft rounded-xl px-3 py-2.5 flex items-center gap-2.5">
          <div className="relative h-2 w-2">
            <span className="absolute inset-0 rounded-full bg-accent-green animate-ping opacity-75" />
            <span className="relative block h-2 w-2 rounded-full bg-accent-green" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[11.5px] text-white/85 font-medium">Dane GPW · live</div>
            <div className="text-[10.5px] text-white/40 mono truncate">{fmtRelTime(asOf)}</div>
          </div>
          <Icon name="globe" size={13} className="text-white/40" />
        </div>
        <div className="mt-3 px-1 text-[10px] leading-relaxed text-white/30">
          Narzędzie badawcze. Nie stanowi porady inwestycyjnej.
        </div>
      </div>
    </aside>
  );
}

window.Sidebar = Sidebar;
window.SIDEBAR_NAV = SIDEBAR_NAV;
