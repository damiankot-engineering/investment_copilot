/* UI primitives: Card, Button, Badge, Tabs, Dialog, Skeleton, Tooltip, Toast,
   CountUp, Markdown renderer. Exposed on window for the other modules. */

const { motion, AnimatePresence, useAnimation, useMotionValue, useTransform, animate } = window.Motion || {};
const { useEffect, useRef, useState, useMemo, useCallback, createContext, useContext, Fragment } = React;

/* ---------------------------- Card ---------------------------- */
function Card({ children, className = '', glow = false, ...rest }) {
  return (
    <div
      className={`glass rounded-2xl ${glow ? 'shadow-glow' : ''} ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
}

/* ---------------------------- Button ---------------------------- */
function Button({
  children,
  variant = 'default',
  size = 'md',
  icon,
  iconRight,
  loading = false,
  disabled = false,
  className = '',
  onClick,
  type = 'button',
  ...rest
}) {
  const base =
    'inline-flex items-center justify-center gap-2 font-medium tracking-tight rounded-xl transition-all duration-150 select-none whitespace-nowrap';
  const sizes = {
    sm: 'h-8 px-3 text-[12.5px]',
    md: 'h-9 px-3.5 text-[13px]',
    lg: 'h-11 px-5 text-[14px]',
  };
  const variants = {
    default: 'bg-white/[0.06] hover:bg-white/[0.1] text-white/90 border border-white/[0.08] hover:border-white/[0.14]',
    primary:
      'text-ink-950 bg-gradient-to-br from-accent-cyan to-accent-violet hover:brightness-110 shadow-[0_8px_30px_-12px_rgba(167,139,250,0.55)] border border-white/10',
    ghost:    'text-white/70 hover:text-white hover:bg-white/[0.06]',
    danger:   'text-accent-red bg-accent-red/10 hover:bg-accent-red/15 border border-accent-red/20',
    outline:  'text-white/85 border border-white/[0.12] hover:border-white/[0.22] hover:bg-white/[0.04]',
  };
  return (
    <motion.button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      whileHover={!disabled ? { y: -1 } : undefined}
      whileTap={!disabled ? { scale: 0.97 } : undefined}
      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
      className={`${base} ${sizes[size]} ${variants[variant]} ${disabled ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}
      {...rest}
    >
      {loading ? (
        <span className="inline-flex items-center gap-2">
          <span className="h-3.5 w-3.5 rounded-full border-2 border-current/30 border-t-current animate-spin" />
          <span className="opacity-80">{children}</span>
        </span>
      ) : (
        <>
          {icon && <Icon name={icon} size={size === 'lg' ? 16 : 14} />}
          <span>{children}</span>
          {iconRight && <Icon name={iconRight} size={size === 'lg' ? 16 : 14} />}
        </>
      )}
    </motion.button>
  );
}

/* ---------------------------- Badge ---------------------------- */
function Badge({ children, variant = 'default', pulse = false, icon, className = '' }) {
  const variants = {
    default: 'bg-white/[0.06] text-white/80 border-white/[0.08]',
    low:     'bg-accent-blue/10 text-accent-blue border-accent-blue/25',
    medium:  'bg-accent-amber/10 text-accent-amber border-accent-amber/25',
    high:    'bg-accent-red/10 text-accent-red border-accent-red/30',
    green:   'bg-accent-green/10 text-accent-green border-accent-green/25',
    red:     'bg-accent-red/10 text-accent-red border-accent-red/25',
    violet:  'bg-accent-violet/10 text-accent-violet border-accent-violet/25',
    cyan:    'bg-accent-cyan/10 text-accent-cyan border-accent-cyan/25',
    on_track: 'bg-accent-green/10 text-accent-green border-accent-green/25',
    watch:    'bg-accent-amber/10 text-accent-amber border-accent-amber/25',
    at_risk:  'bg-accent-red/10 text-accent-red border-accent-red/25',
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium border ${variants[variant] || variants.default} ${className}`}
    >
      {icon && <Icon name={icon} size={11} />}
      <span className={`relative ${pulse ? 'flex items-center gap-1.5' : ''}`}>
        {pulse && (
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full rounded-full bg-current opacity-60 animate-ping" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
          </span>
        )}
        {children}
      </span>
    </span>
  );
}

/* ---------------------------- Tabs ---------------------------- */
function Tabs({ value, onChange, items, className = '' }) {
  return (
    <div className={`relative inline-flex p-1 rounded-xl bg-white/[0.04] border border-white/[0.06] ${className}`}>
      {items.map(it => {
        const active = it.value === value;
        return (
          <button
            key={it.value}
            onClick={() => onChange(it.value)}
            className={`relative z-10 px-3.5 h-8 text-[12.5px] font-medium rounded-lg transition-colors ${
              active ? 'text-white' : 'text-white/55 hover:text-white/80'
            }`}
          >
            {active && (
              <motion.span
                layoutId="seg-bg"
                className="absolute inset-0 rounded-lg bg-gradient-to-br from-white/[0.12] to-white/[0.04] border border-white/[0.1]"
                transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              />
            )}
            <span className="relative z-10 inline-flex items-center gap-1.5">
              {it.icon && <Icon name={it.icon} size={12} />}
              {it.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/* ---------------------------- Dialog ---------------------------- */
function Dialog({ open, onClose, title, subtitle, children, footer, width = 'max-w-2xl' }) {
  useEffect(() => {
    if (!open) return;
    const onKey = e => { if (e.key === 'Escape') onClose && onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center p-6"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
        >
          <motion.div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={onClose}
          />
          <motion.div
            className={`relative w-full ${width} max-h-[88vh] flex flex-col glass rounded-2xl overflow-hidden`}
            initial={{ y: 12, opacity: 0, scale: 0.98 }}
            animate={{ y: 0,  opacity: 1, scale: 1 }}
            exit={{    y: 8,  opacity: 0, scale: 0.98 }}
            transition={{ type: 'spring', stiffness: 360, damping: 32 }}
          >
            <div className="flex items-start justify-between px-6 pt-5 pb-4 border-b border-white/[0.06]">
              <div>
                {title && <h3 className="text-[15px] font-semibold tracking-tight text-white">{title}</h3>}
                {subtitle && <p className="text-[12.5px] text-white/55 mt-0.5">{subtitle}</p>}
              </div>
              <button
                onClick={onClose}
                className="p-1.5 rounded-lg text-white/50 hover:text-white hover:bg-white/[0.06] transition-colors"
              >
                <Icon name="close" size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-6 py-5">{children}</div>
            {footer && (
              <div className="px-6 py-4 border-t border-white/[0.06] flex items-center justify-end gap-2 bg-black/20">
                {footer}
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* ---------------------------- Skeleton ---------------------------- */
function Skeleton({ className = '', style }) {
  return <div className={`shimmer rounded-md ${className}`} style={style} />;
}

/* ---------------------------- Tooltip ---------------------------- */
function Tooltip({ label, children, side = 'top' }) {
  const [hover, setHover] = useState(false);
  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {children}
      <AnimatePresence>
        {hover && (
          <motion.span
            initial={{ opacity: 0, y: side === 'top' ? 4 : -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{    opacity: 0, y: side === 'top' ? 4 : -4 }}
            transition={{ duration: 0.12 }}
            className={`pointer-events-none absolute z-40 ${side === 'top' ? 'bottom-full mb-1.5' : 'top-full mt-1.5'} left-1/2 -translate-x-1/2 whitespace-nowrap px-2 py-1 text-[11px] rounded-md bg-ink-900 border border-white/[0.08] text-white/85 shadow-lg`}
          >
            {label}
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}

/* ---------------------------- Toast ---------------------------- */
const ToastCtx = createContext(null);
function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const push = useCallback((toast) => {
    const id = Math.random().toString(36).slice(2);
    setToasts(t => [...t, { id, ...toast }]);
    setTimeout(() => {
      setToasts(t => t.filter(x => x.id !== id));
    }, toast.duration || 3800);
  }, []);
  const toast = useMemo(() => ({
    success: (title, description) => push({ kind: 'success', title, description }),
    error:   (title, description) => push({ kind: 'error',   title, description }),
    info:    (title, description) => push({ kind: 'info',    title, description }),
  }), [push]);
  return (
    <ToastCtx.Provider value={toast}>
      {children}
      <div className="fixed bottom-5 right-5 z-[60] flex flex-col gap-2 items-end">
        <AnimatePresence>
          {toasts.map(t => (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, x: 30, scale: 0.96 }}
              animate={{ opacity: 1, x: 0,  scale: 1 }}
              exit={{    opacity: 0, x: 30, scale: 0.96 }}
              transition={{ type: 'spring', stiffness: 380, damping: 32 }}
              className="glass min-w-[260px] max-w-[360px] rounded-xl px-4 py-3 flex items-start gap-3"
            >
              <div className={`mt-0.5 h-7 w-7 flex items-center justify-center rounded-lg ${
                t.kind === 'success' ? 'bg-accent-green/15 text-accent-green' :
                t.kind === 'error'   ? 'bg-accent-red/15 text-accent-red' :
                                       'bg-accent-cyan/15 text-accent-cyan'
              }`}>
                <Icon name={t.kind === 'success' ? 'check' : t.kind === 'error' ? 'alertTriangle' : 'info'} size={14} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-[13px] font-medium text-white">{t.title}</div>
                {t.description && <div className="text-[12px] text-white/55 mt-0.5">{t.description}</div>}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastCtx.Provider>
  );
}
const useToast = () => useContext(ToastCtx);

/* ---------------------------- CountUp ---------------------------- */
function CountUp({ value, decimals = 0, duration = 1.1, format, prefix = '', suffix = '', className = '' }) {
  const ref = useRef(null);
  const prev = useRef(0);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const from = prev.current;
    const to = Number(value) || 0;
    const start = performance.now();
    const ms = duration * 1000;
    let raf;
    const step = (now) => {
      const t = Math.min(1, (now - start) / ms);
      const eased = 1 - Math.pow(1 - t, 3);
      const v = from + (to - from) * eased;
      node.textContent = (prefix || '') + (format ? format(v) : v.toFixed(decimals)) + (suffix || '');
      if (t < 1) raf = requestAnimationFrame(step);
      else prev.current = to;
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value, decimals, duration, format, prefix, suffix]);
  return <span ref={ref} className={`num ${className}`}>{prefix}0{suffix}</span>;
}

/* ---------------------------- Formatters ---------------------------- */
const fmtPLN = (n, opts = {}) => {
  const { decimals = 0, signed = false } = opts;
  const sign = signed && n > 0 ? '+' : '';
  return (
    sign +
    new Intl.NumberFormat('pl-PL', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(n) +
    ' zł'
  );
};
const fmtPct = (n, opts = {}) => {
  const { decimals = 2, signed = false } = opts;
  const sign = signed && n > 0 ? '+' : '';
  return sign + n.toFixed(decimals) + '%';
};
const fmtNum = (n, decimals = 0) =>
  new Intl.NumberFormat('pl-PL', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).format(n);
const fmtBytes = (b) => b < 1024 ? `${b} B` : b < 1024 * 1024 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1024 / 1024).toFixed(2)} MB`;
const fmtRelTime = (iso) => {
  const d = new Date(iso);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60)   return 'przed chwilą';
  if (diff < 3600) return `${Math.round(diff / 60)} min temu`;
  if (diff < 86400) return `${Math.round(diff / 3600)} godz. temu`;
  return `${Math.round(diff / 86400)} dni temu`;
};
const fmtDateTime = (iso) => {
  const d = new Date(iso);
  return d.toLocaleString('pl-PL', { dateStyle: 'medium', timeStyle: 'short' });
};
const fmtDate = (iso) => new Date(iso).toLocaleDateString('pl-PL', { dateStyle: 'medium' });

/* ---------------------------- Markdown (tiny) ---------------------------- */
function renderMarkdown(src) {
  if (!src) return '';
  const escape = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const lines = src.split('\n');
  let html = '', inList = false;
  const flushList = () => { if (inList) { html += '</ul>'; inList = false; } };
  const inline = (s) =>
    escape(s)
      .replace(/`([^`]+)`/g,       '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g,     '<em>$1</em>');

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (/^###\s/.test(line)) { flushList(); html += `<h3>${inline(line.replace(/^###\s/, ''))}</h3>`; continue; }
    if (/^##\s/.test(line))  { flushList(); html += `<h2>${inline(line.replace(/^##\s/, ''))}</h2>`;  continue; }
    if (/^#\s/.test(line))   { flushList(); html += `<h1>${inline(line.replace(/^#\s/, ''))}</h1>`;   continue; }
    if (/^---+$/.test(line)) { flushList(); html += '<hr/>'; continue; }
    if (/^\s*[-*]\s/.test(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${inline(line.replace(/^\s*[-*]\s/, ''))}</li>`;
      continue;
    }
    if (/^\s*\d+\.\s/.test(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${inline(line.replace(/^\s*\d+\.\s/, ''))}</li>`;
      continue;
    }
    if (line.trim() === '') { flushList(); continue; }
    flushList();
    html += `<p>${inline(line)}</p>`;
  }
  flushList();
  return html;
}
function Markdown({ children, className = '' }) {
  return <div className={`md ${className}`} dangerouslySetInnerHTML={{ __html: renderMarkdown(children) }} />;
}

/* ---------------------------- Inputs ---------------------------- */
function Input({ value, onChange, placeholder, className = '', type = 'text', icon, ...rest }) {
  return (
    <div className={`relative ${className}`}>
      {icon && (
        <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-white/40">
          <Icon name={icon} size={13} />
        </span>
      )}
      <input
        type={type}
        value={value ?? ''}
        onChange={onChange}
        placeholder={placeholder}
        className={`w-full h-9 ${icon ? 'pl-8' : 'pl-3'} pr-3 rounded-lg bg-white/[0.03] border border-white/[0.08] hover:border-white/[0.14] focus:border-accent-violet/60 focus:bg-white/[0.05] text-[13px] text-white placeholder-white/30 outline-none transition-colors`}
        {...rest}
      />
    </div>
  );
}
function Textarea({ value, onChange, placeholder, rows = 3, className = '', ...rest }) {
  return (
    <textarea
      rows={rows}
      value={value ?? ''}
      onChange={onChange}
      placeholder={placeholder}
      className={`w-full px-3 py-2 rounded-lg bg-white/[0.03] border border-white/[0.08] hover:border-white/[0.14] focus:border-accent-violet/60 focus:bg-white/[0.05] text-[13px] text-white placeholder-white/30 outline-none transition-colors resize-y leading-relaxed ${className}`}
      {...rest}
    />
  );
}

/* ---------------------------- KeywordsInput ---------------------------- */
function KeywordsInput({ value = [], onChange, placeholder = 'Dodaj słowo kluczowe…' }) {
  const [draft, setDraft] = useState('');
  const commit = () => {
    const v = draft.trim();
    if (!v || value.includes(v)) return;
    onChange([...value, v]);
    setDraft('');
  };
  return (
    <div className="flex flex-wrap items-center gap-1.5 p-1.5 rounded-lg bg-white/[0.03] border border-white/[0.08] focus-within:border-accent-violet/60">
      {value.map((k, i) => (
        <span key={k + i} className="inline-flex items-center gap-1 pl-2 pr-1 py-0.5 rounded-md bg-accent-violet/10 text-accent-violet text-[11.5px] border border-accent-violet/20">
          <span>{k}</span>
          <button
            type="button"
            onClick={() => onChange(value.filter((_, j) => j !== i))}
            className="p-0.5 rounded hover:bg-white/[0.08] text-accent-violet/80 hover:text-white"
          >
            <Icon name="close" size={10} />
          </button>
        </span>
      ))}
      <input
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); commit(); }
          if (e.key === 'Backspace' && !draft && value.length) onChange(value.slice(0, -1));
        }}
        onBlur={commit}
        placeholder={value.length ? '' : placeholder}
        className="flex-1 min-w-[140px] bg-transparent text-[12.5px] text-white placeholder-white/30 outline-none px-1.5"
      />
    </div>
  );
}

/* ---------------------------- Section heading ---------------------------- */
function SectionHeader({ eyebrow, title, description, right }) {
  return (
    <div className="flex items-end justify-between gap-4 mb-4">
      <div>
        {eyebrow && (
          <div className="text-[11px] uppercase tracking-[0.16em] text-white/40 mb-1.5">{eyebrow}</div>
        )}
        <h2 className="text-[20px] font-semibold tracking-tight text-white">{title}</h2>
        {description && <p className="text-[13px] text-white/55 mt-1 max-w-2xl">{description}</p>}
      </div>
      {right}
    </div>
  );
}

/* ---------------------------- Empty state ---------------------------- */
function EmptyState({ emoji = '🪐', title, description, cta }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-14 px-6">
      <div className="text-[40px] mb-3 grayscale-[20%]">{emoji}</div>
      <div className="text-[15px] font-medium text-white">{title}</div>
      {description && <div className="text-[12.5px] text-white/50 mt-1.5 max-w-sm">{description}</div>}
      {cta && <div className="mt-4">{cta}</div>}
    </div>
  );
}

Object.assign(window, {
  Card, Button, Badge, Tabs, Dialog, Skeleton, Tooltip,
  ToastProvider, useToast,
  CountUp, Markdown,
  Input, Textarea, KeywordsInput,
  SectionHeader, EmptyState,
  fmtPLN, fmtPct, fmtNum, fmtBytes, fmtRelTime, fmtDateTime, fmtDate,
});
