/* CompanyReport renderer — renders the JSON returned by /api/companies/{t}/...
   Adapted from the cyberfolks_raport template to the dashboard's dark/glass
   aesthetic. Sections are conditionally rendered when data is present. */

const trendClass = (t) =>
  t === 'pos' ? 'text-accent-green'
  : t === 'neg' ? 'text-accent-red'
  : 'text-white/45';

function ReportHeader({ report }) {
  return (
    <div className="flex items-end justify-between gap-4 pb-4 border-b border-white/[0.08]">
      <div className="min-w-0">
        <div className="mono text-[10px] uppercase tracking-[0.22em] text-white/40">
          {report.eyebrow || 'Monitoring portfela · Snapshot'}
        </div>
        <h2 className="font-serif text-[28px] leading-tight text-white mt-1.5 truncate">
          {report.company_name}
        </h2>
        {report.sector && (
          <div className="text-[12px] text-white/55 mt-1">{report.sector}</div>
        )}
      </div>
      <div className="text-right mono text-[11px] text-white/45 leading-[1.6] shrink-0">
        <div>Ticker · <span className="text-white/85 font-medium">{report.ticker}</span></div>
        <div>Raport · <span className="text-white/85 font-medium">{report.report_date}</span></div>
        {report.confidence > 0 && (
          <div>Conf · <span className="text-white/85 font-medium">{report.confidence}/10</span></div>
        )}
      </div>
    </div>
  );
}

function KpiGrid({ kpis }) {
  if (!kpis?.length) return null;
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 border border-white/[0.07] rounded-lg overflow-hidden">
      {kpis.map((k, i) => (
        <div
          key={i}
          className="p-3.5 border-r border-b border-white/[0.05] last:border-r-0 [&:nth-child(4n)]:border-r-0 [&:nth-last-child(-n+4)]:border-b-0"
        >
          <div className="mono text-[9.5px] uppercase tracking-[0.14em] text-white/40 mb-1.5">
            {k.label}
          </div>
          <div className="font-serif text-[20px] leading-none text-white">
            {k.value}
          </div>
          {k.delta && (
            <div className={`mono text-[10.5px] mt-1.5 ${trendClass(k.trend)}`}>
              {k.delta}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function MarketGrid({ market }) {
  if (!market?.length) return null;
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-1">
      {market.map((m, i) => (
        <div key={i} className="flex justify-between gap-3 py-2 border-b border-dotted border-white/[0.07]">
          <span className="mono text-[10px] uppercase tracking-[0.1em] text-white/40 shrink-0">
            {m.label}
          </span>
          <span className="text-[12.5px] text-white/85 text-right truncate">
            {m.value}
          </span>
        </div>
      ))}
    </div>
  );
}

function BulletList({ items, kind, emptyText }) {
  if (!items?.length) {
    return (
      <div className="text-[12px] text-white/40 italic px-1 py-2">
        {emptyText || 'Brak danych.'}
      </div>
    );
  }
  const symbol = kind === 'strengths' ? '+' : (kind === 'risks' ? '△' : '—');
  const color  = kind === 'strengths' ? 'text-accent-green'
              : kind === 'risks'      ? 'text-accent-red'
              : 'text-accent-violet';
  return (
    <ul className="flex flex-col">
      {items.map((it, i) => {
        const isObj = typeof it === 'object';
        const text  = isObj ? it.text : it;
        // Support both the new `citations` array and the legacy single `citation`.
        const cites = isObj
          ? (Array.isArray(it.citations) ? it.citations : (it.citation ? [it.citation] : []))
          : [];
        const hl    = isObj ? it.highlight : null;
        return (
          <li
            key={i}
            className="flex items-start gap-2 py-2 border-b border-dotted border-white/[0.06] last:border-b-0"
          >
            <span className={`mono text-[12px] shrink-0 w-3 mt-0.5 ${color}`}>{symbol}</span>
            <div className="flex-1 min-w-0">
              <div className="text-[13px] text-white/85 leading-relaxed">
                {hl && <strong className="text-white mr-1">{hl}</strong>}
                {text}
                {cites.map((c, ci) => (
                  <span key={ci} className="ml-1.5 mono text-[10px] text-white/35 align-middle">
                    [{c}]
                  </span>
                ))}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function SectionHeading({ num, title }) {
  return (
    <div className="flex items-center justify-between pb-1.5 mb-3 border-b border-white/[0.05]">
      <span className="font-serif text-[10.5px] uppercase tracking-[0.22em] text-white/55">
        {title}
      </span>
      <span className="mono text-[10px] text-white/30">{num}</span>
    </div>
  );
}

function CompanyReport({ report, isFactsheet }) {
  if (!report) return null;
  return (
    <div className="flex flex-col gap-6">
      <ReportHeader report={report} />

      {/* 01 · Streszczenie */}
      <div>
        <SectionHeading num="01" title="Streszczenie" />
        <p className="font-serif text-[15.5px] leading-relaxed text-white/85">
          {report.tldr}
        </p>
        {isFactsheet && (
          <div className="mt-2 text-[11.5px] text-white/40 italic">
            Streszczenie generowane przez AI — kliknij „Generuj raport AI".
          </div>
        )}
      </div>

      {/* 02 · KPI */}
      <div>
        <SectionHeading
          num="02"
          title={report.kpi_section_title || 'Kluczowe wskaźniki'}
        />
        <KpiGrid kpis={report.kpis} />
      </div>

      {/* 03 · Market */}
      {report.market?.length > 0 && (
        <div>
          <SectionHeading num="03" title="Dane rynkowe i pozycja" />
          <MarketGrid market={report.market} />
        </div>
      )}

      {/* 04 · Strengths vs risks */}
      <div>
        <SectionHeading num="04" title="Mocne strony vs. ryzyka" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
          <div>
            <h4 className="font-serif text-[12.5px] uppercase tracking-[0.06em] text-white/65 mb-1.5">
              Mocne strony
            </h4>
            <BulletList
              items={report.strengths}
              kind="strengths"
              emptyText="Brak — wygeneruj raport AI."
            />
          </div>
          <div>
            <h4 className="font-serif text-[12.5px] uppercase tracking-[0.06em] text-white/65 mb-1.5">
              Ryzyka
            </h4>
            <BulletList
              items={report.risks}
              kind="risks"
              emptyText="Brak — wygeneruj raport AI."
            />
          </div>
        </div>
      </div>

      {/* 05 · Calendar */}
      {report.calendar?.length > 0 && (
        <div>
          <SectionHeading num="05" title="Kalendarz" />
          <BulletList items={report.calendar} kind="calendar" />
        </div>
      )}

      {/* Warnings */}
      {report.warnings?.length > 0 && (
        <div className="rounded-lg border border-accent-amber/25 bg-accent-amber/[0.04] px-3.5 py-2.5">
          {report.warnings.map((w, i) => (
            <div key={i} className="text-[12px] text-accent-amber/90 flex items-start gap-2">
              <Icon name="alertTriangle" size={12} className="mt-0.5 shrink-0" />
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="pt-3 mt-1 border-t border-white/[0.05] mono text-[10px] text-white/35 leading-[1.7]">
        <div>{report.sources}</div>
        <div className="mt-1">{report.disclaimer}</div>
      </div>
    </div>
  );
}

window.CompanyReport = CompanyReport;
