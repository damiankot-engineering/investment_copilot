/* Watchlist tab */

const { motion: WMot } = window.Motion;

function WatchlistRow({ item, i }) {
  const hasTarget = item.target_buy_price != null;
  const hasPrice = item.last_price != null;
  return (
    <WMot.tr
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.05 + i * 0.04 }}
      whileHover={{ y: -1, backgroundColor: 'rgba(255,255,255,0.025)' }}
      className={`group border-b border-white/[0.03] last:border-0 ${item.alert ? 'bg-accent-green/[0.04]' : ''}`}
    >
      <td className="px-5 py-3.5">
        <div className="flex items-center gap-2.5">
          <div className="h-7 w-7 rounded-md bg-white/[0.04] border border-white/[0.06] flex items-center justify-center mono text-[10.5px] text-white/85">
            {item.display_ticker.slice(0, 3)}
          </div>
          <span className="mono text-[12.5px] font-semibold text-white" title={item.ticker}>
            {item.display_ticker}
          </span>
        </div>
      </td>
      <td className="px-3 py-3.5 text-[12.5px] text-white/80">{item.name || '—'}</td>
      <td className="px-3 py-3.5 text-right mono text-[12.5px] text-white/55">
        {fmtDate(item.added_date)}
      </td>
      <td className="px-3 py-3.5 text-right mono text-[12.5px] text-white">
        {hasPrice ? fmtPLN(item.last_price, { decimals: 2 }) : <span className="text-white/30">brak danych</span>}
      </td>
      <td className="px-3 py-3.5 text-right mono text-[12.5px] text-white/70">
        {hasTarget ? fmtPLN(item.target_buy_price, { decimals: 2 }) : '—'}
      </td>
      <td className="px-3 py-3.5 text-right">
        {hasPrice && hasTarget ? (
          <span
            className={`mono text-[12.5px] font-medium inline-flex items-center gap-1 ${
              item.alert ? 'text-accent-green' : 'text-accent-amber'
            }`}
          >
            {item.alert && <Icon name="check" size={11} />}
            {fmtPct(item.distance_to_target_pct, { signed: true, decimals: 1 })}
          </span>
        ) : (
          <span className="text-white/30">—</span>
        )}
      </td>
      <td className="px-3 py-3.5 text-right">
        {item.news_count_30d > 0 ? (
          <span
            className="inline-flex items-center gap-1 mono text-[11.5px] text-accent-cyan bg-accent-cyan/10 border border-accent-cyan/20 px-1.5 py-0.5 rounded"
            title="Newsy w cache (ostatnie 30 dni)"
          >
            <Icon name="bell" size={10} />
            {item.news_count_30d}
          </span>
        ) : (
          <span className="text-white/30">—</span>
        )}
      </td>
      <td className="px-5 py-3.5 text-[12px] text-white/60 whitespace-pre-wrap leading-relaxed min-w-[280px] max-w-[480px]">
        {item.notes || '—'}
      </td>
    </WMot.tr>
  );
}

function EditWatchlistDialog({ open, onClose, items, onSave }) {
  const today = new Date().toISOString().slice(0, 10);
  const [rows, setRows] = React.useState(() => items.map(it => ({ ...it })));
  React.useEffect(() => { if (open) setRows(items.map(it => ({ ...it }))); }, [open, items]);

  const update = (i, patch) => setRows(r => r.map((x, j) => (j === i ? { ...x, ...patch } : x)));
  const remove = (i) => setRows(r => r.filter((_, j) => j !== i));
  const add = () => setRows(r => [
    ...r,
    { ticker: '', name: '', added_date: today, target_buy_price: null, notes: '', keywords: [] },
  ]);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Edytuj watchlist"
      subtitle="Tickery śledzone, ale nieposiadane"
      width="max-w-3xl"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>Anuluj</Button>
          <Button variant="primary" icon="check" onClick={() => onSave(rows)}>Zapisz zmiany</Button>
        </>
      }
    >
      <div className="flex flex-col gap-3">
        {rows.map((r, i) => (
          <WMot.div
            key={i}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.03 }}
            className="glass-soft rounded-xl p-4"
          >
            <div className="flex items-start gap-3 mb-3">
              <div className="grid grid-cols-12 gap-2 flex-1">
                <div className="col-span-3">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Ticker</label>
                  <Input value={r.ticker} onChange={e => update(i, { ticker: e.target.value })} placeholder="np. ccc.pl" className="mt-1" />
                </div>
                <div className="col-span-4">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Nazwa</label>
                  <Input value={r.name || ''} onChange={e => update(i, { name: e.target.value })} placeholder="CCC" className="mt-1" />
                </div>
                <div className="col-span-3">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Cena docelowa</label>
                  <Input
                    type="number"
                    step="0.01"
                    value={r.target_buy_price ?? ''}
                    onChange={e => update(i, { target_buy_price: e.target.value === '' ? null : Number(e.target.value) })}
                    placeholder="opcjonalnie"
                    className="mt-1"
                  />
                </div>
                <div className="col-span-2">
                  <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Dodano</label>
                  <Input type="date" value={r.added_date} onChange={e => update(i, { added_date: e.target.value })} className="mt-1" />
                </div>
              </div>
              <button
                onClick={() => remove(i)}
                className="mt-5 h-9 w-9 rounded-lg text-white/40 hover:text-accent-red hover:bg-accent-red/10 border border-white/[0.06] hover:border-accent-red/30 flex items-center justify-center transition-colors"
                title="Usuń"
              >
                <Icon name="trash" size={14} />
              </button>
            </div>

            <div>
              <label className="text-[10.5px] uppercase tracking-[0.14em] text-white/40">Notatki</label>
              <Textarea
                value={r.notes || ''}
                onChange={e => update(i, { notes: e.target.value })}
                placeholder="Dlaczego śledzisz tę spółkę?…"
                className="mt-1"
                rows={2}
              />
            </div>
          </WMot.div>
        ))}

        <button
          onClick={add}
          className="group flex items-center justify-center gap-2 h-12 rounded-xl border border-dashed border-white/[0.12] hover:border-accent-violet/40 text-white/55 hover:text-white text-[13px] transition-colors"
        >
          <Icon name="plus" size={14} />
          Dodaj ticker
        </button>
      </div>
    </Dialog>
  );
}

function WatchlistTab() {
  const [status, setStatus] = React.useState({ as_of: new Date().toISOString(), items: [], missing_data: [] });
  const [loading, setLoading] = React.useState(true);
  const [editOpen, setEditOpen] = React.useState(false);
  const toast = useToast();

  const loadStatus = React.useCallback(async () => {
    try {
      const s = await window.API.getWatchlistStatus();
      setStatus(s);
    } catch (err) {
      console.error(err);
      toast.error('Nie można wczytać watchlisty', err.detail || err.message);
    } finally {
      setLoading(false);
    }
  }, [toast]);

  React.useEffect(() => { loadStatus(); }, [loadStatus]);

  const onSave = async (rows) => {
    const payload = {
      items: rows.map(r => ({
        ticker: r.ticker.trim(),
        name: r.name || null,
        added_date: r.added_date,
        target_buy_price: r.target_buy_price == null || r.target_buy_price === '' ? null : Number(r.target_buy_price),
        notes: r.notes || '',
        keywords: Array.isArray(r.keywords) ? r.keywords : [],
      })).filter(r => r.ticker),  // skip empty rows
    };
    try {
      await window.API.putWatchlist(payload);
      await loadStatus();
      setEditOpen(false);
      toast.success('Watchlist zapisana', `${payload.items.length} pozycji.`);
    } catch (err) {
      toast.error('Zapis nie powiódł się', err.detail || err.message);
    }
  };

  const alertsCount = status.items.filter(it => it.alert).length;

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader
        eyebrow="Śledzone"
        title="Watchlist"
        description="Tickery które obserwujesz, ale jeszcze nie posiadasz. Ustaw cenę docelową, aby dostać alert gdy spadnie."
        right={
          <div className="flex items-center gap-2">
            <Button variant="outline" icon="edit" onClick={() => setEditOpen(true)}>
              Edytuj watchlist
            </Button>
          </div>
        }
      />

      {/* Summary strip */}
      <div className="flex items-center gap-3 flex-wrap text-[12px]">
        <Badge variant="default" icon="eye">{status.items.length} pozycji</Badge>
        {alertsCount > 0 && (
          <Badge variant="on_track" icon="check" pulse>
            {alertsCount} osiągniętych cen docelowych
          </Badge>
        )}
        {status.missing_data.length > 0 && (
          <span className="text-white/40">
            Brak danych OHLCV dla: <span className="mono text-white/55">{status.missing_data.join(', ')}</span>
          </span>
        )}
        <span className="text-white/35 ml-auto">Stan na {fmtDateTime(status.as_of)}</span>
      </div>

      {loading ? (
        <Card><Skeleton className="h-[200px] w-full" /></Card>
      ) : status.items.length === 0 ? (
        <EmptyState
          emoji="👁️"
          title="Watchlist jest pusta"
          description={
            <>
              Dodaj tickery które chciałbyś obserwować ale jeszcze nie posiadasz.
              Po dodaniu uruchom <span className="text-white/80 font-medium">Odśwież dane rynkowe</span> w zakładce Portfel,
              aby pobrać OHLCV i widzieć aktualne ceny.
            </>
          }
          cta={<Button variant="primary" icon="plus" onClick={() => setEditOpen(true)}>Dodaj pierwszy ticker</Button>}
        />
      ) : (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-[10.5px] uppercase tracking-[0.14em] text-white/40 border-y border-white/[0.05]">
                  <th className="text-left  font-medium px-5 py-2.5">Ticker</th>
                  <th className="text-left  font-medium px-3 py-2.5">Nazwa</th>
                  <th className="text-right font-medium px-3 py-2.5">Dodano</th>
                  <th className="text-right font-medium px-3 py-2.5">Ostatnia</th>
                  <th className="text-right font-medium px-3 py-2.5">Cena docelowa</th>
                  <th className="text-right font-medium px-3 py-2.5">Δ do celu</th>
                  <th className="text-right font-medium px-3 py-2.5">Newsy 30d</th>
                  <th className="text-left  font-medium px-5 py-2.5">Notatki</th>
                </tr>
              </thead>
              <tbody>
                {status.items.map((it, i) => <WatchlistRow key={it.ticker} item={it} i={i} />)}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <EditWatchlistDialog
        open={editOpen}
        onClose={() => setEditOpen(false)}
        items={status.items}
        onSave={onSave}
      />
    </div>
  );
}

window.WatchlistTab = WatchlistTab;
