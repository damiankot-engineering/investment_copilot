/* Portfolio switcher — dropdown in the TopBar + a manage (CRUD) dialog.
   Backed by /api/portfolios; switching threads ?portfolio=<id> into every call
   via window.API.setActivePortfolio (see api.jsx). */

function ManagePortfoliosDialog({ open, onClose, items, onChanged, activeId, onSwitch }) {
  const [newId, setNewId] = React.useState('');
  const [newName, setNewName] = React.useState('');
  const [newAccount, setNewAccount] = React.useState('standard');
  const [busy, setBusy] = React.useState(false);
  const toast = useToast();

  React.useEffect(() => { if (open) { setNewId(''); setNewName(''); setNewAccount('standard'); } }, [open]);

  const guard = async (fn, okMsg) => {
    setBusy(true);
    try {
      await fn();
      if (okMsg) toast.success(okMsg);
      await onChanged();
    } catch (err) {
      toast.error('Operacja nie powiodła się', err.detail || err.message);
    } finally {
      setBusy(false);
    }
  };

  const onCreate = () =>
    guard(async () => {
      if (!newId.trim()) throw new window.API.ApiError('Podaj id portfela', 0, 'Podaj id portfela');
      await window.API.createPortfolio({
        id: newId.trim(), name: newName.trim() || null, accountType: newAccount,
      });
      setNewId(''); setNewName(''); setNewAccount('standard');
    }, 'Portfel utworzony');

  const onRename = (it) => {
    const name = window.prompt(`Nowa nazwa dla „${it.name || it.id}":`, it.name || '');
    if (name === null) return;
    guard(() => window.API.renamePortfolio(it.id, name.trim() || null), 'Nazwa zmieniona');
  };

  const onDuplicate = (it) => {
    const newId = window.prompt(`Nowe id dla kopii „${it.name || it.id}":`, `${it.id}-kopia`);
    if (!newId) return;
    guard(() => window.API.duplicatePortfolio(it.id, newId.trim()), 'Portfel zduplikowany');
  };

  const onDelete = (it) => {
    if (!window.confirm(`Usunąć portfel „${it.name || it.id}"?\n\nPlik trafi do .trash (można odzyskać ręcznie).`)) return;
    guard(async () => {
      await window.API.deletePortfolio(it.id);
      if (activeId === it.id) onSwitch('default'); // leave a deleted active portfolio
    }, 'Portfel usunięty');
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Zarządzaj portfelami"
      subtitle="Twórz, zmieniaj nazwę, duplikuj i usuwaj portfele"
      width="max-w-2xl"
      footer={<Button variant="ghost" onClick={onClose}>Zamknij</Button>}
    >
      <div className="flex flex-col gap-4">
        {/* Create row */}
        <div className="glass-soft rounded-xl p-4">
          <div className="text-[10.5px] uppercase tracking-[0.14em] text-white/40 mb-2">Nowy portfel</div>
          <div className="grid grid-cols-12 gap-2 items-end">
            <div className="col-span-3">
              <label className="text-[10.5px] text-white/40">id (a-z, 0-9, -, _)</label>
              <Input value={newId} onChange={e => setNewId(e.target.value)} placeholder="np. ike" className="mt-1" />
            </div>
            <div className="col-span-4">
              <label className="text-[10.5px] text-white/40">Nazwa (opcjonalnie)</label>
              <Input value={newName} onChange={e => setNewName(e.target.value)} placeholder="IKE" className="mt-1" />
            </div>
            <div className="col-span-3">
              <label className="text-[10.5px] text-white/40">Typ konta</label>
              <select
                value={newAccount}
                onChange={e => setNewAccount(e.target.value)}
                className="mt-1 w-full h-9 px-2 rounded-lg bg-white/[0.03] border border-white/[0.08] hover:border-white/[0.15] text-[13px] text-white outline-none focus:border-accent-violet/50"
              >
                <option value="standard" className="bg-ink-900">Standard (PIT 19%)</option>
                <option value="ike" className="bg-ink-900">IKE (bez podatku)</option>
                <option value="ikze" className="bg-ink-900">IKZE (bez podatku)</option>
              </select>
            </div>
            <div className="col-span-2">
              <Button variant="primary" icon="plus" loading={busy} onClick={onCreate} className="w-full">Utwórz</Button>
            </div>
          </div>
        </div>

        {/* List */}
        <div className="flex flex-col gap-2">
          {items.map((it) => (
            <div key={it.id} className="glass rounded-xl px-4 py-3 flex items-center gap-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-[13px] font-medium text-white truncate">{it.name || it.id}</span>
                  <span className="mono text-[11px] text-white/40">{it.id}</span>
                  {it.is_default && <Badge variant="default">default</Badge>}
                  {it.account_type && it.account_type !== 'standard' && (
                    <Badge variant="on_track">{it.account_type.toUpperCase()}</Badge>
                  )}
                  {activeId === it.id && <Badge variant="violet" icon="check">aktywny</Badge>}
                </div>
                <div className="text-[11px] text-white/40 mt-0.5">{it.n_holdings} pozycji</div>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                <Button variant="ghost" size="sm" icon="edit" onClick={() => onRename(it)}>Nazwa</Button>
                <Button variant="ghost" size="sm" icon="fileText" onClick={() => onDuplicate(it)}>Duplikuj</Button>
                <button
                  onClick={() => onDelete(it)}
                  disabled={it.is_default}
                  title={it.is_default ? 'Domyślnego nie można usunąć' : 'Usuń portfel'}
                  className={`h-7 w-7 rounded-md border flex items-center justify-center transition-colors ${
                    it.is_default
                      ? 'text-white/20 border-white/[0.05] cursor-not-allowed'
                      : 'text-white/45 hover:text-accent-red hover:bg-accent-red/10 border-white/[0.06] hover:border-accent-red/25'
                  }`}
                >
                  <Icon name="trash" size={13} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Dialog>
  );
}

function PortfolioSwitcher({ activeId, onSwitch }) {
  const [open, setOpen] = React.useState(false);
  const [manageOpen, setManageOpen] = React.useState(false);
  const [items, setItems] = React.useState([]);
  const ref = React.useRef(null);

  const refresh = React.useCallback(async () => {
    try {
      setItems(await window.API.listPortfolios());
    } catch (err) {
      console.error('listPortfolios failed:', err);
    }
  }, []);

  React.useEffect(() => { refresh(); }, [refresh]);

  // Close dropdown on outside click
  React.useEffect(() => {
    if (!open) return;
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const active = items.find(p => p.id === activeId) || { id: activeId, name: null };

  const pick = (id) => { setOpen(false); if (id !== activeId) onSwitch(id); };

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-2 h-9 pl-2.5 pr-2 rounded-lg bg-white/[0.03] border border-white/[0.08] hover:border-white/[0.16] text-[13px] text-white transition-colors"
        title="Przełącz portfel"
      >
        <Icon name="wallet" size={14} className="text-accent-violet" />
        <span className="font-medium truncate max-w-[160px]">{active.name || active.id}</span>
        <Icon name="chevronDown" size={13} className="text-white/40" />
      </button>

      {open && (
        <div className="absolute left-0 top-[calc(100%+6px)] z-50 w-64 glass rounded-xl border border-white/[0.08] p-1.5 shadow-xl">
          <div className="px-2 py-1.5 text-[10px] uppercase tracking-[0.14em] text-white/35">Portfele</div>
          <div className="flex flex-col max-h-[300px] overflow-y-auto">
            {items.map((p) => (
              <button
                key={p.id}
                onClick={() => pick(p.id)}
                className={`flex items-center gap-2 px-2 py-2 rounded-lg text-left text-[13px] transition-colors hover:bg-white/[0.05] ${
                  p.id === activeId ? 'text-white' : 'text-white/70'
                }`}
              >
                <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${p.id === activeId ? 'bg-accent-violet' : 'bg-white/20'}`} />
                <span className="flex-1 min-w-0 truncate">{p.name || p.id}</span>
                <span className="mono text-[10.5px] text-white/35">{p.n_holdings}</span>
              </button>
            ))}
          </div>
          <div className="border-t border-white/[0.06] mt-1 pt-1">
            <button
              onClick={() => { setOpen(false); setManageOpen(true); }}
              className="w-full flex items-center gap-2 px-2 py-2 rounded-lg text-left text-[12.5px] text-white/70 hover:text-white hover:bg-white/[0.05] transition-colors"
            >
              <Icon name="settings" size={13} className="text-white/40" />
              Zarządzaj portfelami…
            </button>
          </div>
        </div>
      )}

      <ManagePortfoliosDialog
        open={manageOpen}
        onClose={() => setManageOpen(false)}
        items={items}
        onChanged={refresh}
        activeId={activeId}
        onSwitch={onSwitch}
      />
    </div>
  );
}

window.PortfolioSwitcher = PortfolioSwitcher;
