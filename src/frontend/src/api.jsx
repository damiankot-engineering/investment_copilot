/* API client for the Investment Copilot FastAPI backend.

   Exposes `window.API` with one method per endpoint. All methods return
   parsed JSON or throw an `ApiError` (with .status and .detail). The base
   URL defaults to the current origin (FastAPI serves the frontend), but
   you can override with `window.API_BASE = 'http://localhost:8000'` if
   you run the frontend on a different host (e.g. via `python -m http.server`). */

(function () {
  const BASE = (window.API_BASE || '').replace(/\/$/, '');

  // Active portfolio id (multi-portfolio). When set, it's threaded into every
  // request as `?portfolio=<id>`; the backend resolver maps it to a file.
  // `null`/'default' → the default portfolio (param omitted).
  let activePortfolio = null;
  function setActivePortfolio(id) {
    activePortfolio = id && id !== 'default' ? id : null;
  }
  // Append `?portfolio=` to a raw URL string (for the non-`request()` paths:
  // EventSource + downloadable report URLs).
  function withPortfolio(urlStr) {
    if (!activePortfolio) return urlStr;
    const u = new URL(urlStr, window.location.origin);
    if (!u.searchParams.has('portfolio')) u.searchParams.set('portfolio', activePortfolio);
    return u.toString();
  }

  class ApiError extends Error {
    constructor(message, status, detail) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
      this.detail = detail;
    }
  }

  async function request(method, path, { body, query } = {}) {
    const url = new URL(BASE + path, window.location.origin);
    if (query) {
      Object.entries(query).forEach(([k, v]) => {
        if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
      });
    }
    // Multi-portfolio: scope every call to the active portfolio. Endpoints
    // that don't declare a `portfolio` param simply ignore it.
    if (activePortfolio && !url.searchParams.has('portfolio')) {
      url.searchParams.set('portfolio', activePortfolio);
    }
    const init = { method, headers: { Accept: 'application/json' } };
    if (body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    let resp;
    try {
      resp = await fetch(url.toString(), init);
    } catch (err) {
      throw new ApiError('Network error: ' + err.message, 0, null);
    }
    const ct = resp.headers.get('content-type') || '';
    const isJson = ct.includes('application/json');
    const payload = isJson ? await resp.json().catch(() => null) : await resp.text();
    if (!resp.ok) {
      const detail = isJson && payload && payload.detail ? payload.detail : payload;
      throw new ApiError(
        `HTTP ${resp.status} ${resp.statusText}`,
        resp.status,
        detail,
      );
    }
    return payload;
  }

  const API = {
    ApiError,
    // Meta
    health:        () => request('GET', '/api/health'),
    strategies:    () => request('GET', '/api/strategies'),
    getConfig:     () => request('GET', '/api/config'),

    // Portfolio
    getPortfolio:        () => request('GET', '/api/portfolio'),
    putPortfolio:        (portfolio) => request('PUT', '/api/portfolio', { body: portfolio }),
    getPortfolioStatus:  () => request('GET', '/api/portfolio/status'),

    // Watchlist
    getWatchlist:        () => request('GET', '/api/watchlist'),
    putWatchlist:        (wl) => request('PUT', '/api/watchlist', { body: wl }),
    getWatchlistStatus:  () => request('GET', '/api/watchlist/status'),
    refreshWatchlist:    (newsDaysBack = 14) =>
      request('POST', '/api/watchlist/refresh', { query: { news_days_back: newsDaysBack } }),

    // Calendar
    getCalendar:         () => request('GET', '/api/calendar'),

    // Data
    updateData:    (newsDaysBack = 14) =>
      request('POST', '/api/data/update', { query: { news_days_back: newsDaysBack } }),
    streamUpdateData: ({ newsDaysBack = 14, onEvent } = {}) => {
      // EventSource only does GET; the streaming endpoint mirrors POST's
      // side effects but uses GET for SSE compatibility.
      const url = new URL(`${BASE}/api/data/update/stream`, window.location.origin);
      url.searchParams.set('news_days_back', String(newsDaysBack));
      if (activePortfolio) url.searchParams.set('portfolio', activePortfolio);
      const es = new EventSource(url.toString());
      let settled = false;
      const result = new Promise((resolve, reject) => {
        es.onmessage = (msg) => {
          let ev;
          try { ev = JSON.parse(msg.data); } catch (_) { return; }
          if (onEvent) onEvent(ev);
          if (ev.type === 'done') {
            settled = true;
            es.close();
            resolve(ev);
          } else if (ev.type === 'error') {
            settled = true;
            es.close();
            reject(new ApiError(ev.message || 'stream error', 0, ev.message));
          }
        };
        es.onerror = () => {
          if (settled) return;
          es.close();
          reject(new ApiError('SSE connection lost', 0, null));
        };
      });
      result.close = () => es.close();
      return result;
    },

    // Backtest
    runBacktest:   (strategy = 'ma_crossover', { startDate, endDate, includeBenchmark = true, benchmark } = {}) =>
      request('POST', '/api/backtest', {
        query: {
          strategy,
          start_date: startDate,
          end_date: endDate,
          include_benchmark: includeBenchmark,
          benchmark,
        },
      }),

    // Analysis
    runAnalysis:   ({ includeRisks = true, newsDaysBack = 14 } = {}) =>
      request('POST', '/api/analysis', {
        query: { include_risks: includeRisks, news_days_back: newsDaysBack },
      }),
    getCachedAnalysis: () => request('GET', '/api/analysis/cached'),

    // Reports
    listReports:    () => request('GET', '/api/reports'),
    getReport:      (name) => request('GET', `/api/reports/${encodeURIComponent(name)}`),
    downloadReportUrl: (name) => `${BASE}/api/reports/${encodeURIComponent(name)}/download`,
    generateReport: ({ strategy = 'ma_crossover', newsDaysBack = 14, filename } = {}) =>
      request('POST', '/api/reports', {
        body: { strategy, news_days_back: newsDaysBack, filename },
      }),
    deleteReport:   (name) => request('DELETE', `/api/reports/${encodeURIComponent(name)}`),

    // Monitoring (legacy bulk — endpoints stay alive but UI no longer calls them)
    runMonitoring:        ({ newsDaysBack = 30 } = {}) =>
      request('POST', '/api/monitoring', { body: { news_days_back: newsDaysBack } }),
    listMonitoringReports: () => request('GET', '/api/monitoring/reports'),
    monitoringReportUrl:   (name) =>
      `${BASE}/api/monitoring/reports/${encodeURIComponent(name)}`,
    deleteMonitoringReport: (name) =>
      request('DELETE', `/api/monitoring/reports/${encodeURIComponent(name)}`),

    // Per-company report (new monitoring architecture)
    getCompanyFactsheet: (ticker) =>
      request('GET', `/api/companies/${encodeURIComponent(ticker)}/factsheet`),
    getCachedCompanyReport: (ticker) =>
      request('GET', `/api/companies/${encodeURIComponent(ticker)}/report`),
    generateCompanyReport: (ticker) =>
      request('POST', `/api/companies/${encodeURIComponent(ticker)}/report`),
    companyReportHtmlUrl: (ticker) =>
      withPortfolio(`${BASE}/api/companies/${encodeURIComponent(ticker)}/report.html`),
    listUpcomingReports: () =>
      request('GET', '/api/companies/upcoming'),

    // Multi-portfolio registry
    setActivePortfolio: (id) => setActivePortfolio(id),
    getActivePortfolio: () => activePortfolio || 'default',
    listPortfolios:   () => request('GET', '/api/portfolios'),
    createPortfolio:  ({ id, name, baseCurrency = 'PLN' }) =>
      request('POST', '/api/portfolios', { body: { id, name, base_currency: baseCurrency } }),
    renamePortfolio:  (id, name) =>
      request('PATCH', `/api/portfolios/${encodeURIComponent(id)}`, { body: { name } }),
    duplicatePortfolio: (id, newId, name) =>
      request('POST', `/api/portfolios/${encodeURIComponent(id)}/duplicate`, {
        body: { new_id: newId, name },
      }),
    deletePortfolio:  (id) =>
      request('DELETE', `/api/portfolios/${encodeURIComponent(id)}`),
  };

  window.API = API;
})();
