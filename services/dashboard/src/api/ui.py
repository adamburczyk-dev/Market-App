"""Self-contained HTML dashboard — six sections, no build step, no external assets.

Charts are inline SVG built from the section JSON. That is a deliberate choice
over a charting library: the page is served from a container with no CDN access
and no bundler in the toolchain, so a library would have to be vendored and
version-pinned by hand for four chart types. Paths are computed in a dozen lines
of vanilla JS instead.

Sections mirror Plan_Rozwoju Week 21: portfolio, risk, strategy attribution,
backtest, ML, system health. Each polls its own endpoint and fails
independently — an unreachable upstream greys out one card, never the page.

Every value that can be absent renders as "—" with a reason. A dashboard that
draws a VaR from twelve observations is worse than one that says it cannot.
"""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trading System — Dashboard</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#0f1217; color:#e6e9ef; }
  header { padding:14px 24px; border-bottom:1px solid #232a35; display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  header h1 { font-size:17px; margin:0; font-weight:600; }
  nav { display:flex; gap:4px; flex-wrap:wrap; }
  nav button { background:none; border:1px solid #232a35; color:#8a93a3; padding:5px 12px; border-radius:6px; font-size:12px; cursor:pointer; }
  nav button.active { color:#e6e9ef; border-color:#3d4756; background:#1b2129; }
  #updated { color:#8a93a3; font-size:12px; margin-left:auto; }
  main { padding:20px 24px 40px; }
  section { display:none; }
  section.active { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:16px; align-items:start; }
  .card { background:#161b22; border:1px solid #232a35; border-radius:10px; padding:16px; min-width:0; }
  .card.wide { grid-column:1/-1; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:#8a93a3; margin:0 0 12px; }
  .kv { display:flex; justify-content:space-between; gap:12px; padding:4px 0; font-size:14px; }
  .kv span:last-child { font-variant-numeric:tabular-nums; }
  .big { font-size:26px; font-variant-numeric:tabular-nums; margin:2px 0 10px; }
  .up { color:#3fb950; } .down { color:#f85149; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; }
  .chip { font-size:12px; padding:3px 10px; border-radius:999px; border:1px solid #2c3442; }
  .ok { color:#3fb950; border-color:#234430; } .bad { color:#f85149; border-color:#4b2526; }
  .warn { color:#d29922; border-color:#4a3a15; }
  .sev-critical { color:#f85149; } .sev-warning { color:#d29922; } .sev-info { color:#58a6ff; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { text-align:left; padding:5px 6px; border-bottom:1px solid #232a35; white-space:nowrap; }
  th { color:#8a93a3; font-weight:500; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .scroll { overflow-x:auto; }
  .alert { padding:6px 0; border-bottom:1px solid #232a35; font-size:13px; }
  .muted { color:#8a93a3; font-size:13px; }
  .level-yellow { color:#d29922; } .level-red,.level-black { color:#f85149; } .level-none { color:#3fb950; }
  svg { display:block; width:100%; height:auto; }
  .grid-line { stroke:#232a35; stroke-width:1; }
  form.inline { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
  form.inline input, form.inline select { background:#0f1217; border:1px solid #2c3442; color:#e6e9ef; padding:5px 8px; border-radius:6px; font-size:13px; }
  form.inline button { background:#1f6feb; border:none; color:#fff; padding:5px 14px; border-radius:6px; font-size:13px; cursor:pointer; }
  form.inline button:disabled { opacity:.5; cursor:default; }
</style>
</head>
<body>
<header>
  <h1>Trading System</h1>
  <nav id="tabs"></nav>
  <span id="updated">loading…</span>
</header>
<main>
  <section id="tab-portfolio"></section>
  <section id="tab-risk"></section>
  <section id="tab-strategy"></section>
  <section id="tab-backtest"></section>
  <section id="tab-ml"></section>
  <section id="tab-health"></section>
</main>
<script>
const TABS = [
  ["portfolio","Portfolio"], ["risk","Risk"], ["strategy","Strategies"],
  ["backtest","Backtest"], ["ml","ML"], ["health","Health"],
];
let active = "portfolio";

// --- formatting -----------------------------------------------------------
const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num = (v,d=2) => v==null||Number.isNaN(v) ? "—" : Number(v).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d});
const pct = (v,d=2) => v==null||Number.isNaN(v) ? "—" : (v*100).toFixed(d)+"%";
const kv = (k,v,cls="") => `<div class="kv"><span>${esc(k)}</span><span class="${cls}">${v}</span></div>`;
const card = (title,body,wide=false) => `<div class="card${wide?" wide":""}"><h2>${esc(title)}</h2>${body}</div>`;
const none = why => `<div class="muted">${esc(why)}</div>`;

// --- charts: inline SVG, no library --------------------------------------
function linePath(values, w, h, pad){
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = (hi - lo) || 1;
  const dx = values.length > 1 ? (w - 2*pad) / (values.length - 1) : 0;
  return values.map((v,i) => {
    const x = pad + i*dx;
    const y = h - pad - ((v - lo) / span) * (h - 2*pad);
    return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join("");
}

function lineChart(values, {height=170, colour="#3fb950", fill=true, baseline=null} = {}){
  if (!values || values.length < 2) return none("not enough points to draw a line");
  const w = 640, pad = 8;
  const path = linePath(values, w, height, pad);
  // Colour by direction of the whole path — the one thing a glance should read.
  const rising = values[values.length-1] >= values[0];
  const stroke = colour === "auto" ? (rising ? "#3fb950" : "#f85149") : colour;
  const area = fill
    ? `<path d="${path}L${w-pad},${height-pad}L${pad},${height-pad}Z" fill="${stroke}" opacity="0.10"/>`
    : "";
  const base = baseline!=null
    ? `<line class="grid-line" x1="${pad}" x2="${w-pad}" y1="${height-pad}" y2="${height-pad}"/>` : "";
  return `<svg viewBox="0 0 ${w} ${height}" preserveAspectRatio="none" role="img">
    ${base}${area}<path d="${path}" fill="none" stroke="${stroke}" stroke-width="2"
      stroke-linejoin="round" stroke-linecap="round"/></svg>`;
}

function heatmap(symbols, matrix){
  if (!symbols.length) return none("no open positions to correlate");
  const cell = 34, labelW = 66, w = labelW + symbols.length*cell, h = 22 + symbols.length*cell;
  let out = `<svg viewBox="0 0 ${w} ${h}" role="img">`;
  symbols.forEach((s,i) => {
    out += `<text x="${labelW + i*cell + cell/2}" y="14" font-size="9" fill="#8a93a3"
      text-anchor="middle">${esc(s.slice(0,5))}</text>`;
    out += `<text x="${labelW-6}" y="${22 + i*cell + cell/2 + 3}" font-size="9" fill="#8a93a3"
      text-anchor="end">${esc(s.slice(0,7))}</text>`;
  });
  for (let i=0;i<symbols.length;i++){
    for (let j=0;j<symbols.length;j++){
      const v = matrix[i][j];
      // null is NOT zero: an unmeasurable pair is hatched, not painted neutral.
      const f = v==null ? "#20262f" : (v>=0 ? `rgba(63,185,80,${Math.abs(v).toFixed(2)})`
                                            : `rgba(248,81,73,${Math.abs(v).toFixed(2)})`);
      out += `<rect x="${labelW + j*cell}" y="${22 + i*cell}" width="${cell-2}" height="${cell-2}"
        fill="${f}" rx="3"/>`;
      out += `<text x="${labelW + j*cell + (cell-2)/2}" y="${22 + i*cell + cell/2 + 3}"
        font-size="9" fill="${v==null?"#5b6472":"#e6e9ef"}" text-anchor="middle">${v==null?"·":v.toFixed(2)}</text>`;
    }
  }
  return out + "</svg>";
}

function bars(rows){
  if (!rows.length) return none("nothing to compare");
  const max = Math.max(...rows.map(r => Math.abs(r.value ?? 0)), 1e-9);
  return `<table>` + rows.map(r => {
    const w = r.value==null ? 0 : Math.abs(r.value)/max*100;
    return `<tr><td>${esc(r.label)}</td>
      <td style="width:60%"><div style="background:#1f6feb;opacity:.75;height:9px;border-radius:3px;width:${w.toFixed(1)}%"></div></td>
      <td class="num">${r.value==null ? "—" : pct(r.value)}</td></tr>`;
  }).join("") + `</table>`;
}

// --- sections -------------------------------------------------------------
function renderPortfolio(d){
  const cards = [];
  const pnlCls = (d.pnl_abs ?? 0) >= 0 ? "up" : "down";
  cards.push(card("Equity curve",
    (d.available
      ? `<div class="big ${pnlCls}">${num(d.curve[d.curve.length-1]?.equity)}</div>` +
        lineChart(d.curve.map(p => p.equity), {colour:"auto"}) +
        `<div class="muted" style="margin-top:8px">${d.sessions} sessions · ${esc(d.labels[0]??"")} → ${esc(d.labels[d.labels.length-1]??"")}</div>`
      : none("no equity history yet — the broker records one point per session")), true));

  cards.push(card("P&L",
    kv("Absolute", num(d.pnl_abs), pnlCls) +
    kv("Percent", pct(d.pnl_pct), pnlCls) +
    kv("Sharpe (realized)", d.sharpe==null ? "— (needs variance)" : num(d.sharpe)) +
    kv("Sessions", d.sessions)));

  const b = d.broker;
  cards.push(card("Broker", b
    ? kv("Cash", num(b.cash)) + kv("Equity", num(b.equity)) +
      kv("Exposure", pct(b.exposure_pct)) + kv("Drawdown", pct(b.drawdown_pct)) +
      kv("Daily loss", pct(b.daily_loss_pct))
    : none("execution unavailable")));

  const pos = Object.entries(d.positions||{});
  cards.push(card("Positions", pos.length
    ? `<div class="scroll"><table><tr><th>Symbol</th><th class="num">Qty</th><th class="num">Last</th><th class="num">Stop</th><th class="num">Target</th></tr>` +
      pos.map(([s,v]) => `<tr><td>${esc(s)}</td><td class="num">${num(v.quantity)}</td>
        <td class="num">${num(v.last_price)}</td><td class="num">${num(v.stop_loss)}</td>
        <td class="num">${num(v.take_profit)}</td></tr>`).join("") + `</table></div>`
    : none("no open positions")));
  return cards.join("");
}

function renderRisk(d){
  const cards = [];
  const enough = d.samples >= 20;
  cards.push(card("Value at Risk (95%, historical)",
    kv("VaR", d.var_95==null ? `— (${d.samples} of 20 sessions)` : pct(d.var_95), "down") +
    kv("CVaR (expected shortfall)", d.cvar_95==null ? "—" : pct(d.cvar_95), "down") +
    kv("Observations", d.samples) +
    (enough ? "" : `<div class="muted" style="margin-top:8px">An empirical quantile from
      fewer than 20 returns rests on one point — reported as unavailable rather than as a number.</div>`)));

  cards.push(card("Drawdown",
    kv("Maximum", d.max_drawdown==null ? "—" : pct(d.max_drawdown), "down") +
    (d.drawdown_curve && d.drawdown_curve.length > 1
      ? lineChart(d.drawdown_curve.map(v => -v), {colour:"#f85149", height:130})
      : none("no equity history yet"))));

  const c = d.correlation || {symbols:[], matrix:[], coverage:1, samples:0};
  const held = (d.held_symbols||[]).length, withHist = (d.correlated_symbols||[]).length;
  const gap = held > withHist
    ? `<div class="muted">${held - withHist} of ${held} held names have no price history
       here — market-data did not answer for them, which is not the same as their
       correlation being zero.</div>` : "";
  cards.push(card("Correlation of held names",
    heatmap(c.symbols||[], c.matrix||[]) + gap +
    kv("Average pairwise", d.avg_correlation==null ? "—" : num(d.avg_correlation)) +
    kv("Effective bets (1/ρ)", d.avg_correlation && d.avg_correlation > 0
      ? num(1/d.avg_correlation, 1) : "—") +
    kv("Grid coverage", pct(c.coverage ?? 0, 0)) +
    (c.coverage < 1
      ? `<div class="muted">Blank cells are pairs with too little overlapping history to
         measure — not pairs measured as uncorrelated.</div>` : ""), true));

  const cb = d.circuit_breaker;
  cards.push(card("Circuit breaker", cb
    ? `<div class="kv"><span>Level</span><span class="level-${esc(cb.level||"none")}">${esc((cb.level||"none").toUpperCase())}</span></div>` +
      kv("Tripped", cb.tripped===true ? "yes" : "no") +
      kv("Latched", cb.latched===true ? "yes — needs a human reset" : "no")
    : none("risk-mgmt unavailable")));
  return cards.join("");
}

function renderStrategy(d){
  if (!d.available) return card("Strategies", none("strategy service unavailable"), true);
  const rows = d.strategies.map(s => ({label: s.name, value: s.weight}));
  return card("Decision weight per strategy",
      bars(rows) +
      `<div class="muted" style="margin-top:8px">Weights are learned from realized outcomes
       (adaptive weighting). "—" means the source has no outcomes recorded yet — not a
       weight of zero.</div>`, true) +
    card("Status", `<div class="scroll"><table>
      <tr><th>Strategy</th><th>Status</th><th>Reads ranks</th><th class="num">Weight</th></tr>` +
      d.strategies.map(s => {
        const cls = s.status==="active" ? "ok" : (s.status==="probation" ? "warn" : "bad");
        return `<tr><td>${esc(s.name)}</td>
          <td><span class="chip ${cls}">${esc(s.status)}</span></td>
          <td>${(s.required_ranks||[]).length ? esc((s.required_ranks||[]).join(", ")) : "—"}</td>
          <td class="num">${s.weight==null ? "—" : pct(s.weight)}</td></tr>`;
      }).join("") + `</table></div>`, true) +
    card("Other decision sources", Object.keys(d.other_sources||{}).length
      ? Object.entries(d.other_sources).map(([k,v]) => kv(k, pct(v))).join("")
      : none("none"));
}

function renderMl(d){
  const runs = d.runs || {};
  const cards = [];
  cards.push(card("Registered models", (d.models||[]).length
    ? (d.models||[]).map(m => kv(typeof m==="string" ? m : JSON.stringify(m), "")).join("")
    : none(d.available ? "no registered models" : "ml-pipeline unavailable")));
  const s = d.serving;
  cards.push(card("Serving", s
    ? Object.entries(s).map(([k,v]) => kv(k, typeof v==="object" ? JSON.stringify(v) : esc(String(v)))).join("")
    : none("unavailable")));
  // /runs is an INDEX: [{operation, completed_at}] — reading it as a map put
  // array positions ("0", "1") in the Operation column.
  const runList = Array.isArray(runs) ? runs : Object.values(runs||{});
  cards.push(card("Last completed runs", runList.length
    ? `<div class="scroll"><table><tr><th>Operation</th><th>Completed</th></tr>` +
      runList.map(r =>
        `<tr><td>${esc(r.operation ?? "?")}</td><td>${esc(r.completed_at ?? "—")}</td></tr>`).join("") +
      `</table></div>`
    : none("no long-running operation has completed in this container")), true);
  cards.push(card("Feature importance", renderImportance(d.importance||{}), true));
  return cards.join("");
}

function importanceRows(rows, bar){
  return `<div class="scroll"><table>
    <tr><th>Feature</th><th>Δ IC</th><th class="num">t</th><th class="num">Δ AUC</th><th>Note</th></tr>` +
    rows.map(r => {
      const strong = Math.abs(r.t) > bar;
      // Sign matters: permuting a feature the model leans on the WRONG way
      // improves the ranking, and a chart of |Δ| would hide exactly that.
      const w = Math.min(100, Math.abs(r.ic_drop)/Math.max(...rows.map(x=>Math.abs(x.ic_drop)),1e-9)*100);
      const colour = r.ic_drop >= 0 ? "#1f6feb" : "#f85149";
      return `<tr><td>${esc(r.feature)}</td>
        <td style="width:45%"><div style="background:${colour};opacity:${strong?".85":".3"};
          height:9px;border-radius:3px;width:${w.toFixed(1)}%"></div></td>
        <td class="num">${num(r.t,2)}</td><td class="num">${num(r.auc_drop,4)}</td>
        <td>${r.redundant ? `<span class="chip warn">~${esc(r.most_correlated_with||"")}</span>` : ""}</td></tr>`;
    }).join("") + `</table></div>`;
}

function renderImportance(imp){
  if (!imp.table) return none(imp.reason || "not measured");
  const t = imp.table, bar = t.tstat_bar ?? 2;
  const noise = t.noise_control;
  return `<div class="muted">${esc(imp.source||"")}${imp.measured_at ? " · "+esc(imp.measured_at) : ""}</div>` +
    kv("Baseline IC (holdout)", num(t.base_ic,5)) +
    kv("Sessions paired", t.n_sessions) +
    kv("Significance bar (|t|, corrected)", num(bar,2)) +
    (noise ? kv("Planted noise column scored", num(noise.t,2)+" t") : "") +
    `<h4>By feature</h4>` + importanceRows(t.features||[], bar) +
    ((t.groups||[]).length ? `<h4>By family</h4>` + importanceRows(t.groups, bar) : "") +
    `<div class="muted" style="margin-top:8px">${esc(t.verdict||"")}</div>`;
}

function renderHealth(d){
  const svc = d.services || {};
  const rows = Object.entries(svc);
  const chips = rows.map(([k,v]) => {
    const cls = v.status==="up" ? "ok" : (v.status==="degraded" ? "warn" : "bad");
    return `<span class="chip ${cls}">${esc(k)}</span>`;
  }).join("");
  return card("Services", `<div class="big">${d.up} / ${d.total} up</div><div class="chips">${chips}</div>`, true) +
    card("Latency", rows.length
      ? `<div class="scroll"><table><tr><th>Service</th><th>Status</th><th class="num">ms</th></tr>` +
        rows.sort((a,b)=>(b[1].latency_ms||0)-(a[1].latency_ms||0)).map(([k,v]) =>
          `<tr><td>${esc(k)}</td><td>${esc(v.status)}</td><td class="num">${num(v.latency_ms,1)}</td></tr>`
        ).join("") + `</table></div>` +
        kv("Slowest", d.slowest_ms==null ? "—" : num(d.slowest_ms,1)+" ms")
      : none("no services configured"), true);
}

function backtestForm(){
  return `<form class="inline" id="bt-form">
      <input id="bt-strategy" value="sma_ema_crossover" size="22" aria-label="strategy"/>
      <input id="bt-symbol" value="AAPL" size="8" aria-label="symbol"/>
      <input id="bt-limit" value="500" size="6" type="number" aria-label="bars"/>
      <button type="submit" id="bt-run">Run backtest</button>
    </form><div id="bt-result"><div class="muted">Backtests are run on request — they cost real
    work, so nothing here refreshes on a timer.</div></div>`;
}

function renderBacktestResult(status, body){
  if (status === 404) return `<div class="muted">Unknown strategy. ${esc(body.detail||"")}</div>`;
  if (status === 422) return `<div class="muted">${esc(body.detail||"This strategy cannot be
    backtested one symbol at a time.")}</div>`;
  if (status !== 200) return `<div class="muted">Backtest failed (${status}). ${esc(body.detail||"")}</div>`;
  const curve = body.equity_curve || [];
  const up = (body.total_return ?? 0) >= 0;
  return `<div class="big ${up?"up":"down"}">${pct(body.total_return)}</div>` +
    lineChart(curve, {colour:"auto"}) +
    kv("Sharpe", num(body.sharpe_ratio)) +
    kv("Max drawdown", pct(body.max_drawdown)) +
    kv("Trades", body.n_trades) +
    kv("Bars scored", body.n_bars);
}

// --- wiring ---------------------------------------------------------------
const RENDER = {
  portfolio: ["sections/portfolio", renderPortfolio],
  risk:      ["sections/risk", renderRisk],
  strategy:  ["sections/strategy", renderStrategy],
  ml:        ["sections/ml", renderMl],
  health:    ["sections/health", renderHealth],
};

async function loadSection(name){
  const el = document.getElementById("tab-"+name);
  if (name === "backtest"){
    if (!el.dataset.ready){ el.innerHTML = card("Run a backtest", backtestForm(), true); el.dataset.ready="1"; bindBacktest(); }
    return;
  }
  const [path, fn] = RENDER[name];
  try {
    const resp = await fetch(path);
    el.innerHTML = fn(await resp.json());
  } catch (e) {
    el.innerHTML = card(name, `<div class="muted">could not load: ${esc(e.message)}</div>`, true);
  }
}

function bindBacktest(){
  const form = document.getElementById("bt-form");
  form.addEventListener("submit", async ev => {
    ev.preventDefault();
    const btn = document.getElementById("bt-run");
    const out = document.getElementById("bt-result");
    btn.disabled = true; out.innerHTML = `<div class="muted">running…</div>`;
    const q = new URLSearchParams({
      strategy: document.getElementById("bt-strategy").value.trim(),
      symbol: document.getElementById("bt-symbol").value.trim(),
      limit: document.getElementById("bt-limit").value.trim() || "500",
    });
    try {
      const resp = await fetch("sections/backtest?"+q, {method:"POST"});
      out.innerHTML = renderBacktestResult(resp.status, await resp.json());
    } catch (e) {
      out.innerHTML = `<div class="muted">request failed: ${esc(e.message)}</div>`;
    } finally { btn.disabled = false; }
  });
}

function selectTab(name){
  active = name;
  document.querySelectorAll("section").forEach(s => s.classList.remove("active"));
  document.getElementById("tab-"+name).classList.add("active");
  document.querySelectorAll("nav button").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  loadSection(name);
}

document.getElementById("tabs").innerHTML = TABS.map(([id,label]) =>
  `<button data-tab="${id}">${label}</button>`).join("");
document.querySelectorAll("nav button").forEach(b =>
  b.addEventListener("click", () => selectTab(b.dataset.tab)));

async function tick(){
  // Only the visible section is refreshed: the health probe and the
  // correlation grid cost real upstream work, and polling all six every few
  // seconds would make the dashboard the heaviest client in the system.
  await loadSection(active);
  document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString();
}
selectTab("portfolio");
setInterval(tick, 10000);
</script>
</body>
</html>
"""
