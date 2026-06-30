"""Self-contained HTML dashboard (no build step, no external assets).

Vanilla JS polls the sibling ``overview`` endpoint and renders cards. Served by
the ``GET /api/v1/dashboard/ui`` route; the relative fetch resolves to
``/api/v1/dashboard/overview``.
"""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trading System — Dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#0f1217; color:#e6e9ef; }
  header { padding:16px 24px; border-bottom:1px solid #232a35; display:flex; align-items:center; gap:16px; }
  header h1 { font-size:18px; margin:0; font-weight:600; }
  #updated { color:#8a93a3; font-size:12px; margin-left:auto; }
  main { padding:24px; display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:16px; }
  .card { background:#161b22; border:1px solid #232a35; border-radius:10px; padding:16px; }
  .card h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:#8a93a3; margin:0 0 12px; }
  .kv { display:flex; justify-content:space-between; padding:4px 0; font-size:14px; }
  .kv span:last-child { font-variant-numeric:tabular-nums; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; }
  .chip { font-size:12px; padding:3px 10px; border-radius:999px; border:1px solid #2c3442; }
  .ok { color:#3fb950; border-color:#234430; } .bad { color:#f85149; border-color:#4b2526; }
  .sev-critical { color:#f85149; } .sev-warning { color:#d29922; } .sev-info { color:#58a6ff; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { text-align:left; padding:5px 6px; border-bottom:1px solid #232a35; }
  .alert { padding:6px 0; border-bottom:1px solid #232a35; font-size:13px; }
  .muted { color:#8a93a3; font-size:13px; }
  .level-yellow { color:#d29922; } .level-red,.level-black { color:#f85149; } .level-none { color:#3fb950; }
</style>
</head>
<body>
<header>
  <h1>Trading System — Dashboard</h1>
  <span id="updated">loading…</span>
</header>
<main id="root"><div class="muted">Loading overview…</div></main>
<script>
const fmtPct = v => (v==null? "—" : (v*100).toFixed(2)+"%");
const fmtNum = v => (v==null? "—" : Number(v).toLocaleString(undefined,{maximumFractionDigits:2}));
function kv(k,v){ return `<div class="kv"><span>${k}</span><span>${v}</span></div>`; }
function card(title, body){ return `<div class="card"><h2>${title}</h2>${body}</div>`; }

function render(d){
  const root = document.getElementById("root");
  const cards = [];

  const src = d.sources||{};
  cards.push(card("Services", `<div class="chips">` +
    Object.entries(src).map(([k,v])=>`<span class="chip ${v==='ok'?'ok':'bad'}">${k}: ${v}</span>`).join("") +
    `</div>`));

  const p = d.portfolio;
  cards.push(card("Portfolio (risk-mgmt)", p ?
    kv("Value", fmtNum(p.value)) + kv("Exposure", fmtPct(p.exposure_pct)) +
    kv("Drawdown", fmtPct(p.drawdown_pct)) + kv("Daily loss", fmtPct(p.daily_loss_pct)) +
    kv("Regime", p.regime ?? "—") : `<div class="muted">unavailable</div>`));

  const cb = d.circuit_breaker;
  const lvl = cb ? (cb.level ?? "none") : "—";
  cards.push(card("Circuit breaker", cb ?
    `<div class="kv"><span>Level</span><span class="level-${(cb.level||'none')}">${(cb.level||'none').toUpperCase()}</span></div>` +
    kv("Tripped", cb.tripped===true ? "yes" : "no") : `<div class="muted">unavailable</div>`));

  const ex = d.execution;
  cards.push(card("Broker (execution)", ex ?
    kv("Cash", fmtNum(ex.cash)) + kv("Equity", fmtNum(ex.equity)) +
    kv("Exposure", fmtPct(ex.exposure_pct)) : `<div class="muted">unavailable</div>`));

  const pos = d.positions||{};
  const posRows = Object.entries(pos);
  cards.push(card("Positions", posRows.length ?
    `<table><tr><th>Symbol</th><th>Qty</th><th>Last</th></tr>` +
    posRows.map(([s,v])=>`<tr><td>${s}</td><td>${fmtNum(v.quantity)}</td><td>${fmtNum(v.last_price)}</td></tr>`).join("") +
    `</table>` : `<div class="muted">no open positions</div>`));

  const al = d.recent_alerts||[];
  cards.push(card("Recent alerts", al.length ?
    al.slice(-10).reverse().map(a=>`<div class="alert"><span class="sev-${a.severity}">[${a.severity}]</span> ${a.title}</div>`).join("")
    : `<div class="muted">no alerts</div>`));

  const m = d.models||[];
  cards.push(card("ML models", m.length ? m.map(x=>`<div class="kv"><span>${x}</span><span></span></div>`).join("")
    : `<div class="muted">no registered models</div>`));

  root.innerHTML = cards.join("");
  document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString();
}

async function tick(){
  try { const r = await fetch("overview"); render(await r.json()); }
  catch(e){ document.getElementById("updated").textContent = "fetch error"; }
}
tick(); setInterval(tick, 5000);
</script>
</body>
</html>
"""
