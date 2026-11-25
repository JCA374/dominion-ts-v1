// Training page: cards, the main progress chart with an ETA, the goal table, a stage timeline, extra charts from other
// stats.csv columns, the run comparison table, and links to analysis reports. Config: site.json (status.config).
async function main() {
  const [status, stats] = await Promise.all([get('status.json'), get('stats.json')]);
  if (!status) { $('updated').textContent = 'No data yet.'; return; }
  const C = status.config || {}, G = C.goal || {};
  $('updated').textContent = `Updated ${status.updated}.`;
  const metric = G.metric || 'landed', target = G.target;
  goalTable(status, C, metric, target);
  const runs = status.runs || [];
  const main = runs.find((r) => r.alive) || runs[runs.length - 1];
  if (main) {
    const l = main.last || {};
    const card = (c) => { let v = l[c.key]; if (c.fmt === 'pct') v = pct(+v); if (c.fmt === 'M') v = fmtM(v); return [c.label, v ?? '–', c.sub || '']; };
    $('cards').innerHTML = [['Run', main.run, main.alive ? 'training now' : 'stopped'], ...(main.n_stages || l.stage ? [['Stage', l.stage || '–', `${main.passed.length} of ${main.n_stages || '?'} stage(s) passed`]] : []),
      ...(C.cards || []).map(card), ['Best check', main.best_goal ? pct(main.best_goal[metric]) : '–', 'best of many: flatters']]
      .map(([k, v, s]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`).join('');
    const rows = (stats || {})[main.run] || [];
    lineChart($('chart'), rows, C.chart || {}, target, $('chartnote'));
    $('chartnote').textContent = `Run ${main.run}. ` + $('chartnote').textContent;
    $('eta').innerHTML = eta(rows, C.chart || {}, target, main);
    extraCharts(rows, C);
  }
  timeline(status);
  runTable(runs, C, metric);
  const reps = status.reports || [];
  $('reports').innerHTML = reps.length ? reps.map((r) => `<li><a href="${esc(r.href)}">${esc(r.title)}</a> <span class="muted small">${esc(r.t)}</span></li>`).join('') : '<li class="muted">None yet.</li>';
}
function goalTable(status, C, metric, target) {
  const rows = [];
  for (const [k, r] of Object.entries(status.goals || {})) rows.push([r.label || k, r, r.note || 'goal check']);
  for (const [k, r] of Object.entries(status.baselines || {})) rows.push([(C.names || {})[k] || r.label || k, r, 'baseline']);
  for (const [k, r] of Object.entries(status.bakeoff || {})) rows.push([(C.names || {})[k] || r.label || k, r, 'bake-off']);
  rows.sort((a, b) => (b[1][metric] ?? 0) - (a[1][metric] ?? 0));
  const extra = (C.goal || {}).columns || [];
  $('goaltable').innerHTML = `<tr><th>Agent</th><th>${esc((C.goal || {}).label || metric)}</th><th>95% interval</th>${extra.map((c) => `<th>${esc(c[1])}</th>`).join('')}<th>n</th></tr>` +
    rows.map(([n, r, note]) => `<tr><td>${esc(n)}<div class="muted small">${esc(note)}</div></td><td class="num"><b>${pct(r[metric])}</b></td>
      <td class="num">${r.ci ? pct(r.ci[0]) + ' – ' + pct(r.ci[1]) : '–'}</td>${extra.map((c) => `<td class="num">${pct(r[c[0]])}</td>`).join('')}<td class="num">${r.n ?? ''}</td></tr>`).join('') +
    (target ? `<tr><td><b>Goal</b></td><td class="num"><b>${pct(target)}</b></td><td colspan="${2 + extra.length}" class="muted">${esc(C.goal_text || '')}</td></tr>` : '') ||
    '<tr><td class="muted">No checks yet.</td></tr>';
}
/** ETA: a straight line through the last 30% of the main series (the first series of the chart) extended to the goal.
 *  It is an extrapolation and says so. If stats.csv has a wall-clock column (t, time, elapsed_s, wall_s) it also gives hours. */
function eta(rows, cfg, target, run) {
  const s = (cfg.series || [])[0]; if (!s || !target || !rows.length) return '';
  const xk = cfg.x || 'it';
  const pts = rows.filter((r) => r[s.key] !== '' && r[s.key] != null && !isNaN(r[s.key])).map((r) => [+r[xk], +r[s.key], r]);
  if (pts.length < 10) return '';
  const tail = pts.slice(Math.floor(pts.length * 0.7));
  const n = tail.length, mx = tail.reduce((a, p) => a + p[0], 0) / n, my = tail.reduce((a, p) => a + p[1], 0) / n;
  const sxx = tail.reduce((a, p) => a + (p[0] - mx) ** 2, 0), sxy = tail.reduce((a, p) => a + (p[0] - mx) * (p[1] - my), 0);
  const slope = sxx ? sxy / sxx : 0, last = pts[pts.length - 1];
  if (last[1] >= target) return `<b>The goal line is reached</b> in the training measure; the goal table says whether it holds on unseen episodes.`;
  if (slope <= 0) return `<b>ETA:</b> none. Over the last ${n} points the ${esc(s.label || s.key)} is flat or falling (slope ${(slope * (cfg.x_scale || 1) * 100).toFixed(2)} % per ${esc(cfg.x_label || xk)}).`;
  const dx = (target - last[1]) / slope, xs = (cfg.x_scale || 1);
  let out = `<b>ETA (extrapolation):</b> at the rate of the last ${n} points, ${esc(s.label || s.key)} reaches ${pct(target, 0)} after another ${(dx * xs).toFixed(1)} ${esc(cfg.x_label || xk)} (at ${((last[0] + dx) * xs).toFixed(1)}).`;
  const tk = ['t', 'time', 'elapsed_s', 'wall_s', 'seconds'].find((k) => last[2][k] != null && last[2][k] !== '');
  if (tk) {
    const first = tail[0][2], dt = (+last[2][tk] - +first[tk]), dxt = last[0] - tail[0][0];
    if (dt > 0 && dxt > 0) out += ` That is about <b>${(dx / dxt * dt / 3600).toFixed(1)} h</b> at the recent pace${run.alive ? '' : ' (the run is stopped)'}.`;
  }
  return out + ' Learning curves bend, so treat this as a guess that is updated every 15 minutes.';
}
/** Extra charts: site.json "charts": [{title, y_label, percent, series:[{key,label}]}], or, if absent, every numeric
 *  column of stats.csv that the main chart does not show (up to 6), one chart each. */
function extraCharts(rows, C) {
  let charts = C.charts;
  const xk = (C.chart || {}).x || 'it';
  if (!charts) {
    const used = new Set([xk, ...((C.chart || {}).series || []).map((s) => s.key)]);
    const keys = Object.keys(rows[rows.length - 1] || {}).filter((k) => !used.has(k) && rows.some((r) => r[k] !== '' && !isNaN(r[k]) && r[k] !== null)).slice(0, 6);
    charts = keys.map((k) => ({ title: k.replace(/_/g, ' '), percent: false, series: [{ key: k, label: k }] }));
    $('morenote').textContent = keys.length ? 'Every other numeric column of stats.csv (pick them in site.json "charts" to choose and label).' : 'No other columns in stats.csv.';
  }
  else $('morenote').textContent = 'Charts chosen in site.json "charts".';
  $('charts').innerHTML = charts.map((c, i) => `<div class="chartbox"><h3>${esc(c.title || '')}</h3><div class="h"><canvas id="xc${i}"></canvas></div></div>`).join('');
  charts.forEach((c, i) => lineChart($('xc' + i), rows, { x: xk, x_scale: (C.chart || {}).x_scale, x_label: (C.chart || {}).x_label, percent: false, ...c }, null, null));
}
/** Stage timeline per run from events: a bar from stage_start to the next stage_start / stage_pass / stage_cap / now. */
function timeline(status) {
  const tl = status.timeline || {};
  const runs = Object.keys(tl);
  if (!runs.length) { $('timeline').innerHTML = '<p class="muted">No stage events yet.</p>'; return; }
  $('timeline').innerHTML = runs.map((run) => {
    const st = tl[run]; if (!st.length) return '';
    const t0 = Math.min(...st.map((s) => s.start)), t1 = Math.max(...st.map((s) => s.end));
    const span = Math.max(1, t1 - t0);
    return `<div class="tl"><div class="tlrun">run ${esc(run)} <span class="muted small">${((t1 - t0) / 3600).toFixed(1)} h</span></div>` +
      st.map((s) => `<div class="tlrow"><div class="tlname">${esc(s.stage)}</div><div class="tlbar"><div class="tlseg ${esc(s.status)}" style="left:${(100 * (s.start - t0) / span).toFixed(2)}%;width:${Math.max(0.5, 100 * (s.end - s.start) / span).toFixed(2)}%" title="${esc(s.stage)}: ${esc(s.start_text)} → ${esc(s.end_text)} (${((s.end - s.start) / 3600).toFixed(1)} h, ${esc(s.status)})"></div></div></div>`).join('') + '</div>';
  }).join('');
}
function runTable(runs, C, metric) {
  if (!runs.length) { $('runs').innerHTML = '<tr><td class="muted">No runs yet.</td></tr>'; return; }
  const xk = (C.chart || {}).x || 'it';
  $('runs').innerHTML = `<tr><th>Run</th><th>State</th><th>Stages</th><th>${esc((C.chart || {}).x_label || xk)}</th><th>Best check</th><th>Note</th></tr>` +
    runs.slice().reverse().map((r) => `<tr><td><b>${esc(r.run)}</b></td><td>${r.alive ? '<span class="dot on"></span>training' : 'stopped'}</td>
      <td class="num">${r.passed.length} / ${r.n_stages || '?'}</td><td class="num">${fmtM(+(r.last || {})[xk])}</td>
      <td class="num">${r.best_goal ? pct(r.best_goal[metric]) : '–'}</td><td class="muted small">${esc(r.note || '')}</td></tr>`).join('');
}
main();
