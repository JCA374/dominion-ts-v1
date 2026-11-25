// Line charts from stats.csv rows (Chart.js). cfg: {x, x_scale, x_label, y_label, y_max, series: [{key, label, style, color, scale}]}
// The y axis autoscales to the data; the box around the canvas has a fixed height (.chartbox .h) so it never grows.
function lineChart(canvas, rows, cfg, target, noteEl) {
  if (!window.Chart || !rows.length || !cfg.series || !cfg.series.length) return null;
  const X = (r) => +r[cfg.x || 'it'] * (cfg.x_scale || 1);
  const ds = cfg.series.map((s, i) => {
    const sc = s.scale ?? (cfg.percent === false ? 1 : 100);
    const pts = rows.filter((r) => r[s.key] !== '' && r[s.key] != null && !isNaN(r[s.key])).map((r) => ({ x: X(r), y: sc * r[s.key] }));
    const col = cssv(s.color || ['--accent', '--warn', '--good', '--bad', '--muted'][i % 5]);
    return { label: s.label || s.key, data: pts, borderColor: col, backgroundColor: col, showLine: s.style !== 'points', pointRadius: s.style === 'line' || !s.style ? 0 : 2.5, borderWidth: 1.5, tension: .2 };
  });
  const ys = ds.flatMap((d) => d.data.map((p) => p.y));
  if (!ys.length) return null;
  const pc = cfg.percent !== false;
  const yMin = cfg.y_min ?? (pc ? Math.max(0, Math.floor((Math.min(...ys) - 2) / 5) * 5) : undefined);
  const yMax = cfg.y_max ?? (pc ? Math.ceil((Math.max(...ys, target ? 100 * target : 0) + 1) / 2) * 2 : undefined);
  if (target && pc) ds.push({ label: `goal ${pct(target, 0)}`, data: [{ x: X(rows[0]), y: 100 * target }, { x: X(rows[rows.length - 1]), y: 100 * target }], borderColor: cssv('--bad'), borderDash: [5, 4], borderWidth: 1, pointRadius: 0 });
  if (noteEl) noteEl.textContent = `x axis: ${cfg.x_label || cfg.x || 'iteration'}.`;
  return new Chart(canvas, { type: 'line', data: { datasets: ds }, options: { animation: false, maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: false },
    scales: { x: { type: 'linear', ticks: { color: cssv('--muted') }, grid: { color: cssv('--line') } },
      y: { min: yMin, max: yMax, title: { display: !!cfg.y_label, text: cfg.y_label || '', color: cssv('--muted') }, ticks: { color: cssv('--muted') }, grid: { color: cssv('--line') } } },
    plugins: { legend: { labels: { color: cssv('--ink'), boxWidth: 12 } } } } });
}
