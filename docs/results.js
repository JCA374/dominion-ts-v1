// Results page: data/ts.json (site_hooks.extra) as tables.
const tbl = (head, rows) => `<div class="tablewrap"><table><tr>${head.map((h) => `<th>${esc(h)}</th>`).join('')}</tr>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</table></div>`;
get('ts.json').then((d) => {
  if (!d) return;
  $('arms').innerHTML = tbl(['arm', 'what', 'opponent', 'draws', 'leaf', 'halving', 'kingdoms', 'score', '95% interval', 'buy / action overrides', 's per game'],
    d.arms.map((a) => [`<code>${esc(a.name)}</code>`, esc(a.note || ''), esc(a.args.opp || 'c2'), esc(a.args.k1 || ''), esc(a.args.leaf || '0'),
      a.args.alloc === '1' ? 'yes' : 'no', `${a.kingdoms} / ${a.target}${a.done ? '' : (a.kingdoms ? ' (playing)' : ' (queued)')}`,
      a.score != null ? pct(a.score) : '–', a.lo != null ? `${pct(a.lo)} – ${pct(a.hi)}` : '–',
      a.override_rate != null ? `${pct(a.override_rate)} / ${pct(a.act_override_rate)}` : '–', a.secs != null ? a.secs.toFixed(0) : '–']));
  const S = d.side || {};
  let h = '';
  if (S.paired && S.paired.rows && S.paired.rows.length) h += `<h3>Paired against ${esc(S.paired.base)}</h3><p class="muted small">Only the kingdoms both arms have finished; difference in points per kingdom (both seats), 95% bootstrap interval. An interval above 0 means the arm beats the baseline on the same games.</p>` +
    tbl(['arm', 'kingdoms in common', 'arm score', 'baseline score', 'difference', '95% interval'],
      S.paired.rows.map((r) => [`<code>${esc(r.arm)}</code>`, r.kingdoms, pct(r.arm_score), pct(r.base_score), (r.diff >= 0 ? '+' : '') + (100 * r.diff).toFixed(1) + ' pts', `${(100 * r.lo).toFixed(1)} to ${(100 * r.hi).toFixed(1)}`]));
  if (S.leaf_check) {
    const L = S.leaf_check;
    h += '<h3>Value-net leaves against full rollouts</h3><p class="muted small">The same label-only games (c2 plays, every buy graded with 128 draws), graded with rollouts to the end and with the value net after D turns.</p>' +
      tbl(['leaf after', 'seconds', 'cost vs full', 'level correlation', 'within-position correlation', 'same best option'],
        Object.entries(L).filter(([k]) => k !== 'full').map(([k, v]) => [k.replace('leaf', '') + ' turns', v.secs, (v.secs / L.full.secs).toFixed(2) + '×', v.corr_level, v.corr_within, pct(v.best_same)]));
  }
  for (const [k, v] of Object.entries(S)) if (k !== 'leaf_check' && k !== 'paired' && v && v.summary) h += `<h3>${esc(k)}</h3><p>${esc(v.summary)}</p>`;
  $('side').innerHTML = h || '<p class="muted">None yet.</p>';
});
