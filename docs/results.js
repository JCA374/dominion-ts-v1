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
  if (S.bank) h += `<h3>Decision bank</h3><p>${esc(S.bank.summary)}. ${S.bank.buys} buys, ${S.bank.actions} action plays from ${S.bank.games} c2-vs-c2 games; truth = ${S.bank.truth_draws} rollouts to the end per option.</p>`;
  const E = S.bank_eval;
  if (E) {
    const rows = Object.entries(E).filter(([k, v]) => v && v.gates);
    h += '<h3>Teacher settings on the bank</h3><p class="muted small">Regret = points of win chance a choice gives up against the truth, per decision (lower is better), mean of 3 repetitions. "Today\'s gate" = margin 0.02, z 2. Seconds per question measured with the machine busy (compare them with each other only).</p>' +
      tbl(['setting', 'leaf', 'halving', 'draws', 's / question', 'regret, today\'s gate', 'gain over c2', 'overrides', 'best gate', 'regret there'],
        [['c2 alone', '', '', '', '', (100 * E.c2.regret).toFixed(2), '–', '0', '', '']].concat(rows.map(([k, v]) => { const g = v.gates['m0.02_z2.0'], b = v.gates[v.best_gate];
          return [`<code>${esc(k)}</code>`, v.leaf || 'end', v.alloc ? 'yes' : 'no', v.draws, v.secs_per_question, (100 * g.regret).toFixed(2), `${(100 * g.gain_vs_c2).toFixed(2)} [${(100 * g.gain_ci[0]).toFixed(2)}, ${(100 * g.gain_ci[1]).toFixed(2)}]`, pct(g.override_rate), esc(v.best_gate), (100 * b.regret).toFixed(2)]; })));
  }
  for (const [k, v] of Object.entries(S)) if (!['leaf_check', 'paired', 'bank', 'bank_eval'].includes(k) && v && v.summary) h += `<h3>${esc(k)}</h3><p>${esc(v.summary)}</p>`;
  $('side').innerHTML = h || '<p class="muted">None yet.</p>';
});
