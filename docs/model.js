// The model page: generated figures (data/model.json "figures", written by the model_viz hook) and the champion timeline.
(async function () {
  const m = await get('model.json');
  const figs = (m && m.figures) || [];
  $('figs').innerHTML = figs.length ? figs.map((f) => `<figure class="fig"><a href="${esc(f.src)}?v=${esc(f.v || '')}" target="_blank" title="open full size"><img loading="lazy" src="${esc(f.src)}?v=${esc(f.v || '')}" alt="${esc(f.title)}"></a>
    <figcaption><b>${esc(f.title)}</b> ${esc(f.caption || '')}</figcaption></figure>`).join('')
    : '<p class="muted">No figures yet: implement <code>model_viz(out_dir)</code> in site_hooks.py.</p>';
  const ch = (m && m.timeline) || [];
  $('champions').innerHTML = ch.length ? '<tr><th>When</th><th>Run</th><th>Stage</th><th>Check</th><th>Model</th></tr>' +
    ch.map((c) => `<tr><td>${esc(c.t)}</td><td>${esc(c.run)}</td><td>${esc(c.stage || '')}</td><td class="num">${esc(c.check_text || '')}</td><td class="muted small">${esc(c.champion || '')}</td></tr>`).join('')
    : '<tr><td class="muted">No champions yet.</td></tr>';
})();
