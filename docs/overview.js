// Overview page: status cards, the decisions log, and the guide to the tabs (all from data/status.json).
const TAB_HELP = {
  'index.html': 'What the project is, how it works, the decisions taken, and where things are.',
  'log.html': 'What is happening, newest first: notes, stage passes, best checks, crashes, with replays.',
  'training.html': 'Progress charts, the goal table, the stage timeline, run comparison and an estimate of when the goal is reached.',
  'curriculum.html': 'The stages each run goes through: world, reward, pass check, cap and what was expected, written before the stage ran.',
  'senses.html': 'Every input the model gets, what it may not see, and the test that checks it.',
  'inputs.html': 'Every input the model gets: where it comes from, what each feature means, the splits and the leakage rule.',
  'model.html': 'What the model is and how it learns, with generated pictures of the real network and a timeline of its champions.',
  'play.html': 'Watch the newest models on a fixed set of games and compare them side by side.',
  'speed.html': 'The machine, the profiles and the before/after timings of each optimisation.',
  'study.html': 'For someone learning from the project: a reading order, the key concepts, a glossary, how to reproduce it, exercises.',
  'lessons.html': 'What this project taught, as rules for the next one.',
  'podcast/index.html': 'Short spoken episodes about the progress, with transcripts.',
};
document.addEventListener('status', (ev) => {
  const st = ev.detail; if (!st) { $('cards').innerHTML = '<p class="muted">No data yet.</p>'; return; }
  const C = st.config || {}, s = st.strip || {};
  if (C.deadline) {
    const tick = () => {
      const sec = Math.max(0, (new Date(C.deadline) - new Date()) / 1000);
      const d = Math.floor(sec / 86400), h = Math.floor(sec % 86400 / 3600), m = Math.floor(sec % 3600 / 60);
      $('countdown').textContent = sec ? `⏱ ${d} d ${h} h ${m} min left${C.deadline_text ? ' until ' + C.deadline_text : ''}` : 'Deadline reached.';
    };
    tick(); setInterval(tick, 30000);
  }
  const cards = [['Run', s.run || '–', s.alive ? 'training now' : 'stopped'], ...(s.stage ? [['Stage', s.stage, s.stages_text || '']] : []),
    ['Best verified', s.verified_text || '–', s.verified_sub || 'goal check on unseen episodes'],
    ['Best check', s.best_text || '–', 'best of many: flatters'],
    ['Goal', s.goal_text || '–', C.goal_text || '']];
  $('cards').innerHTML = cards.map(([k, v, sub]) => `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${esc(sub)}</div></div>`).join('');
  const dec = st.decisions || [];
  $('decisions').innerHTML = dec.length ? dec.map((d) => `<div class="box dec"><h3>${esc(d.title)} <span class="muted small">${esc(d.t)}</span></h3>
    <p>${esc(d.text)}</p>${d.rejected ? `<p class="muted"><b>Rejected:</b> ${esc(d.rejected)}</p>` : ''}</div>`).join('') : '<p class="muted">No decisions logged yet.</p>';
  $('guide').innerHTML = '<dl class="guide">' + TABS.map(([h, t]) => `<dt><a href="${h}">${esc(t)}</a></dt><dd>${esc(TAB_HELP[h] || '')}</dd>`).join('') + '</dl>';
});
