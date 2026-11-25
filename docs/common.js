// Shared by every page: navigation, the status strip, helpers, formatters, the event renderer. Site settings come from
// site.js (written by publish.py from site.json: title, icon, repo, tabs). A project can override worldText /
// rewardText / passText / eventText in project.js (loaded after this).
const SITE = window.SITE || { title: 'Project', repo: '', tabs: [] };
const TABS = SITE.tabs && SITE.tabs.length ? SITE.tabs : [['index.html', 'Overview'], ['log.html', 'Log'],
  ['training.html', 'Training'], ['model.html', 'The model'], ['play.html', 'Play'], ['speed.html', 'Speed'],
  ['study.html', 'Study'], ['lessons.html', 'Lessons'], ['podcast/index.html', 'Podcast']];
const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x === '' || x == null || isNaN(x)) ? '–' : (100 * x).toFixed(d) + '%';
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const get = (f) => fetch('data/' + f + '?t=' + Date.now()).then((r) => r.ok ? r.json() : null).catch(() => null);
const cssv = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const kv = (o = {}) => Object.entries(o).map(([k, v]) => `${k.replace(/_/g, ' ')} ${Array.isArray(v) ? v.join('→') : v}`).join(', ');
const fmtM = (v) => v == null || v === '' || isNaN(v) ? '–' : +v >= 1e6 ? (v / 1e6).toFixed(1) + ' M' : +v >= 1e3 ? (v / 1e3).toFixed(0) + ' k' : String(v);
const ago = (sec) => sec < 90 ? 'just now' : sec < 5400 ? Math.round(sec / 60) + ' min ago' : sec < 172800 ? (sec / 3600).toFixed(1) + ' h ago' : Math.round(sec / 86400) + ' d ago';
window.worldText = window.worldText || ((w, ramp) => kv(w) + (ramp && Object.keys(ramp).length ? '; ramps: ' + kv(ramp) : ''));
window.rewardText = window.rewardText || ((r) => kv(r) || 'default');
window.passText = window.passText || ((p) => kv(p));

(function () {
  const here = location.pathname.split('/').pop() || 'index.html';
  const nav = document.createElement('nav'); nav.className = 'top';
  nav.innerHTML = `<a class="brand" href="index.html">${SITE.icon || '🧪'} ${SITE.title}</a>` + TABS.map(([h, t]) =>
    `<a href="${h}" class="${h === here ? 'on' : ''}">${t}</a>`).join('') + (SITE.repo ? `<a href="https://github.com/${SITE.repo}">Code</a>` : '');
  document.body.prepend(nav);
  const strip = document.createElement('div'); strip.className = 'strip'; strip.id = 'strip';
  nav.after(strip);
  // replay GIFs play once; click to watch again
  document.addEventListener('click', (e) => {
    if (e.target.tagName === 'IMG' && e.target.closest('.ev')) { const s = e.target.src.split('#')[0]; e.target.src = s + '#' + Date.now(); }
  });
  get('status.json').then((st) => { window.STATUS = st; statusStrip(st); document.dispatchEvent(new CustomEvent('status', { detail: st })); });
})();

/** The one-line status on every page: run alive or not, stage, best result, when the site last updated.
 *  A red banner when the site has not updated for `stale_minutes` (default 60): the publish loop may have died. */
function statusStrip(st) {
  const el = $('strip'); if (!el) return;
  if (!st) { el.innerHTML = '<span class="muted">No data published yet.</span>'; return; }
  const s = st.strip || {}, C = st.config || {};
  const age = st.updated_ts ? Date.now() / 1000 - st.updated_ts : null;
  const staleAfter = 60 * (C.stale_minutes || 60);
  const dot = s.alive ? '<span class="dot on"></span>training now' : '<span class="dot"></span>' + (s.run ? 'stopped' : 'no run');
  const parts = [dot];
  if (s.run) parts.push(`run <b>${esc(s.run)}</b>`);
  if (s.stage) parts.push(`stage <b>${esc(s.stage)}</b>`);
  if (s.best_text) parts.push(`best <b>${esc(s.best_text)}</b>`);
  if (s.goal_text) parts.push(`goal ${esc(s.goal_text)}`);
  parts.push(`<a href="training.html">updated ${age != null ? ago(age) : esc(st.updated)}</a>`);
  el.innerHTML = parts.map((x) => `<span>${x}</span>`).join('<span class="sep">·</span>');
  if (age != null && age > staleAfter) {
    const b = document.createElement('div'); b.className = 'stale';
    b.textContent = `⚠️ This site has not updated since ${st.updated} (${ago(age)}). The publish loop may have stopped: check publish_loop.pid and logs/publish.log on the training machine.`;
    el.after(b);
  }
}

function chk(c) { return c ? Object.entries(c).filter(([k, v]) => typeof v === 'number' && k !== 'n').map(([k, v]) => `${k} ${v <= 1 ? pct(v) : v.toFixed(1)}`).join(', ') + (c.n ? ` (n=${c.n})` : '') : ''; }
/** One log event as HTML. Projects add kinds (or reword these) with eventText(e) -> [title, body] in project.js. */
function eventHtml(e) {
  let [title, body] = (window.eventText && window.eventText(e)) || [null, null];
  if (title == null) {
    switch (e.kind) {
      case 'journal': title = e.title; body = e.text; break;
      case 'decision': title = `Decision: ${e.title}`; body = `${e.text}${e.rejected ? '\nRejected: ' + e.rejected : ''}`; break;
      case 'report': title = `Analysis report: ${e.title}`; body = e.text || ''; break;
      case 'run_start': title = `Run ${e.run} started`; body = e.note; break;
      case 'stage_start': title = `Stage started: ${e.stage}`; body = e.expect ? 'Expected: ' + e.expect : ''; break;
      case 'goal_best': title = 'Best check so far'; body = chk(e.check) + "\nBest of many checks: flattered by luck (winner's curse)."; break;
      case 'check_pass': title = `Check passed in ${e.stage}, confirming on a fresh block`; body = `First: ${chk(e.check)}\nConfirm: ${chk(e.confirm)}`; break;
      case 'stage_pass': title = `✅ Stage passed: ${e.stage}`; body = `Check: ${chk(e.check)}\nConfirm: ${chk(e.confirm)}`; break;
      case 'stage_cap': title = `Stage ${e.stage} reached its cap; moving on`; break;
      case 'crash': title = '⚠️ Training crashed and was restarted from its last checkpoint'; body = e.note; break;
      case 'override': title = `Manual change: ${e.what}`; body = e.reason; break;
      case 'goal_check': title = `🎯 Goal check: ${pct(e.landed ?? e.value)} on ${e.n} unseen episodes`; body = `${e.label || e.policy || ''}. 95% interval ${e.ci ? pct(e.ci[0]) + ' – ' + pct(e.ci[1]) : '–'}. ${e.note || ''}`; break;
      default: title = e.title || e.kind.replace(/_/g, ' '); body = e.note || e.msg || e.text || '';
    }
  }
  const img = e.replay ? `<img loading="lazy" src="${e.replay}" title="click to replay" alt="replay of the agent at this moment">` : '';
  const link = e.href ? `<p><a href="${esc(e.href)}">Open the report →</a></p>` : '';
  return `<div class="ev" data-kind="${esc(e.kind)}"><div class="body"><div class="t">${esc(e.t)}${e.run ? ' · run ' + esc(e.run) : ''}</div>
    <h4><span class="tag ${e.kind}">${e.kind.replace(/_/g, ' ')}</span>${esc(title)}</h4><p>${esc(body)}</p>${link}</div>${img}</div>`;
}
