// Log page: the event feed (runs' events.jsonl + journal + decisions + reports), with a filter by kind.
(async function () {
  const log = await get('log.json');
  $('log').innerHTML = (log || []).map(eventHtml).join('') || '<p class="muted">Nothing yet.</p>';
  $('filter').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    [...$('filter').children].forEach((x) => x.classList.toggle('on', x === b));
    const kinds = b.dataset.f ? b.dataset.f.split(',') : null;
    document.querySelectorAll('#log .ev').forEach((el) => { el.style.display = !kinds || kinds.includes(el.dataset.kind) ? '' : 'none'; });
  });
})();
