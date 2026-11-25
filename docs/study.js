// Study page: the glossary from data/glossary.json (publish.py converts glossary.md), alphabetical.
(async function () {
  const g = await get('glossary.json');
  const items = (g || []).slice().sort((a, b) => a.term.localeCompare(b.term));
  $('glossary').innerHTML = items.length ? items.map((x) => `<dt>${esc(x.term)}</dt><dd>${esc(x.meaning)}</dd>`).join('')
    : '<dd class="muted">No glossary yet: create glossary.md in the repo.</dd>';
})();
