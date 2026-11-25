"""Build the project site's data from the runs and push it (every 15 min via publish_loop.sh).
From the ml-project-site skill (~/.claude/skills/ml-project-site). Project-specific parts live in site.json and
site_hooks.py; this file stays generic, so it can be re-copied from the skill when the skill improves.

Reads (conventions):
  runs/<run>/stats.csv          one row per iteration/generation (any columns; site.json says which to chart)
  runs/<run>/events.jsonl       {"t", "kind", ...}: run_start, stage_start, stage_pass, check_pass, goal_best, champion, ...
  runs/<run>/curriculum.json    {"note", "stages": [{"name", "world", "reward", "pass", "cap", "expect", ...}]}
  experiments/journal.jsonl     {"t", "title", "text"}: notes for the log
  results/goal_*.json           goal checks {"t", "n", "landed"/metric, "ci", "label", "note", "policy", ...}
  results/baseline_*.json, results/bake_*.json   tables on the Training page
  glossary.md                   `**term** — meaning` per line, for the Study page
  docs/reports/*.html           analysis reports (failure-analysis-report), listed on Training and in the Log
Writes docs/data/{status,stats,log,curricula,play,model,glossary}.json, docs/replays/*.gif (play once),
docs/play/*.mp4 (play once), docs/model/* (figures from the model_viz hook), docs/site.js (title, tabs).

Tabs: a tab appears when its page exists in docs/ (see TAB_ORDER); Curriculum also needs a curriculum.json in some run.
site.json "tabs" (a list of [file, label]) overrides that; "tab_labels" renames single tabs.

    python publish.py            # build only
    python publish.py --push     # build, commit docs/ + run logs, push (never stashes)
"""
import argparse
import csv
import glob
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
DOCS = os.path.join(ROOT, 'docs')
DATA = os.path.join(DOCS, 'data')
REPLAYS = os.path.join(DOCS, 'replays')
PLAY = os.path.join(DOCS, 'play')
CFG = json.load(open(os.path.join(ROOT, 'site.json')))
REPLAY_KINDS = set(CFG.get('replay_kinds', ['goal_best', 'stage_pass', 'goal_check', 'champion']))
MODEL = os.path.join(DOCS, 'model')
# Tab order and default labels. index.html is the Overview when log.html exists, else (older projects) the Log.
TAB_ORDER = [('index.html', 'Overview'), ('log.html', 'Log'), ('training.html', 'Training'), ('curriculum.html', 'Curriculum'),
             ('senses.html', 'How it sees'), ('inputs.html', None), ('model.html', 'The model'), ('play.html', 'Play'),
             ('speed.html', 'Speed'), ('study.html', 'Study'), ('lessons.html', 'Lessons'), ('podcast/index.html', 'Podcast')]


def hooks():
    try:
        return importlib.import_module('site_hooks')
    except ImportError:
        return None


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))           # never awk -F, : stage names may contain commas


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def to_ts(t):
    """'YYYY-MM-DD HH:MM[:SS]' (local time, as the events write it) -> epoch seconds; None if unparsable."""
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M:%S'):
        try:
            return time.mktime(time.strptime(str(t)[:19], fmt))
        except (ValueError, TypeError):
            pass
    return None


def page_title(path):
    m = re.search(r'<title>(.*?)</title>', open(path).read(), re.S)
    t = m.group(1).strip() if m else os.path.basename(path)
    return re.sub(rf'^{re.escape(CFG.get("title", ""))}\s*[:·-]\s*', '', t) if CFG.get('title') else t


def tabs(have_curriculum):
    """The navigation: site.json "tabs" if given, else every page of TAB_ORDER that exists in docs/."""
    if CFG.get('tabs'):
        return CFG['tabs']
    labels = CFG.get('tab_labels', {})
    out = []
    for f, label in TAB_ORDER:
        p = os.path.join(DOCS, f)
        if not os.path.exists(p):
            continue
        if f == 'curriculum.html' and not have_curriculum and not CFG.get('curriculum'):
            continue
        if f == 'index.html' and not os.path.exists(os.path.join(DOCS, 'log.html')):
            label = 'Log'
        out.append([f, labels.get(f) or label or page_title(p)])
    return out


def thin(rows, n=600, keep_keys=()):
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    keep = {int(i * step) for i in range(n)} | {len(rows) - 1}
    return [r for i, r in enumerate(rows) if i in keep or any(r.get(k) for k in keep_keys)]


def replay_for(h, policy, name, seed):
    """A GIF of one episode of `policy`, rendered once and cached. GIFs play ONCE (no loop); the page replays on click."""
    path = os.path.join(REPLAYS, name)
    if not os.path.exists(path) and h and hasattr(h, 'replay_gif'):
        try:
            h.replay_gif(policy, seed, path)
        except Exception as ex:                  # a replay must never stop the publisher
            print('replay failed', name, ex)
    return f'replays/{name}' if os.path.exists(path) else None


def build_play(h):
    """The Play tab: the newest models on a fixed set of games, one MP4 per (model, game), rendered once and cached.
    Clips of models that dropped off the list are deleted."""
    if not h or not hasattr(h, 'models') or not hasattr(h, 'games'):
        return
    os.makedirs(PLAY, exist_ok=True)
    models, games = h.models(), h.games()
    out = {'models': [], 'games': games}
    keep = set()
    for m in models:
        entry = dict(m, clips={})
        for g in games:
            name = f"{m['id']}__{g['id']}.mp4"
            path = os.path.join(PLAY, name)
            meta = path + '.json'
            keep |= {name, name + '.json'}
            if not os.path.exists(meta):
                try:
                    res = h.render_play(m['path'], g, path)
                    json.dump(res, open(meta, 'w'))
                except Exception as ex:
                    print('play render failed', name, ex)
                    continue
            entry['clips'][g['id']] = {'src': f'play/{name}', **json.load(open(meta))}
        out['models'].append(entry)
    for f in os.listdir(PLAY):
        if f not in keep:
            os.remove(os.path.join(PLAY, f))
    json.dump(out, open(os.path.join(DATA, 'play.json'), 'w'), separators=(',', ':'))


def build_model(h, log):
    """The model tab: figures from the model_viz hook (cached by the hook; a failure never stops the publisher) and the
    timeline of champions from goal_best / stage_pass events."""
    figures = []
    if h and hasattr(h, 'model_viz'):
        os.makedirs(MODEL, exist_ok=True)
        try:
            for f in h.model_viz(MODEL) or []:
                p = os.path.join(DOCS, f['src'])
                if os.path.exists(p):
                    figures.append(dict(f, v=hashlib.md5(open(p, 'rb').read()).hexdigest()[:8]))
        except Exception as ex:
            print('model_viz failed', ex)
    timeline = []
    for e in log:
        if e['kind'] in ('goal_best', 'stage_pass') and (e.get('champion') or e.get('check')):
            c = e.get('check') or {}
            key = CFG.get('goal', {}).get('metric', 'landed')
            val = c.get(key, c.get('score')) if isinstance(c, dict) else c      # older events: a plain string
            timeline.append({'t': e['t'], 'run': e.get('run'), 'stage': e.get('stage', ''), 'champion': e.get('champion', ''),
                             'check_text': (f'{100 * val:.1f}%' if isinstance(val, (int, float)) and val <= 1 else str(val)) if val is not None else ''})
    json.dump({'figures': figures, 'timeline': timeline}, open(os.path.join(DATA, 'model.json'), 'w'), separators=(',', ':'))


def build_glossary():
    """glossary.md -> data/glossary.json. Lines like `**term** — meaning`, `term: meaning` or `- term — meaning`."""
    path = os.path.join(ROOT, 'glossary.md')
    items = []
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r'^\s*(?:[-*]\s*)?\**([^*:—–-]+?)\**\s*(?:—|–|:|-)\s+(.+?)\s*$', line)
            if m and not line.startswith('#'):
                items.append({'term': m.group(1).strip(), 'meaning': m.group(2).strip()})
    json.dump(items, open(os.path.join(DATA, 'glossary.json'), 'w'), separators=(',', ':'))


def reports():
    """docs/reports/*.html, newest first, title from the page."""
    out = []
    for p in glob.glob(os.path.join(DOCS, 'reports', '*.html')):
        m = re.search(r'<title>(.*?)</title>', open(p, errors='replace').read(2000), re.S)
        out.append({'href': 'reports/' + os.path.basename(p), 'title': (m.group(1).strip() if m else os.path.basename(p)),
                    't': time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p))), 'ts': os.path.getmtime(p)})
    return sorted(out, key=lambda r: -r['ts'])


def stage_timeline(events, alive):
    """Per run: [{stage, start, end, status}] from stage_start / stage_pass / stage_cap events (epoch seconds)."""
    out = []
    for e in events:
        ts = to_ts(e.get('t'))
        if ts is None:
            continue
        if e['kind'] == 'stage_start':
            if out and out[-1]['end'] is None:
                out[-1]['end'] = ts
            out.append({'stage': e.get('stage', '?'), 'start': ts, 'end': None, 'status': 'stopped'})
        elif e['kind'] in ('stage_pass', 'stage_cap') and out:
            cur = next((s for s in reversed(out) if s['stage'] == e.get('stage')), out[-1])
            cur['end'] = cur['end'] or ts
            cur['status'] = 'passed' if e['kind'] == 'stage_pass' else 'capped'
    for s_ in out:
        if s_['end'] is None:
            s_['end'] = time.time() if alive else max(s_['start'], to_ts(events[-1].get('t')) or s_['start'])
            s_['status'] = 'running' if alive else s_['status']
        s_['start_text'] = time.strftime('%Y-%m-%d %H:%M', time.localtime(s_['start']))
        s_['end_text'] = time.strftime('%Y-%m-%d %H:%M', time.localtime(s_['end']))
    return out


def build():
    h = hooks()
    for d in (DATA, REPLAYS):
        os.makedirs(d, exist_ok=True)
    chart_keys = [s['key'] for s in CFG.get('chart', {}).get('series', [])]
    log, stats_all, curricula, runs, timeline = [], {}, {}, [], {}
    for rdir in sorted(glob.glob(os.path.join(ROOT, 'runs', '*')), key=os.path.getmtime):
        name = os.path.basename(rdir)
        if name.startswith('test_') or not os.path.exists(os.path.join(rdir, 'stats.csv')):
            continue
        cpath = os.path.join(rdir, 'curriculum.json')
        cur = json.load(open(cpath)) if os.path.exists(cpath) else None
        if cur:
            curricula[name] = cur
        rows = read_csv(os.path.join(rdir, 'stats.csv'))
        stats_all[name] = thin(rows, keep_keys=chart_keys)
        events = read_jsonl(os.path.join(rdir, 'events.jsonl'))
        for i, e in enumerate(events):
            e = dict(e, run=name)
            if e['kind'] in REPLAY_KINDS and e.get('champion'):
                e['replay'] = replay_for(h, os.path.join(ROOT, e['champion']), f'{name}_{i:04d}.gif',
                                         CFG.get('replay_seed', 3_000_777_000) + i)
            e.pop('args', None)
            log.append(e)
        alive = subprocess.run(['pgrep', '-f', CFG.get('alive_pattern', '^.venv/bin/python .* --run {run}( |$)').format(run=name)],
                               capture_output=True).returncode == 0
        best_key = CFG.get('best_event', 'goal_best')
        best = max((e['check'] for e in events if e['kind'] == best_key and isinstance(e.get('check'), dict)), default=None,
                   key=lambda c: c.get('landed', c.get('score', 0)))
        runs.append({'run': name, 'note': (cur or {}).get('note', ''), 'alive': alive, 'last': rows[-1] if rows else {},
                     'n_stages': len((cur or {}).get('stages', [])), 'best_goal': best,
                     'passed': [e['stage'] for e in events if e['kind'] == 'stage_pass']})
        if events:
            timeline[name] = stage_timeline(events, alive)
    decisions = []
    for e in read_jsonl(os.path.join(ROOT, 'experiments', 'journal.jsonl')):
        e = dict(e, kind=e.get('kind', 'journal'))
        log.append(e)
        if e['kind'] == 'decision':
            decisions.append(e)
    reps = reports()
    for r in reps:
        log.append({'kind': 'report', 't': r['t'], 'title': r['title'], 'href': r['href']})
    for g in sorted(glob.glob(os.path.join(ROOT, 'results', 'goal_*.json'))):
        r = json.load(open(g))
        e = {'kind': 'goal_check', **{k: v for k, v in r.items() if k != 'failed_seeds'}}
        if r.get('policy'):
            e['replay'] = replay_for(h, os.path.join(ROOT, r['policy']), 'goal_' + os.path.basename(g).replace('.json', '.gif'),
                                     CFG.get('replay_seed', 3_000_777_000) + 1000)
        log.append(e)
    log.sort(key=lambda e: e['t'], reverse=True)

    def res(pat):
        return {os.path.basename(p)[:-5]: {k: v for k, v in json.load(open(p)).items() if k != 'failed_seeds'}
                for p in sorted(glob.glob(os.path.join(ROOT, 'results', pat)))}
    goals = res('goal_*.json')
    status = {'updated': time.strftime('%Y-%m-%d %H:%M'), 'updated_ts': time.time(), 'config': {k: CFG.get(k) for k in
              ('title', 'deadline', 'deadline_text', 'goal', 'goal_text', 'chart', 'charts', 'cards', 'names', 'repo', 'stale_minutes')},
              'runs': runs, 'baselines': res('baseline_*.json'), 'bakeoff': res('bake_*.json'), 'goals': goals,
              'timeline': timeline, 'decisions': sorted(decisions, key=lambda d: d['t'], reverse=True), 'reports': reps}
    status['strip'] = strip(runs, goals)
    for name, obj in [('status', status), ('stats', stats_all), ('log', log), ('curricula', curricula)]:
        json.dump(obj, open(os.path.join(DATA, f'{name}.json'), 'w'), separators=(',', ':'))
    site = {k: CFG.get(k) for k in ('title', 'icon', 'repo')}
    site['tabs'] = tabs(bool(curricula))
    open(os.path.join(DOCS, 'site.js'), 'w').write('window.SITE = ' + json.dumps(site) + ';\n')
    build_play(h)
    build_model(h, log)
    build_glossary()
    if h and hasattr(h, 'extra'):
        h.extra(DATA)                            # project-specific data files (e.g. a senses or speed page)


def strip(runs, goals):
    """The one-line status shown on every page (common.js): current run, stage, best check, best verified goal check."""
    G = CFG.get('goal', {})
    metric, target = G.get('metric', 'landed'), G.get('target')
    fmt = lambda v: (f'{100 * v:.1f}%' if isinstance(v, (int, float)) and v <= 1 else str(v)) if v is not None else None
    main = next((r for r in runs if r['run'] == CFG.get('main_run')), None) or \
           next((r for r in runs if r['alive']), runs[-1] if runs else None)   # site.json main_run: the run to show
    out = {'alive': bool(main and main['alive']), 'run': main['run'] if main else None,
           'stage': (main or {}).get('last', {}).get('stage'), 'goal_text': fmt(target),
           'stages_text': f"{len(main['passed'])} of {main['n_stages'] or '?'} stages passed" if main else '',
           'best_text': fmt((main or {}).get('best_goal', {}).get(metric)) if main and main.get('best_goal') else None}
    best = max(goals.values(), key=lambda g: g.get(metric) or 0, default=None)
    if best:
        out['verified_text'] = fmt(best.get(metric))
        out['verified_sub'] = f"{best.get('label') or 'goal check'}, n={best.get('n', '?')}"
    return out


def bust_caches():
    """Pages are cached for 10 minutes: version scripts and styles so a browser never runs an old one."""
    assets = [f for f in os.listdir(DOCS) if f.endswith(('.js', '.css'))]
    for page in glob.glob(os.path.join(DOCS, '*.html')):
        html = open(page).read()
        for a in assets:
            v = hashlib.md5(open(os.path.join(DOCS, a), 'rb').read()).hexdigest()[:8]
            html = re.sub(rf'(src|href)="{re.escape(a)}(\?v=\w+)?"', rf'\1="{a}?v={v}"', html)
        open(page, 'w').write(html)


def push():
    """Commit and push. Never `git pull --rebase --autostash`: its autostash once failed to re-apply and silently
    reverted uncommitted code for an hour. Only this machine pushes; if a push is rejected, merge and push again."""
    os.chdir(ROOT)
    for f in ['docs', 'experiments/journal.jsonl', 'results', 'curricula', 'site.json']:
        if os.path.exists(f):
            subprocess.run(['git', 'add', f])
    for pat in CFG.get('push_globs', ['runs/*/stats.csv', 'runs/*/events.jsonl', 'runs/*/curriculum.json']):
        for f in glob.glob(pat):
            if '/test_' not in f:
                subprocess.run(['git', 'add', '-f', f])
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode == 0:
        return print('nothing to publish')
    subprocess.run(['git', 'commit', '-q', '-m', f'Site update {time.strftime("%Y-%m-%d %H:%M")}'], check=True)
    if subprocess.run(['git', 'push', '-q'], check=False).returncode != 0:
        subprocess.run(['git', 'pull', '-q', '--no-rebase', '--no-edit'], check=False)
        subprocess.run(['git', 'push', '-q'], check=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--push', action='store_true')
    a = ap.parse_args()
    build()
    bust_caches()
    if a.push:
        push()
