"""The site's view of the match arms (called by site_hooks before every build): one run per arm of queue.txt,
runs/<arm>/stats.csv (the running paired score after every 10 completed kingdoms, with its bootstrap 95% interval) and
runs/<arm>/events.jsonl (start, finish), rebuilt from results/match_<arm>.jsonl each time (deterministic)."""
import csv
import json
import os
import shlex
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")


def arms():
    """[(name, args dict, comment above it)] in queue order."""
    out, note = [], ""
    for line in open(os.path.join(ROOT, "queue.txt")):
        s = line.strip()
        if s.startswith("#stopped "):                  # an arm stopped early: still shown, never resumed
            s = s[len("#stopped "):]
        elif s.startswith("#"):
            note = s.lstrip("# ")
            continue
        s = s.split("#")[0].strip()
        if not s:
            continue
        a = shlex.split(s)
        d = {a[i][2:]: a[i + 1] for i in range(0, len(a) - 1) if a[i].startswith("--") and not a[i + 1].startswith("--")}
        out.append((d["name"], d, note))
        note = ""
    return out


def rows_of(name):
    p = os.path.join(RESULTS, f"match_{name}.jsonl")
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def pair_points(rows):
    """[(i, mean points of both seats, secs of both)] for completed kingdoms, in completion order."""
    by, out = {}, []
    for r in rows:
        by.setdefault(r["i"], {})[r["seat"]] = r
        if len(by[r["i"]]) == 2:
            v = by[r["i"]].values()
            out.append((r["i"], float(np.mean([x["points"] for x in v])), sum(x["secs"] for x in v)))
    return out


def ci(pts, n=2000):
    a = np.array(pts)
    rng = np.random.default_rng(0)
    b = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def update():
    for name, args, note in arms():
        rows = rows_of(name)
        if not rows:
            continue
        d = os.path.join(ROOT, "runs", name)
        os.makedirs(d, exist_ok=True)
        pp = pair_points(rows)
        with open(os.path.join(d, "stats.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kingdoms", "games", "score", "lo", "hi", "secs_per_game", "t"])
            for j in range(10, len(pp) + 1, 10):
                pts = [p for _, p, _ in pp[:j]]
                lo, hi = ci(pts)
                w.writerow([j, 2 * j, round(float(np.mean(pts)), 4), round(lo, 4), round(hi, 4),
                            round(sum(s for _, _, s in pp[:j]) / (2 * j), 1), round(sum(s for _, _, s in pp[:j]) / 7)])
        summ = os.path.join(RESULTS, f"match_{name}.json")
        s = json.load(open(summ)) if os.path.exists(summ) else {}
        ev = [{"t": s.get("t", ""), "kind": "run_start", "text": note or name, "args": args}]
        if s.get("done"):
            ev.append({"t": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(summ))),
                       "kind": "arm_done", "text": f"{name}: {100 * s['score']:.1f}% vs {args.get('opp', 'c2')} "
                       f"[{100 * s['ci95'][0]:.1f}, {100 * s['ci95'][1]:.1f}] over {s['games']} games"})
        with open(os.path.join(d, "events.jsonl"), "w") as f:
            for e in ev:
                f.write(json.dumps(e) + "\n")
        json.dump({"note": note, "stages": [{"name": name, "world": note, "pass": "a full arm of the queue",
                                             "expect": ""}]}, open(os.path.join(d, "arm.json"), "w"))


if __name__ == "__main__":
    update()
