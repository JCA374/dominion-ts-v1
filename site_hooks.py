"""Project hooks for publish.py (ml-project-site): the arms' run files (track.py) and the Results data. No Play tab
(no game clips in this project). Drawing lives here only; no policy imports this file."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
try:
    import track
    track.update()
except Exception as ex:                        # never stop the publisher
    print("track.update failed:", ex)


def models():
    return []


def games():
    return []


def model_viz(out_dir):
    return []


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def extra(data_dir):
    """data/ts.json for the Results page: every arm of the queue with its live paired score, and the side results."""
    out = {"arms": [], "side": {}}
    for name, args, note in track.arms():
        pp = track.pair_points(track.rows_of(name))
        s = _load(os.path.join(ROOT, "results", f"match_{name}.json")) or {}
        row = {"name": name, "note": note, "args": args, "kingdoms": len(pp), "done": bool(s.get("done")),
               "target": int(args.get("kingdoms", 0))}
        if len(pp) >= 5:
            pts = [p for _, p, _ in pp]
            lo, hi = track.ci(pts)
            row.update(score=sum(pts) / len(pts), lo=lo, hi=hi, secs=sum(x for _, _, x in pp) / (2 * len(pp)),
                       override_rate=s.get("override_rate"), act_override_rate=s.get("act_override_rate"))
        out["arms"].append(row)
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        b = os.path.basename(p)
        if not b.startswith("match_"):
            out["side"][b[:-5]] = _load(p)
    json.dump(out, open(os.path.join(data_dir, "ts.json"), "w"), indent=0, default=str)
