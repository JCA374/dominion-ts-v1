"""Do value-net leaves grade like full rollouts? The same label-only games (teacher mode 2: c2 plays, every buy graded,
so the games are identical across settings) graded with full rollouts (K=128) and with leaves after D turns (K=128).
Prints per D: correlation of the option values with the full-rollout ones, mean level, agreement on the best option,
and rollout cost.

    python experiments/leaf_check.py --games 8 --d 2 4 8
"""
import argparse, json, os, random, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ts import lib
core = lib.core
ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=8)
ap.add_argument("--k", type=int, default=128)
ap.add_argument("--d", type=int, nargs="+", default=[2, 4, 8])
a = ap.parse_args()


def grade(leaf):
    lib.configure(leaf=leaf, k0=a.k, k1=a.k, margin=0.02, z=2, act=0, obs=1)
    rng = random.Random("ts-v1 leaf check")
    Q, t = [], time.time()
    for i in range(a.games):
        k, seed = core.random_kingdom(rng), rng.getrandbits(32)
        r = core.play([core.seat_for("c2", k, 2), core.seat_for("c2", k)], k, seed, seed ^ 0x5EED)
        Q.append(r["rec"]["Q"])
    return np.concatenate(Q), time.time() - t


ref, tr = grade(0)
out = {"full": {"secs": round(tr, 1), "rows": len(ref)}}
for d in a.d:
    q, t = grade(d)
    assert q.shape == ref.shape
    m = ~np.isnan(ref)
    rows = [i for i in range(len(ref)) if m[i].sum() >= 2]
    best_same = np.mean([np.nanargmax(ref[i]) == np.nanargmax(q[i]) for i in rows])
    # within-position differences (what the teacher decides on): option minus the position's mean
    dr = np.concatenate([ref[i][m[i]] - ref[i][m[i]].mean() for i in rows])
    dq = np.concatenate([q[i][m[i]] - q[i][m[i]].mean() for i in rows])
    out[f"leaf{d}"] = {"secs": round(t, 1), "corr_level": round(float(np.corrcoef(ref[m], q[m])[0, 1]), 3),
                       "mean_full": round(float(ref[m].mean()), 3), "mean_leaf": round(float(q[m].mean()), 3),
                       "corr_within": round(float(np.corrcoef(dr, dq)[0, 1]), 3),
                       "spread_full": round(float(dr.std()), 4), "spread_leaf": round(float(dq.std()), 4),
                       "best_same": round(float(best_same), 3)}
    print(d, out[f"leaf{d}"], flush=True)
print(json.dumps(out))
json.dump(out, open(os.path.join(ROOT, "results", "leaf_check.json"), "w"), indent=1)
