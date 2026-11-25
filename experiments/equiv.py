"""ts.so with both ts-v1 options off must play bv-v1's teacher exactly: the same games, scores and rollout counts as
bv-v1's own library (bv.so built from bv-v1/c/bv.c), teacher on buys and actions, public sampler.

    python experiments/equiv.py [--games 6]      (prints one line per game per library; run once per library)
"""
import argparse, json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=6)
ap.add_argument("--lib", default="ts", help="ts or bv")
ap.add_argument("--leaf", type=int, default=0)
ap.add_argument("--alloc", type=int, default=0)
a = ap.parse_args()
if a.lib == "bv":
    os.environ["BV_SO"] = os.path.join(ROOT, "build", "bv_ref.so")
    sys.path.insert(0, os.path.join(ROOT, "bv-v1"))
    from bv import core
    core.configure(k0=16, k1=16, margin=0.02, z=2, act=1, obs=1)
else:
    from ts import lib
    core = lib.core
    lib.configure(leaf=a.leaf, alloc=a.alloc, k0=16, k1=16, margin=0.02, z=2, act=1, obs=1)
rng = random.Random("ts-v1 equivalence")
for i in range(a.games):
    k, seed = core.random_kingdom(rng), rng.getrandbits(32)
    s0 = core.stats()
    r = core.play([core.seat_for("c2", k, 1), core.seat_for("c2", k)], k, seed, seed ^ 0x5EED)
    s1 = core.stats()
    print(json.dumps({"i": i, "scores": r["scores"], "turns": r["turns"],
                      "rollouts": s1["rollouts"] - s0["rollouts"], "overrides": s1["overrides"] - s0["overrides"]}))
