"""A paired match: model A (c2 with the ts-v1 teacher) against an opponent, both seats on each kingdom, per-game results
saved. bv-v1's match.py with ts-v1's settings (--leaf, --alloc) and its library (ts.so).

    python match.py --name k128 --act 1 --k0 128 --k1 128 --margin 0.02 --z 2 --obs 1 --opp c2 --kingdoms 300 [--leaf 4] [--alloc 1]

Kingdoms come from a seeded stream per --stream (gate / dev / reserved), so different runs play the same games, or
from a frozen manifest file (--games file.json: {"games": [{"i", "kingdom", "seed"}, ...]}, both seats each).
The manifest (settings, model, source and library hashes, the game list's hash) is written to results/match_<name>.json
before the first game; --resume refuses a different one and checks every kept row against its game (kingdom, seed,
seat). Scores and intervals count completed seat pairs only.
Out: results/match_<name>.json (summary) + results/match_<name>.jsonl (games).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
from ts import lib as _ts  # noqa: E402  (builds ts.so, puts bv-v1 on the path)
_A: dict = {}


def init(a):
    from ts import lib
    _A.update(a)
    st = np.load(a["student"]).astype(np.float32) if a["student"] else None
    lib.configure(leaf=a["leaf"], alloc=a["alloc"], sub=a["sub"], k0=a["k0"], k1=a["k1"], se_mult=a["se"], topn=a["topn"],
                  margin=a["margin"], student=st, z=a["z"], act=a["act"], obs=a["obs"], kval=a["kval"])


def job(g):
    from bv import core
    k, s = g["kingdom"], g["seat"]
    me = core.seat_for(_A["base"], k, _A["bv"])
    op = core.seat_for(_A["opp"], k)
    t = time.time()
    before = core.stats()
    r = core.play([me, op] if s == 0 else [op, me], k, g["seed"], g["seed"] ^ 0x5EED)
    after = core.stats()
    import ctypes
    acts = (ctypes.c_long * 5)()
    core.lib.bv_act_stats(acts)
    act = list(acts)
    prev = _A.setdefault("_act", [0] * 5)
    dact = [x - y for x, y in zip(act, prev)]
    _A["_act"] = act
    from ts import lib as _l
    sub = _l.sub_stats()
    sprev = _A.setdefault("_sub", {k: 0 for k in sub})
    dsub = {k: sub[k] - sprev[k] for k in sub}
    _A["_sub"] = sub
    return {**g, "sub_graded": dsub["graded"], "sub_overrides": dsub["overrides"], "sub_rejected": dsub["rejected"],
            "act_graded": dact[0], "act_overrides": dact[1], "act_play_not_stop": dact[2],
            "act_stop_not_play": dact[3], "act_other_action": dact[4],
            "lat": [round(float(x), 5) for x in core.latencies()],
            "act_log": [[round(float(v), 4) for v in row] for row in core.action_log()], "points": core.points(r, s), "secs": round(time.time() - t, 2), "capped": r["capped"],
            "error": r["error"], **{x: after[x] - before[x] for x in ("rollouts", "overrides", "buys")}}


def games(stream, n, skip=0, path=""):
    if path:                                       # a frozen manifest: its games in order, both seats
        out = []
        for g in json.load(open(path))["games"][skip:skip + n]:
            out += [{"i": g["i"], "kingdom": g["kingdom"], "seed": g["seed"], "seat": s} for s in (0, 1)]
        return out
    rng = random.Random(f"bv-v1 {stream} kingdoms")
    from bv.core import random_kingdom
    out = []
    for i in range(skip + n):
        k, seed = random_kingdom(rng), rng.getrandbits(32)
        if i >= skip:
            out += [{"i": i, "kingdom": k, "seed": seed, "seat": 0}, {"i": i, "kingdom": k, "seed": seed, "seat": 1}]
    return out


STREAMS = {"gate": 200, "dev": 300, "dev2": 300, "reserved": 2000, "confirm": 2000, "suite": 300}


def eval_kingdoms():
    """Canonical keys of every kingdom any evaluation stream can deal (collect.py never trains on them)."""
    from bv.core import random_kingdom
    out = set()
    for st, n in STREAMS.items():
        rng = random.Random(f"bv-v1 {st} kingdoms")
        for _ in range(n):
            out.add(",".join(sorted(random_kingdom(rng))))
            rng.getrandbits(32)
    return out


def summary(res):
    by = {}
    for r in res:
        by.setdefault(r["i"], {})[r["seat"]] = r
    pairs = [v for v in by.values() if len(v) == 2]           # completed seat pairs only
    res_all, res = res, [r for v in pairs for r in v.values()]
    blocks = np.array([np.mean([r["points"] for r in v.values()]) for v in pairs])
    rng = np.random.default_rng(0)
    boot = [blocks[rng.integers(0, len(blocks), len(blocks))].mean() for _ in range(2000)] if len(blocks) else [0]
    m = float(np.mean([r["points"] for r in res])) if res else 0.0
    b = sum(r["buys"] for r in res)
    lat = np.array([x for r in res for x in r.get("lat", [])])
    alog = [row for r in res for row in r.get("act_log", [])]
    from bv.core import NAME
    nm = lambda a: "end actions" if int(a) == 33 else NAME[int(a)]  # noqa: E731
    changes = {}
    for row in alog:
        kk = f"{nm(row[1])} -> {nm(row[2])}"
        changes[kk] = changes.get(kk, 0) + 1
    return {"latency_s": ({"decisions": int(len(lat)), "p50": round(float(np.percentile(lat, 50)), 4),
                           "p95": round(float(np.percentile(lat, 95)), 4)} if len(lat) else None),
            "action_overrides": ({"n": len(alog), "mean_gain": round(float(np.mean([r[3] for r in alog])), 4),
                                  "top_changes": sorted(changes.items(), key=lambda kv: -kv[1])[:12]} if alog else None),"games": len(res), "score": round(m, 4), "ci95": [round(float(np.percentile(boot, 2.5)), 4),
            round(float(np.percentile(boot, 97.5)), 4)], "override_rate": round(sum(r["overrides"] for r in res) / max(b, 1), 4),
            "secs_per_game": round(float(np.mean([r["secs"] for r in res])), 2) if res else 0,
            "rollouts": sum(r["rollouts"] for r in res), "capped": sum(r["capped"] for r in res),
            "errors": sum(r["error"] for r in res), "pairs": len(pairs), "unpaired_games": len(res_all) - len(res),
            "buy_overrides": sum(r["overrides"] for r in res), "buys": b,
            "act_graded": sum(r.get("act_graded", 0) for r in res), "act_overrides": sum(r.get("act_overrides", 0) for r in res),
            "act_override_rate": round(sum(r.get("act_overrides", 0) for r in res) / max(sum(r.get("act_graded", 0) for r in res), 1), 4),
            "score_uncapped_pairs": (round(float(np.mean([np.mean([r["points"] for r in v.values()]) for v in pairs
                                                          if not any(r["capped"] for r in v.values())])), 4)
                                     if any(not any(r["capped"] for r in v.values()) for v in pairs) else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="c2")
    ap.add_argument("--bv", type=int, default=1)
    ap.add_argument("--opp", default="c2")
    ap.add_argument("--k0", type=int, default=16)
    ap.add_argument("--k1", type=int, default=64)
    ap.add_argument("--se", type=float, default=1.5)
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--student", default="")
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--act", type=int, default=0, help="1: also grade actions, 2: actions only")
    ap.add_argument("--resume", action="store_true", help="keep the games already in the .jsonl")
    ap.add_argument("--stream", default="gate")
    ap.add_argument("--games", default="", help="a frozen game manifest (JSON) instead of a stream")
    ap.add_argument("--obs", type=int, default=0, help="sampler: 0 simulator, 1 public")
    ap.add_argument("--kval", type=int, default=0, help="validation draws for an override")
    ap.add_argument("--leaf", type=int, default=0, help="turns before a value-net leaf (0: rollouts to the end)")
    ap.add_argument("--alloc", type=int, default=0, help="0 uniform draws, 1 successive halving (same budget)")
    ap.add_argument("--sub", type=int, default=0, help="1: also grade card choices inside my actions (gains, trashes)")
    ap.add_argument("--kingdoms", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    a = vars(ap.parse_args())
    gs = games(a["stream"], a["kingdoms"], path=a["games"])
    out = os.path.join(ROOT, "results", f"match_{a['name']}")
    from bv import core as _c
    import hashlib
    manifest = {k: v for k, v in a.items() if k not in ("workers", "resume")}
    manifest["models"] = {m: _c.sha(m) for m in ("rl3_c2_2654.npy", "rl3_c1_2300.npy", "rl1_r2_2200.npy")}
    manifest["seat_sha"] = {x: hashlib.sha256(open(_c.net_file(a[x]), "rb").read()).hexdigest()[:16]
                            for x in ("base", "opp") if a[x] not in ("ga", "ga2")}
    manifest["student_sha"] = (hashlib.sha256(open(a["student"], "rb").read()).hexdigest()[:16]
                               if a["student"] else None)
    manifest["sources"] = {os.path.relpath(d, ROOT): hashlib.sha256(open(d, "rb").read()).hexdigest()[:16]
                           for d in _ts.DEPS}
    manifest["library"] = {"path": os.path.basename(_c.SO), "sha": hashlib.sha256(open(_c.SO, "rb").read()).hexdigest()[:16]}
    manifest["games_sha"] = hashlib.sha256(json.dumps([[g["i"], g["kingdom"], g["seed"], g["seat"]] for g in gs])
                                           .encode()).hexdigest()[:16]
    res = []
    if a["resume"] and os.path.exists(out + ".json"):
        old = json.load(open(out + ".json")).get("manifest")
        if old is not None and old != manifest:          # never mix games of different settings
            diff = sorted(k for k in set(old) | set(manifest) if old.get(k) != manifest.get(k))
            raise SystemExit(f"resume refused: manifest differs from {out}.json in {diff}")
    if a["resume"] and os.path.exists(out + ".jsonl"):
        res = [json.loads(line) for line in open(out + ".jsonl")]
        want = {(g["i"], g["seat"]): g for g in gs}
        for r in res:                                    # every kept row must be the game it claims to be
            g = want.get((r["i"], r["seat"]))
            if g is None or g["kingdom"] != r["kingdom"] or g["seed"] != r["seed"]:
                raise SystemExit(f"resume refused: kept row {r['i']}/{r['seat']} is not that game of the manifest")
        have = {(r["i"], r["seat"]) for r in res}
        gs = [g for g in gs if (g["i"], g["seat"]) not in have]
    elif not a["resume"] or not os.path.exists(out + ".json"):     # the manifest exists before game one
        json.dump({"name": a["name"], "args": a, "manifest": manifest, "t": time.strftime("%Y-%m-%d %H:%M"),
                   "done": False, "games": 0}, open(out + ".json", "w"), indent=1)
    t = time.time()
    n0 = len(res)
    with mp.Pool(a["workers"], init, (a,)) as pool, open(out + ".jsonl", "a" if n0 else "w") as f:
        for i, r in enumerate(pool.imap_unordered(job, gs), start=n0):
            res.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 50 == 0 or i + 1 == n0 + len(gs):
                s = summary(res)
                json.dump({"name": a["name"], "args": a, "manifest": manifest, "t": time.strftime("%Y-%m-%d %H:%M"), "done": i + 1 == n0 + len(gs),
                           "wall_secs": round(time.time() - t), **s}, open(out + ".json", "w"), indent=1)
                print(f"{a['name']}: {i + 1}/{n0 + len(gs)} score {100 * s['score']:.1f}% ci {s['ci95']} "
                      f"override {s['override_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
