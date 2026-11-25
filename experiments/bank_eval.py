"""Teacher settings on the decision bank (bank.py): each position regraded with a setting (3 repetitions, fresh seeds),
the override gate applied for a grid of (margin, z), and the choice scored against the 2,048-draw truth:
regret = truth of the best legal option - truth of the choice (points of win chance given up). Lower is better;
c2's own pick (no teacher) is the reference. Time per question is measured, so settings compare at equal cost.

    python experiments/bank_eval.py --workers 3 [--configs full128 leaf4_640 ...]
Out: results/bank_eval.json (+ data/bank/eval_<config>.npz raw grades).
"""
import argparse, json, multiprocessing as mp, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CONFIGS = {   # name: (leaf, alloc, draws per option)
    "full32": (0, 0, 32), "full128": (0, 0, 128), "full320": (0, 0, 320),
    "full128_sh": (0, 1, 128),
    "leaf2_400": (2, 0, 400), "leaf2_1600": (2, 0, 1600),
    "leaf4_160": (4, 0, 160), "leaf4_640": (4, 0, 640), "leaf4_1600": (4, 0, 1600), "leaf4_640_sh": (4, 1, 640),
    "leaf8_320": (8, 0, 320), "leaf8_800": (8, 0, 800),
}
GATES = [(m, z) for m in (0.0, 0.01, 0.02, 0.04) for z in (0.0, 1.0, 2.0, 3.0)]
REPS = 3
B = {}


def init():
    B.update(np.load(os.path.join(ROOT, "data", "bank", "bank.npz")))


def job(args):
    from ts import lib
    core = lib.core
    i, name, rep = args
    leaf, alloc, k = CONFIGS[name]
    lib.configure(leaf=leaf, alloc=alloc, k0=k, k1=k, act=1, obs=1, margin=0.02, z=2)
    key = json.loads(str(B["KEYS"][i]))
    t = time.time()
    g = core.regrade(key, int(B["DIDX"][i]), k=k, seed=(7919 * i + 104729 * rep + 1) & 0xFFFFFFFF, all_legal=False,
                     kind=int(B["KIND"][i]))
    secs = time.time() - t
    if g is None or not np.array_equal(g["X"], B["X"][i]):
        return i, name, rep, None
    return i, name, rep, {"Q": g["Q"], "SE": g["SE"], "CAND": g["CAND"], "secs": secs}


def choose(Q, SE, CAND, inc, m, z):
    c = np.where(CAND.astype(bool))[0]
    b = c[np.argmax(Q[c])]
    return b if b != inc and Q[b] - Q[inc] > m and Q[b] - Q[inc] > z * SE[b] else inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    a = ap.parse_args()
    init()
    T, INC, KIND = B["T"], B["INC"], B["KIND"]
    n = len(T)
    best = np.nanmax(T, 1)
    inc_reg = best - T[np.arange(n), INC]
    out_p = os.path.join(ROOT, "results", "bank_eval.json")
    out = json.load(open(out_p)) if os.path.exists(out_p) else {}
    out["c2"] = {"regret": round(float(inc_reg.mean()), 5), "regret_buy": round(float(inc_reg[KIND == 1].mean()), 5),
                 "regret_action": round(float(inc_reg[KIND == 0].mean()), 5), "positions": n}
    with mp.Pool(a.workers, init) as pool:
        for name in a.configs:
            jobs = [(i, name, r) for i in range(n) for r in range(REPS)]
            grades = {}
            for i, nm, r, g in pool.imap_unordered(job, jobs, chunksize=4):
                grades[(i, r)] = g
            if any(g is None for g in grades.values()):
                raise SystemExit(f"{name}: a position did not replay identically")
            res = {"leaf": CONFIGS[name][0], "alloc": CONFIGS[name][1], "draws": CONFIGS[name][2],
                   "secs_per_question": round(float(np.mean([g["secs"] for g in grades.values()])), 3), "gates": {}}
            for m, z in GATES:
                reg = np.array([[best[i] - T[i, choose(grades[(i, r)]["Q"], grades[(i, r)]["SE"], grades[(i, r)]["CAND"],
                                                         INC[i], m, z)] for r in range(REPS)] for i in range(n)])
                ov = np.mean([[choose(grades[(i, r)]["Q"], grades[(i, r)]["SE"], grades[(i, r)]["CAND"], INC[i], m, z)
                               != INC[i] for r in range(REPS)] for i in range(n)])
                d = (reg.mean(1) - inc_reg)                     # per position: teacher minus c2
                rng = np.random.default_rng(0)
                bs = d[rng.integers(0, n, (2000, n))].mean(1)
                res["gates"][f"m{m}_z{z}"] = {"regret": round(float(reg.mean()), 5),
                                              "regret_buy": round(float(reg[KIND == 1].mean()), 5),
                                              "regret_action": round(float(reg[KIND == 0].mean()), 5),
                                              "gain_vs_c2": round(float(-d.mean()), 5),
                                              "gain_ci": [round(float(-np.percentile(bs, 97.5)), 5), round(float(-np.percentile(bs, 2.5)), 5)],
                                              "override_rate": round(float(ov), 4)}
            g0 = res["gates"]["m0.02_z2.0"]
            bestg = min(res["gates"].items(), key=lambda kv: kv[1]["regret"])
            res["best_gate"] = bestg[0]
            out[name] = res
            json.dump(out, open(out_p, "w"), indent=1)
            print(f"{name:>13}: {res['secs_per_question']:.2f} s/q, regret at m.02 z2 {100 * g0['regret']:.2f} "
                  f"(c2 {100 * inc_reg.mean():.2f}), best gate {bestg[0]} {100 * bestg[1]['regret']:.2f}", flush=True)
    s = {k: v for k, v in out.items() if k != "c2"}
    if s:
        top = min(s.items(), key=lambda kv: kv[1]["gates"][kv[1]["best_gate"]]["regret"])
        out["summary"] = (f"c2 alone gives up {100 * out['c2']['regret']:.2f} points per decision; best setting so far "
                          f"{top[0]} with gate {top[1]['best_gate']}: {100 * top[1]['gates'][top[1]['best_gate']]['regret']:.2f}")
        json.dump(out, open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main()
