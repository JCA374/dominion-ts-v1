"""The decision bank: positions from c2-vs-c2 games (buys and action plays with a real choice), each graded with
2 x 1,024 rollouts to the end over EVERY legal option -- the "truth" a teacher setting is judged against (bank_eval.py).

    python experiments/bank.py --games 40 --n 240 --workers 3

Positions: c2 plays both seats (teacher label mode with 1 draw, only to list the decisions); a buy needs >= 3 legal
options, an action >= 2. Every regraded position must be the recorded one (same features, kind, incumbent), else the
bank fails. Out: data/bank/bank.npz + results/bank.json.
"""
import argparse, json, multiprocessing as mp, os, random, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
KT = 1024


def collect(i):
    from ts import lib
    core = lib.core
    lib.configure(k0=1, k1=1, act=1, obs=1, margin=0.02, z=2)
    rng = random.Random(f"ts-v1 bank {i}")
    k, seed = core.random_kingdom(rng), rng.getrandbits(32)
    seat = i % 2
    seats = [core.seat_for("c2", k, 2), core.seat_for("c2", k)]
    if seat:
        seats = seats[::-1]
    r = core.play(seats, k, seed, seed ^ 0x5EED)
    rec, out = r["rec"], []
    for j in range(len(rec["DIDX"])):
        nl = int(rec["LEGAL"][j].sum())
        kind = int(rec["KIND"][j])
        if nl >= (3 if kind == core.BUY else 2):
            out.append({"key": {"kingdom": k, "seed": seed, "seat": seat, "opp": "c2"}, "didx": int(rec["DIDX"][j]),
                        "kind": kind, "X": rec["X"][j], "INC": int(rec["INC"][j]), "TURN": int(rec["TURN"][j]),
                        "LEGAL": rec["LEGAL"][j]})
    return out


def truth(p):
    from ts import lib
    core = lib.core
    lib.configure(k0=KT, k1=KT, act=1, obs=1, margin=0.02, z=2)
    qs = []
    for s in (1, 2):
        g = core.regrade(p["key"], p["didx"], k=KT, seed=hash((p["key"]["seed"], p["didx"], p["kind"], s)) & 0xFFFFFFFF,
                         all_legal=True, kind=p["kind"])
        if g is None or not np.array_equal(g["X"], p["X"]) or int(g["INC"]) != p["INC"]:
            return {"bad": True}
        qs.append(g["Q"])
    return {"T": (qs[0] + qs[1]) / 2, "T1": qs[0], "T2": qs[1], "CAND": g["CAND"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        pos = [p for ps in pool.map(collect, range(a.games)) for p in ps]
        n_all = {k: sum(p["kind"] == k for p in pos) for k in (0, 1)}
        random.Random("ts-v1 bank sample").shuffle(pos)
        pos = pos[:a.n]
        print(f"{len(pos)} positions sampled ({n_all} with a choice) in {time.time() - t0:.0f} s", flush=True)
        res = []
        for i, r in enumerate(pool.imap(truth, pos)):
            res.append(r)
            if (i + 1) % 20 == 0:
                print(f"truth {i + 1}/{len(pos)} {time.time() - t0:.0f} s", flush=True)
    bad = sum(r.get("bad", False) for r in res)
    if bad:
        raise SystemExit(f"{bad} positions did not replay identically")
    os.makedirs(os.path.join(ROOT, "data", "bank"), exist_ok=True)
    T = np.array([r["T"] for r in res])
    d = np.array([np.nanmax(r["T1"]) - r["T1"][np.nanargmax(r["T2"])] for r in res])   # truth's own noise
    np.savez(os.path.join(ROOT, "data", "bank", "bank.npz"), T=T, T1=np.array([r["T1"] for r in res]),
             T2=np.array([r["T2"] for r in res]), CAND=np.array([r["CAND"] for r in res]),
             INC=np.array([p["INC"] for p in pos]), KIND=np.array([p["kind"] for p in pos]),
             TURN=np.array([p["TURN"] for p in pos]), DIDX=np.array([p["didx"] for p in pos]),
             KEYS=np.array([json.dumps(p["key"]) for p in pos]), X=np.array([p["X"] for p in pos]))
    inc_reg = [float(np.nanmax(t) - t[i]) for t, i in zip(T, [p["INC"] for p in pos])]
    s = {"positions": len(pos), "buys": int(sum(p["kind"] == 1 for p in pos)),
         "actions": int(sum(p["kind"] == 0 for p in pos)), "with_choice_in_games": n_all, "games": a.games,
         "truth_draws": 2 * KT, "c2_regret": round(float(np.mean(inc_reg)), 4),
         "c2_best_share": round(float(np.mean([r < 1e-9 for r in inc_reg])), 3),
         "truth_split_regret": round(float(np.mean(d)), 4), "secs": round(time.time() - t0),
         "summary": f"{len(pos)} positions; c2's own pick gives up {100 * np.mean(inc_reg):.2f} points of win chance on "
                    f"average against 2,048-draw truth"}
    json.dump(s, open(os.path.join(ROOT, "results", "bank.json"), "w"), indent=1)
    print(json.dumps(s))


if __name__ == "__main__":
    main()
