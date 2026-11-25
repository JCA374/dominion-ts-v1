"""Paired comparison of match arms on the kingdoms both have completed (both seats): the difference in mean points per
kingdom, with a bootstrap 95% interval over kingdoms. Every arm against the first (the baseline, default A_full_k128).
Writes results/paired.json (the site's Results tab shows it).

    python experiments/compare.py [--base A_full_k128]
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import track  # noqa: E402


def per_kingdom(name):
    return {i: p for i, p, _ in track.pair_points(track.rows_of(name))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="A_full_k128")
    a = ap.parse_args()
    base = per_kingdom(a.base)
    out = {"base": a.base, "rows": []}
    for name, args, note in track.arms():
        if name == a.base:
            continue
        other = per_kingdom(name)
        common = sorted(set(base) & set(other))
        if len(common) < 10:
            continue
        d = np.array([other[i] - base[i] for i in common])
        rng = np.random.default_rng(0)
        b = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
        out["rows"].append({"arm": name, "kingdoms": len(common), "diff": round(float(d.mean()), 4),
                            "lo": round(float(np.percentile(b, 2.5)), 4), "hi": round(float(np.percentile(b, 97.5)), 4),
                            "arm_score": round(float(np.mean([other[i] for i in common])), 4),
                            "base_score": round(float(np.mean([base[i] for i in common])), 4)})
        r = out["rows"][-1]
        print(f"{name:>18} vs {a.base}: {100 * r['diff']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] "
              f"on {r['kingdoms']} kingdoms ({100 * r['arm_score']:.1f} vs {100 * r['base_score']:.1f})")
    json.dump(out, open(os.path.join(ROOT, "results", "paired.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
