# CLAUDE.md

ts-v1: teacher search for the Steam advisor (memory note ts-v1). `c/ts.c` = bv-v1's bv.c + leaves + halving;
`ts/lib.py` builds ts.so and loads bv-v1's bv.core with BV_SO=ts.so.

- Python: `.venv` is a symlink to rl-v3's venv (`../rl-v3/.venv`).
- Arms: add a line to `queue.txt`, start with `experiments/arm.sh <name> <workers>` (unit ts-<name>); results in
  `results/match_<name>.json(l)`; `track.py` turns them into `runs/<name>/` for the site.
- Site: `publish_loop.sh` (every 15 min). Never put personal names or home paths on the site, in commits or podcasts.
- Don't change c/ts.c while an arm runs on it: match.py records the source hashes; build a copy (BV_SO) for tests.
