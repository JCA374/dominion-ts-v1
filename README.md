# Dominion ts-v1: Teacher Search

**🌐 Live site: https://jca374.github.io/dominion-ts-v1/**

A stronger teacher for the Steam advisor within its 3-second budget. The teacher (from
[dominion-bv-v1](https://github.com/JCA374/dominion-bv-v1)) grades each buy and action play by playing every option out
with the RL-v3 c2 net on the same random deals of the hidden cards. This repo tests value-net leaves, successive
halving, several processes and the override gate, each as a paired match arm against frozen c2.

- `c/ts.c` — bv-v1's teacher + `ts_set_leaf` (value-net leaves) + `ts_set_alloc` (successive halving); with both off it
  plays bv-v1's games exactly (`experiments/equiv.py`).
- `match.py`, `queue.txt`, `experiments/arm.sh` — match arms (`results/match_<arm>.json[l]`).
- Submodules: `engine/` (rules), `rl-v3/` (c2's net code), `bv-v1/` (the original teacher and its Python side).

Clone with `--recursive`. Python: any venv with numpy (this repo uses rl-v3's).
