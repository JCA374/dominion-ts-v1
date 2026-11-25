# Rollout speed (ts-v1)

`bench.c` (from bv-v1's bench): G teacher-label games of c2 vs c2 on a fixed kingdom; prints rollouts/s and a
checksum of every graded value (exactness). Build: `gcc -O3 -march=native -ffast-math -fno-finite-math-only -I
../../engine/c -I ../../c bench.c ../../engine/c/engine.c -lm -o bench && ./bench c2.f32 2` (c2.f32 = the raw
float32 weights of RL-v3 c2). Profile: add `-pg -fno-inline-functions`, then `gprof -b -p`.
