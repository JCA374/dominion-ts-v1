/* Standalone teacher benchmark: G games of c2 (teacher labels, mode 2) vs c2 on a fixed kingdom; prints rollouts/s.
 * cc -O3 -march=native -ffast-math -I ../engine/c bench.c ../engine/c/engine.c -lm -o bench && ./bench c2.f32 4 */
#include "../../c/ts.c"
#include <stdio.h>
#include <time.h>

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    static float prm[400000];
    size_t n = fread(prm, sizeof(float), 400000, f);
    fclose(f);
    int G = argc > 2 ? atoi(argv[2]) : 4;
    int nh = 256, nf = RL_NF;
    if (rl_num_params(nh, nf) != (int)n) { fprintf(stderr, "size %zu\n", n); return 1; }
    int kingdom[10] = {6, 7, 8, 20, 21, 11, 15, 19, 9, 28};
    int mode[2] = {RL_NET, RL_NET}, nhs[2] = {nh, nh}, nfs[2] = {nf, nf}, bvm[2] = {2, 0}, out[6];
    static float X[200 * RL_NF], Q[200 * RL_NA], SE[200 * RL_NA];
    static int INC[200], CH[200], K[200], SEAT[200], TURN[200];
    bv_set_cfg(16, 16, 1.5f, 5, 0);
    bv_set_fast(argc > 3 ? atoi(argv[3]) : 1);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long states = 0;
    double qsum = 0;
    for (int g = 0; g < G; g++) {
        int r = bv_play(mode, prm, prm, nhs, nfs, NULL, NULL, bvm, kingdom, 10, 1000 + g, 77 + g,
                        X, Q, SE, INC, CH, K, SEAT, TURN, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 200, out);
        states += r;
        for (int i = 0; i < r; i++) qsum += Q[i * RL_NA + INC[i]];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    long st[6];
    bv_stats(st);
    printf("%ld rollouts, %ld states in %.2f s: %.0f rollouts/s, %.2f states/s, checksum %.6f\n",
           st[0], states, s, st[0] / s, states / s, qsum);
    return 0;
}
