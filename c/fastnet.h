/* fastnet.h -- the rl-v3 net's forward pass for rollouts: same maths as rl_eval (relu(W1 x + b1) -> relu(W2 h + b2) ->
 * logits), but with transposed weights, skipping zero inputs and zero hidden units, and only the legal options' logits
 * (no value head). Summation order differs from rl_eval, so results agree to float rounding, not bit for bit. */
#ifndef BV_FASTNET_H
#define BV_FASTNET_H
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

/* y[o] = b[o] + sum over the n listed rows i (in order) of v[i] * W[idx[i] * nh + o], for all o < nh (a multiple of
 * 64): 64 outputs at a time held in 8 registers, so the sums are not stored and reloaded for every row. Each output
 * gets the same fused multiply-adds in the same order as the row-by-row loop it replaces (which the compiler also
 * contracted to FMA under -ffast-math): identical floats. */
static inline void gemv_rows(float *y, const float *b, const float *W, int nh, const int *idx, const float *v, int n) {
    for (int o = 0; o < nh; o += 64) {
        __m256 a0 = _mm256_load_ps(b + o), a1 = _mm256_load_ps(b + o + 8), a2 = _mm256_load_ps(b + o + 16),
               a3 = _mm256_load_ps(b + o + 24), a4 = _mm256_load_ps(b + o + 32), a5 = _mm256_load_ps(b + o + 40),
               a6 = _mm256_load_ps(b + o + 48), a7 = _mm256_load_ps(b + o + 56);
        for (int k = 0; k < n; k++) {
            const float *r = W + (size_t)idx[k] * nh + o;
            __m256 x = _mm256_set1_ps(v[k]);
            a0 = _mm256_fmadd_ps(x, _mm256_load_ps(r), a0);      a1 = _mm256_fmadd_ps(x, _mm256_load_ps(r + 8), a1);
            a2 = _mm256_fmadd_ps(x, _mm256_load_ps(r + 16), a2); a3 = _mm256_fmadd_ps(x, _mm256_load_ps(r + 24), a3);
            a4 = _mm256_fmadd_ps(x, _mm256_load_ps(r + 32), a4); a5 = _mm256_fmadd_ps(x, _mm256_load_ps(r + 40), a5);
            a6 = _mm256_fmadd_ps(x, _mm256_load_ps(r + 48), a6); a7 = _mm256_fmadd_ps(x, _mm256_load_ps(r + 56), a7);
        }
        _mm256_store_ps(y + o, a0); _mm256_store_ps(y + o + 8, a1); _mm256_store_ps(y + o + 16, a2);
        _mm256_store_ps(y + o + 24, a3); _mm256_store_ps(y + o + 32, a4); _mm256_store_ps(y + o + 40, a5);
        _mm256_store_ps(y + o + 48, a6); _mm256_store_ps(y + o + 56, a7);
    }
}

typedef struct { const float *src; int nh, nf; float *w1t, *w2t; } fnet;

static fnet g_fnets[8];
static int g_nfnets;

static const fnet *fnet_get(const float *prm, int nh, int nf) {
    for (int i = 0; i < g_nfnets; i++)
        if (g_fnets[i].src == prm && g_fnets[i].nh == nh && g_fnets[i].nf == nf) return &g_fnets[i];
    fnet *n = &g_fnets[g_nfnets < 8 ? g_nfnets++ : 7];
    free(n->w1t); free(n->w2t);
    n->src = prm; n->nh = nh; n->nf = nf;
    n->w1t = aligned_alloc(64, ((size_t)nf * nh * sizeof(float) + 63) / 64 * 64);
    n->w2t = aligned_alloc(64, ((size_t)nh * nh * sizeof(float) + 63) / 64 * 64);
    const float *w1 = prm, *w2 = prm + nh * nf + nh;
    for (int o = 0; o < nh; o++)
        for (int i = 0; i < nf; i++) n->w1t[(size_t)i * nh + o] = w1[(size_t)o * nf + i];
    for (int o = 0; o < nh; o++)
        for (int i = 0; i < nh; i++) n->w2t[(size_t)i * nh + o] = w2[(size_t)o * nh + i];
    return n;
}

/* lg[a] for ok[a] only (others untouched); na = number of options */
static void fnet_logits(const fnet *n, const float *x, const int *ok, int na, float *lg) {
    const int nh = n->nh, nf = n->nf;
    if (nh <= 0 || nh > 1024 || nf <= 0 || nf > 1024) return;   /* the buffers below hold 1024 */
    const float *prm = n->src, *b1 = prm + nh * nf, *b2 = b1 + nh + nh * nh, *wp = b2 + nh, *bp = wp + na * nh;
    float h1[1024] __attribute__((aligned(64))), h2[1024] __attribute__((aligned(64)));
    int idx[1024], m = 0;
    float val[1024] __attribute__((aligned(64))), bb[1024] __attribute__((aligned(64)));
    if (nh % 64 == 0) {                           /* blocked (the c2 shape: nh = 256) */
        for (int i = 0; i < nf; i++) if (x[i] != 0.f) { idx[m] = i; val[m++] = x[i]; }
        memcpy(bb, b1, (size_t)nh * sizeof(float));   /* aligned copy of the bias */
        gemv_rows(h1, bb, n->w1t, nh, idx, val, m);
        m = 0;
        for (int i = 0; i < nh; i++) if (h1[i] > 0.f) { idx[m] = i; val[m++] = h1[i]; }
        memcpy(bb, b2, (size_t)nh * sizeof(float));
        gemv_rows(h2, bb, n->w2t, nh, idx, val, m);
    } else {
        memcpy(h1, b1, nh * sizeof(float));
        for (int i = 0; i < nf; i++) {
            float xi = x[i];
            if (xi == 0.f) continue;
            const float *r = n->w1t + (size_t)i * nh;
            for (int o = 0; o < nh; o++) h1[o] += xi * r[o];
        }
        memcpy(h2, b2, nh * sizeof(float));
        for (int i = 0; i < nh; i++) {
            float v = h1[i];
            if (v <= 0.f) continue;
            const float *r = n->w2t + (size_t)i * nh;
            for (int o = 0; o < nh; o++) h2[o] += v * r[o];
        }
    }
    for (int o = 0; o < nh; o++) h2[o] = h2[o] < 0 ? 0 : h2[o];
    for (int a = 0; a < na; a++) {
        if (!ok[a]) continue;
        const float *r = wp + (size_t)a * nh;
        float s = bp[a];
        for (int i = 0; i < nh; i++) s += r[i] * h2[i];
        lg[a] = s;
    }
}
#endif
