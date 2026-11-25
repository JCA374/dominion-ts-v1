/* fastnet.h -- the rl-v3 net's forward pass for rollouts: same maths as rl_eval (relu(W1 x + b1) -> relu(W2 h + b2) ->
 * logits), but with transposed weights, skipping zero inputs and zero hidden units, and only the legal options' logits
 * (no value head). Summation order differs from rl_eval, so results agree to float rounding, not bit for bit. */
#ifndef BV_FASTNET_H
#define BV_FASTNET_H
#include <stdlib.h>
#include <string.h>

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
    const float *prm = n->src, *b1 = prm + nh * nf, *b2 = b1 + nh + nh * nh, *wp = b2 + nh, *bp = wp + na * nh;
    float h1[1024] __attribute__((aligned(64))), h2[1024] __attribute__((aligned(64)));
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
