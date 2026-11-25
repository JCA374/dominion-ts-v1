/*
 * bv.c -- the buy-value teacher: at a buy decision, every candidate buy (and PASS) is played out to the end of the
 * game on the same K hidden-state draws (paired), with frozen policies continuing for both players. The mean final
 * score (win 1, tie 1/2, loss 0) per candidate is the label; the teacher can also play its best candidate.
 *
 * ts-v1 (teacher search): this is bv-v1's c/bv.c with two search options added, off by default (then every result is
 * bv-v1's): value-net leaves (ts_set_leaf: rollouts stop D turns after the decision and score the position with the
 * rollout net's value) and successive halving (ts_set_alloc 1: the same rollout budget, spent on the leaders).
 *
 * Built into ts.so with engine/c/engine.c. Includes rl-v3's rl.c (its net, features and seats; which includes
 * az-v1's az.c and dominion-v1's ga_agent.c). Adds no rules: only de_hide + de_resume_buy on copies.
 */
#include "../rl-v3/c/rl.c"
#include "fastnet.h"
#include <time.h>

/* ── the student (Deep Sets over card tokens), forward pass; see bv/student.py for the layout ── */
#define BV_TOK 17           /* per-card token features (bv_tokens) */
#define BV_GLB 16           /* global features */
#define BV_W 64             /* width */
#define BV_EMB 8            /* learned card-id embedding */

int bv_tok(void) { return BV_TOK; }
int bv_glb(void) { return BV_GLB; }

/* Tokens read off the rl features (x = rl_features, RL_NF) plus the card rules. Row c, columns:
 * 0 own, 1 own draw pile, 2 opp owns, 3 supply left, 4 in game, 5 in my hand, 6 in my play, 7 cost/8, 8 +cards/3,
 * 9 +actions/2, 10 +coins/3, 11 +buys, 12 action, 13 attack, 14 victory, 15 treasure, 16 c2's probability of
 * buying it (softmax of c2's logits over the candidates; 0 if not a candidate).  Global: coins, buys, my turns, opp
 * turns, vp, opp vp, vp diff, provinces, empty piles, seat, actions left, c2's PASS probability, then 4 zero. */
void bv_tokens(const float *x, const float *pol, float *tok, float *glb) {
    const int N = DE_NUM_CARDS;
    for (int c = 0; c < N; c++) {
        float *t = tok + c * BV_TOK;
        const de_card *d = &DE_CARDS[c];
        t[0] = x[c]; t[1] = x[N + c]; t[2] = x[2 * N + c]; t[3] = x[3 * N + c]; t[4] = x[4 * N + c];
        t[5] = x[AZ_NF + c]; t[6] = x[AZ_NF + N + c];
        t[7] = d->cost / 8.0f; t[8] = d->cards / 3.0f; t[9] = d->actions / 2.0f; t[10] = d->coins / 3.0f;
        t[11] = (float)d->buys; t[12] = (d->types & DE_ACTION) != 0; t[13] = (d->types & DE_ATTACK) != 0;
        t[14] = (d->types & DE_VICTORY) != 0; t[15] = (d->types & DE_TREASURE) != 0;
        t[16] = pol[c];
    }
    for (int i = 0; i < 10; i++) glb[i] = x[5 * N + i];
    glb[10] = x[AZ_NF + 2 * N];
    glb[11] = pol[RL_PASS];
    for (int i = 12; i < BV_GLB; i++) glb[i] = 0;
}

static void lin(const float *w, const float *b, const float *x, int nin, int nout, float *y, int relu) {
    for (int o = 0; o < nout; o++) {
        float s = b[o];
        const float *r = w + (size_t)o * nin;
        for (int i = 0; i < nin; i++) s += r[i] * x[i];
        y[o] = relu && s < 0 ? 0 : s;
    }
}

/* q[RL_NA]: sigmoid action values (a card, RL_PASS). Layout (row-major torch Linear weights, then bias):
 * emb[N][BV_EMB]; g0: (BV_TOK+BV_EMB+BV_GLB)->W; for r in 0,1: u_r: (W + 2W) -> W  (token, mean pool, max pool);
 * head: (W + 2W + BV_GLB) -> W -> 1 per card; pass: (2W + BV_GLB) -> W -> 1. */
void bv_student(const float *prm, const float *x, const float *pol, float *q) {
    const int N = DE_NUM_CARDS, W = BV_W;
    float tok[DE_NUM_CARDS * BV_TOK], glb[BV_GLB], h[DE_NUM_CARDS][BV_W], h2[DE_NUM_CARDS][BV_W];
    float in[BV_TOK + BV_EMB + BV_GLB + 3 * BV_W], pool[2 * BV_W], hid[BV_W];
    bv_tokens(x, pol, tok, glb);
    const float *emb = prm, *p = emb + N * BV_EMB;
    const int n0 = BV_TOK + BV_EMB + BV_GLB;
    for (int c = 0; c < N; c++) {
        memcpy(in, tok + c * BV_TOK, BV_TOK * sizeof(float));
        memcpy(in + BV_TOK, emb + c * BV_EMB, BV_EMB * sizeof(float));
        memcpy(in + BV_TOK + BV_EMB, glb, BV_GLB * sizeof(float));
        lin(p, p + W * n0, in, n0, W, h[c], 1);
    }
    p += W * n0 + W;
    for (int r = 0; r < 2; r++) {
        int cnt = 0;
        for (int j = 0; j < W; j++) { pool[j] = 0; pool[W + j] = 0; }
        for (int c = 0; c < N; c++) {
            if (x[4 * N + c] <= 0) continue;          /* pool over the cards in this game */
            cnt++;
            for (int j = 0; j < W; j++) { pool[j] += h[c][j]; if (h[c][j] > pool[W + j]) pool[W + j] = h[c][j]; }
        }
        for (int j = 0; j < W; j++) pool[j] /= cnt ? cnt : 1;
        for (int c = 0; c < N; c++) {
            memcpy(in, h[c], W * sizeof(float));
            memcpy(in + W, pool, 2 * W * sizeof(float));
            lin(p, p + W * 3 * W, in, 3 * W, W, h2[c], 1);
            for (int j = 0; j < W; j++) h2[c][j] += h[c][j];   /* residual */
        }
        memcpy(h, h2, sizeof h);
        p += W * 3 * W + W;
    }
    int cnt = 0;
    for (int j = 0; j < W; j++) { pool[j] = 0; pool[W + j] = 0; }
    for (int c = 0; c < N; c++) {
        if (x[4 * N + c] <= 0) continue;
        cnt++;
        for (int j = 0; j < W; j++) { pool[j] += h[c][j]; if (h[c][j] > pool[W + j]) pool[W + j] = h[c][j]; }
    }
    for (int j = 0; j < W; j++) pool[j] /= cnt ? cnt : 1;
    const int nh = 3 * W + BV_GLB;
    const float *hw1 = p, *hb1 = hw1 + W * nh, *hw2 = hb1 + W, *hb2 = hw2 + W;
    p = hb2 + 1;
    for (int c = 0; c < N; c++) {
        memcpy(in, h[c], W * sizeof(float));
        memcpy(in + W, pool, 2 * W * sizeof(float));
        memcpy(in + 3 * W, glb, BV_GLB * sizeof(float));
        lin(hw1, hb1, in, nh, W, hid, 1);
        float s;
        lin(hw2, hb2, hid, W, 1, &s, 0);
        q[c] = 1.0f / (1.0f + expf(-s));
    }
    const int np = 2 * W + BV_GLB;
    memcpy(in, pool, 2 * W * sizeof(float));
    memcpy(in + 2 * W, glb, BV_GLB * sizeof(float));
    lin(p, p + W * np, in, np, W, hid, 1);
    p += W * np + W;
    float s;
    lin(p, p + W, hid, W, 1, &s, 0);
    q[RL_PASS] = 1.0f / (1.0f + expf(-s));
}

int bv_student_params(void) {
    const int W = BV_W, n0 = BV_TOK + BV_EMB + BV_GLB, nh = 3 * W + BV_GLB, np = 2 * W + BV_GLB;
    return DE_NUM_CARDS * BV_EMB + (W * n0 + W) + 2 * (W * 3 * W + W) + (W * nh + W + W + 1) + (W * np + W + W + 1);
}

/* ── the teacher ── */
typedef struct {
    int k0, k1;                 /* draws: first k0, then up to k1 while the top two are within se_mult SE */
    float se_mult;
    int topn;                   /* candidates: the net's top-n legal buys (+ PASS); all if legal <= topn + 3 */
    float margin;               /* teacher play: override the incumbent only if the paired gain > margin */
} bv_cfg;
static bv_cfg g_cfg = {16, 64, 1.5f, 5, 0.0f};
void bv_set_cfg(int k0, int k1, float se_mult, int topn, float margin) {
    g_cfg = (bv_cfg){k0, k1, se_mult, topn, margin};
}

/* per seat: 0 plain net, 1 teacher plays (margin), 2 teacher labels only (incumbent plays), 3 student plays
 * (margin vs incumbent), 4 student plays + teacher labels (DAgger) */
static int g_bv_mode[2];
static float g_z;               /* teacher play: the gain must also exceed g_z paired SEs */
void bv_set_z(float z) { g_z = z; }
static const float *g_student;
static long g_rollouts, g_capped, g_errors, g_graded, g_overrides, g_buys;
void bv_set_student(const float *prm) { g_student = prm; }
void bv_stats(long *o) {
    o[0] = g_rollouts; o[1] = g_capped; o[2] = g_errors; o[3] = g_graded; o[4] = g_overrides; o[5] = g_buys;
}

/* label records */
typedef struct {
    float *X, *Q, *SE;          /* features (RL_NF); per option mean score (NAN = not a candidate); SE of the
                                 * paired difference to the incumbent */
    int *INC, *CHOSE, *K, *SEAT, *TURN;
    unsigned char *CAND;        /* explicit candidate mask (never infer it from NaN: -ffast-math may drop isnan) */
    float *POL;                 /* c2's prior over the candidates, exactly as the live student reads it */
    int *DIDX;                  /* replay key: index of this graded decision in its game (bv_set_regrade) */
    unsigned char *CAPS;        /* per option: rollouts that hit the round cap */
    unsigned char *LEGAL;       /* per option: legal at this decision */
    int *KIND;                  /* DE_BUY or DE_PLAY_ACTION */
    float *VG, *VSE;            /* validation: the graded winner's paired gain over the incumbent on independent draws,
                                 * its SE (0, 0 when the winner is the incumbent or validation is off) */
    int *VK;                    /* validation draws (0 = none) */
    int *KN;                    /* my known top cards at this decision (the sampler kept them in place) */
    int max, n, overflow;
} bv_rec;

/* Regrade one recorded decision (replay): with target >= 0, bv_play plays the same game (labels-only mode 2 plays c2's
 * buys, so the game does not depend on grading), grades only decision number `target`, with k draws from `seed`, and
 * all legal buys as candidates if all = 1. Off with target -1. */
static int g_rg_target = -1, g_rg_k, g_rg_all, g_didx, g_aidx, g_rg_kind = DE_BUY;
static uint64_t g_rg_seed;
static long g_rejected;
void bv_set_regrade(int target, int k, uint64_t seed, int all) { g_rg_target = target; g_rg_k = k; g_rg_seed = seed; g_rg_all = all; }
/* which decision kind bv_set_regrade's target counts (DE_BUY, the default, or DE_PLAY_ACTION) */
void bv_set_regrade_kind(int kind) { g_rg_kind = kind; }
long bv_rejected(void) { return g_rejected; }

/* per-decision latency (seconds) of graded / student decisions, read and reset by bv_lat */
static float g_lat[8192];
static int g_nlat;
int bv_lat(float *out, int max) {
    int n = g_nlat < max ? g_nlat : max;
    memcpy(out, g_lat, n * sizeof(float));
    g_nlat = 0;
    return n;
}
static double bv_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static bv_rec *g_bvrec;

/* Known top cards. The draw pile is a shuffled block, and cards put on top later (Harbinger, Artisan, Sentry,
 * Bureaucrat's Silver) are known to their owner until drawn. Tracked from the live game at every decision: u = the
 * unknown part of the pile (its bottom); a shuffle (the player's rng moves) makes it all unknown, a draw takes the top
 * first, so u = min(u, deck_n) between decisions and the rest is known. Every top-deck move asks a decision before it
 * (or has no draw between the move and the next decision), so counting at decisions is exact for this card set.
 * Only the owner's known cards are kept; the opponent's (Bureaucrat reveals) are still sampled. */
static int g_kn_on, g_kn_u[2], g_known;        /* g_known: known top cards of the player being graded */
static uint64_t g_kn_rng[2];
static long g_kn_stats[2];                      /* decisions graded, of which with known top cards */
void bv_known_stats(long *o) { o[0] = g_kn_stats[0]; o[1] = g_kn_stats[1]; }
static void kn_observe(const de_game *g) {
    for (int i = 0; i < 2; i++) {
        const de_player *p = &g->p[i];
        if (p->rng != g_kn_rng[i]) { g_kn_u[i] = p->deck_n; g_kn_rng[i] = p->rng; }
        else if (p->deck_n < g_kn_u[i]) g_kn_u[i] = p->deck_n;
    }
}
static int kn_known(const de_game *g, int me) {
    int k = g_kn_on ? g->p[me].deck_n - g_kn_u[me] : 0;
    return k > 0 ? k : 0;
}

/* Public top cards. Some cards put on a draw pile are seen by both players: Bureaucrat's Silver (gained onto the
 * attacker's deck) and the Victory card a Bureaucrat victim puts back. Harbinger, Artisan and Sentry put cards back
 * that only their owner sees. Per player and deck position, g_pub says whether everyone knows the card there. Tracked
 * at every decision of the live game: a shuffle (the player's rng moves) makes the pile unknown, a draw takes the top,
 * and growth since the last decision is private exactly when that decision was the owner's Harbinger / Artisan topdeck
 * or Sentry choice (each such move asks a decision right before it); otherwise it came from a Bureaucrat (public). */
static unsigned char g_pub[2][DE_ZONE];
static int g_pub_n[2], g_last_kind = -1, g_last_player = -1;
static uint64_t g_pub_rng[2];
static long g_pub_stats[2];                     /* graded decisions, of which the opponent had public top cards */
void bv_pub_stats(long *o) { o[0] = g_pub_stats[0]; o[1] = g_pub_stats[1]; }
static void pub_observe(const de_game *g) {
    for (int i = 0; i < 2; i++) {
        const de_player *p = &g->p[i];
        if (p->rng != g_pub_rng[i]) { memset(g_pub[i], 0, DE_ZONE); g_pub_rng[i] = p->rng; }
        else if (p->deck_n > g_pub_n[i]) {
            int priv = g_last_player == i && (g_last_kind == DE_HARBINGER_TOPDECK || g_last_kind == DE_ARTISAN_TOPDECK ||
                                              g_last_kind == DE_SENTRY_CARD || g_last_kind == DE_SENTRY_TOP);
            for (int k = g_pub_n[i]; k < p->deck_n; k++) g_pub[i][k] = !priv;
        }
        g_pub_n[i] = p->deck_n;
    }
}
static int pub_count(const de_game *g, int i) {
    int n = 0;
    for (int k = 0; k < g->p[i].deck_n; k++) n += g_pub[i][k];
    return n;
}

/* The sampler's view of the opponent. 0 "simulator": the opponent's discard pile and the multiset of its draw pile +
 * hand are kept (more than a player can know: the cards it discarded from hand at cleanup were never shown). 1
 * "public": only what a player can reconstruct (the Steam autopilot tracks the opponent's owned cards and the pile
 * sizes): the opponent's draw pile, hand and discard pile are pooled and dealt again at their sizes, except its
 * public top cards (g_pub), which stay in place. My own zones are the same in both: I know my discard pile and my draw
 * pile's contents, and my known top cards (g_known) stay in place. */
static int g_obs;
void bv_set_obs(int obs) { g_obs = obs; }

/* hidden zones in a canonical order, so equal observations give equal samples; known top cards stay in place */
static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }
static void canon(de_game *c, int me) {
    for (int i = 0; i < c->nplayers; i++) {
        de_player *p = &c->p[i];
        if (i == me) { qsort(p->deck, p->deck_n - g_known, sizeof(int), cmp_int); continue; }
        int all[3 * DE_ZONE], n = 0;
        for (int k = 0; k < p->deck_n; k++) if (!(g_obs && g_pub[i][k])) all[n++] = p->deck[k];
        for (int k = 0; k < p->hand_n; k++) all[n++] = p->hand[k];
        if (g_obs) for (int k = 0; k < p->discard_n; k++) all[n++] = p->discard[k];
        qsort(all, n, sizeof(int), cmp_int);
        int j = 0;
        for (int k = 0; k < p->deck_n; k++) if (!(g_obs && g_pub[i][k])) p->deck[k] = all[j++];
        for (int k = 0; k < p->hand_n; k++) p->hand[k] = all[j++];
        if (g_obs) for (int k = 0; k < p->discard_n; k++) p->discard[k] = all[j++];
    }
}

static void bv_shuffle(int *a, int n, uint64_t *rng) {     /* = engine.c's shuffle */
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(de_rng_next(rng) % (uint64_t)(i + 1));
        int t = a[i]; a[i] = a[j]; a[j] = t;
    }
}
/* de_hide, except that my g_known top cards keep their place (identical to de_hide, bit for bit, when g_known = 0 and
 * g_obs = 0); with g_obs = 1 the opponent's hidden cards (all but its public top cards) are dealt again over its draw
 * pile, hand and discard pile */
static void bv_hide(de_game *g, int me, uint64_t seed) {
    if (!g_known && !g_obs) { de_hide(g, me, seed); return; }
    uint64_t master = seed ? seed : 1;
    for (int i = 0; i < g->nplayers; i++) {
        de_player *p = &g->p[i];
        p->rng = de_rng_next(&master) | 1;
        if (i != me && g_obs) {
            int all[3 * DE_ZONE], n = 0;
            for (int k = 0; k < p->deck_n; k++) if (!g_pub[i][k]) all[n++] = p->deck[k];
            for (int k = 0; k < p->hand_n; k++) all[n++] = p->hand[k];
            for (int k = 0; k < p->discard_n; k++) all[n++] = p->discard[k];
            bv_shuffle(all, n, &p->rng);
            int j = 0;
            for (int k = 0; k < p->deck_n; k++) if (!g_pub[i][k]) p->deck[k] = all[j++];
            for (int k = 0; k < p->hand_n; k++) p->hand[k] = all[j++];
            for (int k = 0; k < p->discard_n; k++) p->discard[k] = all[j++];
        } else if (i != me) {
            int h = p->hand_n;
            memcpy(p->deck + p->deck_n, p->hand, (size_t)h * sizeof(int));
            p->deck_n += h;
            p->hand_n = 0;
            bv_shuffle(p->deck, p->deck_n, &p->rng);
            for (int k = 0; k < h; k++) p->hand[p->hand_n++] = p->deck[--p->deck_n];   /* = draw: deck >= h */
        } else {
            bv_shuffle(p->deck, p->deck_n - g_known, &p->rng);
        }
    }
}

static rl_seat g_cont[2];       /* the continuation policies of this game (greedy, no records) */

/* Opponent mixture for the rollouts: with g_nmix > 0, draw k continues the opponent with g_mix[k % g_nmix] instead of
 * the actual opponent's policy (a check that an override does not depend on knowing who the opponent is). */
static rl_seat g_mix[4];
static int g_nmix;
void bv_set_mix(int n, const int *mode, const float **prm, const int *nh, const int *nf, const int **genome) {
    g_nmix = n < 4 ? n : 4;
    for (int i = 0; i < g_nmix; i++) {
        memset(&g_mix[i], 0, sizeof g_mix[i]);
        g_mix[i].mode = mode[i]; g_mix[i].prm = prm[i]; g_mix[i].nh = nh[i]; g_mix[i].nf = nf[i];
        g_mix[i].genome = genome[i]; g_mix[i].temp = 1;
    }
}

/* Fast continuation agent: a NET seat's greedy choice with fastnet (sparse, legal logits only); same decision as
 * rl_decide's greedy path inside a rollout (no records, no planner, no card school, no chain rule), up to float
 * rounding in the logits. GA seats go to rl_decide. bv_set_fast: 0 = rl_decide everywhere (reference),
 * 1 = fast in rollouts (default), 2 = fast in the main game as well (equivalence tests). */
static int g_fast = 1;
void bv_set_fast(int f) { g_fast = f; }
/* ts-v1 card choices (ts_set_sub 1): a forced sequence for the grading player's first decisions in a rollout -- my
 * choices since the action was played, then the candidate. A rollout that asks anything else of me first (the replay
 * diverged) is an error: the decision is not graded. */
#define TS_HIST 64
static int g_fs_on, g_fs_me, g_fs_n, g_fs_i, g_fs_bad;
static int g_fs_kind[TS_HIST + 1], g_fs_opt[TS_HIST + 1];

/* ts-v1 leaves: D > 0 = a rollout stops at its first decision D turns (both players' turns counted) after the graded
 * one and scores (value + 1) / 2 of the rollout net, from the grading player's side; games that end sooner score
 * their result. 0 = to the end (bv-v1). Needs the fast continuation (g_fast >= 1) and net seats. */
static int g_leaf_d, g_leaf_on, g_leaf_hit, g_leaf_t0, g_leaf_me;
static float g_leaf_v;
static long g_leaf_cuts;
void ts_set_leaf(int d) { g_leaf_d = d > 0 ? d : 0; }
long ts_leaf_cuts(void) { return g_leaf_cuts; }
static int bv_fast_decide(void *ctx, const de_game *g, int kind, int me, int subject,
                          const int *opts, int n, int may_pass) {
    rl_seat *s = ctx;
    if (g_fs_on && me == g_fs_me && g_fs_i < g_fs_n) {   /* replaying my choices, then the candidate */
        int k = g_fs_i++, o = g_fs_opt[k], ok = kind == g_fs_kind[k] && ((o == -1 && may_pass) || 0);
        for (int i = 0; i < n && !ok && kind == g_fs_kind[k]; i++) ok = opts[i] == o;
        if (ok) return o;
        g_fs_bad = 1; g_fs_on = 0;                       /* diverged: play on, the draw is thrown away */
    }
    if (g_leaf_on) {                                   /* value-net leaf: the first decision D turns on scores it */
        if (g_leaf_hit) return may_pass ? -1 : opts[0];     /* the cut-off game only finishes its round, any legal move */
        if (g->p[0].turns + g->p[1].turns >= g_leaf_t0 + g_leaf_d) {
            float x[RL_NF], lg[RL_NA];
            rl_features(g, me, kind, subject, x);
            float v = rl_eval(s->prm, s->nh, s->nf, x, lg);   /* the decider's view, in [-1, 1] */
            if (me != g_leaf_me) v = -v;
            g_leaf_v = 0.5f * (v + 1.f);
            g_leaf_hit = 1;
            ((de_game *)g)->over = 1;                    /* ends the game after this round (no rules change: a copy) */
            return may_pass ? -1 : opts[0];
        }
    }
    if (s->mode != RL_NET || s->sample || s->record) return rl_decide(ctx, g, kind, me, subject, opts, n, may_pass);
    if (n + (may_pass != 0) == 1) return n ? opts[0] : -1;
    int ok[RL_NA] = {0};
    for (int i = 0; i < n; i++) ok[opts[i]] = 1;
    if (may_pass) ok[RL_PASS] = 1;
    float x[RL_NF], lg[RL_NA];
    rl_features(g, me, kind, subject, x);
    fnet_logits(fnet_get(s->prm, s->nh, s->nf), x, ok, RL_NA, lg);
    int a = -1;
    for (int i = 0; i < RL_NA; i++) if (ok[i] && (a < 0 || lg[i] > lg[a])) a = i;
    s->decisions++;
    return a == RL_PASS ? -1 : a;
}

/* One hidden-state draw: a copy of the game with the unseen cards dealt from seed (canonical order first). Every
 * candidate of a draw continues from the same prepared state. g_prep 1 (default): prepared once per draw; 0: once per
 * candidate (the old path, kept as the reference for the equivalence test: the states are identical). */
static int g_prep = 1;
void bv_set_prep(int on) { g_prep = on; }
static long g_preps;
long bv_preps(void) { return g_preps; }
static void prepare(const de_game *g, int me, uint64_t seed, de_game *out) {
    *out = *g;
    canon(out, me);
    bv_hide(out, me, seed);
    g_preps++;
}

static int g_roll_kind = DE_BUY;    /* which decision the rollouts resume from: DE_BUY or DE_PLAY_ACTION */
/* ts-v1 card choices: per seat, the game just before my last action play (the card still in hand), that card, the
 * known-top count then, and my decisions since (kind, choice) */
static int g_sub, g_sub_on, g_sub_kind;
static de_game g_pre[2];
static int g_snap_ok[2], g_snap_card[2], g_snap_known[2], g_nh[2], g_hk[2][TS_HIST], g_ho[2][TS_HIST];
static long g_sub_stats[4];         /* graded, overrides, rejected (diverged / error), card choices seen */
void ts_set_sub(int on) { g_sub = on; }
void ts_sub_stats(long *o) { memcpy(o, g_sub_stats, sizeof g_sub_stats); }
static long g_sub_rej[64], g_sub_seen[64];   /* per decision kind: rejected, seen */
void ts_sub_kinds(long *rej, long *seen) { memcpy(rej, g_sub_rej, sizeof g_sub_rej); memcpy(seen, g_sub_seen, sizeof g_sub_seen); }
static int g_fault;                 /* tests: every g_fault-th rollout reports an engine error (0 = off) */
void bv_set_fault(int every) { g_fault = every; }
static int g_last_cap, g_last_err;  /* the last rollout hit the round cap / an engine error */
static float rollout(const de_game *prepared, int me, int cand, uint64_t seed, int k) {
    static de_game c;
    c = *prepared;
    rl_seat rs[2] = {g_cont[0], g_cont[1]};
    if (g_nmix) rs[1 - me] = g_mix[k % g_nmix];
    rs[0].rng = rs[1].rng = seed ^ 0x9E3779B97F4A7C15ULL;
    de_decide_fn fn = g_fast ? bv_fast_decide : rl_decide;
    de_agent ag[2] = {{fn, &rs[0]}, {fn, &rs[1]}};
    g_leaf_on = g_leaf_d > 0 && g_fast; g_leaf_hit = 0; g_leaf_me = me;
    g_leaf_t0 = c.p[0].turns + c.p[1].turns;
    if (g_sub_on) {                                    /* card choice: from before the action, replay, force cand */
        memcpy(g_fs_kind, g_hk[me], sizeof(int) * g_nh[me]);
        memcpy(g_fs_opt, g_ho[me], sizeof(int) * g_nh[me]);
        g_fs_kind[g_nh[me]] = g_sub_kind; g_fs_opt[g_nh[me]] = cand == RL_PASS ? -1 : cand;
        g_fs_n = g_nh[me] + 1; g_fs_i = 0; g_fs_bad = 0; g_fs_me = me; g_fs_on = 1;
        de_resume_action(&c, ag, g_snap_card[me], 1);
        if (g_fs_bad || g_fs_i < g_fs_n) c.error = 1;
        g_fs_on = 0;
    } else if (g_roll_kind == DE_PLAY_ACTION) de_resume_action(&c, ag, cand == RL_PASS ? -1 : cand, 1);
    else de_resume_buy(&c, ag, cand == RL_PASS ? -1 : cand);
    g_leaf_on = 0;
    g_rollouts++;
    if (g_leaf_hit) {                                  /* cut off: the leaf's value, not the unfinished game */
        g_leaf_cuts++;
        g_last_cap = 0; g_last_err = c.error;
        if (c.error) g_errors++;
        return g_leaf_v;
    }
    if (g_fault && g_rollouts % g_fault == 0) c.error = 1;
    g_last_cap = c.capped; g_last_err = c.error;
    if (c.capped) g_capped++;
    if (c.error) g_errors++;
    int w = de_winners(&c);
    return w == 3 ? 0.5f : (w >> me & 1) ? 1.f : 0.f;
}

/* one draw for every candidate: V[a] per candidate; returns 1 if any rollout hit an engine error */
static int draw_all(const de_game *g, int me, const int *cand, int nc, uint64_t seed, int k, float *V,
                    unsigned char *caps) {
    static de_game P;
    int err = 0;
    if (g_prep) prepare(g, me, seed, &P);
    for (int i = 0; i < nc; i++) {
        int a = cand[i];
        if (!g_prep) prepare(g, me, seed, &P);
        V[a] = rollout(&P, me, a, seed, k);
        if (g_last_cap && caps && caps[a] < 255) caps[a]++;
        err |= g_last_err;
    }
    return err;
}

/* the candidates: buys -- all legal options if few, else the incumbent, PASS and the net's top-n other buys;
 * actions -- every legal option (an action card or PASS = end the action phase) */
static int bv_cands(const int *ok, const float *lg, int inc, int *cand) {
    int nc = 0, nl = 0, used[RL_NA] = {0};
    for (int a = 0; a < RL_NA; a++) nl += ok[a];
    if (nl <= g_cfg.topn + 3 || g_roll_kind == DE_PLAY_ACTION) {
        for (int a = 0; a < RL_NA; a++) if (ok[a]) cand[nc++] = a;
        return nc;
    }
    cand[nc++] = inc; used[inc] = 1;
    if (ok[RL_PASS] && !used[RL_PASS]) { cand[nc++] = RL_PASS; used[RL_PASS] = 1; }
    for (int t = 0; t < g_cfg.topn; t++) {
        int b = -1;
        for (int a = 0; a < RL_PASS; a++) if (ok[a] && !used[a] && (b < 0 || lg[a] > lg[b])) b = a;
        if (b < 0) break;
        cand[nc++] = b; used[b] = 1;
    }
    return nc;
}

/* c2's probabilities over the candidates (the student's prior input) */
static void bv_pol(const float *lg, const int *cand, int nc, float *pol) {
    float mx = -1e30f, sum = 0;
    for (int a = 0; a < RL_NA; a++) pol[a] = 0;
    for (int i = 0; i < nc; i++) if (lg[cand[i]] > mx) mx = lg[cand[i]];
    for (int i = 0; i < nc; i++) { pol[cand[i]] = expf(lg[cand[i]] - mx); sum += pol[cand[i]]; }
    for (int i = 0; i < nc; i++) pol[cand[i]] /= sum;
}

/* Grade the candidates of a decision of kind g_roll_kind; q, se: per option (NAN if not a candidate; the explicit mask
 * says which are), caps: rollouts that hit the round cap per option; returns draws used, or -1 if any rollout hit an
 * engine error (the decision is then not a label and the teacher keeps the incumbent: one policy for buys and
 * actions). Adaptive draws: after k0, add 16 at a time (up to k1) while the paired difference between the actual top
 * two candidates is within se_mult of its own standard error; a tie with zero observed SE is unresolved (another
 * batch may separate them), not a stop. */
#define BV_KMAX 1024
static double g_deadline;           /* bv_advise: stop adding draws after this time (bv_now; 0 = no cap) */

/* ts-v1 successive halving (ts_set_alloc 1): the budget of the uniform grader (k1 draws per candidate) in rounds;
 * after each round the better half of the candidates (by mean) goes on, the incumbent always (every comparison is
 * paired with it), until the leader and the incumbent are left, who share what remains. Draw k is the same hidden
 * state for every candidate that reaches it. q = each candidate's mean over its own draws; se = the paired difference
 * with the incumbent over those draws. Stops at the deadline like bv_grade. */
#define TS_KMAX 2048
static int g_alloc;
void ts_set_alloc(int a) { g_alloc = a; }
static long g_sh_final;             /* draws the final pair reached (sum over graded decisions) */
long ts_sh_final(void) { return g_sh_final; }
static int ts_grade_sh(const de_game *g, int me, int *cand, int nc, int inc, uint64_t *rng, float *q, float *se,
                       unsigned char *mask, unsigned char *caps) {
    static float V[TS_KMAX][RL_NA];
    int budget = g_cfg.k1 * nc, used = 0, K = 0, err = 0, stop = 0;
    int alive[RL_NA], na = nc, nd[RL_NA] = {0};
    double s[RL_NA] = {0};
    for (int i = 0; i < nc; i++) alive[i] = cand[i];
    int rounds = 1;                                            /* rounds left, the final pair's included */
    for (int m = nc; m > 2; m = (m + 1) / 2 < 2 ? 2 : (m + 1) / 2) rounds++;
    for (;;) {
        int left = budget - used;
        if (left < na) break;
        int per = na > 2 ? left / (na * rounds) : left / na;
        if (per < 1) per = 1;
        if (K + per > TS_KMAX) per = TS_KMAX - K;
        if (per <= 0) break;
        for (int k = 0; k < per; k++) {
            if (g_deadline > 0 && bv_now() > g_deadline) { stop = 1; break; }
            int d = K++;                                       /* draw d: one hidden state for every survivor */
            err |= draw_all(g, me, alive, na, de_rng_next(rng) | 1, d, V[d], caps);
            for (int i = 0; i < na; i++) { s[alive[i]] += V[d][alive[i]]; nd[alive[i]]++; }
            used += na;
        }
        if (stop || na <= 2) break;
        int order[RL_NA];                                      /* best mean first */
        for (int i = 0; i < na; i++) {
            int x = alive[i], j = i;
            double mx = s[x] / nd[x];
            while (j > 0 && s[order[j - 1]] / nd[order[j - 1]] < mx) { order[j] = order[j - 1]; j--; }
            order[j] = x;
        }
        int keep = (na + 1) / 2 < 2 ? 2 : (na + 1) / 2, has_inc = 0;
        for (int i = 0; i < keep; i++) { alive[i] = order[i]; has_inc |= order[i] == inc; }
        if (!has_inc) alive[keep - 1] = inc;                   /* the incumbent stays in */
        na = keep;
        rounds--;
    }
    for (int a = 0; a < RL_NA; a++) { q[a] = NAN; se[a] = NAN; }
    if (!K) return 0;
    for (int i = 0; i < nc; i++) {
        int a = cand[i], n = nd[a];
        if (!n) { mask[a] = 0; continue; }
        double m = 0, m2 = 0;
        for (int k = 0; k < n; k++) { double x = V[k][a] - V[k][inc]; m += x; m2 += x * x; }  /* inc has every draw */
        m /= n;
        double var = m2 / n - m * m;
        q[a] = (float)(s[a] / n);
        se[a] = (float)sqrt((var > 0 ? var : 0) / n);
    }
    for (int i = 0; i < na; i++) if (alive[i] != inc) g_sh_final += nd[alive[i]];
    for (int i = 0; i < nc; i++) {                             /* only the last survivors are graded options */
        int a = cand[i], in = 0;
        for (int j = 0; j < na; j++) in |= alive[j] == a;
        if (!in) mask[a] = 0;
    }
    g_graded++;
    return err ? -1 : K;
}

static int bv_grade(const de_game *g, int me, const int *ok, const float *lg, int inc, uint64_t *rng,
                    float *q, float *se, unsigned char *mask, unsigned char *caps) {
    int cand[RL_NA], nc = bv_cands(ok, lg, inc, cand);
    memset(mask, 0, RL_NA);
    memset(caps, 0, RL_NA);
    for (int i = 0; i < nc; i++) mask[cand[i]] = 1;
    if (g_alloc == 1) {
        g_in_plan++;
        int K = ts_grade_sh(g, me, cand, nc, inc, rng, q, se, mask, caps);
        g_in_plan--;
        return K;
    }
    static float V[BV_KMAX][RL_NA];
    double s[RL_NA] = {0};
    int K = 0, err = 0, k1 = g_cfg.k1 < BV_KMAX ? g_cfg.k1 : BV_KMAX;
    g_in_plan++;
    while (K < k1) {
        int step = K == 0 ? g_cfg.k0 : 16;
        for (int k = 0; k < step && K < k1; k++, K++) {
            if (g_deadline > 0 && bv_now() > g_deadline) break;
            uint64_t seed = de_rng_next(rng) | 1;
            err |= draw_all(g, me, cand, nc, seed, K, V[K], caps);
            for (int i = 0; i < nc; i++) s[cand[i]] += V[K][cand[i]];
        }
        int b1 = -1, b2 = -1;
        for (int i = 0; i < nc; i++) {
            int a = cand[i];
            if (b1 < 0 || s[a] > s[b1]) { b2 = b1; b1 = a; } else if (b2 < 0 || s[a] > s[b2]) b2 = a;
        }
        if (b2 < 0 || K >= k1 || (g_deadline > 0 && bv_now() > g_deadline)) break;
        double m = 0, m2 = 0;
        for (int k = 0; k < K; k++) { double x = V[k][b1] - V[k][b2]; m += x; m2 += x * x; }
        m /= K;
        double var = m2 / K - m * m, sd = sqrt((var > 0 ? var : 0) / K);
        if (sd == 0 ? m > 0 : m > g_cfg.se_mult * sd) break;
    }
    g_in_plan--;
    for (int a = 0; a < RL_NA; a++) { q[a] = NAN; se[a] = NAN; }
    if (!K) return 0;
    for (int i = 0; i < nc; i++) {
        int a = cand[i];
        double m = 0, m2 = 0;
        for (int k = 0; k < K; k++) { double x = V[k][a] - V[k][inc]; m += x; m2 += x * x; }
        m /= K;
        double var = m2 / K - m * m;
        q[a] = (float)(s[a] / K);
        se[a] = (float)sqrt((var > 0 ? var : 0) / K);
    }
    g_graded++;
    return err ? -1 : K;
}

/* Validation: kval fresh paired draws (independent of the grading draws: the rng has moved on) of the graded winner b
 * against the incumbent. gain, vse: mean paired difference and its SE. Returns -1 on an engine error. */
static int g_kval;
void bv_set_kval(int k) { g_kval = k < BV_KMAX ? k : BV_KMAX; }
static int bv_validate(const de_game *g, int me, int b, int inc, uint64_t *rng, float *gain, float *vse) {
    int cand[2] = {b, inc}, err = 0;
    float V[RL_NA];
    double m = 0, m2 = 0;
    g_in_plan++;
    for (int k = 0; k < g_kval; k++) {
        uint64_t seed = de_rng_next(rng) | 1;
        err |= draw_all(g, me, cand, 2, seed, k, V, NULL);
        double x = V[b] - V[inc];
        m += x; m2 += x * x;
    }
    g_in_plan--;
    m /= g_kval;
    double var = m2 / g_kval - m * m;
    *gain = (float)m;
    *vse = (float)sqrt((var > 0 ? var : 0) / g_kval);
    return err ? -1 : g_kval;
}

/* The action teacher: bv_set_act(0) buys only, 1 = buys and action decisions (DE_PLAY_ACTION: every legal action and
 * PASS = end the action phase, each played to the end of the game), 2 = actions only (buys plain c2). Applies to
 * seats with a bv mode, through the same grading, failure policy and records as buys: mode 1 plays the teacher's
 * choice (z-gated), mode 2 (and 4) only labels (the incumbent plays). Counters: graded, overrides, overrides that play
 * an action where c2 stopped, that stop where c2 played, that pick a different action. */
static int g_act;
static long g_act_stats[5];
/* action overrides of the current game: turn, c2's choice, the teacher's, paired gain, its SE (bv_act_log reads + resets) */
static float g_alog[512][5];
static int g_nalog;
int bv_act_log(float *out, int max) {
    int n = g_nalog < max ? g_nalog : max;
    memcpy(out, g_alog, (size_t)n * 5 * sizeof(float));
    g_nalog = 0;
    return n;
}
void bv_set_act(int a) { g_act = a; }
void bv_act_stats(long *o) { memcpy(o, g_act_stats, sizeof g_act_stats); }

static int bv_decide_main(void *ctx, const de_game *g, int kind, int me, int subject,
                          const int *opts, int n, int may_pass) {
    rl_seat *s = ctx;
    int mode = g_bv_mode[me & 1];
    if (g_kn_on && !g_in_plan) { kn_observe(g); g_known = kn_known(g, me); }
    if (kind == DE_BUY && !g_in_plan && s->mode == RL_NET && n + (may_pass != 0) > 1) {
        int f = rl_forced_buy(g, me, opts, n);          /* rl_set_force (card school, tests): as in rl_decide, a forced */
        if (f >= 0) return f;                           /* buy is neither graded nor recorded */
    }
    int graded_kind = (kind == DE_BUY && g_act != 2) || (kind == DE_PLAY_ACTION && g_act);
    if (g_in_plan || !graded_kind || mode == 0 || s->mode != RL_NET || n + (may_pass != 0) < 2 ||
        (kind == DE_PLAY_ACTION && mode == 3))
        return (g_fast == 2 && !g_in_plan ? bv_fast_decide : rl_decide)(ctx, g, kind, me, subject, opts, n, may_pass);
    int ok[RL_NA] = {0};
    for (int i = 0; i < n; i++) ok[opts[i]] = 1;
    if (may_pass) ok[RL_PASS] = 1;
    float x[RL_NF], lg[RL_NA];
    rl_features(g, me, kind, subject, x);
    rl_eval(s->prm, s->nh, s->nf, x, lg);
    int inc = -1;
    for (int a = 0; a < RL_NA; a++) if (ok[a] && (inc < 0 || lg[a] > lg[inc])) inc = a;
    int a = inc;
    if (kind == DE_BUY) g_buys++;
    double t0 = bv_now();
    float q[RL_NA], se[RL_NA], pol[RL_NA] = {0}, vg = 0, vse = 0;
    unsigned char mask[RL_NA] = {0}, caps[RL_NA] = {0};
    int K = 0, didx = -1, vk = 0;
    g_roll_kind = kind;
    if (mode == 1 || mode == 2 || mode == 4 || kind == DE_PLAY_ACTION) {
        didx = kind == DE_BUY ? g_didx++ : g_aidx++;
        int grade = g_rg_target < 0 || (didx == g_rg_target && kind == g_rg_kind);
        if (grade) {
            bv_cfg keep = g_cfg;
            uint64_t rr = g_rg_seed | 1, *rng = &s->rng;
            if (g_rg_target >= 0) {
                g_cfg.k0 = g_cfg.k1 = g_rg_k;
                if (g_rg_all) g_cfg.topn = RL_NA;
                rng = &rr;
            }
            K = bv_grade(g, me, ok, lg, inc, rng, q, se, mask, caps);
            g_kn_stats[0]++; g_kn_stats[1] += g_known > 0;
            g_pub_stats[0]++; g_pub_stats[1] += pub_count(g, 1 - me) > 0;
            g_cfg = keep;
            if (K < 0) { g_rejected++; K = 0; }          /* an engine error in a rollout: not a label, no override */
            if (K) {
                int b = inc;
                for (int c = 0; c < RL_NA; c++) if (mask[c] && q[c] > q[b]) b = c;
                if (b != inc && g_kval > 0) {
                    vk = bv_validate(g, me, b, inc, rng, &vg, &vse);
                    if (vk < 0) { g_rejected++; K = 0; vk = 0; }
                }
                int play = mode == 1 && b != inc && q[b] - q[inc] > g_cfg.margin && q[b] - q[inc] > g_z * se[b] &&
                           (g_kval <= 0 || (vg > g_cfg.margin && vg > g_z * vse));
                if (K && play) a = b;
            }
        }
        if (K) {
            int cl[RL_NA], ncl = 0;
            for (int c = 0; c < RL_NA; c++) if (mask[c]) cl[ncl++] = c;
            bv_pol(lg, cl, ncl, pol);
        }
    }
    if (kind == DE_BUY && (mode == 3 || mode == 4) && g_student) {
        float sq[RL_NA], spol[RL_NA];
        int cand[RL_NA], nc = bv_cands(ok, lg, inc, cand);
        bv_pol(lg, cand, nc, spol);
        bv_student(g_student, x, spol, sq);
        int b = inc;
        for (int i = 0; i < nc; i++) if (sq[cand[i]] > sq[b]) b = cand[i];
        if (sq[b] - sq[inc] > g_cfg.margin) a = b;
    }
    g_roll_kind = DE_BUY;
    if (mode != 2 && mode != 4 && g_nlat < 8192) g_lat[g_nlat++] = (float)(bv_now() - t0);
    if (kind == DE_BUY && a != inc) g_overrides++;
    if (kind == DE_PLAY_ACTION && K) {
        g_act_stats[0]++;
        if (a != inc) {
            g_act_stats[1]++;
            g_act_stats[inc == RL_PASS ? 2 : a == RL_PASS ? 3 : 4]++;
            if (g_nalog < 512) {
                float *e = g_alog[g_nalog++];
                e[0] = g->p[me].turns + 1; e[1] = inc; e[2] = a; e[3] = q[a] - q[inc]; e[4] = se[a];
            }
        }
    }
    if (K && g_bvrec) {
        bv_rec *r = g_bvrec;
        if (r->n >= r->max) r->overflow = 1;
        else {
            int k = r->n++;
            memcpy(r->X + (size_t)k * RL_NF, x, sizeof x);
            memcpy(r->Q + (size_t)k * RL_NA, q, sizeof q);
            memcpy(r->SE + (size_t)k * RL_NA, se, sizeof se);
            if (r->CAND) memcpy(r->CAND + (size_t)k * RL_NA, mask, RL_NA);
            if (r->POL) memcpy(r->POL + (size_t)k * RL_NA, pol, sizeof pol);
            if (r->CAPS) memcpy(r->CAPS + (size_t)k * RL_NA, caps, RL_NA);
            if (r->DIDX) r->DIDX[k] = didx;
            if (r->LEGAL) for (int c = 0; c < RL_NA; c++) r->LEGAL[(size_t)k * RL_NA + c] = (unsigned char)ok[c];
            if (r->KIND) r->KIND[k] = kind;
            if (r->VG) { r->VG[k] = vg; r->VSE[k] = vse; r->VK[k] = vk; }
            if (r->KN) r->KN[k] = g_known;
            r->INC[k] = inc; r->CHOSE[k] = a; r->K[k] = K; r->SEAT[k] = me; r->TURN[k] = g->p[me].turns + 1;
        }
    }
    if (kind == DE_BUY && !g_in_plan) {                 /* clip trace, as rl_decide */
        float pi[AZ_NA] = {0};
        pi[a == RL_PASS ? AZ_PASS : a] = 1;
        trace(g, me, a == RL_PASS ? AZ_PASS : a, pi, 0);
    }
    return a == RL_PASS ? -1 : a;
}

/* every live decision: public top-card tracking around the decision itself */
/* The card choices graded (ts_set_sub 1): made inside my own action before any hidden card is seen, so a rollout from
 * just before the action, replaying my choices, reaches the same decision exactly. */
static int ts_sub_kind(int kind) {
    return kind == DE_THRONE || kind == DE_CHAPEL_TRASH || kind == DE_CELLAR_DISCARD || kind == DE_MONEYLENDER_TRASH ||
           kind == DE_MINE_TRASH || kind == DE_MINE_GAIN || kind == DE_REMODEL_TRASH || kind == DE_REMODEL_GAIN ||
           kind == DE_WORKSHOP_GAIN || kind == DE_ARTISAN_GAIN || kind == DE_ARTISAN_TOPDECK;
}

static int ts_sub_decide(void *ctx, const de_game *g, int kind, int me, int subject, const int *opts, int n,
                         int may_pass) {
    rl_seat *s = ctx;
    int ok[RL_NA] = {0};
    for (int i = 0; i < n; i++) ok[opts[i]] = 1;
    if (may_pass) ok[RL_PASS] = 1;
    float x[RL_NF], lg[RL_NA], q[RL_NA], se[RL_NA];
    rl_features(g, me, kind, subject, x);
    rl_eval(s->prm, s->nh, s->nf, x, lg);
    int inc = -1;
    for (int a = 0; a < RL_NA; a++) if (ok[a] && (inc < 0 || lg[a] > lg[inc])) inc = a;
    unsigned char mask[RL_NA], caps[RL_NA];
    int keep_known = g_known, keep_kind = g_roll_kind;
    g_known = g_snap_known[me];
    g_roll_kind = DE_PLAY_ACTION;                      /* every legal option is a candidate */
    g_sub_on = 1; g_sub_kind = kind;
    double t0 = bv_now();
    int K = bv_grade(&g_pre[me], me, ok, lg, inc, &s->rng, q, se, mask, caps);
    g_sub_on = 0; g_known = keep_known; g_roll_kind = keep_kind;
    if (g_nlat < 8192) g_lat[g_nlat++] = (float)(bv_now() - t0);
    g_sub_stats[0] += K > 0;
    g_sub_seen[kind & 63]++;
    if (K < 0) { g_sub_stats[2]++; g_sub_rej[kind & 63]++; K = 0; }
    int a = inc;
    if (K) {
        int b = inc;
        for (int c = 0; c < RL_NA; c++) if (mask[c] && q[c] > q[b]) b = c;
        if (b != inc && q[b] - q[inc] > g_cfg.margin && q[b] - q[inc] > g_z * se[b]) { a = b; g_sub_stats[1]++; }
    }
    return a == RL_PASS ? -1 : a;
}

static int bv_decide(void *ctx, const de_game *g, int kind, int me, int subject,
                     const int *opts, int n, int may_pass) {
    if (!g_in_plan) pub_observe(g);
    int a;
    rl_seat *s = ctx;
    int sub = g_sub && !g_in_plan && g_bv_mode[me & 1] == 1 && s->mode == RL_NET;
    if (sub && g_snap_ok[me] && ts_sub_kind(kind) && n + (may_pass != 0) >= 2 && g_nh[me] < TS_HIST) {
        g_sub_stats[3]++;
        a = ts_sub_decide(ctx, g, kind, me, subject, opts, n, may_pass);
    } else a = bv_decide_main(ctx, g, kind, me, subject, opts, n, may_pass);
    if (sub) {
        if (kind == DE_PLAY_ACTION && g->current == me) {
            g_snap_ok[me] = a >= 0;                     /* the next card choices hang off this play */
            if (a >= 0) { g_pre[me] = *g; g_snap_card[me] = a; g_snap_known[me] = g_known; g_nh[me] = 0; }
        } else if (kind == DE_BUY || g->current != me) g_snap_ok[me] = 0;
        else if (g_snap_ok[me]) {
            if (g_nh[me] < TS_HIST) { g_hk[me][g_nh[me]] = kind; g_ho[me][g_nh[me]] = a; g_nh[me]++; }
            else g_snap_ok[me] = 0;
        }
    }
    if (!g_in_plan) { g_last_kind = kind; g_last_player = me; }
    return a;
}

/* One game, as rl_play (no AZ seats), with bv modes per seat. Labels into the record buffers.
 * out: [winners mask, score0, score1, turns0, turns1, capped | error<<1 | label overflow<<2]. Returns label records
 * written. */
int bv_play(const int *mode, const float *prm0, const float *prm1, const int *nh, const int *nf,
            const int *genome0, const int *genome1, const int *bvmode,
            const int *kingdom, int kn, uint64_t seed, uint64_t rng_seed,
            float *X, float *Q, float *SE, int *INC, int *CHOSE, int *K, int *SEAT, int *TURN,
            unsigned char *CAND, float *POL, int *DIDX, unsigned char *CAPS,
            unsigned char *LEGAL, int *KIND, float *VG, float *VSE, int *VK, int *KN, int max_rec, int *out) {
    bv_rec rec = {X, Q, SE, INC, CHOSE, K, SEAT, TURN, CAND, POL, DIDX, CAPS, LEGAL, KIND, VG, VSE, VK, KN, max_rec, 0, 0};
    g_didx = g_aidx = 0;
    g_bvrec = &rec;
    r_rec = NULL;
    const float *prm[2] = {prm0, prm1};
    const int *genome[2] = {genome0, genome1};
    uint64_t r = rng_seed | 1;
    rl_seat st[2];
    for (int i = 0; i < 2; i++) {
        memset(&st[i], 0, sizeof st[i]);
        st[i].mode = mode[i]; st[i].genome = genome[i]; st[i].prm = prm[i]; st[i].nh = nh[i]; st[i].nf = nf[i];
        st[i].temp = 1; st[i].rng = de_rng_next(&r) | 1;
        g_cont[i] = st[i];
        g_bv_mode[i] = bvmode[i];
    }
    g_plan_k[0] = g_plan_k[1] = 0;
    de_agent agents[2] = {{bv_decide, &st[0]}, {bv_decide, &st[1]}};
    de_game g;
    de_init(&g, kingdom, kn, 2, seed);
    g_kn_on = 1; g_kn_rng[0] = g_kn_rng[1] = 0; g_known = 0;
    memset(g_pub, 0, sizeof g_pub); g_pub_rng[0] = g_pub_rng[1] = 0; g_pub_n[0] = g_pub_n[1] = 0;
    g_last_kind = g_last_player = -1;
    g_snap_ok[0] = g_snap_ok[1] = 0;
    de_run(&g, agents);
    g_kn_on = 0; g_known = 0;
    out[0] = de_winners(&g);
    out[1] = de_score(&g, 0);
    out[2] = de_score(&g, 1);
    out[3] = g.p[0].turns;
    out[4] = g.p[1].turns;
    out[5] = g.capped | (g.error << 1) | (rec.overflow << 2);
    g_bvrec = NULL;
    return rec.n;
}

/* c2's candidate probabilities for a recorded state (training): lg from its features, candidates = the graded ones */
void bv_pol_of(const float *prm, int nh, int nf, const float *x, const unsigned char *mask, float *pol) {
    float lg[RL_NA];
    rl_eval(prm, nh, nf, x, lg);
    int cand[RL_NA], nc = 0;
    for (int a = 0; a < RL_NA; a++) if (mask[a]) cand[nc++] = a;
    bv_pol(lg, cand, nc, pol);
}

/* test hook: the canonical + hidden copy's my-deck and opponent-hand, for the sampler tests */
void bv_hide_probe(const de_game *g, int me, uint64_t seed, int *my_deck, int *opp_hand) {
    static de_game c;
    c = *g;
    canon(&c, me);
    bv_hide(&c, me, seed);
    memcpy(my_deck, c.p[me].deck, c.p[me].deck_n * sizeof(int));
    memcpy(opp_hand, c.p[1 - me].hand, c.p[1 - me].hand_n * sizeof(int));
}

/* Observation-equivalence self-test: plays c2 vs c2 until seat 0's first buy with at least `turn` turns done, then grades
 * that state (k draws, all legal buys) three ways from one seed: as is; with my draw pile in another order and the
 * opponent's hidden cards split differently between hand and deck (the same observation); and with one opponent card
 * replaced by another (a different observation). out[0] = max |q diff| of the same observation (0 expected), out[1] =
 * of the different one (> 0 expected); out[2], out[3]: the same with my top 2 cards known (rest reordered: 0 expected;
 * a different known top card: > 0 expected). Returns 1 if a state was found. */
static de_game g_snap;
static int g_snap_on, g_snap_got, g_snap_turn;
static int snap_decide(void *ctx, const de_game *g, int kind, int me, int subject, const int *opts, int n, int may_pass) {
    if (g_snap_on && !g_snap_got && kind == DE_BUY && me == 0 && g->p[0].turns >= g_snap_turn && g->p[0].deck_n > 2 &&
        g->p[1].deck_n > 1 && g->p[1].hand_n > 0) { g_snap = *g; g_snap_got = 1; }
    return rl_decide(ctx, g, kind, me, subject, opts, n, may_pass);
}
int bv_canon_test(const float *prm, int nh, int nf, const int *kingdom, int kn, uint64_t seed, int turn, int k, float *out) {
    rl_seat st[2];
    for (int i = 0; i < 2; i++) {
        memset(&st[i], 0, sizeof st[i]);
        st[i].mode = RL_NET; st[i].prm = prm; st[i].nh = nh; st[i].nf = nf; st[i].temp = 1; st[i].rng = 7 + i;
        g_cont[i] = st[i];
    }
    de_agent ag[2] = {{snap_decide, &st[0]}, {snap_decide, &st[1]}};
    de_game g;
    g_snap_on = 1; g_snap_got = 0; g_snap_turn = turn;
    de_init(&g, kingdom, kn, 2, seed);
    de_run(&g, ag);
    g_snap_on = 0;
    if (!g_snap_got) return 0;
    de_game v[3] = {g_snap, g_snap, g_snap};
    de_player *p = &v[1].p[0], *o = &v[1].p[1];
    for (int i = 0; i < p->deck_n / 2; i++) { int t = p->deck[i]; p->deck[i] = p->deck[p->deck_n - 1 - i]; p->deck[p->deck_n - 1 - i] = t; }
    for (int i = 0; i < o->deck_n; i++)              /* swap a hand card with a different deck card */
        if (o->deck[i] != o->hand[0]) { int t = o->deck[i]; o->deck[i] = o->hand[0]; o->hand[0] = t; break; }
    de_player *o2 = &v[2].p[1];
    o2->deck[0] = o2->deck[0] == DE_GOLD ? DE_CURSE : DE_GOLD;
    int ok[RL_NA] = {0};
    for (int c = 0; c < DE_NUM_CARDS; c++) ok[c] = v[0].supply[c] > 0 && DE_CARDS[c].cost <= v[0].coins;
    ok[RL_PASS] = 1;
    float x[RL_NF], lg[RL_NA], q[3][RL_NA], se[RL_NA];
    unsigned char mask[RL_NA], caps[RL_NA];
    rl_features(&v[0], 0, DE_BUY, -1, x);
    rl_eval(prm, nh, nf, x, lg);
    int inc = -1;
    for (int a = 0; a < RL_NA; a++) if (ok[a] && (inc < 0 || lg[a] > lg[inc])) inc = a;
    bv_cfg keep = g_cfg;
    g_cfg.k0 = g_cfg.k1 = k; g_cfg.topn = RL_NA;
    for (int j = 0; j < 3; j++) { uint64_t r = 12345; bv_grade(&v[j], 0, ok, lg, inc, &r, q[j], se, mask, caps); }
    /* known top cards (2 of them, as after a Sentry): the unknown rest in another order is the same observation; the
     * same cards with a different card on top is a different one */
    float qk[3][RL_NA];
    de_game w[3] = {g_snap, g_snap, g_snap};
    de_player *p1 = &w[1].p[0], *p2 = &w[2].p[0];
    int un = p1->deck_n - 2;
    for (int i = 0; i < un / 2; i++) { int t = p1->deck[i]; p1->deck[i] = p1->deck[un - 1 - i]; p1->deck[un - 1 - i] = t; }
    int top = p2->deck_n - 1, sw = -1;
    for (int i = 0; i < un; i++) if (p2->deck[i] != p2->deck[top]) { sw = i; break; }
    if (sw >= 0) { int t = p2->deck[sw]; p2->deck[sw] = p2->deck[top]; p2->deck[top] = t; }
    g_known = 2;
    for (int j = 0; j < 3; j++) { uint64_t r = 12345; bv_grade(&w[j], 0, ok, lg, inc, &r, qk[j], se, mask, caps); }
    g_known = 0;
    g_cfg = keep;
    out[0] = out[1] = out[2] = out[3] = 0;
    for (int a = 0; a < RL_NA; a++) {
        if (!mask[a]) continue;
        float d[4] = {fabsf(q[1][a] - q[0][a]), fabsf(q[2][a] - q[0][a]), fabsf(qk[1][a] - qk[0][a]),
                      sw >= 0 ? fabsf(qk[2][a] - qk[0][a]) : 0};
        for (int j = 0; j < 4; j++) if (d[j] > out[j]) out[j] = d[j];
    }
    return 1;
}

/* Public-sampler self-test on the same snapshot as bv_canon_test (seat 0's first buy after `turn` turns, c2 vs c2):
 * grades (k draws, all legal buys) of variants of the state, each against the state itself, as max |q diff|:
 * out[0] obs 1, an opponent discard card swapped with a different hidden draw-pile card (same public view: 0 expected)
 * out[1] obs 0, the same swap (the simulator sampler keeps the discard pile: > 0 expected)
 * out[2] obs 1, the opponent's top card marked public, two hidden draw-pile cards below it swapped (0 expected)
 * out[3] obs 1, that public top card swapped with a different hidden card (a different public view: > 0 expected)
 * Returns 1 if a state with an opponent discard pile and a draw pile of 3+ cards was found. */
int bv_obs_test(const float *prm, int nh, int nf, const int *kingdom, int kn, uint64_t seed, int turn, int k, float *out) {
    rl_seat st[2];
    for (int i = 0; i < 2; i++) {
        memset(&st[i], 0, sizeof st[i]);
        st[i].mode = RL_NET; st[i].prm = prm; st[i].nh = nh; st[i].nf = nf; st[i].temp = 1; st[i].rng = 7 + i;
        g_cont[i] = st[i];
    }
    de_agent ag[2] = {{snap_decide, &st[0]}, {snap_decide, &st[1]}};
    de_game g;
    g_snap_on = 1; g_snap_got = 0; g_snap_turn = turn;
    de_init(&g, kingdom, kn, 2, seed);
    de_run(&g, ag);
    g_snap_on = 0;
    if (!g_snap_got) return 0;
    const de_player *o0 = &g_snap.p[1];
    if (o0->discard_n < 1 || o0->deck_n < 3) return 0;
    int ok[RL_NA] = {0};
    for (int c = 0; c < DE_NUM_CARDS; c++) ok[c] = g_snap.supply[c] > 0 && DE_CARDS[c].cost <= g_snap.coins;
    ok[RL_PASS] = 1;
    float x[RL_NF], lg[RL_NA], q0[RL_NA], q1[RL_NA], se[RL_NA];
    unsigned char mask[RL_NA], caps[RL_NA];
    rl_features(&g_snap, 0, DE_BUY, -1, x);
    rl_eval(prm, nh, nf, x, lg);
    int inc = -1;
    for (int a = 0; a < RL_NA; a++) if (ok[a] && (inc < 0 || lg[a] > lg[inc])) inc = a;
    bv_cfg keep = g_cfg;
    g_cfg.k0 = g_cfg.k1 = k; g_cfg.topn = RL_NA;
    int keep_obs = g_obs;
    memset(g_pub, 0, sizeof g_pub);
    #define GRADE(state, obs, qq) do { g_obs = obs; uint64_t r = 12345; bv_grade(state, 0, ok, lg, inc, &r, qq, se, mask, caps); } while (0)
    #define MAXD(qa, qb, o) do { o = 0; for (int a = 0; a < RL_NA; a++) if (mask[a] && fabsf(qa[a] - qb[a]) > o) o = fabsf(qa[a] - qb[a]); } while (0)
    /* a discard card and a different draw-pile card of the opponent */
    de_game v = g_snap;
    de_player *o = &v.p[1];
    int di = 0, kk = -1;
    for (int i = 0; i < o->deck_n && kk < 0; i++) if (o->deck[i] != o->discard[di]) kk = i;
    out[0] = out[1] = out[2] = out[3] = -1;
    if (kk >= 0) {
        int t = o->deck[kk]; o->deck[kk] = o->discard[di]; o->discard[di] = t;
        GRADE(&g_snap, 1, q0); GRADE(&v, 1, q1); MAXD(q0, q1, out[0]);
        GRADE(&g_snap, 0, q0); GRADE(&v, 0, q1); MAXD(q0, q1, out[1]);
    }
    /* the opponent's top card public */
    const int top = o0->deck_n - 1;
    g_pub[1][top] = 1;
    de_game w = g_snap;
    de_player *ow = &w.p[1];
    int a1 = -1, a2 = -1;
    for (int i = 0; i < top && a2 < 0; i++) {
        if (a1 < 0) a1 = i;
        else if (ow->deck[i] != ow->deck[a1]) a2 = i;
    }
    if (a2 >= 0) {
        int t = ow->deck[a1]; ow->deck[a1] = ow->deck[a2]; ow->deck[a2] = t;
        GRADE(&g_snap, 1, q0); GRADE(&w, 1, q1); MAXD(q0, q1, out[2]);
    }
    de_game u = g_snap;
    de_player *ou = &u.p[1];
    int sw = -1;
    for (int i = 0; i < top; i++) if (ou->deck[i] != ou->deck[top]) { sw = i; break; }
    if (sw >= 0) {
        int t = ou->deck[sw]; ou->deck[sw] = ou->deck[top]; ou->deck[top] = t;
        GRADE(&g_snap, 1, q0); GRADE(&u, 1, q1); MAXD(q0, q1, out[3]);
    }
    #undef GRADE
    #undef MAXD
    memset(g_pub, 0, sizeof g_pub);
    g_obs = keep_obs;
    g_cfg = keep;
    return 1;
}

/* The advisor: grade one position read from a live game (the Steam advisor), with a time cap. The caller describes the
 * position: supply[DE_NUM_CARDS] (-1 = not in this game), me (0/1, the player to move), zones = the cards of
 * [my deck, my hand, my discard, my play, their deck, their hand, their discard, their play] one after another with
 * their sizes in zn[8], turns[2] (turns completed), kind (DE_BUY or DE_PLAY_ACTION), coins, buys, actions, merchants.
 * The order of my deck is unknown (shuffled per draw); the opponent's cards are pooled and dealt again at their zone
 * sizes (obs 1: only what a player can know). Rollouts continue both players with the net prm. inc: the advisor's own
 * pick (-1 = the net's arg-max), always a candidate. Draws: g_cfg (bv_set_cfg), stopped at `secs` seconds.
 * q, se: per option (NAN if not a candidate), mask: the candidates. out: [incumbent, net's arg-max, draws,
 * milliseconds, net's value]. Returns draws used (0: no time for one draw), -1 on an engine error in a rollout,
 * -2 if the position has fewer than two options. */
int bv_advise(const float *prm, int nh, int nf, const int *supply, int me, const int *zones, const int *zn,
              const int *turns, int kind, int coins, int buys, int actions, int merchants, int inc, float secs,
              uint64_t seed, float *q, float *se, unsigned char *mask, float *out) {
    double t0 = bv_now();
    static de_game g;
    memset(&g, 0, sizeof g);
    g.nplayers = 2;
    memcpy(g.supply, supply, sizeof g.supply);
    const int *z = zones;
    for (int i = 0; i < 2; i++) {
        de_player *p = &g.p[i == 0 ? me : 1 - me];
        int *dst[4] = {p->deck, p->hand, p->discard, p->play}, *len[4] = {&p->deck_n, &p->hand_n, &p->discard_n, &p->play_n};
        for (int k = 0; k < 4; k++) {
            int n = zn[4 * i + k];
            if (n < 0 || n > DE_ZONE) return -2;
            memcpy(dst[k], z, (size_t)n * sizeof(int));
            *len[k] = n;
            z += n;
        }
        p->turns = turns[i == 0 ? me : 1 - me];
        p->rng = 1;
    }
    g.current = me;
    g.coins = coins; g.buys = buys; g.actions = actions; g.merchants = merchants;
    g.trace = 0xcbf29ce484222325ULL;
    int ok[RL_NA] = {0}, nl = 0;
    if (kind == DE_BUY) {
        if (buys > 0)
            for (int c = 0; c < DE_NUM_CARDS; c++) if (g.supply[c] > 0 && DE_CARDS[c].cost <= coins) ok[c] = 1;
    } else if (kind == DE_PLAY_ACTION) {
        if (actions > 0)
            for (int k = 0; k < g.p[me].hand_n; k++) if (DE_CARDS[g.p[me].hand[k]].types & DE_ACTION) ok[g.p[me].hand[k]] = 1;
    } else return -2;
    ok[RL_PASS] = 1;
    for (int a = 0; a < RL_NA; a++) nl += ok[a];
    float x[RL_NF], lg[RL_NA];
    rl_features(&g, me, kind, -1, x);
    out[4] = rl_eval(prm, nh, nf, x, lg);
    int best = -1;
    for (int a = 0; a < RL_NA; a++) if (ok[a] && (best < 0 || lg[a] > lg[best])) best = a;
    if (inc < 0 || inc >= RL_NA || !ok[inc]) inc = best;
    out[0] = inc; out[1] = best; out[2] = 0;
    for (int a = 0; a < RL_NA; a++) { q[a] = NAN; se[a] = NAN; }
    memset(mask, 0, RL_NA);
    if (nl < 2) { out[3] = (float)(1000 * (bv_now() - t0)); return -2; }
    for (int i = 0; i < 2; i++) {
        memset(&g_cont[i], 0, sizeof g_cont[i]);
        g_cont[i].mode = RL_NET; g_cont[i].prm = prm; g_cont[i].nh = nh; g_cont[i].nf = nf; g_cont[i].temp = 1;
    }
    int keep_obs = g_obs, keep_kn = g_kn_on, keep_nmix = g_nmix;
    g_obs = 1; g_kn_on = 0; g_known = 0; g_nmix = 0;
    memset(g_pub, 0, sizeof g_pub);
    g_roll_kind = kind;
    g_deadline = t0 + secs;
    unsigned char caps[RL_NA];
    uint64_t rng = seed | 1;
    int K = bv_grade(&g, me, ok, lg, inc, &rng, q, se, mask, caps);
    g_deadline = 0;
    g_roll_kind = DE_BUY;
    g_obs = keep_obs; g_kn_on = keep_kn; g_nmix = keep_nmix;
    out[2] = K > 0 ? K : 0;
    out[3] = (float)(1000 * (bv_now() - t0));
    return K;
}

