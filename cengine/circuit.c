/* FullCircuit engine (spiking connectome engine).
 *
 * Per-target in-degree normalization, CSR fan-out delivery, flat delay
 * queues, xorshift128+ noise, Hassenstein-Reichardt T4/T5 correlators on
 * real preferred-direction subtypes, dopamine-gated three-factor plasticity
 * on the descending pool + slow Turrigiano scaling.
 */
#include "circuit.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------- biophysics table */

typedef struct { const char *name; float v[6]; } BioRow;

static const BioRow FULL_BIOS[] = {
    { "lamina",        { 10, -50, 8, 1.5f, 0.25f, 6 } },
    { "Tm",            { 20, -52, 6, 2.5f, 0.2f, 6 } },
    { "TmY",           { 25, -53, 6, 2.5f, 0.2f, 5 } },
    { "T4",            { 20, -52, 6, 2.0f, 0.2f, 5 } },
    { "T5",            { 20, -52, 6, 2.0f, 0.2f, 5 } },
    { "LC",            { 12, -50, 5, 1.5f, 0.2f, 5 } },
    { "LPLC",          { 12, -50, 5, 1.5f, 0.2f, 5 } },
    { "LT",            { 12, -51, 5, 1.5f, 0.2f, 4 } },
    { "lp-tangential", { 15, -52, 5, 2.0f, 0.2f, 4 } },
    { "optic-other",   { 20, -53, 7, 2.5f, 0.2f, 4 } },
    { "descending",    { 15, -52, 5, 1.5f, 0.25f, 4 } },
    { "ascending",     { 20, -53, 6, 2.5f, 0.2f, 4 } },
    { "DAN",           { 30, -55, 4, 2.0f, 0.1f, 2 } },
    { "MBON",          { 25, -54, 4, 2.0f, 0.15f, 3 } },
    { "KC",            { 12, -45, 10, 1.0f, 0.0f, 0.15f } },
    { "motor",         { 15, -52, 4, 1.5f, 0.25f, 4 } },
    { "neck-motor",    { 15, -52, 4, 1.5f, 0.25f, 4 } },
    { "sensory",       { 10, -50, 8, 1.5f, 0.3f, 6 } },
    { "central-other", { 25, -54, 6, 2.5f, 0.2f, 3 } },
    { "vnc-other",     { 20, -53, 6, 2.5f, 0.2f, 4 } },
    { "other",         { 25, -54, 6, 2.5f, 0.2f, 3 } },
    { "adult-specific",{ 25, -54, 5, 2.0f, 0.15f, 3 } },
};
#define NBIO ((int)(sizeof(FULL_BIOS) / sizeof(FULL_BIOS[0])))
static const float LEGACY_BIO[6] = { 20, -55, 6, 2.5f, 0.25f, 4 };

static int is_delay1(const char *g) {
    static const char *names[] = { "lamina", "T4", "T5", "Tm", "TmY", "optic-other",
                                   "LC", "LPLC", "LT", "lp-tangential", "sensory" };
    for (int i = 0; i < 11; i++)
        if (strcmp(g, names[i]) == 0) return 1;
    return 0;
}

static int is_delay2(const char *g) {
    static const char *names[] = { "vnc-other", "motor", "neck-motor", "descending",
                                   "ascending", "central-other" };
    for (int i = 0; i < 6; i++)
        if (strcmp(g, names[i]) == 0) return 1;
    return 0;
}

static int optic_group(const char *g) {
    static const char *names[] = { "Tm", "TmY", "optic-other", "LC", "LPLC",
                                   "LT", "lp-tangential" };
    for (int i = 0; i < 7; i++)
        if (strcmp(g, names[i]) == 0) return 1;
    return 0;
}

void fb_params_default(FbParams *p) {
    memset(p, 0, sizeof(*p));
    p->g_scale = 0.30f;
    p->tgt_budget = 36.0f;
    p->ph_tonic = 5.0f;
    p->ph_optic_gain = 2.0f;
    p->emd_gain = 10.0f;
    p->ph_abs = 1.2f;
    p->ph_contrast = 6.0f;
    p->ph_tau_ms = 800.0f;
    p->inh_gain = 0.6f;
    p->dan_mod_gain = 0.6f;
    p->dopa_gain = 0.09f;
    p->dan_base_tau_ms = 8000.0f;
    p->dan_tonic_mv = 3.0f;
    p->tau_elig_ms = 600.0f;
    p->tau_dopa_ms = 300.0f;
    p->a_ltp = 0.08f;
    p->a_ltd = 0.03f;
    p->w_min = 0.6f;
    p->w_max = 1.6f;
    p->w_floor = 0.1f;
    p->k_scale = 0.001f;
    p->scale_every = 10;
    p->seed = 0x9E3779B9u;
}

/* ------------------------------------------------------------- utilities */

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* xorshift128+ */
static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rng_next(FbCircuit *n) {
    uint64_t x = n->rng_s0, y = n->rng_s1;
    n->rng_s0 = y;
    x ^= x << 23;
    n->rng_s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return n->rng_s1 + y;
}

static inline float rng_float01(FbCircuit *n) {
    return (float)((rng_next(n) >> 11) * (1.0 / 9007199254740992.0));
}

static void push_i64(int64_t **arr, int *n, int *cap, int64_t v) {
    if (*n >= *cap) {
        *cap = *cap ? *cap * 2 : 256;
        *arr = (int64_t *)realloc(*arr, (size_t)*cap * sizeof(int64_t));
    }
    (*arr)[(*n)++] = v;
}

static int cmp_i64(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

typedef struct { int64_t tgt; int pos; } TgtPos;

/* motor pools: DIRECT per-neuron actuator channels (readout.pools). A pool
 * is a selected subset of one group's neurons; its spikes integrate into a
 * leaky accumulator (motor-unit temporal summation) and that integral is
 * the raw actuator signal. With "learn":true each pool neuron owns a
 * plastic weight, driven by the SAME self-regulated dopamine error
 * (dopa_self) that gates R-STDP: sustained dopa>0 (doing better than the
 * adapting baseline) potentiates whichever neurons are actually firing,
 * dopa<0 depresses them. The (spike - tonic) term means a constant
 * situation teaches nothing — the same anti-wedging rule the synapses
 * follow. Weights persist in the memory file alongside the synapse
 * multipliers, keyed by neuron id + group name.
 *
 * Pools clamp to [w_floor, w_max], NOT to w_min: the pool integral IS the
 * actuator signal, so weights stuck at w_min output a large constant
 * offset that railed the client's channel clamp — the body could then
 * never earn dopa>0 to re-potentiate (the "thrust is always 0" failure).
 * w_floor sits where the pool output is ≈ its floor: fully-depressed =
 * silent = neutral stick, with full headroom to recover. */
struct FbPool {
    int64_t *ids;         /* member neuron ids (circuit indices) */
    int n;
    int64_t *widx;        /* parallel plastic weight idx (== ids when learn) */
    double *w;            /* plastic multipliers (only when learn) */
    int learn;
    float lr_scale;       /* per-pool multiplier on circuit pool_lr (1.0)
                          * — lets one actuator learn faster (e.g. throttle
                          * 40x) without touching the others */
    float set_hz;         /* group tonic rate: (spike - tonic) learning ref */
    float acc;            /* leaky spike integral = raw actuator signal */
    float decay;          /* per-tick multiplier from integrate_ms */
};
static void fb_pools_tick(FbCircuit *n, float dopa);

static int cmp_tgtpos(const void *a, const void *b) {
    const TgtPos *x = (const TgtPos *)a, *y = (const TgtPos *)b;
    if (x->tgt != y->tgt) return x->tgt < y->tgt ? -1 : 1;
    return x->pos - y->pos;
}

/* ------------------------------------------------------------------ new */

FbCircuit *fb_circuit_new(const FbConnectome *c, const FbParams *p) {
    FbCircuit *n = (FbCircuit *)calloc(1, sizeof(FbCircuit));
    if (!n) return NULL;
    n->c = c;
    n->p = *p;
    n->G = c->n_groups;
    int N = c->N, E = c->E;
    n->rng_s0 = p->seed;
    n->rng_s1 = p->seed ^ 0xD1B54A32D192ED03ull;

    n->grp = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    for (int i = 0; i < N; i++) n->grp[i] = (int32_t)c->group_of[i];

    /* --- per-group biophysics -> per-neuron vectors --- */
    n->bio_tau = (float *)malloc((size_t)N * 4); n->bio_vth = (float *)malloc((size_t)N * 4);
    n->bio_noise = (float *)malloc((size_t)N * 4); n->bio_ref = (float *)malloc((size_t)N * 4);
    n->bio_adapt = (float *)malloc((size_t)N * 4); n->bio_set_hz = (float *)malloc((size_t)N * 4);
    n->bio_base = (float *)malloc((size_t)N * 4); n->bio_margin = (float *)malloc((size_t)N * 4);
    n->decay_m = (float *)malloc((size_t)N * 4);
    for (int g = 0; g < n->G; g++) {
        const float *b = NULL;
        for (int r = 0; r < NBIO; r++)
            if (strcmp(c->groups[g], FULL_BIOS[r].name) == 0) { b = FULL_BIOS[r].v; break; }
        if (!b) b = LEGACY_BIO;
        for (int i = 0; i < N; i++) {
            if (n->grp[i] != g) continue;
            n->bio_tau[i] = b[0]; n->bio_vth[i] = b[1]; n->bio_noise[i] = b[2];
            n->bio_ref[i] = b[3]; n->bio_adapt[i] = b[4]; n->bio_set_hz[i] = b[5];
        }
    }
    /* fluctuation margin + baseline (no assumed drive) */
    for (int i = 0; i < N; i++) {
        float set = n->bio_set_hz[i];
        float z = set >= 6 ? 2.2f : set >= 5 ? 2.4f : set >= 4 ? 2.6f
                : set >= 3 ? 2.75f : set >= 2 ? 2.9f : 3.7f;
        float d = expf(-FB_DT_MS / n->bio_tau[i]);
        float f = sqrtf((1 - d) / (1 + d));
        float margin = 0.55f * z * f * n->bio_noise[i] / sqrtf(3.0f);
        n->bio_margin[i] = margin;
        n->bio_base[i] = (n->bio_vth[i] - FB_V_REST) - margin;
        n->decay_m[i] = expf(-FB_DT_MS / n->bio_tau[i]);
    }

    /* --- per-target in-degree normalization --- */
    int *in_deg = (int *)calloc((size_t)N, sizeof(int));
    for (int e = 0; e < E; e++) in_deg[c->edge_dst[e]]++;
    float *scale = (float *)malloc((size_t)N * 4);
    for (int i = 0; i < N; i++)
        scale[i] = in_deg[i] > 0 ? n->p.tgt_budget / (float)in_deg[i] : 0.0f;

    /* --- CSR by source (counting sort over 25.5M edges) --- */
    n->out_start = (int64_t *)calloc((size_t)N + 1, sizeof(int64_t));
    for (int e = 0; e < E; e++) n->out_start[c->edge_src[e] + 1]++;
    for (int i = 0; i < N; i++) n->out_start[i + 1] += n->out_start[i];
    n->out_edge = (int64_t *)malloc((size_t)E * sizeof(int64_t));
    n->out_dst = (int64_t *)malloc((size_t)E * sizeof(int64_t));
    n->out_w = (float *)malloc((size_t)E * 4);
    n->out_sign = (int8_t *)malloc((size_t)E);
    {
        int64_t *cursor = (int64_t *)malloc(((size_t)N + 1) * sizeof(int64_t));
        memcpy(cursor, n->out_start, ((size_t)N + 1) * sizeof(int64_t));
        for (int e = 0; e < E; e++) {
            int src = (int)c->edge_src[e];
            int64_t pos = cursor[src]++;
            n->out_edge[pos] = e;
            n->out_dst[pos] = c->edge_dst[e];
            float w = c->edge_w[e] * scale[c->edge_dst[e]];
            n->out_w[pos] = w < 2.0f ? w : 2.0f;
            n->out_sign[pos] = c->nt[c->edge_src[e]] >= 0 ? 1 : -1;
        }
        free(cursor);
    }
    free(in_deg);
    free(scale);

    /* --- delays, depression, masks --- */
    n->delay_ticks = (int8_t *)malloc((size_t)N);
    n->optic_mask = (uint8_t *)calloc((size_t)N, 1);
    n->sensory_set = (uint8_t *)calloc((size_t)N, 1);
    for (int i = 0; i < N; i++) {
        const char *g = c->groups[n->grp[i]];
        n->delay_ticks[i] = is_delay1(g) ? 1 : (is_delay2(g) ? 2 : 0);
        n->optic_mask[i] = (uint8_t)optic_group(g);
        if (strcmp(g, "sensory") == 0) n->sensory_set[i] = 1;
    }
    n->dep_res = (float *)malloc((size_t)E * 4);
    for (int e = 0; e < E; e++) n->dep_res[e] = 1.0f;
    n->wm_edge = (double *)malloc((size_t)E * sizeof(double));
    for (int e = 0; e < E; e++) n->wm_edge[e] = 1.0;

    /* --- state --- */
    n->vm = (float *)malloc((size_t)N * 4);
    for (int i = 0; i < N; i++) n->vm[i] = FB_V_REST;
    n->ge = (float *)calloc((size_t)N, 4);
    n->gi = (float *)calloc((size_t)N, 4);
    n->ge_inc = (float *)calloc((size_t)N, 4);
    n->gi_inc = (float *)calloc((size_t)N, 4);
    n->ref_until = (double *)calloc((size_t)N, sizeof(double));
    n->spiked = (uint8_t *)calloc((size_t)N, 1);
    n->hz_ema = (float *)calloc((size_t)N, 4);
    n->spike_counts = (int32_t *)calloc((size_t)N, sizeof(int32_t));
    n->grp_n = (int64_t *)calloc((size_t)n->G, sizeof(int64_t));
    for (int i = 0; i < N; i++) n->grp_n[n->grp[i]]++;
    n->grp_spikes = (double *)calloc((size_t)n->G, sizeof(double));
    n->pop_rate = (double *)calloc((size_t)n->G, sizeof(double));

    /* --- lamina retinotopy: per-side column maps --- */
    {
        /* first pass: sizes */
        int lam_g = fb_group_index(n, "lamina");
        int maxw = 0, maxh = 0;
        for (int side = 0; side < 2; side++) {
            for (int i = 0; i < N; i++) {
                if (n->grp[i] != lam_g || c->side[i] != side) continue;
                if (c->hex[i * 2] >= 0 && c->hex[i * 2 + 1] >= 0) {
                    if (c->hex[i * 2] + 1 > maxw) maxw = c->hex[i * 2] + 1;
                    if (c->hex[i * 2 + 1] + 1 > maxh) maxh = c->hex[i * 2 + 1] + 1;
                }
            }
        }
        n->lam_w = maxw; n->lam_h = maxh;
        int count[2] = { 0, 0 };
        for (int i = 0; i < N; i++) {
            if (n->grp[i] != lam_g) continue;
            if (c->hex[i * 2] >= 0 && c->hex[i * 2 + 1] >= 0 &&
                c->hex[i * 2] < n->lam_w && c->hex[i * 2 + 1] < n->lam_h)
                count[c->side[i]]++;
        }
        n->lam_n = count[0] + count[1];
        n->lam_side = (int32_t *)malloc((size_t)n->lam_n * 4);
        n->lam_neurons = (int64_t *)malloc((size_t)n->lam_n * 8);
        n->lam_cell = (int64_t *)malloc((size_t)n->lam_n * 8);
        n->lam_col_buf = (float *)calloc((size_t)n->lam_n, 4);
        for (int k = 0; k < n->lam_n; k++) n->lam_col_buf[k] = 0.4f;
        int cl = 0, cr = count[0]; /* left fills first, right after */
        for (int i = 0; i < N; i++) {
            if (n->grp[i] != lam_g) continue;
            int hx = c->hex[i * 2], hy = c->hex[i * 2 + 1];
            if (hx < 0 || hy < 0 || hx >= n->lam_w || hy >= n->lam_h) continue;
            int side = c->side[i];
            int pos = side ? cr++ : cl++;
            n->lam_side[pos] = side;
            n->lam_neurons[pos] = i;
            n->lam_cell[pos] = (int64_t)hy * n->lam_w + hx;
        }
    }

    /* --- T4/T5 + EMD state --- */
    {
        int t4g = fb_group_index(n, "T4"), t5g = fb_group_index(n, "T5");
        int c4 = 0, c5 = 0;
        for (int i = 0; i < N; i++) {
            if (n->grp[i] == t4g) c4++;
            if (n->grp[i] == t5g) c5++;
        }
        n->t4_n = c4; n->t5_n = c5;
        n->t4_idx = c4 ? (int64_t *)malloc((size_t)c4 * 8) : NULL;
        n->t5_idx = c5 ? (int64_t *)malloc((size_t)c5 * 8) : NULL;
        n->t4_dir = c4 ? (int32_t *)malloc((size_t)c4 * 4) : NULL;
        n->t5_dir = c5 ? (int32_t *)malloc((size_t)c5 * 4) : NULL;
        n->t4_hex = c4 ? (int16_t *)malloc((size_t)c4 * 2 * 2) : NULL;
        n->t5_hex = c5 ? (int16_t *)malloc((size_t)c5 * 2 * 2) : NULL;
        n->t4_side = c4 ? (int8_t *)malloc((size_t)c4) : NULL;
        n->t5_side = c5 ? (int8_t *)malloc((size_t)c5) : NULL;
        n->t4_drive = c4 ? (float *)calloc((size_t)c4, 4) : NULL;
        n->t5_drive = c5 ? (float *)calloc((size_t)c5, 4) : NULL;
        int i4 = 0, i5 = 0;
        for (int i = 0; i < N; i++) {
            if (n->grp[i] == t4g && i4 < c4) {
                n->t4_idx[i4] = i;
                n->t4_dir[i4] = c->dir[i];
                n->t4_hex[i4 * 2] = c->hex[i * 2];
                n->t4_hex[i4 * 2 + 1] = c->hex[i * 2 + 1];
                n->t4_side[i4] = (int8_t)c->side[i];
                i4++;
            } else if (n->grp[i] == t5g && i5 < c5) {
                n->t5_idx[i5] = i;
                n->t5_dir[i5] = c->dir[i];
                n->t5_hex[i5 * 2] = c->hex[i * 2];
                n->t5_hex[i5 * 2 + 1] = c->hex[i * 2 + 1];
                n->t5_side[i5] = (int8_t)c->side[i];
                i5++;
            }
        }
        for (int k = 0; k < 2; k++) {
            n->lum_lag[k] = (float *)calloc((size_t)n->lam_w * n->lam_h, 4);
            n->lag_ok[k] = 0;
        }
        n->drv_scratch = (float *)malloc((size_t)(c4 > c5 ? c4 : c5) * 4);
    }

    /* --- DAN pathway --- */
    n->dan_base_hz = 0.0f;
    n->dopa_self = 0.0f;
    n->warming = 1;
    {
        int dg = fb_group_index(n, "DAN");
        int cnt = 0;
        for (int i = 0; i < N; i++) if (n->grp[i] == dg) cnt++;
        n->dan_n = cnt;
        n->dan_idx = cnt ? (int64_t *)malloc((size_t)cnt * 8) : NULL;
        int di = 0;
        for (int i = 0; i < N; i++) if (n->grp[i] == dg) n->dan_idx[di++] = i;
        n->dan_set = (uint8_t *)calloc((size_t)N, 1);
        n->dan_tgt = (uint8_t *)calloc((size_t)N, 1);
        for (int k = 0; k < cnt; k++) n->dan_set[n->dan_idx[k]] = 1;
        for (int e = 0; e < E; e++)
            if (n->dan_set[c->edge_src[e]]) n->dan_tgt[c->edge_dst[e]] = 1;
    }

    /* --- plasticity: strongest real inputs into descending --- */
    {
        int dn_g = fb_group_index(n, "descending");
        n->plastic_n = c->plastic_count;
        n->plastic_idx = (int64_t *)malloc((size_t)n->plastic_n * 8);
        for (int k = 0; k < n->plastic_n; k++)
            n->plastic_idx[k] = (int64_t)c->plastic_idx[k];
        n->w_mult = (double *)malloc((size_t)n->plastic_n * 8);
        n->elig = (double *)calloc((size_t)n->plastic_n, 8);
        for (int k = 0; k < n->plastic_n; k++) n->w_mult[k] = 1.0;
        /* targets + scaling blocks */
        TgtPos *tp = (TgtPos *)malloc((size_t)n->plastic_n * sizeof(TgtPos));
        for (int k = 0; k < n->plastic_n; k++) {
            tp[k].tgt = c->edge_dst[n->plastic_idx[k]];
            tp[k].pos = k;
        }
        qsort(tp, (size_t)n->plastic_n, sizeof(TgtPos), cmp_tgtpos);
        n->scale_pos = (int *)malloc((size_t)n->plastic_n * 4);
        n->n_targets = 0;
        int64_t last = -1;
        for (int k = 0; k < n->plastic_n; k++) {
            n->scale_pos[k] = tp[k].pos;
            if (tp[k].tgt != last) { n->n_targets++; last = tp[k].tgt; }
        }
        n->scale_bounds = (int *)calloc((size_t)n->n_targets + 1, sizeof(int));
        int tb = 0;
        last = -1;
        for (int k = 0; k < n->plastic_n; k++) {
            if (tp[k].tgt != last) { n->scale_bounds[tb++] = k; last = tp[k].tgt; }
        }
        n->scale_bounds[n->n_targets] = n->plastic_n;
        free(tp);
        /* sorted plastic idx + order for memory import */
        n->plastic_sorted = (int64_t *)malloc((size_t)n->plastic_n * 8);
        n->plastic_order = (int *)malloc((size_t)n->plastic_n * 4);
        for (int k = 0; k < n->plastic_n; k++) n->plastic_order[k] = k;
        memcpy(n->plastic_sorted, n->plastic_idx, (size_t)n->plastic_n * 8);
        /* insertion-free sort: qsort pairs (idx, order) */
        {
            TgtPos *pp = (TgtPos *)malloc((size_t)n->plastic_n * sizeof(TgtPos));
            for (int k = 0; k < n->plastic_n; k++) { pp[k].tgt = n->plastic_idx[k]; pp[k].pos = k; }
            qsort(pp, (size_t)n->plastic_n, sizeof(TgtPos), cmp_tgtpos);
            for (int k = 0; k < n->plastic_n; k++) {
                n->plastic_sorted[k] = pp[k].tgt;
                n->plastic_order[k] = pp[k].pos;
            }
            free(pp);
        }
    }

    /* --- per-side DN masks (steering readout) --- */
    {
        int dn_g = fb_group_index(n, "descending");
        n->dn_l_mask = (uint8_t *)calloc((size_t)N, 1);
        n->dn_r_mask = (uint8_t *)calloc((size_t)N, 1);
        int cl = 0, cr = 0;
        for (int i = 0; i < N; i++) {
            if (n->grp[i] != dn_g) continue;
            if (c->side[i] == 1) { n->dn_r_mask[i] = 1; cr++; }
            else { n->dn_l_mask[i] = 1; cl++; }
        }
        n->dn_l_n = cl > 0 ? cl : 1;
        n->dn_r_n = cr > 0 ? cr : 1;
    }

    /* --- scratch --- */
    n->i_ext = (float *)malloc((size_t)N * 4);
    n->noise = (float *)malloc((size_t)N * 4);
    n->vm_next = (float *)malloc((size_t)N * 4);
    n->vth_eff = (float *)malloc((size_t)N * 4);
    n->fire = (uint8_t *)malloc((size_t)N);
    n->fired = (int64_t *)malloc((size_t)N * 8);
    n->fired_n = 0;

    n->sim_ms = 0;
    n->last_dn_l = n->last_dn_r = 0;
    n->touch_blast_mV = 0.0f;
    return n;
}

void fb_circuit_free(FbCircuit *n) {
    if (!n) return;
    free(n->grp);
    free(n->bio_tau); free(n->bio_vth); free(n->bio_noise);
    free(n->bio_ref); free(n->bio_adapt); free(n->bio_set_hz);
    free(n->bio_base); free(n->bio_margin); free(n->decay_m);
    free(n->out_start); free(n->out_edge); free(n->out_dst);
    free(n->out_w); free(n->out_sign);
    free(n->dep_res); free(n->wm_edge);
    free(n->vm); free(n->ge); free(n->gi); free(n->ge_inc); free(n->gi_inc);
    free(n->ref_until); free(n->spiked); free(n->hz_ema); free(n->spike_counts);
    free(n->grp_n); free(n->grp_spikes); free(n->pop_rate);
    free(n->lam_side); free(n->lam_neurons); free(n->lam_cell); free(n->lam_col_buf);
    free(n->t4_idx); free(n->t5_idx); free(n->t4_dir); free(n->t5_dir);
    free(n->t4_hex); free(n->t5_hex); free(n->t4_side); free(n->t5_side);
    free(n->t4_drive); free(n->t5_drive);
    free(n->lum_lag[0]); free(n->lum_lag[1]);
    free(n->drv_scratch);
    free(n->dan_idx); free(n->dan_set); free(n->dan_tgt);
    free(n->plastic_idx); free(n->w_mult); free(n->elig);
    free(n->scale_pos); free(n->scale_bounds);
    free(n->plastic_sorted); free(n->plastic_order);
    free(n->dn_l_mask); free(n->dn_r_mask);
    for (int i = 0; i < n->n_pools; i++) {
        free(n->pools[i].ids);
        free(n->pools[i].widx);
        free(n->pools[i].w);
    }
    free(n->pools);
    free(n->i_ext); free(n->noise); free(n->vm_next); free(n->vth_eff);
    free(n->fire); free(n->fired);
    free(n->sensory_set);
    free(n->pend0_cur); free(n->pend0_next);
    for (int d = 0; d < 2; d++) { free(n->dcur[d]); free(n->dnext[d]); }
    free(n);
}

/* -------------------------------------------------------------- group api */

int fb_group_index(const FbCircuit *n, const char *name) {
    for (int g = 0; g < n->G; g++)
        if (strcmp(n->c->groups[g], name) == 0) return g;
    return -1;
}

float fb_rate_of(const FbCircuit *n, const char *name) {
    int g = fb_group_index(n, name);
    return g >= 0 ? (float)n->pop_rate[g] : 0.0f;
}

float fb_dn_steer(const FbCircuit *n) {
    float s = n->dn_l_rate + n->dn_r_rate;
    float v = (n->dn_l_rate - n->dn_r_rate) / (s > 4.0f ? s : 4.0f);
    return clampf(v, -1.0f, 1.0f);
}

void fb_emd_flow(const FbCircuit *n, float *hL, float *hR, float *vL, float *vR) {
    *hL = n->flow_hL; *hR = n->flow_hR; *vL = n->flow_vL; *vR = n->flow_vR;
}

void fb_apply_reward(FbCircuit *n, float r) {
    n->reward = clampf(n->reward + r, -2.0f, 2.0f);
    n->dan_drive = clampf(n->dan_drive + r, -1.5f, 1.5f);
}

/* brief mechanosensory burst (the collision/tap sensory pathway) */
void fb_circuit_sensory_burst(FbCircuit *n, float mv) {
    if (mv == 0.0f) return;
    n->touch_blast_mV = mv;
}

/* EXTERNAL signals (environment reward, REST/WS injections) enter only as a
 * DAN-excitability bias — a sensory pathway, like PPL1 input to the real
 * mushroom body. They never write the plasticity signal directly: whether
 * learning happens depends on whether the bias actually moves DAN firing,
 * which the circuit measures itself (dopa_self). */
void fb_apply_dan_bias(FbCircuit *n, float r) {
    n->dan_drive = clampf(n->dan_drive + r, -1.5f, 1.5f);
}

void fb_set_learning(FbCircuit *n, int on) { n->learn_on = on ? 1 : 0; }

/* -------------------------------------------------------------- delivery */

/* fan out a batch of presynaptic ids through the CSR into ge_inc/gi_inc */
static void deliver(FbCircuit *n, const int64_t *srcs, int count) {
    const FbConnectome *c = n->c;
    const float g_scale = n->p.g_scale;
    for (int k = 0; k < count; k++) {
        int64_t src = srcs[k];
        int64_t s = n->out_start[src], e = n->out_start[src + 1];
        for (int64_t p = s; p < e; p++) {
            int64_t eid = n->out_edge[p];
            float res = n->dep_res[eid];
            res = res < 0.98f ? res + (1 - res) * 0.02f : 1.0f;
            float w = n->out_w[p] * res;
            res *= 0.82f;
            n->dep_res[eid] = res > 0.05f ? res : 0.05f;
            w = (float)(w * n->wm_edge[eid]) * g_scale;
            int64_t dst = n->out_dst[p];
            if (n->out_sign[p] > 0) n->ge_inc[dst] += w;
            else n->gi_inc[dst] += w;
        }
        (void)c;
    }
}

/* ------------------------------------------------------------------- tick */

int fb_tick(FbCircuit *n, const float *rgb_l, const float *rgb_r, int gw, int gh) {
    const int N = n->c->N;
    const float dt = (float)FB_DT_MS;
    int spike_count = 0;
    n->fired_n = 0;

    /* --- conductances: decay + fold in last tick's deliveries --- */
    const float decay_s = expf(-dt / 10.0f);
    for (int i = 0; i < N; i++) {
        n->ge[i] = n->ge[i] * decay_s + n->ge_inc[i];
        n->gi[i] = n->gi[i] * decay_s + n->gi_inc[i];
        n->ge_inc[i] = 0.0f;
        n->gi_inc[i] = 0.0f;
    }

    /* --- deliver due spike batches --- */
    if (n->pend0_cur_n) {
        deliver(n, n->pend0_cur, n->pend0_cur_n);
        n->pend0_cur_n = 0;
    }
    for (int d = 0; d < 2; d++) {
        if (n->dcur_n[d]) {
            deliver(n, n->dcur[d], n->dcur_n[d]);
            n->dcur_n[d] = 0;
        }
    }

    /* --- external current --- */
    float inh = n->p.inh_gain * tanhf((float)n->last_tick_spikes / (float)N / 0.05f);
    float dan_mod = 0.0f;
    if (n->dan_drive != 0.0f) {
        /* external reward bias = DAN excitability. No satiety clamp here:
         * the slow baseline (dan_base_hz) already self-regulates the
         * teaching signal, and a living pacemaker is meant to be pushable. */
        dan_mod = n->p.dan_mod_gain * n->dan_drive;
    }
    const double now = n->sim_ms;
    for (int i = 0; i < N; i++) {
        float base = n->bio_base[i] - inh;
        if (dan_mod != 0.0f) {
            if (n->dan_set[i]) base += dan_mod;
            else if (n->dan_tgt[i]) base += 0.8f * dan_mod;
        }
        /* DAN pacemaker: PAM-cluster dopamine neurons are spontaneously
         * active (~2-8 Hz tonic). This tonic depolarization is what makes
         * the SELF-regulated dopamine loop meaningful — external signals
         * modulate an already-living system instead of creating it. */
        if (n->dan_set[i]) base += n->p.dan_tonic_mv;
        /* mechanosensory startle: a bump/tap injects a brief depolarizing
         * burst into the sensory population — like bristle/joint receptor
         * currents. The connectome's own wiring propagates the event into
         * behavior (VNC/DNs) and reward circuits; nothing is scripted. */
        if (n->touch_blast_mV != 0.0f && n->sensory_set[i])
            base += n->touch_blast_mV;
        n->i_ext[i] = base;
    }

    /* --- photoreceptors: retinotopic drive every tick (light is light).
     * Single-eye mode: both sides sample the ONE frame — the connectome's
     * left/right lamina columns just view the left/right half of it. --- */
    const int W = n->lam_w, H = n->lam_h;
    int have_vis = n->lam_n > 0 && (rgb_l != NULL || rgb_r != NULL);
    if (have_vis && (gw < W || gh < H)) return 0; /* grid too small: skip frame */

    float l_bar = 0.0f;
    if (have_vis) {
        const float k_ph = dt / n->p.ph_tau_ms;
        const int one_eye = (rgb_l == NULL || rgb_r == NULL || rgb_l == rgb_r);
        for (int k = 0; k < n->lam_n; k++) {
            int side = n->lam_side[k];
            const float *g;
            int cell;
            if (one_eye) {
                /* split the shared frame: left columns read the left half */
                cell = (int)n->lam_cell[k];
                int cx = cell % W;
                int half = W / 2;
                cx = side ? cx / 2 + half : cx / 2;
                cell = (cell / W) * W + cx;
                g = rgb_l ? rgb_l : rgb_r;
            } else {
                g = side ? rgb_r : rgb_l;
                cell = (int)n->lam_cell[k];
            }
            float l = g[cell * 3 + 0] * 0.299f + g[cell * 3 + 1] * 0.587f + g[cell * 3 + 2] * 0.114f;
            n->lam_col_buf[k] += (l - n->lam_col_buf[k]) * k_ph;
            float ph = n->p.ph_tonic * l +
                       n->p.ph_abs * tanhf(n->p.ph_contrast * (l - n->lam_col_buf[k]));
            n->i_ext[n->lam_neurons[k]] += ph;
        }
        /* ambient optic-lobe tone */
        {
            double s = 0;
            for (int k = 0; k < n->lam_n; k++) s += n->lam_col_buf[k];
            l_bar = n->lam_n ? (float)(s / n->lam_n) : 0.0f;
        }
        if (l_bar > 0.01f) {
            const float tone = n->p.ph_optic_gain * l_bar;
            for (int i = 0; i < N; i++)
                if (n->optic_mask[i]) n->i_ext[i] += tone;
        }
    }

    /* --- T4/T5 EMDs: correlate ONCE per new frame pair, hold the result ---
     * (at real camera cadence the 25 ms lag line re-converges between
     * frames; per-tick correlation averages to ~0. Frame-locked bursts are
     * what lobula plate tangential cells actually integrate.) */
    {
        int any_new = n->frame_new[0] || n->frame_new[1];
        const int one_eye_mode = (rgb_l == NULL || rgb_r == NULL || rgb_l == rgb_r);
        if (have_vis && any_new) {
        /* --- T4/T5 Hassenstein-Reichardt correlators (frame-locked) ---
         * single-eye mode: both sides correlate the SAME grid (turn signal
         * becomes the left/right half-field difference = relative motion) */
        float hsum[2] = { 0, 0 };
        float vsumP[2] = { 0, 0 }, vsumN[2] = { 0, 0 }; /* +y-pref / -y-pref */
        int hn[2] = { 0, 0 }, vnP[2] = { 0, 0 }, vnN[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            int cnt = which ? n->t5_n : n->t4_n;
            if (!cnt) continue;
            int64_t *idx = which ? n->t5_idx : n->t4_idx;
            int32_t *dirs = which ? n->t5_dir : n->t4_dir;
            int16_t *hexes = which ? n->t5_hex : n->t4_hex;
            int8_t *sides = which ? n->t5_side : n->t4_side;
            float *drv_out = which ? n->t5_drive : n->t4_drive;
            float *drv = n->drv_scratch;
            memset(drv, 0, (size_t)cnt * 4);
            int hexed = 0;
            for (int k = 0; k < cnt; k++) {
                int side = sides[k];
                const float *g = one_eye_mode ? rgb_l : (side ? rgb_r : rgb_l);
                const float *lag = n->lum_lag[one_eye_mode ? 0 : side];
                int hx = hexes[k * 2], hy = hexes[k * 2 + 1];
                if (hx < 0 || hy < 0 || hx >= W || hy >= H) continue;
                hexed++;
                int d = dirs[k];
                int dx = d == 0 ? 1 : (d == 1 ? -1 : 0);
                int dy = d == 2 ? 1 : (d == 3 ? -1 : 0);
                int bx = hx - dx, by = hy - dy;
                if (bx < 0) bx = 0; if (bx >= W) bx = W - 1;
                if (by < 0) by = 0; if (by >= H) by = H - 1;
                float now_c = g[(hy * W + hx) * 3 + 0] * 0.299f +
                              g[(hy * W + hx) * 3 + 1] * 0.587f +
                              g[(hy * W + hx) * 3 + 2] * 0.114f;
                float back_now = g[(by * W + bx) * 3 + 0] * 0.299f +
                                 g[(by * W + bx) * 3 + 1] * 0.587f +
                                 g[(by * W + bx) * 3 + 2] * 0.114f;
                float back_lag = lag[by * W + bx];
                float now_lag = lag[hy * W + hx];
                drv[k] = now_c * back_lag - back_now * now_lag;
            }
            if (which == 0) n->t4_hcount = hexed; else n->t5_hcount = hexed;
            memcpy(drv_out, drv, (size_t)cnt * 4);
            /* drive T4/T5 + accumulate per-side flow summary.
             * single-eye mode: bucket by FRAME HALF (hx < W/2 = "left"),
             * so the flow difference stays a meaningful turn signal. */
            for (int k = 0; k < cnt; k++) {
                n->i_ext[idx[k]] += n->p.emd_gain * (drv[k] > 0 ? drv[k] : 0);
                int side = sides[k];
                if (one_eye_mode) side = (hexes[k * 2] >= W / 2) ? 1 : 0;
                int d = dirs[k];
                if (d == 0 || d == 1) {
                    hsum[side] += drv[k] > 0 ? drv[k] : 0;
                    hn[side]++;
                } else if (d == 2) {
                    vsumP[side] += drv[k]; /* signed */
                    vnP[side]++;
                } else if (d == 3) {
                    vsumN[side] += drv[k]; /* signed */
                    vnN[side]++;
                }
            }
        }
        /* flow summary: energy for horizontal (both signs pooled),
         * signed bias for vertical — HELD until the next frame */
        float hL = 0, hR = 0, vL = 0, vR = 0;
        if (hn[0]) hL = hsum[0] / (float)hn[0];
        if (hn[1]) hR = hsum[1] / (float)hn[1];
        /* vertical bias = mean(+y-pref) - mean(-y-pref), per eye */
        if (vnP[0] || vnN[0])
            vL = (vnP[0] ? vsumP[0] / (float)vnP[0] : 0) -
                 (vnN[0] ? vsumN[0] / (float)vnN[0] : 0);
        if (vnP[1] || vnN[1])
            vR = (vnP[1] ? vsumP[1] / (float)vnP[1] : 0) -
                 (vnN[1] ? vsumN[1] / (float)vnN[1] : 0);
        n->flow_hL = hL; n->flow_hR = hR; n->flow_vL = vL; n->flow_vR = vR;

        /* --- lag lines: full step per frame (1 frame = 1 correlation
         * interval), so the lag is exactly one frame old --- */
        const float k_lag = 1.0f; /* replaced by exact hold below */
        (void)k_lag;
        for (int side = 0; side < 2; side++) {
            if (!n->frame_new[side]) continue;
            const float *g = side ? rgb_r : rgb_l;
            if (!g) continue;
            if (one_eye_mode) g = rgb_l; /* shared lag line for both sides */
            float *lag = n->lum_lag[one_eye_mode ? 0 : side];
            /* the lag line IS the previous frame (pure delay — the clean
             * Hassenstein-Reichardt formulation at frame cadence) */
            for (int c = 0; c < W * H; c++) {
                float l = g[c * 3 + 0] * 0.299f + g[c * 3 + 1] * 0.587f + g[c * 3 + 2] * 0.114f;
                lag[c] = l;
            }
            n->lag_ok[side] = 1;
            n->frame_new[side] = 0;
        }
        }
    } /* end frame-locked EMD block */

    /* --- noise + integrate + spike --- */
    for (int i = 0; i < N; i++) {
        n->noise[i] = (rng_float01(n) * 2.0f - 1.0f) * n->bio_noise[i];
        float target = FB_V_REST + n->ge[i] - n->gi[i] + n->i_ext[i] + n->noise[i];
        float vmn = n->vm[i] * n->decay_m[i] + (1.0f - n->decay_m[i]) * target;
        float adapt = n->bio_adapt[i] * (n->hz_ema[i] - n->bio_set_hz[i]);
        if (adapt > n->bio_margin[i]) adapt = n->bio_margin[i];
        if (adapt < -n->bio_margin[i]) adapt = -n->bio_margin[i];
        n->vth_eff[i] = n->bio_vth[i] + adapt;
        n->vm_next[i] = vmn;
    }
    for (int i = 0; i < N; i++) {
        if (n->vm_next[i] >= n->vth_eff[i] && now >= n->ref_until[i]) {
            n->fire[i] = 1;
            n->fired[n->fired_n++] = i;
            n->vm[i] = FB_V_RESET;
            n->ref_until[i] = now + n->bio_ref[i];
            n->spike_counts[i]++;
            n->grp_spikes[n->grp[i]] += 1.0;
            spike_count++;
            /* per-side DN EMA */
            float a_dn = 1.0f - expf(-dt / 400.0f);
            if (n->dn_l_mask[i])
                n->dn_l_rate += (1.0f / ((float)n->dn_l_n) / (dt / 1000.0f) - n->dn_l_rate) * a_dn;
            if (n->dn_r_mask[i])
                n->dn_r_rate += (1.0f / ((float)n->dn_r_n) / (dt / 1000.0f) - n->dn_r_rate) * a_dn;
            float k_adapt = dt / 2000.0f;
            n->hz_ema[i] += (1000.0f / dt - n->hz_ema[i]) * k_adapt;
        } else {
            n->fire[i] = 0;
            n->vm[i] = n->vm_next[i];
            float k_adapt = dt / 2000.0f;
            n->hz_ema[i] += (0.0f - n->hz_ema[i]) * k_adapt;
        }
    }

    /* --- route fired neurons by delay --- */
    for (int k = 0; k < n->fired_n; k++) {
        int64_t i = n->fired[k];
        int8_t d = n->delay_ticks[i];
        if (d == 0) {
            push_i64(&n->pend0_next, &n->pend0_next_n, &n->pend0_next_cap, i);
        } else if (d == 1) {
            push_i64(&n->dnext[0], &n->dnext_n[0], &n->dnext_cap[0], i);
        } else if (d == 2) {
            push_i64(&n->dnext[1], &n->dnext_n[1], &n->dnext_cap[1], i);
        }
    }
    /* advance queue rotation */
    { int64_t *t = n->pend0_cur; n->pend0_cur = n->pend0_next; n->pend0_next = t;
      int tn = n->pend0_cur_n; n->pend0_cur_n = n->pend0_next_n; n->pend0_next_n = tn;
      int tc = n->pend0_cap; n->pend0_cap = n->pend0_next_cap; n->pend0_next_cap = tc; }
    for (int d = 0; d < 2; d++) {
        int64_t *t = n->dcur[d]; n->dcur[d] = n->dnext[d]; n->dnext[d] = t;
        int tn = n->dcur_n[d]; n->dcur_n[d] = n->dnext_n[d]; n->dnext_n[d] = tn;
        int tc = n->dcur_cap[d]; n->dcur_cap[d] = n->dnext_cap[d]; n->dnext_cap[d] = tc;
    }

    n->last_tick_spikes = spike_count;
    /* group rates (400 ms EMA) */
    {
        float alpha = 1.0f - expf(-dt / 400.0f);
        for (int g = 0; g < n->G; g++) {
            if (!n->grp_n[g]) continue;
            float hz = (float)(n->grp_spikes[g] / (double)n->grp_n[g]) / (dt / 1000.0f);
            n->pop_rate[g] += ((double)hz - n->pop_rate[g]) * (double)alpha;
            n->grp_spikes[g] = 0.0;
        }
    }

    /* DAN EMA + SELF-REGULATED dopamine + drive decay.
     * dopa_self is the DAN population's own fast-minus-slow activity: an
     * INTERNAL prediction error. dan_base_hz adapts (tau 8 s) to whatever
     * DAN rate is sustained, so constant situations stop teaching while
     * improvements (DAN up) LTP and deterioration (DAN down) LTD — with or
     * without any external reward signal. External input only biases DAN
     * excitability via dan_drive (the sensory pathway), never the error
     * directly. dopa_gain ~0.09/Hz converts deviation to the ±1.5 range. */
    if (n->dan_n) {
        int ds = 0;
        for (int k = 0; k < n->dan_n; k++) ds += n->fire[n->dan_idx[k]];
        float hz_d = (float)ds / (float)n->dan_n / (dt / 1000.0f);
        n->dan_ema_hz += (hz_d - n->dan_ema_hz) * 0.05f;          /* fast ~0.4 s */
        /* WARMUP: while the circuit spins up (pacemaker ramping, external
         * biases decaying) the baseline tracks the fast EMA directly and
         * plasticity is GATED — a start-up or regime shift must never
         * register as reward or punishment. Warmup ends when the baseline
         * has actually converged to the DAN rate. */
        float rel_err = fabsf(n->dan_ema_hz - n->dan_base_hz) /
                        (n->dan_base_hz > 1.0f ? n->dan_base_hz : 1.0f);
        if (n->warming && n->sim_ms > 2000.0f && rel_err < 0.03f)
            n->warming = 0;
        if (n->warming) {
            n->dan_base_hz = n->dan_ema_hz;
            n->dopa_self = 0.0f;
        } else {
            float kb = dt / n->p.dan_base_tau_ms;
            n->dan_base_hz += (n->dan_ema_hz - n->dan_base_hz) * kb; /* slow ~8 s */
            if (n->dan_base_hz < 0.05f) n->dan_base_hz = 0.05f;
            float err = (n->dan_ema_hz - n->dan_base_hz) * n->p.dopa_gain;
            if (err > 1.5f) err = 1.5f;
            if (err < -1.5f) err = -1.5f;
            n->dopa_self = err;
        }
    }
    if (n->dan_drive != 0.0f) {
        n->dan_drive *= expf(-dt / 300.0f);
        if (fabsf(n->dan_drive) < 1e-3f) n->dan_drive = 0.0f;
    }
    /* mechanosensory burst decays with receptor-like kinetics (~40 ms) */
    if (n->touch_blast_mV != 0.0f) {
        n->touch_blast_mV *= expf(-dt / 40.0f);
        if (fabsf(n->touch_blast_mV) < 1e-3f) n->touch_blast_mV = 0.0f;
    }
    if (n->reward != 0.0f) {
        n->reward *= expf(-dt / n->p.tau_dopa_ms);
        if (fabsf(n->reward) < 1e-3f) n->reward = 0.0f;
    }

    /* motor pools: integrate member spikes + move plastic weights with
     * THIS tick's dopamine error (the teaching signal R-STDP follows) */
    if (n->n_pools > 0) fb_pools_tick(n, n->dopa_self);

    /* corruption tripwire */
    if (!isfinite(n->vm[0]) || !isfinite(n->reward)) {
        n->corrupted = 1;
        for (int i = 0; i < N; i++) { n->vm[i] = FB_V_REST; n->ge[i] = n->gi[i] = 0; n->hz_ema[i] = n->bio_set_hz[i]; }
        n->reward = 0; n->dan_drive = 0; n->last_tick_spikes = 0;
    } else {
        n->corrupted = 0;
    }

    n->sim_ms += dt;
    n->tick_count++;
    n->ticks_last_frame++;
    return spike_count;
}

/* -------------------------------------------------------------- learning */

void fb_learn_step(FbCircuit *n) {
    const FbConnectome *c = n->c;
    if (n->plastic_n == 0) {
        memset(n->spike_counts, 0, (size_t)c->N * sizeof(int32_t));
        n->ticks_last_frame = 0;
        return;
    }
    float frame_ms = (float)n->ticks_last_frame * (float)FB_DT_MS;
    if (frame_ms <= 0) return;
    float decay_e = expf(-frame_ms / n->p.tau_elig_ms);
    /* eligibility: coactivity of pre/post spike counts this frame */
    for (int k = 0; k < n->plastic_n; k++) {
        int64_t e = n->plastic_idx[k];
        int pre = n->spike_counts[c->edge_src[e]];
        int post = n->spike_counts[c->edge_dst[e]];
        if (pre > 0 && post > 0) {
            float add = (float)(pre < 8 ? pre : 8) *
                        (1.0f + 0.2f * (float)(post < 8 ? post : 8));
            /* cap the eligibility trace: with a persistent reward the sum
             * grows linearly and saturates every synapse to w_max within
             * seconds; real eligibility is a DECAYING coincidence signal */
            double e2 = n->elig[k] + (double)add;
            n->elig[k] = e2 > 1.5 ? 1.5 : e2;
        }
        n->elig[k] *= (double)decay_e;
        if (fabs(n->elig[k]) < 1e-3) n->elig[k] = 0.0;
    }
    /* per-side DN frame counts for telemetry */
    {
        double sl = 0, sr = 0;
        int nl = 0, nr = 0;
        for (int k = 0; k < n->plastic_n; k++) {
            int64_t e = n->plastic_idx[k];
            int64_t tgt = c->edge_dst[e];
            if (c->side[tgt] == 1) { sr += n->spike_counts[tgt]; nr++; }
            else { sl += n->spike_counts[tgt]; nl++; }
        }
        n->last_dn_l = nl ? (float)(sl / nl) : 0.0f;
        n->last_dn_r = nr ? (float)(sr / nr) : 0.0f;
    }
    if (n->learn_on && !n->warming) {
        /* teaching signal = SELF-REGULATED dopamine (DAN deviation from its
         * internal baseline). External reward already did its job upstream:
         * it biased DAN excitability, which moves dan_ema_hz, which moves
         * this error. A reward injection with no effect on DAN activity
         * teaches nothing — that is the point. */
        float g = n->dopa_self;
        if (g != 0.0f) {
        float a = (g > 0 ? n->p.a_ltp : n->p.a_ltd) * g * (frame_ms / 1000.0f);
        for (int k = 0; k < n->plastic_n; k++) {
            double eff = g > 0 ? n->elig[k] : (n->elig[k] > 0 ? n->elig[k] : 0.0);
            if (eff == 0.0) continue;
            double w = n->w_mult[k];
            /* SOFT bounds: movement scales with the room left in that
             * direction (so a synapse never freezes at a wall and an
             * already-edited synapse remains fully editable), and the
             * reverse direction always moves at FULL strength (so any
             * synapse can always be pulled back). */
            double room = (a > 0) ? (n->p.w_max - w) : (w - n->p.w_min);
            double full = n->p.w_max - n->p.w_min;
            double move = a * eff;
            if (move > 0) move *= room / full;
            double wm = w + move;
            if (wm < n->p.w_min) wm = n->p.w_min;
            if (wm > n->p.w_max) wm = n->p.w_max;
            if (wm != n->w_mult[k]) {
                n->w_mult[k] = wm;
                n->wm_edge[n->plastic_idx[k]] = wm;
            }
        }
        }
    }
    n->cum_reward += (double)n->dopa_self * (double)(frame_ms / 1000.0f);

    /* Turrigiano scaling, every scale_every learning steps */
    if (++n->scale_phase >= n->p.scale_every) {
        n->scale_phase = 0;
        for (int t = 0; t < n->n_targets; t++) {
            int lo = n->scale_bounds[t], hi = n->scale_bounds[t + 1];
            if (hi - lo < 8) continue;
            double s = 0;
            for (int k = lo; k < hi; k++) s += n->w_mult[n->scale_pos[k]];
            double want = (double)(hi - lo);
            double sc = 1.0 + (double)n->p.k_scale * (want / (s > 1e-6 ? s : 1e-6) - 1.0);
            if (sc > 0.9999 && sc < 1.0001) continue;
            for (int k = lo; k < hi; k++) {
                int pos = n->scale_pos[k];
                double wm = n->w_mult[pos] * sc;
                if (wm < n->p.w_min) wm = n->p.w_min;
                if (wm > n->p.w_max) wm = n->p.w_max;
                if (wm != n->w_mult[pos]) {
                    n->w_mult[pos] = wm;
                    n->wm_edge[n->plastic_idx[pos]] = wm;
                }
            }
        }
    }
    memset(n->spike_counts, 0, (size_t)c->N * sizeof(int32_t));
    n->ticks_last_frame = 0;
}

/* ---------------------------------------------------------------- memory */

int fb_learn_stats(const FbCircuit *n, float *lo, float *hi) {
    if (n->plastic_n == 0) { *lo = *hi = 1.0f; return 0; }
    int edited = 0;
    /* count only synapses whose change is meaningful (>1% of range):
     * Turrigiano homeostasis perturbs nearly every synapse by epsilon, and
     * reporting that would make "edited" jump to ~100% within seconds */
    double thr = 0.01 * ((double)n->p.w_max - (double)n->p.w_min);
    for (int k = 0; k < n->plastic_n; k++) {
        double d = n->w_mult[k] - 1.0;
        if (d > thr || d < -thr) edited++;
    }
    *lo = (float)n->w_mult[0];
    *hi = (float)n->w_mult[0];
    for (int k = 1; k < n->plastic_n; k++) {
        if (n->w_mult[k] < *lo) *lo = (float)n->w_mult[k];
        if (n->w_mult[k] > *hi) *hi = (float)n->w_mult[k];
    }
    return edited;
}

int fb_export_memory(const FbCircuit *n, int **out_idx, double **out_w, int *out_n) {
    int cnt = 0;
    for (int k = 0; k < n->plastic_n; k++)
        if (n->w_mult[k] != 1.0) cnt++;
    *out_n = cnt;
    if (!cnt) { *out_idx = NULL; *out_w = NULL; return 0; }
    *out_idx = (int *)malloc((size_t)cnt * sizeof(int));
    *out_w = (double *)malloc((size_t)cnt * sizeof(double));
    int j = 0;
    for (int k = 0; k < n->plastic_n; k++) {
        if (n->w_mult[k] != 1.0) {
            (*out_idx)[j] = (int)n->plastic_idx[k];
            (*out_w)[j] = n->w_mult[k];
            j++;
        }
    }
    return cnt;
}

void fb_reset_plasticity(FbCircuit *n) {
    for (int k = 0; k < n->plastic_n; k++) n->w_mult[k] = 1.0;
    memset(n->elig, 0, (size_t)n->plastic_n * sizeof(double));
    for (int e = 0; e < n->c->E; e++) n->wm_edge[e] = 1.0;
    fb_pools_reset(n);   /* learned body map resets with the synapses */
    n->reward = 0; n->dan_drive = 0; n->cum_reward = 0;
    /* re-warm: re-baseline the DAN so the wipe itself teaches nothing */
    n->warming = 1;
    n->dopa_self = 0;
}

/* binary search in plastic_sorted; returns position or -1 */
static int find_plastic(const FbCircuit *n, int64_t edge_id) {
    int lo = 0, hi = n->plastic_n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (n->plastic_sorted[mid] == edge_id) return mid;
        if (n->plastic_sorted[mid] < edge_id) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

int fb_load_memory(FbCircuit *n, const int *idx, const double *w, int count) {
    fb_reset_plasticity(n);
    if (!idx || !w || count <= 0) return 0;
    int hits = 0;
    for (int i = 0; i < count; i++) {
        int sorted_pos = find_plastic(n, (int64_t)idx[i]);
        if (sorted_pos < 0) continue;
        int k = n->plastic_order[sorted_pos];
        double wc = w[i];
        if (wc < n->p.w_min) wc = n->p.w_min;
        if (wc > n->p.w_max) wc = n->p.w_max;
        n->w_mult[k] = wc;
        n->wm_edge[n->plastic_idx[k]] = wc;
        hits++;
    }
    return hits;
}

/* ------------------------------------------------- motor pools (see top) */

int fb_pools_configure(FbCircuit *n, const FbPoolCfg *cfgs, int n_cfgs,
                       int integrate_ms, int learn, float pool_lr) {
    for (int i = 0; i < n->n_pools; i++) {
        free(n->pools[i].ids);
        free(n->pools[i].widx);
        free(n->pools[i].w);
    }
    free(n->pools);
    n->pools = NULL;
    n->n_pools = 0;
    if (!cfgs || n_cfgs <= 0) return 0;
    n->pool_lr = pool_lr > 0.0f ? pool_lr : 0.08f;   /* a_ltp per second */
    int tau_ms = integrate_ms > 0 ? integrate_ms : 80;
    FbPool *p = (FbPool *)calloc((size_t)n_cfgs, sizeof(FbPool));
    if (!p) return 0;
    int built = 0;
    for (int i = 0; i < n_cfgs; i++) {
        const FbPoolCfg *pc = &cfgs[i];
        int g = fb_group_index(n, pc->group);
        if (g < 0) continue;
        int total = 0;
        for (int j = 0; j < n->c->N; j++) if (n->grp[j] == g) total++;
        if (total <= 0) continue;
        int64_t *ids = (int64_t *)malloc((size_t)total * sizeof(int64_t));
        if (!ids) continue;
        int m = 0;
        if (pc->split_lr) {
            /* partition by connectome side; motor (VNC) neurons are all
             * side==1, so fall back to every-2 there (still a signed pair) */
            int with_side = 0, ones = 0;
            for (int j = 0; j < n->c->N; j++) {
                if (n->grp[j] != g) continue;
                with_side++;
                if (n->c->side[j] == (uint8_t)pc->which) ones++;
            }
            int want = pc->which ? ones : with_side - ones;
            if (want > 0 && want < with_side) {
                for (int j = 0; j < n->c->N; j++)
                    if (n->grp[j] == g && n->c->side[j] == (uint8_t)pc->which)
                        ids[m++] = j;
            } else {
                /* side is degenerate on this group: id-order half split */
                int want = (pc->which ? (total + 1) / 2 : total / 2);
                int seen = 0;
                for (int j = 0; j < n->c->N; j++) {
                    if (n->grp[j] != g) continue;
                    if (pc->which ? seen >= total - want : seen < want)
                        ids[m++] = j;
                    seen++;
                }
            }
        } else if (pc->every > 1) {
            for (int j = 0; j < n->c->N; j++) {
                if (n->grp[j] == g && (j % pc->every) == 0) ids[m++] = j;
            }
        } else {
            for (int j = 0; j < n->c->N; j++)
                if (n->grp[j] == g) ids[m++] = j;
        }
        if (m <= 0) { free(ids); continue; }
        p[built].ids = ids;
        p[built].n = m;
        p[built].learn = learn ? 1 : 0;
        p[built].lr_scale = pc->lr_scale > 0.0f ? pc->lr_scale : 1.0f;
        p[built].set_hz = n->bio_set_hz[ids[0]] > 0.5f
            ? n->bio_set_hz[ids[0]] : 2.0f;
        if (learn) {
            p[built].widx = (int64_t *)malloc((size_t)m * sizeof(int64_t));
            p[built].w = (double *)malloc((size_t)m * sizeof(double));
            if (!p[built].widx || !p[built].w) {
                free(p[built].widx); free(p[built].w);
                free(ids);
                p[built].ids = NULL;
                continue;
            }
            for (int k = 0; k < m; k++) { p[built].widx[k] = ids[k]; p[built].w[k] = 1.0; }
        }
        p[built].decay = expf(-(float)FB_DT_MS / (float)tau_ms);
        built++;
    }
    if (!built) { free(p); return 0; }
    n->pools = p;
    n->n_pools = built;
    return built;
}

FbPool *fb_runtime_pools(FbCircuit *n) { return n->pools; }
int fb_pools_count(const FbCircuit *n) { return n->n_pools; }

/* per tick: fold this tick's pool-neuron spikes into the accumulators and
 * learn. weight_t = weight_{t-1} + lr*dopa*(spike_t - tonic_rate*dt).
 * The (spike - tonic) term is the anti-wedging rule: a neuron at its tonic
 * rate contributes ~0 per tick regardless of dopa sign, so weights stop
 * moving in constant situations; dopa<0 plus ABOVE-tonic firing depresses
 * the pool. lr matches the synapse LTP rate per second. */
static void fb_pools_tick(FbCircuit *n, float dopa) {
    const float dt = (float)FB_DT_MS;
    const float lr = n->pool_lr / 1000.0f * dt;   /* a_ltp per tick (readout.poolLr, default 0.08) */
    for (int i = 0; i < n->n_pools; i++) {
        FbPool *pl = &n->pools[i];
        const float plr = lr * pl->lr_scale;
        pl->acc *= pl->decay;
        if (pl->acc < 1e-5f) pl->acc = 0.0f;
        if (dopa != 0.0f && pl->learn) {
            const float base = pl->set_hz * dt / 1000.0f; /* tonic spikes/tick */
            for (int k = 0; k < pl->n; k++) {
                const int64_t j = pl->ids[k];
                const float act = (float)(n->spike_counts[j] > 0) - base;
                if (act == 0.0f) continue;
                double w = pl->w[k];
                double room = (dopa > 0.0f) ? (n->p.w_max - w)
                                            : (w - n->p.w_floor);
                double full = (double)n->p.w_max - (double)n->p.w_floor;
                double move = (double)plr * (double)dopa * (double)act;
                if (move > 0) move *= room / full;
                double wm = w + move;
                if (wm < n->p.w_floor) wm = n->p.w_floor;
                if (wm > n->p.w_max) wm = n->p.w_max;
                pl->w[k] = wm;
            }
        }
        for (int k = 0; k < pl->n; k++) {
            const int64_t j = pl->ids[k];
            if (n->spike_counts[j] > 0)
                pl->acc += pl->w ? (float)pl->w[k] : 1.0f;
        }
    }
}

/* frame-end: hand the leaky integrals to the runtime readout.
 * NORMALIZED by member count: a pool's value is the mean per-neuron spike
 * integral over the window (rate x tau), so pools of different sizes are
 * comparable and the embodiment gain means the same thing for any pick. */
void fb_pools_snapshot(const FbCircuit *n, float *out, int cap) {
    int m = n->n_pools < cap ? n->n_pools : cap;
    for (int i = 0; i < m; i++)
        out[i] = n->pools[i].n > 0 ? n->pools[i].acc / (float)n->pools[i].n : 0.0f;
}

int fb_pools_export(FbCircuit *n, char (*names)[32], int max_names,
                    double **out_w, int *out_n) {
    int cnt = 0;
    for (int i = 0; i < n->n_pools; i++) {
        FbPool *pl = &n->pools[i];
        if (!pl->learn || !pl->w) continue;
        for (int k = 0; k < pl->n; k++) {
            if (pl->w[k] == 1.0) continue;
            if (cnt < max_names) {
                snprintf(names[cnt], 32, "%d:%s", (int)pl->widx[k],
                         n->c->groups[n->grp[pl->widx[k]]]);
            }
            cnt++;
        }
    }
    *out_n = cnt;
    if (!cnt) { *out_w = NULL; return 0; }
    if (cnt > max_names) cnt = max_names;
    double *w = (double *)malloc((size_t)cnt * sizeof(double));
    if (!w) { *out_n = 0; return 0; }
    int j = 0;
    for (int i = 0; i < n->n_pools && j < cnt; i++) {
        FbPool *pl = &n->pools[i];
        if (!pl->learn || !pl->w) continue;
        for (int k = 0; k < pl->n && j < cnt; k++) {
            if (pl->w[k] == 1.0) continue;
            w[j++] = pl->w[k];
        }
    }
    *out_w = w;
    return cnt;
}

int fb_pools_import(FbCircuit *n, char (*names)[32], const double *w, int count) {
    int hits = 0;
    for (int i = 0; i < n->n_pools; i++) {
        FbPool *pl = &n->pools[i];
        if (!pl->learn || !pl->w) continue;
        for (int k = 0; k < pl->n; k++) {
            char key[36];
            snprintf(key, sizeof(key), "%d:%s", (int)pl->widx[k],
                     n->c->groups[n->grp[pl->widx[k]]]);
            for (int m = 0; m < count; m++) {
                if (strncmp(names[m], key, 32) == 0) {
                    double wc = w[m];
                    if (wc < n->p.w_min) wc = n->p.w_min;
                    if (wc > n->p.w_max) wc = n->p.w_max;
                    pl->w[k] = wc;
                    hits++;
                    break;
                }
            }
        }
    }
    return hits;
}

void fb_pools_reset(FbCircuit *n) {
    for (int i = 0; i < n->n_pools; i++) {
        FbPool *pl = &n->pools[i];
        if (pl->w)
            for (int k = 0; k < pl->n; k++) pl->w[k] = 1.0;
        pl->acc = 0.0f;
    }
}
