/* FullCircuit: the entire MaleCNS connectome, spiking (spiking connectome engine). */
#ifndef FB_CIRCUIT_H
#define FB_CIRCUIT_H

#include "flybrain.h"

#define FB_DT_MS 2.0
#define FB_V_REST (-70.0f)
#define FB_V_RESET (-62.0f)

typedef struct FbPool FbPool;   /* defined in circuit.c (motor pools) */

typedef struct {
    float g_scale;        /* peak PSP (mV) per unit normalized weight */
    float tgt_budget;     /* total PSP stamp per target neuron (mV) */
    float ph_tonic;       /* photoreceptor tonic gain (mV per luminance) */
    float ph_optic_gain;  /* ambient optic-lobe tonic (mV per luminance) */
    float emd_gain;       /* T4/T5 motion detector gain (mV per corr) */
    float ph_abs;         /* photoreceptor transient gain */
    float ph_contrast;    /* contrast tanh slope */
    float ph_tau_ms;      /* luminance adaptation */
    float inh_gain;       /* mV global inhibition per unit fraction */
    float dan_mod_gain;   /* DAN neuromodulation mV */
    float dopa_gain;      /* self-dopa: 1/(Hz) scaling of DAN deviation */
    float dan_base_tau_ms;/* slow baseline the DAN adapts to (ms) */
    float dan_tonic_mv;   /* DAN pacemaker depolarization (spontaneous firing) */
    float tau_elig_ms;    /* eligibility trace window */
    float tau_dopa_ms;    /* dopamine decay */
    float a_ltp, a_ltd;   /* learning rates */
    float w_min, w_max;   /* multiplicative weight bounds (synapses + pools) */
    float w_floor;        /* POOL-only lower bound << w_min: a fully-depressed
                           * pool goes near-silent (≈0 drive) but keeps
                           * headroom to re-potentiate; clamping pools at
                           * w_min pinned the rectified readout at its rail
                           * and starved the reward loop (thrust=0 bug) */
    float k_scale;        /* Turrigiano per learning step */
    int scale_every;      /* learning steps between scaling passes */
    uint32_t seed;        /* xorshift state */
} FbParams;

typedef struct {
    /* per-group biophysics (tau_m, v_th, noise, t_ref, adapt, set_hz) */
    float bio_tau, bio_vth, bio_noise, bio_ref, bio_adapt, bio_set_hz;
} FbBio;

typedef struct {
    const FbConnectome *c;
    FbParams p;

    int G;                /* group count */

    /* per-neuron (N floats/ints) */
    int32_t *grp;         /* group index per neuron */
    float *bio_tau, *bio_vth, *bio_noise, *bio_ref, *bio_adapt;
    float *bio_set_hz, *bio_base, *bio_margin, *decay_m;
    float *vm, *ge, *gi, *hz_ema;
    double *ref_until;
    uint8_t *spiked;
    int32_t *spike_counts;
    int8_t *delay_ticks;

    /* edge arrays (CSR by source) */
    int64_t *out_edge;    /* CSR order edge ids */
    int64_t *out_dst;     /* CSR order dst */
    float *out_w;         /* CSR order base weight (normalized, capped 2.0) */
    int8_t *out_sign;     /* CSR order sign */
    int64_t *out_start;   /* N+1 */

    /* synapse state */
    float *dep_res;       /* E, Tsodyks-Markram resource */

    /* weight multiplier per edge (learning): 1.0 = untouched */
    double *wm_edge;

    /* per-group stats */
    int64_t *grp_n;
    double *grp_spikes, *pop_rate;

    /* photoreceptor mapping: lamina neurons with hex coords, per side */
    int32_t *lam_side;    /* per mapped lamina neuron: 0 left / 1 right */
    int64_t *lam_neurons; /* neuron ids */
    int64_t *lam_cell;    /* cell index in the (lam_h x lam_w) grid */
    int lam_n;
    int lam_w, lam_h;

    /* T4/T5 directional drive */
    int64_t *t4_idx, *t5_idx;
    int32_t *t4_dir, *t5_dir;
    int16_t *t4_hex, *t5_hex;   /* pairs (hex1, hex2), raw incl. -1 */
    int t4_n, t5_n;
    int8_t *t4_side, *t5_side;
    float *t4_drive, *t5_drive;
    int t4_hcount, t5_hcount;   /* hexed counts per side (telemetry sanity) */

    /* EMD lag lines per eye (lam_h x lam_w) */
    float *lum_lag[2];
    int lag_ok[2];
    /* set by the runtime when a NEW frame arrived for that eye: the EMD
     * correlator + lag update run once per frame, and the flow summary holds
     * for the frame duration (motion signals are frame-locked, not per-tick) */
    int frame_new[2];

    /* adapted luminance per mapped lamina neuron (photoreceptor state) */
    float *lam_col_buf;

    /* optic-lobe ambient mask */
    uint8_t *optic_mask;

    /* mechanosensory population ("sensory" superclass) + burst state */
    uint8_t *sensory_set;
    float touch_blast_mV;

    /* DAN pathway */
    int64_t *dan_idx;
    int dan_n;
    uint8_t *dan_set, *dan_tgt;
    float dan_ema_hz, dan_drive;
    /* SELF-REGULATED dopamine: the teaching signal is the DAN population's
     * own fast activity (dan_ema_hz) relative to its slow internal baseline
     * (dan_base_hz) — an internal prediction error, not an external scalar.
     * External reward signals only bias DAN excitability (dan_drive). */
    float dan_base_hz;
    float dopa_self;
    int warming;          /* baseline still converging: plasticity gated */

    /* plasticity (subset of edges into descending) */
    int64_t *plastic_idx;   /* edge ids (owned copy) */
    int plastic_n;
    double *w_mult, *elig;
    int64_t *plastic_targets;
    int *scale_pos;         /* positions sorted by target */
    int *scale_bounds;      /* boundaries into scale_pos (n_targets+1) */
    int n_targets;

    /* per-side descending EMAs (steering readout) */
    int dn_l_n, dn_r_n;
    uint8_t *dn_l_mask, *dn_r_mask;
    float dn_l_rate, dn_r_rate;

    /* motor pools: DIRECT per-neuron actuator channels (readout.pools).
     * Each pool integrates spikes from its selected neurons into a leaky
     * accumulator — motor-unit temporal summation — and that integral is
     * the raw actuator signal (per-embodiment gain/offset are applied in
     * the runtime readout). Pool weights are PLASTIC: shaped by the same
     * self-regulated dopamine signal that drives R-STDP, so the circuit
     * learns its own body map. The pool array is owned by the circuit;
     * the runtime points fb_runtime_pools at it and configures it. */
    FbPool *pools;
    int n_pools;
    float pool_lr;        /* pool weight learning rate (a_ltp per second;
                          * readout.poolLr, default 0.08) */

    /* delay queues: flat id arrays; slot[k] delivers after k+1 more ticks */
    int64_t *pend0_cur, *pend0_next;
    int pend0_cur_n, pend0_next_n, pend0_cap, pend0_next_cap;
    int64_t *dcur[2], *dnext[2]; /* d=0 -> 1-tick, d=1 -> 2-tick */
    int dcur_n[2], dnext_n[2], dcur_cap[2], dnext_cap[2];

    /* RNG (xorshift128+) */
    uint64_t rng_s0, rng_s1;

    /* incoming conductance accumulators (delivered -> ge/gi each tick) */
    float *ge_inc, *gi_inc;

    /* scalar state */
    double sim_ms;
    int64_t tick_count;
    int ticks_last_frame;
    int last_tick_spikes;
    int corrupted;
    float reward, cum_reward;
    int learn_on;
    int scale_phase;
    float last_dn_l, last_dn_r;
    float flow_hL, flow_hR, flow_vL, flow_vR; /* EMD summary per tick */

    /* plasticity lookup: plastic_idx sorted + order (for memory import) */
    int64_t *plastic_sorted;
    int *plastic_order;

    /* scratch (N-sized) reused per tick */
    float *i_ext, *noise, *vm_next, *vth_eff;
    uint8_t *fire;
    int64_t *fired;
    int fired_n;
    float *drv_scratch;     /* max(t4_n, t5_n) */
} FbCircuit;

/* default params mirroring CircuitParams() in circuit.py */
void fb_params_default(FbParams *p);

FbCircuit *fb_circuit_new(const FbConnectome *c, const FbParams *p);
void fb_circuit_free(FbCircuit *n);

/* one dt=2 ms tick; rgb_l/rgb_r are (h, w, 3) float 0..1 or NULL.
 * returns spikes this tick. */
int fb_tick(FbCircuit *n, const float *rgb_l, const float *rgb_r,
            int gw, int gh);

/* plasticity step (once per frame) */
void fb_learn_step(FbCircuit *n);

void fb_apply_reward(FbCircuit *n, float r);
void fb_circuit_sensory_burst(FbCircuit *n, float mv);
void fb_apply_dan_bias(FbCircuit *n, float r);
void fb_set_learning(FbCircuit *n, int on);

/* group helpers */
int fb_group_index(const FbCircuit *n, const char *name);
float fb_rate_of(const FbCircuit *n, const char *name);

/* steering + flow readouts */
float fb_dn_steer(const FbCircuit *n);
void fb_emd_flow(const FbCircuit *n, float *hL, float *hR, float *vL, float *vR);

/* memory export/import (JSON-able arrays) */
int fb_learn_stats(const FbCircuit *n, float *lo, float *hi);
int fb_export_memory(const FbCircuit *n, int **out_idx, double **out_w, int *out_n);
int fb_load_memory(FbCircuit *n, const int *idx, const double *w, int count);
void fb_reset_plasticity(FbCircuit *n);

/* ---- motor pools (readout.pools): direct neuron->actuator channels ----
 *
 * A pool is a set of neurons from ONE group whose spikes integrate into a
 * leaky accumulator; the accumulator IS the actuator signal (the runtime
 * readout scales it per embodiment). With "learn":true the pool's neurons
 * own a plastic dopamine-gated weight each (multiplicative, same soft
 * bounds as synapse learning), so the brain calibrates its own body map.
 *
 * config:  pools JSON array -> FbPoolCfg descriptions
 * returns: number of pools built (0 = none declared / no group)
 *
 * pick selects from the group's neuron ids in ascending id order:
 *   {"every":k}            one neuron of every k (even coverage)
 *   {"split":"lr"}         neurons partitioned by connectome side:
 *                          pool 0 = left half, pool 1 = right half
 *                          (a pool with "pool":1 takes the right half)
 *   default               the whole group
 * Side note: "motor" neurons have side==1 (VNC); split-lr falls back to
 * "every":2 there, still giving a signed differential pair. */
typedef struct {
    char group[32];
    int every;        /* >0: take one of every k */
    int split_lr;     /* 1: partition by connectome side (pool 0 = left) */
    int which;        /* which half for split_lr (0 = left, 1 = right) */
    float lr_scale;   /* per-pool learning-rate multiplier (default 1.0) */
} FbPoolCfg;

int fb_pools_configure(FbCircuit *n, const FbPoolCfg *cfgs, int n_cfgs,
                       int integrate_ms, int learn, float pool_lr);
FbPool *fb_runtime_pools(FbCircuit *n);   /* live pool array (or NULL) */
int fb_pools_count(const FbCircuit *n);
/* current leaky integrals (called with the runtime lock held) */
void fb_pools_snapshot(const FbCircuit *n, float *out, int cap);

/* memory (pool weights ride in the same JSON file as synapse memory) */
int fb_pools_export(FbCircuit *n, char (*names)[32], int max_names,
                    double **out_w, int *out_n);
int fb_pools_import(FbCircuit *n, char (*names)[32], const double *w, int count);
void fb_pools_reset(FbCircuit *n);

#endif
