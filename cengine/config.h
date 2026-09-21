/* Server config (JSON), mirroring config/flybrain.json + profiles. */
#ifndef FB_CONFIG_H
#define FB_CONFIG_H

#include "json.h"

/* One decode-map entry: channel = offset + gain * signal (linear), or
 * offset + gain * tanh(signal) when "shape":"tanh" is declared. The tanh
 * option is an OUTPUT nonlinearity for the embodiment: neural differentials
 * (steering asymmetry, optic flow) are millivolts-wide next to a 0..1
 * channel range, so an embodiment may declare amplification-with-
 * saturation. The set of signals is fixed (population readouts from
 * circuit.h's decoders); which population feeds which actuator name is
 * per-embodiment DECLARATION in config (readout.map) — anatomy, not
 * policy. */
typedef struct {
    char channel[32];
    char signal[32];
    float gain, offset;
    int shape;                /* 0 = linear, 1 = tanh */
    int has_norm;             /* 1 = rescale to [-1,1] (readout norm) */
    float norm_span;          /* half-range pre-normalization */
    float norm_center;        /* subtracted first: bool true = 0.5 (unipolar
                               * pools), a number = that value (measured
                               * resting integral), absent = 0 */
} FbMapEntry;

#define FB_MAX_MAP 32

/* One motor-pool declaration (readout.pools[]): a DIRECT neuron->actuator
 * channel. The pool's neurons are selected from one connectome group;
 * their spikes integrate into a leaky accumulator that BECOMES the raw
 * channel signal (per-embodiment gain/offset are applied on top). With
 * "learn":true the pool's weights are plastic — shaped by the circuit's
 * own dopamine error like R-STDP synapses — so the brain calibrates its
 * own body map. Pool i maps to actuator "pool0"/"pool1"/... in order. */
typedef struct {
    char group[32];
    int every;        /* >1: one neuron of every k (ascending id order) */
    int split_lr;     /* 1: partition by connectome side (which: 0 left) */
    int which;        /* split half selector (0 = left, 1 = right) */
    float lr_scale;   /* per-pool learning-rate multiplier (default 1.0) */
} FbPoolEntry;

typedef struct {
    /* engine */
    char binary[512];
    float g_scale, tgt_budget, ph_tonic, ph_optic_gain, emd_gain, dan_mod_gain;
    float dopa_gain, dan_base_tau_s, dan_tonic_mv;
    int learning;
    float tick_cost_seed_ms;
    /* runtime */
    float sim_speed, max_bio_ms;
    float autosave_s;
    char memory_path[512];
    /* sensors */
    int eye_count;            /* 2 = stereo pair, 1 = single forward-facing eye */
    int eye_left_id, eye_right_id; /* which client eye ids map to slots 0/1 */
    float vision_hz;          /* expected eye-frame rate (drive quality / introspection) */
    float range_alt[2], range_speed[2], range_vy[2], range_clr[2];
    float tau_scale; /* optic-flow->clearance gain: clr = tau_scale / flow */
    /* mechanosensory (touch/collision) pathway */
    float touch_gain;         /* burst amplitude in mV of extra sensory current
                                 (decays with ~40 ms receptor kinetics) */
    /* readout: DN rate normalization + the DECLARED decode map
     * (readout.map[]). There are no behavioral knobs: all behavior is the
     * circuit's own; the map only says which population a channel listens
     * to and how loudly (channel = offset + gain*signal). */
    float dn_hz_scale;
    FbMapEntry map[FB_MAX_MAP];
    int n_map;
    /* direct motor pools (readout.pools): neuron-level actuator channels */
    FbPoolEntry pools[8];
    int n_pools;
    int pool_integrate_ms;   /* leaky-accumulator window (default 80 ms) */
    int pool_learn;          /* plastic pool weights (default: engine learning) */
    float pool_lr;           /* pool weight learning rate (readout.poolLr,
                             * default 0.08 — the historical hard-coded value) */
    /* actuators (channel order matters: it defines the WS action frame) */
    char channels[8][32];
    float ch_lo[8], ch_hi[8], ch_slew[8], ch_default[8];
    int n_channels;
    /* reward pathway (NO reward shaping: only how /reward scales into the
     * DAN excitability input; the bias decays inside the circuit) */
    float reward_gain;        /* /reward v -> DAN bias v*reward_gain */
    /* network */
    int ws_port, rest_port;
} FbConfig;

/* load config + optional profile overlay; returns 0 on success */
int fb_config_load(FbConfig *cfg, const char *path, const char *profile_path);

/* forward declaration: runtime.h includes this header and completes the type */
struct FbRuntime;

/* Re-apply an embodiment profile ON TOP of an already-loaded config and
 * push its embodiment-scoped parts (actuator channels, readout map,
 * sensors, reward) into a live runtime. The engine/neural sections are
 * intentionally ignored (changing them mid-flight would corrupt circuit
 * invariants). Memory/ports are untouched. */
void fb_config_apply_profile_to_runtime(FbConfig *cfg, struct FbRuntime *rt,
                                        const char *profile_path);

#endif
