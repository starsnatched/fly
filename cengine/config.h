/* Server config (JSON), mirroring config/flybrain.json + profiles/*.json. */
#ifndef FB_CONFIG_H
#define FB_CONFIG_H

#include "json.h"

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
    /* readout: embodiment channel SEMANTICS + optomotor reflex gains only.
     * All behavior originates in the circuit (EMD flow, DN steering, R-STDP
     * memory); these knobs shape how neural state maps onto actuator names. */
    float dn_hz_scale;
    float hover_throttle, hover_pitch;
    float alt_damp;           /* body-rate damping from the body's own IMU */
    float cruise_pitch;       /* forward tilt at full brain drive */
    float turn_gain;          /* DN/EMD turn signal -> yaw channel gain */
    float opto_yaw_gain;      /* whole-field horizontal flow -> gaze stabilization */
    float opto_fwd_gain;      /* whole-field flow -> forward-speed regulation */
    float opto_climb_gain;    /* vertical flow -> climb response (height control) */
    /* actuators (channel order matters: it defines the WS action frame) */
    char channels[8][32];
    float ch_lo[8], ch_hi[8], ch_slew[8], ch_default[8];
    int n_channels;
    /* reward pathway (NO reward shaping: only how /reward scales into the
     * DAN excitability input and how fast that input decays) */
    float reward_gain;        /* /reward v -> DAN bias v*reward_gain */
    float reward_decay_per_s; /* decay of the DAN bias (pulse semantics) */
    /* network */
    int ws_port, rest_port;
} FbConfig;

/* load config + optional profile overlay; returns 0 on success */
int fb_config_load(FbConfig *cfg, const char *path, const char *profile_path);

#endif
