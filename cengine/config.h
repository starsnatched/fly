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
    float reward_period_s, autosave_s;
    char memory_path[512];
    /* sensors */
    int eye_count;            /* 2 = stereo pair, 1 = single forward-facing eye */
    int eye_left_id, eye_right_id; /* which client eye ids map to slots 0/1 */
    float range_alt[2], range_speed[2], range_vy[2], range_clr[2];
    /* readout */
    float dn_hz_scale;
    float hover_throttle, hover_pitch;
    /* flight structure: altitude hold + spontaneous saccades/wander
     * (all embodiment-tunable via config/profiles) */
    float alt_gain, alt_damp, target_alt;   /* altitude hold: climb-rate = altGain*(target-alt), damped by altDamp*(climbTarget-vy) */
    float cruise_pitch;                     /* forward cruise command */
    float turn_gain;                        /* EMD flow imbalance -> avoid strength */
    float saccade_rate;                     /* spontaneous saccades per second */
    float saccade_yaw, saccade_roll;        /* saccade strength per channel */
    float wander_tau, wander_amp;           /* Ornstein-Uhlenbeck heading bias */
    /* actuators (channel order matters: it defines the WS action frame) */
    char channels[8][32];
    float ch_lo[8], ch_hi[8], ch_slew[8], ch_default[8];
    int n_channels;
    /* reward */
    float open_sky_start, open_sky_end, open_sky_reward, bump_penalty;
    /* network */
    int ws_port, rest_port;
} FbConfig;

/* load config + optional profile overlay; returns 0 on success */
int fb_config_load(FbConfig *cfg, const char *path, const char *profile_path);

#endif
