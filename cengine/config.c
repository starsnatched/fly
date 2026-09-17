#include "config.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_map_entry(FbMapEntry *e, const FbJson *m) {
    snprintf(e->channel, sizeof(e->channel), "%s", fb_json_str(m, "channel", ""));
    snprintf(e->signal, sizeof(e->signal), "%s", fb_json_str(m, "signal", ""));
    e->gain = (float)fb_json_num(m, "gain", 0.0);
    e->offset = (float)fb_json_num(m, "offset", 0.0);
    return e->channel[0] && e->signal[0];
}

static void copy_json_or_default(FbConfig *cfg, FbJson *root) {
    const FbJson *eng = fb_json_get(root, "engine");
    if (eng) {
        snprintf(cfg->binary, sizeof(cfg->binary), "%s",
                 fb_json_str(eng, "binary", "data/fly-brain-full.bin"));
        cfg->g_scale = (float)fb_json_num(eng, "gScale", 0.30);
        cfg->tgt_budget = (float)fb_json_num(eng, "tgtBudget", 36.0);
        cfg->ph_tonic = (float)fb_json_num(eng, "phTonic", 5.0);
        cfg->ph_optic_gain = (float)fb_json_num(eng, "phOpticGain", 2.0);
        cfg->emd_gain = (float)fb_json_num(eng, "emdGain", 10.0);
        cfg->dan_mod_gain = (float)fb_json_num(eng, "danModGain", 0.6);
        cfg->dopa_gain = (float)fb_json_num(eng, "dopaGain", 0.09);
        cfg->dan_base_tau_s = (float)fb_json_num(eng, "danBaseTauS", 8.0);
        cfg->dan_tonic_mv = (float)fb_json_num(eng, "danTonicMv", 3.0);
        cfg->learning = fb_json_bool(eng, "learning", true) ? 1 : 0;
        cfg->tick_cost_seed_ms = (float)fb_json_num(eng, "tickCostSeedMs", 30.0);
    }
    cfg->sim_speed = (float)fb_json_num(root, "simSpeed", 1.0);
    cfg->max_bio_ms = (float)fb_json_num(root, "maxBioMsPerTick", 20.0);
    cfg->autosave_s = (float)fb_json_num(root, "autosaveS", 5.0);
    snprintf(cfg->memory_path, sizeof(cfg->memory_path), "%s",
             fb_json_str(root, "memoryPath", "state/brain-memory.json"));

    const FbJson *sens = fb_json_get(root, "sensors");
    if (sens) {
        const FbJson *eyes = fb_json_get(sens, "eyes");
        if (eyes) {
            cfg->eye_count = (int)fb_json_num(eyes, "count", (double)cfg->eye_count);
            cfg->eye_left_id = (int)fb_json_num(eyes, "leftId", 0);
            cfg->eye_right_id = (int)fb_json_num(eyes, "rightId", 1);
        }
        cfg->vision_hz = (float)fb_json_num(sens, "visionHz", (double)cfg->vision_hz);
        const FbJson *touch = fb_json_get(sens, "touch");
        if (touch) {
            cfg->touch_gain = (float)fb_json_num(touch, "burstMv", (double)cfg->touch_gain);
        }
        const FbJson *rng = fb_json_get(sens, "scalarRanges");
        if (rng) {
            const FbJson *a;
            if ((a = fb_json_get(rng, "altitude")) && a->type == FB_JSON_ARR && a->n == 2) {
                cfg->range_alt[0] = (float)a->items[0]->num;
                cfg->range_alt[1] = (float)a->items[1]->num;
            }
            if ((a = fb_json_get(rng, "speed")) && a->type == FB_JSON_ARR && a->n == 2) {
                cfg->range_speed[0] = (float)a->items[0]->num;
                cfg->range_speed[1] = (float)a->items[1]->num;
            }
            if ((a = fb_json_get(rng, "vy")) && a->type == FB_JSON_ARR && a->n == 2) {
                cfg->range_vy[0] = (float)a->items[0]->num;
                cfg->range_vy[1] = (float)a->items[1]->num;
            }
            if ((a = fb_json_get(rng, "clearance")) && a->type == FB_JSON_ARR && a->n == 2) {
                cfg->range_clr[0] = (float)a->items[0]->num;
                cfg->range_clr[1] = (float)a->items[1]->num;
            }
            cfg->tau_scale = (float)fb_json_num(rng, "tauScale", (double)cfg->tau_scale);
        }
    }

    const FbJson *ro = fb_json_get(root, "readout");
    if (ro) {
        cfg->dn_hz_scale = (float)fb_json_num(ro, "dnHzScale", (double)cfg->dn_hz_scale);
        const FbJson *map = fb_json_get(ro, "map");
        if (map && map->type == FB_JSON_ARR) {
            int n = map->n > FB_MAX_MAP ? FB_MAX_MAP : map->n;
            int kept = 0;
            for (int i = 0; i < n; i++)
                if (parse_map_entry(&cfg->map[kept], map->items[i])) kept++;
            /* overlay: a profile with a map REPLACES the base map */
            cfg->n_map = kept;
        }
    }

    const FbJson *act = fb_json_get(root, "actuators");
    if (act) {
        const FbJson *chs = fb_json_get(act, "channels");
        if (chs && chs->type == FB_JSON_ARR) {
            int n = chs->n > 8 ? 8 : chs->n;
            for (int i = 0; i < n; i++) {
                const FbJson *ch = chs->items[i];
                snprintf(cfg->channels[i], 32, "%s", fb_json_str(ch, "name", "ch"));
                cfg->ch_lo[i] = (float)fb_json_num(ch, "lo", -1.0);
                cfg->ch_hi[i] = (float)fb_json_num(ch, "hi", 1.0);
                cfg->ch_slew[i] = (float)fb_json_num(ch, "slewPerS", 8.0);
                cfg->ch_default[i] = (float)fb_json_num(ch, "default", 0.0);
            }
            cfg->n_channels = n;
        }
    }

    const FbJson *rew = fb_json_get(root, "reward");
    if (rew) {
        /* reward PATHWAY knob only — there is no reward shaping anywhere */
        cfg->reward_gain = (float)fb_json_num(rew, "rewardGain", (double)cfg->reward_gain);
    }
    const FbJson *net = fb_json_get(root, "network");
    if (net) {
        cfg->ws_port = (int)fb_json_num(net, "wsPort", (double)cfg->ws_port);
        cfg->rest_port = (int)fb_json_num(net, "restPort", (double)cfg->rest_port);
    }
}

int fb_config_load(FbConfig *cfg, const char *path, const char *profile_path) {
    memset(cfg, 0, sizeof(*cfg));
    /* hard defaults (mirroring config/flybrain.json) */
    snprintf(cfg->binary, sizeof(cfg->binary), "data/fly-brain-full.bin");
    cfg->g_scale = 0.30f; cfg->tgt_budget = 36.0f;
    cfg->ph_tonic = 5.0f; cfg->ph_optic_gain = 2.0f; cfg->emd_gain = 10.0f;
    cfg->dan_mod_gain = 0.6f; cfg->learning = 1; cfg->tick_cost_seed_ms = 30.0f;
    cfg->dopa_gain = 0.09f; cfg->dan_base_tau_s = 8.0f; cfg->dan_tonic_mv = 3.0f;
    cfg->sim_speed = 1.0f; cfg->max_bio_ms = 20.0f;
    cfg->autosave_s = 5.0f;
    cfg->reward_gain = 0.30f;
    snprintf(cfg->memory_path, sizeof(cfg->memory_path), "state/brain-memory.json");
    cfg->eye_count = 2;
    cfg->eye_left_id = 0; cfg->eye_right_id = 1;
    cfg->vision_hz = 120.0f;
    cfg->range_alt[0] = 0; cfg->range_alt[1] = 30;
    cfg->range_speed[0] = 0; cfg->range_speed[1] = 18;
    cfg->range_vy[0] = -8; cfg->range_vy[1] = 8;
    cfg->range_clr[0] = 0; cfg->range_clr[1] = 60;
    cfg->tau_scale = 9.0f;
    cfg->touch_gain = 6.0f;
    cfg->dn_hz_scale = 8.0f;
    cfg->n_map = 0;
    cfg->n_channels = 4;
    snprintf(cfg->channels[0], 32, "throttle");
    snprintf(cfg->channels[1], 32, "pitch");
    snprintf(cfg->channels[2], 32, "roll");
    snprintf(cfg->channels[3], 32, "yaw");
    cfg->ch_lo[0] = 0; cfg->ch_hi[0] = 1; cfg->ch_slew[0] = 6; cfg->ch_default[0] = 0.29f;
    for (int i = 1; i < 4; i++) {
        cfg->ch_lo[i] = -1; cfg->ch_hi[i] = 1; cfg->ch_slew[i] = 8;
        cfg->ch_default[i] = 0;
    }
    cfg->ch_default[1] = -0.35f;
    cfg->ws_port = 8787;
    cfg->rest_port = 8788;

    size_t len;
    uint8_t *text = NULL;
    if (path) text = fb_read_file(path, &len);
    if (text) {
        FbJson *root = fb_json_parse((char *)text);
        if (root) {
            copy_json_or_default(cfg, root);
            fb_json_free(root);
        } else {
            fprintf(stderr, "config: parse error in %s\n", path);
        }
        free(text);
    }
    if (profile_path && profile_path[0]) {
        uint8_t *ptext = fb_read_file(profile_path, &len);
        if (ptext) {
            FbJson *root = fb_json_parse((char *)ptext);
            if (root) {
                copy_json_or_default(cfg, root);
                fb_json_free(root);
            } else {
                fprintf(stderr, "config: parse error in profile %s\n", profile_path);
            }
            free(ptext);
        } else {
            fprintf(stderr, "config: profile %s not found (using base)\n", profile_path);
        }
    }
    return 0;
}
