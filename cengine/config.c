#include "config.h"
#include "runtime.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_map_entry(FbMapEntry *e, const FbJson *m) {
    snprintf(e->channel, sizeof(e->channel), "%s", fb_json_str(m, "channel", ""));
    snprintf(e->signal, sizeof(e->signal), "%s", fb_json_str(m, "signal", ""));
    e->gain = (float)fb_json_num(m, "gain", 0.0);
    e->offset = (float)fb_json_num(m, "offset", 0.0);
    const char *shape = fb_json_str(m, "shape", "");
    e->shape = strcmp(shape, "tanh") == 0 ? 1 : 0;
    /* "norm": {"span": s, "center": c} — readout normalization.
     * Rescales the signal so span s maps to the full [-1,1] stick, after
     * subtracting the center: "center": true = 0.5 (unipolar rectified
     * pool signals), a NUMBER = that value (calibrate it to the signal's
     * measured resting level — a wrong center rails the channel and the
     * body can then never earn positive dopamine), absent = 0. */
    const FbJson *nm = fb_json_get(m, "norm");
    e->has_norm = 0;
    if (nm && nm->type == FB_JSON_OBJ) {
        double span = fb_json_num(nm, "span", 0.0);
        if (span > 1e-6f) {
            e->has_norm = 1;
            e->norm_span = (float)span;
            const FbJson *ct = fb_json_get(nm, "center");
            if (ct && ct->type == FB_JSON_NUM)
                e->norm_center = (float)ct->num;
            else if (ct && ct->type == FB_JSON_BOOL)
                e->norm_center = ct->b ? 0.5f : 0.0f;
            else
                e->norm_center = 0.0f;
        }
    }
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
        /* direct motor pools (readout.pools): same overlay semantics — a
         * profile declaring pools REPLACES the base set. Pool i drives the
         * actuator channel named "pool<i>" (declare it in actuators). */
        const FbJson *pools = fb_json_get(ro, "pools");
        if (pools && pools->type == FB_JSON_ARR) {
            int n = pools->n > 8 ? 8 : pools->n;
            int kept = 0;
            for (int i = 0; i < n; i++) {
                const FbJson *p = pools->items[i];
                FbPoolEntry *pe = &cfg->pools[kept];
                snprintf(pe->group, sizeof(pe->group), "%s", fb_json_str(p, "group", ""));
                pe->every = (int)fb_json_num(p, "every", 0.0);
                const char *split = fb_json_str(p, "split", "");
                pe->split_lr = strcmp(split, "lr") == 0 ? 1 : 0;
                pe->which = (int)fb_json_num(p, "which", 0.0);
                pe->lr_scale = (float)fb_json_num(p, "lrScale", 1.0);
                if (pe->group[0]) kept++;
            }
            cfg->n_pools = kept;
        }
        cfg->pool_integrate_ms =
            (int)fb_json_num(ro, "integrateMs", (double)cfg->pool_integrate_ms);
        cfg->pool_learn =
            fb_json_bool(ro, "learn", cfg->pool_learn ? true : false) ? 1 : 0;
        cfg->pool_lr =
            (float)fb_json_num(ro, "poolLr", (double)cfg->pool_lr);
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
    cfg->n_pools = 0;
    cfg->pool_integrate_ms = 80;
    cfg->pool_learn = 1;
    cfg->pool_lr = 0.08f;
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
            /* process-scoped + engine settings must survive the overlay:
             * copy_json_or_default hardcodes defaults, so a profile that
             * omits them would otherwise RESET them (this is how a bench
             * config with its own memoryPath got silently pointed back at
             * the default memory file — and its wipe with it) */
            FbConfig keep = *cfg;
            FbJson *root = fb_json_parse((char *)ptext);
            if (root) {
                copy_json_or_default(cfg, root);
                snprintf(cfg->memory_path, sizeof(cfg->memory_path), "%s", keep.memory_path);
                cfg->sim_speed = keep.sim_speed;
                cfg->max_bio_ms = keep.max_bio_ms;
                cfg->autosave_s = keep.autosave_s;
                cfg->g_scale = keep.g_scale;
                cfg->tgt_budget = keep.tgt_budget;
                cfg->ph_tonic = keep.ph_tonic;
                cfg->ph_optic_gain = keep.ph_optic_gain;
                cfg->emd_gain = keep.emd_gain;
                cfg->dan_mod_gain = keep.dan_mod_gain;
                cfg->dopa_gain = keep.dopa_gain;
                cfg->dan_base_tau_s = keep.dan_base_tau_s;
                cfg->dan_tonic_mv = keep.dan_tonic_mv;
                cfg->learning = keep.learning;
                cfg->tick_cost_seed_ms = keep.tick_cost_seed_ms;
                cfg->ws_port = keep.ws_port;
                cfg->rest_port = keep.rest_port;
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

void fb_config_apply_profile_to_runtime(FbConfig *cfg, FbRuntime *rt,
                                        const char *profile_path) {
    if (!profile_path || !profile_path[0]) return;
    size_t len = 0;
    uint8_t *ptext = fb_read_file(profile_path, &len);
    if (!ptext) {
        fprintf(stderr, "config: profile %s not found (keeping current)\n",
                profile_path);
        return;
    }
    FbJson *root = fb_json_parse((char *)ptext);
    free(ptext);
    if (!root) {
        fprintf(stderr, "config: parse error in profile %s\n", profile_path);
        return;
    }

    /* Snapshot the runtime-scoped + engine sections, overlay the profile,
     * then restore the snapshot: profiles may only change EMBODIMENT
     * anatomy (sensors, readout map, actuator channels, reward pathway).
     * Engine/neural knobs would desync the already-built circuit; memory
     * path and ports belong to the process, not the body. */
    const FbConfig pre = *cfg;
    /* copy_json_or_default REPLACES the readout map when the profile
     * declares one (documented profile-overlay semantics) */
    copy_json_or_default(cfg, root);
    fb_json_free(root);

    snprintf(cfg->binary, sizeof(cfg->binary), "%s", pre.binary);
    snprintf(cfg->memory_path, sizeof(cfg->memory_path), "%s", pre.memory_path);
    cfg->g_scale = pre.g_scale;
    cfg->tgt_budget = pre.tgt_budget;
    cfg->ph_tonic = pre.ph_tonic;
    cfg->ph_optic_gain = pre.ph_optic_gain;
    cfg->emd_gain = pre.emd_gain;
    cfg->dan_mod_gain = pre.dan_mod_gain;
    cfg->dopa_gain = pre.dopa_gain;
    cfg->dan_base_tau_s = pre.dan_base_tau_s;
    cfg->dan_tonic_mv = pre.dan_tonic_mv;
    cfg->learning = pre.learning;
    cfg->tick_cost_seed_ms = pre.tick_cost_seed_ms;
    cfg->sim_speed = pre.sim_speed;
    cfg->max_bio_ms = pre.max_bio_ms;
    cfg->autosave_s = pre.autosave_s;
    cfg->ws_port = pre.ws_port;
    cfg->rest_port = pre.rest_port;

    /* push the embodiment-scoped overlay into the live runtime (the
     * loop thread reads rt->cfg + rt->ch[] under this lock) */
    fb_rt_lock(rt);
    rt->cfg = *cfg;
    rt->n_channels = cfg->n_channels;
    for (int i = 0; i < cfg->n_channels; i++) {
        snprintf(rt->ch[i].name, sizeof(rt->ch[i].name), "%s", cfg->channels[i]);
        rt->ch[i].value = cfg->ch_default[i];
    }
    /* motor pools are circuit-attached: rebuild them from the new
     * declaration, carrying learned pool weights across the switch (they
     * persist like synapse memory). Same lock: fb_pools_* run on the loop
     * thread's data between its ticks. */
    {
        char (*nm)[32] = (char (*)[32])malloc(4096 * sizeof(*nm));
        double *wv = NULL;
        int nw = 0;
        int had = nm ? fb_pools_export(rt->net, nm, 4096, &wv, &nw) : 0;
        FbPoolCfg pcfg[8];
        int npcfg = 0;
        for (int i = 0; i < cfg->n_pools; i++) {
            snprintf(pcfg[npcfg].group, sizeof(pcfg[npcfg].group), "%s",
                     cfg->pools[i].group);
            pcfg[npcfg].every = cfg->pools[i].every;
            pcfg[npcfg].split_lr = cfg->pools[i].split_lr;
            pcfg[npcfg].which = cfg->pools[i].which;
            pcfg[npcfg].lr_scale = cfg->pools[i].lr_scale;
            npcfg++;
        }
        fb_pools_configure(rt->net, pcfg, npcfg,
                           cfg->pool_integrate_ms, cfg->pool_learn, cfg->pool_lr);
        if (had > 0 && nw > 0) fb_pools_import(rt->net, nm, wv, nw);
        free(wv);
        free(nm);
        /* fresh file that predates any pool config: back-fill from disk
         * (file read + import run under the held lock — the loop thread
         * must not tick pools concurrently with the import). Only a
         * successful import sets pools_restored: a file that predates
         * pools, or a transient read failure, must stay retryable on the
         * next profile apply, or learned weights could never arrive. */
        if (!rt->pools_restored) fb_runtime_pools_restore_from_disk(rt);
    }
    fb_rt_unlock(rt);

    fprintf(stderr, "config: applied embodiment profile %s (%d channels:",
            profile_path, cfg->n_channels);
    for (int i = 0; i < cfg->n_channels; i++)
        fprintf(stderr, " %s", cfg->channels[i]);
    fprintf(stderr, ", %d map entries, %d pools)\n", cfg->n_map, cfg->n_pools);
}
