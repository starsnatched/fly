#include "runtime.h"
#include "util.h"
#include "sockopt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

#if defined(_WIN32)
#include <windows.h>
static CRITICAL_SECTION *rt_mutex(void **p) {
    if (!*p) { *p = malloc(sizeof(CRITICAL_SECTION)); InitializeCriticalSection((CRITICAL_SECTION *)*p); }
    return (CRITICAL_SECTION *)*p;
}
static void rt_lock(FbRuntime *rt) { EnterCriticalSection(rt_mutex(&rt->_lock)); }
static void rt_unlock(FbRuntime *rt) { LeaveCriticalSection(rt_mutex(&rt->_lock)); }
#else
#include <pthread.h>
static pthread_mutex_t *rt_mutex(void **p) {
    if (!*p) { *p = malloc(sizeof(pthread_mutex_t)); pthread_mutex_init((pthread_mutex_t *)*p, NULL); }
    return (pthread_mutex_t *)*p;
}
static void rt_lock(FbRuntime *rt) { pthread_mutex_lock(rt_mutex(&rt->_lock)); }
static void rt_unlock(FbRuntime *rt) { pthread_mutex_unlock(rt_mutex(&rt->_lock)); }
#endif

/* ------------------------------------------------------------- sensors */

static void sensors_init(FbSensors *s) {
    memset(s, 0, sizeof(*s));
    s->altitude = 2.0f;
    s->clearance = 30.0f;
}

static void sensors_free(FbSensors *s) {
    free(s->img[0]); free(s->img[1]);
    free(s->grid[0]); free(s->grid[1]);
    free(s->samp[0]); free(s->samp[1]);
    free(s->sample);
    memset(s, 0, sizeof(*s));
}

/* box-downsample a raw uint8 RGB frame to (gw x gh) float 0..1, returns
 * malloc'd (caller frees) or NULL */
static float *box_sample(const uint8_t *img, int w, int h, int gw, int gh) {
    float *out = (float *)malloc((size_t)gw * gh * 3 * sizeof(float));
    if (!out) return NULL;
    for (int gy = 0; gy < gh; gy++) {
        int y0 = (int)((long long)gy * h / gh);
        int y1 = (int)((long long)(gy + 1) * h / gh);
        if (y1 <= y0) y1 = y0 + 1;
        for (int gx = 0; gx < gw; gx++) {
            int x0 = (int)((long long)gx * w / gw);
            int x1 = (int)((long long)(gx + 1) * w / gw);
            if (x1 <= x0) x1 = x0 + 1;
            double rs = 0, gs = 0, bs = 0;
            int cnt = 0;
            for (int y = y0; y < y1 && y < h; y++) {
                const uint8_t *row = img + ((size_t)y * w) * 3;
                for (int x = x0; x < x1 && x < w; x++) {
                    rs += row[x * 3]; gs += row[x * 3 + 1]; bs += row[x * 3 + 2];
                    cnt++;
                }
            }
            float inv = cnt ? 1.0f / (255.0f * (float)cnt) : 0.0f;
            size_t o = ((size_t)gy * gw + gx) * 3;
            out[o] = (float)(rs * inv);
            out[o + 1] = (float)(gs * inv);
            out[o + 2] = (float)(bs * inv);
        }
    }
    return out;
}

/* float-grid resample (any client size -> gw x gh), in-place into dst */
static void box_sample_f(const float *img, int w, int h, int gw, int gh, float *dst) {
    for (int gy = 0; gy < gh; gy++) {
        int y0 = (int)((long long)gy * h / gh);
        int y1 = (int)((long long)(gy + 1) * h / gh);
        if (y1 <= y0) y1 = y0 + 1;
        for (int gx = 0; gx < gw; gx++) {
            int x0 = (int)((long long)gx * w / gw);
            int x1 = (int)((long long)(gx + 1) * w / gw);
            if (x1 <= x0) x1 = x0 + 1;
            double rs = 0, gs = 0, bs = 0;
            int cnt = 0;
            for (int y = y0; y < y1 && y < h; y++) {
                const float *row = img + ((size_t)y * w) * 3;
                for (int x = x0; x < x1 && x < w; x++) {
                    rs += row[x * 3]; gs += row[x * 3 + 1]; bs += row[x * 3 + 2];
                    cnt++;
                }
            }
            float inv = cnt ? 1.0f / (float)cnt : 0.0f;
            size_t o = ((size_t)gy * gw + gx) * 3;
            dst[o] = (float)(rs * inv);
            dst[o + 1] = (float)(gs * inv);
            dst[o + 2] = (float)(bs * inv);
        }
    }
}

/* ------------------------------------------------------------- readout */

typedef struct {
    float hL, hR, vL, vR;
} EmdFlow;

/* Readout: NEURAL STATE -> ACTUATOR CHANNELS, and nothing else.
 * Every signal here is a population activity INSIDE the circuit — the same
 * thing an electrophysiologist would decode from descending neurons and the
 * optic lobe. There are no scripted behaviors, no set points, no reflex
 * gains computed from raw sensors: vision and touch enter the circuit as
 * neural input, and all behavior is what the connectome (plus R-STDP
 * memory) does with them.
 *
 * Which population feeds which channel is a per-embodiment DECLARATION in
 * config (readout.map), because that mapping is anatomy, not policy:
 *   dnDrive    descending-population rate (normalized 0..1)
 *   motorDrive motor-population rate (normalized 0..1)
 *   dnSteer    left-vs-right descending rate asymmetry (-1..1)
 *   flowYaw    T4/T5 horizontal flow, right-vs-left difference
 *   flowRoll   T4/T5 whole-field horizontal flow (optic-lobe consensus)
 *   flowPitch  T4/T5 vertical flow (optic-lobe consensus)
 *   touch      mechanosensory burst envelope (0..1)
 * A channel maps to: offset + gain*signal1 + gain*signal2 + ... (summed),
 * then clamped to the channel's configured range and slewed. */

enum {
    SIG_DN_DRIVE, SIG_MOTOR_DRIVE, SIG_DN_STEER,
    SIG_FLOW_YAW, SIG_FLOW_ROLL, SIG_FLOW_PITCH, SIG_TOUCH,
    SIG_COUNT
};

static const char *SIG_NAMES[SIG_COUNT] = {
    "dnDrive", "motorDrive", "dnSteer",
    "flowYaw", "flowRoll", "flowPitch", "touch",
};

static int signal_index(const char *name) {
    for (int i = 0; i < SIG_COUNT; i++)
        if (strcmp(SIG_NAMES[i], name) == 0) return i;
    return -1;
}

static void readout_compute(FbRuntime *rt, float *out /* n_channels */) {
    const FbConfig *cfg = &rt->cfg;
    FbCircuit *net = rt->net;
    for (int i = 0; i < rt->n_channels; i++) out[i] = 0.0f;

    /* ---- decode the circuit's own state ---- */
    float sig[SIG_COUNT] = {0};
    float dn = fb_rate_of(net, "descending");
    float motor = fb_rate_of(net, "motor");
    sig[SIG_DN_DRIVE] = clampf(dn / cfg->dn_hz_scale, 0.0f, 1.0f);
    sig[SIG_MOTOR_DRIVE] = clampf(motor / cfg->dn_hz_scale, 0.0f, 1.0f);
    sig[SIG_DN_STEER] = fb_dn_steer(net);
    EmdFlow f;
    fb_emd_flow(net, &f.hL, &f.hR, &f.vL, &f.vR);
    sig[SIG_FLOW_YAW] = f.hR - f.hL;
    sig[SIG_FLOW_ROLL] = f.hL + f.hR;
    sig[SIG_FLOW_PITCH] = 0.5f * (f.vL + f.vR);
    sig[SIG_TOUCH] = cfg->touch_gain > 0.0f
        ? clampf(net->touch_blast_mV / cfg->touch_gain, 0.0f, 1.0f) : 0.0f;

    /* ---- the declared decode map: population -> channel ----
     * Entries name a SIGNAL (a population readout above) and a CHANNEL
     * (an actuator from the profile); channel = offset + gain*signal.
     * Unknown names are skipped, never invented. */
    float yaw_cmd = 0.0f;
    for (int m = 0; m < cfg->n_map; m++) {
        const FbMapEntry *e = &cfg->map[m];
        int s = signal_index(e->signal);
        if (s < 0) continue;
        int ch = -1;
        for (int i = 0; i < rt->n_channels; i++)
            if (strcmp(rt->ch[i].name, e->channel) == 0) { ch = i; break; }
        if (ch < 0) continue;
        out[ch] += e->offset + e->gain * sig[s];
    }
    /* clamp here too; actuators_apply applies the channel slew after */
    for (int i = 0; i < rt->n_channels; i++) {
        if (out[i] < rt->cfg.ch_lo[i]) out[i] = rt->cfg.ch_lo[i];
        if (out[i] > rt->cfg.ch_hi[i]) out[i] = rt->cfg.ch_hi[i];
        if (strcmp(rt->ch[i].name, "yaw") == 0) yaw_cmd = out[i];
    }
    rt->last_turn_cmd = yaw_cmd;

    /* tau (time-to-contact) percept for telemetry — an internal estimate
     * from the EMD flow (looming = close); drives NO behavior directly */
    if (rt->sens.coll_hold_s > 0.0) {
        rt->sens.coll_hold_s -= 1.0 / 60.0;
    } else {
        float flow = fmaxf(fabsf(f.hL), fmaxf(fabsf(f.hR), fabsf(sig[SIG_FLOW_PITCH])));
        float clr = flow > 1e-4f ? cfg->tau_scale / flow : cfg->range_clr[1];
        if (clr < cfg->range_clr[0]) clr = cfg->range_clr[0];
        if (clr > cfg->range_clr[1]) clr = cfg->range_clr[1];
        rt->sens.clearance += 0.35f * (clr - rt->sens.clearance);
    }
}

static void actuators_apply(FbRuntime *rt, const float *raw, double dt) {
    for (int i = 0; i < rt->n_channels; i++) {
        float v = raw[i];
        if (!isfinite(v)) v = rt->cfg.ch_default[i];
        float lo = rt->cfg.ch_lo[i], hi = rt->cfg.ch_hi[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        float cur = rt->ch[i].value;
        float step = rt->cfg.ch_slew[i] * (float)(dt > 1e-3 ? dt : 1e-3);
        float d = v - cur;
        if (d > step) d = step;
        if (d < -step) d = -step;
        cur += d;
        if (cur < lo) cur = lo;
        if (cur > hi) cur = hi;
        rt->ch[i].value = cur;
    }
    rt->actions_serial++;
}

/* ------------------------------------------------------------- reward */
/* NOTHING is shaped here. The dopamine teaching signal is computed inside
 * the circuit from the DAN population's own firing vs. its adapting
 * baseline (dopa_self). External influences are strictly sensory:
 *   - POST /reward -> DAN excitability pathway (fb_apply_dan_bias),
 *     exactly like an appetitive/aversive sensory input to PPL1/PAM;
 *   - collisions  -> mechanosensory current into the sensory population
 *     (fb_circuit_sensory_burst); the connectome's own wiring turns the
 *     tap into a behavior- and reward-relevant event.
 * What the brain LEARNS from these events is R-STDP + its own dopa. */

/* ------------------------------------------------------------- memory */

static void memory_save(FbRuntime *rt, int force) {
    if (!rt->learning && !force) return;
    int *idx = NULL; double *w = NULL; int cnt = 0;
    fb_export_memory(rt->net, &idx, &w, &cnt);
    FbStr s;
    fb_str_init(&s);
    fb_str_append(&s, "{\"v\":1,\"idx\":[");
    for (int i = 0; i < cnt; i++) {
        if (i) fb_str_push(&s, ',');
        fb_str_append_int(&s, idx[i]);
    }
    fb_str_append(&s, "],\"w\":[");
    for (int i = 0; i < cnt; i++) {
        if (i) fb_str_push(&s, ',');
        fb_str_append_f(&s, w[i], 5);
    }
    fb_str_append(&s, "],\"simTimeS\":");
    fb_str_append_f(&s, rt->net->sim_ms / 1000.0, 2);
    fb_str_append(&s, ",\"rewardSum\":");
    fb_str_append_f(&s, rt->net->cum_reward, 3);
    fb_str_append(&s, ",\"circuit\":{\"neurons\":");
    fb_str_append_int(&s, rt->net->c->N);
    fb_str_append(&s, ",\"plasticEdges\":");
    fb_str_append_int(&s, rt->net->plastic_n);
    fb_str_append(&s, "}}");

    FILE *f = fopen(rt->cfg.memory_path, "wb");
    if (f) {
        fwrite(s.buf, 1, s.len, f);
        fclose(f);
    }
    fb_str_free(&s);
    free(idx); free(w);
}

static int memory_restore(FbRuntime *rt) {
    size_t len;
    uint8_t *text = fb_read_file(rt->cfg.memory_path, &len);
    if (!text) return 0;
    FbJson *m = fb_json_parse((char *)text);
    free(text);
    if (!m) return 0;
    const FbJson *cir = fb_json_get(m, "circuit");
    if (cir) {
        if ((int)fb_json_num(cir, "neurons", -1) != rt->net->c->N ||
            (int)fb_json_num(cir, "plasticEdges", -1) != rt->net->plastic_n) {
            fb_json_free(m);
            return 0;
        }
    }
    const FbJson *ia = fb_json_get(m, "idx");
    const FbJson *wa = fb_json_get(m, "w");
    int n = 0;
    if (ia && wa && ia->type == FB_JSON_ARR && wa->type == FB_JSON_ARR) {
        n = ia->n < wa->n ? ia->n : wa->n;
        int *idx = (int *)malloc((size_t)n * sizeof(int));
        double *w = (double *)malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; i++) {
            idx[i] = (int)ia->items[i]->num;
            w[i] = wa->items[i]->num;
        }
        n = fb_load_memory(rt->net, idx, w, n);
        free(idx); free(w);
    }
    fb_json_free(m);
    return n;
}

/* ------------------------------------------------------------- runtime */

FbRuntime *fb_runtime_new(const FbConfig *cfg) {
    FbRuntime *rt = (FbRuntime *)calloc(1, sizeof(FbRuntime));
    if (!rt) return NULL;
    rt->cfg = *cfg;
    rt->con = fb_connectome_load(cfg->binary);
    if (!rt->con) { free(rt); return NULL; }
    FbParams p;
    fb_params_default(&p);
    p.g_scale = cfg->g_scale;
    p.tgt_budget = cfg->tgt_budget;
    p.ph_tonic = cfg->ph_tonic;
    p.ph_optic_gain = cfg->ph_optic_gain;
    p.emd_gain = cfg->emd_gain;
    p.dan_mod_gain = cfg->dan_mod_gain;
    p.dopa_gain = cfg->dopa_gain;
    p.dan_base_tau_ms = cfg->dan_base_tau_s * 1000.0f;
    p.dan_tonic_mv = cfg->dan_tonic_mv;
    rt->net = fb_circuit_new(rt->con, &p);
    if (!rt->net) { fb_connectome_free(rt->con); free(rt); return NULL; }
    rt->n_channels = cfg->n_channels;
    for (int i = 0; i < cfg->n_channels; i++) {
        snprintf(rt->ch[i].name, sizeof(rt->ch[i].name), "%s", cfg->channels[i]);
        rt->ch[i].value = cfg->ch_default[i];
    }
    rt->learning = cfg->learning;
    fb_set_learning(rt->net, rt->learning);
    rt->tick_cost_ema = cfg->tick_cost_seed_ms;
    rt->last_wall = fb_now();
    rt->mem_timer = cfg->autosave_s;
    rt->rng = (uint64_t)(fb_now() * 1e6) ^ 0x9E3779B97F4A7C15ULL;
    if (!rt->rng) rt->rng = 0x2545F4914F6CDD1DULL;
    for (int i = 0; i < 8; i++) {
        rt->rng ^= rt->rng << 13; rt->rng ^= rt->rng >> 7; rt->rng ^= rt->rng << 17;
    }
    /* hand the runtime's seed to the circuit so there is one noise source,
     * inside the brain — nothing behavioral is seeded here */
    rt->net->rng_s0 ^= rt->rng;
    rt->net->rng_s1 ^= ~rt->rng;
    rt->last_turn_cmd = 0.0f;
    sensors_init(&rt->sens);
    rt->frames = 0;
    rt->restored = memory_restore(rt);
    return rt;
}

void fb_runtime_free(FbRuntime *rt) {
    if (!rt) return;
    fb_runtime_stop(rt);
    sensors_free(&rt->sens);
    fb_circuit_free(rt->net);
    fb_connectome_free(rt->con);
    free(rt->telemetry_json);
    free(rt->_lock);
    free(rt);
}

/* ------------------------------------------------------------- loop */

#if defined(_WIN32)
static DWORD WINAPI loop_main(LPVOID arg);
#else
static void *loop_main(void *arg);
#endif

int fb_runtime_start(FbRuntime *rt) {
    if (rt->_thread) return 0;
#if defined(_WIN32)
    HANDLE h = CreateThread(NULL, 0, loop_main, rt, 0, NULL);
    if (!h) return -1;
    rt->_thread = (void *)h;
#else
    pthread_t *t = (pthread_t *)malloc(sizeof(pthread_t));
    if (pthread_create(t, NULL, loop_main, rt) != 0) { free(t); return -1; }
    rt->_thread = (void *)t;
#endif
    return 0;
}

void fb_runtime_stop(FbRuntime *rt) {
    if (!rt->_thread) return;
    rt->stop = 1;
#if defined(_WIN32)
    WaitForSingleObject((HANDLE)rt->_thread, 3000);
    CloseHandle((HANDLE)rt->_thread);
#else
    pthread_join(*(pthread_t *)rt->_thread, NULL);
    free(rt->_thread);
#endif
    rt->_thread = NULL;
    rt_lock(rt);
    memory_save(rt, 1);
    rt_unlock(rt);
}

#if defined(_WIN32)
static DWORD WINAPI loop_main(LPVOID arg)
#else
static void *loop_main(void *arg)
#endif
{
    FbRuntime *rt = (FbRuntime *)arg;
    FbCircuit *net = rt->net;
    int W = net->lam_w, H = net->lam_h;
    int cells = W * H;
    while (!rt->stop) {
        double t0 = fb_now();
        double wall_ms = (t0 - rt->last_wall) * 1000.0;
        if (wall_ms > 50.0) wall_ms = 50.0;
        rt->last_wall = t0;

        rt_lock(rt);
        /* drain coalescing slots -> per-eye circuit-resolution grids.
         * single-eye embodiment (cfg.eye_count == 1): both circuit slots
         * point at the ONE sampled grid; the circuit splits it internally. */
        const float *rgb[2] = { NULL, NULL };
        const int one_eye = rt->cfg.eye_count <= 1;
        for (int eye = 0; eye < 2; eye++) {
            int refreshed = 0;
            if (rt->sens.grid_dirty[eye] && rt->sens.grid[eye] &&
                rt->sens.grid_w[eye] > 0 && rt->sens.grid_h[eye] > 0) {
                /* resample the client grid (any size) into circuit resolution */
                float *dst = (float *)realloc(rt->sens.samp[eye], (size_t)cells * 3 * 4);
                if (dst) {
                    rt->sens.samp[eye] = dst;
                    box_sample_f(rt->sens.grid[eye], rt->sens.grid_w[eye],
                                 rt->sens.grid_h[eye], W, H, dst);
                    refreshed = 1;
                    rt->net->frame_new[eye] = 1; /* frame-locked EMD update */
                }
                rt->sens.grid_dirty[eye] = 0;
            }
            if (one_eye && eye == 1) continue; /* slot 1 mirrors slot 0 below */
            if (!refreshed && rt->sens.dirty[eye] && rt->sens.img[eye]) {
                float *dst = (float *)realloc(rt->sens.samp[eye], (size_t)cells * 3 * 4);
                if (dst) {
                    rt->sens.samp[eye] = dst;
                    float *smp = box_sample(rt->sens.img[eye], rt->sens.img_w[eye],
                                            rt->sens.img_h[eye], W, H);
                    if (smp) {
                        memcpy(dst, smp, (size_t)cells * 3 * 4);
                        free(smp);
                        refreshed = 1;
                        rt->net->frame_new[eye] = 1; /* frame-locked EMD update */
                    } else {
                        free(dst);
                        rt->sens.samp[eye] = NULL;
                    }
                }
                rt->sens.dirty[eye] = 0;
            }
            if (rt->sens.samp[eye]) rgb[eye] = rt->sens.samp[eye];
            (void)refreshed;
        }
        if (one_eye) { rgb[1] = rgb[0]; rt->sens.samp_ok[1] = rt->sens.samp_ok[0]; }

        /* readout + actuator slew: turn brain state into commands (~60 Hz) */
        {
            float raw[8];
            readout_compute(rt, raw);
            actuators_apply(rt, raw, 1.0 / 60.0);
        }

        /* adaptive tick budget */
        double want_bio = wall_ms * rt->cfg.sim_speed;
        if (want_bio > rt->cfg.max_bio_ms) want_bio = rt->cfg.max_bio_ms;
        double cost = rt->tick_cost_ema > 2.0 ? rt->tick_cost_ema : 2.0;
        int ticks = (int)(want_bio / cost);
        if (ticks < 1) ticks = 1;
        double t1 = fb_now();
        for (int k = 0; k < ticks; k++)
            fb_tick(net, rgb[0], rgb[1], W, H);
        double used = (fb_now() - t1) * 1000.0;
        fb_learn_step(net);
        rt->frames++;
        double per_tick = used / ticks;
        double a = fabs(per_tick - rt->tick_cost_ema) > 0.5 * rt->tick_cost_ema ? 0.2 : 0.02;
        rt->tick_cost_ema += (per_tick - rt->tick_cost_ema) * a;

        /* collision -> mechanosensory startle: a brief current burst into
         * the sensory population. The connectome's own wiring (sensory ->
         * VNC -> descending, sensory -> central) propagates the event; any
         * learning about it happens through the circuit's own dopa/R-STDP. */
        if (rt->sens.collision) {
            rt->sens.collision = 0;
            fb_circuit_sensory_burst(net, rt->cfg.touch_gain);
        }
        rt->mem_timer -= wall_ms / 1000.0;
        if (rt->mem_timer <= 0) {
            rt->mem_timer = rt->cfg.autosave_s;
            if (rt->learning) memory_save(rt, 0);
        }
        rt_unlock(rt);
        double spend = fb_now() - t0;
        int sleep_ms = (int)((0.016 - spend) * 1000.0);
        if (sleep_ms > 0) fb_sleep_ms(sleep_ms);
    }
    return 0;
}

/* ------------------------------------------------------------- client api */

void fb_runtime_ingest_frame(FbRuntime *rt, int eye, int w, int h, const uint8_t *rgb) {
    if (eye < 0 || eye > 1 || !rgb || w <= 0 || h <= 0) return;
    if (w > 4096 || h > 4096) return; /* sane sensor ceiling */
    size_t bytes = (size_t)w * h * 3;
    rt_lock(rt);
    uint8_t *slot = (uint8_t *)realloc(rt->sens.img[eye], bytes);
    if (slot) {
        memcpy(slot, rgb, bytes);
        rt->sens.img[eye] = slot;
        rt->sens.img_w[eye] = w;
        rt->sens.img_h[eye] = h;
        rt->sens.dirty[eye] = 1;
    }
    rt_unlock(rt);
}

void fb_runtime_ingest_grid(FbRuntime *rt, int eye, int w, int h, const float *rgb) {
    if (eye < 0 || eye > 1 || !rgb || w <= 0 || h <= 0) return;
    size_t bytes = (size_t)w * h * 3 * sizeof(float);
    rt_lock(rt);
    float *slot = (float *)realloc(rt->sens.grid[eye], bytes);
    if (slot) {
        memcpy(slot, rgb, bytes);
        rt->sens.grid[eye] = slot;
        rt->sens.grid_w[eye] = w;
        rt->sens.grid_h[eye] = h;
        rt->sens.grid_dirty[eye] = 1;
    }
    rt_unlock(rt);
}

void fb_runtime_ingest_state(FbRuntime *rt, float altitude, float speed, float vy,
                             float clearance, int collision) {
    const FbConfig *cfg = &rt->cfg;
    rt_lock(rt);
    if (isfinite(altitude)) {
        if (altitude < cfg->range_alt[0]) altitude = cfg->range_alt[0];
        if (altitude > cfg->range_alt[1]) altitude = cfg->range_alt[1];
        rt->sens.altitude = altitude;
    }
    if (isfinite(speed)) {
        if (speed < cfg->range_speed[0]) speed = cfg->range_speed[0];
        if (speed > cfg->range_speed[1]) speed = cfg->range_speed[1];
        rt->sens.speed = speed;
    }
    if (isfinite(vy)) {
        if (vy < cfg->range_vy[0]) vy = cfg->range_vy[0];
        if (vy > cfg->range_vy[1]) vy = cfg->range_vy[1];
        rt->sens.vy = vy;
    }
    /* NOTE: `clearance` is no longer accepted from clients. A real drone has
     * no obstacle rangefinder; the brain estimates clearance itself from
     * optic flow (time-to-contact) in readout_compute(). */
    (void)clearance;
    if (collision) {
        rt->sens.collision = 1;
        rt->sens.coll_hold_s = 0.5; /* bumper memory: penalty window */
    }
    rt_unlock(rt);
}

void fb_runtime_set_learning(FbRuntime *rt, int on) {
    rt_lock(rt);
    rt->learning = on ? 1 : 0;
    fb_set_learning(rt->net, rt->learning);
    if (on && rt->mem_timer > 1.0) rt->mem_timer = 1.0;
    rt_unlock(rt);
}

/* Reset the brain's memory to factory defaults: all learned multipliers
 * back to 1.0, eligibility/reward state cleared, and the on-disk memory
 * REMOVED so the reset survives restarts. */
void fb_runtime_wipe_memory(FbRuntime *rt) {
    rt_lock(rt);
    fb_reset_plasticity(rt->net);
    rt_unlock(rt);
    fb_remove_file(rt->cfg.memory_path);
    memory_save(rt, 1);
    rt->restored = 0;
}

int fb_runtime_import_memory(FbRuntime *rt, const FbJson *mem) {
    const FbJson *ia = fb_json_get(mem, "idx");
    const FbJson *wa = fb_json_get(mem, "w");
    if (!ia || !wa || ia->type != FB_JSON_ARR || wa->type != FB_JSON_ARR) return 0;
    int n = ia->n < wa->n ? ia->n : wa->n;
    int *idx = (int *)malloc((size_t)n * sizeof(int));
    double *w = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        idx[i] = (int)ia->items[i]->num;
        w[i] = wa->items[i]->num;
    }
    rt_lock(rt);
    int hits = fb_load_memory(rt->net, idx, w, n);
    rt_unlock(rt);
    free(idx); free(w);
    return hits;
}

void fb_runtime_apply_reward(FbRuntime *rt, float r) {
    rt_lock(rt);
    /* enters ONLY through the DAN excitability pathway — the circuit
     * decides what (if anything) it means via its own dopa signal */
    fb_apply_dan_bias(rt->net, rt->cfg.reward_gain * r);
    rt_unlock(rt);
}

int fb_runtime_actions_copy(FbRuntime *rt, FbChannel *out, int cap) {
    rt_lock(rt);
    int n = rt->n_channels < cap ? rt->n_channels : cap;
    for (int i = 0; i < n; i++) out[i] = rt->ch[i];
    rt_unlock(rt);
    return n;
}

void fb_rt_lock(FbRuntime *rt) { rt_lock(rt); }
void fb_rt_unlock(FbRuntime *rt) { rt_unlock(rt); }

char *fb_runtime_telemetry_json(FbRuntime *rt) {
    FbStr s;
    fb_str_init(&s);
    rt_lock(rt);
    FbCircuit *net = rt->net;
    float lo, hi;
    int edited = fb_learn_stats(net, &lo, &hi);
    fb_str_append(&s, "{\"neurons\":");
    fb_str_append_int(&s, net->c->N);
    fb_str_append(&s, ",\"edges\":");
    fb_str_append_int(&s, net->c->E);
    fb_str_append(&s, ",\"plastic\":");
    fb_str_append_int(&s, net->plastic_n);
    fb_str_append(&s, ",\"dans\":");
    fb_str_append_int(&s, net->dan_n);
    fb_str_append(&s, ",\"groups\":[");
    for (int g = 0; g < net->G; g++) {
        if (g) fb_str_push(&s, ',');
        fb_str_append_json_str(&s, net->c->groups[g]);
    }
    fb_str_append(&s, "],\"rates\":{");
    const char *keys[] = { "lamina", "Tm", "T4", "T5", "LC", "LPLC",
                           "descending", "motor", "KC", "MBON", "DAN" };
    for (int i = 0; i < 11; i++) {
        if (i) fb_str_push(&s, ',');
        fb_str_append_json_str(&s, keys[i]);
        fb_str_push(&s, ':');
        fb_str_append_f(&s, fb_rate_of(net, keys[i]), 3);
    }
    fb_str_append(&s, "},\"spikesPerTick\":");
    fb_str_append_int(&s, net->last_tick_spikes);
    fb_str_append(&s, ",\"simMs\":");
    fb_str_append_f(&s, net->sim_ms, 1);
    fb_str_append(&s, ",\"danHz\":");
    fb_str_append_f(&s, net->dan_ema_hz, 3);
    fb_str_append(&s, ",\"dopa\":");
    fb_str_append_f(&s, net->dopa_self, 3);
    fb_str_append(&s, ",\"danBase\":");
    fb_str_append_f(&s, net->dan_base_hz, 2);
    fb_str_append(&s, ",\"dnSteer\":");
    fb_str_append_f(&s, net->last_dn_l - net->last_dn_r, 4);
    fb_str_append(&s, ",\"turn\":");
    fb_str_append_f(&s, rt->last_turn_cmd, 4);
    fb_str_append(&s, ",\"flowH\":");
    fb_str_append_f(&s, net->flow_hR - net->flow_hL, 4);
    fb_str_append(&s, ",\"flowV\":");
    fb_str_append_f(&s, 0.5 * (net->flow_vL + net->flow_vR), 4);
    fb_str_append(&s, ",\"alt\":");
    fb_str_append_f(&s, rt->sens.altitude, 2);
    fb_str_append(&s, ",\"clr\":");
    fb_str_append_f(&s, rt->sens.clearance, 1);
    fb_str_append(&s, ",\"speed\":");
    fb_str_append_f(&s, rt->sens.speed, 2);
    fb_str_append(&s, ",\"learning\":");
    fb_str_append(&s, rt->learning ? "true" : "false");
    fb_str_append(&s, ",\"memEdited\":");
    fb_str_append_int(&s, edited);
    fb_str_append(&s, ",\"memLo\":");
    fb_str_append_f(&s, lo, 4);
    fb_str_append(&s, ",\"memHi\":");
    fb_str_append_f(&s, hi, 4);
    fb_str_append(&s, ",\"corrupt\":");
    fb_str_append(&s, net->corrupted ? "true" : "false");
    fb_str_append(&s, ",\"tickCostMs\":");
    fb_str_append_f(&s, rt->tick_cost_ema, 2);
    fb_str_append(&s, ",\"simSpeedTarget\":");
    fb_str_append_f(&s, rt->cfg.sim_speed, 2);
    fb_str_append(&s, ",\"frames\":");
    fb_str_append_int(&s, (long long)rt->frames);
    fb_str_append(&s, ",\"restored\":");
    fb_str_append_int(&s, rt->restored);
    fb_str_append(&s, "}");
    rt_unlock(rt);
    return s.buf;
}

int fb_runtime_actions_json(FbRuntime *rt, FbStr *out) {
    FbChannel ch[8];
    int n = fb_runtime_actions_copy(rt, ch, 8);
    fb_str_append(out, "{\"channels\":{");
    for (int i = 0; i < n; i++) {
        if (i) fb_str_push(out, ',');
        fb_str_append_json_str(out, ch[i].name);
        fb_str_push(out, ':');
        fb_str_append_f(out, ch[i].value, 4);
    }
    fb_str_append(out, "}}");
    return n;
}

/* binary action frame: u8 type=10, u16 nameLen, names JSON, then f32s */
int fb_runtime_actions_binary(FbRuntime *rt, uint8_t *buf, int cap) {
    FbChannel ch[8];
    int n = fb_runtime_actions_copy(rt, ch, 8);
    FbStr s;
    fb_str_init(&s);
    fb_str_push(&s, '[');
    for (int i = 0; i < n; i++) {
        if (i) fb_str_push(&s, ',');
        fb_str_append_json_str(&s, ch[i].name);
    }
    fb_str_push(&s, ']');
    int need = 3 + (int)s.len + n * 4;
    if (need > cap) { fb_str_free(&s); return -1; }
    int off = 0;
    buf[off++] = 10;
    buf[off++] = (uint8_t)(s.len & 0xff);
    buf[off++] = (uint8_t)((s.len >> 8) & 0xff);
    memcpy(buf + off, s.buf, s.len);
    off += (int)s.len;
    for (int i = 0; i < n; i++) {
        union { float f; uint32_t u; } v;
        v.f = ch[i].value;
        buf[off++] = (uint8_t)(v.u & 0xff);
        buf[off++] = (uint8_t)((v.u >> 8) & 0xff);
        buf[off++] = (uint8_t)((v.u >> 16) & 0xff);
        buf[off++] = (uint8_t)((v.u >> 24) & 0xff);
    }
    fb_str_free(&s);
    return off;
}

char *fb_runtime_memory_json(FbRuntime *rt) {
    int *idx = NULL; double *w = NULL; int cnt = 0;
    rt_lock(rt);
    fb_export_memory(rt->net, &idx, &w, &cnt);
    double sim = rt->net->sim_ms / 1000.0, rw = rt->net->cum_reward;
    int nn = rt->net->c->N, pn = rt->net->plastic_n;
    rt_unlock(rt);
    FbStr s;
    fb_str_init(&s);
    fb_str_append(&s, "{\"v\":1,\"idx\":[");
    for (int i = 0; i < cnt; i++) {
        if (i) fb_str_push(&s, ',');
        fb_str_append_int(&s, idx[i]);
    }
    fb_str_append(&s, "],\"w\":[");
    for (int i = 0; i < cnt; i++) {
        if (i) fb_str_push(&s, ',');
        fb_str_append_f(&s, w[i], 5);
    }
    fb_str_append(&s, "],\"simTimeS\":");
    fb_str_append_f(&s, sim, 2);
    fb_str_append(&s, ",\"rewardSum\":");
    fb_str_append_f(&s, rw, 3);
    fb_str_append(&s, ",\"circuit\":{\"neurons\":");
    fb_str_append_int(&s, nn);
    fb_str_append(&s, ",\"plasticEdges\":");
    fb_str_append_int(&s, pn);
    fb_str_append(&s, "}}");
    free(idx); free(w);
    return s.buf;
}

int fb_runtime_client_count(FbRuntime *rt) {
    (void)rt;
    return 0; /* filled by api.c */
}
