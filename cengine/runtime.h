/* BrainRuntime: circuit + sensors + readout + tick loop + memory store. */
#ifndef FB_RUNTIME_H
#define FB_RUNTIME_H

#include "circuit.h"
#include "config.h"
#include "json.h"

typedef struct {
    /* latest frames per eye (raw client grids, coalescing slots) */
    uint8_t *img[2];
    int img_w[2], img_h[2];
    int dirty[2];

    /* grids pushed straight in float form (any client size) */
    float *grid[2];
    int grid_w[2], grid_h[2];
    int grid_dirty[2];

    /* per-eye circuit-resolution float grids (lam_w x lam_h x 3) */
    float *samp[2];
    int samp_ok[2];

    /* scalar state */
    float altitude, speed, vy, clearance; /* clearance is ESTIMATED from optic flow */
    int collision;
    double coll_hold_s; /* bumper memory: contact-range window after a hit */

    /* downsampled float grids for the circuit (lam_w x lam_h x 3) */
    float *sample;
    int sample_cap;
} FbSensors;

typedef struct {
    char name[32];
    float value;
} FbChannel;

typedef struct FbRuntime {
    FbConfig cfg;
    FbConnectome *con;
    FbCircuit *net;

    FbSensors sens;

    /* actuator state */
    FbChannel ch[8];
    int n_channels;
    uint64_t actions_serial;

    /* memory */
    int learning;
    double tick_cost_ema;
    double last_wall;
    double mem_timer;
    int64_t frames;
    int restored;
    int pools_restored;   /* pool weights imported from the memory file */

    /* seed noise only; ALL behavior comes from the circuit's own state
     * (the circuit's per-neuron RNG is seeded from this) */
    uint64_t rng;
    float last_turn_cmd;    /* for telemetry */

    /* thread-safety: the tick loop owns the circuit; API threads only touch
     * the coalescing slots + snapshots guarded by this mutex */
    void *_lock; /* platform mutex */
    int stop;
    void *_thread;

    /* telemetry snapshot cache (refreshed by the loop) */
    char *telemetry_json;
} FbRuntime;

FbRuntime *fb_runtime_new(const FbConfig *cfg);
void fb_runtime_free(FbRuntime *rt);

/* start/stop the background tick loop */
int fb_runtime_start(FbRuntime *rt);
void fb_runtime_stop(FbRuntime *rt);

/* client API (thread-safe) */
void fb_runtime_ingest_frame(FbRuntime *rt, int eye, int w, int h, const uint8_t *rgb);
void fb_runtime_ingest_grid(FbRuntime *rt, int eye, int w, int h, const float *rgb);
void fb_runtime_ingest_state(FbRuntime *rt, float altitude, float speed, float vy,
                             float clearance, int collision);
void fb_runtime_set_learning(FbRuntime *rt, int on);
void fb_runtime_wipe_memory(FbRuntime *rt);
int fb_runtime_import_memory(FbRuntime *rt, const FbJson *mem);
void fb_runtime_apply_reward(FbRuntime *rt, float r); /* DAN excitability pathway */

/* Switch the embodiment profile at runtime: actuator channels, readout map
 * and sensors change; the circuit and its learned memory do not. cfg is the
 * master config owned by main() (kept in sync with the runtime). */
void fb_runtime_switch_profile(FbRuntime *rt, FbConfig *cfg,
                               const char *profile_path);

/* Back-fill pool weights from the memory file (CALLER HOLDS the runtime
 * lock; does not lock internally). Used when pools are configured after
 * the boot-time restore. */
void fb_runtime_pools_restore_from_disk(FbRuntime *rt);

/* snapshots for the API layer (thread-safe; caller frees) */
char *fb_runtime_telemetry_json(FbRuntime *rt);       /* malloc'd */
int fb_runtime_actions_json(FbRuntime *rt, FbStr *out); /* JSON channel map */
int fb_runtime_actions_binary(FbRuntime *rt, uint8_t *buf, int cap); /* WS frame */
char *fb_runtime_memory_json(FbRuntime *rt);          /* malloc'd */
int fb_runtime_client_count(FbRuntime *rt);

/* live channel values (thread-safe copy), returns channel count */
int fb_runtime_actions_copy(FbRuntime *rt, FbChannel *out, int cap);

/* explicit runtime lock (short critical sections in the API layer) */
void fb_rt_lock(FbRuntime *rt);
void fb_rt_unlock(FbRuntime *rt);

#endif
