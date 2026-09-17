/* FLYBRAIN1 binary connectome loader (produced by scripts/extract_full_brain.py). */
#ifndef FB_FLYBRAIN_H
#define FB_FLYBRAIN_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int N;                /* neurons */
    int E;                /* edges */
    int plastic_count;    /* plastic edge ids */
    int n_groups;
    char **groups;        /* owned strings */
    uint8_t *group_of;    /* N */
    int8_t *nt;           /* N, neurotransmitter sign (+1 exc, -1 inh) */
    int8_t *dir;          /* N, T4/T5 preferred-direction class 0..3, else -1 */
    uint8_t *side;        /* N, 0 = left, 1 = right */
    int16_t *hex;         /* N*2, retinotopic column (hex1, hex2), -1 = none */
    float *nt_conf;       /* N */
    uint32_t *edge_src;   /* E */
    uint32_t *edge_dst;   /* E */
    float *edge_w;        /* E */
    uint32_t *plastic_idx;/* plastic_count edge ids */
} FbConnectome;

/* Load from the FLYBRAIN1 file. Returns NULL on failure (prints reason). */
FbConnectome *fb_connectome_load(const char *path);

void fb_connectome_free(FbConnectome *c);

#endif
