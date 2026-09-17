#include "flybrain.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC2 0x4E455552u /* "NURE" little-endian */

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int pad4(size_t n) { return (int)((4 - (n % 4)) % 4); }

void fb_connectome_free(FbConnectome *c) {
    if (!c) return;
    for (int i = 0; i < c->n_groups; i++) free(c->groups[i]);
    free(c->groups);
    free(c->group_of);
    free(c->nt);
    free(c->dir);
    free(c->side);
    free(c->hex);
    free(c->nt_conf);
    free(c->edge_src);
    free(c->edge_dst);
    free(c->edge_w);
    free(c->plastic_idx);
    free(c);
}

FbConnectome *fb_connectome_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "connectome: cannot open %s\n", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long fsz = ftell(f);
    if (fsz < 0) { fclose(f); return NULL; }
    rewind(f);
    size_t size = (size_t)fsz;
    uint8_t *raw = (uint8_t *)malloc(size);
    if (!raw || fread(raw, 1, size, f) != size) {
        fprintf(stderr, "connectome: read failed\n");
        free(raw); fclose(f); return NULL;
    }
    fclose(f);

    if (size < 40 || memcmp(raw, "FLYBRAIN1", 9) != 0) {
        fprintf(stderr, "connectome: bad magic\n");
        free(raw); return NULL;
    }
    if (rd_u32(raw + 12) != MAGIC2) {
        fprintf(stderr, "connectome: bad magic2\n");
        free(raw); return NULL;
    }
    FbConnectome *c = (FbConnectome *)calloc(1, sizeof(FbConnectome));
    if (!c) { free(raw); return NULL; }
    c->N = (int)rd_u32(raw + 16);
    c->E = (int)rd_u32(raw + 20);
    c->plastic_count = (int)rd_u32(raw + 28);
    uint32_t name_len = rd_u32(raw + 32);
    size_t off = 36;

    /* group names: NUL-free '|'-joined string */
    {
        char *names = (char *)malloc(name_len + 1);
        memcpy(names, raw + off, name_len);
        names[name_len] = 0;
        off += name_len + (size_t)pad4(name_len);
        /* count groups */
        int ng = 1;
        for (uint32_t i = 0; i < name_len; i++)
            if (names[i] == '|') ng++;
        c->n_groups = ng;
        c->groups = (char **)calloc((size_t)ng, sizeof(char *));
        int gi = 0;
        char *start = names;
        for (uint32_t i = 0; i <= name_len && gi < ng; i++) {
            if (i == name_len || names[i] == '|') {
                names[i] = 0;
                c->groups[gi++] = strdup(start);
                start = names + i + 1;
            }
        }
        free(names);
    }

#define TAKE(dst, ctype, count)                                            \
    do {                                                                   \
        size_t bytes = (size_t)(count) * sizeof(ctype);                    \
        if (off + bytes > size) {                                          \
            fprintf(stderr, "connectome: truncated at section %s\n", #dst);\
            fb_connectome_free(c); free(raw); return NULL;                 \
        }                                                                  \
        dst = (ctype *)malloc(bytes);                                      \
        memcpy(dst, raw + off, bytes);                                     \
        off += bytes + (size_t)pad4(bytes);                                \
    } while (0)

    TAKE(c->group_of, uint8_t, c->N);
    TAKE(c->nt, int8_t, c->N);
    TAKE(c->dir, int8_t, c->N);
    TAKE(c->side, uint8_t, c->N);
    TAKE(c->hex, int16_t, (size_t)c->N * 2);
    TAKE(c->nt_conf, float, c->N);
    TAKE(c->edge_src, uint32_t, c->E);
    TAKE(c->edge_dst, uint32_t, c->E);
    TAKE(c->edge_w, float, c->E);
    TAKE(c->plastic_idx, uint32_t, c->plastic_count);
#undef TAKE

    free(raw);
    fprintf(stderr, "connectome: %d neurons, %d edges, %d plastic, %d groups\n",
            c->N, c->E, c->plastic_count, c->n_groups);
    return c;
}
