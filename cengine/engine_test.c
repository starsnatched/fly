/* In-process engine test: feed moving gratings to fb_tick, print EMD flow. */
#include "circuit.h"
#include "flybrain.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float *grating(int w, int h, float phase, float yphase) {
    float *g = (float *)malloc((size_t)w * h * 3 * sizeof(float));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float v = 0.5f + 0.45f * sinf(2.0f * (float)M_PI * (x + 0.3f * y) / 6.0f + phase + yphase);
            if (v < 0) v = 0;
            if (v > 1) v = 1;
            size_t o = ((size_t)y * w + x) * 3;
            g[o] = g[o + 1] = g[o + 2] = v;
        }
    }
    return g;
}

int main(int argc, char **argv) {
    const char *bin = argc > 1 ? argv[1] : "public/fly-brain-full.bin";
    FbConnectome *c = fb_connectome_load(bin);
    if (!c) return 1;
    FbParams p;
    fb_params_default(&p);
    FbCircuit *n = fb_circuit_new(c, &p);
    if (!n) return 1;
    int W = n->lam_w, H = n->lam_h;
    printf("lamina grid %dx%d, t4=%d t5=%d\n", W, H, n->t4_n, n->t5_n);

    /* cases: (label, lrate, rrate, yrate) at 0.045 rad/tick like Python probe */
    const float R = 0.50f; /* realistic approach: ~0.5 rad/frame */
    const char *labels[6] = { "static", "L-only +x", "R-only +x", "L-only -x", "R-only -x", "both +y drift" };
    float lr[6] = { 0, R, 0, -R, 0, 0 };
    float rr[6] = { 0, 0, R, 0, -R, 0 };
    float yr[6] = { 0, 0, 0, 0, 0, R };
    for (int case_i = 0; case_i < 6; case_i++) {
        /* frame cadence: 6 ticks per frame (~real camera 30 fps at 2 ms dt) */
        for (int i = 0; i < 80; i++) {
            float *gl = grating(W, H, lr[case_i] * (float)(i / 6), yr[case_i] * (float)(i / 6));
            float *gr = grating(W, H, rr[case_i] * (float)(i / 6), yr[case_i] * (float)(i / 6));
            n->frame_new[0] = n->frame_new[1] = (i % 6 == 0);
            fb_tick(n, gl, gr, W, H);
            free(gl); free(gr);
        }
        double hs = 0, vs = 0;
        int frames = 60;
        for (int i = 80; i < 80 + frames * 6; i++) {
            float *gl = grating(W, H, lr[case_i] * (float)(i / 6), yr[case_i] * (float)(i / 6));
            float *gr = grating(W, H, rr[case_i] * (float)(i / 6), yr[case_i] * (float)(i / 6));
            n->frame_new[0] = n->frame_new[1] = (i % 6 == 0);
            fb_tick(n, gl, gr, W, H);
            if (i % 6 == 5) { /* sample once per frame (held value) */
                float hL, hR, vL, vR;
                fb_emd_flow(n, &hL, &hR, &vL, &vR);
                hs += hR - hL;
                vs += 0.5 * (vL + vR);
            }
            free(gl); free(gr);
        }
        printf("%-22s flowH=%+.4f flowV=%+.4f (t4hexed=%d)\n", labels[case_i],
               hs / frames, vs / frames, n->t4_hcount);
    }
    fb_circuit_free(n);
    fb_connectome_free(c);
    return 0;
}
