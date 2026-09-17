#include "api.h"
#include "util.h"
#include "sockopt.h"
#include "sha1.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MSG_FRAME 1
#define MSG_STATE 2
#define OP_TEXT 0x1
#define OP_BIN 0x2
#define OP_CLOSE 0x8
#define OP_PING 0x9
#define OP_PONG 0xA

/* ------------------------------------------------------------ ws client */

typedef struct WsClient WsClient;
struct WsClient {
    int fd;
    int want_close;
    WsClient *next;
};

struct FbApi {
    FbRuntime *rt;
    int ws_fd, rest_fd;
    WsClient *clients;
    int n_clients;
    /* outgoing action broadcast buffer */
    uint8_t act[1024];
    int act_len;
    double last_push;
};

/* ------------------------------------------------------------ ws frames */

/* read exactly len bytes (busy-wait with select; only used during handshake
 * and small frames — the main loop is non-blocking) */
static int read_exact(int fd, uint8_t *buf, size_t len, int timeout_ms) {
    size_t got = 0;
    while (got < len) {
        int r = fb_wait_readable(fd, timeout_ms);
        if (r <= 0) return -1;
        int n = fb_recv(fd, buf + got, len - got);
        if (n <= 0) return -1;
        got += (size_t)n;
    }
    return 0;
}

static void ws_mask_copy(uint8_t *dst, const uint8_t *src, size_t len, const uint8_t *key) {
    for (size_t i = 0; i < len; i++) dst[i] = src[i] ^ key[i & 3];
}

/* build one WS frame; returns total length, or -1 if it doesn't fit */
static int ws_frame(uint8_t *out, int cap, int opcode, const uint8_t *payload, size_t plen) {
    int hdr = 2;
    if (plen > 125) hdr = 4;
    if (plen > 0xFFFF) hdr = 10;
    if (cap < hdr + (int)plen) return -1;
    int off = 0;
    out[off++] = (uint8_t)(0x80 | opcode);
    if (plen <= 125) {
        out[off++] = (uint8_t)plen;
    } else if (plen <= 0xFFFF) {
        out[off++] = 126;
        out[off++] = (uint8_t)((plen >> 8) & 0xff);
        out[off++] = (uint8_t)(plen & 0xff);
    } else {
        out[off++] = 127;
        for (int i = 7; i >= 0; i--) out[off++] = (uint8_t)(((uint64_t)plen >> (8 * i)) & 0xff);
    }
    memcpy(out + off, payload, plen);
    return off + (int)plen;
}

static void ws_send(FbApi *api, WsClient *c, int opcode, const uint8_t *payload, size_t plen) {
    size_t cap = plen + 32;
    uint8_t *frame = (uint8_t *)malloc(cap);
    if (!frame) return;
    int len = ws_frame(frame, (int)cap, opcode, payload, plen);
    if (len > 0) fb_send_all(c->fd, frame, (size_t)len);
    free(frame);
    (void)api;
}

/* ------------------------------------------------------- ws handshake */

static void http_send(int fd, const char *status, const char *ctype, const char *body, int blen) {
    char hdr[256];
    int hlen = snprintf(hdr, sizeof(hdr),
                        "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n"
                        "Connection: close\r\n\r\n",
                        status, ctype, blen);
    fb_send_all(fd, (const uint8_t *)hdr, (size_t)hlen);
    if (body && blen) fb_send_all(fd, (const uint8_t *)body, (size_t)blen);
}

static void ws_handshake(FbApi *api, int fd, const char *req, size_t reqlen) {
    (void)reqlen;
    /* find Sec-WebSocket-Key */
    const char *key_hdr = strstr(req, "Sec-WebSocket-Key:");
    if (!key_hdr) key_hdr = strstr(req, "sec-websocket-key:");
    if (!key_hdr) {
        http_send(fd, "400 Bad Request", "text/plain", "no key", 6);
        return;
    }
    key_hdr += strlen("Sec-WebSocket-Key:");
    while (*key_hdr == ' ') key_hdr++;
    const char *end = strstr(key_hdr, "\r\n");
    if (!end) end = key_hdr + strlen(key_hdr);
    char key[128];
    size_t klen = (size_t)(end - key_hdr);
    if (klen >= sizeof(key)) klen = sizeof(key) - 1;
    memcpy(key, key_hdr, klen);
    key[klen] = 0;

    /* accept = b64(sha1(key + GUID)) */
    char accept_src[160];
    snprintf(accept_src, sizeof(accept_src), "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", key);
    uint8_t digest[20];
    fb_sha1((const uint8_t *)accept_src, strlen(accept_src), digest);
    char accept[64];
    fb_base64_encode(digest, 20, accept);

    char resp[256];
    int rlen = snprintf(resp, sizeof(resp),
                        "HTTP/1.1 101 Switching Protocols\r\n"
                        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                        "Sec-WebSocket-Accept: %s\r\n\r\n", accept);
    fb_send_all(fd, (const uint8_t *)resp, (size_t)rlen);

    WsClient *c = (WsClient *)calloc(1, sizeof(WsClient));
    c->fd = fd;
    c->next = api->clients;
    api->clients = c;
    api->n_clients++;
    fprintf(stderr, "api: client connected (%d total)\n", api->n_clients);

    /* immediate hello: circuit size */
    char *tel = fb_runtime_telemetry_json(api->rt);
    FbStr hello;
    fb_str_init(&hello);
    fb_str_append(&hello, "{\"type\":\"hello\",\"circuit\":{\"neurons\":");
    /* reuse telemetry numbers */
    fb_str_append(&hello, tel);
    fb_str_append(&hello, "}");
    /* the above embeds the full telemetry under "circuit" too — harmless,
     * but build a clean minimal hello instead: */
    fb_str_free(&hello);
    fb_str_init(&hello);
    /* extract neurons/edges from telemetry JSON by re-serializing minimal */
    fb_str_append(&hello, "{\"type\":\"hello\",\"circuit\":{\"neurons\":");
    {
        FbRuntime *rt = api->rt;
        fb_rt_lock(rt);
        fb_str_append_int(&hello, rt->net->c->N);
        fb_str_append(&hello, ",\"edges\":");
        fb_str_append_int(&hello, rt->net->c->E);
        fb_str_append(&hello, ",\"plastic\":");
        fb_str_append_int(&hello, rt->net->plastic_n);
        fb_str_append(&hello, ",\"dans\":");
        fb_str_append_int(&hello, rt->net->dan_n);
        fb_rt_unlock(rt);
        fb_str_append(&hello, "}}");
    }
    ws_send(api, c, OP_TEXT, (const uint8_t *)hello.buf, hello.len);
    fb_str_free(&hello);
    free(tel);
}

/* ------------------------------------------------------- json handling */

static void handle_json_msg(FbApi *api, WsClient *c, char *text, size_t len) {
    text[len] = 0;
    FbJson *obj = fb_json_parse(text);
    if (!obj) return;
    const char *type = fb_json_str(obj, "type", "");
    if (strcmp(type, "hello") == 0) {
        char *tel = fb_runtime_telemetry_json(api->rt);
        FbStr s;
        fb_str_init(&s);
        fb_str_append(&s, "{\"type\":\"hello\",\"circuit\":{\"neurons\":");
        fb_rt_lock(api->rt);
        fb_str_append_int(&s, api->rt->net->c->N);
        fb_str_append(&s, ",\"edges\":");
        fb_str_append_int(&s, api->rt->net->c->E);
        fb_str_append(&s, ",\"plastic\":");
        fb_str_append_int(&s, api->rt->net->plastic_n);
        fb_str_append(&s, ",\"dans\":");
        fb_str_append_int(&s, api->rt->net->dan_n);
        fb_rt_unlock(api->rt);
        fb_str_append(&s, "}}");
        ws_send(api, c, OP_TEXT, (const uint8_t *)s.buf, s.len);
        fb_str_free(&s);
        free(tel);
    } else if (strcmp(type, "telemetry") == 0) {
        char *tel = fb_runtime_telemetry_json(api->rt);
        /* wrap with type field */
        FbStr s;
        fb_str_init(&s);
        fb_str_append(&s, "{\"type\":\"telemetry\",");
        fb_str_append(&s, tel + 1); /* strip leading '{' */
        ws_send(api, c, OP_TEXT, (const uint8_t *)s.buf, s.len);
        fb_str_free(&s);
        free(tel);
    } else if (strcmp(type, "state") == 0) {
        fb_runtime_ingest_state(api->rt,
                                (float)fb_json_num(obj, "altitude", 2.0),
                                (float)fb_json_num(obj, "speed", 0.0),
                                (float)fb_json_num(obj, "vy", 0.0),
                                (float)fb_json_num(obj, "clearance", 30.0),
                                fb_json_bool(obj, "collision", false) ? 1 : 0);
    } else if (strcmp(type, "control") == 0) {
        const FbJson *v = fb_json_get(obj, "learning");
        if (v && v->type == FB_JSON_BOOL)
            fb_runtime_set_learning(api->rt, v->b ? 1 : 0);
        if (fb_json_bool(obj, "wipe", false))
            fb_runtime_wipe_memory(api->rt);
        const FbJson *mem = fb_json_get(obj, "memory");
        if (mem) {
            int n = fb_runtime_import_memory(api->rt, mem);
            char resp[64];
            int rl = snprintf(resp, sizeof(resp), "{\"type\":\"memoryApplied\",\"n\":%d}", n);
            ws_send(api, c, OP_TEXT, (const uint8_t *)resp, (size_t)rl);
        }
    } else if (strcmp(type, "reward") == 0) {
        fb_runtime_apply_reward(api->rt, (float)fb_json_num(obj, "value", 0.5));
    }
    fb_json_free(obj);
}

/* ------------------------------------------------------- binary handling */

static void handle_binary_msg(FbApi *api, const uint8_t *data, size_t len) {
    if (len < 6) return;
    uint8_t mtype = data[0];
    if (mtype == MSG_FRAME) {
        uint8_t eye = data[1];
        uint16_t w = (uint16_t)(data[2] | (data[3] << 8));
        uint16_t h = (uint16_t)(data[4] | (data[5] << 8));
        /* channel count implied 3; eye index is config-driven (left/right ids)
         * single-eye embodiments send everything as eye 0 */
        if (len < (size_t)6 + (size_t)w * h * 3) return;
        const FbConfig *cfg = &api->rt->cfg;
        int slot = eye == cfg->eye_left_id ? 0 : (eye == cfg->eye_right_id ? 1 : -1);
        if (slot < 0) return;
        if (cfg->eye_count <= 1 && slot == 1) slot = 0;
        fb_runtime_ingest_frame(api->rt, slot, w, h, data + 6);
    } else if (mtype == MSG_STATE) {
        if (len < 18) return;
        uint8_t flags = data[1];
        float alt, spd, vy, clr;
        memcpy(&alt, data + 2, 4);
        memcpy(&spd, data + 6, 4);
        memcpy(&vy, data + 10, 4);
        memcpy(&clr, data + 14, 4);
        fb_runtime_ingest_state(api->rt, alt, spd, vy, clr, (flags & 1) ? 1 : 0);
    }
}

/* ------------------------------------------------------- ws message pump */

/* process all buffered WS messages for a client; consumes from buf */
#define FB_JSON_MAX 65536  /* max JSON control message (64 KB) */
#define FB_WS_MSG_MAX (8 * 1024 * 1024) /* max single WS frame (8 MB eye frames) */
static void ws_process(FbApi *api, WsClient *c, uint8_t *buf, size_t *len) {
    size_t off = 0;
    for (;;) {
        if (*len - off < 2) break;
        uint8_t *p = buf + off;
        int fin = p[0] & 0x80;
        int opcode = p[0] & 0x0F;
        int masked = p[1] & 0x80;
        uint64_t plen = p[1] & 0x7F;
        size_t hdr = 2;
        if (plen == 126) {
            if (*len - off < 4) break;
            plen = ((uint64_t)p[2] << 8) | p[3];
            hdr = 4;
        } else if (plen == 127) {
            if (*len - off < 10) break;
            plen = 0;
            for (int i = 0; i < 8; i++) plen = (plen << 8) | p[2 + i];
            hdr = 10;
        }
        if (plen > FB_WS_MSG_MAX) { /* absurd frame: drop the connection */
            c->want_close = 1;
            return;
        }
        size_t mask_len = masked ? 4 : 0;
        if (*len - off < hdr + mask_len + plen) break;
        uint8_t *payload = p + hdr + mask_len;
        if (masked) {
            /* unmask in place using the 4-byte key right after the header */
            ws_mask_copy(payload, payload, (size_t)plen, p + hdr);
        }
        if (opcode == OP_PING) {
            ws_send(api, c, OP_PONG, payload, (size_t)plen);
        } else if (opcode == OP_CLOSE) {
            c->want_close = 1;
        } else if (opcode == OP_TEXT || opcode == OP_BIN) {
            if (opcode == OP_BIN) {
                handle_binary_msg(api, payload, (size_t)plen);
            } else if (plen <= FB_JSON_MAX) {
                /* parse JSON in place (needs NUL); save/restore the byte so
                 * pipelined messages after this one are untouched */
                uint8_t saved = payload[plen];
                payload[plen] = 0;
                handle_json_msg(api, c, (char *)payload, (size_t)plen);
                payload[plen] = saved;
            }
        }
        (void)fin;
        off += hdr + mask_len + (size_t)plen;
    }
    if (off) {
        memmove(buf, buf + off, *len - off);
        *len -= off;
    }
}

/* ------------------------------------------------------------ rest api */

static void rest_handle(FbApi *api, int fd, const char *req) {
    /* ---- POST /control: memory reset + learning + external reward ----
     * (universal: any embodiment/environment can shape the brain) */
    char post_path[256];
    if (sscanf(req, "POST %255s", post_path) == 1 &&
        strcmp(post_path, "/control") == 0) {
        const char *body = strstr(req, "\r\n\r\n");
        if (!body) { http_send(fd, "400 Bad Request", "application/json", "{\"error\":\"no body\"}", 22); return; }
        body += 4;
        FbJson *obj = fb_json_parse((char *)body);
        if (!obj) { http_send(fd, "400 Bad Request", "application/json", "{\"error\":\"bad json\"}", 22); return; }
        if (fb_json_bool(obj, "wipe", false) || fb_json_bool(obj, "reset", false))
            fb_runtime_wipe_memory(api->rt);
        const FbJson *v = fb_json_get(obj, "learning");
        if (v && v->type == FB_JSON_BOOL)
            fb_runtime_set_learning(api->rt, v->b ? 1 : 0);
        const FbJson *rew = fb_json_get(obj, "reward");
        if (rew && rew->type == FB_JSON_NUM)
            fb_runtime_apply_reward(api->rt, (float)rew->num);
        fb_json_free(obj);
        http_send(fd, "200 OK", "application/json", "{\"ok\":true}", 11);
        return;
    }
    /* ---- POST /reward: artificial reward / punishment ----------------
     * v > 0 rewards (DAN excitability up, the circuit's own dopa signal
     * turns positive -> LTP); v < 0 punishes. Pulses decay over
     * ~1/rewardDecayPerS seconds so constant inputs stop teaching. */
    char rew_path[256];
    float rew_val;
    if (sscanf(req, "POST %255s %f", rew_path, &rew_val) == 2 &&
        strcmp(rew_path, "/reward") == 0) {
        fb_runtime_apply_reward(api->rt, rew_val);
        char body[96];
        int bl = snprintf(body, sizeof(body),
                          "{\"ok\":true,\"reward\":%.3f}", rew_val);
        http_send(fd, "200 OK", "application/json", body, (size_t)bl);
        return;
    }
    char path[256];
    if (sscanf(req, "GET %255s", path) != 1) {
        http_send(fd, "400 Bad Request", "text/plain", "bad", 3);
        return;
    }
    if (strcmp(path, "/health") == 0) {
        char body[128];
        int bl = snprintf(body, sizeof(body), "{\"ok\":true,\"clients\":%d}", api->n_clients);
        http_send(fd, "200 OK", "application/json", body, bl);
    } else if (strcmp(path, "/telemetry") == 0) {
        char *tel = fb_runtime_telemetry_json(api->rt);
        http_send(fd, "200 OK", "application/json", tel, (int)strlen(tel));
        free(tel);
    } else if (strcmp(path, "/actions") == 0) {
        FbStr s;
        fb_str_init(&s);
        fb_runtime_actions_json(api->rt, &s);
        http_send(fd, "200 OK", "application/json", s.buf, (int)s.len);
        fb_str_free(&s);
    } else if (strcmp(path, "/memory") == 0) {
        char *mem = fb_runtime_memory_json(api->rt);
        http_send(fd, "200 OK", "application/json", mem, (int)strlen(mem));
        free(mem);
    } else {
        http_send(fd, "404 Not Found", "application/json", "{\"error\":\"not found\"}", 22);
    }
}

/* ------------------------------------------------------------ main loop */

int fb_api_serve(FbApi *api) {
    api->last_push = fb_now();
    /* per-connection input buffers (index parallel to client list, but we
     * stash the buffer in a small side table keyed by fd) */
    typedef struct { int fd; uint8_t *buf; size_t len, cap; int is_ws; } ConnBuf;
    ConnBuf conns[64];
    int n_conns = 0;

    while (1) {
        fd_set rfds;
        FD_ZERO(&rfds);
        int maxfd = 0;
        FD_SET((unsigned int)api->ws_fd, &rfds);
        if (api->ws_fd > maxfd) maxfd = api->ws_fd;
        FD_SET((unsigned int)api->rest_fd, &rfds);
        if (api->rest_fd > maxfd) maxfd = api->rest_fd;
        for (int i = 0; i < n_conns; i++) {
            FD_SET((unsigned int)conns[i].fd, &rfds);
            if (conns[i].fd > maxfd) maxfd = conns[i].fd;
        }
        struct timeval tv = { 0, 4000 }; /* 4 ms: bounds action-frame latency */
        int ready = select(maxfd + 1, &rfds, NULL, NULL, &tv);
        double now = fb_now();

        if (ready > 0) {
            /* new WS connections */
            if (FD_ISSET((unsigned int)api->ws_fd, &rfds)) {
                int fd = fb_tcp_accept(api->ws_fd);
                if (fd >= 0 && n_conns < 64) {
                    /* read the HTTP request (blocking, bounded) */
                    char req[2048];
                    size_t rl = 0;
                    double t0 = fb_now();
                    while (fb_now() - t0 < 2.0 && rl < sizeof(req) - 1) {
                        int r = fb_recv(fd, (uint8_t *)req + rl, sizeof(req) - 1 - rl);
                        if (r > 0) {
                            rl += (size_t)r;
                            req[rl] = 0;
                            if (strstr(req, "\r\n\r\n")) break;
                        } else if (r == 0) break;
                        fb_sleep_ms(1);
                    }
                    req[rl] = 0;
                    if (strstr(req, "Upgrade: websocket") || strstr(req, "upgrade: websocket")) {
                        fb_set_nonblocking(fd);
                        ws_handshake(api, fd, req, rl);
                        conns[n_conns].fd = fd;
                        conns[n_conns].buf = NULL;
                        conns[n_conns].len = conns[n_conns].cap = 0;
                        conns[n_conns].is_ws = 1;
                        n_conns++;
                    } else {
                        http_send(fd, "404 Not Found", "text/plain", "ws only", 7);
                        fb_close(fd);
                    }
                } else if (fd >= 0) {
                    fb_close(fd);
                }
            }
            /* new REST connections */
            if (FD_ISSET((unsigned int)api->rest_fd, &rfds)) {
                int fd = fb_tcp_accept(api->rest_fd);
                if (fd >= 0) {
                    char req[2048];
                    size_t rl = 0;
                    double t0 = fb_now();
                    while (fb_now() - t0 < 2.0 && rl < sizeof(req) - 1) {
                        int r = fb_recv(fd, (uint8_t *)req + rl, sizeof(req) - 1 - rl);
                        if (r > 0) {
                            rl += (size_t)r;
                            req[rl] = 0;
                            if (strstr(req, "\r\n\r\n")) break;
                        } else if (r == 0) break;
                        fb_sleep_ms(1);
                    }
                    req[rl] = 0;
                    rest_handle(api, fd, req);
                    fb_close(fd);
                }
            }
            /* client data */
            for (int i = 0; i < n_conns; i++) {
                ConnBuf *cb = &conns[i];
                if (!FD_ISSET((unsigned int)cb->fd, &rfds) || !cb->is_ws) continue;
                uint8_t tmp[65536];
                int r = fb_recv(cb->fd, tmp, sizeof(tmp));
                if (r == 0) {
                    /* closed */
                    fb_close(cb->fd);
                    free(cb->buf);
                    *cb = conns[n_conns - 1];
                    n_conns--;
                    /* remove from client list too */
                    for (WsClient **pp = &api->clients; *pp; pp = &(*pp)->next) {
                        if ((*pp)->fd == cb->fd) {
                            WsClient *dead = *pp;
                            *pp = dead->next;
                            free(dead);
                            api->n_clients--;
                            fprintf(stderr, "api: client disconnected (%d total)\n", api->n_clients);
                            break;
                        }
                    }
                    continue;
                }
                if (r < 0) continue;
                if (cb->len + (size_t)r > cb->cap) {
                    size_t need = cb->len + (size_t)r;
                    cb->cap = need * 2 + 1024;
                    cb->buf = (uint8_t *)realloc(cb->buf, cb->cap + 16);
                }
                memcpy(cb->buf + cb->len, tmp, (size_t)r);
                cb->len += (size_t)r;
                /* find client struct for this fd */
                for (WsClient *c = api->clients; c; c = c->next) {
                    if (c->fd == cb->fd) {
                        ws_process(api, c, cb->buf, &cb->len);
                        break;
                    }
                }
            }
        }

        /* action broadcast at ~60 Hz (API thread owns ch[] via actions_copy) */
        if (now - api->last_push >= 1.0 / 60.0) {
            api->last_push = now;
            int len = fb_runtime_actions_binary(api->rt, api->act, sizeof(api->act));
            if (len > 0) {
                for (WsClient *c = api->clients; c; c = c->next) {
                    uint8_t frame[512];
                    int fl = ws_frame(frame, sizeof(frame), OP_BIN, api->act, (size_t)len);
                    if (fl > 0) fb_send_all(c->fd, frame, (size_t)fl);
                }
            }
        }
    }
    return 0;
}

FbApi *fb_api_new(FbRuntime *rt, int ws_port, int rest_port) {
    FbApi *api = (FbApi *)calloc(1, sizeof(FbApi));
    api->rt = rt;
    api->ws_fd = fb_tcp_listen(ws_port);
    api->rest_fd = fb_tcp_listen(rest_port);
    if (api->ws_fd < 0 || api->rest_fd < 0) {
        fprintf(stderr, "api: cannot bind ports %d/%d\n", ws_port, rest_port);
        fb_api_free(api);
        return NULL;
    }
    return api;
}

void fb_api_free(FbApi *api) {
    if (!api) return;
    fb_close(api->ws_fd);
    fb_close(api->rest_fd);
    WsClient *c = api->clients;
    while (c) {
        WsClient *next = c->next;
        fb_close(c->fd);
        free(c);
        c = next;
    }
    free(api);
}
