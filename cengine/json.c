#include "json.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>

/* ------------------------------------------------------------------ parse */

typedef struct {
    const char *p;
    const char *end;
    int depth;
    int failed;
} Parser;

#define FB_JSON_MAX_DEPTH 64

static FbJson *parse_value(Parser *ps);

static FbJson *json_new(FbJsonType t) {
    FbJson *j = (FbJson *)calloc(1, sizeof(FbJson));
    if (j) j->type = t;
    return j;
}

void fb_json_free(FbJson *j) {
    if (!j) return;
    free(j->str);
    for (int i = 0; i < j->n; i++) {
        if (j->items) fb_json_free(j->items[i]);
        if (j->keys) free(j->keys[i]);
    }
    free(j->items);
    free(j->keys);
    free(j);
}

static void skip_ws(Parser *ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' ||
                               *ps->p == '\n' || *ps->p == '\r'))
        ps->p++;
}

static void obj_push(FbJson *j, char *key, FbJson *val) {
    if (j->n >= j->cap) {
        j->cap = j->cap ? j->cap * 2 : 8;
        j->items = (FbJson **)realloc(j->items, (size_t)j->cap * sizeof(FbJson *));
        j->keys = (char **)realloc(j->keys, (size_t)j->cap * sizeof(char *));
    }
    j->items[j->n] = val;
    j->keys[j->n] = key;
    j->n++;
}

static int hex4(const char *p) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else return -1;
    }
    return v;
}

static char *parse_string_raw(Parser *ps) {
    if (ps->p >= ps->end || *ps->p != '"') { ps->failed = 1; return NULL; }
    ps->p++;
    FbStr s;
    fb_str_init(&s);
    while (ps->p < ps->end && *ps->p != '"') {
        char c = *ps->p;
        if (c == '\\') {
            ps->p++;
            if (ps->p >= ps->end) break;
            char e = *ps->p;
            switch (e) {
            case '"': fb_str_push(&s, '"'); break;
            case '\\': fb_str_push(&s, '\\'); break;
            case '/': fb_str_push(&s, '/'); break;
            case 'b': fb_str_push(&s, '\b'); break;
            case 'f': fb_str_push(&s, '\f'); break;
            case 'n': fb_str_push(&s, '\n'); break;
            case 'r': fb_str_push(&s, '\r'); break;
            case 't': fb_str_push(&s, '\t'); break;
            case 'u': {
                if (ps->end - ps->p < 5) { ps->failed = 1; break; }
                int cp = hex4(ps->p + 1);
                if (cp < 0) { ps->failed = 1; break; }
                ps->p += 4;
                /* surrogate pair */
                if (cp >= 0xD800 && cp <= 0xDBFF && ps->end - ps->p >= 7 &&
                    ps->p[1] == '\\' && ps->p[2] == 'u') {
                    int lo = hex4(ps->p + 3);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        ps->p += 6;
                    }
                }
                /* UTF-8 encode */
                if (cp < 0x80) fb_str_push(&s, (char)cp);
                else if (cp < 0x800) {
                    fb_str_push(&s, (char)(0xC0 | (cp >> 6)));
                    fb_str_push(&s, (char)(0x80 | (cp & 0x3F)));
                } else if (cp < 0x10000) {
                    fb_str_push(&s, (char)(0xE0 | (cp >> 12)));
                    fb_str_push(&s, (char)(0x80 | ((cp >> 6) & 0x3F)));
                    fb_str_push(&s, (char)(0x80 | (cp & 0x3F)));
                } else {
                    fb_str_push(&s, (char)(0xF0 | (cp >> 18)));
                    fb_str_push(&s, (char)(0x80 | ((cp >> 12) & 0x3F)));
                    fb_str_push(&s, (char)(0x80 | ((cp >> 6) & 0x3F)));
                    fb_str_push(&s, (char)(0x80 | (cp & 0x3F)));
                }
                break;
            }
            default: ps->failed = 1; break;
            }
            ps->p++;
        } else {
            fb_str_push(&s, c);
            ps->p++;
        }
    }
    if (ps->p >= ps->end) { ps->failed = 1; fb_str_free(&s); return NULL; }
    ps->p++; /* closing quote */
    if (ps->failed) { fb_str_free(&s); return NULL; }
    if (!s.buf) { s.buf = (char *)malloc(1); s.buf[0] = 0; s.len = 0; }
    return s.buf;
}

static FbJson *parse_value(Parser *ps) {
    if (ps->depth >= FB_JSON_MAX_DEPTH) { ps->failed = 1; return NULL; }
    skip_ws(ps);
    if (ps->p >= ps->end) { ps->failed = 1; return NULL; }
    char c = *ps->p;
    if (c == '{') {
        ps->depth++;
        FbJson *obj = json_new(FB_JSON_OBJ);
        if (!obj) { ps->failed = 1; ps->depth--; return NULL; }
        ps->p++;
        skip_ws(ps);
        if (ps->p < ps->end && *ps->p == '}') { ps->p++; ps->depth--; return obj; }
        for (;;) {
            skip_ws(ps);
            char *key = parse_string_raw(ps);
            if (ps->failed) { fb_json_free(obj); ps->depth--; return NULL; }
            skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != ':') {
                free(key); fb_json_free(obj); ps->failed = 1; ps->depth--; return NULL;
            }
            ps->p++;
            FbJson *val = parse_value(ps);
            if (!val) { free(key); fb_json_free(obj); ps->depth--; return NULL; }
            obj_push(obj, key, val);
            skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == '}') { ps->p++; ps->depth--; return obj; }
            fb_json_free(obj); ps->failed = 1; ps->depth--; return NULL;
        }
    }
    if (c == '[') {
        ps->depth++;
        FbJson *arr = json_new(FB_JSON_ARR);
        if (!arr) { ps->failed = 1; ps->depth--; return NULL; }
        ps->p++;
        skip_ws(ps);
        if (ps->p < ps->end && *ps->p == ']') { ps->p++; ps->depth--; return arr; }
        for (;;) {
            FbJson *val = parse_value(ps);
            if (!val) { fb_json_free(arr); ps->depth--; return NULL; }
            obj_push(arr, NULL, val);
            skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == ']') { ps->p++; ps->depth--; return arr; }
            fb_json_free(arr); ps->failed = 1; ps->depth--; return NULL;
        }
    }
    if (c == '"') {
        char *s = parse_string_raw(ps);
        if (!s) return NULL;
        FbJson *j = json_new(FB_JSON_STR);
        if (!j) { free(s); ps->failed = 1; return NULL; }
        j->str = s;
        return j;
    }
    if (c == 't' && ps->end - ps->p >= 4 && strncmp(ps->p, "true", 4) == 0) {
        ps->p += 4;
        FbJson *j = json_new(FB_JSON_BOOL);
        if (j) j->b = true;
        else ps->failed = 1;
        return j;
    }
    if (c == 'f' && ps->end - ps->p >= 5 && strncmp(ps->p, "false", 5) == 0) {
        ps->p += 5;
        FbJson *j = json_new(FB_JSON_BOOL);
        if (j) j->b = false;
        else ps->failed = 1;
        return j;
    }
    if (c == 'n' && ps->end - ps->p >= 4 && strncmp(ps->p, "null", 4) == 0) {
        ps->p += 4;
        return json_new(FB_JSON_NULL);
    }
    /* number */
    {
        char *endp = NULL;
        double v = strtod(ps->p, &endp);
        if (endp == ps->p) { ps->failed = 1; return NULL; }
        ps->p = endp;
        FbJson *j = json_new(FB_JSON_NUM);
        if (j) j->num = v;
        else ps->failed = 1;
        return j;
    }
}

FbJson *fb_json_parse(char *text_with_nul) {
    Parser ps;
    ps.p = text_with_nul;
    ps.end = text_with_nul + strlen(text_with_nul);
    ps.depth = 0;
    ps.failed = 0;
    FbJson *j = parse_value(&ps);
    if (!j || ps.failed) { fb_json_free(j); return NULL; }
    return j;
}

const FbJson *fb_json_get(const FbJson *obj, const char *key) {
    if (!obj || obj->type != FB_JSON_OBJ) return NULL;
    for (int i = 0; i < obj->n; i++)
        if (obj->keys[i] && strcmp(obj->keys[i], key) == 0) return obj->items[i];
    return NULL;
}

double fb_json_num(const FbJson *obj, const char *key, double def) {
    const FbJson *v = fb_json_get(obj, key);
    return (v && v->type == FB_JSON_NUM && isfinite(v->num)) ? v->num : def;
}

bool fb_json_bool(const FbJson *obj, const char *key, bool def) {
    const FbJson *v = fb_json_get(obj, key);
    return (v && v->type == FB_JSON_BOOL) ? v->b : def;
}

const char *fb_json_str(const FbJson *obj, const char *key, const char *def) {
    const FbJson *v = fb_json_get(obj, key);
    return (v && v->type == FB_JSON_STR && v->str) ? v->str : def;
}

/* ------------------------------------------------------------------ build */

void fb_str_init(FbStr *s) {
    s->buf = NULL;
    s->len = 0;
    s->cap = 0;
}

void fb_str_free(FbStr *s) {
    free(s->buf);
    s->buf = NULL;
    s->len = s->cap = 0;
}

void fb_str_push(FbStr *s, char c) {
    if (s->len + 1 >= s->cap) {
        s->cap = s->cap ? s->cap * 2 : 64;
        s->buf = (char *)realloc(s->buf, s->cap);
    }
    s->buf[s->len++] = c;
    s->buf[s->len] = 0;
}

void fb_str_append(FbStr *s, const char *t) {
    while (*t) fb_str_push(s, *t++);
}

void fb_str_append_int(FbStr *s, long long v) {
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%lld", v);
    fb_str_append(s, tmp);
}

void fb_str_append_f(FbStr *s, double v, int prec) {
    char tmp[64];
    snprintf(tmp, sizeof(tmp), "%.*f", prec, v);
    fb_str_append(s, tmp);
}

void fb_str_append_json_str(FbStr *s, const char *t) {
    fb_str_push(s, '"');
    for (; *t; t++) {
        unsigned char c = (unsigned char)*t;
        switch (c) {
        case '"': fb_str_append(s, "\\\""); break;
        case '\\': fb_str_append(s, "\\\\"); break;
        case '\n': fb_str_append(s, "\\n"); break;
        case '\r': fb_str_append(s, "\\r"); break;
        case '\t': fb_str_append(s, "\\t"); break;
        default:
            if (c < 0x20) {
                char tmp[8];
                snprintf(tmp, sizeof(tmp), "\\u%04x", c);
                fb_str_append(s, tmp);
            } else {
                fb_str_push(s, (char)c);
            }
        }
    }
    fb_str_push(s, '"');
}
