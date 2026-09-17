/* Minimal JSON: DOM parse + string building. Enough for the flybrain API. */
#ifndef FB_JSON_H
#define FB_JSON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef enum {
    FB_JSON_NULL, FB_JSON_BOOL, FB_JSON_NUM, FB_JSON_STR,
    FB_JSON_ARR, FB_JSON_OBJ
} FbJsonType;

typedef struct FbJson FbJson;
struct FbJson {
    FbJsonType type;
    bool b;
    double num;
    char *str;        /* owned, NUL-terminated (STR) */
    FbJson **items;   /* owned (ARR/OBJ values) */
    char **keys;      /* owned (OBJ) */
    int n;
    int cap;
};

/* Parse; returns NULL on error. Needs len+1 readable bytes (we NUL it). */
FbJson *fb_json_parse(char *text_with_nul);

void fb_json_free(FbJson *j);

/* lookups (OBJ only; NULL if missing) */
const FbJson *fb_json_get(const FbJson *obj, const char *key);
double fb_json_num(const FbJson *obj, const char *key, double def);
bool fb_json_bool(const FbJson *obj, const char *key, bool def);
const char *fb_json_str(const FbJson *obj, const char *key, const char *def);

/* growable string builder */
typedef struct {
    char *buf;
    size_t len, cap;
} FbStr;

void fb_str_init(FbStr *s);
void fb_str_free(FbStr *s);
void fb_str_push(FbStr *s, char c);
void fb_str_append(FbStr *s, const char *t);
void fb_str_append_int(FbStr *s, long long v);
void fb_str_append_f(FbStr *s, double v, int prec);
/* append a JSON-escaped string with quotes */
void fb_str_append_json_str(FbStr *s, const char *t);

#endif
