#include "runtime.h"
#include "api.h"
#include "util.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *config_path = "config/flybrain.json";
    const char *profile = NULL;
    int port = 8787;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) config_path = argv[++i];
        else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            profile = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
    }
    char profile_path[512] = "";
    if (profile && profile[0]) {
        if (strchr(profile, '/') || strchr(profile, '\\') ||
            (strlen(profile) > 5 && strcmp(profile + strlen(profile) - 5, ".json") == 0))
            snprintf(profile_path, sizeof(profile_path), "%s", profile); /* full path */
        else
            snprintf(profile_path, sizeof(profile_path), "config/profiles/%s.json", profile);
    }

    if (fb_net_init() != 0) {
        fprintf(stderr, "network init failed\n");
        return 1;
    }

    FbConfig cfg;
    fb_config_load(&cfg, config_path, profile_path);
    fprintf(stderr, "flybrain-c: config %s profile %s\n", config_path,
            profile_path[0] ? profile_path : "(none)");

    FbRuntime *rt = fb_runtime_new(&cfg);
    if (!rt) {
        fprintf(stderr, "runtime init failed\n");
        return 1;
    }
    fb_runtime_start(rt);

    int ws_port = port, rest_port = port + 1;
    if (cfg.rest_port > 0 && port == 8787) { /* ports from config unless overridden */
        ws_port = cfg.ws_port;
        rest_port = cfg.rest_port;
    }
    FbApi *api = fb_api_new(rt, ws_port, rest_port);
    if (!api) {
        fb_runtime_free(rt);
        return 1;
    }
    fprintf(stderr,
            "flybrain-c: ws://localhost:%d/stream + REST http://localhost:%d/\n"
            "  %d neurons, %d synapses, %d plastic, memory %s\n",
            ws_port, rest_port, rt->net->c->N, rt->net->c->E, rt->net->plastic_n,
            rt->learning ? "on" : "off");

    fb_api_serve(api);

    fb_api_free(api);
    fb_runtime_free(rt);
    return 0;
}
