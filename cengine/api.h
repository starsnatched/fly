/* flybrain API server: WS /stream + REST /telemetry /actions /memory /health */
#ifndef FB_API_H
#define FB_API_H

#include "runtime.h"

typedef struct FbApi FbApi;

FbApi *fb_api_new(FbRuntime *rt, int ws_port, int rest_port);
void fb_api_free(FbApi *api);

/* run forever (blocks); returns only on fatal error */
int fb_api_serve(FbApi *api);

#endif
