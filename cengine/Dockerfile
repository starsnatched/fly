# flybrain-c: the full MaleCNS connectome engine + streaming API, in C.
FROM alpine:3.20 AS build
RUN apk add --no-cache zig
WORKDIR /src
COPY cengine/ ./cengine/
# zig cc is a complete C toolchain: musl static build, no runtime deps
RUN zig cc -O2 \
      cengine/main.c cengine/api.c cengine/runtime.c cengine/circuit.c \
      cengine/config.c cengine/flybrain.c cengine/json.c cengine/sha1.c \
      cengine/util.c \
      -o /out/flybrain-server -lc -lm -lpthread

FROM alpine:3.20
WORKDIR /app
COPY --from=build /out/flybrain-server ./flybrain-server
# connectome + config live outside the image, bind-mounted at /data
VOLUME ["/data", "/state"]
EXPOSE 8787 8788
# container paths mirror the repo layout: config/, public/, state/
CMD ["./flybrain-server", "--config", "/data/config/flybrain.json", "--profile", "/data/config/profiles/drone.json"]
