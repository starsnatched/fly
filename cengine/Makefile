# flybrain-c — build and test the native brain server.
#
# The only toolchain dependency is zig (any recent version); if you don't
# have it on PATH, `pip install ziglang` works and ZIG can point at
# `python -m ziglang`.

ZIG ?= zig
CC = $(ZIG) cc
SRC = main.c api.c runtime.c circuit.c config.c flybrain.c json.c sha1.c util.c
BUILD = ../build

UNAME_S := $(shell uname -s 2>/dev/null || echo Windows_NT)
ifeq ($(UNAME_S),Windows_NT)
  TARGET = -target x86_64-windows-gnu
  EXE = .exe
  WS_LIB = -lws2_32 -ladvapi32
else
  TARGET =
  EXE =
  WS_LIB = -lpthread
endif

SERVER = $(BUILD)/flybrain-server$(EXE)
TEST = $(BUILD)/engine_test$(EXE)

.PHONY: all build test engine-test run clean

all: build

build:
	@mkdir -p $(BUILD)
	$(CC) $(addprefix ../cengine/, $(SRC)) -o $(SERVER) -O2 $(TARGET) -lc -lm $(WS_LIB)
	@echo "built $(SERVER)"

engine-test:
	@mkdir -p $(BUILD)
	$(CC) ../cengine/engine_test.c ../cengine/circuit.c ../cengine/flybrain.c \
		../cengine/json.c ../cengine/util.c -o $(TEST) -O2 $(TARGET) -lc -lm $(WS_LIB)
	@echo "--- engine self-test (EMD steering on the real connectome) ---"
	./$(TEST)

run: build
	cd .. && ./$(notdir $(SERVER)) --config config/flybrain.json --profile config/profiles/drone.json

test: engine-test

clean:
	rm -f $(SERVER) $(TEST)
