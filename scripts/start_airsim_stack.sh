#!/usr/bin/env bash
# Launch the full AirSim + flybrain stack:
#   1. brain server (cengine) on ws://localhost:8787  [skipped if already up]
#   2. Blocks.exe (AirSim) with the fly-drone settings [skipped if already up]
#   3. the bridge: streams the center camera to the connectome, flies the
#      drone with FPV joystick semantics (body-frame velocity + yaw rate)
#
# Usage: bash scripts/start_airsim_stack.sh [--eyes stereo] [--no-assist] ...
#        (extra args are passed to scripts/airsim_drone.py)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=python3

# 1. brain -----------------------------------------------------------------
if ! curl -s --max-time 2 http://127.0.0.1:8788/health | grep -q '"ok":true'; then
  echo "[stack] starting brain server..."
  nohup ./build/flybrain-server.exe --config config/flybrain.json \
      --profile config/profiles/drone.json > airsim/brain.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -s --max-time 2 http://127.0.0.1:8788/health | grep -q '"ok":true' && break
    sleep 1
  done
else
  echo "[stack] brain already up on :8787/:8788"
fi

# 2. AirSim ----------------------------------------------------------------
if ! netstat -ano | grep -q ":41451.*LISTENING"; then
  BLOCKS=$(ls airsim/*/WindowsNoEditor/*.exe 2>/dev/null | grep -viE "Binaries|Engine" | head -1)
  if [ -z "$BLOCKS" ]; then
    echo "[stack] no AirSim environment found; run: python scripts/get_airsim.py" >&2
    exit 1
  fi
  echo "[stack] starting $BLOCKS (AirSim)..."
  nohup "./$BLOCKS" -opengl4 > airsim/blocks.log 2>&1 &
  for _ in $(seq 1 60); do
    netstat -ano | grep -q ":41451.*LISTENING" && break
    sleep 2
  done
else
  echo "[stack] AirSim already up on :41451"
fi

# 3. bridge ----------------------------------------------------------------
echo "[stack] starting bridge: $PY scripts/airsim_drone.py $*"
exec "$PY" scripts/airsim_drone.py "$@"
