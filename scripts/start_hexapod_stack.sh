#!/usr/bin/env bash
# Launch the MuJoCo hexapod + flybrain stack:
#   1. hexapod brain server (ports 8793 WS / 8794 REST, memory
#      state/hexa-memory.json)          [skipped if already up]
#   2. the bridge: the preconfigured 18-servo hexapod (config/
#      hexapod_world.xml) whose forward-facing stereo eyes + 18-joint
#      proprioceptive state stream to the brain and whose 18 joint channels
#      (one DISJOINT motor pool of 45 descending neurons per servo) drive
#      every servo directly — no gait library.
#
# Usage: bash scripts/start_hexapod_stack.sh [extra bridge args...]
#   e.g. bash scripts/start_hexapod_stack.sh --wind-gain 0 --no-viewer
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=python3

port_open() {
  "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); \
sys.exit(0 if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0 else 1)" "$1"
}

# 1. hexapod brain -----------------------------------------------------------
if ! curl -s --max-time 2 http://127.0.0.1:8794/health | grep -q '"ok":true'; then
  echo "[hexa-stack] starting hexapod brain (ports 8793/8794)..."
  nohup ./build/flybrain-server.exe --config config/hexa-brain.json \
      --profile hexapod > airsim/hexa-brain.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -s --max-time 2 http://127.0.0.1:8794/health | grep -q '"ok":true' && break
    sleep 1
  done
else
  echo "[hexa-stack] hexapod brain already up on :8793/:8794"
fi

# 2. bridge --------------------------------------------------------------------
echo "[hexa-stack] starting bridge: $PY scripts/hexapod_sim.py $*"
exec "$PY" -u scripts/hexapod_sim.py "$@"
