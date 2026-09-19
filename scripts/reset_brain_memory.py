"""Reset the fly brain's learned memory.

Sends {"wipe": true} over the brain's WebSocket protocol (the same control
path the browser clients use), resetting every R-STDP weight and motor-pool
weight to factory values in the live circuit. Verifies the wipe by hashing
GET /memory before and after, with learning paused so nothing else can move
the weights mid-test.

  --full  additionally moves state/brain-memory.json aside (to .bak) so the
          next boot is guaranteed clean. NOTE: the running brain autosaves
          every ~5 s, so for a permanently clean disk file, stop the brain
          first (or delete the .bak after it restarts).

Usage:
  python scripts/reset_brain_memory.py           # wipe the running brain
  python scripts/reset_brain_memory.py --full    # wipe + set aside the file
  python scripts/reset_brain_memory.py --port 8787
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMORY = ROOT / "state" / "brain-memory.json"


def rest_get(port: int, path: str, timeout: float = 30.0):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=timeout) as r:
        return json.loads(r.read())


def memory_hash(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/memory",
                                timeout=30) as r:
        return hashlib.md5(r.read()).hexdigest()[:12]


async def ws_control(port: int, msg: dict):
    """Fire a control message. The server acks only memory imports; for
    wipe/learning the hash/telemetry check afterwards is the verification."""
    async with websockets.connect(f"ws://127.0.0.1:{port}/stream",
                                  open_timeout=30, max_size=32 * 2**20) as ws:
        await ws.send(json.dumps({"type": "hello", "profile": "drone"}))
        await ws.recv()                       # hello ack
        await ws.send(json.dumps(msg))
        await asyncio.sleep(0.3)              # let the runtime apply it


async def main_async(args: argparse.Namespace) -> int:
    ws_port, rest_port = args.port, args.port + 1
    try:
        health = rest_get(rest_port, "/health", timeout=5)
        assert health.get("ok") is True
    except Exception as exc:
        sys.exit(f"no brain answering on ws:{ws_port}/rest:{rest_port} ({exc}); "
                 "start it first (bash scripts/start_airsim_stack.sh)")
    print(f"brain up ({health.get('clients', '?')} client(s) connected)")

    tel0 = rest_get(rest_port, "/telemetry")
    was_learning = bool(tel0.get("learning"))
    print(f"before: memEdited={tel0.get('memEdited')} dopa={tel0.get('dopa', 0):+.3f} "
          f"learning={was_learning}")

    # pause learning so the hash proof isolates the wipe's effect
    await ws_control(ws_port, {"type": "control", "learning": False})
    h0 = memory_hash(rest_port)
    print(f"memory hash before wipe: {h0}")

    await ws_control(ws_port, {"type": "control", "wipe": True})
    await asyncio.sleep(1.0)
    h1 = memory_hash(rest_port)
    print(f"memory hash after  wipe: {h1}")
    wiped = h0 != h1
    print("wipe:", "CONFIRMED — weights reset to factory" if wiped
          else "NO CHANGE — wipe did not apply")

    await ws_control(ws_port, {"type": "control", "learning": was_learning})
    print(f"learning restored to {was_learning}")

    if args.full and wiped:
        if DEFAULT_MEMORY.exists():
            backup = DEFAULT_MEMORY.with_suffix(".json.bak")
            DEFAULT_MEMORY.replace(backup)
            print(f"memory file set aside -> {backup.name}")
            print("(the running brain will autosave its clean state shortly; "
                  "delete the .bak for good)")
        else:
            print(f"no memory file at {DEFAULT_MEMORY} (nothing to set aside)")
    return 0 if wiped else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8787, help="brain WS port")
    ap.add_argument("--full", action="store_true",
                    help="also set aside state/brain-memory.json")
    args = ap.parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\naborted")
        return 1


if __name__ == "__main__":
    sys.exit(main())
