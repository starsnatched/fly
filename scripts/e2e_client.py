"""E2E client test: connect to an EXTERNAL flybrain server (C or Python),
stream binary eye frames + state, assert action frames, telemetry, controls.
Protocol-only, so it verifies any conforming implementation.

Usage: .venv/Scripts/python.exe scripts/e2e_client.py [port]
"""
import asyncio
import json
import struct
import sys
import urllib.request

import numpy as np
import websockets

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787


def eye_frame(eye: int, w: int, h: int, lum: np.ndarray) -> bytes:
    header = struct.pack("<BBHHB", 1, eye, w, h, 3)
    return header + lum.tobytes()


def state_msg(alt: float = 2.0, clr: float = 30.0, coll: bool = False) -> bytes:
    return struct.pack("<BBffff", 2, 1 if coll else 0, alt, 1.0, 0.0, clr)


def parse_action_frame(msg: bytes) -> dict:
    nl = struct.unpack_from("<H", msg, 1)[0]
    names = json.loads(msg[3:3 + nl])
    vals = struct.unpack_from("<" + "f" * len(names), msg, 3 + nl)
    return dict(zip(names, map(float, vals)))


async def main() -> int:
    # REST sanity first
    with urllib.request.urlopen(f"http://localhost:{PORT + 1}/health", timeout=5) as r:
        health = json.loads(r.read())
    assert health.get("ok") is True, health
    print("health:", health)

    ok = False
    async with websockets.connect(f"ws://localhost:{PORT}/stream",
                                  max_size=8 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "hello"}))
        hello = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert hello["type"] == "hello", hello
        circ = hello["circuit"]
        print("hello:", circ)
        assert circ["neurons"] == 165122 and circ["dans"] == 340

        # stream 90 binary eye frames (~3 s at 30 fps): drifting light bar
        w, h = 64, 48
        xs = np.arange(w, dtype=np.float32)
        for i in range(90):
            x0 = int((i * 0.04 * w * 8 / 37) % w)
            lum = (np.full((h, w), 0.35, dtype=np.uint8))
            lum[:, (xs.astype(int) + x0) % w < 5] = 255
            rgb = np.repeat(lum[:, :, None], 3, axis=2)
            await ws.send(eye_frame(0, w, h, rgb))
            await ws.send(eye_frame(1, w, h, rgb))
            await ws.send(state_msg())
            await asyncio.sleep(1 / 30)

        # expect binary action frames broadcast by the push loop
        actions = None
        deadline = asyncio.get_event_loop().time() + 8
        while asyncio.get_event_loop().time() < deadline:
            msg = await asyncio.wait_for(ws.recv(), 5)
            if isinstance(msg, bytes) and msg[0] == 10:
                actions = parse_action_frame(msg)
                break
        assert actions is not None, "no action frame received"
        print("actions:", {k: round(v, 3) for k, v in actions.items()})
        assert set(actions) == {"throttle", "pitch", "roll", "yaw"}
        assert all(np.isfinite(v) for v in actions.values())
        assert 0.0 <= actions["throttle"] <= 1.0

        async def recv_json():
            """Next JSON message, skipping binary broadcasts."""
            while True:
                msg = await asyncio.wait_for(ws.recv(), 10)
                if isinstance(msg, (bytes, bytearray)):
                    if msg[0] == 10:
                        continue
                    return json.loads(bytes(msg).decode())
                return json.loads(msg)

        # telemetry over the same socket
        await ws.send(json.dumps({"type": "telemetry"}))
        tel = await recv_json()
        while tel.get("type") != "telemetry":
            tel = await recv_json()
        print("telemetry: neurons", tel["neurons"], "edges", tel["edges"],
              "spikes/tick", tel.get("spikesPerTick"),
              "dn", round(tel.get("rates", {}).get("descending", 0), 2))
        assert tel["neurons"] == 165122

        # control: enable learning + manual reward pulse
        await ws.send(json.dumps({"type": "control", "learning": True}))
        await ws.send(json.dumps({"type": "reward", "value": 0.8}))
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "telemetry"}))
        tel2 = await recv_json()
        while tel2.get("type") != "telemetry":
            tel2 = await recv_json()
        print("telemetry after reward: dopa", round(tel2.get("dopa", 0), 3),
              "learning", tel2.get("learning"))
        assert tel2.get("learning") is True
        ok = True
    print("E2E CLIENT OK" if ok else "E2E CLIENT FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
