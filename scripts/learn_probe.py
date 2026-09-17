"""Watch self-regulated learning: edits grow, synapses re-edit after wipe."""

import asyncio
import json
import struct
import sys

import websockets

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787


def eye_frame(w=96, h=54, phase=0.0):
    hdr = struct.pack("<BBHHB", 1, 0, w, h, 3)
    body = bytearray()
    for y in range(h):
        for x in range(w):
            v = 60 + 40 * ((x + int(phase * 12)) % 16 < 8)
            body += bytes((v, v, v))
    return hdr + bytes(body)


def state_msg(clr=20.0, alt=5.0, vy=0.0, collide=False):
    return json.dumps(
        {
            "type": "state",
            "clearance": clr,
            "altitude": alt,
            "velY": vy,
            "collision": collide,
            "speed": 3.0,
        }
    )


def parse_action_frame(msg: bytes) -> dict:
    nl = struct.unpack_from("<H", msg, 1)[0]
    names = json.loads(msg[3 : 3 + nl])
    vals = struct.unpack_from("<" + "f" * len(names), msg, 3 + nl)
    return dict(zip(names, map(float, vals)))


async def tele(ws):
    await ws.send(json.dumps({"type": "telemetry"}))
    while True:
        m = await ws.recv()
        if isinstance(m, bytes):
            continue  # binary action frame
        msg = json.loads(m)
        if msg.get("type") == "telemetry":
            return msg


async def main():
    async with websockets.connect(f"ws://localhost:{PORT}/stream") as ws:
        await ws.send(json.dumps({"type": "hello"}))
        hello = json.loads(await ws.recv())
        print("hello:", hello.get("circuit", {}).get("plastic"), "plastic synapses")

        # conditioning: alternate BAD epochs (collisions -> punishment) and
        # GOOD epochs (open sky -> reward); the brain should edit both ways
        async def epoch(label, secs, bad):
            for i in range(int(secs / 0.033)):
                await ws.send(eye_frame(phase=i * 0.05))
                if bad and i % 10 == 0:
                    await ws.send(state_msg(clr=0.5, collide=True))
                else:
                    await ws.send(state_msg(clr=30.0))
                await asyncio.sleep(0.033)
            t = await tele(ws)
            print(
                f"{label}: dopa={t['dopa']:+.3f} danHz={t['danHz']:.1f} "
                f"edited={t['memEdited']} lo={t['memLo']:.3f} hi={t['memHi']:.3f}"
            )
            return t

        await epoch("learn +4s", 4.0, bad=True)
        await ws.send(json.dumps({"type": "control", "wipe": True}))
        await asyncio.sleep(0.5)
        t = await tele(ws)
        print(
            f"wiped : edited={t['memEdited']} lo={t['memLo']:.3f} hi={t['memHi']:.3f}"
        )
        t = await epoch("re-edit+4s", 4.0, bad=True)
        t2 = await epoch("re-edit+8s", 4.0, bad=False)
        print(
            "OK"
            if t["memEdited"] > 0 or t2["memEdited"] > 0
            else "FAIL: no re-edits after wipe"
        )


asyncio.run(main())
