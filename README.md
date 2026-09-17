# FlyBrain — a fruit-fly brain as an API

The **entire traced MaleCNS v1.0 male fruit-fly CNS** (Janelia/Google, Cell
2026 — CC-BY) — **165,122 spiking neurons wired by 25.5M real connectome
synapses** — runs as a standalone C service. Any client that can send eye
images and body state can use it: a drone in a browser, a rover, a robot arm,
a headless benchmark.

```
eye RGB + body state ──WS :8787/stream──▶  C engine
                                           retinotopic photoreceptors
                                           T4/T5 motion detectors (EMDs)
                                           DAN reward + R-STDP memory
actuator channels ◀────60 Hz────────────  165,122-neuron LIF connectome
```

## Repository layout

```
cengine/              the brain: C sources, Dockerfile, engine self-test
cengine/docker-compose.yml   one-command full stack (brain + example client)
config/               engine config + embodiment profiles (drone, rover)
data/                 fly-brain-full.bin — the ~300 MB connectome
                      (not in git; scripts/extract_full_brain.py rebuilds it)
examples/drone-web/   one example embodiment: three.js drone client —
                      the only TypeScript in the repo, talks only the API
scripts/              connectome extractor + protocol test clients
state/                learned weights (created at runtime, not in git)
```

## Quickstart (C brain API)

**With Docker:**

```bash
docker compose -f cengine/docker-compose.yml up --build
# brain: ws://localhost:8787/stream + REST :8788 · client: http://localhost:5199
```

**Without Docker** (only toolchain dependency: [zig](https://ziglang.org),
or `pip install ziglang`):

```bash
cd cengine && make run                          # brain on :8787 (WS) + :8788 (REST)
cd examples/drone-web && npm i && npm run dev   # example client on :5199
```

**Talk to the brain yourself** — any WebSocket client works:

```python
import asyncio, json, struct, websockets


async def main():
    async with websockets.connect("ws://localhost:8787/stream") as ws:
        await ws.send(json.dumps({"type": "hello"}))
        print(await ws.recv())  # circuit info
        while True:
            for eye in (0, 1):  # one frame per eye
                w, h = 192, 108
                rgb = bytearray(w * h * 3)  # your camera here
                await ws.send(struct.pack("<BBHHB", 1, eye, w, h, 3) + rgb)
            await ws.send(struct.pack("<BBffff", 2, 0, 4.0, 3.0, 0.0, 30.0))
            msg = await ws.recv()  # action frame at 60 Hz
            nl = struct.unpack_from("<H", msg, 1)[0]
            names = json.loads(msg[3 : 3 + nl])
            vals = struct.unpack_from("<" + "f" * len(names), msg, 3 + nl)
            print(dict(zip(names, vals)))  # {"throttle": .., "yaw": ..}


asyncio.run(main())
```

## The API

Everything lives behind one WebSocket plus a read-only REST surface.
See `cengine/README.md` for the full protocol and `config/flybrain.json`
for every knob.

| WS `/stream` | direction | format |
|---|---|---|
| eye frames | client → brain | `[1][eye][w16][h16][3] + w*h*3 RGB bytes` (any resolution; the brain resamples through the retina's real hex coordinates. Stereo: eye 0 = LEFT camera (+35°), eye 1 = RIGHT (−35°). Single-eye embodiments send eye 0 only) |
| body state | client → brain | `[2][flags u8][alt f32][speed f32][vy f32][clearance f32]` (flags bit0 = collision) |
| action frame | brain → client (60 Hz) | `[10][nameLen u16][names JSON][f32 × n]` |
| JSON | both ways | `hello`, `telemetry`, `control` (learning/wipe/memory), `reward` |

| REST `:8788` | returns |
|---|---|
| `GET /health` | liveness + client count |
| `GET /telemetry` | firing rates per group, dopa, memory stats, flow readouts |
| `GET /actions` | latest actuator channels |
| `GET /memory` | learned-weight export (R-STDP) |
| `POST /control` | `{"wipe":true}` reset memory to defaults · `{"learning":bool}` · `{"reward":v}` shape from your environment |

## Modular & universal

The brain has no idea what body it is flying — that is all config:

- **`config/flybrain.json`** — engine, sensors, readout, reward, ports.
- **`config/profiles/*.json`** — per-embodiment overlays. `drone.json`
  (throttle/pitch/roll/yaw) and `rover.json` (throttle/steer) ship as
  examples. A hexapod, boat, or cursor is another JSON file: name your
  actuator channels, set their ranges/slew, declare the `readout.map`
  (which neural population drives which channel, with what gain), set
  `sensors.eyes.count` (2 = stereo pair, left eye mounted +35° / right −35°;
  1 = one forward camera split across the retina's two hemispheres), and pick
  which scalar state you send.
- The readout maps *neural state → named channels* generically — no scripted
  behavior, no reflex ladders: vision and touch enter the circuit as neural
  input, and everything the body does is what the connectome (plus R-STDP
  memory) does with them. Channels the map does not drive stay at their
  configured default.
- **Channel contract (drone):** `throttle` tracks motor-population drive
  (+ vertical optic flow), `pitch` tracks descending drive, `yaw` and `roll`
  track descending left/right asymmetry and horizontal optic flow. There are
  no built-in altitude hold, saccades, or escape sequences — steering away
  from looming obstacles is a property of the connectome's own T4/T5 → DN
  wiring, and it can be retrained via R-STDP.
- **Learning is self-regulated**: dopamine is computed *inside* the brain —
  the DAN population's own firing deviation from its adapting internal
  baseline (a prediction error). External signals (embodiment rewards,
  `POST /control {"reward":v}`) only bias DAN excitability; the circuit
  itself decides whether that counts as teaching. Constant situations stop
  teaching, boot transients never teach (warmup gating), edited synapses
  stay editable forever (soft bounds), and `{"wipe":true}` re-warms a
  clean brain.

## Architecture

- **`cengine/`** — the native brain service (C). LIF dynamics with per-group
  biophysics, axonal delays, synaptic depression, retinotopic photoreceptors,
  frame-locked Hassenstein–Reichardt EMDs on the connectome's real T4/T5
  preferred-direction subtypes, dopaminergic reward, R-STDP memory on
  descending synapses, mushroom-body circuit. Details: `cengine/README.md`.
- **`examples/drone-web/`** — one example embodiment (three.js): renders the
  world, captures **two 192×108 RGB eye cameras** (±35°, streamed at 30 fps),
  sends proprioception, applies the returned channels to drone physics.
  Pure sensor/actuator — no neural code; swap it for your own client.
- **`scripts/extract_full_brain.py`** — builds `data/fly-brain-full.bin`
  from the raw connectome, including 2-hop retinotopy inheritance so all
  13,585 T4/T5 columns are located.
- **`scripts/e2e_client.py`** — implementation-agnostic protocol test.

## Verification

```bash
cd cengine && make test                       # engine self-test (EMD steering)
python scripts/e2e_client.py 8787               # protocol e2e vs the live server
```

## Keys (demo client)

`WASD` nudge · `C` manual override · `R` respawn · `L` toggle learning ·
`M` wipe memory.
