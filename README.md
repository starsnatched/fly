# FlyBrain — a fruit-fly brain as an API

## Hardware requirements

I respect your time so I decided to put this up front.

I have measured 1.1GB of memory footprint for the C brain engine server. Running this on Intel Core Ultra 7 256V, I could reach 60hz motor output with less than 10ms of processing latency.

I will test it on a physical embodiment with Raspberry Pi 5, processing the brain on the chip. I will update it here once I get meaningful results.

Enjoy :)

---

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

The brain has no idea what body it is flying. It sees eyes and proprioception,
spikes 165k neurons through the real connectome, and answers with actuator
channels. What the body *is* — channels, ranges, sensor layout — is a JSON
profile, negotiated at runtime.

## Quickstart

**With Docker:**

```bash
docker compose -f cengine/docker-compose.yml up --build
# brain: ws://localhost:8787/stream + REST :8788 · clients: drone :5199, rover :5200
```

**Without Docker** (only toolchain dependency: [zig](https://ziglang.org),
or `pip install ziglang`):

```bash
cd cengine && make run                          # brain on :8787 (WS) + :8788 (REST)
cd examples/drone-web && npm i && npm run dev   # drone client on :5199
cd examples/rover-web && npm i && npm run dev   # rover client on :5200
```

Open a client and click launch. Each page declares its body in the opening
`hello` (`{"profile":"rover"}`) and the running brain switches its
actuator/readout anatomy to match — one brain, several bodies. Both shipped
profiles drive their channels through **direct motor pools**: the decode is
learned, not configured (no restart needed to switch bodies).

## AirSim embodiment (physics-real drone)

The browser drone flies a kinematic model. For a *physics-real* body — Unreal
Engine aerodynamics, drag, collisions — the same brain also flies
[AirSim](https://github.com/microsoft/AirSim) (v1.8.1 prebuilt binaries).

```
Unreal world ──center cam──▶ bridge ──eyes 0+1 + state──▶ connectome brain
Unreal physics ◀──velocity sticks── bridge ◀──throttle/pitch/roll/yaw────┘
```

**One-time setup** (AirSim binaries are ~260 MB–1.7 GB, not in git):

```bash
python scripts/get_airsim.py            # Blocks (light, blocky obstacles)
python scripts/get_airsim.py airsimnh   # realistic suburban neighborhood (1.7 GB)
python scripts/get_airsim.py --list     # other environments
```

This also installs `config/airsim.settings.json` to `~/Documents/AirSim/settings.json`
(use `--force-settings` to overwrite). The settings declare the Fly1 drone with
SimpleFlight physics, collisions, and three 192×108 cameras (left/right ±35°
and a forward center).

**Every run:**

```bash
bash scripts/start_airsim_stack.sh          # brain + Blocks + bridge, skips what's up
# or piecewise:
./airsim/Blocks/WindowsNoEditor/Blocks.exe -opengl4 &   # simulator
.venv/Scripts/python.exe scripts/airsim_drone.py        # bridge
```

**How it flies.** The bridge mirrors the browser embodiment's protocol
(`hello profile=drone`, eye frames at `--vision-hz`, barometer/IMU state), but
actuates FPV-joystick style via `moveByVelocityBodyFrameAsync` — SimpleFlight's
flight controller does all prop mixing and attitude stabilization:

| brain channel | stick | range |
|---|---|---|
| `pitch` | forward/backward body velocity | ±12 m/s |
| `roll` | left/right body velocity | ±9 m/s |
| `throttle` | **altitude lane selector** (position control, expo curve) | 1–30 m lane |
| `yaw` | yaw rate | ±200°/s |

Sticks get real FPV feel: `--stick-gain` (default 2.0) amplifies the brain's
small channel wiggles into full deflections, and `--stick-tau` (default 0.15 s)
gives each stick RC-style inertia so sustained channel output integrates into
motion. Altitude is **position control** (`--control lane`, default): the
throttle channel selects a target altitude lane between `--min-alt` and
`--max-alt` and a P controller flies there. A rate-mode climb stick
(`--control rate`) is available, but the untrained throttle bias kept it
pushing skyward — under lane control "climb bias" just means holding a
higher lane, and ceiling punishments map onto the throttle values that chose
them (clean credit assignment for R-STDP). The lane curve uses an RC expo
(`LANE_EXPO` 0.6): the low half of the throttle range is compressed toward
the floor, so small throttle dips select street-level lanes — throttle 0.29
(hover) sits at 4.8 m, 0.10 at 2.2 m, 0.0 at 1 m, where the parked cars are.

Vision defaults to the **forward center camera streamed to both eyes** (the
retina's two hemispheres each read half the image); `--eyes stereo` switches to
the ±35° left/right pair. The brain's throttle fully owns altitude; a safety
band only pushes back within ~1 m of `--min-alt` (1.0 m — below car-roof
height, so car touches are physically possible) / `--max-alt` (30 m).
A takeoff/recovery routine lifts the drone if it gets knocked to the ground.

**Collision policy:** every collision sends a strong punishment pulse
(default **−2.5**) to the brain and respawns the drone at the start point —
*except* collisions with parked cars (`Car_*` in AirSimNH), which send a
strong **+2.5 reward** (touching cars is the task). Riding the ceiling
(`--max-alt`) for more than `--ceil-ride` seconds (2 s) or leaving the spawn
leash (`--border-radius`, 450 m) punishes and respawns the same way.
`--punish-mag` / `--reward-mag` tune the magnitudes. Each event names the
object hit (or `ceiling` / `map border`) in the bridge log. `--reward alt`
additionally enables classic reward shaping (altitude + forward progress) on
top of the event pulses.

**Proximity reward:** every 0.5 s the bridge also streams a small shaping
pulse proportional to closeness of the nearest parked car (3D distance), on
two scales — a **far gradient** (linear to 0 beyond `--prox-radius` 40 m,
`--prox-max` 0.5 at contact) that guides the brain toward a street with cars
from cruise distance, and a **near gradient** (steeper, within
`--near-radius` 8 m, up to `--near-max` 1.0 at contact) for the final
approach. The near max stays below the +2.5 car touch, so touching a car
always pays more than hovering over one. The
brain's relative reward shaping (τ = 12 s) adapts to any steady value, so
this reinforces the *gradient*: closing in on a car is good, drifting away
is bad. The HUD line shows `near <d> m`, `min <d> m` (episode best) and
`prox <+v>` live; `--prox-gain 0` disables it. The 70 parked-car poses are
cached once at startup.

**Car-crash training recipe** — to train the brain to crash into cars:

```bash
python scripts/reset_brain_memory.py --full   # fresh start
bash scripts/start_airsim_stack.sh --cars     # brain + AirSimNH + car curriculum
```

`--cars` is the training curriculum: every respawn teleports the drone to a
fresh 14–22 m start next to the current target car, facing it, and near the
target the shaping switches to pure progress — each 0.5 s tick that *closes*
distance pulses a small reward (`--closing-gain` 0.1/m, capped 0.5), while
hovering or retreating sends nothing. Without it, episodes start from one
fixed spawn and the approach gradient alone never bridged the last meters
to the jackpot (359 episodes, 0 touches).

Watch the `[episode]` lines the bridge prints on every respawn (duration,
hits, car touches, closest approach, closing pulses, mean dopamine) —
`closing` should climb toward double digits per episode and `cars` should
tick up once approaches start connecting with the +2.5 jackpot.

**Resetting learned memory:**

```bash
python scripts/reset_brain_memory.py           # wipe the live brain (hash-verified)
python scripts/reset_brain_memory.py --full    # wipe + set aside the disk memory file
```

Useful flags: `--vision-hz 120` (web client's retina rate), `--min-alt 1.0`
/ `--max-alt 30` (safety band), `--control lane|rate` (altitude scheme),
`--stick-gain 2.0` / `--stick-tau 0.15` (FPV stick feel),
`--ceil-ride 2` / `--border-radius 450` (bounds policy),
`--prox-max 0.5` / `--near-max 1.0` / `--near-radius 8` (shaping scales),
`--no-assist`, `--eyes stereo|center`, `--reward alt`.

## Talking to the brain

Everything lives behind one WebSocket (`/stream`) plus a small REST surface.
Both speak the same protocol as the browser clients — nothing is special
about them.

### Minimal client

```python
import asyncio, json, struct, websockets

W, H = 192, 108   # any resolution works; the brain resamples through the retina

def eye_frame(eye, rgb_bytes):
    return struct.pack("<BBHHB", 1, eye, W, H, 3) + rgb_bytes   # [1][eye][w][h][3]+RGB

def state_frame(alt, speed, vy, clearance, collision=False):
    return struct.pack("<BBffff", 2, 1 if collision else 0, alt, speed, vy, clearance)

async def main():
    async with websockets.connect("ws://localhost:8787/stream") as ws:
        await ws.send(json.dumps({"type": "hello"}))
        print(await ws.recv())                     # circuit info: neurons, edges

        while True:
            await ws.send(eye_frame(0, bytes(W * H * 3)))    # your camera here
            await ws.send(state_frame(4.0, 3.0, 0.0, 30.0))
            msg = await ws.recv()                          # action frame, 60 Hz
            if not isinstance(msg, bytes):
                continue                                   # JSON ack/reply
            nl = struct.unpack_from("<H", msg, 1)[0]
            names = json.loads(msg[3:3 + nl])
            vals = struct.unpack_from("<" + "f" * len(names), msg, 3 + nl)
            print(dict(zip(names, vals)))                  # {"throttle": .., "yaw": ..}

asyncio.run(main())
```

Run that and you are flying the connectome. `hello` can also declare a body:
`{"type": "hello", "profile": "rover"}` — or a path to your own profile JSON.
The reply echoes the applied anatomy (channels, map, pools).

### Frame formats

| message | direction | format |
|---|---|---|
| eye frame | client → brain | `[1 u8][eye u8][w u16][h u16][3 u8] + w*h*3 RGB bytes`. Stereo: eye 0 = LEFT camera (+35°), eye 1 = RIGHT (−35°). Single-eye bodies send eye 0 only — the retina's two hemispheres each view half the image, so turning reads as flow asymmetry. |
| body state | client → brain | binary `[2 u8][flags u8][alt f32][speed f32][vy f32][clearance f32]` (flags bit0 = collision), or JSON `{"type":"state","altitude":..,"vy":..,"collision":..}` |
| action frame | brain → client (60 Hz) | `[10 u8][nameLen u16][names as JSON][f32 × n]` — named actuator channels for the negotiated body |
| JSON | both ways | `hello`, `telemetry`, `control`, `reward` |

### Control, reward, telemetry

```python
await ws.send(json.dumps({"type": "control", "learning": True}))   # toggle R-STDP
await ws.send(json.dumps({"type": "reward", "value": 0.8}))        # reward pulse
await ws.send(json.dumps({"type": "control", "wipe": True}))       # factory memory

tel = json.loads(await ws.recv())    # after {"type":"telemetry"}
# {"neurons":165122,"edges":25542380,"rates":{"T4":5.3,"DAN":9.6,...},
#  "dopa":0.01,"dnSteer":-0.04,"memEdited":1522,"learning":true,"simMs":2.1}
```

The same actions over REST:

```bash
curl -s localhost:8788/health                 # {"ok":true,"clients":2}
curl -s localhost:8788/telemetry              # rates per group, dopa, flow readouts
curl -s localhost:8788/actions                # latest actuator channels
curl -s localhost:8788/memory > learned.json  # export R-STDP memory (see below)
curl -X POST localhost:8788/control -d '{"wipe":true}'
curl -X POST localhost:8788/reward/0.8        # reward; /reward/-1.0 punishes
```

Learning is self-regulated: dopamine is computed *inside* the brain (the DAN
population's deviation from its own adapting baseline — a prediction error).
External rewards only bias DAN excitability; the circuit decides whether that
counts as teaching. Constant situations stop teaching, boot transients never
teach, and `wipe` re-warms a clean brain.

### Saving and restoring learned memory

`GET /memory` returns every learned weight — R-STDP synapse multipliers plus
the plastic motor-pool weights. Send it back to restore:

```python
mem = json.loads(open("learned.json").read())
await ws.send(json.dumps({"type": "control", "learning": False}))
await ws.send(json.dumps({"type": "control", "memory": mem}))
```

The server acks with `{"type":"memoryApplied","n":44277}` (n = weights
applied). Turn learning off first if you want the weights to stick verbatim —
with learning on, an active circuit immediately keeps editing them, which is
usually what you want.

The server also autosaves to `memoryPath` (`config/flybrain.json`) and reloads
it on boot, so a demo survives a restart on its own.## Downloading the full brain data

The connectome binary is ~295 MB (`data/fly-brain-full.bin`, not in git).
**`GET /brain` serves it from localhost because that is the one host that
already has it** — it's how a running instance hands the exact binary it
booted from to a client, container, or analysis script, with no out-of-band
copying. It is not a distribution channel: a fresh machine has no brain to
ask yet. For that:

**Fresh clone — one command:**

```bash
python scripts/get_brain.py
```

Hash-checks what's on disk, then pulls `fly-brain-full.bin` from this repo's
GitHub Releases (set `GITHUB_TOKEN` if the repo is private), or any URL via
`FLY_BRAIN_URL`. It verifies the SHA256 before installing and tells you how
to rebuild from upstream if no release asset exists yet.

**From any running brain** — the server serves the binary it booted from:

```bash
curl -OJ http://localhost:8788/brain            # ~295 MB, application/octet-stream
```

```python
import urllib.request
urllib.request.urlretrieve("http://localhost:8788/brain", "fly-brain-full.bin")
```

The little-endian format (`FLYBRAIN1` header, neuron/population/group tables,
CSR edge arrays) is documented in the docstring of
`scripts/extract_full_brain.py`; `data/fly-brain-full.meta.json` holds the
counts and the dataset/license metadata.

**Rebuild from the raw connectome** — the extractor reads the MaleCNS
exports in `data/` (`malecns-connectome.feather`, `malecns-annotations.feather`,
`neuron-nt.json`, `tbar-nt.feather`) and regenerates the binary, including
2-hop retinotopy inheritance so all 13,585 T4/T5 columns are located. The raw
exports are published by Janelia at
[janelia.org/project-team/flyem/male-cns-connectome](https://www.janelia.org/project-team/flyem/male-cns-connectome)
(CC-BY 4.0):

```bash
python scripts/extract_full_brain.py
```

## Bodies: one brain, any embodiment

A body is a JSON overlay in `config/profiles/`. `drone.json`
(throttle/pitch/roll/yaw) and `rover.json` (throttle/steer) ship as examples,
and both drive channels through **direct motor pools** —
individual connectome motor neurons whose plastic, dopamine-learned spike
integrals ARE the raw channel signal. A hexapod, boat, or cursor is another
JSON file:

- name your actuator `channels` (ranges, slew, defaults),
- declare the `readout.map`: which signal (a population readout like
  `dnSteer`, `flowRoll`, or a motor pool `pool0` / differential `pool1-pool2`)
  drives which channel, with what gain,
- set `sensors.eyes.count` (2 = stereo pair, 1 = forward camera) and which
  scalar state the body can sense.

The readout maps *neural state → named channels* generically — no scripted
behavior, no reflex ladders. Steering away from looming obstacles is a
property of the connectome's own T4/T5 → DN wiring, and it can be retrained
via R-STDP. Channels the map does not drive stay at their configured default.

## Benchmarks & verification

```bash
python scripts/rover_bench.py --fresh          # closed-loop driving twin:
python scripts/drone_bench.py --fresh          #   learn → test-mem → wipe → test-clean,
                                               #   impacts/100m with vs without memory
cd cengine && make test                        # engine self-test (EMD steering)
python scripts/e2e_client.py 8787              # protocol e2e vs the live server
```

Both benches spawn their own brain on their own ports with their own memory
file — your demo memory is untouched. The `-pools` benches measure what
learning is worth: e.g. the drone twin flies 1.71 hits/100 m with learned
pool weights vs 3.50 after wiping them.

`examples/rover-web/?course=gaps` opens the gap-crossing trial in the browser:
pits punish a fall (−1, soft reset), a clean crossing rewards (+0.6).

## Repository layout

```
cengine/              the brain: C sources, Dockerfile, engine self-test
cengine/docker-compose.yml   one-command full stack (brain + example client)
config/               engine config + embodiment profiles (drone, rover, -pools)
data/                 fly-brain-full.bin (~295 MB connectome, not in git) +
                      raw MaleCNS exports; scripts/extract_full_brain.py
                      rebuilds the binary from them
examples/drone-web/   example embodiments (three.js): drone + rover clients —
examples/rover-web/   the only TypeScript in the repo, talks only the API
scripts/              connectome extractor, bootstrap downloader (get_brain.py),
                      AirSim installer (get_airsim.py) + AirSim bridge
                      (airsim_drone.py, with vendored airsim client and a
                      modern msgpack-RPC shim in scripts/msgpackrpc/),
                      protocol test client, benchmarks
state/                learned weights (created at runtime, not in git)
```

## Keys (demo clients)

Both examples: `WASD` nudge · `C` manual override · `R` respawn ·
`L` toggle learning · `M` wipe memory · `U` reward +1 · `J` punish −1.

## Data credit

MaleCNS v1.0 — HHMI Janelia / Google Research, Cell 2026, CC-BY 4.0.
