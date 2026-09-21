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

**Where the learning lives (226,293 plastic synapses):** the strongest 128
inputs per descending neuron (165,083 — what visual features drive
steering) **plus every KC→MBON synapse in the mushroom body (61,210)** —
the fly's canonical DAN-gated associative site. Sparse Kenyon codes mean
the memory center costs almost nothing to train; MBON→head wiring stays
fixed anatomy. Rebuild via `scripts/extract_full_brain.py` (the
`MB_PLASTIC` flag), or flip it off to return to the DN-only brain
(pre-MB connectome backed up as `data/fly-brain-full.preMB.bin`).

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
| `throttle` | **altitude lane selector** (position control, expo curve) | ground–60 m lane |
| `yaw` | yaw rate | ±200°/s |

Sticks get real FPV feel: `--stick-gain` (default 3.0) amplifies the brain's
small channel wiggles into full deflections, and `--stick-tau` (default 0.15 s)
gives each stick RC-style inertia so sustained channel output integrates into
motion.

**Auto-trim — why the quad can now actually hover.** The brain's motor
channels are *potentials*, not centered sticks: the untrained circuit rests
with a standing bias on pitch/roll/yaw (it wanders: −0.31 → −0.35 across
episodes), so the naive mapping (raw channel = stick deflection) meant the
drone always crept forward, spun, and drifted — it could never sit still.
At every respawn the bridge calibrates each attitude channel's **resting
value** (1.5 s median of quiet samples; active windows are rejected and
resampled) and then **freezes** the trim for the whole episode. Frozen is
the crucial property: a continuous adaptive trim absorbs sustained channel
offsets — but flight *commands are* sustained offsets, and an adaptive trim
ate them (the drone could only spin and change altitude). Deflections from
the frozen center are real stick input for the entire episode: forward,
backward, lateral, climb, spin all work; a resting brain = centered sticks =
true hover. `--no-trim` restores the old raw mapping.

The **yaw stick** is special-cased, because the untrained circuit's yaw
channel swings ±0.2–0.35 stick at ~0.3 Hz — too slow for any usable
low-pass to remove. The stick is smoothed at τ=2 s, then gated by an RC-style
deadzone on the *smoothed* value (zero below 0.3 stick, full authority by
0.5), and demand under 40°/s commands **heading hold** (`YawMode` rate off)
instead of a rate — so the nose sits rock-still exactly like a real quad in
heading-hold mode, while genuine sustained spin intent still turns it.

**Stagnation pressure.** A collapsed policy that never deflects anything
(park the quad, bob in place) generates no events, no gradients, no
learning — refusal must cost. If the quad covers less than
`--stagnate-disp` (1.5 m) of net XY ground in `--stagnate-s` (45 s), that's
punished like a collision (`-punish-mag`, scene held for the dopamine tail)
and respawned. Measured as net horizontal displacement, not speed: the lane
controller's ±2 m/s vertical bobbing would defeat any speed threshold while
the drone goes exactly nowhere. Brief hovers never trigger it.

**Motor babble — the carrot that makes it fly.** The brain is *reactive*,
not exploratory: it has no action sampler, and a parked drone staring at a
static scene produces no retinal change → no sensory drive → motor pools at
rest → no movement. A degenerate fixed point (up/down still worked only
because lane control is position control — the P-controller flies any
throttle number without the brain commanding it). The bridge therefore
injects a course-following "exploration wind": a persistent course whose
turn-rate random-walks (up to 120°/s, hard-turn-biased sampling, 25% of
retargets hold straight) on a ~4.5 s clock, cruised at 4–9 m/s — FPV-fast,with a 150°/s nose-chase so
the camera whips through turns without pirouetting faster than the drone
flies. A yaw-rate wind chases the nose onto
the course, so flight reads as **forward cruise with real turns** when the
course changes, and the camera looks where it flies. Only the forward part
of the wind enters the body-frame velocity — an earlier version injected
lateral wind too, which turned every course change into sideways crabbing
(the drone seemed to only strafe). The brain's own commands ADD to the
wind, and inside 15 m of the target car the wind yields to 35% so the
brain owns the close-in game (wind yields to 60% there — 35% made close-in
forward slower than the vertical controller and the drone crawled exactly
where the game happens) (`--no-dither` disables). As the carrot side:
sustained horizontal cruise outside the near zone earns a small periodic
pulse (+0.05 every 2 s) — moving must pay before any approach gradient can
be discovered. Together with the stagnation punish
this keeps the drone in the regime where optic flow, collisions, proximity
pulses and the car jackpot all actually occur — the event stream R-STDP
learns from.

Altitude is **position control** (`--control lane`, default): the
throttle channel selects a target altitude lane between `--min-alt` and
`--max-alt` and a P controller flies there. A rate-mode climb stick
(`--control rate`) is available, but the untrained throttle bias kept it
pushing skyward — under lane control "climb bias" just means holding a
higher lane, and ceiling punishments map onto the throttle values that chose
them (clean credit assignment for R-STDP). The lane curve uses an RC expo
(`LANE_EXPO` 0.6): the low half of the throttle range is compressed toward
the floor, so small throttle dips select street-level lanes — throttle 0.29
(hover) sits near 7 m, ~0.20 dips to street level, 0.0 reaches the ground
lane where the parked cars are.

Vision defaults to **fly vision** (`--eyes fly`): the drone is fitted with a
wrap-around compound eye — four 130° facets (front pair + rear pair at
±115°) stitched into a 360°×120° panorama, and each eye samples a
**240°-wide window centered 30° off boresight** (left eye left, right eye
right): the pair covers ~300° with true stereo overlap, and forward optic
flow radiates from near the retina CENTER — the geometry the connectome
has always seen (an earlier hemisphere-per-eye layout put the front at the
retina edge, broke the brain's forward prior, and it preferred flying
backward). Then the fly visual processing:

1. **Spectral weighting** — R1-6 photoreceptor response, strongly
   green-weighted (the fly's peak sensitivity), linearized, Naka-Rushton
   soft-saturated with a photon-shot noise floor.
2. **Ommatidial optics** — each eye's window is blurred by the facet
   PSF (σ≈0.9 px @192×108): low acuity, wide acceptance angles.
3. **Phasic motion channel** — an LMC-style high-pass (τ≈55 ms) mixed over
   the sustained response: the retina responds to *change*, the lobula's
   input.

Each eye's window is the exact 192×108 frame in the same wire format —
the connectome genuinely sees through fly eyes. `--eyes stereo` restores
the plain ±35° camera pair, `--eyes center` the single forward camera
duplicated to both eyes.

The drone's **own body is hidden from its eyes** (Unreal `ke <pawn> 0`
console toggle): the rear fly facets stare straight back at the airframe.
Pose sets (takeoff, stuck-recovery, respawns) re-show the pawn, so the
bridge re-hides after every one of them AND re-asserts every 5 s; the
other vehicle is parked 400 m away so it can't pose as a body either.
Verified with a frozen-world toggle test (scene paused, so the body is
the only variable). The vehicle stays fully RPC-controllable while
hidden.

**Supersampled captures** (`config/airsim.settings.json`): cameras render
384×216 and the fly-optics stage area-downsamples once to the 192×108
retina — sharper facet averaging without changing the brain's retina
grid.

**Low-graphics mode** (`--gfx`, default on): on connect the bridge floors
the game's scalability settings (resolution 25%, shadows/effects/
postprocess/textures/view-distance/AA at 0) so the sim renders faster.
Re-applied on every sim (re)connect; `--no-gfx` keeps stock visuals.

**`ViewMode`** (`$USERPROFILE/Documents/AirSim/settings.json`): the default
`"FlyWithMe"` renders the third-person chase camera in the game window so
you can watch the drone fly. Setting `"NoDisplay"` disables the main
viewport entirely — it was eating nearly half the frame budget (the eyes
are scene captures; you'd watch the viewer instead) — measured: capture
batches 120–140 ms → **74–78 ms**. With the viewport ON, the bridge's
parallel capture pool still delivers **24–28 captures/s serial-limited,
~66–70 eye-frames/s end-to-end**.

**Parallel eye capture** — the sim's RPC serializes requests per
connection, but its render pipeline takes captures CONCURRENTLY, so the
bridge opens **one connection + one thread per camera**: measured 10–15
captures/s on one socket → **~58–66/s across four**.

**Net eye rate: ~2.6 → 66–70 eye-frames/s** with the third-person view
rendering (up to ~80+ with `NoDisplay`; the remainder of the budget is the
fly optics + viewer JPEG + brain-tick cost). `--vision-hz` remains a cap
(default uncapped = sim-paced).

**Fly-vision viewer:** while flying, the bridge serves what the brain
literally receives at `http://localhost:8795` — both retinas on top (after
the full optics), the raw wrap-around panorama below (`--viewer-port 0`
disables). The brain's throttle fully owns altitude; a safety
band only pushes back within ~1 m of `--min-alt` (0 m = ground; a climb-out
guard keeps it from burrowing below 0.8 m) / `--max-alt` (60 m).
A takeoff/recovery routine lifts the drone if it gets knocked to the ground.

**Fixed-wing aircraft:** stock AirSim has no fixed-wing physics, so the bridge
adds a point-mass wing model on top of the velocity API
(`--airframe wing`, vehicle `Wing1`): the same four brain channels become
plane controls — throttle → airspeed (10–24 m/s, **stall below 10**), pitch →
elevator (climb/sink, climbing bleeds airspeed), roll → bank angle (banked
coordinated turns, `g·tan(bank)/V`, capped 45°), yaw → rudder. Respawns are
airborne catapult launches at cruise speed, 6 m higher (planes can't
hover-takeoff), and the ceiling policy gives the wing +5 m and 3× the ride
tolerance (a 45° bank bulges turns ~1.6× and it can't stop).
`bash scripts/start_airsim_stack.sh --wing` runs **both aircraft in the same
world**: a second brain (`config/wing-brain.json`, ports 8789/8790, memory
`state/wing-memory.json`) flies Wing1 while the quad brain flies Fly1 — each
with its own connectome, R-STDP memory and car curriculum. The wing flew the
project's first car touch within minutes of its first training session.

**Collision policy (dopa-tail-aware stun protocol):** every collision sends
one deep punishment pulse (default **−5.0** — measured live, a −5 pulse
reaches the brain's `dan_drive` clamp and produces a **77% deeper dopamine
teaching window** than the old −2.5; pulse *trains* were tested and are
worse, since the drive decays with τ=300 ms and just sustains). Then the
bridge **holds the crash scene** — the quad freezes in place, the wing
levels and glides straight ahead (a plane can't hover), with no proximity
or shaping pulses — until the brain's dopamine error recovers above
`DOPA_RECOVER` (−0.03; hard cap `--stun-max` 6.5 s, minimum hold
`--stun-hold` 4.5 s). This matters because the measured negative-dopa tail
lasts **4.5–6 s**: respawning immediately dumps the tail onto the *next*
episode's good flying while the crash-context synapses get less LTD than
they should. Holding the scene keeps the negative window overlapped with
the synapses that caused the crash — clean credit assignment. Car touches
(`Car_*` in AirSimNH) send **+2.5** — plus a one-time **+2.5 jackpot** for
the first touch of each distinct target car (per-target, once per run;
`jackpot on/off` via the command interface) — with a short 1.5 s hold so
the positive tail doesn't spuriously reinforce the next episode's opening
moves. Riding the ceiling (`--max-alt`) for more than `--ceil-ride`
seconds (2 s) or leaving the per-episode spawn leash (`--border-radius`,
450 m) punishes the same way. Collision punish defaults to **−2.5** (was
−5: a crash stream that outgunned every reward made the dopamine
environment net-negative and put the throttle pool on a depression
treadmill — the "keeps falling" era). `--punish-mag` / `--reward-mag`
tune the magnitudes; the altitude hill (`--alt-gain`, default 0.35) is
the throttle pool's main positive teacher. Each event names the object
hit (or `ceiling` / `map border`) in the bridge log, and the HUD shows
`STUN[...]` with remaining hold time and live dopa during the hold.

**Approach shaping (progress-only):** there is deliberately NO reward for
being near a car — an earlier continuous closeness gradient (far/near
scales) paid for hovering over the target and let altitude bobbing count
as value, and it was removed. What remains is pure progress: each 0.5 s
tick that **closes** horizontal distance to the current target car by at
least `--closing-min` meters pulses `--closing-gain` per meter (capped at
0.5). Hovering and retreating send nothing — the brain's relative reward
shaping (τ = 12 s) turns the pulses into "my approach produced this."
Car-touch events stay the jackpot; `near <d> m` / `min <d> m` in the HUD
are telemetry only, no reward attached. The 70 parked-car poses are
cached once at startup; `--prox-gain` and the gradient knobs are gone.

**Car-crash training recipe** — to train the brain to crash into cars:

```bash
python scripts/reset_brain_memory.py --full   # fresh start
bash scripts/start_airsim_stack.sh            # brain + AirSimNH + car curriculum
```

`--cars` (default **on** since the reward-geometry fix; `--no-cars`
disables) is the training curriculum: every respawn teleports the drone to a
fresh 14–22 m start next to the current target car, facing it, and near the
target the shaping switches to pure progress — each 0.5 s tick that *closes*
distance pulses a small reward (`--closing-gain` 0.1/m, capped 0.5, gated
by `--closing-min` 0.10 m/tick and measured in **horizontal** distance only
— 3D distance let altitude bobbing register as false progress), while
hovering or retreating sends nothing. This is also what teaches *forward*
flight: closing pulses are the only signal that aligns the pitch pool's
arbitrary sign with actual approach. Without it, episodes start from one
fixed spawn and the approach gradient alone never bridged the last meters
to the jackpot (359 episodes, 0 touches).

**Altitude shaping is a signed hill** (`--alt-gain` 0.2): +gain at 8 m,
grading to 0 at ground level and at 16 m, then a growing **penalty** above
(−gain by 24 m). An earlier neutral band let a throttle-pool ratchet park
the drone at 22 m+ ("only goes up"); now altitude above the band costs
reward every tick, so descending — shrinking an active penalty — is itself
rewarded. The hill applies everywhere, including inside the approach zone.
Negative shaping values flow through the brain's reward line as dopamine
drops (punish-lite), which the engine handles natively.

Watch the `[episode]` lines the bridge prints on every respawn (duration,
hits, car touches, closest approach, closing pulses, mean dopamine) —
`closing` should climb toward double digits per episode and `cars` should
tick up once approaches start connecting with the +2.5 jackpot.

**Training from video — no labels, no simulator.** The circuit's plasticity
is dopamine-gated R-STDP, so raw video alone changes nothing — but the video
itself can be the teacher: `scripts/train_from_video.py` streams any video
file (or a webcam) into the retina in the same wire format the flying bridge
uses, and pulses the reward line when something visually notable happens
(frame-difference spikes: scene cuts, sudden appearances). The synapses that
were co-active at that moment get strengthened; when a busy scene goes
static again, the engine's own baseline adaptation produces a negative error
and depresses what stopped mattering. No actions are commanded or needed —
the brain's motor output is ignored.

```bash
# dedicated video brain (ports 8791/8792, memory state/video-memory.json -
# your flight memory in state/brain-memory.json stays untouched)
./build/flybrain-server.exe --config config/video-brain.json --profile config/profiles/drone.json &
python scripts/train_from_video.py myflight.mp4 --loop

python scripts/train_from_video.py myflight.mp4 --brain-ws ws://127.0.0.1:8787/stream   # train the flying quad brain itself
python scripts/train_from_video.py --camera 0                     # live webcam
```

Verified live: 45 s of a test clip with 6 scene changes (+3.0 pulses) moved
**371 plastic synapses** past the 1% meaningful-edit bar (`memEdited 1 ->
372`) with zero control labels. Flags: `--cut-thresh` (event sensitivity,
default 0.08 mean-abs-diff), `--reward-mag` (pulse size, default 3.0 —
magnitude is the teaching lever, same as flight), `--cooldown` (min seconds
between pulses), `--speed`, `--fps`, `--duration`. The `[done]` summary
reports `memEdited` before/after so you can confirm learning happened; the
resulting visual memory then rides along in whichever brain you fly later
(same retina, same connectome).

**Resetting learned memory:**

```bash
python scripts/reset_brain_memory.py           # wipe the live brain (hash-verified)
python scripts/reset_brain_memory.py --full    # wipe + set aside the disk memory file
```

Useful flags: `--vision-hz 0` (default: stream eyes as fast as the sim
renders — the capture latency IS the pacing on slow machines; the value is
only ever a cap), `--min-alt 0.0`
/ `--max-alt 60` (safety band), `--control lane|rate` (altitude scheme),
`--stick-gain 3.0` / `--stick-tau 0.15` (FPV stick feel),
`--deadband 0.05` / `--no-trim` (auto-trim around the brain's resting
channel values), `--yaw-tau 2.0` (heading-still smoothing),
`--ceil-ride 2` / `--border-radius 450` (bounds policy),
`--prox-max 0.5` / `--near-max 1.0` / `--near-radius 8` (shaping scales),
`--no-assist`, `--eyes fly|stereo|center`, `--no-gfx`, `--viewer-port 0`,
`--reward alt`.

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
  `dnSteer`, or a motor pool `pool0` / differential `pool1-pool2`)
  drives which channel, with what gain,
- set `sensors.eyes.count` (2 = stereo pair, 1 = forward camera) and which
  scalar state the body can sense.

The readout maps *neural state → named channels* generically — no scripted
behavior, no reflex ladders. Retinal optic-flow signals (`flowPitch`,
`flowRoll`, `flowYaw`) are **inputs to the connectome only**; they are never
wired into the readout map, so no sensory→motor reflex bypasses learning.
Steering away from looming obstacles is a
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

## Human command interface (simple commands)The fly-eye viewer server doubles as a command endpoint — curriculum and
training signals only, never actuator values (the connectome keeps 100% of
the flying):

```bash
curl -X POST http://localhost:8795/cmd -d '{"cmd":"car 3"}'   # target car #3
curl -X POST http://localhost:8795/cmd -d '{"cmd":"new car"}' # random new target
curl -X POST http://localhost:8795/cmd -d '{"cmd":"reset"}'   # respawn now
curl -X POST http://localhost:8795/cmd -d '{"cmd":"hover"}'   # pause brain 5 s
curl -X POST http://localhost:8795/cmd -d '{"cmd":"go"}'      # resume
curl -X POST http://localhost:8795/cmd -d '{"cmd":"jackpot off"}'
curl http://localhost:8795/status                              # live JSON state
```

`GET /status` returns altitude, distance to the current target car, target
index, car touches, collisions, and live dopamine — enough for a dashboard
or a future natural-language front-end that compiles sentences into these
same primitives.

### Readout normalization (engine)

Map entries accept `"norm": {"span": s, "center": c}` — a range
calibration that rescales the signal so `s` maps to the full [-1,1] stick,
after subtracting `c`. **`center` is a number you calibrate** (or `true`
meaning 0.5): it must be the signal's *measured resting value* — read
`poolsRaw` from `/telemetry` and use the channel's quiescent integral.
A wrong center rails the channel against the client's clamp, and a railed
channel is a learning dead end: no event gradient, no dopamine, no thrust.
(The "thrust is always 0" bug was exactly this — pool weights ratcheted to
the floor under a center guessed at 0.5, the raw channel went to −1, and
web clients clamp throttle to ≥ 0, so the body sat at zero thrust where no
reward can be earned. Pools now also carry their own much lower weight
floor `w_floor` (0.1) instead of the synapse floor `w_min` (0.6), so a
fully-depressed pool goes near-silent — neutral stick, recoverable — rather
than pinned at a rail.) It is per-entry, in `config/profiles/*.json`, and
changes no timing or sign — the circuit keeps ownership of every
deflection. The drone profile normalizes **only the throttle channel**
(`pool0`, span 0.45, center 0.3 ≈ its measured rest): rectified pools are
unipolar, so without a norm their output is a large constant — the gain and
offset then map the *deflections around rest* onto the stick with the
calibrated hover at the configured default. Attitude channels
(`poolA-poolB` differentials) deliberately carry **no** norm — they rest at
0 already, the per-episode trim centers them, and a guessed span saturates
the tanh into pinned full-rate sticks (the "spins to face one direction"
regression; measured differentials run 0.25–0.4, far above the 0.2 span
that caused it). If you ever re-add it, measure the live differential span
first.

## Keys (demo clients)

Both examples: `WASD` nudge · `C` manual override · `R` respawn ·
`L` toggle learning · `M` wipe memory · `U` reward +1 · `J` punish −1.

## Data credit

MaleCNS v1.0 — HHMI Janelia / Google Research, Cell 2026, CC-BY 4.0.
