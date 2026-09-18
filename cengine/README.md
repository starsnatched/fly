# flybrain-c — the connectome brain as a native service

The entire MaleCNS v1.0 connectome (**165,122 neurons / 25.5M synapses**)
spiking in plain C, served over a small streaming API. Any embodiment sends
**per-eye RGB frames** and receives **actuator channels**; all the
neuroscience happens here.

## Layout

| file | role |
|---|---|
| `flybrain.c` | FLYBRAIN1 connectome loader (4-byte-aligned sections, LE) |
| `circuit.c`  | LIF engine: per-group biophysics, per-target in-degree normalization, CSR delivery, axonal delays, Tsodyks-Markram depression, retinotopic photoreceptors, **frame-locked Hassenstein–Reichardt T4/T5 EMDs** on real preferred-direction subtypes, DAN neuromodulation, dopamine-gated R-STDP + Turrigiano scaling |
| `runtime.c`  | tick loop (adaptive bio-budget, chunked so API threads never starve behind a full-connectome frame), lock-free frame coalescing, neural readout (declared decode map only — no scripted behavior), live embodiment-profile switching, memory persistence |
| `api.c`      | WebSocket `/stream` (binary eye frames in, action frames out at 60 Hz) + REST `/telemetry /actions /memory /health` |
| `config.c`   | JSON config + embodiment profiles (`config/flybrain.json`, `config/profiles/*.json`) |
| `engine_test.c` | in-process self-test: directional selectivity + symmetry on the real connectome |

## Wire protocol

Client → server (WS `/stream`):
- `0x01` eye frame: `u8 type, u8 eye, u16 w, u16 h, u8 ch=3, RGB bytes`
  (any resolution up to 4096²; the brain resamples through the retina's real
  hex coordinates — eye ids 0/1 map to the config's `sensors.eyes.leftId/rightId`;
  single-eye embodiments just send eye 0)
- `0x02` state: `u8 type, u8 flags(bit0=collision), f32 altitude, speed, vy, clearance`
- JSON: `{"type":"hello"|"telemetry"|"control"|"reward", ...}`
  - `hello`: optional `{"profile":"rover"}` (or a path to a profile JSON)
    switches the EMBODIMENT live — actuator channels, readout map, sensors —
    so one running brain can serve a drone and a rover at different times.
    The circuit and its learned memory are untouched.
  - `control`: `{"learning":bool}`, `{"wipe":true}` (reset memory to defaults),
    or `{"memory":{...}}` (import learned weights)
  - `reward`: `{"value":float}` — external reward bias (see "Self-regulated dopamine")

REST:
- `GET /health | /telemetry | /actions | /memory | /brain` (`/brain` streams
  the connectome binary the engine booted from, ~295 MB,
  `Content-Disposition` filename set for `curl -OJ`)
- `POST /control` with a JSON body: `{"wipe":true}` (memory → factory defaults,
  also deletes the on-disk memory), `{"learning":false|true}`,
  `{"reward":-1.0..1.0}` (external reward bias — DAN excitability, not a
  direct teaching signal)

## Self-regulated dopamine

The teaching signal is computed **inside the circuit**, not injected:

- DANs are spontaneous pacemakers (`engine.danTonicMv`, ~9 Hz tonic).
- `dopa = danFastEma − danSlowBaseline` (`engine.dopaGain`,
  `engine.danBaseTauS`): the DAN population's own deviation from its
  adapting expectation — a biological prediction error.
- External signals (open-sky/collision rewards from an embodiment, or
  `reward` messages) only bias DAN **excitability**. Whether learning
  happens depends on whether the bias actually moves DAN firing, which the
  circuit measures itself.
- A warmup phase gates plasticity until the baseline has converged, so boot
  transients and regime shifts never register as reward.
- `POST /control {"wipe":true}` re-warms and zeroes memory; edited
  synapses remain fully editable forever (soft bounds, no freezing).

Eye configuration (`sensors.eyes.count`): `2` = stereo pair — eye 0 is the
body's LEFT camera (mounted +35°), eye 1 the RIGHT (−35°), and the
connectome's left/right lamina columns each view their own camera, so
binocular flow differences steer directly; `1` = one forward-facing camera
whose left/right HALVES feed the left/right columns — turning then reads
optic-flow differences across the field, which is how many insects steer.

Server → client:
- `0x0A` action frame: `u8 type, u16 nameLen, names JSON, f32 per channel` at ~60 Hz
- JSON: `hello` (circuit size), telemetry snapshots

## Build & run

Any machine with [zig](https://ziglang.org) (or `pip install ziglang` and use
`python -m ziglang cc`):

```bash
cd cengine && make            # builds ../build/flybrain-server
make engine-test              # EMD steering self-test on the real connectome
./../build/flybrain-server --config ../config/flybrain.json   # drone profile is the default
```

Flags: `--config <path>` (default `config/flybrain.json`), `--profile <name|path>`
(a bare name resolves to `config/profiles/<name>.json`; default `drone`), `--port <ws>` (REST = ws+1,
overriding the config's `network` block).

Docker (no local toolchain needed):

```bash
docker compose -f cengine/docker-compose.yml up --build
# brain on ws://localhost:8787/stream + REST :8788, client on :5199
```

## Readout: neural state → actuator channels

The readout computes NO behavior. Every signal is a population activity
INSIDE the circuit — the same thing an electrophysiologist would decode from
descending neurons and the optic lobe. Which population feeds which actuator
channel is a per-embodiment DECLARATION in config (`readout.map`): each entry
is `{ "channel", "signal", "gain", "offset" }` with
`channel = offset + gain · signal`, summed over entries, then clamped to the
channel's range and slewed. Unknown signals or channels are skipped, never
invented. A new body = a new JSON profile: name its channels, declare the map.

| signal | source |
|---|---|
| `dnDrive` | descending-population firing rate (normalized by `dnHzScale`) |
| `motorDrive` | motor-population firing rate (normalized) |
| `dnSteer` | left-vs-right descending rate asymmetry (−1..1) |
| `flowYaw` | T4/T5 horizontal optic flow, right-vs-left difference |
| `flowRoll` | T4/T5 whole-field horizontal flow (optic-lobe consensus) |
| `flowPitch` | T4/T5 vertical flow (optic-lobe consensus) |
| `touch` | mechanosensory burst envelope (0..1) |
| `pool0`..`pool7` | **direct motor pools** (below) |
| `poolA-poolB` | signed differential between two pools (steering) |

### Direct motor pools (`readout.pools`)

The population signals above decode whole-group averages. A body may instead
wire **individual motor neurons** to its actuators — pool `i` is a selected
subset of one connectome group whose spikes integrate into a leaky
accumulator (motor-unit temporal summation, `readout.integrateMs`, default
90 ms). The pool integrals enter the same map as `pool0`.. signals:

```json
"readout": {
  "integrateMs": 90, "learn": true,
  "pools": [
    { "group": "motor", "every": 7 },
    { "group": "motor", "split": "lr", "which": 0 },
    { "group": "motor", "split": "lr", "which": 1 }
  ],
  "map": [
    { "channel": "throttle", "signal": "pool0", "gain": 5.0, "offset": -1.6 },
    { "channel": "steer", "signal": "pool1-pool2", "gain": 40.0, "shape": "tanh" }
  ]
}
```

`every:k` picks one neuron of every k (ascending id); `split:"lr"`
partitions by connectome side (falling back to an id-order half split where
side is degenerate, as in the VNC). Pool values are mean-per-neuron spike
integrals, so pools of different sizes are comparable. With `"learn":true`
each member's weight is PLASTIC — moved per tick by the same self-regulated
dopamine error that gates R-STDP (`w += lr·dopa·(spike − tonic)`), with the
same soft bounds and the same rule that a constant situation teaches
nothing. The learned body map persists in the memory file (`"pools":[…]`)
alongside the synapse multipliers, survives restarts, and is carried across
live profile switches. `config/profiles/rover.json` and
`drone.json` ARE the pool decode (the `-pools` naming was retired) — both
verified closed-loop
(`scripts/rover_bench.py`, `scripts/drone_bench.py`): the learned pool map
flies the twin drone at 3.9 m/s with 1.71 hits/100 m vs 3.50 after wipe,
and steers the rover via its motor L/R differential.

Per embodiment (`config/profiles/`): the **drone** declares
throttle/pitch/roll/yaw; the **rover** declares throttle/steer; both drive
their channels through direct motor pools (population signals like `dnSteer`
or `flowRoll` remain legal in the map for mixing). Gains in
the map shape how strongly a signal drives a channel — they are anatomy
annotations, not behavioral controllers: all behavior originates in the
circuit (EMD flow, DN steering, R-STDP memory, pool plasticity) and its own
dopamine.

Browser embodiments live in `examples/`: `drone-web/` drives the drone
profile, `rover-web/` drives the rover profile (single forward eye, so the
retina's two hemispheres each view half of the same image — run the server
with `--profile rover`).

## Verified

- `make engine-test`: L-half motion → `flowH < 0`, R-half → `flowH > 0`,
  vertical bias signed correctly, static scene → 0.
- `scripts/e2e_client.py` (protocol test, implementation-agnostic):
  handshake, binary frames, action broadcast, telemetry, control, reward —
  passing against this C server.
- `scripts/learn_probe.py`: conditioning proof — collision epochs drive
  dopa negative → internal LTD; open-sky epochs drive dopa positive →
  ~20k graded re-edits **after a full memory wipe** (soft bounds keep every
  synapse editable); dopa fluctuates around 0 in steady state (no runaway).
- Memory stays bounded across sessions (prediction-error dopamine + capped
  eligibility); telemetry `dopa`/`danBase` expose the internal signal.
