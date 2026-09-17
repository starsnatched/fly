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
| `runtime.c`  | tick loop (adaptive bio-budget), lock-free frame coalescing, gait readout (altitude hold, obstacle avoidance, spontaneous saccades + heading wander), memory persistence |
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
  - `control`: `{"learning":bool}`, `{"wipe":true}` (reset memory to defaults),
    or `{"memory":{...}}` (import learned weights)
  - `reward`: `{"value":float}` — external reward injection (R-STDP)

REST:
- `GET /health | /telemetry | /actions | /memory`
- `POST /control` with a JSON body: `{"wipe":true}` (memory → factory defaults,
  also deletes the on-disk memory), `{"learning":false|true}`,
  `{"reward":-1.0..1.0}` (external reward shaping from any environment)

Eye configuration (`sensors.eyes.count`): `2` = stereo pair, the connectome's
left/right lamina columns each view their own camera; `1` = one forward-facing
camera whose left/right HALVES feed the left/right columns — turning then reads
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
./../build/flybrain-server --config ../config/flybrain.json --profile drone
```

Flags: `--config <path>` (default `config/flybrain.json`), `--profile <name|path>`
(a bare name resolves to `config/profiles/<name>.json`), `--port <ws>` (REST = ws+1,
overriding the config's `network` block).

Docker (no local toolchain needed):

```bash
docker compose -f cengine/docker-compose.yml up --build
# brain on ws://localhost:8787/stream + REST :8788, client on :5199
```

## Readout: neural state → actuator channels

The readout is generic: it computes a small set of neural signals and maps
them onto **whatever channels the profile names** (`throttle`, `pitch`,
`roll`, `yaw`, `steer`, …). Unknown channels hold their configured default.
All gains live in the config's `readout` block.

| signal | source |
|---|---|
| `drive` | DN + motor group firing rates (arousal) |
| `avoid` | `tanh(turnGain · (flowR − flowL))` — steer away from the eye with more optic flow |
| `v_flow` | vertical EMD bias — ground expanding → climb |
| saccades | spontaneous, rate `saccadeRate`/s (real flies interleave straight flight with rapid turns) |
| `wander_bias` | Ornstein–Uhlenbeck heading set point (`wanderTau`, `wanderAmp`) |
| altitude hold | climb set point `altGain·(targetAlt − alt) − altDamp·vy` |

Per embodiment (`config/profiles/`): the **drone** gets throttle/pitch/roll/yaw
with altitude hold and saccadic flight; the **rover** gets throttle/steer with
ground steering only. A new body = a new JSON profile.

## Verified

- `make engine-test`: L-half motion → `flowH < 0`, R-half → `flowH > 0`,
  vertical bias signed correctly, static scene → 0.
- `scripts/e2e_client.py` (protocol test, implementation-agnostic):
  handshake, binary frames, action broadcast, telemetry, control, reward —
  passing against this C server.
- `scripts/verify_api.py`: 8 s single-eye streaming probe — throttle rides
  0.27–0.34 around hover (no floor/ceiling relay), yaw spans −0.9…+0.6
  (saccades + wander + EMD), `POST /control {"wipe":true}` → `memEdited: 0`.
- Memory stays bounded across sessions: weight spread ~1.0–1.08 after many
  minutes (prediction-error reward + capped eligibility), no saturation.
