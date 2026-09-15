# FlyBrain FPV 🪰🎮

**A quadcopter in a browser, flown by a circuit of real neurons extracted from the
male fruit-fly brain connectome (MaleCNS v1.0).**

Inspired by the "fly controls an FPV drone" demos — but the wiring here is not
hand-made: it comes from the complete electron-microscopy connectome of an adult
male *Drosophila* central nervous system, published by HHMI Janelia / Google
Research in *Cell* (Berg et al., 2026), licensed CC-BY 4.0.

![pipeline](docs/pipeline.svg)

## How it works

```
WebGL scene (FPV camera, 2 eyes at ±35°)
   └─ 96×54 render per eye → 48×27 luminance grid
        └─ Hassenstein–Reichardt elementary motion detectors  (≈ T4/T5)
             └─ central dark-fraction + growth = looming       (≈ LPLC2)
                  └─ 20,461 spiking LIF neurons, 12 populations
                     (lamina → Tm/TmY → T4/T5 → LC/LPLC/LT → DNs)
                     wiring = 343,241 REAL per-neuron connectome edges
                        └─ descending-neuron pools → throttle/pitch/roll/yaw
                             └─ quadrotor physics → dodge or bump
```

- **Vision** — two eye renders per frame, downsampled, run through
  Hassenstein–Reichardt EMDs (the canonical fly motion detector).
- **Looming detection** — the fly's escape trigger: angular size of the nearest
  dark silhouette *and* its growth above an adapted baseline. Saccades suppress
  detection mid-turn and reset adaptation afterwards (post-saccadic reset).
- **Brain** — a **spiking leaky integrate-and-fire** network: every one of the
  20,461 neurons has its own membrane voltage, threshold, refractory period,
  and spikes. Synapses are exponential-decay conductances delivered through
  343,241 real connectome edges (weights = per-target K-max-normalized synapse
  counts; signs from per-neuron neurotransmitter probabilities aggregated from
  **45.7M real T-bars** — 18% of circuit neurons are GABAergic/inhibitory,
  peaking at 32% in the `inter` population; mean NT confidence 87%). Sensory
  drive injects current per population; T4/T5 additionally get direction-tuned
  EMD current (real a/b/c/d directional subtypes from the data). A saturating
  feedback-inhibition pool keeps recurrent excitation in the fluctuation-driven
  regime (~1-8 Hz spontaneous, bursts on looming).
- **Readout** — descending-neuron pools: `avert` (LPLC-driven escape), `steer`
  (lateral flow asymmetry), `lift` (PD altitude hold near 2 m). Looming
  triggers a rapid **body saccade** away from the threat — the fly's own
  collision-avoidance maneuver.
- **Physics** — simple quadrotor model: thrust along body-up, drag, tilt-based
  acceleration, ground/ceiling clamps, capsule-vs-obstacle collisions.

## The data

| file | rows | what |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 211,577 neurons | type, superclass, side, status |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 151,856,684 edges | neuron→neuron synapse counts |

`scripts/extract_lif_circuit.py` (Python + pyarrow):

1. Filters to `status == "Traced"` (165,122 neurons).
2. Seeds = T4/T5 (13,585) + `visual_projection` (9,201); targets = 1,360 DNs.
3. Keeps neurons on a **seed→DN path of length ≤ 3** via CSR BFS (79,066 survive).
4. Assigns populations, caps sprawling classes, prunes edges per target.
5. Merges per-neuron neurotransmitter profiles from `data/neuron-nt.json`
   (`scripts/aggregate_nt.py`), real T4/T5 a/b/c/d directional subtypes, and
   hex retinotopy where available.
6. Emits `public/fly-lif.json` (~5 MB): 20,461 neurons with per-neuron NT
   sign + confidence, and 343,241 per-neuron edges as flat triplets.

Reproduce:

```bash
python -m venv .venv
.venv/Scripts/pip install pyarrow numpy pandas      # (Linux/macOS: .venv/bin/pip)
.venv/Scripts/python scripts/aggregate_nt.py           # optional: real NT signs (2.7 GB download)
.venv/Scripts/python scripts/extract_lif_circuit.py    # downloads ~1 GB, then extracts
npm install
npm run dev
```

## Controls

| key | action |
|---|---|
| click | launch |
| `W A S D` | nudge thrust |
| `C` | toggle manual override |
| `R` | respawn |

## Files

- `src/vision.ts` — compound eyes + EMDs + looming readout
- `src/lif.ts` — spiking LIF engine (per-neuron Vm, spikes, exponential synapses)
- `src/lifbrain.ts` — LIF controller (sensory drive → ticks → pool readout)
- `src/shared.ts` — shared looming / saccade / wander / command circuit
- `src/drone.ts` — quadrotor physics
- `src/world.ts` — obstacle world generation + collision queries
- `src/main.ts` — glue: loop, cameras, HUD, spike raster
- `scripts/extract_lif_circuit.py` — connectome → per-neuron LIF circuit JSON
  (hex retinotopy, T4/T5 directional subtypes, real NT signs)
- `scripts/aggregate_nt.py` — stream 45.7M T-bar NT probabilities →
  per-neuron mean profiles (`data/neuron-nt.json`)

## Credits & license

- Connectome data: **MaleCNS v1.0**, FlyEM team @ HHMI Janelia, Google Research,
  and collaborators — [male-cns.janelia.org](https://male-cns.janelia.org/) —
  **CC-BY 4.0**. Citation: Berg et al., *"Sexual dimorphism in the complete
  connectome of the Drosophila male central nervous system"*, Cell (2026).
- Modeling conventions follow the connectome-constrained LIF approach of
  Shiu et al. 2024 (signs of influence are a modeling choice; magnitudes and
  topology are the data's).
- Code in this repo: MIT.
