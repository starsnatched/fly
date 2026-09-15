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
                  └─ 20,461-neuron circuit, 12 populations
                     (lamina → Tm/TmY → T4/T5 → LC/LPLC/LT → DNs)
                     wiring = REAL synapse counts aggregated from 5.5M synapses
                        └─ descending-neuron pools → throttle/pitch/roll/yaw
                             └─ quadrotor physics → dodge or crash
```

- **Vision** — two eye renders per frame, downsampled, run through
  Hassenstein–Reichardt EMDs (the canonical fly motion detector).
- **Looming detection** — the fly's escape trigger: angular size of the nearest
  dark silhouette *and* its growth above an adapted baseline. Saccades suppress
  detection mid-turn and reset adaptation afterwards (post-saccadic reset).
- **Brain** — two interchangeable controllers (press **B** to hot-swap):
  - **RATE** (default): leaky-integrator population network, 12 populations,
    wiring = the connectome's own aggregated synapse counts.
  - **SPIKING LIF**: every one of the 20,461 neurons has its own membrane
    voltage, threshold, refractory period, and spikes. Synapses are
    exponential-decay conductances delivered through 343,241 real connectome
    edges (weights = per-target K-max-normalized synapse counts, signs from a
    documented neurotransmitter heuristic). Sensory drive injects current per
    population; T4/T5 additionally get direction-tuned EMD current (real
    a/b/c/d directional subtypes from the data). A saturating feedback-
    inhibition pool keeps recurrent excitation in the fluctuation-driven
    regime (~1-8 Hz spontaneous, bursts on looming).
  - Looming, saccades, wander, and command synthesis are shared
    (`src/shared.ts`), so flight character is preserved across brains.
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

`scripts/extract_circuit.py` (Python + pyarrow):

1. Filters to `status == "Traced"` (165,122 neurons).
2. Seeds = T4/T5 (13,585) + `visual_projection` (9,201); targets = 1,360 DNs.
3. Keeps neurons on a **seed→DN path of length ≤ 3** via CSR BFS (79,066 survive).
4. Assigns populations, caps sprawling classes, prunes edges per target.
5. Emits `public/fly-circuit.json` (~6 MB): 20,461 neurons,
   343,241 visualization edges, 131 population-pair weights.

Reproduce:

```bash
python -m venv .venv
.venv/Scripts/pip install pyarrow numpy pandas      # (Linux/macOS: .venv/bin/pip)
.venv/Scripts/python scripts/extract_circuit.py     # downloads ~1 GB, then extracts
npm install
npm run dev
```

## Controls

| key | action |
|---|---|
| click | launch |
| `W A S D` | nudge thrust |
| `C` | toggle manual override |
| `B` | hot-swap rate ↔ spiking LIF brain |
| `R` | respawn |

## Files

- `src/vision.ts` — compound eyes + EMDs + looming readout
- `src/brain.ts` — rate-based population network (connectome-wired)
- `src/lif.ts` — spiking LIF engine (per-neuron Vm, spikes, exponential synapses)
- `src/lifbrain.ts` — LIF controller adapter (same interface as FlyBrain)
- `src/shared.ts` — shared looming / saccade / wander / command circuit
- `src/drone.ts` — quadrotor physics
- `src/world.ts` — obstacle world generation + collision queries
- `src/main.ts` — glue: loop, cameras, HUD, spike raster
- `scripts/extract_circuit.py` — connectome → population circuit JSON
- `scripts/extract_lif_circuit.py` — connectome → per-neuron LIF circuit JSON
  (adds hex retinotopy, T4/T5 directional subtypes, NT signs)

## Credits & license

- Connectome data: **MaleCNS v1.0**, FlyEM team @ HHMI Janelia, Google Research,
  and collaborators — [male-cns.janelia.org](https://male-cns.janelia.org/) —
  **CC-BY 4.0**. Citation: Berg et al., *"Sexual dimorphism in the complete
  connectome of the Drosophila male central nervous system"*, Cell (2026).
- Modeling conventions follow the connectome-constrained LIF approach of
  Shiu et al. 2024 (signs of influence are a modeling choice; magnitudes and
  topology are the data's).
- Code in this repo: MIT.
