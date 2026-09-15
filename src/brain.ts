import type { CircuitPayload } from "./types";
import { SharedCircuit, type SensoryState, type EyeStats } from "./shared";

/**
 * Rate-based fly-brain controller: a leaky-integrator POPULATION network
 * whose wiring comes from the MaleCNS v1.0 connectome (population-aggregated
 * synapse counts, 131 pairs from 5.5M traced synapses).
 *
 * Looming, saccades, wander, and command synthesis live in shared.ts;
 * this class provides the population dynamics that produce the
 * descending-pool activations.
 */
export interface BrainInputs {
  stats: EyeStats;
  altitude: number;
  /** vertical velocity m/s (positive = climbing) */
  vy: number;
  /** current heading (yaw, radians; 0 = course direction -Z) */
  heading: number;
  speed: number;
  dt: number;
}

export interface BrainOutputs {
  throttle: number;
  pitch: number;
  roll: number;
  yaw: number;
  pools: Record<string, number>;
}

const TAU = 0.08; // 80 ms membrane time constant

export class FlyBrain {
  private popNames: string[];
  private pidx: Map<string, number>;
  private W: Float64Array;
  private N: number;
  private act: Float64Array;
  private meta: CircuitPayload["meta"];
  private neuronCount_: number;
  private edgeCount_: number;
  private shared = new SharedCircuit();
  /** live internals for debugging */
  debug: Record<string, number> = {};

  constructor(payload: CircuitPayload) {
    this.meta = payload.meta;
    this.popNames = payload.meta.populations;
    this.N = this.popNames.length;
    this.pidx = new Map(this.popNames.map((p, i) => [p, i]));
    this.W = new Float64Array(this.N * this.N);
    for (const [pre, post, w] of payload.popMatrix) {
      const a = this.pidx.get(pre), b = this.pidx.get(post);
      if (a !== undefined && b !== undefined) this.W[b * this.N + a] = w;
    }
    for (let b = 0; b < this.N; b++) {
      let s = 0;
      for (let a = 0; a < this.N; a++) s += Math.abs(this.W[b * this.N + a]);
      if (s > 0) for (let a = 0; a < this.N; a++) this.W[b * this.N + a] /= s;
    }
    for (let i = 0; i < this.W.length; i++) this.W[i] *= 0.65;
    this.act = new Float64Array(this.N);
    this.neuronCount_ = payload.neurons.length;
    this.edgeCount_ = payload.edges.length;
  }

  get populationCount(): number { return this.N; }
  get neuronCount(): number { return this.neuronCount_; }
  get edgeCount(): number { return this.edgeCount_; }
  get synapses(): number {
    return this.meta ? 14_000_000 : 0; // reported from extraction metadata
  }

  /** Called when the body collides with something. */
  notifyBump(): void {
    this.shared.notifyBump();
  }

  step(inp: BrainInputs): BrainOutputs {
    const dt = Math.min(inp.dt, 0.05);
    const s: SensoryState = this.shared.update(inp.stats, dt);
    const { stats } = inp;
    const flowHL = stats.flowH.L, flowHR = stats.flowH.R;

    // sensory drive into optic-lobe populations
    const drive: Record<string, number> = {
      lamina: 0.45 + 0.3 * Math.min(1, Math.abs(flowHL) + Math.abs(flowHR)),
      Tm: 0.5, TmY: 0.5,
      T4: Math.min(1, 0.35 + Math.abs(flowHL)),
      T5: Math.min(1, 0.35 + Math.abs(flowHR)),
      LPLC: clamp01((s.loomingL + s.loomingR) * 1.2),
      LC: clamp01(0.5 + 0.6 * (flowHR - flowHL)),
      LT: clamp01(0.35 + 0.5 * Math.abs(flowHR - flowHL)),
      "lp-tangential": clamp01(0.3 + 0.5 * (Math.abs(flowHL) + Math.abs(flowHR)) / 2),
      "optic-other": 0.5,
      inter: 0.5,
      descending: 0,
    };

    // leaky step with global inhibition (lumped GABAergic pool)
    let mean = 0;
    for (let i = 0; i < this.N; i++) mean += this.act[i];
    mean /= this.N;
    const inhib = 0.55 * mean;
    for (let b = 0; b < this.N; b++) {
      let sum = drive[this.popNames[b]] ?? 0;
      const row = b * this.N;
      for (let a = 0; a < this.N; a++) {
        const w = this.W[row + a];
        if (w !== 0) sum += w * this.act[a];
      }
      sum = clamp01(sum - inhib);
      this.act[b] += (dt / TAU) * (sum - this.act[b]);
      this.act[b] = clamp01(this.act[b]);
    }
    const actOf = (p: string) => this.act[this.pidx.get(p) ?? 0];

    // descending readout pools (baseline-subtracted so idle = 0)
    const avert = 1.5 * actOf("LPLC");
    const pools = { avert: clamp01(avert), steer: s.dodge, lift: 0 };

    const cmd = this.shared.commands(
      { avert: pools.avert, steer: s.steer, lift: 0.5 + 0.7 * (actOf("TmY") - 0.5) },
      s, inp.altitude, inp.vy,
    );

    this.debug = {
      ...this.shared.debug,
      lplc: actOf("LPLC"),
      lc: actOf("LC"),
      t4: actOf("T4"),
      t5: actOf("T5"),
      avertRaw: avert,
      pitch: cmd.pitch,
    };

    return {
      throttle: cmd.throttle,
      pitch: cmd.pitch,
      roll: cmd.roll,
      yaw: cmd.yaw,
      pools: { avert: pools.avert, steer: s.dodge, lift: cmd.throttle },
    };
  }
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
function clamp01(x: number): number {
  return clamp(x, 0, 1);
}
