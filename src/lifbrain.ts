import type { LifPayload, BrainInputs, BrainOutputs } from "./types";
import { LifNetwork, DEFAULT_PARAMS, type LifParams } from "./lif";
import { SharedCircuit } from "./shared";

/**
 * Spiking-LIF controller (the only brain): a leaky-integrator network over
 * 20,461 individual neurons wired by the real MaleCNS connectome.
 *
 * Pool activations for command synthesis come from real per-neuron spike
 * rates in the LIF network; looming/saccade/wander/commands are shared.
 */
export class LifBrain {
  private net: LifNetwork;
  private shared = new SharedCircuit();
  private drive: Float64Array;
  debug: Record<string, number> = {};

  constructor(payload: LifPayload, params: Partial<LifParams> = {}) {
    this.net = new LifNetwork(payload, { ...DEFAULT_PARAMS, ...params });
    this.net.finalize();
    this.drive = new Float64Array(payload.populations.length);
  }

  get populationCount(): number { return this.net.populations.length; }
  get neuronCount(): number { return this.net.N; }
  get edgeCount(): number { return this.net.edgeCount; }
  get network(): LifNetwork { return this.net; }

  get synapses(): number {
    return this.net.edgeCount; // edges (each = real synapse-count weight)
  }

  notifyBump(): void {
    this.shared.notifyBump();
  }

  step(inp: BrainInputs): BrainOutputs {
    const dt = Math.min(inp.dt, 0.05);
    const s = this.shared.update(inp.stats, dt);
    const { stats } = inp;

    // sensory drive per population (maps the shared statistics)
    const driveOf = (name: string): number => {
      switch (name) {
        case "lamina": return 0.45 + 0.3 * Math.min(1, Math.abs(stats.flowH.L) + Math.abs(stats.flowH.R));
        case "Tm": return 0.5;
        case "TmY": return 0.5;
        case "T4": return Math.min(1, 0.35 + Math.abs(stats.flowH.L));
        case "T5": return Math.min(1, 0.35 + Math.abs(stats.flowH.R));
        case "LPLC": return Math.min(1, (s.loomingL + s.loomingR) * 1.2);
        case "LC": return Math.min(1, Math.max(0, 0.5 + 0.6 * (stats.flowH.R - stats.flowH.L)));
        case "LT": return Math.min(1, Math.max(0, 0.35 + 0.5 * Math.abs(stats.flowH.R - stats.flowH.L)));
        case "lp-tangential": return Math.min(1, Math.max(0, 0.3 + 0.5 * (Math.abs(stats.flowH.L) + Math.abs(stats.flowH.R)) / 2));
        case "optic-other": return 0.5;
        case "inter": return 0.5;
        case "descending": return 0;
        default: return 0.5;
      }
    };
    for (let i = 0; i < this.net.populations.length; i++) {
      this.drive[i] = driveOf(this.net.populations[i]);
    }

    // run multiple LIF ticks per frame to reach ~1 ms budget of biology
    // (dt is real seconds; simulate up to 8 ms per frame to stay real-time-ish)
    const bioMs = Math.min(dt * 1000, 10);
    const ticks = Math.max(1, Math.round(bioMs / this.net.params.dtMs));
    let spikes = 0;
    for (let t = 0; t < ticks; t++) {
      spikes += this.net.tick(this.drive, stats);
    }

    const pools = this.net.poolActivations();
    const cmd = this.shared.commands(pools, s, inp.altitude, inp.vy);

    this.debug = {
      ...this.shared.debug,
      spikesThisFrame: spikes,
      lplcHz: this.net.rateOf("LPLC"),
      lcHz: this.net.rateOf("LC"),
      t4Hz: this.net.rateOf("T4"),
      t5Hz: this.net.rateOf("T5"),
      dnHz: this.net.rateOf("descending"),
      pitch: cmd.pitch,
    };

    return {
      throttle: cmd.throttle,
      pitch: cmd.pitch,
      roll: cmd.roll,
      yaw: cmd.yaw,
      pools: { avert: pools.avert, steer: pools.steer, lift: cmd.throttle },
    };
  }
}
