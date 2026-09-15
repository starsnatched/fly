/**
 * Spiking leaky integrate-and-fire engine over the MaleCNS per-neuron circuit.
 *
 * Every one of the 20,461 neurons has its own membrane voltage and spikes:
 *   Vm[k+1] = Vm[k]*decayM + (1-decayM) * (Vrest + I_syn + I_drive + noise)
 *   spike when Vm >= Vth(neuron), then Vm = Vreset for tRef ms
 *   I_syn: spikes from the previous tick deliver exponential synapses
 *   through the REAL connectome edges (weights = normalized synapse counts).
 *
 * Weight normalization: per-target K-max (K = 40th-largest incoming synapse
 * count). Robust to single giant inputs, preserves relative strength --
 * analogous to K-winner normalization in fly circuits.
 *
 * Sensory drive: each population receives a pooled current computed from
 * the vision system (drive[pop] 0..1), roughly how fly sensory neurons are
 * driven by graded photoreceptor input. T4/T5 additionally receive
 * direction-tuned current from EMD flow: a=up, b=left, c=down, d=right.
 *
 * Execution: fixed 2 ms ticks; spikes found on tick t deliver on t+1.
 */
import type { EyeStats } from "./shared";
import type { LifPayload } from "./types";

export interface LifParams {
  dtMs: number;      // integration timestep (ms of biology)
  tauM: number;      // membrane time constant (ms)
  tauS: number;      // synaptic decay (ms)
  vRest: number;     // mV
  vReset: number;    // mV
  vTh: number;       // base threshold (mV)
  tRef: number;      // refractory period (ms)
  gScale: number;    // peak PSP (mV) for a unit normalized weight
  inputGain: number; // sensory current scale (mV per unit drive)
  noiseAmp: number;  // uniform current noise half-width (mV)
  inhGain: number;   // mV of global inhibition per 100 Hz network rate
}

export const DEFAULT_PARAMS: LifParams = {
  dtMs: 2,
  tauM: 20,
  tauS: 10,
  vRest: -70,
  vReset: -62,
  vTh: -55,
  tRef: 2.5,
  gScale: 0.05,
  inputGain: 12,
  noiseAmp: 6,
  inhGain: 14,
};

export interface LifDrive {
  /** per-population sensory current, 0..~1.5 */
  drive: Float64Array;
  stats: EyeStats;
}

export class LifNetwork {
  readonly N: number;
  readonly params: LifParams;
  readonly populations: string[];

  private popIndexOf: Map<string, number>;
  private meta: LifPayload["meta"];

  // neuron state
  private vm: Float32Array;
  private ge: Float32Array;
  private gi: Float32Array;
  private refUntil: Float64Array;
  private spikedLast: Uint8Array;

  // circuit
  private popOf: Uint8Array;
  private ntSign: Int8Array;
  private dirOf: Int8Array;
  private hexOf: Int16Array;
  private edgeSrc: Uint16Array;
  private edgeDst: Uint16Array;
  private edgeW: Float32Array;
  private edgeSign: Int8Array;
  private outStart: Uint32Array;
  private outList: Uint32Array;

  // spike delivery double buffer
  private spikeBuf: Uint16Array[] = [];
  private spikeLen = [0, 0];

  private simTimeMs = 0;
  private lastTickFrac = 0;
  private popRate: Float64Array;
  private lastTickSpikes = 0;
  /** which sensory drive each population expects (index into stats) */
  private rngState = 0x9e3779b9;

  constructor(payload: LifPayload, params: Partial<LifParams> = {}) {
    this.params = { ...DEFAULT_PARAMS, ...params };
    this.meta = payload.meta;
    this.populations = payload.populations;
    this.popIndexOf = new Map(this.populations.map((p, i) => [p, i]));
    const n = payload.neurons.id.length;
    this.N = n;

    this.vm = new Float32Array(n).fill(this.params.vRest);
    this.ge = new Float32Array(n);
    this.gi = new Float32Array(n);
    this.refUntil = new Float64Array(n);
    this.spikedLast = new Uint8Array(n);

    this.popOf = new Uint8Array(payload.neurons.pop);
    this.ntSign = new Int8Array(payload.neurons.nt);
    this.dirOf = new Int8Array(payload.neurons.dir);
    this.hexOf = new Int16Array(n * 2);
    payload.neurons.hex.forEach((h, i) => {
      this.hexOf[i * 2] = h[0];
      this.hexOf[i * 2 + 1] = h[1];
    });
    this.popRate = new Float64Array(this.populations.length);

    // ---- edges ----
    const ne = payload.edges.length / 3;
    this.edgeSrc = new Uint16Array(ne);
    this.edgeDst = new Uint16Array(ne);
    this.edgeW = new Float32Array(ne);
    this.edgeSign = new Int8Array(ne);
    // gather per-target weights for K-max normalization
    const byTarget = new Map<number, number[]>();
    for (let e = 0; e < ne; e++) {
      const s = payload.edges[e * 3];
      const d = payload.edges[e * 3 + 1];
      const w = payload.edges[e * 3 + 2];
      this.edgeSrc[e] = s;
      this.edgeDst[e] = d;
      this.edgeSign[e] = this.ntSign[s] >= 0 ? 1 : -1;
      let arr = byTarget.get(d);
      if (!arr) { arr = []; byTarget.set(d, arr); }
      arr.push(w);
    }
    const K = 40;
    const refOf = new Map<number, number>();
    for (const [d, arr] of byTarget) {
      if (arr.length <= K) {
        refOf.set(d, Math.max(...arr));
      } else {
        // K-th largest without full sort: quickselect-ish via partial sort
        const sorted = Array.from(arr).sort((a, b) => b - a);
        refOf.set(d, sorted[K - 1]);
      }
    }
    for (let e = 0; e < ne; e++) {
      const d = this.edgeDst[e];
      const ref = refOf.get(d) || 1;
      this.edgeW[e] = payload.edges[e * 3 + 2] / Math.max(1, ref);
    }

    // group edges by source for fast fan-out
    const bySrc = new Array<number[]>(n);
    for (let i = 0; i < n; i++) bySrc[i] = [];
    for (let e = 0; e < ne; e++) bySrc[this.edgeSrc[e]].push(e);
    this.outStart = new Uint32Array(n + 1);
    const flat: number[] = [];
    for (let i = 0; i < n; i++) {
      this.outStart[i] = flat.length;
      for (const e of bySrc[i]) flat.push(e);
    }
    this.outStart[n] = flat.length;
    this.outList = new Uint32Array(flat);

    // spike double buffers (worst case: every neuron spikes in one tick)
    this.spikeBuf = [new Uint16Array(n), new Uint16Array(n)];
  }

  get metaInfo(): LifPayload["meta"] { return this.meta; }
  get timeMs(): number { return this.simTimeMs; }
  get lastSpikeCount(): number { return this.lastTickSpikes; }

  /** Smoothed firing rate of a population in Hz. */
  rateOf(pop: string): number {
    const i = this.popIndexOf.get(pop);
    return i === undefined ? 0 : this.popRate[i];
  }

  /** Per-population smoothed rates (Hz). */
  allRates(): Float64Array {
    return this.popRate;
  }

  spiked(i: number): boolean { return this.spikedLast[i] === 1; }
  voltageOf(i: number): number { return this.vm[i]; }

  /**
   * Advance the network by one tick (params.dtMs).
   * drive: per-population sensory current 0..~1.5
   */
  tick(drive: Float64Array, stats: EyeStats): number {
    const p = this.params;
    const decayM = Math.exp(-p.dtMs / p.tauM);
    const decayS = Math.exp(-p.dtMs / p.tauS);
    const tNow = this.simTimeMs;
    let spikeCount = 0;

    // 1. decay synapses
    for (let i = 0; i < this.N; i++) {
      this.ge[i] *= decayS;
      this.gi[i] *= decayS;
    }

    // 2. deliver spikes from previous tick
    const deliverIdx = (this.simTimeMs / p.dtMs) % 2 === 0 ? 1 : 0;
    const buf = this.spikeBuf[deliverIdx];
    const len = this.spikeLen[deliverIdx];
    for (let k = 0; k < len; k++) {
      const src = buf[k];
      const lo = this.outStart[src];
      const hi = this.outStart[src + 1];
      for (let oi = lo; oi < hi; oi++) {
        const e = this.outList[oi];
        const w = this.edgeW[e] * p.gScale;
        if (this.edgeSign[e] > 0) this.ge[this.edgeDst[e]] += w;
        else this.gi[this.edgeDst[e]] += w;
      }
    }
    this.spikeLen[deliverIdx] = 0;

    // 3. integrate + spike
    const nextBuf = this.spikeBuf[1 - deliverIdx];
    let nextLen = this.spikeLen[1 - deliverIdx];
    const piT4 = this.popIndexOf.get("T4") ?? -1;
    const piT5 = this.popIndexOf.get("T5") ?? -1;
    const driveGain = p.inputGain;
    // global feedback inhibition (lumped GABAergic pool), driven by the
    // PREVIOUS tick's spike count (2 ms synaptic delay) and saturating --
    // this is what keeps recurrent excitation from avalanching
    const inhib = p.inhGain * Math.tanh(this.lastTickFrac / 0.05);

    for (let i = 0; i < this.N; i++) {
      const pop = this.popOf[i];
      // baseline 8.6 mV leaves headroom to threshold (15 mV) so noise
      // fires neurons at ~1-4 Hz spontaneous; drive adds the rest
      let iExt = 8.6 + drive[pop] * driveGain - inhib;

      // direction-tuned EMD drive into T4/T5 (a=up b=left c=down d=right)
      if (pop === piT4 || pop === piT5) {
        const dir = this.dirOf[i];
        if (dir >= 0) {
          const tune = pop === piT4
            ? DIR_TUNE_T4[dir](stats)
            : DIR_TUNE_T5[dir](stats);
          iExt += tune * driveGain * 0.5;
        }
      }

      // approx-Gaussian noise: sum of two uniforms (triangular), sigma~amp*0.41
      const noise = ((this.rng() + this.rng()) - 1) * p.noiseAmp;
      const iSyn = this.ge[i] - this.gi[i] + iExt + noise;
      const target = p.vRest + iSyn;
      const vmNext = this.vm[i] * decayM + (1 - decayM) * target;

      if (vmNext >= p.vTh && tNow >= this.refUntil[i]) {
        this.spikedLast[i] = 1;
        this.vm[i] = p.vReset;
        this.refUntil[i] = tNow + p.tRef;
        if (nextLen < nextBuf.length) nextBuf[nextLen++] = i;
        spikeCount++;
      } else {
        this.spikedLast[i] = 0;
        this.vm[i] = vmNext;
      }
    }
    this.spikeLen[1 - deliverIdx] = nextLen;
    this.lastTickSpikes = spikeCount;
    this.lastTickFrac = spikeCount / this.N;

    // 4. population rate EMA (Hz), computed from per-pop spike counts
    const alpha = 1 - Math.exp(-p.dtMs / 400); // ~400 ms effective window
    for (let pi = 0; pi < this.populations.length; pi++) {
      const idxs = this.popNeuronIdx[pi];
      if (!idxs.length) continue;
      let spikes = 0;
      for (let k = 0; k < idxs.length; k++) {
        if (this.spikedLast[idxs[k]]) spikes++;
      }
      const hz = (spikes / idxs.length) / (p.dtMs / 1000);
      this.popRate[pi] += (hz - this.popRate[pi]) * alpha;
    }

    this.simTimeMs += p.dtMs;
    return spikeCount;
  }

  private popNeuronIdx: number[][] = [];

  /** Precompute neuron index lists per population (call once after load). */
  finalize(): void {
    this.popNeuronIdx = this.populations.map(() => []);
    for (let i = 0; i < this.N; i++) {
      this.popNeuronIdx[this.popOf[i]].push(i);
    }
  }

  /**
   * Descending-pool activations from LIF rates. Rates are smoothed per
   * population; pool functions mirror the rate-based brain's readouts
   * (avert = LPLC-driven escape, steer = T4/T5 imbalance, lift = TmY).
   */
  poolActivations(): { avert: number; steer: number; lift: number } {
    // per-neuron rates in flight run ~0-8 Hz (sensory-limited regime);
    // normalize on that scale
    const n = (hz: number) => Math.min(1, hz / 8);
    const lplc = this.rateOf("LPLC");
    const lc = this.rateOf("LC");
    const t4 = this.rateOf("T4");
    const t5 = this.rateOf("T5");
    const tmy = this.rateOf("TmY");
    return {
      avert: Math.min(1, 1.6 * n(lplc) + 0.6 * n(lc)),
      steer: 1.2 * (n(t4) - n(t5)),
      lift: 0.45 + 1.0 * (n(tmy) - 0.3),
    };
  }

  /** Evenly-strided sample of neuron indices in a population (up to max). */
  indicesOfPop(popName: string, max: number): number[] {
    const pi = this.popIndexOf.get(popName);
    if (pi === undefined) return [];
    const idxs = this.popNeuronIdx[pi] || [];
    const out: number[] = [];
    const stride = Math.max(1, Math.floor(idxs.length / Math.max(1, max)));
    for (let k = 0; k < idxs.length && out.length < max; k += stride) {
      out.push(idxs[k]);
    }
    return out;
  }

  /** Number of edges (fan-out entries). */
  get edgeCount(): number {
    return this.outList.length;
  }

  /** Fraction of T4/T5 with real hex retinotopy (quality metric). */
  retinotopicFraction(): number {
    let ok = 0, tot = 0;
    const pT4 = this.popIndexOf.get("T4"), pT5 = this.popIndexOf.get("T5");
    for (let i = 0; i < this.N; i++) {
      if (this.popOf[i] === pT4 || this.popOf[i] === pT5) {
        tot++;
        if (this.hexOf[i * 2] >= 0) ok++;
      }
    }
    return tot ? ok / tot : 0;
  }

  /** Deterministic rng for Poisson background (no external deps). */
  rng(): number {
    this.rngState ^= this.rngState << 13;
    this.rngState ^= this.rngState >>> 17;
    this.rngState ^= this.rngState << 5;
    return ((this.rngState >>> 0) % 100000) / 100000;
  }
}

/** Direction-tuning functions for T4 (ON) and T5 (OFF) channels. */
const DIR_TUNE_T4: ((s: EyeStats) => number)[] = [
  (s) => -s.agg.flowVMean * 1.5,                        // a: up (neg v-flow)
  (s) => -s.agg.flowHLeft * 1.5,                        // b: left
  (s) => s.agg.flowVMean * 1.5,                         // c: down
  (s) => s.agg.flowHLeft * 1.5,                         // d: right
];
const DIR_TUNE_T5: ((s: EyeStats) => number)[] = [
  (s) => s.agg.flowVMean * 1.0,                         // a: up
  (s) => s.agg.flowHLeft * 1.0,                         // b: left
  (s) => 0.5 * (1 - Math.min(1, s.agg.darkFraction)) + s.agg.flowVMean * 0.5, // c: looming
  (s) => 0.5 * Math.min(1, s.agg.darkFraction) + s.agg.flowHLeft * 0.5,       // d
];
