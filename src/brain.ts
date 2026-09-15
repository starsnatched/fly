import type { CircuitPayload } from "./types";

/**
 * The fly-brain controller: a leaky-integrator network whose wiring comes
 * from the MaleCNS v1.0 connectome (population-aggregated synapse counts,
 * 131 population pairs aggregated from 5.5M traced synapses).
 *
 * Topology (real, from data):
 *   retina(drive) -> lamina -> Tm/TmY -> T4/T5 (EMD input) -> LC/LPLC/LT
 *   -> lp-tangential / inter -> descending neurons (flight readouts)
 *
 * Synapse counts give magnitudes; signs are a documented modeling choice
 * (looming/avoidance channels excite escape; wide-field channels stabilize),
 * as in Shiu et al. 2024.
 */

export interface EyeStats {
  flowH: { L: number; R: number; U: number; D: number };
  flowV: { L: number; R: number; U: number; D: number };
  /** dark-pixel fraction per eye (silhouette size proxy) */
  looming: { L: number; R: number; U: number; D: number };
  /** mean horizontal EMD flow on each half of each eye (expansion readout) */
  halfFlow: { L: { left: number; right: number }; R: { left: number; right: number } };
}

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
  /** live internals for debugging */
  debug: Record<string, number> = {};
  private synapseTotal: number;

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
    // normalize per-target incoming weight, then temper recurrence so the
    // excitatory loop cannot run away to saturation
    for (let b = 0; b < this.N; b++) {
      let s = 0;
      for (let a = 0; a < this.N; a++) s += Math.abs(this.W[b * this.N + a]);
      if (s > 0) for (let a = 0; a < this.N; a++) this.W[b * this.N + a] /= s;
    }
    for (let i = 0; i < this.W.length; i++) this.W[i] *= 0.65;
    this.act = new Float64Array(this.N);
    this.neuronCount_ = payload.neurons.length;
    this.edgeCount_ = payload.edges.length;
    this.synapseTotal = payload.popMatrix.reduce((s, m) => s + m[2], 0);
  }

  /** Called when the body collides with something: immediate escape turn. */
  notifyBump(): void {
    if (!this.saccade.active && this.saccade.cooldown <= 0) {
      this.saccade.active = true;
      this.saccade.tLeft = 0.6;
      this.saccade.dir = this.rng() < 0.5 ? -1 : 1;
      this.saccade.cooldown = 1.0;
    }
    // also re-roll the wander heading so it doesn't grind along the obstacle
    this.wanderBias = (this.rng() - 0.5) * 1.2;
  }

  get populationCount(): number { return this.N; }
  get neuronCount(): number { return this.neuronCount_; }
  get edgeCount(): number { return this.edgeCount_; }
  get synapses(): number { return this.synapseTotal; }
  get datasetMeta(): CircuitPayload["meta"] { return this.meta; }

  // smoothed looming signal (fast rise, slower decay)
  private loomSm: { L: number; R: number } = { L: 0, R: 0 };
  // adapted baseline of central dark fraction (angular size of the
  // nearest silhouette); only GROWTH above it means "collision course"
  private darkBase: { L: number; R: number } | null = null;
  // saccade state: rapid escape turn, like the fly's body saccades
  private saccade = { active: false, dir: 1, tLeft: 0, cooldown: 0 };
  private lastDodge = 1; // memory for symmetric threats
  private justEnded = false; // post-saccadic detection holdoff
  private justEndTimer = 0;
  private sinceSaccadeEnd = 99; // seconds since last saccade ended
  // spontaneous exploration: mean-reverting random turn-bias + wander saccades
  private wanderBias = 0; // OU process, rad/s of preferred yaw command
  private wanderSaccadeTimer = 2 + Math.random() * 3;
  private wanderSaccade = { active: false, dir: 1, tLeft: 0 };
  private rng = () => {
    // xorshift for reproducible-ish randomness without Math.random deps
    this.rngState ^= this.rngState << 13;
    this.rngState ^= this.rngState >>> 17;
    this.rngState ^= this.rngState << 5;
    return ((this.rngState >>> 0) % 100000) / 100000;
  };
  private rngState = 0x9e3779b9;

  step(inp: BrainInputs): BrainOutputs {
    const dt = Math.min(inp.dt, 0.05);
    const { flowH, looming, halfFlow } = inp.stats;
    const flowHL = flowH.L, flowHR = flowH.R;

    // LPLC2-style looming: the angular size of the nearest dark silhouette
    // (central dark fraction) and its GROWTH. Static scenery gives a steady
    // baseline; an object on a collision course grows super-linearly, so
    // growth above the adapted baseline triggers the escape.
    // During a saccade the whole scene sweeps across the eye, so detection
    // is held; on saccade end the baseline resets (post-saccadic reset).
    const darkL = looming.L, darkR = looming.R;
    if (!this.darkBase) this.darkBase = { L: darkL, R: darkR };
    // fast re-adaptation shortly after a saccade (baseline was force-reset
    // mid-sweep; tau=0.3s lets it settle to the new scene quickly, then
    // slow tau=2s gives stable looming detection in cruise)
    const postSacc = Math.max(0, 2.0 - this.sinceSaccadeEnd) > 0;
    const baseTau = postSacc ? 0.3 : 2.0;
    if (this.saccade.active) {
      this.darkBase.L = darkL;
      this.darkBase.R = darkR;
    } else {
      const dA = Math.min(1, dt / baseTau);
      this.darkBase.L += (darkL - this.darkBase.L) * dA;
      this.darkBase.R += (darkR - this.darkBase.R) * dA;
      const rawL = clamp01((darkL - this.darkBase.L - 0.02) * 7);
      const rawR = clamp01((darkR - this.darkBase.R - 0.02) * 7);
      const smooth = (cur: number, raw: number) =>
        cur + (raw - cur) * (raw > cur ? 0.45 : 0.06);
      this.loomSm.L = smooth(this.loomSm.L, rawL);
      this.loomSm.R = smooth(this.loomSm.R, rawR);
    }
    const loomingL = this.loomSm.L, loomingR = this.loomSm.R;
    void halfFlow;

    // sensory drive into optic-lobe populations
    const drive: Record<string, number> = {
      lamina: 0.45 + 0.3 * Math.min(1, Math.abs(flowHL) + Math.abs(flowHR)),
      Tm: 0.5, TmY: 0.5,
      T4: Math.min(1, 0.35 + Math.abs(flowHL)),
      T5: Math.min(1, 0.35 + Math.abs(flowHR)),
      LPLC: clamp01((loomingL + loomingR) * 1.2),
      LC: clamp01(0.5 + 0.6 * (flowHR - flowHL)),
      LT: clamp01(0.35 + 0.5 * Math.abs(flowHR - flowHL)),
      "lp-tangential": clamp01(0.3 + 0.5 * (Math.abs(flowHL) + Math.abs(flowHR)) / 2),
      "optic-other": 0.5,
      inter: 0.5,
      descending: 0,
    };

    // leaky step with global inhibition (lumped GABAergic pool -- the fly's
    // wide-field inhibitory neurons): act += dt/tau * (drive + W*act - k*mean - act)
    let mean = 0;
    for (let i = 0; i < this.N; i++) mean += this.act[i];
    mean /= this.N;
    const inhib = 0.55 * mean;
    for (let b = 0; b < this.N; b++) {
      let s = drive[this.popNames[b]] ?? 0;
      const row = b * this.N;
      for (let a = 0; a < this.N; a++) {
        const w = this.W[row + a];
        if (w !== 0) s += w * this.act[a];
      }
      s = clamp01(s - inhib);
      this.act[b] += (dt / TAU) * (s - this.act[b]);
      this.act[b] = clamp01(this.act[b]);
    }
    const actOf = (p: string) => this.act[this.pidx.get(p) ?? 0];

    // descending readout pools (baseline-subtracted so idle = 0)
    const avert = 1.5 * actOf("LPLC");
    // dodge away from the more threatened side (positive = threat on right)
    const dodge = loomingR - loomingL;
    if (Math.abs(dodge) > 0.05) this.lastDodge = Math.sign(dodge);
    const steer = 1.4 * (flowHR - flowHL) * (0.4 + actOf("LT"));
    const loomMean = (loomingL + loomingR) / 2;

    // --- saccade state machine: strong expansion triggers a rapid turn away ---
    if (this.saccade.cooldown > 0) this.saccade.cooldown -= dt;
    if (!this.saccade.active && loomMean > 0.25 && this.saccade.cooldown <= 0 && !this.justEnded) {
      this.saccade.active = true;
      this.saccade.tLeft = 0.85;
      // symmetric head-on threat: use lateral-flow bias, else memory
      this.saccade.dir = Math.abs(dodge) > 0.08
        ? Math.sign(dodge)
        : (Math.abs(steer) > 0.05 ? Math.sign(steer) : this.lastDodge);
      this.saccade.cooldown = 1.5;
    }
    let saccYaw = 0, saccRoll = 0, saccPitch = 0;
    if (this.saccade.active) {
      this.saccade.tLeft -= dt;
      const envelope = Math.min(1, this.saccade.tLeft / 0.85 + 0.3);
      // positive cmd.yaw turns LEFT; dodge>0 means threat on the right
      saccYaw = this.saccade.dir * 1.5 * envelope;
      saccRoll = -this.saccade.dir * 1.0 * envelope;
      saccPitch = -0.06 * envelope; // level off during the turn
      if (this.saccade.tLeft <= 0 || loomMean < 0.05) {
        this.saccade.active = false;
        this.justEnded = true;
        this.justEndTimer = 0.5;
        this.sinceSaccadeEnd = 0;
      }
    }
    this.sinceSaccadeEnd += dt;
    if (this.justEnded) {
      this.justEndTimer -= dt;
      if (this.justEndTimer <= 0) this.justEnded = false;
    }

    // PD altitude hold around 2 m (hover throttle = 9.81/34 = 0.29)
    const lift =
      0.29 + 0.22 * Math.tanh((2.0 - inp.altitude) * 0.5) -
      0.06 * inp.vy - 0.08 * loomMean;

    // cruise forward; looming brakes the drone (flies stop at expansion)
    const pitch = clamp(-0.35 * (1 - 0.75 * loomMean) + saccPitch, -1, 0.1);

    // --- spontaneous exploration (real flies turn saccades every ~2-5 s) ---
    // OU process on turn bias: bias += (-bias/tau + sigma*noise) * dt
    this.wanderBias += (-this.wanderBias / 2.5 + (this.rng() - 0.5) * 1.6) * dt;
    this.wanderBias = clamp(this.wanderBias, -0.6, 0.6);
    this.wanderSaccadeTimer -= dt;
    let wanderYaw = 0, wanderRoll = 0;
    if (this.wanderSaccade.active) {
      this.wanderSaccade.tLeft -= dt;
      const env = Math.min(1, this.wanderSaccade.tLeft / 0.4 + 0.3);
      wanderYaw = this.wanderSaccade.dir * 1.1 * env;
      wanderRoll = -this.wanderSaccade.dir * 0.7 * env;
      if (this.wanderSaccade.tLeft <= 0) this.wanderSaccade.active = false;
    } else if (this.wanderSaccadeTimer <= 0 && !this.saccade.active) {
      this.wanderSaccade.active = true;
      this.wanderSaccade.tLeft = 0.4;
      this.wanderSaccade.dir = this.rng() < 0.5 ? -1 : 1;
      this.wanderSaccadeTimer = 2 + this.rng() * 4;
    }

    // course attraction is replaced by the wander bias in explore mode;
    // yaw: escape saccade dominates, then wander saccade, then drift
    const yaw = clamp(
      saccYaw + 2.0 * dodge + 0.4 * steer + wanderYaw + 1.2 * this.wanderBias,
      -1, 1,
    );
    // roll: bank into the turn
    const roll = clamp(
      saccRoll - 1.8 * dodge - 0.7 * steer + wanderRoll - 0.8 * this.wanderBias,
      -1, 1,
    );
    const throttle = clamp(lift, 0, 1);

    this.debug = {
      loomingL,
      loomingR,
      darkL: inp.stats.looming.L,
      darkR: inp.stats.looming.R,
      baseL: this.darkBase?.L ?? -1,
      baseR: this.darkBase?.R ?? -1,
      lplc: actOf("LPLC"),
      lc: actOf("LC"),
      t4: actOf("T4"),
      t5: actOf("T5"),
      avertRaw: avert,
      saccading: this.saccade.active ? 1 : 0,
      saccDir: this.saccade.dir,
      dodge,
      attract: 0,
      wanderBias: this.wanderBias,
      wanderSaccading: this.wanderSaccade.active ? 1 : 0,
      pitch,
    };

    return {
      throttle,
      pitch,
      roll,
      yaw,
      pools: { avert: clamp01(avert), steer: dodge, lift: throttle },
    };
  }
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
function clamp01(x: number): number {
  return clamp(x, 0, 1);
}
