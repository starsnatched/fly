/**
 * Shared sensory-processing front-end used by both controllers:
 *  - looming detection (dark-silhouette growth vs adapted baseline)
 *  - escape-saccade state machine (with post-saccadic reset + holdoff)
 *  - spontaneous wander (OU turn-bias + spontaneous saccades)
 *  - command synthesis from descending-pool activations
 *
 * The rate-based brain (brain.ts) and the spiking LIF brain (lif.ts) both
 * compose this, so switching between them preserves flight behavior while
 * swapping where pool activations come from.
 */

export interface EyeStats {
  flowH: { L: number; R: number; U: number; D: number };
  flowV: { L: number; R: number; U: number; D: number };
  /** dark-pixel fraction per eye (silhouette size proxy) */
  looming: { L: number; R: number; U: number; D: number };
  /** mean horizontal EMD flow on each half of each eye (expansion readout) */
  halfFlow: { L: { left: number; right: number }; R: { left: number; right: number } };
  // aggregates for the LIF engine's direction-tuned input
  agg: {
    flowHLeft: number;
    flowHMean: number;
    flowVMean: number;
    darkFraction: number;
  };
}

export interface PoolActivations {
  avert: number; // 0..1 escape drive
  steer: number; // -1..1 turn preference
  lift: number;  // 0..1 altitude drive
}

export interface SensoryState {
  loomingL: number;
  loomingR: number;
  dodge: number;
  loomMean: number;
  steer: number;
  escape: { active: boolean; dir: number; yaw: number; roll: number; pitch: number };
  wander: { active: boolean; dir: number; yaw: number; roll: number; bias: number };
}

export class SharedCircuit {
  // looming state
  private darkBase: { L: number; R: number } | null = null;
  private loomSm: { L: number; R: number } = { L: 0, R: 0 };

  // escape saccade state
  private saccade = { active: false, dir: 1, tLeft: 0, cooldown: 0 };
  private lastDodge = 1;
  private justEnded = false;
  private justEndTimer = 0;
  private sinceSaccadeEnd = 99;

  // wander state
  private wanderBias = 0;
  private wanderSaccadeTimer = 2 + Math.random() * 3;
  private wanderSaccade = { active: false, dir: 1, tLeft: 0 };

  // attract convenience field for debugging compatibility
  debug: Record<string, number> = {};

  /** Update looming detection + saccade/wander state machines. */
  update(stats: EyeStats, dt: number): SensoryState {
    // -- looming: central dark fraction growth above adapted baseline --
    const darkL = stats.looming.L, darkR = stats.looming.R;
    if (!this.darkBase) this.darkBase = { L: darkL, R: darkR };
    const postSacc = this.sinceSaccadeEnd < 2.0;
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
    const loomMean = (loomingL + loomingR) / 2;
    const dodge = loomingR - loomingL;
    if (Math.abs(dodge) > 0.05) this.lastDodge = Math.sign(dodge);

    const steer = 1.4 * (stats.flowH.R - stats.flowH.L) * 0.9;

    // -- escape saccade state machine --
    if (this.saccade.cooldown > 0) this.saccade.cooldown -= dt;
    if (!this.saccade.active && loomMean > 0.25 && this.saccade.cooldown <= 0 && !this.justEnded) {
      this.saccade.active = true;
      this.saccade.tLeft = 0.85;
      this.saccade.dir = Math.abs(dodge) > 0.08
        ? Math.sign(dodge)
        : (Math.abs(steer) > 0.05 ? Math.sign(steer) : this.lastDodge);
      this.saccade.cooldown = 1.5;
    }
    let escYaw = 0, escRoll = 0, escPitch = 0;
    if (this.saccade.active) {
      this.saccade.tLeft -= dt;
      const env = Math.min(1, this.saccade.tLeft / 0.85 + 0.3);
      escYaw = this.saccade.dir * 1.5 * env;
      escRoll = -this.saccade.dir * 1.0 * env;
      escPitch = -0.06 * env;
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

    // -- wander (OU turn-bias + spontaneous saccades) --
    this.wanderBias += (-this.wanderBias / 2.5 + (Math.random() - 0.5) * 1.6) * dt;
    this.wanderBias = clamp(this.wanderBias, -0.6, 0.6);
    this.wanderSaccadeTimer -= dt;
    let wandYaw = 0, wandRoll = 0;
    if (this.wanderSaccade.active) {
      this.wanderSaccade.tLeft -= dt;
      const env = Math.min(1, this.wanderSaccade.tLeft / 0.4 + 0.3);
      wandYaw = this.wanderSaccade.dir * 1.1 * env;
      wandRoll = -this.wanderSaccade.dir * 0.7 * env;
      if (this.wanderSaccade.tLeft <= 0) this.wanderSaccade.active = false;
    } else if (this.wanderSaccadeTimer <= 0 && !this.saccade.active) {
      this.wanderSaccade.active = true;
      this.wanderSaccade.tLeft = 0.4;
      this.wanderSaccade.dir = Math.random() < 0.5 ? -1 : 1;
      this.wanderSaccadeTimer = 2 + Math.random() * 4;
    }

    this.debug = {
      loomingL, loomingR,
      darkL, darkR,
      baseL: this.darkBase.L, baseR: this.darkBase.R,
      dodge, loomMean,
      saccading: this.saccade.active ? 1 : 0,
      saccDir: this.saccade.dir,
      wanderSaccading: this.wanderSaccade.active ? 1 : 0,
      wanderBias: this.wanderBias,
    };

    return {
      loomingL,
      loomingR,
      dodge,
      loomMean,
      steer,
      escape: { active: this.saccade.active, dir: this.saccade.dir, yaw: escYaw, roll: escRoll, pitch: escPitch },
      wander: { active: this.wanderSaccade.active, dir: this.wanderSaccade.dir, yaw: wandYaw, roll: wandRoll, bias: this.wanderBias },
    };
  }

  /** Called on body collision: escape turn + re-roll wander. */
  notifyBump(): void {
    if (!this.saccade.active && this.saccade.cooldown <= 0) {
      this.saccade.active = true;
      this.saccade.tLeft = 0.6;
      this.saccade.dir = Math.random() < 0.5 ? -1 : 1;
      this.saccade.cooldown = 1.0;
    }
    this.wanderBias = (Math.random() - 0.5) * 1.2;
  }

  /** Synthesize flight commands from pool activations + shared state. */
  commands(_pools: PoolActivations, s: SensoryState, altitude: number, vy: number): {
    throttle: number; pitch: number; roll: number; yaw: number;
  } {
    const loomMean = s.loomMean;
    const lift =
      0.29 + 0.22 * Math.tanh((2.0 - altitude) * 0.5) -
      0.06 * vy - 0.08 * loomMean;
    const pitch = clamp(-0.35 * (1 - 0.75 * loomMean) + s.escape.pitch, -1, 0.1);
    const yaw = clamp(
      s.escape.yaw + 2.0 * s.dodge + 0.4 * s.steer + s.wander.yaw + 1.2 * s.wander.bias,
      -1, 1,
    );
    const roll = clamp(
      s.escape.roll - 1.8 * s.dodge - 0.7 * s.steer + s.wander.roll - 0.8 * s.wander.bias,
      -1, 1,
    );
    return { throttle: clamp01(lift), pitch, roll, yaw };
  }
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
function clamp01(x: number): number {
  return clamp(x, 0, 1);
}
