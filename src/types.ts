import type { EyeStats } from "./shared";

/** Sensory + body state handed to the controller each frame. */
export interface BrainInputs {
  stats: EyeStats;
  /** height above ground (m) */
  altitude: number;
  /** vertical velocity m/s (positive = climbing) */
  vy: number;
  /** current heading (yaw, radians; 0 = -Z) */
  heading: number;
  speed: number;
  dt: number;
}

/** Flight commands emitted by the controller. */
export interface BrainOutputs {
  /** 0..1 vertical thrust */
  throttle: number;
  /** -1..1 nose down/up */
  pitch: number;
  /** -1..1 */
  roll: number;
  /** -1..1 yaw rate command */
  yaw: number;
  /** named descending-pool activations (telemetry) */
  pools: { avert: number; steer: number; lift: number };
}

export interface LifPayload {
  meta: {
    dataset: string;
    license: string;
    maxHops: number;
    minSynapseWeight: number;
    ntHeuristic: string;
    ntSource: string;
    hexFallback: number[];
    dirConvention: string;
  };
  populations: string[];
  neurons: {
    id: number[];
    pop: number[];
    side: number[];
    hex: number[][];
    nt: number[];
    dir: number[];
    /** confidence (max mean NT probability) per neuron */
    ntConf: number[];
  };
  /** flat triplets [src, dst, weight] */
  edges: number[];
}

/** Per-eye motion energy: horizontal (rightward) and vertical (upward) flow. */
export interface EyeFlow {
  h: Float32Array; // GRID_W * GRID_H, positive = rightward motion
  v: Float32Array; // GRID_W * GRID_H, positive = upward motion
  lum: Float32Array; // current luminance grid
}
