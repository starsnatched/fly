/** Sensory + body state handed to a controller each frame. */
export interface BrainInputs {
  /** height above ground (m) */
  altitude: number;
  /** vertical velocity m/s (positive = climbing) */
  vy: number;
  /** current heading (yaw, radians; 0 = -Z) */
  heading: number;
  speed: number;
  dt: number;
  /** clearance ahead along heading (m); drives the reward signal if present */
  clearance?: number;
  /** raw per-eye RGB grids (RENDER_W*RENDER_H*3, stride 3 = R,G,B) — the only
   *  visual signal the brain receives: the server's hex-consistent sampler
   *  accepts any client resolution */
  rgb?: { L: Float32Array; R: Float32Array };
}

/** Flight commands emitted by a controller. */
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

/** One compound eye's processed flow fields (client-side visualization). */
export interface EyeFlow {
  h: Float32Array; // GRID_W * GRID_H, positive = rightward motion
  v: Float32Array; // GRID_W * GRID_H, positive = upward motion
  lum: Float32Array; // GRID_W * GRID_H, 0..1 luminance
}
