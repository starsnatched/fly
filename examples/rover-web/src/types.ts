/** Sensory + body state handed to the brain each frame (rover flavor). */
export interface BrainInputs {
  /** chassis height above ground (m) — near-constant for a rover */
  altitude: number;
  /** wheel odometry speed (m/s, forward positive) */
  speed: number;
  /** bumper contact this instant */
  collision: boolean;
  /** raw forward-eye RGB grid (RENDER_W*RENDER_H*3, stride 3 = R,G,B) —
   *  the only visual signal the brain receives; the server's hex-consistent
   *  sampler accepts any client resolution and splits it across the
   *  retina's two hemispheres (sensors.eyes.count = 1) */
  rgb: Float32Array;
}

/** Drive commands emitted by the brain (rover channels, config/profiles/rover.json). */
export interface BrainOutputs {
  /** 0..1 drive effort */
  throttle: number;
  /** -1..1 skid-steer rate (positive = left) */
  steer: number;
  /** named descending-pool activations (telemetry mirror for the HUD) */
  pools: { drive: number; steer: number };
}
