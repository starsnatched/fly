export interface CircuitNeuron {
  id: number;
  pop: string;
  type: string;
  side: string;
}

export interface CircuitMeta {
  dataset: string;
  license: string;
  maxHops: number;
  minSynapseWeight: number;
  populations: string[];
}

export interface CircuitPayload {
  meta: CircuitMeta;
  neurons: CircuitNeuron[];
  /** [srcIdx, tgtIdx, synapseCount] into neurons[] */
  edges: [number, number, number][];
  /** [prePop, postPop, aggregatedSynapses] */
  popMatrix: [string, string, number][];
}

export interface LifPayload {
  meta: {
    dataset: string;
    license: string;
    maxHops: number;
    minSynapseWeight: number;
    ntHeuristic: string;
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

export interface DroneState {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
  alive: boolean;
  crashCount: number;
}
