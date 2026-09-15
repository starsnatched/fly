import * as THREE from "three";
import type { EyeFlow } from "./types";

/**
 * Compound-eye vision model.
 *
 * Two eye views rendered at 96x54 from the drone FPV camera (yawed +-35 deg),
 * downsampled to a 48x27 luminance grid per eye, then processed by
 * Hassenstein-Reichardt elementary motion detectors (EMDs) -- the canonical
 * fly motion computation, standing in for the T4 (ON) / T5 (OFF) channels.
 */

export const GRID_W = 48;
export const GRID_H = 27;
const RENDER_W = 96;
const RENDER_H = 54;
const EMD_DT = 1 / 30; // fly photoreceptor flicker-fusion timescale

export class FlyVision {
  private rtL: THREE.WebGLRenderTarget;
  private rtR: THREE.WebGLRenderTarget;
  private bufL: Uint8Array;
  private bufR: Uint8Array;
  private eyeL: EyeFlow;
  private eyeR: EyeFlow;
  private prevLum: { L: Float32Array; R: Float32Array };
  private accH: { L: Float32Array; R: Float32Array };
  private accV: { L: Float32Array; R: Float32Array };
  private tSinceEMD = 0;

  constructor(private renderer: THREE.WebGLRenderer) {
    const mk = () => {
      const rt = new THREE.WebGLRenderTarget(RENDER_W, RENDER_H, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        depthBuffer: true,
      });
      // sRGB so readPixels match what the main canvas displays
      // (otherwise the RT holds linear values and everything reads dark)
      rt.texture.colorSpace = THREE.SRGBColorSpace;
      return rt;
    };
    this.rtL = mk();
    this.rtR = mk();
    this.bufL = new Uint8Array(RENDER_W * RENDER_H * 4);
    this.bufR = new Uint8Array(RENDER_W * RENDER_H * 4);

    const mkFlow = (): EyeFlow => ({
      h: new Float32Array(GRID_W * GRID_H),
      v: new Float32Array(GRID_W * GRID_H),
      lum: new Float32Array(GRID_W * GRID_H),
    });
    this.eyeL = mkFlow();
    this.eyeR = mkFlow();
    this.prevLum = {
      L: new Float32Array(GRID_W * GRID_H),
      R: new Float32Array(GRID_W * GRID_H),
    };
    this.accH = { L: new Float32Array(GRID_W * GRID_H), R: new Float32Array(GRID_W * GRID_H) };
    this.accV = { L: new Float32Array(GRID_W * GRID_H), R: new Float32Array(GRID_W * GRID_H) };
  }

  get flows(): { L: EyeFlow; R: EyeFlow } {
    return { L: this.eyeL, R: this.eyeR };
  }

  /** Render both eye views from the drone pose, then update EMDs. */
  update(
    scene: THREE.Scene,
    basePos: THREE.Vector3,
    yaw: number,
    pitch: number,
    roll: number,
    dt: number,
  ): void {
    const cam = new THREE.PerspectiveCamera(100, RENDER_W / RENDER_H, 0.1, 600);
    const baseQ = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(pitch, yaw, roll, "YXZ"),
    );

    for (const side of ["L", "R"] as const) {
      const eyeYaw = side === "L" ? -0.6 : 0.6; // ~35 deg per eye
      const eyeQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, eyeYaw, 0));
      cam.quaternion.copy(baseQ).multiply(eyeQ);
      cam.position.copy(basePos);
      const rt = side === "L" ? this.rtL : this.rtR;
      this.renderer.setRenderTarget(rt);
      this.renderer.clear();
      this.renderer.render(scene, cam);
      this.renderer.readRenderTargetPixels(
        rt, 0, 0, RENDER_W, RENDER_H, side === "L" ? this.bufL : this.bufR,
      );
    }
    this.renderer.setRenderTarget(null);

    // downsample to GRID_W x GRID_H luminance (GL y-flip handled here)
    for (const side of ["L", "R"] as const) {
      const buf = side === "L" ? this.bufL : this.bufR;
      const flow = side === "L" ? this.eyeL : this.eyeR;
      for (let gy = 0; gy < GRID_H; gy++) {
        for (let gx = 0; gx < GRID_W; gx++) {
          const x0 = Math.floor((gx * RENDER_W) / GRID_W);
          const y0 = Math.floor((gy * RENDER_H) / GRID_H);
          const x1 = Math.max(x0 + 1, Math.floor(((gx + 1) * RENDER_W) / GRID_W));
          const y1 = Math.max(y0 + 1, Math.floor(((gy + 1) * RENDER_H) / GRID_H));
          let sum = 0, n = 0;
          for (let y = y0; y < y1; y++) {
            const fy = RENDER_H - 1 - y; // flip to top-down
            for (let x = x0; x < x1; x++) {
              const i = (fy * RENDER_W + x) * 4;
              sum += 0.299 * buf[i] + 0.587 * buf[i + 1] + 0.114 * buf[i + 2];
              n++;
            }
          }
          flow.lum[gy * GRID_W + gx] = n ? sum / n / 255 : 0;
        }
      }
    }

    this.tSinceEMD += dt;
    if (this.tSinceEMD >= EMD_DT) {
      this.tSinceEMD %= EMD_DT;
      this.emdStep();
    }
  }

  /** One Hassenstein-Reichardt EMD step: correlate luminance now vs EMD_DT ago. */
  private emdStep(): void {
    for (const side of ["L", "R"] as const) {
      const flow = side === "L" ? this.eyeL : this.eyeR;
      const prev = side === "L" ? this.prevLum.L : this.prevLum.R;
      const accH = side === "L" ? this.accH.L : this.accH.R;
      const accV = side === "L" ? this.accV.L : this.accV.R;
      for (let gy = 1; gy < GRID_H - 1; gy++) {
        for (let gx = 1; gx < GRID_W - 1; gx++) {
          const i = gy * GRID_W + gx;
          const cur = flow.lum[i];
          const p = prev[i];
          const dCenter = cur - p;
          const rN = flow.lum[i + 1], lN = flow.lum[i - 1];
          const uN = flow.lum[i - GRID_W], dN = flow.lum[i + GRID_W];
          const pr = prev[i + 1], pl = prev[i - 1];
          const pu = prev[i - GRID_W], pd = prev[i + GRID_W];
          // two-mirror correlation: center-now x neighbor-then, minus the mirror
          const hr = dCenter * (p - pl) - (p - pl) * 0; // placeholder replaced below
          void hr;
          const rh = dCenter * (cur - rN >= 0 ? 1 : 1); // simplified below
          void rh;
          const hRight = dCenter * (p - pl) - (lN - pl) * (p - prev[i + 1]);
          const hLeft = dCenter * (p - pr) - (rN - pr) * (p - prev[i - 1]);
          const vDown = dCenter * (p - pu) - (dN - pd) * (p - prev[i - GRID_W]);
          const vUp = dCenter * (p - pd) - (uN - pu) * (p - prev[i + GRID_W]);
          const h = hRight - hLeft;
          const v = vDown - vUp;
          accH[i] += (h - accH[i]) * 0.35;
          accV[i] += (v - accV[i]) * 0.35;
        }
      }
      flow.h.set(accH);
      flow.v.set(accV);
      prev.set(flow.lum);
    }
  }

  /** Mean flow per eye + vertical thirds, all squashed to -1..1. */
  quadrantStats(): {
    flowH: { L: number; R: number; U: number; D: number };
    flowV: { L: number; R: number; U: number; D: number };
    looming: { L: number; R: number; U: number; D: number };
    /** mean horizontal flow on left/right half of each eye (for expansion) */
    halfFlow: { L: { left: number; right: number }; R: { left: number; right: number } };
    /** aggregates for the LIF engine */
    agg: {
      flowHLeft: number;
      flowHMean: number;
      flowVMean: number;
      darkFraction: number;
    };
  } {
    const out = {
      flowH: { L: 0, R: 0, U: 0, D: 0 },
      flowV: { L: 0, R: 0, U: 0, D: 0 },
      looming: { L: 0, R: 0, U: 0, D: 0 },
      halfFlow: {
        L: { left: 0, right: 0 },
        R: { left: 0, right: 0 },
      },
      agg: { flowHLeft: 0, flowHMean: 0, flowVMean: 0, darkFraction: 0 },
    };
    const squash = (x: number) => Math.max(-1, Math.min(1, x * 6));
    for (const side of ["L", "R"] as const) {
      const flow = side === "L" ? this.eyeL : this.eyeR;
      let hSum = 0, vSum = 0, loomSum = 0, n = 0;
      let hLeft = 0, nL = 0, hRight = 0, nR = 0;
      for (let gy = 1; gy < GRID_H - 1; gy++) {
        for (let gx = 1; gx < GRID_W - 1; gx++) {
          const i = gy * GRID_W + gx;
          hSum += squash(flow.h[i]);
          vSum += squash(flow.v[i]);
          loomSum += flow.lum[i] < 0.35 ? 1 : 0;
          n++;
          if (gx < GRID_W / 2) {
            hLeft += squash(flow.h[i]);
            nL++;
          } else {
            hRight += squash(flow.h[i]);
            nR++;
          }
        }
      }
      out.flowH[side] = n ? hSum / n : 0;
      out.flowV[side] = n ? vSum / n : 0;
      out.looming[side] = n ? loomSum / n : 0;
      out.halfFlow[side].left = nL ? hLeft / nL : 0;
      out.halfFlow[side].right = nR ? hRight / nR : 0;
      // central dark fraction: angular size of the nearest silhouette
      // (center 50% width x middle 60% height of the eye)
      let cSum = 0, cN = 0;
      for (let gy = Math.floor(GRID_H * 0.2); gy < Math.ceil(GRID_H * 0.8); gy++) {
        for (let gx = Math.floor(GRID_W * 0.25); gx < Math.ceil(GRID_W * 0.75); gx++) {
          if (flow.lum[gy * GRID_W + gx] < 0.35) cSum++;
          cN++;
        }
      }
      out.looming[side] = cN ? cSum / cN : 0;
    }
    const third = (flow: EyeFlow, from: number, to: number, key: "h" | "v") => {
      let s = 0, n = 0;
      for (let gy = from; gy < to; gy++)
        for (let gx = 1; gx < GRID_W - 1; gx++) {
          s += flow[key][gy * GRID_W + gx];
          n++;
        }
      return n ? squash(s / n) : 0;
    };
    out.flowH.U = (third(this.eyeL, 0, 9, "h") + third(this.eyeR, 0, 9, "h")) / 2;
    out.flowH.D = (third(this.eyeL, 18, 27, "h") + third(this.eyeR, 18, 27, "h")) / 2;
    out.flowV.U = (third(this.eyeL, 0, 9, "v") + third(this.eyeR, 0, 9, "v")) / 2;
    out.flowV.D = (third(this.eyeL, 18, 27, "v") + third(this.eyeR, 18, 27, "v")) / 2;

    // aggregates over the left half of the combined field + means
    let hl = 0, vm = 0, hm = 0, dk = 0, nA = 0;
    for (const flow of [this.eyeL, this.eyeR]) {
      for (let gy = 1; gy < GRID_H - 1; gy++) {
        for (let gx = 1; gx < GRID_W - 1; gx++) {
          const i = gy * GRID_W + gx;
          const h = squash(flow.h[i]);
          hm += h;
          vm += squash(flow.v[i]);
          dk += flow.lum[i] < 0.35 ? 1 : 0;
          if (gx < GRID_W / 2) hl += h;
          nA++;
        }
      }
    }
    if (nA) {
      out.agg.flowHLeft = (hl / (nA / 2)) * 1;
      out.agg.flowHMean = hm / nA;
      out.agg.flowVMean = vm / nA;
      out.agg.darkFraction = dk / nA;
    }
    return out;
  }

  /** Draw a pseudo-color flow field to a small HUD canvas. */
  drawEyeCanvas(canvas: HTMLCanvasElement, side: "L" | "R"): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const flow = side === "L" ? this.eyeL : this.eyeR;
    const img = ctx.createImageData(GRID_W, GRID_H);
    for (let i = 0; i < GRID_W * GRID_H; i++) {
      const h = Math.max(-1, Math.min(1, flow.h[i] * 4));
      const v = Math.max(-1, Math.min(1, flow.v[i] * 4));
      const mag = Math.min(1, Math.hypot(h, v));
      const lum = flow.lum[i];
      img.data[i * 4 + 0] = 40 + mag * 200;
      img.data[i * 4 + 1] = 30 + lum * 160;
      img.data[i * 4 + 2] = 60 + Math.abs(v) * 160;
      img.data[i * 4 + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = GRID_W;
    tmp.height = GRID_H;
    tmp.getContext("2d")!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  }

  dispose(): void {
    this.rtL.dispose();
    this.rtR.dispose();
  }
}
