import * as THREE from "three";
import type { EyeFlow } from "./types";

/**
 * Vision model — a single forward-facing eye.
 *
 * One view rendered at RENDER_W x RENDER_H from the drone FPV camera. The
 * FULL-resolution RGB is what streams to the brain backend (the server's
 * hex-consistent sampler takes any client size); inside the brain, the
 * connectome's left/right lamina columns view the left/right HALF of this
 * one frame, so optic-flow differences across the field still steer it.
 */

export const GRID_W = 48;
export const GRID_H = 27;
export const RENDER_W = 192;
export const RENDER_H = 108;
const EMD_DT = 1 / 30; // fly photoreceptor flicker-fusion timescale

export class FlyVision {
  private rt: THREE.WebGLRenderTarget;
  private buf: Uint8Array;
  private eyeL: EyeFlow;
  private eyeR: EyeFlow;
  /** Full-resolution RGB of the forward eye (0..1, stride 3 = R,G,B) — the
   *  exact bytes streamed to the brain backend. */
  private rgbL: Float32Array;
  private rgbR: Float32Array;
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
    this.rt = mk();
    this.buf = new Uint8Array(RENDER_W * RENDER_H * 4);

    const mkFlow = (): EyeFlow => ({
      h: new Float32Array(GRID_W * GRID_H),
      v: new Float32Array(GRID_W * GRID_H),
      lum: new Float32Array(GRID_W * GRID_H),
    });
    this.eyeL = mkFlow();
    this.eyeR = mkFlow();
    this.rgbL = new Float32Array(RENDER_W * RENDER_H * 3);
    this.rgbR = new Float32Array(RENDER_W * RENDER_H * 3);
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

  /** Mean RGB per grid cell, per eye (0..1, stride 3 = R,G,B). */
  get rgb(): { L: Float32Array; R: Float32Array } {
    return { L: this.rgbL, R: this.rgbR };
  }

  /** Render the forward-eye view from the drone pose, then update EMDs. */
  update(
    scene: THREE.Scene,
    basePos: THREE.Vector3,
    yaw: number,
    pitch: number,
    roll: number,
    dt: number,
  ): void {
    const cam = new THREE.PerspectiveCamera(100, RENDER_W / RENDER_H, 0.1, 600);
    cam.quaternion.setFromEuler(new THREE.Euler(pitch, yaw, roll, "YXZ"));
    cam.position.copy(basePos);
    this.renderer.setRenderTarget(this.rt);
    this.renderer.clear();
    this.renderer.render(scene, cam);
    this.renderer.readRenderTargetPixels(
      this.rt, 0, 0, RENDER_W, RENDER_H, this.buf,
    );
    this.renderer.setRenderTarget(null);

    // pass 1: fill the full-resolution RGB buffer (the streamed signal)
    // pass 2: box-downsample to the GRID_W x GRID_H luminance working grid
    //         (GL y-flip handled here)
    {
      const buf = this.buf;
      const rgb = this.rgbL;
      for (let y = 0; y < RENDER_H; y++) {
        const fy = RENDER_H - 1 - y; // flip to top-down
        const row = fy * RENDER_W * 4;
        const outRow = y * RENDER_W * 3;
        for (let x = 0; x < RENDER_W; x++) {
          const i = row + x * 4;
          const o = outRow + x * 3;
          rgb[o] = buf[i] / 255;
          rgb[o + 1] = buf[i + 1] / 255;
          rgb[o + 2] = buf[i + 2] / 255;
        }
      }
      // both flow grids track the same image (left/right halves differ only)
      for (let gy = 0; gy < GRID_H; gy++) {
        for (let gx = 0; gx < GRID_W; gx++) {
          const x0 = Math.floor((gx * RENDER_W) / GRID_W);
          const y0 = Math.floor((gy * RENDER_H) / GRID_H);
          const x1 = Math.max(x0 + 1, Math.floor(((gx + 1) * RENDER_W) / GRID_W));
          const y1 = Math.max(y0 + 1, Math.floor(((gy + 1) * RENDER_H) / GRID_H));
          let sum = 0, n = 0;
          for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) {
              const i = ((RENDER_H - 1 - y) * RENDER_W + x) * 4;
              sum += 0.299 * buf[i] + 0.587 * buf[i + 1] + 0.114 * buf[i + 2];
              n++;
            }
          }
          const lum = n ? sum / n / 255 : 0;
          this.eyeL.lum[gy * GRID_W + gx] = lum;
          this.eyeR.lum[gy * GRID_W + gx] = lum;
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

  /** Draw the raw eye image — exactly what the brain/backend receives — to a HUD canvas. */
  drawRgbCanvas(canvas: HTMLCanvasElement, side: "L" | "R"): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const rgb = side === "L" ? this.rgbL : this.rgbR; // same frame; halves only differ
    const img = ctx.createImageData(RENDER_W, RENDER_H);
    const data = img.data;
    for (let i = 0; i < RENDER_W * RENDER_H; i++) {
      data[i * 4 + 0] = (rgb[i * 3] * 255) | 0;
      data[i * 4 + 1] = (rgb[i * 3 + 1] * 255) | 0;
      data[i * 4 + 2] = (rgb[i * 3 + 2] * 255) | 0;
      data[i * 4 + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = RENDER_W;
    tmp.height = RENDER_H;
    tmp.getContext("2d")!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true; // full-res now; smooth scale reads better
    ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  }

  dispose(): void {
    this.rt.dispose();
  }
}
