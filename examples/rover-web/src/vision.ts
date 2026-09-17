import * as THREE from "three";

/**
 * Vision model — a single forward-facing eye.
 *
 * One view rendered at RENDER_W x RENDER_H straight ahead (config/profiles/
 * rover.json sets sensors.eyes.count = 1). The FULL-resolution RGB frame is
 * what streams to the brain backend as eye 0; the server's hex-consistent
 * sampler resamples it through the retina's real coordinates and its
 * left/right lamina HALVES each view half of the image, so turning reads as
 * optic-flow differences across the field — how many ground insects steer.
 */

export const GRID_W = 48;
export const GRID_H = 27;
export const RENDER_W = 192;
export const RENDER_H = 108;

/** Wide, ground-vehicle camera: see both wheel tracks and the horizon. */
const VFOV = 62;
/** Mount pitch (radians): nose-down tilt to keep near terrain in frame. */
const MOUNT_PITCH = -0.12;
/** Eye position in the body frame — the mast-top sensor (matches rover.ts:
 *  mast at z=-0.35, eye sphere at y=0.98). Must sit AHEAD of the mast so
 *  the rover never films its own hardware. */
const EYE_X = 0;
const EYE_Y = 0.98;
const EYE_Z = -0.55;

export class RoverVision {
  private rt: THREE.WebGLRenderTarget;
  private buf: Uint8Array;
  /** Full-resolution RGB (0..1, stride 3 = R,G,B) — the exact bytes
   *  streamed to the brain backend as eye 0. */
  private rgb: Float32Array;
  private lum: Float32Array;

  constructor(private renderer: THREE.WebGLRenderer) {
    this.rt = new THREE.WebGLRenderTarget(RENDER_W, RENDER_H, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
    });
    // sRGB so readPixels match what the main canvas displays (otherwise the
    // RT holds linear values and everything reads dark)
    this.rt.texture.colorSpace = THREE.SRGBColorSpace;
    this.buf = new Uint8Array(RENDER_W * RENDER_H * 4);
    this.rgb = new Float32Array(RENDER_W * RENDER_H * 3);
    this.lum = new Float32Array(GRID_W * GRID_H);
  }

  /** Mean RGB per pixel (0..1, stride 3 = R,G,B). */
  get frame(): Float32Array {
    return this.rgb;
  }

  /** Box-downsampled luminance grid (client-side view of the eye). */
  get luminance(): Float32Array {
    return this.lum;
  }

  /** Render the forward eye from the chassis pose and extract RGB. */
  update(
    scene: THREE.Scene,
    pos: THREE.Vector3,
    yaw: number,
    pitch: number,
    roll: number,
  ): void {
    const cam = new THREE.PerspectiveCamera(VFOV, RENDER_W / RENDER_H, 0.1, 600);
    cam.quaternion.setFromEuler(
      new THREE.Euler(pitch + MOUNT_PITCH, yaw, roll, "YXZ"),
    );
    // the camera IS the eye: mast-top sensor position, yawed with the body
    cam.position.set(
      pos.x + EYE_X * Math.cos(yaw) + EYE_Z * Math.sin(yaw),
      pos.y + EYE_Y,
      pos.z - EYE_X * Math.sin(yaw) + EYE_Z * Math.cos(yaw),
    );

    this.renderer.setRenderTarget(this.rt);
    this.renderer.clear();
    this.renderer.render(scene, cam);
    this.renderer.readRenderTargetPixels(
      this.rt, 0, 0, RENDER_W, RENDER_H, this.buf,
    );
    this.renderer.setRenderTarget(null);

    // pass 1: fill the full-resolution RGB buffer (the streamed signal)
    // pass 2: box-downsample luminance to the GRID_W x GRID_H working grid
    //         (GL y-flip handled here)
    for (let y = 0; y < RENDER_H; y++) {
      const fy = RENDER_H - 1 - y; // flip to top-down
      const row = fy * RENDER_W * 4;
      const outRow = y * RENDER_W * 3;
      const gy = Math.min(GRID_H - 1, Math.floor((y * GRID_H) / RENDER_H));
      for (let x = 0; x < RENDER_W; x++) {
        const i = row + x * 4;
        const o = outRow + x * 3;
        const r = this.buf[i] / 255;
        const g = this.buf[i + 1] / 255;
        const b = this.buf[i + 2] / 255;
        this.rgb[o] = r;
        this.rgb[o + 1] = g;
        this.rgb[o + 2] = b;
        const gx = Math.min(GRID_W - 1, Math.floor((x * GRID_W) / RENDER_W));
        this.lum[gy * GRID_W + gx] += 0.299 * r + 0.587 * g + 0.114 * b;
      }
    }
    const cells = (RENDER_W / GRID_W) * (RENDER_H / GRID_H); // 16 exactly
    for (let i = 0; i < this.lum.length; i++) this.lum[i] /= cells;
  }

  /** Draw the raw eye image — exactly what the brain receives — to a HUD canvas. */
  drawRgbCanvas(canvas: HTMLCanvasElement): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(RENDER_W, RENDER_H);
    const data = img.data;
    for (let i = 0; i < RENDER_W * RENDER_H; i++) {
      data[i * 4 + 0] = (this.rgb[i * 3] * 255) | 0;
      data[i * 4 + 1] = (this.rgb[i * 3 + 1] * 255) | 0;
      data[i * 4 + 2] = (this.rgb[i * 3 + 2] * 255) | 0;
      data[i * 4 + 3] = 255;
    }
    const tmp = document.createElement("canvas");
    tmp.width = RENDER_W;
    tmp.height = RENDER_H;
    tmp.getContext("2d")!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  }

  dispose(): void {
    this.rt.dispose();
  }
}
