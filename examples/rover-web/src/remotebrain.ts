/**
 * Remote brain client: streams the forward-eye RGB to the C connectome server
 * (cengine) over a WebSocket and receives actuator channels + telemetry back.
 * The browser is a pure embodiment: eye + proprioception in, actuator
 * channels out. Rover flavor: one eye frame + JSON state, and the action
 * frame's throttle/steer channels are what config/profiles/rover.json
 * declares.
 */
import { RENDER_W, RENDER_H } from "./vision";

/** Drive command the embodiment's physics consumes. */
export interface Cmd {
  throttle: number;
  steer: number;
  pools: { drive: number; steer: number };
}

/** Minimal telemetry snapshot for the HUD. */
export interface RemoteTelemetry {
  neurons: number;
  edges: number;
  rates: Record<string, number>;
  dopa: number;
  dnSteer: number;
  memEdited: number;
  learning: boolean;
  simMs: number;
}

/** Eye-frame streaming rate — mirrors sensors.visionHz in config/profiles/rover.json. */
const VISION_HZ = 120;

export class RemoteBrain {
  readonly remote = true;
  readonly fullMode = true;
  learning = false;

  private ws: WebSocket;
  private seq = 0;
  private lastSend = 0;
  private latest: Cmd = RemoteBrain.idle();
  private tel: RemoteTelemetry | null = null;
  private gotActions = false;
  private url: string;
  private closedByUser = false;
  private retryTimer = 0;
  private telTimer = 0;
  /** pending collision pulse, sent with the next state frame */
  private bumpPending = false;

  constructor(
    url: string,
    private stats: () => {
      altitude: number; speed: number; collision: boolean;
      rgb: Float32Array;
    },
    onReady?: () => void,
  ) {
    this.url = url;
    this.ws = this.connect(onReady);
  }

  /** One WebSocket lifetime: open, wire handlers, schedule reconnect. */
  private connect(onReady?: () => void): WebSocket {
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      // declare the embodiment so the brain drives THIS body (rover channels);
      // the learned motor-pool decode is the standard rover profile
      const wanted = new URLSearchParams(location.search).get("profile") || "rover";
      ws.send(JSON.stringify({ type: "hello", profile: wanted }));
      onReady?.();
    };
    ws.onmessage = (ev) => this.onMessage(ev.data);
    ws.onerror = () => { /* surfaced via status() */ };
    ws.onclose = () => {
      if (this.closedByUser) return;
      this.gotActions = false;
      // reconnect with backoff; retry forever (server may be restarting)
      if (!this.retryTimer) {
        this.retryTimer = window.setTimeout(() => {
          this.retryTimer = 0;
          this.ws = this.connect(onReady);
        }, 1200);
      }
    };
    // poll telemetry ~4/s so the HUD tracks the remote circuit
    this.telTimer = window.setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "telemetry" }));
      }
    }, 250);
    return ws;
  }

  static idle(): Cmd {
    return { throttle: 0.3, steer: 0, pools: { drive: 0, steer: 0 } };
  }

  private onMessage(data: unknown): void {
    if (data instanceof ArrayBuffer) {
      if (data.byteLength < 3) return;
      const view = new DataView(data);
      if (view.getUint8(0) !== 10) return;
      const nl = view.getUint16(1, true);
      const names: string[] = JSON.parse(
        new TextDecoder().decode(new Uint8Array(data, 3, nl)));
      const vals: number[] = [];
      for (let i = 0; i < names.length; i++) {
        vals.push(view.getFloat32(3 + nl + i * 4, true));
      }
      const get = (n: string, d: number) => {
        const i = names.indexOf(n);
        return i >= 0 ? vals[i] : d;
      };
      this.latest = {
        throttle: get("throttle", 0.3),
        steer: get("steer", 0),
        // remote server has no legacy pool semantics; mirror channels so
        // HUD traces keep moving
        pools: { drive: get("throttle", 0.3), steer: get("steer", 0) },
      };
      this.gotActions = true;
      return;
    }
    try {
      const obj = JSON.parse(String(data));
      if (obj.type === "hello" && obj.circuit) {
        this.tel = {
          neurons: obj.circuit.neurons, edges: obj.circuit.edges,
          rates: {}, dopa: 0, dnSteer: 0, memEdited: 0,
          learning: false, simMs: 0,
        };
      } else if (obj.type === "telemetry" && obj.rates) {
        this.tel = {
          neurons: obj.neurons, edges: obj.edges, rates: obj.rates,
          dopa: obj.dopa, dnSteer: obj.dnSteer, memEdited: obj.memEdited,
          learning: obj.learning, simMs: obj.simMs,
        };
        this.learning = !!obj.learning;
      }
    } catch { /* ignore malformed */ }
  }

  /** Called each frame; streams eye + state at VISION_HZ, returns latest command. */
  step(dt: number): Cmd {
    this.lastSend += dt;
    // stream the eye at VISION_HZ (matches sensors.visionHz in the profile;
    // 120 Hz keeps the EMD correlators well above frame rate)
    if (this.lastSend >= 1 / VISION_HZ && this.ws.readyState === WebSocket.OPEN) {
      this.lastSend = 0;
      const s = this.stats();
      this.sendEye(0, s.rgb); // single forward eye: everything is eye 0
      // body state: ONLY what a real rover can sense — chassis height
      // (suspension), wheel odometry, contact events (bumper). No clearance:
      // the brain estimates obstacle distance itself from optic flow.
      this.ws.send(JSON.stringify({
        type: "state", altitude: s.altitude, speed: s.speed,
        vy: 0, collision: s.collision || this.bumpPending,
      }));
      this.bumpPending = false;
      this.seq++;
    }
    return this.latest;
  }

  /** Flag a bumper contact; sent with the next state frame. */
  notifyBump(): void {
    this.bumpPending = true;
  }

  /** Artificial reward / punishment through the REST API.
   * v > 0 rewards (LTP window), v < 0 punishes; the server scales by
   * rewardGain and the circuit's own dopamine decides whether to teach. */
  async pulseReward(v: number): Promise<void> {
    if (!this.connected) return;
    try {
      const base = this.url.replace(/^ws/, "http").replace(/\/stream$/, "");
      await fetch(`${base}/reward/${v}`, { method: "POST" });
    } catch { /* offline; surfaced via status() */ }
  }

  /** Stream one eye frame ([1][eye][w][h] + RGB; channels implied 3). */
  private sendEye(eye: number, arr: Float32Array): void {
    const w = RENDER_W, h = RENDER_H;
    const pkt = new Uint8Array(6 + w * h * 3);
    pkt[0] = 1; // MSG_FRAME
    pkt[1] = eye;
    pkt[2] = w & 0xff; pkt[3] = (w >> 8) & 0xff;
    pkt[4] = h & 0xff; pkt[5] = (h >> 8) & 0xff;
    for (let i = 0; i < w * h * 3; i++) {
      const v = arr[i];
      pkt[6 + i] = v <= 0 ? 0 : v >= 1 ? 255 : (v * 255) | 0;
    }
    this.ws.send(pkt.buffer);
  }

  get seqSent(): number { return this.seq; }
  get connected(): boolean { return this.ws.readyState === WebSocket.OPEN; }
  get hasActions(): boolean { return this.gotActions; }

  status(): string {
    if (this.ws.readyState === WebSocket.CONNECTING) return "connecting…";
    if (this.ws.readyState === WebSocket.OPEN) {
      return this.gotActions ? "streaming" : "handshaking…";
    }
    return "offline";
  }

  telemetry(): RemoteTelemetry | null { return this.tel; }

  // ---- control surface ----
  setLearning(on: boolean): void {
    this.learning = on;
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "control", learning: on }));
    }
  }

  wipeMemory(): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "control", wipe: true }));
    }
  }

  dispose(): void {
    this.closedByUser = true;
    if (this.telTimer) window.clearInterval(this.telTimer);
    this.telTimer = 0;
    if (this.retryTimer) window.clearTimeout(this.retryTimer);
    this.retryTimer = 0;
    try { this.ws.close(); } catch { /* noop */ }
  }
}
