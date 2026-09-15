import * as THREE from "three";
import type { CircuitPayload, LifPayload } from "./types";
import { FlyBrain } from "./brain";
import { LifBrain } from "./lifbrain";
import { FlyVision } from "./vision";
import { Drone, WORLD } from "./drone";
import { buildScene, checkCollision, resolveCollision, clearanceAhead } from "./world";

let renderer: THREE.WebGLRenderer;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let vision: FlyVision;
let rateBrain: FlyBrain;
let lifBrain: LifBrain | null = null;
let useLif = false;
let brain: FlyBrain | LifBrain;
let drone: Drone;
let manual = false;
const keys = new Set<string>();
let started = false;

const el = (id: string) => document.getElementById(id)!;
const bar = (id: string, v: number) => {
  (el(id) as HTMLElement).style.width = `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%`;
};

async function init(): Promise<void> {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  el("app").appendChild(renderer.domElement);

  scene = buildScene();
  camera = new THREE.PerspectiveCamera(
    95, window.innerWidth / window.innerHeight, 0.1, 700,
  );

  drone = new Drone();
  drone.pos.set(0, 4, 20);
  drone.yaw = 0; // body -Z = world -Z: face down the course
  void WORLD;

  // load the connectome-derived circuits (rate + spiking)
  const res = await fetch("/fly-circuit.json");
  const payload = (await res.json()) as CircuitPayload;
  rateBrain = new FlyBrain(payload);
  brain = rateBrain;
  vision = new FlyVision(renderer);

  // LIF circuit loads in the background (adds ~5 MB)
  fetch("/fly-lif.json")
    .then((r) => r.json() as Promise<LifPayload>)
    .then((lp) => {
      lifBrain = new LifBrain(lp);
      el("lif-status").textContent = `LIF ready (B to switch)`;
      updateCircuitInfo();
    })
    .catch(() => {
      el("lif-status").textContent = "LIF unavailable";
    });

  updateCircuitInfo();

  window.addEventListener("keydown", (e) => {
    keys.add(e.key.toLowerCase());
    if (e.key.toLowerCase() === "c") manual = !manual;
    if (e.key.toLowerCase() === "r") drone.respawn();
    if (e.key.toLowerCase() === "b" && lifBrain) {
      useLif = !useLif;
      brain = useLif ? lifBrain : rateBrain;
      updateCircuitInfo();
    }
  });
  window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  started = true;
  requestAnimationFrame(loop);
}

let last = performance.now();
let launchTime = performance.now();
const traceHist: { avert: number; steer: number; lift: number; thr: number }[] = [];

function loop(now: number): void {
  requestAnimationFrame(loop);
  const dt = Math.min((now - last) / 1000, 0.06);
  last = now;
  if (!drone.alive) {
    render();
    return;
  }

  // manual nudges
  drone.nudge.fwd = (keys.has("w") ? 1 : 0) - (keys.has("s") ? 1 : 0);
  drone.nudge.right = (keys.has("d") ? 1 : 0) - (keys.has("a") ? 1 : 0);

  // vision -> brain
  vision.update(scene, drone.pos, drone.yaw, drone.pitch, drone.roll, dt);
  const stats = vision.quadrantStats();
  const speed = drone.vel.length();
  const cmd = brain.step({
    stats,
    altitude: drone.pos.y - WORLD.groundY,
    vy: drone.vel.y,
    heading: drone.yaw,
    speed,
    dt,
  });
  (window as unknown as { __flybrainDebug: Record<string, number> }).__flybrainDebug = brain.debug;
  (window as unknown as { __lifBrain: LifBrain | null }).__lifBrain = lifBrain;

  if (manual) {
    cmd.throttle = 0.55 + drone.nudge.fwd * 0;
    cmd.pitch = -0.2 + (keys.has("i") ? -0.6 : 0) + (keys.has("k") ? 0.6 : 0);
    cmd.roll = drone.nudge.right;
    cmd.yaw = (keys.has("j") ? -0.8 : 0) + (keys.has("l") ? 0.8 : 0);
    if (keys.has("shift")) cmd.throttle += 0.3;
    if (keys.has("control")) cmd.throttle -= 0.3;
  }

  drone.step(cmd, dt);

  // collisions: soft bump + bounce (explore mode - no restarts).
  // Escape-saccade cooldown bumps the brain so it picks a new heading.
  if (checkCollision(drone.pos)) {
    if (resolveCollision(drone.pos, drone.vel)) {
      drone.bumpCount++;
      brain.notifyBump();
    }
  }

  // keep inside the world bounds (perimeter walls): slide along them
  const LIM = 480;
  if (Math.abs(drone.pos.x) > LIM) {
    drone.pos.x = Math.sign(drone.pos.x) * LIM;
    drone.vel.x *= -0.5;
    brain.notifyBump();
  }
  if (Math.abs(drone.pos.z) > LIM) {
    drone.pos.z = Math.sign(drone.pos.z) * LIM;
    drone.vel.z *= -0.5;
    brain.notifyBump();
  }

  // chase camera
  const back = new THREE.Vector3(0, 0, 1).applyEuler(
    new THREE.Euler(drone.pitch, drone.yaw, drone.roll, "YXZ"),
  );
  camera.position.copy(drone.pos).addScaledVector(back, 6.5).add(new THREE.Vector3(0, 2.2, 0));
  const look = drone.pos.clone().addScaledVector(
    new THREE.Vector3(0, 0, -1).applyEuler(
      new THREE.Euler(drone.pitch, drone.yaw, drone.roll, "YXZ"),
    ), 12,
  );
  camera.lookAt(look);

  // telemetry
  const alt = drone.pos.y - WORLD.groundY;
  bar("bar-alt", alt / WORLD.ceilingY);
  el("val-alt").textContent = `${alt.toFixed(1)}m`;
  bar("bar-spd", speed / 18);
  el("val-spd").textContent = `${speed.toFixed(1)}m/s`;
  bar("bar-thr", cmd.throttle);
  el("val-thr").textContent = `${Math.round(cmd.throttle * 100)}%`;
  const clr = clearanceAhead(drone.pos, drone.yaw);
  bar("bar-clr", 1 - clr / 60);
  el("val-clr").textContent = `${clr.toFixed(0)}m`;
  el("mode-badge").textContent = manual ? "MANUAL OVERRIDE" : "EXPLORE · CONNECTOME";
  el("mode-badge").style.color = manual ? "#ffb347" : "#7cf7ff";

  const flightTime = (performance.now() - launchTime) / 1000;
  let poolRows: string;
  if (useLif && lifBrain) {
    const net = lifBrain.network;
    const gabaPct = Math.round((100 * net.inhibCount) / net.N);
    const topGaba = net.populations
      .map((p, i) => ({ p, f: net.gabaFraction[i] }))
      .sort((a, b) => b.f - a.f)[0];
    poolRows =
      `<div>spikes/frame <b style="color:#ffd166">${lifBrain.debug.spikesThisFrame ?? 0}</b></div>` +
      `<div>LPLC <b style="color:#ff7d6b">${(lifBrain.debug.lplcHz ?? 0).toFixed(1)}Hz</b> · LC <b style="color:#ff7d6b">${(lifBrain.debug.lcHz ?? 0).toFixed(1)}Hz</b></div>` +
      `<div>T4 <b style="color:#7cf7ff">${(lifBrain.debug.t4Hz ?? 0).toFixed(1)}Hz</b> · T5 <b style="color:#7cf7ff">${(lifBrain.debug.t5Hz ?? 0).toFixed(1)}Hz</b></div>` +
      `<div>avert <b style="color:#ff7d6b">${cmd.pools.avert.toFixed(2)}</b> · lift <b style="color:#9dff87">${cmd.pools.lift.toFixed(2)}</b></div>` +
      `<div style="opacity:0.75;margin-top:4px">GABAergic <b style="color:#c792ea">${gabaPct}%</b> (${net.inhibCount.toLocaleString()}) · peak ${topGaba.p} ${Math.round(100 * topGaba.f)}%</div>` +
      `<div style="opacity:0.5">NT conf ${(net.meanNtConf * 100).toFixed(0)}% (45.7M T-bars)</div>`;
  } else {
    poolRows =
      `<div>avert pool <b style="color:#ff7d6b">${cmd.pools.avert.toFixed(2)}</b></div>` +
      `<div>steer pool <b style="color:#7cf7ff">${cmd.pools.steer >= 0 ? "+" : ""}${cmd.pools.steer.toFixed(2)}</b></div>` +
      `<div>lift pool <b style="color:#9dff87">${cmd.pools.lift.toFixed(2)}</b></div>`;
  }
  el("pop-stats").innerHTML = poolRows +
    `<div style="opacity:0.6;margin-top:4px">airtime ${flightTime.toFixed(0)}s · bumps ${drone.bumpCount} · ` +
    `pos ${drone.pos.x.toFixed(0)}, ${drone.pos.z.toFixed(0)}m${lifBrain ? " · [B] brain" : ""}</div>`;

  traceHist.push({ avert: cmd.pools.avert, steer: cmd.pools.steer, lift: cmd.pools.lift, thr: cmd.throttle });
  if (traceHist.length > 150) traceHist.shift();
  drawTraces();
  drawRaster();

  vision.drawEyeCanvas(el("eyeL") as HTMLCanvasElement, "L");
  vision.drawEyeCanvas(el("eyeR") as HTMLCanvasElement, "R");

  render();
}

function updateCircuitInfo(): void {
  const mode = useLif ? "SPIKING LIF" : "RATE";
  const syn = brain.synapses >= 1e6
    ? `${Math.round(brain.synapses / 1e6)}M`
    : `${Math.round(brain.synapses / 1e3)}k`;
  el("circuit-info").textContent =
    `${mode}: ${brain.neuronCount.toLocaleString()} neurons / ${brain.edgeCount.toLocaleString()} connections / ` +
    `${syn} synapses - MaleCNS v1.0 [r4]`;
  el("mode-badge").textContent = manual ? "MANUAL OVERRIDE" : (useLif ? "EXPLORE · LIF" : "EXPLORE · CONNECTOME");
}

function render(): void {
  renderer.render(scene, camera);
}

// ---- spike raster ----
interface RasterRow { popIdx: number; neurons: number[]; spikes: boolean[][]; }
const RASTER_POPS = ["T4", "T5", "LC", "LPLC", "TmY", "descending"];
const RASTER_N = 8;
const RASTER_COLS = 240;
let rasterRows: RasterRow[] | null = null;
let rasterCol = 0;

function ensureRaster(): void {
  if (rasterRows || !useLif || !lifBrain) return;
  const net = lifBrain.network;
  rasterRows = RASTER_POPS.map((p) => {
    const popIdx = net.populations.indexOf(p);
    const neurons = popIdx >= 0
      ? net.indicesOfPop(p, RASTER_N)
      : [];
    while (neurons.length < RASTER_N) neurons.push(-1);
    return { popIdx, neurons, spikes: Array.from({ length: RASTER_N }, () => new Array(RASTER_COLS).fill(false)) };
  });
}

function drawRaster(): void {
  ensureRaster();
  const c = el("raster") as HTMLCanvasElement;
  const ctx = c.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = "rgba(4, 8, 10, 0.85)";
  ctx.fillRect(0, 0, c.width, c.height);
  if (!rasterRows || !useLif || !lifBrain) {
    ctx.fillStyle = "rgba(150,190,170,0.4)";
    ctx.font = "11px monospace";
    ctx.fillText("switch to LIF brain with [B] to see spikes", 12, c.height / 2);
    return;
  }
  const net = lifBrain.network;
  // record this frame's spikes
  for (const row of rasterRows) {
    for (let k = 0; k < row.neurons.length; k++) {
      const idx = row.neurons[k];
      row.spikes[k][rasterCol] = idx >= 0 && net.spiked(idx);
    }
  }
  rasterCol = (rasterCol + 1) % RASTER_COLS;

  const rowH = c.height / rasterRows.length;
  RASTER_POPS.forEach((p, r) => {
    const y0 = r * rowH;
    ctx.fillStyle = "rgba(150,190,170,0.55)";
    ctx.font = "9px monospace";
    ctx.fillText(p, 4, y0 + rowH / 2 + 3);
    const row = rasterRows![r];
    for (let k = 0; k < RASTER_N; k++) {
      const yy = y0 + (k / RASTER_N) * rowH;
      for (let ci = 0; ci < RASTER_COLS; ci++) {
        if (row.spikes[k][ci]) {
          const x = (ci / RASTER_COLS) * c.width;
          const fade = ci === (rasterCol - 1 + RASTER_COLS) % RASTER_COLS ? 1 : 0.75;
          ctx.fillStyle = `rgba(60, 255, 170, ${fade})`;
          ctx.fillRect(x + 22, yy, 1.4, Math.max(1.2, rowH / RASTER_N - 0.4));
        }
      }
    }
    ctx.strokeStyle = "rgba(60, 255, 160, 0.08)";
    ctx.beginPath();
    ctx.moveTo(0, y0);
    ctx.lineTo(c.width, y0);
    ctx.stroke();
  });
  // sweep line
  const sx = (rasterCol / RASTER_COLS) * c.width;
  ctx.strokeStyle = "rgba(255, 209, 102, 0.5)";
  ctx.beginPath();
  ctx.moveTo(sx + 22, 0);
  ctx.lineTo(sx + 22, c.height);
  ctx.stroke();
}

function drawTraces(): void {
  const c = el("traces") as HTMLCanvasElement;
  const ctx = c.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, c.width, c.height);
  const series: [keyof (typeof traceHist)[0], string][] = [
    ["avert", "#ff7d6b"], ["steer", "#7cf7ff"], ["lift", "#9dff87"], ["thr", "#ffd166"],
  ];
  for (const [key, color] of series) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    traceHist.forEach((t, i) => {
      const x = (i / 149) * c.width;
      const y = c.height / 2 - t[key] * (c.height / 2 - 6);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

el("startup").addEventListener("click", () => {
  if (started) return;
  el("startup").style.display = "none";
  init().catch((e) => {
    el("startup").innerHTML = `<div class="card"><h1>Failed to start</h1><pre>${String(e)}</pre></div>`;
  });
});
