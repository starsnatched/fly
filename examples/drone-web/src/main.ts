import * as THREE from "three";
import { RemoteBrain } from "./remotebrain";
import { FlyVision } from "./vision";
import { Drone, WORLD } from "./drone";
import { buildScene, checkCollision, resolveCollision, clearanceAhead } from "./world";

/**
 * FlyBrain FPV client: a pure embodiment.
 *
 * The browser renders the world, captures both compound eyes at full
 * resolution, streams them (plus proprioception) to the C connectome
 * server, and applies the actuator channels it receives back. All
 * cognition lives behind the API — see cengine/README.md.
 */

const BRAIN_SERVER = "ws://localhost:8787/stream";

let renderer: THREE.WebGLRenderer;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let vision: FlyVision;
let brain: RemoteBrain;
let drone: Drone;
let manual = false;
const keys = new Set<string>();
let started = false;

const el = (id: string) => document.getElementById(id)!;
const bar = (id: string, v: number) => {
  (el(id) as HTMLElement).style.width = `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%`;
};

/** State the remote brain needs each frame (eye RGB + proprioception).
 * ONLY drone-obtainable data: cameras, barometer altitude, IMU climb rate,
 * bumper contact events. Clearance is the brain's own optic-flow estimate. */
function remoteStats(): {
  altitude: number; vy: number; collision: boolean;
  rgbL: Float32Array; rgbR: Float32Array;
} {
  const rgb = vision.rgb;
  return {
    altitude: drone.pos.y - WORLD.groundY,
    vy: drone.vel.y,
    collision: false,
    rgbL: rgb.L, rgbR: rgb.R, // stereo pair: both eyes streamed
  };
}

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

  vision = new FlyVision(renderer);

  // The browser is an embodiment, nothing more: it always streams to the
  // brain server and waits until the backend answers.
  brain = new RemoteBrain(BRAIN_SERVER, remoteStats, () => {
    el("mode-badge").textContent = "REMOTE BRAIN";
    updateCircuitInfo();
  });

  window.addEventListener("keydown", (e) => {
    keys.add(e.key.toLowerCase());
    if (e.key.toLowerCase() === "c") manual = !manual;
    if (e.key.toLowerCase() === "r") drone.respawn();
    if (e.key.toLowerCase() === "l") {
      brain.setLearning(!brain.learning);
      flashBadge(brain.learning ? "LEARNING ON" : "LEARNING OFF");
    }
    if (e.key.toLowerCase() === "m") {
      brain.wipeMemory();
      flashBadge("MEMORY WIPED");
    }
    if (e.key.toLowerCase() === "u") {
      void brain.pulseReward(1.0);
      flashBadge("REWARD +1.0");
    }
    if (e.key.toLowerCase() === "j") {
      void brain.pulseReward(-1.0);
      flashBadge("PUNISH −1.0");
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

function flashBadge(msg: string): void {
  el("mode-badge").textContent = msg;
  setTimeout(() => {
    el("mode-badge").textContent = manual ? "MANUAL OVERRIDE" : "REMOTE BRAIN";
  }, 2500);
}

function updateCircuitInfo(): void {
  const t = brain.telemetry();
  el("circuit-info").textContent = t
    ? `REMOTE FULL BRAIN: ${t.neurons.toLocaleString()} neurons / ` +
      `${((t.edges ?? 0) / 1e6).toFixed(1)}M synapses @ C engine backend`
    : "REMOTE BRAIN: C connectome server";
  el("mode-badge").textContent = manual ? "MANUAL OVERRIDE" : "REMOTE BRAIN";
}

let last = performance.now();
let launchTime = performance.now();
const traceHist: { avert: number; steer: number; lift: number; thr: number; dopa: number; steerLearn: number }[] = [];

function loop(): void {
  requestAnimationFrame(loop);
  const now = performance.now();
  const dt = Math.min((now - last) / 1000, 0.06);
  last = now;

  // manual nudges
  drone.nudge.fwd = (keys.has("w") ? 1 : 0) - (keys.has("s") ? 1 : 0);
  drone.nudge.right = (keys.has("d") ? 1 : 0) - (keys.has("a") ? 1 : 0);

  // vision -> brain: full-resolution per-eye RGB + proprioception
  vision.update(scene, drone.pos, drone.yaw, drone.pitch, drone.roll, dt);
  const speed = Number.isFinite(drone.vel.length()) ? drone.vel.length() : 0;
  const clr = clearanceAhead(drone.pos, drone.yaw);

  // wait for the backend before flying: no server, no commands
  if (!brain.hasActions) {
    el("circuit-info").textContent =
      `waiting for brain server at ${BRAIN_SERVER} — run: ` +
      `docker compose up  (cengine)`;
    el("pop-stats").innerHTML =
      `<div style="color:#ffd166">○ brain offline — retrying…</div>`;
    renderEyePanels();
    render();
    return;
  }

  const cmd = brain.step(dt);
  // command sanitizer: the physics must never see a non-finite command
  cmd.throttle = Number.isFinite(cmd.throttle) ? Math.max(0, Math.min(1, cmd.throttle)) : 0.29;
  cmd.pitch = Number.isFinite(cmd.pitch) ? Math.max(-1, Math.min(1, cmd.pitch)) : 0;
  cmd.roll = Number.isFinite(cmd.roll) ? Math.max(-1, Math.min(1, cmd.roll)) : 0;
  cmd.yaw = Number.isFinite(cmd.yaw) ? Math.max(-1, Math.min(1, cmd.yaw)) : 0;

  (window as unknown as { __flybrainDebug: Record<string, number> }).__flybrainDebug =
    remoteDebug(brain.telemetry());
  (window as unknown as { __flyBrain: RemoteBrain }).__flyBrain = brain;
  (window as unknown as { __flyVision: FlyVision }).__flyVision = vision;
  (window as unknown as { __flyScene: THREE.Scene }).__flyScene = scene;
  (window as unknown as { __flyDrone: Drone }).__flyDrone = drone;

  if (manual) {
    cmd.throttle = 0.29;
    cmd.pitch = -0.2 + (keys.has("i") ? -0.6 : 0) + (keys.has("k") ? 0.6 : 0);
    cmd.roll = drone.nudge.right;
    cmd.yaw = (keys.has("j") ? -0.8 : 0) + (keys.has("l") ? 0.8 : 0);
    if (keys.has("shift")) cmd.throttle += 0.3;
    if (keys.has("control")) cmd.throttle -= 0.3;
  }

  drone.step(cmd, dt);

  // NaN tripwire: if drone state ever goes non-finite (should be impossible
  // with sanitized commands), respawn instead of rendering nothing forever
  if (!Number.isFinite(drone.pos.x + drone.pos.y + drone.pos.z + drone.vel.x + drone.vel.y + drone.vel.z)) {
    console.warn("[drone] non-finite state — respawn");
    drone.respawn();
  }

  // collisions: soft bump + bounce (explore mode - no restarts).
  // Bumps notify the brain so it picks a new heading.
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

  // telemetry bars
  const alt = drone.pos.y - WORLD.groundY;
  bar("bar-alt", alt / WORLD.ceilingY);
  el("val-alt").textContent = `${alt.toFixed(1)}m`;
  bar("bar-spd", speed / 18);
  el("val-spd").textContent = `${speed.toFixed(1)}m/s`;
  bar("bar-thr", cmd.throttle);
  el("val-thr").textContent = `${Math.round(cmd.throttle * 100)}%`;
  bar("bar-clr", 1 - clr / 60);
  el("val-clr").textContent = `${clr.toFixed(0)}m`;
  if (!manual) el("mode-badge").textContent = "REMOTE BRAIN";
  el("mode-badge").style.color = manual ? "#ffb347" : "#7cf7ff";

  updateTelemetry(cmd);
  renderEyePanels();
  drawTraces();
  render();
}

/** Synthesize the debug record from remote telemetry. */
function remoteDebug(t: ReturnType<RemoteBrain["telemetry"]>): Record<string, number> {
  if (!t) return {};
  const r = t.rates;
  return {
    t4Hz: r["T4"] ?? 0, t5Hz: r["T5"] ?? 0,
    lcHz: r["LC"] ?? 0, lplcHz: r["LPLC"] ?? 0,
    kcHz: r["KC"] ?? 0, danHz: r["DAN"] ?? 0, dnHz: r["descending"] ?? 0,
    dopa: t.dopa, dnSteer: t.dnSteer, simSpeed: 1,
  };
}

function updateTelemetry(cmd: ReturnType<RemoteBrain["step"]>): void {
  const t = brain.telemetry();
  const d = t?.rates ?? {};
  const flightTime = (performance.now() - launchTime) / 1000;
  const poolRows =
    `<div style="color:#9dff87">● REMOTE BRAIN · ${brain.status()}</div>` +
    `<div>LPLC <b style="color:#ff7d6b">${(d["LPLC"] ?? 0).toFixed(1)}Hz</b> · LC <b style="color:#ff7d6b">${(d["LC"] ?? 0).toFixed(1)}Hz</b></div>` +
    `<div>T4 <b style="color:#7cf7ff">${(d["T4"] ?? 0).toFixed(1)}Hz</b> · T5 <b style="color:#7cf7ff">${(d["T5"] ?? 0).toFixed(1)}Hz</b></div>` +
    `<div style="opacity:0.8">KC <b style="color:#c792ea">${(d["KC"] ?? 0).toFixed(2)}Hz</b> · DAN <b style="color:#ff4fd8">${(d["DAN"] ?? 0).toFixed(1)}Hz</b> · DN <b style="color:#ffd166">${(d["descending"] ?? 0).toFixed(1)}Hz</b></div>` +
    `<div style="margin-top:4px">memory <b style="color:${t?.learning ? "#1de9a0" : "#888"}">${t?.learning ? "● LEARNING" : "○ idle"}</b> · ` +
    `<b style="color:#ffd166">${(t?.memEdited ?? 0).toLocaleString()}</b> syn edited</div>` +
    `<div style="opacity:0.75">dopa <b style="color:#ff7d6b">${(t?.dopa ?? 0).toFixed(2)}</b> · ` +
    `turn <b style="color:#7cf7ff">${(t?.dnSteer ?? 0).toFixed(2)}</b> · ` +
    `bumps ${drone.bumpCount}</div>`;
  el("pop-stats").innerHTML = poolRows +
    `<div style="opacity:0.6;margin-top:4px">airtime ${flightTime.toFixed(0)}s · bumps ${drone.bumpCount} · ` +
    `pos ${drone.pos.x.toFixed(0)}, ${drone.pos.z.toFixed(0)}m</div>`;
  el("circuit-info").textContent =
    `REMOTE FULL BRAIN: ${(t?.neurons ?? 0).toLocaleString()} neurons / ` +
    `${((t?.edges ?? 0) / 1e6).toFixed(1)}M synapses @ C engine backend`;
  traceHist.push({
    avert: Math.abs(cmd.roll), steer: cmd.pools.steer, lift: cmd.pools.lift,
    thr: cmd.throttle, dopa: t?.dopa ?? 0, steerLearn: t?.dnSteer ?? 0,
  });
  if (traceHist.length > 150) traceHist.shift();
}

function renderEyePanels(): void {
  vision.drawRgbCanvas(el("eyeLrgb") as HTMLCanvasElement, "L");
  vision.drawRgbCanvas(el("eyeRrgb") as HTMLCanvasElement, "R");
}

function render(): void {
  renderer.render(scene, camera);
}

function drawTraces(): void {
  const c = el("traces") as HTMLCanvasElement;
  const ctx = c.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, c.width, c.height);
  const series: [keyof (typeof traceHist)[0], string][] = [
    ["avert", "#ff7d6b"], ["steer", "#7cf7ff"], ["lift", "#9dff87"], ["thr", "#ffd166"],
    ["dopa", "#ff4fd8"], ["steerLearn", "#c792ea"],
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
