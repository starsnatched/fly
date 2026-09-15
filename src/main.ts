import * as THREE from "three";
import type { CircuitPayload } from "./types";
import { FlyBrain } from "./brain";
import { FlyVision } from "./vision";
import { Drone, WORLD } from "./drone";
import { buildScene, checkCollision, clearanceAhead } from "./world";

let renderer: THREE.WebGLRenderer;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let vision: FlyVision;
let brain: FlyBrain;
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

  // load the connectome-derived circuit
  const res = await fetch("/fly-circuit.json");
  const payload = (await res.json()) as CircuitPayload;
  brain = new FlyBrain(payload);
  vision = new FlyVision(renderer);

  el("circuit-info").textContent =
    `${brain.neuronCount.toLocaleString()} neurons / ${brain.edgeCount.toLocaleString()} connections / ` +
    `${Math.round(brain.synapses / 1e6)}M synapses - MaleCNS v1.0 [r3]`;

  window.addEventListener("keydown", (e) => {
    keys.add(e.key.toLowerCase());
    if (e.key.toLowerCase() === "c") manual = !manual;
    if (e.key.toLowerCase() === "r") drone.respawn();
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

  if (manual) {
    cmd.throttle = 0.55 + drone.nudge.fwd * 0;
    cmd.pitch = -0.2 + (keys.has("i") ? -0.6 : 0) + (keys.has("k") ? 0.6 : 0);
    cmd.roll = drone.nudge.right;
    cmd.yaw = (keys.has("j") ? -0.8 : 0) + (keys.has("l") ? 0.8 : 0);
    if (keys.has("shift")) cmd.throttle += 0.3;
    if (keys.has("control")) cmd.throttle -= 0.3;
  }

  drone.step(cmd, dt);

  // collisions
  if (checkCollision(drone.pos)) {
    drone.crash("obstacle");
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
  el("mode-badge").textContent = manual ? "MANUAL OVERRIDE" : "CONNECTOME";
  el("mode-badge").style.color = manual ? "#ffb347" : "#7cf7ff";

  el("pop-stats").innerHTML =
    `<div>avert pool <b style="color:#ff7d6b">${cmd.pools.avert.toFixed(2)}</b></div>` +
    `<div>steer pool <b style="color:#7cf7ff">${cmd.pools.steer >= 0 ? "+" : ""}${cmd.pools.steer.toFixed(2)}</b></div>` +
    `<div>lift pool <b style="color:#9dff87">${cmd.pools.lift.toFixed(2)}</b></div>` +
    `<div style="opacity:0.6;margin-top:4px">crashes: ${drone.crashCount} · z: ${drone.pos.z.toFixed(0)}m</div>`;

  traceHist.push({ avert: cmd.pools.avert, steer: cmd.pools.steer, lift: cmd.pools.lift, thr: cmd.throttle });
  if (traceHist.length > 150) traceHist.shift();
  drawTraces();

  vision.drawEyeCanvas(el("eyeL") as HTMLCanvasElement, "L");
  vision.drawEyeCanvas(el("eyeR") as HTMLCanvasElement, "R");

  render();
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
