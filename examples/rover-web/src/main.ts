import * as THREE from "three";
import { RemoteBrain } from "./remotebrain";
import { RoverVision } from "./vision";
import { Rover } from "./rover";
import {
  buildScene, checkCollision, resolveCollision, clearanceAhead,
  nearestPitAhead, pitAt, WORLD, type CourseKind,
} from "./world";

/**
 * FlyBrain rover client: a pure embodiment.
 *
 * The browser renders the world, captures the forward eye at full
 * resolution, streams it (plus wheel odometry) to the C connectome server,
 * and applies the actuator channels it receives back. All cognition lives
 * behind the API — see cengine/README.md. Run the brain with the rover
 * profile:  ./flybrain-server --config config/flybrain.json --profile rover
 */

const BRAIN_SERVER = "ws://localhost:8787/stream";

let renderer: THREE.WebGLRenderer;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let vision: RoverVision;
let brain: RemoteBrain;
let rover: Rover;
let manual = false;
const keys = new Set<string>();
let started = false;
let gapMode = false; // ?course=gaps

const el = (id: string) => document.getElementById(id)!;
const bar = (id: string, v: number) => {
  (el(id) as HTMLElement).style.width = `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%`;
};

/** State the remote brain needs each frame (eye RGB + proprioception).
 * ONLY rover-obtainable data: the camera, wheel odometry, bumper contacts.
 * Clearance is the brain's own optic-flow estimate. */
function remoteStats(): {
  altitude: number; speed: number; collision: boolean;
  rgb: Float32Array;
} {
  return {
    altitude: rover.pos.y,
    speed: Math.abs(rover.forwardSpeed),
    collision: false,
    rgb: vision.frame,
  };
}

async function init(): Promise<void> {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  el("app").appendChild(renderer.domElement);

  // ?course=gaps opens pits along the route; default is the rock desert
  const course: CourseKind =
    new URLSearchParams(window.location.search).get("course") === "gaps"
      ? "gaps"
      : "desert";
  gapMode = course === "gaps";
  scene = buildScene(course);
  camera = new THREE.PerspectiveCamera(
    62, window.innerWidth / window.innerHeight, 0.1, 700,
  );

  rover = new Rover();
  rover.attach(scene);

  vision = new RoverVision(renderer);

  // The browser is an embodiment, nothing more: it always streams to the
  // brain server and waits until the backend answers.
  brain = new RemoteBrain(BRAIN_SERVER, remoteStats, () => {
    el("mode-badge").textContent = "REMOTE BRAIN";
    updateCircuitInfo();
  });

  window.addEventListener("keydown", (e) => {
    keys.add(e.key.toLowerCase());
    if (e.key.toLowerCase() === "c") manual = !manual;
    if (e.key.toLowerCase() === "r") rover.respawn();
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
  // Headless/occluded tabs suspend requestAnimationFrame entirely; a real
  // user who backgrounds the tab gets the same. Keep streaming to the brain
  // with a timer watchdog that takes over whenever rAF stalls. loop() is
  // reentrancy-guarded, so the two drivers coexist safely; all timing is
  // wall-clock dt, so an extra driver only changes the sampling rate.
  window.setInterval(() => {
    if (performance.now() - lastLoopAt > 120) loop();
  }, 50);
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
let lastLoopAt = 0; // watchdog: last time the loop actually ran
const launchTime = performance.now();
const traceHist: { steer: number; drive: number; thr: number; spd: number; dopa: number; steerLearn: number; cleared: number; pit: number }[] = [];

function loop(): void {
  lastLoopAt = performance.now();
  requestAnimationFrame(loop);
  const now = performance.now();
  const dt = Math.min((now - last) / 1000, 0.06);
  last = now;

  // manual nudges
  rover.nudge.fwd = (keys.has("w") ? 1 : 0) - (keys.has("s") ? 1 : 0);
  rover.nudge.steer = (keys.has("a") ? 1 : 0) - (keys.has("d") ? 1 : 0);

  // vision -> brain: full-resolution forward-eye RGB + proprioception
  vision.update(scene, rover.pos, rover.yaw, rover.pitch, rover.roll);
  const speed = Math.abs(rover.forwardSpeed);
  const clr = clearanceAhead(rover.pos, rover.yaw);

  // wait for the backend before rolling: no server, no commands
  if (!brain.hasActions) {
    el("circuit-info").textContent =
      `waiting for brain server at ${BRAIN_SERVER} — run: ` +
      `./flybrain-server --profile rover  (cengine)`;
    el("pop-stats").innerHTML =
      `<div style="color:#ffd166">○ brain offline — retrying…</div>`;
    renderEyePanel();
    render();
    return;
  }

  const cmd = brain.step(dt);
  // command sanitizer: the physics must never see a non-finite command
  cmd.throttle = Number.isFinite(cmd.throttle) ? Math.max(0, Math.min(1, cmd.throttle)) : 0.3;
  cmd.steer = Number.isFinite(cmd.steer) ? Math.max(-1, Math.min(1, cmd.steer)) : 0;

  // expose for console debugging even before the first action frame
  (window as unknown as { __flyBrain: RemoteBrain }).__flyBrain = brain;
  (window as unknown as { __flyVision: RoverVision }).__flyVision = vision;
  (window as unknown as { __flyRover: Rover }).__flyRover = rover;
  (window as unknown as { __flyScene: THREE.Scene }).__flyScene = scene;

  (window as unknown as { __flybrainDebug: Record<string, number> }).__flybrainDebug =
    remoteDebug(brain.telemetry());
  (window as unknown as { __flyBrain: RemoteBrain }).__flyBrain = brain;
  (window as unknown as { __flyVision: RoverVision }).__flyVision = vision;
  (window as unknown as { __flyRover: Rover }).__flyRover = rover;
  (window as unknown as { __flyScene: THREE.Scene }).__flyScene = scene;

  if (manual) {
    cmd.throttle = rover.nudge.fwd > 0 ? 1.0 : rover.nudge.fwd < 0 ? 0.0 : 0.62;
    cmd.steer = rover.nudge.steer * 0.9;
  }

  rover.step(cmd, dt);

  // NaN tripwire: if rover state ever goes non-finite (should be impossible
  // with sanitized commands), respawn instead of rendering nothing forever
  if (!Number.isFinite(rover.pos.x + rover.pos.z + rover.vel.x + rover.vel.z)) {
    console.warn("[rover] non-finite state — respawn");
    rover.respawn();
  }

  // gap course: falling punishes and resets the trial; clearing one rewards.
  // The brain only ever sees its reward stream + the eye — no scripted steer.
  if (gapMode) {
    if (!rover.paused && pitAt(rover.pos.x, rover.pos.z)) {
      rover.falls++;
      rover.lastPit = { x: rover.pos.x, z: rover.pos.z };
      void brain.pulseReward(-1.0);
      // soft reset: back before the same gap, facing it again
      rover.pos.set(
        THREE.MathUtils.clamp(rover.lastPit.x + (Math.random() - 0.5) * 4, -24, 24),
        0, rover.lastPit.z + 8);
      rover.vel.set(0, 0, 0);
      rover.yaw = 0;
      rover.pitch = rover.roll = 0;
      rover.startPause();
    } else {
      // crossing detection: engage the pit ahead, count when passed
      const engaged = nearestPitAhead(rover.pos, rover.yaw);
      if (engaged && engaged.dist < 3) rover.pendingPit = engaged.pit;
      if (rover.pendingPit) {
        const p = rover.pendingPit;
        if (rover.pos.z < p.z - p.l / 2 - 1) {
          rover.crossings++;
          rover.clearedFlag = true;
          rover.pendingPit = null;
          void brain.pulseReward(0.6);
        } else if (rover.pos.z > p.z + p.l / 2 + 14) {
          rover.pendingPit = null; // drifted away without crossing
        }
      }
    }
  }

  // collisions: soft bump + bounce (explore mode - no restarts).
  // Bumps notify the brain so it picks a new heading.
  if (checkCollision(rover.pos)) {
    if (resolveCollision(rover.pos, rover.vel)) {
      rover.bumpCount++;
      brain.notifyBump();
    }
  }

  // keep inside the world bounds: slide along the perimeter
  const LIM = WORLD.limit;
  if (Math.abs(rover.pos.x) > LIM) {
    rover.pos.x = Math.sign(rover.pos.x) * LIM;
    rover.vel.x *= -0.5;
    brain.notifyBump();
  }
  if (Math.abs(rover.pos.z) > LIM) {
    rover.pos.z = Math.sign(rover.pos.z) * LIM;
    rover.vel.z *= -0.5;
    brain.notifyBump();
  }

  rover.syncMesh();

  // chase camera
  const back = new THREE.Vector3(0, 0, 1).applyEuler(new THREE.Euler(0, rover.yaw, 0, "YXZ"));
  const camTarget = rover.pos.clone().addScaledVector(back, 5.5).add(new THREE.Vector3(0, 3.0, 0));
  camera.position.lerp(camTarget, Math.min(1, dt * 4));
  const look = rover.pos.clone().addScaledVector(
    new THREE.Vector3(0, 0, -1).applyEuler(new THREE.Euler(0, rover.yaw, 0, "YXZ")), 10,
  );
  camera.lookAt(look);

  // telemetry bars
  bar("bar-spd", speed / 4);
  el("val-spd").textContent = `${speed.toFixed(1)}m/s`;
  bar("bar-thr", cmd.throttle);
  el("val-thr").textContent = `${Math.round(cmd.throttle * 100)}%`;
  bar("bar-str", (cmd.steer + 1) / 2);
  el("val-str").textContent = cmd.steer.toFixed(2);
  bar("bar-clr", 1 - clr / 90);
  el("val-clr").textContent = `${clr.toFixed(0)}m`;
  if (!manual) el("mode-badge").textContent = "REMOTE BRAIN";
  el("mode-badge").style.color = manual ? "#ffb347" : "#ffd166";

  updateTelemetry(cmd);
  renderEyePanel();
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
  const runTime = (performance.now() - launchTime) / 1000;
  const poolRows =
    `<div style="color:#9dff87">● REMOTE BRAIN · ${brain.status()}</div>` +
    `<div>LPLC <b style="color:#ff7d6b">${(d["LPLC"] ?? 0).toFixed(1)}Hz</b> · LC <b style="color:#ff7d6b">${(d["LC"] ?? 0).toFixed(1)}Hz</b></div>` +
    `<div>T4 <b style="color:#7cf7ff">${(d["T4"] ?? 0).toFixed(1)}Hz</b> · T5 <b style="color:#7cf7ff">${(d["T5"] ?? 0).toFixed(1)}Hz</b></div>` +
    `<div style="opacity:0.8">KC <b style="color:#c792ea">${(d["KC"] ?? 0).toFixed(2)}Hz</b> · DAN <b style="color:#ff4fd8">${(d["DAN"] ?? 0).toFixed(1)}Hz</b> · DN <b style="color:#ffd166">${(d["descending"] ?? 0).toFixed(1)}Hz</b></div>` +
    `<div style="margin-top:4px">memory <b style="color:${t?.learning ? "#1de9a0" : "#888"}">${t?.learning ? "● LEARNING" : "○ idle"}</b> · ` +
    `<b style="color:#ffd166">${(t?.memEdited ?? 0).toLocaleString()}</b> syn edited</div>` +
    `<div style="opacity:0.75">dopa <b style="color:#ff7d6b">${(t?.dopa ?? 0).toFixed(2)}</b> · ` +
    `turn <b style="color:#7cf7ff">${(t?.dnSteer ?? 0).toFixed(2)}</b> · ` +
    `bumps ${rover.bumpCount}` +
    (gapMode ? ` · falls ${rover.falls} · crossings ${rover.crossings}` : "") +
    `</div>`;
  el("pop-stats").innerHTML = poolRows +
    `<div style="opacity:0.6;margin-top:4px">runtime ${runTime.toFixed(0)}s · odometer ${rover.odometer.toFixed(0)}m · ` +
    `pos ${rover.pos.x.toFixed(0)}, ${rover.pos.z.toFixed(0)}m</div>`;
  el("circuit-info").textContent =
    `REMOTE FULL BRAIN: ${(t?.neurons ?? 0).toLocaleString()} neurons / ` +
    `${((t?.edges ?? 0) / 1e6).toFixed(1)}M synapses @ C engine backend`;
  traceHist.push({
    steer: (cmd.steer + 1) / 2, drive: cmd.pools.drive, thr: cmd.throttle,
    spd: speed01(), dopa: (t?.dopa ?? 0), steerLearn: (t?.dnSteer ?? 0),
    cleared: rover.clearedFlag ? 1 : 0, pit: rover.paused ? 1 : 0,
  });
  rover.clearedFlag = false;
  if (traceHist.length > 150) traceHist.shift();
}

function speed01(): number {
  return Math.min(1, Math.abs(rover.forwardSpeed) / 4);
}

function renderEyePanel(): void {
  vision.drawRgbCanvas(el("eyeFwd") as HTMLCanvasElement);
}

function render(): void {
  renderer.render(scene, camera);
}

function drawTraces(): void {
  const c = el("traces") as HTMLCanvasElement;
  const ctx = c.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, c.width, c.height);
  // draw dopa around its own midline (it is a small prediction-error signal)
  const mid = c.height * 0.75;
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(c.width, mid);
  ctx.stroke();
  const series: [keyof (typeof traceHist)[0], string, number][] = [
    ["steer", "#7cf7ff", c.height / 4],
    ["thr", "#ffd166", c.height / 4],
    ["spd", "#9dff87", c.height / 4],
    ["dopa", "#ff4fd8", c.height / 8],
    ["steerLearn", "#c792ea", c.height / 8],
    ["cleared", "#1de9a0", c.height / 3],
    ["pit", "#ff5252", c.height / 3],
  ];
  for (const [key, color, scale] of series) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    const base = key === "dopa" || key === "steerLearn" || key === "cleared" || key === "pit" ? mid : 0;
    traceHist.forEach((t, i) => {
      const x = (i / 149) * c.width;
      const y = base + t[key] * scale;
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
