"""MuJoCo hexapod <-> FlyBrain bridge.

A physics-real embodiment for the connectome brain: the preconfigured 18-servo
hexapod (config/hexapod_world.xml) walking through a walled obstacle arena,
eyes and body state streamed to the brain over the SAME protocol every other
body uses (README "Frame formats"), the returned joint channels driving every
servo DIRECTLY — no gait library, no directional abstraction.

  MuJoCo world ──eyeL/eyeR 192x108 + body state──▶ connectome brain (:8793)
  servo ctrl  ◀──── 18 channels, one per joint ──────────────────────────┘

Channel semantics (decode is learned, not configured): the profile gives each
of the 18 servos its own DISJOINT pool of 90 descending motor neurons
("servo:k" <- "pool{k-1}"), so the circuit drives every joint individually.

Training loop mirrors the AirSim bridge's discipline:
  - reach the red puck  -> +2.5 jackpot, fresh episode (puck respawns)
  - closing on the puck -> small progress pulses (progress-only: standing
    still or retreating pays nothing)
  - obstacle/wall contact -> punish + stun: the scene is HELD (sim paused)
    until the brain's dopamine recovers, so the negative window overlaps the
    synapses that caused the touch (clean credit assignment); the touch also
    drives the state collision flag -> the brain's tactile burst channel
  - stagnation pressure: < --stagnate-disp net displacement in --stagnate-s
    is punished and the episode is respawned
  - auto-trim: each episode calibrates every channel's resting value and
    freezes it for the episode (deflections from rest = real intent)

MuJoCo rendering notes: call mj_forward BEFORE Renderer.update_scene, and
create renderers ONCE per process (per-episode GL context churn crashes
native MuJoCo on Windows).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import random
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

W, H = 192, 108
# servo:k = joint k of the robot, k = 3*(leg-1) + segment (0 coxa, 1 femur,
# 2 tibia): the profile's 18 disjoint pools map 1:1 onto these channels.
CHANNELS = tuple(f"servo:{i}" for i in range(1, 19))
SERVO_ACTUATORS = {f"servo:{3 * (k - 1) + s + 1}": f"{seg}_{k}"
                   for k in range(1, 7)
                   for s, seg in enumerate(("coxa", "femur", "tibia"))}
TOUCH_COOLDOWN_S = 1.0
TOUCH_FLAG_WINDOW_S = 0.25     # how long the state collision flag stays set


def pack_eye_frame(eye: int, rgb: np.ndarray) -> bytes:
    return struct.pack("<BBHHB", 1, eye, W, H, 3) + np.ascontiguousarray(rgb).tobytes()


def decode_action(msg: bytes) -> dict[str, float]:
    """[10 u8][nameLen u16][names JSON][f32*n] -> {channel: value}."""
    if len(msg) < 3 or msg[0] != 10:
        return {}
    (nl,) = struct.unpack_from("<H", msg, 1)
    names = json.loads(msg[3:3 + nl])
    vals = struct.unpack_from("<" + "f" * len(names), msg, 3 + nl)
    return dict(zip(names, vals))


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def deadband(x: float, db: float) -> float:
    return 0.0 if abs(x) < db else x


class HexaBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = mujoco.MjModel.from_xml_path(args.world)
        self.data = mujoco.MjData(self.model)
        self.acts = {
            ch: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for ch, name in SERVO_ACTUATORS.items()}
        assert all(v >= 0 for v in self.acts.values()), "actuator name mismatch"
        self.home = np.zeros(18)   # keyframe home pose, captured at first reset
        self.camL = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eyeL")
        self.camR = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eyeR")
        self.camChase = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "chase")
        self.goal_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.goal_mocapid = self.model.body_mocapid[self.goal_body]
        assert self.goal_mocapid >= 0, "goal body must be a mocap body"

        self._renderer = None
        self._view_renderer = None
        self._view_jpeg = b""
        self._view_img: np.ndarray | None = None
        self._view_lock = threading.Lock()
        self._sim_lock = threading.Lock()
        self._render_stop = False
        self._render_task: asyncio.Task | None = None

        # episode / bridge state
        self.trim: dict[str, float] = {c: 0.0 for c in CHANNELS}
        self.trim_samples: dict[str, list[float]] = {c: [] for c in CHANNELS}
        self.trimmed = not args.trim
        self.cmd: dict[str, float] = {c: 0.0 for c in CHANNELS}
        self._latest_channels: dict[str, float] | None = None
        self.stick: dict[str, float] | None = None
        # exploration wind: JOINT-SPACE motor babble — a slow random walk on
        # every servo target (the brain's deflections ADD to it) so the
        # reactive circuit keeps producing optic flow, touches and puck
        # approaches — the event stream R-STDP learns from. Same role as the
        # AirSim bridge's motor-babble dither, in joint space.
        self.wind = np.zeros(18)
        self.wind_target = np.zeros(18)
        self.wind_next = 0.0
        self._last_carrot = 0.0
        self.got_actions = False
        self.ep_start = (0.0, 0.0)
        self.ep_t = 0.0
        self.ep_touches = 0
        self.ep_pucks = 0
        self.ep_closing = 0.0
        self.ep_best = float("inf")
        self.last_puck_dist = float("inf")
        self.last_touch_t = -1e9
        self.touch_flag_until = 0.0
        self.pending_contact = False
        self.last_prog_t = 0.0
        self.stagnate_since: float | None = None
        self.stun_start: float | None = None
        self.dopa = 0.0
        self.learning = False
        self.telemetry: dict | None = None
        self._last_loop = time.time()
        self._last_hud = 0.0
        self._last_tel = 0.0
        self._last_jpeg = 0.0

    # ---- episode -------------------------------------------------------------
    def reset_episode(self) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[0] = random.uniform(-1.2, 1.2)
        d.qpos[1] = random.uniform(-1.2, 1.2)
        yaw = random.uniform(-math.pi, math.pi)
        d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qvel[:] = 0.0
        d.mocap_pos[self.goal_mocapid] = [
            random.uniform(-1.3, 1.3), random.uniform(-1.3, 1.3), 0.004]
        d.ctrl[:] = d.qpos[7:25]           # home servo pose
        self.home = d.qpos[7:25].copy()
        mujoco.mj_forward(m, d)
        self.trimmed = not self.args.trim
        self.trim = {c: 0.0 for c in CHANNELS}
        self.trim_samples = {c: [] for c in CHANNELS}
        self._latest_channels = None
        self.stick = None
        self.ep_start = (float(d.qpos[0]), float(d.qpos[1]))
        self.ep_t = 0.0
        self.ep_touches = 0
        self.ep_pucks = 0
        self.ep_closing = 0.0
        self.ep_best = float("inf")
        self.last_puck_dist = float("inf")
        self.last_touch_t = -1e9
        self.touch_flag_until = 0.0
        self.pending_contact = False
        self.last_prog_t = 0.0
        self.stagnate_since = None
        self.stun_start = None
        # NOTE: renderers are created once per process and kept. Destroying
        # and recreating GL contexts per episode exhausts the Windows OpenGL
        # driver and hard-crashes the process (native, no traceback).
        # update_scene() rebuilds the scene from live data on every call, so
        # a persistent renderer is always current.
        print(f"[episode] respawn at ({d.qpos[0]:+.2f}, {d.qpos[1]:+.2f}) "
              f"yaw {math.degrees(yaw):+.0f}deg, puck at "
              f"({d.mocap_pos[self.goal_mocapid][0]:+.2f}, "
              f"{d.mocap_pos[self.goal_mocapid][1]:+.2f})")

    def robot_pose(self) -> tuple[float, float, float]:
        with self._sim_lock:
            d = self.data
            w, qx, qy, qz = d.qpos[3:7]
            yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            return float(d.qpos[0]), float(d.qpos[1]), yaw

    def puck_dist(self) -> float:
        d = self.data
        gp = d.mocap_pos[self.goal_mocapid]
        return math.hypot(d.qpos[0] - gp[0], d.qpos[1] - gp[1])

    # ---- sensing ---------------------------------------------------------------
    def render_eye(self, cam_id: int) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=H, width=W)
        self._renderer.update_scene(self.data, camera=cam_id)
        return self._renderer.render()

    def refresh_view_jpeg(self) -> None:
        """Encode + publish the LATEST rendered frame. Rendering itself runs
        on a dedicated thread (see render_thread): MuJoCo renders take tens
        of ms and would otherwise stall the sim/eye-stream loop — the source
        of the old viewer's lag."""
        with self._view_lock:
            img = self._view_img
        if img is None:
            return
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, self.args.view_quality])
        if ok:
            self._view_jpeg = buf.tobytes()

    def render_thread(self) -> None:
        """Dedicated chase-cam renderer: snapshots MuJoCo state under lock,
        renders OFF the sim loop, at the viewer's own cadence."""
        r = mujoco.Renderer(self.model, height=self.args.view_height,
                            width=self.args.view_width)
        snap = mujoco.MjData(self.model)
        while not self._render_stop:
            t0 = time.perf_counter()
            with self._sim_lock:
                snap.qpos[:] = self.data.qpos
                snap.qvel[:] = self.data.qvel
                snap.act[:] = self.data.act
                snap.time = self.data.time
                snap.mocap_pos[:] = self.data.mocap_pos
                snap.mocap_quat[:] = self.data.mocap_quat
            mujoco.mj_forward(self.model, snap)
            r.update_scene(snap, camera=self.camChase)
            img = r.render()
            with self._view_lock:
                self._view_img = img
            # hold the viewer cadence (default ~12 fps), minus render cost
            delay = 1.0 / self.args.view_fps - (time.perf_counter() - t0)
            if delay > 0:
                time.sleep(delay)

    def geom_name(self, geom_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""

    def detect_touch(self) -> str | None:
        """Name of the obstacle a body part touched this step, or None."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1, n2 = self.geom_name(c.geom1), self.geom_name(c.geom2)
            names = (n1, n2)
            for name in names:
                if name.startswith("rock") or name.startswith("wall"):
                    return name
            # feet/legs touching the floor is normal walking; the chassis
            # scraping anything = toppled or rammed into an obstacle
            if ("lower" in names or "upper" in names) and "floor" not in names:
                return "body-vs-" + (n2 if ("lower" in n1 or "upper" in n1)
                                     else n1)
        return None

    def joint_state(self) -> tuple[list[float], list[float]]:
        """Normalized per-DoF proprioception: qpos/qvel of the 18 hinges in
        [-1, 1] (range +/-90 deg). Order matches the servo channels: joint k =
        3*(leg-1)+segment, exactly the profile's pool -> servo mapping."""
        d = self.data
        state = [float(np.clip(d.qpos[7 + i] / 1.5708, -1.0, 1.0)) for i in range(18)]
        vel = [float(np.clip(d.qvel[6 + i] / 12.0, -1.0, 1.0)) for i in range(18)]
        return state, vel

    def body_state(self) -> dict:
        """Joints ONLY: everything in this frame is proprioception a real
        robot can measure with encoders. Altitude/speed/vy/collision were
        removed — they are simulation knowledge, not body knowledge. Body
        collision still reaches the brain as a touch REWARD event from the
        episode logic (as it would from a bumper's nerves), never as state."""
        state, vel = self.joint_state()
        contact = self.pending_contact
        self.pending_contact = False
        return {
            # proprioception: [[state, vel], ...] per DoF, normalized
            "joints": [[round(s, 3), round(v, 3)]
                       for s, v in zip(state, vel)],
            # one-shot chassis-contact event (a bumper nerve, not a flag)
            "contact": contact,
        }

    # ---- joint channels -> servo ctrl ------------------------------------------
    def apply_channels(self, ch: dict[str, float], dt: float) -> None:
        args = self.args
        if not self.trimmed and self.ep_t < args.trim_window:
            for k in CHANNELS:
                if k in ch:
                    self.trim_samples[k].append(float(ch[k]))
        elif not self.trimmed:
            for k in CHANNELS:
                xs = self.trim_samples[k]
                self.trim[k] = float(np.median(xs)) if len(xs) >= 20 else 0.0
            self.trimmed = True
            print("[trim] " + "  ".join(f"{k} {v:+.3f}" for k, v in self.trim.items()))

        def raw(k: str) -> float:
            return ch.get(k, self.trim[k]) - self.trim[k]

        # per-joint command smoothing: the untrained circuit's channel
        # deflections arrive at tick rate; first-order smoothing (tau ~0.35 s)
        # keeps servo targets coherent instead of lurching — the AirSim
        # bridge's --stick-tau, in joint space.
        if self.stick is None:                    self.stick = {k: clamp(raw(k), -1.0, 1.0) for k in CHANNELS}
        alpha = min(1.0, dt / max(self.args.stick_tau, 1e-3))  # wall-dt OK: RC feel
        for k in CHANNELS:
            self.stick[k] += (clamp(raw(k), -1.0, 1.0) - self.stick[k]) * alpha

        def eff(k: str) -> float:
            return deadband(self.stick[k], args.deadband)

        # joint babble wind (see __init__): retargets on a clock, eases in,
        # yields inside the puck's near zone so the brain owns the approach
        now = time.time()
        if now >= self.wind_next:
            for i in range(18):
                if (i % 3) == 0:                    # coxa: gentle turn-like sway
                    self.wind_target[i] = random.uniform(-0.35, 0.35)
                else:                               # femur/tibia: lift/plant steps
                    self.wind_target[i] = (random.uniform(-0.5, 0.5)
                                           if random.random() > 0.3 else 0.0)
            self.wind_next = now + random.uniform(3.0, 7.0)
        w_alpha = min(1.0, dt / 1.5)
        near = self.puck_dist() < self.args.near_radius
        wind_scale = args.wind_gain * (args.near_yield if near else 1.0)
        self.wind += (self.wind_target * wind_scale - self.wind) * w_alpha

        self.cmd = {k: round(eff(k), 3) for k in CHANNELS}
        # servo targets: home pose + brain deflection + babble wind
        for i, chan in enumerate(CHANNELS):
            a = self.acts[chan]
            self.data.ctrl[a] = clamp(self.home[i] + eff(chan) + self.wind[i],
                                      -1.5708, 1.5708)

    def step_servos(self) -> None:
        with self._sim_lock:      # physics state is read by the render thread
            for _ in range(self.args.physics_substeps):
                mujoco.mj_step(self.model, self.data)
            self.ep_t += 0.002 * self.args.physics_substeps

    # ---- viewer -------------------------------------------------------------
    async def viewer_server(self, host: str, port: int) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path.startswith("/view.jpg"):
                    body, ctype = bridge._view_jpeg, "image/jpeg"
                elif self.path.startswith("/status"):
                    x, y, yaw = bridge.robot_pose()
                    body = json.dumps({
                        "x": round(x, 3), "y": round(y, 3),
                        "yawDeg": round(math.degrees(yaw), 1),
                        "episodeT": round(bridge.ep_t, 1),
                        "touches": bridge.ep_touches, "pucks": bridge.ep_pucks,
                        "bestPuck": (None if bridge.ep_best == float("inf")
                                     else round(bridge.ep_best, 3)),
                        "dopa": round(bridge.dopa, 3),
                        "stunned": bridge.stun_start is not None,
                        "cmd": bridge.cmd,
                        "jme": round(float(np.abs(bridge.data.qvel[6:24]).mean()), 3),
                        "learning": bridge.learning,
                    }).encode()
                    ctype = "application/json"
                else:
                    body, ctype = HEXA_VIEWER_HTML.encode(), "text/html"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a) -> None:
                pass

        server = ThreadingHTTPServer((host, port), Handler)
        print(f"[viewer] http://localhost:{port} (chase cam + status)")
        await asyncio.to_thread(server.serve_forever)

    # ---- brain ---------------------------------------------------------------
    async def handshake(self, ws) -> None:
        await ws.send(json.dumps({"type": "hello", "profile": "hexapod"}))
        ack = json.loads(await ws.recv())
        circ = ack.get("circuit", ack)
        print(f"[brain] {circ.get('neurons', '?')} neurons / "
              f"{circ.get('edges', '?')} synapses; hexapod profile applied")

    async def reader_loop(self, ws) -> None:
        async for msg in ws:
            if isinstance(msg, bytes):
                ch = decode_action(msg)
                if ch:
                    self._latest_channels = ch
                    self.got_actions = True
            elif msg:
                try:
                    obj = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "telemetry":
                    self.telemetry = obj
                    self.dopa = float(obj.get("dopa", 0.0))
                    self.learning = bool(obj.get("learning", False))

    # ---- training signals ------------------------------------------------------
    async def shaping_tick(self, ws) -> None:
        dist = self.puck_dist()
        self.ep_best = min(self.ep_best, dist)
        now = time.time()
        if self.last_puck_dist != float("inf") and now - self.last_prog_t >= 0.5:
            gained = self.last_puck_dist - dist
            if gained >= self.args.closing_min:
                v = min(self.args.closing_gain * gained, 0.5)
                self.ep_closing += v
                await ws.send(json.dumps({"type": "reward", "value": round(v, 3)}))
                self.last_prog_t = now
        self.last_puck_dist = dist
        # the carrot: sustained actual motion (outside the near zone) pays a
        # small periodic pulse — moving must earn before any approach
        # gradient can be discovered (mirrors the AirSim cruise pulse)
        speed = float(math.hypot(self.data.qvel[0], self.data.qvel[1]))
        if (speed >= self.args.carrot_speed and dist >= self.args.near_radius
                and now - self._last_carrot >= 2.0):
            self._last_carrot = now
            await ws.send(json.dumps({"type": "reward",
                                      "value": self.args.carrot_mag}))

    async def punish(self, ws, mag: float, reason: str) -> None:
        self.ep_touches += 1
        self.stun_start = time.time()
        self.touch_flag_until = self.stun_start + TOUCH_FLAG_WINDOW_S
        self.pending_contact = True   # one-shot body-contact nerve event
        await ws.send(json.dumps({"type": "reward", "value": mag}))
        print(f"[sim] touch [{reason}] {mag:+.1f}; scene held until dopa recovers")

    async def stun_hold(self, ws) -> None:
        """Hold the scene until the dopamine tail recovers (dopa-aware stun)."""
        while self.stun_start is not None:
            held = time.time() - self.stun_start
            if (held >= self.args.stun_hold and self.dopa >= self.args.dopa_recover) \
                    or held >= self.args.stun_hold + 2.0:
                print(f"[sim] stun over after {held:.1f}s (dopa {self.dopa:+.3f})")
                self.stun_start = None
                return
            if time.time() - self._last_tel > 0.5:
                await ws.send(json.dumps({"type": "telemetry"}))
                self._last_tel = time.time()
            await asyncio.sleep(0.05)

    async def check_stagnation(self, ws) -> None:
        x, y, _ = self.robot_pose()
        disp = math.hypot(x - self.ep_start[0], y - self.ep_start[1])
        now = time.time()
        if disp < self.args.stagnate_disp:
            if self.stagnate_since is None:
                self.stagnate_since = now
            elif now - self.stagnate_since >= self.args.stagnate_s:
                await self.punish(ws, -self.args.punish_mag, "stagnation")
        else:
            self.stagnate_since = None

    # ---- main loop ------------------------------------------------------------
    async def run(self) -> None:
        args = self.args
        print(f"[hexapod] MuJoCo {mujoco.__version__} | world {args.world} | "
              f"brain {args.brain_ws}")
        self.reset_episode()
        async with websockets.connect(args.brain_ws, max_size=2 ** 22) as ws:
            await self.handshake(ws)
            reader = asyncio.create_task(self.reader_loop(ws))
            viewer = (asyncio.create_task(
                self.viewer_server("127.0.0.1", args.viewer_port))
                if args.viewer_port else None)
            if args.viewer_port:
                self._render_task = asyncio.create_task(
                    asyncio.to_thread(self.render_thread))
            eye_period = 1.0 / args.vision_hz if args.vision_hz > 0 else 0.0
            next_eye = 0.0
            try:
                while True:
                    now = time.time()
                    dt = clamp(now - self._last_loop, 0.001, 0.05)
                    self._last_loop = now

                    if self.stun_start is not None:
                        await self.stun_hold(ws)
                        self.reset_episode()
                        next_eye = 0.0
                        continue

                    ch = self._latest_channels or {c: 0.0 for c in CHANNELS}
                    self.apply_channels(ch, dt)
                    self.step_servos()

                    reason = self.detect_touch()
                    if reason and now - self.last_touch_t > TOUCH_COOLDOWN_S:
                        self.last_touch_t = now
                        await self.punish(ws, -args.punish_mag, reason)
                        continue

                    if self.puck_dist() < args.puck_radius:
                        self.ep_pucks += 1
                        await ws.send(json.dumps({"type": "reward", "value": 2.5}))
                        print(f"[sim] PUCK reached +2.5 (pucks {self.ep_pucks})")
                        self.reset_episode()
                        next_eye = 0.0
                        continue

                    if now >= next_eye:
                        await ws.send(pack_eye_frame(0, self.render_eye(self.camL)))
                        await ws.send(pack_eye_frame(1, self.render_eye(self.camR)))
                        st = self.body_state()
                        await ws.send(json.dumps({"type": "state", **st}))
                        next_eye = now + eye_period if eye_period else now + 1 / 60

                    await self.shaping_tick(ws)
                    await self.check_stagnation(ws)

                    if now - self._last_tel > 1.0:
                        await ws.send(json.dumps({"type": "telemetry"}))
                        self._last_tel = now

                    if args.viewer_port and now - self._last_jpeg > 0.4:
                        self.refresh_view_jpeg()
                        self._last_jpeg = now

                    if now - self._last_hud > 2.0:
                        x, y, yaw = self.robot_pose()
                        print(f"[hexa] x {x:+.2f} y {y:+.2f} "
                              f"yaw {math.degrees(yaw):+5.0f}deg z {self.data.qpos[2]:.3f} "
                              f"| puck {self.puck_dist():.2f}m best {self.ep_best:.2f}m "
                              f"| touches {self.ep_touches} pucks {self.ep_pucks} "
                              f"| dopa {self.dopa:+.3f} learn {int(self.learning)} "
                              f"| max|cmd| {max(abs(v) for v in self.cmd.values()):.2f} "
                              f"| jME {float(np.abs(self.data.qvel[6:24]).mean()):.2f} "
                              f"| wind {float(np.abs(self.wind).max()):.2f}")
                        self._last_hud = now

                    await asyncio.sleep(0)
            finally:
                reader.cancel()
                if viewer is not None:
                    viewer.cancel()
                self._render_stop = True


HEXA_VIEWER_HTML = """<!doctype html>
<html><head><title>hexapod viewer</title>
<style>
  body { margin:0; background:#111; color:#ddd; font:13px system-ui; }
  img { display:block; width:100%; image-rendering:auto; }
  #status { padding:6px 10px; white-space:pre; font-family:ui-monospace,monospace; }
</style></head>
<body>
<div id="status">connecting...</div>
<img id="chase" src="/view.jpg">
<script>
async function poll() {
  for (;;) {
    try {
      const s = await (await fetch('/status')).json();
      document.getElementById('status').textContent =
        `x ${s.x}  y ${s.y}  yaw ${s.yawDeg}deg  epT ${s.episodeT}s | ` +
        `puck best ${s.bestPuck}m | touches ${s.touches} pucks ${s.pucks} | ` +
        `dopa ${s.dopa} ${s.stunned ? "STUNNED" : ""} | learn ${s.learning ? 1 : 0} | ` +
        `jME ${s.jme} | cmd ${JSON.stringify(s.cmd)}`;
    } catch (e) {}
    await new Promise(r => setTimeout(r, 80));
    document.getElementById('chase').src = '/view.jpg?t=' + Date.now();
  }
}
poll();
</script>
</body></html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MuJoCo hexapod <-> FlyBrain bridge")
    p.add_argument("--world", default=str(ROOT / "config/hexapod_world.xml"))
    p.add_argument("--brain-ws", default="ws://127.0.0.1:8793/stream")
    p.add_argument("--vision-hz", type=float, default=30.0,
                   help="eye frame cap (0 = uncapped, sim-paced)")
    p.add_argument("--physics-substeps", type=int, default=4,
                   help="2ms physics steps per loop tick (4 = 500 Hz control)")
    p.add_argument("--trim", action="store_true", default=True)
    p.add_argument("--no-trim", dest="trim", action="store_false")
    p.add_argument("--trim-window", type=float, default=1.5)
    p.add_argument("--stick-tau", type=float, default=0.35,
                   help="per-joint command smoothing time constant (s)")
    p.add_argument("--deadband", type=float, default=0.05)
    p.add_argument("--punish-mag", type=float, default=2.5)
    p.add_argument("--stun-hold", type=float, default=4.5)
    p.add_argument("--dopa-recover", type=float, default=-0.03)
    p.add_argument("--closing-gain", type=float, default=0.12)
    p.add_argument("--closing-min", type=float, default=0.02)
    p.add_argument("--puck-radius", type=float, default=0.12)
    p.add_argument("--wind-gain", type=float, default=1.0,
                   help="exploration wind strength (0 disables)")
    p.add_argument("--near-radius", type=float, default=0.5,
                   help="puck distance inside which the wind yields")
    p.add_argument("--near-yield", type=float, default=0.35)
    p.add_argument("--carrot-mag", type=float, default=0.05)
    p.add_argument("--carrot-speed", type=float, default=0.02,
                   help="sustained body speed (m/s) that earns the carrot")
    p.add_argument("--stagnate-disp", type=float, default=0.15,
                   help="net displacement (m) under which the episode stagnates")
    p.add_argument("--stagnate-s", type=float, default=60.0)
    p.add_argument("--viewer-port", type=int, default=8795)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--view-fps", type=float, default=12.0,
                   help="chase-cam render cadence (dedicated thread)")
    p.add_argument("--view-width", type=int, default=960)
    p.add_argument("--view-height", type=int, default=540)
    p.add_argument("--view-quality", type=int, default=82,
                   help="JPEG quality for the viewer feed")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    if args.no_viewer:
        args.viewer_port = 0
    bridge = HexaBridge(args)
    await bridge.run()


if __name__ == "__main__":
    asyncio.run(main())
