"""AirSim <-> FlyBrain bridge.

Streams AirSim camera captures to the C connectome brain (cengine) over its
WebSocket protocol and applies the returned actuator channels to the drone.

Brain protocol (README "Frame formats"):
  client->brain: eye frame  [1 u8][eye u8][w u16][h u16][3 u8] + w*h*3 RGB
                 state JSON {"type":"state", altitude, vy, collision}
  brain->client: action     [10 u8][nameLen u16][names JSON][f32*n]
                 JSON       hello ack / telemetry

Embodiment parity with examples/drone-web:
  - eye 0 = LEFT (+35 deg off boresight), eye 1 = RIGHT (-35 deg). AirSim
    (NED) yaws the opposite direction from three.js, so the left camera is
    configured at Yaw -35 and the right at +35 (see config/airsim.settings.json).
  - state sends ONLY what a real drone senses: altitude, vy, collision.
    No clearance: the brain estimates distance from optic flow itself.

AirSim side:
  - SimpleFlight physics, collisions enabled.
  - JOYSTICK control scheme: the brain's channels are mapped onto FPV-style
    sticks and flown through moveByVelocityBodyFrameAsync — the flight
    controller does all prop mixing and attitude stabilization:
      pitch stick  -> forward/backward body-frame velocity
      roll stick   -> left/right body-frame velocity
      throttle     -> climb/descent rate, stick NORMALIZED around hover so
                      down authority == up authority (a raw mapping gives
                      down-sticks only hover*100% of the up range, which is
                      why the drone could climb but never descend)
      yaw stick    -> yaw RATE
      every stick is amplified (--stick-gain) then low-passed (--stick-tau)
      to behave like real RC sticks: the brain's brief wiggles become
      gliding deflections and sustained outputs integrate into motion
  - vision: ONE forward-facing center camera (camera "2"); its frame is
    streamed to BOTH eyes (0 and 1) so the connectome's full bilateral
    retina sees the same view. --eyes stereo restores the +-35 deg pair.
  - the connectome decode cannot hover by construction (no altitude
    feedback in the motor pools), so the bridge keeps the drone inside a
    ground/ceiling safety band: the brain's throttle FULLY owns climb and
    descent; the assist only pushes back within 1 m of the band edges.
    --no-assist removes even that.
  - collisions punish (default -2.5) and respawn; parked-car touches reward
    (default +2.5) and respawn. --punish-mag / --reward-mag tune magnitudes.

Usage:
  python scripts/airsim_drone.py                     # defaults, assist on
  python scripts/airsim_drone.py --no-assist         # raw brain channels
  python scripts/airsim_drone.py --reward alt        # external reward stream
  python scripts/airsim_drone.py --vision-hz 120     # more retina bandwidth
  python scripts/airsim_drone.py --eyes stereo       # +-35 deg camera pair
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import sys
import time
from pathlib import Path
import re

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import airsim  # noqa: E402  (vendored in scripts/airsim, MIT licensed)

# ---- embodiment constants (mirrors examples/drone-web) ---------------------
W, H = 192, 108
DEFAULT_VISION_HZ = 60          # --vision-hz 120 matches the web client

# ---- AirSim joystick envelope (SimpleFlight) -------------------------------
V_FWD_MAX = 12.0       # m/s at |pitch stick| = 1 (web convention: -pitch = fwd)
V_LAT_MAX = 9.0        # m/s at |roll stick| = 1 — strong avoidance authority
VZ_MAX = 7.0           # m/s climb/descent demand at |climb stick| = 1
YAW_RATE_MAX = 200.0   # deg/s at |yaw stick| = 1 — drastic turns
STICK_GAIN = 2.0       # sensitivity: amplify the brain's small channel wiggles
STICK_TAU_S = 0.15     # RC stick inertia (first-order low-pass on each stick)
CMD_HOLD_S = 0.25      # velocity command hold (re-sent every CMD_PERIOD_S)

BRAIN_HOVER = 0.29                     # brain throttle default (drone.json)

RESPAWN_ALT = 8.0      # respawn height above ground (m)
GROUND_Z = 0.0         # world-frame z of the ground plane (--ground-z)
RESPAWN_COOLDOWN_S = 2.5   # ignore collisions this long after a respawn

CEILING_RIDE_S = 2.0    # pinned at/above the ceiling this long -> punish
BORDER_RADIUS_M = 450.0 # horizontal leash from the spawn point (map border)

RETRY_BASE_S = 1.0
RETRY_MAX_S = 8.0

STATUS_PERIOD_S = 5.0


CAR_NAME_RE = re.compile(r"^car[_]?\d", re.IGNORECASE)  # AirSimNH: Car_10, Car_01_32, ...


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def pack_eye_frame(eye: int, rgb: np.ndarray) -> bytes:
    """[1 u8][eye u8][w u16][h u16][3 u8] + w*h*3 RGB bytes."""
    return struct.pack("<BBHHB", 1, eye, W, H, 3) + rgb.tobytes()


def decode_action(msg: bytes) -> dict[str, float]:
    """[10 u8][nameLen u16][names JSON][f32*n] -> {channel: value}."""
    if len(msg) < 3 or msg[0] != 10:
        return {}
    (nl,) = struct.unpack_from("<H", msg, 1)
    names = json.loads(msg[3:3 + nl])
    vals = struct.unpack_from("<" + "f" * len(names), msg, 3 + nl)
    return dict(zip(names, vals))


class Bridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cmd = {"throttle": BRAIN_HOVER, "pitch": 0.0, "roll": 0.0, "yaw": 0.0}
        self.got_actions = False
        self.telemetry: dict | None = None
        self.dopa = 0.0
        self.learning = False
        self.collision_count = 0
        self.car_bumps = 0
        self.ceiling_hits = 0
        self.border_hits = 0
        self._ceil_since = 0.0
        self._last_col_ts = 0.0
        self._last_respawn = 0.0
        self._reward_prog = 0.0
        self._last_reward_t = 0.0
        self._last_tel_req = 0.0
        self._last_status = 0.0
        self._actions_at = 0.0
        self._last_col_print = 0.0
        self._last_cmd_at = 0.0
        self._grounded_since = 0.0
        self._ws_ref = None
        self.stick = {"fwd": 0.0, "lat": 0.0, "climb": 0.0, "yaw": 0.0}
        self._stick_at = 0.0
        self._hud = (0.0, 0.0, 0.0, 0.0)

    # ---- brain link ------------------------------------------------------
    async def run(self) -> int:
        args = self.args
        client = airsim.MultirotorClient(ip=args.airsim_ip, port=args.airsim_port)
        client.confirmConnection()
        vehicles = client.listVehicles()
        print(f"[airsim] connected; vehicles: {vehicles}")
        if args.vehicle not in vehicles:
            print(f"[airsim] vehicle {args.vehicle!r} not in {vehicles}")
            return 1
        self.spawn = client.simGetGroundTruthKinematics(args.vehicle).position
        self.setup_vehicle(client)

        delay = RETRY_BASE_S
        while True:
            print(f"[bridge] connecting to brain at {args.brain_ws}")
            try:
                async with websockets.connect(
                        args.brain_ws, open_timeout=30, max_size=32 * 2**20) as ws:
                    await self.handshake(ws)
                    delay = RETRY_BASE_S
                    await self.flight_loop(client, ws)
            except (websockets.exceptions.ConnectionClosed, OSError) as exc:
                print(f"[bridge] brain link lost ({exc}); retrying", flush=True)
            except KeyboardInterrupt:
                return 0
            except Exception as exc:  # keep the sim alive across brain restarts
                print(f"[bridge] error: {exc!r}; retrying", flush=True)
            await asyncio.sleep(delay)
            delay = min(RETRY_MAX_S, delay * 1.5)

    def setup_vehicle(self, client) -> None:
        client.enableApiControl(True, self.args.vehicle)
        client.armDisarm(True, self.args.vehicle)
        self.takeoff(client)

    def takeoff(self, client) -> None:
        """Un-stick (spawn can be below the collision surface, which pins the
        physics resolver), teleport to a clean absolute altitude, then climb
        a little. Absolute z (ground-z - 8) instead of relative, so a spawn
        captured mid-air can never push the drone above the ceiling."""
        print("[airsim] takeoff", flush=True)
        pose = client.simGetVehiclePose(self.args.vehicle)
        pose.position.z_val = self.args.ground_z - RESPAWN_ALT  # NED: up = -z
        client.simSetVehiclePose(pose, True, self.args.vehicle)
        client.moveByVelocityBodyFrameAsync(
            0.0, 0.0, -2.0, 4.0,   # climb at 2 m/s for 4 s (NED: -vz = up)
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, 0.0),
            self.args.vehicle,
        ).join()

    def recover_if_stuck(self, client, kin, now: float) -> None:
        """Grounded recovery: if we sit at/below ground with no motion for a
        while (wedged, or knocked down), teleport up and take off again."""
        alt = -kin.position.z_val
        speed = math.sqrt(kin.linear_velocity.x_val ** 2
                          + kin.linear_velocity.y_val ** 2
                          + kin.linear_velocity.z_val ** 2)
        if alt < 0.5 and speed < 0.6:
            if self._grounded_since == 0.0:
                self._grounded_since = now
            elif now - self._grounded_since > 3.0:
                print("[sim] grounded recovery: taking off", flush=True)
                self.takeoff(client)
                self._grounded_since = 0.0
        else:
            self._grounded_since = 0.0

    def check_bounds(self, client, kin, now: float) -> None:
        """Ceiling and map-border policy: riding the ceiling or leaving the
        spawn leash is a punish + respawn, exactly like a collision."""
        if now - self._last_respawn < RESPAWN_COOLDOWN_S:
            return
        alt = -kin.position.z_val
        dx = kin.position.x_val - self.spawn.x_val
        dy = kin.position.y_val - self.spawn.y_val
        dist = math.hypot(dx, dy)
        reason = None
        # ceiling punish sits 25 cm inside the blended band edge, so a
        # saturated climb that equilibrates right at the ceiling still counts
        if alt >= self.args.max_alt - 0.25:
            if self._ceil_since == 0.0:
                self._ceil_since = now
            elif now - self._ceil_since > self.args.ceil_ride:
                reason = "ceiling"
        else:
            self._ceil_since = 0.0
        if reason is None and dist > self.args.border_radius:
            reason = "map border"
        if reason is not None:
            if reason == "ceiling":
                self.ceiling_hits += 1
            else:
                self.border_hits += 1
            asyncio.get_running_loop().create_task(
                self.send_reward_now(self.args.punish_mag))
            print(f"[sim] {reason} -> punish {self.args.punish_mag:+.1f}; "
                  f"respawning (hits {self.collision_count}, ceilings "
                  f"{self.ceiling_hits}, borders {self.border_hits})",
                  flush=True)
            self.respawn(client)
            self._last_respawn = now

    async def handshake(self, ws) -> None:
        await ws.send(json.dumps({"type": "hello", "profile": "drone"}))
        ack = json.loads(await ws.recv())
        circ = ack.get("circuit", ack)
        print(f"[brain] {circ.get('neurons', '?')} neurons / "
              f"{circ.get('edges', '?')} synapses; profile applied")

    # ---- main loop -------------------------------------------------------
    async def flight_loop(self, client, ws) -> None:
        args = self.args
        self._ws_ref = ws
        send_dt = 1.0 / args.vision_hz
        next_send = 0.0
        print(f"[bridge] streaming eyes at {args.vision_hz} Hz; "
              f"assist={'off' if args.no_assist else 'on'}; reward={args.reward}")

        reader = asyncio.create_task(self.reader_loop(ws))
        try:
            while True:
                now = time.perf_counter()

                # stream the eye pair + body state
                if now >= next_send:
                    next_send = now + send_dt
                    await self.sense_and_stream(client, ws)

                # periodic control / telemetry / status
                self.apply_command(client, now)
                kin = client.simGetGroundTruthKinematics(self.args.vehicle)
                self.recover_if_stuck(client, kin, now)
                self.check_bounds(client, kin, now)
                await self.maybe_reward(client, ws, now)
                if now - self._last_tel_req > 1.0:
                    self._last_tel_req = now
                    await ws.send(json.dumps({"type": "telemetry"}))
                if now - self._last_status > STATUS_PERIOD_S:
                    self._last_status = now
                    self.print_status(client)

                await asyncio.sleep(0.002)
        finally:
            reader.cancel()

    async def reader_loop(self, ws) -> None:
        """Consume brain messages continuously; updates cmd/telemetry state."""
        try:
            async for msg in ws:
                self.handle_brain_message(msg)
                if isinstance(msg, bytes) and msg and msg[0] == 10:
                    self._actions_at = time.perf_counter()
                    self.got_actions = True
        except websockets.exceptions.ConnectionClosed:
            pass

    # ---- sense -----------------------------------------------------------
    async def sense_and_stream(self, client, ws) -> None:
        col = client.simGetCollisionInfo(self.args.vehicle)
        self.track_collision(client, col)
        kin = client.simGetGroundTruthKinematics(self.args.vehicle)
        alt = -kin.position.z_val                    # NED z is down
        vy_up = -kin.linear_velocity.z_val
        speed = math.sqrt(kin.linear_velocity.x_val ** 2
                          + kin.linear_velocity.y_val ** 2
                          + kin.linear_velocity.z_val ** 2)
        collided = bool(col.has_collided)

        if self.args.eyes == "stereo":
            requests = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
            resps = client.simGetImages(requests)
            frames = [pack_eye_frame(eye, rgb)
                      for eye, resp in enumerate(resps)
                      if (rgb := self.rgb_from_response(resp)) is not None]
        else:
            # ONE forward center camera ("2"); its frame feeds BOTH eyes so
            # the connectome's full bilateral retina sees the same view.
            resp = client.simGetImages(
                [airsim.ImageRequest("2", airsim.ImageType.Scene, False, False)])[0]
            rgb = self.rgb_from_response(resp)
            frames = ([pack_eye_frame(0, rgb), pack_eye_frame(1, rgb)]
                      if rgb is not None else [])
        if len(frames) == 2:
            for f in frames:
                await ws.send(f)
            await ws.send(json.dumps({
                "type": "state",
                "altitude": round(alt, 3),
                "speed": round(speed, 3),
                "vy": round(vy_up, 3),
                "collision": collided,
            }))

    @staticmethod
    def rgb_from_response(resp) -> np.ndarray | None:
        data = bytes(resp.image_data_uint8)
        if not data:
            return None
        n = resp.width * resp.height
        if len(data) >= n * 3:
            arr = np.frombuffer(data[:n * 3], dtype=np.uint8)
            return arr.reshape(H, W, 3)
        return None

    def track_collision(self, client, col) -> None:
        """Collision event -> punish OR reward (car), then respawn.
        Punish: anything that is not a car. Reward: AirSimNH parked cars
        (Car_*). Both respawn at the start point facing down the street."""
        now = time.perf_counter()
        if now - self._last_respawn < RESPAWN_COOLDOWN_S:
            return
        # has_collided latches after an impact; a NEW event is a new time_stamp
        ts = float(col.time_stamp) if col is not None else 0.0
        fresh = (col is not None and col.has_collided
                 and ts != self._last_col_ts)
        if not fresh:
            return
        self._last_col_ts = ts
        obj = (col.object_name or "")
        is_car = bool(obj) and bool(CAR_NAME_RE.match(obj))
        if is_car:
            self.car_bumps += 1
            mag = self.args.reward_mag
            sign = "car reward"
        else:
            self.collision_count += 1
            mag = self.args.punish_mag
            sign = "punish"
        asyncio.get_running_loop().create_task(
            self.send_reward_now(mag))
        print(f"[sim] collision [{obj or 'unknown'}] -> {sign} {mag:+.1f}; "
              f"respawning (hits {self.collision_count}, "
              f"cars {self.car_bumps})", flush=True)
        self.respawn(client)
        self._last_respawn = now

    def respawn(self, client) -> None:
        """Back to the spawn XY facing down the street, at an ABSOLUTE
        respawn altitude above the ground plane (never relative to wherever
        the drone was when the bridge started)."""
        pose = client.simGetVehiclePose(self.args.vehicle)
        sp = self.spawn
        pose.position.x_val = sp.x_val
        pose.position.y_val = sp.y_val
        pose.position.z_val = self.args.ground_z - RESPAWN_ALT  # NED: up = -z
        pose.orientation = airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)  # level, initial heading
        client.simSetVehiclePose(pose, True, self.args.vehicle)
        # kill any stale brain stick bias so the fresh episode starts calm
        self.cmd = {"throttle": BRAIN_HOVER, "pitch": 0.0, "roll": 0.0, "yaw": 0.0}
        self.stick = {"fwd": 0.0, "lat": 0.0, "climb": 0.0, "yaw": 0.0}
        self._ceil_since = 0.0
        client.moveByVelocityBodyFrameAsync(
            0.0, 0.0, -1.5, 1.0,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, 0.0),
            self.args.vehicle,
        )

    # ---- act -------------------------------------------------------------
    CMD_PERIOD_S = 0.05  # 20 Hz command refresh (SimpleFlight holds the rest)

    def compute_sticks(self, now: float) -> dict[str, float]:
        """Brain channels -> RC sticks, with real FPV feel:
        - the throttle stick is NORMALIZED around BRAIN_HOVER: hover-scaled
          down deflections get the same authority as up deflections (the old
          raw mapping capped down-sticks at hover of the up range, so the
          drone could climb but never genuinely descend);
        - --stick-gain amplifies the brain's small channel wiggles;
        - a first-order low-pass (--stick-tau) gives the stick real inertia,
          so brief channel dips integrate into actual stick deflections the
          flight controller can act on, instead of being averaged away."""
        c = self.cmd
        gain = self.args.stick_gain
        d = clamp(c["throttle"], 0.0, 1.0) - BRAIN_HOVER
        t = d / (1.0 - BRAIN_HOVER) if d >= 0.0 else d / BRAIN_HOVER
        raw = {
            "fwd":   clamp(-c["pitch"] * gain, -1.0, 1.0) * V_FWD_MAX,
            "lat":   clamp(c["roll"] * gain, -1.0, 1.0) * V_LAT_MAX,
            "climb": clamp(t * gain, -1.0, 1.0) * VZ_MAX,
            "yaw":   -clamp(c["yaw"] * gain, -1.0, 1.0) * YAW_RATE_MAX,
        }
        dt = now - self._stick_at
        self._stick_at = now
        if dt <= 0.0:
            return self.stick
        a = 1.0 if self.args.stick_tau <= 0 else min(1.0, dt / self.args.stick_tau)
        for k, tgt in raw.items():
            self.stick[k] += (tgt - self.stick[k]) * a
        return self.stick

    def apply_command(self, client, now: float) -> None:
        """FPV joystick scheme: smoothed RC sticks demand body-frame
        velocities + yaw rate; SimpleFlight mixes props and stabilizes."""
        if now - self._last_cmd_at < self.CMD_PERIOD_S:
            return
        self._last_cmd_at = now
        s = self.compute_sticks(now)
        climb = s["climb"]
        if not self.args.no_assist:
            # brain owns altitude; the band blends demand toward a safe rate
            # within 1 m of an edge and FULLY overrides past it, so even a
            # saturated climb/descent stick cannot fly out of the band
            kin = client.simGetGroundTruthKinematics(self.args.vehicle)
            alt = -kin.position.z_val
            over = (self.args.min_alt + 1.0) - alt
            if over > 0.0:                      # floor: blend to +2 m/s climb
                f = clamp(over, 0.0, 1.0)
                climb = climb * (1.0 - f) + 2.0 * f
            over = alt - (self.args.max_alt - 1.0)
            if over > 0.0:                      # ceiling: blend to -0.8 m/s
                f = clamp(over, 0.0, 1.0)
                climb = climb * (1.0 - f) - 0.8 * f
        self._hud = (s["fwd"], s["lat"], s["climb"], s["yaw"])
        # NED: vz is down-positive, so pass -climb
        client.moveByVelocityBodyFrameAsync(
            s["fwd"], s["lat"], -climb, CMD_HOLD_S + self.CMD_PERIOD_S,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, s["yaw"]),
            self.args.vehicle,
        )

    def handle_brain_message(self, msg) -> None:
        if isinstance(msg, bytes):
            action = decode_action(msg)
            if action:
                self.cmd["throttle"] = float(action.get("throttle", BRAIN_HOVER))
                self.cmd["pitch"] = float(action.get("pitch", 0.0))
                self.cmd["roll"] = float(action.get("roll", 0.0))
                self.cmd["yaw"] = float(action.get("yaw", 0.0))
            return
        try:
            obj = json.loads(msg)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if obj.get("type") == "telemetry":
            self.telemetry = obj
            self.dopa = float(obj.get("dopa", 0.0) or 0.0)
            self.learning = bool(obj.get("learning", False))

    # ---- reward shaping (optional) ----------------------------------------
    async def maybe_reward(self, client, ws, now: float) -> None:
        if not self.args.reward:
            return
        if now - self._last_reward_t < 0.5:
            return
        self._last_reward_t = now
        kin = client.simGetGroundTruthKinematics(self.args.vehicle)
        alt = -kin.position.z_val
        speed = math.hypot(kin.linear_velocity.x_val, kin.linear_velocity.y_val)
        self._reward_prog += (speed - self._reward_prog) * 0.05
        v = 0.6 * clamp((alt - 4.0) / 4.0, -1.0, 1.0) \
            + 0.4 * clamp((speed - self._reward_prog) / max(self._reward_prog, 0.5),
                          -1.0, 1.0)
        await ws.send(json.dumps({"type": "reward", "value": round(v, 3)}))

    async def send_reward_now(self, value: float) -> None:
        ws = getattr(self, "_ws_ref", None)
        if ws is not None:
            try:
                await ws.send(json.dumps({"type": "reward", "value": value}))
            except websockets.exceptions.ConnectionClosed:
                pass

    # ---- status ------------------------------------------------------------
    def print_status(self, client) -> None:
        kin = client.simGetGroundTruthKinematics(self.args.vehicle)
        alt = -kin.position.z_val
        speed = math.hypot(kin.linear_velocity.x_val, kin.linear_velocity.y_val)
        tel = self.telemetry or {}
        sim_ms = tel.get("simMs", 0.0)
        stale = time.perf_counter() - self._actions_at
        sticks = self._hud
        print(f"[fly] alt {alt:5.1f} m  spd {speed:4.1f} m/s  "
              f"sticks fwd {sticks[0]:+5.1f} lat {sticks[1]:+5.1f} "
              f"up {sticks[2]:+5.1f} m/s yaw {sticks[3]:+6.1f} deg/s  "
              f"hits {self.collision_count}  cars {self.car_bumps}  "
              f"dopa {self.dopa:+.3f} "
              f"learn {'on' if self.learning else 'off'}  "
              f"sim {sim_ms:.1f} ms  act {stale*1000:.0f} ms ago", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brain-ws", default="ws://127.0.0.1:8787/stream")
    ap.add_argument("--airsim-ip", default="127.0.0.1")
    ap.add_argument("--airsim-port", type=int, default=41451)
    ap.add_argument("--vehicle", default="Fly1")
    ap.add_argument("--vision-hz", type=int, default=DEFAULT_VISION_HZ,
                    help="eye streaming rate (web client uses 120)")
    ap.add_argument("--eyes", choices=["center", "stereo"], default="center",
                    help="center: one forward camera to both eyes; "
                         "stereo: +-35 deg left/right pair")
    ap.add_argument("--min-alt", type=float, default=2.5,
                    help="ground safety band lower edge (m)")
    ap.add_argument("--max-alt", type=float, default=30.0,
                    help="ceiling safety band upper edge (m)")
    ap.add_argument("--stick-gain", type=float, default=STICK_GAIN,
                    help="sensitivity multiplier on every stick (pre-clamp)")
    ap.add_argument("--stick-tau", type=float, default=STICK_TAU_S,
                    help="RC stick inertia time constant (s); 0 = raw sticks")
    ap.add_argument("--ceil-ride", type=float, default=CEILING_RIDE_S,
                    help="seconds riding the ceiling before punish+respawn")
    ap.add_argument("--border-radius", type=float, default=BORDER_RADIUS_M,
                    help="horizontal leash from spawn; beyond it = map border")
    ap.add_argument("--ground-z", type=float, default=GROUND_Z,
                    help="world-frame z of the ground plane at the spawn area")
    ap.add_argument("--no-assist", action="store_true",
                    help="apply raw brain channels, no altitude hold")
    ap.add_argument("--punish-mag", type=float, default=-2.5,
                    help="reward pulse value sent on a collision")
    ap.add_argument("--reward-mag", type=float, default=2.5,
                    help="reward pulse value sent on a car touch")
    ap.add_argument("--reward", choices=["alt"], default=None,
                    help="optional additional reward shaping (altitude+progress)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bridge = Bridge(args)
    try:
        return asyncio.run(bridge.run())
    except KeyboardInterrupt:
        print("\n[bridge] stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
