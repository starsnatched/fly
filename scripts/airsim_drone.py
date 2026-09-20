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
      throttle     -> SELECTS a target altitude lane (min_alt..max_alt);
                      a P controller climbs/descends to hold it. Position
                      control: the untrained throttle bias can no longer
                      produce endless climbing — a wiggle changes the lane
                      instead of integrating into continuous ascent, and
                      ceiling punishment maps directly onto the throttle
                      values that caused it. --control rate restores the
                      old climb-rate stick.
      yaw stick    -> yaw RATE
      every stick is amplified (--stick-gain) then low-passed (--stick-tau)
      to behave like real RC sticks: the brain's brief wiggles become
      gliding deflections and sustained outputs integrate into motion
  - vision: ONE forward-facing center camera (camera "2"); its frame is
    streamed to BOTH eyes (0 and 1) so the connectome's full bilateral
    retina sees the same view. --eyes stereo restores the +-35 deg pair.
  - --airframe wing swaps the quad for a fixed-wing point-mass flight
    model on the same brain channels: throttle -> airspeed (10-24 m/s,
    stall below 10), pitch -> elevator (climb/sink), roll -> bank angle
    (banked coordinated turns), yaw -> rudder. SimpleFlight does attitude
    animation; the model does the flight physics. Respawns are airborne
    catapult launches at cruise speed (planes don't hover-takeoff).
  - the connectome decode cannot hover by construction (no altitude
    feedback in the motor pools), so the bridge keeps the drone inside a
    ground/ceiling safety band: the brain's throttle FULLY owns climb and
    descent; the assist only pushes back within 1 m of the band edges.
    --no-assist removes even that.
  - collisions punish (default -2.5) and respawn; parked-car touches reward
    (default +2.5) and respawn. --punish-mag / --reward-mag tune magnitudes.
  - proximity shaping: every 0.5 s a small reward pulse proportional to
    closeness to the nearest parked car (3D distance), on two scales — a
    far gradient (40 m, small) to guide the brain toward a street with
    cars, and a steeper near gradient (8 m, stronger) for the final
    approach. The brain's relative reward shaping turns this into
    "closing in on cars is good, retreating is bad". --prox-gain 0 off.
  - --cars turns this into a full car-crash curriculum: the respawn point
    becomes a fresh 14-22 m start next to the target car facing it, and
    near the target the shaping switches to pure progress — every tick
    that CLOSES distance pulses a small reward; hovering/retreating sends
    nothing. Run 1 (fixed spawn, 359 episodes) proved approach rewards
    alone never bridge the last meters to the jackpot.

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
import random
import struct
import sys
import time
from pathlib import Path
import re

import numpy as np
import websockets

# ---- the training objective (car-crash curriculum) -------------------------
# Everything below turns "crash into cars" into a shaped learning problem:
#   touch car  -> +2.5 (reward-mag) jackpot + respawn
#   near field -> approach gradient up to +1.0 (near-max) within 8 m
#   far field  -> approach gradient up to +0.5 (prox-max) within 40 m
#   anything else -> -2.5 (punish-mag) and respawn

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
ALT_P_GAIN = 0.9       # lane-mode altitude controller (m/s per m of error)
YAW_RATE_MAX = 200.0   # deg/s at |yaw stick| = 1 — drastic turns
STICK_GAIN = 2.0       # sensitivity: amplify the brain's small channel wiggles
STICK_TAU_S = 0.15     # RC stick inertia (first-order low-pass on each stick)
CMD_HOLD_S = 0.25      # velocity command hold (re-sent every CMD_PERIOD_S)
LANE_EXPO = 0.60       # lane curve: lane = min + range * expo(throttle, 0.60)
                       # (0 = linear; <1 pushes low lanes together so small
                       # throttle dips reach street level where the cars are)

# ---- fixed-wing flight model (--airframe wing) ------------------------------
# Stock AirSim has no fixed-wing physics, so the bridge flies the Wing1
# SimpleFlight vehicle through a point-mass wing model implemented on top of
# the velocity API: banked coordinated turns, throttle -> airspeed, and
# climb-bleeds-speed energy coupling. The brain flies something that behaves
# like a plane: it cannot hover, must keep flying to stay up, and turns by
# banking.
WING_V_MIN = 10.0      # stall speed (m/s): below this the wing drops
WING_V_MAX = 24.0      # airspeed at full throttle
WING_CLIMB_MAX = 4.0   # m/s climb at full up-elevator (at cruise speed)
WING_SINK_MAX = 5.0    # m/s descent at full down-elevator
WING_ROLL_MAX = 45.0   # bank angle at full aileron
WING_TURN_RATE = 70.0  # deg/s cap on the banked turn rate
WING_ENERGY = 0.55     # climb costs airspeed (m/s per m/s of climb)
WING_RUDDER_RATE = 15.0  # deg/s of extra yaw from the rudder stick

BRAIN_HOVER = 0.29                     # brain throttle default (drone.json)


def expo(x: float, e: float) -> float:
    """RC expo curve: e=0 -> linear, 0<e<1 -> compressed near 0."""
    s = clamp(x, 0.0, 1.0)
    return (1.0 - e) * s + e * s * s * s

RESPAWN_ALT = 8.0      # respawn height above ground (m)
CAR_SPAWN_ALT = 6.0    # respawn height in cars mode (m) — cars visible early
CAR_SPAWN_DIST = (14.0, 22.0)  # spawn-distance range from the target car (m)
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
        self._ep_origin = (0.0, 0.0)   # XY the current episode started at
        self._ws_ref = None
        self.stick = {"fwd": 0.0, "lat": 0.0, "climb": 0.0, "yaw": 0.0}
        self._stick_at = 0.0
        self._hud = (0.0, 0.0, 0.0, 0.0)
        # fixed-wing state
        self._wing_v = 15.0    # current airspeed (m/s)
        self._wing_roll = 0.0  # current bank (deg)
        self._last_wing_at = 0.0
        self.lane = self.lane_from_throttle(BRAIN_HOVER)
        self._lane_at = 0.0
        self.car_poses: list[tuple[float, float, float]] = []
        self._target_idx = 0        # which parked car is the current target
        self._car_pulses = 0        # approach pulses on the current target
        self._last_target_near = float("inf")
        self._near_car = float("inf")
        self._prox_val = 0.0
        self._last_prox_t = 0.0
        # per-episode stats (reset in end_episode)
        self._ep_start = time.perf_counter()
        self._ep_near_sum = 0.0
        self._ep_near_n = 0
        self._ep_near_min = float("inf")
        self._ep_dopa_sum = 0.0
        self._ep_dopa_n = 0

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
        # car poses BEFORE setup_vehicle: the wing's catapult respawn already
        # needs the car list (and the quad's takeoff runs a velocity climb)
        self.cache_car_poses(client)
        self.setup_vehicle(client)
        if args.airframe == "wing":
            print(f"[wing] airframe=wing: throttle->airspeed, elevator->"
                  f"climb, aileron->banked turn; stall below {WING_V_MIN:.0f} "
                  f"m/s; catapult respawns", flush=True)
        if args.cars and self.car_poses:
            self._target_idx = random.randrange(len(self.car_poses))
            print(f"[cars] training target: car #{self._target_idx} "
                  f"(of {len(self.car_poses)}); each respawn starts a fresh "
                  f"14-22 m approach", flush=True)
        self._ep_origin = (self.spawn.x_val, self.spawn.y_val)

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

    def cache_car_poses(self, client) -> None:
        """World poses of the parked Car_* meshes (static scenery in
        AirSimNH). Fetched once at startup; the per-tick nearest-car distance
        is then pure math, no RPC. Powers the proximity reward shaping."""
        try:
            names = [n for n in client.simListSceneObjects("[Cc]ar.*")
                     if CAR_NAME_RE.match(n)]
            for name in names:
                p = client.simGetObjectPose(name).position
                self.car_poses.append((p.x_val, p.y_val, p.z_val))
        except Exception as exc:
            print(f"[airsim] car pose cache failed ({exc!r}); "
                  f"proximity reward disabled", flush=True)
        print(f"[airsim] tracking {len(self.car_poses)} parked cars "
              f"for proximity reward", flush=True)

    def setup_vehicle(self, client) -> None:
        client.enableApiControl(True, self.args.vehicle)
        client.armDisarm(True, self.args.vehicle)
        self.takeoff(client)

    def takeoff(self, client) -> None:
        """Quad: un-stick (spawn can be below the collision surface, which
        pins the physics resolver), teleport to a clean absolute altitude,
        then climb a little. Absolute z (ground-z - 8) instead of relative,
        so a spawn captured mid-air can never push the drone above the
        ceiling. Wing: airplanes don't take off from hover — catapult."""
        if self.args.airframe == "wing":
            self.respawn(client)
            return
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
        # leash is measured from where THIS episode started (cars mode
        # teleports anywhere in the map; a fixed spawn leash would instantly
        # border-punish every episode)
        dx = kin.position.x_val - self._ep_origin[0]
        dy = kin.position.y_val - self._ep_origin[1]
        dist = math.hypot(dx, dy)
        reason = None
        # ceiling punish sits 25 cm inside the blended band edge, so a
        # saturated climb that equilibrates right at the ceiling still counts.
        # The wing turns with up to 45 deg of bank, which bulges its path
        # outward ~1.6x, and it cannot stop — so its ceiling sits higher and
        # the ride tolerance is 3x longer before punishing.
        wing = self.args.airframe == "wing"
        ceil_m = self.args.max_alt + (5.0 if wing else 0.0)
        ceil_ride = self.args.ceil_ride * (3.0 if wing else 1.0)
        if alt >= ceil_m - 0.25:
            if self._ceil_since == 0.0:
                self._ceil_since = now
            elif now - self._ceil_since > ceil_ride:
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
            self.end_episode(reason)
            self.respawn(client)
            self._last_respawn = now

    def end_episode(self, why: str) -> None:
        """One stats line per episode, so training runs are comparable."""
        dur = time.perf_counter() - self._ep_start
        tel = self.telemetry or {}
        print(f"[episode] {why}  dur {dur:5.1f} s  hits {self.collision_count}  "
              f"cars {self.car_bumps}  ceilings {self.ceiling_hits}  "
              f"borders {self.border_hits}  "
              f"near-avg {self._ep_near_sum / max(self._ep_near_n, 1):5.1f} m  "
              f"near-min {self._ep_near_min:5.1f} m  "
              f"closing {self._car_pulses}  "
              f"dopa-avg {self._ep_dopa_sum / max(self._ep_dopa_n, 1):+.3f}",
              flush=True)
        self._ep_start = time.perf_counter()
        self._ep_near_sum = 0.0
        self._ep_near_n = 0
        self._ep_near_min = float("inf")
        self._ep_dopa_sum = 0.0
        self._ep_dopa_n = 0

    async def proximity_reward(self, ws, kin, now: float) -> None:
        """Continuous shaping stream: a small reward pulse every 0.5 s,
        proportional to closeness of the nearest parked car (3D distance).
        The brain's relative shaping (tau 12 s) adapts to any steady value,
        so what actually gets reinforced is the GRADIENT: closing in on a
        car drives dopamine up, drifting away drives it down — a guidance
        signal toward the +2.5 car-touch event, without drowning it.

        Two scales (car-crash training):
          far  — linear over --prox-radius (40 m), small (--prox-max 0.5):
                 gets the brain into the right street from cruise distance;
          near — steeper over --near-radius (8 m), stronger (--near-max 1.0):
                 a clear approach gradient once a car is actually in sight.
        The near max stays below the car-touch reward, so touching a car
        always pays more than hovering over it."""
        if self.args.prox_gain <= 0.0 or not self.car_poses:
            self._prox_val = 0.0
            return
        if now - self._last_prox_t < 0.5:
            return
        self._last_prox_t = now
        p = kin.position
        d2 = min((p.x_val - x) ** 2 + (p.y_val - y) ** 2 + (p.z_val - z) ** 2
                 for x, y, z in self.car_poses)
        self._near_car = math.sqrt(d2)
        if (self.args.cars and self.args.prox_gain > 0.0
                and self._near_car <= self.args.prox_radius):
            # near the training target: pure PROGRESS signal. Hovering and
            # retreating send nothing (the brain's relative shaping treats
            # silence as neutral); each 0.5 s tick that CLOSED distance
            # pulses a small reward proportional to meters gained. This is
            # the dense approach signal that leads to the +2.5 jackpot.
            if self._last_target_near == float("inf"):
                pass                        # first reading: baseline only
            elif self._near_car < self._last_target_near:
                gained = self._last_target_near - self._near_car
                v = min(self.args.closing_gain * gained, 0.5)
                self._car_pulses += 1
                self._prox_val = v
                await ws.send(json.dumps(
                    {"type": "reward", "value": round(v, 3)}))
            self._last_target_near = self._near_car
            return
        v = self.args.prox_gain * (
            self.args.prox_max
            * clamp(1.0 - self._near_car / self.args.prox_radius, 0.0, 1.0)
            + self.args.near_max
            * clamp(1.0 - self._near_car / self.args.near_radius, 0.0, 1.0))
        self._prox_val = v
        if v > 0.0:
            await ws.send(json.dumps({"type": "reward", "value": round(v, 3)}))

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
                await self.proximity_reward(ws, kin, now)
                if math.isfinite(self._near_car):
                    self._ep_near_sum += self._near_car
                    self._ep_near_n += 1
                    self._ep_near_min = min(self._ep_near_min, self._near_car)
                self._ep_dopa_sum += self.dopa
                self._ep_dopa_n += 1
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
        self.end_episode("collision" if not is_car else "car touch")
        self.respawn(client)
        self._last_respawn = now

    def respawn(self, client) -> None:
        """Cars mode: teleport next to a random parked car (the target is
        picked at startup), facing it, close enough that the car is inside
        the retina and the near-field gradient. Street mode: back to the
        spawn XY at the ABSOLUTE respawn altitude (never relative to
        wherever the drone was when the bridge started)."""
        pose = client.simGetVehiclePose(self.args.vehicle)
        wing = self.args.airframe == "wing"
        spawn_alt = (CAR_SPAWN_ALT if self.args.cars else RESPAWN_ALT) \
            + (6.0 if wing else 0.0)   # planes spawn higher: they can't hover
        if self.args.cars and self.car_poses:
            x, y, _z = self.car_poses[self._target_idx]
            ang = random.uniform(0.0, 2.0 * math.pi)
            dist = random.uniform(*CAR_SPAWN_DIST)
            pose.position.x_val = x + dist * math.cos(ang)
            pose.position.y_val = y + dist * math.sin(ang)
            pose.position.z_val = self.args.ground_z - spawn_alt  # NED
            yaw = math.atan2(y - pose.position.y_val, x - pose.position.x_val)
            pose.orientation = airsim.Quaternionr(
                0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        else:
            sp = self.spawn
            pose.position.x_val = sp.x_val
            pose.position.y_val = sp.y_val
            pose.position.z_val = self.args.ground_z - spawn_alt  # NED: up = -z
            pose.orientation = airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)  # level, initial heading
        client.simSetVehiclePose(pose, True, self.args.vehicle)
        self._ep_origin = (pose.position.x_val, pose.position.y_val)
        if self.args.airframe == "wing":
            # catapult launch: airborne at cruise speed, level wings
            self._wing_v = WING_V_MIN + 5.0
            self._wing_roll = 0.0
            client.moveByVelocityBodyFrameAsync(
                self._wing_v, 0.0, 0.0, 0.4,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, 0.0),
                self.args.vehicle,
            )
        # kill any stale brain stick bias so the fresh episode starts calm
        self.cmd = {"throttle": BRAIN_HOVER, "pitch": 0.0, "roll": 0.0, "yaw": 0.0}
        self.stick = {"fwd": 0.0, "lat": 0.0, "climb": 0.0, "yaw": 0.0}
        self._ceil_since = 0.0
        self._last_target_near = float("inf")   # fresh approach baseline
        self._car_pulses = 0
        self.lane = self.lane_from_throttle(BRAIN_HOVER)
        client.moveByVelocityBodyFrameAsync(
            0.0, 0.0, -1.5, 1.0,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, 0.0),
            self.args.vehicle,
        )

    # ---- act -------------------------------------------------------------
    CMD_PERIOD_S = 0.05  # 20 Hz command refresh (SimpleFlight holds the rest)

    def apply_wing_model(self, client, now: float) -> None:
        """Fixed-wing point-mass model on top of the SimpleFlight velocity
        API. The brain's channels become plane controls:
          throttle -> airspeed setpoint (WING_V_MIN..WING_V_MAX)
          pitch    -> elevator: climb (+) / sink (-); climbing bleeds speed
          roll     -> bank angle; bank rate = g*tan(bank)/V, capped — the
                      classic coordinated turn
          yaw      -> rudder: small extra yaw rate
        below stall speed the wing drops until speed recovers. SimpleFlight
        animates the attitude; this model does the flight dynamics."""
        if now - self._last_wing_at < self.CMD_PERIOD_S:
            return
        dt = now - self._last_wing_at if self._last_wing_at else 0.05
        self._last_wing_at = now
        c = self.cmd
        # airspeed follows the throttle channel
        v_t = WING_V_MIN + clamp(c["throttle"], 0.0, 1.0) \
            * (WING_V_MAX - WING_V_MIN)
        self._wing_v += (v_t - self._wing_v) * min(1.0, dt * 0.6)
        # elevator + energy coupling: climbing bleeds airspeed, diving adds
        climb_dem = clamp(c["pitch"], -1.0, 1.0)
        climb = climb_dem * (WING_CLIMB_MAX if climb_dem > 0 else WING_SINK_MAX)
        self._wing_v -= WING_ENERGY * max(climb, 0.0) * dt
        self._wing_v = clamp(self._wing_v, 6.0, WING_V_MAX + 2.0)
        stall = self._wing_v < WING_V_MIN
        if stall:
            climb = min(climb, -3.0)      # the wing drops
        # bank -> turn rate (coordinated turn), capped
        roll_t = WING_ROLL_MAX * clamp(c["roll"] * self.args.stick_gain,
                                       -1.0, 1.0)
        self._wing_roll += (roll_t - self._wing_roll) * min(1.0, dt * 2.5)
        turn = math.degrees(math.tan(math.radians(abs(self._wing_roll)))) \
            * 9.81 / max(self._wing_v, 8.0)
        turn = clamp(turn, 0.0, WING_TURN_RATE) * (1 if self._wing_roll >= 0 else -1)
        rudder = -clamp(c["yaw"], -1.0, 1.0) * WING_RUDDER_RATE
        # body-frame velocity: forward at airspeed, sideslip = bank
        fwd = self._wing_v * math.cos(math.radians(abs(self._wing_roll)))
        lat = self._wing_v * math.sin(math.radians(self._wing_roll))
        kin = client.simGetGroundTruthKinematics(self.args.vehicle)
        alt = -kin.position.z_val
        if alt < 0.8:                     # never terrace-crash into the ground
            climb = max(climb, 2.0)
        self._hud = (fwd, lat, climb, turn + rudder)
        client.moveByVelocityBodyFrameAsync(
            fwd, lat, -climb, CMD_HOLD_S + self.CMD_PERIOD_S,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(False, 0.0),   # yaw rate comes from bank + rudder
            self.args.vehicle,
        )

    def lane_from_throttle(self, throttle: float) -> float:
        """Throttle -> target altitude. RC expo curve (LANE_EXPO): the low
        half of the throttle range is compressed toward the floor, so the
        brain's small down-dips actually select street-level lanes where the
        cars are — without that, a high-lane cruising brain can never touch
        one (episode 1: 41 collisions, 0 car touches)."""
        frac = expo(clamp(throttle, 0.0, 1.0), LANE_EXPO)
        return self.args.min_alt + frac * (self.args.max_alt - self.args.min_alt)

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
        """FPV joystick scheme. fwd/lat/yaw are amplified, low-passed rate
        sticks; altitude depends on --control:
          lane (default) — the throttle channel SELECTS a target altitude
              lane and a P controller flies there (position control: an
              untrained climb bias holds a high lane instead of ascending
              forever, and ceiling punishes map onto the throttle values
              that chose them — clean credit assignment for R-STDP).
          rate — the old normalized climb-rate stick (needs --no-assist
              to be fully raw)."""
        if self.args.airframe == "wing":
            self.apply_wing_model(client, now)
            return
        if now - self._last_cmd_at < self.CMD_PERIOD_S:
            return
        self._last_cmd_at = now
        s = self.compute_sticks(now)
        if self.args.control == "lane" and not self.args.no_assist:
            kin = client.simGetGroundTruthKinematics(self.args.vehicle)
            alt = -kin.position.z_val
            # the selected lane glides RC-style, then the controller chases it
            lane_t = self.lane_from_throttle(self.cmd["throttle"])
            dt = clamp(now - self._lane_at, 0.001, 0.2) if self._lane_at else 0.05
            self._lane_at = now
            a = (1.0 if self.args.stick_tau <= 0
                 else min(1.0, dt / self.args.stick_tau))
            self.lane += (lane_t - self.lane) * a
            climb = clamp(ALT_P_GAIN * (self.lane - alt), -VZ_MAX, VZ_MAX)
            if alt < 0.8:                       # never burrow into the ground
                climb = max(climb, 2.0)
        else:
            climb = s["climb"]
            if not self.args.no_assist:
                # brain owns altitude; the band blends demand toward a safe
                # rate within 1 m of an edge and FULLY overrides past it
                kin = client.simGetGroundTruthKinematics(self.args.vehicle)
                alt = -kin.position.z_val
                over = (self.args.min_alt + 1.0) - alt
                if over > 0.0:                  # floor: blend to +2 m/s climb
                    f = clamp(over, 0.0, 1.0)
                    climb = climb * (1.0 - f) + 2.0 * f
                over = alt - (self.args.max_alt - 1.0)
                if over > 0.0:                  # ceiling: blend to -0.8 m/s
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
        wing = (f"v {self._wing_v:4.1f} m/s bank {self._wing_roll:+5.1f} deg  "
                if self.args.airframe == "wing" else "")
        print(f"[{'wing' if self.args.airframe == 'wing' else 'fly'}] "
              f"alt {alt:5.1f} m  lane {self.lane:5.1f} m  "
              f"spd {speed:4.1f} m/s  {wing}"
              f"sticks fwd {sticks[0]:+5.1f} lat {sticks[1]:+5.1f} "
              f"vz {sticks[2]:+5.1f} m/s yaw {sticks[3]:+6.1f} deg/s  "
              f"hits {self.collision_count}  cars {self.car_bumps}  "
              f"near {self._near_car:5.1f} m  min {self._ep_near_min:4.1f} m  "
              f"pulses {self._car_pulses:3d}  prox {self._prox_val:+.2f}  "
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
    ap.add_argument("--min-alt", type=float, default=1.0,
                    help="ground safety band lower edge (m); below car-roof "
                         "height (~1.5 m) so car touches are possible")
    ap.add_argument("--max-alt", type=float, default=30.0,
                    help="ceiling safety band upper edge (m); the wing gets "
                         "+5 m and 3x the ride tolerance automatically")
    ap.add_argument("--airframe", choices=["quad", "wing"], default="quad",
                    help="quad: SimpleFlight multirotor; wing: fixed-wing "
                         "flight model (throttle->airspeed, elevator->climb, "
                         "banked turns, stall, catapult respawns)")
    ap.add_argument("--stick-gain", type=float, default=STICK_GAIN,
                    help="sensitivity multiplier on every stick (pre-clamp)")
    ap.add_argument("--stick-tau", type=float, default=STICK_TAU_S,
                    help="RC stick inertia time constant (s); 0 = raw sticks")
    ap.add_argument("--control", choices=["lane", "rate"], default="lane",
                    help="lane: throttle selects a target altitude (position "
                         "control); rate: climb-rate stick (legacy)")
    ap.add_argument("--ceil-ride", type=float, default=CEILING_RIDE_S,
                    help="seconds riding the ceiling before punish+respawn")
    ap.add_argument("--border-radius", type=float, default=BORDER_RADIUS_M,
                    help="horizontal leash from spawn; beyond it = map border")
    ap.add_argument("--ground-z", type=float, default=GROUND_Z,
                    help="world-frame z of the ground plane at the spawn area")
    ap.add_argument("--prox-gain", type=float, default=1.0,
                    help="proximity-reward gain (0 disables the shaping)")
    ap.add_argument("--prox-radius", type=float, default=40.0,
                    help="nearest-car distance beyond which prox reward is 0 (m)")
    ap.add_argument("--prox-max", type=float, default=0.5,
                    help="proximity reward pulse value at 0 m distance")
    ap.add_argument("--near-radius", type=float, default=8.0,
                    help="near-field approach-gradient radius (m)")
    ap.add_argument("--near-max", type=float, default=1.0,
                    help="near-field reward at 0 m (must stay below reward-mag "
                         "so touching a car still pays more than hovering over it)")
    ap.add_argument("--cars", action="store_true",
                    help="car-crash curriculum: every respawn starts a fresh "
                         "14-22 m approach to the current target car")
    ap.add_argument("--closing-gain", type=float, default=0.1,
                    help="cars mode: reward per meter closed toward the "
                         "target car (capped at 0.5 per pulse)")
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
