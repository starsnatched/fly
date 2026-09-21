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
      throttle     -> climb-rate demand, relative to the EPISODE'S
                      calibrated hover point (per-episode frozen trim,
                      like pitch/roll/yaw). Rate control: a resting
                      throttle pool holds altitude, deviations climb or
                      descend proportionally.
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
    descent; the assist only pushes back within ~1 m of the ceiling and
    0.8 m of the ground (it used to reach 3 m up and swallow every
    low-altitude descent command — the "never comes down" bug).
    --no-assist removes even that.
  - collisions punish (default -2.5) and respawn; parked-car touches reward
    (default +2.5) and respawn. --punish-mag / --reward-mag tune magnitudes.
  - approach shaping is PROGRESS-ONLY: each 0.5 s tick that closes
    horizontal distance to the target car pulses a small reward
    (--closing-gain/m, capped). Hovering/retreating sends nothing. The
    old continuous closeness gradient (paid for being near) is removed —
    it rewarded hovering and let altitude bobbing count as value.
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
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re

import numpy as np
import cv2
import websockets

# ---- the training objective (car-crash curriculum) -------------------------
# Everything below turns "crash into cars" into a shaped learning problem:
#   touch car  -> +2.5 (reward-mag) jackpot + respawn
#   progress   -> closing pulses (proximity_reward): paid for approach,
#                 never for position
#   anything else -> -2.5 (punish-mag) and respawn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import airsim  # noqa: E402  (vendored in scripts/airsim, MIT licensed)

# ---- embodiment constants (mirrors examples/drone-web) ---------------------
W, H = 192, 108
DEFAULT_VISION_HZ = 0           # 0 = stream eyes AS FAST as the sim renders
                                # (the value is a cap, never adds delay on
                                # top of capture time; this iGPU sim renders
                                # ~3.7 fps, which is the real ceiling)

# ---- fly vision (--eyes fly) ------------------------------------------------
# A housefly's visual system, as far as reasonably simulable with two perspective
# cameras: near-panoramic wrap-around binocular field (Musca sees ~270 deg front
# + a rear sliver), ommatidial optics (each facet's own PSF -> low acuity),
# and photoreceptor response tuned to a green world with a fast phasic
# (band-pass) transient channel — the motion pathway that feeds the lobula.
# The brain receives the RESULT (192x108 per eye, same wire format), so the
# connectome genuinely sees through fly eyes.
FLY_EYE_FOV_DEG = 130.0   # per-camera FOV: 2 x 130 front + 2 x ~55 rear wraps
                          # most of the sphere around the drone
FLY_EYE_REAR_YAW = 115.0  # rear pair yaw: 360 - 2*130 = 100 deg gap, covered
                          # by two 130-deg rear cameras at +-115 deg
FLY_EYE_PITCH = 0.0       # level with the horizon (flies ride level)
FLY_PHOT_GAIN = 1.25      # photoreceptor gain (bumped: R1-6 press the signal
                          # into a narrower range -> darker eyes than a camera)
FLY_PHOT_SIGMA = 0.42     # photoreceptor soft-saturation (Naka-Rushton-ish)
FLY_GREEN_W = (0.25, 0.70, 0.05)  # R1-6 weighting: strongly green-weighted
                          # (the fly's peak sensitivity, and its favorite color)
FLY_TRANSIENT_TAU_S = 0.055  # LMC-style high-pass: this much of the previous
                          # response is subtracted each frame (phasic channel)
FLY_TRANSIENT_GAIN = 2.2  # transient (motion) channel strength in the mix
FLY_BLUR_SIGMA = 0.9      # ommatidial PSF blur (pixels @192x108): facet
                          # diffraction/aberration -> acuity ~a few degrees
FLY_NOISE = 0.015         # photon-shot-like noise floor
FLY_VIEWER_PORT = 8795    # built-in MJPEG viewer (http://localhost:8795)
FLY_PANO_W = 720          # panorama width: 2 deg per pixel over 360 deg
FLY_PANO_H = 120          # panorama height: +-60 deg vertical field
FLY_EYE_CAMS = ("0", "2", "3", "1")   # front-left, front-right,
                          # rear-left, rear-right (wrap/camera order)
FLY_EYE_WINDOW_DEG = 240.0  # per-eye angular width: wide fly-like field
FLY_EYE_OFFBORE_DEG = 30.0  # each window is centered this far off
                            # boresight (left eye left, right eye right)
                            # for stereo; forward flow then radiates from
                            # near the retina CENTER — the geometry the
                            # brain has always seen (hemisphere-per-eye
                            # put the front at the retina edge and the
                            # brain's forward prior broke: it flew
                            # backward)

# ---- AirSim joystick envelope (SimpleFlight) -------------------------------
V_FWD_MAX = 12.0       # m/s at |pitch stick| = 1 (web convention: -pitch = fwd)
CMD_HOLD_S = 0.5       # SimpleFlight velocity-command hold (s): > 1/cmd period
                       # so consecutive 20 Hz commands chain without gaps
V_LAT_MAX = 9.0        # m/s at |roll stick| = 1 — strong avoidance authority
VZ_MAX = 9.0           # m/s climb/descent demand at |climb stick| = 1
ALT_P_GAIN = 1.6       # lane-mode altitude controller (m/s per m of error)
YAW_RATE_MAX = 350.0   # deg/s at |yaw stick| = 1 — race-quad yaw
YAW_HOLD_EPS = 40.0    # below this yaw demand: rate mode off (the
                       # flight controller damps rotation) — a units
                       # translation, applied to the brain's own value
STICK_GAIN = 3.0       # sensitivity: amplify the brain's small channel wiggles
                       # (0.33 channel deviation from trim = full deflection)
STICK_TAU_S = 0.15     # RC stick inertia (first-order low-pass on each stick)
TRIM_SAMPLE_S = 1.5    # per-episode trim calibration window (s): the median
                       # of resting samples becomes the channel center for
                       # the WHOLE episode (frozen — unlike a continuous EMA
                       # trim, sustained flight commands are never absorbed)
TRIM_QUIET_STDEV = 0.06  # a calibration window with more channel movement
                       # than this is a command, not rest — resample
TRIM_MAX_WINDOWS = 4   # calibration attempts before accepting the median
STICK_DEADBAND = 0.05  # channel deadband around the trim center (stick units)
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

BRAIN_HOVER = 0.29                     # fallback hover throttle if the
                                       # physical probe fails. Throttle is
                                       # ABSOLUTE brain output: 0 = no
                                       # throttle = sink at the aircraft's
                                       # physical rate; the MEASURED hover
                                       # point (below) is the zero-climb
                                       # reference, not any brain statistic.
PHYS_HOVER_DEFAULT = 0.30              # probe overwrites at sim connect


def expo(x: float, e: float) -> float:
    """RC expo curve: e=0 -> linear, 0<e<1 -> compressed near 0."""
    s = clamp(x, 0.0, 1.0)
    return (1.0 - e) * s + e * s * s * s

RESPAWN_ALT = 8.0      # respawn height above ground (m)
CAR_SPAWN_ALT = 6.0    # respawn height in cars mode (m) — cars visible early
CAR_SPAWN_DIST = (14.0, 22.0)  # spawn-distance range from the target car (m)
GROUND_Z = 0.0         # world-frame z of the ground plane (--ground-z)
RESPAWN_COOLDOWN_S = 2.5   # ignore collisions this long after a respawn
SPAWN_HOLD_S = 3.0   # hold-level at episode start until the brain commands
                     # up (anti-cold-start; the brain owns the aircraft the
                     # moment it wants to climb or the timer ends)

# ---- punishment protocol (dopa-tail-aware) ----------------------------------
# Measured live on the wing brain (cengine source + dopamine probes):
#   * dan_drive clamps at +-1.5 and decays with tau 300 ms, so a pulse TRAIN
#     only sustains the suppression — the MAGNITUDE of one deep step is the
#     teaching lever (-5.0 gave a 77% deeper dopamine window than -2.5);
#   * the negative dopamine error then lasts 4.5-6 s (fast drive + slow
#     baseline re-adaptation), yet the old protocol respawned ~0.3 s after
#     the crash — dumping the tail onto the NEXT episode's good flying and
#     giving the crash-context synapses less LTD than they should get.
# New protocol: one deep pulse, then HOLD the crash scene (retina keeps the
# context, motors frozen so recovery isn't accidentally punished) until the
# dopamine error recovers — only then respawn onto a clean slate.
PUNISH_MAG_DEFAULT = -2.5   # was -5: crash streams outgunned every reward
                            # (net-negative dopamine => pool treadmill). -2.5
                            # still dwarfs the +0.2 altitude hill but lets
                            # minutes of good flight outweigh one mistake
REWARD_MAG_DEFAULT = 2.5
ALT_GAIN_DEFAULT = 0.35     # was 0.2: the altitude hill is the throttle
                            # pool's main positive teacher
STUN_HOLD_S = 4.5           # minimum scene-hold after a punish (dopa tail 4.5-6 s)
STUN_MAX_S = 6.5            # hard cap; dopa recovery usually releases earlier
DOPA_RECOVER = -0.03        # respawn gate: dopa error above this = tail drained
POSITIVE_HOLD_S = 1.5       # shorter hold after car rewards (positive tail)

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


# ---- fly-vision optics (--eyes fly) -----------------------------------------
# The four fisheye captures are stitched into a wrap-around panorama, then
# processed the way a fly's visual system is: spectral weighting (R1-6 favor
# green), ommatidial blur (facet PSF -> low acuity, wide acceptance angle),
# Naka-Rushton photoreceptor saturation, and a phasic high-pass channel
# (LMC-style) that emphasizes CHANGE - the motion pathway the lobula reads.

def _fly_wrap(rgbs: list[np.ndarray | None]) -> np.ndarray:
    """Stitch the four fisheye captures into one 360x120 deg panorama.
    Panorama x maps 180 deg per half-width (retina pixel density), y maps
    +-60 deg. Each camera only paints the sector its optical axis covers;
    remap with BORDER_TRANSPARENT leaves the rest untouched."""
    pano = np.zeros((FLY_PANO_H, FLY_PANO_W, 3), dtype=np.uint8)
    ys, xs = np.mgrid[0:FLY_PANO_H, 0:FLY_PANO_W].astype(np.float32)
    dx = (xs - 0.5 * FLY_PANO_W) / (0.5 * FLY_PANO_W) * 180.0   # -180..180
    py = (0.5 - ys / FLY_PANO_H) * 120.0                        # +60..-60
    for img, yc in zip(rgbs, (0.0, 180.0,
                              -FLY_EYE_REAR_YAW, FLY_EYE_REAR_YAW)):
        if img is None:
            continue
        hs, ws_ = img.shape[:2]
        cx = (dx - yc) / (0.5 * FLY_EYE_FOV_DEG) * (0.5 * ws_) + 0.5 * ws_
        cy = (0.5 * FLY_EYE_FOV_DEG - py) / FLY_EYE_FOV_DEG * hs
        cv2.remap(img, cx, cy, cv2.INTER_LINEAR, dst=pano,
                  borderMode=cv2.BORDER_TRANSPARENT)
    return pano


_FLY_PREV: dict[str, np.ndarray | None] = {"L": None, "R": None}


def _fly_photoreceptors(rgb: np.ndarray) -> np.ndarray:
    """Green-weighted R1-6 response with soft saturation and shot noise.
    Returns float32 [0,1] at retina resolution."""
    lin = (rgb.astype(np.float32) / 255.0) ** 2.2          # sRGB -> linear
    g = lin @ np.array(FLY_GREEN_W, dtype=np.float32)      # spectral weighting
    v = np.clip(g * FLY_PHOT_GAIN, 0.0, None)
    resp = v / (v + FLY_PHOT_SIGMA)                        # Naka-Rushton
    resp = np.clip(resp + np.random.normal(0.0, FLY_NOISE, resp.shape),
                   0.0, 1.0).astype(np.float32)
    return cv2.resize(resp, (W, H), interpolation=cv2.INTER_AREA)


def _fly_eye(pano: np.ndarray, right: bool, key: str) -> np.ndarray:
    """One eye's full pipeline: wide off-boresight window, photoreceptors,
    ommatidial blur, phasic transient mix. Output u8 (H, W) - exactly the
    frame the brain's retina receives.

    Geometry: each eye samples a 240-deg-wide, full-height window of the
    wrap-around panorama centered FLY_EYE_OFFBORE_DEG off boresight (left
    eye to the left, right eye to the right). The front therefore lands
    near the retina CENTER in both eyes, so forward optic flow expands
    centrally the way the connectome has always seen it, while the pair
    still wraps ~300 deg around the drone (rear +-30 deg gap). Images are
    unmirrored (world-left = image-left, camera convention)."""
    az_c = -FLY_EYE_OFFBORE_DEG if not right else FLY_EYE_OFFBORE_DEG
    cx = 0.5 * FLY_PANO_W + az_c / (360.0 / FLY_PANO_W)   # pano col (2 deg/px)
    off = (np.arange(W, dtype=np.float32) - 0.5 * W) \
        * (FLY_EYE_WINDOW_DEG / W)                        # deg from center
    cols = np.clip(cx + off / (360.0 / FLY_PANO_W), 0, FLY_PANO_W - 1)
    rows = np.arange(H, dtype=np.float32) * (FLY_PANO_H / float(H))
    map_x = np.tile(cols[None, :], (H, 1)).astype(np.float32)
    map_y = np.tile(rows[:, None], (1, W)).astype(np.float32)
    crop = cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR)
    phot = _fly_photoreceptors(crop)
    lam = cv2.GaussianBlur(phot, (0, 0), FLY_BLUR_SIGMA)
    prev = _FLY_PREV[key]
    _FLY_PREV[key] = lam
    if prev is not None:
        tran = np.clip((lam - prev) * FLY_TRANSIENT_GAIN, -1.0, 1.0)
    else:
        tran = np.zeros_like(lam)
    out = np.clip(lam + 0.35 * tran, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def fly_eye_left(pano: np.ndarray) -> np.ndarray:
    return _fly_eye(pano, right=False, key="L")


def fly_eye_right(pano: np.ndarray) -> np.ndarray:
    return _fly_eye(pano, right=True, key="R")


_FLY_VIEWER_HTML = b"""<!doctype html>
<html><head><title>fly vision</title><style>
 body{background:#0c0f0c;color:#9f9;font-family:ui-monospace,monospace;margin:16px}
 h3{margin:4px 0} img{image-rendering:pixelated;width:768px;max-width:96vw;
 border:1px solid #2a3a2a;border-radius:6px} p{color:#5f7f5f;max-width:768px}
</style></head><body>
<h3>what the fly brain sees</h3>
<img id=f src="/frame?t=0">
<p>top: the two retinas after full fly-vision processing (300-deg
wrap-around windows centered 30 deg off boresight, ommatidial optics,
green-weighted photoreceptors, phasic motion channel) &mdash; LEFT EYE |
RIGHT EYE, each the exact 192x108 frame streamed to the connectome.
bottom: the raw wrap-around panorama before the optics.</p>
<script>setInterval(()=>{f.src='/frame?t='+Date.now()},66)</script>
</body></html>"""


_FLY_NOFRAME_JPEG: bytes | None = None   # placeholder before first frame


class _FlyViewerHandler(BaseHTTPRequestHandler):
    bridge: "Bridge" | None = None

    def do_GET(self):  # noqa: N802 (http.server API)
        try:
            self._do_get()
        except (BrokenPipeError, ConnectionResetError):
            pass   # viewer tab polling faster than we produce frames

    def do_POST(self):  # noqa: N802 (http.server API)
        # /cmd lands here (do_GET's router checks self.command)
        self.do_GET()

    def _do_get(self):
        if self.path.startswith("/frame"):
            with self.bridge._fly_state_lock:
                jpeg = self.bridge._fly_jpeg
            if jpeg is None:
                jpeg = _FLY_NOFRAME_JPEG   # placeholder, never an error
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
        elif self.path in ("/", "/fly"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_FLY_VIEWER_HTML)))
            self.end_headers()
            self.wfile.write(_FLY_VIEWER_HTML)
        elif self.path == "/cmd" and self.command == "POST":
            # simple human commands: {"cmd": "..."} or {"cmd": "car N"} /
            # {"cmd": "hover"}. Curriculum + training signals only.
            try:
                n = int(self.headers.get("Content-Length", 0) or 0)
                obj = json.loads(self.rfile.read(n) or b"{}")
                text = str(obj.get("cmd", "")).strip()
                reply = self.bridge._handle_command(text)
            except Exception as exc:                      # noqa: BLE001
                reply = f"error: {exc!r}"
            body = json.dumps({"ok": not reply.startswith("error"),
                               "reply": reply}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            b = self.bridge
            body = json.dumps({
                "alt": round(b._last_alt or 0.0, 1),
                "near": (round(b._near_car, 1)
                         if b._near_car != float("inf") else None),
                "target": b._target_idx,
                "cars": b.car_bumps,
                "hits": b.collision_count,
                "dopa": round(b.dopa, 3),
                "commands": b._cmd_q,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *args):  # silence request logging
        pass


def start_fly_viewer(bridge: "Bridge", port: int) -> None:
    if port <= 0:
        return
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), _FlyViewerHandler)
    except OSError as exc:
        print(f"[viewer] port {port} unavailable ({exc!r}); "
              f"fly viewer disabled", flush=True)
        return
    global _FLY_NOFRAME_JPEG
    if _FLY_NOFRAME_JPEG is None:
        ok, buf = cv2.imencode(
            ".jpg", np.full((228, 384, 3), 30, np.uint8))
        _FLY_NOFRAME_JPEG = buf.tobytes() if ok else b""
    _FlyViewerHandler.bridge = bridge
    threading.Thread(target=srv.serve_forever, daemon=True,
                     name="fly-viewer").start()
    print(f"[viewer] fly vision at http://localhost:{port} "
          f"(exactly what the brain's retinas receive)", flush=True)


def _stdev(xs: list[float]) -> float:
    """Population standard deviation (trim quietness check)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


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
        self._last_tel_req = 0.0
        self._last_status = 0.0
        self._actions_at = 0.0
        self._last_col_print = 0.0
        self._last_cmd_at = 0.0
        self._grounded_since = 0.0
        self._stun_until = 0.0    # event stun: scene held until this time
        self._stun_pending = 0.0  # stun start ts (0 = no respawn pending)
        self._sim_ready = False   # one-time sim setup done (survives retries)
        self._stun_mag = 0.0      # signed event magnitude, for the status line
        self._fly_state: dict | None = None  # fly-vision viewer state
        self._fly_state_lock = threading.Lock()
        self._fly_jpeg: bytes | None = None
        self._vision_hz_ema = 0.0   # achieved eye rate (captures are the
                                    # pacing on slow sims — see flight_loop)
        self._last_hide_assert = 0.0
        self._stun_why = ""
        self._ep_origin = (0.0, 0.0)   # XY the current episode started at
        self._ws_ref = None
        self.stick = {"fwd": 0.0, "lat": 0.0, "climb": 0.0, "yaw": 0.0}
        self._stick_at = 0.0
        # auto-trim: each attitude channel's resting value (None = pending
        # first brain action); centered sticks are deflections FROM this
        self.trim = {"pitch": None, "roll": None, "yaw": None}
        self._trim_frozen = False
        self._trim_at = 0.0
        self._trim_samples: list[tuple[float, dict[str, float]]] = []
        self._trim_windows = 0
        self._hud = (0.0, 0.0, 0.0, 0.0)
        # parallel eye capture pool: one RPC connection+thread per camera.
        # The sim's RPC serializes per connection, but its render pipeline
        # takes captures CONCURRENTLY — 4 sockets fetch ~4x faster than one
        # (measured 58-66 vs 10-15 captures/s). Threads die with the sim and
        # are rebuilt on reconnect (the old client objects just fail).
        self._eye_pool: list | None = None
        self._jpeg_n = 0      # viewer JPEG encoded every 2nd eye frame
        self._hover_thr = PHYS_HOVER_DEFAULT  # MEASURED at sim connect
        self._hover_thr = PHYS_HOVER_DEFAULT  # MEASURED at sim connect
        # fixed-wing state
        self._wing_v = 15.0    # current airspeed (m/s)
        self._wing_roll = 0.0  # current bank (deg)
        self._last_wing_at = 0.0
        self.car_poses: list[tuple[float, float, float]] = []
        self._target_idx = 0        # which parked car is the current target
        self._car_pulses = 0        # approach pulses on the current target
        self._last_target_near = float("inf")
        # simple human command interface: viewer HTTP POSTs land on this
        # queue and are consumed between control ticks. Commands set the
        # CURRICULUM and TRAINING SIGNALS (which car is the goal, spawn
        # where, jackpot on/off); they never write actuator values — the
        # connectome keeps 100% of the flying.
        self._cmd_q: list[dict] = []
        self._jackpot_enabled = True
        self._car_paid: set[int] = set()   # targets already paid this run
        # spawn hold: for the first moments of an episode the aircraft is
        # HELD LEVEL (never sinks) unless the brain commands climb. This is
        # spawn initialization (like the face-car heading), not behavior:
        # the instant the brain commands up — or the timer ends — it owns
        # the aircraft completely, and NO shaping is paid during the hold,
        # so the circuit isn't rewarded for the bridge's help either.
        self._spawn_hold_until = 0.0
        self._cmd_client = None            # sim client for manual reset
        self._near_car = float("inf")
        self._prox_val = 0.0
        self._last_alt: float | None = None
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
        """Everything — sim connect, one-time setup, brain link — lives inside
        the retry loop, so a sim crash (the iGPU AirSimNH build dies under
        long sessions) heals automatically: the bridge waits for the RPC
        port, re-attaches, and resumes training without a restart."""
        args = self.args
        client = airsim.MultirotorClient(ip=args.airsim_ip, port=args.airsim_port)

        delay = RETRY_BASE_S
        while True:
            try:
                client.confirmConnection()
                vehicles = client.listVehicles()
                if args.vehicle not in vehicles:
                    print(f"[airsim] vehicle {args.vehicle!r} not in "
                          f"{vehicles}; waiting", flush=True)
                    raise RuntimeError("vehicle missing")
                if not self._sim_ready:
                    print(f"[airsim] connected; vehicles: {vehicles}", flush=True)
                    self.spawn = client.simGetGroundTruthKinematics(
                        args.vehicle).position
                    # car poses BEFORE setup_vehicle: the wing's catapult
                    # respawn already needs the car list (and the quad's
                    # takeoff runs a velocity climb)
                    self.cache_car_poses(client)
                    self.setup_vehicle(client)
                    if args.eyes == "fly":
                        self.apply_fly_cameras(client)
                    if args.gfx:
                        self.apply_low_gfx(client)
                    if args.airframe == "wing":
                        print(f"[wing] airframe=wing: throttle->airspeed, "
                              f"elevator->climb, aileron->banked turn; stall "
                              f"below {WING_V_MIN:.0f} m/s; catapult respawns",
                              flush=True)
                    if args.cars and self.car_poses:
                        self._target_idx = random.randrange(len(self.car_poses))
                        print(f"[cars] training target: car #{self._target_idx} "
                              f"(of {len(self.car_poses)}); each respawn starts "
                              f"a fresh 14-22 m approach", flush=True)
                    self._ep_origin = (self.spawn.x_val, self.spawn.y_val)
                    self._cmd_client = client   # for manual reset commands
                    self._sim_ready = True
                else:
                    # Sim may have restarted while the bridge lived on: a
                    # fresh sim resets cameras/quality/visibility. Detect
                    # it cheaply via the fly-eye FOV and redo full setup.
                    fresh = False
                    try:
                        fov = client.simGetCurrentFieldOfView("0", args.vehicle)
                        fresh = (args.eyes == "fly"
                                 and abs(fov - FLY_EYE_FOV_DEG) > 1.0)
                    except Exception:
                        pass
                    if fresh:
                        print("[airsim] fresh sim detected (camera FOV "
                              f"{fov:.0f} != fly {FLY_EYE_FOV_DEG:.0f}); "
                              "re-running setup", flush=True)
                        self._sim_ready = False
                        self._eye_pool = None   # workers ride the dead sim
                        continue
                    if args.eyes == "fly":
                        self._hide_own_body(client)
                print(f"[bridge] connecting to brain at {args.brain_ws}")
                async with websockets.connect(
                        args.brain_ws, open_timeout=30, max_size=32 * 2**20) as ws:
                    await self.handshake(ws)
                    delay = RETRY_BASE_S
                    await self.flight_loop(client, ws)
            except KeyboardInterrupt:
                return 0
            except Exception as exc:  # sim crash / brain restart / link loss
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
            self.car_poses.clear()   # reconnects re-enter here; never dupe
            for name in names:
                p = client.simGetObjectPose(name).position
                self.car_poses.append((p.x_val, p.y_val, p.z_val))
        except Exception as exc:
            print(f"[airsim] car pose cache failed ({exc!r}); "
                  f"proximity reward disabled", flush=True)
        print(f"[airsim] tracking {len(self.car_poses)} parked cars "
              f"for proximity reward", flush=True)

    def apply_fly_cameras(self, client) -> None:
        """Re-point the sim's four fisheye captures into a wrap-around
        compound-eye arrangement (two front, two rear, 130-deg facets).
        Runs once per sim session; settings.json keeps a plain copy so
        other clients are unaffected."""
        fov = FLY_EYE_FOV_DEG
        for name, yaw in zip(FLY_EYE_CAMS,
                             (0.0, 180.0, -FLY_EYE_REAR_YAW,
                              FLY_EYE_REAR_YAW)):
            client.simSetCameraPose(name, airsim.Pose(
                airsim.Vector3r(0.10, 0, 0),
                airsim.to_quaternion(FLY_EYE_PITCH, 0.0, math.radians(yaw))))
            try:
                client.simSetCameraFov(name, fov, self.args.vehicle)
            except Exception:
                pass   # older builds: settings.json default (90) still works
        print(f"[fly] compound-eye optics: 4 x {fov:.0f} deg facets "
              f"(front pair + rear pair at +-{FLY_EYE_REAR_YAW:.0f} deg) "
              f"-> wrap-around panorama", flush=True)
        # the other vehicle parked at the spawn area is a pseudo-body
        # staring into the rear facets — park it far from the training
        # neighborhood
        other = "Wing1" if self.args.vehicle != "Wing1" else "Fly1"
        try:
            op = client.simGetObjectPose(other)
            op.position.x_val += 400.0
            op.position.y_val += 400.0
            client.simSetObjectPose(other, op, True)
            print(f"[fly] parked {other} 400 m away (was in the eye view)",
                  flush=True)
        except Exception:
            pass
        self._hide_own_body(client)
        self._probe_hover_throttle(client)
    def apply_low_gfx(self, client) -> None:
        """Drop the game's rendering cost so the eye rate rises. Measured on
        the iGPU box: scalability floor 3.6 -> 8.3 captures/s (2.3x). Scene
        captures ignore window resolution and r.ScreenPercentage (no gain
        there), so only the quality floors are touched. Re-applied on every
        sim (re)connect."""
        for cmd in ("sg.ResolutionQuality 10", "sg.ShadowQuality 0",
                    "sg.EffectsQuality 0", "sg.PostProcessQuality 0",
                    "sg.TextureQuality 0", "sg.ViewDistanceQuality 0",
                    "sg.AntiAliasingQuality 0", "r.VSync 0",
                    "r.SSRQuality 0", "r.MotionBlurQuality 0",
                    "r.BloomQuality 0", "r.AmbientOcclusionLevels 0",
                    "t.IdleWhenNotForeground 0", "t.MaxFPS 0"):
            try:
                client.simRunConsoleCommand(cmd)
            except Exception:
                pass
        print("[gfx] render floor applied (scalability mins, vsync off, "
              "SSR/blur/bloom/AO off, no background throttle)", flush=True)

    def _probe_hover_throttle(self, client) -> None:
        """Measure the AIRCRAFT's hover throttle (an aircraft constant,
        like its mass — not a brain statistic): hold a test throttle with
        the low-level angle/throttle API and read the resulting climb rate.
        Two points give the hover point by interpolation and the climb
        slope sanity-checks VZ scaling. Falls back to PHYS_HOVER_DEFAULT
        on any failure. Runs during sim setup, BEFORE the brain gets
        control, so the flight controller's hover hold has settled."""
        try:
            client.enableApiControl(True, self.args.vehicle)
            client.armDisarm(True, self.args.vehicle)
            z0 = client.simGetGroundTruthKinematics(
                self.args.vehicle).position.z_val
            client.moveToZAsync(z0 - 8.0, 3.0, 12,
                                airsim.YawMode(False, 0.0), -1, 1,
                                self.args.vehicle).join()
            probe = []
            for thr in (0.50, 0.70):
                client.moveByRollPitchYawrateThrottleAsync(
                    0.0, 0.0, 0.0, thr, 1.6, self.args.vehicle)
                time.sleep(1.6)
                vz0 = client.simGetGroundTruthKinematics(
                    self.args.vehicle).linear_velocity.z_val
                client.moveByRollPitchYawrateThrottleAsync(
                    0.0, 0.0, 0.0, thr, 1.4, self.args.vehicle)
                time.sleep(1.4)
                vz1 = client.simGetGroundTruthKinematics(
                    self.args.vehicle).linear_velocity.z_val
                probe.append((thr, -(vz0 + vz1) * 0.5))  # NED z -> up+
            (t0, c0), (t1, c1) = probe
            if c1 > c0:      # climbing with more throttle: sane physics
                self._hover_thr = clamp(t0 + (0.0 - c0) * (t1 - t0) / (c1 - c0),
                                        0.15, 0.75)
            print(f"[fly] measured hover throttle {self._hover_thr:.3f} "
                  f"(probe: thr .50 -> climb {c0:+.1f} m/s, thr .70 -> "
                  f"climb {c1:+.1f} m/s)", flush=True)
        except Exception as exc:
            print(f"[fly] hover probe failed ({exc!r}); using fallback "
                  f"{PHYS_HOVER_DEFAULT}", flush=True)
            self._hover_thr = PHYS_HOVER_DEFAULT

    def _hide_own_body(self, client) -> None:
        """Hide this vehicle's pawn from every camera (--hide-body opt-in).
        The rear fly facets would stare straight back at the drone body;
        the off-boresight eye windows already exclude the rear hemisphere,
        so the default keeps the pawn VISIBLE for the third-person chase
        view. Uses the Unreal console ('ke <pawn> 0'); the vehicle stays
        fully RPC-controllable while hidden. Pose resets can restore the
        pawn, so this is re-run after every respawn."""
        if not self.args.hide_body:
            return
        try:
            client.simRunConsoleCommand(f"ke {self.args.vehicle} 0")
        except Exception as exc:
            print(f"[fly] body-hide failed ({exc!r}); rear facets may "
                  f"see the drone body", flush=True)

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
        if self.args.eyes == "fly":
            self._hide_own_body(client)   # pose sets re-show the pawn
        client.moveByVelocityBodyFrameAsync(
            0.0, 0.0, -2.0, 4.0,   # climb at 2 m/s for 4 s (NED: -vz = up)
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, 0.0),
            self.args.vehicle,
        ).join()

    def recover_if_stuck(self, client, kin, now: float) -> None:
        """Grounded recovery: if we sit at/below ground with no motion for a
        while (wedged, or knocked down), teleport up and take off again.
        Suppressed during a stun: a dazed crashed aircraft stays put until
        the dopamine tail has drained onto the crash context."""
        if self.in_stun():
            return
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
        if self.in_stun() or self._stun_pending:
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
            self.begin_event_stun(self.args.punish_mag, reason)
            print(f"[sim] {reason} -> punish {self.args.punish_mag:+.1f}; "
                  f"holding scene until dopa recovers (hits "
                  f"{self.collision_count}, ceilings {self.ceiling_hits}, "
                  f"borders {self.border_hits})",
                  flush=True)
            self.end_episode(reason)
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
        """Event-and-progress shaping only. The continuous Closeness
        gradient (a reward for merely BEING near a car, on far/near
        scales) was removed: it paid for hovering over the target, made
        altitude bobbing register as value, and taught "cars make dopamine
        happen" instead of "approaching cars is my doing". What remains:

          closing pulses — each 0.5 s tick that CLOSES horizontal distance
                to the current target car (by --closing-min) pulses a
                small reward proportional to meters gained. Progress, not
                position: hovering and retreating send nothing.
          altitude hill — the SIGNED band (peak +alt_gain at 8 m, penalty
                above 16 m), applied everywhere including the approach
                zone. Staying airborne is the prerequisite skill.
          car-touch jackpot — the +2.5 event (+2.5 one-time per target).

        The brain's relative shaping (tau 12 s) turns the closing pulses
        into "my approach produced this" credit; the HUD's `near` readout
        stays as pure telemetry (no reward attached)."""
        if not self.car_poses:
            self._prox_val = 0.0
            return
        if self.in_stun():
            return    # keep the event window clean of shaping pulses
        if now - self._last_prox_t < 0.5:
            return
        self._last_prox_t = now
        if now < self._spawn_hold_until:
            return    # bridge is holding the aircraft: pay no shaping, so
                      # the circuit isn't rewarded for the bridge's help
        p = kin.position
        d2 = min((p.x_val - x) ** 2 + (p.y_val - y) ** 2 + (p.z_val - z) ** 2
                 for x, y, z in self.car_poses)
        self._near_car = math.sqrt(d2)
        # altitude-band shaping (SIGNED hill, peak +alt_gain at 8 m): below
        # 16 m it grades to zero toward both ground and 16 m; ABOVE 16 m it
        # turns into a growing penalty (−alt_gain by 24 m). A neutral
        # ceiling taught nothing — the ratchet era ("only goes up") thrived
        # on rewards that stayed positive at altitude. Descent now shrinks
        # an active penalty, which the brain's relative shaping reads as a
        # positive rate-of-change: coming down is itself rewarded.
        alt_v = 0.0
        if self._last_alt is not None:
            alt = self._last_alt
            if alt <= 16.0:
                alt_v = self.args.alt_gain * clamp(
                    1.0 - abs(alt - 8.0) / 8.0, 0.0, 1.0)
            else:
                alt_v = -self.args.alt_gain * clamp(
                    (alt - 16.0) / 8.0, 0.0, 1.0)
        if alt_v != 0.0:
            await ws.send(json.dumps(
                {"type": "reward", "value": round(alt_v, 3)}))
        if not self.args.cars:
            return
        # closing-progress pulses on the CURRENT TARGET, horizontal only —
        # the approach task is XY, and using 3D distance let altitude
        # bobbing register as false progress. Each tick that closes at
        # least --closing-min meters pulses reward proportional to meters
        # gained; hovering and retreating send nothing.
        tx, ty, _tz = self.car_poses[self._target_idx]
        near_h = math.sqrt((p.x_val - tx) ** 2 + (p.y_val - ty) ** 2)
        if self._last_target_near == float("inf"):
            pass                        # first reading: baseline only
        elif self._last_target_near - near_h >= self.args.closing_min:
            gained = self._last_target_near - near_h
            v = min(self.args.closing_gain * gained, 0.5)
            self._car_pulses += 1
            self._prox_val = v
            await ws.send(json.dumps(
                {"type": "reward", "value": round(v, 3)}))
        self._last_target_near = near_h

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
        send_dt = (1.0 / args.vision_hz) if args.vision_hz > 0 else 0.0
        next_send = 0.0
        print(f"[bridge] streaming eyes (cap {args.vision_hz or 'uncapped'} Hz; "
              f"sim-paced on this machine); "
              f"assist={'off' if args.no_assist else 'on'}")

        reader = asyncio.create_task(self.reader_loop(ws))
        try:
            while True:
                now = time.perf_counter()

                # human commands (viewer /cmd) set curriculum + signals
                self._drain_commands()

                # stream the eye pair + body state
                if now >= next_send:
                    t_vision = now
                    await self.sense_and_stream(client, ws)
                    # vision-hz is a CAP: never add a sleep on top of
                    # capture time. On slow sims the capture IS the
                    # pacing; measure the achieved rate for the HUD.
                    dt_v = time.perf_counter() - t_vision
                    if dt_v > 0:
                        hz = 1.0 / dt_v
                        self._vision_hz_ema = (
                            hz if self._vision_hz_ema == 0.0
                            else 0.8 * self._vision_hz_ema + 0.2 * hz)
                    next_send = ((t_vision + send_dt)
                                 if send_dt > 0 else 0.0)

                # periodic control / telemetry / status
                self.apply_command(client, now)
                kin = client.simGetGroundTruthKinematics(self.args.vehicle)
                self.recover_if_stuck(client, kin, now)
                self.check_bounds(client, kin, now)
                # stun release: hold ended AND dopamine recovered -> respawn
                if self._stun_pending and not self.in_stun() \
                        and (self.dopa > DOPA_RECOVER
                             or now - self._stun_pending > self.args.stun_max):
                    print(f"[sim] stun over -> respawn "
                          f"(dopa {self.dopa:+.3f}, held "
                          f"{now - self._stun_pending:.1f} s)", flush=True)
                    self._stun_pending = 0.0
                    self.respawn(client)
                    self._last_respawn = now
                await self.proximity_reward(ws, kin, now)
                if math.isfinite(self._near_car):
                    self._ep_near_sum += self._near_car
                    self._ep_near_n += 1
                    self._ep_near_min = min(self._ep_near_min, self._near_car)
                self._ep_dopa_sum += self.dopa
                self._ep_dopa_n += 1
                if (self.args.eyes == "fly"
                        and now - self._last_hide_assert > 5.0):
                    self._last_hide_assert = now
                    self._hide_own_body(client)   # pose sets re-show it
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
        self._last_alt = alt
        vy_up = -kin.linear_velocity.z_val
        speed = math.sqrt(kin.linear_velocity.x_val ** 2
                          + kin.linear_velocity.y_val ** 2
                          + kin.linear_velocity.z_val ** 2)
        collided = bool(col.has_collided)

        if self.args.eyes == "fly":
            frames, pano = self.fly_eye_frames(client)
            self._hud_vision = pano
        elif self.args.eyes == "stereo":
            requests = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
            resps = client.simGetImages(requests)
            frames = [pack_eye_frame(eye, self._to_retina(rgb))
                      for eye, resp in enumerate(resps)
                      if (rgb := self.rgb_from_response(resp)) is not None]
        else:
            # ONE forward center camera ("2"); its frame feeds BOTH eyes so
            # the connectome's full bilateral retina sees the same view.
            resp = client.simGetImages(
                [airsim.ImageRequest("2", airsim.ImageType.Scene, False, False)])[0]
            rgb = self.rgb_from_response(resp)
            frames = ([pack_eye_frame(0, self._to_retina(rgb)),
                       pack_eye_frame(1, self._to_retina(rgb))]
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

    def _new_rpc_client(self):
        """A fresh RPC connection to the sim (parallel eye pool uses one
        per camera; see _fetch_rgb_parallel)."""
        return airsim.MultirotorClient(ip=self.args.airsim_ip,
                                       port=self.args.airsim_port)

    def _fetch_rgb_parallel(self) -> list[np.ndarray | None]:
        """Fetch one frame per eye camera CONCURRENTLY (a dedicated
        connection + worker thread per camera). Returns RGBs in
        FLY_EYE_CAMS order (None for a camera that failed).

        The bridge is async; these calls are BLOCKING, so the workers are
        background daemon threads with the latest result dropped into a
        slot — the event loop never waits on a capture."""
        if self._eye_pool is None:
            pool = []
            for name in FLY_EYE_CAMS:
                c = self._new_rpc_client()
                c.confirmConnection()
                req = [airsim.ImageRequest(
                    name, airsim.ImageType.Scene, False, False)]
                out: dict = {"rgb": None}
                ev = threading.Event()
                th = threading.Thread(
                    target=self._eye_worker, args=(c, req, out, ev),
                    daemon=True)
                th.start()
                pool.append((out, ev))
            self._eye_pool = pool
        for out, ev in self._eye_pool:
            ev.set()                      # request a fresh capture
        time.sleep(0.004)                 # workers run; loop stays async
        with self._fly_state_lock:
            return [out["rgb"] for out, _ in self._eye_pool]

    @staticmethod
    def _eye_worker(c, req, out: dict, ev: threading.Event) -> None:
        """Dedicated capture thread: one camera, one connection, forever
        (until the sim dies, which raises and quietly ends the thread; the
        pool is rebuilt on the next reconnect)."""
        while True:
            ev.wait()
            ev.clear()
            try:
                resp = c.simGetImages(req)[0]
                data = bytes(resp.image_data_uint8)
                n = resp.width * resp.height
                rgb = (np.frombuffer(data[:n * 3], dtype=np.uint8)
                       .reshape(resp.height, resp.width, 3)
                       if len(data) >= n * 3 else None)
                out["rgb"] = rgb          # atomic ref swap under the GIL
            except Exception:
                time.sleep(0.05)          # sim gone: idle until rebuilt

    def fly_eye_frames(self, client) -> tuple[list[bytes], np.ndarray]:
        """Fetch the four wrap-around captures, build the panorama, run
        the full fly-optics pipeline per eye, and return (packed retina
        frames for the brain, panorama for the viewer)."""
        rgbs = self._fetch_rgb_parallel()
        pano = _fly_wrap(rgbs)
        left = fly_eye_left(pano)
        right = fly_eye_right(pano)
        # viewer JPEG: retinas on top, raw panorama below
        lw = left.shape[1]
        top = np.concatenate([left, right], axis=1)
        cv2.putText(top, "LEFT EYE", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        cv2.putText(top, "RIGHT EYE", (lw + 4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        pano_bgr = cv2.cvtColor(pano, cv2.COLOR_RGB2BGR)
        pano_bgr = cv2.resize(pano_bgr, (top.shape[1], 120),
                              interpolation=cv2.INTER_AREA)
        self._jpeg_n += 1
        if self._jpeg_n % 2 == 0:         # viewer needs ~25 fps, not 60:
            vis = np.concatenate([        # skip alternate encodes
                cv2.cvtColor(top, cv2.COLOR_GRAY2BGR), pano_bgr], axis=0)
            ok, buf = cv2.imencode(".jpg", vis,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._fly_state_lock:
                    self._fly_jpeg = buf.tobytes()
        return [pack_eye_frame(0, np.repeat(left[:, :, None], 3, axis=2)),
                pack_eye_frame(1, np.repeat(right[:, :, None], 3, axis=2))], pano

    @staticmethod
    def rgb_from_response(resp) -> np.ndarray | None:
        """Native-size RGB capture (dimensions come from the response; the
        callers decide whether to keep them for supersampled optics or
        resize to the retina for the plain modes)."""
        data = bytes(resp.image_data_uint8)
        if not data:
            return None
        n = resp.width * resp.height
        if len(data) >= n * 3:
            return np.frombuffer(data[:n * 3], dtype=np.uint8).reshape(
                resp.height, resp.width, 3)
        return None

    @staticmethod
    def _to_retina(rgb: np.ndarray) -> np.ndarray:
        """Resize any capture to the exact (H, W, 3) retina frame the
        plain (stereo/center) eye modes pack."""
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        return rgb

    def track_collision(self, client, col) -> None:
        """Collision event -> punish OR reward (car), then respawn.
        Punish: anything that is not a car. Reward: AirSimNH parked cars
        (Car_*). Both respawn at the start point facing down the street."""
        now = time.perf_counter()
        if now - self._last_respawn < RESPAWN_COOLDOWN_S:
            return
        if self.in_stun() or self._stun_pending:
            return    # event already registered; a wedged aircraft keeps
                      # producing fresh collision timestamps every tick —
                      # they're the SAME crash, not new events
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
            if self._target_idx not in self._car_paid:
                self._car_paid.add(self._target_idx)
                if self._jackpot_enabled:
                    mag += 2.5    # first touch of THIS target: one-time bonus
                    sign = "car reward + jackpot"
                else:
                    sign = "car reward (jackpot off)"
            else:
                sign = "car reward (repeat)"
        else:
            self.collision_count += 1
            mag = self.args.punish_mag
            sign = "punish"
        asyncio.get_running_loop().create_task(
            self.send_reward_now(mag))
        self.begin_event_stun(mag, "car touch" if is_car else "collision")
        print(f"[sim] {sign} [{obj or 'unknown'}] {mag:+.1f}; "
              f"holding scene until dopa recovers "
              f"(hits {self.collision_count}, cars {self.car_bumps})", flush=True)
        self.end_episode("collision" if not is_car else "car touch")

    # ---- punishment / reward protocol ------------------------------------
    def begin_event_stun(self, mag: float, why: str) -> None:
        """After a big event (punish or car reward), HOLD the current scene
        for a while before respawning.

        Why (measured on the live brain): one deep pulse leaves the dopamine
        error negative for 4.5-6 s — respawning immediately dumps that tail
        onto the NEXT episode's opening moves, punishing good flying, while
        the crash-context synapses get less LTD than they should. Holding
        the scene keeps the negative window overlapped with the synapses
        that caused the event: clean credit assignment. Rewards get a short
        hold so their positive tail can't spuriously reinforce the next
        episode's first moves either."""
        wing = self.args.airframe == "wing"
        hold = POSITIVE_HOLD_S if mag > 0.0 else self.args.stun_hold
        if wing and mag < 0.0:
            hold = max(hold, 2.5)     # a plane can't freeze; still hold the view
        self._stun_until = time.perf_counter() + hold
        self._stun_pending = time.perf_counter()
        self._stun_mag = mag
        self._stun_why = why

    def in_stun(self) -> bool:
        return time.perf_counter() < self._stun_until

    def respawn(self, client) -> None:
        """Cars mode: teleport next to the current target car, facing it
        (atan2 bearing -> Z-rotation quaternion; body +x lands on the
        car), close enough that the car is inside the retina and the
        closing-progress signal. Street mode: back to the spawn XY at the
        ABSOLUTE respawn altitude (never relative to wherever the drone
        was when the bridge started)."""
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
        if self.args.eyes == "fly":
            self._hide_own_body(client)   # pose reset can restore the pawn
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
        # CARRY the previous episode's trim across the respawn: the circuit's
        # resting state doesn't change at an episode boundary, and starting
        # with trim=None mutes pitch/roll/yaw (centered() refuses to act on
        # raw bias) until a quiet calibration window lands — seconds of no
        # steering while absolute sub-hover throttle dove the drone into the
        # ground. With the carried trim the brain controls from frame 0; the
        # quiet-window sampler below still refreshes the estimate for future
        # episodes, so long-term drift stays tracked.
        if not self._trim_frozen:
            self.trim = {"pitch": None, "roll": None, "yaw": None}
        self._trim_frozen = False
        self._trim_at = 0.0
        self._trim_samples: list[tuple[float, dict[str, float]]] = []
        self._trim_windows = 0
        self.stick = {"fwd": 0.0, "lat": 0.0, "climb": 0.0, "yaw": 0.0}
        self._ceil_since = 0.0
        self._last_target_near = float("inf")   # fresh approach baseline
        self._car_pulses = 0
        self._spawn_hold_until = time.perf_counter() + SPAWN_HOLD_S
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

    def compute_sticks(self, now: float) -> dict[str, float]:
        """Brain channels -> RC sticks. IDENTICAL rule on every axis:
        trimmed center -> deadband -> gain -> low-pass. No per-axis
        special cases anywhere; the brain's commands are the control."""
        c = self.cmd
        gain = self.args.stick_gain

        def centered(name: str) -> float:
            """Channel value relative to its trim center, deadbanded."""
            trim = self.trim.get(name)
            if trim is None:
                return 0.0          # no trim yet: never act on raw bias
            d = clamp(c[name], -1.0, 1.0) - trim
            if abs(d) < self.args.deadband:
                return 0.0
            return d

        # climb: throttle is the brain's ABSOLUTE output. Zero throttle
        # means zero throttle — the aircraft sinks at its physical rate.
        # The MEASURED hover point (aircraft constant, probed at sim
        # connect) is where climb demand crosses zero; above it climbs,
        # below it descends. Nothing about the brain's resting statistics
        # is subtracted, so holding altitude is learned, not trimmed in.
        thr = clamp(c["throttle"], 0.0, 1.0)
        hover = self._hover_thr
        k_up = VZ_MAX / (1.0 - hover)     # full stick up == +VZ_MAX
        climb = (thr - hover) * k_up
        raw = {
            "fwd":   clamp(-centered("pitch") * gain, -1.0, 1.0) * V_FWD_MAX,
            "lat":   clamp(centered("roll") * gain, -1.0, 1.0) * V_LAT_MAX,
            "climb": clamp(climb, -VZ_MAX, VZ_MAX),
            "yaw":   -clamp(centered("yaw") * gain, -1.0, 1.0) * YAW_RATE_MAX,
        }
        dt = now - self._stick_at
        self._stick_at = now
        if dt <= 0.0:
            return self.stick
        for k, tgt in raw.items():
            a = 1.0 if self.args.stick_tau <= 0                 else min(1.0, dt / self.args.stick_tau)
            self.stick[k] += (tgt - self.stick[k]) * a
        return self.stick

    def update_trim(self) -> None:
        """Calibrate each attitude channel's resting value ONCE per episode
        (RC transmitter trim, frozen after arming).

        For the first TRIM_SAMPLE_S of an episode we collect resting samples
        and take the median as the center; from then on the trim is FROZEN.
        This is the crucial property: a continuous adaptive trim absorbs
        sustained channel offsets, but flight COMMANDS *are* sustained
        offsets — an adaptive trim ate them (the drone could only spin and
        change altitude).

        The frozen center is CARRIED across respawns: the previous episode's
        trim governs from frame 0 (no mute gap — the sampler can't be trusted
        to land a quiet window quickly, and it rejects windows whenever the
        brain is active), while this sampler keeps re-deriving a fresh
        estimate so it tracks the circuit's slowly drifting resting state.
        """
        now = time.perf_counter()
        if self.in_stun() or not self.got_actions:
            return
        if self._trim_frozen:
            return                        # calibrated this episode: frozen
        self._trim_samples.append(
            (now, {k: clamp(self.cmd[k], -1.0, 1.0)
                   for k in ("pitch", "roll", "yaw")}))
        t0 = self._trim_samples[0][0]
        if now - t0 < TRIM_SAMPLE_S:
            return                        # still sampling
        # reject an ACTIVE window: if any channel moved (stdev over the
        # window) the brain was commanding, and its median would freeze a
        # command in as the center — discard and resample. After a few
        # failed windows, accept anyway (a biased-but-stable center still
        # beats never calibrating; the next respawn retries).
        active = any(
            _stdev([s[1][k] for s in self._trim_samples]) > TRIM_QUIET_STDEV
            for k in ("pitch", "roll", "yaw"))
        if active and self._trim_windows < TRIM_MAX_WINDOWS:
            self._trim_windows += 1
            self._trim_samples = []
            return
        for name in ("pitch", "roll", "yaw"):
            vals = sorted(s[1][name] for s in self._trim_samples)
            self.trim[name] = vals[len(vals) // 2]
        self._trim_frozen = True
        self._trim_samples = []
        print(f"[trim] calibrated (frozen for episode): "
              f"pitch {self.trim['pitch']:+.3f} "
              f"roll {self.trim['roll']:+.3f} "
              f"yaw {self.trim['yaw']:+.3f}  |  phys hover "
              f"{self._hover_thr:.3f}", flush=True)

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
            if self.in_stun():
                # dazed plane: wings level, gentle climb out, glide straight
                # ahead at cruise (a fixed wing cannot hover) — nothing new
                # commanded; the climb usually exits ground/canopy wedges
                self._hud = (self._wing_v, 0.0, 2.0, 0.0)
                client.moveByVelocityBodyFrameAsync(
                    self._wing_v, 0.0, -2.0, CMD_HOLD_S + self.CMD_PERIOD_S,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    airsim.YawMode(False, 0.0),
                    self.args.vehicle,
                )
            else:
                self.apply_wing_model(client, now)
            return
        if now - self._last_cmd_at < self.CMD_PERIOD_S:
            return
        self._last_cmd_at = now
        s = self.compute_sticks(now)
        if self.in_stun():
            # dazed quad: freeze the stick demands so nothing further is
            # punished and the crash view stays put for the retina
            self.stick = {"fwd": 0.0, "lat": 0.0, "climb": 0.0, "yaw": 0.0}
            s = self.stick
            climb = 0.0
        else:
            climb = s["climb"]
            if now < self._spawn_hold_until and climb < 0.5:
                # spawn hold: block descent demands only; the first climb
                # command hands control straight back to the brain
                climb = 0.3
            if not self.args.no_assist:
                # brain owns altitude; the band is a tiny anti-smash guard
                # near the ground ONLY (0.8 m blend, full override at
                # min_alt). It used to reach 3 m — which swallowed every
                # descent command below 3 m, bounced the drone off an
                # invisible floor in a climb-sawtooth, and made voluntary
                # low-altitude descent unlearnable (no consequence, no
                # gradient). Now the brain owns everything above ~1 m.
                kin = client.simGetGroundTruthKinematics(self.args.vehicle)
                alt = -kin.position.z_val
                over = (self.args.min_alt + 0.8) - alt
                if over > 0.0:                  # floor: blend to +2 m/s climb
                    f = clamp(over, 0.0, 1.0)
                    climb = climb * (1.0 - f) + 2.0 * f
                over = alt - (self.args.max_alt - 1.0)
                if over > 0.0:                  # ceiling: blend to -0.8 m/s
                    f = clamp(over, 0.0, 1.0)
                    climb = climb * (1.0 - f) - 0.8 * f
        self._hud = (s["fwd"], s["lat"], s["climb"], s["yaw"])
        # NED: vz is down-positive, so pass -climb. Yaw: tiny demand =
        # heading hold (YawMode is_rate=False damps rotation), like a real
        # quad's heading-hold mode; otherwise a rate command.
        yaw_mode = (abs(s["yaw"]) >= YAW_HOLD_EPS)
        # BRAIN-ONLY ACTUATION (no hard-coded behavior): the connectome's
        # four channels are the ONLY flight commands. Yaw follows the same
        # stick rule as every other axis — centered = heading hold
        # (YawMode is_rate=False), deflected = yaw RATE. The former
        # exploration wind / nose-chase injected actuator commands the
        # brain never made (the repeated spin-in-place was exactly that,
        # bridge-side, not the circuit) — gone. Reward/punish signals
        # (car curriculum, collision punish, stagnation pressure) are
        # TRAINING signals, not actuator control, and remain.
        client.moveByVelocityBodyFrameAsync(
            s["fwd"],
            s["lat"],
            -climb, CMD_HOLD_S + self.CMD_PERIOD_S,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(abs(s["yaw"]) >= YAW_HOLD_EPS, s["yaw"]),
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
                self.update_trim()
            return
        try:
            obj = json.loads(msg)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if obj.get("type") == "telemetry":
            self.telemetry = obj
            self.dopa = float(obj.get("dopa", 0.0) or 0.0)
            self.learning = bool(obj.get("learning", False))

    # ---- human command interface (curriculum only, never actuators) -------
    def _handle_command(self, text: str) -> str:
        """Map a typed command to queue actions. Everything here sets the
        TASK (goal, spawn, signals) the same way flags do at startup; no
        line decides a motor output, so the brain still earns the behavior
        itself through R-STDP."""
        t = text.strip().lower()
        if not t:
            return "error: empty command"
        if t == "reset":
            self._cmd_q.append({"t": "reset"})
            return "respawning now"
        if t in ("new car", "next car"):
            self._cmd_q.append({"t": "newcar"})
            return "picking a new target car at the next respawn"
        m = re.fullmatch(r"car #?(\d+)", t)
        if m:
            idx = int(m.group(1))
            if not self.car_poses or not (0 <= idx < len(self.car_poses)):
                return (f"error: car index out of range "
                        f"(0..{max(0, len(self.car_poses) - 1)})")
            self._cmd_q.append({"t": "car", "idx": idx})
            return f"target set to car #{idx}; takes effect at the next respawn"
        if t == "hover":
            self._cmd_q.append({"t": "hold"})
            return "brain paused for 5 s (affects the next episode), then flying again"
        if t == "go":
            self._cmd_q.append({"t": "resume"})
            return "training resumed (if it was held)"
        if t == "training on":
            self._cmd_q.append({"t": "resume"})
            return "training resumed"
        if t == "training off":
            self._cmd_q.append({"t": "hold"})
            return "brain paused for 5 s (affects the next episode)"
        if t == "jackpot on":
            self._jackpot_enabled = True
            return "car-touch jackpot on"
        if t == "jackpot off":
            self._jackpot_enabled = False
            return "car-touch jackpot off"
        return ("error: unknown command (try: car N, new car, hover, go, "
                "reset, jackpot on/off)")

    def _drain_commands(self) -> None:
        """Consume queued commands on the control thread."""
        while self._cmd_q:
            c = self._cmd_q.pop(0)
            k = c["t"]
            if k == "car":
                self._target_idx = c["idx"]
                self._car_paid.discard(c["idx"])   # fresh jackpot for it
                print(f"[cmd] target car -> #{self._target_idx}", flush=True)
            elif k == "newcar" and self.car_poses:
                others = [i for i in range(len(self.car_poses))
                          if i != self._target_idx]
                self._target_idx = random.choice(others) if others else 0
                self._car_paid.discard(self._target_idx)
                print(f"[cmd] new target car -> #{self._target_idx}",
                      flush=True)
            elif k == "reset":
                self._stun_until = 0.0
                self._spawn_hold_until = 0.0
                self.end_episode("command reset")
                try:
                    client = self._cmd_client
                except AttributeError:
                    client = None
                if client is not None:
                    self.respawn(client)
                print("[cmd] manual reset", flush=True)
            elif k == "hold":
                self._spawn_hold_until = time.perf_counter() + 5.0
                print("[cmd] holding (brain paused 5 s)", flush=True)
            elif k == "resume":
                self._spawn_hold_until = 0.0
                print("[cmd] resumed", flush=True)

    # ---- reward shaping (optional) ----------------------------------------
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
        stun = ""
        if self.in_stun():
            left = self._stun_until - time.perf_counter()
            stun = (f"  STUN[{self._stun_why} {self._stun_mag:+.1f}] "
                    f"{left:4.1f}s dopa {self.dopa:+.2f}")
        sticks = self._hud
        wing = (f"v {self._wing_v:4.1f} m/s bank {self._wing_roll:+5.1f} deg  "
                if self.args.airframe == "wing" else "")
        print(f"[{'wing' if self.args.airframe == 'wing' else 'fly'}] "
              f"alt {alt:5.1f} m  "
              f"spd {speed:4.1f} m/s  {wing}"
              f"sticks fwd {sticks[0]:+5.1f} lat {sticks[1]:+5.1f} "
              f"vz {sticks[2]:+5.1f} m/s yaw {sticks[3]:+6.1f} deg/s  "
              f"hits {self.collision_count}  cars {self.car_bumps}  "
              f"near {self._near_car:5.1f} m  min {self._ep_near_min:4.1f} m  "
              f"pulses {self._car_pulses:3d}  prox {self._prox_val:+.2f}  "
              f"dopa {self.dopa:+.3f} eyes {self._vision_hz_ema:4.1f}/s "
              f"learn {'on' if self.learning else 'off'}  "
              f"sim {sim_ms:.1f} ms  act {stale*1000:.0f} ms ago", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brain-ws", default="ws://127.0.0.1:8787/stream")
    ap.add_argument("--airsim-ip", default="127.0.0.1")
    ap.add_argument("--airsim-port", type=int, default=41451)
    ap.add_argument("--vehicle", default="Fly1")
    ap.add_argument("--vision-hz", type=int, default=DEFAULT_VISION_HZ,
                    help="eye-rate CAP; 0 (default) = stream as fast as the "
                         "sim renders (capture latency dominates on slow "
                         "machines; an artificial sleep is never added)")
    ap.add_argument("--eyes", choices=["center", "stereo", "fly"],
                    default="fly",
                    help="fly (default): wrap-around compound-eye optics "
                         "with ommatidial blur, green-weighted photoreceptors "
                         "and a phasic motion channel, one hemisphere per "
                         "eye; stereo: +-35 deg camera pair; center: one "
                         "forward camera to both eyes")
    ap.add_argument("--no-gfx", dest="gfx", action="store_false",
                    default=True,
                    help="drop the game's rendering quality so the sim "
                         "renders faster and the eyes stream faster "
                         "(--no-gfx to keep stock visuals)")
    ap.add_argument("--hide-body", action="store_true",
                    help="hide the drone's own mesh from every camera "
                         "(Unreal 'ke <pawn> 0'); default keeps it visible "
                         "for the third-person chase view")
    ap.add_argument("--viewer-port", type=int, default=FLY_VIEWER_PORT,
                    help="fly-vision viewer HTTP port (0 disables; "
                         "http://localhost:8795)")
    ap.add_argument("--min-alt", type=float, default=0.0,
                    help="ground safety band lower edge (m); 0 = ground "
                         "level (the <0.8 m climb-out guard still applies)")
    ap.add_argument("--max-alt", type=float, default=60.0,
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
    ap.add_argument("--deadband", type=float, default=STICK_DEADBAND,
                    help="attitude-channel deadband around trim (stick "
                         "units) — jitter inside it commands nothing")
    ap.add_argument("--no-trim", action="store_true",
                    help="disable auto-trim (treat channel 0 as center, "
                         "the old always-drifting behavior)")
    ap.add_argument("--ceil-ride", type=float, default=CEILING_RIDE_S,
                    help="seconds riding the ceiling before punish+respawn")
    ap.add_argument("--border-radius", type=float, default=BORDER_RADIUS_M,
                    help="horizontal leash from spawn; beyond it = map border")
    ap.add_argument("--ground-z", type=float, default=GROUND_Z,
                    help="world-frame z of the ground plane at the spawn area")
    ap.add_argument("--prox-gain", type=float, default=1.0,
                    help="proximity-reward gain (0 disables the shaping)")
    ap.add_argument("--alt-gain", type=float, default=ALT_GAIN_DEFAULT,
                    help="signed altitude-hill reward: +gain at 8 m "
                         "grading to 0 at 0 m and 16 m, then a growing "
                         "penalty above (−gain by 24 m) — makes coming "
                         "down itself rewarding; 0 disables")
    ap.add_argument("--prox-radius", type=float, default=40.0,
                    help="nearest-car distance beyond which prox reward is 0 (m)")
    ap.add_argument("--prox-max", type=float, default=0.5,
                    help="proximity reward pulse value at 0 m distance")
    ap.add_argument("--near-radius", type=float, default=8.0,
                    help="near-field approach-gradient radius (m)")
    ap.add_argument("--near-max", type=float, default=1.0,
                    help="near-field reward at 0 m (must stay below reward-mag "
                         "so touching a car still pays more than hovering over it)")
    ap.add_argument("--cars", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="car-crash curriculum: every respawn starts a fresh "
                         "14-22 m approach to the current target car "
                         "(default on; --no-cars disables)")
    ap.add_argument("--closing-gain", type=float, default=0.1,
                    help="cars mode: reward per meter closed toward the "
                         "target car (capped at 0.5 per pulse)")
    ap.add_argument("--closing-min", type=float, default=0.10,
                    help="cars mode: minimum meters closed per 0.5 s tick "
                         "for a closing pulse (jitter gate — a hover must "
                         "stay silent)")
    ap.add_argument("--no-assist", action="store_true",
                    help="apply raw brain channels, no altitude hold")
    ap.add_argument("--punish-mag", type=float, default=PUNISH_MAG_DEFAULT,
                    help="collision punish pulse; -5.0 reaches the brain's "
                         "dan_drive clamp (-1.5) = deepest teaching window "
                         "(measured 77%% deeper dopamine than -2.5)")
    ap.add_argument("--reward-mag", type=float, default=REWARD_MAG_DEFAULT,
                    help="reward pulse value sent on a car touch")
    ap.add_argument("--stun-hold", type=float, default=STUN_HOLD_S,
                    help="minimum seconds to hold the crash scene before "
                         "respawn (the negative dopa window is 4.5-6 s); "
                         "respawn also waits for dopa recovery (cap "
                         f"{STUN_MAX_S} s)")
    ap.add_argument("--stun-max", type=float, default=STUN_MAX_S,
                    help="hard cap on the scene hold even if dopa has not "
                         "recovered (s)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bridge = Bridge(args)
    start_fly_viewer(bridge, args.viewer_port if args.eyes == "fly" else 0)
    try:
        return asyncio.run(bridge.run())
    except KeyboardInterrupt:
        print("\n[bridge] stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
