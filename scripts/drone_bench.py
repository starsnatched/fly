"""Drone benchmark: does R-STDP memory change flight behavior?

Closed-loop, protocol-only twin of examples/drone-web (same generation seed,
same kinematics, same collision model). Streams a synthetic stereo eye pair —
pillar darkness in whichever eye it sees, plus a vertical loom band when an
obstacle closes — at 60 Hz plus barometer/IMU state, and applies the returned
throttle/pitch/roll/yaw channels.

Phases (identical design to the rover bench):
  1. learn       reward stream ON  (altitude hold + clearance + speed - impacts)
  2. test-mem    reward OFF, learning OFF, memory AS LEARNED
  3. wipe        POST-equivalent control {wipe:true}  (factory defaults)
  4. test-clean  reward OFF, learning OFF, memory CLEAN

Paired trials: both test phases start at the same pose on the same generated
world, so the comparison is course-identical.

Server spawns on its own ports with its own memory file — your normal brain
memory is untouched.

Usage:
  python scripts/drone_bench.py [--learn 120] [--test 45] [--fresh]
                                [--readout population|pools]
Requires: the built server (build/flybrain-server), websockets, numpy.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "build" / "flybrain-server.exe"
if not SERVER.exists():
    SERVER = ROOT / "build" / "flybrain-server"

WS_PORT, REST_PORT = 8809, 8810
BENCH_CONFIG = ROOT / "state" / "drone-bench.config.json"
BENCH_MEMORY = ROOT / "state" / "drone-bench-memory.json"

W, H = 192, 108
EYE_MOUNT = 35.0                 # degrees off-boresight per eye (vision.ts)
EYE_MERGE_HALF = 47.0            # merged binocular half-FOV
MAX_THRUST = 34.0                # m/s^2 at throttle 1.0 (drone.ts)
DRAG_LIN, DRAG_QUAD = 0.24, 0.015
MAX_TILT = 0.9                   # rad
HOVER_THRUST = 9.81 / MAX_THRUST  # throttle for level flight

LIMIT = 480.0
CEILING = 26.0


class World:
    """Mirror of examples/drone-web/src/world.ts (seed 7 LCG)."""

    def __init__(self):
        self.obs: list[dict] = []   # {x,z,r,h}
        self.seed = 7
        z = -30.0
        while z > -420.0:
            k = self.rnd()
            if k < 0.45:
                n = 1 + int(self.rnd() * 3)
                for _ in range(n):
                    r = 1.2 + self.rnd() * 2.6
                    h = 6 + self.rnd() * 17
                    x = (self.rnd() - 0.5) * 70
                    zz = z + (self.rnd() - 0.5) * 16
                    self.add(x, zz, r * 1.12, h)
            elif k < 0.75:
                gap_x = (self.rnd() - 0.5) * 44
                h = 7 + self.rnd() * 9
                wl = gap_x + 18 + 35
                if wl > 1.5:
                    self.add(-35 + wl / 2, z, wl / 2, h)
                wr = 35 - (gap_x + 24)
                if wr > 1.5:
                    self.add(gap_x + 24 + wr / 2, z, wr / 2, h)
            else:
                y = 5.5 + self.rnd() * 5
                self.add(0.0, z, 0.8, y + 0.8)
                sx = -26.0 if self.rnd() < 0.5 else 26.0
                self.add(sx, z, 1.0, y)
            z -= 16 + self.rnd() * 22
        for _ in range(420):
            x = (self.rnd() - 0.5) * 900
            zz = (self.rnd() - 0.5) * 900
            if abs(x) < 40 and zz < 10:
                continue
            r = 1.5 + self.rnd() * 5
            h = 5 + self.rnd() * 20
            self.add(x, zz, r * 1.15, h)

    def rnd(self) -> float:
        self.seed = (self.seed * 16807) % 2147483647
        return self.seed / 2147483647.0

    def add(self, x: float, z: float, r: float, h: float) -> None:
        self.obs.append({"x": x, "z": z, "r": r, "h": h})


def eye_frames(x: float, y: float, z: float, yaw: float):
    """Stereo pair: pillar darkness in whichever eye sees it (mirrors the
    browser's render — obstacles are dark against bright sky/ground)."""
    imgs = [np.full((H, W), 150, np.uint8) for _ in range(2)]
    for im in imgs:
        im[: H // 2] = 205                      # sky
    merged_best = (1e9, None)                   # (dist, bearing) for loom
    for eye in (0, 1):
        side = 1 if eye == 0 else -1            # left eye mounts +35 deg
        eye_yaw = yaw + math.radians(side * EYE_MOUNT)
        fx, fz = -math.sin(eye_yaw), -math.cos(eye_yaw)
        best = (1e9, None)                      # (dist, bearing in eye)
        for o in world.obs:
            dx, dz = o["x"] - x, o["z"] - z
            flat = math.hypot(dx, dz)
            if flat < 0.5 or flat - o["r"] > 90:
                continue
            if y > o["h"] + 0.5:                # flying above it: invisible
                continue
            along = (dx * fx + dz * fz) / flat
            if along < math.cos(math.radians(EYE_MERGE_HALF + EYE_MOUNT)):
                continue
            bearing = math.atan2(dx * -fz + dz * fx, dx * fx + dz * fz)
            d = flat - o["r"]
            if d < best[0]:
                best = (d, bearing)
            if d < merged_best[0]:
                merged_best = (d, bearing - math.radians(side * EYE_MOUNT))
        if best[1] is not None and best[0] < 60:
            d, bearing = best
            col = int((bearing / math.radians(EYE_MERGE_HALF) + 1) / 2 * (W - 1))
            im = imgs[eye]
            im[H // 2:, max(0, col - 8): col + 8] = 40
    d, _bm = merged_best
    if d < 1e8 and d < 60:
        # vertical loom hint: a near obstacle darkens a band ABOVE the
        # horizon in both eyes (its top edge), stronger the closer it is
        rows = max(0, min(H // 2 - 2, int((1 - d / 60) * (H // 2) * 0.55)))
        if rows > 0:
            for im in imgs:
                im[H // 2 - rows: H // 2,
                   max(0, W // 2 - 24): W // 2 + 24] = 55
    out = []
    for im in imgs:
        rgb = np.empty((H, W, 3), np.uint8)
        rgb[..., 0] = im
        rgb[..., 1] = im
        rgb[..., 2] = im
        out.append(struct.pack("<BBHHB", 1, 0, W, H, 3) + rgb.tobytes())
    return out


class DronePhys:
    """Mirror of examples/drone-web/src/drone.ts step() (second-order
    attitude: pitchRate/rollRate/yawRate are state, exactly as in the
    browser; Euler order YXZ)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.x, self.y, self.z = 0.0, 4.0, 20.0
        self.vx = self.vy = self.vz = 0.0
        self.yaw = 0.0
        self.pitch = self.roll = 0.0
        self.pitchRate = self.rollRate = self.yawRate = 0.0
        self.odo = 0.0
        self.overlapping = False
        self.last_thr, self.last_pitch = HOVER_THRUST, -0.35
        self.last_roll, self.last_yaw = 0.0, 0.0

    def step(self, thr: float, pitch: float, roll: float, yaw: float,
             dt: float) -> bool:
        tp = pitch * MAX_TILT
        tr = roll * MAX_TILT
        tyr = yaw * 1.9
        self.pitchRate += ((tp - self.pitch) * 8 - self.pitchRate * 6) * dt
        self.rollRate += ((-tr - self.roll) * 8 - self.rollRate * 6) * dt
        self.yawRate += (tyr - self.yawRate) * 9 * dt
        self.pitch += self.pitchRate * dt
        self.roll += self.rollRate * dt
        self.yaw += self.yawRate * dt

        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        # R = Ry(yaw) Rx(pitch) Rz(roll) applied to (0,1,0) and (0,0,-1)
        ux = -sr * cy + cr * sp * sy
        uy = cr * cp
        uz = sr * sy + cr * sp * cy
        fx, fy, fz = -sy * cp, sp, -cy * cp

        acc = MAX_THRUST * thr
        ax = acc * ux
        ay = acc * uy - 9.81
        az = acc * uz
        spd = math.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
        k = DRAG_LIN + DRAG_QUAD * spd
        ax -= k * self.vx
        ay -= k * self.vy
        az -= k * self.vz
        self.vx += ax * dt
        self.vy += ay * dt
        self.vz += az * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        bumped = False
        if self.y < 0.25:
            self.y = 0.25
            self.vy = abs(self.vy) * 0.4
            self.vx *= 0.92
            self.vz *= 0.92
            bumped = True
        if self.y > CEILING:
            self.y = CEILING
            self.vy = min(0.0, self.vy)
        if abs(self.x) > LIMIT:
            self.x = math.copysign(LIMIT, self.x)
            self.vx *= -0.5
            bumped = True
        if abs(self.z) > LIMIT:
            self.z = math.copysign(LIMIT, self.z)
            self.vz *= -0.5
            bumped = True
        hs = math.hypot(self.vx, self.vz)
        if hs > 1e-4:
            self.odo += hs * dt
        return bumped

    def resolve(self) -> bool:
        hit = False
        for o in world.obs:
            if self.y > o["h"] + 0.1:
                continue
            dx, dz = self.x - o["x"], self.z - o["z"]
            dist = math.hypot(dx, dz)
            rr = o["r"] + 0.45
            if dist >= rr:
                continue
            nx = dx / dist if dist > 1e-4 else 1.0
            nz = dz / dist if dist > 1e-4 else 0.0
            self.x = o["x"] + nx * rr
            self.z = o["z"] + nz * rr
            vn = self.vx * nx + self.vz * nz
            if vn < 0:
                self.vx -= 1.35 * vn * nx
                self.vz -= 1.35 * vn * nz
            self.vx *= 0.72
            self.vz *= 0.72
            hit = True
        return hit

    def clearance(self) -> float:
        best = 120.0
        fx, fz = -math.sin(self.yaw), -math.cos(self.yaw)
        for o in world.obs:
            if self.y > o["h"] + 0.5:
                continue
            dx, dz = o["x"] - self.x, o["z"] - self.z
            flat = math.hypot(dx, dz)
            if flat < 0.5:
                continue
            dist = flat - o["r"]
            if dist < 0.1:
                continue
            if (dx * fx + dz * fz) / flat > 0.86:
                best = min(best, dist)
        return best


world = World()
phys = DronePhys()


def rest_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return json.loads(r.read()).get("ok") is True
    except Exception:
        return False


async def run_phase(ws, label: str, seconds: float, reward: bool, learn: bool):
    await ws.send(json.dumps({"type": "control", "learning": learn}))
    t_start = time.perf_counter()
    x0, z0 = phys.x, phys.z
    hits = stops = wedges = 0
    low_s = 0.0
    impact_pulse = 0.0
    impact_cool = 0.0
    recovering_s = 0.0
    dist = 0.0
    prog = 0.0
    last = time.perf_counter()
    next_report = t_start + 15
    while time.perf_counter() - t_start < seconds:
        now = time.perf_counter()
        dt = min(now - last, 0.06)
        last = now
        hit = False
        spd = math.hypot(phys.vx, phys.vz)

        if recovering_s > 0.0:
            # low-altitude recovery reflex: climb out; reward withheld
            recovering_s -= dt
            phys.step(0.5, -0.35, 0.0, 0.0, dt)
            phys.resolve()
        else:
            phys.step(phys.last_thr, phys.last_pitch, phys.last_roll,
                      phys.last_yaw, dt)
            hit = phys.resolve()
            if hit and not phys.overlapping and impact_cool <= 0.0:
                hits += 1
                impact_pulse = 1.0
                impact_cool = 0.4
            phys.overlapping = hit
            impact_cool = max(0.0, impact_cool - dt)
            spd = math.hypot(phys.vx, phys.vz)
            dist += spd * dt
            if phys.y < 1.2:
                low_s += dt
                if low_s > 1.5:
                    wedges += 1
                    stops += 1
                    recovering_s = 1.0
                    low_s = 0.0
            else:
                low_s = 0.0

        if reward and recovering_s <= 0.0:
            impact_pulse *= 0.9
            alt_err = max(-1.0, min(1.0, (5.0 - phys.y) / 4.0))
            clr = phys.clearance()
            clr_r = max(-1.0, min(1.0, (18.0 - clr) / 30.0))
            prog += (spd - prog) * min(1.0, dt / 12.0)
            d = (spd - prog) / max(prog, 0.5)
            v = 0.7 * alt_err + 0.7 * clr_r + 0.5 * max(-1.0, min(1.0, d)) \
                - 1.2 * impact_pulse
            await ws.send(json.dumps(
                {"type": "reward", "value": round(max(-1.0, min(1.0, v)), 3)}))

        fl, fr = eye_frames(phys.x, phys.y, phys.z, phys.yaw)
        await ws.send(fl)
        await ws.send(fr)
        await ws.send(json.dumps({
            "type": "state", "altitude": round(phys.y, 3),
            "vy": round(phys.vy, 3), "collision": bool(hit)}))

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
            except (asyncio.TimeoutError, TimeoutError):
                break
            if isinstance(msg, bytes) and msg and msg[0] == 10:
                nl = struct.unpack_from("<H", msg, 1)[0]
                names = json.loads(msg[3:3 + nl])
                vals = struct.unpack_from("<" + "f" * len(names), msg, 3 + nl)
                d2 = dict(zip(names, vals))
                phys.last_thr = float(np.clip(d2.get("throttle", HOVER_THRUST), 0, 1))
                phys.last_pitch = float(np.clip(d2.get("pitch", 0), -1, 1))
                phys.last_roll = float(np.clip(d2.get("roll", 0), -1, 1))
                phys.last_yaw = float(np.clip(d2.get("yaw", 0), -1, 1))
        await asyncio.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))
        if time.perf_counter() > next_report:
            next_report += 15
            print(f"    [{label}] alt {phys.y:5.1f} m  odo {phys.odo:6.1f} m  "
                  f"hits {hits:3d}  ({hits / max(dist, 1) * 100:5.1f}/100m)")
    net = math.hypot(phys.x - x0, phys.z - z0)
    return {"label": label, "hits": hits, "stops": stops, "dist": dist,
            "net": net, "mean_speed": dist / seconds,
            "mean_alt": phys.y, "odo": phys.odo}


async def amain(args):
    global world
    world = World()
    BENCH_CONFIG.parent.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / "config" / "flybrain.json").read_text())
    cfg["memoryPath"] = str(BENCH_MEMORY).replace("\\", "/")
    BENCH_CONFIG.write_text(json.dumps(cfg, indent=2))
    if args.fresh and BENCH_MEMORY.exists():
        BENCH_MEMORY.unlink()

    # motor pools are the standard decode; --readout is kept for compat
    profile = "drone"
    proc = subprocess.Popen(
        [str(SERVER), "--config", str(BENCH_CONFIG), "--profile", profile,
         "--port", str(WS_PORT)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while not rest_ok(REST_PORT):
            if proc.poll() is not None:
                sys.exit("server exited during boot")
            if time.time() - t0 > 60:
                sys.exit("server never became healthy")
            await asyncio.sleep(0.5)
        print(f"brain up on ws://127.0.0.1:{WS_PORT} (profile {profile}, "
              f"memory: {BENCH_MEMORY.name}, fresh={args.fresh})")

        phys.reset()
        async with websockets.connect(f"ws://127.0.0.1:{WS_PORT}/stream",
                                      open_timeout=30, max_size=20 * 2**20) as ws:
            await ws.send(json.dumps({"type": "hello", "profile": profile}))
            await ws.recv()

            print(f"\nLEARN phase ({args.learn}s) — reward stream ON, learning ON")
            r1 = await run_phase(ws, "learn", args.learn, reward=True, learn=True)

            with urllib.request.urlopen(f"http://127.0.0.1:{REST_PORT}/memory",
                                        timeout=60) as r:
                mem_json = r.read().decode()

            print(f"\nTEST with learned memory ({args.test}s) — reward OFF, learning OFF")
            await ws.send(json.dumps({"type": "control", "learning": False}))
            phys.reset()
            r2 = await run_phase(ws, "mem", args.test, reward=False, learn=False)

            await ws.send(json.dumps({"type": "control", "wipe": True}))
            await asyncio.sleep(0.5)
            print(f"\nTEST after wipe ({args.test}s) — memory reset, learning OFF")
            phys.reset()
            r3 = await run_phase(ws, "clean", args.test, reward=False, learn=False)

            # put the learned memory back so the bench is non-destructive
            await ws.send(json.dumps({"type": "control", "learning": True}))
            await ws.send(json.dumps({"type": "control", "memory": json.loads(mem_json)}))
            await asyncio.sleep(0.5)

        def row(r):
            per100 = r["hits"] / max(r["dist"], 1) * 100
            return (f"{r['label']:>6}: hits/100m {per100:6.2f}  "
                    f"low-alt stops {r['stops']:3d}  mean {r['mean_speed']:4.2f} m/s  "
                    f"net {r['net']:6.1f} m  odo {r['odo']:6.1f} m")

        print("\n=== RESULTS ===")
        print(row(r2))
        print(row(r3))
        d_mem = r2["hits"] / max(r2["dist"], 1)
        d_clean = r3["hits"] / max(r3["dist"], 1)
        if d_clean > 0:
            ratio = d_mem / d_clean
            verdict = "better" if ratio < 1 else ("worse" if ratio > 1 else "same")
            print(f"\nimpact-rate with memory vs clean: {ratio:.2f}x ({verdict})")
        print("BENCH DONE")
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if not args.keep:
            BENCH_CONFIG.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", type=float, default=120)
    ap.add_argument("--test", type=float, default=45)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--readout", choices=["population", "pools"], default="population")
    ap.add_argument("--keep", action="store_true")
    asyncio.run(amain(ap.parse_args()))
