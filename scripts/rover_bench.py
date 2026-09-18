"""Rover benchmark: does R-STDP memory actually change rover behavior?

Closed-loop, protocol-only (no browser): a numeric twin of examples/rover-web's
desert world streams a forward-eye RGB frame + proprioception to a dedicated
brain server at 60 Hz and applies the returned throttle/steer channels.

Phases
  1. learn       reward stream ON  (speed - impacts - stuck), learning ON
  2. test-mem    reward OFF, learning OFF, memory AS LEARNED
  3. wipe        POST-equivalent control {wipe:true}  (factory defaults)
  4. test-clean  reward OFF, learning OFF, memory CLEAN

Report: obstacles hit / 100 m, wall hits, stops, mean + net progress per phase.

The server is spawned on its own ports with its own memory file
(state/rover-bench-memory.json) — your normal brain memory is untouched.

Usage:
  python scripts/rover_bench.py [--learn 120] [--test 45] [--fresh] [--keep]
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
import time
import urllib.request
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "build" / "flybrain-server.exe"
if not SERVER.exists():
    SERVER = ROOT / "build" / "flybrain-server"

WS_PORT, REST_PORT = 8805, 8806
BENCH_CONFIG = ROOT / "state" / "rover-bench.config.json"
BENCH_MEMORY = ROOT / "state" / "rover-bench-memory.json"

W, H = 192, 108
FOV_RAD = math.radians(58)          # must match examples/rover-web/src/vision.ts
EYE_H = 0.98                        # eye height above ground
GRID = 4                            # eye-bar bearing quantization

# ---- world twin (mirrors examples/rover-web/src/world.ts generation) ----
ROVER_R = 0.9
COURSE_HALF = 30.0
LIMIT = 220.0
MAX_SPEED = 4.0


class World:
    """Obstacles as circles: ('wall' | 'boulder', x, z, r)."""

    def __init__(self, seed: int = 20260917, density: float = 1.0):
        self.obs: list[tuple[str, float, float, float]] = []
        self.rng_state = seed
        self.rnd()

        def add_boulder(x, z, r):
            self.obs.append(("boulder", x, z, r * 1.02))

        def add_wall(x, z, w, d=1.6):
            n = max(1, round(w / (d * 0.9)))
            for i in range(n):
                t = (i / (n - 1) - 0.5) * (w - d) if n > 1 else 0.0
                self.obs.append(("wall", x + t, z, d * 0.55))

        z = -18.0
        while z > -170.0:
            if self.rnd() < 0.32:
                gap_x = (self.rnd() - 0.5) * 26.0
                add_wall(gap_x - 19.0, z, 20 + self.rnd() * 8)
                add_wall(gap_x + 19.0, z, 20 + self.rnd() * 8)
            else:
                for _ in range(2 + int(self.rnd() * 4)):
                    r = 0.6 + self.rnd() * 1.8
                    add_boulder((self.rnd() - 0.5) * 60.0,
                                z + (self.rnd() - 0.5) * 10.0, r)
            z -= (9 + self.rnd() * 10) / max(density, 0.1)

        for _ in range(int(340 * density)):
            x = (self.rnd() - 0.5) * 440.0
            zz = (self.rnd() - 0.5) * 440.0
            if abs(x) < 34 and zz < 10:
                continue
            add_boulder(x, zz, 0.5 + self.rnd() * 2.6)

    def rnd(self) -> float:
        self.rng_state = (self.rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.rng_state / 4294967296.0

    def resolve(self, x: float, z: float, vx: float, vz: float):
        """Push out + bounce; returns (nx, nz, hit_wall, hit_any)."""
        hit = False
        hit_wall = False
        for kind, ox, oz, r in self.obs:
            rr = r + ROVER_R
            dx, dz = x - ox, z - oz
            d2 = dx * dx + dz * dz
            if d2 >= rr * rr:
                continue
            d = math.hypot(dx, dz)
            nx = dx / d if d > 1e-4 else 1.0
            nz = dz / d if d > 1e-4 else 0.0
            x, z = ox + nx * rr, oz + nz * rr
            vn = vx * nx + vz * nz
            if vn < 0:
                vx -= 1.3 * vn * nx
                vz -= 1.3 * vn * nz
            vx *= 0.7
            vz *= 0.7
            hit = True
            hit_wall = hit_wall or kind == "wall"
        return x, z, vx, vz, hit_wall, hit


class RoverPhys:
    """Mirror of examples/rover-web/src/rover.ts."""

    def __init__(self):
        self.x, self.z = 0.0, 20.0
        self.vx = self.vz = 0.0
        self.yaw = 0.0
        self.odo = 0.0

    def step(self, throttle: float, steer: float, dt: float) -> float:
        accel = 5.0
        roll = 2.2
        skid = 8.0
        # heading turns first; old momentum resolves onto the NEW body axes
        # (lateral velocity scrubs off fast — the chassis cannot strafe)
        self.yaw += steer * 1.6 * dt * min(1.0, math.hypot(self.vx, self.vz) / 0.6 + 0.25)
        fx, fz = -math.sin(self.yaw), -math.cos(self.yaw)
        rx, rz = math.cos(self.yaw), -math.sin(self.yaw)
        v_f = self.vx * fx + self.vz * fz
        v_l = self.vx * rx + self.vz * rz
        sp0 = abs(v_f)
        resist = roll * min(1.0, sp0 / 0.5) + 0.5 * sp0
        dv = (throttle * accel - resist) * dt
        if throttle * accel < resist and v_f + dv < 0:
            dv = -v_f
        v_f += dv
        v_l *= math.exp(-skid * dt)
        self.vx = fx * v_f + rx * v_l
        self.vz = fz * v_f + rz * v_l
        sp = math.hypot(self.vx, self.vz)
        if sp > MAX_SPEED:
            k = MAX_SPEED / sp
            self.vx *= k
            self.vz *= k
        self.x += self.vx * dt
        self.z += self.vz * dt
        # forward speed = velocity along body-forward (fx, fz); positive ahead
        fwd_sp = self.vx * fx + self.vz * fz
        if fwd_sp > 0:
            self.odo += fwd_sp * dt
        return fwd_sp

    def forward_speed(self) -> float:
        fx, fz = -math.sin(self.yaw), -math.cos(self.yaw)
        return self.vx * fx + self.vz * fz

    def clamp_bounds(self):
        bumped = False
        if abs(self.x) > LIMIT:
            self.x = math.copysign(LIMIT, self.x)
            self.vx *= -0.5
            bumped = True
        if abs(self.z) > LIMIT:
            self.z = math.copysign(LIMIT, self.z)
            self.vz *= -0.5
            bumped = True
        return bumped


def eye_frame(x: float, z: float, yaw: float) -> bytes:
    """Forward-eye RGB: sky/ground split + one dark bar at the nearest
    obstacle's bearing (quantized to GRID columns). Same signal idea as the
    browser's render: obstacles loom as dark pixels against bright ground."""
    img = np.full((H, W, 3), 150, np.uint8)          # bright regolith
    img[: H // 2] = 205                                # sky
    # bearing of nearest obstacle inside the FOV
    best_d, best_bearing = 1e9, None
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    for _, ox, oz, r in world.obs:
        dx, dz = ox - x, oz - z
        flat = math.hypot(dx, dz)
        if flat < 0.5 or flat - r > 60:
            continue
        along = (dx * fx + dz * fz) / flat
        if along < math.cos(FOV_RAD / 2):
            continue
        d = flat - r
        if d < best_d:
            best_d = d
            # signed angle from heading to obstacle
            best_bearing = math.atan2(
                dx * -fz + dz * fx, dx * fx + dz * fz)
    if best_bearing is not None and best_d < 40:
        col = int((best_bearing / (FOV_RAD / 2) + 1) / 2 * (W - 1))
        x0 = max(0, col - 8)
        img[H // 2:, x0: col + 8] = 40
    return struct.pack("<BBHHB", 1, 0, W, H, 3) + img.tobytes()





world = World()
phys = RoverPhys()


def phys_fwd_speed() -> float:
    return phys.forward_speed()


async def rest_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return json.loads(r.read()).get("ok") is True
    except Exception:
        return False


async def wait_healthy(proc, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during boot")
        if await rest_ok(REST_PORT):
            return
        await asyncio.sleep(0.5)
    raise RuntimeError("server did not become healthy")


async def run_phase(ws, label: str, seconds: float, reward: bool, learn: bool):
    await ws.send(json.dumps({"type": "control", "learning": learn}))
    t_start = time.perf_counter()
    x0, z0 = phys.x, phys.z
    hits = walls = stops = wedges = 0
    stop_s = 0.0
    impact_pulse = 0.0
    impact_cool = 0.0
    prog = 0.0   # EWMA of progress rate: the reward baseline (tau ~12 s)
    overlapping = False
    stuck = False
    recovering_s = 0.0
    dist = 0.0
    last = time.perf_counter()
    next_report = t_start + 15
    while time.perf_counter() - t_start < seconds:
        now = time.perf_counter()
        dt = min(now - last, 0.06)
        last = now
        hit_now = False
        speed_now = abs(phys_fwd_speed())

        if recovering_s > 0.0:
            # stuck recovery reflex: spin out of the wedge; reward is
            # withheld (a constant reward teaches nothing) and impacts are
            # not double-counted while grinding free
            recovering_s -= dt
            steer = 1.0 if wedges % 2 == 0 else -1.0
            phys.step(0.4, steer, dt)
            world.resolve(phys.x, phys.z, phys.vx, phys.vz)
            phys.clamp_bounds()
        else:
            fwd_sp = phys.step(phys.last_thr, phys.last_steer, dt)
            speed_now = abs(fwd_sp)
            nx, nz, nvx, nvz, hit_wall, hit = world.resolve(
                phys.x, phys.z, phys.vx, phys.vz)
            phys.x, phys.z, phys.vx, phys.vz = nx, nz, nvx, nvz
            phys.clamp_bounds()
            hit_now = hit
            # edge-triggered impact counting: a wedged contact is ONE hit
            if hit and not overlapping and impact_cool <= 0.0:
                hits += 1
                if hit_wall:
                    walls += 1
                impact_pulse = 1.0
                impact_cool = 0.4
            overlapping = hit
            impact_cool = max(0.0, impact_cool - dt)
            dist += abs(fwd_sp) * dt
            if abs(fwd_sp) < 0.15:
                stop_s += dt
                if stop_s > 2.0:
                    wedges += 1
                    stops += 1
                    recovering_s = 1.2
                    stop_s = 0.0
            else:
                stop_s = 0.0

        # reward: RELATIVE progress (rate-of-rate, tau ~12 s) minus impact
        # pulses. A constant speed teaches nothing by construction — only
        # getting faster/slower than the recent baseline moves the DAN
        # (matches the circuit's own adapting-baseline dopamine).
        if reward and recovering_s <= 0.0:
            impact_pulse *= 0.9
            prog += (speed_now - prog) * min(1.0, dt / 12.0)
            d = (speed_now - prog) / max(prog, 0.5)
            v = 1.2 * max(-1.0, min(1.0, d)) - 1.2 * impact_pulse
            await ws.send(json.dumps({"type": "reward", "value": round(max(-1.0, min(1.0, v)), 3)}))
        # sensory stream
        await ws.send(eye_frame(phys.x, phys.z, phys.yaw))
        await ws.send(json.dumps({"type": "state", "altitude": 0.0,
                                  "speed": round(speed_now, 3), "vy": 0.0,
                                  "collision": hit_now}))
        # consume the latest action frame (drain to newest)
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
            except (asyncio.TimeoutError, TimeoutError):
                break
            if isinstance(msg, bytes) and msg and msg[0] == 10:
                nl = struct.unpack_from("<H", msg, 1)[0]
                names = json.loads(msg[3:3 + nl])
                vals = struct.unpack_from("<" + "f" * len(names), msg, 3 + nl)
                d = dict(zip(names, vals))
                phys.last_thr = float(np.clip(d.get("throttle", 0.3), 0, 1))
                phys.last_steer = float(np.clip(d.get("steer", 0.0), -1, 1))
        await asyncio.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))
        if time.perf_counter() > next_report:
            next_report += 15
            print(f"    [{label}] odo {phys.odo:6.1f} m  hits {hits:3d}  "
                  f"({hits / max(dist, 1) * 100:5.1f}/100m)")
    net = math.hypot(phys.x - x0, phys.z - z0)
    return {
        "label": label, "hits": hits, "walls": walls, "stops": stops,
        "dist": dist, "net": net, "mean_speed": dist / seconds,
    }


def reset_rover():
    """Paired-trial design: both test phases start at the same pose so they
    traverse the identical generated course."""
    phys.x, phys.z = 0.0, 20.0
    phys.vx = phys.vz = 0.0
    phys.yaw = 0.0
    phys.last_thr, phys.last_steer = 0.3, 0.0


async def amain(args):
    global world
    world = World(seed=20260917, density=args.density)
    # ---- isolated config: flybrain.json with bench memoryPath ----
    BENCH_CONFIG.parent.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / "config" / "flybrain.json").read_text())
    cfg["memoryPath"] = str(BENCH_MEMORY).replace("\\", "/")
    BENCH_CONFIG.write_text(json.dumps(cfg, indent=2))
    if args.fresh and BENCH_MEMORY.exists():
        BENCH_MEMORY.unlink()

    # motor pools are the standard decode; --readout is kept for compat
    profile = "rover"
    proc = subprocess.Popen(
        [str(SERVER), "--config", str(BENCH_CONFIG), "--profile", profile,
         "--port", str(WS_PORT)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        await wait_healthy(proc)
        print(f"brain up on ws://127.0.0.1:{WS_PORT} (profile {profile}, memory: "
              f"{BENCH_MEMORY.name}, fresh={args.fresh})")

        phys.last_thr, phys.last_steer = 0.3, 0.0
        async with websockets.connect(f"ws://127.0.0.1:{WS_PORT}/stream",
                                      open_timeout=30, max_size=20 * 2**20) as ws:
            await ws.send(json.dumps({"type": "hello", "profile": profile}))
            hello = json.loads(await ws.recv())

            print(f"\nLEARN phase ({args.learn}s) — reward stream ON, learning ON")
            r1 = await run_phase(ws, "learn", args.learn, reward=True, learn=True)

            # export learned memory (REST) so the wipe below is reversible
            with urllib.request.urlopen(f"http://127.0.0.1:{REST_PORT}/memory",
                                        timeout=60) as r:
                mem_json = r.read().decode()

            print(f"\nTEST with learned memory ({args.test}s) — reward OFF, learning OFF")
            await ws.send(json.dumps({"type": "control", "learning": False}))
            reset_rover()
            r2 = await run_phase(ws, "mem", args.test, reward=False, learn=False)

            await ws.send(json.dumps({"type": "control", "wipe": True}))
            await asyncio.sleep(0.5)
            print(f"\nTEST after wipe ({args.test}s) — memory reset, learning OFF")
            reset_rover()
            r3 = await run_phase(ws, "clean", args.test, reward=False, learn=False)

            # put the learned memory back so the bench is non-destructive
            await ws.send(json.dumps({"type": "control", "learning": True}))
            await ws.send(json.dumps({"type": "control", "memory": json.loads(mem_json)}))
            await asyncio.sleep(0.5)

        def row(r, ref_dist):
            per100 = r["hits"] / max(r["dist"], 1) * 100
            wper = r["walls"] / max(r["dist"], 1) * 100
            return (f"{r['label']:>6}: hits/100m {per100:6.2f}  walls/100m {wper:6.2f}  "
                    f"stops {r['stops']:3d}  mean {r['mean_speed']:4.2f} m/s  "
                    f"net {r['net']:6.1f} m")

        print("\n=== RESULTS ===")
        print(row(r2, None))
        print(row(r3, None))
        d_mem, d_clean = r2["hits"] / max(r2["dist"], 1), r3["hits"] / max(r3["dist"], 1)
        if d_clean > 0:
            print(f"\nimpact-rate with memory vs clean: {d_mem / d_clean:.2f}x "
                  f"({'better' if d_mem < d_clean else 'no improvement — the connectome needs more learn time; try --learn 300+'})")
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
    ap.add_argument("--learn", type=float, default=120, help="learn phase seconds")
    ap.add_argument("--test", type=float, default=45, help="test phase seconds")
    ap.add_argument("--fresh", action="store_true", help="start with wiped bench memory")
    ap.add_argument("--density", type=float, default=1.0,
                    help="course obstacle density multiplier (1.0 = browser desert)")
    ap.add_argument("--readout", choices=["population", "pools"], default="population",
                    help="rover decode: population map or direct motor pools (option 2)")
    ap.add_argument("--keep", action="store_true", help="keep bench config file")
    asyncio.run(amain(ap.parse_args()))
