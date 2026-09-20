#!/usr/bin/env python3
"""Train the fly brain from raw video — no control labels, no simulator.

How learning without labels works here: the circuit's plasticity is
dopamine-gated R-STDP (cengine/circuit.c — weight change = a_ltp/ltd x
dopa_error x eligibility). Eligibility is built from pre/post spike
coactivity while a frame is processed; the dopamine error is SELF-regulated
(DAN fast-minus-slow activity), moved by reward pulses. So a video becomes
training data through EVENTS: this script watches the stream and pulses the
reward line when something visually notable happens — a scene cut, an
abrupt appearance, a sudden change. The retina/association synapses that
were co-active right then get strengthened; when a sustained stimulus goes
static again, the engine's own baseline adaptation produces a negative
error and depresses what stopped mattering. The video is the only teacher:
no actions are commanded, none are needed, and the brain's motor output is
simply ignored (optionally recorded to CSV for inspection).

Wire format is identical to scripts/airsim_drone.py:
  eye frame: [1 u8][eye u8][w u16][h u16][3 u8] + w*h*3 RGB bytes, x2 eyes
  state:     {"type":"state","altitude":..,"speed":..,"vy":..,"collision":false}
  reward:    {"type":"reward","value":v}     (engine gain 0.3, dan clamp +-1.5)

Usage:
  .venv/Scripts/python.exe scripts/train_from_video.py clip.mp4
  .venv/Scripts/python.exe scripts/train_from_video.py clip.mp4 --loop
  .venv/Scripts/python.exe scripts/train_from_video.py --camera 0
  .venv/Scripts/python.exe scripts/train_from_video.py clip.mp4 --speed 2

Defaults target the dedicated video brain (config/video-brain.json,
ws 8791) so flight memory in state/brain-memory.json is untouched. Point
--brain-ws at ws://127.0.0.1:8787/stream to train the flying brain itself.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time

import cv2
import numpy as np
import websockets

W, H = 192, 108            # retina size (matches scripts/airsim_drone.py)
MAX_FPS = 120.0            # engine visionHz cap; faster is wasted frames
DIFF_DOWNSCALE = (64, 36)  # frame-difference analysis resolution


def pack_eye_frame(eye: int, rgb: np.ndarray) -> bytes:
    return struct.pack("<BBHHB", 1, eye, W, H, 3) + rgb.tobytes()


class VideoSource:
    """cv2 capture wrapper: file (with optional loop + speed) or camera."""

    def __init__(self, path: str | None, camera: int | None, fps_override: float,
                 loop: bool):
        self.camera = camera
        self.path = path
        self.fps_override = fps_override
        self.loop = loop
        self.cap = self._open()
        self.native_fps = fps_override
        self.total_frames = 0
        if camera is None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.native_fps = fps_override or (fps if fps and fps > 1 else 30.0)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def _open(self) -> cv2.VideoCapture:
        if self.camera is not None:
            cap = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            src = f"camera {self.camera}" if self.camera is not None else self.path
            raise SystemExit(f"cannot open video source: {src}")
        return cap

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.path is not None and self.loop:
            self.cap.release()
            self.cap = self._open()
            ok, frame = self.cap.read()
        return ok, frame

    def release(self) -> None:
        self.cap.release()


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.src = VideoSource(args.video, args.camera, args.fps, args.loop)
        self.dopa = 0.0
        self.learning = None
        self.mem_edited = 0
        self.frames_sent = 0
        self.pulses_sent = 0
        self.cuts = 0
        self._last_pulses = 0.0
        self._prev_gray: np.ndarray | None = None
        self._last_tel = 0.0
        self._last_status = 0.0

    # ---- wire helpers ----------------------------------------------------
    @staticmethod
    async def handshake(ws) -> None:
        await ws.send(json.dumps({"type": "hello", "profile": "drone"}))
        ack = json.loads(await ws.recv())
        circ = ack.get("circuit", ack)
        print(f"[brain] {circ.get('neurons', '?')} neurons / "
              f"{circ.get('edges', '?')} synapses; profile applied")

    async def reader_loop(self, ws) -> None:
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue                     # action frames: ignored
                try:
                    obj = json.loads(msg)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if obj.get("type") == "telemetry":
                    self.dopa = float(obj.get("dopa", 0.0) or 0.0)
                    self.learning = bool(obj.get("learning", False))
                    self.mem_edited = int(obj.get("memEdited", 0) or 0)
        except websockets.exceptions.ConnectionClosed:
            pass

    # ---- reward: the video is the teacher --------------------------------
    def frame_novelty(self, frame: np.ndarray) -> float:
        """Mean absolute difference from the previous frame, 0..1.
        Slow pans land ~0.005-0.03; scene cuts ~0.2-0.6."""
        small = cv2.resize(frame, DIFF_DOWNSCALE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None or gray.shape != self._prev_gray.shape:
            self._prev_gray = gray
            return 0.0
        d = float(np.mean(np.abs(gray.astype(np.int16)
                               - self._prev_gray.astype(np.int16)))) / 255.0
        self._prev_gray = gray
        return d

    async def maybe_pulse(self, ws, novelty: float, now: float) -> None:
        if novelty < self.args.cut_thresh:
            return
        if now - self._last_pulses < self.args.cooldown:
            return
        self._last_pulses = now
        self.cuts += 1
        v = self.args.reward_mag
        await ws.send(json.dumps({"type": "reward", "value": round(v, 3)}))
        self.pulses_sent += 1
        print(f"[video] scene change (d={novelty:.2f}) -> reward +{v:.2f} "
              f"(cuts {self.cuts}, pulses {self.pulses_sent})", flush=True)

    # ---- main loop -------------------------------------------------------
    async def run(self) -> int:
        period = 1.0 / min(self.src.native_fps * self.args.speed, MAX_FPS)
        mem_start = None
        print(f"[video] source: "
              f"{'camera ' + str(self.args.camera) if self.args.camera is not None else self.args.video}"
              f"  native {self.src.native_fps:.1f} fps  "
              f"playback x{self.args.speed:g} ({period * 1000:.0f} ms/frame)")
        async with websockets.connect(self.args.brain_ws, open_timeout=30,
                                      max_size=32 * 2**20) as ws:
            await self.handshake(ws)
            reader = asyncio.create_task(self.reader_loop(ws))
            next_t = time.perf_counter()
            frame_idx = 0
            try:
                while True:
                    if self.args.duration and frame_idx * period >= self.args.duration:
                        print(f"[video] --duration {self.args.duration:g}s reached")
                        break
                    ok, frame = self.src.read()
                    if not ok:
                        print("[video] source exhausted (no --loop)")
                        break
                    rgb = cv2.cvtColor(cv2.resize(frame, (W, H),
                                                  interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2RGB)
                    await ws.send(pack_eye_frame(0, rgb))
                    await ws.send(pack_eye_frame(1, rgb))
                    await ws.send(json.dumps({
                        "type": "state", "altitude": 10.0, "speed": 5.0,
                        "vy": 0.0, "collision": False}))
                    self.frames_sent += 1
                    await self.maybe_pulse(
                        ws, self.frame_novelty(frame), time.perf_counter())

                    now = time.perf_counter()
                    if mem_start is None and self.mem_edited:
                        mem_start = self.mem_edited
                        print(f"[brain] learning {'ON' if self.learning else 'OFF!'}"
                              f"  memEdited {mem_start}")
                    if now - self._last_status > 2.0:
                        self._last_status = now
                        pos = (f" frame {frame_idx}/{self.src.total_frames}"
                               if self.src.total_frames else f" frame {frame_idx}")
                        print(f"[video]{pos}  cuts {self.cuts}  "
                              f"pulses {self.pulses_sent}  dopa {self.dopa:+.3f}  "
                              f"memEdited {self.mem_edited}", flush=True)
                    if now - self._last_tel > 1.0:
                        self._last_tel = now
                        await ws.send(json.dumps({"type": "telemetry"}))

                    frame_idx += 1
                    next_t += period
                    sleep = next_t - time.perf_counter()
                    if sleep > 0:
                        await asyncio.sleep(sleep)
                    else:
                        next_t = time.perf_counter()   # fell behind: no debt
            finally:
                reader.cancel()
                self.src.release()
        edited = self.mem_edited - (mem_start or 0)
        print(f"[done] frames {self.frames_sent}  scene-change pulses "
              f"{self.pulses_sent}  memEdited {mem_start or 0} -> "
              f"{self.mem_edited} ({'+' if edited >= 0 else ''}{edited})")
        if edited <= 0:
            print("[warn] no weight edits above the 1% reporting bar - try a "
                  "larger --reward-mag, more loops, or lower --cut-thresh; "
                  "also check learning is ON")
        return 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", nargs="?", default=None,
                    help="video file (mp4/avi/mkv/mov/webm)")
    ap.add_argument("--camera", type=int, default=None, metavar="IDX",
                    help="use a live camera instead of a file (0 = default)")
    ap.add_argument("--brain-ws", default="ws://127.0.0.1:8791/stream",
                    help="brain stream url (default: the dedicated video "
                         "brain; use ws://127.0.0.1:8787/stream to train the "
                         "flying quad brain)")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="override native fps (0 = keep video fps)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier (2 = twice as fast)")
    ap.add_argument("--loop", action="store_true",
                    help="restart the video when it ends")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after this many seconds of video time "
                         "(0 = until the source ends)")
    ap.add_argument("--cut-thresh", type=float, default=0.08,
                    help="frame-difference threshold that counts as an "
                         "event (0..1)")
    ap.add_argument("--reward-mag", type=float, default=3.0,
                    help="reward pulse magnitude on a detected event; 3.0 "
                         "matches the flight brain's reward scale, so video "
                         "edits cross the same 1-percent reporting bar")
    ap.add_argument("--cooldown", type=float, default=1.0,
                    help="minimum seconds between reward pulses")
    args = ap.parse_args(argv)
    if args.camera is None and not args.video:
        ap.error("give a video file or --camera IDX")
    if args.camera is not None and args.video:
        ap.error("give either a video file or --camera, not both")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(Trainer(args).run())
    except KeyboardInterrupt:
        print("\n[video] stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
