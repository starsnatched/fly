import * as THREE from "three";
import type { BrainOutputs } from "./brain";

export const WORLD = {
  groundY: 0,
  ceilingY: 26,
  courseLength: 420,
};

const THRUST_MAX = 34; // m/s^2 at full throttle
const DRAG_LIN = 0.24;
const DRAG_QUAD = 0.015;
const MAX_TILT = 0.9; // rad

export class Drone {
  pos = new THREE.Vector3(0, 3.5, 0);
  vel = new THREE.Vector3();
  yaw = 0;
  pitch = 0;
  roll = 0;
  yawRate = 0;
  pitchRate = 0;
  rollRate = 0;
  alive = true;
  crashCount = 0;
  lastCrashAt = -10;

  /** Manual override nudge (WASD), decayed over time. */
  nudge = { fwd: 0, right: 0, up: 0, yaw: 0 };

  step(cmd: BrainOutputs, dt: number): void {
    const targetPitch = cmd.pitch * MAX_TILT;
    const targetRoll = cmd.roll * MAX_TILT;
    const targetYawRate = cmd.yaw * 1.9;

    this.pitchRate += ((targetPitch - this.pitch) * 8 - this.pitchRate * 6) * dt;
    this.rollRate += ((-targetRoll - this.roll) * 8 - this.rollRate * 6) * dt;
    this.yawRate += (targetYawRate - this.yawRate) * 9 * dt;
    this.pitch += this.pitchRate * dt;
    this.roll += this.rollRate * dt;
    this.yaw += this.yawRate * dt;

    const q = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(this.pitch, this.yaw, this.roll, "YXZ"),
    );
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(q);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(q);

    const acc = new THREE.Vector3();
    acc.addScaledVector(up, cmd.throttle * THRUST_MAX);
    acc.y -= 9.81;

    // manual nudges
    acc.addScaledVector(fwd, this.nudge.fwd * 6);
    acc.addScaledVector(right, this.nudge.right * 6);
    acc.y += this.nudge.up * 8;
    this.yawRate += this.nudge.yaw * 1.2;

    const sp = this.vel.length();
    acc.addScaledVector(this.vel, -(DRAG_LIN + DRAG_QUAD * sp));

    this.vel.addScaledVector(acc, dt);
    this.pos.addScaledVector(this.vel, dt);

    if (this.pos.y < WORLD.groundY + 0.25) {
      this.pos.y = WORLD.groundY + 0.25;
      if (this.vel.y < -6) this.crash("ground"); // only hard slams count
      this.vel.y = Math.max(0, this.vel.y);
      this.vel.multiplyScalar(0.92);
    }
    if (this.pos.y > WORLD.ceilingY) {
      this.pos.y = WORLD.ceilingY;
      this.vel.y = Math.min(0, this.vel.y);
    }
  }

  crash(kind: string): void {
    this.crashCount++;
    this.lastCrashAt = performance.now() / 1000;
    this.alive = false;
    setTimeout(() => this.respawn(), 1600);
    console.warn(`[drone] ${kind} impact #${this.crashCount}`);
  }

  respawn(): void {
    this.pos.set((Math.random() - 0.5) * 20, 3.5 + Math.random() * 2, 20);
    this.vel.set(0, 0, 0);
    this.yaw = 0; // body -Z = world -Z: face down the course
    this.pitch = this.roll = this.pitchRate = this.rollRate = this.yawRate = 0;
    this.alive = true;
  }
}
