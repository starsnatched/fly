import * as THREE from "three";
import type { BrainOutputs } from "./types";

const MAX_SPEED = 4.0; // m/s at full throttle (matches the profile's speed range)
const ACCEL = 5.0; // m/s^2
const STEER_RATE = 1.6; // rad/s at full steer
const ROLL_FRICTION = 2.2; // m/s^2 rolling resistance
const SKID_FRICTION = 8.0; // 1/s lateral scrub — tires don't roll sideways

/** Skid-steer rover embodiment: throttle drives both wheel pairs, steer is a
 *  differential. All cognition lives behind the API — this is pure mechanics. */
export class Rover {
  /** chassis origin at wheel-contact height (0 = wheels on the ground) */
  pos = new THREE.Vector3(0, 0, 20);
  vel = new THREE.Vector3();
  yaw = 0; // body -Z = world -Z: face down the course
  pitch = 0; // derived: suspension pitch from acceleration
  roll = 0; // derived: suspension roll from steering
  odometer = 0; // meters travelled (wheel odometry the brain senses)
  bumpCount = 0;
  /** gap-course trial state (managed by main.ts) */
  falls = 0;
  crossings = 0;
  pauseS = 0; // > 0 while reset-paused after a fall
  paused = false;
  /** one-shot "cleared a gap" flag, consumed by the main loop */
  clearedFlag = false;
  /** the pit the rover last fell into (respawn stays in front of it) */
  lastPit: { x: number; z: number } | null = null;
  /** pit currently being approached (crossing detection) */
  pendingPit: { x: number; z: number; l: number } | null = null;

  /** Manual override nudge (WASD), decayed over time. */
  nudge = { fwd: 0, steer: 0 };

  private chassis: THREE.Group;
  private wheelFL: THREE.Mesh;
  private wheelFR: THREE.Mesh;
  private wheelRL: THREE.Mesh;
  private wheelRR: THREE.Mesh;
  private wheelSpin = 0;

  constructor() {
    this.chassis = new THREE.Group();

    // body plate + sensor mast
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(1.1, 0.28, 1.5),
      new THREE.MeshStandardMaterial({ color: 0xcfa14e, roughness: 0.5, metalness: 0.35 }),
    );
    body.position.y = 0.36;
    this.chassis.add(body);

    const mast = new THREE.Mesh(
      new THREE.CylinderGeometry(0.04, 0.05, 0.5, 8),
      new THREE.MeshStandardMaterial({ color: 0x8a8f98, roughness: 0.4, metalness: 0.6 }),
    );
    mast.position.set(0, 0.72, -0.35);
    this.chassis.add(mast);

    const eye = new THREE.Mesh(
      new THREE.SphereGeometry(0.09, 12, 12),
      new THREE.MeshStandardMaterial({ color: 0x111418, roughness: 0.2, metalness: 0.1 }),
    );
    // matches the capture camera in vision.ts (EYE_Z): this IS the sensor
    eye.position.set(0, 0.98, -0.55);
    this.chassis.add(eye);

    // four wheels
    const wheelGeo = new THREE.CylinderGeometry(0.26, 0.26, 0.18, 14);
    wheelGeo.rotateZ(Math.PI / 2);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x2c2c30, roughness: 0.9 });
    const mk = (x: number, z: number) => {
      const w = new THREE.Mesh(wheelGeo, wheelMat);
      w.position.set(x, 0.26, z);
      this.chassis.add(w);
      return w;
    };
    this.wheelFL = mk(-0.62, -0.52);
    this.wheelFR = mk(0.62, -0.52);
    this.wheelRL = mk(-0.62, 0.52);
    this.wheelRR = mk(0.62, 0.52);

    this.chassis.position.copy(this.pos);
  }

  /** Add the rover's visual representation to the scene. */
  attach(scene: THREE.Scene): void {
    scene.add(this.chassis);
  }

  get forwardSpeed(): number {
    // velocity along body -Z
    return -(this.vel.x * Math.sin(this.yaw) + this.vel.z * Math.cos(this.yaw));
  }

  /** Freeze commands while paused after a fall (trial reset). */
  startPause(): void {
    this.paused = true;
    this.pauseS = 1.0;
  }

  step(cmd: BrainOutputs, dt: number): void {
    if (this.paused) {
      this.pauseS -= dt;
      if (this.pauseS <= 0) this.paused = false;
      return; // hold still: a stopped rover sends no new optics to learn from
    }
    const throttle = Math.max(0, Math.min(1, cmd.throttle));
    const steer = Math.max(-1, Math.min(1, cmd.steer));

    // skid-steer kinematics: the heading turns FIRST, then the old momentum
    // is resolved onto the NEW body axes. Lateral velocity bleeds off fast
    // (tires scrub — the chassis cannot strafe); forward velocity follows
    // the drive model: resistance ramps from rest, brake-to-zero, never
    // reverse-accelerate from a forward-only throttle.
    this.yaw += steer * STEER_RATE * dt * Math.min(1, this.vel.length() / 0.6 + 0.25);

    const fx = -Math.sin(this.yaw), fz = -Math.cos(this.yaw); // body forward
    const rx = Math.cos(this.yaw), rz = -Math.sin(this.yaw);  // body right
    let vF = this.vel.x * fx + this.vel.z * fz;
    let vL = this.vel.x * rx + this.vel.z * rz;

    const sp0 = Math.abs(vF);
    const resist = ROLL_FRICTION * Math.min(1, sp0 / 0.5) + 0.5 * sp0;
    let dv = (throttle * ACCEL - resist) * dt;
    if (throttle * ACCEL < resist && vF + dv < 0) {
      dv = -vF; // brake to zero, never past it
    }
    vF += dv;
    vL *= Math.exp(-SKID_FRICTION * dt);

    this.vel.set(fx * vF + rx * vL, 0, fz * vF + rz * vL);
    const sp = this.vel.length();
    if (sp > MAX_SPEED) this.vel.multiplyScalar(MAX_SPEED / sp);

    this.pos.addScaledVector(this.vel, dt);
    this.odometer += vF > 0 ? vF * dt : 0;

    // suspension lean: pitch back under acceleration, roll into turns
    const accel = (throttle * ACCEL - ROLL_FRICTION);
    this.pitch += ((accel / 40) - this.pitch) * 6 * dt;
    this.roll += ((-steer * Math.min(1, sp / MAX_SPEED)) / 14 - this.roll) * 6 * dt;

    // wheel spin animation from odometry rate
    this.wheelSpin += (this.forwardSpeed / 0.26) * dt;
    for (const w of [this.wheelFL, this.wheelFR, this.wheelRL, this.wheelRR]) {
      w.rotation.x = this.wheelSpin;
    }
    // steering splay on the front pair (visual only)
    this.wheelFL.rotation.y = steer * 0.45;
    this.wheelFR.rotation.y = steer * 0.45;
  }

  syncMesh(): void {
    this.chassis.position.copy(this.pos);
    this.chassis.rotation.set(this.pitch, this.yaw, this.roll, "YXZ");
  }

  respawn(): void {
    this.pos.set((Math.random() - 0.5) * 16, 0, 20);
    this.vel.set(0, 0, 0);
    this.yaw = 0;
    this.pitch = this.roll = 0;
    this.paused = false;
    this.pauseS = 0;
  }
}
