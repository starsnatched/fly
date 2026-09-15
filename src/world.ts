import * as THREE from "three";
import { WORLD } from "./drone";

interface Obstacle {
  x: number;
  z: number;
  r: number;
  h: number;
}

const obstacles: Obstacle[] = [];

export function buildScene(): THREE.Scene {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x7d8ba0);
  scene.fog = new THREE.Fog(0x7d8ba0, 70, 380);

  const sun = new THREE.DirectionalLight(0xfff2dd, 2.0);
  sun.position.set(-60, 80, -40);
  scene.add(sun);
  scene.add(new THREE.HemisphereLight(0xaebdd4, 0x3a4150, 1.1));

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(1200, 1200),
    new THREE.MeshStandardMaterial({ color: 0x707a8b, roughness: 0.95 }),
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const grid = new THREE.GridHelper(1200, 120, 0x39404e, 0x525c6d);
  const gm = grid.material as THREE.Material;
  gm.transparent = true;
  gm.opacity = 0.4;
  grid.position.y = 0.02;
  scene.add(grid);

  const group = new THREE.Group();
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0x1c2230, roughness: 0.7, metalness: 0.15 });
  const pillarMat2 = new THREE.MeshStandardMaterial({ color: 0x282031, roughness: 0.75, metalness: 0.1 });
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x0d4d3a, transparent: true, opacity: 0.5 });

  let z = -30;
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

  const addObstacleMesh = (m: THREE.Mesh, r: number, h: number) => {
    group.add(m);
    obstacles.push({ x: m.position.x, z: m.position.z, r, h });
  };

  while (z > -WORLD.courseLength) {
    const kind = rnd();
    if (kind < 0.45) {
      // pillar cluster
      const n = 1 + Math.floor(rnd() * 3);
      for (let i = 0; i < n; i++) {
        const r = 1.2 + rnd() * 2.6;
        const h = 6 + rnd() * 17;
        const x = (rnd() - 0.5) * 70;
        const zz = z + (rnd() - 0.5) * 16;
        const m = new THREE.Mesh(
          new THREE.CylinderGeometry(r, r * 1.12, h, 10),
          rnd() < 0.5 ? pillarMat : pillarMat2,
        );
        m.position.set(x, h / 2, zz);
        addObstacleMesh(m, r * 1.12, h);
      }
    } else if (kind < 0.75) {
      // wall with a gap, marked by posts
      const gapX = (rnd() - 0.5) * 44;
      const h = 7 + rnd() * 9;
      const wL = gapX + 18 + 35;
      if (wL > 1.5) {
        const mL = new THREE.Mesh(new THREE.BoxGeometry(wL, h, 1.4), pillarMat2);
        mL.position.set(-35 + wL / 2, h / 2, z);
        addObstacleMesh(mL, wL / 2, h);
      }
      const wR = 35 - (gapX + 24);
      if (wR > 1.5) {
        const mR = new THREE.Mesh(new THREE.BoxGeometry(wR, h, 1.4), pillarMat2);
        mR.position.set(gapX + 24 + wR / 2, h / 2, z);
        addObstacleMesh(mR, wR / 2, h);
      }
      for (const px of [gapX, gapX + 6]) {
        const post = new THREE.Mesh(
          new THREE.CylinderGeometry(0.22, 0.22, h + 2, 6),
          new THREE.MeshBasicMaterial({ color: 0x35f0b0 }),
        );
        post.position.set(px, (h + 2) / 2, z);
        group.add(post);
      }
    } else {
      // overhead beam forcing low flight + one support pillar
      const y = 5.5 + rnd() * 5;
      const m = new THREE.Mesh(new THREE.BoxGeometry(70, 1.6, 1.6), pillarMat);
      m.position.set(0, y, z);
      group.add(m);
      obstacles.push({ x: 0, z, r: 0.8, h: y + 0.8 });
      const sx = rnd() < 0.5 ? -26 : 26;
      const sup = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 0.9, y, 8), pillarMat);
      sup.position.set(sx, y / 2, z);
      addObstacleMesh(sup, 1.0, y);
    }
    z -= 16 + rnd() * 22;
  }

  // sparse pillar scatter across the whole field so open-world exploration
  // always has something on the horizon
  for (let i = 0; i < 420; i++) {
    const x = (rnd() - 0.5) * 900;
    const zz = (rnd() - 0.5) * 900;
    // keep the course strip itself as generated (skip if inside it)
    if (Math.abs(x) < 40 && zz < 10) continue;
    const r = 1.5 + rnd() * 5;
    const h = 5 + rnd() * 20;
    const m = new THREE.Mesh(
      new THREE.CylinderGeometry(r, r * 1.15, h, 9),
      rnd() < 0.5 ? pillarMat : pillarMat2,
    );
    m.position.set(x, h / 2, zz);
    addObstacleMesh(m, r * 1.15, h);
  }

  // edge outlines for readability
  const meshes: THREE.Mesh[] = [];
  group.traverse((o) => {
    if ((o as THREE.Mesh).isMesh && (o as THREE.Mesh).geometry instanceof THREE.BoxGeometry) {
      meshes.push(o as THREE.Mesh);
    }
    if ((o as THREE.Mesh).isMesh && (o as THREE.Mesh).geometry instanceof THREE.CylinderGeometry) {
      const g = o as THREE.Mesh;
      if ((g.material as THREE.Material) !== (postMat() as THREE.Material)) meshes.push(g);
    }
  });
  function postMat(): THREE.Material {
    return new THREE.MeshBasicMaterial();
  }
  for (const m of meshes) {
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(m.geometry, 25),
      edgeMat,
    );
    edges.position.copy(m.position);
    group.add(edges);
  }

  scene.add(group);
  return scene;
}

/** Capsule-vs-obstacle collision test; true on impact. */
export function checkCollision(pos: THREE.Vector3): boolean {
  for (const o of obstacles) {
    if (pos.y > o.h) continue;
    const dx = pos.x - o.x;
    const dz = pos.z - o.z;
    const rr = o.r + 0.4;
    if (dx * dx + dz * dz < rr * rr) return true;
  }
  return false;
}

/**
 * Push the drone out of any obstacle it penetrates and bounce it off
 * (soft bump, not death). Returns true if a bump occurred.
 */
export function resolveCollision(pos: THREE.Vector3, vel: THREE.Vector3): boolean {
  let bumped = false;
  for (const o of obstacles) {
    if (pos.y > o.h + 0.1) continue;
    const dx = pos.x - o.x;
    const dz = pos.z - o.z;
    const dist = Math.hypot(dx, dz);
    const rr = o.r + 0.45;
    if (dist >= rr) continue;
    // push out along the radial normal (or +x if dead center)
    const nx = dist > 1e-4 ? dx / dist : 1;
    const nz = dist > 1e-4 ? dz / dist : 0;
    pos.x = o.x + nx * rr;
    pos.z = o.z + nz * rr;
    // reflect velocity with damping (restitution 0.35)
    const vn = vel.x * nx + vel.z * nz;
    if (vn < 0) {
      vel.x -= 1.35 * vn * nx;
      vel.z -= 1.35 * vn * nz;
    }
    vel.multiplyScalar(0.72);
    bumped = true;
  }
  return bumped;
}

/** Nearest obstacle surface distance within a forward cone (HUD "clear"). */
export function clearanceAhead(pos: THREE.Vector3, yaw: number): number {
  let best = 120;
  const fx = -Math.sin(yaw), fz = -Math.cos(yaw);
  for (const o of obstacles) {
    if (pos.y > o.h + 0.5) continue;
    const dx = o.x - pos.x, dz = o.z - pos.z;
    const flat = Math.hypot(dx, dz);
    if (flat < 0.5) continue;
    const dist = flat - o.r;
    if (dist < 0.1) continue;
    if ((dx * fx + dz * fz) / flat > 0.86) best = Math.min(best, dist);
  }
  return best;
}
