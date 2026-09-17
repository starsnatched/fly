import * as THREE from "three";

/** World constants shared by physics and HUD. */
export const WORLD = {
  /** collision cylinder radius of the chassis */
  roverRadius: 0.9,
  /** keep-in bounds */
  limit: 220,
};

/** Course modes: "desert" = rocks + walls; "gaps" = rock-strewn floor with
 *  open pits to steer around (fall in and the trial resets, with reward). */
export type CourseKind = "desert" | "gaps";

export interface Pit {
  x: number;
  z: number;
  /** extent across (x) */
  w: number;
  /** extent along the route (z) */
  l: number;
  depth: number;
}

export const pits: Pit[] = [];

interface Obstacle {
  x: number;
  z: number;
  /** collision radius */
  r: number;
}

const obstacles: Obstacle[] = [];

/** Deterministic RNG so every reload drives the same desert. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function buildScene(kind: CourseKind = "desert"): THREE.Scene {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xc9a86a);
  scene.fog = new THREE.Fog(0xd8b984, 60, 340);

  const sun = new THREE.DirectionalLight(0xfff1d6, 2.4);
  sun.position.set(-70, 90, -30);
  scene.add(sun);
  scene.add(new THREE.HemisphereLight(0xd8c9a8, 0x6b543a, 1.15));

  // ground: warm regolith
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(1200, 1200),
    new THREE.MeshStandardMaterial({ color: 0xb08d5e, roughness: 1.0 }),
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const grid = new THREE.GridHelper(1200, 240, 0x8a6c44, 0x9d7d51);
  const gm = grid.material as THREE.Material;
  gm.transparent = true;
  gm.opacity = 0.35;
  grid.position.y = 0.02;
  scene.add(grid);

  const rockMat = new THREE.MeshStandardMaterial({ color: 0x8a6f52, roughness: 0.9, flatShading: true });
  const darkRockMat = new THREE.MeshStandardMaterial({ color: 0x5f4a36, roughness: 0.85, flatShading: true });
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x7a6248, roughness: 0.8 });
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x3f2f1e, transparent: true, opacity: 0.5 });

  const rnd = makeRng(kind === "gaps" ? 20260918 : 20260917);
  const group = new THREE.Group();

  const addObstacleMesh = (m: THREE.Mesh, r: number) => {
    group.add(m);
    obstacles.push({ x: m.position.x, z: m.position.z, r });
  };

  // a granite boulder: irregular icosahedron, sunk into the regolith
  const addBoulder = (x: number, z: number, r: number, dark: boolean) => {
    const geo = new THREE.IcosahedronGeometry(r, 1);
    const pos = geo.getAttribute("position") as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const k = 1 + (rnd() - 0.5) * 0.35;
      pos.setXYZ(i, pos.getX(i) * k, pos.getY(i) * k * 0.8, pos.getZ(i) * k);
    }
    geo.computeVertexNormals();
    const m = new THREE.Mesh(geo, dark ? darkRockMat : rockMat);
    m.position.set(x, r * 0.62, z);
    m.rotation.y = rnd() * Math.PI * 2;
    addObstacleMesh(m, r * 1.02);
  };

  // a megalith slab wall with a gap to thread
  const addWall = (x: number, z: number, w: number, h: number, d: number) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), wallMat);
    m.position.set(x, h / 2, z);
    m.rotation.y = (rnd() - 0.5) * 0.35;
    group.add(m);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry, 30), edgeMat);
    edges.position.copy(m.position);
    edges.rotation.copy(m.rotation);
    group.add(edges);
    // circular-ish footprint: approximate the slab with a few discs
    const along = Math.abs(Math.sin(m.rotation.y)) * d + Math.cos(m.rotation.y) * w;
    const across = Math.abs(Math.cos(m.rotation.y)) * d + Math.sin(m.rotation.y) * w;
    const n = Math.max(1, Math.round(along / (across * 0.9)));
    for (let i = 0; i < n; i++) {
      const t = n === 1 ? 0 : (i / (n - 1) - 0.5) * (along - across);
      const ox = Math.cos(m.rotation.y) * t;
      const oz = -Math.sin(m.rotation.y) * t;
      obstacles.push({ x: x + ox, z: z + oz, r: across * 0.55 });
    }
  };

  // the course: dense clutter down the -Z corridor (where the rover starts)
  if (kind === "desert") {
    for (let z = -18; z > -170; z -= 9 + rnd() * 10) {
      if (rnd() < 0.32) {
        // slab wall with a threadable gap
        const gapX = (rnd() - 0.5) * 26;
        const h = 1.6 + rnd() * 1.4;
        addWall(gapX - 13 - 6, z, 20 + rnd() * 8, h, 1.6);
        addWall(gapX + 13 + 6, z, 20 + rnd() * 8, h, 1.6);
      } else {
        // boulder cluster
        const n = 2 + Math.floor(rnd() * 4);
        for (let i = 0; i < n; i++) {
          const r = 0.6 + rnd() * 1.8;
          addBoulder((rnd() - 0.5) * 60, z + (rnd() - 0.5) * 10, r, rnd() < 0.4);
        }
      }
    }
  } else {
    // gap course: pits open along the route — dark, unavoidable-looking
    // openings the eye sees as looming voids against the bright floor
    let pz = -20;
    while (pz > -190) {
      const w = 3 + rnd() * 5; // 3..8 m across
      const l = 6 + rnd() * 6; // 6..12 m along the route
      const px = (rnd() - 0.5) * 44;
      pits.push({ x: px, z: pz - l / 2, w, l, depth: 4 + rnd() * 2 });
      pz -= l + 16 + rnd() * 16; // spacing leaves room to steer around
    }
    for (const p of pits) {
      const hole = new THREE.Mesh(
        new THREE.BoxGeometry(p.w, 0.2, p.l),
        new THREE.MeshBasicMaterial({ color: 0x0a0703 }), // unlit = reads as void
      );
      hole.position.set(p.x, 0.02, p.z);
      group.add(hole);
      const rim = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(p.w, 0.2, p.l)),
        edgeMat,
      );
      rim.position.set(p.x, 0.03, p.z);
      group.add(rim);
    }
  }

  // sparse scatter over the whole field so open wandering always has terrain
  for (let i = 0; i < 340; i++) {
    const x = (rnd() - 0.5) * 440;
    const z = (rnd() - 0.5) * 440;
    if (Math.abs(x) < 34 && z < 10) continue; // keep the course strip as generated
    addBoulder(x, z, 0.5 + rnd() * 2.6, rnd() < 0.4);
  }

  // small pebbles: pure decoration, no collision (the eye sees texture)
  const pebbleGeo = new THREE.IcosahedronGeometry(1, 0);
  const pebbles = new THREE.InstancedMesh(pebbleGeo, darkRockMat, 900);
  const dummy = new THREE.Object3D();
  for (let i = 0; i < 900; i++) {
    dummy.position.set((rnd() - 0.5) * 460, 0.05 + rnd() * 0.08, (rnd() - 0.5) * 460);
    dummy.rotation.set(rnd() * Math.PI, rnd() * Math.PI, rnd() * Math.PI);
    dummy.scale.setScalar(0.12 + rnd() * 0.3);
    dummy.updateMatrix();
    pebbles.setMatrixAt(i, dummy.matrix);
  }
  group.add(pebbles);

  scene.add(group);
  return scene;
}

/** Circle-vs-obstacle collision test; true when the chassis overlaps one. */
export function checkCollision(pos: THREE.Vector3): boolean {
  for (const o of obstacles) {
    const dx = pos.x - o.x;
    const dz = pos.z - o.z;
    const rr = o.r + WORLD.roverRadius;
    if (dx * dx + dz * dz < rr * rr) return true;
  }
  return false;
}

/**
 * Push the rover out of any obstacle it penetrates and bounce it off
 * (soft bump, not death). Returns true if a bump occurred.
 */
export function resolveCollision(pos: THREE.Vector3, vel: THREE.Vector3): boolean {
  let bumped = false;
  for (const o of obstacles) {
    const dx = pos.x - o.x;
    const dz = pos.z - o.z;
    const dist = Math.hypot(dx, dz);
    const rr = o.r + WORLD.roverRadius;
    if (dist >= rr) continue;
    const nx = dist > 1e-4 ? dx / dist : 1;
    const nz = dist > 1e-4 ? dz / dist : 0;
    pos.x = o.x + nx * rr;
    pos.z = o.z + nz * rr;
    const vn = vel.x * nx + vel.z * nz;
    if (vn < 0) {
      vel.x -= 1.3 * vn * nx;
      vel.z -= 1.3 * vn * nz;
    }
    vel.multiplyScalar(0.7);
    bumped = true;
  }
  return bumped;
}

/** Pit containing (x,z), if any. */
export function pitAt(x: number, z: number): Pit | null {
  for (const p of pits) {
    if (Math.abs(x - p.x) <= p.w / 2 && Math.abs(z - p.z) <= p.l / 2) return p;
  }
  return null;
}

/** Nearest pit in a forward cone (threat zone); null when clear. */
export function nearestPitAhead(
  pos: THREE.Vector3,
  yaw: number,
): { pit: Pit; dist: number } | null {
  let best: { pit: Pit; dist: number } | null = null;
  const fx = -Math.sin(yaw), fz = -Math.cos(yaw);
  for (const p of pits) {
    const dx = p.x - pos.x, dz = p.z - pos.z;
    const flat = Math.hypot(dx, dz);
    if (flat > 30) continue;
    if ((dx * fx + dz * fz) / flat < 0.5) continue; // ~60° cone
    const d = Math.max(0, flat - Math.max(p.w, p.l) / 2);
    if (!best || d < best.dist) best = { pit: p, dist: d };
  }
  return best;
}

/** Nearest obstacle surface distance within a forward cone (HUD "clear"). */
export function clearanceAhead(pos: THREE.Vector3, yaw: number): number {
  let best = 90;
  const fx = -Math.sin(yaw), fz = -Math.cos(yaw);
  for (const o of obstacles) {
    const dx = o.x - pos.x, dz = o.z - pos.z;
    const flat = Math.hypot(dx, dz);
    if (flat < 0.5) continue;
    const dist = flat - o.r;
    if (dist < 0.1) continue;
    if ((dx * fx + dz * fz) / flat > 0.9) best = Math.min(best, dist);
  }
  // pits count too: the rim is the surface
  for (const p of pits) {
    const dx = p.x - pos.x, dz = p.z - pos.z;
    const flat = Math.hypot(dx, dz);
    if (flat < 0.5) continue;
    const dist = flat - Math.max(p.w, p.l) / 2;
    if (dist < 0.1) continue;
    if ((dx * fx + dz * fz) / flat > 0.9) best = Math.min(best, dist);
  }
  return best;
}
