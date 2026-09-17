// Headless gap-course geometry validation (run with node from repo root)
// Requires: cd examples/rover-web && npx esbuild src/world.ts --bundle
//           --format=cjs --outfile=../../build/world-test.cjs
const w = require("../build/world-test.cjs");

w.buildScene("gaps");

// discover pits by probing pitAt on a grid (the array itself is private)
const pits = [];
for (let z = -200; z <= 5; z += 0.5) {
  for (let x = -60; x <= 60; x += 0.5) {
    const p = w.pitAt(x, z);
    if (p && !pits.some((q) => q.x === p.x && q.z === p.z)) pits.push(p);
  }
}
console.log("pits discovered:", pits.length);
if (pits.length < 4) throw new Error("gap course should open several pits");

const byZ = pits.slice().sort((a, b) => b.z - a.z);
const p0 = byZ[0];
console.log("first pit:", JSON.stringify(p0));

// containment: inside vs just outside
const inside = w.pitAt(p0.x, p0.z);
const outsideX = w.pitAt(p0.x + p0.w / 2 + 1, p0.z);
const outsideZ = w.pitAt(p0.x, p0.z + p0.l / 2 + 1);
console.log(
  "containment inside:", !!inside,
  "| outside(x+):", !outsideX,
  "| outside(z+):", !outsideZ,
);
if (!inside || outsideX || outsideZ) throw new Error("pitAt containment wrong");

// forward cone: stand before the pit facing -Z (yaw 0), see it at rim distance
const pos = { x: p0.x, y: 0, z: p0.z + p0.l / 2 + 12 };
const ahead = w.nearestPitAhead(pos, 0);
const want = 12; // flat - half-extent along z
console.log("pit ahead dist:", ahead ? +ahead.dist.toFixed(1) : null,
  "(expect ~" + want + ")");
if (!ahead || Math.abs(ahead.dist - want) > 1.0) {
  throw new Error("nearestPitAhead distance wrong");
}

// facing away: the same pit must be invisible
const behind = w.nearestPitAhead(pos, Math.PI);
console.log("facing away sees pit:", behind !== null, "(expect false)");
if (behind) throw new Error("cone detection leaks backward");

// clearanceAhead accounts for pits: much closer than the 90 m default
const clr = w.clearanceAhead(pos, 0);
console.log("clearanceAhead:", +clr.toFixed(1), "m (expect ~= pit dist)");
if (clr > 30) throw new Error("clearanceAhead ignores pits");

console.log("GAP GEOMETRY OK");
