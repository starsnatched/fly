#!/usr/bin/env python3
"""
Aggregate per-neuron neurotransmitter probabilities from the MaleCNS
T-bar NT table (45.7M rows, 2.65 GB) via streaming batches.

Output: data/neuron-nt.json  { "<bodyId>": [pACh, pGABA, pGlu, pDA, p5HT, pOA, pHA], ... }
  only bodies belonging to the LIF circuit (passed via --bodies file).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

DATA = Path(__file__).parent.parent / "data"
SRC = DATA / "tbar-nt.feather"
OUT = DATA / "neuron-nt.json"

NT_COLS = [
    "nt_acetylcholine_prob", "nt_gaba_prob", "nt_glutamate_prob",
    "nt_dopamine_prob", "nt_serotonin_prob", "nt_octopamine_prob",
    "nt_histamine_prob",
]


def main():
    bodies_file = DATA / "lif-bodies.txt"
    if not bodies_file.exists():
        sys.exit("run extract_lif_circuit.py first (writes lif-bodies.txt)")
    wanted = set(int(x) for x in bodies_file.read_text().split())
    print(f"{len(wanted):,} bodies wanted")

    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}

    t = feather.read_table(SRC, columns=["body"] + NT_COLS)
    print(f"table: {t.num_rows:,} rows")
    total = t.num_rows
    batch_size = 4_000_000
    done = 0
    for batch in t.to_batches(max_chunksize=batch_size):
        b = np.asarray(batch.column("body"))
        probs = np.stack([np.asarray(batch.column(c)).astype(np.float32) for c in NT_COLS], axis=1)
        # mask to wanted bodies
        mask = np.fromiter((int(x) in wanted for x in b), dtype=bool, count=len(b))
        if not mask.any():
            done += len(b)
            print(f"  {done / 1e6:.0f}M / {total / 1e6:.0f}M rows (none wanted)")
            continue
        bodies = b[mask].astype("int64")
        probs = probs[mask]
        # np.add.at accumulate per unique body
        uniq, inv = np.unique(bodies, return_inverse=True)
        for j, u in enumerate(uniq):
            u = int(u)
            sel = probs[inv == j]
            if u in sums:
                sums[u] += sel.sum(axis=0)
                counts[u] += len(sel)
            else:
                sums[u] = sel.sum(axis=0)
                counts[u] = len(sel)
        done += len(b)
        print(f"  {done / 1e6:.0f}M / {total / 1e6:.0f}M rows, {len(sums):,} bodies so far")

    # write out: normalized mean probabilities
    out = {}
    for u, s in sums.items():
        c = max(1, counts[u])
        mean = (s / c).astype(float)
        total_p = mean.sum()
        if total_p > 0:
            mean = mean / total_p  # renormalize (table may not sum to 1)
        out[str(u)] = [round(float(x), 4) for x in mean]
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(out):,} bodies)")


if __name__ == "__main__":
    main()
