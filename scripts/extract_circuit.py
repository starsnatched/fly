#!/usr/bin/env python3
"""
Extract a flight-relevant visual-motion circuit from the MaleCNS v1.0 connectome.

Seeds:
  - T4/T5 neurons (optic-lobe elementary-motion detectors, ON and OFF channels)
  - visual_projection neurons (LC / LPLC / LT / TmY cells leaving the optic lobe)
Expansion:
  - keep neurons on a seed -> DN path of length <= MAX_HOPS
    (forward BFS depths from seeds + backward BFS depths from DNs,
    via CSR adjacency over the 23M strong edges)
  - per-population caps on sprawling classes
Output:
  public/fly-circuit.json
    neurons[{id, pop, type, side}], pruned edges for visualization,
    plus a population-level aggregated weight matrix used by the controller.

Data: HHMI Janelia / Google Research, MaleCNS v1.0, CC-BY 4.0.
      Berg et al., "Sexual dimorphism in the complete connectome of the
      Drosophila male central nervous system", Cell (2026).
"""

import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

DATA = Path(__file__).parent.parent / "data"
OUT = Path(__file__).parent.parent / "public" / "fly-circuit.json"

MAX_HOPS = 3
MIN_EDGE_W = 3

POP_CAPS = {
    "T4": 5000, "T5": 5000, "lamina": 1500, "Tm": 2500, "TmY": 2000,
    "optic-other": 4000, "inter": 3000,
}

POP_RULES = [
    (r"^T4", "T4"), (r"^T5", "T5"),
    (r"^(HS|VS|CH|FD|H1|H2)", "lp-tangential"),
    (r"^LPLC", "LPLC"), (r"^LC([0-9]|[A-Z])", "LC"), (r"^LT", "LT"),
    (r"^TmY", "TmY"), (r"^Tm", "Tm"), (r"^L[1-5]$", "lamina"),
    (r"^DN", "descending"),
]


def col(df, *needles):
    for c in df.columns:
        lc = c.lower()
        if all(n.lower() in lc for n in needles):
            return c
    return None


def build_csr(src: np.ndarray, dst: np.ndarray):
    """Group edges by src: returns (order, indptr) so that
    dst[order[indptr[i]:indptr[i+1]]] are targets of node i (node = dense index)."""
    n_nodes = int(max(src.max(), dst.max())) + 1
    order = np.argsort(src, kind="stable")
    sorted_src = src[order]
    indptr = np.searchsorted(sorted_src, np.arange(n_nodes + 1))
    return order, indptr, n_nodes


def bfs_depths(sources, adj_order, adj_indptr, adj_dst, allowed, max_depth):
    """Multi-source BFS over CSR; returns depth dict for nodes in `allowed`."""
    depth = {}
    q = deque()
    for s in sources:
        if s in allowed and s not in depth:
            depth[s] = 0
            q.append(s)
    while q:
        n = q.popleft()
        d = depth[n]
        if d >= max_depth:
            continue
        lo, hi = adj_indptr[n], adj_indptr[n + 1]
        if hi <= lo:
            continue
        for t in adj_dst[adj_order[lo:hi]].tolist():
            ti = int(t)
            if ti in allowed and ti not in depth:
                depth[ti] = d + 1
                q.append(ti)
    return depth


def population_of(type_name: str, superclass: str) -> str:
    for pat, pop in POP_RULES:
        if re.match(pat, type_name):
            return pop
    if "descending" in superclass:
        return "descending"
    if "visual" in superclass or superclass.startswith("ol_"):
        return "optic-other"
    return "inter"


def main():
    print("Loading annotations...")
    ann = feather.read_table(DATA / "malecns-annotations.feather").to_pandas()
    print(f"  {len(ann):,} neurons")

    idc = "bodyId"
    df = ann[ann["status"] == "Traced"].copy()
    print(f"  status==Traced: {len(df):,}")

    ty = df["type"].astype(str)
    sc = df["superclass"].astype(str)
    seeds_motion = df[ty.str.match(r"^(T4|T5)", na=False)]
    seeds_projection = df[sc.eq("visual_projection")]
    dns = df[sc.eq("descending_neuron") | ty.str.match(r"^DN", na=False)]
    print(f"  seeds: T4/T5={len(seeds_motion):,} visual_projection={len(seeds_projection):,} DNs={len(dns):,}")

    seed_ids = frozenset(seeds_motion[idc].astype("int64")) | frozenset(seeds_projection[idc].astype("int64"))
    dn_ids = frozenset(dns[idc].astype("int64"))
    if not seed_ids or not dn_ids:
        sys.exit("FATAL: empty seed/DN sets")

    id_to_idx = {int(b): i for i, b in enumerate(df[idc].astype("int64").tolist())}
    type_of = df["type"].astype(str).to_dict()
    side_of = {int(k): ("" if v != v else str(v)) for k, v in df["somaSide"].items()}
    super_of = df["superclass"].astype(str).to_dict()
    universe = frozenset(id_to_idx)

    print("Loading connectome (~1.1 GB)...")
    edges = feather.read_table(DATA / "malecns-connectome.feather").to_pandas()
    print(f"  {len(edges):,} edges")
    pre_c = col(edges, "pre") or edges.columns[0]
    post_c = col(edges, "post") or edges.columns[1]
    w_col = col(edges, "weight") or edges.columns[-1]
    pre = edges[pre_c].astype("int64").to_numpy()
    post = edges[post_c].astype("int64").to_numpy()
    w = edges[w_col].fillna(0).astype("int64").to_numpy()
    strong = w >= MIN_EDGE_W
    sp, st, sw = pre[strong], post[strong], w[strong]
    print(f"  strong edges (w>={MIN_EDGE_W}): {len(sp):,}")

    # CSR indexes on dense-ish id space (ids are large; use searchsorted mapping)
    all_ids = np.array(sorted(id_to_idx), dtype="int64")
    print("Building adjacency indexes...")
    sp_m = np.searchsorted(all_ids, sp)
    st_m = np.searchsorted(all_ids, st)
    valid = (sp_m < len(all_ids)) & (all_ids[np.minimum(sp_m, len(all_ids) - 1)] == sp) \
        & (st_m < len(all_ids)) & (all_ids[np.minimum(st_m, len(all_ids) - 1)] == st)
    sp_m, st_m, sw_v = sp_m[valid], st_m[valid], sw[valid]
    print(f"  edges within annotated set: {len(sp_m):,}")

    fwd_order, fwd_ptr, n_nodes = build_csr(sp_m, st_m)
    rev_order, rev_ptr, _ = build_csr(st_m, sp_m)
    print(f"  CSR ready ({n_nodes} slots)")

    seed_idx = {id_to_idx[n] for n in seed_ids}
    dn_idx = {id_to_idx[n] for n in dn_ids}

    d_seed = bfs_depths(seed_idx, fwd_order, fwd_ptr, st_m, universe_idx := universe, MAX_HOPS) \
        if False else None
    # (simpler: BFS directly with dicts below)
    d_seed = bfs_depths_idx(seed_idx, fwd_order, fwd_ptr, st_m, MAX_HOPS)
    d_back = bfs_depths_idx(dn_idx, rev_order, rev_ptr, sp_m, MAX_HOPS)
    print(f"  forward-reachable <= {MAX_HOPS} hops: {len(d_seed):,}")
    print(f"  backward-reachable <= {MAX_HOPS} hops: {len(d_back):,}")

    keep = {n for n in d_seed if n in d_back and d_seed[n] + d_back[n] <= MAX_HOPS}
    keep |= dn_idx
    print(f"  on seed->DN paths of length <= {MAX_HOPS}: {len(keep):,}")

    kmask = np.isin(sp_m, np.fromiter(keep, dtype="int64", count=len(keep))) \
        & np.isin(st_m, np.fromiter(keep, dtype="int64", count=len(keep)))
    sub_pre, sub_post, sub_w = sp_m[kmask], st_m[kmask], sw_v[kmask]
    print(f"  subgraph edges: {len(sub_pre):,}")

    fanout_of = defaultdict(int)
    for p in sub_pre.tolist():
        fanout_of[p] += 1

    idx_to_id = {i: b for b, i in id_to_idx.items()}
    pop_of = {}
    by_pop = defaultdict(list)
    for n in keep:
        nid = idx_to_id[n]
        tname = type_of.get(nid, "")
        sname = super_of.get(nid, "")
        by_pop[population_of(tname if isinstance(tname, str) else "",
                             sname if isinstance(sname, str) else "")].append(n)
    neurons = []
    for pop, members in sorted(by_pop.items()):
        cap = POP_CAPS.get(pop)
        if cap and len(members) > cap:
            members = sorted(members, key=lambda n: -fanout_of.get(n, 0))[:cap]
            print(f"  pop {pop}: capped to {len(members):,}")
        for n in members:
            pop_of[n] = pop
            nid = idx_to_id[n]
            neurons.append({
                "id": int(nid),
                "pop": pop,
                "type": (type_of.get(nid, "") if isinstance(type_of.get(nid, ""), str) else "")[:32],
                "side": (side_of.get(nid) or "?")[:1],
            })
    idset = frozenset(pop_of)
    print("Populations:", {k: sum(1 for v in pop_of.values() if v == k)
                           for k in sorted(set(pop_of.values()))})

    pop_names = sorted(set(pop_of.values()))
    pop_idx = {p: i for i, p in enumerate(pop_names)}
    agg = defaultdict(int)
    vis_by_target = defaultdict(list)
    for a, b, wt in zip(sub_pre.tolist(), sub_post.tolist(), sub_w.tolist()):
        if a in idset and b in idset:
            pa, pb = pop_of[a], pop_of[b]
            if pa != pb:
                agg[(pop_idx[pa], pop_idx[pb])] += int(wt)
            vis_by_target[b].append((a, int(wt)))
    pop_matrix = [[pop_names[pa], pop_names[pb], wt] for (pa, pb), wt in sorted(agg.items())]

    final_edges = []
    for tgt, lst in vis_by_target.items():
        lst.sort(key=lambda x: -x[1])
        k = max(6, int(len(lst) * 0.25))
        for src, wt in lst[:k]:
            final_edges.append((src, tgt, wt))

    ordered = sorted(idset)
    idx = {b: i for i, b in enumerate(ordered)}
    payload = {
        "meta": {
            "dataset": "MaleCNS v1.0 - HHMI Janelia / Google Research, Cell 2026",
            "license": "CC-BY 4.0",
            "maxHops": MAX_HOPS,
            "minSynapseWeight": MIN_EDGE_W,
            "populations": pop_names,
        },
        "neurons": neurons,
        "edges": [[idx[a], idx[b], wt] for (a, b, wt) in final_edges],
        "popMatrix": pop_matrix,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, "
          f"{len(neurons)} neurons, {len(final_edges):,} vis edges, "
          f"{len(pop_matrix)} pop-pairs)")


def bfs_depths_idx(sources, adj_order, adj_ptr, adj_dst, max_depth):
    depth = {}
    q = deque()
    for s in sources:
        if s not in depth:
            depth[s] = 0
            q.append(s)
    while q:
        n = q.popleft()
        d = depth[n]
        if d >= max_depth:
            continue
        lo, hi = adj_ptr[n], adj_ptr[n + 1]
        if hi <= lo:
            continue
        for t in adj_dst[adj_order[lo:hi]].tolist():
            ti = int(t)
            if ti not in depth:
                depth[ti] = d + 1
                q.append(ti)
    return depth


if __name__ == "__main__":
    main()
