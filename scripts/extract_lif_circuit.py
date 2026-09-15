#!/usr/bin/env python3
"""
Extract a per-neuron LIF-ready circuit from MaleCNS v1.0 for spiking simulation.

Same node selection as extract_circuit.py (T4/T5 + visual_projection seeds,
descending neurons as targets, seed->DN paths <= 3 hops, population caps),
but keeps per-neuron data the spiking engine needs:

  - retinotopy: assignedOlHex1/2 (the optic lobe's hexagonal column grid)
  - neurotransmitter sign heuristic (documented):
      type ^C[0-9]        -> GABAergic  (inhibitory)
      superclass *tbc/glia excluded
      everything else     -> cholinergic (excitatory) [optic-lobe default]
  - per-neuron pruned edges (strongest 25% per target, min 6) -> ~1-2M edges

Output: public/fly-lif.json
  { meta, populations[], neurons: {id[], pop[], side[], hex[][], nt[]},
    edges: flat [src,dst,w, ...] }
"""

import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

DATA = Path(__file__).parent.parent / "data"
OUT = Path(__file__).parent.parent / "public" / "fly-lif.json"

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
    n_nodes = int(max(src.max(), dst.max())) + 1
    order = np.argsort(src, kind="stable")
    sorted_src = src[order]
    indptr = np.searchsorted(sorted_src, np.arange(n_nodes + 1))
    return order, indptr, n_nodes


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


def population_of(type_name: str, superclass: str) -> str:
    for pat, pop in POP_RULES:
        if re.match(pat, type_name):
            return pop
    if "descending" in superclass:
        return "descending"
    if "visual" in superclass or superclass.startswith("ol_"):
        return "optic-other"
    return "inter"


def nt_sign(type_name: str) -> int:
    """+1 excitatory (cholinergic default), -1 inhibitory (GABA, C-types)."""
    if re.match(r"^C[0-9]", type_name):
        return -1
    return +1


def sval(x) -> str:
    return x if isinstance(x, str) else ""


def dir_of(type_name: str) -> int:
    """T4/T5 directional subtype: a=up, b=left, c=down, d=right (looming/T4c).
    Returns 0..3, or -1 if not directional."""
    m = re.match(r"^T[45]([abcd])", type_name)
    if not m:
        return -1
    return {"a": 0, "b": 1, "c": 2, "d": 3}[m.group(1)]


def main():
    print("Loading annotations...")
    ann = feather.read_table(DATA / "malecns-annotations.feather").to_pandas()
    idc = "bodyId"
    df = ann[ann["status"] == "Traced"].copy()
    print(f"  traced: {len(df):,}")

    ty = df["type"].astype(str)
    sc = df["superclass"].astype(str)
    hex1 = df["assignedOlHex1"]
    hex2 = df["assignedOlHex2"]
    seeds_motion = df[ty.str.match(r"^(T4|T5)", na=False)]
    seeds_projection = df[sc.eq("visual_projection")]
    dns = df[sc.eq("descending_neuron") | ty.str.match(r"^DN", na=False)]
    print(f"  seeds: T4/T5={len(seeds_motion):,} visual_projection={len(seeds_projection):,} DNs={len(dns):,}")

    seed_ids = frozenset(seeds_motion[idc].astype("int64")) | frozenset(seeds_projection[idc].astype("int64"))
    dn_ids = frozenset(dns[idc].astype("int64"))

    id_to_idx = {int(b): i for i, b in enumerate(df[idc].astype("int64").tolist())}
    type_of = {int(k): sval(v) for k, v in df["type"].items()}
    side_of = {int(k): ("" if v != v else sval(v)) for k, v in df["somaSide"].items()}
    super_of = {int(k): sval(v) for k, v in df["superclass"].items()}
    hex_of: dict[int, tuple[int, int]] = {}
    n_hex = 0
    for k, h1, h2 in zip(df[idc].astype("int64"), hex1, hex2):
        try:
            if h1 == h1 and h2 == h2:  # non-NaN
                hex_of[int(k)] = (int(h1), int(h2))
                n_hex += 1
        except (TypeError, ValueError):
            pass
    print(f"  with hex retinotopy: {n_hex:,}")

    print("Loading connectome (~1.1 GB)...")
    edges = feather.read_table(DATA / "malecns-connectome.feather").to_pandas()
    pre = edges["body_pre"].astype("int64").to_numpy()
    post = edges["body_post"].astype("int64").to_numpy()
    w = edges["weight"].fillna(0).astype("int64").to_numpy()
    strong = w >= MIN_EDGE_W
    sp, st, sw = pre[strong], post[strong], w[strong]

    all_ids = np.array(sorted(id_to_idx), dtype="int64")
    print("Building adjacency indexes...")
    sp_m = np.searchsorted(all_ids, sp)
    st_m = np.searchsorted(all_ids, st)
    n = len(all_ids)
    valid = (sp_m < n) & (all_ids[np.minimum(sp_m, n - 1)] == sp) \
        & (st_m < n) & (all_ids[np.minimum(st_m, n - 1)] == st)
    sp_m, st_m, sw_v = sp_m[valid], st_m[valid], sw[valid]
    print(f"  edges in annotated set: {len(sp_m):,}")

    fwd_order, fwd_ptr, n_slots = build_csr(sp_m, st_m)
    rev_order, rev_ptr, _ = build_csr(st_m, sp_m)

    seed_idx = {id_to_idx[b] for b in seed_ids}
    dn_idx = {id_to_idx[b] for b in dn_ids}
    d_seed = bfs_depths_idx(seed_idx, fwd_order, fwd_ptr, st_m, MAX_HOPS)
    d_back = bfs_depths_idx(dn_idx, rev_order, rev_ptr, sp_m, MAX_HOPS)
    keep = {i for i in d_seed if i in d_back and d_seed[i] + d_back[i] <= MAX_HOPS}
    keep |= dn_idx
    print(f"  on seed->DN paths <= {MAX_HOPS}: {len(keep):,}")

    keep_arr = np.fromiter(keep, dtype="int64", count=len(keep))
    kmask = np.isin(sp_m, keep_arr) & np.isin(st_m, keep_arr)
    sub_pre, sub_post, sub_w = sp_m[kmask], st_m[kmask], sw_v[kmask]
    print(f"  subgraph edges: {len(sub_pre):,}")

    fanout_of = defaultdict(int)
    for p in sub_pre.tolist():
        fanout_of[p] += 1

    idx_to_id = {i: b for b, i in id_to_idx.items()}
    pop_of = {}
    by_pop = defaultdict(list)
    for i in keep:
        nid = idx_to_id[i]
        t = type_of.get(nid, "")
        s = super_of.get(nid, "")
        by_pop[population_of(t, s)].append(i)
    members_of: dict[int, str] = {}
    for pop, members in sorted(by_pop.items()):
        cap = POP_CAPS.get(pop)
        if cap and len(members) > cap:
            members = sorted(members, key=lambda x: -fanout_of.get(x, 0))[:cap]
        for i in members:
            pop_of[i] = pop
    print("Populations:", {k: sum(1 for v in pop_of.values() if v == k)
                           for k in sorted(set(pop_of.values()))})

    idset = frozenset(pop_of)
    # per-target prune for the LIF engine
    by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a, b, wt in zip(sub_pre.tolist(), sub_post.tolist(), sub_w.tolist()):
        if a in idset and b in idset:
            by_target[b].append((a, int(wt)))
    final: list[tuple[int, int, int]] = []
    for tgt, lst in by_target.items():
        lst.sort(key=lambda x: -x[1])
        k = max(6, int(len(lst) * 0.25))
        for src, wt in lst[:k]:
            final.append((src, tgt, wt))
    print(f"  LIF edges: {len(final):,}")

    # payload: SoA arrays for compactness
    ordered = sorted(idset)
    # merge real per-neuron NT probabilities if available (aggregate_nt.py)
    nt_file = DATA / "neuron-nt.json"
    nt_lookup: dict[str, list[float]] = {}
    if nt_file.exists():
        nt_lookup = json.loads(nt_file.read_text())
        print(f"  real NT profiles: {len(nt_lookup):,} bodies")

    # write body-id list for the NT aggregation step (for future refreshes)
    (DATA / "lif-bodies.txt").write_text("\n".join(str(idx_to_id[b]) for b in ordered))
    remap = {b: i for i, b in enumerate(ordered)}
    pop_names = sorted(set(pop_of.values()))
    pop_index = {p: i for i, p in enumerate(pop_names)}

    ids_out: list[int] = []
    pop_out: list[int] = []
    side_out: list[int] = []
    hex_out: list[list[int]] = []
    nt_out: list[int] = []
    dir_out: list[int] = []
    nt_conf_out: list[float] = []
    heuristic_signs = 0
    for b in ordered:
        nid = idx_to_id[b]
        t = type_of.get(nid, "")
        h = hex_of.get(nid, (-1, -1))
        ids_out.append(int(nid))
        pop_out.append(pop_index[pop_of[b]])
        side_out.append(1 if (side_of.get(nid) or "")[:1] == "R" else 0)
        hex_out.append([h[0], h[1]])
        dir_out.append(dir_of(t))
        prof = nt_lookup.get(str(nid))
        if prof:
            ach, gaba, glu = prof[0], prof[1], prof[2]
            if gaba >= 0.5:
                nt_out.append(-1)
            elif ach >= 0.5 or glu >= 0.5:
                nt_out.append(1)
            else:
                nt_out.append(nt_sign(t))  # weak profile: fall back to type rule
                heuristic_signs += 1
            nt_conf_out.append(round(max(prof), 3))
        else:
            nt_out.append(nt_sign(t))
            nt_conf_out.append(0.0)
            heuristic_signs += 1
    if heuristic_signs:
        print(f"  {heuristic_signs:,} neurons used type-rule fallback (weak/absent NT profile)")

    flat_edges: list[int] = []
    for a, b, wt in final:
        flat_edges.append(remap[a])
        flat_edges.append(remap[b])
        flat_edges.append(wt)

    payload = {
        "meta": {
            "dataset": "MaleCNS v1.0 - HHMI Janelia / Google Research, Cell 2026",
            "license": "CC-BY 4.0",
            "maxHops": MAX_HOPS,
            "minSynapseWeight": MIN_EDGE_W,
            "ntHeuristic": "per-neuron mean NT probabilities from 45.7M T-bars (tbar-neurotransmitters table); GABA>=0.5 -> inhibitory, else excitatory; type-rule fallback for weak profiles",
            "ntSource": "tbar-neurotransmitters-male-cns-v1.0.feather",
            "hexFallback": [-1, -1],
            "dirConvention": "T4/T5 subtype: a=up b=left c=down d=right, -1=n/a",
        },
        "populations": pop_names,
        "neurons": {
            "id": ids_out,
            "pop": pop_out,
            "side": side_out,
            "hex": hex_out,
            "nt": nt_out,
            "dir": dir_out,
            "ntConf": nt_conf_out,
        },
        "edges": flat_edges,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, "
          f"{len(ordered)} neurons, {len(final):,} edges, "
          f"{sum(1 for h in hex_out if h[0] >= 0):,} retinotopic)")


if __name__ == "__main__":
    main()
