#!/usr/bin/env python3
"""
Extract the ENTIRE traced MaleCNS v1.0 brain for full-scale spiking simulation.

Unlike extract_lif_circuit.py (which keeps a pruned seed->DN subgraph), this
keeps EVERY traced neuron and every traced->traced synapse (weight >= 1):
  - 165,122 neurons, ~25.5M synapses (the whole CNS connectome)
  - population = real superclass (ol_intrinsic, cb_intrinsic, vnc_motor, ...)
  - functional group per neuron (lamina, T4/T5, LC/LPLC/LT, descending,
    mushroom-body KC/MBON/DAN, motor, ...) for biophysics + sensory mapping
  - real NT signs from neuron-nt.json (45.7M T-bars), type-rule fallback
  - hex retinotopy + T4/T5 a/b/c/d directional subtypes where annotated
  - a plastic sub-circuit for reward learning: strongest 128 synapses per
    descending neuron (R-STDP memory attaches there)

Output (data/fly-brain-full.bin + .meta.json), little-endian:
  header   "FLYBRAIN1" (9 bytes)
  u32      magic 0x4E455552 ('NEUR' little-endian)
  u32      N, u32 E, u32 P (populations), u32 G (groups)
  str32+P  population names
  str32+G  group names
  N*u8     group per neuron
  N*u8     population per neuron
  N*i8     nt sign (+1 exc, -1 inh)
  N*i8     T4/T5 direction 0..3, -1 n/a
  N*u8     side (1 = right)
  N*i16x2  hex retinotopy (-1,-1 fallback)
  N*f32    NT confidence
  (E)*u32  edgeSrc CSR order, (E)*u32 edgeDst, (E)*f32 edgeW (synapse count)

Run:  .venv/Scripts/python scripts/extract_full_brain.py
"""

import json
import re
import struct
import time
from pathlib import Path

import numpy as np
from pyarrow import feather

DATA = Path(__file__).parent.parent / "data"
OUT_BIN = Path(__file__).parent.parent / "public" / "fly-brain-full.bin"
OUT_META = Path(__file__).parent.parent / "public" / "fly-brain-full.meta.json"

MIN_EDGE_W = 1  # keep every traced synapse
PLASTIC_K = 128  # strongest inputs per DN kept plastic (R-STDP)
TOP_INPUTS_CAP = 4096  # safety cap on one neuron's inputs (guards dense hubs)


# ---- functional groups (drive the biophysics + sensory mapping) ----


def group_of(t: str, sc: str) -> str:
    if re.match(r"^T4", t):
        return "T4"
    if re.match(r"^T5", t):
        return "T5"
    if re.match(r"^L[1-5]$", t):
        return "lamina"
    if re.match(r"^L([6-9]|N|W|U)", t):
        return "lamina"
    if re.match(r"^TmY", t):
        return "TmY"
    if re.match(r"^Tm", t):
        return "Tm"
    if re.match(r"^LPLC", t):
        return "LPLC"
    if re.match(r"^LC([0-9]|[A-Z])", t):
        return "LC"
    if re.match(r"^LT", t):
        return "LT"
    if re.match(r"^(HS|VS|CH|FD|H[12])", t):
        return "lp-tangential"
    if re.match(r"^DN", t) or sc == "descending_neuron":
        return "descending"
    if re.match(r"^AN", t) or sc == "ascending_neuron":
        return "ascending"
    if re.match(r"^(PAM|PPL|PPL1|PPL2|DAN)", t):
        return "DAN"
    if re.match(r"^MBON", t):
        return "MBON"
    if re.match(r"^(Kenyon|KC)", t) or ("kenyon" in t.lower()):
        return "KC"
    if re.match(r"^(aSP|aIP|pC1|vMS|SP-|SIFa|IPN)", t):
        return "adult-specific"
    if re.match(r"^vpo", t) or "neck" in t.lower():
        return "neck-motor"
    if sc == "vnc_motor" or sc == "cb_motor":
        return "motor"
    if sc == "vnc_sensory" or sc == "cb_sensory" or sc == "ol_sensory":
        return "sensory"
    if sc == "visual_projection":
        return "optic-other"
    if sc == "visual_centrifugal":
        return "optic-other"
    if sc.startswith("ol_"):
        return "optic-other"
    if sc.startswith("cb_"):
        return "central-other"
    if sc.startswith("vnc_"):
        return "vnc-other"
    return "other"


def nt_sign_fallback(t: str) -> int:
    # type-rule: C0/C1/C2... glutamatergic/GABAergic tabs in FlyWire-style naming
    if re.match(r"^C[0-9]", t):
        return -1
    return +1


def dir_of(t: str) -> int:
    m = re.match(r"^T[45]([abcd])", t)
    if not m:
        return -1
    return {"a": 0, "b": 1, "c": 2, "d": 3}[m.group(1)]


def sval(x) -> str:
    return x if isinstance(x, str) else ""


def write_str32(buf, s: str):
    b = s.encode("utf-8")
    buf += struct.pack("<I", len(b))
    buf += b
    buf += b"\x00" * ((-len(b)) % 4)  # keep u32/i16/f32 views aligned


def pad4(buf, nbytes: int):
    buf += b"\x00" * ((-nbytes) % 4)


def main():
    t0 = time.time()

    print("Loading annotations...")
    ann = feather.read_table(DATA / "malecns-annotations.feather").to_pandas()
    df = ann[ann["status"] == "Traced"].copy()
    idc = "bodyId"
    df = df.sort_values(idc).reset_index(drop=True)
    n = len(df)
    ids = df[idc].astype("int64").to_numpy()
    print(f"  traced neurons: {n:,} ({time.time() - t0:.0f}s)")

    ty = df["type"].fillna("").astype(str).to_numpy()
    sc = df["superclass"].fillna("").astype(str).to_numpy()
    remap = {int(b): i for i, b in enumerate(ids.tolist())}

    groups = [group_of(t, s) for t, s in zip(ty, sc)]
    group_names = sorted(set(groups))
    gidx = {g: i for i, g in enumerate(group_names)}
    grp = np.array([gidx[g] for g in groups], dtype=np.int64)

    # real NT signs
    nt_lookup = {}
    nt_file = DATA / "neuron-nt.json"
    if nt_file.exists():
        nt_lookup = json.loads(nt_file.read_text())
        print(f"  real NT profiles: {len(nt_lookup):,}")
    nt_out = np.zeros(n, dtype=np.int8)
    nt_conf = np.zeros(n, dtype=np.float32)
    fallbacks = 0
    for i, (nid, t) in enumerate(zip(ids.tolist(), ty)):
        prof = nt_lookup.get(str(nid))
        if prof:
            ach, gaba, glu = prof[0], prof[1], prof[2]
            if gaba >= 0.5:
                nt_out[i] = -1
            elif ach >= 0.5 or glu >= 0.5:
                nt_out[i] = 1
            else:
                nt_out[i] = nt_sign_fallback(t)
                fallbacks += 1
            nt_conf[i] = round(max(prof), 3)
        else:
            nt_out[i] = nt_sign_fallback(t)
            fallbacks += 1
    print(f"  NT fallbacks: {fallbacks:,}")

    hex1 = df["assignedOlHex1"].to_numpy()
    hex2 = df["assignedOlHex2"].to_numpy()
    side = np.array(
        [1 if sval(x)[:1] == "R" else 0 for x in df["somaSide"].fillna("")],
        dtype=np.uint8,
    )
    dir_out = np.array([dir_of(t) for t in ty], dtype=np.int8)

    print("Loading connectome (~1.1 GB)...")
    e = feather.read_table(
        DATA / "malecns-connectome.feather", columns=["body_pre", "body_post", "weight"]
    ).to_pandas()
    pre = e["body_pre"].astype("int64").to_numpy()
    post = e["body_post"].astype("int64").to_numpy()
    w = e["weight"].fillna(0).astype("int64").to_numpy()
    del e
    m = (w >= MIN_EDGE_W) & np.isin(pre, ids) & np.isin(post, ids)
    pre_m = np.array([remap[b] for b in pre[m].tolist()], dtype=np.int64)
    post_m = np.array([remap[b] for b in post[m].tolist()], dtype=np.int64)
    w_m = w[m].astype(np.float32)

    # T4/T5 columnar retinotopy: real T4/T5 dendrites are confined to a
    # single medulla column, so each inherits the hex of its strongest
    # hexed upstream input (2-hop BFS from lamina through Tm/TmY etc.),
    # letting their a/b/c/d direction classes read that column's EMDs.
    hex1f = np.nan_to_num(hex1.astype(float), nan=-1.0)
    hex2f = np.nan_to_num(hex2.astype(float), nan=-1.0)
    t4_id, t5_id = gidx.get("T4", -1), gidx.get("T5", -1)
    is_t45 = (grp == t4_id) | (grp == t5_id)
    if is_t45.any():
        h1s = np.where(hex1f >= 0, hex1f, -1).astype(np.float32)
        h2s = np.where(hex1f >= 0, hex2f, -1).astype(np.float32)
        have = hex1f >= 0
        # 2 propagation hops: neuron takes the hex of its strongest hexed
        # presynaptic partner (weighted by synapse count)
        for _hop in range(2):
            need = is_t45 & ~have
            if not need.any():
                break
            sel = need[post_m] & have[pre_m]
            if not sel.any():
                break
            b_post, b_w = post_m[sel], w_m[sel]
            b_h1, b_h2 = h1s[pre_m[sel]], h2s[pre_m[sel]]
            o = np.lexsort((-b_w, b_post))
            b_post, b_h1, b_h2 = b_post[o], b_h1[o], b_h2[o]
            first = np.ones(b_post.size, dtype=bool)
            first[1:] = b_post[1:] != b_post[:-1]
            h1s[b_post[first]] = b_h1[first]
            h2s[b_post[first]] = b_h2[first]
            have[b_post[first]] = True
        hex1f = np.where(is_t45, h1s, hex1f)
        hex2f = np.where(is_t45, h2s, hex2f)
        got = int((is_t45 & have).sum())
        print(f"  T4/T5 hex inheritance: {got:,}/{int(is_t45.sum()):,} neurons")
    hex1, hex2 = hex1f, hex2f
    E = len(pre_m)
    print(
        f"  kept {E:,} traced->traced synapses (weight >= {MIN_EDGE_W}) ({time.time() - t0:.0f}s)"
    )

    # CSR order by source, per-target cap, plastic sub-circuit
    print("Building CSR + caps + plastic subset...")
    order = np.argsort(pre_m, kind="stable")
    pre_s, post_s, w_s = pre_m[order], post_m[order], w_m[order]
    indptr = np.searchsorted(pre_s, np.arange(n + 1))
    kept_mask = np.ones(E, dtype=bool)
    plastic_mask = np.zeros(E, dtype=bool)

    dn_group = gidx["descending"]
    grp_arr = np.array([gidx[g] for g in groups], dtype=np.uint8)
    is_dn = grp_arr == dn_group
    # cap per-target fan-in + choose plastic edges for DNs
    # vectorized per target over CSR of reverse edges
    r_order = np.argsort(post_s, kind="stable")
    r_pre, r_w = pre_s[r_order], w_s[r_order]
    r_ptr = np.searchsorted(post_s[r_order], np.arange(n + 1))
    for tgt in range(n):
        lo, hi = r_ptr[tgt], r_ptr[tgt + 1]
        k = hi - lo
        if k == 0:
            continue
        if k > TOP_INPUTS_CAP:
            seg = slice(lo, hi)
            w_seg = r_w[seg]
            thr = np.partition(w_seg, -TOP_INPUTS_CAP)[-TOP_INPUTS_CAP]
            kept_mask[r_order[lo:hi][w_seg < thr]] = False
        if is_dn[tgt]:
            # take strongest PLASTIC_K inputs by weight
            idxs = r_order[lo:hi]
            w_seg = r_w[lo:hi]
            if len(w_seg) > PLASTIC_K:
                top = np.argpartition(-w_seg, PLASTIC_K)[:PLASTIC_K]
                plastic_mask[idxs[top]] = True
            else:
                plastic_mask[idxs] = True
    # build CSR-order masks
    r_order_kept = r_order[kept_mask[r_order]]
    kept_sorted = np.sort(r_order_kept)
    kept_mask = np.zeros(E, dtype=bool)
    kept_mask[kept_sorted] = True
    del r_order, r_pre, r_w, r_ptr

    final_idx = np.nonzero(kept_mask)[0]
    Ef = len(final_idx)
    print(f"  final edges after caps: {Ef:,}")

    print("Writing binary...")
    plastic_idx = np.nonzero(plastic_mask[final_idx])[0].astype(np.uint32)
    # 12-byte magic keeps every subsequent field 4-byte aligned for JS views
    header = b"FLYBRAIN1\0\0\0" + struct.pack(
        "<IIIII", 0x4E455552, n, Ef, len(group_names), len(plastic_idx)
    )
    buf = bytearray(header)
    write_str32(buf, "|".join(group_names))
    buf += grp_arr.tobytes()
    pad4(buf, n)
    buf += nt_out.tobytes()
    pad4(buf, n)
    buf += dir_out.tobytes()
    pad4(buf, n)
    buf += side.tobytes()
    pad4(buf, n)
    h1f = np.nan_to_num(hex1.astype(float), nan=-1.0)
    h2f = np.nan_to_num(hex2.astype(float), nan=-1.0)
    buf += np.stack([h1f, h2f], axis=1).astype(np.int16).tobytes()
    buf += nt_conf.tobytes()
    buf += pre_s[final_idx].astype(np.uint32).tobytes()
    buf += post_s[final_idx].astype(np.uint32).tobytes()
    buf += w_s[final_idx].astype(np.float32).tobytes()
    buf += plastic_idx.tobytes()
    OUT_BIN.write_bytes(bytes(buf))

    meta = {
        "format": "FLYBRAIN1",
        "neurons": int(n),
        "edges": int(Ef),
        "plasticEdges": int(plastic_mask[final_idx].sum()),
        "groups": group_names,
        "groupOf": {g: int(sum(1 for x in groups if x == g)) for g in group_names},
        "minSynapseWeight": MIN_EDGE_W,
        "plasticK": PLASTIC_K,
        "dataset": "MaleCNS v1.0 - HHMI Janelia / Google Research, Cell 2026",
        "license": "CC-BY 4.0",
    }
    OUT_META.write_text(json.dumps(meta, indent=1))
    print(
        f"Wrote {OUT_BIN} ({OUT_BIN.stat().st_size / 1e6:.1f} MB) + meta ({time.time() - t0:.0f}s total)"
    )


if __name__ == "__main__":
    main()
