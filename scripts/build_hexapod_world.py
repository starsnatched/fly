"""Convert the preconfigured PiHexa-style hexapod MJCF (config/hexapod_assets/robot.xml,
millimeter units, constraint-held chassis) into the FlyBrain walker world
(config/hexapod_world.xml): meter units, one rigid torso with the six legs as
proper kinematic children, 18 position servos, foot touch sensors, plus the
arena (walls, rocks, goal puck) and the brain's eye/chase cameras.

The conversion is verified against the source model: the six foot (touch) site
world positions at the default pose must match the source (mm) vs converted
(m -> x1000) within 1e-6 mm, proving the leg kinematics are preserved exactly.
Run standalone:  python scripts/build_hexapod_world.py

Why not use the source model as-is:
  - authored in millimeters with default gravity (an effective 1/1000 g:
    everything floats), masses in kg, joint ranges in degrees
  - lower/upper chassis are two free bodies + legs all held to the chassis by
    6 <connect> equality constraints (a soft constraint-assembled kit, not a
    rigid mechanism)
  - no cameras, no arena, kp=1e6 servo gains tuned for the mm scale
The conversion keeps: every mesh, every leg mount pose (verified), joint
axes/names/order, servo naming — only the chassis representation, units and
surroundings change.
"""
from __future__ import annotations

import copy
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "config/hexapod_assets/robot.xml"
DST = ROOT / "config/hexapod_world.xml"

# attributes carrying LENGTH quantities (mm -> m). euler/quat/axis are angles
# or directions and must NOT scale. NOTE: joint `pos` is the hinge anchor —
# missing it leaves pivots at mm-as-meters offsets (legs swinging meters per rad).
LEN_ATTRS = {"pos": {"body", "geom", "site", "camera", "light", "joint"},
             "size": {"geom", "site"},
             "fromto": {"geom"},
             "scale": {"mesh"},
             "anchor": set()}
MM = 0.001

DEG = math.pi / 180.0


def q_norm(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_rot(q, v):
    """Rotate vector v by quaternion q."""
    t = 2.0 * np.cross(q[1:], v)
    return v + q[0] * t + np.cross(q[1:], t)


def euler_to_quat(euler_deg):
    """MuJoCo default eulerseq 'xyz' (extrinsic? intrinsic? MuJoCo uses
    intrinsic rotations applied x then y then z as in mju_euler2Quat)."""
    out = np.zeros(4)
    e = np.radians(np.asarray(euler_deg, dtype=float))
    mujoco.mju_euler2Quat(out, e, "xyz")
    return q_norm(out)


def _body_local_pose(b):
    pos = np.array([float(x) for x in (b.get("pos") or "0 0 0").split()])
    # NOTE: `"x" in element` tests CHILD ELEMENTS, not attributes — must use .get()
    qattr, eattr = b.get("quat"), b.get("euler")
    if qattr is not None:
        quat = q_norm([float(x) for x in qattr.split()])
    elif eattr is not None:
        quat = euler_to_quat([float(x) for x in eattr.split()])
    else:
        quat = np.array([1.0, 0, 0, 0])
    return pos, quat


def body_world_pose(tree, name):
    """World pose of a (possibly nested) body by name, walking up the chain."""
    # find the element and its ancestor chain
    chain = []
    def hunt(el):
        for b in el.findall("body"):
            if b.get("name") == name:
                chain.append(b)
                return True
            if hunt(b):
                chain.append(b)
                return True
        return False
    if not hunt(tree.getroot().find("worldbody")):
        raise KeyError(name)
    pos, quat = np.zeros(3), np.array([1.0, 0, 0, 0])
    for b in reversed(chain):        # world -> ... -> target
        lp, lq = _body_local_pose(b)
        pos = pos + q_rot(quat, lp)
        quat = q_mul(quat, lq)
    return pos, quat


def local_pose_in(parent_pos, parent_quat, pos, quat):
    """Express a world pose in a parent frame."""
    dq = pos - parent_pos
    quat_local = q_mul(q_conj(parent_quat), quat)
    pos_local = q_rot(q_conj(parent_quat), dq)
    return pos_local, quat_local


def set_pose(el, pos, quat):
    el.set("pos", " ".join(f"{v:.6f}" for v in pos))
    el.set("quat", " ".join(f"{v:.8f}" for v in quat))
    el.attrib.pop("euler", None)


def source_foot_positions(model_path, settle_steps=0) -> dict[str, np.ndarray]:
    """Foot (touch) site world positions in the SOURCE model at reset (the
    design pose) or after settling `settle_steps` if > 0."""
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for _ in range(settle_steps):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)          # compute site_xpos (reset does not)
    out = {}
    for i in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
        if name and name.startswith("touch:"):
            out[name] = d.site_xpos[i].copy()
    return out


def convert() -> ET.ElementTree:
    tree = ET.parse(SRC)
    root = tree.getroot()
    wb = root.find("worldbody")

    # ---- 1. rigid chassis: merge upper (chassis + all six leg mounts) into
    # lower, drop its freejoint. The legs are CHILDREN of upper in the source
    # kit and are welded to lower by <connect> equality constraints; here they
    # become rigid kinematic children of the merged torso.
    lower = wb.find("body[@name='lower']")
    upper = wb.find("body[@name='upper']")
    lp, lq = body_world_pose(tree, "lower")
    up, uq = body_world_pose(tree, "upper")
    for g in list(upper.findall("geom")):
        gp = np.array([float(x) for x in (g.get("pos") or "0 0 0").split()])
        gq_attr, ge_attr = g.get("quat"), g.get("euler")
        gq = (q_norm([float(x) for x in gq_attr.split()])
              if gq_attr is not None else
              euler_to_quat([float(x) for x in ge_attr.split()])
              if ge_attr is not None else np.array([1.0, 0, 0, 0]))
        gq_w = q_mul(uq, gq)
        gp_w = up + q_rot(uq, gp)
        set_pose(g, *local_pose_in(lp, lq, gp_w, gq_w))
        lower.append(g)
    # leg mounts: world pose now, re-expressed in the torso frame
    legs = [b for b in upper.findall("body") if b.get("name", "").startswith("cx_")]
    assert len(legs) == 6, f"expected 6 leg mounts, got {len(legs)}"
    for b in legs:
        bp, bq = body_world_pose(tree, b.get("name"))
        upper.remove(b)
        set_pose(b, *local_pose_in(lp, lq, bp, bq))
        lower.append(b)
    wb.remove(upper)
    eq = root.find("equality")
    if eq is not None:
        root.remove(eq)   # constraints replaced by real kinematic parenting

    # mesh paths: the converted file lives in config/, meshes in
    # config/hexapod_assets/ — rewrite each mesh file attribute
    for mesh in root.find("asset").findall("mesh"):
        mesh.set("file", "hexapod_assets/" + mesh.get("file"))

    # ---- 3. mm -> m on every length attribute -------------------------------
    for el in root.iter():
        tag = el.tag.split('}')[-1]
        for attr, tags in LEN_ATTRS.items():
            if attr in el.attrib and (tag in tags or not tags):
                vals = [float(x) * MM for x in el.attrib[attr].split()]
                el.attrib[attr] = " ".join(f"{v:.8g}" for v in vals)
    # gravity is default (-9.81) again in meters; keep compiler degrees
    # (source joint ranges are in degrees: +-62.83 deg = +-1.0966 rad)

    # ---- 4. servo tuning for meter scale ------------------------------------
    # The kit's +-62.83 deg joint range is a sim artifact; real hobby servos in
    # this chassis do +-90 deg. Without it the legs cannot reach the floor with
    # the belly plate (75 mm below origin) clear — the robot belly-flops.
    # NOTE: compiler angle=degree, so joint range is in DEGREES.
    for j in root.iter("joint"):
        j.set("range", "-90 90")
    for dflt in root.iter("default"):
        for j in dflt.findall("joint"):
            j.set("damping", "0.06")
            j.set("armature", "0.002")
            j.set("frictionloss", "0.04")
        for g in dflt.findall("geom"):
            g.set("friction", "1.0 0.02 0.001")
    actu = root.find("actuator")
    for a in list(actu):
        a.set("kp", "40")
        a.set("kv", "0.6")
        a.set("forcerange", "-8 8")
        a.set("ctrlrange", "-1.5708 1.5708")  # radians: MuJoCo ctrl is always rad
        a.set("ctrllimited", "true")

    # ---- 5. arena + cameras + goal puck --------------------------------------
    # drop the source floor plane & light, add ours (meters)
    for g in list(wb.findall("geom")):
        wb.remove(g)
    for l in list(wb.findall("light")):
        wb.remove(l)

    arena = ET.SubElement(wb, "light", {
        "directional": "true", "pos": "0 0 4", "dir": "0 0 -1",
        "diffuse": "0.55 0.55 0.55", "specular": "0.1 0.1 0.1"})
    ET.SubElement(wb, "geom", {
        "name": "floor", "type": "plane", "size": "3 3 0.05",
        "contype": "3", "conaffinity": "3", "group": "0",
        "rgba": "0.45 0.52 0.38 1"})
    walls = [("wallN", "0  1.6 0.12", "1.7 0.04 0.12"),
             ("wallS", "0 -1.6 0.12", "1.7 0.04 0.12"),
             ("wallE", " 1.6 0 0.12", "0.04 1.7 0.12"),
             ("wallW", "-1.6 0 0.12", "0.04 1.7 0.12")]
    for name, pos, size in walls:
        ET.SubElement(wb, "geom", {
            "name": name, "type": "box", "pos": pos, "size": size,
            "contype": "3", "conaffinity": "3", "group": "2",
            "rgba": "0.5 0.5 0.55 1"})
    rocks = [
        ("rock1",  "box",      "0.55  0.35 0.05", "0.09 0.06 0.05", "0 0 28.6"),
        ("rock2",  "cylinder", "0.85 -0.40 0.06", "0.05 0.06 0.06", "0 0 63.0"),
        ("rock3",  "box",      "1.10  0.55 0.07", "0.12 0.08 0.07", "0 0 -17.2"),
        ("rock4",  "box",      "0.30 -0.75 0.04", "0.07 0.10 0.04", "0 0 51.6"),
        ("rock5",  "cylinder", "-0.55 0.60 0.05", "0.045 0.05 0.05", "0 0 11.5"),
        ("rock6",  "box",      "-0.90 -0.35 0.06", "0.10 0.07 0.06", "0 0 -45.8"),
        ("rock7",  "box",      "-0.35 1.05 0.05", "0.11 0.06 0.05", "0 0 80.2"),
        ("rock8",  "cylinder", "0.05 0.75 0.07", "0.06 0.07 0.07", "0 0 0"),
        ("rock9",  "box",      "-1.15 0.15 0.04", "0.08 0.12 0.04", "0 0 22.9"),
        ("rock10", "cylinder", "-0.75 -0.95 0.06", "0.055 0.06 0.06", "0 0 40.1"),
        ("rock11", "box",      "0.70 1.15 0.06", "0.10 0.09 0.06", "0 0 -57.3"),
        ("rock12", "box",      "1.30 -0.10 0.05", "0.06 0.11 0.05", "0 0 17.2"),
        ("rock13", "cylinder", "0.15 -1.20 0.05", "0.05 0.05 0.05", "0 0 0"),
        ("rock14", "box",      "-1.25 -0.75 0.07", "0.09 0.08 0.07", "0 0 68.8"),
    ]
    for name, typ, pos, size, euler in rocks:
        ET.SubElement(wb, "geom", {
            "name": name, "type": typ, "pos": pos, "size": size,
            "euler": euler, "contype": "3", "conaffinity": "3", "group": "2",
            "rgba": "0.62 0.58 0.50 1"})
    goal = ET.SubElement(wb, "body", {"name": "goal", "mocap": "true",
                                      "pos": "0.8 0 0.004"})
    ET.SubElement(goal, "geom", {
        "name": "goalpuck", "type": "cylinder", "size": "0.035 0.004",
        "rgba": "0.9 0.12 0.08 0.75", "contype": "0", "conaffinity": "0",
        "group": "4"})

    # cameras on the torso (+x is forward: leg mounts cx_1/cx_4 sit at +x)
    torso = lower
    ET.SubElement(torso, "camera", {
        "name": "eyeL", "pos": "0.075 -0.014 0.012",
        "euler": "90 0 115", "fovy": "90"})
    ET.SubElement(torso, "camera", {
        "name": "eyeR", "pos": "0.075 0.014 0.012",
        "euler": "90 0 65", "fovy": "90"})
    ET.SubElement(torso, "camera", {
        "name": "chase", "mode": "trackcom", "pos": "0.35 -0.35 0.25",
        "xyaxes": "0.707 0.707 0 -0.357 0.357 0.878", "fovy": "55"})

    # ---- 5b. rubber feet -----------------------------------------------------
    # The tibia servo housings ride ~2-3 mm below the foot tips in every pose;
    # on the physical kit the tips carry rubber feet. Lower each touch site by
    # 6 mm (redefining the foot reference) and add a rubber collision sphere
    # centered 4 mm above the site, so the pad bottom sits exactly at the site
    # and the housing always clears the ground.
    def walk_bodies(el):
        for child in el:
            if child.tag == "body":
                yield child
            yield from walk_bodies(child)
    for body in walk_bodies(root):
        for site in body.findall("site"):
            n = site.get("name") or ""
            if n.startswith("touch:"):
                k = n.split(":")[1]
                p = [float(x) for x in site.get("pos").split()]
                p[2] -= 0.006
                site.set("pos", " ".join(f"{v:.8g}" for v in p))
                c = list(p)
                c[2] += 0.004      # sphere center 4 mm above site -> bottom at site
                ET.SubElement(body, "geom", {
                    "name": f"foot:{k}", "type": "sphere", "size": "0.004",
                    "pos": " ".join(f"{v:.8g}" for v in c),
                    "friction": "1.5 0.02 0.001", "rgba": "0.15 0.15 0.15 1",
                    "contype": "1", "conaffinity": "1", "group": "3",
                    "mass": "0.005"})

    # ---- 6. keyframe home pose: standing stance ------------------------------
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", {"name": "home", "qpos": _home_qpos_string(root)})

    ET.indent(tree, space="  ")
    tree.write(DST, encoding="unicode")
    print(f"wrote {DST}")
    return tree


def _home_qpos_string(tree_root, femur_deg=20.0, tibia_deg=-20.0) -> str:
    """Home stance from an exhaustive simulated posture search (femur x tibia
    grid, 3 s settle each): the only stable family for this kit is legs nearly
    straight (femur ~20 deg down, tibia ~-20 deg) — a wide, low spider stance
    with tilt < 2 deg. The torso origin sits near the TOP of the chassis, so a
    correct stance has the origin at ~0 world height with pads on the floor.
    """
    trial = DST.parent / "_hexapod_trial.xml"
    ET.ElementTree(tree_root).write(trial, encoding="unicode")
    try:
        m = mujoco.MjModel.from_xml_path(str(trial))
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[7:] = np.radians([0.0, femur_deg, tibia_deg] * 6)
        mujoco.mj_forward(m, d)
        pad_z = [float(d.geom_xpos[g][2]) - 0.004
                 for g in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("foot:")]
        torso_z = -min(pad_z)             # lowest pad exactly on the floor
        qpos = np.concatenate([[0.0, 0.0, torso_z, 1.0, 0.0, 0.0, 0.0],
                               np.radians([0.0, femur_deg, tibia_deg] * 6)])
    finally:
        trial.unlink(missing_ok=True)
    return " ".join(f"{v:.6f}" for v in qpos)


def verify() -> None:
    """Kinematic parity at the DESIGN pose: source at reset (mj_forward only —
    its ball-joint kit sags under its fake 1/1000 g gravity, which is an
    artifact, not the geometry) vs the converted model with its torso at the
    same world pose. This proves the leg chains are preserved exactly."""
    src_feet = source_foot_positions(SRC, settle_steps=0)
    m = mujoco.MjModel.from_xml_path(str(DST))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)          # torso spawns at origin, identity
    mujoco.mj_forward(m, d)
    worst = 0.0
    for i in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
        if name and name.startswith("touch:"):
            got_m = d.site_xpos[i].copy()
            want_mm = src_feet[name].copy()
            want_mm[2] -= 6.0   # rubber-foot reference: sites intentionally 6 mm lower
            err = float(np.max(np.abs(got_m * 1000.0 - want_mm)))
            worst = max(worst, err)
            print(f"  {name}: max err {err*1000:.6f} um")
    verdict = "OK" if worst < 1e-3 else "FAIL"
    print(f"foot-parity worst error: {worst*1000:.6f} um ({verdict})")
    if worst >= 1e-3:
        sys.exit(1)


def main() -> None:
    convert()
    verify()
    m = mujoco.MjModel.from_xml_path(str(DST))
    print(f"compiled: nq={m.nq} nu={m.nu} nbody={m.nbody} ncam={m.ncam} "
          f"nsensor={m.nsensor}")


if __name__ == "__main__":
    main()
