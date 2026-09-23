"""Hexapod leg kinematics + tripod gait generator (MuJoCo MJCF config/hexapod_world.xml).

The robot: six 3-DoF legs (coxa yaw / femur pitch / tibia pitch), insect style.
The bridge converts the fly brain's four actuator channels into body-frame
locomotion velocity (like the AirSim bridge's FPV sticks):

  throttle -> gait cycle rate (stop = feet planted, standing)
  surge    -> forward (+) / backward (-) stride
  lateral  -> sideways stride
  turn     -> rotation rate

Each leg is driven by closed-form IK to a foot target that orbits a rest
position in a stance ellipse (stance = push, swing = lift-and-replace) — the
standard hexapod walking-machine control approach, done in joint space at
1 kHz. The gait phase advances with throttle (power), and the stride geometry
comes from surge/lateral/turn — exactly how a real hexapod RC decoder works.

MJCF conventions used here (see config/hexapod_world.xml):
  legs L0/L1/L2 (+y, left) R0/R1/R2 (-y, right); az angles 55/85/125 deg.
  coxa axis +z (yaw, range +-1 rad), femur/tibia axis +y (pitch).
  keyframe "home": femur -0.6, tibia 1.9 -> all six feet exactly on the floor.
  tibia: larger angle = foot further down; femur: more negative = leg lifted.
"""
from __future__ import annotations

import math

# ---- leg geometry (mirrors config/hexapod_world.xml) ------------------------
L_COXA = 0.030
L_FEMUR = 0.055
L_TIBIA = 0.085

# per-leg mount azimuth (rad) and hip offsets in the MJCF
LEGS = ("L0", "R0", "L1", "R1", "L2", "R2")
AZ = {"L0": 0.9599, "R0": -0.9599, "L1": 1.4835, "R1": -1.4835, "L2": 2.1817, "R2": -2.1817}
HIP = {  # coxa joint origin in torso frame
    "L0": (0.05, 0.03, -0.005), "R0": (0.05, -0.03, -0.005),
    "L1": (0.0, 0.03, -0.005), "R1": (0.0, -0.03, -0.005),
    "L2": (-0.05, 0.03, -0.005), "R2": (-0.05, -0.03, -0.005),
}
# tripod groups: A = L0 R1 L2, B = R0 L1 R2 (alternating triangles)
GROUP_A = {"L0", "R1", "L2"}
GROUP_B = {"R0", "L1", "R2"}

HOME_FEMUR = -0.6
HOME_TIBIA = 1.9

# ---- gait parameters --------------------------------------------------------
SWING_H = 0.014          # swing foot lift (m)
STRIDE = 0.028           # max stride half-length along travel dir (m)
STEP_R = 0.055           # nominal foot radius from hip in the horizontal plane
STANCE_Z = 0.0           # stance feet stay on the floor (IK solves to touch)


def _fk(az: float, coxa: float, femur: float, tibia: float, hip=(0.0, 0.0, 0.0), out=None):
    """Leg FK: foot position in the TORSO frame (pass hip=HIP[leg]) or hip-relative
    (hip=(0,0,0)). Elevation convention: segment elevation = -femur (femur < 0
    lifts), second segment elevation = -femur-tibia (tibia > 0 pushes foot down)."""
    a = az + coxa
    ca, sa = math.cos(a), math.sin(a)
    reach = L_COXA + math.cos(-femur) * L_FEMUR + math.cos(-femur - tibia) * L_TIBIA
    z = hip[2] + math.sin(-femur) * L_FEMUR + math.sin(-femur - tibia) * L_TIBIA
    if out is None:
        out = [0.0, 0.0, 0.0]
    out[0] = hip[0] + ca * reach
    out[1] = hip[1] + sa * reach
    out[2] = z
    return out


def solve_ik(az: float, target: list[float], hip=(0.0, 0.0, -0.005),
             init=(0.0, HOME_FEMUR, HOME_TIBIA)) -> tuple[float, float, float]:
    """Closed-form 3-DoF IK: coxa yaw, femur pitch, tibia pitch.

    az: leg mount azimuth; target: foot position in the TORSO frame.
    Returns the branch that respects the joint limits (femur <= 0.4, i.e. the
    insect "knee above hip" posture of the home keyframe) — verified by the
    FK/IK round-trip test against the MuJoCo model.
    """
    dx, dy, dz = target[0] - hip[0], target[1] - hip[1], target[2] - hip[2]
    coxa = math.atan2(dy, dx) - az
    # normalize into the joint range
    while coxa > math.pi:
        coxa -= 2 * math.pi
    while coxa < -math.pi:
        coxa += 2 * math.pi
    coxa = max(-1.0, min(1.0, coxa))

    # planar 2-link problem in the leg's vertical plane (r from the coxa pivot)
    r = max(math.hypot(dx, dy) - L_COXA, 1e-4)
    z = dz
    D = (r * r + z * z - L_FEMUR * L_FEMUR - L_TIBIA * L_TIBIA) / (2.0 * L_FEMUR * L_TIBIA)
    D = max(-1.0, min(1.0, D))
    tibia = math.acos(D)                 # in [0, pi]: always within [-0.6, 2.6]

    # theta1 = femur elevation = -femur; theta2 = relative elevation = -tibia
    phi = math.atan2(z, r)               # aim at the target
    psi = math.atan2(L_TIBIA * math.sin(-tibia), L_FEMUR + L_TIBIA * math.cos(-tibia))
    femur = -(phi - psi)                 # knee-up branch: femur <= 0 at home

    return coxa, femur, tibia


def _clamp_joints(c: float, f: float, t: float) -> tuple[float, float, float]:
    return (
        max(-1.0, min(1.0, c)),
        max(-1.7, min(0.4, f)),
        max(-0.6, min(2.6, t)),
    )


class GaitEngine:
    """Tripod-gait joint target generator.

    velocity = (vx, vy, wz) in the body frame (m/s, m/s, rad/s).
    throttle = gait cycle rate (0..1). Foot targets orbit the rest pose on a
    stance ellipse; groups A/B alternate with a 50% duty cycle.
    """

    def __init__(self, model):
        self.model = model
        self.phase = 0.0
        # rest foot position (torso frame) per leg from the home keyframe
        self.rest: dict[str, list[float]] = {}
        self._fk = _fk
        for leg in LEGS:
            self.rest[leg] = _fk(AZ[leg], 0.0, HOME_FEMUR, HOME_TIBIA)

    # -- joint addressing ----------------------------------------------------
    def joint_adrs(self) -> dict[str, tuple[int, int, int]]:
        out = {}
        for leg in LEGS:
            q = [
                mujoco_joint_adr(self.model, f"{leg}-cxa"),
                mujoco_joint_adr(self.model, f"{leg}-fmr"),
                mujoco_joint_adr(self.model, f"{leg}-tbs"),
            ]
            out[leg] = (q[0], q[1], q[2])
        return out

    def actuator_adrs(self) -> dict[str, tuple[int, int, int]]:
        out = {}
        for leg in LEGS:
            out[leg] = (
                mujoco_act_adr(self.model, f"{leg}-cxa"),
                mujoco_act_adr(self.model, f"{leg}-fmr"),
                mujoco_act_adr(self.model, f"{leg}-tbs"),
            )
        return out

    # -- the gait --------------------------------------------------------------
    def targets(self, throttle: float, surge: float, lateral: float, turn: float,
                dt: float) -> dict[str, tuple[float, float, float]]:
        """Advance the gait phase and return per-leg joint targets (rad)."""
        rate = max(0.0, min(1.0, throttle)) * 2.2   # Hz, fly-scale stride clock
        self.phase = (self.phase + rate * dt) % 1.0

        turning = abs(turn) > 0.02
        moving = rate > 1e-3 and (abs(surge) > 0.02 or abs(lateral) > 0.02 or turning)
        if not moving:
            # standing: plant all feet exactly at the rest pose
            return {leg: _clamp_joints(*solve_ik(AZ[leg], self.rest[leg], HIP[leg]))
                    for leg in LEGS}

        # body-frame travel direction for this stride
        heading = math.atan2(lateral, surge) if (abs(surge) + abs(lateral)) > 0.02 else 0.0
        stride_scale = math.hypot(surge, lateral) if not turning else 0.35

        targets = {}
        for leg in LEGS:
            group = 0.0 if leg in GROUP_A else 0.5
            ph = (self.phase + group) % 1.0
            in_swing = ph < 0.5
            s = ph * 2.0 if in_swing else (ph - 0.5) * 2.0   # 0..1 within half-cycle

            if in_swing:
                # lift and reach forward along the travel direction
                u = s * 2.0 - 1.0                    # -1..1 through the swing
                along = -math.cos(u * math.pi)       # smooth fwd excursion
                lift = math.sin(s * math.pi) * SWING_H
                # swing legs also rotate the body: push their ground point back
                if turning:
                    along -= 0.5 * turn * math.copysign(1.0, _side(leg))
                tx = self.rest[leg][0] + math.cos(heading) * along * STRIDE * stride_scale
                ty = self.rest[leg][1] + math.sin(heading) * along * STRIDE * stride_scale
                tz = self.rest[leg][2] + lift
            else:
                # stance: foot pinned, body slides over it (relative motion)
                u = s * 2.0 - 1.0
                along = u                            # -1 -> +1 under the body
                tx = self.rest[leg][0] - math.cos(heading) * along * STRIDE * stride_scale
                ty = self.rest[leg][1] - math.sin(heading) * along * STRIDE * stride_scale
                tz = self.rest[leg][2] - 0.002       # slight press for traction
                if turning:
                    # stance feet sweep opposite the turn -> yaw torque
                    tx -= turn * _tangent(leg)[0] * 0.5 * STRIDE * 2.0
                    ty -= turn * _tangent(leg)[1] * 0.5 * STRIDE * 2.0

            sol = solve_ik(AZ[leg], [tx, ty, tz], HIP[leg],
                           init=(0.0, HOME_FEMUR, HOME_TIBIA))
            targets[leg] = _clamp_joints(*sol)
        return targets


def _side(leg: str) -> float:
    return 1.0 if leg.startswith("L") else -1.0


def _tangent(leg: str) -> tuple[float, float]:
    """Unit vector perpendicular to the hip->foot ray (yaw-torque direction)."""
    az = AZ[leg]
    return (-math.sin(az), math.cos(az))


# -- tiny model helpers (kept dependency-light for tests) ---------------------
def mujoco_joint_adr(model, name: str) -> int:
    import mujoco
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_qposadr[j]


def mujoco_act_adr(model, name: str) -> int:
    import mujoco
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return model.actuator_ctrladr[a]
