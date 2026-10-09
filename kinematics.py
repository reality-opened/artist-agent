"""SO-101 URDF forward kinematics and damped-least-squares IK in LeRobot degree space.

Ported from reality-opened/artist (kinematics.py and teleop.py solve_ik; authors galois777,
David Zhang, Hillysilly). The numerics are unchanged so solutions match the artist teleop
daemon; only the interface moved from raw ticks to LeRobot MotorNormMode.DEGREES values,
whose zero (the calibration range midpoint) is also the URDF zero.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np
from lerobot.motors import MotorCalibration

URDF = Path(__file__).parent / "urdf" / "so101_new_calib.urdf"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TICKS_PER_TURN = 4095  # LeRobot's degree conversion divides by STS3215 resolution - 1


def _floats(text: str) -> np.ndarray:
    return np.array(text.split(), dtype=float)


def rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    x, y, z = axis / np.linalg.norm(axis)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


class Kinematics:
    def __init__(self, urdf: Path = URDF):
        root = ET.parse(urdf).getroot()
        self.joints: dict[str, dict] = {}
        self.by_child: dict[str, dict] = {}
        for j in root.findall("joint"):
            origin = j.find("origin")
            roll, pitch, yaw = _floats(origin.get("rpy", "0 0 0"))
            t = np.eye(4)
            t[:3, :3] = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
            t[:3, 3] = _floats(origin.get("xyz", "0 0 0"))
            axis, limit = j.find("axis"), j.find("limit")
            record = dict(
                name=j.get("name"),
                kind=j.get("type"),
                origin=t,
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                axis=_floats(axis.get("xyz")) if axis is not None else None,
                limits=(float(limit.get("lower")), float(limit.get("upper"))) if limit is not None else None,
            )
            self.joints[record["name"]] = record
            self.by_child[record["child"]] = record
        self.inertia = {  # link -> (mass kg, centre of mass in the link frame, m)
            link.get("name"): (float(i.find("mass").get("value")), _floats(i.find("origin").get("xyz")))
            for link in root.findall("link") if (i := link.find("inertial")) is not None}

    def _joints_above(self, link: str) -> set[str]:
        names = set()
        while link in self.by_child:
            names.add(self.by_child[link]["name"])
            link = self.by_child[link]["parent"]
        return names

    def fk(self, radians, link: str = "gripper_frame_link") -> np.ndarray:
        """Link pose in the base frame (m); joints outside JOINTS (the jaw) are taken at zero."""
        q = dict(zip(JOINTS, radians))
        chain = []
        while link in self.by_child:
            joint = self.by_child[link]
            chain.append(joint)
            link = joint["parent"]
        result = np.eye(4)
        for joint in reversed(chain):
            result = result @ joint["origin"]
            if joint["kind"] != "fixed":
                r = np.eye(4)
                r[:3, :3] = rotation(joint["axis"], q.get(joint["name"], 0.0))
                result = result @ r
        return result

    def gravity_torque(self, radians, joint: str, point_masses=()) -> float:
        """Static gravity torque (N.m) about `joint`'s axis from every link beyond it plus `point_masses` carried
        beyond it ((kg, base-frame position in m) pairs), base z up. Positive turns the joint toward larger angles."""
        T = self.fk(radians, link=self.joints[joint]["child"])
        pivot, axis = T[:3, 3], T[:3, :3] @ self.joints[joint]["axis"]
        weight = np.array([0.0, 0.0, -9.81])
        tau = 0.0
        for link, (mass, com) in self.inertia.items():
            if joint in self._joints_above(link):
                L = self.fk(radians, link=link)
                tau += np.cross(L[:3, :3] @ com + L[:3, 3] - pivot, mass * weight) @ axis
        for mass, at in point_masses:
            tau += np.cross(np.asarray(at, float) - pivot, mass * weight) @ axis
        return float(tau)


class IKSolution(NamedTuple):
    joints_deg: np.ndarray  # the five arm joints, LeRobot degrees, JOINTS order
    xyz: list[float]  # model gripper_frame position (m, URDF base frame)
    jaw_axis: list[float]  # gripper_frame z axis in the base frame; [0, 0, -1] = jaws down


def joint_bounds_deg(
    kin: Kinematics, calibration: dict[str, MotorCalibration], margin_ticks: int = 10, offsets_deg=None,
    urdf_exempt: tuple[str, ...] = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Reading-space bounds (LeRobot degrees): URDF limits, which hold for model angle = reading + offset, intersected
    with the calibrated tick range shrunk by a margin (teleop used 8 + 2). urdf_exempt: (joint, "lo"|"hi") style
    names "joint:lo" whose URDF side is replaced by a measured mechanical stop elsewhere."""
    offsets = np.zeros(len(JOINTS)) if offsets_deg is None else np.asarray(offsets_deg, float)
    lo, hi = [], []
    for name, off in zip(JOINTS, offsets):
        cal = calibration[name]
        mid = (cal.range_min + cal.range_max) / 2
        urdf_lo, urdf_hi = np.degrees(kin.joints[name]["limits"]) - off
        if f"{name}:lo" in urdf_exempt:
            urdf_lo = -np.inf
        if f"{name}:hi" in urdf_exempt:
            urdf_hi = np.inf
        lo.append(max(urdf_lo, (cal.range_min + margin_ticks - mid) * 360 / TICKS_PER_TURN))
        hi.append(min(urdf_hi, (cal.range_max - margin_ticks - mid) * 360 / TICKS_PER_TURN))
    return np.array(lo), np.array(hi)


def _dls(kin: Kinematics, q, target, down: bool, lo, hi, w: float):
    for _ in range(200):
        T = kin.fk(q)
        err = [target - T[:3, 3]]
        if down:
            err.append(w * (np.array([0, 0, -1.0]) - T[:3, 2]))
        err = np.concatenate(err)
        if np.linalg.norm(err) < 1e-4:
            break
        J = []
        for i in range(5):
            dq = np.zeros(5)
            dq[i] = 1e-5
            Tp, Tm = kin.fk(q + dq), kin.fk(q - dq)
            col = [(Tp[:3, 3] - Tm[:3, 3]) / 2e-5]
            if down:
                col.append(w * (Tp[:3, 2] - Tm[:3, 2]) / 2e-5)
            J.append(np.concatenate(col))
        J = np.array(J).T
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(err)), err), lo, hi)
    T = kin.fk(q)
    return q, np.linalg.norm(target - T[:3, 3])


def solve_ik(kin: Kinematics, start_deg, xyz, down: bool, lo_deg, hi_deg, multi_seed: bool = True) -> IKSolution:
    """Damped least squares on position (+ jaws pointing down), 28 seeds; the solution nearest start wins.

    multi_seed=False keeps only the start as a seed: for short, warm-started path segments, it is far
    faster and cannot jump to another IK branch. Raises ValueError when no seed lands within 3 mm, or
    when the result leaves the bounds (possible only if the start itself is out of bounds and on target).
    """
    q0 = np.radians(np.asarray(start_deg, dtype=float))
    lo, hi = np.radians(lo_deg), np.radians(hi_deg)
    target = np.asarray(xyz, dtype=float)
    seeds = [q0]
    if multi_seed:
        seeds += [
            np.clip(np.array([q0[0], a, b, c, q0[4]]), lo, hi)
            for a in (-0.8, 0, 0.8)
            for b in (-0.8, 0, 0.8)
            for c in (0, 0.8, 1.4)
        ]
    # Prefer jaws straight down; relax toward position-only when out of reach.
    for w in ((0.3, 0.1, 0.03) if down else (0,)):
        results = [_dls(kin, s.copy(), target, down, lo, hi, w) for s in seeds]
        good = [(np.abs(q - q0).sum(), q) for q, cost in results if cost < 0.003]
        if good:
            break
    if not good:
        q = min(results, key=lambda r: r[1])[0]
        T = kin.fk(q)
        raise ValueError(f"IK unreachable: best {T[:3, 3].round(3).tolist()} axis {T[:3, 2].round(2).tolist()}")
    q = min(good, key=lambda g: g[0])[1]
    for name, angle, low, high in zip(JOINTS, q, lo, hi):
        if not low <= angle <= high:
            raise ValueError(
                f"IK {name} {np.degrees(angle):.2f} deg outside [{np.degrees(low):.2f}, {np.degrees(high):.2f}]"
            )
    T = kin.fk(q)
    return IKSolution(np.degrees(q), T[:3, 3].tolist(), T[:3, 2].tolist())
