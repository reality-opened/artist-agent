"""Offline stand-ins for the hardware: a simulated Feetech bus and a ray-cast RGB-D bench scene.

Base frame = robot base, millimetres, z up, table top at z = 0. Camera frame = OpenCV (x right, y down,
z forward). The algorithms under test never see cam_from_base; tests use it only as ground truth.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from lerobot.motors import MotorCalibration
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

if TYPE_CHECKING:
    from calibration import RigCalibration


# A logged teleop pose (artist status log, 2026-10-02): gripper ~150 mm above the bench in front of the base.
SIM_START_TICKS = {"shoulder_pan": 1945, "shoulder_lift": 2095, "elbow_flex": 2048, "wrist_flex": 3017,
                   "wrist_roll": 3302, "gripper": 870}
# The real arm's rest pose; with the field joint offsets (calibration.FIELD_OFFSETS_DEG) a simulated world that
# matches the hardware (the zero-offset world above has the arm stretched out, 0.67 N.m on the shoulder at rest).
FIELD_REST_TICKS = {"shoulder_pan": 1959, "shoulder_lift": 2052, "elbow_flex": 1981, "wrist_flex": 3447,
                    "wrist_roll": 3900, "gripper": 761}

_READABLE = ("Present_Position", "Goal_Position", "Torque_Enable", "Present_Temperature", "Present_Load",
             "Present_Voltage", "Status", "Torque_Limit", "Goal_Velocity", "Acceleration", "Lock")


class SimBus:
    """Stand-in for lerobot's FeetechMotorsBus (only the calls arm.Arm makes), with STS3215-like behaviour.

    A powered joint starts moving once its goal is more than `breakaway` ticks away and stops `hold` ticks
    short of it (stiction, as on the real arm), travelling at most `speed` ticks/s or Goal_Velocity.
    `sag[m]` is gravity: the joint settles that many ticks below its goal (P-only servo under load).
    Present_Load is the drive duty: it grows with the position error and saturates at Torque_Limit. Tests
    can block a joint inside a tick window (a hand in the way), make the gripper stall on an object,
    inject temperatures and bus failures, and record positions in `trace`; every transaction costs `latency_s`.
    Raw values only.
    """

    def __init__(self, calibration: dict[str, MotorCalibration], start_ticks=None, *, breakaway=20, hold=8,
                 speed=1500.0, latency_s=0.002, duty_per_tick=6.0):
        start = start_ticks or SIM_START_TICKS
        self.calibration = calibration
        self.eeprom = dict(calibration)
        self.pos = {m: float(start[m]) for m in calibration}
        self.reg = {name: {m: 0 for m in calibration} for name in _READABLE}
        self.reg["Goal_Position"] = {m: int(start[m]) for m in calibration}
        self.reg["Torque_Limit"] = {m: 1000 for m in calibration}
        self.reg["Present_Temperature"] = {m: 30 for m in calibration}
        self.reg["Present_Voltage"] = {m: 53 for m in calibration}
        self.blocks: dict[str, tuple[float, float]] = {}
        self.sag = {m: 0.0 for m in calibration}
        self.stiction: dict[str, tuple[float, float]] = {}  # motor -> (breakaway, hold): stronger friction than default
        self.trace: dict[str, list[float]] = {}  # motor -> positions after every transaction, for the motors listed
        self.grip_stall: float | None = None
        self.fail_reads = 0
        self.fail_after = 0  # successful sync_reads still allowed before fail_reads kick in
        self.writes: list[tuple[str, str, int]] = []
        self.breakaway, self.hold, self.speed = breakaway, hold, speed
        self.latency_s, self.duty_per_tick = latency_s, duty_per_tick
        self._moving = {m: False for m in calibration}
        self._lock = threading.Lock()
        self._t = time.monotonic()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, handshake: bool = True) -> None:
        if self._connected:
            raise DeviceAlreadyConnectedError("SimBus already connected")
        self._connected, self._t = True, time.monotonic()

    def disconnect(self, disable_torque: bool = True) -> None:
        if disable_torque:
            self.disable_torque()
        self._connected = False

    def read_calibration(self) -> dict[str, MotorCalibration]:
        self._transaction()
        return dict(self.eeprom)

    def _transaction(self) -> None:
        if not self._connected:
            raise DeviceNotConnectedError("SimBus is not connected")
        time.sleep(self.latency_s)
        with self._lock:
            now = time.monotonic()
            dt, self._t = now - self._t, now
            for m in self.pos:
                self._step(m, dt)

    def _step(self, m: str, dt: float) -> None:
        goal = self.reg["Goal_Position"][m]
        breakaway, hold = self.stiction.get(m, (self.breakaway, self.hold))
        if self.reg["Torque_Enable"][m]:
            err = goal - self.sag[m] - self.pos[m]
            if abs(err) > breakaway:
                self._moving[m] = True
            room = abs(err) - hold
            if self._moving[m] and room > 0:
                cap = self.reg["Goal_Velocity"][m] or self.speed
                self.pos[m] += math.copysign(min(room, min(cap, self.speed) * dt), err)
            elif room <= 0:
                self._moving[m] = False
        lo, hi = self.blocks.get(m, (-math.inf, math.inf))
        if m == "gripper" and self.grip_stall is not None:
            lo = max(lo, self.grip_stall)
        self.pos[m] = min(max(self.pos[m], lo), hi)
        err = goal - self.pos[m]
        duty = min(self.reg["Torque_Limit"][m], self.duty_per_tick * abs(err)) if self.reg["Torque_Enable"][m] else 0
        self.reg["Present_Load"][m] = int(math.copysign(duty, err))
        self.reg["Present_Position"][m] = int(round(self.pos[m]))
        if m in self.trace:
            self.trace[m].append(self.pos[m])

    def _names(self, motors) -> list[str]:
        if motors is None:
            return list(self.pos)
        return [motors] if isinstance(motors, str) else list(motors)

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0) -> dict[str, int]:
        if normalize:
            raise ValueError("SimBus serves raw register values only (normalize=False)")
        if self.fail_reads > 0 and self.fail_after > 0:
            self.fail_after -= 1
        elif self.fail_reads > 0:
            self.fail_reads -= 1
            raise ConnectionError(f"Failed to sync read '{data_name}' (simulated bus failure)")
        self._transaction()
        return {m: self.reg[data_name][m] for m in self._names(motors)}

    def read(self, data_name, motor, *, normalize=True, num_retry=0) -> int:
        return self.sync_read(data_name, motor, normalize=normalize)[motor]

    def sync_write(self, data_name, values, *, normalize=True, num_retry=0) -> None:
        if normalize:
            raise ValueError("SimBus takes raw register values only (normalize=False)")
        self._transaction()
        for m, v in values.items():
            self._set(data_name, m, v)

    def write(self, data_name, motor, value, *, normalize=True, num_retry=0) -> None:
        self.sync_write(data_name, {motor: value}, normalize=normalize)

    def _set(self, data_name: str, m: str, value) -> None:
        if data_name == "Torque_Enable" and value not in (0, 1):
            raise ValueError(f"Torque_Enable={value}: 128 would re-home the servo")
        self.writes.append((data_name, m, int(value)))
        self.reg[data_name][m] = int(value)

    def enable_torque(self, motors=None, num_retry=0) -> None:
        for m in self._names(motors):
            self._set("Torque_Enable", m, 1)
            self._set("Lock", m, 1)

    def disable_torque(self, motors=None, num_retry=0) -> None:
        for m in self._names(motors):
            self._set("Torque_Enable", m, 0)
            self._set("Lock", m, 0)

# Gemini 2 color intrinsics at 1280x720, as served by artist's camera_stream on 2026-10-01.
FIELD_K = dict(fx=686.65, fy=686.52, cx=641.32, cy=359.03, width=1280, height=720)
TAPE_IN_GRIPPER_M = (0.0, 0.015, -0.02)  # tape centre in gripper_frame_link (m): the tracked point is not the TCP

TABLE_RGB = (70, 120, 80)
CAN_RGB = (200, 25, 35)
LID_RGB = (175, 175, 180)
TAPE_RGB = (40, 80, 220)  # OpenCV hue ~112: inside the field tape band 100..125


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """4x4 camera-from-base transform for a camera at `eye` looking at `target` (base mm)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ eye
    return T


@dataclass
class BenchScene:
    K: dict = field(default_factory=lambda: dict(FIELD_K))
    cam_from_base: np.ndarray = field(default_factory=lambda: look_at((620.0, -260.0, 420.0), (250.0, 0.0, 40.0)))
    can_xy: tuple[float, float] = (300.0, 60.0)
    can_radius: float = 33.0
    can_height: float = 122.0
    tape_radius: float = 10.0
    spheres: list[tuple[tuple[float, float, float], float, tuple[int, int, int]]] = field(default_factory=list)
    noise_mm: float = 0.6
    seed: int = 0

    def to_cam(self, p_base) -> np.ndarray:
        p = np.asarray(p_base, float)
        return p @ self.cam_from_base[:3, :3].T + self.cam_from_base[:3, 3]

    def lid_center_cam(self) -> np.ndarray:
        return self.to_cam([self.can_xy[0], self.can_xy[1], self.can_height])

    def table_normal_cam(self) -> np.ndarray:
        return self.cam_from_base[:3, :3] @ np.array([0.0, 0.0, 1.0])

    def render(self, tape_base=None) -> tuple[np.ndarray, np.ndarray]:
        """(RGB uint8 (H, W, 3), depth uint16 mm (H, W)); tape_base = gripper tape centre in base mm or None."""
        K = self.K
        H, W = K["height"], K["width"]
        v, u = np.mgrid[0:H, 0:W]
        rays = np.stack([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], np.ones((H, W))], -1).reshape(-1, 3)
        R, t = self.cam_from_base[:3, :3], self.cam_from_base[:3, 3]
        best = np.full(len(rays), np.inf)
        rgb = np.zeros((len(rays), 3), np.uint8)

        def take(hit_t, color):
            win = (hit_t > 0) & (hit_t < best)
            best[win] = hit_t[win]
            rgb[win] = color

        with np.errstate(divide="ignore", invalid="ignore"):
            n_c = R @ np.array([0.0, 0, 1])
            take((n_c @ t) / (rays @ n_c), TABLE_RGB)  # table z = 0
            a = n_c
            c0 = self.to_cam([self.can_xy[0], self.can_xy[1], 0.0])
            w = rays - np.outer(rays @ a, a)
            m = -c0 + (c0 @ a) * a
            A, B, C = (w * w).sum(1), 2 * (w @ m), m @ m - self.can_radius**2
            disc = B * B - 4 * A * C
            side = (-B - np.sqrt(np.where(disc >= 0, disc, np.nan))) / (2 * A)
            along = side * (rays @ a) - c0 @ a
            take(np.where((along >= 0) & (along <= self.can_height), side, np.nan), CAN_RGB)
            c1 = c0 + a * self.can_height
            cap = (a @ c1) / (rays @ a)
            on_cap = np.linalg.norm(cap[:, None] * rays - c1, axis=1) <= self.can_radius
            take(np.where(on_cap, cap, np.nan), LID_RGB)
            spheres = list(self.spheres)
            if tape_base is not None:
                spheres.append((tuple(tape_base), self.tape_radius, TAPE_RGB))
            for center, radius, color in spheres:
                s = self.to_cam(center)
                b, c = rays @ s, s @ s - radius**2
                disc = b * b - (rays * rays).sum(1) * c
                take((b - np.sqrt(np.where(disc >= 0, disc, np.nan))) / (rays * rays).sum(1), color)

        rng = np.random.default_rng(self.seed)
        hit = np.isfinite(best)
        depth = np.where(hit, best + rng.normal(0, self.noise_mm, len(best)), 0)
        return rgb.reshape(H, W, 3), np.clip(depth, 0, 65535).astype(np.uint16).reshape(H, W)


class SimCamera:
    """The bench scene as the Gemini sees it, with the gripper tape wherever the simulated arm put it."""

    def __init__(self, scene: BenchScene, bus: SimBus, kin, tape_offset_m=TAPE_IN_GRIPPER_M, joint_offsets_deg=None):
        """joint_offsets_deg: this world's true model-minus-reading angles (the real arm's are tens of degrees)."""
        from kinematics import JOINTS

        self.scene, self.bus, self.kin, self.joints = scene, bus, kin, JOINTS
        self.tape_offset = np.r_[np.asarray(tape_offset_m, float), 1.0]
        self.offsets = np.radians(np.zeros(len(JOINTS)) if joint_offsets_deg is None else joint_offsets_deg)
        self.intrinsics = dict(scene.K)
        self.width, self.height = scene.K["width"], scene.K["height"]

    def tape_base_mm(self) -> np.ndarray:
        cal = self.bus.calibration
        q = [(self.bus.pos[m] - (cal[m].range_min + cal[m].range_max) / 2) * 2 * np.pi / 4095 for m in self.joints]
        return (self.kin.fk(np.asarray(q) + self.offsets) @ self.tape_offset)[:3] * 1000.0

    def read_rgbd(self, timeout_ms: float = 1000) -> tuple[np.ndarray, np.ndarray]:
        return self.scene.render(tape_base=self.tape_base_mm())


def sim_rig(scene: BenchScene, motor_calibration_path, joint_offsets_deg=None) -> "RigCalibration":
    """The exact rig calibration of the simulated world: its joint offsets, the scene's true camera pose."""
    from calibration import RigCalibration, file_sha256
    from kinematics import JOINTS

    n = scene.table_normal_cam()
    offsets = np.zeros(len(JOINTS)) if joint_offsets_deg is None else np.asarray(joint_offsets_deg, float)
    return RigCalibration(
        joint_offsets_deg={m: float(v) for m, v in zip(JOINTS, offsets)},
        camera_from_base_mm=np.asarray(scene.cam_from_base).tolist(),
        marker_in_gripper_mm=[v * 1000.0 for v in TAPE_IN_GRIPPER_M], bench_normal_cam=n.tolist(),
        bench_d_cam_mm=float(n @ scene.cam_from_base[:3, 3]), bench_z_base_mm=0.0, tilt_deg=0.0, rms_mm=0.0,
        max_mm=0.0, created="simulation", robot_id="exp23_follower",
        motor_calibration_sha256=file_sha256(motor_calibration_path), camera_serial="SIM")
