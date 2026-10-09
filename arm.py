"""Request-scoped, supervised motion for the SO-101 on a LeRobot Feetech bus: one motion at a time, a receipt each.

Replaces artist's teleop.py daemon (authors galois777, David Zhang, Hillysilly). Kept from the field: enabling
torque adopts the held pose, communication faults keep torque on, a hot gripper relaxes only itself, three
consecutive plausible temperature readings before acting, Torque_Limit 500 / gripper 250. Added for an agent:
per-request receipts; a goal leash, so a blocked joint never banks more than LEASH ticks of travel to release
later; stall detection by lack of progress (load saturates at Torque_Limit, so a load threshold cannot see a
jam); staged thermal handling instead of dropping the arm at 55 C; a stop() that never waits for the motion.

Gravity: with the servos' P gain of 16 a loaded joint holds only while its goal leads its position (measured on
the arm 2026-10-08: the elbow sat 26..55 ticks short). So a motion starts from the goal the servo is holding,
never from the measured position (that would drop the arm), ends by nudging the goal on until the joint really
is at the target (sag compensation, arm joints only), and on stop/timeout simply stops moving the goal.

API units: joints in LeRobot degrees (zero = calibration range midpoint, i.e. the pose the arm was held in during
motor calibration), gripper as an opening percent (0 = empty jaws closed, 100 = fully open, from the field gripper
map), Cartesian targets in URDF base-frame millimetres. The URDF's own zero is a different pose: kinematics uses
model angle = reading + offset, offsets from the rig calibration (calibration.py; this arm: shoulder -103, elbow +87,
wrist -13 deg). Without a rig calibration the offsets are 0, which is wrong on this arm. Internally everything is raw
ticks: deadband, leash and stiction are tick effects.
"""

import math
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from lerobot.motors import MotorCalibration
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from calibration import RigCalibration, file_sha256
from kinematics import JOINTS, TICKS_PER_TURN, IKSolution, Kinematics, joint_bounds_deg, solve_ik

GRIPPER = "gripper"
MOTORS = (*JOINTS, GRIPPER)
TICKS_PER_DEG = TICKS_PER_TURN / 360

# Measured stops (2026-10-08, slow drive into each end until the stall detector fired; the bench guard on):
# shoulder_lift back stop at 2005; elbow folds onto the upper arm at 2084 (the URDF would allow 2157) and opens past
# 97 with no stop, so its low end is kept far from the 0/4095 encoder wrap; wrist_flex reaches its EEPROM limits
# 1710..3790 freely; the jaws touch at 657 and open past 1497 freely; wrist_roll turning down meets a stop tab on the
# printed wrist at 971/972/971 (three runs, two arm poses; a designed stop), which going up only comes round again
# past the encoder wrap; the URDF's 3899 put the rest pose (3887..3901) on the edge. Pan turns freely to its EEPROM
# limits 618..3136 both ways (load ~110), so its calibration range stands.
# The shoulder's encoder wraps (4095 -> 0) with the arm level in front, where gravity pulls it on toward the wrap; a
# loaded joint can sit up to a LEASH (~120 ticks) past its goal, and sag compensation leads a goal as far past its
# target, so the shoulder's top stays 200 ticks clear of the wrap. The roll carries almost no gravity load: the lead
# alone (114) must stay clear of it.
HARD_TICKS = {"shoulder_lift": (2030, 3895), "elbow_flex": (400, 2060), "wrist_roll": (996, 3920)}
GRIPPER_CLOSED, GRIPPER_OPEN = 665, 1200  # 0 % = jaws just closed .. 100 % = wide open
GRIPPER_WINDOW = (655, 1210)
TORQUE_LIMIT, GRIPPER_TORQUE_LIMIT = 500, 250
ACCELERATION = 254
GOAL_VELOCITY = 1500  # servo-side speed cap, steps/s (teleop used 0 = none); bounds a release lunge. Verify on hardware.
LEASH, GRIPPER_LEASH = 120, 120  # duty ~4.2/tick of lead (measured), so ~120 ticks is where Torque_Limit saturates
GRASP_MARGIN, GRASP_MIN_GAP, GRASP_MAX_POS = 38, 30, 1180  # squeeze past the stall; held = stopped >= gap short

CYCLE_S, MAX_DT = 1 / 30, 0.1
STALL_WINDOW_S, STALL_PROGRESS = 0.3, 3
# Contact while moving (a hand, an object): once a joint has moved CONTACT_MOVED ticks in this motion (past any
# breakaway, which takes up to ~100 ticks of lead), it is blocked when it lags its still-advancing goal by CONTACT_LEAD
# ticks and covered under CONTACT_FRACTION of the goal's advance over CONTACT_WINDOW_S. Pan at 6 deg/s (2026-10-08):
# free motion lags ~13 ticks; against a hand the lag grew 33 -> 111 while the joint crept on in stick-slip, which the
# stall rule only caught 2.7 s in (load 456); replayed, this rule fires 0.9 s in (load 288).
CONTACT_WINDOW_S, CONTACT_LEAD, CONTACT_FRACTION, CONTACT_MOVED, CONTACT_MIN_ADVANCE = 0.5, 70, 0.3, 30, 20
SETTLE_CYCLES, SETTLE_TIMEOUT_S = 3, 4.0
# Sag/stiction compensation (ticks/s; lead cap as a leash fraction). The real elbow needed ~100 ticks of lead (load
# ~370 of 500) to break free on small moves, in both directions, so the cap sits just under torque saturation.
COMP_RATE, COMP_CAP, COMP_STUCK_S, COMP_INNER = 90.0, 0.95, 0.2, 4
PINNED_S = 0.6  # at the lead cap this long without moving: not stiction but an obstacle (or the arm's own fold)
SLOW_READ_S = 0.5
WARN_C, LOCKOUT_C, RELEASE_C, GRIPPER_RELAX_C = 50, 58, 66, 55
PLAUSIBLE_C = (5, 100)
CONSECUTIVE = 3
MAX_SEGMENT_MM = 30.0
CLEAR_TIP_MM, CLEAR_LINK_MM = 8.0, 25.0  # least height above the bench: TCP/marker, and wrist/gripper link origins
CLEARANCE_SAMPLES = 8  # poses checked along a joint move
# Static gravity torque (URDF masses + the wrist camera) the shoulder can still lift at Torque_Limit 500 on this 5.3 V
# supply: from rest it lifted 0.71 and 0.78 N.m (peak load 396, 412 of 500) and stalled at 0.93 (2026-10-08). An arm
# stretched out in front past this sinks toward the bench and cannot be raised again, so no plan may end there. A held
# object is not counted: its weight times its reach adds to the load.
SHOULDER_LIFT_NM = 0.7
WRIST_CAMERA_KG = 0.05  # camera + bracket, not weighed
# The arm's body as capsules around the joint-to-joint segments, base first (motor housings reach ~60 mm off the
# joint axes), plus what is bolted on: this arm carries the wrist camera on a ~90 mm bracket off the gripper body
# (link frame, mm; measured with the depth camera 2026-10-08).
BODY_RADIUS_MM = (90.0, 65.0, 65.0, 65.0, 65.0, 50.0)
ATTACHMENTS = {"gripper_link": (((-20.0, -15.0, -10.0), (-95.0, -25.0, 5.0), 40.0),)}


class ArmError(RuntimeError):
    pass


class BusyError(ArmError):
    pass


class LimitError(ArmError):
    pass


class ThermalError(ArmError):
    pass


@dataclass
class MoveReceipt:
    """status: reached | settled (sag compensation ran out of time or lead: still short by `error`) | blocked |
    stopped | timeout | comm_error | thermal | thermal_release. Joint values in degrees, the gripper in opening percent."""

    status: str
    requested: dict[str, float]
    final: dict[str, float]
    error: dict[str, float]
    max_load: dict[str, int]
    max_temp: dict[str, int]
    duration_s: float
    mean_cycle_ms: float
    blocked: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class CartesianReceipt:
    """status: reached (model TCP within tol_mm) | off_target (joints settled inside their deadband but the TCP
    is further out: refine with servo_to) | the joint move's failure status."""

    status: str
    move: MoveReceipt
    target_mm: list[float]
    planned_mm: list[float]  # model TCP the IK reached for the last segment
    final_mm: list[float]  # model TCP of the measured joints after the move ([] if they could not be read)
    tcp_error_mm: float | None
    segments: int
    jaw_axis: list[float]


@dataclass
class GraspReceipt:
    held: bool
    reason: str
    stall_tick: int | None
    hold_tick: int | None
    opening_pct: float
    load: int
    move: MoveReceipt


@dataclass
class ArmState:
    connected: bool
    calibration_ok: bool | None
    calibration_mismatches: list[str]
    joints_deg: dict[str, float]
    gripper_opening_pct: float
    ticks: dict[str, int]
    torque: dict[str, int]
    temperature_c: dict[str, int]
    load: dict[str, int]
    status_bits: dict[str, int]
    voltage_v: dict[str, float]
    tcp_mm: list[float]
    jaw_axis: list[float]
    tcp_above_bench_mm: float
    limits: dict[str, list[float]]
    joint_offsets_deg: dict[str, float]
    rig_calibration: str  # where the offsets / camera pose come from, and how good they are
    busy: bool
    thermal_lockout: bool
    hot: list[str]
    gripper_relaxed: bool
    watchdog_error: str | None
    auto_calibration: str | None = None  # the server's camera-link monitor, when it runs one


class Arm:
    def __init__(self, bus, calibration: dict[str, MotorCalibration], kin: Kinematics | None = None,
                 rest_pose_deg: dict[str, float] | None = None, robot=None, rig: RigCalibration | None = None,
                 motor_calibration_sha256: str | None = None):
        self.bus, self.cal, self.kin = bus, calibration, kin or Kinematics()
        self._robot = robot  # keeps LeRobot's object alive; its __del__ disconnects (torque kept: see open())
        self.rest_pose_deg = rest_pose_deg
        self.mid = {m: (c.range_min + c.range_max) / 2 for m, c in calibration.items()}
        self.motor_calibration_sha256 = motor_calibration_sha256
        self.use_rig(rig)
        self._bus_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._connect_lock = threading.Lock()
        self._motion = threading.Lock()
        self._moving, self._stop, self._held = threading.Event(), threading.Event(), threading.Event()
        self._watch_stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._configured: set[str] = set()
        self._streak: dict[tuple[str, int], int] = {}
        self.calibration_ok: bool | None = None
        self.mismatches: list[str] = []
        self.lockout = False
        self.hot: list[str] = []
        self.gripper_relaxed = False
        self.watchdog_error: str | None = None

    def use_rig(self, rig: RigCalibration | None) -> None:
        """Adopt a rig calibration: joint offsets (kinematics and bounds), bench height, camera pose."""
        self.rig, offsets = rig, np.zeros(len(JOINTS))
        if rig is None:
            self.rig_note = "none: joint offsets 0 and no camera pose (run calibrate_rig on a real arm)"
        elif self.motor_calibration_sha256 is not None and rig.motor_calibration_sha256 != self.motor_calibration_sha256:
            self.rig, self.rig_note = None, ("ignored: fitted to a different motor calibration, offsets 0 "
                                             "(re-run calibrate_rig)")
        else:
            offsets = rig.offsets_array()
            self.rig_note = (f"{rig.created}, {len(rig.samples)} poses, marker rms {rig.rms_mm} mm, "
                             f"camera {rig.camera_serial}")
        self.use_model(offsets, self.rig.bench_z_base_mm if self.rig is not None else 0.0)

    def use_model(self, offsets_deg, bench_z_mm: float) -> None:
        """Joint offsets and bench height without a camera link: kinematics, bounds and the bench guard. A calibration
        that starts with no rig drives the arm on a prior model through this."""
        self.offsets, self.bench_z = np.asarray(offsets_deg, float), float(bench_z_mm)
        lo_deg, hi_deg = joint_bounds_deg(self.kin, self.cal, offsets_deg=self.offsets,
                                          urdf_exempt=("shoulder_lift:lo", "wrist_roll:hi"))  # measured: HARD_TICKS
        self.bounds: dict[str, tuple[int, int]] = {GRIPPER: GRIPPER_WINDOW}
        for m, lo, hi in zip(JOINTS, lo_deg, hi_deg):
            hard = HARD_TICKS.get(m, (0, TICKS_PER_TURN))
            self.bounds[m] = (max(math.ceil(self.to_tick_f(m, lo)), hard[0]), min(math.floor(self.to_tick_f(m, hi)), hard[1]))
        self.lo_deg = np.array([self.to_deg(m, self.bounds[m][0]) for m in JOINTS])
        self.hi_deg = np.array([self.to_deg(m, self.bounds[m][1]) for m in JOINTS])

    @classmethod
    def open(cls, port: str, robot_id: str = "exp23_follower", calibration_dir=None, rig="auto", **kw) -> "Arm":
        """rig: a RigCalibration, None, or "auto" = calibration/rig.json of this repo when present."""
        config = SO101FollowerConfig(port=port, id=robot_id, calibration_dir=calibration_dir,
                                     disable_torque_on_disconnect=False)
        robot = SO101Follower(config)
        if not robot.calibration:
            raise ArmError(f"no calibration file at {robot.calibration_fpath}")
        if rig == "auto":
            rig = RigCalibration.load()
        return cls(robot.bus, robot.calibration, robot=robot, rig=rig,
                   motor_calibration_sha256=file_sha256(robot.calibration_fpath), **kw)

    # --- units ---------------------------------------------------------------------------------------------
    def to_tick_f(self, motor: str, deg: float) -> float:
        return deg * TICKS_PER_DEG + self.mid[motor]

    def to_tick(self, motor: str, deg: float) -> int:
        return int(round(self.to_tick_f(motor, deg)))

    def to_deg(self, motor: str, tick: float) -> float:
        return (tick - self.mid[motor]) / TICKS_PER_DEG

    @staticmethod
    def gripper_tick(opening_pct: float) -> int:
        return int(round(GRIPPER_CLOSED + (GRIPPER_OPEN - GRIPPER_CLOSED) * opening_pct / 100))

    @staticmethod
    def gripper_pct(tick: float) -> float:
        return (tick - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED) * 100

    def api_value(self, motor: str, tick: float) -> float:
        return round(self.gripper_pct(tick) if motor == GRIPPER else self.to_deg(motor, tick), 2)

    def tcp(self, joints_deg) -> tuple[np.ndarray, np.ndarray]:
        """Model TCP (base mm) and jaw axis for the five arm joint readings."""
        T = self.kin.fk(np.radians(np.asarray(joints_deg, float) + self.offsets))
        return T[:3, 3] * 1000.0, T[:3, 2]

    def link_points(self, joints_deg=None) -> list[np.ndarray]:
        """Base-frame mm origins along the arm, base to TCP (the body approximated by segments between them)."""
        if joints_deg is None:
            self.connect()
            pos = self._read("Present_Position", list(JOINTS))
            joints_deg = [self.to_deg(m, pos[m]) for m in JOINTS]
        q = np.radians(np.asarray(joints_deg, float) + self.offsets)
        links = ("shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "gripper_frame_link")
        return [np.zeros(3)] + [self.kin.fk(q, link=link)[:3, 3] * 1000.0 for link in links]

    def _attachment_points(self, joints_deg) -> list[tuple[np.ndarray, np.ndarray, float]]:
        q = np.radians(np.asarray(joints_deg, float) + self.offsets)
        out = []
        for link, items in ATTACHMENTS.items():
            T = self.kin.fk(q, link=link)
            for start, end, radius in items:
                out.append((T[:3, :3] @ np.asarray(start) + T[:3, 3] * 1000.0,
                            T[:3, :3] @ np.asarray(end) + T[:3, 3] * 1000.0, radius))
        return out

    def body_segments(self, joints_deg=None) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """The arm's body as (start, end, radius) capsules in the base frame, mm: links, then attachments."""
        if joints_deg is None:
            self.connect()
            pos = self._read("Present_Position", list(JOINTS))
            joints_deg = [self.to_deg(m, pos[m]) for m in JOINTS]
        pts = self.link_points(joints_deg)
        return [(a, b, r) for (a, b), r in zip(zip(pts[:-1], pts[1:]), BODY_RADIUS_MM)] + \
            self._attachment_points(joints_deg)

    def clearance(self, joints_deg) -> tuple[float, str]:
        """Least margin (mm) by which this pose clears the bench, and which point sets it: the TCP and the tracked
        marker need CLEAR_TIP_MM, the wrist and gripper link origins CLEAR_LINK_MM."""
        q = np.radians(np.asarray(joints_deg, float) + self.offsets)
        T = self.kin.fk(q)
        points = {"tcp": (T[:3, 3] * 1000.0, CLEAR_TIP_MM)}
        if self.rig is not None:
            points["marker"] = (T[:3, :3] @ np.asarray(self.rig.marker_in_gripper_mm) + T[:3, 3] * 1000.0, CLEAR_TIP_MM)
        for link in ("gripper_link", "wrist_link", "lower_arm_link"):  # lower_arm_link = the elbow
            points[link] = (self.kin.fk(q, link=link)[:3, 3] * 1000.0, CLEAR_LINK_MM)
        for k, (_, end, radius) in enumerate(self._attachment_points(joints_deg)):
            points[f"attachment {k} (wrist camera bracket)"] = (end, radius)
        margins = {name: p[2] - self.bench_z - need for name, (p, need) in points.items()}
        worst = min(margins, key=margins.get)
        return float(margins[worst]), worst

    def shoulder_load_nm(self, joints_deg) -> float:
        """Static gravity torque (N.m) on the shoulder in this pose; negative pulls the arm back toward its stop."""
        q = np.radians(np.asarray(joints_deg, float) + self.offsets)
        camera = [(WRIST_CAMERA_KG, (a + b) / 2000.0) for a, b, _ in self._attachment_points(joints_deg)]
        return self.kin.gravity_torque(q, "shoulder_lift", camera)

    def _check_poses(self, poses_deg, what: str, start_deg=None) -> None:
        """Refuse a path that comes too close to the bench or ends where the shoulder could not lift the arm again.
        A path that gets no lower and no heavier than where it starts is allowed, so an arm left low or stretched out
        (pushed there, or after contact) can always be brought back; it must still end in a liftable pose."""
        floor = 0.0 if start_deg is None else min(0.0, self.clearance(start_deg)[0] - 1.0)
        ceiling = SHOULDER_LIFT_NM if start_deg is None else max(SHOULDER_LIFT_NM,
                                                                  abs(self.shoulder_load_nm(start_deg)) + 0.02)
        for k, q in enumerate(poses_deg):
            last = k == len(poses_deg) - 1
            where = "at the target" if last else f"on the way ({k + 1}/{len(poses_deg)})"
            margin, point = self.clearance(q)
            if margin < floor:
                raise LimitError(f"{what} would bring the {point} {abs(margin):.0f} mm too close to the bench {where}")
            load = abs(self.shoulder_load_nm(q))
            if load > (SHOULDER_LIFT_NM if last else ceiling):
                raise LimitError(f"{what} would leave the shoulder holding {load:.2f} N.m {where}; it can lift "
                                 f"{SHOULDER_LIFT_NM} N.m, so the arm would sink and stay down. Keep the arm folded "
                                 "nearer the shoulder (more elbow bend) or aim closer in")

    def limits_api(self) -> dict[str, list[float]]:
        return {m: [self.api_value(m, lo), self.api_value(m, hi)] for m, (lo, hi) in self.bounds.items()}

    # --- bus -----------------------------------------------------------------------------------------------
    def _read(self, name: str, motors=None) -> dict[str, int]:
        with self._bus_lock:
            return self.bus.sync_read(name, motors, normalize=False, num_retry=2)

    def _write_goals(self, goals: dict[str, int]) -> None:
        with self._bus_lock:
            self.bus.sync_write("Goal_Position", goals, normalize=False)

    def connect(self) -> None:
        with self._connect_lock:
            if self.bus.is_connected:
                return
            try:
                with self._bus_lock:
                    self.bus.connect()
                    on_motor = self.bus.read_calibration()
            except ConnectionError as e:
                raise ArmError(f"cannot open the arm's serial port: {e}. Only one program can own it: is "
                               "artist's teleop daemon (or a LeRobot script) still running?") from e
            self.mismatches = []
            for m, c in self.cal.items():
                got, want = on_motor[m], c
                if (got.homing_offset, got.range_min, got.range_max) != (want.homing_offset, want.range_min, want.range_max):
                    self.mismatches.append(f"{m}: motor offset/min/max {got.homing_offset}/{got.range_min}/"
                                           f"{got.range_max}, file {want.homing_offset}/{want.range_min}/{want.range_max}")
            self.calibration_ok = not self.mismatches
            self._watch_stop.clear()
            self._watchdog = threading.Thread(target=self._watch, name="arm-watchdog", daemon=True)
            self._watchdog.start()

    def disconnect(self) -> None:
        """Close the port with torque left exactly as it is."""
        self._watch_stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=3)
        with self._bus_lock:
            if self.bus.is_connected:
                self.bus.disconnect(disable_torque=False)

    # --- state -----------------------------------------------------------------------------------------------
    def state(self) -> ArmState:
        self.connect()
        pos = self._read("Present_Position")
        regs = {name: self._read(name) for name in ("Torque_Enable", "Present_Temperature", "Present_Load", "Status",
                                                     "Present_Voltage")}
        joints = {m: self.api_value(m, pos[m]) for m in JOINTS}
        xyz, axis = self.tcp([joints[m] for m in JOINTS])
        return ArmState(
            connected=True, calibration_ok=self.calibration_ok, calibration_mismatches=self.mismatches,
            joints_deg=joints, gripper_opening_pct=self.api_value(GRIPPER, pos[GRIPPER]), ticks=pos,
            torque=regs["Torque_Enable"], temperature_c=regs["Present_Temperature"],
            load={m: abs(v) for m, v in regs["Present_Load"].items()}, status_bits=regs["Status"],
            voltage_v={m: v / 10 for m, v in regs["Present_Voltage"].items()},
            tcp_mm=[round(float(v), 1) for v in xyz], jaw_axis=[round(float(v), 3) for v in axis],
            tcp_above_bench_mm=round(float(xyz[2] - self.bench_z), 1), limits=self.limits_api(),
            joint_offsets_deg={m: round(float(v), 2) for m, v in zip(JOINTS, self.offsets)},
            rig_calibration=self.rig_note, busy=self._moving.is_set(), thermal_lockout=self.lockout, hot=self.hot,
            gripper_relaxed=self.gripper_relaxed, watchdog_error=self.watchdog_error)

    # --- safety --------------------------------------------------------------------------------------------
    def _check_thermal(self, temps: dict[str, int]) -> str | None:
        """Staged: hot >= 50 C warns, >= 58 locks out new motion, >= 66 releases all torque; the gripper
        relaxes itself at >= 55 (it heats ~1 C/min when stalled on an object). Three plausible readings in a row."""
        with self._state_lock:
            for m, t in temps.items():
                plausible = PLAUSIBLE_C[0] <= t <= PLAUSIBLE_C[1]
                for level in (WARN_C, GRIPPER_RELAX_C, LOCKOUT_C, RELEASE_C):
                    self._streak[(m, level)] = self._streak.get((m, level), 0) + 1 if plausible and t >= level else 0
            def over(level: int, motors) -> list[str]:
                return [m for m in motors if self._streak.get((m, level), 0) >= CONSECUTIVE]

            self.hot = over(WARN_C, MOTORS)
            if over(GRIPPER_RELAX_C, [GRIPPER]) and not self.gripper_relaxed:
                with self._bus_lock:
                    self.bus.disable_torque(GRIPPER, num_retry=2)
                self.gripper_relaxed = True
            if over(RELEASE_C, JOINTS):
                with self._bus_lock:
                    self.bus.disable_torque(num_retry=5)
                self.lockout = True
                return "release_all"
            if over(LOCKOUT_C, JOINTS):
                self.lockout = True
                return "lockout"
            if self.lockout and all(temps.get(m, 0) < WARN_C for m in JOINTS):
                self.lockout = False
            return None

    def _watch(self) -> None:
        while not self._watch_stop.wait(1.0):
            try:
                with self._state_lock:
                    if self._moving.is_set():
                        continue
                    if any(self._read("Torque_Enable").values()):
                        self._check_thermal(self._read("Present_Temperature"))
                self.watchdog_error = None
            except Exception as e:
                self.watchdog_error = repr(e)

    def _require_motion_ok(self) -> None:
        self.connect()
        if not self.calibration_ok:
            raise ArmError("motor EEPROM calibration differs from the calibration file; refusing to move "
                           f"(reads still work): {self.mismatches}")
        if self.lockout:
            raise ThermalError(f"thermal lockout ({self.hot or 'arm joints'} reached {LOCKOUT_C} C): "
                               f"let them cool below {WARN_C} C")

    @property
    def halted(self) -> bool:
        """stop() ended the last motion: nothing may move the arm again until the agent asks for a motion."""
        return self._held.is_set()

    @contextmanager
    def _motion_slot(self):
        if not self._motion.acquire(blocking=False):
            raise BusyError("another motion is running: wait for its receipt or call stop()")
        try:
            with self._state_lock:
                self._moving.set()
            self._stop.clear()
            self._held.clear()
            yield
        finally:
            self._moving.clear()
            self._motion.release()

    def stop(self) -> dict:
        """Halt the running motion now: its goals stop advancing, so each joint keeps only the lead it holds with
        (at most a leash). With no motion running there is nothing to stop, and nothing is written."""
        self._stop.set()
        if not self._moving.is_set():
            return dict(status="idle", note="no motion was running; the arm keeps holding its goals")
        if self._held.wait(1.0):
            return dict(status="held", by="executor")
        if self._bus_lock.acquire(timeout=0.5):  # the executor is stuck in a bus call: stop the joints here
            try:
                pos = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
                torque = self.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
                self.bus.sync_write("Goal_Position", {m: pos[m] for m in pos if torque[m]}, normalize=False)
                return dict(status="held", by="stop",
                            note="executor unresponsive: goals set to the measured positions, loaded joints may sag")
            finally:
                self._bus_lock.release()
        return dict(status="unconfirmed", note="the bus did not answer within 0.5 s")

    def release(self, which: str = "gripper", confirm: bool = False) -> dict:
        """Torque off. 'all' drops the arm, 'gripper' drops what the jaws hold: both need confirm=True."""
        if which not in ("gripper", "all"):
            raise ArmError("which must be 'gripper' or 'all'")
        if not confirm:
            raise ArmError(f"releasing {which} drops {'the arm' if which == 'all' else 'whatever the jaws hold'}: "
                           "call again with confirm=true once that is safe")
        self.connect()
        self.stop()
        with self._bus_lock:
            self.bus.disable_torque(GRIPPER if which == "gripper" else None, num_retry=5)
        return dict(status="released", which=which)

    # --- execution -----------------------------------------------------------------------------------------
    def _adopt(self) -> tuple[dict[str, int], dict[str, int]]:
        """Torque on with zero motion; returns (measured positions, goals to ramp from). A motor whose torque was
        off gets goal := position. One that is on keeps the goal it holds with (that lead carries gravity), only
        clamped to a leash around its position, so a stale goal can never bank travel."""
        with self._bus_lock:
            torque = self.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
            pos = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
            held = self.bus.sync_read("Goal_Position", normalize=False, num_retry=2)
            motors = [m for m in MOTORS if not (m == GRIPPER and self.gripper_relaxed)]
            start, rewrite = dict(pos), {}
            for m in motors:
                leash = GRIPPER_LEASH if m == GRIPPER else LEASH
                start[m] = pos[m] if not torque[m] else int(np.clip(held[m], pos[m] - leash, pos[m] + leash))
                if not torque[m] or start[m] != held[m]:
                    rewrite[m] = start[m]
            if rewrite:
                self.bus.sync_write("Goal_Position", rewrite, normalize=False)
            for m in motors:
                if m not in self._configured:
                    limit = GRIPPER_TORQUE_LIMIT if m == GRIPPER else TORQUE_LIMIT
                    self.bus.write("Torque_Limit", m, limit, normalize=False, num_retry=2)
                    self.bus.write("Acceleration", m, ACCELERATION, normalize=False, num_retry=2)
                    self.bus.write("Goal_Velocity", m, GOAL_VELOCITY, normalize=False, num_retry=2)
                    self._configured.add(m)
                if not torque[m]:
                    self.bus.enable_torque(m, num_retry=2)
        return pos, start

    @staticmethod
    def _progress(hist: deque, now: float, window: float) -> float:
        """Ticks moved over the last `window` seconds of a (time, position) history."""
        for t, p in hist:
            if t >= now - window:
                return abs(hist[-1][1] - p)
        return 0.0

    def _run(self, path: list[dict[str, int]], speed_tps: float, tol: int, timeout_s: float) -> MoveReceipt:
        """Ramp the goals through `path` (tick waypoints, same motors each), then compensate sag until every joint
        is within `tol` ticks of the last waypoint, supervising throughout."""
        motors = list(path[-1])
        t0 = time.monotonic()
        notes: list[str] = []
        status, blocked = None, None
        max_load = {m: 0 for m in motors}
        max_temp: dict[str, int] = {}
        cycles, seg, settled, settle_start = 0, 0, 0, None
        try:
            pos, start = self._adopt()
            goal = {m: float(start[m]) for m in motors}
            hist = {m: deque([(t0, pos[m])]) for m in motors}
            origin = dict(pos)
            track = {m: deque([(t0, pos[m], goal[m])]) for m in motors}  # for the in-motion contact rule
            pinned_since: dict[str, float] = {}
            last, next_slow = t0, t0
            while status is None:
                now = time.monotonic()
                dt, last = min(now - last, MAX_DT), now
                if self._stop.is_set():
                    status = "stopped"
                    break
                if now - t0 > timeout_s:
                    status = "timeout"
                    break
                target = path[seg]
                if settle_start is None:
                    for m in motors:
                        leash = GRIPPER_LEASH if m == GRIPPER else LEASH
                        step = float(np.clip(target[m] - goal[m], -speed_tps * dt, speed_tps * dt))
                        if step > 0:
                            goal[m] = max(goal[m], min(goal[m] + step, pos[m] + leash))
                        elif step < 0:
                            goal[m] = min(goal[m], max(goal[m] + step, pos[m] - leash))
                    if all(abs(goal[m] - target[m]) < 0.5 for m in motors):
                        if seg < len(path) - 1:
                            seg += 1
                        else:
                            settle_start = now
                else:  # sag compensation: walk the goal on, slowly, only while a joint sits stuck short of target
                    for m in motors:
                        if m == GRIPPER:
                            continue
                        err = target[m] - pos[m]
                        if abs(err) > COMP_INNER and (goal[m] - target[m]) * err < 0:
                            goal[m] = float(target[m])  # broke free and crossed the target: stop pushing it on
                            continue
                        if abs(err) <= COMP_INNER or self._progress(hist[m], now, COMP_STUCK_S) >= 2:
                            continue
                        new = goal[m] + math.copysign(COMP_RATE * dt, err)
                        if abs(new - pos[m]) <= COMP_CAP * LEASH:
                            goal[m] = new
                self._write_goals({m: int(round(goal[m])) for m in motors})
                pos = self._read("Present_Position", motors)
                for m, v in self._read("Present_Load", motors).items():
                    max_load[m] = max(max_load[m], abs(v))
                if now >= next_slow:
                    temps = self._read("Present_Temperature")
                    for m in motors:
                        max_temp[m] = max(max_temp.get(m, 0), temps[m])
                    action = self._check_thermal(temps)
                    if action is not None:
                        status = "thermal_release" if action == "release_all" else "thermal"
                        break
                    next_slow = now + SLOW_READ_S
                for m in motors:
                    hist[m].append((now, pos[m]))
                    while len(hist[m]) > 1 and hist[m][1][0] <= now - STALL_WINDOW_S:
                        hist[m].popleft()
                    leash = GRIPPER_LEASH if m == GRIPPER else LEASH
                    limit = leash if settle_start is not None and m != GRIPPER else 0.9 * leash  # comp stays below
                    t_old, p_old = hist[m][0]
                    if (abs(goal[m] - pos[m]) >= limit and now - t_old >= STALL_WINDOW_S
                            and abs(pos[m] - p_old) < STALL_PROGRESS):
                        status, blocked = "blocked", m
                        break
                    track[m].append((now, pos[m], goal[m]))
                    while len(track[m]) > 1 and track[m][1][0] <= now - CONTACT_WINDOW_S:
                        track[m].popleft()
                    if settle_start is None and m != GRIPPER and abs(pos[m] - origin[m]) >= CONTACT_MOVED:
                        t_c, p_c, g_c = track[m][0]
                        advance, lag = goal[m] - g_c, abs(goal[m] - pos[m])
                        covered = (pos[m] - p_c) * math.copysign(1.0, advance)
                        if (now - t_c >= CONTACT_WINDOW_S and abs(advance) >= CONTACT_MIN_ADVANCE
                                and lag >= CONTACT_LEAD and covered < CONTACT_FRACTION * abs(advance)):
                            status, blocked = "blocked", m
                            notes.append(f"{m} met resistance while moving: {lag / TICKS_PER_DEG:.1f} deg behind its "
                                         f"goal, covered {covered / TICKS_PER_DEG:.1f} of "
                                         f"{abs(advance) / TICKS_PER_DEG:.1f} deg in {CONTACT_WINDOW_S} s")
                            break
                    if settle_start is not None and m != GRIPPER:
                        if abs(goal[m] - pos[m]) >= 0.9 * COMP_CAP * LEASH:
                            pinned_since[m] = pinned_since.get(m) or now
                            if (now - pinned_since[m] >= PINNED_S
                                    and self._progress(hist[m], now, STALL_WINDOW_S) < STALL_PROGRESS):
                                status, blocked = "blocked", m
                                break
                        else:
                            pinned_since.pop(m, None)
                if status is not None:
                    break
                if settle_start is not None:
                    at_rest = all(self._progress(hist[m], now, 0.15) < 2 for m in motors)  # not just passing through
                    if at_rest and max(abs(pos[m] - target[m]) for m in motors) <= tol:
                        settled += 1
                        if settled >= SETTLE_CYCLES:
                            status = "reached"
                    else:
                        settled = 0
                        if now - settle_start > SETTLE_TIMEOUT_S:
                            status = "settled"
                cycles += 1
                time.sleep(max(0.0, CYCLE_S - (time.monotonic() - now)))
        except ConnectionError as e:
            status = "comm_error"
            notes.append(f"bus error, torque left on: {e}")
        final = {}
        if status not in ("comm_error", "thermal_release"):
            try:
                final = self._read("Present_Position", motors)
                if status == "blocked":  # contact: no moving joint keeps pushing (loaded ones may sag a few ticks)
                    self._write_goals(final)
            except ConnectionError as e:
                notes.append(f"could not read the final position after {status}: {e}")
        if status == "stopped":
            self._held.set()
        if status == "thermal_release":
            notes.append(f"an arm joint stayed >= {RELEASE_C} C: all torque released")
        want = path[-1]
        duration = time.monotonic() - t0
        return MoveReceipt(
            status=status, requested={m: self.api_value(m, want[m]) for m in motors},
            final={m: self.api_value(m, final[m]) for m in final},
            error={m: round(self.api_value(m, final[m]) - self.api_value(m, want[m]), 2) for m in final},
            max_load=max_load, max_temp=max_temp, duration_s=round(duration, 3),
            mean_cycle_ms=round(1000 * duration / max(cycles, 1), 1), blocked=blocked, notes=notes)

    def _ticks_for(self, targets: dict[str, float]) -> dict[str, int]:
        ticks, problems = {}, []
        for m, value in targets.items():
            if m not in MOTORS:
                raise LimitError(f"unknown joint {m!r}; joints are {list(MOTORS)}")
            ticks[m] = self.gripper_tick(value) if m == GRIPPER else self.to_tick(m, value)
            lo, hi = self.bounds[m]
            if not lo <= ticks[m] <= hi:
                unit = "% open" if m == GRIPPER else " deg"
                problems.append(f"{m} {value:g}{unit} is outside [{self.api_value(m, lo)}, {self.api_value(m, hi)}]")
        if problems:
            raise LimitError("; ".join(problems))
        return ticks

    def move_joints(self, targets: dict[str, float], speed_dps: float = 20.0, tol_deg: float = 1.0,
                    timeout_s: float = 30.0) -> MoveReceipt:
        self._require_motion_ok()
        ticks = self._ticks_for(targets)
        if GRIPPER in ticks and self.gripper_relaxed:
            self._recover_gripper()
        with self._motion_slot():
            if any(m in JOINTS for m in ticks):
                try:
                    pos = self._read("Present_Position", list(JOINTS))
                except ConnectionError as e:
                    raise ArmError(f"cannot read the joints, nothing moved (torque unchanged): {e}") from e
                start = np.array([self.to_deg(m, pos[m]) for m in JOINTS])
                end = np.array([self.to_deg(m, ticks.get(m, pos[m])) for m in JOINTS])
                self._check_poses([start + (end - start) * f for f in np.linspace(0, 1, CLEARANCE_SAMPLES)[1:]],
                                      "this joint move", start)
            return self._run([ticks], speed_dps * TICKS_PER_DEG, round(tol_deg * TICKS_PER_DEG), timeout_s)

    def _recover_gripper(self) -> None:
        t = self._read("Present_Temperature", [GRIPPER])[GRIPPER]
        if t >= WARN_C:
            raise ThermalError(f"gripper relaxed after overheating and still {t} C: wait until it is below {WARN_C} C")
        self.gripper_relaxed = False

    def plan_line(self, start_deg, target_mm, jaws_down: bool, multi_seed: bool = True) -> list:
        """IK for every <= 30 mm segment of the straight TCP line, all before any motion; rejects branch jumps and
        poses near the bench. Returns readings (LeRobot deg, JOINTS order) per segment end. Only the first segment
        searches all IK seeds (multi_seed=False: none do, for short corrective steps)."""
        start_mm, _ = self.tcp(start_deg)
        target_mm = np.asarray(target_mm, float)
        if target_mm[2] - self.bench_z < CLEAR_TIP_MM and target_mm[2] < start_mm[2]:
            raise LimitError(f"the target is {target_mm[2] - self.bench_z:.0f} mm above the bench; the TCP must stay "
                             f"at least {CLEAR_TIP_MM:.0f} mm above it")
        n = max(1, math.ceil(np.linalg.norm(target_mm - start_mm) / MAX_SEGMENT_MM))
        seg_mm = np.linalg.norm(target_mm - start_mm) / n
        lo, hi = self.lo_deg + self.offsets, self.hi_deg + self.offsets  # IK works in model angles
        q, plan = np.asarray(start_deg, float) + self.offsets, []
        for k in range(1, n + 1):
            p = (start_mm + (target_mm - start_mm) * k / n) / 1000.0
            try:
                sol = solve_ik(self.kin, q, p, jaws_down, lo, hi, multi_seed=(k == 1 and multi_seed))
            except ValueError:
                try:
                    sol = solve_ik(self.kin, q, p, jaws_down, lo, hi)
                except ValueError as e:
                    raise LimitError(f"segment {k}/{n} toward {np.round(p * 1000, 1).tolist()} mm: {e}") from e
            jump = np.abs(sol.joints_deg - q)
            # The wrist sits ~100 mm from the TCP, so honest position IK turns it ~17 deg per 30 mm, and up to ~27 deg
            # per 20 mm with the shoulder at its back stop (measured); an IK branch flip swings some joint 60+ deg
            # (wrist_roll was seen flipping 265 deg). 45 deg/30 mm, at least 15, separates the two.
            allowed = np.full(5, max(15.0, 45.0 * seg_mm / MAX_SEGMENT_MM))
            allowed[JOINTS.index("wrist_roll")] = 45.0
            if (jump > allowed).any():
                worst = int(np.argmax(jump / allowed))
                raise LimitError(f"segment {k}/{n} would swing {JOINTS[worst]} by {jump[worst]:.0f} deg "
                                 f"(limit {allowed[worst]:.0f}); use move_joints for large reconfigurations")
            plan.append(IKSolution(sol.joints_deg - self.offsets, sol.xyz, sol.jaw_axis))
            q = sol.joints_deg
        checks, prev = [], np.asarray(start_deg, float)
        for s in plan:
            checks += [prev + (s.joints_deg - prev) * f for f in (0.5, 1.0)]
            prev = s.joints_deg
        self._check_poses(checks, "this line", np.asarray(start_deg, float))
        return plan

    def move_to(self, target_mm, jaws_down: bool = False, speed_dps: float = 20.0, tol_mm: float = 10.0,
                timeout_s: float = 30.0, multi_seed: bool = True, joint_tol_deg: float = 1.0) -> CartesianReceipt:
        self._require_motion_ok()
        with self._motion_slot():
            pos = self._read("Present_Position", list(JOINTS))
            plan = self.plan_line([self.to_deg(m, pos[m]) for m in JOINTS], target_mm, jaws_down, multi_seed)
            path = [{m: self.to_tick(m, d) for m, d in zip(JOINTS, sol.joints_deg)} for sol in plan]
            move = self._run(path, speed_dps * TICKS_PER_DEG, round(joint_tol_deg * TICKS_PER_DEG), timeout_s)
        target = [float(v) for v in target_mm]
        planned = [round(v * 1000, 1) for v in plan[-1].xyz]
        if not all(m in move.final for m in JOINTS):
            return CartesianReceipt(move.status, move, target, planned, [], None, len(plan), [])
        final_mm, axis = self.tcp([move.final[m] for m in JOINTS])
        error = round(float(np.linalg.norm(final_mm - target)), 1)
        status = move.status if move.status not in ("reached", "settled") else (
            "reached" if error <= tol_mm else "off_target")
        return CartesianReceipt(
            status=status, move=move, target_mm=target, planned_mm=planned,
            final_mm=[round(float(v), 1) for v in final_mm], tcp_error_mm=error, segments=len(plan),
            jaw_axis=[round(float(v), 3) for v in axis])

    def tcp_now(self) -> np.ndarray:
        self.connect()
        pos = self._read("Present_Position", list(JOINTS))
        return self.tcp([self.to_deg(m, pos[m]) for m in JOINTS])[0]

    @property
    def moving(self) -> bool:
        return self._moving.is_set()

    @contextmanager
    def between_motions(self):
        """Hold the motion slot without moving, yielding False (and holding nothing) if a motion runs: for swapping
        the calibration where no motion can be planning or running with the old one."""
        free = self._motion.acquire(blocking=False)
        try:
            yield free
        finally:
            if free:
                self._motion.release()

    def joints_now(self) -> tuple[list[float], float]:
        """The five arm joints (deg, JOINTS order) and the gripper opening (%) in one bus read."""
        self.connect()
        pos = self._read("Present_Position", list(MOTORS))
        return [self.to_deg(m, pos[m]) for m in JOINTS], self.gripper_pct(pos[GRIPPER])

    def move_by(self, delta_mm, jaws_down: bool = False, speed_dps: float = 20.0, tol_mm: float = 10.0,
                timeout_s: float = 30.0, multi_seed: bool = True, joint_tol_deg: float = 1.0) -> CartesianReceipt:
        self._require_motion_ok()
        return self.move_to(self.tcp_now() + np.asarray(delta_mm, float), jaws_down, speed_dps, tol_mm, timeout_s,
                            multi_seed, joint_tol_deg)

    def set_gripper(self, opening_pct: float, speed_dps: float = 30.0) -> MoveReceipt:
        return self.move_joints({GRIPPER: opening_pct}, speed_dps, tol_deg=2.0)

    def grasp(self, speed_dps: float = 8.0, timeout_s: float = 20.0) -> GraspReceipt:
        """Close slowly until the jaws stall on something, squeeze GRASP_MARGIN ticks past the stall, verify."""
        self._require_motion_ok()
        if self.gripper_relaxed:
            self._recover_gripper()
        with self._motion_slot():
            move = self._run([{GRIPPER: GRIPPER_CLOSED}], speed_dps * TICKS_PER_DEG, round(2 * TICKS_PER_DEG), timeout_s)
            if move.status != "blocked":
                pos = self._read("Present_Position", [GRIPPER])[GRIPPER]
                reason = "closed without stalling: nothing between the jaws" if move.status in ("reached", "settled") \
                    else f"close ended with status {move.status}"
                return GraspReceipt(False, reason, None, None, round(self.gripper_pct(pos), 1), 0, move)
            stall = self._read("Present_Position", [GRIPPER])[GRIPPER]
            hold = max(stall - GRASP_MARGIN, GRIPPER_WINDOW[0])
            self._write_goals({GRIPPER: hold})
            time.sleep(0.4)
            g = self._read("Present_Position", [GRIPPER])[GRIPPER]
            load = abs(self._read("Present_Load", [GRIPPER])[GRIPPER])
        held = g - hold >= GRASP_MIN_GAP and g <= GRASP_MAX_POS
        reason = "stalled on an object and holding" if held else (
            "jaws kept closing after the stall: object slipped or crushed" if g - hold < GRASP_MIN_GAP
            else "stalled while still wide open")
        return GraspReceipt(held, reason, stall, hold, round(self.gripper_pct(g), 1), load, move)

    def park(self, speed_dps: float = 15.0) -> MoveReceipt:
        if not self.rest_pose_deg:
            raise ArmError("no rest pose configured yet (set one at the lab: a low-load pose to cool down in)")
        return self.move_joints(self.rest_pose_deg, speed_dps)
