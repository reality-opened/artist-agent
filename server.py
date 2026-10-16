"""MCP server for the SO-101 "artist" arm: typed tools, one receipt per request, no docs needed by the agent.

Run: .venv\\Scripts\\python.exe server.py [--sim] [--port COM11] [--wrist-camera VID:PID | --wrist-index N]
--sim swaps in a simulated arm (stiction, gravity sag, load caps, thermal faults) and a rendered bench with a can.
"""

import argparse
import functools
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import vision
from arm import JOINTS, Arm, ArmError, ArmState, CartesianReceipt, GraspReceipt, MoveReceipt
from autocal import AutoCalibration, AutoCalReport
from calibration import FIELD_OFFSETS_DEG, file_sha256, full_calibration, recalibrate_camera
from servo import ServoReceipt, identify_marker
from servo import servo_to as run_servo

INSTRUCTIONS = """\
Drives a real SO-101 arm (5 joints + gripper) watched by a fixed Orbbec RGB-D camera; a person may be nearby.

Frames and units: joints in degrees, gripper as opening percent (0 closed, 100 open). Base frame = the robot
base, mm, z up: move_to / move_by take base-frame points. Camera frame = the depth camera, mm, x right, y down,
z forward: servo_to takes camera-frame points. Image pixels are in the full 1280x720 frame (look(grid=true) labels
them). The rig calibration (arm_status.rig_calibration) links camera and base, so find_objects, locate_point and
find_marker also give base-frame points that move_to can go to directly (open loop, ~8 mm). If they return a
frame_note saying the camera moved, base-frame points are withheld: run calibrate_camera, or stay in the camera
frame with servo_to, which learns the mapping by moving and gets within ~5 mm. calibrate_rig redoes everything
(joint offsets too) after the motors were recalibrated. A monitor (auto_calibration) compares the camera's view of
the gripper with the calibration whenever the arm rests with the jaws as calibrated (gripper at most 40 % open, wrist
within 30 deg of its rest roll): it notes suspect base-frame points, withholds them when the camera has moved, and
re-fits the camera from those rests by itself.

A session starts from the rest pose: its first tool call (any but stop, release_torque and auto_calibration) first
parks the arm and closes a wide-open gripper to 20 %, wherever the last session left it (arm_status.session_reset).

Every motion blocks until it ends and returns a receipt; read its status before the next step. One motion at a
time (a second one is refused as busy); stop() works during a motion. Moves that would bring the gripper close
to the bench, go past a joint limit, or leave the arm stretched out further than the shoulder can lift back up
are refused before anything moves. Typical reach: arm_status, look,
find_objects (or look + locate_point on a pixel), move_to the object's approach_base_mm, then find_marker and
servo_to for the last few millimetres, small move_by steps while checking with look, grasp; release_torque only
with confirm=true.
"""

REACH_MM = 650.0  # objects further than this from the base (horizontally) are not reported
APPROACH_MM = 60.0
RESET_GRIPPER_PCT = 20.0
NO_RESET = frozenset({"stop", "release_torque", "auto_calibration"})  # tools that never trigger the session-start park


@dataclass
class ObjectReport:
    id: int
    kind: str  # "round_top" (a can, cup, ...) or "object"
    top_center_cam_mm: list[float]
    top_center_base_mm: list[float] | None
    approach_cam_mm: list[float]  # APPROACH_MM above the top centre: a servo_to target
    approach_base_mm: list[float] | None  # the same point for move_to
    height_mm: float
    length_mm: float
    width_mm: float
    radius_mm: float | None
    image_px: list[float]
    bbox_px: list[int]
    mean_rgb: list[int]
    hue: int
    saturation: int


@dataclass
class ObjectsReport:
    objects: list[ObjectReport]
    arm_excluded: bool  # the arm's own body was masked out (needs the rig calibration)
    frame_note: str | None


@dataclass
class PointReport:
    point_cam_mm: list[float]
    point_base_mm: list[float] | None
    height_above_bench_mm: float
    source: str  # "depth", or "bench plane (no depth here)"
    frame_note: str | None


@dataclass
class MarkerReport:
    marker_cam_mm: list[float]
    image_px: list[float]
    area_px: int
    candidates: int  # same-colour blobs in view
    confirmed_by_motion: bool
    marker_base_mm: list[float] | None
    frame_note: str | None


@dataclass
class CalibrationReport:
    status: str
    poses: int
    rms_mm: float
    max_mm: float
    saved: str
    note: str


class Rig:
    """Lazily opened hardware (or simulation) shared by all tools."""

    def __init__(self, args):
        self.args = args
        self._lock = threading.Lock()
        self._arm: Arm | None = None
        self._gemini = None
        self._wrist = None
        self.marker: np.ndarray | None = None
        self.marker_px: tuple[float, float] | None = None
        # min_val 65, not vision's 80: the dark navy tip tape (2026-10-16) has brightness 66-106, so 80 kept ~54-113 px of
        # its ~700 and the camera sweep lost it; 65 gives one ~400 px blob, while 50 also let in a navy bottle cap.
        self.marker_spec = dict(hue=(100, 125), min_sat=70, min_val=65, min_mm=200, max_mm=650)
        self.objects: list[vision.BenchObject] = []
        self.jacobian = None
        self.serial = "SIM" if args.sim else args.gemini_serial
        self.holding = False  # the last grasp held something and the jaws have not opened since
        self.auto = None  # the camera-link monitor (autocal.AutoCalibration), set by build()
        self.reset_note: str | None = None
        self._reset_lock = threading.Lock()
        self._reset_done = False

    @property
    def opened(self) -> bool:
        """The arm and the depth camera are both open (the monitor never opens hardware by itself)."""
        return self._arm is not None and self._gemini is not None

    @property
    def arm(self) -> Arm:
        with self._lock:
            if self._arm is None:
                if self.args.sim:
                    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

                    from sim import FIELD_REST_TICKS, BenchScene, SimBus, SimCamera, sim_rig

                    data = Path(__file__).parent / "tests" / "data"
                    config = SO101FollowerConfig(port="sim", id="exp23_follower", disable_torque_on_disconnect=False,
                                                 calibration_dir=data)
                    cal = SO101Follower(config).calibration
                    bus = SimBus(cal, start_ticks=FIELD_REST_TICKS)
                    scene = BenchScene(spheres=[((160.0, -160.0, 40.0), 28.0, (40, 80, 220))])
                    arm = Arm(bus, cal, rig=sim_rig(scene, data / "exp23_follower.json", FIELD_OFFSETS_DEG),
                              motor_calibration_sha256=file_sha256(data / "exp23_follower.json"))
                    arm.rest_pose_deg = {m: round(arm.to_deg(m, FIELD_REST_TICKS[m]), 2) for m in JOINTS}
                    self._gemini = SimCamera(scene, bus, arm.kin, joint_offsets_deg=FIELD_OFFSETS_DEG)
                    self._arm = arm
                else:
                    self._arm = Arm.open(self.args.port, self.args.robot_id,
                                         rest_pose_deg=dict(zip(JOINTS, self.args.rest_pose)))
            return self._arm

    def session_reset(self) -> None:
        """Once per server process (one agent session), before its first tool that uses the hardware: park, then close
        a wide-open gripper to RESET_GRIPPER_PCT, wherever the last session left the arm. Never opens the jaws (they may
        hold something). A refused or failed park lands in reset_note instead of failing the tool; stop() halts it."""
        with self._reset_lock:
            if self._reset_done:
                return
            arm = self.arm
            arm.connect()  # a busy port reaches the agent as an error, and its next call tries again
            self._reset_done = True
            try:
                parked = arm.park()
                note = f"parked ({parked.status})"
                if parked.status in ("reached", "settled") and arm.joints_now()[1] > RESET_GRIPPER_PCT + 5:
                    closed = arm.set_gripper(RESET_GRIPPER_PCT)
                    note += f", gripper closed to {RESET_GRIPPER_PCT:.0f} % ({closed.status})"
            except (ArmError, ConnectionError, TimeoutError) as e:
                note = f"not parked: {e}"
            self.reset_note = f"session start: {note}"

    def gemini(self):
        if self.args.sim:
            _ = self.arm
            return self._gemini
        with self._lock:
            if self._gemini is None:
                from orbbec_camera import OrbbecCamera, OrbbecCameraConfig

                cam = OrbbecCamera(OrbbecCameraConfig(serial_number=self.args.gemini_serial))
                cam.connect()
                self._gemini = cam
            return self._gemini

    def wrist(self):
        if self.args.sim:
            raise ArmError("the simulation has no wrist camera")
        with self._lock:
            if self._wrist is None:
                from lerobot.cameras.configs import Cv2Rotation
                from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

                index, backend = find_wrist_camera(self.args.wrist_index, self.args.wrist_camera)
                rotation = {"none": Cv2Rotation.NO_ROTATION, "cw": Cv2Rotation.ROTATE_90,
                            "ccw": Cv2Rotation.ROTATE_270}[self.args.wrist_rotate]
                width, height = (640, 480) if rotation == Cv2Rotation.NO_ROTATION else (480, 640)  # LeRobot: output size
                cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=index, fps=30, width=width, height=height,
                                                      rotation=rotation, backend=backend))
                cam.connect()
                self._wrist = cam
            return self._wrist

    def rgbd(self, frames: int = 3) -> tuple[np.ndarray, np.ndarray, dict]:
        """Color (RGB) + depth (mm, per-pixel median of `frames` frames) + intrinsics."""
        cam = self.gemini()
        pairs = [cam.read_rgbd() for _ in range(max(1, frames))]
        depth = pairs[0][1] if len(pairs) == 1 else vision.median_depth([d for _, d in pairs])
        return pairs[-1][0], depth, cam.intrinsics

    def blobs(self, frames: int = 2) -> list[vision.Blob]:
        rgb, depth, K = self.rgbd(frames)
        return vision.color_blobs(rgb, depth, K, **self.marker_spec)

    def observe(self, prior, gate_mm: float = 60.0):
        b = vision.nearest_blob(self.blobs(), self.marker if prior is None else prior, gate_mm=gate_mm)
        if b is not None:
            self.marker, self.marker_px = vision.front_point(b), b.px
        return None if b is None else self.marker

    @staticmethod
    def bench_plane(depth, K) -> vision.Plane:
        v, u = np.nonzero((depth > 200) & (depth < 1600))
        return vision.table_plane(vision.backproject(u, v, depth[v, u], K))

    def to_base(self, points_cam, plane: vision.Plane) -> tuple[list | None, str | None]:
        """Camera points in the base frame, or (None, why) when there is no rig calibration or the camera moved."""
        rig = self.arm.rig
        if rig is None:
            return None, f"no rig calibration ({self.arm.rig_note})"
        moved = rig.camera_moved(plane.n, plane.d)
        if moved:
            return None, moved
        withhold, note = self.auto.base_frame() if self.auto is not None else (False, None)
        if withhold:
            return None, note
        return [np.round(rig.to_base(p), 1).tolist() for p in points_cam], note

    def arm_mask(self):
        """exclude(points_cam) -> True for points on the arm's own body (FK + rig camera pose), or None."""
        rig = self.arm.rig
        if rig is None:
            return None
        body = [(rig.to_cam(a), rig.to_cam(b), r) for a, b, r in self.arm.body_segments()]

        def exclude(points):
            hit = np.zeros(len(points), bool)
            for a, b, radius in body:
                hit |= vision.segment_distance(points, a, b) < radius
            return hit

        return exclude


def find_wrist_camera(index: int | None, which: str):
    """(OpenCV index, backend) of the wrist webcam: --wrist-index if given, else the one camera whose USB VID:PID (hex)
    or name matches `which`. OpenCV numbers cameras in plug-in order, so a fixed index breaks when devices change."""
    from cv2_enumerate_cameras import enumerate_cameras
    from lerobot.cameras.configs import Cv2Backends

    if index is not None:
        return index, Cv2Backends.ANY
    cams = enumerate_cameras(cv2.CAP_MSMF)
    want = which.lower()
    hits = [c for c in cams if want == f"{c.vid or 0:04x}:{c.pid or 0:04x}" or want in c.name.lower()]
    if len(hits) != 1:
        seen = "; ".join(f"{c.name} {c.vid or 0:04X}:{c.pid or 0:04X}" for c in cams) or "none"
        raise ArmError(f"{len(hits)} cameras match the wrist camera {which!r} (connected: {seen}): plug it in, or start "
                       "the server with --wrist-camera VID:PID / --wrist-index N")
    return hits[0].index, Cv2Backends(hits[0].backend)


def draw_grid(image: np.ndarray, scale: float) -> None:
    """Pixel grid labelled in full-frame coordinates (the image may be shown downscaled by `scale`)."""
    h, w = image.shape[:2]
    for x in range(0, int(w / scale), 100):
        cv2.line(image, (int(x * scale), 0), (int(x * scale), h), (255, 255, 255), 1)
        cv2.putText(image, str(x), (int(x * scale) + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    for y in range(0, int(h / scale), 100):
        cv2.line(image, (0, int(y * scale)), (w, int(y * scale)), (255, 255, 255), 1)
        cv2.putText(image, str(y), (2, int(y * scale) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)


def build(args) -> MCPServer:
    rig = Rig(args)
    server = MCPServer("artist-arm", instructions=INSTRUCTIONS)
    auto = AutoCalibration(rig, persist=not args.sim, run_sweep=lambda: calibrate(lambda arm, bench, sha: (
        recalibrate_camera(arm, lambda prior: rig.observe(prior, gate_mm=120.0), bench, sha, rig.serial, args.robot_id))))
    rig.auto = auto

    def tool(fn):
        """Register fn; refusals and device errors reach the agent as readable ToolErrors (MCP masks the rest). Unless
        fn is in NO_RESET, the session-start park runs before it."""

        @functools.wraps(fn)
        def guarded(*a, **kw):
            auto.touch()
            try:
                if fn.__name__ not in NO_RESET:
                    rig.session_reset()
                return fn(*a, **kw)
            except (RuntimeError, ConnectionError, TimeoutError, ValueError) as e:
                hint = " (the monitor's calibrate_camera sweep is running: stop() halts it)" \
                    if auto.state == "sweeping" else ""
                raise ToolError(f"{type(e).__name__}: {e}{hint}") from e

        return server.tool()(guarded)

    @tool
    def arm_status() -> ArmState:
        """Joint angles (deg), gripper opening (%), model TCP (base mm) and its height above the bench, torque,
        temperatures, loads, limits, rig calibration, the camera-link monitor (auto_calibration), safety flags
        (calibration_ok, thermal_lockout, busy), and how the session-start park went (session_reset). Call it first:
        as a session's first call it runs that park; after that it only reads."""
        st = rig.arm.state()
        st.auto_calibration = auto.summary()
        st.session_reset = rig.reset_note
        return st

    @tool
    def move_joints(targets: dict[str, float], speed_deg_s: float = 20.0) -> MoveReceipt:
        """Move joints to absolute targets, e.g. {"elbow_flex": -10, "gripper": 50}. Joints: shoulder_pan,
        shoulder_lift, elbow_flex, wrist_flex, wrist_roll (degrees), gripper (opening %). Out-of-range targets,
        moves passing near the bench and poses too stretched out for the shoulder to lift are refused, never clamped
        (arm_status lists the limits)."""
        return rig.arm.move_joints(targets, speed_deg_s)

    @tool
    def move_to(x_mm: float, y_mm: float, z_mm: float, jaws_down: bool = False, speed_deg_s: float = 20.0,
                tol_mm: float = 10.0) -> CartesianReceipt:
        """Move the gripper TCP along a straight line to a base-frame point (mm). The whole line is solved before
        anything moves; a line that needs an IK branch flip, passes near the bench or ends too stretched out for the
        shoulder to lift is refused. jaws_down turns the jaws toward straight down along the way, without twisting
        them (wrist_roll is kept): the wrist reaches straight down only low over the bench (on this arm from about
        4 cm up, 20-30 cm out), so higher up the jaws stay tilted as near down as it goes, and the receipt's notes
        say by how much (jaw_axis [0, 0, -1] = straight down); descending with jaws_down straightens them.
        status off_target = joints stopped inside their deadband short of the point."""
        return rig.arm.move_to([x_mm, y_mm, z_mm], jaws_down, speed_deg_s, tol_mm)

    @tool
    def move_by(dx_mm: float, dy_mm: float, dz_mm: float, jaws_down: bool = False, speed_deg_s: float = 20.0,
                tol_mm: float = 10.0) -> CartesianReceipt:
        """Move the gripper TCP by a base-frame offset (mm): +z is up, away from the bench. jaws_down as in move_to."""
        return rig.arm.move_by([dx_mm, dy_mm, dz_mm], jaws_down, speed_deg_s, tol_mm)

    @tool
    def set_gripper(opening_pct: float, speed_deg_s: float = 30.0) -> MoveReceipt:
        """Open or close the jaws to an opening percent (0 = empty jaws closed, 100 = fully open)."""
        rig.holding = False
        return rig.arm.set_gripper(opening_pct, speed_deg_s)

    @tool
    def grasp() -> GraspReceipt:
        """Close slowly until the jaws stall on an object, then squeeze a fixed margin past the stall.
        held=false means the jaws closed on nothing or the object slipped."""
        g = rig.arm.grasp()
        rig.holding = g.held
        return g

    @tool
    def stop() -> dict[str, str]:
        """Halt any running motion now and hold position (torque stays on)."""
        return {k: str(v) for k, v in rig.arm.stop().items()}

    @tool
    def release_torque(which: Literal["gripper", "all"] = "gripper", confirm: bool = False) -> dict[str, str]:
        """Turn torque off. 'all' lets the arm fall, 'gripper' drops what the jaws hold: requires confirm=true."""
        out = {k: str(v) for k, v in rig.arm.release(which, confirm).items()}
        rig.holding = False
        return out

    @tool
    def park() -> MoveReceipt:
        """Move to the configured low-load rest pose (to cool down or finish)."""
        return rig.arm.park()

    @tool
    def look(camera: Literal["gemini", "wrist"] = "gemini", annotate: bool = True, grid: bool = False) -> Image:
        """A JPEG from the depth camera (gemini) or the wrist camera. annotate: the tracked marker (magenta) and the
        last find_objects results (yellow, with ids). grid: pixel grid labelled in full-frame coordinates, for
        picking a pixel to pass to locate_point."""
        if camera == "wrist":
            image = rig.wrist().read_latest().copy()
            scale = 1.0
        else:
            image, _, K = rig.rgbd(frames=1)
            image = image.copy()
            if annotate and rig.marker is not None:
                u, v = vision.project(rig.marker, K)[0]
                cv2.circle(image, (int(u), int(v)), 18, (255, 0, 255), 3)
                cv2.putText(image, "marker", (int(u) + 22, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            if annotate:
                for k, obj in enumerate(rig.objects):
                    x, y, w, h = obj.bbox
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(image, str(k), (x + 4, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            scale = min(1.0, 960 / image.shape[1])
        if scale < 1.0:
            image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        if grid:
            draw_grid(image, scale)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ArmError("JPEG encoding failed")
        return Image(data=jpg.tobytes(), format="jpeg")

    @tool
    def find_objects(min_height_mm: float = 8.0, max_height_mm: float = 300.0) -> ObjectsReport:
        """Everything standing on the bench within reach, largest first: top centre (camera and base frame), an
        approach point above it for move_to, height, footprint, radius when the top is round, image box and colour.
        The arm's own body is masked out when the rig calibration is valid. look() then draws the ids."""
        rgb, depth, K = rig.rgbd(frames=5)
        plane = rig.bench_plane(depth, K)
        base_ok = rig.to_base([np.zeros(3)], plane)[0] is not None
        objs, _ = vision.find_objects(rgb, depth, K, plane, min_height_mm, max_height_mm,
                                      exclude=rig.arm_mask() if base_ok else None)
        reports, kept = [], []
        for obj in objs:
            top_base, note = rig.to_base([obj.top_center], plane)
            if top_base is not None and np.hypot(*top_base[0][:2]) > REACH_MM:
                continue
            approach_cam = obj.top_center + APPROACH_MM * plane.n
            approach, _ = rig.to_base([approach_cam], plane)
            kept.append(obj)
            reports.append(ObjectReport(
                id=len(reports), kind="round_top" if obj.radius_mm else "object",
                top_center_cam_mm=np.round(obj.top_center, 1).tolist(),
                top_center_base_mm=None if top_base is None else top_base[0],
                approach_cam_mm=np.round(approach_cam, 1).tolist(),
                approach_base_mm=None if approach is None else approach[0], height_mm=round(obj.height_mm, 1),
                length_mm=round(obj.length_mm, 1), width_mm=round(obj.width_mm, 1),
                radius_mm=None if obj.radius_mm is None else round(obj.radius_mm, 1),
                image_px=[round(v, 1) for v in obj.px], bbox_px=list(obj.bbox), mean_rgb=list(obj.mean_rgb),
                hue=obj.hue, saturation=obj.saturation))
        rig.objects = kept
        note = None if base_ok else rig.to_base([np.zeros(3)], plane)[1]
        return ObjectsReport(reports, base_ok, note)

    @tool
    def locate_point(u_px: float, v_px: float) -> PointReport:
        """The 3D point behind an image pixel (full 1280x720 frame, see look(grid=true)), in camera and base frame,
        and its height above the bench. Without depth at that pixel the ray is intersected with the bench plane."""
        _, depth, K = rig.rgbd(frames=5)
        plane = rig.bench_plane(depth, K)
        p, source = vision.pixel_point(depth, K, u_px, v_px, plane=plane)
        if p is None:
            raise ArmError(source)
        base, note = rig.to_base([p], plane)
        return PointReport(np.round(p, 1).tolist(), None if base is None else base[0],
                           round(float(plane.height(p)), 1), source, note)

    @tool
    def find_marker(hue_low: int = 100, hue_high: int = 125, min_saturation: int = 70, near_mm: float = 200.0,
                    far_mm: float = 650.0, confirm_by_motion: bool = True, min_value: int = 65) -> MarkerReport:
        """Lock onto the coloured tape on the gripper (OpenCV hue 0..179; default blue tape; min_value = brightness
        floor 0..255, lower it for dark tape or a dim room). With
        confirm_by_motion the arm lifts 15 mm and returns, and only the blob that moved with it is accepted, so a
        same-coloured object elsewhere cannot be mistaken for the gripper."""
        rig.marker_spec = dict(hue=(hue_low, hue_high), min_sat=min_saturation, min_val=min_value, min_mm=near_mm,
                               max_mm=far_mm)
        blobs = rig.blobs()
        if not blobs:
            raise ArmError("no blob of that colour within the depth window: check look() and the hue band")
        if confirm_by_motion:
            lock = identify_marker(rig.arm, lambda: [b.p for b in rig.blobs()])
            if lock is None:
                raise ArmError("no single blob moved with the arm: the tape may be hidden, or the arm did not move")
            b = vision.nearest_blob(rig.blobs(), lock, gate_mm=60)
            if b is None:
                raise ArmError("lost the marker after the confirming motion")
        else:
            b = vision.nearest_blob(blobs, rig.marker)
        rig.marker, rig.marker_px, rig.jacobian = vision.front_point(b), b.px, None
        _, depth, K = rig.rgbd(frames=3)
        base, note = rig.to_base([rig.marker], rig.bench_plane(depth, K))
        return MarkerReport(np.round(rig.marker, 1).tolist(), [round(v, 1) for v in b.px], b.area, len(blobs),
                            confirm_by_motion, None if base is None else base[0], note)

    @tool
    def servo_to(x_cam_mm: float, y_cam_mm: float, z_cam_mm: float, tol_mm: float = 5.0, max_seconds: float = 40.0,
                 jaws_down: bool = False) -> ServoReceipt:
        """Closed loop: move the arm until the locked marker (find_marker first) sits at a camera-frame point.
        The first call probes +-20 mm per axis to learn how the arm moves in the image; later calls reuse that.
        status budget = time ran out, call again; resolution_floor = stiction stops further gains."""
        if rig.marker is None:
            raise ArmError("no marker locked: call find_marker first")
        start_j, source = rig.jacobian, "learned on an earlier call"
        if start_j is None and rig.arm.rig is not None:
            _, depth, K = rig.rgbd(frames=3)
            plane = rig.bench_plane(depth, K)
            if rig.arm.rig.camera_moved(plane.n, plane.d) is None and auto.trusts_rig():
                start_j, source = rig.arm.rig.R.tolist(), "the rig calibration (no probing needed)"
        receipt = run_servo(rig.arm, rig.observe, [x_cam_mm, y_cam_mm, z_cam_mm], tol_mm=tol_mm,
                            budget_s=max_seconds, jaws_down=jaws_down, jacobian=start_j)
        if start_j is not None:
            receipt.notes.insert(0, f"started from the Jacobian of {source}")
        if receipt.jacobian is not None and receipt.status not in ("jacobian_unreliable", "tracking_lost"):
            rig.jacobian = receipt.jacobian
        return receipt

    @tool
    def calibrate_camera(confirm: bool = False) -> CalibrationReport:
        """Re-measure the camera pose after the camera was moved or bumped (frame_note says so). Parks, closes a
        wide-open gripper to 20 %, locks the marker by moving it, then moves the arm through ~14 poses around the rest
        pose (~1.5 min); keeps the arm's
        joint offsets, the marker on the gripper and the bench height, saves a new versioned rig calibration and uses
        it at once. The arm ends parked. Requires confirm=true."""
        if not confirm:
            raise ArmError("calibrate_camera moves the arm through ~14 poses for about 1.5 minutes: call again with "
                           "confirm=true once the space around the arm is clear")
        return calibrate(lambda arm, bench, sha: recalibrate_camera(
            arm, lambda prior: rig.observe(prior, gate_mm=120.0), bench, sha, rig.serial, args.robot_id))

    @tool
    def calibrate_rig(confirm: bool = False) -> CalibrationReport:
        """Full calibration: the arm's joint offsets, the marker on the gripper and the camera pose, from ~45 poses in
        three widening stages around the rest pose plus 8 with the jaws down near the bench, 6-11 cm over it and
        23-30 cm out (~6 min). Needed after the motors were recalibrated or the marker
        tape moved; when only the camera moved, calibrate_camera is quicker. Parks, closes a wide-open gripper to 20 %,
        locks the marker by moving it, sweeps, and refuses a fit that predicts held-out poses worse than 12 mm (the old
        calibration then stays).
        Saves a versioned rig calibration and uses it at once. Requires confirm=true and a clear space around the arm."""
        if not confirm:
            raise ArmError("calibrate_rig moves the arm through ~45 poses for about 5 minutes: call again with "
                           "confirm=true once the space around the arm is clear")
        return calibrate(lambda arm, bench, sha: full_calibration(
            arm, lambda prior, gate: rig.observe(prior, gate_mm=gate), bench, sha, rig.serial, args.robot_id))

    def calibrate(sweep) -> CalibrationReport:
        """Park, lock the marker by moving it (in the agent's colour band), sweep(arm, bench, motor_sha) -> new rig;
        save it (versioned; a temp file in simulation) and use it at once. The monitor holds still meanwhile."""
        arm = rig.arm
        was_paused, auto.paused = auto.paused, True
        try:
            parked = arm.park()
            if parked.status not in ("reached", "settled"):
                raise ArmError(f"could not park before calibrating: {parked.status} {parked.notes}")
            if not rig.holding and arm.joints_now()[1] > RESET_GRIPPER_PCT + 5:
                arm.set_gripper(RESET_GRIPPER_PCT)  # the tape is calibrated with the jaws nearly closed
            spec = rig.marker_spec
            find_marker(spec["hue"][0], spec["hue"][1], spec["min_sat"], spec["min_mm"], spec["max_mm"],
                        min_value=spec["min_val"])

            def bench():
                _, depth, K = rig.rgbd(frames=5)
                return rig.bench_plane(depth, K)

            new = sweep(arm, bench, arm.motor_calibration_sha256 or "unknown")
            path = new.save(Path(tempfile.gettempdir()) / "artist-agent-sim-rig.json" if args.sim else
                            Path(__file__).parent / "calibration" / "rig.json", archive=not args.sim)
            arm.use_rig(new)
            rig.jacobian = None
        finally:
            auto.paused = was_paused
        auto.reset(f"calibrated by a tool: {new.notes[0]}")
        return CalibrationReport("calibrated", len(new.samples), new.rms_mm, new.max_mm, str(path), new.notes[0])

    @tool
    def auto_calibration(enabled: bool | None = None, period_s: float | None = None,
                         sweep: bool | None = None) -> AutoCalReport:
        """The camera-link monitor (on by default; it never moves the arm unless sweep is armed). Whenever the arm
        rests somewhere new it compares where the camera sees the gripper marker with where the calibration puts it:
        state ok, suspect (base-frame points carry a note) or stale (the camera moved: base-frame points withheld).
        Every period_s (default 60) it re-solves the camera pose from those rests, camera only, and adopts it when
        clearly better. sweep=true lets it run calibrate_camera by itself when stale, after 2 min without tool calls,
        with empty jaws. Omitted arguments keep their value; returns the status."""
        return auto.configure(enabled, period_s, sweep)

    auto.start()
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", action="store_true", help="simulated arm + rendered bench, no hardware")
    ap.add_argument("--port", default="COM11")
    ap.add_argument("--robot-id", default="exp23_follower")
    ap.add_argument("--gemini-serial", default="AY35C3200SX")
    ap.add_argument("--wrist-camera", default="0bda:5844",
                    help="wrist webcam by USB VID:PID (hex) or part of its name; this arm's is 0bda:5844 'USB Camera'")
    ap.add_argument("--wrist-index", type=int, default=None, help="OpenCV index of the wrist webcam (overrides)")
    ap.add_argument("--wrist-rotate", choices=("none", "cw", "ccw"), default="cw")
    ap.add_argument("--rest-pose", type=float, nargs=5, default=[6.95, 0.75, -9.5, 60.4, 162.0],
                    metavar=("PAN", "LIFT", "ELBOW", "WRIST", "ROLL"), help="park pose, LeRobot degrees")
    build(ap.parse_args()).run("stdio")


if __name__ == "__main__":
    main()
