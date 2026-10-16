"""Rig calibration from the arm's own motion: joint zero offsets, camera pose and bench plane, one versioned asset.

LeRobot's motor calibration puts each joint's zero at whatever pose the arm was held in while calibrating; the URDF's
zero is a fixed geometric pose. On this arm the two differ by tens of degrees (first measured 2026-10-08: the raw
readings put the base 121 mm under the bench), and no camera calibration can absorb that. So one fit solves, together,
for model angle = reading + offset on the joints that shape the arm, the camera-from-base transform, and where the
tracked marker sits on the gripper. The pan offset is only a camera yaw and the roll offset only spins the marker
about the roll axis, so neither is observable from a marker at the jaw tips; both stay 0.

The asset keeps its raw samples, so it can be refitted later, and states what it depends on: the motor calibration
(by hash: re-calibrating the motors invalidates the offsets) and the camera (by serial and time: moving the camera
invalidates the camera pose, not the offsets).

Accuracy limit: the marker is measured on its visible surface, a few mm in front of the point it stands for. Much
of that bias is constant in the camera frame, so the fit absorbs it into the camera translation, and every other
camera point converted to the base frame inherits it (simulation, 10 mm ball marker: 7 mm). Base-frame positions of
seen objects are therefore good to about +-10 mm; servo_to, which compares camera measurements with each other,
is the way to the last few millimetres.
"""

import contextlib
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from kinematics import JOINTS, Kinematics, rotation

FIT_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
RIG_DIR = Path(__file__).parent / "calibration"
RIG_FILE = RIG_DIR / "rig.json"


@dataclass
class Sample:
    readings_deg: list[float]  # the five arm joints, LeRobot degrees, JOINTS order
    marker_cam_mm: list[float]
    px: list[float]
    area: int


def kabsch(P, Q) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising |Q - (R P + t)| over rows."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def align_z(n) -> np.ndarray:
    """A rotation taking +z to the unit vector n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    axis = np.cross([0.0, 0.0, 1.0], n)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3) if n[2] > 0 else np.diag([1.0, -1.0, -1.0])
    return rotation(axis / s, np.arctan2(s, n[2]))


def rotvec(R) -> np.ndarray:
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if angle < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return w * angle


def gripper_poses(kin: Kinematics, readings_deg, offsets_deg) -> list[np.ndarray]:
    """Base-from-gripper_frame transforms (mm) for every reading row."""
    out = []
    for q in np.atleast_2d(readings_deg):
        T = kin.fk(np.radians(np.asarray(q, float) + offsets_deg))
        T[:3, 3] *= 1000.0
        out.append(T)
    return out


def handeye(Ts, p_cam, iters=200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternating least squares for p_cam = R (T g) + t with fixed kinematics: (R, t, g)."""
    p_cam = np.asarray(p_cam, float)
    g = np.zeros(3)
    for _ in range(iters):
        P = np.array([T[:3, :3] @ g + T[:3, 3] for T in Ts])
        R, t = kabsch(P, p_cam)
        A = np.vstack([R @ T[:3, :3] for T in Ts])
        b = np.concatenate([p - t - R @ T[:3, 3] for T, p in zip(Ts, p_cam)])
        g_new = np.linalg.lstsq(A, b, rcond=None)[0]
        done = np.linalg.norm(g_new - g) < 1e-7
        g = g_new
        if done:
            break
    P = np.array([T[:3, :3] @ g + T[:3, 3] for T in Ts])
    R, t = kabsch(P, p_cam)
    return R, t, g


def solve(kin: Kinematics, readings_deg, marker_cam, bench_n_cam=None, offsets0=None, level=True,
          fit_joints=FIT_JOINTS, iters=100, g_fixed=None) -> dict:
    """Levenberg-Marquardt on (rotation, t, g, offsets of fit_joints). level=True holds base z on the bench normal
    (the base stands on the bench), leaving only the camera yaw about it free. g_fixed holds the marker offset (a
    camera-only refit: small motions cannot tell a marker offset from a camera shift)."""
    readings = np.atleast_2d(np.asarray(readings_deg, float))
    p_cam = np.asarray(marker_cam, float)
    idx = [JOINTS.index(j) for j in fit_joints]
    base_offsets = np.zeros(5) if offsets0 is None else np.asarray(offsets0, float).copy()
    poses0 = gripper_poses(kin, readings, base_offsets)
    if g_fixed is None:
        R0, t0, g0 = handeye(poses0, p_cam)
    else:
        g0 = np.asarray(g_fixed, float)
        R0, t0 = kabsch(np.array([T[:3, :3] @ g0 + T[:3, 3] for T in poses0]), p_cam)
    if level:
        A = align_z(bench_n_cam)
        M = A.T @ R0
        rot0 = np.array([np.arctan2(M[1, 0], M[0, 0])])
    else:
        rot0 = rotvec(R0)
    n_rot, n_g = len(rot0), 0 if g_fixed is not None else 3
    x = np.r_[rot0, t0, g0[:n_g], base_offsets[idx]]

    def unpack(x):
        R = A @ rotation([0, 0, 1], x[0]) if level else (rotation(x[:3], np.linalg.norm(x[:3]))
                                                         if np.linalg.norm(x[:3]) > 1e-12 else np.eye(3))
        t = x[n_rot:n_rot + 3]
        g = x[n_rot + 3:n_rot + 6] if n_g else g0
        off = base_offsets.copy()
        off[idx] = x[n_rot + 3 + n_g:]
        return R, t, g, off

    def resid(x):
        R, t, g, off = unpack(x)
        P = np.array([T[:3, :3] @ g + T[:3, 3] for T in gripper_poses(kin, readings, off)])
        return (p_cam - (P @ R.T + t)).ravel()

    r = resid(x)
    lam = 1e-2
    steps = np.r_[np.full(n_rot, 1e-4), np.full(3 + n_g, 0.05), np.full(len(idx), 0.02)]
    for _ in range(iters):
        J = np.column_stack([(resid(x + e) - resid(x - e)) / (2 * e[k])
                             for k, e in enumerate(np.diag(steps))])
        H, grad = J.T @ J, J.T @ r
        improved = False
        for _ in range(10):
            dx = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -grad)
            r_new = resid(x + dx)
            if r_new @ r_new < r @ r:
                x, r, lam, improved = x + dx, r_new, max(lam / 3, 1e-7), True
                break
            lam *= 4
        if not improved or np.linalg.norm(dx) < 1e-6:
            break
    R, t, g, off = unpack(x)
    off = (off + 180.0) % 360.0 - 180.0
    res = r.reshape(-1, 3)
    J = np.column_stack([(resid(x + e) - resid(x - e)) / (2 * e[k]) for k, e in enumerate(np.diag(steps))])
    dof = max(len(r) - len(x), 1)
    cov = np.linalg.pinv(J.T @ J) * float(r @ r) / dof
    sigma = np.zeros(5)
    if idx:
        sigma[idx] = np.sqrt(np.clip(np.diag(cov)[-len(idx):], 0.0, None))
    out = dict(R=R, t=t, g=g, offsets=off, offsets_sigma=sigma, residuals=res,
               rms=float(np.sqrt((res ** 2).sum(1).mean())), max=float(np.linalg.norm(res, axis=1).max()), level=level)
    if bench_n_cam is not None:
        up = R.T @ np.asarray(bench_n_cam, float)
        out["tilt_deg"] = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    return out


def best_fit(kin: Kinematics, readings_deg, marker_cam, bench_n_cam, level=True, starts=None,
             max_marker_mm=80.0) -> dict:
    """solve() from several offset starts; lowest rms wins among physical answers. The shoulder/elbow pair has a
    mirrored solution, and a wrist turned 180 deg can fake the data with a marker ~300 mm out along the gripper, so
    any fit putting the marker further than max_marker_mm from the TCP is discarded (it sits on the jaw tips)."""
    starts = starts if starts is not None else [np.zeros(5), np.array([0.0, -77.0, 91.0, -14.0, 0.0]),
                                                np.array([0.0, -104.0, 90.0, -13.0, 0.0]),
                                                np.array([0.0, -90.0, 90.0, 0.0, 0.0]),
                                                np.array([0.0, 77.0, -91.0, 14.0, 0.0])]
    fits = [solve(kin, readings_deg, marker_cam, bench_n_cam, s, level) for s in starts]
    physical = [f for f in fits if np.linalg.norm(f["g"]) <= max_marker_mm]
    if not physical:
        raise ValueError(f"no fit keeps the marker within {max_marker_mm} mm of the TCP: the samples are not "
                         "of one rigid marker (wrong blob?) or the arm moved without being measured")
    return min(physical, key=lambda f: f["rms"])


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class RigCalibration:
    """What the arm and the camera need to agree on, with provenance. Model angle = LeRobot reading + offset."""

    joint_offsets_deg: dict[str, float]
    camera_from_base_mm: list[list[float]]  # 4x4: p_cam = T @ [p_base, 1]
    marker_in_gripper_mm: list[float]
    bench_normal_cam: list[float]
    bench_d_cam_mm: float  # bench plane n.p = d, camera frame
    bench_z_base_mm: float  # bench height along base z
    tilt_deg: float  # base z vs bench normal (0 when fitted level)
    rms_mm: float
    max_mm: float
    created: str
    robot_id: str
    motor_calibration_sha256: str
    camera_serial: str
    samples: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    version: int = 1

    @property
    def R(self) -> np.ndarray:
        return np.asarray(self.camera_from_base_mm, float)[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.camera_from_base_mm, float)[:3, 3]

    def to_base(self, p_cam) -> np.ndarray:
        return np.linalg.solve(self.R, (np.asarray(p_cam, float) - self.t).T).T

    def to_cam(self, p_base) -> np.ndarray:
        return np.asarray(p_base, float) @ self.R.T + self.t

    def offsets_array(self) -> np.ndarray:
        return np.array([self.joint_offsets_deg.get(m, 0.0) for m in JOINTS])

    def camera_moved(self, bench_n_cam, bench_d_cam, max_deg=2.0, max_mm=10.0) -> str | None:
        """Compare the bench plane seen now with the one at calibration time: a reason string when the camera has
        evidently moved (its pose, and so every base-frame coordinate derived from it, is stale), else None."""
        n0 = np.asarray(self.bench_normal_cam, float)
        n = np.asarray(bench_n_cam, float) / np.linalg.norm(bench_n_cam)
        angle = float(np.degrees(np.arccos(np.clip(n @ n0, -1.0, 1.0))))
        shift = abs(float(bench_d_cam) - self.bench_d_cam_mm)
        if angle > max_deg or shift > max_mm:
            return (f"the bench looks {angle:.1f} deg / {shift:.0f} mm different from calibration time "
                    f"({self.created}): the camera moved; re-run calibrate_camera")
        return None

    def save(self, path=RIG_FILE, archive=True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(asdict(self), indent=1)
        path.write_text(text, encoding="utf-8")
        if archive:
            stamp = self.created.replace(":", "").replace("-", "")[:15]
            (path.parent / f"rig-{stamp}.json").write_text(text, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path=RIG_FILE) -> "RigCalibration | None":
        path = Path(path)
        if not path.is_file():
            return None
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def make_rig(fit: dict, samples: list[Sample], bench_n_cam, bench_d_cam, robot_id: str, motor_calibration_sha256: str,
             camera_serial: str, notes=()) -> RigCalibration:
    R, t = fit["R"], fit["t"]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    n = np.asarray(bench_n_cam, float)
    up_base = R.T @ n
    # bench points satisfy n.p_cam = d; with p_cam = R p_base + t the bench height along base z (level base):
    bench_z = float((bench_d_cam - n @ t) / up_base[2])
    return RigCalibration(
        joint_offsets_deg={m: round(float(v), 3) for m, v in zip(JOINTS, fit["offsets"])},
        camera_from_base_mm=np.round(T, 6).tolist(), marker_in_gripper_mm=np.round(fit["g"], 2).tolist(),
        bench_normal_cam=np.round(n, 6).tolist(), bench_d_cam_mm=round(float(bench_d_cam), 3), bench_z_base_mm=round(bench_z, 2),
        tilt_deg=round(float(fit.get("tilt_deg", 0.0)), 3), rms_mm=round(fit["rms"], 2), max_mm=round(fit["max"], 2),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"), robot_id=robot_id,
        motor_calibration_sha256=motor_calibration_sha256, camera_serial=camera_serial,
        samples=[asdict(s) for s in samples], notes=list(notes))


def solve_camera(kin: Kinematics, readings_deg, marker_cam, bench_n_cam, bench_d_cam, bench_z_mm, offsets_deg, g_mm,
                 fit_yaw=True, yaw_rad=0.0, sigma_prior_mm=5.0) -> dict:
    """Camera-only fit with the arm's model held (joint offsets, marker) and its bench height pinned: the bench plane
    the camera sees fixes the camera's tilt and height, leaving its yaw about the bench normal and its sideways shift,
    a closed-form 2-D fit (fit_yaw=False: the shift alone, at yaw_rad). The bench height is the arm's own and must not
    move with the camera: it sets the bench guard. Residuals over max(3x the median, 12 mm) are dropped (twice at
    most); dropping over a quarter is an error. sigma_yaw_deg: how firmly the samples' sideways spread pins the yaw."""
    n = np.asarray(bench_n_cam, float) / np.linalg.norm(bench_n_cam)
    A = align_z(n)
    u = A[:, :2]  # images of base x and y: the bench plane's directions in the camera frame
    P = np.array([T[:3, :3] @ np.asarray(g_mm, float) + T[:3, 3] for T in gripper_poses(kin, readings_deg, offsets_deg)])
    cams = np.asarray(marker_cam, float)
    t_normal = (float(bench_d_cam) - float(bench_z_mm)) * n
    Q = cams @ u
    keep = np.ones(len(P), bool)
    for _ in range(3):
        Pk, Qk = P[keep, :2], Q[keep]
        yaw = yaw_rad
        if fit_yaw:
            H = (Pk - Pk.mean(0)).T @ (Qk - Qk.mean(0))
            yaw = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
        c, s = np.cos(yaw), np.sin(yaw)
        shift = (Qk - Pk @ np.array([[c, s], [-s, c]])).mean(0)
        R, t = A @ rotation([0, 0, 1], yaw), u @ shift + t_normal
        res = cams - (P @ R.T + t)
        norms = np.linalg.norm(res, axis=1)
        rms = float(np.sqrt((norms[keep] ** 2).mean()))
        trimmed = norms <= max(3.0 * float(np.median(norms[keep])), 12.0)  # a median: bad blobs inflate an rms
        if (trimmed == keep).all():
            break
        keep = trimmed
    if keep.sum() < 0.75 * len(P) or keep.sum() < 3:
        raise ValueError(f"only {keep.sum()} of {len(P)} samples agree on one camera pose: wrong blobs, or the camera "
                         "moved during the sweep")
    spread = float(np.sqrt(((P[keep, :2] - P[keep, :2].mean(0)) ** 2).sum()))
    sigma_axis = max(rms, sigma_prior_mm) / np.sqrt(3.0)
    return dict(R=R, t=t, g=np.asarray(g_mm, float), offsets=np.asarray(offsets_deg, float), residuals=res, kept=keep,
                rms=rms, max=float(norms[keep].max()), level=True, tilt_deg=0.0, yaw_rad=yaw,
                sigma_yaw_deg=float(np.degrees(sigma_axis / max(spread, 1e-9))))


CAMERA_SWEEP = (  # deg around the rest pose, where the field sweep saw the tape; pan +-25 and two reaches pin the yaw
    {}, {"shoulder_pan": -25}, {"shoulder_pan": -12}, {"shoulder_pan": 12}, {"shoulder_pan": 25},
    {"shoulder_lift": 12, "elbow_flex": -12}, {"shoulder_lift": 24, "elbow_flex": -28}, {"elbow_flex": -16},
    {"wrist_flex": 15}, {"wrist_flex": -20}, {"shoulder_pan": -20, "shoulder_lift": 18, "elbow_flex": -24},
    {"shoulder_pan": 20, "shoulder_lift": 18, "elbow_flex": -24}, {"wrist_roll": -30}, {},
)


def recalibrate_camera(arm, read_marker, bench_plane, motor_calibration_sha256: str, camera_serial: str,
                       robot_id: str = "exp23_follower", poses=CAMERA_SWEEP, speed_dps=10.0, min_samples=7,
                       say=lambda *_: None) -> RigCalibration:
    """Re-fit the camera pose after the camera moved, keeping the arm's joint offsets, the marker on the gripper and
    the bench height (solve_camera); with no rig yet, the marker too.

    read_marker(prior_cam_mm | None) -> the locked marker's camera position or None; bench_plane() -> vision.Plane
    of the bench as the camera sees it now. Moves the arm through `poses` around the rest pose (each refused by the
    arm itself if it would come near the bench) and back to where it started, unless stop() ended a move.
    """
    start = arm.state().joints_deg
    center = dict(arm.rest_pose_deg or start)
    prev = read_marker(None)
    if prev is None:
        raise ValueError("the marker is not in view: lock it with find_marker first")
    from arm import LimitError

    readings, cams, skipped = [], [], []
    try:
        for off in poses:
            target = {m: center[m] + off.get(m, 0.0) for m in JOINTS}
            try:
                r = arm.move_joints(target, speed_dps)
            except LimitError as e:
                skipped.append(off)
                say(f"pose {off} skipped: {e}")
                continue
            if r.status not in ("reached", "settled"):
                raise RuntimeError(f"calibration move {off} ended {r.status}: {r.notes or r.blocked or ''}")
            q = [r.final[m] for m in JOINTS]
            p = read_marker(prev)
            say(f"pose {off}: {'marker at ' + str(np.round(p, 1).tolist()) if p is not None else 'marker not seen'}")
            if p is not None:
                readings.append(q)
                cams.append(np.asarray(p, float))
                prev = p
    finally:
        if not arm.halted:
            arm.move_joints({m: start[m] for m in JOINTS}, speed_dps)
    if len(cams) < min_samples:
        raise ValueError(f"only {len(cams)} of {len(poses)} poses saw the marker (need {min_samples}); "
                         f"{len(skipped)} were outside the arm's limits: keep the gripper in the camera's view")
    plane = bench_plane()
    if arm.rig is not None:  # the tape and the bench height did not move with the camera
        fit = solve_camera(arm.kin, np.array(readings), np.array(cams), plane.n, plane.d, arm.rig.bench_z_base_mm,
                           arm.offsets, arm.rig.marker_in_gripper_mm)
        note = (f"camera-only recalibration from {fit['kept'].sum()} of {len(cams)} poses, yaw sigma "
                f"{fit['sigma_yaw_deg']:.2f} deg; joint offsets, marker and bench height carried over (rig "
                f"{arm.rig.created})")
    else:
        fit = solve(arm.kin, np.array(readings), np.array(cams), plane.n, offsets0=arm.offsets, level=True,
                    fit_joints=())
        if np.linalg.norm(fit["g"]) > 80.0:
            raise ValueError(f"the fit puts the marker {np.linalg.norm(fit['g']):.0f} mm from the TCP: not one marker")
        note = f"camera-only recalibration from {len(cams)} poses; joint offsets carried over, no rig before it"
    samples = [Sample(list(map(float, q)), list(map(float, p)), [0.0, 0.0], 0) for q, p in zip(readings, cams)]
    return make_rig(fit, samples, plane.n, plane.d, robot_id, motor_calibration_sha256, camera_serial, notes=[note])


# This arm's fit of 2026-10-08 (38-pose sweep): the prior model when an arm has no rig yet (motors recalibrated).
FIELD_OFFSETS_DEG = (0.0, -102.79, 87.19, -12.65, 0.0)
FIELD_MARKER_MM = (1.08, 4.45, -3.0)
# Joint deltas (deg) from the rest pose in three widening stages; each later stage is driven by the model fitted on
# the samples before it. Field-proven 2026-10-08: 38 of these 49 poses saw the tape.
FULL_SWEEP = (
    ({"shoulder_pan": -25}, {"shoulder_pan": -15}, {"shoulder_pan": 15}, {"shoulder_pan": 25},
     {"shoulder_lift": 6, "elbow_flex": -6}, {"shoulder_lift": 12, "elbow_flex": -12}, {"shoulder_lift": 6},
     {"elbow_flex": -16}, {"wrist_flex": 15}, {"wrist_flex": -20}, {"wrist_flex": 25},
     {"shoulder_pan": -15, "shoulder_lift": 8, "elbow_flex": -8}, {"shoulder_pan": 15, "shoulder_lift": 8, "elbow_flex": -8},
     {"shoulder_pan": -20, "wrist_flex": 15}, {"shoulder_pan": 20, "wrist_flex": -15},
     {"shoulder_pan": 10, "elbow_flex": -14, "wrist_flex": 10}, {"wrist_roll": -30}, {"wrist_roll": -60}, {}),
    ({"shoulder_lift": 18, "elbow_flex": -20}, {"shoulder_lift": 24, "elbow_flex": -28},
     {"shoulder_lift": 30, "elbow_flex": -36}, {"shoulder_lift": 24, "elbow_flex": -20, "wrist_flex": -15},
     {"shoulder_pan": -20, "shoulder_lift": 18, "elbow_flex": -24}, {"shoulder_pan": 20, "shoulder_lift": 18, "elbow_flex": -24},
     {"shoulder_pan": -25, "shoulder_lift": 24, "elbow_flex": -30, "wrist_flex": 10},
     {"shoulder_pan": 25, "shoulder_lift": 24, "elbow_flex": -30, "wrist_flex": -10},
     {"elbow_flex": -26, "wrist_flex": 20}, {"shoulder_lift": 12, "elbow_flex": -30, "wrist_flex": 25},
     {"shoulder_pan": -10, "shoulder_lift": 30, "elbow_flex": -30}, {"shoulder_pan": 10, "shoulder_lift": 30, "elbow_flex": -42},
     {"shoulder_lift": 8, "elbow_flex": -4, "wrist_flex": 28}, {}),
    ({"shoulder_pan": -30, "shoulder_lift": 12, "elbow_flex": -18}, {"shoulder_pan": 30, "shoulder_lift": 12, "elbow_flex": -18},
     {"shoulder_pan": -15, "shoulder_lift": 30, "elbow_flex": -40, "wrist_flex": 15},
     {"shoulder_pan": 15, "shoulder_lift": 36, "elbow_flex": -44},
     {"shoulder_pan": -25, "shoulder_lift": 36, "elbow_flex": -48, "wrist_flex": -10},
     {"shoulder_pan": 25, "shoulder_lift": 6, "elbow_flex": -8, "wrist_flex": 26},
     {"shoulder_pan": -25, "shoulder_lift": 6, "elbow_flex": -8, "wrist_flex": 26},
     {"shoulder_pan": -12, "elbow_flex": -20, "wrist_flex": -20}, {"shoulder_pan": 12, "elbow_flex": -20, "wrist_flex": -25},
     {"shoulder_pan": 20, "shoulder_lift": 20, "elbow_flex": -20, "wrist_roll": -45},
     {"shoulder_pan": -20, "shoulder_lift": 20, "elbow_flex": -20, "wrist_roll": -90},
     {"shoulder_pan": -30, "shoulder_lift": 24, "elbow_flex": -36, "wrist_flex": 20},
     {"shoulder_pan": 30, "shoulder_lift": 24, "elbow_flex": -36, "wrist_flex": 20},
     {"shoulder_pan": 18, "shoulder_lift": 14, "elbow_flex": -30, "wrist_flex": -5}, {}),
)
FLOOR_PREDICTED_MM, FLOOR_MEASURED_MM = 45.0, 30.0  # marker height above the bench: skip below / stop below
# Then base-frame TCP points (mm) near the bench, reached with the jaws down by straight lines: where objects are picked
# up. The joint stages keep the jaws forward-down near rest, and a fit from them alone missed these poses by 10-12 mm
# (2026-10-16). Kept 3 cm from that day's check poses, x >= 235 clear of a block left at x <= 208.
BENCH_SWEEP = ((235, -30, 65), (250, -95, 85), (270, -45, 75), (290, -20, 90), (240, 15, 70), (265, 30, 80),
               (280, 55, 95), (295, 35, 110))


def _marker_mm(arm, readings_deg, g_mm) -> np.ndarray:
    T = arm.kin.fk(np.radians(np.asarray(readings_deg, float) + arm.offsets))
    return T[:3, :3] @ np.asarray(g_mm, float) + T[:3, 3] * 1000.0


def leave_one_out_mm(kin: Kinematics, readings_deg, marker_cam, bench_n_cam, fit: dict) -> np.ndarray:
    """How far each sample's marker lands from where a fit on all the other samples predicts it."""
    readings, cams = np.asarray(readings_deg, float), np.asarray(marker_cam, float)
    out = []
    for i in range(len(cams)):
        keep = np.arange(len(cams)) != i
        f = solve(kin, readings[keep], cams[keep], bench_n_cam, fit["offsets"], level=True)
        T = gripper_poses(kin, readings[i], f["offsets"])[0]
        out.append(np.linalg.norm(f["R"] @ (T[:3, :3] @ f["g"] + T[:3, 3]) + f["t"] - cams[i]))
    return np.array(out)


def full_calibration(arm, read_marker, bench_plane, motor_calibration_sha256: str, camera_serial: str,
                     robot_id: str = "exp23_follower", stages=FULL_SWEEP, bench=BENCH_SWEEP, speed_dps=10.0,
                     min_samples=20, max_loo_mm=12.0, say=lambda *_: None) -> RigCalibration:
    """Joint offsets, the marker on the gripper and the camera pose, from a staged sweep around the rest pose.

    read_marker(prior_cam_mm | None, gate_mm) -> camera position of the marker blob nearest the prior (None = the
    locked marker), or None. The first stage tracks the marker from pose to pose (the old camera pose may be stale);
    each later stage looks where the model fitted so far predicts it, and that model also drives the arm's bench
    guard. Then the `bench` points with the jaws down, by straight lines, and 50 mm up again. Start at the rest pose
    with the marker locked there. The arm ends where it started. A fit whose leave-one-out error exceeds max_loo_mm is
    refused, and the arm goes back to the calibration it had."""
    from arm import ArmError, LimitError

    original, start = arm.rig, arm.state().joints_deg
    center = dict(arm.rest_pose_deg or start)
    plane = bench_plane()
    g_prior = arm.rig.marker_in_gripper_mm if arm.rig is not None else FIELD_MARKER_MM
    prev = read_marker(None, 60.0)
    if prev is None:
        raise ValueError("the marker is not in view: lock it with find_marker first")
    if arm.rig is None:
        arm.use_model(FIELD_OFFSETS_DEG, 0.0)
    readings, cams, skipped, model = [], [], 0, None

    def take(q, what: str) -> None:
        """Look for the marker where the model so far puts it (else near the last sighting) and keep the sample."""
        nonlocal prev
        p = None
        if model is not None:
            p = read_marker(model["R"] @ _marker_mm(arm, q, model["g"]) + model["t"], 60.0)
        p = p if p is not None else read_marker(prev, 150.0)
        if p is None:
            say(f"{what}: marker not seen")
            return
        if plane.height(p) < FLOOR_MEASURED_MM:
            raise RuntimeError(f"the marker came {plane.height(p):.0f} mm from the bench at {what}: stopped")
        readings.append(q)
        cams.append(np.asarray(p, float))
        prev = p

    try:
        for k, stage in enumerate(stages):
            for off in stage:
                target = {m: center[m] + off.get(m, 0.0) for m in JOINTS}
                g = model["g"] if model is not None else g_prior
                if _marker_mm(arm, [target[m] for m in JOINTS], g)[2] - arm.bench_z < FLOOR_PREDICTED_MM:
                    skipped += 1
                    continue
                try:
                    r = arm.move_joints(target, speed_dps)
                except LimitError as e:
                    skipped += 1
                    say(f"pose {off} skipped: {e}")
                    continue
                if r.status not in ("reached", "settled"):
                    raise RuntimeError(f"calibration move {off} ended {r.status}: {r.notes or r.blocked or ''}")
                take([r.final[m] for m in JOINTS], f"pose {off}")
            if len(cams) >= 8:
                model = best_fit(arm.kin, np.array(readings), np.array(cams), plane.n, level=True)
                arm.use_rig(make_rig(model, [], plane.n, plane.d, robot_id, motor_calibration_sha256, camera_serial,
                                     notes=[f"provisional, after stage {k + 1} of {len(stages)}"]))
                say(f"stage {k + 1}: {len(cams)} samples, rms {model['rms']:.1f} mm")
        for xyz in bench:
            if xyz[2] - arm.bench_z < FLOOR_PREDICTED_MM:
                skipped += 1
                continue
            try:
                r = arm.move_to(xyz, jaws_down=True, speed_dps=speed_dps)
            except LimitError as e:
                skipped += 1
                say(f"bench point {xyz} skipped: {e}")
                continue
            if r.status not in ("reached", "off_target"):
                raise RuntimeError(f"calibration move to {xyz} ended {r.status}: {r.move.notes or r.move.blocked or ''}")
            take([r.move.final[m] for m in JOINTS], f"bench point {xyz}")
        if bench:
            with contextlib.suppress(ArmError):  # up first: the way back to the start is a joint move
                arm.move_by([0.0, 0.0, 50.0], jaws_down=True, speed_dps=speed_dps)
        if len(cams) < min_samples:
            raise ValueError(f"only {len(cams)} poses saw the marker (need {min_samples}; {skipped} skipped): keep "
                             "the gripper in the camera's view from the rest pose")
        fit = best_fit(arm.kin, np.array(readings), np.array(cams), plane.n, level=True)
        loo = leave_one_out_mm(arm.kin, readings, cams, plane.n, fit)
        loo_rms = float(np.sqrt((loo ** 2).mean()))
        if loo_rms > max_loo_mm:
            raise ValueError(f"the fit predicts held-out poses only to {loo_rms:.1f} mm (limit {max_loo_mm}): wrong "
                             "blobs or a camera that moved during the sweep; the old calibration stays")
        free = solve(arm.kin, readings, cams, plane.n, fit["offsets"], level=False)
    except BaseException:
        arm.use_rig(original)
        if not arm.halted:  # never move again after stop()
            with contextlib.suppress(Exception):  # best effort: the error being raised is the one to report
                arm.move_joints({m: start[m] for m in JOINTS}, speed_dps)
        raise
    samples = [Sample(list(map(float, q)), list(map(float, p)), [0.0, 0.0], 0) for q, p in zip(readings, cams)]
    sigma = np.round(fit["offsets_sigma"][1:4], 2).tolist()
    notes = [f"full calibration from {len(samples)} poses ({skipped} skipped); leave-one-out rms {loo_rms:.1f} mm, "
             f"max {loo.max():.1f}; offset sigma deg {sigma} (lift, elbow, wrist); free-fit base tilt "
             f"{free.get('tilt_deg', 0.0):.2f} deg", "pan and roll offsets are unobservable from a marker at the tips: 0"]
    try:
        back = arm.move_joints({m: start[m] for m in JOINTS}, speed_dps)
        if back.status not in ("reached", "settled"):
            notes.append(f"the return to the start pose ended {back.status}")
    except ArmError as e:
        notes.append(f"did not return to the start pose: {e}")
    return make_rig(fit, samples, plane.n, plane.d, robot_id, motor_calibration_sha256, camera_serial, notes=notes)
