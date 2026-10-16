"""Keeps the camera<->arm link honest while the agent works, without moving the arm for it.

Whenever the arm rests somewhere new, the monitor looks at the gripper marker and compares it with where the rig
calibration predicts it. One look can be fooled (an object in the jaws, a same-coloured thing nearby, the arm still
ringing), so a verdict needs distinct poses: the camera has moved when the disagreements of several poses are one rigid
motion, or when one is beyond any innocent cause. States: ok; suspect (base-frame points still given, with a note);
stale (withheld until fixed). Every period_s the camera pose is re-solved from the rests by calibration.solve_camera,
holding the arm's joint offsets, the marker and the bench height: the sideways shift always, the yaw too once the rests
are spread enough to pin it; the new pose is adopted only if it explains the rests clearly better, and never while a
motion runs. The calibrate_camera sweep runs by itself only if the agent armed it, the link is stale, no tool was called
for SWEEP_IDLE_S, the jaws hold nothing and the gripper is well clear of the bench. Design reviewed 2026-10-08.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np

import vision
from calibration import Sample, align_z, gripper_poses, kabsch, make_rig, solve_camera
from kinematics import JOINTS

CELL_MM, MAX_RESTS, MAX_AGE_S = 40.0, 30, 1800.0  # one rest per 40 mm cell of marker positions, newest wins
GATE_MM, EDGE_PX, STILL_DEG = 60.0, 40.0, 0.2
SUSPECT_MM, STALE_MM, RECOVER_MM = 15.0, 40.0, 10.0  # field fit: rms 4.5, LOO 5.2, max ~10-15 mm
RIGID_FIT_MM, RIGID_SHIFT_MM, RIGID_TURN_DEG = 8.0, 12.0, 1.0
ADOPT_GAIN, ADOPT_SHIFT_MM, ADOPT_RMS_MM, ADOPT_MAX_MM, YAW_SIGMA_DEG = 0.7, 4.0, 8.0, 20.0, 0.5
SWEEP_IDLE_S, SWEEP_GAP_S, SWEEP_ATTEMPTS, SWEEP_CLEAR_MM = 120.0, 600.0, 2, 100.0
RECHECK_S = 300.0  # look again at an unchanged pose this often: a camera bumped while the arm sits still
# Looks only with the jaws as calibrated (tape on the fixed jaw, wrist near its rest roll): on 2026-10-16 an open gripper
# showed the other jaw's tape ~45 mm off and a wrist rolled 94 deg showed another face of the tape 15-25 mm off, and the
# monitor called the camera moved.
LOOK_MAX_GRIP_PCT, LOOK_ROLL_SPAN_DEG = 40.0, 30.0


@dataclass
class Rest:
    t: float
    epoch: int
    readings: list[float]  # five joints, deg
    gripper_pct: float
    base: np.ndarray  # marker by forward kinematics, base mm
    seen: np.ndarray  # marker by the camera, camera mm
    far: bool = False  # the only marker-coloured blob, but nowhere near the prediction: no verdict on its own


@dataclass
class AutoCalReport:
    enabled: bool
    state: str  # ok | suspect | stale | waiting (hardware not opened or no rig yet) | sweeping | off
    period_s: float
    sweep_armed: bool
    distinct_rests: int
    last_miss_mm: float | None  # how far the camera saw the marker from its predicted place, last look
    last_refit: str | None
    note: str | None


class AutoCalibration:
    def __init__(self, rig, period_s: float = 60.0, enabled: bool = True, sweep: bool = False, poll_s: float = 1.0,
                 persist: bool = True, run_sweep=None, clock=time.monotonic):
        self.rig, self.clock = rig, clock  # the server's Rig: arm, rgbd(), bench_plane(), marker_spec, jacobian
        self.period_s, self.enabled, self.sweep_armed, self.poll_s = period_s, enabled, sweep, poll_s
        self.persist, self.run_sweep = persist, run_sweep
        self.state, self.note, self.last_refit, self.last_miss = "waiting", None, None, None
        self.rests: dict[tuple, Rest] = {}
        self.paused = False
        self.epoch = 0
        self._lock = threading.RLock()
        now = clock()
        self._last_pose, self._last_grip, self._last_look, self._last_eval, self._last_tool = None, None, -np.inf, now, now
        self._sweeps: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- agent surface -----------------------------------------------------------------------------------------
    def configure(self, enabled=None, period_s=None, sweep=None) -> AutoCalReport:
        with self._lock:
            if enabled is not None:
                self.enabled = bool(enabled)
            if period_s is not None:
                if period_s < 10:
                    raise ValueError("period_s must be at least 10 s")
                self.period_s = float(period_s)
            if sweep is not None:
                self.sweep_armed, self._sweeps = bool(sweep), []
            return self.report()

    def report(self) -> AutoCalReport:
        with self._lock:
            state = "off" if not self.enabled else self.state
            return AutoCalReport(self.enabled, state, self.period_s, self.sweep_armed, len(self.rests),
                                 None if self.last_miss is None else round(self.last_miss, 1), self.last_refit, self.note)

    def summary(self) -> str:
        r = self.report()
        miss = "" if r.last_miss_mm is None else f", last look {r.last_miss_mm} mm off"
        return f"{r.state}{miss}; {r.distinct_rests} rests" + (f"; {r.note}" if r.note else "")

    def touch(self) -> None:
        """Any tool call: the agent is busy, so no sweep of its own."""
        self._last_tool = self.clock()

    def base_frame(self) -> tuple[bool, str | None]:
        """(withhold base-frame points?, note for frame_note)."""
        with self._lock:
            if not self.enabled or self.state in ("ok", "waiting"):
                return False, None
            if self.state == "suspect":
                return False, f"suspect calibration: {self.note}"
            return True, (f"the camera moved: {self.note}. Base-frame points withheld; keep working (each new rest "
                          "helps re-fit it, every period_s), use servo_to in the camera frame, or run calibrate_camera")

    def trusts_rig(self) -> bool:
        with self._lock:
            return not self.enabled or self.state in ("ok", "waiting")

    def reset(self, why: str) -> None:
        """A calibration was just made by a tool: start afresh from it."""
        with self._lock:
            self.rests.clear()
            self.epoch += 1
            self.state, self.note, self.last_miss = "ok", None, None
            self.last_refit = f"{time.strftime('%H:%M:%S')} {why}"

    # --- loop ----------------------------------------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="autocal", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.poll()
            except Exception as e:  # keep watching; the agent sees why in the status
                self.note = f"monitor error: {type(e).__name__}: {e}"

    def poll(self, now: float | None = None) -> str:
        """One look-and-decide step; returns what it did (for tests and logs)."""
        now = self.clock() if now is None else now
        if not self.enabled or self.paused:
            return "off"
        if not self.rig.opened:
            return "waiting"
        arm = self.rig.arm
        if arm.rig is None:
            self.state = "waiting"
            return "no rig"
        if self.state == "waiting":
            self.state = "ok"
        if arm.moving:
            return "moving"
        q, grip = arm.joints_now()
        moved = (self._last_pose is None or np.abs(np.subtract(q, self._last_pose)).max() > 2.0
                 or abs(grip - self._last_grip) > 5.0)
        if moved or now - self._last_look >= RECHECK_S:
            self._last_pose, self._last_grip, self._last_look = q, grip, now
            what = self._look(arm, q, grip, now)
        else:
            what = "same pose"
        if now - self._last_eval >= self.period_s:
            self._last_eval = now
            what += "; " + self._evaluate(arm, now)
        self._maybe_sweep(arm, now)
        return what

    # --- one look ----------------------------------------------------------------------------------------------
    def _look(self, arm, q, grip, now) -> str:
        if grip > LOOK_MAX_GRIP_PCT:
            return f"no look: the gripper is open past {LOOK_MAX_GRIP_PCT:.0f} %"
        rest_roll = (arm.rest_pose_deg or {}).get("wrist_roll")
        if rest_roll is not None and abs(q[JOINTS.index("wrist_roll")] - rest_roll) > LOOK_ROLL_SPAN_DEG:
            return f"no look: the wrist is rolled more than {LOOK_ROLL_SPAN_DEG:.0f} deg from rest"
        cal = arm.rig
        T = gripper_poses(arm.kin, q, arm.offsets)[0]
        base = T[:3, :3] @ np.asarray(cal.marker_in_gripper_mm, float) + T[:3, 3]
        pred = cal.to_cam(base)
        rgb, depth, K = self.rig.rgbd(frames=2)
        spec = self.rig.marker_spec
        u, v = K["fx"] * pred[0] / pred[2] + K["cx"], K["fy"] * pred[1] / pred[2] + K["cy"]
        h, w = depth.shape[:2]
        if not (EDGE_PX <= u <= w - EDGE_PX and EDGE_PX <= v <= h - EDGE_PX and spec["min_mm"] <= pred[2] <= spec["max_mm"]):
            return "marker out of view"
        blobs = vision.color_blobs(rgb, depth, K, **spec)
        near = [b for b in blobs if np.linalg.norm(b.p - pred) < 2 * GATE_MM]
        if len(near) > 1:
            return "ambiguous: several marker-coloured blobs near the prediction"
        b = near[0] if near else (blobs[0] if len(blobs) == 1 else None)  # far but alone: the camera may have moved a lot
        if b is None:
            return "marker not seen"
        q2, _ = arm.joints_now()
        if arm.moving or np.abs(np.subtract(q2, q)).max() > STILL_DEG:
            return "arm moved during the look"
        rest = Rest(now, self.epoch, list(map(float, q)), float(grip), base, vision.front_point(b), far=not near)
        with self._lock:
            self.rests[tuple(np.floor(base / CELL_MM).astype(int))] = rest
            if len(self.rests) > MAX_RESTS:
                del self.rests[min(self.rests, key=lambda k: self.rests[k].t)]
            self.last_miss = float(np.linalg.norm(rest.seen - pred))
            self._classify(cal, now)
        return f"looked: {self.last_miss:.1f} mm off"

    def _fresh(self, now, epoch_only) -> list[Rest]:
        return [r for r in self.rests.values() if now - r.t < MAX_AGE_S and (not epoch_only or r.epoch == self.epoch)]

    def _classify(self, cal, now) -> None:
        miss = self.last_miss
        latest = max(self.rests.values(), key=lambda r: r.t)
        if self.state == "ok" and miss > max(SUSPECT_MM, 1.5 * cal.max_mm):
            self.epoch += 1  # rests from before this look may be of another camera pose
            latest.epoch = self.epoch
            self.state, self.note = "suspect", f"the camera sees the gripper {miss:.0f} mm from where the arm puts it"
        if self.state != "stale" and miss > STALE_MM and not latest.far:
            self._go_stale(f"the camera sees the gripper {miss:.0f} mm from where the arm puts it")
            return
        rests = self._fresh(now, epoch_only=self.state != "ok")
        if self.state == "suspect" and len(rests) >= 3:
            pred = np.array([cal.to_cam(r.base) for r in rests])
            seen = np.array([r.seen for r in rests])
            R, t = kabsch(pred, seen)
            fitted = pred @ R.T + t
            post = float(np.sqrt(((seen - fitted) ** 2).sum(1).mean()))
            shift = float(np.linalg.norm(fitted - pred, axis=1).max())
            turn = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
            if post < RIGID_FIT_MM and (shift > RIGID_SHIFT_MM or turn > RIGID_TURN_DEG):
                self._go_stale(f"{len(rests)} poses agree the camera moved {shift:.0f} mm / {turn:.1f} deg")
                return
        recent = sorted(rests, key=lambda r: r.t)[-3:]
        if self.state != "ok" and len(recent) == 3 and all(
                np.linalg.norm(r.seen - cal.to_cam(r.base)) < RECOVER_MM for r in recent):
            self.state, self.note = "ok", None

    def _go_stale(self, why: str) -> None:
        self.state, self.note = "stale", why
        for k in [k for k, r in self.rests.items() if r.epoch != self.epoch]:
            del self.rests[k]
        self.rig.jacobian = None

    # --- refit ---------------------------------------------------------------------------------------------------
    def _evaluate(self, arm, now) -> str:
        cal = arm.rig
        with self._lock:
            rests = self._fresh(now, epoch_only=self.state != "ok")
        if len(rests) < 3:
            if self.state == "stale":
                self.note = f"{self.note.split(';')[0]}; {len(rests)} of at least 3 distinct rests to re-fit from"
            return "too few rests"
        _, depth, K = self.rig.rgbd(frames=3)
        plane = self.rig.bench_plane(depth, K)
        readings = np.array([r.readings for r in rests])
        seen = np.array([r.seen for r in rests])
        args = (arm.kin, readings, seen, plane.n, plane.d, cal.bench_z_base_mm, arm.offsets, cal.marker_in_gripper_mm)
        try:
            fit = solve_camera(*args)
            if fit["sigma_yaw_deg"] > YAW_SIGMA_DEG:  # rests too clustered to pin the yaw: keep it, fit the shift
                Rz = align_z(plane.n).T @ cal.R
                fit = solve_camera(*args, fit_yaw=False, yaw_rad=float(np.arctan2(Rz[1, 0], Rz[0, 0])))
        except ValueError as e:
            return f"no re-fit: {e}"
        base = np.array([r.base for r in rests])
        before = np.linalg.norm(seen - cal.to_cam(base), axis=1)
        rms_before = float(np.sqrt((before ** 2).mean()))
        change = float(np.linalg.norm(base @ fit["R"].T + fit["t"] - cal.to_cam(base), axis=1).max())
        if not (fit["rms"] <= ADOPT_RMS_MM and fit["max"] <= ADOPT_MAX_MM and fit["rms"] <= ADOPT_GAIN * rms_before
                and change > ADOPT_SHIFT_MM):
            return f"kept: re-fit rms {fit['rms']:.1f} vs {rms_before:.1f} mm, change {change:.1f} mm"
        return self._adopt(arm, cal, fit, plane, rests, change)

    def _adopt(self, arm, cal, fit, plane, rests, change) -> str:
        yaw = "shift only" if fit.get("sigma_yaw_deg", 0) > YAW_SIGMA_DEG else f"yaw sigma {fit['sigma_yaw_deg']:.2f} deg"
        note = (f"camera re-fitted from {fit['kept'].sum()} of {len(rests)} rests while working ({yaw}), "
                f"rms {fit['rms']:.1f} mm, moved up to {change:.0f} mm; joint offsets, marker and bench height from rig "
                f"{cal.created}")
        samples = [Sample(r.readings, list(map(float, r.seen)), [0.0, 0.0], 0) for r in rests]
        new = make_rig(fit, samples, plane.n, plane.d, cal.robot_id, cal.motor_calibration_sha256, cal.camera_serial,
                       notes=[note])
        with arm.between_motions() as free:
            if not free:
                return "re-fit ready, waiting for the motion to end"
            if self.persist:
                new.save()
            arm.use_rig(new)
        with self._lock:
            self.rig.jacobian = None
            self.state, self.note = "ok", None
            self.last_refit = f"{time.strftime('%H:%M:%S')} {note}"
        return "adopted"

    # --- the armed sweep -----------------------------------------------------------------------------------------
    def _maybe_sweep(self, arm, now) -> None:
        if not (self.sweep_armed and self.state == "stale" and self.run_sweep is not None):
            return
        if now - self._last_tool < SWEEP_IDLE_S or any(now - t < SWEEP_GAP_S for t in self._sweeps):
            return
        if len(self._sweeps) >= SWEEP_ATTEMPTS or getattr(self.rig, "holding", False) or arm.lockout or arm.hot:
            return
        q, _ = arm.joints_now()
        if arm.tcp(q)[0][2] - arm.bench_z < SWEEP_CLEAR_MM:
            return
        self._sweeps.append(now)
        self.state, self.paused = "sweeping", True
        try:
            self.run_sweep()
            self.reset("calibrate_camera sweep run by the monitor")
        except Exception as e:  # the sweep's own error is the news; an arm halted by stop() disarms it
            self.state, self.note = "stale", f"auto sweep failed: {type(e).__name__}: {e}"
            if arm.halted:
                self.sweep_armed, self.note = False, "stop() ended the auto sweep: disarmed"
        finally:
            self.paused = False
