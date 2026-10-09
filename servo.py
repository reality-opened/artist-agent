"""Closed-loop visual servo: drive a tracked marker to a camera-frame point, in Cartesian command space.

Successor to artist's Ctl2 (drawbot/control.py; authors galois777, David Zhang, Hillysilly), keeping its field
lessons: the Jacobian is measured, never assumed, and updated every step (a fixed tick-space Jacobian drove the
elbow to a bad pose); a measurement is checked against the prediction and retried before it is trusted; the blob
tracked is the one nearest the prediction, never the largest. The command is a base-frame Cartesian step through
IK, so the command -> observation map stays close to linear across the workspace; displacement is measured from
the joints (FK) after every step, so deadband shows up as "did not move" instead of corrupting the update.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from arm import Arm, ArmError, LimitError

# Singular values of the camera-mm-per-base-mm map: the field measured 0.52..0.70 (the tape is not the TCP and
# turns with the wrist), so the map is not a rotation; outside these bounds the tracking is not believable.
SV_MIN, SV_MAX, COND_MAX = 0.15, 4.0, 12.0
BROYDEN_STEP, MIN_UPDATE_MM, STILL_MM = 0.4, 5.0, 1.0


@dataclass
class ServoStep:
    commanded_mm: list[float]
    moved_mm: list[float]
    marker_cam_mm: list[float]
    error_mm: float
    prediction_miss_mm: float
    trusted: bool


@dataclass
class ServoReceipt:
    """status: reached | resolution_floor (steps no longer move the joints: deadband) | max_iterations | budget
    (call again with jacobian= to continue) | unreachable (the arm cannot step that way) | probe_unreachable |
    tracking_lost | jacobian_unreliable | marker_not_found | arm_error | move_<status>."""

    status: str
    iterations: int
    error_mm: float | None
    marker_cam_mm: list[float] | None
    target_cam_mm: list[float]
    jacobian: list[list[float]] | None
    singular_values: list[float] | None
    probes: int
    duration_s: float
    steps: list[ServoStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def identify_marker(arm: Arm, blobs, probe_mm=15.0, speed_dps=15.0, min_move_mm=5.0) -> np.ndarray | None:
    """Which same-coloured blob rides on the arm: lift the arm probe_mm and back, keep the blob that moved.

    blobs() -> camera-frame mm centres of every candidate blob. Returns the marker's position before the probe
    (the arm returns there), or None when no single blob moved with the arm.
    """
    before = [np.asarray(p, float) for p in blobs()]
    if not before:
        return None
    up = arm.move_by([0.0, 0.0, probe_mm], speed_dps=speed_dps, tol_mm=1e9, multi_seed=False)
    after = [np.asarray(p, float) for p in blobs()] if up.move.status in ("reached", "settled") else []
    arm.move_by([0.0, 0.0, -probe_mm], speed_dps=speed_dps, tol_mm=1e9, multi_seed=False)
    if not after:
        return None
    moved = [b for b in before if min(np.linalg.norm(a - b) for a in after) >= min_move_mm]
    return moved[0] if len(moved) == 1 else None


def jacobian_ok(J) -> tuple[bool, list[float]]:
    s = np.linalg.svd(np.asarray(J, float), compute_uv=False)
    return bool(s[-1] > SV_MIN and s[0] < SV_MAX and s[0] / s[-1] < COND_MAX), [round(float(v), 3) for v in s]


def servo_to(arm: Arm, observe, target_cam, *, tol_mm=5.0, max_iters=20, gain=0.7, max_step_mm=25.0,
             probe_mm=20.0, gate_mm=20.0, budget_s=40.0, jaws_down=False, speed_dps=15.0, jacobian=None) -> ServoReceipt:
    """observe(prior_cam_mm or None) -> marker position (camera mm) or None. Probes +-probe_mm along each base
    axis to measure J (3x3, camera mm per base mm) unless `jacobian` is given, then steps du = J+ (gain * error)."""
    t0 = time.monotonic()
    target = np.asarray(target_cam, float)
    steps: list[ServoStep] = []
    notes: list[str] = []
    probes, J = 0, None if jacobian is None else np.asarray(jacobian, float)

    def receipt(status, y, iterations=0) -> ServoReceipt:
        ok_sv = jacobian_ok(J)[1] if J is not None else None
        return ServoReceipt(
            status=status, iterations=iterations,
            error_mm=None if y is None else round(float(np.linalg.norm(target - y)), 1),
            marker_cam_mm=None if y is None else [round(float(v), 1) for v in y],
            target_cam_mm=[round(float(v), 1) for v in target],
            jacobian=None if J is None else np.round(J, 4).tolist(), singular_values=ok_sv, probes=probes,
            duration_s=round(time.monotonic() - t0, 2), steps=steps, notes=notes)

    def look(prior, gate):
        for _ in range(3):
            p = observe(prior)
            if p is not None and np.linalg.norm(p - prior) < gate:
                return np.asarray(p, float)
        return None

    def move(delta):  # each step only needs to land roughly: the next observation corrects the rest
        before = arm.tcp_now()
        r = arm.move_by(delta, jaws_down=jaws_down, speed_dps=speed_dps, tol_mm=1e9, multi_seed=False,
                        joint_tol_deg=1.5)
        return r.move.status, arm.tcp_now() - before

    y = observe(None)
    if y is None:
        return receipt("marker_not_found", None)
    y = np.asarray(y, float)
    try:
        if J is None:
            X, Y = [], []
            for axis in range(3):  # out and back along +axis, else -axis, else half as far
                sides = 0
                for scale in (1.0, 0.5):
                    for sign in (1, -1):
                        d = np.zeros(3)
                        d[axis] = sign * scale * probe_mm
                        try:
                            status, dx = move(d)
                        except LimitError:
                            continue
                        probes += 1
                        if status not in ("reached", "settled"):
                            return receipt(f"move_{status}", y)
                        y_new = look(y, 3 * np.linalg.norm(dx) + gate_mm)
                        if y_new is None:
                            return receipt("tracking_lost", y)
                        X.append(dx)
                        Y.append(y_new - y)
                        y = y_new
                        status, dx = move(-d)
                        probes += 1
                        if status not in ("reached", "settled"):
                            return receipt(f"move_{status}", y)
                        y_new = look(y, 3 * np.linalg.norm(dx) + gate_mm)
                        if y_new is None:
                            return receipt("tracking_lost", y)
                        X.append(dx)
                        Y.append(y_new - y)
                        y = y_new
                        sides += 1
                        break
                    if sides:
                        break
                if not sides:
                    notes.append(f"no probe along base axis {'xyz'[axis]} is reachable from here")
                    return receipt("probe_unreachable", y)
            J = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0].T
            if not jacobian_ok(J)[0]:
                notes.append("probe motion and marker motion disagree: wrong blob, or the camera moved")
                return receipt("jacobian_unreliable", y)
        rejects, still, push = 0, 0, 1.0
        for it in range(max_iters + 1):
            err = target - y
            if np.linalg.norm(err) < tol_mm:
                return receipt("reached", y, it)
            if it == max_iters:
                return receipt("max_iterations", y, it)
            if time.monotonic() - t0 > budget_s:
                notes.append("time budget spent: call again with this jacobian to continue")
                return receipt("budget", y, it)
            # A step that moved nothing sat inside the joints' stiction band: retry it 2x, then 4x harder.
            du = np.linalg.lstsq(J, gain * err, rcond=None)[0] * push
            if np.linalg.norm(du) > max_step_mm:
                du *= max_step_mm / np.linalg.norm(du)
            for _ in range(3):  # a step the arm cannot take (reach, bench) is halved before giving up
                try:
                    status, dx = move(du)
                    break
                except LimitError as e:
                    reason = str(e)
                    du = du / 2
            else:
                notes.append(f"the next step is not reachable: {reason}")
                return receipt("unreachable", y, it)
            if status not in ("reached", "settled"):
                return receipt(f"move_{status}", y, it)
            pred = y + J @ dx
            y_new = look(pred, gate_mm)
            trusted = y_new is not None
            if not trusted:
                rejects += 1
                if rejects > 5:
                    return receipt("tracking_lost", y, it + 1)
                y_new = pred
            miss = float(np.linalg.norm((y_new - y) - J @ dx))
            if trusted and np.linalg.norm(dx) >= MIN_UPDATE_MM:
                J_new = J + BROYDEN_STEP * np.outer((y_new - y) - J @ dx, dx) / (dx @ dx)
                if jacobian_ok(J_new)[0]:
                    J = J_new
                else:
                    notes.append(f"step {it + 1}: Broyden update rejected (would make J ill-conditioned)")
            if np.linalg.norm(dx) < STILL_MM and np.linalg.norm(du) >= 2.0:
                still, push = still + 1, min(push * 2, 4.0)
            else:
                still, push = 0, 1.0
            y = y_new
            steps.append(ServoStep([round(float(v), 1) for v in du], [round(float(v), 1) for v in dx],
                                   [round(float(v), 1) for v in y], round(float(np.linalg.norm(target - y)), 1),
                                   round(miss, 1), trusted))
            if still >= 3:
                return receipt("resolution_floor", y, it + 1)
    except ArmError as e:
        notes.append(str(e))
        return receipt("arm_error", y, len(steps))
    return receipt("max_iterations", y, len(steps))
