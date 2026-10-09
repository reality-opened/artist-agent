"""servo_to closed around the simulated arm (stiction included) and the rendered bench: lock onto the gripper tape by
motion despite a bigger same-colour distractor, find the can lid, and drive the tape to a point above it, starting
with no knowledge of where the camera is. Run from the repo root (~1-2 min): .venv\\Scripts\\python.exe -m tests.test_servo
"""

from pathlib import Path

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import vision
from arm import Arm
from servo import identify_marker, jacobian_ok, servo_to
from sim import BenchScene, SimBus, SimCamera

DATA = Path(__file__).parent / "data"
TAPE_HUE = (100, 125)


def make_world():
    config = SO101FollowerConfig(port="unused", id="exp23_follower", calibration_dir=DATA,
                                 disable_torque_on_disconnect=False)
    cal = SO101Follower(config).calibration
    bus = SimBus(cal)
    arm = Arm(bus, cal)
    scene = BenchScene(spheres=[((160.0, -160.0, 40.0), 28.0, (40, 80, 220))])  # bigger same-colour distractor
    return arm, bus, scene, SimCamera(scene, bus, arm.kin)


def blob_centres(cam):
    rgb, depth = cam.read_rgbd()
    return vision.color_blobs(rgb, depth, cam.intrinsics, TAPE_HUE, min_sat=70, min_mm=200, max_mm=900)


def test_identify_marker_ignores_the_bigger_distractor() -> None:
    arm, bus, scene, cam = make_world()
    blobs = blob_centres(cam)
    assert len(blobs) == 2 and np.linalg.norm(blobs[0].p - scene.to_cam(cam.tape_base_mm())) > 50
    distractor = blobs[0].p
    lock = identify_marker(arm, lambda: [b.p for b in blob_centres(cam)])
    tape = scene.to_cam(cam.tape_base_mm())
    # lock = the pre-probe surface point: ~8 mm depth bias + up to ~7 mm settle-back offset (seen 7-15 mm)
    assert lock is not None and np.linalg.norm(lock - tape) < 25, lock
    assert np.linalg.norm(lock - distractor) > 8 * np.linalg.norm(lock - tape), (lock, distractor)
    arm.disconnect()


def test_servo_drives_the_tape_above_the_can_lid() -> None:
    arm, bus, scene, cam = make_world()
    rgb, depth = scene.render(tape_base=cam.tape_base_mm())
    can = vision.locate_can(rgb, depth, cam.intrinsics)
    assert can is not None
    target = can.center + 70.0 * can.table.n  # 70 mm above the lid centre
    last = {"p": identify_marker(arm, lambda: [b.p for b in blob_centres(cam)])}
    assert last["p"] is not None

    def observe(prior):
        b = vision.nearest_blob(blob_centres(cam), last["p"] if prior is None else prior, gate_mm=60)
        if b is not None:
            last["p"] = b.p
        return None if b is None else b.p

    r = servo_to(arm, observe, target, tol_mm=5.0)
    truth = scene.to_cam(cam.tape_base_mm())
    print(f"  {r.status} after {r.probes} probes + {r.iterations} steps in {r.duration_s} s: tracked error "
          f"{r.error_mm} mm, true tape centre {np.linalg.norm(truth - target):.1f} mm from target, J sv {r.singular_values}")
    assert r.status in ("reached", "resolution_floor"), r
    assert r.error_mm <= 8.0 and jacobian_ok(r.jacobian)[0], r
    assert np.linalg.norm(truth - target) < 8.0 + scene.tape_radius, "tracked blob and true tape disagree"
    again = servo_to(arm, observe, target, tol_mm=5.0, jacobian=r.jacobian)  # continuation skips the probes
    assert again.probes == 0 and again.status in ("reached", "resolution_floor"), again
    arm.disconnect()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
