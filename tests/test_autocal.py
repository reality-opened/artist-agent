"""The camera-link monitor on the simulated field-like world (field joint offsets, the real rest pose): quiet while
nothing moves; a camera slid sideways, which the bench-plane check cannot see, turns it suspect then stale and
withholds base-frame points; the arm's ordinary rests then re-fit the camera without a sweep.
Run from the repo root (~1 min): .venv\\Scripts\\python.exe -m tests.test_autocal
"""

import argparse
import time

import numpy as np

from autocal import AutoCalibration
from kinematics import JOINTS
from server import Rig

TOUR = ({"shoulder_pan": -25}, {"shoulder_pan": -12}, {"shoulder_pan": 12}, {"shoulder_pan": 25},
        {"shoulder_lift": 12, "elbow_flex": -12}, {"shoulder_pan": -20, "shoulder_lift": 18, "elbow_flex": -24},
        {"shoulder_pan": 20, "shoulder_lift": 18, "elbow_flex": -24}, {"elbow_flex": -16}, {"wrist_flex": 15},
        {"shoulder_lift": 24, "elbow_flex": -28}, {"shoulder_pan": -15, "wrist_flex": -15}, {"shoulder_pan": 15},
        {"shoulder_pan": 8, "elbow_flex": -10}, {"shoulder_pan": -8, "shoulder_lift": 8}, {})


def make():
    args = argparse.Namespace(sim=True, gemini_serial="SIM", robot_id="exp23_follower", port="sim", wrist_index=None,
                              wrist_rotate="cw", rest_pose=None)
    rig = Rig(args)
    arm = rig.arm
    rig.gemini()
    rig.auto = AutoCalibration(rig, persist=False, period_s=1e9)
    return rig, arm, rig.auto


def visit(arm, auto, poses) -> list[str]:
    seen = []
    for off in poses:
        r = arm.move_joints({m: arm.rest_pose_deg[m] + off.get(m, 0.0) for m in JOINTS}, speed_dps=40)
        assert r.status in ("reached", "settled"), r
        seen.append(f"{auto.poll(time.monotonic())} -> {auto.state}")
    return seen


def camera_error_mm(rig) -> float:
    """Where the rig predicts workspace points in the camera vs where the scene really has them."""
    pts = np.array([[x, y, z] for x in (150, 250, 350) for y in (-150, 0, 150) for z in (20, 120)], float)
    return float(np.linalg.norm(rig.arm.rig.to_cam(pts) - rig._gemini.scene.to_cam(pts), axis=1).max())


def test_monitor_flags_a_slid_camera_and_refits_it_from_ordinary_rests() -> None:
    rig, arm, auto = make()
    quiet = visit(arm, auto, TOUR[:5])
    print("  camera untouched:", quiet)
    assert auto.state == "ok" and all(e.endswith("ok") for e in quiet), quiet
    assert rig.to_base([np.zeros(3)], plane := _plane(rig))[0] is not None

    scene = rig._gemini.scene
    scene.cam_from_base[:3, 3] -= scene.cam_from_base[:3, :3] @ np.array([0.0, 25.0, 0.0])  # tripod slid 25 mm
    plane = _plane(rig)
    assert arm.rig.camera_moved(plane.n, plane.d) is None, "the bench-plane check was expected to miss a slide"
    moved = visit(arm, auto, TOUR[5:9])
    print("  camera slid 25 mm:", moved)
    assert auto.state == "stale", moved
    points, note = rig.to_base([np.zeros(3)], plane)
    assert points is None and "camera moved" in note, (points, note)

    auto.period_s = 0.0  # re-fit at every look from here on
    healing = visit(arm, auto, TOUR[9:] + TOUR[:4])
    print("  ordinary rests:", healing)
    assert any("adopted" in e for e in healing) and auto.state == "ok", healing
    err = camera_error_mm(rig)
    print(f"  after the re-fit: camera-frame error {err:.1f} mm over the workspace; {auto.last_refit}")
    assert err < 8.0, err
    arm.disconnect()


def _plane(rig):
    _, depth, K = rig.rgbd(frames=2)
    return rig.bench_plane(depth, K)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
