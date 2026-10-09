"""Replay the artist teleop daemon's IK results through kinematics.solve_ik.

Cases (data/teleop_ik_replay.jsonl):
- source "log": every {"xyz": ...} command in artist's ctl/status.json.log (2026-10-01..02; 47 accepted,
  all position-only, + 2 rejected jaws-down), started from teleop's targets just before the command.
- source "artist teleop.solve_ik": 18 jaws-down targets solved by artist's own code (12 accepted,
  6 rejected), because no jaws-down command succeeded in the log.
Joint values pass through LeRobot's own tick <-> degree conversion, built from the calibration the
cases were recorded under (data/exp23_follower.json).
Run from the repo root (~4 min; unreachable jaws-down targets try every seed at three weights):
.venv\\Scripts\\python.exe -m tests.test_kinematics
"""

import json
from pathlib import Path

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from kinematics import JOINTS, TICKS_PER_TURN, Kinematics, joint_bounds_deg, solve_ik

DATA = Path(__file__).parent / "data"


def test_replay_matches_teleop() -> None:
    config = SO101FollowerConfig(
        port="unused", id="exp23_follower", calibration_dir=DATA, disable_torque_on_disconnect=False
    )
    bus = SO101Follower(config).bus
    ids = [bus.motors[name].id for name in JOINTS]
    mids = [(bus.calibration[n].range_min + bus.calibration[n].range_max) / 2 for n in JOINTS]
    kin = Kinematics()
    lo, hi = joint_bounds_deg(kin, bus.calibration)
    cases = [json.loads(line) for line in (DATA / "teleop_ik_replay.jsonl").read_text().splitlines()]

    worst_xyz = 0.0
    for case in cases:
        label = (case["source"], case["xyz"], case["down"])
        start = bus._normalize(dict(zip(ids, case["start"])))
        start_deg = [start[i] for i in ids]
        if "error" in case:
            try:
                solve_ik(kin, start_deg, case["xyz"], case["down"], lo, hi)
            except ValueError as e:
                assert str(e) == case["error"], (label, str(e), case["error"])
                continue
            raise AssertionError(f"{label}: teleop rejected this target, the port accepted it")
        sol = solve_ik(kin, start_deg, case["xyz"], case["down"], lo, hi)
        ticks = [round(d * TICKS_PER_TURN / 360 + m) for d, m in zip(sol.joints_deg, mids)]  # teleop rounding
        assert ticks == case["ticks"], (label, ticks, case["ticks"])
        worst_xyz = max(worst_xyz, float(np.abs(np.subtract(sol.xyz, case["model_xyz"])).max()))
        sent = bus._unnormalize(dict(zip(ids, sol.joints_deg)))  # what send_action would write
        assert all(abs(sent[i] - t) <= 1 for i, t in zip(ids, ticks)), (label, sent, ticks)
    assert worst_xyz < 1e-6, worst_xyz
    print(f"{len(cases)} cases match teleop; worst model xyz deviation {worst_xyz:.1e} m")


if __name__ == "__main__":
    test_replay_matches_teleop()
