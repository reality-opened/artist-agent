"""Arm runtime against the simulated bus (stiction, load saturation, blocks, stalls, injected faults).
Run from the repo root (~40 s, motions run in real time): .venv\\Scripts\\python.exe -m tests.test_arm
"""

import dataclasses
import threading
import time
from pathlib import Path

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import arm as arm_module
import calibration
from arm import GRIPPER, JOINTS, TORQUE_LIMIT, Arm, ArmError, BusyError, LimitError, ThermalError
from kinematics import angle_from_down
from sim import FIELD_REST_TICKS, SimBus

DATA = Path(__file__).parent / "data"


def make(**sim_kw) -> tuple[Arm, SimBus]:
    config = SO101FollowerConfig(port="unused", id="exp23_follower", calibration_dir=DATA,
                                 disable_torque_on_disconnect=False)
    cal = SO101Follower(config).calibration
    bus = SimBus(cal, **sim_kw)
    return Arm(bus, cal), bus


def settle_sim(bus: SimBus, seconds: float) -> None:
    time.sleep(seconds)
    bus.read("Present_Position", GRIPPER, normalize=False)


def test_adopt_is_zero_motion_and_pins_goals_first() -> None:
    arm, bus = make()
    st = arm.state()
    assert st.calibration_ok and not any(st.torque.values())
    before = dict(bus.pos)
    r = arm.move_joints({"shoulder_pan": st.joints_deg["shoulder_pan"]})
    assert r.status == "reached", r
    assert all(bus.reg["Torque_Enable"].values())
    assert max(abs(bus.pos[m] - before[m]) for m in bus.pos) < 1
    first_enable = next(i for i, w in enumerate(bus.writes) if w[0] == "Torque_Enable")
    assert any(w[0] == "Goal_Position" for w in bus.writes[:first_enable]), "goal must be pinned before torque on"
    assert ("Torque_Limit", "shoulder_lift", TORQUE_LIMIT) in bus.writes
    arm.disconnect()


def test_receipt_and_limits_are_rejected_not_clamped() -> None:
    arm, bus = make()
    target = arm.state().joints_deg["elbow_flex"] - 15
    r = arm.move_joints({"elbow_flex": target}, speed_dps=40)
    assert r.status in ("reached", "settled") and abs(r.error["elbow_flex"]) <= 2.0, r
    for bad in ({"shoulder_lift": -10.0}, {GRIPPER: 150.0}, {"elbow": 0.0}):
        try:
            arm.move_joints(bad)
        except LimitError as e:
            assert "outside" in str(e) or "unknown" in str(e)
        else:
            raise AssertionError(f"{bad} was accepted")
    arm.disconnect()


def test_calibration_mismatch_refuses_motion_but_reads() -> None:
    arm, bus = make()
    c = bus.eeprom["wrist_flex"]
    bus.eeprom["wrist_flex"] = dataclasses.replace(c, homing_offset=c.homing_offset + 100)
    st = arm.state()
    assert st.calibration_ok is False and st.calibration_mismatches[0].startswith("wrist_flex")
    try:
        arm.move_joints({"wrist_flex": 0.0})
    except ArmError as e:
        assert "calibration" in str(e)
    else:
        raise AssertionError("moved with a calibration mismatch")
    arm.disconnect()


def test_blocked_joint_is_caught_and_does_not_lunge() -> None:
    arm, bus = make()
    start = bus.pos["elbow_flex"]
    bus.blocks["elbow_flex"] = (start - 60, start + 400)  # a hand stops it 60 ticks along (elbow up, clear of the bench)
    r = arm.move_joints({"elbow_flex": arm.to_deg("elbow_flex", start - 600)}, speed_dps=40)
    assert r.status == "blocked" and r.blocked == "elbow_flex", r
    assert r.max_load["elbow_flex"] <= TORQUE_LIMIT, r.max_load
    assert abs(bus.reg["Goal_Position"]["elbow_flex"] - bus.pos["elbow_flex"]) <= 1, "travel stayed banked"
    del bus.blocks["elbow_flex"]
    held = bus.pos["elbow_flex"]
    settle_sim(bus, 0.5)
    assert abs(bus.pos["elbow_flex"] - held) < 2, "the joint lunged once the hand let go"
    arm.disconnect()


def test_a_hand_that_gives_way_is_caught_while_moving() -> None:
    arm, bus = make()
    start = bus.pos["shoulder_pan"]
    hand = [start + 120.0]  # 120 ticks along, then yielding 15 ticks/s: the joint creeps on, as against a real hand
    bus.blocks["shoulder_pan"] = (-float("inf"), hand[0])
    released = threading.Event()

    def give_way() -> None:
        while not released.wait(0.1):
            hand[0] += 1.5
            bus.blocks["shoulder_pan"] = (-float("inf"), hand[0])

    threading.Thread(target=give_way, daemon=True).start()
    r = arm.move_joints({"shoulder_pan": arm.to_deg("shoulder_pan", start + 600)}, speed_dps=6)
    released.set()
    contact_s = 140 / (6 * 4095 / 360)  # breakaway + 120 ticks at 6 deg/s
    assert r.status == "blocked" and r.blocked == "shoulder_pan", r
    assert any("met resistance while moving" in n for n in r.notes), r.notes
    assert r.duration_s < contact_s + 1.5, f"caught {r.duration_s - contact_s:.1f} s after contact"
    del bus.blocks["shoulder_pan"]
    held = bus.pos["shoulder_pan"]
    settle_sim(bus, 0.5)
    assert abs(bus.pos["shoulder_pan"] - held) < 2, "the joint lunged once the hand let go"
    arm.disconnect()


def test_stop_from_another_call_and_busy_rejection() -> None:
    arm, bus = make()
    pan = arm.state().joints_deg["shoulder_pan"]
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("r", arm.move_joints({"shoulder_pan": pan + 60}, speed_dps=10)))
    worker.start()
    time.sleep(0.6)
    try:
        arm.move_joints({"elbow_flex": 0.0})
    except BusyError:
        pass
    else:
        raise AssertionError("a second motion started while one was running")
    s = arm.stop()
    worker.join(5)
    assert s["status"] == "held" and out["r"].status == "stopped", (s, out)
    stopped_at = bus.pos["shoulder_pan"]
    settle_sim(bus, 0.4)
    assert abs(bus.pos["shoulder_pan"] - stopped_at) <= 25, "coasted further than its tracking lag"
    assert 3 < out["r"].final["shoulder_pan"] - pan < 40, out["r"].final
    writes = len(bus.writes)
    assert arm.stop()["status"] == "idle" and len(bus.writes) == writes, "an idle stop() must not touch the goals"
    arm.disconnect()


def test_gravity_sag_is_compensated_and_never_dropped() -> None:
    arm, bus = make()
    bus.sag["elbow_flex"] = 50.0  # the real elbow sat 26..55 ticks below its goal
    elbow = arm.state().joints_deg["elbow_flex"]
    r = arm.move_joints({"elbow_flex": elbow - 6}, speed_dps=20)
    assert r.status == "reached" and abs(r.error["elbow_flex"]) <= 1.0, r
    held = bus.pos["elbow_flex"]
    lead = bus.reg["Goal_Position"]["elbow_flex"] - held
    assert 35 <= lead <= 70, f"should hold against gravity with a ~50 tick lead, has {lead}"
    bus.trace["elbow_flex"] = []
    pan = arm.state().joints_deg["shoulder_pan"]
    assert arm.move_joints({"shoulder_pan": pan + 3}, speed_dps=20).status == "reached"
    assert max(abs(p - held) for p in bus.trace["elbow_flex"]) < 3, "the elbow dropped while another joint moved"
    bus.trace["elbow_flex"] = []
    r = arm.move_joints({"elbow_flex": elbow - 3}, speed_dps=20)  # up, against gravity
    assert r.status == "reached", r
    assert min(bus.trace["elbow_flex"]) > held - 3, "the elbow dropped at the start of its own move"
    assert arm.move_joints({"elbow_flex": elbow - 30}, speed_dps=30).status == "reached"
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("r", arm.move_joints({"elbow_flex": elbow - 5},
                                                                                 speed_dps=8)))
    worker.start()
    time.sleep(0.8)
    arm.stop()
    worker.join(5)
    at_stop = bus.pos["elbow_flex"]
    settle_sim(bus, 0.5)
    assert out["r"].status == "stopped", out
    assert at_stop - 3 <= bus.pos["elbow_flex"] <= at_stop + 25, (at_stop, bus.pos["elbow_flex"])
    arm.disconnect()


def test_thermal_stages() -> None:
    arm, bus = make()
    arm.set_gripper(50.0, speed_dps=60)
    temps = bus.reg["Present_Temperature"]
    temps[GRIPPER] = 56
    for _ in range(3):
        arm._check_thermal(arm._read("Present_Temperature"))
    assert arm.gripper_relaxed and bus.reg["Torque_Enable"][GRIPPER] == 0 and bus.reg["Torque_Enable"]["elbow_flex"] == 1
    temps["shoulder_lift"] = 60
    for _ in range(3):
        arm._check_thermal(arm._read("Present_Temperature"))
    assert arm.lockout and "shoulder_lift" in arm.hot
    try:
        arm.move_joints({"elbow_flex": 0.0})
    except ThermalError:
        pass
    else:
        raise AssertionError("moved during a thermal lockout")
    temps["shoulder_lift"] = 67
    actions = [arm._check_thermal(arm._read("Present_Temperature")) for _ in range(3)]
    assert actions[-1] == "release_all" and not any(bus.reg["Torque_Enable"].values())
    for m in temps:
        temps[m] = 30
    arm._check_thermal(arm._read("Present_Temperature"))
    assert not arm.lockout
    assert arm.set_gripper(60.0, speed_dps=60).status in ("reached", "settled") and not arm.gripper_relaxed
    arm.disconnect()


def test_watchdog_acts_while_idle() -> None:
    arm, bus = make()
    arm.set_gripper(50.0, speed_dps=60)
    bus.reg["Present_Temperature"]["elbow_flex"] = 70
    time.sleep(4.5)
    assert not any(bus.reg["Torque_Enable"].values()) and arm.lockout, "idle watchdog did not act"
    arm.disconnect()


def test_grasp_detects_an_object_and_empty_jaws() -> None:
    arm, bus = make()
    arm.set_gripper(90.0, speed_dps=60)
    bus.grip_stall = 820
    g = arm.grasp()
    assert g.held and abs(g.stall_tick - 820) <= 12 and g.hold_tick == g.stall_tick - 38, g
    bus.grip_stall = None
    arm.set_gripper(90.0, speed_dps=60)
    g = arm.grasp()
    assert not g.held and "nothing" in g.reason, g
    arm.disconnect()


def test_bus_failure_keeps_torque_on() -> None:
    arm, bus = make()
    arm.set_gripper(50.0, speed_dps=60)
    bus.fail_reads = 1000
    try:
        arm.move_joints({"elbow_flex": -5.0})
    except ArmError as e:
        assert "nothing moved" in str(e), e
    else:
        raise AssertionError("moved without being able to read the joints")
    bus.fail_after = 6  # the pre-move read and adopt succeed, the executor's reads then fail mid-motion
    r = arm.move_joints({"elbow_flex": -5.0})
    assert r.status == "comm_error" and all(bus.reg["Torque_Enable"].values()), r
    bus.fail_reads = 0
    assert arm.state().connected
    arm.disconnect()


def test_obstacle_near_the_target_ends_blocked_not_pushed() -> None:
    arm, bus = make()
    start = bus.pos["elbow_flex"]
    bus.blocks["elbow_flex"] = (start - 40, start + 400)  # stops it 40 ticks short of a 60 tick move (up)
    r = arm.move_joints({"elbow_flex": arm.to_deg("elbow_flex", start - 60)}, speed_dps=12)
    assert r.status == "blocked" and r.blocked == "elbow_flex", r
    assert r.duration_s < 3.5, f"pushed for {r.duration_s} s before giving up"
    assert abs(bus.reg["Goal_Position"]["elbow_flex"] - bus.pos["elbow_flex"]) <= 1, "still pushing after the stop"
    arm.disconnect()


def test_strong_stiction_is_overcome_without_hunting() -> None:
    arm, bus = make()
    bus.sag["elbow_flex"] = 45.0  # measured: ~78 ticks of lead lifting against gravity, ~0-3 going with it
    bus.stiction["elbow_flex"] = (40.0, 15.0)
    elbow = arm.state().joints_deg["elbow_flex"]
    for delta in (-3.0, 2.0, -1.5):
        r = arm.move_joints({"elbow_flex": elbow + delta}, speed_dps=12)
        assert r.status in ("reached", "settled") and r.blocked is None, r
        assert abs(r.error["elbow_flex"]) <= 3.0, r
        assert r.duration_s < 8.0, r
        elbow = r.final["elbow_flex"]
    arm.disconnect()


def test_a_joint_restarting_after_waiting_at_a_waypoint_is_not_called_blocked() -> None:
    # 2026-10-16, mid jaws_down move: the elbow's goal waited at a waypoint for the shoulder, the elbow stopped and
    # stuck, and its second breakaway (~90 ticks of lead) was taken for contact: blocked with nothing in the way.
    arm, bus = make()
    bus.stiction["elbow_flex"] = (90.0, 15.0)
    st = arm.state()
    pan, elbow = st.ticks["shoulder_pan"], st.ticks["elbow_flex"]
    path = [{"shoulder_pan": pan + 400, "elbow_flex": elbow - 150}, {"shoulder_pan": pan + 400, "elbow_flex": elbow - 300}]
    with arm._motion_slot():
        r = arm._run(path, 20 * arm_module.TICKS_PER_DEG, 11, 20.0)
    assert r.status in ("reached", "settled") and r.blocked is None, r
    arm.disconnect()


def test_a_joint_slowed_by_load_is_not_called_blocked() -> None:
    # 2026-10-16: the elbow lifting the stretched-out arm covered 33 ticks in 0.5 s against a 20 deg/s goal (29 %, the
    # contact rule's 30 % line) and was called blocked. Steadily slow is not held: a hand lets a joint creep a few ticks.
    arm, bus = make()
    bus.speed = 60.0  # every joint at most ~5 deg/s, as if load-limited
    pan = arm.state().joints_deg["shoulder_pan"]
    r = arm.move_joints({"shoulder_pan": pan + 15}, speed_dps=20)
    assert r.status in ("reached", "settled") and r.blocked is None, r
    arm.disconnect()


def test_rig_offsets_shift_kinematics_and_bounds() -> None:
    config = SO101FollowerConfig(port="unused", id="exp23_follower", calibration_dir=DATA,
                                 disable_torque_on_disconnect=False)
    cal = SO101Follower(config).calibration
    offsets = {"shoulder_pan": 0.0, "shoulder_lift": -103.0, "elbow_flex": 87.0, "wrist_flex": -13.0, "wrist_roll": 0.0}
    rig = calibration.RigCalibration(offsets, np.eye(4).tolist(), [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0, 6.0, 0.0,
                                     4.5, 9.9, "2026-10-08T15:47:05", "exp23_follower", "abc", "SIM")
    plain, shifted = Arm(SimBus(cal), cal), Arm(SimBus(cal), cal, rig=rig, motor_calibration_sha256="abc")
    q = [7.0, 0.8, -9.5, 60.4, 162.0]
    expect = plain.tcp(np.array(q) + np.array(list(offsets.values())))[0]
    np.testing.assert_allclose(shifted.tcp(q)[0], expect, atol=1e-9)
    urdf_wrist_lo = float(np.degrees(plain.kin.joints["wrist_flex"]["limits"][0]))
    assert abs(shifted.limits_api()["wrist_flex"][0] - (urdf_wrist_lo + 13.0)) < 0.2  # URDF limits are model angles
    assert abs(plain.limits_api()["wrist_flex"][0] - (urdf_wrist_lo + 13.0)) > 5
    assert shifted.bench_z == 6.0 and "poses" in shifted.rig_note
    stale = Arm(SimBus(cal), cal, rig=rig, motor_calibration_sha256="other")
    assert stale.rig is None and not stale.offsets.any() and "different motor calibration" in stale.rig_note


def test_bench_clearance_refuses_before_moving() -> None:
    arm, bus = make()
    start = arm.tcp_now()
    before = dict(bus.pos)
    try:
        arm.move_to([start[0], start[1], 3.0])
    except LimitError as e:
        assert "bench" in str(e), e
    else:
        raise AssertionError("a line ending 3 mm above the bench was accepted")
    lift = arm.state().joints_deg["shoulder_lift"]
    low = next(d for d in range(5, 90, 5) if arm.clearance([arm.state().joints_deg[m] + (d if m == "shoulder_lift" else 0)
                                                             for m in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                                       "wrist_flex", "wrist_roll")])[0] < 0)
    saved, arm_module.SHOULDER_LIFT_NM = arm_module.SHOULDER_LIFT_NM, 9.0  # the bench guard alone (shoulder: below)
    try:
        arm.move_joints({"shoulder_lift": lift + low})
    except LimitError as e:
        assert "bench" in str(e), e
    else:
        raise AssertionError("a joint move into the bench was accepted")
    finally:
        arm_module.SHOULDER_LIFT_NM = saved
    assert max(abs(bus.pos[m] - before[m]) for m in bus.pos) < 1, "moved before refusing"
    arm.disconnect()


def test_a_pose_the_shoulder_could_not_lift_is_refused_but_the_way_back_stays_open() -> None:
    arm, bus = make()
    q = arm.state().joints_deg
    before = dict(bus.pos)
    load = arm.shoulder_load_nm([q[m] for m in JOINTS])  # with zero offsets this arm starts stretched out, ~0.67 N.m
    try:
        arm.move_joints({"shoulder_lift": q["shoulder_lift"] + 10})
    except LimitError as e:
        assert "shoulder" in str(e), e
    else:
        raise AssertionError("a pose heavier than the shoulder can lift was accepted")
    assert max(abs(bus.pos[m] - before[m]) for m in bus.pos) < 1, "moved before refusing"
    saved, arm_module.SHOULDER_LIFT_NM = arm_module.SHOULDER_LIFT_NM, load - 0.05  # now it starts overloaded
    try:
        for stays_heavy in ({"shoulder_lift": q["shoulder_lift"] + 5}, {"wrist_roll": q["wrist_roll"] + 10}):
            try:
                arm.move_joints(stays_heavy)
            except LimitError:
                pass
            else:
                raise AssertionError(f"{stays_heavy} ends overloaded but was accepted")
        r = arm.move_joints({"elbow_flex": q["elbow_flex"] - 40})  # folds the load off: allowed
        assert r.status in ("reached", "settled"), r
        assert abs(arm.shoulder_load_nm([arm.state().joints_deg[m] for m in JOINTS])) < load - 0.05
    finally:
        arm_module.SHOULDER_LIFT_NM = saved
    arm.disconnect()


def test_an_arm_left_too_low_can_still_move_up() -> None:
    arm, bus = make()
    arm.bench_z = arm.tcp_now()[2] - 5.0  # as if pushed down: the TCP is already within its 8 mm margin
    try:
        arm.move_by([0.0, 0.0, -10.0])
    except LimitError:
        pass
    else:
        raise AssertionError("moved further down while already too close to the bench")
    r = arm.move_by([0.0, 0.0, 25.0], speed_dps=30)
    assert r.status in ("reached", "off_target"), r
    assert arm.clearance([arm.state().joints_deg[m] for m in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                             "wrist_flex", "wrist_roll")])[0] > 0
    arm.disconnect()


def test_cartesian_line_and_refusal_before_motion() -> None:
    arm, bus = make()
    start = arm.tcp_now()
    r = arm.move_to(start + [0.0, 50.0, -20.0], speed_dps=40)
    assert r.move.status in ("reached", "settled") and r.segments == 2, r
    assert max(abs(e) for e in r.move.error.values()) <= 21 / 11.375, "a joint stopped outside the stiction band"
    assert r.status == ("reached" if r.tcp_error_mm <= 10 else "off_target") and r.tcp_error_mm < 25, r
    print(f"  54 mm line: {r.segments} segments, TCP {r.tcp_error_mm} mm from target under 20-tick stiction ({r.status})")
    before = dict(bus.pos)
    try:
        arm.move_to(start + [600.0, 0.0, 0.0])
    except LimitError as e:
        assert "segment" in str(e)
    else:
        raise AssertionError("an unreachable line was accepted")
    assert max(abs(bus.pos[m] - before[m]) for m in bus.pos) < 1, "moved before the whole line was solved"
    assert np.isfinite(arm.tcp_now()).all()
    arm.disconnect()


def test_jaws_down_from_rest_tilts_as_far_as_the_wrist_goes_and_keeps_the_twist() -> None:
    # The field arm of 2026-10-16 at its rest pose (jaws ~69 deg off vertical). Agents' jaws_down approach 9 cm over
    # the bench was refused at segment 1: straight down there needs wrist_flex ~114 deg, past its 90.5 limit.
    config = SO101FollowerConfig(port="unused", id="exp23_follower", calibration_dir=DATA,
                                 disable_torque_on_disconnect=False)
    cal = SO101Follower(config).calibration
    offsets = {"shoulder_pan": 0.0, "shoulder_lift": -102.37, "elbow_flex": 85.98, "wrist_flex": -14.49,
               "wrist_roll": 0.0}
    rig = calibration.RigCalibration(offsets, np.eye(4).tolist(), [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0, 6.19, 0.0,
                                     4.8, 12.2, "2026-10-16T15:10:13", "exp23_follower", "abc", "SIM")
    arm = Arm(SimBus(cal, start_ticks=FIELD_REST_TICKS), cal, rig=rig, motor_calibration_sha256="abc")
    roll = arm.state().joints_deg["wrist_roll"]
    high = arm.move_to([261.5, -6.0, 89.0], jaws_down=True, speed_dps=40)
    assert high.status in ("reached", "off_target") and high.tcp_error_mm < 10, high
    assert 5 < angle_from_down(high.jaw_axis) < 25 and any("jaws_down" in n for n in high.move.notes), high
    low = arm.move_by([0.0, 0.0, -45.0], jaws_down=True, speed_dps=40)
    assert low.status in ("reached", "off_target") and angle_from_down(low.jaw_axis) < 5, low
    assert abs(arm.state().joints_deg["wrist_roll"] - roll) < 2, "jaws_down twisted the jaws"
    print(f"  jaws_down from rest: {angle_from_down(high.jaw_axis):.0f} deg off vertical 9 cm up, "
          f"{angle_from_down(low.jaw_axis):.1f} deg 4.5 cm up")
    arm.disconnect()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
