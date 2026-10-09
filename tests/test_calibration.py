"""Rig calibration solver on synthetic data with a known answer, and the asset's save/load round trip.
Run from the repo root: .venv\\Scripts\\python.exe -m tests.test_calibration
"""

import tempfile
from pathlib import Path

import numpy as np

import calibration as calib
from kinematics import JOINTS, Kinematics, rotation

DATA = Path(__file__).parent / "data"
TRUE_OFFSETS = np.array([0.0, -77.0, 91.0, -14.0, 0.0])
TRUE_G = np.array([-1.5, -8.0, 4.0])


def synthetic_rig(seed=0, n=24, noise_mm=1.5):
    rng = np.random.default_rng(seed)
    kin = Kinematics()
    bench_n = np.array([-0.0485, -0.9611, -0.2719])
    bench_n /= np.linalg.norm(bench_n)
    R = calib.align_z(bench_n) @ rotation([0, 0, 1], 0.7)  # level base, some camera yaw
    t = np.array([-60.0, 180.0, 520.0])
    home = np.array([6.9, 0.8, -9.5, 60.4, 140.0])
    lift = rng.uniform(0, 22, n)  # shoulder travel paired with an elbow raise, as the bench allows
    readings = home + np.c_[rng.uniform(-20, 20, n), lift, -0.8 * lift + rng.uniform(-8, 8, n),
                            rng.uniform(-25, 20, n), rng.uniform(-40, 0, n)]
    Ts = calib.gripper_poses(kin, readings, TRUE_OFFSETS)
    P = np.array([T[:3, :3] @ TRUE_G + T[:3, 3] for T in Ts])
    cam = P @ R.T + t + rng.normal(0, noise_mm, P.shape)
    return kin, readings, cam, bench_n, R, t


def test_recovers_offsets_camera_and_marker() -> None:
    kin, readings, cam, bench_n, R, t = synthetic_rig()
    fit = calib.best_fit(kin, readings, cam, bench_n, level=True)
    print(f"  level fit: offsets {np.round(fit['offsets'], 2).tolist()} +- {np.round(fit['offsets_sigma'], 2).tolist()} "
          f"g {np.round(fit['g'], 1).tolist()} rms {fit['rms']:.2f}")
    err = np.abs(fit["offsets"] - TRUE_OFFSETS)
    assert (err <= np.maximum(3 * fit["offsets_sigma"], 0.5)).all() and err.max() < 3.0, (err, fit["offsets_sigma"])
    assert fit["rms"] < 3.0 and fit["tilt_deg"] < 1e-6
    # what matters downstream: predicting where the gripper is, at poses the fit never saw
    _, held_readings, held_cam, *_ = synthetic_rig(seed=11, n=12, noise_mm=0.0)
    Ts = calib.gripper_poses(kin, held_readings, fit["offsets"])
    pred = np.array([T[:3, :3] @ fit["g"] + T[:3, 3] for T in Ts]) @ fit["R"].T + fit["t"]
    held_rms = float(np.sqrt(((pred - held_cam) ** 2).sum(1).mean()))
    print(f"  held-out marker prediction rms {held_rms:.2f} mm")
    assert held_rms < 3.5, held_rms
    free = calib.solve(kin, readings, cam, bench_n, fit["offsets"], level=False)
    assert free["tilt_deg"] < 3.0 and free["rms"] <= fit["rms"] + 1e-6, (free["tilt_deg"], free["rms"])


def test_starting_from_zero_offsets_still_finds_them() -> None:
    kin, readings, cam, bench_n, *_ = synthetic_rig(seed=3)
    fit = calib.solve(kin, readings, cam, bench_n, offsets0=np.zeros(5), level=True)
    best = calib.best_fit(kin, readings, cam, bench_n, level=True)
    assert best["rms"] <= fit["rms"] + 1e-9
    err = np.abs(best["offsets"] - TRUE_OFFSETS)
    assert (err <= np.maximum(3 * best["offsets_sigma"], 0.5)).all() and err.max() < 3.0, (err, best["offsets_sigma"])


def test_asset_round_trip_and_frames() -> None:
    kin, readings, cam, bench_n, R, t = synthetic_rig(seed=1)
    fit = calib.best_fit(kin, readings, cam, bench_n, level=True)
    bench_d = float(bench_n @ t)  # the base origin lies on the bench in this rig
    samples = [calib.Sample(list(map(float, q)), list(map(float, p)), [0.0, 0.0], 100) for q, p in zip(readings, cam)]
    rig = calib.make_rig(fit, samples, bench_n, bench_d, "exp23_follower",
                         calib.file_sha256(DATA / "exp23_follower.json"), "SIM")
    assert abs(rig.bench_z_base_mm) < 8.0, rig.bench_z_base_mm
    with tempfile.TemporaryDirectory() as tmp:
        path = rig.save(Path(tmp) / "rig.json")
        assert len(list(Path(tmp).glob("rig-*.json"))) == 1
        back = calib.RigCalibration.load(path)
    assert back.joint_offsets_deg == rig.joint_offsets_deg and len(back.samples) == len(samples)
    p = np.array([120.0, -30.0, 90.0])
    np.testing.assert_allclose(back.to_base(back.to_cam(p)), p, atol=1e-6)
    assert set(back.joint_offsets_deg) == set(JOINTS)


def test_camera_only_refit_pins_the_bench_height_and_finds_a_moved_camera() -> None:
    kin, readings, _, bench_n, R, t = synthetic_rig(seed=7, n=14)
    bench_d = float(bench_n @ t)  # bench height 0 in this rig
    u = calib.align_z(bench_n)[:, :2]
    R_new, t_new = calib.align_z(bench_n) @ rotation([0, 0, 1], 0.7 + np.radians(4.0)), t + u @ [60.0, -25.0]
    Ts = calib.gripper_poses(kin, readings, TRUE_OFFSETS)
    P = np.array([T[:3, :3] @ TRUE_G + T[:3, 3] for T in Ts])
    cam = P @ R_new.T + t_new + np.random.default_rng(2).normal(0, 1.5, P.shape)  # tripod slid 65 mm, turned 4 deg
    cam[[3, 9]] += [60.0, 0.0, 0.0]  # two wrong blobs
    fit = calib.solve_camera(kin, readings, cam, bench_n, bench_d, 0.0, TRUE_OFFSETS, TRUE_G)
    print(f"  camera refit: yaw off {np.degrees(fit['yaw_rad'] - 0.7) - 4.0:+.2f} deg, t off "
          f"{np.linalg.norm(fit['t'] - t_new):.1f} mm, kept {fit['kept'].sum()}/14, yaw sigma {fit['sigma_yaw_deg']:.2f}")
    assert not fit["kept"][[3, 9]].any() and fit["kept"].sum() == 12, fit["kept"]
    yaw_err = np.degrees(fit["yaw_rad"] - 0.7) - 4.0
    assert abs(yaw_err) < 2.5 * fit["sigma_yaw_deg"] < 4.0, (yaw_err, fit["sigma_yaw_deg"])
    miss = np.linalg.norm((P @ fit["R"].T + fit["t"]) - (P @ R_new.T + t_new), axis=1)  # where the arm works
    assert miss.max() < 5.0, miss.max()
    rig = calib.make_rig(fit, [], bench_n, bench_d, "exp23_follower", "x", "SIM")
    assert abs(rig.bench_z_base_mm) < 1e-6, rig.bench_z_base_mm
    narrow = calib.solve_camera(kin, readings[:4], cam[:4], bench_n, bench_d, 0.0, TRUE_OFFSETS, TRUE_G)
    assert narrow["sigma_yaw_deg"] > fit["sigma_yaw_deg"], (narrow["sigma_yaw_deg"], fit["sigma_yaw_deg"])


def test_full_sweep_recovers_the_rig_from_a_prior_in_simulation() -> None:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    import vision
    from arm import Arm
    from sim import FIELD_REST_TICKS, BenchScene, SimBus, SimCamera

    cal = SO101Follower(SO101FollowerConfig(port="unused", id="exp23_follower", calibration_dir=DATA,
                                            disable_torque_on_disconnect=False)).calibration
    truth = np.array(calib.FIELD_OFFSETS_DEG) + [0.0, 3.0, -2.5, 4.0, 0.0]  # the prior is a few degrees off
    bus = SimBus(cal, start_ticks=FIELD_REST_TICKS)
    arm = Arm(bus, cal)  # no rig: the sweep starts from the field prior
    arm.rest_pose_deg = arm.state().joints_deg
    cam = SimCamera(BenchScene(), bus, arm.kin, joint_offsets_deg=truth)
    spec = dict(hue=(100, 125), min_sat=70, min_mm=200, max_mm=900)
    locked = {}

    def tape_now():
        rgb, depth = cam.read_rgbd()
        return vision.color_blobs(rgb, depth, cam.intrinsics, **spec)

    def read_marker(prior, gate_mm):
        b = vision.nearest_blob(tape_now(), locked["p"] if prior is None else prior, gate_mm=gate_mm)
        if b is not None:
            locked["p"] = vision.front_point(b)
        return None if b is None else locked["p"]

    def bench():
        _, depth = cam.read_rgbd()
        v, u = np.nonzero((depth > 200) & (depth < 1600))
        return vision.table_plane(vision.backproject(u, v, depth[v, u], cam.intrinsics))

    (only,) = tape_now()
    locked["p"] = vision.front_point(only)
    stages = tuple(stage[::3] for stage in calib.FULL_SWEEP)
    rig = calib.full_calibration(arm, read_marker, bench, "sim", "SIM", stages=stages, speed_dps=40.0, min_samples=12)
    err = np.abs(rig.offsets_array() - truth)[1:4]
    print(f"  {len(rig.samples)} poses: offsets off by {np.round(err, 2).tolist()} deg (lift, elbow, wrist); {rig.notes[0]}")
    assert err.max() < 1.5, err
    rng = np.random.default_rng(5)
    misses = []
    for _ in range(6):  # poses the sweep never visited, judged against what the camera then sees
        q = np.array([arm.rest_pose_deg[m] for m in JOINTS]) + [rng.uniform(-15, 15), rng.uniform(0, 20),
                                                               rng.uniform(-20, 0), rng.uniform(-15, 15), 0.0]
        for m, deg in zip(JOINTS, q):
            bus.pos[m] = arm.to_tick_f(m, deg)
        T = calib.gripper_poses(arm.kin, q, rig.offsets_array())[0]
        (seen,) = tape_now()
        misses.append(np.linalg.norm(rig.to_cam(T[:3, :3] @ rig.marker_in_gripper_mm + T[:3, 3]) - vision.front_point(seen)))
    print(f"  held-out marker prediction: rms {np.sqrt(np.mean(np.square(misses))):.2f} mm, max {max(misses):.2f}")
    assert max(misses) < 6.0, misses
    arm.disconnect()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
