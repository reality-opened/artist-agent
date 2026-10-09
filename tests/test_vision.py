"""Perception checks on synthetic data: artist's own geometry tests, plus the rendered bench scene from sim.py
(can lid, gripper tape with a bigger same-colour distractor) where ground truth is exact.
Run from the repo root: .venv\\Scripts\\python.exe -m tests.test_vision
"""

import numpy as np

import vision
from sim import BenchScene

ARTIST_K = {"fx": 686.65, "fy": 686.52, "cx": 641.32, "cy": 359.03}
# artist's field paper frame (camera mm), used by its synthetic pick tests
FIELD_FRAME = dict(ctr=[-44.7, 67.23, 426.82], e1=[-0.84033, 0.100997, -0.532582],
                   e2=[0.521402, 0.419329, -0.743171], n=[0.148269, -0.902199, -0.405036], d=-240.16)
FIELD_BASE = np.array([-242.58, -11.69])
TAPE_HUE = (100, 125)


def cylinder_points(c, ax, length=130.0, rad=7.0, step=0.5, rng=None) -> np.ndarray:
    """Upper half of a marker lying on the table, plane coords (x, y, h)."""
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    perp = np.array([-ax[1], ax[0]])
    S, T = np.meshgrid(np.arange(-length / 2, length / 2, step), np.linspace(0, np.pi, 40))
    xy = np.asarray(c, float) + S.reshape(-1, 1) * ax + (rad * np.cos(T)).reshape(-1, 1) * perp
    Q = np.c_[xy, rad + rad * np.sin(T).reshape(-1)]
    return Q + rng.normal(0, 0.5, Q.shape) if rng is not None else Q


def field_frame() -> vision.PlaneFrame:
    return vision.PlaneFrame(*(np.asarray(FIELD_FRAME[k], float) for k in ("ctr", "e1", "e2", "n")), FIELD_FRAME["d"])


def test_fit_plane_large_cloud() -> None:
    rng = np.random.default_rng(1)
    n = np.array([0.15, -0.9, -0.41])
    n /= np.linalg.norm(n)
    a = np.cross(n, [1, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    s, t = rng.uniform(-150, 150, (2, 150_000))
    pts = -240.0 * n + s[:, None] * a + t[:, None] * b + rng.normal(0, 0.5, (150_000, 1)) * n
    pts[:5000] += rng.uniform(-60, 60, (5000, 3))
    nf, df, inl = vision.fit_plane(pts)
    assert nf[2] < 0 and abs(nf @ n) > 0.9999
    assert abs(df * np.sign(nf @ n) - -240.0) < 0.5 and inl.mean() > 0.9


def test_backproject_project_and_ray_plane() -> None:
    u, v, z = np.array([10.0, 640, 1200]), np.array([5.0, 360, 700]), np.array([300.0, 450, 800])
    p = vision.backproject(u, v, z, ARTIST_K)
    np.testing.assert_allclose(vision.project(p, ARTIST_K), np.stack([u, v], 1), atol=1e-9)
    n, d = np.asarray(FIELD_FRAME["n"]), FIELD_FRAME["d"]
    hits = vision.ray_plane(u, v, n, d, ARTIST_K)
    np.testing.assert_allclose(hits @ n, d, atol=1e-9)
    np.testing.assert_allclose(vision.project(hits, ARTIST_K), np.stack([u, v], 1), atol=1e-9)


def test_plane_frame_roundtrip() -> None:
    fr = field_frame()
    xyh = np.array([[10.0, -20, 30], [-100, 50, 0]])
    np.testing.assert_allclose(fr.plane(fr.to_cam(xyh)), xyh, atol=0.05)
    on = vision.PlaneFrame.on(vision.Plane(fr.n, fr.d), [0.0, 0.0, 500.0])
    assert abs(on.plane(on.ctr)).max() < 1e-9 and abs(on.e1 @ on.n) < 1e-12 and on.e1[0] < 0


def test_locate_can_on_rendered_bench() -> None:
    scene = BenchScene()
    rgb, depth = scene.render()
    can = vision.locate_can(rgb, depth, scene.K)
    assert can is not None
    err = np.linalg.norm(can.center - scene.lid_center_cam())
    assert err < 3.0, f"lid centre off by {err:.1f} mm"
    assert abs(can.radius_mm - 33) < 2 and abs(can.height_mm - 122) < 3, (can.radius_mm, can.height_mm)
    assert can.table.n @ scene.table_normal_cam() > 0.999
    print(f"  lid centre error {err:.2f} mm, r {can.radius_mm:.1f}, h {can.height_mm:.1f}")


def test_tape_tracking_ignores_bigger_distractor() -> None:
    tape = np.array([250.0, -40.0, 160.0])
    scene = BenchScene(spheres=[((160.0, -160.0, 40.0), 28.0, (40, 80, 220))])  # bigger, same hue, unoccluded
    rgb, depth = scene.render(tape_base=tape)
    blobs = vision.color_blobs(rgb, depth, scene.K, TAPE_HUE, min_sat=70, min_mm=200, max_mm=900)
    assert len(blobs) == 2 and blobs[0].area > 2 * blobs[1].area, [b.area for b in blobs]
    truth = scene.to_cam(tape)
    assert np.linalg.norm(vision.nearest_blob(blobs).p - truth) > 50, "largest blob should be the distractor"
    hit = vision.nearest_blob(blobs, prior=truth + [15.0, -10.0, 5.0], gate_mm=40)
    assert hit is not None and np.linalg.norm(hit.p - truth) < scene.tape_radius + 2
    assert vision.nearest_blob(blobs, prior=truth + [300.0, 0, 0], gate_mm=40) is None


def test_locate_lying_cylinder_with_clutter() -> None:
    rng = np.random.default_rng(1)
    c, ax = np.array([-150.0, 60.0]), np.array([0.6, 0.8])
    table = np.c_[rng.uniform(-260, -40, (30000, 2)), rng.normal(0, 0.6, 30000)]
    box = np.c_[rng.uniform([-100, 110], [-80, 130], (800, 2)), rng.uniform(40, 80, 800)]
    speck = np.c_[rng.uniform([-200, 20], [-196, 24], (40, 2)), np.full(40, 10.0)]
    Q = np.vstack([table, cylinder_points(c, ax, rng=rng), box, speck])
    r = vision.locate_lying(Q, c + [15, -10], base=FIELD_BASE, radius=90)
    assert r is not None
    np.testing.assert_allclose(r["center"], c, atol=1.5)
    assert abs(r["axis"] @ ax) > np.cos(np.radians(2))
    assert abs(r["length"] - 130 * 0.96) < 6 and abs(r["top"] - 14.0) < 1.5 and r["tall_nearby"] > 100
    assert np.linalg.norm(r["near_end"] - FIELD_BASE) < np.linalg.norm(r["far_end"] - FIELD_BASE)
    assert vision.locate_lying(table, c, base=FIELD_BASE) is None


def test_find_lying_in_a_dark_room() -> None:
    fr = field_frame()
    H, W = 720, 1280
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - ARTIST_K["cx"]) / ARTIST_K["fx"], (v - ARTIST_K["cy"]) / ARTIST_K["fy"],
                     np.ones_like(u, float)], -1)
    depth = (fr.d / (rays @ fr.n)).astype(np.float32)
    img = np.full((H, W, 3), 22, np.uint8)
    c, ax = np.array([-60.0, -40.0]), np.array([0.0, 1.0])
    pts = fr.to_cam(cylinder_points(c, ax, step=0.3))
    px = np.round(vision.project(pts, ARTIST_K)).astype(int)
    order = np.argsort(-pts[:, 2])
    img[px[order, 1], px[order, 0]] = (8, 45, 8)
    depth[px[order, 1], px[order, 0]] = pts[order, 2]
    region = ((-200, 150), (-220, 200))
    found = vision.find_lying(img, depth, ARTIST_K, fr, "green", region=region, base=FIELD_BASE)
    assert found is not None and found["pose"]["source"] == "shape"
    np.testing.assert_allclose(found["pose"]["center"], c, atol=2.5)
    assert abs(found["pose"]["axis"][1]) > 0.99
    assert vision.find_lying(img, depth, ARTIST_K, fr, "blue", region=region) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
