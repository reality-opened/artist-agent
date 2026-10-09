"""RGB-D perception on aligned color + depth: geometry, color blobs, table plane, can lid, objects lying flat.

Pure functions on numpy arrays: no cameras, no IPC. Ported from reality-opened/artist (drawbot/rgbd.py,
drawbot/frame.py, the sensing half of drawbot/pick.py, and the can-opening session's artist_can.py;
authors galois777, David Zhang, Hillysilly).
Conventions: color is RGB uint8 (LeRobot's default; HSV hues match artist's BGR-tuned thresholds), depth is
millimetres with 0 = no reading, K = color intrinsics {fx, fy, cx, cy}, camera frame x right / y down /
z forward in mm.
"""

import warnings
from dataclasses import dataclass

import cv2
import numpy as np

# Marker hue bands (OpenCV hue 0..179, lo > hi wraps through red) and min saturation, after
# normalize_brightness; field-tested on lying markers (artist pick.py BANDS).
LYING_BANDS = {"blue": ((95, 128), 60), "yellow": ((18, 34), 90), "green": ((35, 80), 90),
               "orange": ((172, 12), 120), "pink": ((140, 171), 70)}


def median_depth(frames) -> np.ndarray:
    """Per-pixel median of depth frames (mm); 0 and >= 5 m readings are ignored. Float mm, 0 = none."""
    stack = [np.where((d > 0) & (d < 5000), d, np.nan).astype(np.float32) for d in frames]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
        return np.nan_to_num(np.nanmedian(np.stack(stack), axis=0), nan=0.0)


def backproject(u, v, z, K) -> np.ndarray:
    """Pixel(s) + depth mm -> camera-frame XYZ mm."""
    u, v, z = np.asarray(u, float), np.asarray(v, float), np.asarray(z, float)
    return np.stack([(u - K["cx"]) * z / K["fx"], (v - K["cy"]) * z / K["fy"], z], -1)


def project(p, K) -> np.ndarray:
    p = np.atleast_2d(np.asarray(p, float))
    return np.stack([K["fx"] * p[:, 0] / p[:, 2] + K["cx"], K["fy"] * p[:, 1] / p[:, 2] + K["cy"]], -1)


def ray_plane(u, v, n, d, K) -> np.ndarray:
    """Camera-frame points where pixel rays hit the plane n.p = d (works where depth is missing or unreliable)."""
    rays = backproject(u, v, np.ones(np.shape(u)), K)
    return rays * (d / (rays @ np.asarray(n, float)))[..., None]


def hue_mask(hsv, hue, min_sat=150, min_val=80, max_sat=255) -> np.ndarray:
    """hue=(lo, hi) in OpenCV units (0..179); lo > hi wraps through red."""
    h = hsv[..., 0]
    band = (h >= hue[0]) & (h <= hue[1]) if hue[0] <= hue[1] else (h >= hue[0]) | (h <= hue[1])
    return band & (hsv[..., 1] > min_sat) & (hsv[..., 1] <= max_sat) & (hsv[..., 2] > min_val)


def fit_plane(pts, iters=300, thresh=3.0, seed=0):
    """RANSAC plane: (unit normal n toward the camera (n_z < 0), offset d with n.p = d, inlier mask)."""
    rng = np.random.default_rng(seed)
    best, best_in = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.integers(0, len(pts), 3)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(pts @ n - n @ a) < thresh
        if inl.sum() > best_in:
            best, best_in = inl, inl.sum()
    q = pts[best]
    centroid = q.mean(0)
    # full_matrices=False: with >100k points the full U would be N x N.
    n = np.linalg.svd(q - centroid, full_matrices=False)[2][-1]
    if n[2] > 0:
        n = -n
    return n, float(n @ centroid), best


@dataclass
class Plane:
    """n.p = d with the unit normal n pointing toward the camera, i.e. up out of a table the camera looks down on."""

    n: np.ndarray
    d: float

    def height(self, p) -> np.ndarray:
        """Signed distance (mm) above the plane, toward the camera."""
        return np.asarray(p, float) @ self.n - self.d


def table_plane(pts, seed=0) -> Plane:
    """RANSAC the dominant plane of a cloud (the bench)."""
    n, d, _ = fit_plane(pts[:: max(1, len(pts) // 60000)], iters=400, thresh=4.0, seed=seed)
    return Plane(n, d)


@dataclass
class Blob:
    p: np.ndarray  # median 3D point, camera mm
    px: tuple[float, float]
    area: int
    pts: np.ndarray  # every 3D point of the blob
    uv: tuple[np.ndarray, np.ndarray]


def color_blobs(rgb, depth, K, hue, min_sat=150, min_val=80, max_sat=255, min_mm=250, max_mm=900) -> list[Blob]:
    """Saturated blobs in a hue band within a depth window, largest first (blobs < 100 px dropped once one exists)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = (hue_mask(hsv, hue, min_sat, min_val, max_sat) & (depth > min_mm) & (depth < max_mm)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    blobs = []
    for idx in 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA]):
        if stats[idx, cv2.CC_STAT_AREA] < 100 and blobs:
            break
        v, u = np.nonzero(lab == idx)
        pts = backproject(u, v, depth[v, u], K)
        blobs.append(Blob(np.median(pts, 0), (float(u.mean()), float(v.mean())), int(len(u)), pts, (u, v)))
    return blobs


def front_point(blob: Blob, band_mm=12.0) -> np.ndarray:
    """3D centre of a blob's front surface: the pixels within band_mm of its near depth. A small blob's rim mixes
    in the background behind it, which drags a plain median several mm (and sometimes tens of mm) away."""
    z = blob.pts[:, 2]
    keep = z <= np.percentile(z, 30) + band_mm
    return np.median(blob.pts[keep], 0)


def nearest_blob(blobs, prior=None, gate_mm=None, min_area=100) -> Blob | None:
    """The blob nearest the last known 3D position (within gate_mm), else the largest when there is no prior.

    Never "largest" once a prior exists: in the field a passer-by's bigger same-hue object hijacked the
    tracked tape and made a whole calibration run look like the arm had not moved.
    """
    blobs = [b for b in blobs if b.area >= min_area]
    if not blobs:
        return None
    if prior is None:
        return max(blobs, key=lambda b: b.area)
    prior = np.asarray(prior, float)
    best = min(blobs, key=lambda b: np.linalg.norm(b.p - prior))
    if gate_mm is not None and np.linalg.norm(best.p - prior) > gate_mm:
        return None
    return best


def plane_basis(n) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes (e1, e2) with e1 x e2 = n."""
    a = np.array([1.0, 0.0, 0.0])
    if abs(n @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def fit_circle(xy) -> tuple[float, float, float, float]:
    """Algebraic (Kasa) circle fit: (cx, cy, r, rms radial residual)."""
    x, y = xy[:, 0], xy[:, 1]
    c = np.linalg.lstsq(np.stack([x, y, np.ones_like(x)], 1), x**2 + y**2, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    r = float(np.sqrt(max(c[2] + cx**2 + cy**2, 0.0)))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), r, rms


def fit_rim(xy, bins=36, pct=97.0, rounds=2) -> tuple[float, float, float, float]:
    """Circle through the outer edge of a disk-like point set: a lid seen from above plus some of its wall.

    A Kasa fit over every point of a filled lid returns about R/sqrt(2) and drags the centre toward the
    visible wall (artist's field scan reported a 22.7 mm "radius" for a ~33 mm can; the rendered bench shows
    a 4.7 mm centre bias toward the camera). Fitting the outermost points of each angular sector does not.
    """
    xy = np.asarray(xy, float)
    centre = np.median(xy, 0)
    for _ in range(rounds):
        rel = xy - centre
        sector = (np.floor((np.arctan2(rel[:, 1], rel[:, 0]) + np.pi) / (2 * np.pi) * bins).astype(int)) % bins
        radius = np.hypot(rel[:, 0], rel[:, 1])
        rim = []
        for s in range(bins):
            sel = sector == s
            if sel.sum() >= 5:
                r_s = radius[sel]
                rim.append(xy[sel][r_s >= np.percentile(r_s, pct)].mean(0))
        if len(rim) < bins // 3:
            return fit_circle(xy)
        cx, cy, r, rms = fit_circle(np.array(rim))
        centre = np.array([cx, cy])
    return cx, cy, r, rms


@dataclass
class CanLid:
    center: np.ndarray  # lid centre, camera mm
    radius_mm: float
    height_mm: float  # highest point above the table
    px: tuple[float, float]  # centroid of the standing blob in the image
    saturation: float  # mean HSV saturation of the blob (a red can is strongly saturated)
    rms_mm: float  # circle-fit residual
    table: Plane


def locate_can(rgb, depth, K, min_h=60.0, max_h=200.0, lid_band=15.0, radius=33.0, height=122.0,
               seed=0) -> CanLid | None:
    """A standing can's lid from one aligned RGB-D frame (defaults: 355 ml can, r 33 mm, 122 mm tall).

    Fits the bench plane, keeps whatever stands min_h..max_h above it, takes the top lid_band of each
    blob, fits its rim circle in true plane coordinates, and returns the most can-like blob.
    """
    v, u = np.nonzero((depth > 200) & (depth < 1600))
    if len(u) < 1000:
        return None
    pts = backproject(u, v, depth[v, u], K)
    plane = table_plane(pts, seed)
    h = plane.height(pts)
    tall = (h > min_h) & (h < max_h)
    mask = np.zeros(depth.shape, np.uint8)
    mask[v[tall], u[tall]] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nlab, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    e1, e2 = plane_basis(plane.n)
    cands = []
    for idx in 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:6]:
        if stats[idx, cv2.CC_STAT_AREA] < 400:
            continue
        sel = lab[v, u] == idx
        p, ph = pts[sel], h[sel]
        if len(p) < 50:
            continue
        top = p[ph >= ph.max() - lid_band]
        cx, cy, r, rms = fit_rim(np.stack([top @ e1, top @ e2], 1))
        center = cx * e1 + cy * e2 + plane.n * float(np.median(top @ plane.n))
        cands.append(CanLid(center, r, float(ph.max()), (float(cent[idx][0]), float(cent[idx][1])),
                            float(sat[lab == idx].mean()), rms, plane))
    if not cands:
        return None
    return min(cands, key=lambda c: abs(c.radius_mm - radius) / radius + abs(c.height_mm - height) / height
               - c.saturation / 500)


@dataclass
class BenchObject:
    """Something standing on the bench, as one connected region of the depth image above the bench plane."""

    centroid: np.ndarray  # camera mm, median of its points
    top_center: np.ndarray  # camera mm: the rim-fit centre of a round top, else the median of its top band
    height_mm: float  # highest point above the bench
    length_mm: float  # footprint extent on the bench along its long axis (2..98 percentile)
    width_mm: float
    radius_mm: float | None  # set when the top is round (rim fit residual < 15% of the radius)
    px: tuple[float, float]  # image centroid
    bbox: tuple[int, int, int, int]  # x, y, w, h in the image
    n_points: int
    mean_rgb: tuple[int, int, int]
    hue: int  # OpenCV hue (0..179) of the mean colour
    saturation: int


def find_objects(rgb, depth, K, plane: Plane | None = None, min_h=8.0, max_h=300.0, min_px=150, exclude=None,
                 max_objects=12, seed=0) -> tuple[list[BenchObject], Plane]:
    """Everything standing min_h..max_h above the bench, largest first. exclude(points_cam) -> bool mask drops points
    (e.g. the robot arm itself) before segmentation. Returns (objects, bench plane)."""
    v, u = np.nonzero((depth > 150) & (depth < 1600))
    pts = backproject(u, v, depth[v, u], K)
    plane = plane or table_plane(pts, seed)
    h = plane.height(pts)
    keep = (h > min_h) & (h < max_h)
    if exclude is not None and keep.any():
        keep[keep] &= ~exclude(pts[keep])
    mask = np.zeros(depth.shape, np.uint8)
    mask[v[keep], u[keep]] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nlab, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    index = np.full(depth.shape, -1, np.int64)
    index[v, u] = np.arange(len(u))
    e1, e2 = plane_basis(plane.n)
    hsv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    found = []
    for k in 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA]):
        if stats[k, cv2.CC_STAT_AREA] < min_px or len(found) >= max_objects:
            break
        rows, cols = np.nonzero(lab == k)
        ids = index[rows, cols]
        ids = ids[ids >= 0]
        if len(ids) < min_px // 2:
            continue
        P, H = pts[ids], h[ids]
        top = P[H >= H.max() - 15.0]
        xy = np.stack([P @ e1, P @ e2], 1)
        c2 = np.median(xy, 0)
        axis = np.linalg.svd(xy - c2, full_matrices=False)[2]
        along, across = (xy - c2) @ axis[0], (xy - c2) @ axis[1]
        radius = None
        top_center = np.median(top, 0)
        length = float(np.percentile(along, 98) - np.percentile(along, 2))
        width = float(np.percentile(across, 98) - np.percentile(across, 2))
        if len(top) >= 30 and length <= 1.6 * max(width, 1.0):  # a long thin top (a lying pen) is not a round one
            cx, cy, r, rms = fit_rim(np.stack([top @ e1, top @ e2], 1))
            if 8.0 <= r <= 120.0 and rms < 0.15 * r and 0.6 <= 2 * r / max(length, 1.0) <= 1.5:
                radius = r
                top_center = cx * e1 + cy * e2 + plane.n * float(np.median(top @ plane.n))
        mean = rgb[rows, cols].mean(0)
        hsv = cv2.cvtColor(np.uint8([[mean]]), cv2.COLOR_RGB2HSV)[0, 0]
        x, y, w, hh = (int(s) for s in stats[k, :4])
        found.append(BenchObject(
            centroid=np.median(P, 0), top_center=top_center, height_mm=float(H.max()), length_mm=length,
            width_mm=width, radius_mm=radius,
            px=(float(cent[k][0]), float(cent[k][1])), bbox=(x, y, w, hh), n_points=int(len(ids)),
            mean_rgb=tuple(int(c) for c in mean), hue=int(hsv[0]), saturation=int(hsv_img[rows, cols, 1].mean())))
    return found, plane


def segment_distance(points, a, b) -> np.ndarray:
    """Distance of each point to the segment a-b (all in the same frame)."""
    ab = np.asarray(b, float) - np.asarray(a, float)
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (np.asarray(a, float) + t[:, None] * ab), axis=1)


def pixel_point(depth, K, u, v, window=5, plane: Plane | None = None) -> tuple[np.ndarray | None, str]:
    """3D camera point at pixel (u, v): median depth over a window, else the pixel ray on the bench plane."""
    r = window // 2
    patch = depth[max(0, int(v) - r):int(v) + r + 1, max(0, int(u) - r):int(u) + r + 1]
    z = patch[(patch > 150) & (patch < 5000)]
    if z.size:
        return backproject(float(u), float(v), float(np.median(z)), K), "depth"
    if plane is not None:
        return ray_plane(np.array(float(u)), np.array(float(v)), plane.n, plane.d, K), "bench plane (no depth here)"
    return None, "no depth at this pixel"


@dataclass
class PlaneFrame:
    """Plane-aligned frame: (x, y) in the plane, h = height above it; all camera-frame mm (artist Frame)."""

    ctr: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    n: np.ndarray
    d: float

    @classmethod
    def on(cls, plane: Plane, origin) -> "PlaneFrame":
        """Frame on `plane`, origin = `origin` dropped onto it, e1 = camera -x (image left) laid into the plane."""
        scale = np.linalg.norm(plane.n)  # stored normals are rounded, so not exactly unit
        n, d = np.asarray(plane.n, float) / scale, float(plane.d) / scale
        image_left = np.array([-1.0, 0.0, 0.0])
        e1 = image_left - n * (n @ image_left)
        e1 /= np.linalg.norm(e1)
        o = np.asarray(origin, float)
        return cls(o - n * (n @ o - d), e1, np.cross(n, e1), n, d)

    def plane(self, p) -> np.ndarray:
        """Camera mm (..., 3) -> (x, y, h)."""
        p = np.asarray(p, float)
        q = p - self.ctr
        return np.stack([q @ self.e1, q @ self.e2, p @ self.n - self.d], -1)

    def to_cam(self, xyh) -> np.ndarray:
        xyh = np.asarray(xyh, float)
        x, y, h = xyh[..., 0:1], xyh[..., 1:2], xyh[..., 2:3]
        return self.ctr + x * self.e1 + y * self.e2 + (h - (self.n @ self.ctr - self.d)) * self.n


def largest_cluster(xy, cell=3.0) -> np.ndarray:
    """Indices of the largest 8-connected cluster of points binned on a `cell` mm grid."""
    g = np.floor(xy / cell).astype(int)
    cells: dict[tuple, list[int]] = {}
    for i, k in enumerate(map(tuple, g)):
        cells.setdefault(k, []).append(i)
    seen, best = set(), []
    for k in cells:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            a = stack.pop()
            comp += cells[a]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    b = (a[0] + dx, a[1] + dy)
                    if b in cells and b not in seen:
                        seen.add(b)
                        stack.append(b)
        if len(comp) > len(best):
            best = comp
    return np.asarray(best, int)


def unit2(v) -> np.ndarray:
    v = np.asarray(v, float)[:2]
    return v / max(np.linalg.norm(v), 1e-9)


def locate_lying(Q, guess, base=None, radius=60.0, hmin=6.0, hmax=28.0, exclude_above=40.0, cell=3.0,
                 min_pts=30, axis=None, band=8.0) -> dict | None:
    """A long object lying near plane-xy `guess`, from plane points Q (N, 3) = (x, y, h) - shape only.

    Returns dict(center, axis, length, width, top, n, tall_nearby, near_end, far_end[, r]) or None when fewer
    than min_pts points sit in the hmin..hmax band. With an `axis` prior the search is a strip (|along| <
    radius, |lateral| < band) so neighbours packed alongside are excluded.
    """
    Q = np.asarray(Q, float)
    rel = Q[:, :2] - np.asarray(guess, float)[:2]
    if axis is None:
        near = np.linalg.norm(rel, axis=1) < radius
    else:
        a = unit2(axis)
        near = (np.abs(rel @ a) < radius) & (np.abs(rel @ np.array([-a[1], a[0]])) < band)
    pts = Q[near & (Q[:, 2] > hmin) & (Q[:, 2] < hmax)]
    tall = int((near & (Q[:, 2] >= exclude_above)).sum())
    if len(pts) < min_pts:
        return None
    q = pts[largest_cluster(pts[:, :2], cell)]
    ctr = q[:, :2].mean(0)
    ax = np.linalg.svd(q[:, :2] - ctr, full_matrices=False)[2][0]
    perp = np.array([-ax[1], ax[0]])
    proj, side = (q[:, :2] - ctr) @ ax, (q[:, :2] - ctr) @ perp
    lo, hi = np.percentile(proj, 2), np.percentile(proj, 98)
    slo, shi = np.percentile(side, 2), np.percentile(side, 98)
    # centre of the 2..98 % extent, not the mean: the camera sees more of the near end and of one side
    ctr = ctr + ax * (lo + hi) / 2 + perp * (slo + shi) / 2
    length = float(hi - lo)
    out = dict(center=ctr, axis=ax, length=length, width=float(shi - slo),
               top=float(np.percentile(q[:, 2], 95)), n=int(len(q)), tall_nearby=tall)
    ends = [ctr + ax * length / 2, ctr - ax * length / 2]
    if base is not None:
        base = np.asarray(base, float)[:2]
        ends.sort(key=lambda e: np.linalg.norm(e - base))
        out["r"] = float(np.linalg.norm(ctr - base))
    out["near_end"], out["far_end"] = ends
    return out


def normalize_brightness(img, mean=90.0) -> np.ndarray:
    """Scale the image so its mean is `mean`: makes hue/sat/value thresholds work in a dark room."""
    g = img.astype(np.float32)
    return np.clip(g / max(float(g.mean()), 1.0) * mean, 0, 255).astype(np.uint8)


def find_lying(rgb, depth, K, frame: PlaneFrame, color, region=None, min_area=40, h_band=(3.0, 60.0),
               shape_radius=35.0, base=None) -> dict | None:
    """Dark-robust colour find + depth-shape refinement of an object lying on the plane (artist find_marker).

    color: a LYING_BANDS key. region: ((x0, x1), (y0, y1)) plane-xy window or None. Returns dict(blob, shape,
    pose) with pose = dict(center, axis, top, source) taken from the shape when found, else from the colour.
    """
    hsv = cv2.cvtColor(normalize_brightness(rgb), cv2.COLOR_RGB2HSV)
    (h0, h1), smin = LYING_BANDS[color]
    m = (hue_mask(hsv, (h0, h1), smin, 60) & (depth > 250) & (depth < 900)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    blob = None
    for k in 1 + np.argsort(-st[1:, cv2.CC_STAT_AREA])[:6]:
        if st[k, cv2.CC_STAT_AREA] < min_area:
            continue
        v, u = np.nonzero(lab == k)
        q = frame.plane(backproject(u, v, depth[v, u], K))
        xyh = np.median(q, 0)
        inside = region is None or (region[0][0] < xyh[0] < region[0][1] and region[1][0] < xyh[1] < region[1][1])
        if inside and h_band[0] < xyh[2] < h_band[1]:
            ax = np.linalg.svd(q[:, :2] - q[:, :2].mean(0), full_matrices=False)[2][0]
            blob = dict(c=xyh, axis2=ax, area=int(st[k, cv2.CC_STAT_AREA]))
            break
    if blob is None:
        return None
    v, u = np.nonzero(depth > 0)
    Q = frame.plane(backproject(u, v, depth[v, u], K))
    shape = locate_lying(Q, blob["c"][:2], base, radius=shape_radius)
    if shape is not None:  # second pass along the first axis, so a clipped end does not bias the centre
        full = locate_lying(Q, shape["center"], base, radius=90.0, axis=shape["axis"])
        if full is not None and full["length"] >= shape["length"]:
            shape = full
    if shape is not None:
        pose = dict(center=shape["center"], axis=shape["axis"], top=shape["top"], source="shape")
    else:
        pose = dict(center=blob["c"][:2], axis=blob["axis2"], top=float(blob["c"][2]) + 5.0, source="color")
    return dict(blob=blob, shape=shape, pose=pose)


def wrist_blob(rgb, hue=(170, 14), min_sat=110) -> dict | None:
    """Wrist camera: largest blob of a hue - centroid, axis angle (deg from image vertical), edge contact."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = hue_mask(hsv, hue, min_sat, 80).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    v, u = np.nonzero(lab == k)
    X = np.stack([u, v], 1).astype(float)
    ax = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
    ang = (float(np.degrees(np.arctan2(ax[0], ax[1]))) + 90) % 180 - 90  # 0 = vertical in the image
    edge = u.min() < 3 or v.min() < 3 or u.max() > rgb.shape[1] - 4 or v.max() > rgb.shape[0] - 4
    return dict(cx=float(u.mean()), cy=float(v.mean()), ang=ang, area=int(len(u)), edge=bool(edge),
                w=rgb.shape[1], h=rgb.shape[0])
