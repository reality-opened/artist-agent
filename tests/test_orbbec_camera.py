"""Orbbec adapter: frame decoding, depth scaling, LeRobot wiring, and connect() with or without a Gemini.

With no Orbbec attached, connect() must fail fast and cleanly; with one attached the same test streams a
real aligned frame pair (the Phase 2 hardware acceptance). Run from the repo root:
.venv\\Scripts\\python.exe -m tests.test_orbbec_camera
"""

import time
from pathlib import Path

import cv2
import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.errors import DeviceNotConnectedError

from orbbec_camera import OrbbecCamera, OrbbecCameraConfig, decode_color, depth_to_mm

DATA = Path(__file__).parent / "data"


def gradient_bgr(h=48, w=64) -> np.ndarray:
    v, u = np.mgrid[0:h, 0:w]
    return np.stack([u * 255 // w, v * 255 // h, (u + v) * 255 // (h + w)], -1).astype(np.uint8)


def test_decode_color_formats() -> None:
    bgr = gradient_bgr()
    h, w = bgr.shape[:2]
    np.testing.assert_array_equal(decode_color(bgr.reshape(-1), "BGR", w, h), bgr)
    np.testing.assert_array_equal(decode_color(bgr[..., ::-1].copy().reshape(-1), "RGB", w, h), bgr)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    assert np.abs(decode_color(jpg.reshape(-1), "MJPG", w, h).astype(int) - bgr).mean() < 3
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    assert np.abs(decode_color(i420.reshape(-1), "I420", w, h).astype(int) - bgr).mean() < 4
    try:
        decode_color(bgr.reshape(-1), "H264", w, h)
    except RuntimeError as e:
        assert "Unsupported" in str(e)
    else:
        raise AssertionError("unsupported format accepted")


def test_depth_scaling_and_size_check() -> None:
    raw = np.array([[0, 1000], [65535, 500]], np.uint16).tobytes()
    np.testing.assert_array_equal(depth_to_mm(raw, 2, 2, 1.0), [[0, 1000], [65535, 500]])
    np.testing.assert_array_equal(depth_to_mm(raw, 2, 2, 0.1), [[0, 100], [6553, 50]])
    try:
        depth_to_mm(raw, 3, 2, 1.0)
    except RuntimeError as e:
        assert "expected 3x2" in str(e)
    else:
        raise AssertionError("wrong-size depth buffer accepted")


def test_lerobot_wiring() -> None:
    gemini = OrbbecCameraConfig(serial_number="AY35C3200SX")
    cams = make_cameras_from_configs({"gemini": gemini})
    assert isinstance(cams["gemini"], OrbbecCamera)
    config = SO101FollowerConfig(port="unused", id="exp23_follower", calibration_dir=DATA,
                                 disable_torque_on_disconnect=False, cameras={"gemini": gemini})
    features = SO101Follower(config).observation_features
    assert features["gemini"] == (720, 1280, 3), features
    assert features["gemini_depth"] == (720, 1280, 1), features
    assert features["shoulder_pan.pos"] is float


def test_connect() -> None:
    found = OrbbecCamera.find_cameras()
    cam = OrbbecCamera(OrbbecCameraConfig())
    if not found:
        t0 = time.monotonic()
        try:
            cam.connect()
        except DeviceNotConnectedError as e:
            assert "No Orbbec device" in str(e)
        else:
            raise AssertionError("connect() succeeded with no Orbbec attached")
        assert time.monotonic() - t0 < 10 and not cam.is_connected
        print("  (no Orbbec attached: checked the clean failure path)")
        return
    cam.connect()
    try:
        color, depth = cam.read_rgbd(timeout_ms=2000)
        assert color.shape == (cam.height, cam.width, 3) and color.dtype == np.uint8
        assert depth.shape == (cam.height, cam.width) and depth.dtype == np.uint16
        assert (depth > 0).mean() > 0.3, "under 30% of pixels have depth"
        assert cam.read_latest_depth().shape == (cam.height, cam.width, 1)
        assert set(cam.intrinsics) >= {"fx", "fy", "cx", "cy"}
        print(f"  streamed {found[0]['id']}: {cam.width}x{cam.height}, {(depth > 0).mean():.0%} depth, K={cam.intrinsics}")
    finally:
        cam.disconnect()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
