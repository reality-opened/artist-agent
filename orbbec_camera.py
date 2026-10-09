"""Orbbec Gemini 2 as a LeRobot camera: RGB plus depth aligned to the color image, in millimetres.

Ported from reality-opened/artist camera_stream.py (authors galois777, David Zhang, Hillysilly), keeping
its field fixes: raw Y16 depth is requested explicitly (the Gemini 2 default is RLE-compressed), depth is
aligned to color with AlignFilter, and a pipeline that hit a device error is kept referenced forever,
because destroying it segfaults in uvc_close.

pyorbbecsdk is imported lazily: its native module prints to stdout when loaded, which would corrupt an
MCP stdio stream unless the server has already diverted fd 1.
"""

import logging
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from lerobot.cameras import CameraConfig
from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)

DEPTH_CANDIDATES = ((640, 400), (0, 0))  # Y16 depth modes to try, (0, 0) = any size
_leaked: list[tuple] = []  # pipelines that saw a device error: never destroyed (uvc_close segfault)


def _sdk():
    import pyorbbecsdk

    return pyorbbecsdk


def decode_color(data: np.ndarray, fmt: str, width: int, height: int) -> np.ndarray:
    """One Orbbec color frame (raw bytes, OBFormat name) -> BGR uint8 (H, W, 3)."""
    if fmt == "MJPG":
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("MJPG frame failed to decode")
        return image
    if fmt == "BGR":
        return data.reshape(height, width, 3)
    if fmt == "RGB":
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if fmt in ("YUYV", "UYVY"):
        code = cv2.COLOR_YUV2BGR_YUY2 if fmt == "YUYV" else cv2.COLOR_YUV2BGR_UYVY
        return cv2.cvtColor(data.reshape(height, width, 2), code)
    planar = {"I420": cv2.COLOR_YUV2BGR_I420, "NV12": cv2.COLOR_YUV2BGR_NV12, "NV21": cv2.COLOR_YUV2BGR_NV21}
    if fmt in planar:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), planar[fmt])
    raise RuntimeError(f"Unsupported color format: {fmt}")


def depth_to_mm(raw: bytes | np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
    """Y16 depth buffer -> uint16 millimetres (H, W); 0 = no reading."""
    values = np.frombuffer(raw, dtype=np.uint16)
    if values.size != width * height:
        raise RuntimeError(f"depth buffer has {values.size} values, expected {width}x{height}")
    mm = values.reshape(height, width).astype(np.float32) * scale
    return np.clip(mm, 0, 65535).astype(np.uint16)


@CameraConfig.register_subclass("orbbec")
@dataclass
class OrbbecCameraConfig(CameraConfig):
    """Gemini 2 color (MJPG, default 1280x720@30) + Y16 depth aligned to it.

    serial_number: device serial (None = first device). width/height/fps pick the color mode; the aligned
    depth image has the same size. color_mode: RGB (LeRobot default) or BGR.
    """

    serial_number: str | None = None
    color_mode: ColorMode = ColorMode.RGB
    use_rgb: bool = True
    use_depth: bool = True
    warmup_s: float = 1.0

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of `use_rgb` or `use_depth` must be enabled.")
        values = (self.fps, self.width, self.height)
        if all(v is None for v in values):
            self.fps, self.width, self.height = 30, 1280, 720  # LeRobot robots require all three
        elif any(v is None for v in values):
            raise ValueError("For `fps`, `width` and `height`, either all of them need to be set, or none of them.")


class OrbbecCamera(Camera):
    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.config = config
        self.use_rgb, self.use_depth, self.color_mode = config.use_rgb, config.use_depth, config.color_mode
        self.intrinsics: dict[str, float] | None = None  # color intrinsics, valid for the aligned depth too
        self._refs: tuple | None = None  # (context, device, pipeline, align)
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.error: Exception | None = None
        self.frame_lock = Lock()
        self.new_frame_event = Event()
        self.latest_color_frame: np.ndarray | None = None
        self.latest_depth_frame: np.ndarray | None = None
        self.latest_timestamp: float | None = None

    def __str__(self) -> str:
        return f"OrbbecCamera({self.config.serial_number or 'first device'})"

    @property
    def is_connected(self) -> bool:
        return self._refs is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        ob = _sdk()
        context = ob.Context()  # must outlive the device list, or the device manager is freed
        devices = context.query_devices()
        found = []
        for i in range(devices.get_count()):
            info = devices.get_device_by_index(i).get_device_info()
            found.append(dict(type="Orbbec", name=info.get_name(), id=info.get_serial_number(),
                              firmware=info.get_firmware_version()))
        return found

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        ob = _sdk()
        ob.Context.set_logger_to_console(ob.OBLogLevel.FATAL)  # depth timestamp-anomaly spam
        context = ob.Context()  # must outlive the device list, or the device manager is freed
        devices = context.query_devices()
        if devices.get_count() == 0:
            raise DeviceNotConnectedError("No Orbbec device detected.")
        serial = self.config.serial_number
        device = devices.get_device_by_serial_number(serial) if serial else devices.get_device_by_index(0)
        pipeline = ob.Pipeline(device)
        color_profile = self._profile(pipeline, ob.OBSensorType.COLOR_SENSOR,
                                      [(self.width, self.height, ob.OBFormat.MJPG, self.fps)])
        depth_profile = self._profile(pipeline, ob.OBSensorType.DEPTH_SENSOR,
                                      [(w, h, ob.OBFormat.Y16, self.fps) for w, h in DEPTH_CANDIDATES])
        config = ob.Config()
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        pipeline.enable_frame_sync()
        align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        pipeline.start(config)
        k = color_profile.get_intrinsic()
        self.intrinsics = dict(fx=k.fx, fy=k.fy, cx=k.cx, cy=k.cy, width=k.width, height=k.height)
        self.width, self.height, self.fps = color_profile.get_width(), color_profile.get_height(), color_profile.get_fps()
        self._refs = (context, device, pipeline, align)
        self.error = None
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()
        if warmup and not self.new_frame_event.wait(timeout=self.config.warmup_s + 5.0):
            error = self.error
            self.disconnect()
            raise TimeoutError(f"{self} produced no aligned color+depth frame: {error}")

    @staticmethod
    def _profile(pipeline, sensor, candidates):
        profiles = pipeline.get_stream_profile_list(sensor)
        for width, height, fmt, fps in candidates:
            try:
                return profiles.get_video_stream_profile(width, height, fmt, fps)
            except Exception:
                continue
        return profiles.get_default_video_stream_profile()

    def _read_loop(self) -> None:
        _, _, pipeline, align = self._refs
        last = time.monotonic()
        while not self.stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames(500)
                color = depth = None
                if frames is not None and frames.get_color_frame() and frames.get_depth_frame():
                    aligned = align.process(frames)
                    aligned = aligned.as_frame_set() if aligned else None
                    if aligned:
                        color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
                if not (color and depth):
                    if time.monotonic() - last > 5:
                        raise RuntimeError("no aligned color+depth frames for five seconds")
                    continue
                bgr = decode_color(np.frombuffer(color.get_data(), dtype=np.uint8), color.get_format().name,
                                   color.get_width(), color.get_height())
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if self.color_mode is ColorMode.RGB else bgr
                mm = depth_to_mm(depth.get_data(), depth.get_width(), depth.get_height(), depth.get_depth_scale())
                with self.frame_lock:
                    if self.stop_event.is_set():
                        break
                    self.latest_color_frame, self.latest_depth_frame = image, mm[..., np.newaxis]
                    self.latest_timestamp = time.perf_counter()
                self.new_frame_event.set()
                last = time.monotonic()
            except Exception as e:
                self.error = e
                logger.error(f"{self} read loop stopped: {e!r}")
                return

    def _latest(self, max_age_ms: float, depth: bool) -> np.ndarray:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running: {self.error!r}")
        with self.frame_lock:
            frame = self.latest_depth_frame if depth else self.latest_color_frame
            stamp = self.latest_timestamp
        if frame is None or stamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age_ms = (time.perf_counter() - stamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is {age_ms:.0f} ms old (max {max_age_ms} ms).")
        return frame

    def _next(self, timeout_ms: float, depth: bool) -> np.ndarray:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running: {self.error!r}")
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for a frame from {self} after {timeout_ms} ms.")
        with self.frame_lock:
            frame = self.latest_depth_frame if depth else self.latest_color_frame
            self.new_frame_event.clear()
        return frame

    @check_if_not_connected
    def read(self) -> np.ndarray:
        return self._next(10000, depth=False)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        return self._next(timeout_ms, depth=False)

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        return self._latest(max_age_ms, depth=False)

    @check_if_not_connected
    def async_read_depth(self, timeout_ms: float = 200) -> np.ndarray:
        """Next new depth frame, uint16 millimetres (H, W, 1), aligned to the color image."""
        return self._next(timeout_ms, depth=True)

    @check_if_not_connected
    def read_latest_depth(self, max_age_ms: int = 500) -> np.ndarray:
        return self._latest(max_age_ms, depth=True)

    @check_if_not_connected
    def read_rgbd(self, timeout_ms: float = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Next new (color, depth mm (H, W)) pair from the same frame set."""
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for a frame from {self} after {timeout_ms} ms.")
        with self.frame_lock:
            color, depth = self.latest_color_frame, self.latest_depth_frame
            self.new_frame_event.clear()
        if color is None or depth is None:
            raise RuntimeError(f"{self} read thread is not running: {self.error!r}")
        return color, depth[..., 0]

    def disconnect(self) -> None:
        if self._refs is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        refs, self._refs = self._refs, None
        try:
            refs[2].stop()
        except Exception as e:
            logger.warning(f"{self} pipeline.stop() failed: {e!r}")
        if self.error is not None:
            _leaked.append(refs)
        self.thread = self.stop_event = None
        with self.frame_lock:
            self.latest_color_frame = self.latest_depth_frame = self.latest_timestamp = None
            self.new_frame_event.clear()
