"""RealSense D405 capture -- replaces realsense2_camera + cv_bridge + the CameraInfo topic.

Also fixes, by construction, the pose/image time-association problem that the ROS stack solved
with a tf2 lookup at `msg.header.stamp`. SAM3 takes 1-2 s per frame, so by the time a detection
exists the arm may have moved; fusing it against the arm's CURRENT pose would attribute the
detection to the wrong viewpoint and corrupt the triangulation.

`capture()` therefore returns a Frame that carries the camera pose SAMPLED AT CAPTURE. There is
no later lookup to get wrong, and no tf cache to expire. The pose travels WITH the pixels.
"""

import time

import numpy as np

from .. import log as urlog

log = urlog.get('camera')


class Frame:
    """One captured image plus everything needed to interpret it geometrically."""

    def __init__(self, color, K, D, stamp, T_base_cam=None, depth=None):
        self.color = color              # HxWx3 uint8, BGR
        self.K = K                      # 3x3 intrinsics
        self.D = D                      # distortion coefficients
        self.stamp = stamp              # time.monotonic() at capture
        self.T_base_cam = T_base_cam    # camera pose in base_link AT CAPTURE -- the whole point
        self.depth = depth

    @property
    def rgb(self):
        return self.color[:, :, ::-1]

    @property
    def age(self):
        return time.monotonic() - self.stamp


class RealSenseCamera:
    """Blocking capture from a D405. Pass `pose_fn` (usually robot.camera) to stamp each frame
    with the camera pose at the moment of capture."""

    def __init__(self, cfg, pose_fn=None):
        c = cfg.section('camera')
        self.serial = str(c.get('serial_no', '')) or None
        self.width = int(c.get('width', 1280))
        self.height = int(c.get('height', 720))
        self.fps = int(c.get('fps', 30))
        self.enable_depth = bool(c.get('enable_depth', False))
        self.pose_fn = pose_fn
        self.dry_run = bool(cfg.get_path('robot.dry_run', False))

        if self.dry_run:
            log.warning('DRY RUN: the camera returns a black frame.')
            self.pipeline = None
            self.K = np.array([[600.0, 0, self.width / 2],
                               [0, 600.0, self.height / 2],
                               [0, 0, 1.0]])
            self.D = np.zeros(5)
            return

        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError('pyrealsense2 is not installed. `pip install pyrealsense2`') from exc
        self._rs = rs

        self.pipeline = rs.pipeline()
        profile = self._start_color(rs)

        cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = cp.get_intrinsics()
        # Read the ACTUAL resolution back from the device -- after a fallback it may differ from
        # what the config asked for, and K/width/height must describe what we're really getting.
        self.width, self.height, self.fps = cp.width(), cp.height(), cp.fps()
        self.K = np.array([[intr.fx, 0.0, intr.ppx],
                           [0.0, intr.fy, intr.ppy],
                           [0.0, 0.0, 1.0]])
        self.D = np.asarray(intr.coeffs, dtype=float)
        log.info('D405 %s streaming %dx%d @ %d fps (fx=%.1f, fy=%.1f).',
                 self.serial or '(first found)', self.width, self.height, self.fps,
                 intr.fx, intr.fy)

        for _ in range(5):              # auto-exposure settle
            self.pipeline.wait_for_frames()

    def _start_color(self, rs):
        """Start the pipeline with the configured color mode, falling back to a device-supported
        one if the exact request is not offered.

        The D405's color sensor exposes a specific set of (resolution, fps, format) profiles, and
        an unsupported combination makes pipeline.start() raise 'Couldn't resolve requests'. So we
        try the configured mode, then let librealsense pick its DEFAULT color profile, and only
        then give up -- with the list of modes the device actually supports."""
        def base_config():
            cfg = rs.config()
            if self.serial:
                cfg.enable_device(self.serial)
            if self.enable_depth:
                cfg.enable_stream(rs.stream.depth, rs.format.z16, self.fps)
            return cfg

        # 1. Exactly what the config asked for.
        cfg = base_config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        try:
            return self.pipeline.start(cfg)
        except RuntimeError as exc:
            log.warning("Color %dx%d @ %d fps not supported by this device (%s); falling back to "
                        "its default color mode.", self.width, self.height, self.fps, exc)

        # 2. Let librealsense choose a valid default color profile.
        cfg = base_config()
        cfg.enable_stream(rs.stream.color, rs.format.bgr8)
        try:
            return self.pipeline.start(cfg)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not start the D405 color stream. Supported color modes:\n"
                f"{self._supported_color(rs)}\n"
                f"Set camera.width/height/fps in the config to one of these. Original error: {exc}"
            ) from exc

    def _supported_color(self, rs):
        try:
            devs = rs.context().query_devices()
            if not devs:
                return '  (no device found)'
            modes = sorted({(v.width(), v.height(), p.fps())
                            for s in devs[0].query_sensors()
                            for p in s.get_stream_profiles()
                            if p.stream_type() == rs.stream.color and p.is_video_stream_profile()
                            for v in [p.as_video_stream_profile()]})
            return '\n'.join(f'  {w}x{h} @ {f} fps' for w, h, f in modes) or '  (none)'
        except Exception as exc:                    # noqa: BLE001
            return f'  (could not enumerate: {exc})'

    def capture(self, timeout_ms=5000):
        """One frame, stamped with the camera pose at capture."""
        if self.dry_run:
            color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return Frame(color, self.K, self.D, time.monotonic(),
                         self.pose_fn() if self.pose_fn else None)

        frames = self.pipeline.wait_for_frames(timeout_ms)
        # Sample the pose as close to the capture as we can get. The residual skew is the
        # USB/driver latency (a few ms); at scan speeds the arm has settled anyway, which is the
        # reason the scan holds each view still before detecting.
        T_base_cam = self.pose_fn() if self.pose_fn else None
        stamp = time.monotonic()

        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = None
        if self.enable_depth:
            d = frames.get_depth_frame()
            if d:
                depth = np.asanyarray(d.get_data())
        return Frame(color, self.K, self.D, stamp, T_base_cam, depth)

    def close(self):
        if self.pipeline is not None:
            self.pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
