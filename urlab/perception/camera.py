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

        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0.0, intr.ppx],
                           [0.0, intr.fy, intr.ppy],
                           [0.0, 0.0, 1.0]])
        self.D = np.asarray(intr.coeffs, dtype=float)
        log.info('D405 %s streaming %dx%d @ %d fps (fx=%.1f, fy=%.1f).',
                 self.serial or '(first found)', self.width, self.height, self.fps,
                 intr.fx, intr.fy)

        for _ in range(5):              # auto-exposure settle
            self.pipeline.wait_for_frames()

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
