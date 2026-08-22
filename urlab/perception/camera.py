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
        # RECONNECT. A USB camera that drops out mid-run used to end the run: wait_for_frames
        # raises, and nothing caught it. Losing an hour of a cycle test to a nudged cable is a
        # bad trade for a fault that fixes itself the moment the plug goes back in -- so the
        # default is to WAIT, indefinitely, and carry on where it left off.
        rc = c.get('reconnect', {}) or {}
        self.reconnect_enabled = bool(rc.get('enabled', True))
        self.reconnect_interval_s = float(rc.get('retry_interval_s', 2.0))
        self.reconnect_max_wait_s = float(rc.get('max_wait_s', 0.0))   # 0 = forever

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
            log.warning("Color %dx%d @ %d fps not supported by this device (%s); auto-selecting a "
                        "supported mode.", self.width, self.height, self.fps, exc)

        # 2. Auto-select from the modes the device ACTUALLY reports -- not librealsense's "default"
        #    profile, which for the D405 is 30 fps and does not actually start. Highest resolution
        #    first, then highest fps at that resolution: more pixels sharpen the cable fit, and the
        #    scan holds each view still so fps barely matters. (The D405 tops out at 1280x720 @ 15.)
        modes = self._color_modes(rs)
        if not modes:
            raise RuntimeError('The D405 reports no color profiles (a depth-only unit?).')
        w, h, f = max(modes, key=lambda m: (m[0] * m[1], m[2]))
        log.info('Auto-selected color mode %dx%d @ %d fps.', w, h, f)
        cfg = base_config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
        try:
            return self.pipeline.start(cfg)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not start the D405 color stream even at its own reported {w}x{h} @ {f} fps."
                f" Supported color modes:\n{self._fmt_modes(modes)}\nOriginal error: {exc}") from exc

    def _color_modes(self, rs):
        """Sorted [(w, h, fps)] color modes the connected device supports."""
        try:
            devs = rs.context().query_devices()
            if not devs:
                return []
            return sorted({(v.width(), v.height(), p.fps())
                           for s in devs[0].query_sensors()
                           for p in s.get_stream_profiles()
                           if p.stream_type() == rs.stream.color and p.is_video_stream_profile()
                           for v in [p.as_video_stream_profile()]})
        except Exception:                           # noqa: BLE001
            return []

    @staticmethod
    def _fmt_modes(modes):
        return '\n'.join(f'  {w}x{h} @ {f} fps' for w, h, f in modes) or '  (none)'

    def capture(self, timeout_ms=5000, reconnect=None):
        """One frame, stamped with the camera pose at capture.

        SURVIVES A DISCONNECT. If the device drops out, this waits for it to come back and
        returns the frame it was asked for, rather than raising and taking the run with it.
        `reconnect=False` restores the old raise-immediately behaviour, which is what a
        best-effort BACKGROUND capture wants: a recorder thread that blocks forever on a
        vanished camera would sit there holding its `with` block open long after the grasp it
        was recording finished."""
        if self.dry_run:
            color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return Frame(color, self.K, self.D, time.monotonic(),
                         self.pose_fn() if self.pose_fn else None)
        wait = self.reconnect_enabled if reconnect is None else bool(reconnect)
        try:
            return self._capture_once(timeout_ms)
        except Exception as exc:                        # noqa: BLE001 -- any device fault
            if not wait:
                raise
            log.error('CAMERA LOST (%s). Waiting for it to come back -- the run is PAUSED here, '
                      'the arm is holding position, and nothing is retried until there is a '
                      'frame again.', exc)
        self._await_device()
        return self._capture_once(timeout_ms)

    def _await_device(self):
        """Block until the pipeline yields a frame again, restarting it each attempt.

        RE-READS THE INTRINSICS AND REFUSES A DIFFERENT ONE. A device that comes back at another
        resolution -- or a DIFFERENT CAMERA appearing on the bus when no serial_no is pinned --
        has a different K, and every pose this run has produced or will produce is measured
        through K. Carrying on with the wrong one would not fail; it would quietly return wrong
        answers, which is worse than the disconnect."""
        rs = self._rs
        K0, size0 = np.array(self.K, dtype=float), (self.width, self.height)
        t0, n = time.monotonic(), 0
        while True:
            n += 1
            waited = time.monotonic() - t0
            if self.reconnect_max_wait_s > 0.0 and waited > self.reconnect_max_wait_s:
                raise RuntimeError(
                    'the camera did not come back within camera.reconnect.max_wait_s '
                    f'({self.reconnect_max_wait_s:.0f} s)')
            try:
                self.pipeline.stop()
            except Exception:                           # noqa: BLE001 -- already down
                pass
            time.sleep(self.reconnect_interval_s)
            try:
                self.pipeline = rs.pipeline()
                profile = self._start_color(rs)
                cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = cp.get_intrinsics()
                self.width, self.height, self.fps = cp.width(), cp.height(), cp.fps()
                self.K = np.array([[intr.fx, 0.0, intr.ppx],
                                   [0.0, intr.fy, intr.ppy],
                                   [0.0, 0.0, 1.0]])
                self.D = np.asarray(intr.coeffs, dtype=float)
                for _ in range(5):                      # let auto-exposure settle again
                    self.pipeline.wait_for_frames()
            except Exception as exc:                    # noqa: BLE001 -- still gone
                if n == 1 or n % 15 == 0:
                    log.warning('  camera still down after %.0f s (%s) -- still waiting.',
                                waited, exc)
                continue
            if (self.width, self.height) != size0 or not np.allclose(self.K, K0, atol=1e-6):
                raise RuntimeError(
                    'the camera came back DIFFERENT: %dx%d fx=%.1f fy=%.1f, was %dx%d fx=%.1f '
                    'fy=%.1f. Every pose is measured through K, so continuing would return '
                    'wrong answers rather than fail. Pin camera.serial_no if more than one '
                    'device is on the bus, and re-check the mode.'
                    % (self.width, self.height, self.K[0, 0], self.K[1, 1],
                       size0[0], size0[1], K0[0, 0], K0[1, 1]))
            log.info('CAMERA BACK after %.0f s (%dx%d, fx=%.1f) -- resuming.',
                     time.monotonic() - t0, self.width, self.height, self.K[0, 0])
            return

    def _capture_once(self, timeout_ms):
        """One frame, no reconnect handling. Raises if the device is not there."""
        # DRAIN the pipeline's buffered frames first, THEN wait for a genuinely new one. The
        # pipeline keeps producing frames at 15 fps while the arm moves and settles, so a plain
        # wait_for_frames() can hand back a STALE frame captured mid-move -- which we would then
        # label with the current (settled) pose, corrupting the ray geometry the fusion depends on.
        # (The ROS stack avoided this differently: it looked the pose up at the image's OWN
        # timestamp via tf2. Here we instead guarantee the frame is fresh, so "pose now" == "pose of
        # this frame".) Draining discards the mid-move backlog; the wait then returns a frame
        # exposed after the arm has come to rest, matching the pose we sample right after.
        while self.pipeline.poll_for_frames():
            pass
        frames = self.pipeline.wait_for_frames(timeout_ms)
        T_base_cam = self.pose_fn() if self.pose_fn else None
        stamp = time.monotonic()

        cf = frames.get_color_frame()
        if not cf:
            raise RuntimeError('the pipeline returned no color frame')
        color = np.asanyarray(cf.get_data())
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
