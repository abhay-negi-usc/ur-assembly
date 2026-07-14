"""ArUco marker detection -- the pure-OpenCV core of ur_vision_demo, minus the ROS wrapper.

The old node was 183 lines, of which about 40 were the actual computer vision and the rest was
cv_bridge conversions, CameraInfo plumbing, PoseArray assembly and a TF broadcaster. This is the
40 lines.
"""

import cv2
import numpy as np

from .. import log as urlog
from ..transforms import inverse

log = urlog.get('aruco')


def get_dictionary(name):
    attr = getattr(cv2.aruco, name, None)
    if attr is None:
        raise ValueError(f'Unknown ArUco dictionary {name!r} (e.g. DICT_4X4_50)')
    return cv2.aruco.getPredefinedDictionary(attr)


class ArucoDetector:
    """Detects markers and returns their poses in the CAMERA OPTICAL frame."""

    def __init__(self, cfg):
        a = cfg.section('aruco')
        self.marker_size = float(a.get('marker_size_m', 0.0203))
        self.dictionary = get_dictionary(a.get('dictionary', 'DICT_4X4_50'))
        self.params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

        # ArUco corner order is TL, TR, BR, BL, centred on the marker with +Z out of its face.
        h = self.marker_size / 2.0
        self.obj_points = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
                                   dtype=np.float32)

    def detect(self, frame):
        """{marker_id: T_cam_marker (4x4)} for every marker in the frame."""
        gray = cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return {}

        out = {}
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            img_points = marker_corners.reshape(4, 2).astype(np.float32)
            # IPPE_SQUARE is the analytic planar-square solver -- exact for four coplanar corners,
            # and far better conditioned than the iterative default at these marker sizes.
            ok, rvec, tvec = cv2.solvePnP(
                self.obj_points, img_points, frame.K, frame.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            T = np.eye(4)
            T[:3, :3], _ = cv2.Rodrigues(rvec)
            T[:3, 3] = tvec.flatten()
            out[int(marker_id)] = T
        return out

    def detect_in_base(self, frame):
        """{marker_id: T_base_marker}. Requires the frame to carry its capture pose.

        This one line -- T_base_cam @ T_cam_marker -- is the entire job that ur_tf_demo's
        pose_streamer_node, its hand_eye.yaml static_transform_publisher, and the tf2 tree
        existed to do."""
        if frame.T_base_cam is None:
            raise ValueError('Frame has no camera pose; construct the camera with pose_fn=.')
        return {mid: frame.T_base_cam @ T for mid, T in self.detect(frame).items()}

    def draw(self, frame, poses=None):
        """Annotated copy of the image, for a debug window or a saved scan overlay."""
        img = frame.color.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return img
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        for mid, T in (poses or self.detect(frame)).items():
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            cv2.drawFrameAxes(img, frame.K, frame.D, rvec, T[:3, 3], self.marker_size * 0.5)
        return img


class MarkerTracker:
    """Keeps the latest pose of one marker and publishes it into the frame graph.

    `lookup()` gates on age -- the detector simply stops reporting a marker that has left the
    view, and without an age check the last sighting would keep being returned as if it were
    current. That is not a caching artefact to be tuned away; it is the question "is the marker
    still there?", and it has to be asked explicitly."""

    def __init__(self, camera, detector, frames, marker_id, base_frame='base_link',
                 frame_name=None):
        self.camera = camera
        self.detector = detector
        self.frames = frames
        self.marker_id = int(marker_id)
        self.base_frame = base_frame
        self.frame_name = frame_name or f'marker_{marker_id}'

    def observe(self):
        """Capture, detect, publish. Returns T_base_marker or None."""
        frame = self.camera.capture()
        poses = self.detector.detect_in_base(frame)
        T = poses.get(self.marker_id)
        if T is None:
            return None
        self.frames.set_observed(self.base_frame, self.frame_name, T, stamp=frame.stamp)
        return T

    def acquire(self, timeout_s=5.0, max_age_s=1.0):
        """Keep capturing until the marker is seen. Returns T_base_marker or None."""
        import time
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            T = self.observe()
            if T is not None:
                return T
        log.error("Marker %d not in view after %.1fs.", self.marker_id, timeout_s)
        return None

    def marker_in_camera(self, T_base_marker, T_base_cam):
        """The marker in the optical frame: x/y are the centring error, z the depth."""
        return inverse(T_base_cam) @ T_base_marker
