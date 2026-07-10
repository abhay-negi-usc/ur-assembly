#!/usr/bin/env python3
"""Cable pick-and-place demo for the UR10e + Robotiq 2F-85 (ROS2 Jazzy).

Subclasses ur_pick_place_demo's PickPlace to reuse the whole pick/place machinery (IK, JTC, gripper,
grasp/lift/place geometry). The one difference is DETECTION: instead of reading a single fiducial,
the robot SCANS the cable from several camera views so the external SAM3 cable-connector pose
estimator (see the sam3-abhay repo: cable_neck_ros_node + connector_pose_node) can fuse them into a
3D connector pose, published over TF as base_frame -> connector_frame. This demo reads that pose,
builds a grasp frame, and runs the pick/place sequence. It needs NO torch -- the SAM3 nodes run
separately and are coupled only through TF.

Sequence:
  open -> scan (multi-view) -> estimate connector pose -> grasp-align -> grasp -> close -> lift
  -> pre-place -> place -> open -> retreat -> home

Connector frame convention (built here from the SAM3 axis + the up assumption):
  x = the cable-connector AXIS (SAM3 measures it; it is that TF's z-column),
  z = base +Z (connector_up_axis), re-orthogonalized perpendicular to x,
  y = z x x  (right-handed, horizontal).
The grasp (connector_grasp) is defined relative to this frame.
"""

import os
from datetime import datetime

import numpy as np

import rclpy
from sensor_msgs.msg import Image

from ur_pick_place_demo.pick_place_node import PickPlace, matrix_to_pose, xyzrpy_to_matrix


def _imgmsg_to_bgr(msg):
    """Decode a sensor_msgs/Image (bgr8/rgb8) to an HxWx3 uint8 BGR array (no cv_bridge)."""
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    h, w, step = msg.height, msg.width, msg.step
    enc = msg.encoding.lower()
    if enc == 'bgr8':
        return buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3).copy()
    if enc == 'rgb8':
        return buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)[:, :, ::-1].copy()
    return None


class CablePickPlace(PickPlace):
    """Pick-and-place a cable using a multi-view SAM3 connector-pose estimate."""

    def __init__(self):
        super().__init__(node_name='cable_pick_place', default_config=self._cable_config())
        c = self.cfg
        self.connector_frame = c.get('connector_frame', 'connector')
        self.connector_max_age = float(c.get('connector_max_age_s', 3.0))
        self.connector_wait_s = float(c.get('connector_wait_s', 8.0))
        self.connector_up_axis = np.asarray(
            c.get('connector_up_axis', [0.0, 0.0, 1.0]), dtype=float)
        self.T_connector_grasp = xyzrpy_to_matrix(**self._xyzrpy(c.get('connector_grasp', {})))

        # Fingertip frame: defined w.r.t. the GRIPPER (grasp_tcp_offset, set by PickPlace). The cable
        # grasp is commanded so the fingertip coincides with the connector, so the FINGERTIP becomes
        # the grasp reference the pick machinery uses (self.T_tool0_grasp). Keep the gripper frame.
        T_gripper_fingertip = xyzrpy_to_matrix(**self._xyzrpy(c.get('fingertip_grasp', {})))
        self.T_tool0_gripper = self.T_tool0_grasp.copy()
        self.T_tool0_fingertip = self.T_tool0_gripper @ T_gripper_fingertip
        self.T_tool0_grasp = self.T_tool0_fingertip
        self._static_tf = None
        if bool(c.get('publish_fingertip_tf', True)):
            from tf2_ros import StaticTransformBroadcaster
            self._static_tf = StaticTransformBroadcaster(self)
            self._publish_fingertip_tf()

        scan = c.get('scan', {}) or {}
        self.scan_dwell_s = float(scan.get('dwell_s', 3.0))
        # Scan views are RELATIVE to the camera's pose at the START of the scan (not absolute base
        # poses). Each offset is applied in the CAMERA frame (xyz m, rpy rad) and CLAMPED to
        # relative_bounds so the camera stays in a safe Cartesian box around where it started.
        rb = scan.get('relative_bounds', {}) or {}
        self.scan_bounds_xyz = np.abs(np.asarray(rb.get('xyz', [0.06, 0.06, 0.05]), dtype=float))
        self.scan_bounds_rpy = np.abs(np.asarray(rb.get('rpy', [0.0, 0.15, 0.15]), dtype=float))
        self.scan_offsets = [self._xyzrpy(o) for o in (scan.get('offsets', []) or [])]

        # Save the SAM3 overlay (Node 1's ~/debug_image) at each scan view.
        self.save_scan_images = bool(c.get('save_scan_images', True))
        self.debug_image_topic = c.get('debug_image_topic', '/cable_neck_detector/debug_image')
        data_dir = os.path.expanduser(str(c.get('data_dir', 'data')))
        data_dir = data_dir if os.path.isabs(data_dir) else os.path.abspath(data_dir)
        subdir = str(c.get('scan_images_subdir', 'cable_pick_place'))
        self._scan_dir = os.path.join(data_dir, subdir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        self._latest_debug = None
        if self.save_scan_images:
            self.create_subscription(Image, self.debug_image_topic, self._debug_cb, 1)

        # ---- Pick-up failure detection & recovery ----
        # Detects the "cable not seated in the fingertip groove" failure from the gripper position in
        # COUNTS (0-255, Robotiq). A FULL close reaches ~closed_counts when the cable is seated in the
        # groove (or the fingers are empty); if the cable is caught OUTSIDE the groove the fingers
        # stall SHORT (< closed_counts - tolerance) -> FAILED grasp. (Empty vs seated is NOT
        # distinguished yet -- future work.) On failure: open (drop), return to the initial pose, and
        # retry the whole scan->grasp sequence.
        gc = c.get('grasp_check', {}) or {}
        self.grasp_check_enabled = bool(gc.get('enabled', True))
        self.grasp_closed_counts = int(gc.get('closed_counts', 228))
        self.grasp_tol_counts = int(gc.get('tolerance_counts', 1))
        self.grasp_full_close_rad = float(gc.get('full_close_rad', 0.8))
        self.grasp_check_settle_s = float(gc.get('settle_s', 1.0))
        self.grasp_max_retries = int(gc.get('max_retries', 2))
        # With the check on, COMMAND a full close so the fingers stall at the cable-determined
        # position (needed to tell "seated" from "short"); otherwise use the configured closed_position.
        self.grasp_close = (self.grasp_full_close_rad if self.grasp_check_enabled
                            else self.gripper_closed)

    def _publish_fingertip_tf(self):
        """Broadcast the static tool0 -> fingertip transform (for RViz verification)."""
        from geometry_msgs.msg import TransformStamped
        pose = matrix_to_pose(self.T_tool0_fingertip)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.tip_frame
        t.child_frame_id = 'fingertip'
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation
        self._static_tf.sendTransform(t)

    @staticmethod
    def _cable_config():
        src = os.path.normpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', 'config', 'cable_pick_place.yaml'))
        if os.path.isfile(src):
            return src
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('ur_cable_pick_place_demo'),
                            'config', 'cable_pick_place.yaml')

    # ------------------------------------------------------------------ setup
    def setup(self):
        if not super().setup():
            return False
        if self._tf_matrix(self.tip_frame, self.camera_frame, timeout_s=5.0) is None:
            self.get_logger().error(
                f'No {self.tip_frame} -> {self.camera_frame} tf. Start ur_tf_demo (hand-eye) so the '
                'camera pose is known for the scan.')
            return False
        self.get_logger().warn(
            'This demo READS the connector pose from the SAM3 nodes -- make sure both are running: '
            'cable_neck_ros_node (detector) and connector_pose_node '
            f"(world_frame:={self.base_frame}, connector_frame:={self.connector_frame}).")
        return True

    # ------------------------------------------------------------ connector geometry
    def _connector_frame_from_tf(self, T_base_conn):
        """Rebuild the connector frame to this demo's convention: x = cable axis (SAM3's z-column),
        z = base +Z (connector_up_axis, re-orthogonalized), y = z x x."""
        origin = T_base_conn[:3, 3]
        x = T_base_conn[:3, 2].astype(float)          # SAM3 puts the connector axis in the z-column
        x = x / (np.linalg.norm(x) + 1e-12)
        up = self.connector_up_axis / (np.linalg.norm(self.connector_up_axis) + 1e-12)
        y = np.cross(up, x)
        if np.linalg.norm(y) < 1e-6:                  # axis ~parallel to up; pick another reference
            up = np.array([1.0, 0.0, 0.0]) if abs(x[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            y = np.cross(up, x)
        y = y / (np.linalg.norm(y) + 1e-12)
        z = np.cross(x, y)
        z = z / (np.linalg.norm(z) + 1e-12)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, origin
        return T

    def _read_connector(self):
        T = self._tf_matrix(self.base_frame, self.connector_frame,
                            max_age_s=self.connector_max_age, timeout_s=self.connector_wait_s)
        if T is None:
            self.get_logger().error(
                f"No fresh '{self.base_frame} -> {self.connector_frame}' tf. Is connector_pose_node "
                f"running (world_frame:={self.base_frame})? Did the scan give it enough views + "
                'parallax? Add more scan.offsets or widen them (within scan.relative_bounds).')
        return T

    def _estimate_connector(self):
        """Read the fused connector pose and set the grasp target (self.T_base_grasp)."""
        T_base_conn = self._read_connector()
        if T_base_conn is None:
            return False
        T_frame = self._connector_frame_from_tf(T_base_conn)
        self.T_base_grasp = T_frame @ self.T_connector_grasp
        p = T_frame[:3, 3]
        axis = T_frame[:3, 0]
        self.get_logger().info(
            f'Connector: origin=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) '
            f'axis=({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f}).')
        return True

    # ------------------------------------------------------------ scan phase
    def _go_to_camera_pose(self, T_base_cam, label):
        T_tool0_cam = self._tf_matrix(self.tip_frame, self.camera_frame)
        if T_tool0_cam is None:
            self.get_logger().error(f'No {self.tip_frame} -> {self.camera_frame} tf (hand-eye).')
            return False
        T_base_tool0 = T_base_cam @ np.linalg.inv(T_tool0_cam)
        return self.move_tool0_to(matrix_to_pose(T_base_tool0), label)

    def _debug_cb(self, msg):
        img = _imgmsg_to_bgr(msg)
        if img is not None:
            self._latest_debug = img

    def _save_view_image(self, index):
        """Save the latest SAM3 overlay for the current view to the run's data folder."""
        if not self.save_scan_images:
            return
        if self._latest_debug is None:
            self.get_logger().warn(
                f"No SAM3 overlay on '{self.debug_image_topic}' for view {index} -- is "
                'cable_neck_ros_node running with publish_debug:=true, and dwell_s long enough?')
            return
        try:
            import cv2
            os.makedirs(self._scan_dir, exist_ok=True)
            path = os.path.join(self._scan_dir, f'view_{index:02d}.png')
            cv2.imwrite(path, self._latest_debug)
            self.get_logger().info(f'Saved scan overlay: {path}')
        except Exception as exc:   # noqa: BLE001 - saving is best-effort
            self.get_logger().warn(f'Could not save scan overlay: {exc}')

    def _scan(self):
        if not self.scan_offsets:
            self.get_logger().error('No scan.offsets configured.')
            return False
        # Anchor the scan on the camera's CURRENT pose; every view is an offset from it.
        T_base_cam0 = self._tf_matrix(self.base_frame, self.camera_frame)
        if T_base_cam0 is None:
            self.get_logger().error(
                f'No {self.base_frame} -> {self.camera_frame} tf to anchor the relative scan.')
            return False
        n = len(self.scan_offsets)
        self.get_logger().info(
            f'Scanning the cable from {n} views relative to the current camera pose '
            '(offsets in the camera frame, clamped to relative_bounds; must give parallax)...')
        for i, off in enumerate(self.scan_offsets):
            xyz = np.clip(np.asarray(off['xyz'], dtype=float),
                          -self.scan_bounds_xyz, self.scan_bounds_xyz)
            rpy = np.clip(np.asarray(off['rpy'], dtype=float),
                          -self.scan_bounds_rpy, self.scan_bounds_rpy)
            T_cam_target = T_base_cam0 @ xyzrpy_to_matrix(xyz, rpy)   # offset in the camera frame
            if not self._go_to_camera_pose(T_cam_target, f'scan view {i + 1}/{n}'):
                return False
            self._latest_debug = None            # discard any stale overlay before this view
            self._sleep(self.scan_dwell_s)       # hold still so SAM3 processes a clean frame here
            self._save_view_image(i + 1)         # save the fresh SAM3 overlay for this view
            got = self._tf_matrix(self.base_frame, self.connector_frame,
                                  max_age_s=self.connector_max_age, timeout_s=0.2) is not None
            self.get_logger().info(
                f'  view {i + 1}/{n}: connector estimate '
                f'{"available" if got else "not yet (need more views/parallax)"}.')
        return True

    # ------------------------------------------------------ grasp check & recovery
    def _gripper_counts(self):
        """Actual gripper position in COUNTS (0-255, Robotiq), converted from the joint (radians) in
        /joint_states via full_close_rad. None if the joint isn't published yet."""
        pos = self._joints.get(self.gripper_joint)
        if pos is None or self.grasp_full_close_rad <= 0.0:
            return None if pos is None else 0
        return max(0, min(255, int(round(pos / self.grasp_full_close_rad * 255.0))))

    def _grasp_succeeded(self):
        """Detect the "cable not in the fingertip groove" failure. After a FULL close, the fingers
        reach ~closed_counts when the cable is seated in the groove (or the fingers are empty); if the
        cable is caught OUTSIDE the groove they stall SHORT -> failure. (Seated vs empty is not
        distinguished yet.) Reads the gripper position in counts (settles first for a steady value)."""
        self._sleep(self.grasp_check_settle_s)      # let the gripper stall/seat after closing
        counts = self._gripper_counts()
        if counts is None:
            self.get_logger().warn(
                f"No '{self.gripper_joint}' in /joint_states; can't check the grasp -- assuming OK.")
            return True
        seated = counts >= self.grasp_closed_counts - self.grasp_tol_counts
        self.get_logger().info(
            f'Grasp check: gripper at {counts}/255 (fully closed if >= '
            f'{self.grasp_closed_counts - self.grasp_tol_counts}) -> '
            f'{"CLOSED (seated or empty)" if seated else "SHORT -- cable NOT in the groove: FAILED"}.')
        return seated

    def _recover_to_home(self, home_joints):
        """Failure recovery: open the gripper (drop anything held) and return to the initial pose."""
        return (
            self._do('recover: open gripper (drop)',
                     lambda: self.gripper_to(self.gripper_open, 'open'))
            and self._do('recover: return to initial pose',
                         lambda: self.send_joints(home_joints)))

    def _attempt_grasp(self, home_joints):
        """One pick attempt through closing on the cable. Returns 'ok' (grasp check passed or
        disabled), 'retry' (grasp EMPTY -- recoverable), or 'abort' (a motion/step failed)."""
        steps_ok = (
            self._do('open gripper', lambda: self.gripper_to(self.gripper_open, 'open'))
            and self._do('scan cable (multi-view)', self._scan)
            and self._do('estimate connector pose', self._estimate_connector)
            and self._do('move to grasp-align',
                         lambda: self.move_grasp_tcp_to(self._pre_grasp_pose(), 'grasp-align'))
            and self._do('move to grasp',
                         lambda: self.move_grasp_tcp_to(self.T_base_grasp, 'grasp'))
            and self._do('close gripper (grasp)',
                         lambda: self.gripper_to(self.grasp_close, 'close')))
        if not steps_ok:
            return 'abort'
        if not self.grasp_check_enabled:
            return 'ok'
        return 'ok' if self._grasp_succeeded() else 'retry'

    # --------------------------------------------------------------------- run
    def run(self):
        home_joints = self._current_joints()

        # Pick the cable, with optional failure detection + recovery/retry.
        attempt = 0
        while True:
            result = self._attempt_grasp(home_joints)
            if result == 'ok':
                break
            if result == 'abort':
                return False
            # 'retry': the grasp came up empty.
            if attempt >= self.grasp_max_retries:
                self.get_logger().error(
                    f'Grasp failed on all {self.grasp_max_retries + 1} attempts; aborting.')
                return False
            attempt += 1
            self.get_logger().warn(
                f'Pick-up failed (gripper closed past threshold -- no cable). Recovering and '
                f'retrying (attempt {attempt + 1}/{self.grasp_max_retries + 1})...')
            if not self._recover_to_home(home_joints):
                return False

        # Grasp OK -> lift, place, release, home.
        ok = (
            self._do('lift', lambda: self.move_grasp_tcp_to(self._lift_pose(), 'lift'))
            and self._do('move to pre-place',
                         lambda: self.move_grasp_tcp_to(self._pre_place_pose(), 'pre-place'))
            and self._do('move to place',
                         lambda: self.move_grasp_tcp_to(self._place_pose(), 'place'))
            and self._do('open gripper (release)',
                         lambda: self.gripper_to(self.gripper_open, 'open'))
            and self._do('retreat',
                         lambda: self.move_grasp_tcp_to(self._pre_place_pose(), 'retreat'))
            and self._do('return home', lambda: self.send_joints(home_joints)))
        if ok:
            self.get_logger().info('Cable pick-and-place complete.')
        return ok


def main(args=None):
    rclpy.init(args=args)
    node = CablePickPlace()
    try:
        if node.setup():
            node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
