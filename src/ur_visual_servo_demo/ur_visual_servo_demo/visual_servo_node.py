#!/usr/bin/env python3
"""Eye-in-hand position-based visual servoing (PBVS) demo for the UR10e (ROS2 Jazzy).

The wrist-mounted camera is driven to a desired pose relative to a detected ArUco marker:
directly in front of it, facing it head-on, at a stand-off distance. The control loop closes a
clamped fraction of the remaining pose error each iteration, so the arm converges on that pose
and then TRACKS it -- move the marker and the arm follows.

Loop (rate_hz):
  1. Read T_base_marker from tf (needs ur_vision_demo + ur_tf_demo).
  2. Desired camera pose:  T_base_cam_des = T_base_marker * [Rx(cam_rpy) | (0,0,standoff)].
  3. Pose error current->desired. Within the deadband -> HOLD (no move).
  4. Otherwise step the camera a clamped `gain` fraction toward the desired pose, back-solve
     tool0 via the hand-eye tf, IK, and command a short trajectory.
When the marker leaves view, the arm HOLDS (stops commanding motion).

Prereqs: the arm driver/bringup (scaled_joint_trajectory_controller active), move_group
(/compute_ik), and the vision + hand-eye tf stack. This demo moves the arm autonomously --
keep the e-stop in hand.
"""

import os

import numpy as np
import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf2_ros
from tf_transformations import (
    euler_matrix, quaternion_from_matrix, quaternion_matrix, quaternion_slerp)

from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes

UR_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]


# --------------------------------------------------------------- transform helpers
def xyzrpy_to_matrix(xyz, rpy):
    m = euler_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    m[0, 3], m[1, 3], m[2, 3] = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return m


def transform_to_matrix(t):
    q = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
    m = quaternion_matrix(q)
    m[0, 3], m[1, 3], m[2, 3] = t.translation.x, t.translation.y, t.translation.z
    return m


def matrix_to_pose(m):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = m[0, 3], m[1, 3], m[2, 3]
    q = quaternion_from_matrix(m)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    return pose


def _duration(seconds):
    secs = int(seconds)
    return Duration(sec=secs, nanosec=int((seconds - secs) * 1e9))


class VisualServo(Node):
    """Eye-in-hand PBVS: keep the camera centered/squared on the marker at a stand-off."""

    def __init__(self):
        super().__init__('visual_servo')

        cfg_path = self.declare_parameter('config_file', self._default_config()).value
        with open(cfg_path, 'r') as f:
            c = yaml.safe_load(f) or {}
        self.get_logger().info(f'Config: {cfg_path}')

        self.base_frame = c.get('base_frame', 'base_link')
        self.tip_frame = c.get('tip_frame', 'tool0')
        self.camera_frame = c.get('camera_frame', 'camera1_color_optical_frame')

        m = c.get('marker', {})
        self.marker_frame = m.get('frame') or f"camera1_marker_{int(m.get('id', 0))}"

        self.standoff = float(c.get('standoff_m', 0.20))
        self.cam_rpy_in_marker = c.get('cam_rpy_in_marker', [3.14159, 0.0, 0.0])

        self.gain = float(c.get('gain', 0.4))
        self.max_lin = float(c.get('max_linear_step_m', 0.02))
        self.max_ang = np.radians(float(c.get('max_angular_step_deg', 15.0)))
        self.pos_deadband = float(c.get('pos_deadband_m', 0.005))
        self.ang_deadband = np.radians(float(c.get('ang_deadband_deg', 1.5)))

        self.rate_hz = float(c.get('rate_hz', 2.0))
        self.loop_period = 1.0 / self.rate_hz if self.rate_hz > 0 else 0.5
        self.move_duration = float(c.get('move_duration_s', 0.8))
        self.move_timeout = float(c.get('move_timeout_s', 10.0))
        self.settle_s = float(c.get('settle_s', 0.0))
        self.marker_max_age = float(c.get('marker_max_age_s', 0.5))

        self.ik_timeout = float(c.get('ik_timeout_s', 2.0))
        self.ik_attempts = int(c.get('ik_attempts', 12))
        self.ik_avoid_collisions = bool(c.get('avoid_collisions', False))

        self.controller_action = c.get(
            'controller_action', '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        self.confirm_start = bool(c.get('confirm_start', True))
        self.confirm_each_move = bool(c.get('confirm_each_move', False))
        self.max_iterations = int(c.get('max_iterations', 0))
        self.joint_names = list(UR_JOINTS)
        self.planning_group = c.get('planning_group', 'ur_manipulator')

        # Interfaces
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._joints = {}
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(self, FollowJointTrajectory, self.controller_action)

    @staticmethod
    def _default_config():
        """Config path. Prefer the SOURCE yaml when running from a --symlink-install build, so
        editing the yaml takes effect WITHOUT rebuilding (realpath resolves the symlinked module
        back into src/). Fall back to the installed copy (plain build / no source present)."""
        src = os.path.normpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', 'config', 'visual_servo.yaml'))
        if os.path.isfile(src):
            return src
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory('ur_visual_servo_demo'), 'config', 'visual_servo.yaml')

    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self._joints[name] = pos

    # --------------------------------------------------------------------- setup
    def setup(self):
        # Wait for each dependency separately and log which one, so a stall points at the culprit
        # (the old single 'waiting for...' line couldn't say which check was blocking).
        self.get_logger().info('[1/4] Waiting for /compute_ik service (move_group)...')
        if not self.ik_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error('/compute_ik unavailable (start move_group).')
            return False

        self.get_logger().info(
            f"[2/4] Waiting for controller action '{self.controller_action}'...")
        if not self.traj_client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(
                f"'{self.controller_action}' unavailable -- is scaled_joint_trajectory_controller "
                'ACTIVE? Check: ros2 control list_controllers')
            return False

        self.get_logger().info('[3/4] Waiting for /joint_states (all 6 UR joints)...')
        if not self._wait_joints(10.0):
            have = sorted(self._joints)
            self.get_logger().error(
                f'No /joint_states with all 6 UR joints (have: {have}). Is the driver up?')
            return False

        self.get_logger().info(
            f'[4/4] Waiting for hand-eye tf {self.tip_frame} -> {self.camera_frame}...')
        if self._tf_matrix(self.tip_frame, self.camera_frame, timeout_s=5.0) is None:
            self.get_logger().error(
                f'No {self.tip_frame} -> {self.camera_frame} tf. Start ur_tf_demo '
                '(publishes the hand-eye static transform).')
            return False

        self.get_logger().info('All dependencies ready.')
        return True

    def _wait_joints(self, timeout_s):
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            if all(j in self._joints for j in self.joint_names):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _current_joints(self):
        return [self._joints.get(j, 0.0) for j in self.joint_names]

    # ------------------------------------------------------------------ tf lookups
    def _tf_matrix(self, target, source, max_age_s=None, timeout_s=1.0):
        """Latest target<-source transform as a 4x4, or None. If max_age_s is set, a transform
        older than that counts as unavailable -- the aruco node stops publishing when the marker
        leaves view but tf2 keeps the last transform cached, so a plain lookup would go stale."""
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(target, source, Time())
                if max_age_s is not None:
                    age = (self.get_clock().now()
                           - Time.from_msg(tf.header.stamp)).nanoseconds / 1e9
                    if age > max_age_s:
                        rclpy.spin_once(self, timeout_sec=0.05)
                        continue
                return transform_to_matrix(tf.transform)
            except tf2_ros.TransformException:
                rclpy.spin_once(self, timeout_sec=0.05)
        return None

    # ------------------------------------------------------------- servo control law
    @staticmethod
    def _pose_error(T_cur, T_des):
        """(linear error [m], angular error [rad]) between two 4x4 poses."""
        lin = float(np.linalg.norm(T_des[:3, 3] - T_cur[:3, 3]))
        q_cur = quaternion_from_matrix(T_cur)
        q_des = quaternion_from_matrix(T_des)
        dot = min(1.0, abs(float(np.dot(q_cur, q_des))))
        ang = float(2.0 * np.arccos(dot))
        return lin, ang

    def _interpolate_pose(self, T_cur, T_des):
        """Step a clamped `gain` fraction from T_cur toward T_des (translation + slerp)."""
        p_cur, p_des = T_cur[:3, 3], T_des[:3, 3]
        step = (p_des - p_cur) * self.gain
        n = np.linalg.norm(step)
        if n > self.max_lin:
            step = step / n * self.max_lin

        q_cur = quaternion_from_matrix(T_cur)
        q_des = quaternion_from_matrix(T_des)
        dot = min(1.0, abs(float(np.dot(q_cur, q_des))))
        ang = 2.0 * np.arccos(dot)
        frac = 0.0 if ang < 1e-6 else min(self.gain, self.max_ang / ang)
        q_new = quaternion_slerp(q_cur, q_des, frac)

        T_new = quaternion_matrix(q_new)
        T_new[:3, 3] = p_cur + step
        return T_new

    # --------------------------------------------------------------------- IK + move
    def solve_ik(self, tool0_pose):
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = tool0_pose

        seed = self._current_joints()
        for attempt in range(max(1, self.ik_attempts)):
            seed_positions = (list(seed) if attempt == 0
                              else list(np.random.uniform(-np.pi, np.pi, len(self.joint_names))))
            req = GetPositionIK.Request()
            req.ik_request.group_name = self.planning_group
            req.ik_request.ik_link_name = self.tip_frame
            req.ik_request.avoid_collisions = self.ik_avoid_collisions
            req.ik_request.timeout = _duration(self.ik_timeout)
            req.ik_request.robot_state.joint_state.name = list(self.joint_names)
            req.ik_request.robot_state.joint_state.position = seed_positions
            req.ik_request.pose_stamped = ps

            future = self.ik_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.ik_timeout + 2.0)
            resp = future.result()
            if resp is not None and resp.error_code.val == MoveItErrorCodes.SUCCESS:
                sol = dict(zip(resp.solution.joint_state.name,
                               resp.solution.joint_state.position))
                try:
                    return [sol[j] for j in self.joint_names]
                except KeyError:
                    return None
        return None

    def send_joints(self, positions):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = _duration(self.move_duration)
        goal.trajectory.points.append(point)

        sf = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sf, timeout_sec=10.0)
        gh = sf.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('Trajectory goal rejected / no response.')
            return False
        rf = gh.get_result_async()
        deadline = self.get_clock().now().nanoseconds + int(self.move_timeout * 1e9)
        while rclpy.ok() and not rf.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.get_clock().now().nanoseconds > deadline:
                self.get_logger().error(
                    f'Trajectory not finished after {self.move_timeout:.0f}s -- goal accepted but '
                    'not executing (External Control playing? speed slider up? e-stop?). Canceling.')
                gh.cancel_goal_async()
                return False
        res = rf.result()
        if res is None or res.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error('Move failed.')
            return False
        return True

    def _sleep(self, seconds):
        if seconds <= 0:
            return
        deadline = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _confirm(self, label):
        ans = input(f'\n[{label}] Enter to proceed (q to abort): ')
        return ans.strip().lower() != 'q'

    # --------------------------------------------------------------------- run
    def _servo_once(self):
        """One control iteration. Returns True to keep servoing, False to stop."""
        T_base_marker = self._tf_matrix(
            self.base_frame, self.marker_frame, max_age_s=self.marker_max_age)
        if T_base_marker is None:
            self.get_logger().warn(
                f"Marker '{self.marker_frame}' not in view -- holding.",
                throttle_duration_sec=2.0)
            return True

        T_base_cam = self._tf_matrix(self.base_frame, self.camera_frame)
        T_tool0_cam = self._tf_matrix(self.tip_frame, self.camera_frame)
        if T_base_cam is None or T_tool0_cam is None:
            self.get_logger().warn('Camera tf unavailable -- holding.', throttle_duration_sec=2.0)
            return True

        T_base_cam_des = T_base_marker @ xyzrpy_to_matrix(
            [0.0, 0.0, self.standoff], self.cam_rpy_in_marker)
        lin_err, ang_err = self._pose_error(T_base_cam, T_base_cam_des)

        if lin_err <= self.pos_deadband and ang_err <= self.ang_deadband:
            self.get_logger().info(
                f'On target (lin {lin_err * 1000:.1f} mm, ang {np.degrees(ang_err):.1f} deg) '
                '-- holding.', throttle_duration_sec=2.0)
            return True

        self.get_logger().info(
            f'error lin={lin_err * 1000:.1f} mm ang={np.degrees(ang_err):.1f} deg -> stepping')
        if self.confirm_each_move and not self._confirm('corrective move'):
            return False

        T_cam_cmd = self._interpolate_pose(T_base_cam, T_base_cam_des)
        T_base_tool0 = T_cam_cmd @ np.linalg.inv(T_tool0_cam)
        joints = self.solve_ik(matrix_to_pose(T_base_tool0))
        if joints is None:
            self.get_logger().warn('IK failed for servo target -- holding this cycle.')
            return True
        self.send_joints(joints)
        if self.settle_s > 0:
            self._sleep(self.settle_s)
        return True

    def run(self):
        if self.confirm_start and not self._confirm('start visual servo'):
            self.get_logger().info('Aborted by user.')
            return False
        self.get_logger().info(
            f"Visual servo running (target: {self.standoff * 100:.0f} cm in front of "
            f"'{self.marker_frame}'). Move the marker; Ctrl-C to stop.")
        i = 0
        while rclpy.ok():
            if self.max_iterations and i >= self.max_iterations:
                self.get_logger().info(f'Reached max_iterations ({self.max_iterations}); stopping.')
                break
            i += 1
            if not self._servo_once():
                self.get_logger().info('Stopped by user.')
                break
            self._sleep(self.loop_period)
        return True


def main(args=None):
    rclpy.init(args=args)
    node = VisualServo()
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
