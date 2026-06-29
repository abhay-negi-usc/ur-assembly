#!/usr/bin/env python3
"""Cartesian position control demo for a UR10e (ROS2 Jazzy).

From the robot's current ("initial") TCP pose, this node steps the TCP by +/- a fixed
linear amount along X/Y/Z and +/- a fixed angular amount about roll/pitch/yaw, returning
to the initial pose between every move. The whole sequence is run once per *motion frame*
so you can compare cartesian motion expressed in the world frame vs. the tool0 frame.

Because ur_controllers 3.8.0 no longer ships a Cartesian trajectory controller, this demo
does the Cartesian -> joint mapping itself:

  * the current TCP pose is read from tf2 (``reference_frame`` -> ``tip_frame``),
  * each target pose is solved with MoveIt's ``/compute_ik`` service (uses the driver's
    *calibrated* URDF, seeded with the initial joint state for solution continuity),
  * the resulting joint goal is sent to the already-active
    ``scaled_joint_trajectory_controller`` via ``FollowJointTrajectory`` (smooth, time
    parameterized, speed-scaled -- much safer than the streaming forward controllers).

Recentering commands the *recorded initial joint configuration* directly (no IK), so the
robot returns exactly to where it started.

Requires ``move_group`` to be running, e.g.:
    ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10e
"""

import math
import sys

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf2_ros
from tf_transformations import (
    quaternion_conjugate,
    quaternion_from_euler,
    quaternion_matrix,
    quaternion_multiply,
)

from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes

UR_JOINTS = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


def _normalize(q):
    """Return a unit-length quaternion [x, y, z, w]."""
    n = math.sqrt(sum(c * c for c in q))
    if n == 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    return [c / n for c in q]


def _rotate_vec(q, v):
    """Rotate vector v by quaternion q ([x, y, z, w])."""
    rot = quaternion_matrix(q)[:3, :3]
    return rot.dot(np.array(v, dtype=float))


def _duration(seconds):
    secs = int(seconds)
    return Duration(sec=secs, nanosec=int((seconds - secs) * 1e9))


class CartesianPoseDemo(Node):
    """Drives the +/- X/Y/Z/R/P/Y cartesian demo, once per motion frame."""

    def __init__(self):
        super().__init__('cartesian_pose_demo')

        # --- Parameters ---------------------------------------------------
        self.planning_group = self.declare_parameter(
            'planning_group', 'ur_manipulator').value
        self.reference_frame = self.declare_parameter(
            'reference_frame', 'base_link').value
        self.tip_frame = self.declare_parameter('tip_frame', 'tool0').value
        # Frames whose axes define the +/- X/Y/Z/R/P/Y deltas. The sequence runs once each.
        # 'tool0' is the controlled tool-zero frame (Z out the tool); prefer it over 'flange',
        # whose X/Y are rotated 90 deg about the tool axis relative to tool0.
        self.motion_frames = self.declare_parameter(
            'motion_frames', ['world', 'tool0']).value
        self.joint_names = self.declare_parameter('joint_names', UR_JOINTS).value
        self.controller_action = self.declare_parameter(
            'controller_action',
            '/scaled_joint_trajectory_controller/follow_joint_trajectory').value
        self.linear_step_m = self.declare_parameter('linear_step_m', 0.030).value
        self.angular_step_deg = self.declare_parameter('angular_step_deg', 30.0).value
        self.move_duration_s = self.declare_parameter('move_duration_s', 4.0).value
        self.settle_s = self.declare_parameter('settle_s', 0.5).value
        self.ik_timeout_s = self.declare_parameter('ik_timeout_s', 2.0).value
        self.avoid_collisions = self.declare_parameter('avoid_collisions', True).value
        self.confirm_each_move = self.declare_parameter('confirm_each_move', True).value

        # --- tf2 ----------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- joint state cache -------------------------------------------
        self._joint_positions = {}
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)

        # --- IK service + trajectory action ------------------------------
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(
            self, FollowJointTrajectory, self.controller_action)

    # ------------------------------------------------------------- callbacks
    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self._joint_positions[name] = pos

    # ------------------------------------------------------------------ setup
    def setup(self):
        self.get_logger().info("Waiting for /compute_ik service (is move_group running?)...")
        if not self.ik_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error(
                "/compute_ik unavailable. Start MoveIt, e.g.:\n"
                "  ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10e")
            return False

        self.get_logger().info(
            f"Waiting for trajectory action '{self.controller_action}'...")
        if not self.traj_client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(
                f"Action '{self.controller_action}' unavailable. Is "
                "scaled_joint_trajectory_controller active? Check with "
                "'ros2 control list_controllers'.")
            return False

        self.get_logger().info('Waiting for /joint_states...')
        if not self._wait_for_joints(timeout_s=10.0):
            self.get_logger().error('No /joint_states received.')
            return False
        return True

    def _wait_for_joints(self, timeout_s):
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            if all(j in self._joint_positions for j in self.joint_names):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _current_joint_positions(self):
        return [self._joint_positions[j] for j in self.joint_names]

    # --------------------------------------------------------------- get pose
    def get_initial_pose(self, timeout_s=10.0):
        """Look up reference_frame->tip and return a geometry_msgs/Pose, or None."""
        self.get_logger().info(
            f"Looking up current TCP pose ({self.reference_frame} -> {self.tip_frame})...")
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.reference_frame, self.tip_frame, rclpy.time.Time())
                pose = Pose()
                pose.position.x = tf.transform.translation.x
                pose.position.y = tf.transform.translation.y
                pose.position.z = tf.transform.translation.z
                pose.orientation = tf.transform.rotation
                return pose
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().error('Timed out waiting for TF. Is the driver publishing?')
        return None

    def get_frame_rotation(self, frame, timeout_s=3.0):
        """Return quaternion [x,y,z,w] of `frame` relative to reference_frame, or None."""
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.reference_frame, frame, rclpy.time.Time())
                r = tf.transform.rotation
                return [r.x, r.y, r.z, r.w]
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                rclpy.spin_once(self, timeout_sec=0.1)
        return None

    # ----------------------------------------------------------- pose offset
    def offset_pose(self, base_pose, delta, frame_quat):
        """Apply a 6-DOF delta (expressed in frame_quat's axes) to base_pose.

        frame_quat is the orientation of the motion frame relative to reference_frame.
        """
        dx, dy, dz, droll, dpitch, dyaw = delta

        target = Pose()
        # Translation: express the linear step in the motion frame, then in reference frame.
        t = _rotate_vec(frame_quat, [dx, dy, dz])
        target.position.x = base_pose.position.x + t[0]
        target.position.y = base_pose.position.y + t[1]
        target.position.z = base_pose.position.z + t[2]

        # Rotation: rotate about an axis expressed in the motion frame.
        #   R_new = (R_frame * R_delta * R_frame^-1) * R_current
        q_current = [base_pose.orientation.x, base_pose.orientation.y,
                     base_pose.orientation.z, base_pose.orientation.w]
        q_delta = quaternion_from_euler(droll, dpitch, dyaw)
        q_axis = quaternion_multiply(
            quaternion_multiply(frame_quat, q_delta),
            quaternion_conjugate(frame_quat))
        q_new = _normalize(quaternion_multiply(q_axis, q_current))

        target.orientation.x = q_new[0]
        target.orientation.y = q_new[1]
        target.orientation.z = q_new[2]
        target.orientation.w = q_new[3]
        return target

    # --------------------------------------------------------------- solve IK
    def solve_ik(self, pose, seed_positions):
        """Return joint positions (ordered like self.joint_names) for pose, or None."""
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.planning_group
        req.ik_request.ik_link_name = self.tip_frame
        req.ik_request.avoid_collisions = self.avoid_collisions
        req.ik_request.timeout = _duration(self.ik_timeout_s)

        # Seed the solver with the given joint state for solution continuity.
        req.ik_request.robot_state.joint_state.name = list(self.joint_names)
        req.ik_request.robot_state.joint_state.position = list(seed_positions)

        ps = PoseStamped()
        ps.header.frame_id = self.reference_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = pose
        req.ik_request.pose_stamped = ps

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.ik_timeout_s + 2.0)
        resp = future.result()
        if resp is None:
            self.get_logger().error('IK service call failed (no response).')
            return None
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f'IK failed (error_code={resp.error_code.val}) -- target unreachable?')
            return None

        sol = dict(zip(resp.solution.joint_state.name,
                       resp.solution.joint_state.position))
        try:
            return [sol[j] for j in self.joint_names]
        except KeyError as exc:
            self.get_logger().error(f'IK solution missing joint {exc}.')
            return None

    # ------------------------------------------------------------- send goal
    def send_joint_goal(self, positions, duration_s):
        """Send a single-point joint trajectory and block until done. Returns bool."""
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = _duration(duration_s)
        traj.points.append(point)
        goal.trajectory = traj

        send_future = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected by controller.')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            self.get_logger().error('No result returned for trajectory goal.')
            return False
        if result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(
                f'Move failed: error_code={result.result.error_code} '
                f'{result.result.error_string}')
            return False
        return True

    def _sleep(self, seconds):
        if seconds <= 0:
            return
        deadline = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    # ------------------------------------------------------------- sequence
    def _build_moves(self):
        lin = self.linear_step_m
        ang = math.radians(self.angular_step_deg)
        # (label, dx, dy, dz, droll, dpitch, dyaw)
        return [
            ('X +', +lin, 0, 0, 0, 0, 0),
            ('X -', -lin, 0, 0, 0, 0, 0),
            ('Y +', 0, +lin, 0, 0, 0, 0),
            ('Y -', 0, -lin, 0, 0, 0, 0),
            ('Z +', 0, 0, +lin, 0, 0, 0),
            ('Z -', 0, 0, -lin, 0, 0, 0),
            ('Roll +', 0, 0, 0, +ang, 0, 0),
            ('Roll -', 0, 0, 0, -ang, 0, 0),
            ('Pitch +', 0, 0, 0, 0, +ang, 0),
            ('Pitch -', 0, 0, 0, 0, -ang, 0),
            ('Yaw +', 0, 0, 0, 0, 0, +ang),
            ('Yaw -', 0, 0, 0, 0, 0, -ang),
        ]

    def run_sequence(self, frame_name, frame_quat, initial_pose, initial_joints):
        """Run the full +/- sequence with deltas expressed in the given motion frame."""
        self.get_logger().info(
            f'==== Motion frame: {frame_name} '
            f'({self.linear_step_m * 1000:.0f} mm / {self.angular_step_deg:.0f} deg) ====')
        for label, dx, dy, dz, dr, dp, dyaw in self._build_moves():
            if self.confirm_each_move:
                ans = input(
                    f'\n[{frame_name} | {label}] Press Enter to move (q + Enter to quit): ')
                if ans.strip().lower() == 'q':
                    self.get_logger().info('Quit requested.')
                    return False

            self.get_logger().info(f'--> [{frame_name}] {label}')
            target = self.offset_pose(initial_pose, (dx, dy, dz, dr, dp, dyaw), frame_quat)
            joints = self.solve_ik(target, seed_positions=initial_joints)
            if joints is None:
                self.get_logger().warn(f'Skipping {label} (no IK solution).')
                continue
            if not self.send_joint_goal(joints, self.move_duration_s):
                self.get_logger().error('Stopping demo due to move failure.')
                return False

            self.get_logger().info('--> recentering to initial pose')
            if not self.send_joint_goal(initial_joints, self.move_duration_s):
                self.get_logger().error('Stopping demo due to recenter failure.')
                return False

            self._sleep(self.settle_s)
        return True

    # ----------------------------------------------------------------- run
    def run(self):
        initial_pose = self.get_initial_pose()
        if initial_pose is None:
            return False
        initial_joints = self._current_joint_positions()

        p = initial_pose.position
        self.get_logger().info(
            f'Initial TCP position: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} (m)')

        for frame_name in self.motion_frames:
            frame_quat = self.get_frame_rotation(frame_name)
            if frame_quat is None:
                self.get_logger().warn(
                    f"Frame '{frame_name}' not found in tf; expressing its motion in "
                    f"'{self.reference_frame}' (identity) instead.")
                frame_quat = [0.0, 0.0, 0.0, 1.0]
            if not self.run_sequence(frame_name, frame_quat, initial_pose, initial_joints):
                return False

        self.get_logger().info('Demo complete.')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = CartesianPoseDemo()
    ok = False
    try:
        if node.setup():
            ok = node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
