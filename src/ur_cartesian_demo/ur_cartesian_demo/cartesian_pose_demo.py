#!/usr/bin/env python3
"""Cartesian position control demo for a UR10e (ROS2 Jazzy).

From the robot's current ("initial") TCP pose, this node steps the TCP by +/- a fixed
linear amount along X/Y/Z and +/- a fixed angular amount about roll/pitch/yaw, returning
to the initial pose between every move.

Commands are sent to the UR driver's ``pose_based_cartesian_traj_controller`` via the
``cartesian_control_msgs/action/FollowCartesianTrajectory`` action. The controller/robot
performs the inverse kinematics -- this script never computes joint angles.

The current TCP pose is read from tf2 (``base`` -> ``tool0``), which works on both the real
robot and in simulation (unlike ``tcp_pose_broadcaster``, which is non-functional in sim).
"""

import math
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose

import tf2_ros
from tf_transformations import (
    quaternion_from_euler,
    quaternion_multiply,
)

from cartesian_control_msgs.action import FollowCartesianTrajectory
from cartesian_control_msgs.msg import (
    CartesianTrajectory,
    CartesianTrajectoryPoint,
)

try:
    from controller_manager_msgs.srv import SwitchController
    _HAVE_SWITCH = True
except ImportError:  # pragma: no cover - only needed for auto_switch_controllers
    _HAVE_SWITCH = False


SCALED_JTC = 'scaled_joint_trajectory_controller'


def _normalize(q):
    """Return a unit-length quaternion [x, y, z, w]."""
    n = math.sqrt(sum(c * c for c in q))
    if n == 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    return [c / n for c in q]


class CartesianPoseDemo(Node):
    """Drives the +/- X/Y/Z/R/P/Y cartesian demo sequence."""

    def __init__(self):
        super().__init__('cartesian_pose_demo')

        # --- Parameters ---------------------------------------------------
        self.controller_name = self.declare_parameter(
            'controller_name', 'pose_based_cartesian_traj_controller'
        ).value
        self.base_frame = self.declare_parameter('base_frame', 'base').value
        self.tip_frame = self.declare_parameter('tip_frame', 'tool0').value
        self.linear_step_m = self.declare_parameter('linear_step_m', 0.010).value
        self.angular_step_deg = self.declare_parameter('angular_step_deg', 10.0).value
        self.move_duration_s = self.declare_parameter('move_duration_s', 4.0).value
        self.settle_s = self.declare_parameter('settle_s', 0.5).value
        self.rotate_in_tool_frame = self.declare_parameter(
            'rotate_in_tool_frame', True
        ).value
        self.confirm_each_move = self.declare_parameter('confirm_each_move', True).value
        self.auto_switch_controllers = self.declare_parameter(
            'auto_switch_controllers', False
        ).value

        # --- tf2 ----------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Action client ------------------------------------------------
        action_ns = f'{self.controller_name}/follow_cartesian_trajectory'
        self.action_client = ActionClient(self, FollowCartesianTrajectory, action_ns)
        self._action_ns = action_ns

        self._switched = False  # whether we activated the cartesian controller ourselves

    # ------------------------------------------------------------------ setup
    def setup(self):
        """Switch controllers (optional) and wait for the action server. Returns bool."""
        if self.auto_switch_controllers:
            if not self._switch_controllers(activate=[self.controller_name],
                                            deactivate=[SCALED_JTC]):
                return False
            self._switched = True

        self.get_logger().info(f"Waiting for action server '{self._action_ns}'...")
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"Action server '{self._action_ns}' not available.\n"
                "Is the cartesian controller active? Activate it with:\n"
                f"  ros2 control switch_controllers "
                f"--deactivate {SCALED_JTC} --activate {self.controller_name}\n"
                "or re-run this node with -p auto_switch_controllers:=true"
            )
            return False
        return True

    def _switch_controllers(self, activate, deactivate):
        if not _HAVE_SWITCH:
            self.get_logger().error('controller_manager_msgs not available; '
                                    'cannot auto-switch controllers.')
            return False
        client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/controller_manager/switch_controller unavailable.')
            return False
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.STRICT
        self.get_logger().info(
            f'Switching controllers: activate={activate} deactivate={deactivate}')
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        if result is None or not result.ok:
            self.get_logger().error('Controller switch failed.')
            return False
        return True

    # --------------------------------------------------------------- get pose
    def get_initial_pose(self, timeout_s=10.0):
        """Look up base->tip and return a geometry_msgs/Pose, or None on failure."""
        self.get_logger().info(
            f"Looking up current TCP pose ({self.base_frame} -> {self.tip_frame})...")
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, self.tip_frame, rclpy.time.Time())
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

    # ----------------------------------------------------------- pose offset
    def offset_pose(self, base_pose, dx, dy, dz, droll, dpitch, dyaw):
        """Return base_pose translated by (dx,dy,dz) and rotated by (droll,dpitch,dyaw)."""
        target = Pose()
        target.position.x = base_pose.position.x + dx
        target.position.y = base_pose.position.y + dy
        target.position.z = base_pose.position.z + dz

        q_init = [base_pose.orientation.x, base_pose.orientation.y,
                  base_pose.orientation.z, base_pose.orientation.w]
        q_delta = quaternion_from_euler(droll, dpitch, dyaw)
        if self.rotate_in_tool_frame:
            # Body-frame rotation: rotate about the tool's own axes.
            q_target = quaternion_multiply(q_init, q_delta)
        else:
            # Base-frame rotation: rotate about the fixed base axes.
            q_target = quaternion_multiply(q_delta, q_init)
        q_target = _normalize(q_target)

        target.orientation.x = q_target[0]
        target.orientation.y = q_target[1]
        target.orientation.z = q_target[2]
        target.orientation.w = q_target[3]
        return target

    # ------------------------------------------------------------- send goal
    def send_pose(self, pose, duration_s):
        """Send a single-point cartesian trajectory and block until done. Returns bool."""
        goal = FollowCartesianTrajectory.Goal()
        traj = CartesianTrajectory()
        traj.header.frame_id = self.base_frame
        traj.header.stamp = self.get_clock().now().to_msg()

        point = CartesianTrajectoryPoint()
        point.pose = pose
        secs = int(duration_s)
        point.time_from_start = Duration(sec=secs,
                                         nanosec=int((duration_s - secs) * 1e9))
        traj.points.append(point)
        goal.trajectory = traj

        send_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Goal rejected by controller.')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            self.get_logger().error('No result returned for goal.')
            return False

        error_code = result.result.error_code
        if error_code != 0:
            self.get_logger().error(
                f'Move failed: error_code={error_code} '
                f'{result.result.error_string}')
            return False
        return True

    def _sleep(self, seconds):
        if seconds <= 0:
            return
        deadline = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    # ----------------------------------------------------------------- run
    def run(self):
        initial_pose = self.get_initial_pose()
        if initial_pose is None:
            return False

        p = initial_pose.position
        self.get_logger().info(
            f'Initial TCP position: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} (m)')

        lin = self.linear_step_m
        ang = math.radians(self.angular_step_deg)

        # (label, dx, dy, dz, droll, dpitch, dyaw)
        moves = [
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

        for label, dx, dy, dz, dr, dp, dyaw in moves:
            if self.confirm_each_move:
                ans = input(f'\n[{label}] Press Enter to move (or q + Enter to quit): ')
                if ans.strip().lower() == 'q':
                    self.get_logger().info('Quit requested.')
                    break

            self.get_logger().info(f'--> {label}')
            target = self.offset_pose(initial_pose, dx, dy, dz, dr, dp, dyaw)
            if not self.send_pose(target, self.move_duration_s):
                self.get_logger().error('Stopping demo due to move failure.')
                return False

            self.get_logger().info('--> recentering to initial pose')
            if not self.send_pose(initial_pose, self.move_duration_s):
                self.get_logger().error('Stopping demo due to recenter failure.')
                return False

            self._sleep(self.settle_s)

        self.get_logger().info('Demo complete.')
        return True

    # ------------------------------------------------------------- shutdown
    def teardown(self):
        if self._switched:
            self.get_logger().info('Restoring scaled_joint_trajectory_controller...')
            self._switch_controllers(activate=[SCALED_JTC],
                                     deactivate=[self.controller_name])


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
        node.teardown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
