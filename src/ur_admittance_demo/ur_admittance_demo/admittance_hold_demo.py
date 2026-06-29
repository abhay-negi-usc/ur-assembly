#!/usr/bin/env python3
"""Admittance "compliant hold" demo for a UR10e (ROS2 Jazzy).

Streams the robot's *initial* joint configuration as a constant reference to the
ros2_control ``admittance_controller``. The controller does FK on that reference, reads the
wrist force-torque sensor, and applies Cartesian admittance (F = M*a + D*v + S*(x - x_d)):
push or pull the TCP by hand and it yields to the force, then springs back to the held pose
per the configured stiffness.

The controller's non-chained reference input is *joint space*: a
``trajectory_msgs/msg/JointTrajectoryPoint`` on ``<controller>/joint_references``. The point
has no joint names, so its ``positions`` must be ordered exactly like the controller's
``joints`` parameter (the UR order below).

This demo is only meaningful on the REAL robot -- on fake hardware/sim the FT sensor reads
zero, so there is no force to comply with. Keep the e-stop in hand.
"""

import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

UR_JOINTS = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


class AdmittanceHoldDemo(Node):
    """Publishes a fixed joint reference so the arm holds compliantly."""

    def __init__(self):
        super().__init__('admittance_hold_demo')

        self.controller_name = self.declare_parameter(
            'controller_name', 'admittance_controller').value
        self.joint_names = self.declare_parameter('joint_names', UR_JOINTS).value
        self.publish_rate_hz = self.declare_parameter('publish_rate_hz', 20.0).value
        # 0.0 -> hold until Ctrl-C.
        self.hold_duration_s = self.declare_parameter('hold_duration_s', 0.0).value

        self.reference_topic = f'{self.controller_name}/joint_references'

        self._joint_positions = {}
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self._pub = self.create_publisher(
            JointTrajectoryPoint, self.reference_topic, QoSProfile(depth=10))

        self._reference = None
        self._start_ns = None
        self._timer = None

    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self._joint_positions[name] = pos

    def _wait_for_joints(self, timeout_s):
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            if all(j in self._joint_positions for j in self.joint_names):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def setup(self):
        self.get_logger().info('Waiting for /joint_states...')
        if not self._wait_for_joints(timeout_s=10.0):
            self.get_logger().error('No /joint_states received. Is the driver up?')
            return False

        # Capture the pose to hold.
        self._reference = [self._joint_positions[j] for j in self.joint_names]

        # Warn (don't fail) if the admittance controller isn't subscribed to our reference.
        deadline = self.get_clock().now().nanoseconds + int(3e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._pub.get_subscription_count() == 0:
            self.get_logger().warn(
                f"No subscriber on '{self.reference_topic}'. Is admittance_controller "
                "loaded and active? See the README for spawn/switch commands. Publishing "
                "anyway.")
        return True

    def run(self):
        joints_str = ', '.join(f'{p:.3f}' for p in self._reference)
        self.get_logger().info(f'Holding joint reference: [{joints_str}]')
        self.get_logger().info(
            '>>> Push or pull the TCP by hand: it should yield to the force and spring '
            'back. Ctrl-C to stop. <<<')

        self._start_ns = self.get_clock().now().nanoseconds
        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self._timer = self.create_timer(period, self._tick)
        rclpy.spin(self)

    def _tick(self):
        if self.hold_duration_s > 0.0:
            elapsed = (self.get_clock().now().nanoseconds - self._start_ns) / 1e9
            if elapsed >= self.hold_duration_s:
                self.get_logger().info('Hold duration elapsed; stopping.')
                raise KeyboardInterrupt
        point = JointTrajectoryPoint()
        point.positions = list(self._reference)
        self._pub.publish(point)


def main(args=None):
    rclpy.init(args=args)
    node = AdmittanceHoldDemo()
    try:
        if node.setup():
            node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()
