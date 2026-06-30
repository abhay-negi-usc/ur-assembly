#!/usr/bin/env python3
"""Robotiq 2F-85 gripper demo for a UR control box (ROS2 Jazzy).

Gripper only -- this node never commands the arm.

Drives the gripper through a sweep of open -> partial -> closed -> open positions using the
ros2_control ``robotiq_gripper_controller``. That controller is a
``parallel_gripper_action_controller/GripperActionController``, whose action server
(default ``/robotiq_gripper_controller/gripper_cmd``) speaks
``control_msgs/action/ParallelGripperCommand`` -- NOT the older ``GripperCommand``.

A ParallelGripperCommand goal carries a ``sensor_msgs/JointState`` whose ``position[0]`` is
the target joint angle. For the 2F-85 that runs 0.0 (open) -> ~0.8 (closed). Positions here
are interpolated from ``open_position`` to ``closed_position`` by a "closedness" fraction.
``effort``/``velocity`` are sent too, but are only honored if the hardware exposes those
command interfaces (the stock Robotiq config commands position only).
"""

import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import ParallelGripperCommand
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


class GripperDemo(Node):
    """Cycles a Robotiq gripper through a set of positions via ParallelGripperCommand."""

    def __init__(self):
        super().__init__('gripper_demo')

        self.action_name = self.declare_parameter(
            'action_name', '/robotiq_gripper_controller/gripper_cmd').value
        self.joint_name = self.declare_parameter(
            'joint_name', 'robotiq_85_left_knuckle_joint').value
        self.open_position = self.declare_parameter('open_position', 0.0).value
        self.closed_position = self.declare_parameter('closed_position', 0.8).value
        self.max_effort = self.declare_parameter('max_effort', 50.0).value
        self.max_velocity = self.declare_parameter('max_velocity', 0.5).value
        self.dwell_s = self.declare_parameter('dwell_s', 1.5).value
        self.cycles = self.declare_parameter('cycles', 1).value
        # "Closedness" fractions to step through each cycle (0 = open, 1 = closed).
        self.fractions = self.declare_parameter(
            'fractions', [0.0, 0.25, 0.5, 0.75, 1.0, 0.0]).value
        # Optionally (re)activate the gripper before the demo.
        self.activate_first = self.declare_parameter('activate_first', False).value
        self.activation_service = self.declare_parameter(
            'activation_service',
            '/robotiq_activation_controller/reactivate_gripper').value

        self.client = ActionClient(self, ParallelGripperCommand, self.action_name)

    # ------------------------------------------------------------------ setup
    def setup(self):
        if self.activate_first and not self._activate():
            return False

        self.get_logger().info(f"Waiting for gripper action server '{self.action_name}'...")
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"Action server '{self.action_name}' not available (or wrong type). "
                "This node expects control_msgs/action/ParallelGripperCommand. Check:\n"
                "  ros2 control list_controllers | grep -i grip   (-> active)\n"
                "  ros2 action info /robotiq_gripper_controller/gripper_cmd -t")
            return False
        return True

    def _activate(self):
        client = self.create_client(Trigger, self.activation_service)
        self.get_logger().info(f"Activating gripper via '{self.activation_service}'...")
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                f"Activation service '{self.activation_service}' unavailable; skipping.")
            return True
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        resp = future.result()
        if resp is None or not resp.success:
            self.get_logger().error('Gripper activation failed.')
            return False
        self.get_logger().info('Gripper activated.')
        return True

    # ------------------------------------------------------------- send goal
    def send_position(self, position, label):
        goal = ParallelGripperCommand.Goal()
        cmd = JointState()
        cmd.name = [self.joint_name]
        cmd.position = [float(position)]
        cmd.velocity = [float(self.max_velocity)]
        cmd.effort = [float(self.max_effort)]
        goal.command = cmd

        self.get_logger().info(f'--> {label}: position={position:.3f}')
        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Goal rejected by gripper controller.')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            self.get_logger().error('No result returned for gripper goal.')
            return False

        r = wrapped.result
        pos = r.state.position[0] if r.state.position else float('nan')
        self.get_logger().info(
            f'    reached_goal={r.reached_goal} stalled={r.stalled} position={pos:.3f}')
        # A stall (object grasped) is a normal, successful outcome -- not a failure.
        return True

    def _sleep(self, seconds):
        if seconds <= 0:
            return
        deadline = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    # ----------------------------------------------------------------- run
    def run(self):
        span = self.closed_position - self.open_position
        for cycle in range(max(int(self.cycles), 1)):
            self.get_logger().info(f'==== Gripper cycle {cycle + 1}/{self.cycles} ====')
            for frac in self.fractions:
                frac = min(max(float(frac), 0.0), 1.0)
                position = self.open_position + frac * span
                label = f'{int(round(frac * 100))}% closed'
                if not self.send_position(position, label):
                    return False
                self._sleep(self.dwell_s)
        self.get_logger().info('Gripper demo complete.')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = GripperDemo()
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
