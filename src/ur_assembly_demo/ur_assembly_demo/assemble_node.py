#!/usr/bin/env python3
"""Fiducial-guided pick-and-ASSEMBLE demo for the UR10e + Robotiq 2F-85 (ROS2 Jazzy).

Subclasses ur_pick_place_demo's PickPlace to reuse the ENTIRE pick pipeline (detect -> visual
approach -> estimate -> grasp -> close -> lift), then runs an assembly sequence:

  view pose -> detect target marker -> estimate target object -> pre-align -> stand-off
  -> tare F/T + switch to admittance control -> compliant mate to the assembled pose
  -> open gripper -> switch back to position control -> retract (stand-off -> view -> home)

The compliant mate ramps a JOINT reference on the ros2_control admittance_controller from the
stand-off to the assembled pose, while the controller yields to contact forces. The
admittance_controller must be LOADED (inactive) on the controller_manager beforehand -- see
ur_admittance_demo for install/spawn.

Config: config/assemble.yaml (the FULL pick config + an `assembly:` section).
"""

import os

import numpy as np
import rclpy

from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController
from std_srvs.srv import Trigger

from ur_pick_place_demo.pick_place_node import (
    PickPlace, matrix_to_pose, translation_matrix, xyzrpy_to_matrix)


class Assemble(PickPlace):
    """Pick like pick_place, then compliantly assemble into a fiducial-located target."""

    def __init__(self):
        super().__init__(node_name='assemble', default_config=self._assemble_config())
        a = self.cfg.get('assembly', {}) or {}

        # View pose = desired CAMERA pose w.r.t. base (arm moves tool0 = view * inv(tool0->cam)).
        self.T_base_view = xyzrpy_to_matrix(**self._xyzrpy(a.get('view_pose', {})))

        tm = a.get('target_marker', {}) or {}
        self.target_marker_frame = tm.get('frame') or f"camera1_marker_{int(tm.get('id', 0))}"
        self.T_targetobj_marker = xyzrpy_to_matrix(
            **self._xyzrpy(a.get('target_object_marker', {})))
        self.T_targetobj_held_final = xyzrpy_to_matrix(**self._xyzrpy(a.get('assembled_pose', {})))
        self.assembly_axis = np.array(a.get('approach_axis', [0.0, 0.0, 1.0]), dtype=float)
        self.prealign_dist = float(a.get('prealign_distance_m', 0.10))
        self.standoff_dist = float(a.get('standoff_distance_m', 0.03))

        comp = a.get('compliance', {}) or {}
        self.admittance_controller = comp.get('admittance_controller', 'admittance_controller')
        self.position_controller = comp.get(
            'position_controller', 'scaled_joint_trajectory_controller')
        self.switch_service = comp.get('switch_service', '/controller_manager/switch_controller')
        self.reference_topic = comp.get(
            'reference_topic', '/admittance_controller/joint_references')
        self.ft_zero_service = comp.get(
            'ft_zero_service', '/io_and_status_controller/zero_ftsensor')
        self.insertion_time = float(comp.get('insertion_time_s', 5.0))
        self.insertion_rate = float(comp.get('insertion_rate_hz', 20.0))
        self.insertion_hold = float(comp.get('insertion_hold_s', 1.0))
        # Force-guarded stop: halt the insertion ramp when the (tared) contact wrench reaches these.
        self.wrench_topic = comp.get('wrench_topic', '/force_torque_sensor_broadcaster/wrench')
        self.max_force = float(comp.get('max_force_n', 20.0))     # 0 disables the force check
        self.max_torque = float(comp.get('max_torque_nm', 5.0))   # 0 disables the torque check

        # Compliance interfaces.
        self.switch_client = self.create_client(SwitchController, self.switch_service)
        self.ft_zero_client = self.create_client(Trigger, self.ft_zero_service)
        self.ref_pub = self.create_publisher(JointTrajectoryPoint, self.reference_topic, 10)
        self._wrench = None
        self.create_subscription(WrenchStamped, self.wrench_topic, self._wrench_cb, 10)

        self.T_tool0_held = np.eye(4)      # set after the pick (held object w.r.t. tool0)
        self.T_base_targetobj = None       # set after detecting the target
        self._home_joints = None
        self._in_compliance = False

    @staticmethod
    def _assemble_config():
        """Source-preferring path to config/assemble.yaml (edit the yaml without rebuilding under
        --symlink-install); fall back to the installed copy."""
        src = os.path.normpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', 'config', 'assemble.yaml'))
        if os.path.isfile(src):
            return src
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory('ur_assembly_demo'), 'config', 'assemble.yaml')

    # ------------------------------------------------------------------ setup
    def setup(self):
        if not super().setup():
            return False
        if not self.switch_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                f"'{self.switch_service}' not up yet -- the compliant mate needs the "
                'controller_manager with a LOADED admittance_controller (see ur_admittance_demo).')
        return True

    # -------------------------------------------------------------- assembly geometry
    def _held_assembly_tool0(self, separation):
        """tool0 target (Pose) placing the held object `separation` m along the assembly axis from
        the assembled pose (separation 0 = fully mated). Assembly axis is in the target-obj frame."""
        T_to_held = translation_matrix(self.assembly_axis * separation) @ self.T_targetobj_held_final
        T_base_held = self.T_base_targetobj @ T_to_held
        return matrix_to_pose(T_base_held @ np.linalg.inv(self.T_tool0_held))

    def _move_held_assembly(self, separation, label):
        return self.move_tool0_to(self._held_assembly_tool0(separation), label)

    # ------------------------------------------------------------------ assembly steps
    def _go_to_view_pose(self):
        T_tool0_cam = self._tf_matrix(self.tip_frame, self.camera_frame)
        if T_tool0_cam is None:
            self.get_logger().error(f'No {self.tip_frame} -> {self.camera_frame} tf (hand-eye).')
            return False
        T_base_tool0 = self.T_base_view @ np.linalg.inv(T_tool0_cam)
        return self.move_tool0_to(matrix_to_pose(T_base_tool0), 'view pose')

    def _detect_target(self):
        T_base_marker = self._tf_matrix(
            self.base_frame, self.target_marker_frame,
            max_age_s=self.marker_max_age, timeout_s=self.reacquire_timeout)
        if T_base_marker is None:
            self.get_logger().error(
                f"Target marker '{self.target_marker_frame}' not in view at the view pose.")
            return False
        self.T_base_targetobj = T_base_marker @ np.linalg.inv(self.T_targetobj_marker)
        p = self.T_base_targetobj[:3, 3]
        self.get_logger().info(f'Target object (base): x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}')
        return True

    # ------------------------------------------------------------ compliance / control
    def _tare_ft(self):
        if not self.ft_zero_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                f"FT zero service '{self.ft_zero_service}' unavailable; skipping tare.")
            return False
        future = self.ft_zero_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        resp = future.result()
        if resp is None or not resp.success:
            self.get_logger().warn('F/T tare did not report success; continuing.')
            return False
        self.get_logger().info('F/T sensor tared.')
        return True

    def _switch_controllers(self, activate, deactivate):
        if not self.switch_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"'{self.switch_service}' unavailable.")
            return False
        req = SwitchController.Request()
        req.activate_controllers = list(activate)
        req.deactivate_controllers = list(deactivate)
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = True
        future = self.switch_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        resp = future.result()
        if resp is None or not resp.ok:
            self.get_logger().error(
                f'switch_controller failed (activate={activate}, deactivate={deactivate}). '
                f"Is '{self.admittance_controller}' loaded? (ros2 control list_controllers)")
            return False
        return True

    def _switch_to_position(self):
        if not self._in_compliance:
            return True
        if self._switch_controllers([self.position_controller], [self.admittance_controller]):
            self._in_compliance = False
            return True
        return False

    def _wrench_cb(self, msg):
        f, t = msg.wrench.force, msg.wrench.torque
        self._wrench = ((f.x ** 2 + f.y ** 2 + f.z ** 2) ** 0.5,
                        (t.x ** 2 + t.y ** 2 + t.z ** 2) ** 0.5)

    def _contact_exceeded(self):
        """True once the (tared) contact force/torque magnitude reaches the yaml limit."""
        if self._wrench is None:
            return False
        force, torque = self._wrench
        if self.max_force > 0.0 and force >= self.max_force:
            self.get_logger().info(
                f'Contact force {force:.1f} N >= max_force_n {self.max_force:.1f} N -- stopping.')
            return True
        if self.max_torque > 0.0 and torque >= self.max_torque:
            self.get_logger().info(
                f'Contact torque {torque:.2f} Nm >= max_torque_nm {self.max_torque:.2f} Nm '
                '-- stopping.')
            return True
        return False

    def _compliant_insert(self):
        """Tare F/T, switch to admittance control, and ramp a joint reference from the stand-off
        toward the assembled pose -- STOPPING as soon as the tared contact force/torque reaches the
        yaml limit (force-guarded). Then hold the last reference so the mate settles."""
        j_start = self._current_joints()
        target_pose = self._held_assembly_tool0(0.0)
        j_final = self.solve_ik(target_pose, j_start)
        if j_final is None:
            self._log_ik_failure('assembled pose', target_pose)
            return False

        self._tare_ft()   # warn-only

        if not self._switch_controllers([self.admittance_controller], [self.position_controller]):
            return False
        self._in_compliance = True
        self._sleep(0.3)   # let a fresh (tared) wrench sample arrive before we start pushing
        self.get_logger().info(
            f'Compliance ON; ramping to the assembled pose, force-guarded at '
            f'{self.max_force:.0f} N / {self.max_torque:.1f} Nm...')

        steps = max(1, int(self.insertion_time * self.insertion_rate))
        period = 1.0 / self.insertion_rate
        last_ref = list(j_start)
        seated = False
        for k in range(1, steps + 1):
            if not rclpy.ok():
                break
            if self._contact_exceeded():          # stop advancing on contact
                seated = True
                break
            alpha = k / steps
            last_ref = [(1.0 - alpha) * s + alpha * f for s, f in zip(j_start, j_final)]
            pt = JointTrajectoryPoint()
            pt.positions = last_ref
            self.ref_pub.publish(pt)
            self._sleep(period)

        # Hold the LAST commanded reference (the contact point, or the assembled pose) to settle.
        for _ in range(max(1, int(self.insertion_hold * self.insertion_rate))):
            if not rclpy.ok():
                break
            pt = JointTrajectoryPoint()
            pt.positions = list(last_ref)
            self.ref_pub.publish(pt)
            self._sleep(period)

        if seated:
            self.get_logger().info('Stopped on the contact limit (part seated).')
        else:
            self.get_logger().info('Ramp complete (contact limit not reached).')
        return True

    # --------------------------------------------------------------------- run
    def _assemble(self):
        return (
            self._do('go to view pose', self._go_to_view_pose)
            and self._do('detect target marker', self._detect_target)
            and self._do('pre-align to assembly',
                         lambda: self._move_held_assembly(self.prealign_dist, 'prealign'))
            and self._do('assembly stand-off',
                         lambda: self._move_held_assembly(self.standoff_dist, 'assembly-standoff'))
            and self._do('tare + compliant mate', self._compliant_insert)
            and self._do('open gripper (release)',
                         lambda: self.gripper_to(self.gripper_open, 'open'))
            and self._do('switch to position control', self._switch_to_position)
            and self._do('retract to assembly stand-off',
                         lambda: self._move_held_assembly(self.standoff_dist, 'retract-standoff'))
            and self._do('back to view pose', self._go_to_view_pose)
            and self._do('return home', lambda: self.send_joints(self._home_joints))
        )

    def run(self):
        self._home_joints = self._current_joints()
        if not self._pick(self._home_joints):
            return False
        # Held object pose relative to tool0 -- fixed once grasped.
        self.T_tool0_held = self.T_tool0_grasp @ np.linalg.inv(self.T_object_grasp)
        self.get_logger().info('Picked. Starting assembly.')
        ok = False
        try:
            ok = self._assemble()
        finally:
            if self._in_compliance:   # never leave the arm compliant on exit/abort
                self.get_logger().warn('Restoring position control (was left compliant).')
                self._switch_to_position()
        if ok:
            self.get_logger().info('Pick-and-assemble complete.')
        return ok


def main(args=None):
    rclpy.init(args=args)
    node = Assemble()
    try:
        if node.setup():
            node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
        if node._in_compliance:
            node._switch_to_position()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
