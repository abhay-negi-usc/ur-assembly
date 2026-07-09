#!/usr/bin/env python3
"""Kinematic assembly demo for the UR10e (ROS2 Jazzy) -- no vision, no gripper.

The object is rigidly attached to the flange. Ground truth from config:
  * assembled_pose   -- the TOOL0 pose in base at assembly (poll base_link->tool0 at a good mate;
    directly measurable),
  * held_object_pose -- the held object w.r.t. tool0,
  * an assembly TRAJECTORY -- a CSV of held-object poses w.r.t. the TARGET object.

The target object's pose in base is anchored from the assembled state (the LAST waypoint), where the
held object is simultaneously at traj[-1] w.r.t. the target object and at assembled_pose*held w.r.t.
base:
  T_base_targetobj = assembled_pose * held_object_pose * inv(traj[-1])
and each waypoint is commanded as
  tool0(row) = T_base_targetobj * traj_row * inv(held_object_pose)
(the last row collapses back to assembled_pose exactly). The robot:

  1. moves to an assembly STAND-OFF (the assembled pose backed off `standoff_distance_m` along
     `standoff_axis`, in the TARGET OBJECT frame),
  2. executes the trajectory (IK -> joint motion) under POSITION or ADMITTANCE control (tunable,
     with an optional force-guarded stop),
  3. optionally disassembles (reverse trajectory) and/or returns home.

IK is chained (each waypoint seeded from the previous) so the joint path is continuous.
"""

import csv
import os

import numpy as np
import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tf_transformations import euler_matrix, quaternion_from_matrix

from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes
from controller_manager_msgs.srv import SwitchController
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from std_srvs.srv import Trigger

UR_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]


# --------------------------------------------------------------- transform helpers
def xyzrpy_to_matrix(xyz, rpy):
    m = euler_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    m[0, 3], m[1, 3], m[2, 3] = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return m


def matrix_to_pose(m):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = m[0, 3], m[1, 3], m[2, 3]
    q = quaternion_from_matrix(m)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    return pose


def translation_matrix(vec):
    m = np.eye(4)
    m[0, 3], m[1, 3], m[2, 3] = vec[0], vec[1], vec[2]
    return m


def _duration(seconds):
    secs = int(seconds)
    return Duration(sec=secs, nanosec=int((seconds - secs) * 1e9))


class KinematicAssembly(Node):
    """Ground-truth, trajectory-driven assembly under position or admittance control."""

    def __init__(self, node_name='kinematic_assembly', default_config=None):
        super().__init__(node_name)

        cfg_path = self.declare_parameter(
            'config_file', default_config or self._default_config()).value
        with open(cfg_path, 'r') as f:
            c = yaml.safe_load(f) or {}
        self.get_logger().info(f'Config: {cfg_path}')
        self._cfg_dir = os.path.dirname(cfg_path)

        self.base_frame = c.get('base_frame', 'base_link')
        self.tip_frame = c.get('tip_frame', 'tool0')
        self.planning_group = c.get('planning_group', 'ur_manipulator')
        self.controller_action = c.get(
            'controller_action', '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        self.ik_timeout = float(c.get('ik_timeout_s', 2.0))
        self.ik_attempts = int(c.get('ik_attempts', 12))
        self.ik_avoid_collisions = bool(c.get('avoid_collisions', False))
        self.joint_names = list(UR_JOINTS)

        # assembled_pose is the TOOL0 pose in base at assembly -- poll base_link->tool0 at a good
        # mate and paste it here (directly measurable).
        self.T_base_assembled = xyzrpy_to_matrix(**self._xyzrpy(c.get('assembled_pose', {})))
        # held object w.r.t. tool0 -- needed because the trajectory is authored as held-object poses
        # w.r.t. the TARGET object (see run() for how the target frame is anchored).
        self.T_tool0_held = xyzrpy_to_matrix(**self._xyzrpy(c.get('held_object_pose', {})))
        self.T_base_targetobj = None    # anchored in run() from assembled_pose + held + traj[-1]
        self.T_targetobj_held_assembled = None   # traj[-1]: held pose w.r.t. target at assembly
        self.standoff_dist = float(c.get('standoff_distance_m', 0.05))
        self.standoff_axis = np.array(c.get('standoff_axis', [0.0, 0.0, 1.0]), dtype=float)

        self.trajectory_csv = c.get('trajectory_csv', 'assembly_trajectory.csv')
        self.traj_deg = bool(c.get('trajectory_angles_deg', False))

        self.control_mode = str(c.get('control_mode', 'position')).lower()
        self.standoff_move_duration = float(c.get('standoff_move_duration_s', 4.0))
        self.waypoint_dt = float(c.get('waypoint_dt_s', 1.0))
        self.settle_s = float(c.get('settle_s', 0.5))
        self.move_timeout = float(c.get('move_timeout_s', 120.0))
        self.return_home_after = bool(c.get('return_home_after', True))
        # Disassembly: after assembling, run the trajectory in REVERSE (extraction), then stand-off,
        # then the initial pose. Takes precedence over return_home_after (it also ends at home).
        self.disassemble_after = bool(c.get('disassemble_after', False))
        self.confirm = bool(c.get('confirm_each_step', True))
        self.debug = (bool(c.get('debug', False))
                      or bool(self.declare_parameter('debug', False).value))

        a = c.get('admittance', {}) or {}
        self.adm_controller = a.get('controller', 'admittance_controller')
        self.position_controller = a.get('position_controller', 'scaled_joint_trajectory_controller')
        self.switch_service = a.get('switch_service', '/controller_manager/switch_controller')
        self.reference_topic = a.get('reference_topic', '/admittance_controller/joint_references')
        self.reference_rate = float(a.get('reference_rate_hz', 20.0))
        self.ft_zero_service = a.get('ft_zero_service', '/io_and_status_controller/zero_ftsensor')
        self.tare_before = bool(a.get('tare_before', True))
        self.apply_params = bool(a.get('apply_params', True))
        self.adm_mass = a.get('mass', [5.0] * 3 + [0.5] * 3)
        self.adm_damping = a.get('damping_ratio', [1.0] * 6)
        self.adm_stiffness = a.get('stiffness', [200.0] * 3 + [15.0] * 3)
        self.adm_selected = a.get('selected_axes', [True] * 6)
        self.wrench_topic = a.get('wrench_topic', '/force_torque_sensor_broadcaster/wrench')
        self.max_force = float(a.get('max_force_n', 30.0))
        self.max_torque = float(a.get('max_torque_nm', 8.0))

        # Interfaces
        self._joints = {}
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(self, FollowJointTrajectory, self.controller_action)

        self._in_compliance = False
        if self.control_mode == 'admittance':
            self.switch_client = self.create_client(SwitchController, self.switch_service)
            self.ft_zero_client = self.create_client(Trigger, self.ft_zero_service)
            # In Jazzy each controller runs as its OWN node named after the controller, so its
            # parameters (admittance.*) live there -- NOT under /controller_manager.
            self.setparam_client = self.create_client(
                SetParameters, f'/{self.adm_controller}/set_parameters')
            self.ref_pub = self.create_publisher(JointTrajectoryPoint, self.reference_topic, 10)
            self._wrench = None
            self.create_subscription(WrenchStamped, self.wrench_topic, self._wrench_cb, 10)

    @staticmethod
    def _default_config():
        src = os.path.normpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', 'config',
            'kinematic_assembly.yaml'))
        if os.path.isfile(src):
            return src
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('ur_kinematic_assembly_demo'),
                            'config', 'kinematic_assembly.yaml')

    @staticmethod
    def _xyzrpy(d):
        return {'xyz': d.get('xyz', [0.0, 0.0, 0.0]), 'rpy': d.get('rpy', [0.0, 0.0, 0.0])}

    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self._joints[name] = pos

    def _wrench_cb(self, msg):
        f, t = msg.wrench.force, msg.wrench.torque
        self._wrench = ((f.x ** 2 + f.y ** 2 + f.z ** 2) ** 0.5,
                        (t.x ** 2 + t.y ** 2 + t.z ** 2) ** 0.5)

    # ------------------------------------------------------------------ setup
    def setup(self):
        self.get_logger().info('[1/3] Waiting for /compute_ik (move_group)...')
        if not self.ik_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error('/compute_ik unavailable (start move_group).')
            return False
        self.get_logger().info(f"[2/3] Waiting for '{self.controller_action}'...")
        if not self.traj_client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(
                f"'{self.controller_action}' unavailable -- is the arm controller active?")
            return False
        self.get_logger().info('[3/3] Waiting for /joint_states...')
        if not self._wait_joints(10.0):
            self.get_logger().error('No /joint_states.')
            return False
        if self.control_mode == 'admittance' and not self.switch_client.wait_for_service(
                timeout_sec=3.0):
            self.get_logger().warn(
                f"'{self.switch_service}' not up -- admittance mode needs the controller_manager "
                'with a LOADED admittance_controller (see ur_admittance_demo).')
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

    def _sleep(self, seconds):
        if seconds <= 0:
            return
        deadline = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _confirm(self, label):
        if not self.confirm:
            return True
        return input(f'\n[{label}] Enter to proceed (q to abort): ').strip().lower() != 'q'

    # ------------------------------------------------------------ trajectory / geometry
    def _load_trajectory(self):
        """Read the CSV into a list of T_targetobj_held 4x4 matrices, or None on error.

        Each row is a held-object pose w.r.t. the TARGET object; the LAST row is the assembled
        state (used to anchor the target frame in base -- see run())."""
        path = self.trajectory_csv
        if not os.path.isabs(path):
            cand = os.path.join(self._cfg_dir, path)
            path = cand if os.path.isfile(cand) else path
        if not os.path.isfile(path):
            self.get_logger().error(f'Trajectory CSV not found: {path}')
            return None
        mats = []
        with open(path, newline='') as f:
            for row in csv.reader(f):
                if not row or row[0].lstrip().startswith('#'):
                    continue
                try:
                    vals = [float(v) for v in row[:6]]
                except ValueError:
                    continue                       # header or non-numeric row
                if len(vals) < 6:
                    continue
                rpy = vals[3:6]
                if self.traj_deg:
                    rpy = [np.radians(a) for a in rpy]
                mats.append(xyzrpy_to_matrix(vals[0:3], rpy))
        if not mats:
            self.get_logger().error(f'No trajectory waypoints parsed from {path}.')
            return None
        self.get_logger().info(f'Loaded {len(mats)} trajectory waypoint(s) from {path}.')
        return mats

    def _tool0_at(self, T_targetobj_held):
        """tool0 Pose for a held-object pose expressed w.r.t. the target object (a CSV row)."""
        T_base_held = self.T_base_targetobj @ T_targetobj_held
        return matrix_to_pose(T_base_held @ np.linalg.inv(self.T_tool0_held))

    def _standoff_pose(self):
        # Stand-off = the assembled held-object pose backed off standoff_dist along standoff_axis in
        # the TARGET OBJECT frame, then converted to a tool0 target.
        T_standoff_held = (translation_matrix(self.standoff_axis * self.standoff_dist)
                           @ self.T_targetobj_held_assembled)
        return self._tool0_at(T_standoff_held)

    # --------------------------------------------------------------------- IK
    def solve_ik(self, tool0_pose, seed):
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = tool0_pose
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

    def _ik_chain(self, poses, seed):
        """IK a list of tool0 poses, chaining seeds for a continuous joint path. None on failure."""
        joints, s = [], list(seed)
        for i, pose in enumerate(poses):
            j = self.solve_ik(pose, s)
            if j is None:
                self.get_logger().error(f'IK failed for waypoint {i}.')
                return None
            joints.append(j)
            s = j
        return joints

    # ------------------------------------------------------------ position control
    def send_joint_trajectory(self, joint_points, dt, label):
        """Send one multi-point JointTrajectory (times = cumulative dt) and wait for the result."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(self.joint_names)
        for i, jp in enumerate(joint_points):
            pt = JointTrajectoryPoint()
            pt.positions = list(jp)
            pt.time_from_start = _duration(dt * (i + 1))
            goal.trajectory.points.append(pt)

        sf = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sf, timeout_sec=10.0)
        gh = sf.result()
        if gh is None or not gh.accepted:
            self.get_logger().error(f'[{label}] trajectory goal rejected / no response.')
            return False
        rf = gh.get_result_async()
        deadline = self.get_clock().now().nanoseconds + int(self.move_timeout * 1e9)
        while rclpy.ok() and not rf.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info(f'[{label}] executing...', throttle_duration_sec=5.0)
            if self.get_clock().now().nanoseconds > deadline:
                self.get_logger().error(
                    f'[{label}] not finished after {self.move_timeout:.0f}s -- goal accepted but '
                    'not executing (External Control playing? speed slider up? e-stop?). Canceling.')
                gh.cancel_goal_async()
                return False
        res = rf.result()
        if res is None or res.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            code = None if res is None else res.result.error_code
            self.get_logger().error(f'[{label}] move failed (error_code={code}).')
            return False
        return True

    # ------------------------------------------------------------ admittance control
    def _contact_exceeded(self):
        if self._wrench is None:
            return False
        force, torque = self._wrench
        if self.max_force > 0.0 and force >= self.max_force:
            self.get_logger().info(
                f'Contact force {force:.1f} N >= {self.max_force:.1f} N -- stopping.')
            return True
        if self.max_torque > 0.0 and torque >= self.max_torque:
            self.get_logger().info(
                f'Contact torque {torque:.2f} Nm >= {self.max_torque:.2f} Nm -- stopping.')
            return True
        return False

    def _tare_ft(self):
        if not self.ft_zero_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f"FT zero '{self.ft_zero_service}' unavailable; skipping tare.")
            return
        future = self.ft_zero_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        resp = future.result()
        if resp is None or not resp.success:
            self.get_logger().warn('F/T tare did not report success; continuing.')
        else:
            self.get_logger().info('F/T sensor tared.')

    def _apply_admittance_params(self):
        """Best-effort push of the tunable admittance gains onto the running controller."""
        if not self.apply_params:
            return
        if not self.setparam_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                f"/{self.adm_controller}/set_parameters unavailable (is it loaded?); using the "
                'controller\'s loaded params (set them in ur_admittance_demo yaml instead).')
            return
        # admittance.mass/damping_ratio/stiffness/selected_axes are all dynamic (not read_only),
        # and enable_parameter_update_without_reactivation defaults to true, so these apply live.
        params = []
        for key, vals in (('mass', self.adm_mass), ('damping_ratio', self.adm_damping),
                          ('stiffness', self.adm_stiffness)):
            p = Parameter()
            p.name = f'admittance.{key}'
            p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                                     double_array_value=[float(v) for v in vals])
            params.append(p)
        p = Parameter()
        p.name = 'admittance.selected_axes'
        p.value = ParameterValue(type=ParameterType.PARAMETER_BOOL_ARRAY,
                                 bool_array_value=[bool(v) for v in self.adm_selected])
        params.append(p)
        req = SetParameters.Request()
        req.parameters = params
        future = self.setparam_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        resp = future.result()
        if resp is None or not all(r.successful for r in resp.results):
            reasons = '' if resp is None else '; '.join(
                r.reason for r in resp.results if not r.successful and r.reason)
            self.get_logger().warn(f'Some admittance params were not applied. {reasons}')
        else:
            self.get_logger().info('Applied tunable admittance params (live).')

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
                f"switch_controller failed (activate={activate}, deactivate={deactivate}). "
                f"Is '{self.adm_controller}' loaded?")
            return False
        return True

    def _switch_to_position(self):
        if not self._in_compliance:
            return True
        if self._switch_controllers([self.position_controller], [self.adm_controller]):
            self._in_compliance = False
            return True
        return False

    def run_admittance(self, standoff_joints, waypoint_joints):
        """Stream the trajectory as joint references under admittance, with a force-guarded stop.
        Assumes the arm is already at the stand-off (position control)."""
        self._apply_admittance_params()
        if self.tare_before:
            self._tare_ft()
        if not self._switch_controllers([self.adm_controller], [self.position_controller]):
            return False
        self._in_compliance = True
        self._sleep(0.3)   # let a fresh (tared) wrench sample arrive
        self.get_logger().info(
            f'Compliance ON; streaming {len(waypoint_joints)} waypoints, force-guarded at '
            f'{self.max_force:.0f} N / {self.max_torque:.1f} Nm...')

        period = 1.0 / self.reference_rate
        prev = list(standoff_joints)
        last_ref = list(standoff_joints)
        seated = False
        for i, target in enumerate(waypoint_joints):
            steps = max(1, int(self.waypoint_dt * self.reference_rate))
            for k in range(1, steps + 1):
                if not rclpy.ok():
                    break
                if self._contact_exceeded():
                    seated = True
                    break
                alpha = k / steps
                last_ref = [(1.0 - alpha) * p + alpha * t for p, t in zip(prev, target)]
                pt = JointTrajectoryPoint()
                pt.positions = last_ref
                self.ref_pub.publish(pt)
                self._sleep(period)
            if seated:
                break
            prev = list(target)
            self.get_logger().info(f'  waypoint {i + 1}/{len(waypoint_joints)} reached (compliant)')

        # Hold the last reference so it settles.
        for _ in range(max(1, int(self.reference_rate))):
            if not rclpy.ok():
                break
            pt = JointTrajectoryPoint()
            pt.positions = list(last_ref)
            self.ref_pub.publish(pt)
            self._sleep(period)
        self.get_logger().info(
            'Stopped on contact (seated).' if seated else 'Trajectory complete.')
        return True

    # --------------------------------------------------------------------- run
    def run(self):
        home_joints = self._current_joints()

        traj_mats = self._load_trajectory()
        if traj_mats is None:
            return False
        # Anchor the target-object frame in base: at the assembled state (the LAST waypoint) the
        # held object is at traj[-1] w.r.t. the target object AND at assembled_pose*held w.r.t. base.
        T_base_held_assembled = self.T_base_assembled @ self.T_tool0_held
        self.T_targetobj_held_assembled = traj_mats[-1]
        self.T_base_targetobj = T_base_held_assembled @ np.linalg.inv(self.T_targetobj_held_assembled)
        traj_poses = [self._tool0_at(m) for m in traj_mats]

        # IK the stand-off + trajectory (chained seeds for a continuous joint path).
        standoff_pose = self._standoff_pose()
        standoff_joints = self.solve_ik(standoff_pose, home_joints)
        if standoff_joints is None:
            self.get_logger().error('IK failed for the stand-off pose.')
            return False
        waypoint_joints = self._ik_chain(traj_poses, standoff_joints)
        if waypoint_joints is None:
            return False

        # 1. Move to the stand-off (position control).
        if not self._confirm('move to assembly stand-off'):
            self.get_logger().info('Aborted by user.')
            return False
        if not self.send_joint_trajectory([standoff_joints], self.standoff_move_duration,
                                          'stand-off'):
            return False
        self._sleep(self.settle_s)

        # 2. Execute the assembly trajectory in the selected mode.
        mode = 'ADMITTANCE' if self.control_mode == 'admittance' else 'POSITION'
        if not self._confirm(f'execute assembly trajectory ({mode})'):
            self.get_logger().info('Aborted by user.')
            return False
        if not self._execute_trajectory(standoff_joints, waypoint_joints, 'assembly'):
            return False
        self._sleep(self.settle_s)
        self.get_logger().info('Assembly trajectory done.')

        # 3a. Optional DISASSEMBLY: reverse the trajectory (extraction) -> stand-off -> initial pose.
        if self.disassemble_after:
            reverse_targets = list(reversed(waypoint_joints))[1:]   # from assembled back to start
            if not self._confirm(f'DISASSEMBLE: reverse trajectory ({mode})'):
                self.get_logger().info('Aborted by user.')
                return False
            if not self._execute_trajectory(waypoint_joints[-1], reverse_targets, 'disassembly'):
                return False
            self._sleep(self.settle_s)
            self.get_logger().info('Disassembly trajectory done.')
            if not (self._confirm('retract to assembly stand-off')
                    and self.send_joint_trajectory([standoff_joints], self.standoff_move_duration,
                                                    'stand-off')
                    and self._confirm('return to initial pose')
                    and self.send_joint_trajectory([home_joints], self.standoff_move_duration,
                                                    'home')):
                return False
        # 3b. Otherwise, optional simple wind-down: retract to stand-off -> home.
        elif self.return_home_after:
            if not (self._confirm('retract to stand-off')
                    and self.send_joint_trajectory([standoff_joints], self.standoff_move_duration,
                                                    'retract')
                    and self._confirm('return home')
                    and self.send_joint_trajectory([home_joints], self.standoff_move_duration,
                                                    'home')):
                return False

        self.get_logger().info('Kinematic assembly complete.')
        return True

    def _execute_trajectory(self, start_joints, target_joints, label):
        """Run target_joints in the selected control mode. start_joints = the current config (the
        admittance ramp start). Restores position control afterwards so the arm is never left
        compliant. Returns bool; a no-op (True) if there are no targets."""
        if not target_joints:
            return True
        ok = False
        try:
            if self.control_mode == 'admittance':
                ok = self.run_admittance(start_joints, target_joints)
            else:
                ok = self.send_joint_trajectory(target_joints, self.waypoint_dt, label)
        finally:
            if self._in_compliance:
                self.get_logger().warn('Restoring position control (was left compliant).')
                self._switch_to_position()
        return ok


def main(args=None):
    rclpy.init(args=args)
    node = KinematicAssembly()
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
