#!/usr/bin/env python3
"""Fiducial-guided pick-and-place demo for the UR10e + Robotiq 2F-85 (ROS2 Jazzy).

Pipeline:
  1. Look up the object's marker in the base frame via tf2 (needs ur_vision_demo detecting the
     marker and ur_tf_demo's hand-eye transform connecting camera -> tool0 -> base).
  2. Compose:  T_base_object = T_base_marker * inv(T_object_marker)
               T_base_grasp  = T_base_object * T_object_grasp     (grasp-TCP target in base)
  3. Run the sequence (each arm move = MoveIt /compute_ik for tool0 + a FollowJointTrajectory
     goal to scaled_joint_trajectory_controller; gripper via the ParallelGripperCommand action):
       open -> pre-grasp -> grasp -> close -> lift -> pre-place -> place -> open -> retreat -> home

The grasp pose is expressed for a grasp-TCP between the fingers; since IK solves for tool0, each
grasp-TCP target is converted to a tool0 target via inv(grasp_tcp_offset).

Assumes the integrated bringup (arm + gripper under one controller_manager), move_group (for
/compute_ik), and the vision/tf stack are all running. Place = pick shifted by place_offset.
"""

import os
import sys

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
    euler_from_quaternion, euler_matrix, quaternion_from_matrix, quaternion_matrix)

from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand
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


def translation_matrix(vec):
    m = np.eye(4)
    m[0, 3], m[1, 3], m[2, 3] = vec[0], vec[1], vec[2]
    return m


def _duration(seconds):
    secs = int(seconds)
    return Duration(sec=secs, nanosec=int((seconds - secs) * 1e9))


class PickPlace(Node):
    """Orchestrates the fiducial-guided pick-and-place."""

    def __init__(self):
        super().__init__('pick_place')

        default_cfg = os.path.join(
            self._share_dir(), 'config', 'pick_place.yaml')
        cfg_path = self.declare_parameter('config_file', default_cfg).value
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f) or {}

        c = self.cfg
        self.base_frame = c.get('base_frame', 'base_link')
        self.tip_frame = c.get('tip_frame', 'tool0')
        self.planning_group = c.get('planning_group', 'ur_manipulator')
        self.controller_action = c.get(
            'controller_action', '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        self.ik_timeout = float(c.get('ik_timeout_s', 2.0))
        self.ik_attempts = int(c.get('ik_attempts', 12))
        self.ik_avoid_collisions = bool(c.get('avoid_collisions', False))
        self.joint_names = list(UR_JOINTS)

        m = c.get('marker', {})
        self.marker_frame = m.get('frame') or f"camera1_marker_{int(m.get('id', 0))}"

        self.T_object_marker = xyzrpy_to_matrix(**self._xyzrpy(c.get('object_marker', {})))
        self.T_object_grasp = xyzrpy_to_matrix(**self._xyzrpy(c.get('object_grasp', {})))
        self.T_tool0_grasp = xyzrpy_to_matrix(**self._xyzrpy(c.get('grasp_tcp_offset', {})))

        self.approach_distance = float(c.get('approach_distance_m', 0.10))
        self.approach_axis = np.array(c.get('approach_axis', [0.0, 0.0, -1.0]), dtype=float)
        self.lift_distance = float(c.get('lift_distance_m', 0.10))
        self.lift_axis = np.array(c.get('lift_axis', [0.0, 0.0, 1.0]), dtype=float)
        self.place_offset = xyzrpy_to_matrix(
            c.get('place_offset_xyz', [0.0, 0.20, 0.0]),
            c.get('place_offset_rpy', [0.0, 0.0, 0.0]))

        # Re-servo at the standoff: re-read the marker and update the grasp target; if the marker
        # isn't in view, retract tool0 along -Z and retry.
        self.refine_at_standoff = bool(c.get('refine_at_standoff', True))
        self.marker_max_age = float(c.get('marker_max_age_s', 0.5))
        self.refine_lookup_timeout = float(c.get('refine_lookup_timeout_s', 2.0))
        self.refine_retry_step = float(c.get('refine_retry_step_m', 0.01))
        self.refine_max_retries = int(c.get('refine_max_retries', 5))
        self.T_base_grasp = np.eye(4)   # updated from the marker (initial + standoff refine)

        g = c.get('gripper', {})
        self.gripper_action_name = g.get('action', '/robotiq_gripper_controller/gripper_cmd')
        self.gripper_joint = g.get('joint', 'robotiq_85_left_knuckle_joint')
        self.gripper_open = float(g.get('open_position', 0.0))
        self.gripper_closed = float(g.get('closed_position', 0.8))
        self.gripper_effort = float(g.get('max_effort', 50.0))
        self.gripper_velocity = float(g.get('max_velocity', 0.5))

        self.move_duration = float(c.get('move_duration_s', 4.0))
        self.move_timeout = float(c.get('move_timeout_s', 60.0))
        self.settle_s = float(c.get('settle_s', 0.5))
        self.confirm = bool(c.get('confirm_each_step', True))

        # Interfaces
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._joints = {}
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(self, FollowJointTrajectory, self.controller_action)
        self.gripper_client = ActionClient(
            self, ParallelGripperCommand, self.gripper_action_name)

    @staticmethod
    def _share_dir():
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('ur_pick_place_demo')

    @staticmethod
    def _xyzrpy(d):
        return {'xyz': d.get('xyz', [0.0, 0.0, 0.0]), 'rpy': d.get('rpy', [0.0, 0.0, 0.0])}

    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self._joints[name] = pos

    # ------------------------------------------------------------------ setup
    def setup(self):
        self.get_logger().info('Waiting for /compute_ik, controller, gripper, joint_states...')
        if not self.ik_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error('/compute_ik unavailable (start move_group).')
            return False
        if not self.traj_client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(f"'{self.controller_action}' unavailable.")
            return False
        if not self.gripper_client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(
                f"Gripper action '{self.gripper_action_name}' unavailable "
                '(run the integrated bringup so the gripper controller is active).')
            return False
        if not self._wait_joints(10.0):
            self.get_logger().error('No /joint_states.')
            return False
        return True

    def _wait_joints(self, timeout_s):
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            if all(j in self._joints for j in self.joint_names):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _current_joints(self):
        return [self._joints[j] for j in self.joint_names]

    # ------------------------------------------------------------ marker lookup
    def lookup_marker(self, timeout_s=15.0, max_age_s=None):
        """Return T_base_marker (4x4) once the marker is visible, else None.

        If max_age_s is set, a transform older than that is treated as "not in view": the aruco
        node stops publishing when the marker leaves the frame, but tf2 keeps the last transform
        cached, so a plain latest-lookup would return a stale pose. We check the stamp's age and
        keep waiting for a fresh one until the timeout.
        """
        self.get_logger().info(
            f"Looking up object marker '{self.marker_frame}' in '{self.base_frame}'...")
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, self.marker_frame, Time())
                if max_age_s is not None:
                    age = (self.get_clock().now()
                           - Time.from_msg(tf.header.stamp)).nanoseconds / 1e9
                    if age > max_age_s:
                        rclpy.spin_once(self, timeout_sec=0.1)   # stale; wait for a fresh one
                        continue
                return transform_to_matrix(tf.transform)
            except tf2_ros.TransformException:
                rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().error(
            f"Marker '{self.marker_frame}' not in view (no fresh tf). Is it visible and is the "
            'hand-eye transform published (ur_vision_demo + ur_tf_demo)?')
        return None

    # ------------------------------------------------------------- arm motion
    def solve_ik(self, tool0_pose, seed):
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = tool0_pose

        # MoveIt's default KDL solver is a LOCAL search seeded from one joint state: for a
        # reachable pose whose solution lives in a different IK branch it returns -31. So we retry
        # with random seeds -- attempt 0 uses the given seed (usually the current state, keeps the
        # move small when possible), the rest are random so the solver can reach other branches.
        last_code = None
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
            last_code = None if resp is None else resp.error_code.val

        self.get_logger().error(
            f'IK failed after {self.ik_attempts} attempts (last error_code={last_code}) -- '
            'target unreachable, in self-collision, or at a singularity.')
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
        if gh is None:
            self.get_logger().error('No response to trajectory goal (controller not running?).')
            return False
        if not gh.accepted:
            self.get_logger().error('Trajectory goal rejected by the controller.')
            return False

        # Don't block forever on the result. The scaled_joint_trajectory_controller ACCEPTS a goal
        # even when the robot can't move (External Control not playing, a protective/e-stop, or the
        # speed slider at 0) -- then time-scaling is 0 and the action never completes. Poll with a
        # deadline + liveness log so a stall is visible instead of a silent hang.
        rf = gh.get_result_async()
        deadline = self.get_clock().now().nanoseconds + int(self.move_timeout * 1e9)
        while rclpy.ok() and not rf.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info('executing trajectory...', throttle_duration_sec=5.0)
            if self.get_clock().now().nanoseconds > deadline:
                self.get_logger().error(
                    f'Trajectory not finished after {self.move_timeout:.0f}s -- goal was accepted '
                    'but not executing. On the pendant: is the External Control program PLAYING, '
                    'the speed slider up, and no protective/e-stop? Canceling.')
                gh.cancel_goal_async()
                return False

        res = rf.result()
        if res is None or res.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            code = None if res is None else res.result.error_code
            self.get_logger().error(f'Move failed (error_code={code}).')
            return False
        return True

    @staticmethod
    def _fmt(xyz, rpy_deg):
        return (f'xyz=[{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}] m  '
                f'rpy=[{rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}] deg')

    def _log_ik_failure(self, label, target_pose):
        """On IK failure, log the controlled frame's current vs target pose (in base) + delta."""
        # Current controlled frame (tip_frame, e.g. tool0) in the base frame, from tf.
        cur_xyz, cur_rpy = [float('nan')] * 3, [float('nan')] * 3
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.tip_frame, Time())
            tr, rot = tf.transform.translation, tf.transform.rotation
            cur_xyz = [tr.x, tr.y, tr.z]
            cur_rpy = [np.degrees(a)
                       for a in euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])]
        except tf2_ros.TransformException:
            pass
        t = target_pose.position
        q = target_pose.orientation
        tgt_xyz = [t.x, t.y, t.z]
        tgt_rpy = [np.degrees(a) for a in euler_from_quaternion([q.x, q.y, q.z, q.w])]
        dxyz = [tgt_xyz[i] - cur_xyz[i] for i in range(3)]
        drpy = [tgt_rpy[i] - cur_rpy[i] for i in range(3)]
        dist = (dxyz[0] ** 2 + dxyz[1] ** 2 + dxyz[2] ** 2) ** 0.5

        log = self.get_logger()
        log.error(f'[{label}] IK unreachable. Controlled frame '
                  f"'{self.tip_frame}' vs '{self.base_frame}':")
        log.error(f'  initial: {self._fmt(cur_xyz, cur_rpy)}')
        log.error(f'  target:  {self._fmt(tgt_xyz, tgt_rpy)}')
        log.error(f'  delta:   {self._fmt(dxyz, drpy)}  (translation dist={dist:.3f} m)')

    def move_tool0_to(self, tool0_pose, label):
        """IK + execute a tool0 pose. Returns bool."""
        joints = self.solve_ik(tool0_pose, self._current_joints())
        if joints is None:
            self._log_ik_failure(label, tool0_pose)
            self.get_logger().error(f'[{label}] no IK solution; aborting.')
            return False
        if not self.send_joints(joints):
            self.get_logger().error(f'[{label}] move failed; aborting.')
            return False
        self._sleep(self.settle_s)
        return True

    def move_grasp_tcp_to(self, T_base_grasp_tcp, label):
        """Move so the grasp-TCP reaches T_base_grasp_tcp (convert to tool0 target first)."""
        T_base_tool0 = T_base_grasp_tcp @ np.linalg.inv(self.T_tool0_grasp)
        self.get_logger().info(f'--> {label}')
        return self.move_tool0_to(matrix_to_pose(T_base_tool0), label)

    # ------------------------------------------------------------- gripper
    def gripper_to(self, position, label):
        goal = ParallelGripperCommand.Goal()
        cmd = JointState()
        cmd.name = [self.gripper_joint]
        cmd.position = [float(position)]
        cmd.velocity = [self.gripper_velocity]
        cmd.effort = [self.gripper_effort]
        goal.command = cmd
        self.get_logger().info(f'--> gripper {label} ({position:.3f})')
        sf = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sf)
        gh = sf.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('Gripper goal rejected.')
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        return rf.result() is not None

    def _sleep(self, seconds):
        if seconds <= 0:
            return
        deadline = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _confirm(self, label):
        if not self.confirm:
            return True
        ans = input(f'\n[{label}] Enter to proceed (q to abort): ')
        return ans.strip().lower() != 'q'

    # ------------------------------------------------------ grasp pose helpers
    def _update_grasp_from_marker(self, T_base_marker, label='marker'):
        """Recompute the grasp-TCP target in base from an observed marker pose."""
        T_base_object = T_base_marker @ np.linalg.inv(self.T_object_marker)
        self.T_base_grasp = T_base_object @ self.T_object_grasp
        p = self.T_base_grasp[:3, 3]
        self.get_logger().info(
            f'Grasp target (base) [{label}]: x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}')

    # Grasp/lift/place are derived from self.T_base_grasp on demand, so a standoff refine that
    # updates self.T_base_grasp automatically feeds the grasp, lift and place moves.
    def _pre_grasp_pose(self):
        return self.T_base_grasp @ translation_matrix(self.approach_axis * self.approach_distance)

    def _lift_pose(self):
        return translation_matrix(self.lift_axis * self.lift_distance) @ self.T_base_grasp

    def _place_pose(self):
        return self.place_offset @ self.T_base_grasp

    def _pre_place_pose(self):
        return translation_matrix(self.lift_axis * self.lift_distance) @ self._place_pose()

    def _refine_at_standoff(self):
        """Re-observe the marker at the standoff and update the grasp target. If the marker isn't
        in view, retract tool0 by refine_retry_step along its own -Z and retry, up to
        refine_max_retries. Returns True when the grasp is updated (or refinement is disabled)."""
        if not self.refine_at_standoff:
            return True
        for attempt in range(self.refine_max_retries + 1):
            T_base_marker = self.lookup_marker(
                timeout_s=self.refine_lookup_timeout, max_age_s=self.marker_max_age)
            if T_base_marker is not None:
                self._update_grasp_from_marker(T_base_marker, label=f'refined@{attempt}')
                return True
            if attempt < self.refine_max_retries:
                self.get_logger().warn(
                    f'Marker not in view at standoff; retracting '
                    f'{self.refine_retry_step * 100:.1f} cm along tool0 -Z and retrying '
                    f'({attempt + 1}/{self.refine_max_retries}).')
                if not self._retract_tool0_z(-self.refine_retry_step):
                    self.get_logger().error('Retract move failed; aborting refine.')
                    return False
        self.get_logger().error(
            f'Marker still not in view after {self.refine_max_retries} retract-and-retry '
            'attempts; aborting.')
        return False

    def _retract_tool0_z(self, delta_z):
        """Translate tool0 by delta_z along its OWN Z (negative = back away). IK + execute."""
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.tip_frame, Time())
        except tf2_ros.TransformException as exc:
            self.get_logger().error(f"Can't look up '{self.tip_frame}' to retract: {exc}")
            return False
        T_base_tool0 = transform_to_matrix(tf.transform)
        T_new = T_base_tool0 @ translation_matrix([0.0, 0.0, delta_z])
        return self.move_tool0_to(matrix_to_pose(T_new), 'refine-retract')

    # ----------------------------------------------------------------- run
    def run(self):
        home_joints = self._current_joints()

        # Initial marker observation -> grasp target (sets self.T_base_grasp).
        T_base_marker = self.lookup_marker()
        if T_base_marker is None:
            return False
        self._update_grasp_from_marker(T_base_marker, label='initial')

        # Pre-grasp standoff from the initial estimate. grasp/lift/place read self.T_base_grasp at
        # execution time, so the standoff refine (which updates it) flows into them.
        pre_grasp = self._pre_grasp_pose()

        steps = [
            ('close gripper (grasp)', lambda: self.gripper_to(self.gripper_closed, 'close')),
            ('open gripper', lambda: self.gripper_to(self.gripper_open, 'open')),
            ('move to pre-grasp', lambda: self.move_grasp_tcp_to(pre_grasp, 'pre-grasp')),
            ('refine grasp at standoff', self._refine_at_standoff),
            ('move to grasp', lambda: self.move_grasp_tcp_to(self.T_base_grasp, 'grasp')),
            ('close gripper (grasp)', lambda: self.gripper_to(self.gripper_closed, 'close')),
            ('lift', lambda: self.move_grasp_tcp_to(self._lift_pose(), 'lift')),
            ('move to pre-place', lambda: self.move_grasp_tcp_to(self._pre_place_pose(), 'pre-place')),
            ('move to place', lambda: self.move_grasp_tcp_to(self._place_pose(), 'place')),
            ('open gripper (release)', lambda: self.gripper_to(self.gripper_open, 'open')),
            ('retreat', lambda: self.move_grasp_tcp_to(self._pre_place_pose(), 'retreat')),
            ('return home', lambda: self.send_joints(home_joints)),
        ]

        for label, action in steps:
            if not self._confirm(label):
                self.get_logger().info('Aborted by user.')
                return False
            if not action():
                self.get_logger().error(f'Step failed: {label}. Stopping.')
                return False

        self.get_logger().info('Pick-and-place complete.')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = PickPlace()
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
