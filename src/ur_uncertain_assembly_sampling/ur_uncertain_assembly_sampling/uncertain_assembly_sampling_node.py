#!/usr/bin/env python3
"""Uncertain assembly sampling for the UR10e (ROS2 Jazzy).

Repeatedly executes a CHUNK (a fraction) of the ideal assembly trajectory under uncertainty, then
disassembles along the ideal reverse path, logging the ASSEMBLY data each trial. Subclasses
ur_kinematic_assembly_demo's KinematicAssembly to reuse the geometry (held object w.r.t. target
object), IK, and admittance execution.

Per trial:
  1. Resample the ideal trajectory to a fixed translational/rotational resolution (dense waypoints).
  2. Take a chunk = the first `chunk_fraction` of it.
  3. Perturb each waypoint IN THE TARGET-OBJECT FRAME (bias @ noise @ ideal):
       * bias  -- ONE uniform draw for the whole trial (systematic offset), and
       * noise -- a fresh uniform draw per waypoint (jitter).
     Execute that perturbed chunk under ADMITTANCE, recording continuously at the F/T sensor rate.
     A FORCE- AND TORQUE-guarded stop (thresholds in config) ends the chunk early.
  4. Disassemble (NOT logged): snap to the closest pose on the IDEAL dense trajectory (weighted
     translation + rotation), then run the ideal reverse from there back to the start.
Repeat for `num_trials`; the CSV is flushed after each trial.

CSV columns: trial, timestamp, ACTUAL tool0-in-base and held-in-target poses, COMMANDED tool0-in-base
and held-in-target poses (each as xyz + quaternion + ZYX Euler [yaw,pitch,roll] deg), F/T in the
tool0 frame, and F/T in the held-object frame. One timestamped CSV per run, appended incrementally
(rows are streamed to disk, never accumulated in memory).
"""

import csv
import math
import os
from datetime import datetime

import numpy as np

import rclpy
from rclpy.time import Time

from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectoryPoint

import tf2_ros
from tf_transformations import (
    euler_from_quaternion, quaternion_from_matrix, quaternion_matrix, quaternion_slerp)

from ur_kinematic_assembly_demo.kinematic_assembly_node import (
    KinematicAssembly, xyzrpy_to_matrix)


def transform_to_matrix(t):
    q = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
    m = quaternion_matrix(q)
    m[0, 3], m[1, 3], m[2, 3] = t.translation.x, t.translation.y, t.translation.z
    return m


def _pose_cols(prefix):
    return [f'{prefix}_x', f'{prefix}_y', f'{prefix}_z',
            f'{prefix}_qx', f'{prefix}_qy', f'{prefix}_qz', f'{prefix}_qw',
            f'{prefix}_yaw_deg', f'{prefix}_pitch_deg', f'{prefix}_roll_deg']


def _pose_fields(T):
    """xyz + quaternion (x,y,z,w) + ZYX Euler [yaw, pitch, roll] in degrees for a 4x4 pose."""
    q = quaternion_from_matrix(T)                       # [x, y, z, w]
    roll, pitch, yaw = euler_from_quaternion(q)         # 'sxyz' == intrinsic ZYX (yaw-pitch-roll)
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
            float(q[0]), float(q[1]), float(q[2]), float(q[3]),
            math.degrees(yaw), math.degrees(pitch), math.degrees(roll)]


class UncertainAssemblySampling(KinematicAssembly):
    """Perturbed, chunked assembly + ideal disassembly, repeated and logged at the F/T rate."""

    def __init__(self):
        super().__init__(node_name='uncertain_assembly_sampling',
                         default_config=self._sampling_config())
        s = self.cfg.get('sampling', {}) or {}
        self.num_trials = int(s.get('num_trials', 20))
        self.chunk_fraction = float(s.get('chunk_fraction', 1.0))
        self.res_t = float(s.get('translational_resolution_m', 0.002))
        self.res_r = math.radians(float(s.get('rotational_resolution_deg', 1.0)))
        # Per-dimension half-widths [x, y, z, roll, pitch, yaw]; x/y/z in m, roll/pitch/yaw in deg.
        self.bias_bounds = [float(v) for v in s.get('bias', [0.002, 0.002, 0.002, 1.0, 1.0, 1.0])]
        self.noise_bounds = [float(v) for v
                             in s.get('noise', [0.0005, 0.0005, 0.0005, 0.25, 0.25, 0.25])]
        self.rot_weight = float(s.get('closest_pose_rot_weight_mm_per_deg', 1.0))
        # Cap the reference/trajectory ramp speed so a move is never commanded faster than the joints
        # allow. This is KinematicAssembly's shared joint-velocity cap (speed.max_joint_velocity_rad_s
        # in yaml -> self.max_joint_vel); the old sampling.max_joint_speed_rad_s still works as a
        # fallback for back-compat.
        legacy = float(s.get('max_joint_speed_rad_s', 0.0))
        if self.max_joint_vel <= 0.0 and legacy > 0.0:
            self.max_joint_vel = legacy
        seed = int(s.get('random_seed', 0))
        if seed:
            np.random.seed(seed)

        # One timestamped CSV per run (unique). Relative -> resolved from the working directory.
        raw = os.path.expanduser(str(s.get('csv_path', 'uncertain_assembly_log.csv')))
        raw = raw if os.path.isabs(raw) else os.path.abspath(raw)
        base, ext = os.path.splitext(raw)
        self.csv_path = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext or '.csv'}"

        # F/T: keep the FULL wrench + its frame for logging. Ensure a subscription exists even in
        # position mode (the parent only subscribes for admittance).
        self._wrench_raw = None
        self._wrench_frame = None
        self._T_tool0_wrench = None      # cached wrench-frame -> tool0 (F/T sensor is fixed to wrist)
        if self.control_mode != 'admittance':
            self.create_subscription(WrenchStamped, self.wrench_topic, self._wrench_cb, 10)

        # tf, to read the ACTUAL tool0 pose (the kinematic demo needs no tf; we do).
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Continuous-recording state (the F/T callback writes a row while _recording).
        self._recording = False
        self._current_trial = -1
        self._commanded_held = None
        self._csv_file = None
        self._csv_writer = None
        self._dense = None

    @staticmethod
    def _sampling_config():
        src = os.path.normpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', 'config',
            'uncertain_assembly_sampling.yaml'))
        if os.path.isfile(src):
            return src
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('ur_uncertain_assembly_sampling'),
                            'config', 'uncertain_assembly_sampling.yaml')

    # F/T: store the full wrench + frame + the magnitudes the parent's guard expects, and -- while
    # recording -- write a row on EVERY message (records at the sensor rate, the highest available).
    def _wrench_cb(self, msg):
        w = msg.wrench
        self._wrench_raw = (w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z)
        self._wrench_frame = msg.header.frame_id
        self._wrench = ((w.force.x ** 2 + w.force.y ** 2 + w.force.z ** 2) ** 0.5,
                        (w.torque.x ** 2 + w.torque.y ** 2 + w.torque.z ** 2) ** 0.5)
        if self._recording:
            self._record_row()

    def setup(self):
        if not super().setup():
            return False
        if self._lookup_tool0(timeout_s=5.0) is None:
            self.get_logger().error(
                f'No {self.base_frame} -> {self.tip_frame} tf (is robot_state_publisher up?).')
            return False
        # Detect + report the published wrench frame (so the tool0/held transforms are correct).
        deadline = self.get_clock().now().nanoseconds + int(3e9)
        while rclpy.ok() and self._wrench_frame is None \
                and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._wrench_frame is None:
            self.get_logger().warn(
                f"No wrench on '{self.wrench_topic}'; F/T columns will be zeros.")
        else:
            self.get_logger().info(f"Wrench published in frame '{self._wrench_frame}'.")
        return True

    # ------------------------------------------------------------------ tf / state
    def _lookup_tool0(self, timeout_s=1.0):
        """base -> tool0 as 4x4, spinning until available (use OUTSIDE callbacks)."""
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < deadline:
            m = self._tool0_now()
            if m is not None:
                return m
            rclpy.spin_once(self, timeout_sec=0.05)
        return None

    def _tool0_now(self):
        """Latest base -> tool0 as 4x4, or None. Non-spinning -- safe inside a callback."""
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.tip_frame, Time())
            return transform_to_matrix(tf.transform)
        except tf2_ros.TransformException:
            return None

    def _actual_held_in_target(self):
        T_base_tool0 = self._lookup_tool0()
        if T_base_tool0 is None:
            return None
        return np.linalg.inv(self.T_base_targetobj) @ T_base_tool0 @ self.T_tool0_held

    # ------------------------------------------------------------ resample / perturb
    def _resample(self, mats):
        """Densify so consecutive waypoints are within res_t / res_r (linear + slerp)."""
        dense = [mats[0]]
        for a, b in zip(mats[:-1], mats[1:]):
            dtrans = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
            qa, qb = quaternion_from_matrix(a), quaternion_from_matrix(b)
            dot = min(1.0, abs(float(np.dot(qa, qb))))
            dang = float(2.0 * np.arccos(dot))
            n = 1
            if self.res_t > 0:
                n = max(n, int(np.ceil(dtrans / self.res_t)))
            if self.res_r > 0:
                n = max(n, int(np.ceil(dang / self.res_r)))
            for k in range(1, n + 1):
                alpha = k / n
                T = quaternion_matrix(quaternion_slerp(qa, qb, alpha))
                T[:3, 3] = (1.0 - alpha) * a[:3, 3] + alpha * b[:3, 3]
                dense.append(T)
        return dense

    def _random_delta(self, bounds):
        """Uniform random pose delta with per-dimension half-widths
        [x, y, z (m), roll, pitch, yaw (deg)]."""
        b = np.asarray(bounds, dtype=float)
        t = np.random.uniform(-1.0, 1.0, 3) * b[0:3]
        r = np.radians(np.random.uniform(-1.0, 1.0, 3) * b[3:6])
        return xyzrpy_to_matrix(t, r)

    def _closest_index(self, actual, dense):
        """Closest dense waypoint by weighted translation+rotation (rot_weight mm per degree)."""
        p = actual[:3, 3]
        qa = quaternion_from_matrix(actual)
        best_i, best_c = 0, float('inf')
        for i, d in enumerate(dense):
            dt_mm = float(np.linalg.norm(d[:3, 3] - p)) * 1000.0
            dot = min(1.0, abs(float(np.dot(qa, quaternion_from_matrix(d)))))
            dr_deg = math.degrees(2.0 * math.acos(dot))
            cost = dt_mm + self.rot_weight * dr_deg
            if cost < best_c:
                best_c, best_i = cost, i
        return best_i

    # ------------------------------------------------------------ wrench / logging
    @staticmethod
    def _transform_wrench(f, tau, T_BA):
        """Wrench (f, tau) in frame A -> frame B, where T_BA maps A-coords to B-coords."""
        R, p = T_BA[:3, :3], T_BA[:3, 3]
        fB = R @ np.asarray(f, dtype=float)
        tauB = R @ np.asarray(tau, dtype=float) + np.cross(p, fB)
        return fB, tauB

    def _wrench_to_tool0(self):
        """Cached transform mapping the published wrench frame -> tool0 (identity if same/unknown)."""
        if self._T_tool0_wrench is not None:
            return self._T_tool0_wrench
        frame = self._wrench_frame
        if not frame or frame == self.tip_frame:
            self._T_tool0_wrench = np.eye(4)
            return self._T_tool0_wrench
        try:
            tf = self.tf_buffer.lookup_transform(self.tip_frame, frame, Time())
            self._T_tool0_wrench = transform_to_matrix(tf.transform)
        except tf2_ros.TransformException:
            self.get_logger().warn(
                f"No {self.tip_frame} <- '{frame}' tf for the wrench; assuming identity.")
            self._T_tool0_wrench = np.eye(4)
        return self._T_tool0_wrench

    def _open_csv(self):
        try:
            os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
            new = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path) == 0
            self._csv_file = open(self.csv_path, 'a', newline='')
        except OSError as exc:
            self.get_logger().error(f'Cannot open CSV {self.csv_path}: {exc}')
            return False
        self._csv_writer = csv.writer(self._csv_file)
        if new:
            self._csv_writer.writerow(
                ['trial', 'timestamp']
                + _pose_cols('tool0_base') + _pose_cols('held_target')
                + _pose_cols('cmd_tool0_base') + _pose_cols('cmd_held_target')
                + ['ft_tool0_fx', 'ft_tool0_fy', 'ft_tool0_fz',
                   'ft_tool0_tx', 'ft_tool0_ty', 'ft_tool0_tz']
                + ['ft_held_fx', 'ft_held_fy', 'ft_held_fz',
                   'ft_held_tx', 'ft_held_ty', 'ft_held_tz'])
        self.get_logger().info(f'Logging to {self.csv_path}')
        return True

    def _record_row(self):
        """Write one row from the latest state (called from the F/T callback while recording).
        Non-spinning (tf read from the buffer); rows are streamed straight to the file."""
        if self._csv_writer is None or self._commanded_held is None:
            return
        T_base_tool0 = self._tool0_now()
        if T_base_tool0 is None:
            return
        inv_target = np.linalg.inv(self.T_base_targetobj)
        inv_held = np.linalg.inv(self.T_tool0_held)
        T_held_target = inv_target @ T_base_tool0 @ self.T_tool0_held
        T_cmd_tool0 = self.T_base_targetobj @ self._commanded_held @ inv_held
        w = self._wrench_raw if self._wrench_raw is not None else (0.0,) * 6
        # Published wrench -> tool0 -> held.
        f0, t0 = self._transform_wrench(w[0:3], w[3:6], self._wrench_to_tool0())
        fh, th = self._transform_wrench(f0, t0, inv_held)
        ts = self.get_clock().now().nanoseconds / 1e9
        self._csv_writer.writerow(
            [self._current_trial, ts]
            + _pose_fields(T_base_tool0) + _pose_fields(T_held_target)
            + _pose_fields(T_cmd_tool0) + _pose_fields(self._commanded_held)
            + [float(v) for v in f0] + [float(v) for v in t0]
            + [float(v) for v in fh] + [float(v) for v in th])

    def _flush_csv(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            os.fsync(self._csv_file.fileno())

    def _close_csv(self):
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except OSError:
                pass
            self._csv_file = None

    # ------------------------------------------------------------ motion
    def _goto(self, target_joints, start_joints):
        """Move to target_joints in the current control mode. Returns True if force-stopped."""
        if self.control_mode == 'admittance':
            return self._stream_reference(start_joints, target_joints)
        self.send_joint_trajectory([target_joints], self.waypoint_dt, 'move')
        return False

    def _stream_reference(self, start_joints, target_joints):
        """Ramp a joint reference start->target under admittance, honoring the force/torque guard.
        The ramp duration respects both waypoint_dt AND the joint-velocity cap (max_joint_vel), so a
        large joint move is never commanded faster than the joint velocity limits allow."""
        max_delta = max((abs(t - s) for s, t in zip(start_joints, target_joints)), default=0.0)
        duration = self.waypoint_dt
        if self.max_joint_vel > 0.0:
            duration = max(duration, max_delta / self.max_joint_vel)
        steps = max(1, int(duration * self.reference_rate))
        period = 1.0 / self.reference_rate
        for k in range(1, steps + 1):
            if not rclpy.ok():
                break
            if self._contact_exceeded():            # force AND torque thresholds (config)
                return True
            alpha = k / steps
            pt = JointTrajectoryPoint()
            pt.positions = [(1.0 - alpha) * s + alpha * t
                            for s, t in zip(start_joints, target_joints)]
            self.ref_pub.publish(pt)
            self._sleep(period)
        return False

    # --------------------------------------------------------------------- run
    def run(self):
        home_joints = self._current_joints()

        traj_mats = self._load_trajectory()
        if traj_mats is None:
            return False
        self.T_targetobj_held_assembled = traj_mats[-1]
        self.T_base_targetobj = ((self.T_base_assembled @ self.T_tool0_held)
                                 @ np.linalg.inv(traj_mats[-1]))
        self._dense = self._resample(traj_mats)
        k = max(1, int(np.ceil(self.chunk_fraction * len(self._dense))))
        self.get_logger().info(
            f'{len(self._dense)} dense waypoints; chunk = first {k} '
            f'({self.chunk_fraction:.2f}); {self.num_trials} trials.')

        if not self._open_csv():
            return False

        # 1. Approach the stand-off + trajectory start in POSITION control. These are large
        #    free-space moves; streaming them as admittance references over waypoint_dt exceeds the
        #    joint velocity limits (and faults the controller/driver).
        if not self._confirm('move to stand-off + trajectory start'):
            self.get_logger().info('Aborted by user.')
            self._close_csv()
            return False
        seed = list(home_joints)
        for pose, label in ((self._standoff_pose(), 'stand-off'),
                            (self._tool0_at(self._dense[0]), 'trajectory start')):
            j = self.solve_ik(pose, seed)
            if j is None:
                self.get_logger().error(f'IK failed for {label}.')
                self._close_csv()
                return False
            if not self.send_joint_trajectory([j], self.standoff_move_duration, label):
                self._close_csv()
                return False
            seed = j

        # 2. Switch to admittance for the (small) contact moves of the trials.
        if self.control_mode == 'admittance':
            self._apply_admittance_params()
            if not self._switch_controllers([self.adm_controller], [self.position_controller]):
                self._close_csv()
                return False
            self._in_compliance = True

        ok = False
        try:
            ok = self._run_trials(seed, k)
        finally:
            self._recording = False
            if self._in_compliance:
                self.get_logger().warn('Restoring position control.')
                self._switch_to_position()
            self._close_csv()

        if ok:
            self.send_joint_trajectory([home_joints], self.standoff_move_duration, 'home')
            self.get_logger().info('Sampling complete.')
        return ok

    def _run_trials(self, start_seed, k):
        dense = self._dense
        seed = list(start_seed)   # already at the trajectory start (dense[0]) in the right mode

        for trial in range(self.num_trials):
            if not self._confirm(f'trial {trial + 1}/{self.num_trials}'):
                self.get_logger().info('Aborted by user.')
                return False
            self._current_trial = trial

            if self.control_mode == 'admittance' and self.tare_before:
                self._tare_ft()   # re-zero at the disengaged start of each trial

            # Perturbed chunk: one bias per trial + per-waypoint noise, in the TARGET-OBJECT frame.
            bias = self._random_delta(self.bias_bounds)
            perturbed = [bias @ self._random_delta(self.noise_bounds) @ c
                         for c in dense[:k]]

            # Execute + record continuously (the F/T callback writes rows while _recording).
            self._recording = True
            stopped = False
            for pose_held in perturbed:
                j = self.solve_ik(self._tool0_at(pose_held), seed)
                if j is None:
                    self.get_logger().warn('IK failed on a chunk waypoint; ending chunk.')
                    break
                self._commanded_held = pose_held
                if self._goto(j, seed):
                    stopped = True
                seed = j
                self._sleep(self.settle_s)
                if stopped:
                    self.get_logger().info('Force/torque-guard stop; ending chunk.')
                    break
            self._recording = False

            # Disassembly (NOT logged): closest IDEAL pose (weighted), then ideal reverse to start.
            actual = self._actual_held_in_target()
            j_close = self._closest_index(actual, dense) if actual is not None else (k - 1)
            for idx in range(j_close, -1, -1):
                j = self.solve_ik(self._tool0_at(dense[idx]), seed)
                if j is None:
                    self.get_logger().error(f'IK failed during disassembly at dense[{idx}].')
                    return False
                self._goto(j, seed)
                seed = j

            self._flush_csv()
            self.get_logger().info(f'Trial {trial + 1} done (data flushed).')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = UncertainAssemblySampling()
    try:
        if node.setup():
            node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
        node._recording = False
        if node._in_compliance:
            node._switch_to_position()
        node._close_csv()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
