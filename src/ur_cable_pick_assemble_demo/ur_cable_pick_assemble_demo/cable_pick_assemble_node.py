#!/usr/bin/env python3
"""Cable PICK-and-ASSEMBLE demo for the UR10e + Robotiq 2F-85 (ROS2 Jazzy).

Subclasses ur_cable_pick_place_demo's CablePickPlace, so the PICK is literally the same code path
(scan -> estimate connector -> grasp-align -> grasp -> close -> grasp check + recovery). What replaces
"place" is an ASSEMBLY.

Sequence:
  [PICK -- identical to ur_cable_pick_place_demo]
    -> lift (clear the surface)
    -> move to the assembly STAND-OFF   (position control)
    -> switch to COMPLIANCE             (admittance; toggleable, default on)
    -> execute the ideal stand-off -> target trajectory in fractional CHUNKS (force-guarded)
    -> open the gripper (release)
    -> switch back to POSITION control
    -> RETRACT along a specified frame/axis/distance
    -> return to the initial pose

ASSEMBLY METHODS (assembly.method)
----------------------------------
  'kinematic' -- IMPLEMENTED. The target pose is given outright; no perception is used during the
                 assembly itself. Everything downstream (stand-off, chunked insertion, force guard,
                 compliance, retract) is method-agnostic, so adding a method only means supplying a
                 target pose.
  'vision'    -- placeholder for a perceived target (e.g. a socket located by SAM3 or a fiducial).
                 Not implemented; it fails loudly rather than silently doing something else.

TARGET FRAME (assembly.target.frame)
------------------------------------
The pose is always in base_link, but `frame` says WHICH frame that pose describes -- three different
questions, all reduced to a FINGERTIP command because that is what move_grasp_tcp_to controls:

  'fingertip' -- the pose IS the fingertip's target.                fingertip = target
  'tool0'     -- the pose is where the FLANGE must end up.          fingertip = target * T_tool0_fingertip
  'connector' -- the pose is where the HELD CONNECTOR must end up.  fingertip = target * connector_grasp
                 (the connector is rigidly held: at grasp the fingertip was commanded to
                  connector * connector_grasp, so T_connector_fingertip == connector_grasp)
"""

import math
import os

import numpy as np

import rclpy
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectoryPoint

from controller_manager_msgs.srv import SwitchController
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger

from ur_cable_pick_place_demo.cable_pick_place_node import CablePickPlace
from ur_pick_place_demo.pick_place_node import (
    matrix_to_pose, translation_matrix, xyzrpy_to_matrix)

from tf_transformations import quaternion_from_matrix, quaternion_matrix, quaternion_slerp


class CablePickAssemble(CablePickPlace):
    """Pick the cable exactly as ur_cable_pick_place_demo does, then assemble it."""

    def __init__(self):
        super().__init__(node_name='cable_pick_assemble',
                         default_config=self._assemble_config())
        a = self.cfg.get('assembly', {}) or {}
        self.method = str(a.get('method', 'kinematic')).lower()

        t = a.get('target', {}) or {}
        self.target_frame = str(t.get('frame', 'fingertip')).lower()
        self.T_base_target = xyzrpy_to_matrix(**self._xyzrpy(t))

        s = a.get('standoff', {}) or {}
        self.standoff_axis = np.asarray(s.get('axis', [0.0, 0.0, 1.0]), dtype=float)
        self.standoff_dist = float(s.get('distance_m', 0.05))
        self.lift_after_pick = bool(a.get('lift_after_pick', True))

        c = a.get('compliance', {}) or {}
        self.compliance_enabled = bool(c.get('enabled', True))
        self.adm_controller = c.get('controller', 'admittance_controller')
        self.position_controller = c.get('position_controller',
                                         'scaled_joint_trajectory_controller')
        self.switch_service = c.get('switch_service', '/controller_manager/switch_controller')
        self.reference_topic = c.get('reference_topic', '/admittance_controller/joint_references')
        self.reference_rate = float(c.get('reference_rate_hz', 20.0))
        self.ft_zero_service = c.get('ft_zero_service', '/io_and_status_controller/zero_ftsensor')
        self.tare_before = bool(c.get('tare_before', True))
        self.apply_params = bool(c.get('apply_params', True))
        # How long to PIN the admittance reference to the arm's current pose on each side of a
        # controller switch. This is what stops the joint-velocity fault on the transition -- see
        # _hold_reference. Cheap insurance; there is no reason to shrink it.
        self.switch_settle_s = float(c.get('switch_settle_s', 0.5))
        self.adm_mass = c.get('mass', [5.0] * 3 + [0.5] * 3)
        self.adm_damping = c.get('damping_ratio', [1.0] * 6)
        self.adm_stiffness = c.get('stiffness', [200.0] * 3 + [15.0] * 3)
        self.adm_selected = c.get('selected_axes', [True] * 6)

        fg = a.get('force_guard', {}) or {}
        self.wrench_topic = fg.get('wrench_topic', '/force_torque_sensor_broadcaster/wrench')
        self.max_force = float(fg.get('max_force_n', 20.0))     # 0 disables
        self.max_torque = float(fg.get('max_torque_nm', 5.0))   # 0 disables
        # Guard ARMED OVER EVERY MOTION, not just the insertion (see _abort_move): an unexpected
        # collision during the scan, the traverse to the stand-off, or the retract cancels the
        # trajectory MID-MOVE instead of being discovered after it.
        self.force_guard_enabled = bool(fg.get('enabled', True))
        # TARE BEFORE EVERY TASK (see _do). Not paranoia -- the PAYLOAD CHANGES mid-sequence: the moment
        # the cable is grasped its weight lands on the sensor, and residual bias would otherwise read as
        # 'contact' and trip the guard spuriously (or mask a real contact).
        self.tare_each_task = bool(fg.get('tare_each_task', True))
        self.tare_settle_s = float(fg.get('tare_settle_s', 0.3))
        self._guard_tripped = False

        i = a.get('insertion', {}) or {}
        self.chunk_fraction = float(i.get('chunk_fraction', 0.25))
        self.chunk_time_s = float(i.get('chunk_time_s', 2.0))
        self.chunk_settle_s = float(i.get('settle_s', 0.5))

        r = a.get('retract', {}) or {}
        self.retract_frame = str(r.get('frame', 'target')).lower()
        # A SEQUENCE of displacement steps, each expressed in retract_frame and applied in order. This
        # is what a real escape path needs (back out along the mate, clear laterally, come down) -- a
        # single axis*distance cannot express it. Falls back to the old axis/distance_m form.
        steps = r.get('steps')
        if steps:
            self.retract_steps = [np.asarray(s['xyz'], dtype=float) for s in steps]
        else:
            axis = np.asarray(r.get('axis', [0.0, 0.0, 1.0]), dtype=float)
            dist = float(r.get('distance_m', 0.08))
            self.retract_steps = [axis * dist] if abs(dist) > 1e-9 else []

        # Interfaces
        self._wrench = None
        self.create_subscription(WrenchStamped, self.wrench_topic, self._wrench_cb, 10)
        self._in_compliance = False
        self.switch_client = self.create_client(SwitchController, self.switch_service)
        self.ft_zero_client = self.create_client(Trigger, self.ft_zero_service)
        # In Jazzy each controller runs as its OWN node named after the controller, so its parameters
        # (admittance.*) live there -- NOT under /controller_manager.
        self.setparam_client = self.create_client(
            SetParameters, f'/{self.adm_controller}/set_parameters')
        self.ref_pub = self.create_publisher(JointTrajectoryPoint, self.reference_topic, 10)

        self._T_target_ftip = None     # the FINGERTIP target, resolved once the method runs

    @staticmethod
    def _assemble_config():
        src = os.path.normpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', 'config',
            'cable_pick_assemble.yaml'))
        if os.path.isfile(src):
            return src
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('ur_cable_pick_assemble_demo'),
                            'config', 'cable_pick_assemble.yaml')

    # ------------------------------------------------------------------ setup
    def setup(self):
        if not super().setup():
            return False
        if self.method not in ('kinematic',):
            self.get_logger().error(
                f"assembly.method '{self.method}' is not implemented. Only 'kinematic' is available "
                "('vision' is a placeholder). Aborting rather than silently doing something else.")
            return False
        if self.compliance_enabled and not self.switch_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                f"'{self.switch_service}' not up -- compliance needs the controller_manager with a "
                f"LOADED '{self.adm_controller}' (see ur_admittance_demo). Set "
                'assembly.compliance.enabled: false to insert under position control instead.')
        return True

    # ---------------------------------------------------------------- F/T
    def _wrench_cb(self, msg):
        f, t = msg.wrench.force, msg.wrench.torque
        self._wrench = ((f.x ** 2 + f.y ** 2 + f.z ** 2) ** 0.5,
                        (t.x ** 2 + t.y ** 2 + t.z ** 2) ** 0.5)

    def _contact_exceeded(self):
        """True once the (tared) contact force/torque reaches the configured limit."""
        if self._wrench is None:
            return False
        force, torque = self._wrench
        if self.max_force > 0.0 and force >= self.max_force:
            self.get_logger().info(
                f'Contact force {force:.1f} N >= max_force_n {self.max_force:.1f} N.')
            return True
        if self.max_torque > 0.0 and torque >= self.max_torque:
            self.get_logger().info(
                f'Contact torque {torque:.2f} Nm >= max_torque_nm {self.max_torque:.2f} Nm.')
            return True
        return False

    def _tare_ft(self, quiet=False):
        """Zero the F/T sensor. The caller decides WHEN -- per task (_do) and/or before compliance."""
        if not self.ft_zero_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f"FT zero '{self.ft_zero_service}' unavailable; skipping tare.")
            return False
        future = self.ft_zero_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        resp = future.result()
        if resp is None or not resp.success:
            self.get_logger().warn('F/T tare did not report success; continuing.')
            return False
        if not quiet:
            self.get_logger().info('F/T sensor tared.')
        return True

    # ------------------------------------------------- force guard over EVERY motion
    def _abort_move(self):
        """Force guard, armed over EVERY trajectory (hook in PickPlace.send_joints). True -> cancel.

        This is what makes the guard cover ALL phases rather than just the insertion: a collision during
        the scan, the traverse to the stand-off, or the retract cancels the move immediately.

        Callers tell 'guard tripped' apart from 'move failed' via self._guard_tripped -- and the two
        mean OPPOSITE things depending on the phase: during the insertion a trip means the part is
        SEATED (success); anywhere else it means we hit something we should not have (failure)."""
        if not self.force_guard_enabled:
            return False
        if self._contact_exceeded():
            self._guard_tripped = True
            return True
        return False

    def _do(self, label, fn):
        """Every task: TARE first, then run it with the force guard armed.

        Per-task taring matters because the PAYLOAD CHANGES mid-sequence -- the moment the cable is
        grasped, its weight lands on the sensor. Without re-zeroing, that weight (and any drift) reads
        as 'contact': the guard would trip spuriously on the very next move, or, if the bias went the
        other way, mask a real collision. Zeroing at the start of each task means the guard measures
        only the force THAT task generates."""
        if not self._confirm(label):
            self.get_logger().info('Aborted by user.')
            return False
        if self.tare_each_task and self.force_guard_enabled:
            self._tare_ft(quiet=True)
            self._sleep(self.tare_settle_s)      # let a fresh, tared sample land before arming
        self._guard_tripped = False
        if not fn():
            if self._guard_tripped:
                self.get_logger().error(
                    f'FORCE GUARD tripped during "{label}" (limit {self.max_force:.0f} N / '
                    f'{self.max_torque:.1f} Nm) -- the arm contacted something unexpected. Stopping.')
            else:
                self.get_logger().error(f'Step failed: {label}. Stopping.')
            return False
        return True

    # -------------------------------------------------------- compliance control
    def _apply_admittance_params(self):
        """Best-effort push of the tunable admittance gains onto the running controller.
        admittance.mass/damping_ratio/stiffness/selected_axes are DYNAMIC, and
        enable_parameter_update_without_reactivation defaults true, so these apply live."""
        if not self.apply_params:
            return
        if not self.setparam_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                f'/{self.adm_controller}/set_parameters unavailable; using the gains the controller '
                'was loaded with.')
            return
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
            self.get_logger().warn('Some admittance params were not applied.')
        else:
            self.get_logger().info('Applied admittance params (live).')

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
                f"Is '{self.adm_controller}' loaded? (ros2 control list_controllers)")
            return False
        return True

    def _hold_reference(self, seconds):
        """Publish the arm's CURRENT joint positions as the admittance reference for `seconds`.

        THIS IS WHAT PREVENTS THE JOINT-VELOCITY VIOLATION ON A CONTROLLER SWITCH.

        The admittance controller starts tracking its reference the INSTANT it activates. If that
        reference is anything other than where the arm actually is -- a stale message left in its buffer,
        or a default -- it commands a step change from the real pose to that reference in a single
        control cycle. The driver sees an impossible joint velocity (joint 0 first, since the shoulder
        pan carries the largest excursion) and faults.

        Publishing the current position makes activation a NO-OP by construction: the target IS where the
        arm already is, whatever the controller's own initialisation happens to do. Held on BOTH sides of
        the switch -- before, to seed the buffer; after, to pin it there until we deliberately ramp."""
        j = self._current_joints()
        period = 1.0 / self.reference_rate
        for _ in range(max(1, int(seconds * self.reference_rate))):
            pt = JointTrajectoryPoint()
            pt.positions = list(j)
            self.ref_pub.publish(pt)
            self._sleep(period)

    def _enable_compliance(self):
        """Tare, push the gains, and hand the arm over to the admittance controller.

        NOTE the consequence: the JTC is DEACTIVATED, so move_grasp_tcp_to (which goes through the
        trajectory action) will NOT work until we switch back. Everything between here and
        _switch_to_position must stream joint references instead -- see _stream_to."""
        if not self.compliance_enabled:
            self.get_logger().info('Compliance disabled; inserting under POSITION control.')
            return True
        self._apply_admittance_params()
        if self.tare_before:
            self._tare_ft()
            self._sleep(self.tare_settle_s)

        # Seed the reference with the CURRENT pose BEFORE the switch, so activation cannot jump.
        self._hold_reference(self.switch_settle_s)
        if not self._switch_controllers([self.adm_controller], [self.position_controller]):
            return False
        self._in_compliance = True
        # Pin it there now that the controller is live, until we deliberately start ramping.
        self._hold_reference(self.switch_settle_s)

        # A tared sensor should read ~0. If it does not, the admittance controller will DRIVE the arm to
        # "comply" with that phantom force -- a runaway, not an insertion. This is the other way joint 0
        # runs away on a switch, and no amount of reference seeding fixes it.
        residual = self._wrench[0] if self._wrench else 0.0
        if self.max_force > 0.0 and residual > 0.25 * self.max_force:
            self.get_logger().warn(
                f'Residual force after tare is {residual:.1f} N (limit {self.max_force:.0f} N). The '
                'admittance controller will actively push the arm to null that. Check the tare took '
                "effect, and that the controller's gravity/payload compensation is configured -- an "
                'uncompensated tool weight looks exactly like a constant external force.')

        self.get_logger().info(
            f'Compliance ON. Force-guarded at {self.max_force:.0f} N / {self.max_torque:.1f} Nm.')
        return True

    def _switch_to_position(self):
        """Hand the arm back to the JTC. Hold the reference across the switch for the same reason as
        _enable_compliance: a reference that disagrees with the arm's real pose is a step command."""
        if not self._in_compliance:
            return True
        self._hold_reference(self.switch_settle_s)     # pin the arm before handing it over
        if not self._switch_controllers([self.position_controller], [self.adm_controller]):
            return False
        self._in_compliance = False
        self._sleep(self.switch_settle_s)              # let the JTC latch the current pose as its hold
        self.get_logger().info('Position control restored.')
        return True

    # ------------------------------------------------------------ target geometry
    def _fingertip_target(self):
        """Reduce the configured target to a FINGERTIP pose -- that is what move_grasp_tcp_to commands.

        See the module docstring for what each target frame means. The 'connector' case is the
        interesting one: the connector is RIGIDLY HELD, and at grasp the fingertip was commanded to
        connector * connector_grasp -- so T_connector_fingertip == connector_grasp, and asking for the
        connector to land at T means asking the fingertip to land at T * connector_grasp."""
        T = self.T_base_target
        if self.target_frame == 'fingertip':
            return T
        if self.target_frame == 'tool0':
            return T @ self.T_tool0_grasp          # T_tool0_grasp IS the tool0->fingertip transform
        if self.target_frame == 'connector':
            return T @ self.T_connector_grasp
        self.get_logger().error(
            f"assembly.target.frame '{self.target_frame}' is not one of "
            "'fingertip' | 'tool0' | 'connector'.")
        return None

    def _standoff_of(self, T_target):
        """Back off from the target along standoff.axis by standoff.distance_m, IN THE TARGET FRAME
        (so the axis means the same thing however the target is oriented in base)."""
        return T_target @ translation_matrix(self.standoff_axis * self.standoff_dist)

    @staticmethod
    def _interp(T0, T1, alpha):
        """Pose interpolation: linear in position, SLERP in orientation."""
        q = quaternion_slerp(quaternion_from_matrix(T0), quaternion_from_matrix(T1), float(alpha))
        T = quaternion_matrix(q)
        T[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
        return T

    def _frame_rotation(self, name):
        """3x3 rotation of the named frame, expressed in base. Used to interpret the retract axis."""
        if name == 'base':
            return np.eye(3)
        if name == 'target':
            return self.T_base_target[:3, :3]
        if name == 'tool0':
            T = self._tf_matrix(self.base_frame, self.tip_frame)
            return None if T is None else T[:3, :3]
        if name == 'fingertip':
            T = self._fingertip_now()
            return None if T is None else T[:3, :3]
        return None

    def _fingertip_now(self):
        T = self._tf_matrix(self.base_frame, self.tip_frame)
        return None if T is None else T @ self.T_tool0_grasp

    # ---------------------------------------------------------------- insertion
    def _stream_to(self, T_ftip_target, label):
        """Ramp a JOINT reference toward the target under ADMITTANCE, force-guarded.

        Under compliance the JTC is deactivated, so the trajectory action is unavailable: we IK the
        target and stream interpolated joint references on the admittance controller's reference topic.
        The controller yields to contact, so the arm does not fight a misalignment -- it settles into
        it. Returns (ok, seated)."""
        T_tool0 = T_ftip_target @ np.linalg.inv(self.T_tool0_grasp)
        j_start = self._current_joints()
        j_target = self.solve_ik(matrix_to_pose(T_tool0), j_start)
        if j_target is None:
            self._log_ik_failure(label, matrix_to_pose(T_tool0))
            return False, False

        # Duration honours chunk_time_s AND the joint-velocity cap, so a large chunk is never commanded
        # faster than the joints allow (a streamed reference has no controller-side time scaling).
        max_delta = max((abs(t - s) for s, t in zip(j_start, j_target)), default=0.0)
        duration = self.chunk_time_s
        if self.max_joint_vel > 0.0:
            duration = max(duration, max_delta / self.max_joint_vel)
        steps = max(1, int(duration * self.reference_rate))
        period = 1.0 / self.reference_rate

        for k in range(1, steps + 1):
            if not rclpy.ok():
                break
            if self._contact_exceeded():
                self.get_logger().info(f'[{label}] force guard tripped mid-chunk -- holding here.')
                return True, True
            alpha = k / steps
            pt = JointTrajectoryPoint()
            pt.positions = [(1.0 - alpha) * s + alpha * t for s, t in zip(j_start, j_target)]
            self.ref_pub.publish(pt)
            self._sleep(period)
        return True, False

    def _insert_chunked(self, T_start, T_target):
        """Execute the ideal stand-off -> target trajectory in fractional CHUNKS, force-guarded.

        Chunking (rather than one continuous move) is what makes an insertion inspectable: the contact
        wrench is checked BETWEEN chunks, so a jam is caught after a fraction of the travel instead of
        after all of it -- and with confirm_each_step you get a veto at each fraction. Under compliance
        the arm also yields to contact WITHIN a chunk (see _stream_to)."""
        n = max(1, int(math.ceil(1.0 / max(1e-6, self.chunk_fraction))))
        self.get_logger().info(
            f'Inserting in {n} chunk(s) of {self.chunk_fraction * 100:.0f}% '
            f'({"ADMITTANCE" if self._in_compliance else "POSITION"} control), force-guarded at '
            f'{self.max_force:.0f} N / {self.max_torque:.1f} Nm...')

        for k in range(1, n + 1):
            if self._contact_exceeded():
                self.get_logger().info(
                    f'Force guard already tripped before chunk {k}/{n} -- the part is seated; '
                    'stopping the insertion here.')
                return True
            alpha = min(1.0, k * self.chunk_fraction)
            T = self._interp(T_start, T_target, alpha)
            label = f'insert chunk {k}/{n} ({alpha * 100:.0f}%)'
            if not self._confirm(label):
                self.get_logger().info('Aborted by user.')
                return False

            if self._in_compliance:
                ok, seated = self._stream_to(T, label)
                if not ok:
                    return False
                if seated:
                    self.get_logger().info('Stopped on the contact limit (seated).')
                    return True
            else:
                # Position control: the guard cancels the trajectory MID-CHUNK via _abort_move. Here --
                # and ONLY here -- that trip means the part is SEATED, not that we hit something wrong,
                # so it is a SUCCESS. Everywhere else in the sequence the same trip fails the task.
                self._guard_tripped = False
                if not self.move_grasp_tcp_to(T, label):
                    if self._guard_tripped:
                        self.get_logger().info(
                            f'[{label}] force guard tripped mid-chunk -- the part is SEATED. Stopping '
                            'the insertion here (success, not a failure).')
                        return True
                    return False
            self._sleep(self.chunk_settle_s)     # let the wrench settle before the next check

        self.get_logger().info('Insertion complete (contact limit not reached).')
        return True

    def _retract(self):
        """Retract in a SEQUENCE of displacement steps, each expressed in retract.frame.

        Each step moves the FINGERTIP by its displacement vector, and the steps are applied in order --
        so a multi-leg escape (back out along the mate, clear laterally, come down) is expressible
        without inventing a trajectory format.

        WHICH FRAMES MOVE matters here:
          * 'base' and 'target' are FIXED -- every step means the same world direction regardless of how
            the arm ends up. This is almost always what you want for an escape path.
          * 'tool0'/'fingertip' move WITH the arm, so their axes are re-evaluated at each step: step 2's
            "+X" is relative to wherever step 1 left the tool. Rarely what you want.
        """
        if not self.retract_steps:
            self.get_logger().info('No retract steps configured; skipping.')
            return True

        n = len(self.retract_steps)
        self.get_logger().info(
            f'Retracting in {n} step(s), each expressed in the {self.retract_frame.upper()} frame '
            f'({"FIXED" if self.retract_frame in ("base", "target") else "MOVES WITH THE ARM"}).')

        for k, d in enumerate(self.retract_steps, start=1):
            # Re-read the rotation each step: a no-op for fixed frames, but correct for moving ones.
            R = self._frame_rotation(self.retract_frame)
            if R is None:
                self.get_logger().error(
                    f"assembly.retract.frame '{self.retract_frame}' is not one of "
                    "'base' | 'target' | 'tool0' | 'fingertip' (or its tf is missing).")
                return False
            T_now = self._fingertip_now()
            if T_now is None:
                self.get_logger().error('No fingertip pose to retract from.')
                return False
            T_new = T_now.copy()
            T_new[:3, 3] = T_now[:3, 3] + R @ d
            label = (f'retract {k}/{n}: [{d[0] * 100:+.0f}, {d[1] * 100:+.0f}, {d[2] * 100:+.0f}] cm '
                     f'in {self.retract_frame}')
            if not self.move_grasp_tcp_to(T_new, label):
                self.get_logger().error(
                    f'{label} failed -- unreachable or in collision. Retract steps are LARGE '
                    'free-space moves and avoid_collisions is false: check the path is clear.')
                return False
        return True

    # ---------------------------------------------------------------- assemble
    def _assemble_kinematic(self):
        """Kinematic assembly: the target is given outright -- no perception during the assembly."""
        T_target = self._fingertip_target()
        if T_target is None:
            return False
        self._T_target_ftip = T_target
        T_standoff = self._standoff_of(T_target)
        p, q = T_target[:3, 3], T_standoff[:3, 3]
        self.get_logger().info(
            f"Kinematic assembly. target ({self.target_frame} frame) -> fingertip at "
            f'({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}); stand-off at ({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}).')

        return (
            # Stand-off is reached under POSITION control -- it is a free-space traverse, and the JTC
            # must still be active to run a trajectory.
            self._do('assembly: move to stand-off',
                     lambda: self.move_grasp_tcp_to(T_standoff, 'assembly stand-off'))
            and self._do('assembly: enable compliance', self._enable_compliance)
            and self._do('assembly: insert (chunked)',
                         lambda: self._insert_chunked(T_standoff, T_target))
            and self._do('assembly: open gripper (release)',
                         lambda: self.gripper_to(self.gripper_open, 'open'))
            # Back to position control BEFORE retracting: the retract is a free-space move.
            and self._do('assembly: restore position control', self._switch_to_position)
            and self._do('assembly: retract', self._retract))

    def _assemble(self):
        if self.method == 'kinematic':
            return self._assemble_kinematic()
        self.get_logger().error(f"assembly.method '{self.method}' is not implemented.")
        return False

    # --------------------------------------------------------------------- run
    def run(self):
        home_joints = self._current_joints()

        # 1. PICK -- identical to ur_cable_pick_place_demo (inherited), including the grasp check and
        #    its open/return/retry recovery.
        attempt = 0
        while True:
            result = self._attempt_grasp(home_joints)
            if result == 'ok':
                break
            if result == 'abort':
                return False
            if attempt >= self.grasp_max_retries:
                self.get_logger().error(
                    f'Grasp failed on all {self.grasp_max_retries + 1} attempts; aborting.')
                return False
            attempt += 1
            self.get_logger().warn(
                f'Pick-up failed (cable not seated in the fingertip groove). Recovering and retrying '
                f'(attempt {attempt + 1}/{self.grasp_max_retries + 1})...')
            if not self._recover_to_home(home_joints):
                return False

        # 2. Clear the surface before traversing to the stand-off.
        if self.lift_after_pick and not self._do(
                'lift', lambda: self.move_grasp_tcp_to(self._lift_pose(), 'lift')):
            return False

        # 3. ASSEMBLE. Never leave the arm compliant, even on abort/exception.
        ok = False
        try:
            ok = self._assemble()
        finally:
            if self._in_compliance:
                self.get_logger().warn('Restoring position control (was left compliant).')
                self._switch_to_position()
        if not ok:
            return False

        # 4. Home.
        if not self._do('return home', lambda: self.send_joints(home_joints)):
            return False
        self.get_logger().info('Cable pick-and-assemble complete.')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = CablePickAssemble()
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
