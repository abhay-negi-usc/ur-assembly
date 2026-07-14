"""URArm -- the UR10e over RTDE.

Replaces, in one class, what took four ROS mechanisms:

    MoveIt /compute_ik                      -> getInverseKinematics()
    FollowJointTrajectory action + JTC      -> moveJ()
    force_torque_sensor_broadcaster/wrench  -> getActualTCPForce()
    admittance_controller + switch_controller + set_parameters
                                            -> forceMode() / endForceMode()

Three of those swaps delete a whole bug class rather than just re-homing it:

  * IK. MoveIt's KDL solver is a LOCAL search seeded from one joint state, so a perfectly
    reachable pose whose solution lives in another IK branch returned error -31 and the old code
    papered over it by retrying with up to 12 RANDOM seeds. The UR controller's own IK is
    ANALYTIC -- it enumerates all eight branches and returns the one nearest `qnear`. There is
    nothing to retry, and the answer is the closest solution rather than a lucky one.

  * Compliance. The ros2_control admittance controller had to be installed, loaded inactive,
    parameterised over a service, and ACTIVATED -- which deactivated the trajectory controller.
    Activating it also made the arm jump to whatever reference it had, which is what caused the
    joint-0 velocity fault (the fix was an elaborate dance pinning the reference to the arm's
    current pose on both sides of both switches). forceMode() has no such transition: it is a
    mode of the SAME controller, and it starts from where the arm is. The _hold_reference dance
    is not ported because there is nothing left for it to fix.

  * Speed caps. `_move_duration_for` computed a DURATION from a velocity cap and then trusted the
    controller to respect it, bounding only the AVERAGE velocity (peak ~1.5x). moveJ takes a
    joint-velocity limit directly, and the controller enforces it exactly.

Frames: RTDE speaks the UR `base` frame, this class speaks ROS `base_link` (the frame every
config in this repo is measured in). transforms.rtde_to_matrix / matrix_to_rtde are the only
place that conversion happens -- see the note there.
"""

import time

import numpy as np

from .. import log as urlog
from ..transforms import (
    UR_JOINTS, inverse, matrix_to_rtde, rtde_to_matrix, transform_wrench)

log = urlog.get('arm')


class ArmError(RuntimeError):
    pass


class URArm:
    """Blocking arm control. Every move returns True/False; nothing spins an executor."""

    def __init__(self, cfg, frames=None):
        r = cfg.section('robot')
        self.ip = r.get('ip', '192.168.125.2')
        self.dry_run = bool(r.get('dry_run', False))
        self.joint_names = list(UR_JOINTS)

        # Speed caps. Applied directly as RTDE limits, so these are HARD ceilings, not the
        # average-velocity bounds the old duration-based scheme produced.
        speed = cfg.section('speed')
        self.max_joint_vel = float(speed.get('max_joint_velocity_rad_s', 0.0)) or 1.05
        self.max_cart_vel = float(speed.get('max_cartesian_velocity_m_s', 0.0)) or 0.25
        self.joint_accel = float(speed.get('joint_acceleration_rad_s2', 1.2))
        self.cart_accel = float(speed.get('cartesian_acceleration_m_s2', 0.5))

        self.move_timeout = float(cfg.get('move_timeout_s', 60.0))
        self.settle_s = float(cfg.get('settle_s', 0.5))
        self.debug = bool(cfg.get('debug', False))

        self._force_mode = False
        self._ft_offset = np.zeros(6)
        self._guards = []

        if self.dry_run:
            log.warning('DRY RUN: no robot connection; moves are logged, not executed.')
            self.rtde_c = self.rtde_r = None
            self._sim_q = np.array(r.get('dry_run_joints',
                                         [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]), dtype=float)
        else:
            try:
                from rtde_control import RTDEControlInterface
                from rtde_receive import RTDEReceiveInterface
            except ImportError as exc:
                raise ArmError(
                    'ur_rtde is not installed. `pip install ur_rtde` (Linux wheels exist for '
                    'cp38-cp312). Use --dry-run to plan without a robot.') from exc

            log.info('Connecting to the UR at %s ...', self.ip)
            # RTDEControlInterface uploads its OWN control script to the robot. The External
            # Control URCap program must NOT be playing -- if it is, the two fight over the
            # controller and this constructor hangs or the arm refuses to move.
            self.rtde_c = RTDEControlInterface(self.ip)
            self.rtde_r = RTDEReceiveInterface(self.ip)
            log.info('Connected.')

        if frames is not None:
            self.publish_frames(frames, cfg)

    # ------------------------------------------------------------------ frames
    def publish_frames(self, frames, cfg):
        """Register base_link -> tool0 as a LIVE edge: read from RTDE on every lookup, so it is
        fresh by construction and can never go stale the way a republished tf could."""
        self.base_frame = cfg.get('base_frame', 'base_link')
        self.tip_frame = cfg.get('tip_frame', 'tool0')
        frames.set_live(self.base_frame, self.tip_frame, self.tcp_pose)

    # ------------------------------------------------------------------ state
    def q(self):
        """Current joint positions (rad), in UR_JOINTS order."""
        if self.dry_run:
            return list(self._sim_q)
        return list(self.rtde_r.getActualQ())

    def tcp_pose(self):
        """Current tool0 pose in base_link, as a 4x4."""
        if self.dry_run:
            return self.fk(self._sim_q)
        return rtde_to_matrix(self.rtde_r.getActualTCPPose())

    def fk(self, q):
        """Forward kinematics: joints -> tool0 in base_link."""
        if self.dry_run:
            # No kinematic model offline; a fixed plausible pose keeps dry runs from crashing on
            # a None, but every geometric result in a dry run is therefore meaningless.
            return rtde_to_matrix([0.4, 0.0, 0.4, 0.0, 3.14, 0.0])
        return rtde_to_matrix(self.rtde_c.getForwardKinematics(list(q)))

    def ik(self, T_base_tool0, qnear=None):
        """Inverse kinematics: tool0 pose in base_link -> joints, or None if unreachable.

        The controller's analytic solver returns the branch nearest `qnear` (default: where the
        arm is now), so successive calls along a path stay on one branch and the arm does not
        flip its elbow mid-trajectory."""
        if self.dry_run:
            return list(self._sim_q)
        pose = matrix_to_rtde(T_base_tool0)
        if not self.rtde_c.isPoseWithinSafetyLimits(pose):
            log.error('Pose is outside the robot safety limits: %s', np.round(pose, 4))
            return None
        qnear = list(qnear) if qnear is not None else self.q()
        q = self.rtde_c.getInverseKinematics(pose, qnear)
        if not q or not self.rtde_c.isJointsWithinSafetyLimits(list(q)):
            log.error('No IK solution (unreachable, in collision, or at a singularity).')
            return None
        return list(q)

    # ------------------------------------------------------------------ guards
    def add_guard(self, guard):
        """Arm a callable checked DURING every move; returning True cancels the motion.

        This is the force guard's hook. In the ROS stack the equivalent (_abort_move) could only
        be polled between executor spins; here the move loop owns the thread and polls at 100 Hz,
        so a collision stops the arm within ~10 ms of tripping instead of within a spin."""
        self._guards.append(guard)

    def clear_guards(self):
        self._guards = []

    def _tripped(self):
        for guard in self._guards:
            if guard():
                return True
        return False

    # ------------------------------------------------------------------ motion
    def _speeds(self, q_target):
        """(joint speed, accel) honouring BOTH caps.

        The joint cap is direct. The Cartesian cap is converted into an equivalent joint speed:
        the tool travels `d` metres while the largest joint travels `dj` radians, so holding the
        tool under `max_cart_vel` means holding that joint under `dj/d * max_cart_vel`. Taking
        the min of the two gives a move that respects whichever cap actually binds."""
        q_now = np.asarray(self.q(), dtype=float)
        dj = float(np.max(np.abs(np.asarray(q_target, dtype=float) - q_now)))
        speed = self.max_joint_vel
        if dj > 1e-6 and self.max_cart_vel > 0.0:
            d = float(np.linalg.norm(self.fk(q_target)[:3, 3] - self.tcp_pose()[:3, 3]))
            if d > 1e-6:
                speed = min(speed, dj / d * self.max_cart_vel)
        return max(speed, 1e-3), self.joint_accel

    def move_j(self, q_target, speed=None, accel=None, label='move'):
        """Joint-space move to `q_target`. Blocking, guard-checked. Returns True on success."""
        if self.dry_run:
            log.info('[dry-run] %s -> q=%s', label, np.round(q_target, 3))
            self._sim_q = np.asarray(q_target, dtype=float)
            return True

        auto_speed, auto_accel = self._speeds(q_target)
        speed = auto_speed if speed is None else speed
        accel = auto_accel if accel is None else accel

        # asynchronous=True hands control back immediately so we can poll the guards while the
        # arm is moving. The synchronous form would block until the move finished, which is
        # exactly when a force guard is too late to be useful.
        self.rtde_c.moveJ(list(q_target), speed, accel, True)
        return self._await_move(label)

    def move_l(self, T_base_tool0, speed=None, accel=None, label='move'):
        """Straight-line Cartesian move (the tool travels a line in space, not a joint arc)."""
        if self.dry_run:
            log.info('[dry-run] %s -> linear', label)
            return True
        self.rtde_c.moveL(matrix_to_rtde(T_base_tool0),
                          self.max_cart_vel if speed is None else speed,
                          self.cart_accel if accel is None else accel, True)
        return self._await_move(label)

    def _await_move(self, label):
        """Poll until the async move finishes, the guard trips, or the timeout expires."""
        deadline = time.monotonic() + self.move_timeout
        last_log = 0.0
        while True:
            # < 0 means "no async operation running", i.e. finished. ur_rtde returns the progress
            # of the running op otherwise.
            if self.rtde_c.getAsyncOperationProgress() < 0:
                time.sleep(0.02)                        # let the final setpoint land
                return True

            if self._tripped():
                log.warning('[%s] guard tripped -- stopping the arm.', label)
                self.rtde_c.stopJ(2.0)
                return False

            now = time.monotonic()
            if now > deadline:
                log.error(
                    '[%s] not finished after %.0fs. On the pendant: is the robot in Remote '
                    'Control, is there a protective/e-stop, and is the speed slider up? Stopping.',
                    label, self.move_timeout)
                self.rtde_c.stopJ(2.0)
                return False
            if now - last_log > 5.0:
                log.info('[%s] executing...', label)
                last_log = now
            time.sleep(0.01)

    def move_to(self, T_base_tool0, label='move', qnear=None):
        """IK + joint move to a tool0 pose. The workhorse: every app move goes through here."""
        if self.debug:
            from ..transforms import fmt_delta, fmt_pose
            log.info('[%s] target %s', label, fmt_pose(T_base_tool0))
            log.info('[%s] delta from current: %s', label,
                     fmt_delta(self.tcp_pose(), T_base_tool0))
        q = self.ik(T_base_tool0, qnear)
        if q is None:
            from ..transforms import fmt_delta, fmt_pose
            log.error('[%s] IK UNREACHABLE.', label)
            log.error('  current: %s', fmt_pose(self.tcp_pose()))
            log.error('  target:  %s', fmt_pose(T_base_tool0))
            log.error('  delta:   %s', fmt_delta(self.tcp_pose(), T_base_tool0))
            return False
        if not self.move_j(q, label=label):
            return False
        time.sleep(self.settle_s)
        return True

    def move_frame_to(self, T_base_target, T_tool0_frame, label='move', qnear=None):
        """Move so that the frame rigidly attached to tool0 by `T_tool0_frame` lands on
        `T_base_target`.

        This is how a fingertip, a grasp TCP, or a held object gets commanded: you say where the
        THING should go, and the tool0 pose that puts it there is inv()'d out. Every demo that
        controls something other than the flange goes through this one method."""
        return self.move_to(T_base_target @ inverse(T_tool0_frame), label, qnear)

    def servo_j(self, q, dt, lookahead=0.1, gain=300):
        """One step of a streamed joint reference -- the compliant-insertion inner loop.

        Unlike moveJ this does NOT plan a profile; it drives toward `q` and returns immediately.
        Call it at a steady rate or the motion will be jerky."""
        if self.dry_run:
            self._sim_q = np.asarray(q, dtype=float)
            return
        t0 = self.rtde_c.initPeriod()
        self.rtde_c.servoJ(list(q), 0.0, 0.0, dt, lookahead, gain)
        self.rtde_c.waitPeriod(t0)

    def servo_stop(self):
        if not self.dry_run:
            self.rtde_c.servoStop()

    def stop(self):
        if not self.dry_run:
            self.rtde_c.stopJ(2.0)

    # ------------------------------------------------------------------ force / torque
    def zero_ft(self):
        """Tare the wrist F/T. Everything after this is CONTACT force, with the tool's own weight
        subtracted -- so it must be done with the tool hanging free, not in contact."""
        if self.dry_run:
            return True
        ok = self.rtde_c.zeroFtSensor()
        time.sleep(0.3)                     # let a fresh tared sample arrive before anyone reads
        residual = float(np.linalg.norm(self.wrench()[:3]))
        if residual > 5.0:
            log.warning(
                'Residual force %.1f N after taring. The tool weight is NOT being compensated '
                '(check the payload/CoG on the pendant) -- force thresholds will be wrong and '
                'force mode will drift.', residual)
        return ok

    def wrench(self):
        """TCP wrench [fx, fy, fz, tx, ty, tz] in base_link, tared.

        NOTE this is the BASE frame, whereas the ROS broadcaster published in tool0_controller.
        Force MAGNITUDE is frame-invariant so the force guards carry over unchanged; TORQUE
        magnitude is NOT (it depends on the reference origin), so a torque threshold tuned
        against the ROS stack needs re-checking. Use wrench_in() to get it in a tool frame."""
        if self.dry_run:
            return np.zeros(6)
        return np.asarray(self.rtde_r.getActualTCPForce(), dtype=float)

    def wrench_in(self, T_base_frame):
        """The TCP wrench re-expressed in an arbitrary frame (given as its pose in base_link)."""
        w = self.wrench()
        f, tau = transform_wrench(w[:3], w[3:], inverse(T_base_frame))
        return np.concatenate([f, tau])

    def force(self):
        return float(np.linalg.norm(self.wrench()[:3]))

    def torque(self):
        return float(np.linalg.norm(self.wrench()[3:]))

    # ------------------------------------------------------------------ compliance
    def force_mode(self, T_base_task, selection, wrench, limits, mode=2, damping=0.005,
                   gain_scaling=0.8):
        """Enter compliant (force) control.

        `selection` marks the COMPLIANT axes [x, y, z, rx, ry, rz] of the task frame: 1 = yield to
        contact and regulate toward the requested `wrench`, 0 = stay rigidly position-controlled.
        `limits` bounds the compliant axes' speed (m/s, rad/s) and the non-compliant ones'
        deviation (m, rad).

        This is a mode of the running controller, not a different controller: the arm keeps
        tracking whatever you servo to it, but yields on the selected axes. Nothing is switched
        out, so there is no activation jump -- the joint-0 velocity fault that the ros2_control
        admittance controller produced on every switch simply cannot happen here."""
        if self.dry_run:
            self._force_mode = True
            return True
        self.rtde_c.forceModeSetDamping(damping)
        self.rtde_c.forceModeSetGainScaling(gain_scaling)
        self.rtde_c.forceMode(matrix_to_rtde(T_base_task), list(selection),
                              list(wrench), int(mode), list(limits))
        self._force_mode = True
        return True

    def end_force_mode(self):
        """Leave force mode. Idempotent and safe to call from a finally: -- and it must be, or a
        crash mid-insertion leaves the arm compliant."""
        if self._force_mode and not self.dry_run:
            self.rtde_c.forceModeStop()
        self._force_mode = False

    @property
    def in_force_mode(self):
        return self._force_mode

    # ------------------------------------------------------------------ lifecycle
    def disconnect(self):
        if self.dry_run:
            return
        try:
            self.end_force_mode()
            self.rtde_c.servoStop()
            self.rtde_c.stopScript()
        except Exception as exc:                       # noqa: BLE001 -- best-effort teardown
            log.warning('Teardown: %s', exc)
        finally:
            try:
                self.rtde_c.disconnect()
                self.rtde_r.disconnect()
            except Exception:                          # noqa: BLE001
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.disconnect()
        return False
