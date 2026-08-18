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
    BASE_LINK_FROM_UR_BASE, UR_JOINTS, inverse, matrix_to_rtde, pose_error, rtde_to_matrix)

log = urlog.get('arm')


class ArmError(RuntimeError):
    pass


_DEFAULT_LIMITS = (1.05, 1.2, 0.25, 0.0)     # rad/s, rad/s^2, m/s, rad/s (0 = unbounded)


def parse_limits(spd, base=_DEFAULT_LIMITS):
    """A `speed:`-style mapping -> (joint_vel, joint_accel, cart_vel, cart_rot) in SI units.

    The four canonical keys carry their unit in the name -- max_joint_velocity_deg_s,
    max_joint_acceleration_deg_s2, max_cartesian_translation_mm_s, max_cartesian_rotation_deg_s.
    Precedence preserves older configs' exact behaviour: the deg joint spellings win over the
    legacy rad ones (more explicit), but the legacy max_cartesian_velocity_m_s wins over the mm/s
    key (in older configs m/s governed moveL while mm/s only paced the compliant reference).
    Absent keys fall back to `base` -- pass the global limits there to make a partial override
    block (e.g. assembly.speed) inherit the rest."""
    spd = spd or {}
    jv = (np.radians(float(spd['max_joint_velocity_deg_s']))
          if spd.get('max_joint_velocity_deg_s') is not None
          else float(spd.get('max_joint_velocity_rad_s', 0.0)) or base[0])
    ja = (np.radians(float(spd['max_joint_acceleration_deg_s2']))
          if spd.get('max_joint_acceleration_deg_s2') is not None
          else float(spd.get('joint_acceleration_rad_s2', 0.0)) or base[1])
    mm = spd.get('max_cartesian_translation_mm_s')
    cv = (float(spd.get('max_cartesian_velocity_m_s', 0.0))
          or (float(mm) / 1000.0 if mm is not None else 0.0) or base[2])
    cr = (np.radians(float(spd['max_cartesian_rotation_deg_s']))
          if spd.get('max_cartesian_rotation_deg_s') is not None else base[3])
    return jv, ja, cv, cr


class URArm:
    """Blocking arm control. Every move returns True/False; nothing spins an executor."""

    def __init__(self, cfg, frames=None):
        r = cfg.section('robot')
        self.ip = r.get('ip', '192.168.125.2')
        self.dry_run = bool(r.get('dry_run', False))
        self.joint_names = list(UR_JOINTS)

        # Speed caps. Applied directly as RTDE limits, so these are HARD ceilings, not the
        # average-velocity bounds the old duration-based scheme produced. The GLOBAL `speed:`
        # block sets the four limits for every move; a caller may pass a `caps=` mapping (same
        # schema) to move_j / move_l to override them for one move (e.g. assembly.speed).
        speed = cfg.section('speed')
        (self.max_joint_vel, self.joint_accel,
         self.max_cart_vel, self.max_cart_rot) = parse_limits(speed)
        self.cart_accel = float(speed.get('cartesian_acceleration_m_s2', 0.5))
        self.speed_scale = 1.0            # the CURRENT phase's scaling (set_speed_scale)

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
            self._set_payload(r.get('payload', {}) or {})
            self._report_tcp_offset()

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
    def set_speed_scale(self, scale, phase=''):
        """Set the CURRENT phase's scaling of the global speed limits (speed.phase_scale.<phase>).
        Applies to every following move until changed; a per-move `caps` still overrides it."""
        self.speed_scale = max(float(scale), 1e-3)
        log.info('Speed scale%s: %.2fx the global limits',
                 f' [{phase}]' if phase else '', self.speed_scale)

    def _limits(self, caps):
        """The four limits for one move. None = the global limits x the CURRENT phase scale
        (set_speed_scale). Otherwise `caps` REPLACES the phase scale for this move: either a
        `speed:`-style mapping (absolute overrides; absent keys inherit the global values), or a
        bare NUMBER scaling all four global limits (e.g. 0.2 = a fifth of every limit)."""
        mine = (self.max_joint_vel, self.joint_accel, self.max_cart_vel, self.max_cart_rot)
        if caps is None:
            s = self.speed_scale
            return tuple(v * s for v in mine)
        if isinstance(caps, (int, float)):
            return tuple(v * float(caps) for v in mine)
        return parse_limits(caps, mine)

    def _speeds(self, q_target, caps=None):
        """(joint speed, accel, name-of-binding-cap) honouring EVERY cap -- whichever binds.

        The joint caps are direct. Each Cartesian cap is converted into an equivalent joint
        speed: the tool travels `d` (metres, or radians of tool rotation) while the largest joint
        travels `dj` radians, so holding the tool under the cap means holding that joint under
        `dj/d * cap`. The min over all of them respects every limit simultaneously."""
        jv, ja, cv, cr = self._limits(caps)
        q_now = np.asarray(self.q(), dtype=float)
        dj = float(np.max(np.abs(np.asarray(q_target, dtype=float) - q_now)))
        speed, which = jv, 'joint-velocity'
        if dj > 1e-6 and (cv > 0.0 or cr > 0.0):
            lin, ang = pose_error(self.tcp_pose(), self.fk(q_target))
            # SANITY-GATE the ratios: lin/dj is the move's effective lever arm, physically bounded
            # by the reach (~1.3 m; tool rotation similarly by the sum of joint rotations). A ratio
            # beyond that means the pose delta is NOT this move's motion but the constant few-mm
            # disagreement between getActualTCPPose and getForwardKinematics -- which DOMINATES
            # when the arm is already AT the target (dj ~ 0) and would collapse the bound to the
            # 1e-3 floor ("moveJ 0.00 rad/s"). Sub-mm moves need no cartesian pacing either way.
            if cv > 0.0 and 1e-3 < lin <= 2.0 * dj and dj / lin * cv < speed:
                speed, which = dj / lin * cv, 'cartesian-translation'
            if cr > 0.0 and 1e-3 < ang <= 8.0 * dj and dj / ang * cr < speed:
                speed, which = dj / ang * cr, 'cartesian-rotation'
        return max(speed, 1e-3), ja, which

    def move_j(self, q_target, speed=None, accel=None, label='move', caps=None):
        """Joint-space move to `q_target`. Blocking; returns True on success.

        SYNCHRONOUS unless a guard is armed. A synchronous moveJ blocks until the arm has actually
        finished -- simplest, and immune to the async-progress race: right after an async moveJ,
        ur_rtde's getAsyncOperationProgress() can report 'no operation' for a tick before the move
        registers, and a wait that reads that as 'done' returns instantly, so the next command
        supersedes the move before it happens and the arm never visibly moves. Only a guarded move
        needs to run async, so the guard can be polled while the arm is in motion."""
        if self.dry_run:
            log.info('[dry-run] %s -> q=%s', label, np.round(q_target, 3))
            self._sim_q = np.asarray(q_target, dtype=float)
            return True

        auto_speed, auto_accel, which = self._speeds(q_target, caps)
        if speed is None:
            speed = auto_speed
        else:
            which = 'explicit'
        accel = auto_accel if accel is None else accel
        # Log the COMMANDED speed and the cap that produced it: if the arm visibly moves slower
        # than this line says, the throttle is on the CONTROLLER side (pendant speed slider,
        # safety Reduced mode / restricted limits), not in this code or the config.
        log.info('[%s] moveJ %.2f rad/s (%.0f deg/s leading joint, %s cap), accel %.2f rad/s^2',
                 label, speed, np.degrees(speed), which, accel)

        if not self._guards:
            ok = self.rtde_c.moveJ(list(q_target), speed, accel, False)   # blocks until finished
            if not ok:
                log.error('[%s] move rejected/failed. On the pendant: Remote Control on, brakes '
                          'released (green "Normal"), speed slider up, and NOT in Simulation mode?',
                          label)
            return bool(ok)

        self.rtde_c.moveJ(list(q_target), speed, accel, True)             # async -> pollable guard
        return self._await_move(label)

    def move_l(self, T_base_tool0, speed=None, accel=None, label='move', caps=None):
        """Straight-line Cartesian move (the tool travels a line in space, not a joint arc).
        `caps` (a `speed:`-style mapping) overrides the global limits for this move; moveL's RTDE
        speed is a TCP linear speed, so the translation cap is the one applied.

        Synchronous unless a guard is armed, for the same reason as move_j."""
        if self.dry_run:
            log.info('[dry-run] %s -> linear', label)
            return True
        pose = matrix_to_rtde(T_base_tool0)
        speed = self._limits(caps)[2] if speed is None else speed
        accel = self.cart_accel if accel is None else accel
        log.info('[%s] moveL %.3f m/s, accel %.2f m/s^2', label, speed, accel)
        if not self._guards:
            ok = self.rtde_c.moveL(pose, speed, accel, False)
            if not ok:
                log.error('[%s] linear move rejected/failed (unreachable, or the checks above).',
                          label)
            return bool(ok)
        self.rtde_c.moveL(pose, speed, accel, True)
        return self._await_move(label)

    def _await_move(self, label):
        """Poll an ASYNC move until it finishes, a guard trips, or it times out.

        Waits for the move to actually START before accepting 'no operation' as finished. Right
        after an async move, getAsyncOperationProgress() can read < 0 for a tick before the
        controller registers it; returning then would skip the move. So < 0 counts as 'done' only
        once we have seen it running -- or after a short grace, which covers a genuinely instant
        (near-zero) move that never registers progress at all."""
        deadline = time.monotonic() + self.move_timeout
        start_grace = time.monotonic() + 1.0
        last_log = 0.0
        started = False
        while True:
            prog = self.rtde_c.getAsyncOperationProgress()
            if prog >= 0:
                started = True
            elif started or time.monotonic() > start_grace:
                time.sleep(0.02)                        # let the final setpoint land
                return True

            if self._tripped():
                log.warning('[%s] guard tripped -- stopping the arm.', label)
                self.rtde_c.stopJ(2.0)
                return False

            now = time.monotonic()
            if now > deadline:
                log.error(
                    '[%s] not finished after %.0fs. On the pendant: Remote Control on, no '
                    'protective/e-stop, speed slider up, NOT in Simulation mode? Stopping.',
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

    def servo_l(self, T_base_tool0, dt, lookahead=0.1, gain=300):
        """One step of a streamed CARTESIAN reference -- the software-admittance inner loop.

        Drives tool0 toward the pose (the controller's own IK handles it) and returns immediately.
        Like servo_j it plans no profile, so call it at a steady rate; the initPeriod/waitPeriod
        pair holds that rate. `dt` is the control period."""
        if self.dry_run:
            self._sim_q = self.ik(T_base_tool0) or self._sim_q
            return
        t0 = self.rtde_c.initPeriod()
        self.rtde_c.servoL(matrix_to_rtde(T_base_tool0), 0.0, 0.0, dt, lookahead, gain)
        self.rtde_c.waitPeriod(t0)

    def servo_stop(self):
        if not self.dry_run:
            self.rtde_c.servoStop()

    def stop(self):
        if not self.dry_run:
            self.rtde_c.stopJ(2.0)

    # ------------------------------------------------------------------ force / torque
    def _report_tcp_offset(self):
        """Read and report the controller's configured TCP offset -- it decides where the logged
        MOMENT is referenced, and nothing else in this stack can tell.

        wrench() returns a force plus a moment about a point the controller chooses, and the two
        candidate readings of the documentation disagree about which point that is: the SW5.19
        URScript page for get_tcp_force() says the tool flange, while the RTDE field ur_rtde
        actually reads (actual_TCP_force) is described as being at the TCP. They are the same
        point only when this offset is zero.

        wrench_in() references the moment at tcp_pose(), i.e. the controller's TCP -- correct
        under the RTDE reading unconditionally, and under the URScript reading when the offset is
        zero. A NON-ZERO offset therefore means the two readings differ by exactly that vector,
        and it also means `tip_frame: tool0` ("the pendant all-zeros TCP") is no longer true, so
        every pose in this stack inherits the same shift. Worth a loud line either way."""
        try:
            off = np.asarray(self.rtde_c.getTCPOffset(), dtype=float)
        except Exception as exc:                       # noqa: BLE001 -- never fatal at connect
            log.warning('Could not read the controller TCP offset (%s). Poses assume it is zero '
                        '(tip_frame: tool0) and wrench_in references the moment at tcp_pose().',
                        exc)
            return
        d_mm = float(np.linalg.norm(off[:3])) * 1000.0
        if d_mm < 0.05:
            log.info('Controller TCP offset is zero -- tool flange == TCP == tool0, so the logged '
                     'moment is referenced there and every frame in this repo means what it says.')
        else:
            log.warning(
                'Controller TCP offset is NON-ZERO: %s mm. Two consequences, both silent '
                'otherwise. (1) tip_frame: tool0 is documented as "the pendant all-zeros TCP", '
                'which is no longer true -- every pose read from RTDE is the TCP, shifted by this '
                'from the flange. (2) The logged wrench MOMENT is referenced at whichever point '
                'the firmware uses, and the URScript and RTDE docs disagree by exactly this '
                'vector; wrench_in() references it at tcp_pose(). Zero the TCP on the pendant, or '
                'confirm which end the moment comes from before trusting the torque columns.',
                np.round(off[:3] * 1000.0, 2).tolist())

    def _set_payload(self, payload):
        """Tell the controller the tool's mass + CoG so getActualTCPForce() subtracts the tool's
        own weight and reports true EXTERNAL force. Without this, the wrench reads the tool weight
        (~tens of N) as contact -- and a tare only cancels it at the tare pose and only while idle,
        so a force guard trips spuriously the moment the arm is under active control. Configure
        robot.payload.mass_kg + cog_m for correct readings everywhere (guard, admittance, touch)."""
        mass = float(payload.get('mass_kg', 0.0))
        cog = [float(v) for v in payload.get('cog_m', [0.0, 0.0, 0.0])]
        if mass > 0.0:
            try:
                self.rtde_c.setPayload(mass, cog)
                log.info('Payload set: %.2f kg, CoG %s m.', mass, cog)
            except Exception as exc:                       # noqa: BLE001
                log.warning('setPayload failed: %s', exc)
        else:
            log.warning('No robot.payload configured -- getActualTCPForce() will read the tool '
                        "weight as external force, so the force guard/admittance can't be trusted. "
                        'Set robot.payload.mass_kg and cog_m (mass + centre of gravity of the '
                        'gripper + coupler + camera).')

    def zero_ft(self, settle=True):
        """Tare the wrist F/T. Everything after this is CONTACT force -- so do it with the tool
        hanging free, not in contact. `settle=False` skips the sample-wait + residual check, for
        taring INSIDE a servo loop (where blocking would drop servo mode)."""
        if self.dry_run:
            return True
        ok = self.rtde_c.zeroFtSensor()
        if not settle:
            return ok
        time.sleep(0.3)                     # let a fresh tared sample arrive before anyone reads
        residual = float(np.linalg.norm(self.wrench()[:3]))
        if residual > 5.0:
            log.warning(
                'Residual force %.1f N after taring. The tool weight is NOT being compensated '
                '(set robot.payload) -- force thresholds will be wrong and force mode will drift.',
                residual)
        return ok

    def wrench(self):
        """Contact wrench [fx, fy, fz, tx, ty, tz]: force and the moment ABOUT THE TOOL FLANGE,
        both expressed in ROS base_link axes, tared.

        THE REFERENCE POINT IS THE FLANGE, NOT THE BASE ORIGIN. The UR script manual for
        get_tcp_force says the components are "all measured at the tool flange with the
        orientation of the robot base coordinate system" -- so this is a moment about a point
        roughly a metre from the base origin, and anything that re-expresses it has to move that
        reference point before rotating. wrench_in() does; a bare rotation would not.

        getActualTCPForce() reports in the UR `base` frame, which differs from ROS `base_link` by
        Rz(pi) -- the SAME bridge every pose crosses via rtde_to_matrix. It must be applied here
        too, or the wrench is silently mixed with base_link poses and its x/y components come out
        NEGATED while z is fine. That asymmetry is invisible to anything reading a MAGNITUDE (the
        force guard, force(), torque()) but inverts the compliant axes of the admittance law and
        mislabels the logged wrench columns. Pure rotation about a shared origin -> no cross term.

        NOTE this is the BASE frame, whereas the ROS broadcaster published in tool0_controller.
        Force MAGNITUDE is frame-invariant so the force guards carry over unchanged; TORQUE
        magnitude is NOT (it depends on the reference origin), so a torque threshold tuned
        against the ROS stack needs re-checking. Use wrench_in() to get it in a tool frame."""
        if self.dry_run:
            return np.zeros(6)
        w = np.asarray(self.rtde_r.getActualTCPForce(), dtype=float)
        R = BASE_LINK_FROM_UR_BASE[:3, :3]                   # UR base -> ROS base_link
        return np.concatenate([R @ w[:3], R @ w[3:]])

    def wrench_in(self, T_base_frame, T_base_flange=None):
        """The contact wrench moved TO another frame's origin and expressed IN its axes.

        Two separate operations, and the first one is the easy one to skip:

          1. MOVE THE REFERENCE POINT. wrench() reports the moment about the TOOL FLANGE (the UR
             script manual for get_tcp_force is explicit: "all measured at the tool flange with
             the orientation of the robot base coordinate system"). A moment is only meaningful
             about a stated point, so re-referencing it to the target frame's origin is a
             physical change, not a rotation: tau_new = tau + (p_flange - p_target) x f.
          2. ROTATE into the target frame's axes: R^T applied to both halves.

        SKIPPING STEP 1 IS A SILENT, LARGE ERROR. Composing only the rotation implicitly claims
        the moment was about the BASE ORIGIN, which adds a phantom lever of |base -> flange| --
        about a metre on this arm. Measured on the 2026-08 BNC logs before this fix: the logged
        connector torque was 90% explained (R^2 0.90) by a single cross product p x f with
        |p| = 1024 mm, against a predicted |base -> flange| of 1050 mm, and tau came out
        perpendicular to f in 98.8% of rows -- the signature of a pure lever-arm artefact rather
        than a contact moment. A BNC contact moment should show a lever of a few millimetres.

        `T_base_flange` lets a caller that already read the arm pose pass it in: it saves a
        second RTDE round trip and, more importantly, guarantees the pose and the wrench come
        from the SAME sample rather than two reads a cycle apart.
        """
        w = self.wrench()
        f_b, tau_b = np.asarray(w[:3], dtype=float), np.asarray(w[3:], dtype=float)
        T = np.asarray(T_base_frame, dtype=float)
        p_flange = (np.asarray(T_base_flange, dtype=float)[:3, 3] if T_base_flange is not None
                    else self.tcp_pose()[:3, 3])
        tau_b = tau_b + np.cross(p_flange - T[:3, 3], f_b)     # (1) re-reference the moment
        R = T[:3, :3]
        return np.concatenate([R.T @ f_b, R.T @ tau_b])        # (2) rotate into the frame

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
