"""Grasp helpers shared by the cable demos -- pre-grasp geometry, the counts-based grasp check,
and the open/return/retry recovery.

Ported from PickPlace's grasp/lift/place geometry and CablePickPlace's grasp-check + recovery.
The grasp check is genuinely better here than in the ROS version, for a hardware reason: the
Robotiq talks to us in COUNTS directly (see robot/gripper.py), so the whole rad->counts
conversion through `full_close_rad` -- the config value that was permanently marked "TODO: verify
me" -- is gone. `closed_counts: 228` is now compared against a number the gripper reports.
"""

import numpy as np

from .. import log as urlog
from ..transforms import fmt_delta, inverse, pose_error, translation_matrix

log = urlog.get('pick')


class GraspGeometry:
    """The grasp/pre-grasp/lift/place poses, all derived from one grasp target on demand -- so a
    refined estimate that updates the target automatically feeds every downstream pose."""

    def __init__(self, cfg):
        self.approach_distance = float(cfg.get('approach_distance_m', 0.10))
        self.approach_axis = np.asarray(cfg.get('approach_axis', [0.0, 0.0, -1.0]), dtype=float)
        self.lift_distance = float(cfg.get('lift_distance_m', 0.10))
        self.lift_axis = np.asarray(cfg.get('lift_axis', [0.0, 0.0, 1.0]), dtype=float)
        from ..transforms import xyzrpy_to_matrix
        self.place_offset = xyzrpy_to_matrix(cfg.get('place_offset_xyz', [0.0, 0.20, 0.0]),
                                             cfg.get('place_offset_rpy', [0.0, 0.0, 0.0]))
        self.T_base_grasp = np.eye(4)

    def pre_grasp(self):
        """Stand-off before the grasp, along approach_axis IN THE GRASP FRAME."""
        return self.T_base_grasp @ translation_matrix(self.approach_axis * self.approach_distance)

    def lift(self):
        return translation_matrix(self.lift_axis * self.lift_distance) @ self.T_base_grasp

    def place(self):
        return self.place_offset @ self.T_base_grasp

    def pre_place(self):
        return translation_matrix(self.lift_axis * self.lift_distance) @ self.place()


class GraspCheck:
    """Classify a completed close from the finger position, in counts.

    The current fingertips give THREE distinct closure levels (more obstruction = LESS closed):

        faces (<= faces_max_counts, ~220-223)  cable caught on the flat faces -> MISSED
        groove (~groove_counts, 225)           cable seated in the fingertip groove -> OK
        empty (>= empty_counts, 228)           fingers closed fully on nothing -> EMPTY

    So a seated cable STOPS the fingers short of full closure (unlike the previous fingertips, where
    a seated cable allowed full closure and EMPTY was indistinguishable). Empty is now reliably
    separable by POSITION (228 vs 225), which is why detect_empty defaults on."""

    def __init__(self, cfg):
        gc = cfg.section('grasp_check')
        self.enabled = bool(gc.get('enabled', True))
        self.groove_counts = int(gc.get('groove_counts', 225))     # SUCCESS: target in the groove
        self.empty_counts = int(gc.get('empty_counts', 228))       # full closure on nothing
        self.faces_max_counts = int(gc.get('faces_max_counts', 223))  # <= this: too thick (faces) -> miss
        gm = gc.get('groove_max_counts', None)                     # > this (but < empty): too THIN -> miss
        self.groove_max_counts = int(gm) if gm is not None else None  # (e.g. cable, not the connector)
        self.tolerance = int(gc.get('tolerance_counts', 1))
        self.detect_empty = bool(gc.get('detect_empty', True))
        self.max_retries = int(gc.get('max_retries', 2))
        self.settle_s = float(gc.get('settle_s', 1.0))

    def evaluate(self, gripper):
        """'ok' | 'missed' | 'empty' from the settled finger position."""
        if not self.enabled:
            return 'ok'
        import time
        time.sleep(self.settle_s)
        return gripper.grasp_result(self.groove_counts, self.empty_counts, self.faces_max_counts,
                                    self.tolerance, self.detect_empty, self.groove_max_counts)


class GraspRecovery:
    """Failed-grasp recovery for the CONNECTOR pick, classified purely by the stalled finger COUNT
    (more obstruction = LESS closed). The grasp target is the connector; the count after a close
    says what is between the fingers, and each state maps to a DIRECTED reseat in the junction frame
    (x = connector axis toward the connector's END; z = up, so -z = toward the ground):

      * CONNECTOR band  -> the connector is seated: SUCCESS.
      * CABLE (within tolerance of cable_counts) -> grabbed the thinner cable: open, shift
                           cable_shift_fraction * finger_width in +x (toward the connector end), retry.
      * CLOSED (within tolerance of the closed/empty position) -> nothing grasped: open, drop
                           empty_drop_m in -z (toward the object on the ground), retry.
      * anything else    -> unexpected: open + blind retry (no move).

    The reseat ACCUMULATES into geom.T_base_grasp, so a successful reseat leaves the corrected pose
    for the lift/place. (The old faces/tips reseat is gone.)"""

    def __init__(self, cfg):
        gc = cfg.section('grasp_check')
        rc = (gc.get('recovery', {}) or {})
        self.enabled = bool(rc.get('enabled', True))
        self.max_tries = int(rc.get('max_tries', 5))
        fw = float(rc.get('finger_width_m', 0.02278))
        self.cable_shift = float(rc.get('cable_shift_fraction', 0.8)) * fw   # +x reseat for a cable grab
        self.empty_drop = float(rc.get('empty_drop_m', 0.003))              # -z reseat for an empty close

        # Count bands (from the grasp_check block, set per-cable by apply_cable_profile): the CONNECTOR
        # range is success, cable_counts / the closed position are the two miss states.
        conn = gc.get('connector_counts') or []
        self.connector_lo = int(min(conn)) if conn else int(gc.get('faces_max_counts', 223)) + 1
        self.connector_hi = (int(max(conn)) if conn
                             else int(gc.get('groove_max_counts', gc.get('groove_counts', 225))))
        cc = gc.get('cable_counts')
        self.cable_counts = int(cc) if cc is not None else None
        self.closed_counts = int(gc.get('empty_counts', 228))
        self.tol = int(gc.get('tolerance_counts', 1))
        self.settle_s = float(gc.get('settle_s', 1.0))

        # Optional: save the wrist-camera view at EACH grasp close, labelled with the gripper count
        # (in the filename and drawn on the image), for correlating the visual grasp state with the
        # count bands. Written to <data_dir>/<capture_subdir>/<timestamp>/.
        self.capture_images = bool(gc.get('capture_images', True))
        self._data_root = cfg.get('data_dir', 'data')
        self._capture_subdir = gc.get('capture_subdir', 'grasp_images')
        self._capture_dir = None
        self._capture_seq = 0

    def _capture_grasp(self, camera, count, tag, result):
        """Save the wrist-camera frame labelled with the gripper `count` (filename + on-image).
        Best-effort; a no-op if capture is off or there is no camera."""
        if not self.capture_images or camera is None or getattr(camera, 'dry_run', False):
            return
        try:
            import os
            import cv2
            from datetime import datetime
            if self._capture_dir is None:
                self._capture_dir = os.path.join(self._data_root, self._capture_subdir,
                                                 datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(self._capture_dir, exist_ok=True)
            img = camera.capture().color.copy()
            cv2.putText(img, f'{count} counts  {tag}  {result}', (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            path = os.path.join(self._capture_dir,
                                f'grasp_{self._capture_seq:03d}_{tag}_{count}counts_{result}.png')
            cv2.imwrite(path, img)
            self._capture_seq += 1
            log.info('  saved grasp image %s', path)
        except Exception as exc:                       # noqa: BLE001 -- capture is best-effort
            log.warning('  could not save grasp image: %s', exc)

    def grasp_with_recovery(self, robot, geom, check, camera=None):
        """Close, classify by the finger COUNT, and on a miss reseat in the DIRECTED way for that
        count and retry -- up to max_tries. Returns 'ok' | 'missed' | 'abort'. Leaves the (possibly
        corrected) grasp in geom.T_base_grasp. If `camera` is given and grasp_check.capture_images is
        on, the wrist view is saved (labelled with the count) at each close."""
        import time
        g = robot.gripper
        tries = self.max_tries if self.enabled else 0
        for attempt in range(tries + 1):
            if not g.close('grasp'):
                return 'abort'
            time.sleep(self.settle_s)
            pos = g.position()
            tag = 'grasp' if attempt == 0 else f'reseat{attempt}'

            if self.connector_lo <= pos <= self.connector_hi:      # connector seated -> success
                log.info('Grasp OK: %d counts in the connector band [%d, %d].',
                         pos, self.connector_lo, self.connector_hi)
                self._capture_grasp(camera, pos, tag, 'ok')
                return 'ok'
            if attempt >= tries:                                    # out of retries
                self._capture_grasp(camera, pos, tag, 'missed')
                break

            if self.cable_counts is not None and abs(pos - self.cable_counts) <= self.tol:
                log.warning('Grabbed the CABLE (%d ~ %d) -- open, shift %.1f mm +x toward the '
                            'connector end, retry.', pos, self.cable_counts, self.cable_shift * 1000)
                self._capture_grasp(camera, pos, tag, 'cable')
                delta = translation_matrix([self.cable_shift, 0.0, 0.0])   # +x = toward connector end
                reseat = 'reseat +x (toward connector)'
            elif abs(pos - self.closed_counts) <= self.tol:
                log.warning('EMPTY close (%d ~ closed %d) -- open, drop %.1f mm -z toward the '
                            'object, retry.', pos, self.closed_counts, self.empty_drop * 1000)
                self._capture_grasp(camera, pos, tag, 'empty')
                delta = translation_matrix([0.0, 0.0, -self.empty_drop])   # -z = toward the ground
                reseat = 'reseat -z (toward ground)'
            else:                                                   # not connector/cable/closed
                log.warning('Grasp count %d is not connector/cable/closed -- open + blind retry.', pos)
                self._capture_grasp(camera, pos, tag, 'other')
                delta = None
                reseat = 'blind retry'

            if not g.open('reposition'):
                return 'abort'
            if delta is not None:
                geom.T_base_grasp = geom.T_base_grasp @ delta       # ACCUMULATE the correction
                if not robot.move_fingertip(geom.T_base_grasp, reseat):
                    return 'abort'

        log.error('Grasp not seated in the connector band after %d tries.', self.max_tries)
        return 'missed'


def _compliant_move(robot, adm, guard, T_start_ftip, T_target_ftip, duration,
                    tare_before=True, settle_s=0.5, what='move'):
    """Move the fingertip from its CURRENT pose to T_target under SOFTWARE ADMITTANCE
    (robot/admittance.py) instead of a stiff moveL -- the SAME law the assembly insert uses. The
    fingertip yields to contact (a misplaced cable, the work surface, a cable that resists the lift)
    and springs back toward the reference. Returns True on completion (or an early guard trip =
    contact); leaves the arm OUT of the servo loop.

    CARTESIAN reference: the tool0 pose is slerped start -> target. This is the straight-line tool
    path, but near an IK branch boundary / SINGULARITY the servoL Jacobian inverse spikes the joint
    velocity and the shoulder jerks into the controller's COLLISION-DETECTION protective stop (no
    real contact). Keep the working region away from singularities (mid-range elbow/wrist)."""
    T_t0_ft = robot.T_tool0_fingertip

    def ref(T_ft):
        return T_ft @ inverse(T_t0_ft)                  # fingertip pose -> tool0 servoL reference

    # Tare MID-WARMUP (servo engaged, static) so the guard baseline matches the servo-active reading.
    tare = (lambda: robot.arm.zero_ft(settle=False)) if tare_before else None
    adm.reset()
    adm.warmup(ref(T_start_ftip), tare_fn=tare)         # settle the servo (+ tare) before the guard
    if guard is not None:
        guard.reset()
    log.info('%s under ADMITTANCE (S=%.0f N/m trans, %.0f Nm/rad rot) over %.1fs%s.',
             what, adm.S[0], adm.S[3], duration,
             '' if guard is None
             else f', guarded at {guard.max_force:.0f} N / {guard.max_torque:.1f} Nm')
    try:
        result = adm.ramp(ref(T_start_ftip), ref(T_target_ftip), duration, guard)
        if result == 'seated':
            log.info('  contact reached -- holding here (compliant).')
        else:
            adm.hold(ref(T_target_ftip), settle_s, guard)
        return True
    finally:
        robot.arm.servo_stop()                          # leave the servo loop before the next step


class GraspController:
    """Moves the fingertip in the configured pickup mode -- used for BOTH the grasp DESCENT
    (grasp-align -> grasp) and the LIFT (grasp -> lift), so they always match:

      * 'position'   -- a stiff moveL (default; unchanged behaviour).
      * 'compliance' -- SOFTWARE ADMITTANCE (the same spring-mass-damper law as the assembly
                        insert): the fingertip yields to contact and springs back toward the
                        reference. Parameters mirror assembly.compliance, plus an optional
                        force_guard backstop and the ramp duration.

    Keeping the lift in the SAME mode matters: a compliant grasp leaves the arm holding the cable
    under the admittance law, and a stiff guarded lift then fights the cable's resistance and trips
    the force guard. A compliant lift yields instead.

    Reads the `pickup:` config section. The admittance controller (and optional guard) are built
    lazily on first compliant move, so a position-mode run never touches the servo layer."""

    def __init__(self, cfg):
        p = cfg.section('pickup')
        self.mode = str(p.get('mode', 'position')).lower()
        self.compliance = p.get('compliance', {}) or {}
        self.compliance_enabled = bool(self.compliance.get('enabled', True))
        self.tare_before = bool(self.compliance.get('tare_before', True))
        # NEVER tare at the LIFT by default: the lift starts IN CONTACT (part on the ground), so a
        # tare there bakes the ground reaction into the baseline -- the moment the part lifts off,
        # that baseline reads as a PHANTOM DOWNWARD force and the admittance drives the tool back
        # into the ground (jerky scraping, force climbing to the pendant's safety stop). The
        # descent's FREE-SPACE tare (at grasp-align) stays valid for the lift.
        self.tare_before_lift = bool(p.get('tare_before_lift', False))
        self.descent_time_s = float(p.get('descent_time_s', 2.0))
        # SPEED-based pacing (preferred): when either limit is set, the ramp duration is derived
        # from the ACTUAL distance -- so the touchdown speed no longer changes silently when
        # approach_distance_m does. The pickup keys override; absent, the GLOBAL cartesian limits
        # (speed.max_cartesian_translation_mm_s / _rotation_deg_s) pace the descent and lift like
        # every other pre-assembly motion. descent_time_s is the legacy fallback when neither the
        # pickup nor the global limits are set.
        spd = cfg.section('speed')
        self.descent_v_mm_s = p.get('descent_translation_mm_s',
                                    spd.get('max_cartesian_translation_mm_s'))
        self.descent_w_deg_s = p.get('descent_rotation_deg_s',
                                     spd.get('max_cartesian_rotation_deg_s'))
        # Per-phase scaling of the global limits (speed.phase_scale): the descent runs at the
        # 'pickup' scale, the lift at the 'lift' scale (typically slower -- the cable is in hand).
        scales = spd.get('phase_scale', {}) or {}
        self.pickup_scale = float(scales.get('pickup', 1.0))
        self.lift_scale = float(scales.get('lift', 1.0))
        self.settle_s = float(p.get('settle_s', 0.5))
        self._guard_cfg = p.get('force_guard', {}) or {}
        self._adm = None
        self._guard = None

    @property
    def compliant(self):
        return self.mode == 'compliance' and self.compliance_enabled

    def _lazy_build(self, robot):
        if self._adm is None:
            from ..robot import AdmittanceController, ForceGuard
            self._adm = AdmittanceController(robot.arm, self.compliance)
            if self._guard_cfg.get('enabled', False):
                self._guard = ForceGuard(robot.arm, self._guard_cfg)

    def _duration(self, T_from, T_to, scale=1.0):
        """Ramp seconds for a compliant move: distance / the configured SPEED limits (whichever of
        translation/rotation needs longer), or the legacy fixed descent_time_s when no speed is
        configured. `scale` is the phase's speed.phase_scale factor (0.5 = half speed = double
        time). Floored at 0.1 s so a zero-length move still ramps sanely."""
        scale = max(float(scale), 1e-6)
        if self.descent_v_mm_s is None and self.descent_w_deg_s is None:
            return self.descent_time_s / scale
        lin_m, ang_rad = pose_error(T_from, T_to)
        v = float(self.descent_v_mm_s or 0.0) * scale
        w = float(self.descent_w_deg_s or 0.0) * scale
        t_lin = (lin_m * 1000.0 / v) if v > 0 else 0.0
        t_ang = (np.degrees(ang_rad) / w) if w > 0 else 0.0
        return max(t_lin, t_ang, 0.1)

    def _to(self, robot, T_target, label, what, position_guard=None, tare=None, scale=1.0):
        """Move the fingertip to T_target in the configured mode. In POSITION mode, `position_guard`
        (a callable taking the move thunk) runs it force-guarded; in COMPLIANCE mode the guard is
        ignored -- the admittance bounds the contact force itself (a stiff guarded move is exactly
        what trips on the cable's resistance). `tare` overrides whether to re-tare mid-warmup
        (default: the descent's tare_before) -- a move that STARTS IN CONTACT must not tare.
        `scale` is the phase's speed.phase_scale factor."""
        if self.compliant:
            self._lazy_build(robot)
            T_start = robot.fingertip()
            return _compliant_move(robot, self._adm, self._guard, T_start, T_target,
                                   self._duration(T_start, T_target, scale),
                                   self.tare_before if tare is None else tare,
                                   self.settle_s, what)
        if position_guard is not None:
            return position_guard(lambda: robot.move_fingertip(T_target, label))
        return robot.move_fingertip(T_target, label)

    def descend(self, robot, geom, label='grasp'):
        """Move to the grasp pose (from wherever the arm is -- the grasp-align pose). Tares in
        FREE SPACE (at grasp-align), which is the baseline the lift keeps. Runs at the 'pickup'
        phase scale."""
        return self._to(robot, geom.T_base_grasp, label, 'Grasp descent', scale=self.pickup_scale)

    def lift(self, robot, geom, label='lift', position_guard=None):
        """Lift to geom.lift() in the SAME mode as the descent, at the 'lift' phase scale.
        `position_guard` guards the position-mode lift only. Does NOT re-tare (tare_before_lift,
        default off): the lift starts in contact, and taring there turns the ground reaction into
        a phantom downward force at liftoff -- the arm would chase it back into the ground."""
        return self._to(robot, geom.lift(), label, 'Lift', position_guard=position_guard,
                        tare=self.tare_before_lift, scale=self.lift_scale)


class GraspImageRecorder:
    """Save wrist-camera images at a fixed RATE for the DURATION of a grasp, in a background thread,
    so the whole approach -> close -> reseat is captured (not just the close events that
    GraspRecovery labels with the count). Enabled by grasp_check.capture_images (default TRUE) at
    grasp_check.capture_rate_hz (default 1 Hz); written to <data_dir>/<capture_subdir>/<timestamp>/.

        with recorder.recording(camera):
            ... grasp motions ...
    """

    def __init__(self, cfg):
        gc = cfg.section('grasp_check')
        self.enabled = bool(gc.get('capture_images', True))
        self.rate_hz = max(0.1, float(gc.get('capture_rate_hz', 1.0)))
        self._data_root = cfg.get('data_dir', 'data')
        self._subdir = gc.get('capture_subdir', 'grasp_images')
        self._dir = None
        self._seq = 0

    def recording(self, camera):
        """Context manager: record at self.rate_hz for the length of the `with` block."""
        return _GraspRecording(self, camera)

    def _capture(self, camera):
        """Save one wrist frame; best-effort (a failure never interrupts the grasp)."""
        try:
            import os
            import cv2
            from datetime import datetime
            if self._dir is None:
                self._dir = os.path.join(self._data_root, self._subdir,
                                         datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(self._dir, exist_ok=True)
            img = camera.capture().color.copy()
            cv2.imwrite(os.path.join(self._dir, f'grasp_{self._seq:04d}.png'), img)
            self._seq += 1
        except Exception as exc:                       # noqa: BLE001 -- capture is best-effort
            log.warning('  grasp image capture failed: %s', exc)


class _GraspRecording:
    """Runs GraspImageRecorder._capture at the recorder's rate in a daemon thread for the length of
    the `with` block. No-op if capture is off / no camera / dry-run."""

    def __init__(self, rec, camera):
        self._rec = rec
        self._camera = camera
        self._stop = None
        self._thread = None

    def __enter__(self):
        r = self._rec
        if not r.enabled or self._camera is None or getattr(self._camera, 'dry_run', False):
            return self
        import threading
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info('  recording grasp images at %.1f Hz.', r.rate_hz)
        return self

    def _loop(self):
        period = 1.0 / self._rec.rate_hz
        while not self._stop.is_set():
            self._rec._capture(self._camera)
            self._stop.wait(period)

    def __exit__(self, *exc):
        if self._stop is not None:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        return False


def log_grasp_delta(robot, T_base_grasp, label):
    """Report the fingertip's pose vs the grasp target -- the end-to-end error of the whole
    perception -> IK -> motion chain, in the units that matter (mm at the fingers).

    A report, not a gate: it prints before the grasp's confirm prompt so a bad perception estimate
    can be vetoed before the arm commits, but it never blocks on its own."""
    T_cur = robot.fingertip()
    log.info('[%s] fingertip vs grasp target: %s', label, fmt_delta(T_cur, T_base_grasp))
    return True
