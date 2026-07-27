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
from ..transforms import fmt_delta, inverse, translation_matrix

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
        self.groove_counts = int(gc.get('groove_counts', 225))     # SUCCESS: cable in the groove
        self.empty_counts = int(gc.get('empty_counts', 228))       # full closure on nothing
        self.faces_max_counts = int(gc.get('faces_max_counts', 223))  # <= this: cable on the faces
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
                                    self.tolerance, self.detect_empty)


class GraspRecovery:
    """Failed-grasp recovery for the cable pick. When the grasp check returns 'missed' the cable is
    between the fingers but NOT seated in the groove; the stalled COUNT tells the failure mode
    (see GraspCheck for the bands -- more obstruction = LESS closed):

      * faces (<= faces_counts + band, ~220-223) -- cable on the flat parallel faces, propping the
                                   fingers open: move the gripper AWAY from the cable so it settles
                                   off the flats into the groove (success ~225).
      * tips (higher, but short of the groove) -- cable pinched at the fingertip TIPS: move the
                                   gripper TOWARD it. The CURRENT fingertips do not show this (a miss
                                   is always 'faces'); kept configurable for other fingertips.

    Recovery is: (1) a BLIND retry -- loose grip then full close, no arm motion, which alone
    reseats a cable that was merely nipped; then (2) up to `max_tries` corrective iterations that
    classify the mode, open to the loose grip, nudge the arm `increment_m` (default 0.5 mm) in the
    corrective direction, and close again. The correction is along the grasp-frame approach axis
    ('toward the cable' = deeper along the approach) and ACCUMULATES into the grasp target, so a
    successful reseat leaves the corrected pose in geom.T_base_grasp for the lift/place.
    `faces_band_counts` sets the faces/tips split -- tune it on hardware."""

    def __init__(self, cfg):
        gc = cfg.section('grasp_check')
        rc = (gc.get('recovery', {}) or {})
        self.enabled = bool(rc.get('enabled', True))
        self.max_tries = int(rc.get('max_tries', 5))
        self.loose_counts = int(rc.get('loose_counts', 215))
        self.increment_m = float(rc.get('increment_m', 0.0005))
        self.faces_counts = int(rc.get('faces_counts', 220))
        self.faces_band = int(rc.get('faces_band_counts', 1))
        self._toward_cfg = rc.get('toward_cable_axis', None)   # grasp frame; else -approach_axis

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

    def _toward(self, geom):
        """Unit 'toward the cable' direction in the GRASP frame: the config override, else the
        negated approach axis (continuing the approach = deeper onto the cable)."""
        v = (np.asarray(self._toward_cfg, dtype=float) if self._toward_cfg is not None
             else -np.asarray(geom.approach_axis, dtype=float))
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])

    def _classify(self, pos):
        """'faces' (at/near the count floor) | 'tips' (above it, but short of success)."""
        return 'faces' if pos <= self.faces_counts + self.faces_band else 'tips'

    def grasp_with_recovery(self, robot, geom, check, camera=None):
        """Close, grasp-check, and on a MISS run the blind retry + corrective loop. Returns
        'ok' | 'missed' | 'empty' | 'abort'. Leaves the (possibly corrected) grasp in
        geom.T_base_grasp. If `camera` is given and grasp_check.capture_images is on, the wrist view
        is saved (labelled with the gripper count) at each close."""
        g = robot.gripper
        if not g.close('grasp'):
            return 'abort'
        result = check.evaluate(g)
        self._capture_grasp(camera, g.position(), 'close', result)
        if result != 'missed' or not self.enabled:
            return result

        # 1. Blind retry: loose grip, then full close -- no arm motion.
        log.warning('Grasp missed at %d counts -- blind retry (loose grip -> close).', g.position())
        if not (g.go_to(self.loose_counts, 'loose grip') and g.close('grasp')):
            return 'abort'
        result = check.evaluate(g)
        self._capture_grasp(camera, g.position(), 'blind', result)
        if result != 'missed':
            return result

        # 2. Mode-directed corrective loop.
        toward = self._toward(geom)
        for i in range(self.max_tries):
            pos = g.position()
            mode = self._classify(pos)
            # faces (cable on the flats) -> move AWAY from the cable so it settles into the groove;
            # tips -> move TOWARD it.
            direction = -toward if mode == 'faces' else toward
            geom.T_base_grasp = geom.T_base_grasp @ translation_matrix(direction * self.increment_m)
            log.warning('Recovery %d/%d: %s mode (%d counts) -- reseat %.1f mm %s the cable.',
                        i + 1, self.max_tries, mode, pos, self.increment_m * 1000,
                        'away from' if mode == 'faces' else 'toward')
            if not (g.go_to(self.loose_counts, 'loose grip')
                    and robot.move_fingertip(geom.T_base_grasp, f'reseat ({mode})')
                    and g.close('grasp')):
                return 'abort'
            result = check.evaluate(g)
            self._capture_grasp(camera, g.position(), f'reseat{i + 1}', result)
            if result != 'missed':
                return result

        log.error('Grasp still MISSED after the blind retry and %d corrective tries.',
                  self.max_tries)
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
        self.descent_time_s = float(p.get('descent_time_s', 2.0))
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

    def _to(self, robot, T_target, label, what, position_guard=None):
        """Move the fingertip to T_target in the configured mode. In POSITION mode, `position_guard`
        (a callable taking the move thunk) runs it force-guarded; in COMPLIANCE mode the guard is
        ignored -- the admittance bounds the contact force itself (a stiff guarded move is exactly
        what trips on the cable's resistance)."""
        if self.compliant:
            self._lazy_build(robot)
            return _compliant_move(robot, self._adm, self._guard, robot.fingertip(), T_target,
                                   self.descent_time_s, self.tare_before, self.settle_s, what)
        if position_guard is not None:
            return position_guard(lambda: robot.move_fingertip(T_target, label))
        return robot.move_fingertip(T_target, label)

    def descend(self, robot, geom, label='grasp'):
        """Move to the grasp pose (from wherever the arm is -- the grasp-align pose)."""
        return self._to(robot, geom.T_base_grasp, label, 'Grasp descent')

    def lift(self, robot, geom, label='lift', position_guard=None):
        """Lift to geom.lift() in the SAME mode as the descent. `position_guard` guards the
        position-mode lift only."""
        return self._to(robot, geom.lift(), label, 'Lift', position_guard=position_guard)


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
