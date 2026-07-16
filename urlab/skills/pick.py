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
    """The 'cable not seated in the fingertip groove' detector, in counts."""

    def __init__(self, cfg):
        gc = cfg.section('grasp_check')
        self.enabled = bool(gc.get('enabled', True))
        self.closed_counts = int(gc.get('closed_counts', 228))
        self.tolerance = int(gc.get('tolerance_counts', 1))
        self.detect_empty = bool(gc.get('detect_empty', False))
        self.max_retries = int(gc.get('max_retries', 2))
        self.settle_s = float(gc.get('settle_s', 1.0))

    def evaluate(self, gripper):
        """'ok' | 'missed' | 'empty'. See gripper.grasp_result for the (inverted) logic: reaching
        the target is SUCCESS, stalling short is the failure."""
        if not self.enabled:
            return 'ok'
        import time
        time.sleep(self.settle_s)
        return gripper.grasp_result(self.closed_counts, self.tolerance, self.detect_empty)


class GraspRecovery:
    """Failed-grasp recovery for the cable pick. When the fingers stall short of `closed_counts`
    the cable is between them but NOT seated in the fingertip groove; the stalled COUNT tells the
    two failure modes apart (higher count = more closed):

      * ~`closed_counts` (228)  -- cable in the groove (or empty): SUCCESS, no recovery.
      * ~`faces_counts` (220)   -- cable on the flat parallel faces, propping the fingers widest
                                   apart (the count FLOOR): move the gripper TOWARD the cable so it
                                   drops off the flats into the groove.
      * between the two         -- cable pinched at the fingertip TIPS: move the gripper AWAY.

    Recovery is: (1) a BLIND retry -- loose grip then full close, no arm motion, which alone
    reseats a cable that was merely nipped; then (2) up to `max_tries` corrective iterations that
    classify the mode, open to the loose grip, nudge the arm `increment_m` (default 0.5 mm) in the
    corrective direction, and close again. The correction is along the grasp-frame approach axis
    ('toward the cable' = deeper along the approach) and ACCUMULATES into the grasp target, so a
    successful reseat leaves the corrected pose in geom.T_base_grasp for the lift/place.

    The two failure bands overlap in practice (a tips reading sits just above the faces floor), so
    `faces_band_counts` sets the split -- tune it on hardware; a misclassification just costs one
    iteration."""

    def __init__(self, cfg):
        rc = (cfg.section('grasp_check').get('recovery', {}) or {})
        self.enabled = bool(rc.get('enabled', True))
        self.max_tries = int(rc.get('max_tries', 5))
        self.loose_counts = int(rc.get('loose_counts', 215))
        self.increment_m = float(rc.get('increment_m', 0.0005))
        self.faces_counts = int(rc.get('faces_counts', 220))
        self.faces_band = int(rc.get('faces_band_counts', 1))
        self._toward_cfg = rc.get('toward_cable_axis', None)   # grasp frame; else -approach_axis

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

    def grasp_with_recovery(self, robot, geom, check):
        """Close, grasp-check, and on a MISS run the blind retry + corrective loop. Returns
        'ok' | 'missed' | 'empty' | 'abort'. Leaves the (possibly corrected) grasp in
        geom.T_base_grasp."""
        g = robot.gripper
        if not g.close('grasp'):
            return 'abort'
        result = check.evaluate(g)
        if result != 'missed' or not self.enabled:
            return result

        # 1. Blind retry: loose grip, then full close -- no arm motion.
        log.warning('Grasp missed at %d counts -- blind retry (loose grip -> close).', g.position())
        if not (g.go_to(self.loose_counts, 'loose grip') and g.close('grasp')):
            return 'abort'
        result = check.evaluate(g)
        if result != 'missed':
            return result

        # 2. Mode-directed corrective loop.
        toward = self._toward(geom)
        for i in range(self.max_tries):
            pos = g.position()
            mode = self._classify(pos)
            direction = toward if mode == 'faces' else -toward
            geom.T_base_grasp = geom.T_base_grasp @ translation_matrix(direction * self.increment_m)
            log.warning('Recovery %d/%d: %s mode (%d counts) -- reseat %.1f mm %s the cable.',
                        i + 1, self.max_tries, mode, pos, self.increment_m * 1000,
                        'toward' if mode == 'faces' else 'away from')
            if not (g.go_to(self.loose_counts, 'loose grip')
                    and robot.move_fingertip(geom.T_base_grasp, f'reseat ({mode})')
                    and g.close('grasp')):
                return 'abort'
            result = check.evaluate(g)
            if result != 'missed':
                return result

        log.error('Grasp still MISSED after the blind retry and %d corrective tries.',
                  self.max_tries)
        return 'missed'


def log_grasp_delta(robot, T_base_grasp, label):
    """Report the fingertip's pose vs the grasp target -- the end-to-end error of the whole
    perception -> IK -> motion chain, in the units that matter (mm at the fingers).

    A report, not a gate: it prints before the grasp's confirm prompt so a bad perception estimate
    can be vetoed before the arm commits, but it never blocks on its own."""
    T_cur = robot.fingertip()
    log.info('[%s] fingertip vs grasp target: %s', label, fmt_delta(T_cur, T_base_grasp))
    return True
