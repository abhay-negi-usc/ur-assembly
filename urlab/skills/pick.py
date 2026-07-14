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


def log_grasp_delta(robot, T_base_grasp, label):
    """Report the fingertip's pose vs the grasp target -- the end-to-end error of the whole
    perception -> IK -> motion chain, in the units that matter (mm at the fingers).

    A report, not a gate: it prints before the grasp's confirm prompt so a bad perception estimate
    can be vetoed before the arm commits, but it never blocks on its own."""
    T_cur = robot.fingertip()
    log.info('[%s] fingertip vs grasp target: %s', label, fmt_delta(T_cur, T_base_grasp))
    return True
