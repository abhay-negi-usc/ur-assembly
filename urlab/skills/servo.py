"""Visual-servo skills -- align, centre, and approach a marker.

Lifted from PickPlace's _servo_to / _align_and_center / _approach_to_standoff and the standalone
visual_servo_node, which were three near-identical copies of the same PBVS loop. One copy now.

The control law: put the CAMERA on the marker's normal at a chosen distance, facing it. Because
we always back-solve the tool0 pose from a FRESH marker observation, the loop re-centres and
re-squares every iteration -- there is no accumulating open-loop error.
"""

import numpy as np

from .. import log as urlog
from ..transforms import inverse, step_toward, xyzrpy_to_matrix

log = urlog.get('servo')


def camera_on_marker(T_base_marker, distance, cam_rpy_in_marker):
    """Desired CAMERA pose: on the marker's +Z normal at `distance`, oriented by
    cam_rpy_in_marker (typically [pi, 0, 0] -- the camera looking back down the normal)."""
    return T_base_marker @ xyzrpy_to_matrix([0.0, 0.0, distance], cam_rpy_in_marker)


def servo_to(robot, T_base_marker, distance, cam_rpy_in_marker, label='servo'):
    """Single move putting the camera centred + squared to the marker at `distance`."""
    return robot.move_camera(camera_on_marker(T_base_marker, distance, cam_rpy_in_marker), label)


def align_and_center(robot, tracker, cam_rpy_in_marker, max_age_s=1.0):
    """Face + centre the marker WITHOUT changing distance (hold the current standoff)."""
    T = tracker.acquire(max_age_s=max_age_s)
    if T is None:
        return False
    d = float(np.linalg.norm(robot.camera()[:3, 3] - T[:3, 3]))
    log.info('Aligning + centring at the current distance %.3f m.', d)
    return servo_to(robot, T, d, cam_rpy_in_marker, 'align+center')


def approach_to_standoff(robot, tracker, standoff_m, step_m, cam_rpy_in_marker,
                         max_iterations=20, max_age_s=1.0, confirm=None):
    """Step the camera toward the marker until it is within `standoff_m`, re-centring each step.

    Re-reading the marker every step matters twice: it corrects the aim as the view improves, and
    it means a marker that drifts is tracked rather than lost."""
    log.info('Approaching to a %.0f cm standoff in %.0f cm steps...', standoff_m * 100, step_m * 100)
    for i in range(max_iterations):
        T = tracker.acquire(max_age_s=max_age_s)
        if T is None:
            return False
        d = float(np.linalg.norm(robot.camera()[:3, 3] - T[:3, 3]))
        log.info('  iter %d: camera-marker distance %.3f m', i, d)
        if d <= standoff_m + 1e-3:
            log.info('Reached standoff (%.3f m).', standoff_m)
            return True
        d_next = max(standoff_m, d - step_m)
        if confirm and not confirm(f'approach step -> {d_next:.3f} m'):
            return False
        if not servo_to(robot, T, d_next, cam_rpy_in_marker, f'approach->{d_next:.2f}m'):
            return False
    log.warning('Hit max approach iterations (%d); proceeding.', max_iterations)
    return True


def visual_servo_to_pose(robot, target_fn, gain, max_lin, max_ang, pos_deadband, ang_deadband,
                         max_iterations=20, label='servo', finish_on_loss=False, confirm=None):
    """Closed-loop PBVS to a tool0 target that is RECOMPUTED from a fresh marker each iteration.

    `target_fn()` returns the current tool0 target (4x4) or None if the marker was lost. On loss:
    either finish open-loop to the last known target (finish_on_loss -- for the final grasp where
    the gripper occludes the marker) or give up."""
    from ..transforms import pose_error
    last = None
    for i in range(max_iterations):
        T_target = target_fn()
        if T_target is None:
            if finish_on_loss and last is not None:
                log.warning('[%s] marker lost; finishing OPEN-LOOP to the last target.', label)
                return robot.move_tool0(last, f'{label} open-loop')
            log.warning('[%s] marker not in view.', label)
            return False
        last = T_target

        T_cur = robot.tool0()
        lin, ang = pose_error(T_cur, T_target)
        log.info('[%s] iter %d: err lin=%.1f mm ang=%.1f deg', label, i, lin * 1000, np.degrees(ang))
        if lin <= pos_deadband and ang <= ang_deadband:
            log.info('[%s] converged.', label)
            return True
        if confirm and not confirm(f'{label} step (err {lin * 1000:.0f} mm)'):
            return False
        if not robot.move_tool0(step_toward(T_cur, T_target, gain, max_lin, max_ang), label):
            return False
    log.warning('[%s] hit max iterations (%d); proceeding.', label, max_iterations)
    return True
