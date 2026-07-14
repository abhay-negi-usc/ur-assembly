"""Cartesian jog demo -- exercise each tool0 axis about a home pose.

Port of ur_cartesian_demo. For each motion frame (world, then tool0), nudge the tool +/- along
each of x/y/z and +/- about roll/pitch/yaw, returning to home between every nudge. Every target
is relative to the captured HOME pose, never to the previous one, so errors do not accumulate.
"""

import numpy as np

from .. import log as urlog
from ..transforms import inverse, xyzrpy_to_matrix
from ._runner import run_app

log = urlog.get('cartesian')

# One nonzero component each: +x, -x, +y, -y, +z, -z, then the three rotations both ways.
_MOVES = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1),
          (3, +1), (3, -1), (4, +1), (4, -1), (5, +1), (5, -1)]


def _offset(T_home, frame_R, axis, sign, lin, ang):
    """Home pose nudged along/about `axis`, with the axis taken from `frame_R` (the motion frame's
    rotation in base). Translation and rotation both use the same frame, so 'world' gives base-axis
    motion and 'tool0' gives tool-axis motion."""
    delta = np.zeros(6)
    delta[axis] = sign * (lin if axis < 3 else ang)
    T_delta = xyzrpy_to_matrix(delta[:3], delta[3:])          # in the motion frame
    T_frame = np.eye(4)
    T_frame[:3, :3] = frame_R
    # Rotate the delta into base, apply about the home pose's origin (position offset in base,
    # orientation composed): T = T_frame @ T_delta @ inv(T_frame) @ T_home, but keeping the home
    # translation fixed for a rotation and moving it for a translation falls out of doing the
    # translation in the frame and the rotation as a similarity transform.
    R_similar = T_frame @ T_delta @ inverse(T_frame)
    T = np.eye(4)
    T[:3, :3] = R_similar[:3, :3] @ T_home[:3, :3]
    T[:3, 3] = T_home[:3, 3] + frame_R @ delta[:3]
    return T


def build_and_run(cfg, robot, camera, args):
    lin = float(cfg.get('linear_step_m', 0.03))
    ang = np.radians(float(cfg.get('angular_step_deg', 30.0)))
    frames = cfg.get('motion_frames', ['world', 'tool0'])
    confirm = None if cfg.get('confirm_each_step', True) is False else \
        (lambda label: input(f'[{label}] Enter to proceed (q to abort): ').strip().lower() != 'q')

    T_home = robot.tool0()
    q_home = robot.arm.q()
    log.info('Home captured. Jogging %d frame(s): %s', len(frames), frames)

    for frame in frames:
        # 'world' == base axes here (identity); 'tool0' uses the home orientation.
        frame_R = np.eye(3) if frame == 'world' else T_home[:3, :3]
        log.info('--- motion frame: %s ---', frame)
        for axis, sign in _MOVES:
            comp = 'xyzRPY'[axis]
            label = f'{frame} {comp}{"+" if sign > 0 else "-"}'
            if confirm and not confirm(label):
                log.info('Aborted by the user.')
                return True
            target = _offset(T_home, frame_R, axis, sign, lin, ang)
            if not robot.move_tool0(target, label):
                log.warning('%s: unreachable -- skipping.', label)
                continue
            robot.arm.move_j(q_home, label='recenter')       # exact joint return, no IK drift
    log.info('Cartesian jog complete.')
    return True


def main():
    run_app('Cartesian jog demo (UR10e)', 'cartesian', build_and_run, with_gripper=False)


if __name__ == '__main__':
    main()
