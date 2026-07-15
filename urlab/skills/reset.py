"""Reset behavior -- open the gripper and return to a defined HOME joint pose under admittance.

Run at the START and END of a task. Going home under admittance (a compliant move, see
robot/admittance.py) rather than a stiff moveJ means that if the arm bumps something on the way --
a fixture, the part, a hand -- it yields and the force guard stops it, instead of forcing through.
The home pose is a fixed joint configuration (not the arm's start pose), so a reset always lands
in the same known place regardless of where the demo left it.
"""

import numpy as np

from .. import log as urlog
from ..robot import ForceGuard

log = urlog.get('reset')


def home_joints(cfg):
    """The configured home joint vector (rad), from reset.home_joints_deg."""
    deg = cfg.get_path('reset.home_joints_deg', [90.0, -135.0, -135.0, 0.0, 90.0, 0.0])
    return list(np.radians(deg))


def go_home(robot, cfg, guard=None):
    """Return to the home joint config under POSITION control (moveJ) with the force guard armed.

    NOT Cartesian servoL admittance: a large, arbitrary joint traverse streamed as Cartesian
    servoL can flip IK branches (elbow/wrist), which the arm executes as a fast lunge in the wrong
    direction -- and servoL ignores the velocity caps, so they can't rein it in. moveJ plans a
    proper JOINT trajectory (no branch flips), honours max_joint/cartesian_velocity, and the armed
    guard STOPS it on contact -- which is the actual safety intent of a compliant home. (Cartesian
    admittance stays for the insertion, a small move near the target where servoL is appropriate.)

    Returns True on a clean home, False if the guard tripped (hit something) or the move failed."""
    q_home = home_joints(cfg)
    log.info('Returning home to %s deg (position control, contact-guarded at %.0f N).',
             list(np.round(np.degrees(q_home)).astype(int)),
             float(cfg.get_path('reset.max_force_n', 30.0)))
    if robot.arm.dry_run:
        return robot.arm.move_j(q_home, label='home (dry-run)')

    robot.arm.zero_ft()                              # tare so the guard measures contact only
    if guard is None:
        guard = ForceGuard(robot.arm, {'max_force_n': cfg.get_path('reset.max_force_n', 30.0)})
    guard.reset()
    robot.arm.add_guard(guard)                       # armed -> moveJ runs async and polls the guard
    try:
        ok = robot.arm.move_j(q_home, label='home')
    finally:
        robot.arm.clear_guards()
    if not ok and guard.tripped_by:
        log.error('Home move hit something (guard tripped: %s) -- stopped. Clear the path and '
                  'retry.', guard.tripped_by)
    return ok


def reset_robot(robot, cfg, confirm=None, label='reset'):
    """Open the gripper -> go home under admittance (taring mid-move once the servo is engaged).
    The whole-run bookend."""
    if confirm and not confirm(f'{label}: open gripper -> tare -> go home (admittance)'):
        return False
    if robot.gripper is not None and not robot.gripper.open('reset: open gripper'):
        return False
    return go_home(robot, cfg)
