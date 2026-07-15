"""Reset behavior -- open the gripper and return to a defined HOME joint pose under admittance.

Run at the START and END of a task. Going home under admittance (a compliant move, see
robot/admittance.py) rather than a stiff moveJ means that if the arm bumps something on the way --
a fixture, the part, a hand -- it yields and the force guard stops it, instead of forcing through.
The home pose is a fixed joint configuration (not the arm's start pose), so a reset always lands
in the same known place regardless of where the demo left it.
"""

import numpy as np

from .. import log as urlog
from ..robot import AdmittanceController, ForceGuard

log = urlog.get('reset')


def home_joints(cfg):
    """The configured home joint vector (rad), from reset.home_joints_deg."""
    deg = cfg.get_path('reset.home_joints_deg', [90.0, -135.0, -135.0, 0.0, 90.0, 0.0])
    return list(np.radians(deg))


def go_home(robot, cfg, guard=None):
    """Return to the home joint config under admittance. The F/T is tared MID-WARMUP (once the
    servo is engaged and static) so the guard's baseline matches the reading it will actually see
    -- taring while idle leaves the tool-weight offset that only appears under active control.
    Returns True on a clean home, False if the guard tripped (hit something) or a move failed."""
    q_home = home_joints(cfg)
    log.info('Going home under admittance to %s deg.',
             list(np.round(np.degrees(q_home)).astype(int)))
    if robot.arm.dry_run:
        return robot.arm.move_j(q_home, label='home (dry-run)')

    adm = AdmittanceController(robot.arm, cfg.get_path('reset.compliance', {}))
    if guard is None:
        guard = ForceGuard(robot.arm, {'max_force_n': cfg.get_path('reset.max_force_n', 30.0)})
    guard.reset()

    def tare():
        log.info('Taring the F/T sensor (servo engaged, before the home travel).')
        robot.arm.zero_ft(settle=False)              # settle=False: don't block the servo loop

    duration = float(cfg.get_path('reset.home_duration_s', 6.0))
    try:
        result = adm.ramp_joint_path(robot.arm.q(), q_home, duration, guard, tare_fn=tare)
    finally:
        robot.arm.servo_stop()
    if result == 'seated':
        log.error('Home move hit something (guard tripped: %s) -- stopped. Clear the path and '
                  'retry.', guard.tripped_by)
        return False
    return True


def reset_robot(robot, cfg, confirm=None, label='reset'):
    """Open the gripper -> go home under admittance (taring mid-move once the servo is engaged).
    The whole-run bookend."""
    if confirm and not confirm(f'{label}: open gripper -> tare -> go home (admittance)'):
        return False
    if robot.gripper is not None and not robot.gripper.open('reset: open gripper'):
        return False
    return go_home(robot, cfg)
