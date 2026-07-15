"""Cable pick-and-assemble -- port of ur_cable_pick_assemble_demo.

The PICK is the exact cable-pick-place pipeline. What replaces "place" is a pluggable assembly:
stand-off, compliant chunked insertion (force mode), release, retract. Only 'kinematic' is
implemented (the target pose is given outright); 'vision' fails loudly.

    [PICK] -> lift -> stand-off -> enter compliance -> insert (chunked, guarded)
      -> open (release) -> end compliance -> retract (multi-step) -> home
"""

from .. import log as urlog
from ..log import StepRunner
from ..robot import AdmittanceController, ForceGuard
from ..skills import insert as ins
from ..skills.pick import GraspCheck, GraspGeometry, log_grasp_delta
from ..transforms import from_cfg
from ._cable import build_scanner, make_confirm
from ._runner import run_app

log = urlog.get('cable-assemble')


def _pick(cfg, robot, scanner, geom, check, confirm):
    """The cable pick, returning 'ok' | 'missed' | 'empty' | 'abort'."""
    scanner.estimator.reset()
    q_home = robot.arm.q()
    T_conn_grasp = from_cfg(cfg.section('connector_grasp'))
    if not robot.gripper.open('open'):
        return 'abort'
    T_conn = scanner.scan(confirm=confirm)
    if T_conn is None:
        return 'abort'
    geom.T_base_grasp = T_conn @ T_conn_grasp

    runner = StepRunner(log, confirm=confirm is not None)
    if not runner.run([
        ('return to initial pose (post-scan)', lambda: robot.arm.move_j(q_home, label='home')),
        ('move to grasp-align', lambda: robot.move_fingertip(geom.pre_grasp(), 'grasp-align')),
        ('report pre-grasp delta', lambda: log_grasp_delta(robot, geom.T_base_grasp, 'pre-grasp')),
        ('move to grasp', lambda: robot.move_fingertip(geom.T_base_grasp, 'grasp')),
        ('close gripper (grasp)', robot.gripper.close),
    ]):
        return 'abort'
    return check.evaluate(robot.gripper)


def build_and_run(cfg, robot, camera, args):
    ic = ins.InsertConfig(cfg)
    if ic.method != 'kinematic':
        log.error("assembly.method %r is not implemented (only 'kinematic'); aborting.", ic.method)
        return False

    scanner, _detector, _estimator = build_scanner(cfg, robot, camera)
    geom = GraspGeometry(cfg)
    check = GraspCheck(cfg)
    guard = ForceGuard(robot.arm, cfg.get_path('assembly.force_guard', {}))
    confirm = make_confirm(cfg)
    q_home = robot.arm.q()

    # 1. PICK, with grasp-check retry.
    attempt = 0
    while True:
        result = _pick(cfg, robot, scanner, geom, check, confirm)
        if result == 'ok':
            break
        if result == 'abort':
            return False
        if attempt >= check.max_retries:
            log.error('Grasp failed on all %d attempts; aborting.', check.max_retries + 1)
            return False
        attempt += 1
        log.warning('Grasp %s -- recovering (attempt %d/%d).',
                    result, attempt + 1, check.max_retries + 1)
        if not (robot.gripper.open('drop') and robot.arm.move_j(q_home, label='home')):
            return False

    # 2. Reduce the target to a fingertip pose and derive the stand-off.
    T_target = ins.fingertip_target(ic, from_cfg(cfg.section('connector_grasp')),
                                    robot.T_tool0_fingertip)
    T_standoff = ins.standoff_of(ic, T_target)
    log.info('Kinematic assembly: fingertip target %s, stand-off %s.',
             T_target[:3, 3].round(4), T_standoff[:3, 3].round(4))

    # 3. Assemble. The force guard is armed over the free-space moves (a trip = unexpected
    #    collision); the insertion reads the guard itself (a trip there = SEATED). servo_stop in the
    #    finally guarantees the arm is left out of the servo loop.
    #    Compliance is a SOFTWARE ADMITTANCE law (finite restoring stiffness) -- see the config's
    #    compliance block and robot/admittance.py. insert_compliant runs it; if compliance is
    #    disabled it falls back to a stiff position insert.
    adm = AdmittanceController(robot.arm, cfg.get_path('assembly.compliance', {}))

    def do_insert():
        if ic.compliance_enabled:
            return ins.insert_compliant(robot, adm, guard, ic, T_standoff, T_target)
        return ins.insert_chunked(robot, guard, ic, T_standoff, T_target, confirm)

    runner = StepRunner(log, confirm=confirm is not None)
    ok = False
    try:
        ok = runner.run([
            ('lift', lambda: _guarded(robot, guard, lambda: robot.move_fingertip(geom.lift(), 'lift'))),
            ('move to stand-off',
             lambda: _guarded(robot, guard,
                              lambda: robot.move_fingertip(T_standoff, 'stand-off'))),
            ('insert (admittance)', do_insert),
            ('open gripper (release)', robot.gripper.open),
            ('retract', lambda: ins.retract(robot, ic)),
        ])
    finally:
        robot.arm.servo_stop()          # leave the servo loop no matter how the insertion ended
    if not ok:
        return False

    return robot.arm.move_j(q_home, label='return home')


def _guarded(robot, guard, move_fn):
    """Run a move with the force guard armed as a canceller (a trip = unexpected collision)."""
    guard.reset()
    robot.arm.add_guard(guard)
    try:
        ok = move_fn()
    finally:
        robot.arm.clear_guards()
    if not ok and guard.tripped_by:
        log.error('Force guard tripped during a free-space move (%s) -- hit something unexpected.',
                  guard.tripped_by)
    return ok


def main():
    run_app('Cable pick-and-assemble (kinematic + compliant insert)', 'cable_pick_assemble',
            build_and_run, needs_camera=True)


if __name__ == '__main__':
    main()
