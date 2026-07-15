"""Cable pick-and-place -- port of ur_cable_pick_place_demo.

Scan the cable from several views so SAM3's neck detections fuse into a 3D connector pose, then
grasp at that pose and place. The grasp check (in counts) catches a cable that did not seat in the
fingertip groove and retries the whole scan->grasp sequence.

    scan -> estimate connector -> return home -> grasp-align -> grasp -> close -> [check]
      -> lift -> pre-place -> place -> open -> retreat -> home
"""

from .. import log as urlog
from ..log import StepRunner
from ..skills import reset
from ..skills.pick import GraspCheck, GraspGeometry, log_grasp_delta
from ..transforms import from_cfg, inverse
from ._cable import build_scanner, make_confirm
from ._runner import run_app

log = urlog.get('cable-pick-place')


def _attempt(cfg, robot, scanner, geom, check, confirm):
    """One scan->grasp attempt. Returns 'ok' | 'missed' | 'empty' | 'abort'."""
    scanner.estimator.reset()
    q_home = robot.arm.q()
    T_conn_grasp = from_cfg(cfg.section('connector_grasp'))

    if not robot.gripper.open('open'):
        return 'abort'

    T_conn = scanner.scan(confirm=confirm)
    if T_conn is None:
        return 'abort'
    geom.T_base_grasp = T_conn @ T_conn_grasp
    log.info('Connector origin %s, grasp target set.', T_conn[:3, 3].round(3))

    runner = StepRunner(log, confirm=confirm is not None)
    steps = [
        ('return to initial pose (post-scan)', lambda: robot.arm.move_j(q_home, label='home')),
        ('move to grasp-align', lambda: robot.move_fingertip(geom.pre_grasp(), 'grasp-align')),
        ('report pre-grasp delta', lambda: log_grasp_delta(robot, geom.T_base_grasp, 'pre-grasp')),
        ('move to grasp', lambda: robot.move_fingertip(geom.T_base_grasp, 'grasp')),
        ('report at-grasp delta', lambda: log_grasp_delta(robot, geom.T_base_grasp, 'at-grasp')),
        ('close gripper (grasp)', robot.gripper.close),
    ]
    if not runner.run(steps):
        return 'abort'
    return check.evaluate(robot.gripper)


def build_and_run(cfg, robot, camera, args):
    scanner, _detector, _estimator = build_scanner(cfg, robot, camera)
    geom = GraspGeometry(cfg)
    check = GraspCheck(cfg)
    confirm = make_confirm(cfg)

    # RESET at the start: open the gripper and go to the defined HOME pose under admittance, so the
    # run always begins from the same known configuration. q_home is then that home config.
    if not reset.reset_robot(robot, cfg, confirm, 'start reset'):
        return False
    q_home = robot.arm.q()

    # Pick, with grasp-check retry.
    attempt = 0
    while True:
        result = _attempt(cfg, robot, scanner, geom, check, confirm)
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

    # Place, release, then RESET at the end (open gripper + go home under admittance).
    runner = StepRunner(log, confirm=confirm is not None)
    return runner.run([
        ('lift', lambda: robot.move_fingertip(geom.lift(), 'lift')),
        ('move to pre-place', lambda: robot.move_fingertip(geom.pre_place(), 'pre-place')),
        ('move to place', lambda: robot.move_fingertip(geom.place(), 'place')),
        ('open gripper (release)', robot.gripper.open),
        ('retreat', lambda: robot.move_fingertip(geom.pre_place(), 'retreat')),
        ('end reset (open + home under admittance)',
         lambda: reset.reset_robot(robot, cfg, None, 'end reset')),
    ])


def main():
    run_app('Cable pick-and-place (multi-view SAM3)', 'cable_pick_place', build_and_run,
            needs_camera=True)


if __name__ == '__main__':
    main()
