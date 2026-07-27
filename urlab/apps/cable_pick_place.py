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
from ..skills.pick import (GraspCheck, GraspController, GraspGeometry, GraspImageRecorder,
                           GraspRecovery, log_grasp_delta)
from ..transforms import from_cfg, inverse
from ._cable import build_scanner, make_confirm
from ._runner import run_app

log = urlog.get('cable-pick-place')


def _attempt(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder):
    """One scan->grasp attempt. Returns 'ok' | 'missed' | 'empty' | 'abort'."""
    scanner.estimator.reset()
    T_conn_grasp = from_cfg(cfg.section('connector_grasp'))

    if not robot.gripper.open('open'):
        return 'abort'

    T_conn = scanner.scan(confirm=confirm)
    if T_conn is None:
        return 'abort'
    geom.T_base_grasp = T_conn @ T_conn_grasp
    log.info('Connector origin %s, grasp target set.', T_conn[:3, 3].round(3))

    # Grasp directly from wherever the scan ended (already close to the cable) -- no detour back to
    # the initial pose first.
    runner = StepRunner(log, confirm=confirm is not None)
    steps = [
        ('move to grasp-align', lambda: robot.move_fingertip(geom.pre_grasp(), 'grasp-align')),
        ('report pre-grasp delta', lambda: log_grasp_delta(robot, geom.T_base_grasp, 'pre-grasp')),
        ('move to grasp', lambda: grasp.descend(robot, geom, 'grasp')),
        ('report at-grasp delta', lambda: log_grasp_delta(robot, geom.T_base_grasp, 'at-grasp')),
    ]
    # Record wrist images at grasp_check.capture_rate_hz (default 1 Hz) for the whole descent +
    # close + recovery -- the timed record, alongside the count-labelled frames GraspRecovery saves.
    with recorder.recording(scanner.camera):
        if not runner.run(steps):
            return 'abort'
        # Close + grasp-check + recovery: a blind loose->close retry, then mode-directed reseat nudges
        # (see GraspRecovery) instead of a bare close, so a cable on the fingertip flats/tips is
        # reseated rather than failing the whole scan->grasp attempt.
        return recovery.grasp_with_recovery(robot, geom, check, camera=scanner.camera)


def build_and_run(cfg, robot, camera, args):
    scanner, _detector, _estimator = build_scanner(cfg, robot, camera)
    geom = GraspGeometry(cfg)
    check = GraspCheck(cfg)
    recovery = GraspRecovery(cfg)
    grasp = GraspController(cfg)
    recorder = GraspImageRecorder(cfg)
    confirm = make_confirm(cfg)

    # RESET at the start: open the gripper and go to the defined HOME pose under admittance, so the
    # run always begins from the same known configuration. q_home is then that home config.
    if not reset.reset_robot(robot, cfg, 'start reset'):
        return False
    q_home = robot.arm.q()

    # Pick, with grasp-check retry.
    attempt = 0
    while True:
        result = _attempt(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder)
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
        ('lift', lambda: grasp.lift(robot, geom, 'lift')),   # same mode as the pickup descent
        ('move to pre-place', lambda: robot.move_fingertip(geom.pre_place(), 'pre-place')),
        ('move to place', lambda: robot.move_fingertip(geom.place(), 'place')),
        ('open gripper (release)', robot.gripper.open),
        ('retreat', lambda: robot.move_fingertip(geom.pre_place(), 'retreat')),
        ('end reset (open + home under admittance)',
         lambda: reset.reset_robot(robot, cfg, 'end reset')),
    ])


def main():
    run_app('Cable pick-and-place (multi-view SAM3)', 'cable_pick_place', build_and_run,
            needs_camera=True)


if __name__ == '__main__':
    main()
