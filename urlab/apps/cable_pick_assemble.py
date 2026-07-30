"""Cable pick-and-assemble -- port of ur_cable_pick_assemble_demo.

The PICK is the exact cable-pick-place pipeline. What replaces "place" is a pluggable assembly:
stand-off, compliant chunked insertion (software admittance), release, retract. Only 'kinematic'
is implemented (the target pose is given outright); 'vision' fails loudly.

    [start reset] -> [PICK] -> lift -> stand-off -> insert (admittance, guarded)
      -> open (release) -> retract (multi-step) -> [end reset]
"""

from .. import log as urlog
from ..log import StepRunner
from ..robot import AdmittanceController, ForceGuard
from ..skills import insert as ins
from ..skills import reset
from ..skills.pick import (GraspCheck, GraspController, GraspGeometry, GraspImageRecorder,
                           GraspRecovery, log_grasp_delta, retry_offset_x)
from ..transforms import from_cfg, inverse, translation_matrix
from ._cable import build_scanner, make_confirm
from ._runner import run_app

log = urlog.get('cable-assemble')


def _pick(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder, offset_x_m=0.0):
    """The cable pick, returning 'ok' | 'missed' | 'empty' | 'abort'. `offset_x_m` shifts the
    grasp along the JUNCTION's own x-axis -- the outer-retry perturbation (retry_offset_x) that
    keeps a deterministic scan->grasp->fail loop from retrying the identical pose."""
    scanner.estimator.reset()
    # junction_in_fingertip (from cables.yaml): where the junction sits in the FINGERTIP frame at
    # the grasp -- so the fingertip goes to detected_junction @ its inverse before closing.
    T_ftip_junction = from_cfg(cfg.section('junction_in_fingertip'))
    if not robot.gripper.open('open'):
        return 'abort'
    T_conn = scanner.scan(confirm=confirm)
    if T_conn is None:
        return 'abort'
    if offset_x_m:
        log.info('Retry perturbation: %+.1f mm along the junction x-axis.', offset_x_m * 1000)
        T_conn = T_conn @ translation_matrix([offset_x_m, 0.0, 0.0])
    geom.T_base_grasp = T_conn @ inverse(T_ftip_junction)

    # PICKUP HEIGHT from the gripper model (pickup.height_from_model). ASSUMES the connector
    # rests ON THE GROUND PLANE -- the ground-plane scan puts the junction estimate AT the plane,
    # so the grasp target rises by d_max/2 (the centerline of the connector's thickest section)
    # PLUS the fingertip ADVANCE between the separation fingertip_grasp was calibrated at and the
    # expected grasp stall separation: the physical pad travels ~12.8 mm along the approach axis
    # over the stroke (calibrated circle model), so the STATIC tool0->fingertip transform is
    # exact at ONE separation only. COMPRESSION: all separations are zero-compression values; in
    # practice the pads squeeze (desired -- grip pressure), which the calibrated groove depth
    # already absorbs on average, and near closure the advance is insensitive to it (<0.1 mm) --
    # pad_compression_mm is exposed for completeness.
    hm = cfg.get_path('pickup.height_from_model', {}) or {}
    d_conn = cfg.get_path('grasp_check.connector_diameter_mm')
    if bool(hm.get('enabled', False)) and d_conn:
        from ..robot.gripper_kinematics import pad_forward_from_gap
        d_max = max(float(v) for v in d_conn) / 1000.0
        s_ref = float(hm.get('fingertip_ref_separation_mm', 0.0)) / 1000.0
        s_grasp = max(0.0, d_max - 2.0 * robot.gripper.groove_depth_m
                      - float(hm.get('pad_compression_mm', 0.0)) / 1000.0)
        advance = pad_forward_from_gap(s_grasp) - pad_forward_from_gap(s_ref)
        dz = d_max / 2.0 + advance
        log.info('Pickup height from the gripper model: %+.2f mm '
                 '(centerline %+.2f, fingertip advance %+.2f at %.1f mm separation).',
                 dz * 1000, d_max / 2.0 * 1000, advance * 1000, s_grasp * 1000)
        geom.T_base_grasp = translation_matrix([0.0, 0.0, dz]) @ geom.T_base_grasp

    # Grasp directly from wherever the scan ended (already close to the cable) -- no detour home first.
    # Record wrist images at grasp_check.capture_rate_hz (default 1 Hz) over the descent + close +
    # recovery, alongside the count-labelled frames GraspRecovery saves.
    runner = StepRunner(log, confirm=confirm is not None)
    steps = [
        ('move to grasp-align', lambda: robot.move_fingertip(geom.pre_grasp(), 'grasp-align')),
        ('report pre-grasp delta', lambda: log_grasp_delta(robot, geom.T_base_grasp, 'pre-grasp')),
    ]
    # PHYSICAL-dimension gripper prep: with the connector diameter known (cables.yaml), narrow
    # the fingers to diameter + clearance instead of descending fully open -- less close travel
    # at the grasp. Done AFTER the scan (full open keeps the fingers splayed out of the camera
    # view) and BEFORE the descent.
    d_conn = cfg.get_path('grasp_check.connector_diameter_mm')
    if d_conn:
        gap_m = (max(d_conn) + float(cfg.get_path('gripper.open_clearance_mm', 15.0))) / 1000.0
        steps.append(('narrow to clearance',
                      lambda: robot.gripper.go_to_gap(gap_m, 'clearance')))
    steps.append(('move to grasp', lambda: grasp.descend(robot, geom, 'grasp')))
    with recorder.recording(scanner.camera):
        if not runner.run(steps):
            return 'abort'
        # Close + grasp-check + recovery (blind retry, then mode-directed reseat nudges) -- see
        # GraspRecovery -- so a cable on the fingertip flats/tips is reseated, not failed.
        return recovery.grasp_with_recovery(robot, geom, check, camera=scanner.camera)


def build_and_run(cfg, robot, camera, args):
    ic = ins.InsertConfig(cfg)
    if ic.method != 'kinematic':
        log.error("assembly.method %r is not implemented (only 'kinematic'); aborting.", ic.method)
        return False

    scanner, _detector, _estimator = build_scanner(cfg, robot, camera)
    geom = GraspGeometry(cfg)
    check = GraspCheck(cfg)
    recovery = GraspRecovery(cfg)
    grasp = GraspController(cfg)
    recorder = GraspImageRecorder(cfg)
    guard = ForceGuard(robot.arm, cfg.get_path('assembly.force_guard', {}))
    confirm = make_confirm(cfg)

    # RESET at the start: open the gripper and go to the defined HOME pose under admittance.
    if not reset.reset_robot(robot, cfg, 'start reset'):
        return False
    q_home = robot.arm.q()

    # 1. PICK, with grasp-check retry (each full retry perturbed along the junction x-axis).
    attempt = 0
    while True:
        result = _pick(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder,
                       offset_x_m=retry_offset_x(attempt, check.retry_perturb_x_m))
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
    T_target = ins.fingertip_target(ic, inverse(from_cfg(cfg.section('junction_in_fingertip'))),
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
            # Lift in the SAME mode as the pickup descent. Position mode keeps the force guard (a
            # trip = collision); compliance mode yields to the cable's resistance instead of tripping.
            ('lift', lambda: grasp.lift(robot, geom, 'lift',
                                        position_guard=lambda mv: _guarded(robot, guard, mv))),
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

    # RESET at the end: open the gripper and go home under admittance.
    return reset.reset_robot(robot, cfg, 'end reset')


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
