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
                           GraspRecovery, connector_axis_height_m, fingertip_in_connector,
                           grasp_pose, held_junction_in_fingertip, log_grasp_delta,
                           retry_offset_x)
import numpy as np

from ..transforms import inverse, matrix_to_xyzrpy, translation_matrix
from ._cable import build_scanner, make_confirm
from ._common import guarded as _guarded   # re-exported: sibling apps import it from here
from ._runner import run_app

log = urlog.get('cable-assemble')


def _pick(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder, offset_x_m=0.0):
    """The cable pick, returning 'ok' | 'missed' | 'empty' | 'abort'. `offset_x_m` shifts the
    grasp along the JUNCTION's own x-axis -- the outer-retry perturbation (retry_offset_x) that
    keeps a deterministic scan->grasp->fail loop from retrying the identical pose."""
    scanner.estimator.reset()
    if not robot.gripper.open('open'):
        return 'abort'
    T_conn = scanner.scan(confirm=confirm)
    if T_conn is None:
        return 'abort'
    if offset_x_m:
        log.info('Retry perturbation: %+.1f mm along the junction x-axis.', offset_x_m * 1000)
        T_conn = T_conn @ translation_matrix([offset_x_m, 0.0, 0.0])

    # ---- CORRECT THE ESTIMATE FIRST: the part is a SOLID RESTING ON THE PLANE ----------------
    # The ground_plane scan reports the junction ON the ground plane -- that is what it can see.
    # The connector is lying flat on that plane, so its AXIS is one radius up, and it is the
    # GREATEST radius that decides it: a stepped barrel rests on its fattest section and carries
    # every thinner section clear with it. Fixing it HERE fixes the measurement once, for the
    # grasp and for everything downstream; folding it into the grasp offset would bury a
    # property of the PART inside a choice about the APPROACH.
    if bool(cfg.get_path('pickup.rests_on_ground_plane', True)):
        r_max = connector_axis_height_m(cfg)
        if r_max > 0.0:
            log.info('Connector rests on the ground plane: raising the detected pose %+.2f mm '
                     '(half its greatest diameter, %.2f mm) so the estimate is on the AXIS '
                     'rather than on the plane.', r_max * 1000.0, r_max * 2000.0)
            T_conn = translation_matrix([0.0, 0.0, r_max]) @ T_conn
        else:
            log.warning('pickup.rests_on_ground_plane is on but neither '
                        'grasp_check.connector_diameter_mm nor grasp_check.connector_counts is '
                        'set, so the greatest diameter is unknown -- the detected pose stays ON '
                        'the plane and the pads will aim low. Measure the barrel.')

    # ---- THE WHOLE GRASP COMMAND: the fingertip goes to detected_connector @ this -----------
    ftip_in_conn = fingertip_in_connector(cfg)
    geom.T_base_grasp = grasp_pose(T_conn, ftip_in_conn)
    _oxyz, _orpy = matrix_to_xyzrpy(ftip_in_conn)
    # THE CONNECTOR FRAME, spelled out because every sign here is a physical direction:
    # +x along the connector axis toward its free end, +y across the cable (the jaw-closing
    # direction), +z UP off the ground plane; the rpy aims the approach.
    log.info('fingertip_in_connector -- the target fingertip wrt the detected connector '
             '(+x along the connector, +y across the cable, +z up off the ground): '
             'xyz %s mm, rpy %s deg. The in-hand belief carries it; re-check the grasp_check '
             'band -- the barrel diameter at the new bite point is what the counts read.',
             np.round(_oxyz * 1000.0, 2).tolist(), np.round(np.degrees(_orpy), 2).tolist())
    # WHICH WAY THE FINGERTIP WILL FACE, reported before the arm moves -- nothing downstream
    # can tell you which way it came out. The knob is now fingertip_in_connector's own rpy yaw
    # (180 opposes the detected heading); cables.yaml's junction_in_fingertip no longer steers
    # the arm, it only describes the NOMINAL grip the in-hand belief is written against.
    _d = float(np.dot(geom.T_base_grasp[:3, 0], T_conn[:3, 0]))
    log.info('Grasp orientation: fingertip +X is %s the detected connector heading (dot %+.2f).',
             'ALONG' if _d > 0 else 'OPPOSED to', _d)

    # Grasp directly from wherever the scan ended (already close to the cable) -- no detour home first.
    # Record wrist images at grasp_check.capture_rate_hz (default 1 Hz) over the descent + close +
    # recovery, alongside the count-labelled frames GraspRecovery saves.
    # The descent runs FULLY OPEN deliberately: the open fingers' full capture width (~84 mm) is
    # the robustness margin against LATERAL (y) grasp error -- narrowing to the connector
    # diameter before the close would shrink exactly that margin.
    runner = StepRunner(log, confirm=confirm is not None)
    with recorder.recording(scanner.camera):
        if not runner.run([
            ('move to grasp-align', lambda: grasp.align(robot, geom, 'grasp-align')),
            ('report pre-grasp delta', lambda: log_grasp_delta(robot, geom.T_base_grasp, 'pre-grasp')),
            ('move to grasp', lambda: grasp.descend(robot, geom, 'grasp')),
        ]):
            # 'unreachable' is ACTIONABLE where 'abort' is not: it means the geometry refused
            # this approach, which a caller can answer by changing the approach. Anything else
            # (comms, an operator abort) stays 'abort'.
            return getattr(grasp, 'last_refusal', None) or 'abort'
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
    T_target = ins.fingertip_target(
        ic, inverse(held_junction_in_fingertip(fingertip_in_connector(cfg))),
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


def main():
    run_app('Cable pick-and-assemble (kinematic + compliant insert)', 'cable_pick_assemble',
            build_and_run, needs_camera=True)


if __name__ == '__main__':
    main()
