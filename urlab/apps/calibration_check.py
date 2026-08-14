"""CALIBRATION CHECK -- probe the recorded target repeatedly and measure where contact happens.

The question this answers: is the recorded mate (frames.yaml targets: entry) still where the
physical socket is, and how repeatable is the approach? The 2026-08 hose campaign traced a whole
estimator failure back to a drifted/recalibrated frame, and the v3 map's pitch asymmetry to a
calibration offset -- this script is the two-minute test that would have caught both.

Per cycle:
    standoff   the held connector backed off standoff_distance_m along its own -X from the
               recorded target (free space, planned from the catalogue frames)
    approach   a SLOW straight-line advance along the connector's +X, under admittance, toward
               and slightly past the recorded target (overshoot_m) -- contact is guaranteed
    contact    the force guard trips (force_guard.max_force_n, with the usual optional
               persistence_s debounce); hold settle_s, then record the CONTACT POSE = the held
               connector wrt the recorded target (mm / deg) and the settled wrench
<<<<<<< HEAD
    retract    compliant, un-guarded, straight back out to the standoff. For mates the robot
               cannot back out of on its own (a fully assembled hose needs a human on the unlock
               button) there are two alternatives: retract: false leaves the arm AT the contact
               and ends the run after that single probe; retract: prompt waits at the contact
               for the operator to press the unlock button and confirm with ENTER in the
               terminal, THEN retracts -- so multi-cycle checks still work.
=======
    retract    compliant, un-guarded, straight back out to the standoff -- UNLESS retract: false,
               for mates the robot cannot back out of (a fully assembled hose needs a human on
               the unlock button). Then the arm is left AT the contact and the run ends after
               that single probe.
>>>>>>> 454e0ff2d19d0257f6da4df7e506be47eadf815a

After `cycles` repetitions: per-axis mean / std / min / max of the contact pose, and the DRIFT
(linear slope of contact x vs cycle). Contact x should sit near the physical face-to-face
offset and stay put; a trend means something is moving (fixture, grasp, or calibration).

Output: data/experiments/calibration_check_<stamp>/contacts.csv + a printed summary.
Run:  python -m urlab.apps.calibration_check --config configs/calibration_check.yaml
"""

import csv as _csv
import os
import time
from datetime import datetime

import numpy as np

from .. import log as urlog
from .. import tool_frames
from ..robot import AdmittanceController, ForceGuard
from ..transforms import inverse, matrix_to_xyzrpy, pose_error, translation_matrix
from ._runner import run_app

log = urlog.get('calibration-check')


def line_rows(standoff_m, overshoot_m, resolution_m):
    """Straight-line waypoints along the connector's +X, RELATIVE TO THE TARGET frame: from
    -standoff (backed out) to +overshoot (just past the recorded mate). Pure geometry, so the
    smoke tests can check it without a robot."""
    n = max(int(np.ceil((standoff_m + overshoot_m) / max(resolution_m, 1e-5))), 2)
    xs = np.linspace(-abs(standoff_m), abs(overshoot_m), n + 1)
    return [translation_matrix([float(x), 0.0, 0.0]) for x in xs]


def build_and_run(cfg, robot, camera, args):
    held_name = cfg.get('held_frame')
    if not held_name:
        log.error('held_frame is required.')
        return False
    frames = tool_frames.load_frames(cfg)
    targets = tool_frames.load_targets(cfg)
    if held_name not in frames or held_name not in targets:
        log.error('held_frame %r needs BOTH a frames: and a targets: entry in %s.',
                  held_name, tool_frames.frames_path(cfg))
        return False
    T_held = frames[held_name]                     # tool0 -> held connector
    T_base_tconn = targets[held_name]              # base_link <- target connector (the mate)

    cycles = int(cfg.get('cycles', 10))
    standoff_m = float(cfg.get('standoff_distance_m', 0.03))
    overshoot_m = float(cfg.get('overshoot_m', 0.005))
    res_m = float(cfg.get('resolution_m', 0.001))
    v_mm_s = float(cfg.get_path('speed.approach_translation_mm_s', 2.0))
    rv_mm_s = float(cfg.get_path('speed.retract_translation_mm_s', 20.0))
    settle_s = float(cfg.get_path('compliance.settle_s', 1.0))
<<<<<<< HEAD
    retract = cfg.get('retract', True)
    retract = (retract.strip().lower() if isinstance(retract, str)
               else ('auto' if retract else 'never'))
    if retract not in ('auto', 'never', 'prompt'):
        log.error("retract must be true, false or 'prompt' (got %r).", cfg.get('retract'))
        return False
    if cycles < 1 or standoff_m <= 0 or overshoot_m < 0:
        log.error('cycles must be >= 1, standoff_distance_m > 0, overshoot_m >= 0.')
        return False
    if retract == 'never' and cycles > 1:
=======
    retract = bool(cfg.get('retract', True))
    if cycles < 1 or standoff_m <= 0 or overshoot_m < 0:
        log.error('cycles must be >= 1, standoff_distance_m > 0, overshoot_m >= 0.')
        return False
    if not retract and cycles > 1:
>>>>>>> 454e0ff2d19d0257f6da4df7e506be47eadf815a
        log.warning('retract: false leaves the arm AT the contact, so only ONE probe can run '
                    '(%d cycles requested). Capping to 1.', cycles)
        cycles = 1

    adm = AdmittanceController(robot.arm, cfg.section('compliance'))
    guard = ForceGuard(robot.arm, cfg.section('force_guard'))
    tare = (lambda: robot.arm.zero_ft(settle=False)) \
        if bool(cfg.get_path('compliance.tare_before', True)) else None
    min_seg_s = 1.0 / adm.rate

    rows_line = line_rows(standoff_m, overshoot_m, res_m)
    refs = [T_base_tconn @ r @ inverse(T_held) for r in rows_line]
    log.info('Calibration check: %d cycles, %.0f mm standoff -> %.1f mm past the recorded '
             'target at %.1f mm/s, guard %.0f N (persistence %.2f s), retract %s.', cycles,
             standoff_m * 1000.0, overshoot_m * 1000.0, v_mm_s, guard.max_force,
             guard.persistence_s, retract)

    def seg_time(A, B, v):
        lin_m, ang_rad = pose_error(A, B)
        return max(lin_m * 1000.0 / max(v, 1e-6), min_seg_s)

    def contact_pose():
        """Held connector wrt the recorded target, [x..yaw] in mm / deg, plus the wrench."""
        xyz, rpy = matrix_to_xyzrpy(inverse(T_base_tconn) @ robot.tool0() @ T_held)
        w = robot.arm.wrench()
        return list(xyz * 1000.0) + list(np.degrees(rpy)), list(w)

    out_dir = os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                           f'calibration_check_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    fout = open(os.path.join(out_dir, 'contacts.csv'), 'w', newline='')
    writer = _csv.writer(fout)
    writer.writerow(['cycle', 'contact_x_mm', 'contact_y_mm', 'contact_z_mm',
                     'contact_roll_deg', 'contact_pitch_deg', 'contact_yaw_deg',
                     'force_n', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz',
                     'tripped_by', 'reached_end', 'duration_s'])
    log.info('Output: %s', out_dir)

    q_home = robot.arm.q()
    q = robot.arm.ik(refs[0], q_home)
    if q is None or not robot.arm.move_j(q, label='calibration-check standoff'):
        fout.close()
        return False
    seed_q = q

    contacts = []
    ok = True
    try:
        for cyc in range(1, cycles + 1):
            t0 = time.time()
            q = robot.arm.ik(refs[0], seed_q)
            if q is None or not robot.arm.move_j(q, label=f'cycle {cyc} standoff'):
                log.error('IK/approach failed at cycle %d; stopping.', cyc)
                ok = False
                break
            seed_q = q
            adm.reset()
            adm.warmup(refs[0], tare_fn=tare)
            guard.reset()
            last_ref, tripped = refs[0], False
            for i in range(1, len(refs)):
                res = adm.ramp(refs[i - 1], refs[i], seg_time(refs[i - 1], refs[i], v_mm_s),
                               guard)
                last_ref = refs[i]
                if res == 'seated':
                    tripped = True
                    break
            reached_end = not tripped
            adm.hold(last_ref, settle_s, guard=None)   # settle AT the stop, no re-trip races
            pose6, w = contact_pose()
            fmag = float(np.linalg.norm(w[:3]))
            contacts.append(pose6)
            writer.writerow([cyc] + [f'{v:.3f}' for v in pose6] + [f'{fmag:.2f}']
                            + [f'{v:.2f}' for v in w]
                            + [guard.tripped_by or '', reached_end,
                               f'{time.time() - t0:.1f}'])
            fout.flush()
            log.info('cycle %2d/%d: contact x %+7.2f mm  (y %+.2f, z %+.2f mm | r %+.2f, '
                     'p %+.2f, y %+.2f deg)  |f| %.1f N%s', cyc, cycles, pose6[0], pose6[1],
                     pose6[2], pose6[3], pose6[4], pose6[5], fmag,
                     '  [REACHED END -- no contact before the overshoot!]' if reached_end
                     else '')
<<<<<<< HEAD
            if retract == 'never':
                adm.stop()
                log.warning('retract: false -- the arm is LEFT AT THE CONTACT. Press the '
                            'unlock/release button before commanding any robot motion.')
            else:
                if retract == 'prompt':
                    adm.stop()         # hold position while the human is at the connector
                    try:
                        input(f'   cycle {cyc}/{cycles}: press the unlock/release button, '
                              'then hit ENTER to retract... ')
                    except EOFError:
                        log.warning("retract: prompt needs an interactive terminal and stdin "
                                    "is closed. Leaving the arm AT the contact; press the "
                                    "unlock/release button before commanding any motion.")
                        break
                    adm.reset()
                    adm.warmup(last_ref)   # NO tare: the sensor is loaded at the contact
                # compliant straight retract to the standoff reference
                adm.ramp(last_ref, refs[0], seg_time(last_ref, refs[0], rv_mm_s), guard=None)
                adm.stop()
=======
            if retract:
                # compliant straight retract to the standoff reference
                adm.ramp(last_ref, refs[0], seg_time(last_ref, refs[0], rv_mm_s), guard=None)
                adm.stop()
            else:
                adm.stop()
                log.warning('retract: false -- the arm is LEFT AT THE CONTACT. Press the '
                            'unlock/release button before commanding any robot motion.')
>>>>>>> 454e0ff2d19d0257f6da4df7e506be47eadf815a
    finally:
        fout.close()

    if len(contacts) >= 2:
        C = np.asarray(contacts, dtype=float)
        names = ['x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg']
        log.info('--- calibration summary over %d contacts ---', len(C))
        for j, nm in enumerate(names):
            log.info('   %-10s mean %+8.3f  std %6.3f  min %+8.3f  max %+8.3f', nm,
                     C[:, j].mean(), C[:, j].std(), C[:, j].min(), C[:, j].max())
        slope = float(np.polyfit(np.arange(len(C)), C[:, 0], 1)[0])
        log.info('   contact-x drift: %+.3f mm/cycle (%+.2f mm over the run). Repeatability '
                 '(x std): %.3f mm.', slope, slope * (len(C) - 1), C[:, 0].std())
        if abs(slope) * (len(C) - 1) > 0.5:
            log.warning('Contact x DRIFTED more than 0.5 mm over the run -- fixture, grasp or '
                        'frame is moving.')
    return ok and len(contacts) == cycles


def main():
    # with_gripper=False: the connector is fixtured between the closed fingers.
    run_app('Calibration check: repeated slow probes of the recorded target',
            'calibration_check', build_and_run, with_gripper=False)


if __name__ == '__main__':
    main()
