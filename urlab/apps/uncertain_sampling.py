"""Uncertain-assembly sampling -- data collection for a HELD CONNECTOR.

Repeatedly drive a PERTURBED connector into the mate under compliance, logging each sample, then
disassemble along the ideal path and repeat. The connector is held via the connector_holder frame
(tool0 -> connector_holder -> connector); the assembly TARGET is recorded for the connector_holder
(hand-guide to a good mate, read it off the monitor), and the connector target = that @ the
holder->connector offset.

Per-trial the perturbation is drawn in the CONNECTOR's OWN frame (half-widths along its axes). Each
logged sample is: trial, timestamp, raw tool0-wrt-base, the connector's DEVIATION from the ideal
mate (identity at a perfect mate), and the contact wrench BOTH as recorded (base_link) and
re-expressed in the connector frame.

CONTROL is COMPLIANCE (software admittance, robot/admittance.py), NOT forceMode. The arm FOLLOWS the
(perturbed) assembly trajectory as a position reference and YIELDS to contact through a virtual
spring-mass-damper with finite restoring stiffness, springing back toward the reference when contact
eases. forceMode is pure force control -- no stiffness -- so it floats freely off the path; that is
the drift this replaces.

    for each trial: perturb (connector frame), move to the perturbed start (stiff, free space),
    follow the path under ADMITTANCE (LOGGING at the servo rate, guarded), settle, retract -> CSV.
"""

import os
import time

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from ..robot import AdmittanceController, ForceGuard
from ..skills import trajectory as traj
from ..transforms import from_cfg, inverse, pose_error, translation_matrix
from ._runner import run_app

log = urlog.get('uncertain-sampling')

_POSE = ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw_deg', 'pitch_deg', 'roll_deg')
_WRENCH = ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')
_HEADER = (['trial', 'timestamp']
           + [f'tool0_base_{s}' for s in _POSE]           # raw tool0 wrt base
           + [f'connector_target_{s}' for s in _POSE]     # connector deviation from the ideal mate
           + [f'wrench_base_{s}' for s in _WRENCH]         # wrench as recorded (base_link)
           + [f'wrench_connector_{s}' for s in _WRENCH])   # wrench re-expressed in the connector frame


def build_and_run(cfg, robot, camera, args):
    s = cfg.section('sampling')
    seed = int(s.get('random_seed', 0))
    # A dedicated RNG so the perturbation stream is reproducible independent of anything else --
    # the ROS version shared numpy's global RNG with IK's random restarts, which desynchronised it.
    rng = np.random.default_rng(seed if seed > 0 else None)

    # The held part is the CONNECTOR, via the holder (tool0 -> connector_holder -> connector). The
    # assembly target is recorded for the connector_holder; the connector target = that @ the
    # (holder -> connector) offset. The assembled TOOL0 pose falls out, and the anchoring is unchanged.
    T_tool0_holder = robot.T_tool0_connector_holder
    T_holder_connector = robot.T_connector_holder_connector
    T_tool0_held = robot.T_tool0_connector                       # held part = the connector
    T_base_holder_target = from_cfg(cfg.section('connector_holder_target'))
    T_base_connector_target = T_base_holder_target @ T_holder_connector   # ideal assembled connector
    T_base_assembled = T_base_holder_target @ inverse(T_tool0_holder)     # assembled tool0 pose
    csv_in = urconfig.resolve(cfg, cfg.get('trajectory_csv', 'assembly_trajectory.csv'))
    # The trajectory rows are the CONNECTOR w.r.t. the TARGET CONNECTOR (last row = identity = the
    # mate; a direct -X -> 0 insertion). With an identity last row, anchor_target reduces to
    # T_base_targetobj = T_base_connector_target, so each row directly places the connector.
    mats = traj.load_csv(csv_in, angles_deg=bool(cfg.get('trajectory_angles_deg', False)))
    T_base_targetobj = traj.anchor_target(T_base_assembled, T_tool0_held, mats[-1])

    dense = traj.resample(mats, float(s.get('translational_resolution_m', 0.001)),
                          float(s.get('rotational_resolution_deg', 1.0)))
    k = max(2, int(np.ceil(float(s.get('chunk_fraction', 0.2)) * len(dense))))   # >=2 for a ramp
    num_trials = int(s.get('num_trials', 20))
    log.info('%d ideal rows -> %d dense; chunk = %d waypoints; %d trials.',
             len(mats), len(dense), k, num_trials)
    log.info('Compliant reference limited to %.1f mm/s / %.1f deg/s.',
             float(cfg.get_path('speed.max_cartesian_translation_mm_s', 3.5)),
             float(cfg.get_path('speed.max_cartesian_rotation_deg_s', 5.0)))

    # WHICH error source the trial perturbation models. Validated here so a typo fails before the
    # arm moves, not on the first trial. See trajectory.perturb for the physics of each.
    perturb_frame = str(s.get('perturb_frame', 'connector')).lower()
    if perturb_frame not in traj.PERTURB_FRAMES:
        log.error('sampling.perturb_frame %r must be one of %s.', perturb_frame, traj.PERTURB_FRAMES)
        return False
    log.info("Perturbing in the %s frame (%s), UNIFORM over +/- %s.", perturb_frame,
             'in-hand pose error' if perturb_frame != 'target' else 'target/socket pose error',
             s.get('uncertainty', s.get('bias', [])))

    csv_path = s.get('csv_path', 'data/uncertain_assembly_sampling/log.csv')
    cable = cfg.get('cable')
    if cable:                                    # group each run under a subfolder named by the cable
        head, tail = os.path.split(csv_path)
        csv_path = os.path.join(head, str(cable), tail)
    out_path = _timestamped(cfg, csv_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = open(out_path, 'w', newline='')
    import csv as _csv
    writer = _csv.writer(fout)
    writer.writerow(_HEADER)
    log.info('Logging to %s', out_path)

    # COMPLIANCE = software admittance (follows the reference, yields to contact, springs back). The
    # ForceGuard trips at the contact limit -> the connector SEATED. See robot/admittance.py.
    adm = AdmittanceController(robot.arm, cfg.section('compliance'))
    guard = ForceGuard(robot.arm, cfg.section('force_guard'))
    tare = (lambda: robot.arm.zero_ft(settle=False)) if bool(cfg.get_path('compliance.tare_before', True)) else None
    settle_s = float(cfg.get_path('compliance.settle_s', 0.5))

    # CARTESIAN SPEED LIMITS for the compliant reference (mm/s, deg/s). Each ramp segment is given
    # the time its own geometry needs, so the reference never exceeds either limit -- rather than a
    # fixed total time, which silently changes speed whenever the path length or resolution changes.
    v_mm_s = float(cfg.get_path('speed.max_cartesian_translation_mm_s', 3.5))
    w_deg_s = float(cfg.get_path('speed.max_cartesian_rotation_deg_s', 5.0))
    min_seg_s = 1.0 / adm.rate                       # never below one servo cycle

    def seg_time(A, B):
        """Seconds for the reference to go A -> B without exceeding either cartesian limit."""
        lin_m, ang_rad = pose_error(A, B)
        t_lin = (lin_m * 1000.0 / v_mm_s) if v_mm_s > 0 else 0.0
        t_ang = (np.degrees(ang_rad) / w_deg_s) if w_deg_s > 0 else 0.0
        return max(t_lin, t_ang, min_seg_s)
    decim = max(1, int(s.get('log_decimation', 5)))        # log every Nth servo cycle (125 Hz / N)
    q_home = robot.arm.q()

    # Approach the stand-off under position control (free space, stiff). The stand-off backs off the
    # mate along standoff_axis; the per-trial perturbed start is reached from here each trial.
    standoff_axis = np.asarray(cfg.get('standoff_axis', [-1, 0, 0]), dtype=float)
    T_standoff_held = translation_matrix(standoff_axis * float(cfg.get('standoff_distance_m', 0.2))) \
        @ mats[-1]
    q = robot.arm.ik(traj.tool0_at(T_base_targetobj, T_standoff_held, T_tool0_held), q_home)
    if q is None or not robot.arm.move_j(q, label='approach standoff'):
        fout.close()
        return False
    seed_q = q

    ok = True
    try:
        for trial in range(1, num_trials + 1):
            log.info('--- trial %d/%d ---', trial, num_trials)
            # `uncertainty` (per-DOF half-widths) is the trial's misalignment, drawn ONCE, UNIFORMLY
            # over +/- each half-width; `noise` adds per-waypoint jitter (usually 0). `perturb_frame`
            # picks the ERROR SOURCE being modelled -- 'connector' = in-hand pose error (default),
            # 'target' = socket pose error. See trajectory.perturb.
            perturbed = traj.perturb(dense[:k],
                                     s.get('uncertainty', s.get('bias', [0.001, 0.001, 0, 1, 1, 1])),
                                     s.get('noise', [0] * 6), rng, frame=perturb_frame)
            refs = [traj.tool0_at(T_base_targetobj, p, T_tool0_held) for p in perturbed]

            # Move to the perturbed START under position control (free space) -- this REALIZES the
            # known misalignment; admittance then takes over for the contact phase.
            q = robot.arm.ik(refs[0], seed_q)
            if q is None or not robot.arm.move_j(q, label='approach perturbed start'):
                log.warning('IK/approach failed for the perturbed start; skipping trial %d.', trial)
                continue
            seed_q = q

            # Follow the perturbed path under ADMITTANCE, logging at the servo rate. A guard trip =
            # the connector seated (stop advancing).
            cnt = [0]

            def log_cb(_cnt=cnt, _trial=trial):
                _cnt[0] += 1
                if _cnt[0] % decim == 0:
                    _log_row(writer, robot, _trial, T_base_connector_target, T_tool0_held)

            adm.reset()
            adm.warmup(refs[0], tare_fn=tare)          # engage servo + tare before the guard is armed
            guard.reset()
            last_ref, reached = refs[0], 0
            for i in range(1, len(refs)):
                res = adm.ramp(refs[i - 1], refs[i], seg_time(refs[i - 1], refs[i]),
                               guard, on_step=log_cb)
                last_ref, reached = refs[i], i
                if res == 'seated':
                    log.info('Contact limit reached at waypoint %d/%d -- connector SEATED.', i, len(refs) - 1)
                    break
            adm.hold(last_ref, settle_s, guard, on_step=log_cb)   # settle (records the contact wrench)

            # Retract along the SAME perturbed path (back out the way it came in). The guard is NOT
            # armed here: a seated/jammed connector is already over the limit, so a guarded retract
            # would block the very motion that frees it (see ForceGuard.disable()).
            prev = last_ref
            for idx in range(reached, -1, -1):
                adm.ramp(prev, refs[idx], seg_time(prev, refs[idx]), guard=None)
                prev = refs[idx]
            adm.stop()
            fout.flush()
            os.fsync(fout.fileno())
    except Exception:                              # noqa: BLE001
        ok = False
        log.exception('Sampling error:')
    finally:
        robot.arm.servo_stop()
        fout.close()
    if ok:
        robot.arm.move_j(q_home, label='home')
        log.info('Sampling complete: %s', out_path)
    return ok


def _log_row(writer, robot, trial, T_base_connector_target, T_tool0_connector):
    """One sample: raw tool0 (base), the connector's deviation from the ideal mate, and the contact
    wrench both AS RECORDED (base_link) and re-expressed in the CONNECTOR frame."""
    T_base_tool0 = robot.tool0()
    T_base_connector = T_base_tool0 @ T_tool0_connector
    connector_wrt_target = inverse(T_base_connector_target) @ T_base_connector   # identity at the mate
    w_base = np.asarray(robot.arm.wrench(), dtype=float)          # getActualTCPForce -> base_link
    w_connector = np.asarray(robot.arm.wrench_in(T_base_connector), dtype=float)
    writer.writerow([trial, time.time()]
                    + traj.pose_fields(T_base_tool0)
                    + traj.pose_fields(connector_wrt_target)
                    + list(w_base) + list(w_connector))


def _timestamped(cfg, path):
    from datetime import datetime
    path = urconfig.resolve(cfg, path)
    stem, ext = os.path.splitext(path)
    return f'{stem}_{datetime.now().strftime("%Y%m%d_%H%M%S")}{ext}'


def main():
    run_app('Uncertain-assembly sampling (data collection)', 'uncertain_sampling', build_and_run,
            with_gripper=False)


if __name__ == '__main__':
    main()
