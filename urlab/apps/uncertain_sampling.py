"""Uncertain-assembly sampling -- data collection for a HELD CONNECTOR.

Repeatedly drive a PERTURBED connector into the mate under compliance, logging each sample, then
retract and repeat. The connector is held via the connector_holder frame (tool0 -> connector_holder
-> connector); the assembly TARGET is recorded for the connector_holder (hand-guide to a good mate,
read it off the monitor), and the connector target = that @ the holder->connector offset.

Each logged sample is: trial, timestamp, raw tool0-wrt-base, the connector's DEVIATION from the
ideal mate (identity at a perfect mate), and the contact wrench BOTH as recorded (base_link) and
re-expressed in the connector frame.

CONTROL is COMPLIANCE (software admittance, robot/admittance.py), NOT forceMode. The arm FOLLOWS the
(perturbed) assembly trajectory as a position reference and YIELDS to contact through a virtual
spring-mass-damper with finite restoring stiffness, springing back toward the reference when contact
eases. forceMode is pure force control -- no stiffness -- so it floats freely off the path.

ASSUMPTION -- LINEAR (PEG-IN-HOLE) ASSEMBLY. The mate is taken to be a single-axis insertion along
the connector's +X: the trajectory is a straight -X -> 0 approach, and the escape is simply the
reverse translation along that same axis (`retract_distance_m` back along the connector's OWN -X, so
a perturbed part backs out along its own axis, not the target's). Nothing here handles a curved,
multi-axis, or twist-to-lock mate -- those would need a real reverse-path retract.

    for each trial: perturb (connector frame), move to the perturbed start (stiff, free space),
    follow the path under ADMITTANCE (LOGGING at the servo rate, guarded), settle,
    retract straight back along the connector's -X -> CSV.
"""

import os
import time

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from ..robot import AdmittanceController, ForceGuard
from ..skills import trajectory as traj
from ..transforms import from_cfg, inverse, matrix_to_xyzrpy, pose_error, translation_matrix
from ._runner import run_app

log = urlog.get('uncertain-sampling')

# Column suffixes. Every dimensional field carries its UNIT in the name: translation in MILLIMETRES
# (the logs are read by hand and mm is the unit the monitor and the calibrations use), rotation in
# degrees. Quaternion components are dimensionless. Forces stay N / Nm.
_POSE = ('x_mm', 'y_mm', 'z_mm', 'qx', 'qy', 'qz', 'qw', 'yaw_deg', 'pitch_deg', 'roll_deg')
_WRENCH = ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')
_HEADER = (['trial', 'timestamp']
           + [f'tool0_base_{s}' for s in _POSE]            # raw tool0 wrt base
           + [f'connector_target_{s}' for s in _POSE]      # connector deviation from the ideal mate
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

    # The trajectory rows are the CONNECTOR w.r.t. the TARGET CONNECTOR (last row = identity = the
    # mate; a direct -X -> 0 insertion). With an identity last row, anchor_target reduces to
    # T_base_targetobj = T_base_connector_target, so each row directly places the connector.
    csv_in = urconfig.resolve(cfg, cfg.get('trajectory_csv', 'assembly_trajectory.csv'))
    mats = traj.load_csv(csv_in, angles_deg=bool(cfg.get('trajectory_angles_deg', False)))
    T_base_targetobj = traj.anchor_target(T_base_assembled, T_tool0_held, mats[-1])

    dense = traj.resample(mats, float(s.get('translational_resolution_m', 0.001)),
                          float(s.get('rotational_resolution_deg', 1.0)))
    k = max(2, int(np.ceil(float(s.get('chunk_fraction', 0.2)) * len(dense))))   # >=2 for a ramp

    # UNCERTAINTY as absolute LOWER/UPPER bounds per DOF (not half-widths), so a range need not be
    # centred on zero. 'grid' sweeps every combination at `grid_resolution` and DERIVES the trial
    # count; 'random' draws uniformly inside the bounds for `num_trials`.
    unc = s.get('uncertainty', {}) or {}
    unc_lo, unc_hi = unc.get('lower', [0.0] * 6), unc.get('upper', [0.0] * 6)
    noise = s.get('noise', {}) or {}
    noise_lo, noise_hi = noise.get('lower', [0.0] * 6), noise.get('upper', [0.0] * 6)
    mode = str(s.get('mode', 'random')).lower()
    if mode not in ('random', 'grid'):
        log.error("sampling.mode %r must be 'random' or 'grid'.", mode)
        return False
    try:
        grid = traj.grid_deltas(unc_lo, unc_hi, s.get('grid_resolution', [0.0] * 6)) \
            if mode == 'grid' else None
    except ValueError as exc:
        log.error('%s', exc)
        return False
    num_trials = len(grid) if grid is not None else int(s.get('num_trials', 20))
    log.info('%d ideal rows -> %d dense; chunk = %d waypoints; %d trials (%s sampling%s).',
             len(mats), len(dense), k, num_trials, mode.upper(),
             ' -- trial count derived from the grid' if grid is not None else '')

    # WHICH error source the trial perturbation models. Validated here so a typo fails before the
    # arm moves, not on the first trial. See trajectory.perturb for the physics of each.
    perturb_frame = str(s.get('perturb_frame', 'connector')).lower()
    if perturb_frame not in traj.PERTURB_FRAMES:
        log.error('sampling.perturb_frame %r must be one of %s.', perturb_frame, traj.PERTURB_FRAMES)
        return False
    log.info('Perturbing in the %s frame (%s); bounds lower=%s upper=%s.', perturb_frame,
             'in-hand pose error' if perturb_frame != 'target' else 'target/socket pose error',
             list(unc_lo), list(unc_hi))

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
    tare = (lambda: robot.arm.zero_ft(settle=False)) \
        if bool(cfg.get_path('compliance.tare_before', True)) else None
    settle_s = float(cfg.get_path('compliance.settle_s', 0.5))

    # CARTESIAN SPEED LIMITS for the compliant reference (mm/s, deg/s). Each ramp segment is given
    # the time its own geometry needs, so the reference never exceeds either limit -- rather than a
    # fixed total time, which silently changes speed whenever the path length or resolution changes.
    # The RETRACT gets its own (faster) limits: it is a free-space escape with no contact expected.
    v_mm_s = float(cfg.get_path('speed.max_cartesian_translation_mm_s', 3.5))
    w_deg_s = float(cfg.get_path('speed.max_cartesian_rotation_deg_s', 5.0))
    rv_mm_s = float(cfg.get_path('speed.retract_translation_mm_s', v_mm_s))
    rw_deg_s = float(cfg.get_path('speed.retract_rotation_deg_s', w_deg_s))
    retract_m = float(cfg.get('retract_distance_m', 0.05))   # straight back along the connector's -X
    min_seg_s = 1.0 / adm.rate                               # never below one servo cycle

    def seg_time(A, B, v=None, w=None):
        """Seconds for the reference to go A -> B without exceeding either cartesian limit."""
        v = v_mm_s if v is None else v
        w = w_deg_s if w is None else w
        lin_m, ang_rad = pose_error(A, B)
        t_lin = (lin_m * 1000.0 / v) if v > 0 else 0.0
        t_ang = (np.degrees(ang_rad) / w) if w > 0 else 0.0
        return max(t_lin, t_ang, min_seg_s)

    log.info('Compliant reference: INSERT %.1f mm/s / %.1f deg/s, RETRACT %.1f mm/s / %.1f deg/s.',
             v_mm_s, w_deg_s, rv_mm_s, rw_deg_s)
    decim = max(1, int(s.get('log_decimation', 5)))        # log every Nth servo cycle (125 Hz / N)
    q_home = robot.arm.q()

    # Approach the stand-off under position control (free space, stiff). The stand-off is measured
    # from the START of the assembly trajectory (mats[0]), backed off a further standoff_distance_m
    # along standoff_axis (a TARGET-frame direction) -- so it is always CLEAR of the path start.
    # Measuring it from the MATE instead would let a stand-off shorter than the trajectory's first
    # row land INSIDE the path, which is not a stand-off at all.
    standoff_axis = np.asarray(cfg.get('standoff_axis', [-1, 0, 0]), dtype=float)
    T_standoff_held = translation_matrix(
        standoff_axis * float(cfg.get('standoff_distance_m', 0.05))) @ mats[0]
    q = robot.arm.ik(traj.tool0_at(T_base_targetobj, T_standoff_held, T_tool0_held), q_home)
    if q is None or not robot.arm.move_j(q, label='approach standoff'):
        fout.close()
        return False
    seed_q = q

    ok = True
    try:
        durations = []
        for trial in range(1, num_trials + 1):
            log.info('--- trial %d/%d ---', trial, num_trials)
            t_trial = time.time()
            # The trial's misalignment: the next GRID point, or a uniform draw inside the bounds.
            # `noise` adds per-waypoint jitter (usually 0). `perturb_frame` picks the ERROR SOURCE --
            # 'connector' = in-hand pose error (default), 'target' = socket pose error.
            bias = grid[trial - 1] if grid is not None else traj.random_delta(unc_lo, unc_hi, rng)
            b_xyz, b_rpy = matrix_to_xyzrpy(bias)          # report the offsets BEFORE moving
            log.info('offsets (%s frame): xyz=[%+7.2f, %+7.2f, %+7.2f] mm  '
                     'rpy=[%+6.2f, %+6.2f, %+6.2f] deg',
                     perturb_frame, *(b_xyz * 1000.0), *np.degrees(b_rpy))
            perturbed = traj.perturb(dense[:k], bias, noise_lo, noise_hi, rng, frame=perturb_frame)
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
            last_ref = refs[0]
            for i in range(1, len(refs)):
                res = adm.ramp(refs[i - 1], refs[i], seg_time(refs[i - 1], refs[i]),
                               guard, on_step=log_cb)
                last_ref = refs[i]
                if res == 'seated':
                    log.info('Contact limit reached at waypoint %d/%d -- connector SEATED.',
                             i, len(refs) - 1)
                    break
            adm.hold(last_ref, settle_s, guard, on_step=log_cb)   # settle (records the contact wrench)

            # RETRACT: a LINEAR (peg-in-hole) escape -- straight back along the CONNECTOR's OWN -X by
            # retract_distance_m, from wherever the insert stopped. Still compliant (it yields if it
            # catches), but the guard is NOT armed: a seated/jammed connector is already over the
            # limit, so a guarded retract would block the very motion that frees it (see
            # ForceGuard.disable()).
            T_out = _retract_ref(last_ref, T_tool0_held, retract_m)
            adm.ramp(last_ref, T_out, seg_time(last_ref, T_out, rv_mm_s, rw_deg_s), guard=None)
            adm.stop()
            fout.flush()
            os.fsync(fout.fileno())

            durations.append(time.time() - t_trial)
            mean_s = sum(durations) / len(durations)
            log.info('trial %d/%d took %.1f s | mean cycle %.1f s | %d left, ETA %s (done ~%s)',
                     trial, num_trials, durations[-1], mean_s, num_trials - trial,
                     _fmt_dur(mean_s * (num_trials - trial)),
                     _clock(mean_s * (num_trials - trial)))
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


def _retract_ref(T_ref, T_tool0_held, distance_m):
    """The tool0 reference that backs the HELD PART straight out along ITS OWN -X by `distance_m`.

    ASSUMES LINEAR (PEG-IN-HOLE) ASSEMBLY: the mate is a single-axis insertion along the connector's
    +X, so the escape is simply the reverse translation along that same axis. Expressed in the
    CONNECTOR's frame (right-multiply), so it follows the part's ACTUAL, perturbed orientation --
    a tilted connector backs out along its own axis, not the target's.

    `distance_m` is used as a magnitude: a negative value would drive INTO the socket."""
    back = translation_matrix([-abs(float(distance_m)), 0.0, 0.0])
    return T_ref @ T_tool0_held @ back @ inverse(T_tool0_held)


def _pose_fields_mm(T):
    """traj.pose_fields with the TRANSLATION converted to MILLIMETRES.

    The library works in metres throughout; the conversion happens here, at the logging boundary
    only, so the CSV reads in mm (matching the `_mm` column names) while nothing upstream changes."""
    f = traj.pose_fields(T)                        # [x, y, z (m), qx..qw, yaw, pitch, roll (deg)]
    return [f[0] * 1000.0, f[1] * 1000.0, f[2] * 1000.0] + f[3:]


def _log_row(writer, robot, trial, T_base_connector_target, T_tool0_connector):
    """One sample: raw tool0 (base), the connector's deviation from the ideal mate, and the contact
    wrench both AS RECORDED (base_link) and re-expressed in the CONNECTOR frame.

    Translations are logged in mm (see _pose_fields_mm); wrenches stay N / Nm."""
    T_base_tool0 = robot.tool0()
    T_base_connector = T_base_tool0 @ T_tool0_connector
    connector_wrt_target = inverse(T_base_connector_target) @ T_base_connector   # identity at the mate
    w_base = np.asarray(robot.arm.wrench(), dtype=float)          # tared, bridged into base_link
    w_connector = np.asarray(robot.arm.wrench_in(T_base_connector), dtype=float)
    writer.writerow([trial, time.time()]
                    + _pose_fields_mm(T_base_tool0)
                    + _pose_fields_mm(connector_wrt_target)
                    + list(w_base) + list(w_connector))


def _fmt_dur(seconds):
    """A duration as h:mm:ss / m:ss -- for the per-trial ETA."""
    seconds = int(max(0.0, seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f'{h}:{m:02d}:{sec:02d}' if h else f'{m}:{sec:02d}'


def _clock(seconds_from_now):
    """Wall-clock time the run is expected to finish (HH:MM:SS)."""
    from datetime import datetime, timedelta
    return (datetime.now() + timedelta(seconds=max(0.0, seconds_from_now))).strftime('%H:%M:%S')


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
