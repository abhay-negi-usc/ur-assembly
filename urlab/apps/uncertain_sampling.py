"""Uncertain-assembly sampling -- port of ur_uncertain_assembly_sampling.

Repeatedly drive a PERTURBED held part into the mate under compliance, logging the commanded and
actual poses plus the contact wrench, then disassemble along the ideal path and repeat. This is a
DATA-COLLECTION run: it produces a CSV of (commanded, actual, wrench) samples for studying how a
misaligned part behaves during insertion.

    resample the ideal trajectory -> for each trial: tare, perturb, drive in (compliant, guarded,
    LOGGING), snap to the closest ideal pose, disassemble along the ideal path -> write the CSV.
"""

import os
import time

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from ..skills import trajectory as traj
from ..transforms import from_cfg, inverse, translation_matrix
from ._runner import run_app

log = urlog.get('uncertain-sampling')

_HEADER = (['trial', 'timestamp']
           + [f'tool0_base_{s}' for s in
              ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw_deg', 'pitch_deg', 'roll_deg')]
           + [f'held_target_{s}' for s in
              ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw_deg', 'pitch_deg', 'roll_deg')]
           + [f'cmd_tool0_base_{s}' for s in
              ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw_deg', 'pitch_deg', 'roll_deg')]
           + [f'cmd_held_target_{s}' for s in
              ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw_deg', 'pitch_deg', 'roll_deg')]
           + ['ft_tool0_fx', 'ft_tool0_fy', 'ft_tool0_fz',
              'ft_tool0_tx', 'ft_tool0_ty', 'ft_tool0_tz'])


def build_and_run(cfg, robot, camera, args):
    s = cfg.section('sampling')
    seed = int(s.get('random_seed', 0))
    # A dedicated RNG so the perturbation stream is reproducible independent of anything else --
    # the ROS version shared numpy's global RNG with IK's random restarts, which desynchronised it.
    rng = np.random.default_rng(seed if seed > 0 else None)

    T_tool0_held = from_cfg(cfg.section('held_object_pose'))
    T_base_assembled = from_cfg(cfg.section('assembled_pose'))
    csv_in = urconfig.resolve(cfg, cfg.get('trajectory_csv', 'assembly_trajectory.csv'))
    mats = traj.load_csv(csv_in, angles_deg=bool(cfg.get('trajectory_angles_deg', False)))
    T_base_targetobj = traj.anchor_target(T_base_assembled, T_tool0_held, mats[-1])

    dense = traj.resample(mats, float(s.get('translational_resolution_m', 0.001)),
                          float(s.get('rotational_resolution_deg', 1.0)))
    k = max(1, int(np.ceil(float(s.get('chunk_fraction', 0.2)) * len(dense))))
    log.info('%d ideal rows -> %d dense; chunk = %d waypoints; %d trials.',
             len(mats), len(dense), k, int(s.get('num_trials', 20)))

    out_path = _timestamped(cfg, s.get('csv_path', 'data/uncertain_assembly_sampling/log.csv'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = open(out_path, 'w', newline='')
    import csv as _csv
    writer = _csv.writer(fout)
    writer.writerow(_HEADER)
    log.info('Logging to %s', out_path)

    limits = [0.05] * 3 + [0.17] * 3
    max_force = float(s.get('max_force_n', cfg.get_path('admittance.max_force_n', 30.0)))
    q_home = robot.arm.q()
    seed_q = q_home

    # Traverse to the stand-off and the first dense pose under position control (the ROS comment is
    # explicit: streaming these as compliant references exceeds joint velocity limits).
    standoff_axis = np.asarray(cfg.get('standoff_axis', [0, 0, 1]), dtype=float)
    T_standoff_held = translation_matrix(standoff_axis * float(cfg.get('standoff_distance_m', 0.2))) \
        @ mats[-1]
    for pose in (traj.tool0_at(T_base_targetobj, T_standoff_held, T_tool0_held),
                 traj.tool0_at(T_base_targetobj, dense[0], T_tool0_held)):
        q = robot.arm.ik(pose, seed_q)
        if q is None or not robot.arm.move_j(q, label='approach'):
            fout.close()
            return False
        seed_q = q

    ok = True
    try:
        robot.arm.force_mode(traj.tool0_at(T_base_targetobj, mats[-1], T_tool0_held),
                             [1, 1, 1, 1, 1, 1], [0.0] * 6, limits)
        for trial in range(1, int(s.get('num_trials', 20)) + 1):
            log.info('--- trial %d/%d ---', trial, int(s.get('num_trials', 20)))
            robot.arm.zero_ft()
            perturbed = traj.perturb(dense[:k], s.get('bias', [0.001, 0.001, 0, 1, 1, 1]),
                                     s.get('noise', [0] * 6), rng)
            for pose_held in perturbed:
                if robot.arm.force() >= max_force:
                    break
                q = robot.arm.ik(traj.tool0_at(T_base_targetobj, pose_held, T_tool0_held), seed_q)
                if q is None:
                    log.warning('IK failed mid-chunk; ending this trial early.')
                    break
                robot.arm.move_j(q, label='insert')
                seed_q = q
                _log_row(writer, robot, trial, T_base_targetobj, T_tool0_held, pose_held)
            # Snap to the closest ideal pose, then disassemble along the ideal path.
            actual_held = inverse(T_base_targetobj) @ robot.tool0() @ T_tool0_held
            j_close = traj.closest_index(actual_held, dense,
                                         float(s.get('closest_pose_rot_weight_mm_per_deg', 1.0)))
            for idx in range(j_close, -1, -1):
                q = robot.arm.ik(traj.tool0_at(T_base_targetobj, dense[idx], T_tool0_held), seed_q)
                if q is not None:
                    robot.arm.move_j(q, label='disassemble')
                    seed_q = q
            fout.flush()
            os.fsync(fout.fileno())
    except Exception:                              # noqa: BLE001
        ok = False
        log.exception('Sampling error:')
    finally:
        robot.arm.end_force_mode()
        fout.close()
    if ok:
        robot.arm.move_j(q_home, label='home')
        log.info('Sampling complete: %s', out_path)
    return ok


def _log_row(writer, robot, trial, T_base_targetobj, T_tool0_held, cmd_held):
    T_base_tool0 = robot.tool0()
    held_actual = inverse(T_base_targetobj) @ T_base_tool0 @ T_tool0_held
    cmd_tool0 = traj.tool0_at(T_base_targetobj, cmd_held, T_tool0_held)
    w = robot.arm.wrench()
    writer.writerow([trial, time.time()]
                    + traj.pose_fields(T_base_tool0) + traj.pose_fields(held_actual)
                    + traj.pose_fields(cmd_tool0) + traj.pose_fields(cmd_held)
                    + list(np.asarray(w, dtype=float)))


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
