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

    resample the ideal trajectory -> for each trial: tare, perturb (connector frame), drive in
    (compliant, guarded, LOGGING), snap to the closest ideal pose, disassemble -> write the CSV.
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
            # Perturb in the CONNECTOR's own frame (half-widths along the connector's axes).
            perturbed = traj.perturb(dense[:k], s.get('bias', [0.001, 0.001, 0, 1, 1, 1]),
                                     s.get('noise', [0] * 6), rng, frame='connector')
            for pose_held in perturbed:
                if robot.arm.force() >= max_force:
                    break
                q = robot.arm.ik(traj.tool0_at(T_base_targetobj, pose_held, T_tool0_held), seed_q)
                if q is None:
                    log.warning('IK failed mid-chunk; ending this trial early.')
                    break
                robot.arm.move_j(q, label='insert')
                seed_q = q
                _log_row(writer, robot, trial, T_base_connector_target, T_tool0_held)
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
