"""Kinematic assembly demo -- port of ur_kinematic_assembly_demo.

Follow a CSV trajectory of held-object poses (relative to the target object) from a stand-off to
the mate, optionally under compliance, then disassemble/return home. No perception: the assembled
pose is measured ground truth and the target frame is back-derived from it (see skills/trajectory).

    stand-off -> [position or compliant] follow the trajectory -> mate
      -> [disassemble in reverse | return home]
"""

import time

import numpy as np

from .. import log as urlog
from ..robot import ForceGuard
from ..skills import trajectory as traj
from ..transforms import from_cfg
from .. import config as urconfig
from ._cable import make_confirm
from ._runner import run_app

log = urlog.get('kinematic-assembly')


def build_and_run(cfg, robot, camera, args):
    confirm = make_confirm(cfg)
    q_home = robot.arm.q()

    T_tool0_held = from_cfg(cfg.section('held_object_pose'))
    T_base_assembled = from_cfg(cfg.section('assembled_pose'))

    csv_path = urconfig.resolve(cfg, cfg.get('trajectory_csv', 'assembly_trajectory.csv'))
    mats = traj.load_csv(csv_path, angles_deg=bool(cfg.get('trajectory_angles_deg', False)))
    if len(mats) < 1:
        log.error('Trajectory %s has no usable rows.', csv_path)
        return False
    log.info('Loaded %d trajectory rows from %s.', len(mats), csv_path)

    T_base_targetobj = traj.anchor_target(T_base_assembled, T_tool0_held, mats[-1])
    poses = [traj.tool0_at(T_base_targetobj, m, T_tool0_held) for m in mats]

    standoff_axis = np.asarray(cfg.get('standoff_axis', [0, 0, 1]), dtype=float)
    standoff_dist = float(cfg.get('standoff_distance_m', 0.2))
    from ..transforms import translation_matrix
    T_standoff_held = translation_matrix(standoff_axis * standoff_dist) @ mats[-1]
    standoff_pose = traj.tool0_at(T_base_targetobj, T_standoff_held, T_tool0_held)

    q_standoff = robot.arm.ik(standoff_pose, q_home)
    if q_standoff is None:
        log.error('Stand-off pose is unreachable.')
        return False
    waypoint_q = traj.ik_chain(robot.arm, poses, q_standoff)
    if waypoint_q is None:
        return False

    compliant = str(cfg.get('control_mode', 'position')).lower() == 'admittance'
    guard = ForceGuard(robot.arm, {'max_force_n': cfg.get_path('admittance.max_force_n', 30.0),
                                   'max_torque_nm': cfg.get_path('admittance.max_torque_nm', 5.0)})

    # 1. Traverse to the stand-off under position control (a free-space move).
    if confirm and not confirm('move to stand-off'):
        return False
    if not robot.arm.move_j(q_standoff, label='stand-off'):
        return False
    time.sleep(cfg.get('settle_s', 0.2))

    # 2. Follow the trajectory. Under compliance, forceMode + per-waypoint moves; otherwise plain
    #    position moves. The force guard stops the insert when the part seats.
    ok = True
    try:
        if compliant:
            robot.arm.zero_ft()
            T_task = traj.tool0_at(T_base_targetobj, mats[-1], T_tool0_held)
            robot.arm.force_mode(T_task, [1, 1, 1, 1, 1, 1], [0.0] * 6, [0.05] * 3 + [0.17] * 3)
        for i, q in enumerate(waypoint_q):
            if guard.check():
                log.info('Contact limit reached at waypoint %d -- part seated.', i)
                break
            if confirm and not confirm(f'waypoint {i + 1}/{len(waypoint_q)}'):
                ok = False
                break
            guard.reset()
            robot.arm.add_guard(guard)
            moved = robot.arm.move_j(q, label=f'waypoint {i + 1}')
            robot.arm.clear_guards()
            if not moved:
                if guard.tripped_by:
                    log.info('Guard tripped (%s) -- seated.', guard.tripped_by)
                    break
                ok = False
                break
    finally:
        robot.arm.end_force_mode()
    if not ok:
        return False

    # 3. Disassemble (reverse) or return home.
    if cfg.get('disassemble_after', True):
        for i in range(len(waypoint_q) - 2, -1, -1):
            if not robot.arm.move_j(waypoint_q[i], label=f'disassemble {i}'):
                return False
    if not robot.arm.move_j(q_standoff, label='stand-off'):
        return False
    return robot.arm.move_j(q_home, label='home')


def main():
    run_app('Kinematic assembly (CSV trajectory)', 'kinematic_assembly', build_and_run,
            with_gripper=False)


if __name__ == '__main__':
    main()
