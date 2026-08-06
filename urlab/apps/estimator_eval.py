"""In-hand pose ESTIMATOR EVALUATION -- the manifold ICP against a KNOWN ground truth.

cable_pick_estimate_assemble tests the estimator end-to-end, but after a real pick the true
in-hand pose is unknown, so the estimate cannot be scored. Here the connector is RIGIDLY FIXTURED
between the CLOSED gripper fingers (like uncertain_sampling's held part), so the TRUE tool0->
connector pose IS the shared catalogue's `held_frame:` entry (configs/frames.yaml) -- known
exactly. Each trial then INJECTS a known belief error and lets the estimator try to remove it:

    for each of eval.num_trials trials:
        perturb   the BELIEF only: T_believed = T_true @ delta, delta drawn inside
                  eval.perturbation (the robot then plans with the wrong belief, so the part
                  physically rides the trajectory offset -- exactly a bad grasp)
        LOOP (max eval.max_attempts):
            assemble   admittance-follow the trajectory planned from the BELIEF, collecting
                       observations (believed connector-wrt-target pose + wrench in the believed
                       connector frame) -- same law and logging as cable_pick_estimate_assemble
            retract    straight back along the believed connector's own -X (peg-in-hole)
            estimate   manifold ICP -> T_corr;  T_believed <- T_believed @ T_corr
            score      remaining GROUND-TRUTH error = inverse(T_true) @ T_believed -- logged
                       per attempt; within eval.success_tolerance -> converged (early stop)
        disassemble: back at the stand-off (free space) before the next trial

The GRIPPER IS NEVER OPENED OR CLOSED (with_gripper=False) -- the part is fixtured in the closed
fingers and an open would drop it mid-run.

Output: data/experiments/estimator_eval_<timestamp>/ with trials.csv (one row per attempt:
injected error, ground-truth error before/after the update, the correction, ICP diagnostics),
per-attempt observation CSVs for post-analysis, and summary.csv (per-attempt-index aggregate).

Units: robot poses are metres/radians (repo convention); the manifold space and all logged errors
are mm/deg -- conversions happen only at the observation/logging boundary in this file.
"""

import csv as _csv
import os
import time
from datetime import datetime

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from .. import tool_frames
from ..robot import AdmittanceController, ForceGuard
from ..skills import trajectory as traj
from ..skills.manifold import FORCE_COLS, ManifoldEstimator, POSE_COLS, TORQUE_COLS
from ..transforms import inverse, matrix_to_xyzrpy, pose_error, translation_matrix
from ._runner import run_app
from .uncertain_sampling import _clock, _fmt_dur, _retract_ref

log = urlog.get('estimator-eval')

# Component order of every logged error vector -- matches skills.manifold.DIMS, so the estimate
# dims name the same columns (err_after_x_mm vs estimate_dims 'x_mm', etc.).
_ERR = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')


def _gt_error(T_true, T_believed):
    """The belief's remaining error against the KNOWN truth, in the estimator's own convention:
    the right-multiplied offset E with T_true @ E = T_believed (identity = perfect belief).

    Returns (vec6 [mm, deg] ordered as _ERR, pos_mm, rot_deg)."""
    E = inverse(T_true) @ T_believed
    xyz, rpy = matrix_to_xyzrpy(E)
    lin, ang = pose_error(E, np.eye(4))
    return list(xyz * 1000.0) + list(np.degrees(rpy)), lin * 1000.0, float(np.degrees(ang))


def _corr_to_m(T_corr_mm):
    """The estimator's correction (translation in mm) -> a metre-based transform."""
    T = np.array(T_corr_mm, dtype=float)
    T[:3, 3] /= 1000.0
    return T


def _observe(robot, T_tool0_conn, T_base_tconn):
    """One observation row: BELIEVED connector-wrt-target [mm, deg 6-vec] + raw wrench in the
    believed connector frame [N, Nm] -- same convention as cable_pick_estimate_assemble."""
    T_base_conn = robot.tool0() @ T_tool0_conn
    rel = inverse(T_base_tconn) @ T_base_conn
    xyz, rpy = matrix_to_xyzrpy(rel)
    w = robot.arm.wrench_in(T_base_conn)
    return list(xyz * 1000.0) + list(np.degrees(rpy)) + list(w)


def _save_observations(path, rows):
    with open(path, 'w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
        w.writerows(rows)


def _fieldnames(dims):
    """trials.csv schema -- fixed up-front so the file is written incrementally, row by row."""
    return (['trial', 'attempt', 'n_observations', 'seated', 'check_pos_mm', 'check_rot_deg']
            + [f'inj_{s}' for s in _ERR]
            + [f'err_before_{s}' for s in _ERR] + ['err_before_pos_mm', 'err_before_rot_deg']
            + [f'corr_{d}' for d in dims] + ['icp_inliers', 'icp_residual', 'estimate']
            + [f'err_after_{s}' for s in _ERR] + ['err_after_pos_mm', 'err_after_rot_deg']
            + ['converged'])


def _plot_trial_errors(path, trial, dims, err6, norms, tol_pos_mm, tol_rot_deg):
    """ONE figure per trial, RE-SAVED after every attempt: the GROUND-TRUTH error, all attempts
    co-plotted (x = 0 is the injected error, x = k the error left after attempt k's update).
    One panel per estimated dim (signed, dashed zero) plus the pos/rot norms against the
    convergence tolerances. BEST-EFFORT: a plotting problem (e.g. matplotlib missing on the
    robot box) is logged and skipped, never allowed to kill a hardware run."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        err6 = np.asarray(err6, dtype=float)
        norms = np.asarray(norms, dtype=float)
        x = np.arange(len(err6))
        n = len(dims) + 2
        fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.1 * n), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, dim in zip(axes, dims):
            j = _ERR.index(dim)
            unit = 'deg' if dim.endswith('_deg') else 'mm'
            ax.axhline(0.0, ls='--', lw=1.0, color='#888888', zorder=1)
            ax.plot(x, err6[:, j], 'o-', color='#4C72B0', zorder=2)
            ax.set_ylabel(f'{dim} error [{unit}]')
        for ax, col, label, tol in ((axes[-2], 0, '|pos| error [mm]', tol_pos_mm),
                                    (axes[-1], 1, '|rot| error [deg]', tol_rot_deg)):
            ax.plot(x, norms[:, col], 'o-', color='#DD8452', zorder=2)
            ax.axhline(tol, ls='--', lw=1.0, color='#C44E52', zorder=1,
                       label=f'tolerance {tol:g}')
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0.0)
            ax.legend(loc='upper right', fontsize=8)
        axes[-1].set_xticks(x)
        axes[-1].set_xlabel('attempt (0 = injected error, before any update)')
        fig.suptitle(f'trial {trial}: ground-truth belief error per attempt', y=0.995)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as exc:                       # noqa: BLE001 -- plotting is never fatal
        log.warning('trial error plot skipped (%s)', exc)


def _write_summary(out_dir, rows, dims):
    """Per-attempt-index aggregate of the |ground-truth error| left AFTER each update, plus a
    'final' row (each trial's last attempt) -- the headline estimator numbers."""
    if not rows:
        return
    finals = {}
    for r in rows:
        finals[r['trial']] = r                     # rows arrive in order; the last attempt wins

    def agg(label, rs):
        rec = {'attempt': label, 'n': len(rs),
               'converged_frac': float(np.mean([bool(r['converged']) for r in rs]))}
        for d in dims:
            v = np.abs([float(r[f'err_after_{d}']) for r in rs])
            rec[f'mean_abs_{d}'] = float(v.mean())
            rec[f'median_abs_{d}'] = float(np.median(v))
        return rec

    recs = [agg(k, [r for r in rows if r['attempt'] == k])
            for k in sorted({r['attempt'] for r in rows})]
    recs.append(agg('final', list(finals.values())))
    path = os.path.join(out_dir, 'summary.csv')
    with open(path, 'w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    for rec in recs:
        log.info('summary attempt=%-5s n=%-3d converged=%.0f%%  %s', rec['attempt'], rec['n'],
                 100.0 * rec['converged_frac'],
                 '  '.join(f'|{d}| mean {rec[f"mean_abs_{d}"]:.2f} / med {rec[f"median_abs_{d}"]:.2f}'
                           for d in dims))
    log.info('Summary: %s', path)


def build_and_run(cfg, robot, camera, args):
    ev = cfg.section('eval')

    # Build the ESTIMATOR first -- a missing/stale manifold CSV must fail before the robot moves.
    estimator = ManifoldEstimator(cfg.section('estimation'))

    # ---- GROUND TRUTH from the shared catalogue: the part is fixtured between the closed
    # fingers, so frames.yaml's tool0->frame pose IS the true in-hand pose, and its targets:
    # entry is the recorded mate -- the same pair uncertain_sampling uses. ----
    held_name = cfg.get('held_frame')
    if not held_name:
        log.error('held_frame is required -- the catalogue frame IS the ground truth here.')
        return False
    frames = tool_frames.load_frames(cfg)
    targets = tool_frames.load_targets(cfg)
    if held_name not in frames:
        log.error('held_frame %r is not in %s.', held_name, tool_frames.frames_path(cfg))
        return False
    if held_name not in targets:
        log.error('held_frame %r has no targets: entry in %s -- hand-guide to a good mate, '
                  'read `base_link <- %s` off the monitor, and paste it there.',
                  held_name, tool_frames.frames_path(cfg), held_name)
        return False
    T_true = frames[held_name]                     # ground-truth tool0 -> held connector
    T_base_tconn = targets[held_name]              # the recorded mate (base_link <- connector)
    log.info('Ground truth: held frame %r + target from %s.',
             held_name, tool_frames.frames_path(cfg))

    csv_in = urconfig.resolve(cfg, cfg.get('trajectory_csv', 'assembly_trajectory.csv'))
    mats = traj.load_csv(csv_in, angles_deg=bool(cfg.get('trajectory_angles_deg', False)))
    dense = traj.resample(mats, float(cfg.get('translational_resolution_m', 0.001)),
                          float(cfg.get('rotational_resolution_deg', 1.0)))

    # ---- The injected belief errors. Bounds follow uncertain_sampling's convention: ABSOLUTE
    # lower/upper per DOF [x, y, z (m), roll, pitch, yaw (deg)], applied in the HELD part's OWN
    # frame (right-multiplied) -- the same frame the estimator corrects in. 'grid' sweeps every
    # combination at eval.grid_resolution and DERIVES the trial count; 'random' draws uniformly.
    pert = ev.get('perturbation', {}) or {}
    lo = pert.get('lower', [-0.005, 0.0, -0.005, 0.0, -5.0, 0.0])
    hi = pert.get('upper', [0.005, 0.0, 0.005, 0.0, 5.0, 0.0])
    mode = str(ev.get('mode', 'random')).lower()
    if mode not in ('random', 'grid'):
        log.error("eval.mode %r must be 'random' or 'grid'.", mode)
        return False
    try:
        grid = traj.grid_deltas(lo, hi, ev.get('grid_resolution', [0.0] * 6)) \
            if mode == 'grid' else None
    except ValueError as exc:
        log.error('%s', exc)
        return False
    num_trials = len(grid) if grid is not None else int(ev.get('num_trials', 50))
    max_attempts = int(ev.get('max_attempts', 5))
    seed = int(ev.get('random_seed', 0))
    rng = np.random.default_rng(seed if seed > 0 else None)
    tol = ev.get('success_tolerance', {}) or {}
    tol_pos_mm = float(tol.get('pos_mm', 2.0))
    tol_rot_deg = float(tol.get('rot_deg', 3.0))
    stop_conv = bool(ev.get('stop_when_converged', True))
    decim = max(1, int(ev.get('log_decimation', 5)))
    save_obs = bool(ev.get('save_observations', True))
    log.info('%d trials x max %d attempts (%s perturbations, bounds lower=%s upper=%s).',
             num_trials, max_attempts, mode.upper(), list(lo), list(hi))

    # ONE error figure per trial (trial_TTT_errors.png), re-saved after every attempt -- cheap
    # (N files, not N x M), and watchable live during a run.
    save_plots = bool(ev.get('save_plots', True))

    # COMPLIANCE + guard + speeds: same shape as uncertain_sampling; the config mirrors the pick
    # app's assembly values so the estimator sees production-like observations.
    adm = AdmittanceController(robot.arm, cfg.section('compliance'))
    guard = ForceGuard(robot.arm, cfg.section('force_guard'))
    tare = (lambda: robot.arm.zero_ft(settle=False)) \
        if bool(cfg.get_path('compliance.tare_before', True)) else None
    settle_s = float(cfg.get_path('compliance.settle_s', 0.5))
    v_mm_s = float(cfg.get_path('speed.max_cartesian_translation_mm_s', 5.0))
    w_deg_s = float(cfg.get_path('speed.max_cartesian_rotation_deg_s', 6.0))
    rv_mm_s = float(cfg.get_path('speed.retract_translation_mm_s', v_mm_s))
    rw_deg_s = float(cfg.get_path('speed.retract_rotation_deg_s', w_deg_s))
    retract_m = float(cfg.get('retract_distance_m', 0.05))
    min_seg_s = 1.0 / adm.rate

    def seg_time(A, B, v=None, w=None):
        v = v_mm_s if v is None else v
        w = w_deg_s if w is None else w
        lin_m, ang_rad = pose_error(A, B)
        t_lin = (lin_m * 1000.0 / v) if v > 0 else 0.0
        t_ang = (np.degrees(ang_rad) / w) if w > 0 else 0.0
        return max(t_lin, t_ang, min_seg_s)

    out_dir = os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                           f'estimator_eval_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    log.info('Experiment folder: %s', out_dir)
    try:
        import json
        with open(os.path.join(out_dir, 'eval_config.json'), 'w') as fh:
            json.dump({'held_frame': held_name, 'trajectory_csv': csv_in,
                       'num_trials': num_trials, 'eval': ev,
                       'estimation': cfg.section('estimation'),
                       'compliance': cfg.section('compliance'),
                       'force_guard': cfg.section('force_guard')}, fh, indent=2, default=str)
    except Exception as exc:                       # noqa: BLE001
        log.warning('eval_config.json skipped (%s)', exc)
    fout = open(os.path.join(out_dir, 'trials.csv'), 'w', newline='')
    writer = _csv.DictWriter(fout, fieldnames=_fieldnames(estimator.estimate_dims), restval='')
    writer.writeheader()

    # The stand-off = the DISASSEMBLED rest between trials: the trajectory START backed off a
    # further standoff_distance_m, planned from the TRUE held pose -- one fixed physical spot
    # regardless of what the current (wrong) belief says.
    standoff_axis = np.asarray(cfg.get('standoff_axis', [-1, 0, 0]), dtype=float)
    T_standoff_row = translation_matrix(
        standoff_axis * float(cfg.get('standoff_distance_m', 0.02))) @ mats[0]
    T_standoff_ref = T_base_tconn @ T_standoff_row @ inverse(T_true)
    q_home = robot.arm.q()
    q = robot.arm.ik(T_standoff_ref, q_home)
    if q is None or not robot.arm.move_j(q, label='approach standoff'):
        fout.close()
        return False
    seed_q = q

    rows, ok, durations = [], True, []
    try:
        for trial in range(1, num_trials + 1):
            t_trial = time.time()
            # Corrupt the BELIEF only -- the part never moves in the fingers. The robot plans
            # from T_believed, so the true part physically rides the path offset by exactly the
            # injected error: the same situation as a bad grasp, but with the answer known.
            delta = grid[trial - 1] if grid is not None else traj.random_delta(lo, hi, rng)
            T_believed = T_true @ delta
            inj, inj_pos, inj_rot = _gt_error(T_true, T_believed)
            log.info('--- trial %d/%d --- injected belief error xyz=[%+6.2f, %+6.2f, %+6.2f] mm '
                     'rpy=[%+6.2f, %+6.2f, %+6.2f] deg', trial, num_trials, *inj)
            # The trial's error track for the co-plot: index 0 = the injected error, index k =
            # the error left after attempt k's update.
            track6, trackn = [inj], [(inj_pos, inj_rot)]

            for attempt in range(1, max_attempts + 1):
                errb, errb_pos, errb_rot = _gt_error(T_true, T_believed)
                refs = [T_base_tconn @ row @ inverse(T_believed) for row in dense]

                # To the attempt's start -- stiff, free space (the retract/stand-off cleared it).
                q = robot.arm.ik(refs[0], seed_q)
                if q is None or not robot.arm.move_j(
                        q, label=f'trial {trial} attempt {attempt} start'):
                    log.warning('IK/approach failed; abandoning the rest of trial %d.', trial)
                    break
                seed_q = q

                # ASSEMBLE under admittance, collecting observations (same law as the pick app).
                obs, cnt = [], [0]

                def log_cb(_obs=obs, _cnt=cnt, _T=T_believed):
                    _cnt[0] += 1
                    if _cnt[0] % decim == 0:
                        _obs.append(_observe(robot, _T, T_base_tconn))

                adm.reset()
                adm.warmup(refs[0], tare_fn=tare)
                guard.reset()
                last_ref, seated = refs[0], False
                for i in range(1, len(refs)):
                    res = adm.ramp(refs[i - 1], refs[i], seg_time(refs[i - 1], refs[i]),
                                   guard, on_step=log_cb)
                    last_ref = refs[i]
                    if res == 'seated':
                        seated = True
                        log.info('Contact limit at waypoint %d/%d -- stopped advancing.',
                                 i, len(refs) - 1)
                        break
                adm.hold(last_ref, settle_s, guard, on_step=log_cb)

                # Kinematic check numbers, for the record only -- the GROUND-TRUTH error below is
                # the metric this app exists for.
                lin, ang = pose_error(robot.tool0() @ T_believed, T_base_tconn)

                # RETRACT: compliant, UN-guarded escape along the believed part's own -X (a seated
                # part is already over the guard limit; a guarded retract would block itself).
                T_out = _retract_ref(last_ref, T_believed, retract_m)
                adm.ramp(last_ref, T_out, seg_time(last_ref, T_out, rv_mm_s, rw_deg_s), guard=None)
                adm.stop()

                if save_obs:
                    _save_observations(os.path.join(
                        out_dir, f'trial_{trial:03d}_attempt_{attempt:02d}_observations.csv'), obs)

                # ESTIMATE -- always, even on the last attempt: the estimate IS the thing under
                # test, so every attempt's observations get scored.
                obs_arr = np.asarray(obs, dtype=float)
                vec6, w6 = estimator.prepare_observations(
                    obs_arr[:, :6], obs_arr[:, 6:9], obs_arr[:, 9:12]) if len(obs) else \
                    (np.zeros((0, 6)), np.zeros((0, 6)))
                T_corr_mm, info = estimator.estimate(vec6, w6)
                row = {'trial': trial, 'attempt': attempt, 'n_observations': len(obs),
                       'seated': seated, 'check_pos_mm': lin * 1000.0,
                       'check_rot_deg': float(np.degrees(ang)),
                       'err_before_pos_mm': errb_pos, 'err_before_rot_deg': errb_rot}
                row.update({f'inj_{s}': v for s, v in zip(_ERR, inj)})
                row.update({f'err_before_{s}': v for s, v in zip(_ERR, errb)})
                if T_corr_mm is None:
                    log.warning('Estimation skipped (%s) -- belief unchanged.', info)
                    row['estimate'] = f'skipped: {info}'
                else:
                    T_believed = T_believed @ _corr_to_m(T_corr_mm)   # believed @ corr ~= true
                    row['estimate'] = 'ok'
                    row.update({f'corr_{k}': v for k, v in info['theta_corr'].items()})
                    row.update({'icp_inliers': info['inliers'],
                                'icp_residual': info['final_residual']})
                erra, erra_pos, erra_rot = _gt_error(T_true, T_believed)
                track6.append(erra)
                trackn.append((erra_pos, erra_rot))
                if save_plots:                     # re-saved after EVERY attempt of this trial
                    _plot_trial_errors(os.path.join(out_dir, f'trial_{trial:03d}_errors.png'),
                                       trial, estimator.estimate_dims, track6, trackn,
                                       tol_pos_mm, tol_rot_deg)
                row.update({f'err_after_{s}': v for s, v in zip(_ERR, erra)})
                row.update({'err_after_pos_mm': erra_pos, 'err_after_rot_deg': erra_rot})
                row['converged'] = bool(erra_pos <= tol_pos_mm and erra_rot <= tol_rot_deg)
                log.info('trial %d attempt %d: gt error %.2f mm / %.2f deg -> %.2f mm / %.2f deg'
                         '%s', trial, attempt, errb_pos, errb_rot, erra_pos, erra_rot,
                         '  (CONVERGED)' if row['converged'] else '')
                rows.append(row)
                writer.writerow(row)
                fout.flush()                       # a 50-trial run must survive an abort mid-way
                os.fsync(fout.fileno())
                if row['converged'] and stop_conv:
                    break

            # DISASSEMBLE between trials: back to the fixed stand-off (free space) so every trial
            # starts from the same physically-clear spot, whatever the last belief was.
            q = robot.arm.ik(T_standoff_ref, seed_q)
            if q is None or not robot.arm.move_j(q, label='standoff (disassembled)'):
                log.error('Could not return to the stand-off; stopping.')
                ok = False
                break
            seed_q = q
            durations.append(time.time() - t_trial)
            mean_s = sum(durations) / len(durations)
            log.info('trial %d/%d took %.1f s | mean %.1f s | %d left, ETA %s (done ~%s)',
                     trial, num_trials, durations[-1], mean_s, num_trials - trial,
                     _fmt_dur(mean_s * (num_trials - trial)),
                     _clock(mean_s * (num_trials - trial)))
    except Exception:                              # noqa: BLE001
        ok = False
        log.exception('Evaluation error:')
    finally:
        robot.arm.servo_stop()
        fout.close()
        _write_summary(out_dir, rows, estimator.estimate_dims)
    if ok:
        robot.arm.move_j(q_home, label='home')
        log.info('Evaluation complete: %s (%d attempt rows, %d trials)',
                 out_dir, len(rows), num_trials)
    return ok


def main():
    # with_gripper=False: the connector is FIXTURED between the closed fingers -- the gripper is
    # never constructed, so nothing here can open it and drop the part.
    run_app('In-hand estimator evaluation (known ground truth)', 'estimator_eval', build_and_run,
            with_gripper=False)


if __name__ == '__main__':
    main()
