"""BNC ASSEMBLY -- cable_pick_estimate_assemble's pipeline with estimator_eval's estimator.

Same shape as cable_pick_estimate_assemble (scan, grasp, slip-checked lift, stand-off, then an
assemble/estimate loop that corrects the in-hand belief), but everything downstream of the
grasp is estimator_eval's, because that is where the 2026-08 BNC campaign did its tuning:

  * ESTIMATOR: the same `estimation:` block, including `commit: argmin` (take the dense
    landscape's minimum and skip the multi-start ICP entirely), the per-candidate wrench
    re-basing, per-block kNN bandwidths and the per-channel weights. `_argmin_estimate` and
    `_landscape` are IMPORTED from estimator_eval rather than reimplemented, so the two apps
    cannot drift.
  * COLLECTION: `assembly.collection.mode` = attempts | offset_sweep | peck, with the same
    semantics -- offset_sweep commands one insertion per deliberate offset with the belief held
    FIXED across passes (so their evidence fuses exactly), peck keeps advancing past each force
    stop. Deeper contact was the strongest measured predictor of a good estimate.
  * COMPLIANCE / GUARD: read from the TOP-LEVEL `compliance:` and `force_guard:` blocks, the
    same place and the same names estimator_eval uses, so tuned values copy across verbatim.
  * FINAL INSERTION: `assembly.final_insertion`, the same grouped block -- stiffness, mass,
    damping, settle, dwell and the guard overrides all in one place, each inheriting the shared
    value when null. This is the attempt meant to SEAT, so it gets its own physics.

WHAT IS NECESSARILY DIFFERENT. estimator_eval fixtures the part and injects a KNOWN belief
error, so it can score every estimate against ground truth. Here the part is really picked and
the true in-hand pose is unknown: there is no injected error, no err_before/after, and no
truth to draw. Success is the OPERATOR's call at the check (a dry run falls back to the
kinematic tolerance), exactly as in cable_pick_estimate_assemble. The match diagnostics still
render -- with the committed estimate marked and no truth line.

Units: robot poses are metres/radians; the manifold space is mm/deg. The conversion happens
only at the observation/correction boundary, as in the app this is derived from.
"""

import csv as _csv
import os
from datetime import datetime

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from .. import tool_frames
from ..log import StepRunner
from ..robot import AdmittanceController, ForceGuard
from ..skills import manifold_debug, reset
from ..skills import trajectory as traj
from ..skills.manifold import mats_from_vec6, vec6_from_mats
from ..skills.pick import (GraspCheck, GraspController, GraspGeometry, GraspImageRecorder,
                           GraspRecovery, retry_offset_x, verify_cable_held)
from ..skills.solution_check import CheckedManifoldEstimator
from ..transforms import from_cfg, inverse, matrix_to_xyzrpy, pose_error, translation_matrix
from ._cable import build_scanner, make_confirm
from ._runner import run_app
from .cable_pick_assemble import _guarded, _pick
from .cable_pick_estimate_assemble import (_corr_to_m, _observe, _plot_run, _save_observations)
from .estimator_eval import _argmin_estimate, _landscape
from .uncertain_sampling import _retract_ref

log = urlog.get('bnc-assembly')


def build_and_run(cfg, robot, camera, args):
    a = cfg.section('assembly')

    # The ESTIMATOR first: a missing/stale manifold CSV or a bad metric key must fail before
    # the robot moves, not after a part is in the fingers.
    estimator = CheckedManifoldEstimator(cfg.section('estimation'))
    commit = str(cfg.get_path('estimation.commit', 'aggregator')).strip().lower()
    if commit not in ('aggregator', 'argmin'):
        log.error("estimation.commit %r must be 'aggregator' or 'argmin'.", commit)
        return False
    log.info('Estimator: commit %s, dims %s, manifold %s.', commit, estimator.estimate_dims,
             cfg.get_path('estimation.manifold_csv'))

    scanner, _detector, _est = build_scanner(cfg, robot, camera)
    geom, check = GraspGeometry(cfg), GraspCheck(cfg)
    recovery, grasp = GraspRecovery(cfg), GraspController(cfg)
    recorder = GraspImageRecorder(cfg)
    confirm = make_confirm(cfg)

    # COMPLIANCE + GUARD from the top-level blocks -- same names estimator_eval reads, so a
    # tuned pair of blocks can be copied between the two configs without translation.
    adm = AdmittanceController(robot.arm, cfg.section('compliance'))
    guard_shared = ForceGuard(robot.arm, cfg.section('force_guard'))
    tare = (lambda: robot.arm.zero_ft(settle=False)) \
        if bool(cfg.get_path('compliance.tare_before', True)) else None
    settle_shared = float(cfg.get_path('compliance.settle_s', 0.5))
    hold_shared = float(cfg.get_path('compliance.hold_after_insertion_s', 0.0))

    # ---- FINAL INSERTION: one grouped block, each key inheriting the shared value when null.
    fi = a.get('final_insertion', {}) or {}
    fi_on = bool(fi.get('enabled', True))
    fi_settle = None if fi.get('settle_s') is None else float(fi['settle_s'])
    fi_hold = None if fi.get('hold_after_insertion_s') is None \
        else float(fi['hold_after_insertion_s'])
    # Pacing, dwell and noise for the COMMIT, in the same grouped block. The commit is
    # deliberately ZERO-NOISE by default -- the jitter exists to gather varied contact while
    # probing and has no place in the attempt meant to seat -- so enabling
    # final_insertion.trajectory_noise is an explicit opt-out of that.
    fi_v = None if fi.get('speed_translation_mm_s') is None \
        else float(fi['speed_translation_mm_s'])
    fi_wr = None if fi.get('speed_rotation_deg_s') is None \
        else float(fi['speed_rotation_deg_s'])
    fi_pause = float(fi.get('pause_s', 0.0) or 0.0)
    fin = fi.get('trajectory_noise', {}) or {}
    fi_noise_on = bool(fin.get('enabled', False))
    fi_noise_std = [float(v) for v in (fin.get('std') or [0.0] * 6)]
    fi_noise_w = max(1, int(fin.get('smooth_window', 10)))
    if fi_noise_on and len(fi_noise_std) != 6:
        log.error('final_insertion.trajectory_noise.std must have 6 entries.')
        return False
    for nm, v in (('speed_translation_mm_s', fi_v), ('speed_rotation_deg_s', fi_wr),
                  ('pause_s', fi_pause)):
        if v is not None and v < 0:
            log.error('assembly.final_insertion.%s must be >= 0 (got %.2f).', nm, v)
            return False
    adm_final, guard_final = adm, None
    if fi_on:
        comp_final = dict(cfg.section('compliance'))
        for k in ('stiffness', 'mass', 'damping_ratio'):
            if fi.get(k) is not None:
                comp_final[k] = [float(v) for v in fi[k]]
        adm_final = AdmittanceController(robot.arm, comp_final)
        gsec = dict(cfg.section('force_guard'))
        over_g = {k: fi[k] for k in ('max_force_n', 'max_torque_nm', 'persistence_s')
                  if fi.get(k) is not None}
        guard_final = ForceGuard(robot.arm, {**gsec, **over_g}) if over_g else None
        log.info('Final insertion ON: stiffness %s, guard %.0f N.', comp_final.get('stiffness'),
                 float(over_g.get('max_force_n', gsec.get('max_force_n', 0.0))))

    # ---- Target from the SHARED catalogue (the same record estimator_eval assembles to) ----
    tname = a.get('target_frame')
    targets = tool_frames.load_targets(cfg)
    if not tname or tname not in targets:
        log.error('assembly.target_frame %r needs a targets: entry in %s.',
                  tname, tool_frames.frames_path(cfg))
        return False
    T_base_tconn = targets[tname]
    csv_in = urconfig.resolve(cfg, a.get('trajectory_csv', 'assembly_trajectory.csv'))
    mats = traj.load_csv(csv_in, angles_deg=bool(a.get('trajectory_angles_deg', False)))
    dense = traj.resample(mats, float(a.get('translational_resolution_m', 0.001)),
                          float(a.get('rotational_resolution_deg', 1.0)))

    # ---- COLLECTION MODE (estimator_eval.eval.collection semantics) ----
    col = a.get('collection', {}) or {}
    col_mode = str(col.get('mode', 'attempts')).strip().lower()
    if col_mode not in ('attempts', 'offset_sweep', 'peck'):
        log.error("assembly.collection.mode %r must be 'attempts', 'offset_sweep' or 'peck'.",
                  col_mode)
        return False
    sweep_offsets = col.get('sweep_offsets')
    if sweep_offsets is None:
        sweep_offsets = [[0.0, 0.0, 0.0, 0.0, float(p), 0.0]
                         for p in np.arange(-4.0, 4.01, 2.0)]
    sweep_offsets = [[float(v) for v in o] for o in sweep_offsets]
    if col_mode == 'offset_sweep' and (not sweep_offsets
                                       or any(len(o) != 6 for o in sweep_offsets)):
        log.error('assembly.collection.sweep_offsets must be a non-empty list of 6-vectors '
                  '[x, y, z (m), roll, pitch, yaw (deg)].')
        return False
    peck_back_m = float(col.get('peck_retract_mm', 5.0)) / 1000.0
    peck_timeout = float(col.get('peck_timeout_s', 30.0))
    if col_mode == 'offset_sweep':
        log.info('Collection: offset_sweep, %d passes per attempt (pitch %s).',
                 len(sweep_offsets), [round(o[4], 1) for o in sweep_offsets])
    elif col_mode == 'peck':
        log.info('Collection: peck, %.1f mm back-off per stop, %.0f s budget.',
                 peck_back_m * 1000.0, peck_timeout)

    # ---- Trajectory noise (same semantics as estimator_eval.eval.trajectory_noise) ----
    tn = a.get('trajectory_noise', {}) or {}
    tn_on = bool(tn.get('enabled', False))
    tn_std = tn.get('std') or [0.0] * 6
    tn_std = [float(v) for v in tn_std]
    if len(tn_std) != 6:
        log.error('assembly.trajectory_noise.std must have 6 entries.')
        return False
    tn_w = max(1, int(tn.get('smooth_window', 10)))
    tn_da, tn_dt = float(tn.get('noise_decay_attempt', 0.0)), \
        float(tn.get('noise_decay_traj', 0.0))
    noise_rng = np.random.default_rng()

    spd = cfg.section('speed')
    scales = spd.get('phase_scale', {}) or {}

    def phase(name):
        robot.arm.set_speed_scale(float(scales.get(name, 1.0)), name)

    g_v = float(spd.get('max_cartesian_translation_mm_s', 3.5))
    g_w = float(spd.get('max_cartesian_rotation_deg_s', 5.0))
    s_asm = float(scales.get('assemble', 1.0))
    s_ret = float(scales.get('retract', 1.0))
    min_seg_s = 1.0 / adm.rate

    def seg_time(A, B, v=None, w=None):
        v = g_v * s_asm if v is None else v
        w = g_w * s_asm if w is None else w
        lin_m, ang_rad = pose_error(A, B)
        return max((lin_m * 1000.0 / v) if v > 0 else 0.0,
                   (np.degrees(ang_rad) / w) if w > 0 else 0.0, min_seg_s)

    retract_m = float(a.get('retract_distance_m', 0.05))
    decim = max(1, int(a.get('log_decimation', 5)))
    tol = a.get('success_tolerance', {}) or {}
    tol_pos_m, tol_rot_rad = float(tol.get('pos_mm', 2.0)) / 1000.0, \
        np.radians(float(tol.get('rot_deg', 3.0)))
    max_attempts = int(a.get('max_attempts', 5))
    accumulate = bool(a.get('accumulate_observations', True))
    dbg = a.get('debug_match', {}) or {}
    dbg_on = bool(dbg.get('enabled', False))
    dbg_live = dbg.get('live', True)
    dbg_live = (os.path.join(cfg.get('data_dir', 'data'), 'experiments')
                if dbg_live is True else (dbg_live or None))

    init = cfg.get_path('estimation.initial_connector_in_fingertip')
    T_ftip_conn = from_cfg(init) if init else from_cfg(cfg.section('junction_in_fingertip'))

    live = a.get('live_plot', True)
    live_path = None
    if live:
        live_path = live if isinstance(live, str) else os.path.join(
            cfg.get('data_dir', 'data'), 'experiments', 'bnc_assembly_live.png')
        os.makedirs(os.path.dirname(live_path) or '.', exist_ok=True)
    out_dir = os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                           f'bnc_assembly_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    log.info('Experiment folder: %s', out_dir)

    def run_insertion(adm_ctl, refs, T_tool0_conn, peck=False, guard_ctl=None, settle=None,
                      hold=None, speed=None, pause=None):
        """One admittance insertion along refs, collecting observations; then the compliant
        UN-GUARDED retract along the believed connector's own -X. Mirrors estimator_eval's
        run_insertion, including peck (a force stop backs off and advances again rather than
        ending the pass) and the un-guarded post-insertion dwell."""
        import time as _time
        guard = guard_ctl if guard_ctl is not None else guard_shared
        settle_s = settle if settle is not None else settle_shared
        hold_s = hold if hold is not None else hold_shared
        sv, sw = speed if speed is not None else (None, None)
        obs, cnt = [], [0]

        def log_cb():
            cnt[0] += 1
            if cnt[0] % decim == 0:
                obs.append(_observe(robot, T_tool0_conn, T_base_tconn))

        adm_ctl.reset()
        adm_ctl.warmup(refs[0], tare_fn=tare)
        guard.reset()
        last_ref, i, t0, stops = refs[0], 1, _time.time(), []
        prev = refs[0]
        while i < len(refs):
            res = adm_ctl.ramp(prev, refs[i], seg_time(prev, refs[i], sv, sw), guard,
                               on_step=log_cb)
            last_ref = refs[i]
            if res == 'seated':
                stops.append(float(matrix_to_xyzrpy(
                    inverse(T_base_tconn) @ robot.tool0() @ T_tool0_conn)[0][0] * 1000.0))
                if not peck or _time.time() - t0 > peck_timeout:
                    break
                # back off along the BELIEVED connector's -X, un-guarded (we are at the limit
                # by construction), then keep advancing from where we are.
                T_out = _retract_ref(last_ref, T_tool0_conn, peck_back_m)
                adm_ctl.ramp(last_ref, T_out, seg_time(last_ref, T_out, g_v * s_ret,
                                                       g_w * s_ret), guard=None,
                             on_step=log_cb)
                guard.reset()
                prev = T_out
                i += 1
                continue
            prev = refs[i]
            i += 1
        adm_ctl.hold(last_ref, settle_s, guard, on_step=log_cb)
        if hold_s > 0:                                 # dwell: un-guarded, unlogged
            adm_ctl.hold(last_ref, hold_s, guard=None)
        if pause:
            # PAUSE: servo released, the arm just stands still. Unlike the dwell above it
            # applies no force -- inspection/measurement time, not press time.
            adm_ctl.stop()
            log.info('   pausing %.1f s at the seat (servo stopped).', float(pause))
            _time.sleep(float(pause))
        lin, ang = pose_error(robot.tool0() @ T_tool0_conn, T_base_tconn)
        T_out = _retract_ref(last_ref, T_tool0_conn, retract_m)
        adm_ctl.ramp(last_ref, T_out, seg_time(last_ref, T_out, g_v * s_ret, g_w * s_ret),
                     guard=None)
        robot.arm.servo_stop()
        return obs, lin, ang, stops

    # ---- RESET + PICK + slip-checked LIFT (identical to cable_pick_estimate_assemble) ----
    phase('reset')
    if not reset.reset_robot(robot, cfg, 'start reset'):
        return False
    q_home = robot.arm.q()
    attempt = 0
    runner = StepRunner(log, confirm=confirm is not None)
    while True:
        phase('scan')
        result = _pick(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder,
                       offset_x_m=retry_offset_x(attempt, check.retry_perturb_x_m))
        if result == 'ok':
            status = {}

            def do_lift(_s=status):
                _s['r'] = grasp.lift_verified(
                    robot, geom, check, 'lift',
                    position_guard=lambda mv: _guarded(robot, guard_shared, mv))
                return _s['r'] == 'ok'

            if runner.run([('lift (slip-checked)', do_lift)]):
                break
            result = status.get('r')
            if result != 'slipped':
                return False
        if result == 'abort':
            return False
        if attempt >= check.max_retries:
            log.error('Grasp failed on all %d attempts; aborting.', check.max_retries + 1)
            return False
        attempt += 1
        log.warning('Grasp %s -- recovering (attempt %d/%d).', result, attempt + 1,
                    check.max_retries + 1)
        if result == 'slipped' and hasattr(scanner, 'reselect'):
            phase('scan')
            T_up = translation_matrix([0.0, 0.0, check.slip_raise_m]) @ robot.tool0()
            if not (robot.gripper.open('drop')
                    and _guarded(robot, guard_shared,
                                 lambda: robot.arm.move_l(T_up, label='slip recovery (up)'))):
                return False
            scanner.reselect()
        else:
            phase('reset')
            if not (robot.gripper.open('drop') and robot.arm.move_j(q_home, label='home')):
                return False

    # ---- Stand-off, held check, and the unconditional human gate before contact ----
    st = a.get('standoff', {}) or {}
    T_standoff_row = translation_matrix(
        np.asarray(st.get('axis', [-1, 0, 0]), dtype=float)
        * float(st.get('distance_m', 0.01))) @ mats[0]

    def tool0_ref(row, T_tool0_conn):
        return T_base_tconn @ row @ inverse(T_tool0_conn)

    phase('standoff')
    seed_q = robot.arm.q()
    T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
    q = robot.arm.ik(tool0_ref(T_standoff_row, T_tool0_conn), seed_q)
    if q is None or not _guarded(robot, guard_shared,
                                 lambda: robot.arm.move_j(q, label='stand-off')):
        return False
    seed_q = q
    if not verify_cable_held(robot, check, 'stand-off'):
        return False
    if not robot.arm.dry_run:
        try:
            ans = input('\n[stand-off] Ready to ASSEMBLE (contact ahead). '
                        'Enter to continue (q to abort): ')
        except EOFError:
            ans = ''
        if ans.strip().lower() in ('q', 'quit', 'n', 'no'):
            log.info('Aborted at the stand-off by the user.')
            return False

    # ---- The collect / estimate / update loop ----
    est_rows, success = [], False
    acc = np.zeros((0, 12))
    T_cum = np.eye(4)
    trackc, trackr, trackg = [np.zeros(len(estimator.estimate_dims))], [], []
    try:
        for it in range(1, max_attempts + 1):
            T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
            e_xyz, e_rpy = matrix_to_xyzrpy(T_ftip_conn)
            log.info('--- attempt %d/%d --- in-hand estimate xyz=%s mm rpy=%s deg', it,
                     max_attempts, np.round(e_xyz * 1000, 2).tolist(),
                     np.round(np.degrees(e_rpy), 2).tolist())

            # COLLECTION PASSES: the sweep commands one insertion per deliberate offset with
            # the belief held FIXED across passes, so their evidence fuses exactly.
            passes = ([list(o) for o in sweep_offsets] if col_mode == 'offset_sweep'
                      else [None])
            obs, lin, ang = [], 0.0, 0.0
            for pi, poff in enumerate(passes):
                if tn_on or poff is not None:
                    rows_t = traj.noised(dense, noise_rng, tn_std if tn_on else [0.0] * 6,
                                         tn_w, tn_dt, (1.0 - tn_da) ** (it - 1), poff)
                else:
                    rows_t = dense
                refs = [tool0_ref(row, T_tool0_conn) for row in rows_t]
                phase('standoff')
                label = (f'attempt {it}'
                         + (f' sweep {pi + 1}/{len(passes)}' if poff is not None else '')
                         + ' start')
                q = robot.arm.ik(refs[0], seed_q)
                if q is None or not _guarded(robot, guard_shared,
                                             lambda: robot.arm.move_j(q, label=label)):
                    log.error('Could not reach the pass start; aborting.')
                    return False
                seed_q = q
                phase('assemble')
                obs_i, lin, ang, stops = run_insertion(adm, refs, T_tool0_conn,
                                                       peck=(col_mode == 'peck'))
                obs.extend(obs_i)
                if poff is not None:
                    log.info('  sweep %d/%d (pitch %+.1f deg, z %+.1f mm): %d obs, stop %s mm.',
                             pi + 1, len(passes), poff[4], poff[2] * 1000.0, len(obs_i),
                             [round(s, 1) for s in stops])
                if not verify_cable_held(robot, check, f'attempt {it} pass {pi + 1}'):
                    return False

            log.info('check: believed connector vs target: %.2f mm, %.2f deg (tol %.2f mm, '
                     '%.2f deg)', lin * 1000, np.degrees(ang), tol_pos_m * 1000,
                     np.degrees(tol_rot_rad))
            _save_observations(os.path.join(out_dir, f'attempt_{it:02d}_observations.csv'), obs)
            row = {'attempt': it, 'n_observations': len(obs), 'n_passes': len(passes),
                   'check_pos_mm': lin * 1000.0, 'check_rot_deg': float(np.degrees(ang))}

            # SUCCESS is the operator's call -- they can see the physical mate; the kinematic
            # numbers only see the belief. A dry run has no operator.
            if robot.arm.dry_run:
                row['success'] = bool(lin <= tol_pos_m and ang <= tol_rot_rad)
            else:
                try:
                    ans = input(f'[check attempt {it}] Was the assembly SUCCESSFUL? '
                                '(y = done / Enter = retry / q = abort): ').strip().lower()
                except EOFError:
                    ans = ''
                if ans in ('q', 'quit'):
                    est_rows.append(row)
                    return False
                row['success'] = ans in ('y', 'yes')
            if row['success']:
                est_rows.append(row)
                log.info('ASSEMBLY COMPLETE on attempt %d.', it)
                success = True
                break
            if it == max_attempts:
                est_rows.append(row)
                log.error('Attempt limit reached (%d) without a successful mate.', max_attempts)
                break

            # ---- ESTIMATE. Accumulated rows are re-projected into the belief after every
            # update, exactly as in estimator_eval, so old evidence stays valid.
            obs_arr = np.asarray(obs, dtype=float).reshape(-1, 12)
            full = np.vstack([acc, obs_arr]) if accumulate and len(acc) else obs_arr
            vec6, w6 = estimator.prepare_observations(full[:, :6], full[:, 6:9], full[:, 9:12])
            if commit == 'argmin':
                T_corr_mm, info, land_pack = _argmin_estimate(estimator, vec6, w6)
            else:
                T_corr_mm, info = estimator.estimate(vec6, w6)
                land_pack = None
            if T_corr_mm is None:
                log.warning('Estimation skipped (%s) -- retrying with the UNCHANGED belief.',
                            info)
                row['estimate'] = f'skipped: {info}'
                est_rows.append(row)
                trackc.append(trackc[-1])
                trackr.append(float('nan'))
                trackg.append(np.zeros(0))
                if accumulate:
                    acc = full
                continue
            log.info('belief correction: %s  (residual %.3f, %d obs, %s)',
                     {k: round(v, 3) for k, v in info['theta_corr'].items()},
                     info['final_residual'], info['n_observations'], commit)
            if dbg_on:
                try:
                    manifold_debug.figures(
                        estimator, vec6, w6, dict(info['theta_corr']), None,
                        os.path.join(out_dir, f'attempt_{it:02d}_match.png'),
                        title=f'attempt {it} (no ground truth)',
                        max_rows=int(dbg.get('max_rows', 250)),
                        grid_n=int(dbg.get('grid_points', 41)), live_dir=dbg_live)
                except Exception as exc:               # noqa: BLE001 -- never fatal
                    log.warning('match diagnostics skipped (%s)', exc)

            T_ftip_conn = T_ftip_conn @ _corr_to_m(T_corr_mm)      # believed @ corr ~= true
            T_cum = T_cum @ np.asarray(T_corr_mm, dtype=float)
            if accumulate:
                from .estimator_eval import _rebase_rows
                acc = _rebase_rows(full, T_corr_mm) if len(full) else full
            trackc.append(vec6_from_mats(T_cum)[estimator.idx])
            trackr.append(float(info['final_residual']))
            trackg.append(np.asarray(info['res_hist'], dtype=float)[:, -1]
                          if info.get('res_hist') is not None else np.zeros(0))
            _plot_run(os.path.join(out_dir, 'run_corrections.png'), estimator.estimate_dims,
                      trackc, trackr, trackg,
                      f'attempt {it}/{max_attempts} | check {lin * 1000:.1f} mm', live_path)
            row.update({f'corr_{k}': v for k, v in info['theta_corr'].items()})
            row.update({'icp_residual': info['final_residual'], 'commit': commit})
            est_rows.append(row)

        # ---- FINAL INSERTION -- the COMMIT, from the final corrected belief, zero noise ----
        if fi_on and not success:
            log.info('FINAL INSERTION from the corrected belief (zero noise).')
            T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
            rows_f = (traj.noised(dense, noise_rng, fi_noise_std, fi_noise_w, 0.0, 1.0)
                      if fi_noise_on else dense)
            refs = [tool0_ref(row_, T_tool0_conn) for row_ in rows_f]
            phase('standoff')
            q = robot.arm.ik(refs[0], seed_q)
            if q is None or not _guarded(robot, guard_shared,
                                         lambda: robot.arm.move_j(q, label='final start')):
                log.warning('IK/approach failed for the final insertion.')
            else:
                seed_q = q
                phase('assemble')
                obs_f, lin, ang, _ = run_insertion(
                    adm_final, refs, T_tool0_conn, guard_ctl=guard_final, settle=fi_settle,
                    hold=fi_hold, speed=(fi_v, fi_wr) if (fi_v or fi_wr) else None,
                    pause=fi_pause)
                _save_observations(os.path.join(out_dir, 'final_insertion_observations.csv'),
                                   obs_f)
                frow = {'attempt': 'final_insertion', 'n_observations': len(obs_f),
                        'check_pos_mm': lin * 1000.0,
                        'check_rot_deg': float(np.degrees(ang))}
                if not robot.arm.dry_run:
                    try:
                        ansf = input('[final insertion] SUCCESSFUL? (y/n): ').strip().lower()
                    except EOFError:
                        ansf = ''
                    frow['success'] = ansf in ('y', 'yes')
                    success = success or frow['success']
                est_rows.append(frow)
    finally:
        robot.arm.servo_stop()
        if est_rows:
            keys = sorted({k for r in est_rows for k in r}, key=str)
            with open(os.path.join(out_dir, 'estimates.csv'), 'w', newline='') as fh:
                w = _csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(est_rows)
            log.info('Per-attempt log: %s', os.path.join(out_dir, 'estimates.csv'))

    if not success:
        return False
    d_out = float(a.get('release_retract_distance_m', 0.08))
    back = -(robot.tool0() @ (robot.T_tool0_fingertip @ T_ftip_conn))[:3, 0] * d_out

    def release_escape():
        return _guarded(robot, guard_shared,
                        lambda: robot.arm.move_l(translation_matrix(back) @ robot.tool0(),
                                                 label='retract (connector -X)'))

    phase('retract')
    ok = runner.run([('open gripper (release)', robot.gripper.open),
                     ('retract (connector -X)', release_escape)])
    phase('reset')
    return ok and reset.reset_robot(robot, cfg, 'end reset')


def main():
    run_app('BNC assembly: pick + estimator_eval-tuned manifold estimation',
            'bnc_assembly', build_and_run, needs_camera=True)


if __name__ == '__main__':
    main()
