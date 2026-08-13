"""Cable pick, then ESTIMATE-while-ASSEMBLING -- cable_pick_assemble + the contact manifold.

The PICK is exactly the cable_pick_assemble pipeline (scan, grasp, check, recovery). The assembly
differs: the robot KNOWS the target connector pose (assembly.target_frame names a
configs/frames.yaml frame whose targets: entry is the recorded mate -- the same catalogue
uncertain_sampling assembles to) and holds an
ESTIMATE of the connector-in-hand (fingertip -> connector, initialised from the grasp geometry),
but that estimate carries in-hand error. Each attempt runs the assembly trajectory under software
admittance exactly like uncertain_sampling -- following the (believed) path, yielding to contact,
LOGGING observations (believed connector-wrt-target pose + wrench in the believed connector frame).

    [pick] -> lift (slip-checked) -> stand-off (held check) ->
        LOOP (max assembly.max_attempts):
            assemble (admittance, guarded, observing)
            check    (believed connector pose vs target, success_tolerance)
              -> within tolerance: release, retract, done
            retract  (linear, back along the connector's own -X -- peg-in-hole assumption)
            held check (re-close + counts: the insertion/retract can strip the part out)
            estimate (ICP of the observations against the CONTACT MANIFOLD -- skills/manifold.py,
                      the same algorithm analysis/manifold_icp_validation.py validates offline)
            update   (T_fingertip_connector <- T_fingertip_connector @ T_corr)
            realign  (recompute the trajectory references from the new estimate)

WHY THE CHECK WORKS: the check uses the BELIEVED pose, but under admittance a wrong belief cannot
fake success -- if the part jams short of the mate, the arm DEFLECTS off the reference, the actual
tool0 (and therefore the believed connector pose) lags the target, and the check fails. The failed
attempt's observations are exactly what the manifold estimator needs to correct the belief.

Units: robot poses are metres/radians (repo convention); the manifold space is mm/deg -- the
conversions happen only at the observation/correction boundary in this file.
"""

import csv as _csv
import itertools
import os
from datetime import datetime

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from .. import tool_frames
from ..log import StepRunner
from ..robot import AdmittanceController, ForceGuard
from ..skills import reset
from ..skills import trajectory as traj
from ..skills.manifold import (FORCE_COLS, ManifoldEstimator, POSE_COLS, TORQUE_COLS,
                               vec6_from_mats)
from ..skills.pick import (GraspCheck, GraspController, GraspGeometry, GraspImageRecorder,
                           GraspRecovery, retry_offset_x, verify_cable_held)
from ..transforms import from_cfg, inverse, matrix_to_xyzrpy, pose_error, translation_matrix
from ._cable import build_scanner, make_confirm
from ._runner import run_app
from .cable_pick_assemble import _guarded, _pick
from .uncertain_sampling import _retract_ref

log = urlog.get('cable-est-assemble')


def _corr_to_m(T_corr_mm):
    """The estimator's correction (translation in mm) -> a metre-based transform."""
    T = np.array(T_corr_mm, dtype=float)
    T[:3, 3] /= 1000.0
    return T


def _observe(robot, T_tool0_conn, T_base_tconn):
    """One observation row: believed connector-wrt-target [mm, deg 6-vec] + raw wrench in the
    believed connector frame [N, Nm]."""
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


def _plot_estimate(path, dims, info):
    """Per-attempt convergence figure, same layout as analysis/manifold_icp_validation: one panel
    per estimated dim (correction vs ICP iteration -- every guess faint, RANSAC consensus bold,
    dashed zero) plus the log-scale NN residual. BEST-EFFORT: a plotting problem (e.g. seaborn not
    installed on the robot box) is logged and skipped, never allowed to kill a hardware run."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        th, inl = info['theta_hist'], info['inlier_mask']
        res = np.maximum(info['res_hist'], 1e-6)
        sns.set_theme(style='whitegrid')
        fig, axes = plt.subplots(len(dims) + 1, 1, figsize=(9.0, 2.6 * (len(dims) + 1)),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        it = np.arange(th.shape[1])
        for j, (ax, dim) in enumerate(zip(axes[:-1], dims)):
            unit = 'deg' if dim.endswith('_deg') else 'mm'
            ax.axhline(0.0, ls='--', lw=1.0, color='#888888', zorder=1)
            for g in range(th.shape[0]):
                ax.plot(it, th[g, :, j], color='#4C72B0', alpha=0.07, lw=1.0, zorder=2)
            ax.plot(it, th[inl, :, j].mean(axis=0), color='#DD8452', lw=2.4, zorder=3)
            lim = max(float(np.abs(th[..., j]).max()), 1e-3) * 1.05
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(f'{dim} corr [{unit}]')
            ax.set_title(f'correction[{dim}] = {info["theta_corr"][dim]:+.3f} {unit}',
                         fontsize=10, loc='left')
        ax = axes[-1]
        it_r = np.arange(1, res.shape[1] + 1)
        for g in range(res.shape[0]):
            ax.plot(it_r, res[g], color='#4C72B0', alpha=0.07, lw=1.0, zorder=2)
        ax.plot(it_r, res[inl].mean(axis=0), color='#DD8452', lw=2.4, zorder=3)
        ax.set_yscale('log')
        ax.set_ylabel('mean NN residual [mm-eq]')
        ax.set_xlabel('ICP iteration')
        fig.suptitle(f'belief correction ({info["inliers"]}/{info["guesses"]} inliers, '
                     f'residual {info["final_residual"]:.3f})', y=0.995)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as exc:                       # noqa: BLE001 -- plotting is never fatal
        log.warning('estimate plot skipped (%s)', exc)


def _plot_run(path, dims, corr_track, res_agg, res_all, status=None, live_path=None):
    """RUN-LEVEL figure, RE-SAVED after every estimate (+ optional atomic LIVE copy, like
    estimator_eval's live figure). No ground truth exists after a real pick, so the tracks show
    the BELIEF's movement instead: the CUMULATIVE applied correction per estimated dim
    (x = attempt, 0 = the initial in-hand estimate), the ICP residual per attempt (every guess
    faint, the aggregated pick bold, log y), and for 2+ dims a PHASE plot of the cumulative
    correction pairs (origin = no correction). BEST-EFFORT: never fatal to a hardware run."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ct = np.asarray(corr_track, dtype=float)       # (attempts+1, len(dims)), row 0 = zeros
        x = np.arange(len(ct))
        n_left = len(dims)
        pairs = list(itertools.combinations(range(len(dims)), 2))
        ncols = 3 if pairs else 2
        fig = plt.figure(figsize=(4.6 * ncols + 1.0, max(2.3 * n_left, 6.0)))
        gs = fig.add_gridspec(2 * n_left, ncols, width_ratios=[1.2, 1.0, 0.95][:ncols])
        axes = []
        for i in range(n_left):
            axes.append(fig.add_subplot(gs[2 * i:2 * i + 2, 0],
                                        sharex=axes[0] if axes else None))
        for j, (ax, dim) in enumerate(zip(axes, dims)):
            unit = 'deg' if dim.endswith('_deg') else 'mm'
            ax.axhline(0.0, ls='--', lw=1.0, color='#888888', zorder=1)
            ax.plot(x, ct[:, j], 'o-', color='#4C72B0', zorder=2)
            lim = max(float(np.abs(ct[:, j]).max()), 1e-3) * 1.15
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(f'cumulative {dim} corr [{unit}]')
            ax.tick_params(labelbottom=(j == n_left - 1))
        axes[-1].set_xticks(x)
        axes[-1].set_xlabel('attempt (0 = initial in-hand estimate)')

        ax_r = fig.add_subplot(gs[:, 1])
        labelled = False
        for k, rg in enumerate(res_all or []):
            rg = np.maximum(np.asarray(rg, dtype=float), 1e-6)
            if not len(rg):
                continue
            jit = (np.arange(len(rg)) / max(len(rg) - 1, 1) - 0.5) * 0.3
            ax_r.scatter(k + 1 + jit, rg, s=7, color='#55A868', alpha=0.25, lw=0, zorder=1,
                         label=None if labelled else 'all guesses')
            labelled = True
        r = np.maximum(np.asarray(res_agg, dtype=float), 1e-6)
        ax_r.plot(np.arange(1, len(r) + 1), r, 'o-', color='#55A868', zorder=2,
                  label='aggregated')
        if labelled:
            ax_r.legend(fontsize=8)
        ax_r.set_yscale('log')
        ax_r.set_xticks(np.arange(1, len(r) + 1))
        ax_r.set_xlabel('attempt')
        ax_r.set_ylabel('ICP residual [mm-eq]')

        bounds = np.linspace(0, 2 * n_left, len(pairs) + 1).astype(int) if pairs else []
        for pi, (pa, pb) in enumerate(pairs):
            axp = fig.add_subplot(gs[bounds[pi]:bounds[pi + 1], 2])
            axp.axhline(0.0, ls=':', lw=0.8, color='#aaaaaa', zorder=1)
            axp.axvline(0.0, ls=':', lw=0.8, color='#aaaaaa', zorder=1)
            axp.plot(ct[:, pa], ct[:, pb], '-', color='#4C72B0', lw=1.0, zorder=2)
            axp.scatter(ct[1:, pa], ct[1:, pb], s=20, color='#4C72B0', zorder=3)
            axp.scatter([0.0], [0.0], s=40, marker='s', color='#DD8452', zorder=4,
                        label='initial')
            axp.scatter([ct[-1, pa]], [ct[-1, pb]], s=80, marker='*', color='#55A868',
                        zorder=5, label='latest')
            for k in range(1, len(ct)):
                axp.annotate(str(k), (ct[k, pa], ct[k, pb]), textcoords='offset points',
                             xytext=(4, 3), fontsize=7, color='#444444')
            la = max(float(np.abs(ct[:, pa]).max()), 1e-3) * 1.15
            lb = max(float(np.abs(ct[:, pb]).max()), 1e-3) * 1.15
            axp.set_xlim(-la, la)                  # symmetric: no-correction is the centre
            axp.set_ylim(-lb, lb)
            axp.set_xlabel(f'{dims[pa]} corr '
                           f'[{"deg" if dims[pa].endswith("_deg") else "mm"}]', fontsize=8)
            axp.set_ylabel(f'{dims[pb]} corr '
                           f'[{"deg" if dims[pb].endswith("_deg") else "mm"}]', fontsize=8)
            axp.tick_params(labelsize=7)
            if pi == 0:
                axp.legend(fontsize=7, loc='best')

        fig.suptitle('assembly run: belief corrections per attempt (no ground truth)', y=0.995)
        if status:
            fig.text(0.99, 0.965, status, ha='right', fontsize=9, color='#333333')
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        if live_path:
            tmp = live_path + '.tmp'               # temp + os.replace: viewers never see a
            fig.savefig(tmp, dpi=110, format='png')   # half-written PNG
            os.replace(tmp, live_path)
        plt.close(fig)
    except Exception as exc:                       # noqa: BLE001 -- plotting is never fatal
        log.warning('run plot skipped (%s)', exc)


def build_and_run(cfg, robot, camera, args):
    a = cfg.section('assembly')

    # Build the ESTIMATOR first -- a missing/stale manifold CSV must fail before the robot moves.
    estimator = ManifoldEstimator(cfg.section('estimation'))

    scanner, _detector, _estimator = build_scanner(cfg, robot, camera)
    geom = GraspGeometry(cfg)
    check = GraspCheck(cfg)
    recovery = GraspRecovery(cfg)
    grasp = GraspController(cfg)
    recorder = GraspImageRecorder(cfg)
    guard = ForceGuard(robot.arm, a.get('force_guard', {}))
    adm = AdmittanceController(robot.arm, a.get('compliance', {}))
    confirm = make_confirm(cfg)

    # ---- Known target from the SHARED frames catalogue: assembly.target_frame names a
    # configs/frames.yaml frame whose targets: entry is the recorded mate (base_link <- connector
    # -- the SAME record uncertain_sampling and estimator_eval assemble to, one per socket). ----
    tname = a.get('target_frame')
    if not tname:
        log.error('assembly.target_frame is required -- name a %s frame whose targets: entry '
                  'records the mate (hand-guide to a good mate, read `base_link <- <frame>` off '
                  'the monitor, paste it under targets:).', tool_frames.frames_path(cfg))
        return False
    targets = tool_frames.load_targets(cfg)
    if tname not in targets:
        log.error('assembly.target_frame %r has no targets: entry in %s.',
                  tname, tool_frames.frames_path(cfg))
        return False
    T_base_tconn = targets[tname]
    if a.get('target_connector') or cfg.get('connector_holder_target'):
        log.warning('assembly.target_connector / connector_holder_target are IGNORED -- the '
                    'target now comes from the frames catalogue (targets: %r). Remove the old '
                    'keys.', tname)
    csv_in = urconfig.resolve(cfg, a.get('trajectory_csv', 'assembly_trajectory.csv'))
    mats = traj.load_csv(csv_in, angles_deg=bool(a.get('trajectory_angles_deg', False)))
    if float(np.abs(mats[-1] - np.eye(4)).max()) > 1e-6:
        log.warning('trajectory last row is not identity -- rows are still applied relative to '
                    'the recorded connector target.')
    dense = traj.resample(mats, float(a.get('translational_resolution_m', 0.001)),
                          float(a.get('rotational_resolution_deg', 1.0)))

    # ---- The in-hand ESTIMATE (fingertip -> connector). The grasp geometry is the initial
    # belief: at grasp the fingertip is posed so the junction lands at junction_in_fingertip
    # (cables.yaml), so ftip->conn IS that pose. estimation.initial_connector_in_fingertip
    # (m/rad) overrides it when set. ----
    init = cfg.get_path('estimation.initial_connector_in_fingertip')
    T_ftip_conn = from_cfg(init) if init else from_cfg(cfg.section('junction_in_fingertip'))

    # ---- Speed limits: ONE global `speed:` block; EVERY phase of this app applies its own
    # scale to all four limits (speed.phase_scale.<phase>). Free-space moves inherit the scale
    # from arm.set_speed_scale -- `phase(...)` marks each boundary below -- while the compliant
    # ramps are paced here (assemble / retract) and in GraspController (pickup / lift). ----
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
        t_lin = (lin_m * 1000.0 / v) if v > 0 else 0.0
        t_ang = (np.degrees(ang_rad) / w) if w > 0 else 0.0
        return max(t_lin, t_ang, min_seg_s)

    comp = a.get('compliance', {}) or {}
    settle_s = float(comp.get('settle_s', 0.5))
    tare = (lambda: robot.arm.zero_ft(settle=False)) if bool(comp.get('tare_before', True)) else None
    retract_m = float(a.get('retract_distance_m', 0.05))
    decim = max(1, int(a.get('log_decimation', 5)))
    tol = a.get('success_tolerance', {}) or {}
    tol_pos_m = float(tol.get('pos_mm', 2.0)) / 1000.0
    tol_rot_rad = np.radians(float(tol.get('rot_deg', 3.0)))
    max_attempts = int(a.get('max_attempts', 5))

    # OPTIONAL trajectory noising (same semantics as estimator_eval's eval.trajectory_noise):
    # smoothed per-waypoint Gaussian offsets in the connector's OWN frame, redrawn per attempt,
    # per-DOF std [x,y,z (m), r,p,y (deg)]; noise_decay_attempt shrinks the whole perturbation
    # by (1-f)^(attempt-1); noise_decay_traj sheds it linearly along the path (converge onto
    # the nominal reference near the seat).
    tn = a.get('trajectory_noise', {}) or {}
    tn_on = bool(tn.get('enabled', False))
    tn_std = tn.get('std')
    if tn_std is None:
        tn_std = [float(tn.get('translation_m', 0.0005))] * 3 \
            + [float(tn.get('rotation_deg', 0.5))] * 3
    tn_std = [float(v) for v in tn_std]
    if len(tn_std) != 6:
        log.error('assembly.trajectory_noise.std must have 6 entries [x,y,z (m), r,p,y (deg)].')
        return False
    tn_w = max(1, int(tn.get('smooth_window', 25)))
    tn_da = float(tn.get('noise_decay_attempt', 0.0))
    tn_dt = float(tn.get('noise_decay_traj', 0.0))
    if tn.get('alternate_pitch_deg'):
        log.warning('trajectory_noise.alternate_pitch_deg was REMOVED (2026-08-13). Ignored.')
    noise_rng = np.random.default_rng()
    if tn_on:
        log.info('Trajectory noise ON: std %s, smooth %d, decay/attempt %.2f, decay/traj %.2f.',
                 tn_std, tn_w, tn_da, tn_dt)

    # LIVE run figure (like estimator_eval's): ONE fixed path outside the experiment folder,
    # atomically overwritten after every estimate. true = data/experiments/cable_pick_live.png;
    # a string = explicit path; false disables.
    live = a.get('live_plot', True)
    live_path = None
    if live:
        live_path = live if isinstance(live, str) else os.path.join(
            cfg.get('data_dir', 'data'), 'experiments', 'cable_pick_live.png')
        os.makedirs(os.path.dirname(live_path) or '.', exist_ok=True)

    # Every run gets its own EXPERIMENT subdirectory: per-attempt observation CSVs, per-attempt
    # convergence plots, and estimates.csv.
    out_dir = os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                           f'cable_pick_estimate_assemble_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    log.info('Experiment folder: %s', out_dir)

    # ---- RESET + PICK + slip-checked LIFT (pick identical to cable_pick_assemble). The lift is
    # INSIDE the retry loop: a cable that slips out during the lift ('slipped', detected by the
    # partial-lift re-close in GraspController.lift_verified) restarts the whole scan->grasp,
    # and each full retry perturbs the grasp along the junction +/-x (retry_offset_x). ----
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
                _s['r'] = grasp.lift_verified(robot, geom, check, 'lift',
                                              position_guard=lambda mv: _guarded(robot, guard, mv))
                return _s['r'] == 'ok'

            if runner.run([('lift (slip-checked)', do_lift)]):
                break
            result = status.get('r')
            if result != 'slipped':
                return False               # a move failed, or the user aborted at the step gate
        if result == 'abort':
            return False
        if attempt >= check.max_retries:
            log.error('Grasp failed on all %d attempts; aborting.', check.max_retries + 1)
            return False
        attempt += 1
        log.warning('Grasp %s -- recovering (attempt %d/%d).',
                    result, attempt + 1, check.max_retries + 1)
        if result == 'slipped' and hasattr(scanner, 'reselect'):
            # SLIP RECOVERY -- lighter than the full reset: the cable fell somewhere below, so
            # open, rise slip_raise_m STRAIGHT UP from here, and let the next scan re-image,
            # re-number, and RE-PROMPT the operator from this vantage (reselect() drops the
            # cached junction selection -- it is stale, the cable moved when it dropped).
            phase('scan')
            T_up = translation_matrix([0.0, 0.0, check.slip_raise_m]) @ robot.tool0()
            if not (robot.gripper.open('drop')
                    and _guarded(robot, guard,
                                 lambda: robot.arm.move_l(T_up, label='slip recovery (up)'))):
                return False
            scanner.reselect()
        else:
            phase('reset')
            if not (robot.gripper.open('drop') and robot.arm.move_j(q_home, label='home')):
                return False

    # ---- PHYSICAL payload check: the stalled counts -> held width through the calibrated
    # gripper model (+ groove depth). Purely informational next to the counts-band check, but in
    # units a human can sanity-check against the datasheet with calipers. ----
    d_conn = cfg.get_path('grasp_check.connector_diameter_mm')
    if d_conn and not robot.arm.dry_run:
        w_mm = robot.gripper.held_width_m() * 1000.0
        log.info('Payload width: %.2f mm (expected connector %.2f-%.2f mm).',
                 w_mm, min(d_conn), max(d_conn))

    # ---- Approach the stand-off (beyond the trajectory START, target frame) ----

    standoff_axis = np.asarray((a.get('standoff', {}) or {}).get('axis', [-1, 0, 0]), dtype=float)
    standoff_m = float((a.get('standoff', {}) or {}).get('distance_m', 0.01))
    T_standoff_row = translation_matrix(standoff_axis * standoff_m) @ mats[0]

    def tool0_ref(row, T_tool0_conn):
        return T_base_tconn @ row @ inverse(T_tool0_conn)

    phase('standoff')
    seed_q = robot.arm.q()
    T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
    q = robot.arm.ik(tool0_ref(T_standoff_row, T_tool0_conn), seed_q)
    if q is None or not _guarded(robot, guard, lambda: robot.arm.move_j(q, label='stand-off')):
        return False
    seed_q = q

    # CABLE-IN-GRIPPER check at the stand-off: the transit from the lift can lose the part
    # without any force signature (re-close + counts, like the slip check -- no motion).
    if not verify_cable_held(robot, check, 'stand-off'):
        return False

    # UNCONDITIONAL pause at the stand-off (like the reset gate): the next motion drives the held
    # part into contact, so a human confirms the scene is ready -- regardless of confirm_each_step.
    if not robot.arm.dry_run:
        try:
            answer = input('\n[stand-off] Ready to ASSEMBLE (contact ahead). '
                           'Enter to continue (q to abort): ')
        except EOFError:
            answer = ''
        if answer.strip().lower() in ('q', 'quit', 'n', 'no'):
            log.info('Aborted at the stand-off by the user.')
            return False

    # ---- The assemble / check / retract / estimate loop ----
    est_rows, success = [], False
    try:
        # Run-level tracks for _plot_run: the cumulative applied correction (row 0 = the
        # initial in-hand estimate) + per-attempt residuals (aggregated and every guess).
        T_cum = np.eye(4)
        trackc = [np.zeros(len(estimator.estimate_dims))]
        trackr, trackg = [], []

        for it in range(1, max_attempts + 1):
            T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
            e_xyz, e_rpy = matrix_to_xyzrpy(T_ftip_conn)
            log.info('--- attempt %d/%d --- in-hand estimate xyz=%s mm rpy=%s deg', it, max_attempts,
                     np.round(e_xyz * 1000, 2).tolist(), np.round(np.degrees(e_rpy), 2).tolist())
            if tn_on:
                rows_t = traj.noised(dense, noise_rng, tn_std, tn_w, tn_dt,
                                     (1.0 - tn_da) ** (it - 1))
            else:
                rows_t = dense
            refs = [tool0_ref(row, T_tool0_conn) for row in rows_t]

            # Realign with the START of the (re-estimated) trajectory -- stiff, free space, guarded.
            phase('standoff')
            q = robot.arm.ik(refs[0], seed_q)
            if q is None or not _guarded(robot, guard,
                                         lambda: robot.arm.move_j(q, label=f'align start {it}')):
                log.error('Could not reach the trajectory start; aborting.')
                return False
            seed_q = q

            # ASSEMBLE under admittance, collecting observations (same law as uncertain_sampling).
            obs, cnt = [], [0]

            def log_cb(_obs=obs, _cnt=cnt, _Ttc=T_tool0_conn):
                _cnt[0] += 1
                if _cnt[0] % decim == 0:
                    _obs.append(_observe(robot, _Ttc, T_base_tconn))

            adm.reset()
            adm.warmup(refs[0], tare_fn=tare)
            guard.reset()
            last_ref = refs[0]
            for i in range(1, len(refs)):
                res = adm.ramp(refs[i - 1], refs[i], seg_time(refs[i - 1], refs[i]),
                               guard, on_step=log_cb)
                last_ref = refs[i]
                if res == 'seated':
                    log.info('Contact limit at waypoint %d/%d -- stopped advancing.', i, len(refs) - 1)
                    break
            adm.hold(last_ref, settle_s, guard, on_step=log_cb)

            # CHECK: the kinematic numbers are computed and logged for the record (compliance
            # deflection makes a wrong belief show as a real error here), but the SUCCESS DECISION
            # is the OPERATOR's -- they can see the physical mate; the numbers only see the belief.
            # A dry run has no operator, so it falls back to the tolerance check.
            T_conn_now = robot.tool0() @ T_tool0_conn
            lin, ang = pose_error(T_conn_now, T_base_tconn)
            log.info('check: believed connector vs target: %.2f mm, %.2f deg (reference tol '
                     '%.2f mm, %.2f deg)', lin * 1000, np.degrees(ang),
                     tol_pos_m * 1000, np.degrees(tol_rot_rad))
            _save_observations(os.path.join(out_dir, f'attempt_{it:02d}_observations.csv'), obs)

            row = {'attempt': it, 'n_observations': len(obs),
                   'check_pos_mm': lin * 1000.0, 'check_rot_deg': float(np.degrees(ang))}
            if robot.arm.dry_run:
                row['success'] = bool(lin <= tol_pos_m and ang <= tol_rot_rad)
            else:
                try:
                    ans = input(f'[check attempt {it}] Was the assembly SUCCESSFUL? '
                                '(y = done / Enter = retry / q = abort): ').strip().lower()
                except EOFError:
                    ans = ''
                if ans in ('q', 'quit'):
                    log.info('Aborted at the check by the user.')
                    est_rows.append(row)
                    return False
                row['success'] = ans in ('y', 'yes')
            if row['success']:
                est_rows.append(row)
                log.info('Within tolerance -- ASSEMBLY COMPLETE on attempt %d.', it)
                success = True
                break

            # RETRACT: linear escape along the connector's own -X (compliant, un-guarded), at the
            # 'retract' phase scale.
            T_out = _retract_ref(last_ref, T_tool0_conn, retract_m)
            adm.ramp(last_ref, T_out, seg_time(last_ref, T_out, g_v * s_ret, g_w * s_ret),
                     guard=None)
            robot.arm.servo_stop()

            # CABLE-IN-GRIPPER check after the attempt: an insertion/retract can strip the part
            # out of the fingers (it may even be left IN the socket). Without the part, further
            # attempts -- and the estimate from this attempt's observations -- are meaningless.
            if not verify_cable_held(robot, check, f'attempt {it} retract'):
                row['cable_held'] = False
                est_rows.append(row)
                return False

            if it == max_attempts:
                est_rows.append(row)
                log.error('Attempt limit reached (%d) without a successful mate.', max_attempts)
                break

            # ESTIMATE the belief error from this attempt's observations, against the manifold.
            obs_arr = np.asarray(obs, dtype=float)
            vec6, w6 = estimator.prepare_observations(obs_arr[:, :6], obs_arr[:, 6:9],
                                                      obs_arr[:, 9:12]) if len(obs) else \
                (np.zeros((0, 6)), np.zeros((0, 6)))
            T_corr_mm, info = estimator.estimate(vec6, w6)
            if T_corr_mm is None:
                log.warning('Estimation skipped (%s) -- retrying with the UNCHANGED estimate.', info)
                row['estimate'] = f'skipped: {info}'
                est_rows.append(row)
                trackc.append(trackc[-1])          # belief unchanged this attempt
                trackr.append(float('nan'))
                trackg.append(np.zeros(0))
                _plot_run(os.path.join(out_dir, 'run_corrections.png'),
                          estimator.estimate_dims, trackc, trackr, trackg,
                          f'attempt {it}/{max_attempts} (estimate skipped)', live_path)
                continue
            log.info('estimated belief correction: %s  (inliers %d/%d, residual %.3f, %d obs)',
                     {k: round(v, 3) for k, v in info['theta_corr'].items()},
                     info['inliers'], info['guesses'], info['final_residual'],
                     info['n_observations'])
            _plot_estimate(os.path.join(out_dir, f'attempt_{it:02d}_estimate.png'),
                           estimator.estimate_dims, info)
            T_ftip_conn = T_ftip_conn @ _corr_to_m(T_corr_mm)     # believed @ corr ~= true
            T_cum = T_cum @ np.asarray(T_corr_mm, dtype=float)    # cumulative, mm units
            trackc.append(vec6_from_mats(T_cum)[estimator.idx])
            trackr.append(float(info['final_residual']))
            trackg.append(np.asarray(info['res_hist'], dtype=float)[:, -1])
            _plot_run(os.path.join(out_dir, 'run_corrections.png'),
                      estimator.estimate_dims, trackc, trackr, trackg,
                      f'attempt {it}/{max_attempts} | check {lin * 1000:.1f} mm / '
                      f'{np.degrees(ang):.1f} deg', live_path)
            row.update({f'corr_{k}': v for k, v in info['theta_corr'].items()})
            row.update({'icp_inliers': info['inliers'], 'icp_residual': info['final_residual']})
            est_rows.append(row)
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

    # ---- Release, escape along the CONNECTOR's OWN -X (peg-in-hole -- the direction comes from
    # the connector frame at the mate, executed as a pure world translation), then reset. ----
    d_out = float(a.get('release_retract_distance_m', 0.08))
    T_conn_final = robot.tool0() @ T_tool0_conn
    back = -T_conn_final[:3, 0] * d_out                   # connector -X, in base coordinates

    def release_escape():
        T_new = translation_matrix(back) @ robot.tool0()
        return _guarded(robot, guard, lambda: robot.arm.move_l(T_new, label='retract (connector -X)'))

    phase('retract')
    ok = runner.run([('open gripper (release)', robot.gripper.open),
                     ('retract (connector -X)', release_escape)])
    phase('reset')
    return ok and reset.reset_robot(robot, cfg, 'end reset')


def main():
    run_app('Cable pick + estimate-while-assemble (contact-manifold ICP)',
            'cable_pick_estimate_assemble', build_and_run, needs_camera=True)


if __name__ == '__main__':
    main()
