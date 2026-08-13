"""TWO-STAGE estimator evaluation: estimate the MODE first, then the POSE within it.

The failure this app exists to fix (observed on the v3 hose data, replacing
estimator_eval_probe): the single-energy estimator often commits to the WRONG mode because an
incorrect-but-similar contact mode explains the observations BETTER than the truth -- what
should be a local minimum becomes the global one (aliasing), the correction diverges, and the
next attempt collects even less explainable contact. The pose channels are the aliased ones;
the WRENCH signature is what actually separates contact modes -- but in a single metric the
wrench is drowned by pose proximity exactly when it matters.

So the metric is SPLIT:

    STAGE 1 -- MODE  (estimation_mode: wrench-HEAVY metric, pose down-weighted): build the
        energy over the shared candidate grid, optionally fuse the stop-signature energies
        (the one channel aliasing cannot touch), partition into modes (mass-filtered
        watershed, skills/mixture), and pick the top-mass mode. This stage is tuned for
        CLASSIFICATION -- it does not need to be accurate, only to put the truth's cell in
        the chosen SET.
    STAGE 2 -- POSE  (estimation_pose: wrench-LIGHT metric, stop-signature fused): a second
        energy over the SAME grid, UNRESTRICTED argmin, scaled by alpha. The mode partition
        is a pure DIAGNOSTIC -- gating stage 2 on it was measured to hurt (|z'| 7.4 vs 2.0
        mm) while the classifier (78% truth-in-mode) trails the fused refiner's implicit
        mode accuracy (~84%); the mode columns exist to tune the classifier toward the
        crossover, not to constrain the estimate.

Because the truth is known here, trials.csv scores the two stages SEPARATELY:
`truth_in_mode` / `truth_mode_rank` grade stage 1 as a classifier (tune estimation_mode until
these are high), `err_after_*` grades stage 2. Collection is offset-sweep by default
(trajectory diversity is what feeds stage 1). Divergence bounds terminate runaway trials; the
trial always ends with ONE zero-noise insertion from the final belief.

Run:  python -m urlab.apps.mode_pose_estimator_eval --config configs/mode_pose_estimator_eval.yaml
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
from ..skills.grid_estimator import GridManifoldEstimator
from ..skills.manifold import mats_from_vec6, vec6_from_mats
from ..skills.mixture import from_energy
from ..skills.success_basin import SuccessBasin
from ..transforms import inverse, matrix_to_xyzrpy, pose_error, translation_matrix
from ._runner import run_app
from .estimator_eval import (_ERR, _corr_to_m, _gt_error, _observe, _rebase_rows,
                             _save_observations)
from .uncertain_sampling import _clock, _fmt_dur, _retract_ref

log = urlog.get('mode-pose-eval')


def merged_estimation(cfg):
    """(mode_cfg, pose_cfg): estimation_shared overlaid by the per-stage blocks. The GRID and
    estimate_dims come from shared ONLY -- both stages must index the SAME candidate cells."""
    shared = dict(cfg.section('estimation_shared'))
    if not shared.get('manifold_csv'):
        raise ValueError('estimation_shared.manifold_csv is required')
    out = []
    for key in ('estimation_mode', 'estimation_pose'):
        over = dict(cfg.section(key))
        for frozen in ('grid', 'estimate_dims'):
            if frozen in over:
                raise ValueError(f'{key}.{frozen} must live in estimation_shared -- both '
                                 'stages share one candidate grid')
        c = dict(shared)
        c.update(over)
        out.append(c)
    return out


def two_stage(est_mode, est_pose, rows12, e_stop=None, stop_weight=2.0,
              temp=0.05, min_weight=0.02, alpha=1.0, stop_in_mode=False):
    """The two-stage estimate, PURE (no robot): rows12 = pooled observation rows.

    Returns (theta6, info) or (None, reason). Verified against the 2026-08-13 AM data:

      * the stage-1 PARTITION must stay UNFUSED (stop_in_mode=False): adding the smooth
        stop-signature basin fragments the watershed and destroys the mode classifier
        (truth-in-top-mode 78% -> 0% when fused);
      * the stop signature belongs in STAGE 2 instead -- `e_stop` fuses into the refinement
        energy (median-normalised), which is the measured-best point estimator;
      * there is NO GATE (removed 2026-08-13): restricting stage 2 to the chosen mode was
        measured to HURT (|z'| 7.4 vs 2.0 mm) because the classifier (78% truth-in-mode) is
        below the fused refiner's own implicit mode accuracy (~84%). Stage 1's partition is
        a pure DIAGNOSTIC: the mode outputs grade the classifier, the refinement runs
        unrestricted."""
    vm, wm = est_mode.prepare_observations(rows12[:, :6], rows12[:, 6:9], rows12[:, 9:12])
    vp, wp = est_pose.prepare_observations(rows12[:, :6], rows12[:, 6:9], rows12[:, 9:12])
    E_m, n_m, _ = est_mode.energy(vm, wm)
    E_p, n_p, _ = est_pose.energy(vp, wp)
    if E_m is None or E_p is None:
        return None, f'too few observations (mode {n_m}, pose {n_p})'
    E_dec = E_m / max(float(np.median(E_m)), 1e-12)
    E_p = np.asarray(E_p, dtype=float) / max(float(np.median(E_p)), 1e-12)
    if e_stop is not None:
        e_n = stop_weight * np.asarray(e_stop, dtype=float) \
            / max(float(np.median(e_stop)), 1e-12)
        E_p = E_p + e_n                            # refinement: stop fusion belongs HERE
        if stop_in_mode:
            E_dec = E_dec + e_n                    # partition fusion: measured HARMFUL
    mix = from_energy(est_mode.grid6[:, est_mode.idx], E_dec, est_mode.grid_shape,
                      temp=temp, min_weight=min_weight, dims=est_mode.estimate_dims)
    chosen = mix.components[0]
    # STAGE 2: unrestricted refinement argmin; the stage-1 partition is diagnostic only
    k_in = int(np.argmin(E_p))
    theta = alpha * est_pose.grid6[k_in]
    info = {
        'theta_corr': {d: float(theta[j])
                       for d, j in zip(est_pose.estimate_dims, est_pose.idx)},
        'mixture': mix, 'n_modes': mix.n_modes, 'mode_mass': float(chosen.weight),
        'mode_centre': {d: float(v)
                        for d, v in zip(est_mode.estimate_dims, chosen.mean)},
        'ambiguity': mix.ambiguity, 'separation': mix.separation,
        'E_mode': E_dec, 'E_pose': np.asarray(E_p, dtype=float),
        'pose_argmin_global': {d: float(v) for d, v in zip(
            est_pose.estimate_dims, est_pose.grid6[int(np.argmin(E_p))][est_pose.idx])},
        'n_observations': int(n_p),
    }
    return theta, info


def truth_mode_rank(mix, est, errb6):
    """Which mode (by mass rank, 1 = chosen) holds the TRUTH cell? len+1 if folded away."""
    t6 = vec6_from_mats(np.linalg.inv(mats_from_vec6(np.asarray(errb6, dtype=float))))
    cell = 0
    for a, (j, ax) in enumerate(zip(est.idx, est.grid_axes)):
        cell = cell * est.grid_shape[a] if a else 0
    # compute flat index properly (row-major over grid_shape)
    sub = []
    for j, ax in zip(est.idx, est.grid_axes):
        sub.append(int(np.argmin(np.abs(ax - t6[j]))))
    cell = int(np.ravel_multi_index(tuple(sub), est.grid_shape))
    for i, c in enumerate(mix.components):
        if c.idx is not None and cell in set(int(v) for v in c.idx):
            return i + 1
    return len(mix.components) + 1


def _plot_live(path, live_path, est_mode, est_pose, info, errb, dims, hist, status=None):
    """ONE figure per trial, re-saved after every attempt and mirrored ATOMICALLY to the
    shared live path (the same file estimator_eval writes -- one open viewer serves all
    eval apps).

    LEFT: the stage-1 MODE landscape (wrench-heavy energy) in the ERROR frame (truth =
    origin) with the mass-filtered partition -- the chosen mode's cells shaded, every mode
    centre sized by its mass. MIDDLE: the stage-2 fused POSE landscape (same frame) with the
    committed correction and the truth. RIGHT: the trial history -- |error| before/after per
    attempt, coloured by whether the truth was inside the chosen mode (the stage-1 grade).
    BEST-EFFORT: a plotting problem is logged, never fatal."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        mix = info['mixture']
        idx = list(est_mode.idx)
        gth = est_mode.grid6[:, idx]
        # candidate corrections -> the REMAINING ERROR they would leave (truth at origin)
        th6 = np.zeros((len(gth), 6))
        th6[:, idx] = gth
        gerr = vec6_from_mats(mats_from_vec6(np.asarray(errb, dtype=float))
                              @ mats_from_vec6(th6))[:, idx]
        unit = ['deg' if d.endswith('_deg') else 'mm' for d in dims]

        fig = plt.figure(figsize=(16.5, 6.2))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0])
        axM = fig.add_subplot(gs[:, 0])
        axP = fig.add_subplot(gs[:, 1])
        axT = fig.add_subplot(gs[0, 2])
        axB = fig.add_subplot(gs[1, 2])

        def landscape(ax, E, title):
            if len(dims) == 2:
                sh = est_mode.grid_shape
                im = ax.pcolormesh(gerr[:, 1].reshape(sh), gerr[:, 0].reshape(sh),
                                   np.asarray(E, dtype=float).reshape(sh),
                                   shading='auto', cmap='viridis')
                fig.colorbar(im, ax=ax, label='energy [norm]')
                ax.axhline(0.0, ls=':', lw=0.9, color='white', alpha=0.6)
                ax.axvline(0.0, ls=':', lw=0.9, color='white', alpha=0.6)
                ax.plot([0.0], [0.0], '*', ms=16, mfc='#55A868', mec='white', zorder=8,
                        label='truth (origin)')
                ax.set_xlabel(f'{dims[1]} ERROR remaining [{unit[1]}]')
                ax.set_ylabel(f'{dims[0]} ERROR remaining [{unit[0]}]')
            else:
                o = np.argsort(gerr[:, 0])
                ax.plot(gerr[o, 0], np.asarray(E, dtype=float).ravel()[o], lw=1.5,
                        color='#4C72B0')
                ax.axvline(0.0, color='#55A868', ls='--', lw=1.8, label='truth')
                ax.set_xlabel(f'{dims[0]} ERROR remaining [{unit[0]}]')
                ax.set_ylabel('energy [norm]')
            ax.set_title(title, fontsize=10)

        # ---- LEFT: mode landscape + partition ----
        landscape(axM, info['E_mode'],
                  f"stage 1 MODE (wrench-heavy): {info['n_modes']} modes, "
                  f"chosen mass {info['mode_mass']:.0%}")
        if len(dims) == 2:
            chosen = mix.components[0]
            cells = np.asarray([int(j) for j in chosen.idx], dtype=int)
            step = max(len(cells) // 400, 1)       # bounded scatter
            axM.scatter(gerr[cells[::step], 1], gerr[cells[::step], 0], s=4,
                        color='#e377c2', alpha=0.25, zorder=3, label='chosen mode cells')
            for m, comp in enumerate(mix.components[:4]):
                c6 = np.zeros(6)
                c6[idx] = comp.mean
                ce = vec6_from_mats(mats_from_vec6(np.asarray(errb, dtype=float))
                                    @ mats_from_vec6(c6))[idx]
                axM.plot([ce[1]], [ce[0]], 'D', ms=5 + 9 * comp.weight, mfc='none',
                         mec='#e377c2', mew=1.8, zorder=7,
                         label='mode centres' if m == 0 else None)
                axM.annotate(f'{comp.weight:.0%}', (ce[1], ce[0]),
                             textcoords='offset points', xytext=(6, -10), fontsize=7,
                             color='#e377c2')
        axM.legend(fontsize=7, loc='best')

        # ---- MIDDLE: fused pose landscape + committed correction ----
        landscape(axP, info['E_pose'], 'stage 2 POSE (wrench-light + stop, committed)')
        a6 = np.zeros(6)
        a6[idx] = [info['theta_corr'][d] for d in dims]
        ae = vec6_from_mats(mats_from_vec6(np.asarray(errb, dtype=float))
                            @ mats_from_vec6(a6))[idx]
        if len(dims) == 2:
            axP.plot([ae[1]], [ae[0]], 'o', ms=9, mfc='#DD8452', mec='white', zorder=8,
                     label=f"applied (alpha'd)")
        else:
            axP.axvline(ae[0], color='#DD8452', lw=2.0, label="applied (alpha'd)")
        axP.legend(fontsize=7, loc='best')

        # ---- RIGHT: history -- error track + truth-in-mode grades ----
        eb = [h['err_before'] for h in hist]
        ea = [h['err_after'] for h in hist]
        tim = [h['truth_in_mode'] for h in hist]
        xs = np.arange(1, len(hist) + 1)
        axT.plot(xs, eb, 'o--', color='#999999', label='|err| before')
        axT.plot(xs, ea, 'o-', color='#4C72B0', label='|err| after')
        for x, ok, y in zip(xs, tim, ea):
            axT.plot([x], [y], 'o', ms=10, mfc='none', mew=2,
                     mec='#55A868' if ok else '#C44E52')
        axT.set_xlabel('attempt (all trials)')
        axT.set_ylabel('combined error [mm-eq]')
        axT.set_title('error per attempt (ring: truth in chosen mode?)', fontsize=9)
        axT.legend(fontsize=7)
        axT.grid(alpha=0.25)
        rate = float(np.mean(tim)) if tim else float('nan')
        axB.bar(['truth_in_mode'], [100 * rate], color='#55A868', width=0.4)
        axB.axhline(84, ls='--', lw=1.2, color='#4C72B0')
        axB.annotate('refiner implicit ~84%\n(gate crossover)', (0.02, 85),
                     xycoords=('axes fraction', 'data'), fontsize=8, color='#4C72B0')
        axB.set_ylim(0, 100)
        axB.set_ylabel('%')
        axB.set_title(f'stage-1 classifier rate: {rate:.0%} (n={len(tim)})', fontsize=9)

        if status:
            fig.text(0.99, 0.965, status, ha='right', fontsize=9, color='#333333')
        fig.suptitle('mode_pose_estimator_eval -- ERROR frame (truth = origin)', y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(path, dpi=110)
        if live_path:
            tmp = live_path + '.tmp.png'
            fig.savefig(tmp, dpi=110)
            os.replace(tmp, live_path)             # atomic: the viewer never sees a partial
        plt.close(fig)
    except Exception as exc:                       # noqa: BLE001 -- plotting is never fatal
        log.warning('live plot skipped (%s)', exc)


def _fieldnames(dims):
    return (['trial', 'attempt', 'n_observations', 'n_passes']
            + [f'inj_{s}' for s in _ERR]
            + [f'err_before_{s}' for s in _ERR] + ['err_before_pos_mm', 'err_before_rot_deg']
            + ['n_modes', 'mode_mass', 'ambiguity', 'separation',
               'truth_in_mode', 'truth_mode_rank']
            + [f'mode_centre_{d}' for d in dims]
            + [f'corr_{d}' for d in dims] + [f'pose_argmin_global_{d}' for d in dims]
            + ['stop_fused', 'x_stop_mm']
            + [f'err_after_{s}' for s in _ERR] + ['err_after_pos_mm', 'err_after_rot_deg']
            + ['converged', 'diverged', 'estimate']
            + ['insert_depth_mm', 'insert_seated', 'insert_success']
            + [f'seat_{s}' for s in _ERR])


def build_and_run(cfg, robot, camera, args):
    ev = cfg.section('eval')
    try:
        cfg_m, cfg_p = merged_estimation(cfg)
        est_mode = GridManifoldEstimator(cfg_m)
        est_pose = GridManifoldEstimator(cfg_p)
    except ValueError as exc:
        log.error('%s', exc)
        return False
    if est_mode.grid_shape != est_pose.grid_shape:
        log.error('stage grids differ -- estimation grids must be identical.')
        return False
    dims = est_pose.estimate_dims
    log.info('Two-stage estimator: MODE metric s_force %.2f s_torque %.2f dim_w %s | POSE '
             'metric s_force %.2f s_torque %.2f dim_w %s | grid %s.',
             est_mode.s_force, est_mode.s_torque, np.round(est_mode.dim_w, 2).tolist(),
             est_pose.s_force, est_pose.s_torque, np.round(est_pose.dim_w, 2).tolist(),
             'x'.join(map(str, est_pose.grid_shape)))

    ms = cfg.section('mode_stage')
    ps = cfg.section('pose_stage')
    temp = float(ms.get('posterior_temp', 0.05))
    min_w = float(ms.get('mode_min_weight', 0.02))
    # VERIFIED 2026-08-13 (analysis/mode_pose): the stop signature fuses into STAGE 2 (the
    # refinement -- the measured-best point estimator), and must NOT enter the stage-1
    # partition (it fragments the watershed: truth-in-top-mode 78% -> 0%). There is NO mode
    # gate (removed): restricting stage 2 to the chosen mode measurably hurt (|z'| 7.4 vs
    # 2.0) -- the mode outputs are DIAGNOSTICS for tuning the classifier.
    stop_in_mode = bool(ms.get('stop_fusion', False))
    stop_w = float(ps.get('stop_weight', 2.0))
    alpha = float(ps.get('alpha', 0.75))
    if ps.get('mode_gate'):
        log.warning('pose_stage.mode_gate was REMOVED (2026-08-13) -- the mode partition is '
                    'diagnostic only. Ignored.')
    basin = None
    if bool(ps.get('stop_fusion', True)) or stop_in_mode:
        try:
            basin = SuccessBasin(urconfig.resolve(cfg, cfg_m['manifold_csv']), dims,
                                 dict(ms.get('basin', {}) or {},
                                      scaling_constant_deg_to_mm=est_mode.s_rot))
        except Exception as exc:                   # noqa: BLE001
            log.error('Stop model unavailable (%s) -- stages run unfused.', exc)
    log.info('Two-stage wiring: stop->stage2 %s (w %.1f), stop->partition %s, '
             'alpha %.2f.', basin is not None, stop_w, stop_in_mode, alpha)

    held_name = cfg.get('held_frame')
    frames = tool_frames.load_frames(cfg)
    targets = tool_frames.load_targets(cfg)
    if not held_name or held_name not in frames or held_name not in targets:
        log.error('held_frame %r needs BOTH frames: and targets: entries in %s.',
                  held_name, tool_frames.frames_path(cfg))
        return False
    T_true = frames[held_name]
    T_base_tconn = targets[held_name]

    csv_in = urconfig.resolve(cfg, cfg.get('trajectory_csv', 'assembly_trajectory.csv'))
    mats = traj.load_csv(csv_in, angles_deg=bool(cfg.get('trajectory_angles_deg', False)))
    dense = traj.resample(mats, float(cfg.get('translational_resolution_m', 0.001)),
                          float(cfg.get('rotational_resolution_deg', 1.0)))

    # injected errors: 'random' inside the bounds, or 'grid' at grid_resolution
    seed = int(ev.get('random_seed', 0))
    rng = np.random.default_rng(seed if seed > 0 else None)
    pert = ev.get('perturbation', {}) or {}
    lo, hi = pert.get('lower', [0.0] * 6), pert.get('upper', [0.0] * 6)
    mode_sel = str(ev.get('mode', 'random')).lower()
    fixed = None
    if mode_sel == 'grid':
        fixed = traj.grid_deltas(lo, hi, ev.get('grid_resolution', [0.0] * 6))
    num_trials = len(fixed) if fixed is not None else int(ev.get('num_trials', 10))
    max_attempts = int(ev.get('max_attempts', 2))

    sweep = ev.get('sweep_offsets')
    if sweep is None:
        sweep = [[0.0, 0.0, 0.0, 0.0, float(p), 0.0] for p in np.arange(-4.0, 4.01, 2.0)]
    sweep = [[float(v) for v in o] for o in sweep]
    if any(len(o) != 6 for o in sweep) or not sweep:
        log.error('eval.sweep_offsets must be a non-empty list of 6-vectors.')
        return False
    tn = ev.get('trajectory_noise', {}) or {}
    tn_on = bool(tn.get('enabled', True))
    tn_std = [float(v) for v in (tn.get('std') or [0.0, 0.0005, 0.005, 0.0, 5.0, 0.0])]
    tn_w = max(1, int(tn.get('smooth_window', 1)))
    noise_rng = np.random.default_rng(seed + 1 if seed > 0 else None)

    ab = ev.get('abort_bounds', {}) or {}
    abort_pos = float(ab.get('pos_mm', 10.0))
    abort_rot = float(ab.get('rot_deg', 15.0))
    tol = ev.get('success_tolerance', {}) or {}
    tol_pos, tol_rot = float(tol.get('pos_mm', 1.0)), float(tol.get('rot_deg', 1.0))
    succ_tol = [float(v) for v in ev.get('success_pose_tol', [2, 1, 5, 5, 5, 1])]
    decim = max(1, int(ev.get('log_decimation', 5)))
    save_obs = bool(ev.get('save_observations', True))
    accumulate = bool(ev.get('accumulate_observations', True))
    save_plots = bool(ev.get('save_plots', True))
    live = ev.get('live_plot', True)
    live_path = None
    if save_plots and live:
        live_path = live if isinstance(live, str) else os.path.join(
            cfg.get('data_dir', 'data'), 'experiments', 'estimator_eval_live.png')
        os.makedirs(os.path.dirname(live_path) or '.', exist_ok=True)
        log.info('Live figure: %s', live_path)

    adm = AdmittanceController(robot.arm, cfg.section('compliance'))
    guard = ForceGuard(robot.arm, cfg.section('force_guard'))
    tare = (lambda: robot.arm.zero_ft(settle=False)) \
        if bool(cfg.get_path('compliance.tare_before', True)) else None
    settle_s = float(cfg.get_path('compliance.settle_s', 0.5))
    v_mm_s = float(cfg.get_path('speed.max_cartesian_translation_mm_s', 5.0))
    w_deg_s = float(cfg.get_path('speed.max_cartesian_rotation_deg_s', 30.0))
    rv = float(cfg.get_path('speed.retract_translation_mm_s', 50.0))
    rw = float(cfg.get_path('speed.retract_rotation_deg_s', 50.0))
    retract_m = float(cfg.get('retract_distance_m', 0.05))
    min_seg_s = 1.0 / adm.rate

    def seg_time(A, B, v=None, w=None):
        v = v_mm_s if v is None else v
        w = w_deg_s if w is None else w
        lin_m, ang_rad = pose_error(A, B)
        return max((lin_m * 1000.0 / v) if v > 0 else 0.0,
                   (np.degrees(ang_rad) / w) if w > 0 else 0.0, min_seg_s)

    def run_insertion(refs, T_bel):
        obs, cnt = [], [0]

        def log_cb():
            cnt[0] += 1
            if cnt[0] % decim == 0:
                obs.append(_observe(robot, T_bel, T_base_tconn))

        adm.reset()
        adm.warmup(refs[0], tare_fn=tare)
        guard.reset()
        last_ref, seated = refs[0], False
        for i in range(1, len(refs)):
            if adm.ramp(refs[i - 1], refs[i], seg_time(refs[i - 1], refs[i]), guard,
                        on_step=log_cb) == 'seated':
                seated = True
                last_ref = refs[i]
                break
            last_ref = refs[i]
        adm.hold(last_ref, settle_s, guard, on_step=log_cb)
        lin, ang = pose_error(robot.tool0() @ T_bel, T_base_tconn)
        xyz, rpy = matrix_to_xyzrpy(inverse(T_base_tconn) @ robot.tool0() @ T_true)
        seat6 = list(xyz * 1000.0) + list(np.degrees(rpy))
        T_out = _retract_ref(last_ref, T_bel, retract_m)
        adm.ramp(last_ref, T_out, seg_time(last_ref, T_out, rv, rw), guard=None)
        adm.stop()
        return obs, seated, lin, ang, seat6

    out_dir = os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                           f'mode_pose_eval_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    log.info('Experiment folder: %s', out_dir)
    fout = open(os.path.join(out_dir, 'trials.csv'), 'w', newline='')
    writer = _csv.DictWriter(fout, fieldnames=_fieldnames(dims), restval='')
    writer.writeheader()

    standoff_axis = np.asarray(cfg.get('standoff_axis', [-1, 0, 0]), dtype=float)
    T_standoff_ref = T_base_tconn @ (translation_matrix(
        standoff_axis * float(cfg.get('standoff_distance_m', 0.03))) @ mats[0]) \
        @ inverse(T_true)
    q = robot.arm.ik(T_standoff_ref, robot.arm.q())
    if q is None or not robot.arm.move_j(q, label='approach standoff'):
        fout.close()
        return False
    seed_q = q

    ok, durations = True, []
    hist = []                                      # per-attempt history for the live figure
    try:
        for trial in range(1, num_trials + 1):
            t_trial = time.time()
            delta = fixed[trial - 1] if fixed is not None else traj.random_delta(lo, hi, rng)
            T_believed = T_true @ delta
            inj, inj_pos, inj_rot = _gt_error(T_true, T_believed)
            log.info('--- trial %d/%d --- injected xyz=[%+6.2f, %+6.2f, %+6.2f] mm '
                     'rpy=[%+6.2f, %+6.2f, %+6.2f] deg', trial, num_trials, *inj)
            acc = np.zeros((0, 12))
            diverged = abandoned = False
            for attempt in range(1, max_attempts + 1):
                errb, errb_pos, errb_rot = _gt_error(T_true, T_believed)
                obs, xstops = [], []
                seated, seat6 = False, [0.0] * 6
                for pi, poff in enumerate(sweep):
                    rows_t = traj.noised(dense, noise_rng, tn_std if tn_on else [0.0] * 6,
                                         tn_w, 0.0, 1.0, poff)
                    refs = [T_base_tconn @ r @ inverse(T_believed) for r in rows_t]
                    q = robot.arm.ik(refs[0], seed_q)
                    if q is None or not robot.arm.move_j(
                            q, label=f'trial {trial} attempt {attempt} pass {pi + 1}'):
                        abandoned = True
                        break
                    seed_q = q
                    obs_i, seated, lin, ang, seat6 = run_insertion(refs, T_believed)
                    obs.extend(obs_i)
                    if obs_i:
                        xstops.append(float(np.max(
                            np.asarray(obs_i, dtype=float).reshape(-1, 12)[:, 0])))
                    log.info('  pass %d/%d (pitch %+.1f): %d obs, stop %s.', pi + 1,
                             len(sweep), poff[4], len(obs_i),
                             [round(x, 1) for x in xstops[-1:]])
                if abandoned:
                    break
                obs_arr = np.asarray(obs, dtype=float).reshape(-1, 12)
                if save_obs:
                    _save_observations(os.path.join(
                        out_dir, f'trial_{trial:03d}_attempt_{attempt:02d}_'
                                 'observations.csv'), obs)
                full = np.vstack([acc, obs_arr]) if accumulate else obs_arr
                e_stop = None
                if basin is not None and xstops:
                    e_stop = sum(basin.stop_energy(xs, est_mode.grid6[:, est_mode.idx])
                                 for xs in xstops)
                theta6, info = two_stage(est_mode, est_pose, full, e_stop, stop_w,
                                         temp, min_w, alpha,
                                         stop_in_mode=stop_in_mode)
                row = {'trial': trial, 'attempt': attempt,
                       'n_passes': len(sweep), 'err_before_pos_mm': errb_pos,
                       'err_before_rot_deg': errb_rot,
                       'stop_fused': e_stop is not None,
                       'x_stop_mm': max(xstops) if xstops else ''}
                row.update({f'inj_{s}': v for s, v in zip(_ERR, inj)})
                row.update({f'err_before_{s}': v for s, v in zip(_ERR, errb)})
                if theta6 is None:
                    row['estimate'] = f'skipped: {info}'
                    log.warning('estimation skipped (%s)', info)
                    if accumulate:
                        acc = full
                else:
                    rank = truth_mode_rank(info['mixture'], est_mode, errb)
                    row.update({
                        'n_observations': info['n_observations'],
                        'n_modes': info['n_modes'], 'mode_mass': info['mode_mass'],
                        'ambiguity': info['ambiguity'], 'separation': info['separation'],
                        'truth_in_mode': rank == 1, 'truth_mode_rank': rank,
                        'estimate': 'ok'})
                    row.update({f'mode_centre_{d}': v
                                for d, v in info['mode_centre'].items()})
                    row.update({f'corr_{d}': v for d, v in info['theta_corr'].items()})
                    row.update({f'pose_argmin_global_{d}': v
                                for d, v in info['pose_argmin_global'].items()})
                    T_corr_mm = mats_from_vec6(theta6)
                    T_believed = T_believed @ _corr_to_m(T_corr_mm)
                    acc = _rebase_rows(full, T_corr_mm) if accumulate else acc
                    log.info('MODE: %d modes, chosen mass %.0f%%, centre %s, truth rank %d%s',
                             info['n_modes'], 100 * info['mode_mass'],
                             {d: round(v, 2) for d, v in info['mode_centre'].items()}, rank,
                             '' if rank == 1 else '  *** TRUTH NOT IN CHOSEN MODE ***')
                    log.info('POSE (within mode): %s | global pose argmin was %s',
                             {d: round(v, 2) for d, v in info['theta_corr'].items()},
                             {d: round(v, 2)
                              for d, v in info['pose_argmin_global'].items()})
                erra, erra_pos, erra_rot = _gt_error(T_true, T_believed)
                row.update({f'err_after_{s}': v for s, v in zip(_ERR, erra)})
                row.update({'err_after_pos_mm': erra_pos, 'err_after_rot_deg': erra_rot})
                row['converged'] = bool(erra_pos <= tol_pos and erra_rot <= tol_rot)
                diverged = bool((abort_pos > 0 and erra_pos > abort_pos)
                                or (abort_rot > 0 and erra_rot > abort_rot))
                row['diverged'] = diverged
                writer.writerow(row)
                fout.flush()
                os.fsync(fout.fileno())
                if theta6 is not None:
                    # mm-eq combined error track for the live figure (rot via the shared
                    # deg->mm conversion), plus the stage-1 grade for the ring markers
                    s_r = est_pose.s_rot
                    hist.append({
                        'err_before': float(np.hypot(np.linalg.norm(errb[:3]),
                                                     s_r * np.linalg.norm(errb[3:]))),
                        'err_after': float(np.hypot(np.linalg.norm(erra[:3]),
                                                    s_r * np.linalg.norm(erra[3:]))),
                        'truth_in_mode': bool(row.get('truth_in_mode'))})
                    if save_plots:
                        n_tim = sum(1 for h in hist if h['truth_in_mode'])
                        _plot_live(os.path.join(out_dir, f'trial_{trial:03d}.png'),
                                   live_path, est_mode, est_pose, info,
                                   np.asarray(errb, dtype=float), dims, hist,
                                   status=f'trial {trial}/{num_trials}  |  truth-in-mode '
                                          f'{n_tim}/{len(hist)}')
                log.info('trial %d attempt %d: %.2f mm / %.2f deg -> %.2f mm / %.2f deg%s',
                         trial, attempt, errb_pos, errb_rot, erra_pos, erra_rot,
                         '  (DIVERGED -- trial terminated)' if diverged else '')
                if diverged or row['converged']:
                    break

            # FINAL zero-noise insertion from the final belief (all collection modes end here)
            if not abandoned and not diverged:
                refs = [T_base_tconn @ r @ inverse(T_believed) for r in dense]
                q = robot.arm.ik(refs[0], seed_q)
                if q is not None and robot.arm.move_j(q, label=f'trial {trial} final'):
                    seed_q = q
                    obs, seated, lin, ang, seat6 = run_insertion(refs, T_believed)
                    errf = _gt_error(T_true, T_believed)[0]
                    depth = ''
                    if obs:
                        oa = np.asarray(obs, dtype=float).reshape(-1, 12)
                        rel = vec6_from_mats(mats_from_vec6(oa[:, :6]) @ np.linalg.inv(
                            mats_from_vec6(np.asarray(errf, dtype=float))))
                        depth = float(rel[:, 0].max())
                    frow = {'trial': trial, 'attempt': 'final_insertion',
                            'insert_depth_mm': depth, 'insert_seated': seated,
                            'insert_success': all(abs(v) <= t
                                                  for v, t in zip(seat6, succ_tol)),
                            'estimate': 'none (final insertion)'}
                    frow.update({f'seat_{s}': v for s, v in zip(_ERR, seat6)})
                    frow.update({f'err_before_{s}': v for s, v in zip(_ERR, errf)})
                    writer.writerow(frow)
                    fout.flush()
            q = robot.arm.ik(T_standoff_ref, seed_q)
            if q is None or not robot.arm.move_j(q, label='standoff'):
                ok = False
                break
            seed_q = q
            durations.append(time.time() - t_trial)
            mean_s = sum(durations) / len(durations)
            log.info('trial %d/%d took %.0f s | ETA %s (done ~%s)', trial, num_trials,
                     durations[-1], _fmt_dur(mean_s * (num_trials - trial)),
                     _clock(mean_s * (num_trials - trial)))
    except Exception:                              # noqa: BLE001
        ok = False
        log.exception('mode-pose eval error:')
    finally:
        robot.arm.servo_stop()
        fout.close()
    log.info('Done: %s', out_dir)
    return ok


def main():
    # with_gripper=False: the connector is fixtured between the closed fingers.
    run_app('Two-stage mode -> pose estimator evaluation', 'mode_pose_estimator_eval',
            build_and_run, with_gripper=False)


if __name__ == '__main__':
    main()
