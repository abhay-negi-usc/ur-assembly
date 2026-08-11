"""Estimator evaluation with a PROBE phase and an INSERTION phase (known ground truth).

The 2026-08 hose campaign's recommendations, made testable on hardware. Per trial:

    inject a known belief error (the part is FIXTURED, so the truth is exact)

    PROBE PHASE -- probe.attempts insertions whose ONLY job is to gather information:
        * their own STIFFNESS (probe.stiffness) and force guard (probe.max_force_n)
        * a LONG settle at the stop (probe.settle_s): the contact manifold was collected with
          uncertain_sampling's 5 s hold, and this arm's admittance needs ~1.2 s of true rest to
          reach 95% of the settled wrench. A short settle logs TRANSIENTS, which is what put the
          production wrench field 40.8 deg off the manifold (self-consistency floor 12.3 deg) and
          made every estimator score at corr ~0. THIS IS THE FIX -- the estimate cannot work
          until the probe process matches the manifold's.
        * a deliberate per-probe BIAS, alternating +/- probe.alternate_pitch_deg (plus an
          optional z jog): probes at different biases break aliases. On hard cases 3 alternating
          probes cut the 75th-percentile error from 11.08 to 1.63 deg.
        * the BELIEF IS NOT UPDATED between probes. With belief error E and bias B the logged
          rows are the biased path and the correction that realigns them is inverse(E)
          REGARDLESS of B -- so every probe measures the same correction and their energies
          simply ADD (skills/grid_estimator.py fuses them).

    ESTIMATE -- exhaustive grid over the fused energy (no multi-start, no aggregator choice),
        rows weighted by informativeness-vs-depth; uncertainty = curvature at the minimum
        (2 extra evaluations, AUROC 0.89-0.91) + a multi-modality FLAG (multi-modal solutions
        fail 44% of the time vs 6%). ONE correction is applied.

    INSERTION PHASE -- one guarded insertion from the corrected belief along the NOMINAL path
        with the PRODUCTION stiffness (insertion.stiffness) and settle: does the corrected
        belief actually seat? Scored against eval.success_pose_tol.

Output: data/experiments/estimator_eval_probe_<stamp>/ with trials.csv (one row per trial: the
injected error, every probe's diagnostics, the fused estimate, its uncertainty, the ground-truth
error before/after, and the insertion outcome), per-probe observation CSVs, and summary.csv.

Run:  python -m urlab.apps.estimator_eval_probe --config configs/estimator_eval_probe.yaml
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
from ..transforms import inverse, matrix_to_xyzrpy, pose_error, translation_matrix
from ._runner import run_app
from .estimator_eval import _ERR, _corr_to_m, _gt_error, _observe, _save_observations
from .uncertain_sampling import _clock, _fmt_dur, _retract_ref

log = urlog.get('estimator-eval-probe')


def _fieldnames(dims, n_probes):
    """trials.csv schema -- fixed up front so rows stream out as the run proceeds."""
    cols = ['trial', 'n_probes', 'n_observations']
    cols += [f'inj_{s}' for s in _ERR]
    cols += ['err_before_pos_mm', 'err_before_rot_deg']
    cols += [f'err_before_{s}' for s in _ERR]
    for k in range(1, n_probes + 1):
        cols += [f'probe{k}_bias_pitch_deg', f'probe{k}_bias_z_mm', f'probe{k}_n_obs',
                 f'probe{k}_seated', f'probe{k}_settled_rows', f'probe{k}_max_force_n',
                 f'probe{k}_residual']
    cols += [f'corr_{d}' for d in dims]
    cols += ['fused_residual', 'uncertainty', 'modes', 'multimodal', 'width_frac']
    cols += [f'unc_{d}' for d in dims]             # 1/curvature -- the ranking metric
    cols += [f'sigma_{d}' for d in dims]           # sqrt(E_min/curvature) -- mm / deg, plottable
    cols += [f'err_after_{s}' for s in _ERR] + ['err_after_pos_mm', 'err_after_rot_deg']
    cols += ['converged', 'insert_seated', 'insert_success', 'insert_check_pos_mm',
             'insert_check_rot_deg'] + [f'seat_{s}' for s in _ERR]
    return cols


def _plot_trial(path, est, fused, dims, truth_corr, hist, status=None, live_path=None):
    """ONE figure per trial, mirrored ATOMICALLY to a fixed live path (like estimator_eval's).

    LEFT: the FUSED energy landscape the estimate came from -- a curve for one estimated dim, a
    heatmap for two -- with the chosen correction, its per-dim +/- sigma (from the curvature, in
    the dim's own units), and the TRUE correction marked, so a wrong pick is instantly visible as
    "truth sits in a different basin" vs "truth is inside the bar".
    RIGHT (top): estimated vs TRUE correction over every trial so far, with sigma error bars and
    the ideal y = x line -- the single plot that answers "does the estimator track truth?".
    RIGHT (bottom): per-trial ground-truth error before -> after, and the uncertainty reported
    for each trial (bar), coloured by whether that trial's insertion seated.

    BEST-EFFORT: a plotting problem is logged and never allowed to kill a hardware run."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14.0, 6.4))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])
        axL = fig.add_subplot(gs[:, 0])
        axT = fig.add_subplot(gs[0, 1])
        axB = fig.add_subplot(gs[1, 1])

        info = hist[-1] if hist else None
        sig = (info or {}).get('sigma', {})
        est_corr = (info or {}).get('corr', {})

        # ---- LEFT: the fused energy landscape --------------------------------------------
        if fused is not None and len(dims) == 1:
            ax0 = est.grid_axes[0]
            axL.plot(ax0, fused, color='#4C72B0', lw=1.6)
            e = est_corr.get(dims[0])
            s = sig.get(dims[0])
            if e is not None:
                axL.axvline(e, color='#DD8452', lw=2.0, label=f'estimate {e:+.2f}')
                if s is not None and np.isfinite(s):
                    axL.axvspan(e - s, e + s, color='#DD8452', alpha=0.18,
                                label=f'+/- sigma ({s:.2f})')
            t = truth_corr.get(dims[0])
            if t is not None:
                axL.axvline(t, color='#55A868', ls='--', lw=2.0, label=f'truth {t:+.2f}')
            axL.set_xlabel(f'{dims[0]} correction')
            axL.set_ylabel('fused energy [mm-eq]')
            axL.legend(fontsize=8)
        elif fused is not None and len(dims) >= 2:
            a0, a1 = est.grid_axes[0], est.grid_axes[1]
            Eg = np.asarray(fused).reshape(est.grid_shape)
            if Eg.ndim > 2:                       # collapse any extra dims at their best slice
                Eg = Eg.reshape(len(a0), len(a1), -1).min(axis=2)
            im = axL.pcolormesh(a1, a0, Eg, shading='auto', cmap='viridis')
            fig.colorbar(im, ax=axL, label='fused energy [mm-eq]')
            e0, e1 = est_corr.get(dims[0]), est_corr.get(dims[1])
            if e0 is not None and e1 is not None:
                s0, s1 = sig.get(dims[0], np.nan), sig.get(dims[1], np.nan)
                axL.errorbar([e1], [e0],
                             xerr=[[s1], [s1]] if np.isfinite(s1) else None,
                             yerr=[[s0], [s0]] if np.isfinite(s0) else None,
                             fmt='o', ms=9, mfc='#DD8452', mec='white', ecolor='#DD8452',
                             elinewidth=2, capsize=4, label='estimate +/- sigma')
            t0, t1 = truth_corr.get(dims[0]), truth_corr.get(dims[1])
            if t0 is not None and t1 is not None:
                axL.plot([t1], [t0], '*', ms=18, mfc='#55A868', mec='white', label='truth')
            axL.set_xlabel(f'{dims[1]} correction')
            axL.set_ylabel(f'{dims[0]} correction')
            axL.legend(fontsize=8, loc='upper right')
        else:
            axL.text(0.5, 0.5, 'no fused energy', ha='center', va='center')
        axL.set_title('fused probe energy (the landscape the estimate came from)', fontsize=10)

        # ---- RIGHT TOP: estimated vs true correction, with sigma bars ---------------------
        d0 = dims[0]
        tv = [h['truth'].get(d0) for h in hist if h.get('corr')]
        ev = [h['corr'].get(d0) for h in hist if h.get('corr')]
        sv = [h.get('sigma', {}).get(d0, np.nan) for h in hist if h.get('corr')]
        if tv:
            sv = np.array([s if np.isfinite(s) else 0.0 for s in sv], dtype=float)
            axT.errorbar(tv, ev, yerr=sv, fmt='o', ms=5, color='#4C72B0',
                         ecolor='#9ab4d4', elinewidth=1.2, capsize=2)
            lim = max(np.max(np.abs(tv)), np.max(np.abs(ev)), 1e-3) * 1.15
            axT.plot([-lim, lim], [-lim, lim], ls='--', lw=1.0, color='#888888')
            axT.set_xlim(-lim, lim)
            axT.set_ylim(-lim, lim)
            if len(tv) > 2 and np.std(tv) > 1e-9 and np.std(ev) > 1e-9:
                axT.set_title(f'estimate vs truth ({d0}): corr '
                              f'{np.corrcoef(tv, ev)[0, 1]:+.2f}, n={len(tv)}', fontsize=10)
            else:
                axT.set_title(f'estimate vs truth ({d0}), n={len(tv)}', fontsize=10)
        axT.set_xlabel(f'TRUE {d0} correction')
        axT.set_ylabel('estimated')

        # ---- RIGHT BOTTOM: error before/after + the reported uncertainty ------------------
        n = np.arange(1, len(hist) + 1)
        eb = [h['err_before'] for h in hist]
        ea = [h['err_after'] for h in hist]
        axB.plot(n, eb, 'o-', color='#C44E52', lw=1.2, ms=4, label='|err| before')
        axB.plot(n, ea, 'o-', color='#55A868', lw=1.6, ms=5, label='|err| after')
        axB.set_xlabel('trial')
        axB.set_ylabel('ground-truth error [mm]')
        axB.set_ylim(bottom=0.0)
        axB.set_xticks(n if len(n) <= 20 else n[:: max(len(n) // 20, 1)])
        axB.legend(fontsize=8, loc='upper left')
        axU = axB.twinx()
        unc = [h.get('sigma', {}).get(d0, np.nan) for h in hist]
        cols = ['#4C72B0' if h.get('insert_success') else '#bbbbbb' for h in hist]
        axU.bar(n, unc, width=0.55, color=cols, alpha=0.45, zorder=0)
        axU.set_ylabel(f'reported sigma [{ "deg" if d0.endswith("_deg") else "mm" }]  '
                       '(blue = seated)', fontsize=8)
        axU.set_ylim(bottom=0.0)
        for i, h in enumerate(hist):
            if h.get('multimodal'):
                axU.annotate('M', (i + 1, unc[i] if np.isfinite(unc[i]) else 0.0),
                             textcoords='offset points', xytext=(0, 3), ha='center',
                             fontsize=7, color='#C44E52')

        fig.suptitle('estimator eval: probe phase + insertion phase', y=0.99)
        if status:
            fig.text(0.99, 0.965, status, ha='right', fontsize=9, color='#333333')
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        if live_path:
            tmp = live_path + '.tmp'               # temp + replace: viewers never see a
            fig.savefig(tmp, dpi=110, format='png')   # half-written PNG
            os.replace(tmp, live_path)
        plt.close(fig)
    except Exception as exc:                       # noqa: BLE001 -- plotting is never fatal
        log.warning('trial plot skipped (%s)', exc)


def _write_summary(out_dir, rows, dims):
    if not rows:
        return
    rec = {'n': len(rows)}
    for key in ('err_before_pos_mm', 'err_after_pos_mm', 'err_before_rot_deg',
                'err_after_rot_deg'):
        v = np.array([float(r[key]) for r in rows if r.get(key) not in (None, '')])
        if len(v):
            rec[f'mean_{key}'] = float(v.mean())
            rec[f'median_{key}'] = float(np.median(v))
    for d in dims:
        v = np.abs([float(r[f'err_after_{d}']) for r in rows if r.get(f'err_after_{d}') != ''])
        if len(v):
            rec[f'median_abs_after_{d}'] = float(np.median(v))
    for key, label in (('insert_success', 'insert_success_frac'),
                       ('converged', 'converged_frac'), ('multimodal', 'multimodal_frac')):
        v = [bool(r.get(key)) for r in rows if r.get(key) not in (None, '')]
        if v:
            rec[label] = float(np.mean(v))
    path = os.path.join(out_dir, 'summary.csv')
    with open(path, 'w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rec))
        w.writeheader()
        w.writerow(rec)
    log.info('SUMMARY  n=%d  gt error %.2f -> %.2f mm  insert success %.0f%%  multimodal %.0f%%',
             rec['n'], rec.get('median_err_before_pos_mm', float('nan')),
             rec.get('median_err_after_pos_mm', float('nan')),
             100.0 * rec.get('insert_success_frac', 0.0),
             100.0 * rec.get('multimodal_frac', 0.0))
    log.info('Summary: %s', path)


def build_and_run(cfg, robot, camera, args):
    ev = cfg.section('eval')
    pb = cfg.section('probe')
    ins = cfg.section('insertion')

    # The estimator first -- a bad grid/manifold config must fail before the robot moves.
    estimator = GridManifoldEstimator(cfg.section('estimation'))
    dims = estimator.estimate_dims

    held_name = cfg.get('held_frame')
    if not held_name:
        log.error('held_frame is required -- the catalogue frame IS the ground truth here.')
        return False
    frames = tool_frames.load_frames(cfg)
    targets = tool_frames.load_targets(cfg)
    if held_name not in frames or held_name not in targets:
        log.error('held_frame %r needs BOTH a frames: and a targets: entry in %s.',
                  held_name, tool_frames.frames_path(cfg))
        return False
    T_true = frames[held_name]
    T_base_tconn = targets[held_name]
    log.info('Ground truth: held frame %r + its recorded mate.', held_name)

    csv_in = urconfig.resolve(cfg, cfg.get('trajectory_csv', 'assembly_trajectory.csv'))
    mats = traj.load_csv(csv_in, angles_deg=bool(cfg.get('trajectory_angles_deg', False)))
    dense = traj.resample(mats, float(cfg.get('translational_resolution_m', 0.001)),
                          float(cfg.get('rotational_resolution_deg', 1.0)))

    # ---- injected belief errors (same conventions as estimator_eval) ----
    pert = ev.get('perturbation', {}) or {}
    lo = pert.get('lower', [0.0, 0.0, -0.005, 0.0, -5.0, 0.0])
    hi = pert.get('upper', [0.0, 0.0, 0.005, 0.0, 5.0, 0.0])
    mode = str(ev.get('mode', 'bounds')).lower()
    if mode not in ('random', 'grid', 'bounds'):
        log.error("eval.mode %r must be 'random', 'grid' or 'bounds'.", mode)
        return False
    try:
        if mode == 'grid':
            fixed = traj.grid_deltas(lo, hi, ev.get('grid_resolution', [0.0] * 6))
        elif mode == 'bounds':
            fixed = traj.bounds_deltas(lo, hi,
                                       simultaneous=bool(ev.get('bounds_simultaneous', False)))
        else:
            fixed = None
    except ValueError as exc:
        log.error('%s', exc)
        return False
    if fixed is not None and not fixed:
        log.error('eval.mode %r produced 0 trials -- every perturbation bound is zero.', mode)
        return False
    num_trials = len(fixed) if fixed is not None else int(ev.get('num_trials', 20))
    seed = int(ev.get('random_seed', 0))
    rng = np.random.default_rng(seed if seed > 0 else None)
    tol = ev.get('success_tolerance', {}) or {}
    tol_pos_mm = float(tol.get('pos_mm', 1.0))
    tol_rot_deg = float(tol.get('rot_deg', 1.0))
    succ_tol = [float(v) for v in (ev.get('success_pose_tol')
                                   or [2.0, 1.0, 5.0, 5.0, 5.0, 1.0])]
    if len(succ_tol) != 6:
        log.error('eval.success_pose_tol must have 6 entries.')
        return False
    decim = max(1, int(ev.get('log_decimation', 5)))
    save_obs = bool(ev.get('save_observations', True))
    save_plots = bool(ev.get('save_plots', True))
    # LIVE figure: ONE fixed path OUTSIDE the per-experiment folder, atomically overwritten after
    # every trial -- keep it open in an image viewer across trials and runs. true =
    # data/experiments/estimator_eval_live.png (the SAME file estimator_eval writes, so one
    # viewer serves both apps); a string = explicit path; false disables.
    live = ev.get('live_plot', True)
    live_path = None
    if save_plots and live:
        live_path = live if isinstance(live, str) else os.path.join(
            cfg.get('data_dir', 'data'), 'experiments', 'estimator_eval_live.png')
        os.makedirs(os.path.dirname(live_path) or '.', exist_ok=True)
        log.info('Live figure: %s', live_path)

    # ---- PROBE phase settings ----
    n_probes = max(1, int(pb.get('attempts', 3)))
    alt_pitch = float(pb.get('alternate_pitch_deg', 5.0))
    z_jog_mm = float(pb.get('alternate_z_mm', 0.0))
    probe_settle_s = float(pb.get('settle_s', 2.0))
    if probe_settle_s < 1.0:
        log.warning('probe.settle_s = %.2f s is SHORT: this arm needs ~1.2 s of rest to reach '
                    '95%% of the settled contact wrench, and the manifold was collected with a '
                    '5 s hold. Expect the manifold match -- and the estimate -- to degrade.',
                    probe_settle_s)
    probe_retract_m = float(pb.get('retract_distance_m', cfg.get('retract_distance_m', 0.05)))

    # PER-WAYPOINT TRAJECTORY NOISE -- process-critical, not a nicety. The manifold was collected
    # by uncertain_sampling with a fresh random offset on EVERY waypoint, and that jitter is what
    # walks a misaligned cylindrical peg off the hole's rim and into the bore. A clean path does
    # not: the peg's leading face lands on the rim and the insertion stops dead. Measured on the
    # 2026-08-11 runs -- early-landing (max true x < -14 mm) was 6% for the manifold, 32% for the
    # noised estimator_eval, and 68% for the first, NOISE-FREE version of this app. Keep this
    # matched to the map's collection process or the probes never reach the contact the map
    # describes.
    tn = pb.get('trajectory_noise', {}) or {}
    tn_on = bool(tn.get('enabled', True))
    tn_std = [float(v) for v in (tn.get('std') or [0.0, 0.0005, 0.005, 0.0, 5.0, 0.0])]
    if len(tn_std) != 6:
        log.error('probe.trajectory_noise.std must have 6 entries [x,y,z (m), r,p,y (deg)].')
        return False
    tn_w = max(1, int(tn.get('smooth_window', 1)))
    tn_dt = float(tn.get('decay_traj', 0.0))
    noise_rng = np.random.default_rng(seed + 17 if seed > 0 else None)
    if tn_on:
        log.info('Probe trajectory noise ON: per-waypoint std [%.1f, %.1f, %.1f] mm / '
                 '[%.1f, %.1f, %.1f] deg, smooth window %d.', tn_std[0] * 1000,
                 tn_std[1] * 1000, tn_std[2] * 1000, tn_std[3], tn_std[4], tn_std[5], tn_w)
    else:
        log.warning('Probe trajectory noise is OFF. A clean path lands a misaligned peg on the '
                    'hole RIM instead of walking it in -- expect most probes to stop ~15 mm '
                    'short of the contact the manifold describes.')

    comp_probe = dict(cfg.section('compliance'))
    if pb.get('stiffness') is not None:
        comp_probe['stiffness'] = [float(v) for v in pb['stiffness']]
    comp_probe['settle_s'] = probe_settle_s
    adm_probe = AdmittanceController(robot.arm, comp_probe)
    guard_probe = ForceGuard(robot.arm, dict(cfg.section('force_guard'),
                                             **({'max_force_n': float(pb['max_force_n'])}
                                                if pb.get('max_force_n') is not None else {})))

    comp_ins = dict(cfg.section('compliance'))
    if ins.get('stiffness') is not None:
        comp_ins['stiffness'] = [float(v) for v in ins['stiffness']]
    ins_settle_s = float(ins.get('settle_s', comp_ins.get('settle_s', 0.5)))
    comp_ins['settle_s'] = ins_settle_s
    adm_ins = AdmittanceController(robot.arm, comp_ins)
    guard_ins = ForceGuard(robot.arm, dict(cfg.section('force_guard'),
                                           **({'max_force_n': float(ins['max_force_n'])}
                                              if ins.get('max_force_n') is not None else {})))
    log.info('PROBE  phase: %d probes, stiffness %s, settle %.2f s, guard %.0f N, '
             'alternating pitch %+.1f deg / z %+.1f mm.', n_probes,
             comp_probe.get('stiffness'), probe_settle_s,
             float(guard_probe.max_force_n) if hasattr(guard_probe, 'max_force_n')
             else float('nan'), alt_pitch, z_jog_mm)
    log.info('INSERT phase: stiffness %s, settle %.2f s.', comp_ins.get('stiffness'),
             ins_settle_s)

    tare = (lambda: robot.arm.zero_ft(settle=False)) \
        if bool(cfg.get_path('compliance.tare_before', True)) else None
    v_mm_s = float(cfg.get_path('speed.max_cartesian_translation_mm_s', 5.0))
    w_deg_s = float(cfg.get_path('speed.max_cartesian_rotation_deg_s', 30.0))
    rv_mm_s = float(cfg.get_path('speed.retract_translation_mm_s', 50.0))
    rw_deg_s = float(cfg.get_path('speed.retract_rotation_deg_s', 50.0))
    min_seg_s = 1.0 / adm_probe.rate

    def seg_time(A, B, v=None, w=None):
        v = v_mm_s if v is None else v
        w = w_deg_s if w is None else w
        lin_m, ang_rad = pose_error(A, B)
        return max((lin_m * 1000.0 / v) if v > 0 else 0.0,
                   (np.degrees(ang_rad) / w) if w > 0 else 0.0, min_seg_s)

    def run_insertion(adm_ctl, guard, refs, T_bel, settle_s, retract_m):
        """One admittance insertion + hold + compliant retract. Returns the observation rows,
        whether contact stopped the advance, the kinematic check, the TRUE seat pose, and how
        many logged rows were STATIC (the settled-wrench evidence)."""
        obs, cnt = [], [0]

        def log_cb():
            cnt[0] += 1
            if cnt[0] % decim == 0:
                obs.append(_observe(robot, T_bel, T_base_tconn))

        adm_ctl.reset()
        adm_ctl.warmup(refs[0], tare_fn=tare)
        guard.reset()
        last_ref, seated = refs[0], False
        for i in range(1, len(refs)):
            res = adm_ctl.ramp(refs[i - 1], refs[i], seg_time(refs[i - 1], refs[i]),
                               guard, on_step=log_cb)
            last_ref = refs[i]
            if res == 'seated':
                seated = True
                log.info('  contact limit at waypoint %d/%d.', i, len(refs) - 1)
                break
        n_before = len(obs)
        adm_ctl.hold(last_ref, settle_s, guard, on_step=log_cb)
        # settled-row evidence: of the rows logged during the hold, how many barely moved
        settled = 0
        if len(obs) - n_before >= 2:
            hold_rows = np.asarray(obs[n_before:], dtype=float)
            step = (np.linalg.norm(np.diff(hold_rows[:, :3], axis=0), axis=1)
                    + 0.2 * np.linalg.norm(np.diff(hold_rows[:, 3:6], axis=0), axis=1))
            settled = int(np.sum(step < 0.15))
        lin, ang = pose_error(robot.tool0() @ T_bel, T_base_tconn)
        xyz, rpy = matrix_to_xyzrpy(inverse(T_base_tconn) @ robot.tool0() @ T_true)
        seat6 = list(xyz * 1000.0) + list(np.degrees(rpy))
        T_out = _retract_ref(last_ref, T_bel, retract_m)
        adm_ctl.ramp(last_ref, T_out, seg_time(last_ref, T_out, rv_mm_s, rw_deg_s), guard=None)
        adm_ctl.stop()
        return obs, seated, lin, ang, seat6, settled

    out_dir = os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                           f'estimator_eval_probe_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    log.info('Experiment folder: %s', out_dir)
    try:
        import json
        with open(os.path.join(out_dir, 'eval_config.json'), 'w') as fh:
            json.dump({'held_frame': held_name, 'trajectory_csv': csv_in,
                       'num_trials': num_trials, 'eval': ev, 'probe': pb, 'insertion': ins,
                       'estimation': cfg.section('estimation'),
                       'compliance': cfg.section('compliance'),
                       'force_guard': cfg.section('force_guard')}, fh, indent=2, default=str)
    except Exception as exc:                       # noqa: BLE001
        log.warning('eval_config.json skipped (%s)', exc)

    fout = open(os.path.join(out_dir, 'trials.csv'), 'w', newline='')
    writer = _csv.DictWriter(fout, fieldnames=_fieldnames(dims, n_probes), restval='')
    writer.writeheader()

    standoff_axis = np.asarray(cfg.get('standoff_axis', [-1, 0, 0]), dtype=float)
    T_standoff_row = translation_matrix(
        standoff_axis * float(cfg.get('standoff_distance_m', 0.025))) @ mats[0]
    T_standoff_ref = T_base_tconn @ T_standoff_row @ inverse(T_true)
    q_home = robot.arm.q()
    q = robot.arm.ik(T_standoff_ref, q_home)
    if q is None or not robot.arm.move_j(q, label='approach standoff'):
        fout.close()
        return False
    seed_q = q

    rows, ok, durations, n_succ = [], True, [], 0
    hist = []                                      # per-trial plot record (see _plot_trial)
    try:
        for trial in range(1, num_trials + 1):
            t_trial = time.time()
            delta = fixed[trial - 1] if fixed is not None else traj.random_delta(lo, hi, rng)
            T_believed = T_true @ delta
            inj, inj_pos, inj_rot = _gt_error(T_true, T_believed)
            errb, errb_pos, errb_rot = inj, inj_pos, inj_rot
            log.info('--- trial %d/%d --- injected xyz=[%+6.2f, %+6.2f, %+6.2f] mm '
                     'rpy=[%+6.2f, %+6.2f, %+6.2f] deg', trial, num_trials, *inj)

            row = {'trial': trial, 'n_probes': n_probes,
                   'err_before_pos_mm': errb_pos, 'err_before_rot_deg': errb_rot}
            row.update({f'inj_{s}': v for s, v in zip(_ERR, inj)})
            row.update({f'err_before_{s}': v for s, v in zip(_ERR, errb)})

            # ---------------- PROBE PHASE (belief held FIXED throughout) ----------------
            fused = None
            n_obs_total = 0
            abandoned = False
            for k in range(1, n_probes + 1):
                sign = 1.0 if k % 2 else -1.0
                bias = [0.0, 0.0, sign * z_jog_mm / 1000.0, 0.0, sign * alt_pitch, 0.0] \
                    if (alt_pitch or z_jog_mm) else None
                # The deliberate bias AND the per-waypoint jitter, redrawn for every probe (the
                # jitter is what gets the peg past the rim -- see the note where it is read).
                if tn_on or bias:
                    rows_t = traj.noised(dense, noise_rng, tn_std if tn_on else [0.0] * 6,
                                         tn_w, tn_dt, 1.0, bias)
                else:
                    rows_t = dense
                refs = [T_base_tconn @ r @ inverse(T_believed) for r in rows_t]
                q = robot.arm.ik(refs[0], seed_q)
                if q is None or not robot.arm.move_j(q, label=f'trial {trial} probe {k} start'):
                    log.warning('IK/approach failed; abandoning trial %d.', trial)
                    abandoned = True
                    break
                seed_q = q
                obs, seated, lin, ang, seat6, settled = run_insertion(
                    adm_probe, guard_probe, refs, T_believed, probe_settle_s, probe_retract_m)
                obs_arr = np.asarray(obs, dtype=float).reshape(-1, 12)
                fmax = float(np.linalg.norm(obs_arr[:, 6:9], axis=1).max()) if len(obs_arr) else 0.0
                if save_obs:
                    _save_observations(os.path.join(
                        out_dir, f'trial_{trial:03d}_probe_{k:02d}_observations.csv'), obs)
                row[f'probe{k}_bias_pitch_deg'] = sign * alt_pitch
                row[f'probe{k}_bias_z_mm'] = sign * z_jog_mm
                row[f'probe{k}_n_obs'] = len(obs)
                row[f'probe{k}_seated'] = seated
                row[f'probe{k}_settled_rows'] = settled
                row[f'probe{k}_max_force_n'] = fmax
                if settled < 3:
                    log.warning('  probe %d logged only %d settled rows -- the hold is too '
                                'short for this admittance; the wrench is a TRANSIENT and will '
                                'not match the manifold.', k, settled)
                if len(obs_arr) < estimator.min_observations:
                    log.warning('  probe %d: only %d observations, skipped.', k, len(obs_arr))
                    continue
                vec6, w6 = estimator.prepare_observations(
                    obs_arr[:, :6], obs_arr[:, 6:9], obs_arr[:, 9:12])
                E, n = estimator.energy(vec6, w6)
                if E is None:
                    log.warning('  probe %d: %d rows after filtering, skipped.', k, n)
                    continue
                n_obs_total += n
                fused = E if fused is None else fused + E
                _, pinfo = estimator.solve(E, n)
                row[f'probe{k}_residual'] = pinfo['final_residual']
                log.info('  probe %d (bias pitch %+.1f deg): %d obs (%d settled), '
                         'max |f| %.1f N, this-probe corr %s', k, sign * alt_pitch, n, settled,
                         fmax, {d: round(v, 2) for d, v in pinfo['theta_corr'].items()})
            if abandoned:
                rows.append(row)
                writer.writerow(row)
                fout.flush()
                continue

            # ---------------- FUSED ESTIMATE (one correction) ----------------
            row['n_observations'] = n_obs_total
            # The TRUE correction: believed @ C = true, so C = inverse(err_before).
            truth6 = vec6_from_mats(inverse(mats_from_vec6(np.asarray(errb, dtype=float))))
            truth_corr = {d: float(truth6[j]) for d, j in zip(dims, estimator.idx)}
            rec = {'truth': truth_corr, 'err_before': errb_pos, 'err_after': errb_pos}
            if fused is None:
                log.error('trial %d: no usable probe observations -- belief unchanged.', trial)
            else:
                T_corr_mm, info = estimator.solve(fused, n_obs_total)
                T_believed = T_believed @ _corr_to_m(T_corr_mm)
                row.update({f'corr_{d}': v for d, v in info['theta_corr'].items()})
                row.update({'fused_residual': info['final_residual'],
                            'uncertainty': info['uncertainty'], 'modes': info['modes'],
                            'multimodal': info['multimodal'],
                            'width_frac': info['width_frac']})
                row.update({f'unc_{d}': v for d, v in info['curvature_uncertainty'].items()})
                row.update({f'sigma_{d}': v for d, v in info['sigma'].items()})
                rec.update({'corr': info['theta_corr'], 'sigma': info['sigma'],
                            'multimodal': info['multimodal']})
                log.info('FUSED estimate over %d probes: %s | sigma %s | residual %.3f  '
                         'modes %d%s', n_probes,
                         {d: round(v, 3) for d, v in info['theta_corr'].items()},
                         {d: round(v, 2) for d, v in info['sigma'].items()},
                         info['final_residual'], info['modes'],
                         '  *** MULTI-MODAL: treat as suspect ***' if info['multimodal'] else '')
                log.info('   truth would be %s',
                         {d: round(v, 3) for d, v in truth_corr.items()})
            erra, erra_pos, erra_rot = _gt_error(T_true, T_believed)
            row.update({f'err_after_{s}': v for s, v in zip(_ERR, erra)})
            row.update({'err_after_pos_mm': erra_pos, 'err_after_rot_deg': erra_rot})
            row['converged'] = bool(erra_pos <= tol_pos_mm and erra_rot <= tol_rot_deg)
            rec['err_after'] = erra_pos
            hist.append(rec)
            log.info('gt error %.2f mm / %.2f deg -> %.2f mm / %.2f deg%s',
                     errb_pos, errb_rot, erra_pos, erra_rot,
                     '  (CONVERGED)' if row['converged'] else '')

            # ---------------- INSERTION PHASE (production stiffness, nominal path) ----------
            refs = [T_base_tconn @ r @ inverse(T_believed) for r in dense]
            q = robot.arm.ik(refs[0], seed_q)
            if q is None or not robot.arm.move_j(q, label=f'trial {trial} insertion'):
                log.warning('IK/approach failed for the insertion of trial %d.', trial)
            else:
                seed_q = q
                obs, seated, lin, ang, seat6, settled = run_insertion(
                    adm_ins, guard_ins, refs, T_believed, ins_settle_s,
                    float(cfg.get('retract_distance_m', 0.05)))
                if save_obs:
                    _save_observations(os.path.join(
                        out_dir, f'trial_{trial:03d}_insertion_observations.csv'), obs)
                succ = all(abs(v) <= t for v, t in zip(seat6, succ_tol))
                n_succ += int(succ)
                row.update({'insert_seated': seated, 'insert_success': succ,
                            'insert_check_pos_mm': lin * 1000.0,
                            'insert_check_rot_deg': float(np.degrees(ang))})
                row.update({f'seat_{s}': v for s, v in zip(_ERR, seat6)})
                rec['insert_success'] = bool(succ)
                log.info('INSERTION: seated=%s success=%s  seat xyz=[%+5.2f, %+5.2f, %+5.2f] mm '
                         'rpy=[%+5.2f, %+5.2f, %+5.2f] deg  (running %d/%d)',
                         seated, succ, *seat6, n_succ, trial)

            if save_plots:
                status = (f'trial {trial}/{num_trials}  |  seated {n_succ}/{trial} '
                          f'({n_succ / max(trial, 1):.0%})')
                _plot_trial(os.path.join(out_dir, f'trial_{trial:03d}.png'), estimator, fused,
                            dims, truth_corr, hist, status, live_path)

            rows.append(row)
            writer.writerow(row)
            fout.flush()
            os.fsync(fout.fileno())

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
        _write_summary(out_dir, rows, dims)
    if ok:
        robot.arm.move_j(q_home, label='home')
        log.info('Evaluation complete: %s (%d trials)', out_dir, len(rows))
    return ok


def main():
    # with_gripper=False: the connector is FIXTURED between the closed fingers.
    run_app('Estimator evaluation: probe phase + insertion phase', 'estimator_eval_probe',
            build_and_run, with_gripper=False)


if __name__ == '__main__':
    main()
