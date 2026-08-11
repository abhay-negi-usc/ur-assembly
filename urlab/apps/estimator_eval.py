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
            estimate   manifold ICP over the trial's ACCUMULATED observations (default;
                       eval.accumulate_observations -- prior attempts are RE-PROJECTED into
                       the current belief) -> T_corr;  T_believed <- T_believed @ T_corr
            score      remaining GROUND-TRUTH error = inverse(T_true) @ T_believed -- logged
                       per attempt; within eval.success_tolerance -> converged (early stop)
            success    the TRUE connector pose wrt the TARGET at the end of the hold within
                       eval.success_pose_tol per DOF -> the trial TERMINATES; the live figure
                       shows the running trial count + success rate
        final      OPTIONAL (eval.final_insertion): ONE more guarded insertion from the FINAL
                   corrected belief under a DIFFERENT stiffness -- seats-or-not, no estimation
        disassemble: back at the stand-off (free space) before the next trial

Options: eval.trajectory_noise adds smoothed per-waypoint noise (redrawn per attempt, its own
random stream); eval.live_plot mirrors the current trial's figure to ONE fixed path outside the
experiment folder, atomically, so it can stay open in an image viewer across trials and runs.

The GRIPPER IS NEVER OPENED OR CLOSED (with_gripper=False) -- the part is fixtured in the closed
fingers and an open would drop it mid-run.

Output: data/experiments/estimator_eval_<timestamp>/ with trials.csv (one row per attempt:
injected error, ground-truth error before/after the update, the correction, ICP diagnostics),
per-attempt observation CSVs for post-analysis, and summary.csv (per-attempt-index aggregate).

Units: robot poses are metres/radians (repo convention); the manifold space and all logged errors
are mm/deg -- conversions happen only at the observation/logging boundary in this file.
"""

import csv as _csv
import itertools
import os
import time
from datetime import datetime

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from .. import tool_frames
from ..robot import AdmittanceController, ForceGuard
from ..skills import trajectory as traj
from ..skills.manifold import (FORCE_COLS, POSE_COLS, TORQUE_COLS,
                               mats_from_vec6, scaled12, vec6_from_mats)
from ..skills.solution_check import CheckedManifoldEstimator
from ..skills.success_basin import SuccessBasin
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


def _rebase_rows(rows12, T_corr_mm):
    """Re-express logged observation rows under the belief UPDATED by T_corr.

    A logged pose is inverse(T_target) @ tool0 @ T_believed_old and the update is
    T_believed_new = T_believed_old @ T_corr, so rel_new = rel_old @ T_corr exactly (the
    physical robot pose in the log never changes). The wrench columns live in the believed
    connector frame, which moves the same way: transform_wrench's convention with
    T_ba = inverse(T_corr), vectorized over rows (metres for the cross term)."""
    pose_new = vec6_from_mats(mats_from_vec6(rows12[:, :6]) @ np.asarray(T_corr_mm, dtype=float))
    Tinv = inverse(_corr_to_m(np.asarray(T_corr_mm, dtype=float)))
    R, p = Tinv[:3, :3], Tinv[:3, 3]
    f_new = rows12[:, 6:9] @ R.T
    tau_new = rows12[:, 9:12] @ R.T + np.cross(p, f_new)
    return np.hstack([pose_new, f_new, tau_new])


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


# The solution check's signal names (urlab/skills/solution_check.py) -- fixed here so the
# trials.csv schema is stable whether or not a given signal is available on an attempt
# (u_cfg needs check.config_disagreement; absent signals write as '').
_TRUST_SIGNALS = ('u_post', 'u_spread', 'u_split', 'u_res', 'u_depth', 'u_cfg')


def _fieldnames(dims):
    """trials.csv schema -- fixed up-front so the file is written incrementally, row by row."""
    return (['trial', 'attempt', 'n_observations', 'seated', 'check_pos_mm', 'check_rot_deg']
            + [f'seat_{s}' for s in _ERR] + ['success']
            + [f'inj_{s}' for s in _ERR]
            + [f'err_before_{s}' for s in _ERR] + ['err_before_pos_mm', 'err_before_rot_deg']
            + [f'corr_{d}' for d in dims] + [f'sigma_{d}' for d in dims]
            + [f'cov_{a}{b}' for a in range(len(dims)) for b in range(a, len(dims))]
            + ['p_seat', 'gate_opened', 'insert_depth_mm', 'seat_depth_mm', 'seated_basin']
            + ['icp_inliers', 'icp_residual', 'estimate']
            + ['trust_rankavg2', 'trust_cauchy'] + [f'trust_{s}' for s in _TRUST_SIGNALS]
            + [f'err_after_{s}' for s in _ERR] + ['err_after_pos_mm', 'err_after_rot_deg']
            + ['converged'])


def _landscape(estimator, vec6, w6, max_pts=2600, row_cap=120):
    """The ICP energy over a DENSE GRID of candidate corrections, in the estimator's own metric.

    The multi-start solver never builds this -- it only samples it from random starts -- so the
    landscape is what shows whether the applied correction sits in the true basin, whether the
    basin is flat (uncertain) and whether a rival mode is deeper. Returns
    (axes, E, sigma, argmin) with E shaped like the mesh, sigma per dim in that dim's own units
    (sqrt(E_min / curvature), the 2026-08 campaign's uncertainty winner) and argmin the grid's
    best correction. Cost is bounded by subsampling the grid and the observation rows."""
    dims = estimator.estimate_dims
    halves = [max(float(estimator.init_range.get(d, 8.0)), 6.0 if d.endswith('_mm') else 10.0)
              for d in dims]
    n_per = max(int(round(max_pts ** (1.0 / max(len(dims), 1)))), 9)
    axes = [np.linspace(-h, h, n_per) for h in halves]
    mesh = np.meshgrid(*axes, indexing='ij')
    G = np.zeros((mesh[0].size, 6))
    for j, m in zip(estimator.idx, mesh):
        G[:, j] = m.ravel()
    v6, w = np.asarray(vec6, dtype=float), np.asarray(w6, dtype=float)
    if len(v6) > row_cap:                          # bound the cost; the shape is unaffected
        sel = np.linspace(0, len(v6) - 1, row_cap).astype(int)
        v6, w = v6[sel], w[sel]
    C = np.einsum('nij,kjl->knil', mats_from_vec6(v6), mats_from_vec6(G))
    pts = scaled12(vec6_from_mats(C), w, estimator.s_rot).reshape(-1, 12)
    dist, nn = estimator.tree.query(pts, k=estimator.interp_neighbors, workers=-1)
    if estimator.interp_neighbors > 1:
        bw = np.exp(-(dist - dist[:, :1]) / estimator.interp_tau)
        bw /= bw.sum(axis=1, keepdims=True)
        tgt = np.einsum('mk,mkd->md', bw, estimator.M12[nn])
        d1 = np.linalg.norm(tgt - pts, axis=1)
    else:
        d1 = dist if dist.ndim == 1 else dist[:, 0]
    E = d1.reshape(len(G), len(v6)).mean(axis=1).reshape(mesh[0].shape)

    k = np.unravel_index(int(np.argmin(E)), E.shape)
    Emin = float(E[k])
    n = len(dims)
    argmin = {d: float(ax[k[a]]) for a, (d, ax) in enumerate(zip(dims, axes))}

    # FULL Hessian at the minimum -> covariance. The per-axis sigma alone is misleading whenever
    # the dims trade off (the z-pitch valley): it reports the CONDITIONAL width and hides the
    # correlation. Cov = E_min * inv(H) gives the MARGINAL widths on its diagonal and, through
    # its eigenvectors, the stiff/sloppy directions -- which is what the ellipse draws.
    H = np.zeros((n, n))
    probe = []
    for a, ax in enumerate(axes):
        step = ax[1] - ax[0]
        probe.append(max(int(round(3.0 / step)), 1))   # +-3 mm / deg: validated probe distance
    for a in range(n):
        lo, hi = list(k), list(k)
        lo[a] = max(k[a] - probe[a], 0)
        hi[a] = min(k[a] + probe[a], len(axes[a]) - 1)
        h = 0.5 * (axes[a][hi[a]] - axes[a][lo[a]])
        H[a, a] = ((float(E[tuple(lo)]) - 2.0 * Emin + float(E[tuple(hi)])) / (h ** 2)
                   if h > 0 else 0.0)
    for a in range(n):
        for b in range(a + 1, n):
            ia, ib = probe[a], probe[b]
            def at(sa, sb):
                q = list(k)
                q[a] = int(np.clip(k[a] + sa * ia, 0, len(axes[a]) - 1))
                q[b] = int(np.clip(k[b] + sb * ib, 0, len(axes[b]) - 1))
                return float(E[tuple(q)])
            ha = axes[a][int(np.clip(k[a] + ia, 0, len(axes[a]) - 1))] - axes[a][k[a]]
            hb = axes[b][int(np.clip(k[b] + ib, 0, len(axes[b]) - 1))] - axes[b][k[b]]
            if ha > 0 and hb > 0:
                H[a, b] = H[b, a] = (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / (
                    4.0 * ha * hb)
    try:
        w_h, V_h = np.linalg.eigh(H)
        floor = max(1e-9, 1e-6 * max(abs(w_h).max(), 1e-9))
        cov = (V_h * (max(Emin, 1e-12) / np.maximum(w_h, floor))) @ V_h.T
    except np.linalg.LinAlgError:
        cov = np.eye(n) * float(axes[0][-1]) ** 2
    sigma = {d: float(np.sqrt(max(cov[a, a], 0.0)))
             for a, d in enumerate(dims)}
    for a, d in enumerate(dims):                   # never claim more than the box we searched
        sigma[d] = float(min(sigma[d], axes[a][-1]))
    return axes, E, sigma, argmin, cov


def _as_error(errb, theta, idx):
    """Corrections -> the REMAINING ERROR they would leave, in the estimated dims.

    Applying correction T to belief B gives B @ T, so the error left against the truth is
    err_before (.) theta -- exact, not a subtraction. Plotting in this frame makes every attempt
    and every trial share ONE set of axes with TRUTH AT THE ORIGIN, instead of a correction frame
    that shifts every time the belief updates."""
    th = np.atleast_2d(np.asarray(theta, dtype=float))
    th6 = np.zeros((len(th), 6))
    th6[:, idx] = th
    rem = vec6_from_mats(mats_from_vec6(np.asarray(errb, dtype=float)) @ mats_from_vec6(th6))
    return rem[:, idx]


def _plot_trial_errors(path, trial, dims, err6, residuals, s_rot, live_path=None,
                       res_all=None, l2_all=None, status=None, land=None):
    """ONE figure per trial, RE-SAVED after every attempt: the GROUND-TRUTH error, all attempts
    co-plotted (x = 0 is the injected error, x = k the error left after attempt k's update).

    LEFT column, sharing the attempt axis: one panel per estimated dim (SIGNED error, symmetric
    ylim so the dashed zero line is the centre), then the combined L2 error in the estimator's
    own mm-equivalent metric (rotation scaled by scaling_constant_deg_to_mm). RIGHT column: the
    ICP residual per attempt on a LOG y axis -- EVERY guess's final residual as a faint column
    (res_all, one array per attempt: the population the aggregator votes over, so consensus
    spread and outlier guesses are visible) with the AGGREGATED residual bold on top (nan =
    estimation skipped) -- then
    residual vs the L2 error LEFT AFTER applying that attempt's correction -- EVERY guess as a
    faint point (l2_all: the error each guess's own correction WOULD have left, computable here
    because the truth is known) with the aggregated pick bold and labelled by attempt. The
    residual is only trustworthy if that cloud trends up-right. THIRD column (2+ estimated
    dims): PHASE plots, one per pairwise dim combination -- the trial's error TRAJECTORY in
    that error plane (square = injected, numbered dots = after each attempt, star = latest),
    axes symmetric so the ORIGIN (zero error in both dims) sits at the centre; convergence
    reads as the path spiralling into the crosshair. If live_path is
    given, the same figure is ALSO written there ATOMICALLY (temp file + os.replace), so one
    fixed file outside the experiment folder can stay open in an image viewer. BEST-EFFORT: a
    plotting problem (e.g. matplotlib missing on the robot box) is logged and skipped, never
    allowed to kill a hardware run."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        err6 = np.asarray(err6, dtype=float)
        x = np.arange(len(err6))
        # Combined L2 error in the SAME mm-equivalent metric the estimator matches in:
        # sqrt(|t|^2 + |s_rot * rot|^2), rotation folded in via scaling_constant_deg_to_mm.
        l2 = np.sqrt((err6[:, :3] ** 2).sum(axis=1) + ((s_rot * err6[:, 3:]) ** 2).sum(axis=1))
        res = np.maximum(np.asarray(residuals, dtype=float), 1e-6)   # nan stays nan (gaps)

        n_left = len(dims) + 1
        pairs = list(itertools.combinations(range(len(dims)), 2))
        ncols = (3 if pairs else 2) + (1 if land else 0)
        widths = ([1.25, 1.0] + ([0.95] if pairs else []) + ([1.15] if land else []))
        fig = plt.figure(figsize=(11.0 + (4.2 if pairs else 0.0) + (5.0 if land else 0.0),
                                  max(2.1 * n_left, 6.5)))
        gs = fig.add_gridspec(2 * n_left, ncols, width_ratios=widths)
        axes = []
        for i in range(n_left):
            axes.append(fig.add_subplot(gs[2 * i:2 * i + 2, 0],
                                        sharex=axes[0] if axes else None))
        for ax, dim in zip(axes, dims):
            j = _ERR.index(dim)
            unit = 'deg' if dim.endswith('_deg') else 'mm'
            ax.axhline(0.0, ls='--', lw=1.0, color='#888888', zorder=1)
            ax.plot(x, err6[:, j], 'o-', color='#4C72B0', zorder=2)
            lim = max(float(np.abs(err6[:, j]).max()), 1e-3) * 1.1
            ax.set_ylim(-lim, lim)                 # symmetric: the zero line is the centre
            ax.set_ylabel(f'{dim} error [{unit}]')
            ax.tick_params(labelbottom=False)

        ax = axes[-1]
        ax.plot(x, l2, 'o-', color='#DD8452', zorder=2)
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel(f'L2 error [mm-eq]\n(deg x {s_rot:g})')
        ax.set_xticks(x)
        ax.set_xlabel('attempt (0 = injected error, before any update)')

        # ICP residuals, one column per ATTEMPT (none for the injected point), log scale:
        # every guess faint, the aggregated residual bold on top.
        ax_r = fig.add_subplot(gs[:n_left, 1])
        labelled = False
        for k, rg in enumerate(res_all or []):
            rg = np.maximum(np.asarray(rg, dtype=float), 1e-6)
            if not len(rg):
                continue
            jit = (np.arange(len(rg)) / max(len(rg) - 1, 1) - 0.5) * 0.3   # deterministic spread
            ax_r.scatter(k + 1 + jit, rg, s=7, color='#55A868', alpha=0.25, lw=0, zorder=1,
                         label=None if labelled else 'all guesses')
            labelled = True
        ax_r.plot(np.arange(1, len(res) + 1), res, 'o-', color='#55A868', zorder=2,
                  label='aggregated')
        if labelled:
            ax_r.legend(fontsize=8, loc='best')
        ax_r.set_yscale('log')
        ax_r.set_xticks(np.arange(1, len(res) + 1))
        ax_r.set_xlabel('attempt')
        ax_r.set_ylabel('ICP residual [mm-eq]')

        # Residual vs the L2 error LEFT AFTER that attempt's update: every guess faint (its own
        # would-be outcome), the aggregated pick bold -- attempt k's aggregate pairs with l2[k].
        ax_s = fig.add_subplot(gs[n_left:, 1])
        for rg, lg in zip(res_all or [], l2_all or []):
            rg = np.maximum(np.asarray(rg, dtype=float), 1e-6)
            lg = np.asarray(lg, dtype=float)
            m2 = min(len(rg), len(lg))
            if m2:
                ax_s.scatter(rg[:m2], lg[:m2], s=7, color='#55A868', alpha=0.25, lw=0,
                             zorder=1)
        m = min(len(res), len(l2) - 1)
        fin = np.flatnonzero(np.isfinite(res[:m]))
        ax_s.scatter(res[fin], l2[fin + 1], color='#55A868', zorder=2)
        for k in fin:
            ax_s.annotate(str(k + 1), (res[k], l2[k + 1]), textcoords='offset points',
                          xytext=(4, 3), fontsize=8, color='#444444')
        ax_s.set_xscale('log')
        ax_s.set_ylim(bottom=0.0)
        ax_s.set_xlabel('ICP residual [mm-eq]')
        ax_s.set_ylabel('L2 error after update [mm-eq]')

        # PHASE plots: the error trajectory per pairwise dim combination, origin = zero error.
        bounds = np.linspace(0, 2 * n_left, len(pairs) + 1).astype(int) if pairs else []
        for pi, (a, b) in enumerate(pairs):
            axp = fig.add_subplot(gs[bounds[pi]:bounds[pi + 1], 2])
            ja, jb = _ERR.index(dims[a]), _ERR.index(dims[b])
            ea, eb = err6[:, ja], err6[:, jb]
            axp.axhline(0.0, ls=':', lw=0.8, color='#aaaaaa', zorder=1)
            axp.axvline(0.0, ls=':', lw=0.8, color='#aaaaaa', zorder=1)
            axp.plot(ea, eb, '-', color='#4C72B0', lw=1.0, zorder=2)
            axp.scatter(ea[1:], eb[1:], s=20, color='#4C72B0', zorder=3)
            axp.scatter([ea[0]], [eb[0]], s=40, marker='s', color='#DD8452', zorder=4,
                        label='injected')
            axp.scatter([ea[-1]], [eb[-1]], s=80, marker='*', color='#55A868', zorder=5,
                        label='latest')
            for k in range(1, len(ea)):
                axp.annotate(str(k), (ea[k], eb[k]), textcoords='offset points',
                             xytext=(4, 3), fontsize=7, color='#444444')
            la = max(float(np.abs(ea).max()), 1e-3) * 1.15
            lb = max(float(np.abs(eb).max()), 1e-3) * 1.15
            axp.set_xlim(-la, la)                  # symmetric: the origin is the centre
            axp.set_ylim(-lb, lb)
            axp.set_xlabel(f'{dims[a]} error '
                           f'[{"deg" if dims[a].endswith("_deg") else "mm"}]', fontsize=8)
            axp.set_ylabel(f'{dims[b]} error '
                           f'[{"deg" if dims[b].endswith("_deg") else "mm"}]', fontsize=8)
            axp.tick_params(labelsize=7)
            if pi == 0:
                axp.legend(fontsize=7, loc='best')

        # LANDSCAPE, in the ERROR FRAME: every candidate correction is drawn at the REMAINING
        # ERROR it would leave, so TRUTH IS THE ORIGIN and the axes mean the same thing on every
        # attempt and every trial (the correction frame shifts each time the belief updates).
        # Separates "the solver picked badly" -- truth sits in a deep basin it missed -- from
        # "the landscape's minimum is in the wrong place", where the DATA is at fault.
        if land:
            (axes_l, E, sigma, argmin, est_corr, errb_l, finals, idx_l, track_l, cov_l,
             pseat_l) = land
            col = ncols - 1
            axl = fig.add_subplot(gs[:, col])
            unit = [('deg' if d.endswith('_deg') else 'mm') for d in dims]
            mesh = np.meshgrid(*axes_l, indexing='ij')
            gth = np.stack([m.ravel() for m in mesh], axis=1)
            gerr = _as_error(errb_l, gth, idx_l)     # grid, as remaining error
            e_app = _as_error(errb_l, [est_corr[d] for d in dims], idx_l)[0]
            e_arg = _as_error(errb_l, [argmin[d] for d in dims], idx_l)[0]
            e_fin = (_as_error(errb_l, finals, idx_l)
                     if finals is not None and len(finals) else None)
            tr = np.asarray(track_l, dtype=float) if track_l is not None else None
            if len(dims) == 1:
                d0 = dims[0]
                o = np.argsort(gerr[:, 0])
                axl.plot(gerr[o, 0], E.ravel()[o], color='#4C72B0', lw=1.6, zorder=2)
                if e_fin is not None:
                    axl.plot(e_fin[:, 0], np.full(len(e_fin), float(np.min(E))), '|',
                             color='#ff7f0e', ms=9, alpha=0.8, zorder=3, label='ICP finals')
                axl.axvline(0.0, color='#55A868', ls='--', lw=2.0, zorder=5, label='truth')
                axl.axvline(e_app[0], color='#DD8452', lw=2.0, zorder=4,
                            label=f'applied ({e_app[0]:+.2f} left)')
                s = sigma.get(d0)
                if s is not None and np.isfinite(s):
                    axl.axvspan(e_app[0] - s, e_app[0] + s, color='#DD8452', alpha=0.18,
                                zorder=0, label=f'+/- 1 sigma ({s:.2f})')
                axl.plot([e_arg[0]], [float(np.min(E))], 'x', color='#C44E52', ms=9, mew=2,
                         zorder=6, label='grid argmin')
                if tr is not None and len(tr):
                    y = float(np.min(E)) + 0.03 * float(np.ptp(E))
                    axl.plot(tr[:, 0], np.full(len(tr), y), '.-', color='#4C72B0', lw=1.0,
                             ms=6, alpha=0.7, zorder=4, label='estimate trajectory')
                axl.set_xlabel(f'{d0} ERROR remaining [{unit[0]}]   (0 = truth)')
                axl.set_ylabel('ICP energy [mm-eq]')
            else:
                d0, d1 = dims[0], dims[1]
                sh = mesh[0].shape
                Eg = E if E.ndim == 2 else E.reshape(sh[0], sh[1], -1).min(2)
                im = axl.pcolormesh(gerr[:, 1].reshape(sh), gerr[:, 0].reshape(sh), Eg,
                                    shading='auto', cmap='viridis')
                fig.colorbar(im, ax=axl, label='ICP energy [mm-eq]')
                if e_fin is not None and e_fin.shape[1] >= 2:
                    axl.scatter(e_fin[:, 1], e_fin[:, 0], s=14, facecolors='none',
                                edgecolors='#ff7f0e', linewidths=0.7, alpha=0.75, zorder=3,
                                label='ICP finals')
                if tr is not None and len(tr) > 1:
                    axl.plot(tr[:, 1], tr[:, 0], '-', color='#7fb3ff', lw=1.4, zorder=4)
                    axl.scatter(tr[:-1, 1], tr[:-1, 0], s=22, color='#7fb3ff', zorder=5)
                    for i in range(len(tr)):
                        axl.annotate(str(i), (tr[i, 1], tr[i, 0]), textcoords='offset points',
                                     xytext=(4, 3), fontsize=7, color='#eaf2ff')
                # 1-sigma COVARIANCE ELLIPSE with its eigen-axes drawn: the marginal 2x2 block
                # of the full covariance, so with 3+ estimated dims this is the correct marginal
                # for the two shown. The axes make the stiff/sloppy split visible directly.
                if cov_l is not None and len(cov_l) >= 2:
                    sub = np.array([[cov_l[0, 0], cov_l[0, 1]], [cov_l[1, 0], cov_l[1, 1]]])
                    try:
                        wv, Vv = np.linalg.eigh(sub)
                        wv = np.maximum(wv, 0.0)
                        th = np.linspace(0, 2 * np.pi, 120)
                        pts = (Vv * np.sqrt(wv)) @ np.vstack([np.cos(th), np.sin(th)])
                        axl.plot(e_app[1] + pts[1], e_app[0] + pts[0], color='#DD8452',
                                 lw=1.8, zorder=6, label='1-sigma covariance')
                        for a2 in range(2):        # the eigen-axes (stiff / sloppy directions)
                            v = Vv[:, a2] * np.sqrt(wv[a2])
                            axl.plot([e_app[1] - v[1], e_app[1] + v[1]],
                                     [e_app[0] - v[0], e_app[0] + v[0]], color='#DD8452',
                                     lw=1.0, ls='--', alpha=0.85, zorder=6)
                        ratio = (np.sqrt(wv[1] / max(wv[0], 1e-12)) if wv[0] > 0 else np.inf)
                        axl.plot([], [], ' ', label=f'sloppy/stiff = {ratio:.1f}x')
                    except np.linalg.LinAlgError:
                        pass
                axl.plot([e_app[1]], [e_app[0]], 'o', ms=8, mfc='#DD8452', mec='white',
                         zorder=7, label='applied')
                axl.plot([e_arg[1]], [e_arg[0]], 'x', color='#C44E52', ms=10, mew=2, zorder=7,
                         label='grid argmin')
                axl.axhline(0.0, ls=':', lw=0.9, color='#ffffff', alpha=0.6, zorder=2)
                axl.axvline(0.0, ls=':', lw=0.9, color='#ffffff', alpha=0.6, zorder=2)
                axl.plot([0.0], [0.0], '*', ms=18, mfc='#55A868', mec='white', zorder=8,
                         label='truth (origin)')
                axl.set_xlabel(f'{d1} ERROR remaining [{unit[1]}]   (0 = truth)')
                axl.set_ylabel(f'{d0} ERROR remaining [{unit[0]}]   (0 = truth)')
            axl.legend(fontsize=7, loc='best')
            ttl = 'energy landscape in the ERROR frame (truth = origin)'
            if pseat_l is not None and np.isfinite(pseat_l):
                ttl += f'   |   P(seat) = {pseat_l:.0%}'
            axl.set_title(ttl, fontsize=10)

        fig.suptitle(f'trial {trial}: ground-truth belief error per attempt', y=0.995)
        if status:                                 # run progress: trials done + success rate
            fig.text(0.99, 0.965, status, ha='right', fontsize=9, color='#333333')
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        if live_path:
            # Temp-then-replace so a viewer polling the live file never reads a half-written PNG.
            tmp = live_path + '.tmp'
            fig.savefig(tmp, dpi=110, format='png')
            os.replace(tmp, live_path)
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
               'converged_frac': float(np.mean([bool(r['converged']) for r in rs])),
               'success_frac': float(np.mean([bool(r.get('success')) for r in rs]))}
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
    # The CHECKED estimator (estimation.check) is OBSERVATIONAL: it computes the rankavg2 +
    # Cauchy trust scores after every estimate, which land in trials.csv (trust_* columns) --
    # here, with the ground truth known, is exactly where those scores get VALIDATED against
    # the realized error. The correction is always applied, same as the plain estimator.
    estimator = CheckedManifoldEstimator(cfg.section('estimation'))

    # ---- SUCCESS BASIN: P(seat | offset) learned from the SAME manifold, plus ONE definition
    # of success (reach within seat_margin_mm of the manifold's deepest insertion) used both to
    # gate the insertion and to score it afterwards. ----
    sg = ev.get('seat_gate', {}) or {}
    basin, seat_gate, seat_temp = None, 1.0, 0.05
    if bool(sg.get('enabled', True)):
        try:
            basin = SuccessBasin(urconfig.resolve(cfg, cfg.get_path('estimation.manifold_csv')),
                                 estimator.estimate_dims,
                                 dict(sg, scaling_constant_deg_to_mm=estimator.s_rot))
            seat_gate = float(sg.get('p_seat_threshold', 0.95))
            seat_temp = float(sg.get('posterior_temp', 0.05))
            log.info('P(seat) gate at %.0f%%: once the posterior clears it the trial commits to '
                     'ONE insertion with NO trajectory noise.', 100 * seat_gate)
        except Exception as exc:                   # noqa: BLE001
            log.error('Success basin unavailable (%s) -- gating disabled.', exc)
            basin = None

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
    # combination at eval.grid_resolution and DERIVES the trial count; 'bounds' tests the +/-
    # bound extremes (count derived) -- one DOF at a time by default, or every corner of the
    # box at once with eval.bounds_simultaneous; 'random' draws uniformly.
    pert = ev.get('perturbation', {}) or {}
    lo = pert.get('lower', [-0.005, 0.0, -0.005, 0.0, -5.0, 0.0])
    hi = pert.get('upper', [0.005, 0.0, 0.005, 0.0, 5.0, 0.0])
    mode = str(ev.get('mode', 'random')).lower()
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
        log.error("eval.mode 'bounds' produced 0 trials -- every perturbation bound is zero.")
        return False
    num_trials = len(fixed) if fixed is not None else int(ev.get('num_trials', 50))
    max_attempts = int(ev.get('max_attempts', 5))
    seed = int(ev.get('random_seed', 0))
    rng = np.random.default_rng(seed if seed > 0 else None)
    tol = ev.get('success_tolerance', {}) or {}
    tol_pos_mm = float(tol.get('pos_mm', 2.0))
    tol_rot_deg = float(tol.get('rot_deg', 3.0))
    stop_conv = bool(ev.get('stop_when_converged', True))
    # PHYSICAL success: the TRUE connector pose wrt the TARGET within these per-DOF tolerances
    # [x, y, z (mm), roll, pitch, yaw (deg)] at the end of an insertion. Success TERMINATES
    # the trial (independent of the belief-convergence gate above).
    succ_tol = [float(v) for v in
                (ev.get('success_pose_tol') or [2.0, 1.0, 5.0, 5.0, 5.0, 1.0])]
    if len(succ_tol) != 6:
        log.error('eval.success_pose_tol must have 6 entries [x,y,z (mm), r,p,y (deg)].')
        return False
    decim = max(1, int(ev.get('log_decimation', 5)))
    save_obs = bool(ev.get('save_observations', True))
    log.info('%d trials x max %d attempts (%s perturbations, bounds lower=%s upper=%s).',
             num_trials, max_attempts, mode.upper(), list(lo), list(hi))

    # ONE error figure per trial (trial_TTT_errors.png), re-saved after every attempt -- cheap
    # (N files, not N x M), and watchable live during a run.
    save_plots = bool(ev.get('save_plots', True))
    # The dense energy landscape behind each estimate (an extra column on the trial figure and
    # therefore on the live one). Costs one grid x observations kNN sweep per attempt -- both are
    # capped in _landscape -- so it can be turned off on a slow box.
    plot_land = bool(ev.get('plot_landscape', True))
    # LIVE figure: ONE fixed path OUTSIDE the per-experiment folder, atomically overwritten with
    # the current trial's figure after every attempt -- keep it open in an image viewer across
    # trials and runs. true = data/experiments/estimator_eval_live.png; a string = explicit path.
    live = ev.get('live_plot', True)
    live_path = None
    if save_plots and live:
        live_path = live if isinstance(live, str) else os.path.join(
            cfg.get('data_dir', 'data'), 'experiments', 'estimator_eval_live.png')
        os.makedirs(os.path.dirname(live_path) or '.', exist_ok=True)
        log.info('Live figure: %s', live_path)

    # OPTIONAL trajectory noising: smoothed zero-mean offsets in the connector's OWN frame,
    # REDRAWN per attempt -- varied contact instead of the same nominal path every time. Its OWN
    # random stream: toggling noise must not disturb the injected-error draws, so noised and
    # nominal runs stay pairable trial-for-trial.
    tn = ev.get('trajectory_noise', {}) or {}
    tn_on = bool(tn.get('enabled', False))
    # Per-DIMENSION std [x, y, z (m), roll, pitch, yaw (deg)]; the legacy scalar keys
    # (translation_m / rotation_deg) broadcast to their three axes when 'std' is absent.
    tn_std = tn.get('std')
    if tn_std is None:
        tn_std = [float(tn.get('translation_m', 0.0005))] * 3 \
            + [float(tn.get('rotation_deg', 0.5))] * 3
    tn_std = [float(v) for v in tn_std]
    if len(tn_std) != 6:
        log.error('eval.trajectory_noise.std must have 6 entries [x,y,z (m), r,p,y (deg)].')
        return False
    tn_w = max(1, int(tn.get('smooth_window', 25)))
    # DECAYS + alternation: shrink the whole perturbation by noise_decay_attempt each attempt
    # ((1-f)^(k-1)), shed it linearly along the path by noise_decay_traj (1 -> 1-f at the last
    # waypoint), and optionally add a deterministic initial pitch offset whose SIGN alternates
    # per attempt (probe the valley from both sides).
    tn_da = float(tn.get('noise_decay_attempt', 0.0))
    tn_dt = float(tn.get('noise_decay_traj', 0.0))
    tn_alt = float(tn.get('alternate_pitch_deg', 0.0))
    noise_rng = np.random.default_rng(seed + 1 if seed > 0 else None)
    if tn_on:
        log.info('Trajectory noise ON: std [%.2f, %.2f, %.2f] mm / [%.2f, %.2f, %.2f] deg '
                 'per waypoint, smooth window %d.', tn_std[0] * 1000.0, tn_std[1] * 1000.0,
                 tn_std[2] * 1000.0, tn_std[3], tn_std[4], tn_std[5], tn_w)

    # ACCUMULATE observations across a trial's attempts (default ON): the part is FIXTURED, so
    # the rigid-belief-error assumption holds for the whole trial, and earlier attempts sample
    # DIFFERENT manifold regions -- extra constraint diversity exactly where one attempt is
    # degenerate (the z-pitch valley). After each correction every stored row is RE-PROJECTED
    # into the updated belief (_rebase_rows), and the estimator's recency weighting decays the
    # older attempts naturally (they sit earlier in the concatenated sequence).
    accumulate = bool(ev.get('accumulate_observations', True))

    # COMPLIANCE + guard + speeds: same shape as uncertain_sampling; the config mirrors the pick
    # app's assembly values so the estimator sees production-like observations.
    adm = AdmittanceController(robot.arm, cfg.section('compliance'))
    guard = ForceGuard(robot.arm, cfg.section('force_guard'))
    # OPTIONAL FINAL INSERTION: one extra guarded assemble per trial from the FINAL corrected
    # belief under a DIFFERENT stiffness (same compliance section otherwise) -- does the
    # corrected belief actually seat? Built here so a bad stiffness list fails pre-motion.
    fi = ev.get('final_insertion', {}) or {}
    fi_on = bool(fi.get('enabled', False))
    adm_final = None
    if fi_on:
        comp_final = dict(cfg.section('compliance'))
        if fi.get('stiffness') is not None:
            comp_final['stiffness'] = [float(v) for v in fi['stiffness']]
        adm_final = AdmittanceController(robot.arm, comp_final)
        log.info('Final insertion ON: stiffness %s.', comp_final.get('stiffness'))
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

    def run_insertion(adm_ctl, refs, T_bel):
        """One admittance-followed insertion along refs, collecting observations (same law and
        logging as cable_pick_estimate_assemble), the seated kinematic check, then the compliant
        UN-guarded retract along the believed part's own -X (a seated part is already over the
        guard limit; a guarded retract would block itself). Returns (obs, seated, lin, ang)."""
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
                log.info('Contact limit at waypoint %d/%d -- stopped advancing.',
                         i, len(refs) - 1)
                break
        adm_ctl.hold(last_ref, settle_s, guard, on_step=log_cb)

        # Kinematic check numbers (BELIEVED pose), for the record only.
        lin, ang = pose_error(robot.tool0() @ T_bel, T_base_tconn)
        # PHYSICAL seat pose: the TRUE connector wrt the TARGET connector at the end of the
        # hold (the part is fixtured, so T_true is exact) -- the per-DOF success measure.
        xyz, rpy = matrix_to_xyzrpy(inverse(T_base_tconn) @ robot.tool0() @ T_true)
        seat6 = list(xyz * 1000.0) + list(np.degrees(rpy))

        T_out = _retract_ref(last_ref, T_bel, retract_m)
        adm_ctl.ramp(last_ref, T_out, seg_time(last_ref, T_out, rv_mm_s, rw_deg_s), guard=None)
        adm_ctl.stop()
        return obs, seated, lin, ang, seat6

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

    rows, ok, durations, n_succ = [], True, [], 0
    try:
        for trial in range(1, num_trials + 1):
            trial_succ = False
            t_trial = time.time()
            # Corrupt the BELIEF only -- the part never moves in the fingers. The robot plans
            # from T_believed, so the true part physically rides the path offset by exactly the
            # injected error: the same situation as a bad grasp, but with the answer known.
            delta = fixed[trial - 1] if fixed is not None else traj.random_delta(lo, hi, rng)
            T_believed = T_true @ delta
            inj, inj_pos, inj_rot = _gt_error(T_true, T_believed)
            log.info('--- trial %d/%d --- injected belief error xyz=[%+6.2f, %+6.2f, %+6.2f] mm '
                     'rpy=[%+6.2f, %+6.2f, %+6.2f] deg', trial, num_trials, *inj)
            # The trial's error track for the co-plot: index 0 = the injected error, index k =
            # the error left after attempt k's update. trackr = the AGGREGATED ICP residual per
            # attempt (nan where estimation was skipped); trackg = every guess's final residual
            # per attempt (the population the aggregator votes over); trackl2 = the L2 error
            # each guess's OWN correction would have left (computable: the truth is known).
            # acc = this trial's accumulated observations, always in the CURRENT belief.
            track6, trackr, trackg, trackl2 = [inj], [], [], []
            acc = np.zeros((0, 12))

            abandoned = False
            gate_open = False
            for attempt in range(1, max_attempts + 1):
                errb, errb_pos, errb_rot = _gt_error(T_true, T_believed)
                if tn_on:
                    bias = ([0.0, 0.0, 0.0, 0.0,
                             tn_alt * (1.0 if attempt % 2 else -1.0), 0.0]
                            if tn_alt else None)
                    rows_t = traj.noised(dense, noise_rng, tn_std, tn_w, tn_dt,
                                         (1.0 - tn_da) ** (attempt - 1), bias)
                else:
                    rows_t = dense
                refs = [T_base_tconn @ row @ inverse(T_believed) for row in rows_t]

                # To the attempt's start -- stiff, free space (the retract/stand-off cleared it).
                q = robot.arm.ik(refs[0], seed_q)
                if q is None or not robot.arm.move_j(
                        q, label=f'trial {trial} attempt {attempt} start'):
                    log.warning('IK/approach failed; abandoning the rest of trial %d.', trial)
                    abandoned = True
                    break
                seed_q = q

                # ASSEMBLE under admittance (same law as the pick app), check, retract.
                obs, seated, lin, ang, seat6 = run_insertion(adm, refs, T_believed)
                succ = all(abs(v) <= t for v, t in zip(seat6, succ_tol))
                if succ and not trial_succ:
                    trial_succ = True
                    n_succ += 1

                if save_obs:
                    _save_observations(os.path.join(
                        out_dir, f'trial_{trial:03d}_attempt_{attempt:02d}_observations.csv'), obs)

                # ESTIMATE -- always, even on the last attempt: the estimate IS the thing under
                # test, so every attempt's observations get scored. With accumulation the input
                # is prior attempts (already re-projected into the current belief) + this one,
                # oldest first, so recency weighting decays the carried rows.
                obs_arr = np.asarray(obs, dtype=float).reshape(-1, 12)
                full = np.vstack([acc, obs_arr]) if accumulate else obs_arr
                if accumulate and len(acc):
                    log.info('Estimating on %d observations (%d carried from prior attempts).',
                             len(full), len(acc))
                vec6, w6 = estimator.prepare_observations(
                    full[:, :6], full[:, 6:9], full[:, 9:12])
                T_corr_mm, info = estimator.estimate(vec6, w6)
                land, p_seat = None, float('nan')
                row = {'trial': trial, 'attempt': attempt, 'n_observations': len(obs),
                       'seated': seated, 'check_pos_mm': lin * 1000.0,
                       'check_rot_deg': float(np.degrees(ang)), 'success': succ,
                       'err_before_pos_mm': errb_pos, 'err_before_rot_deg': errb_rot}
                row.update({f'seat_{s}': v for s, v in zip(_ERR, seat6)})
                row.update({f'inj_{s}': v for s, v in zip(_ERR, inj)})
                row.update({f'err_before_{s}': v for s, v in zip(_ERR, errb)})
                if T_corr_mm is None:
                    log.warning('Estimation skipped (%s) -- belief unchanged.', info)
                    row['estimate'] = f'skipped: {info}'
                    trackr.append(float('nan'))
                    trackg.append(np.zeros(0))
                    trackl2.append(np.zeros(0))
                    if accumulate:
                        acc = full                 # belief unchanged -- rows stay valid as-is
                else:
                    T_believed = T_believed @ _corr_to_m(T_corr_mm)   # believed @ corr ~= true
                    if accumulate:
                        # keep every stored row expressed in the belief JUST updated
                        acc = _rebase_rows(full, T_corr_mm) if len(full) else full
                    row['estimate'] = 'ok'
                    row.update({f'corr_{k}': v for k, v in info['theta_corr'].items()})
                    row.update({'icp_inliers': info['inliers'],
                                'icp_residual': info['final_residual']})
                    # TRUST readout (observational): both scores + the raw signals, so the
                    # run's ground truth can score the uncertainty estimates themselves.
                    chk = info.get('check')
                    if chk:
                        row['trust_rankavg2'] = chk['rankavg2']
                        row['trust_cauchy'] = chk['cauchy']
                        row.update({f'trust_{k}': v for k, v in chk['signals'].items()
                                    if k in _TRUST_SIGNALS})
                    trackr.append(float(info['final_residual']))
                    trackg.append(np.asarray(info['res_hist'], dtype=float)[:, -1])
                    # Per-guess would-be OUTCOME: the ground-truth L2 error left if guess g's
                    # correction had been applied to the PRE-update belief (errb, mm/deg).
                    th6 = np.zeros((len(info['theta_hist']), 6))
                    th6[:, estimator.idx] = info['theta_hist'][:, -1]
                    rem6 = vec6_from_mats(
                        mats_from_vec6(np.asarray(errb)) @ mats_from_vec6(th6))
                    trackl2.append(np.sqrt(
                        (rem6[:, :3] ** 2).sum(axis=1)
                        + ((estimator.s_rot * rem6[:, 3:]) ** 2).sum(axis=1)))
                    # The landscape the correction was picked from, with the TRUE correction
                    # (believed @ C = true, so C = inverse(err_before)) for reference.
                    if (save_plots and plot_land) or basin is not None:
                        try:
                            ax_l, E_l, sig_l, amin_l, cov_l = _landscape(estimator, vec6, w6)
                            # P(SEAT) of the POSTERIOR, not of the point estimate: the robot
                            # never knows its remaining error, so the decision quantity is
                            # E_p[P(seat)] over the landscape's Gibbs posterior. Hypothesis
                            # "theta was the right correction" leaves remaining error
                            # inverse(theta) (.) theta_applied once we apply theta_applied.
                            if basin is not None:
                                mesh = np.meshgrid(*ax_l, indexing='ij')
                                gth = np.stack([m.ravel() for m in mesh], axis=1)
                                g6 = np.zeros((len(gth), 6))
                                g6[:, estimator.idx] = gth
                                a6 = np.zeros(6)
                                a6[estimator.idx] = [info['theta_corr'][d]
                                                     for d in estimator.estimate_dims]
                                rem = vec6_from_mats(
                                    np.linalg.inv(mats_from_vec6(g6))
                                    @ mats_from_vec6(a6)[None, :, :])
                                Ef = np.asarray(E_l, dtype=float).ravel()
                                wgt = np.exp(-(Ef - Ef.min())
                                             / max(seat_temp * float(Ef.min()), 1e-12))
                                p_seat = basin.p_seat_posterior(
                                    basin.offset_of_error(rem), wgt)
                                row['p_seat'] = p_seat
                            trk = np.asarray(track6, dtype=float)[:, estimator.idx]
                            land = (ax_l, E_l, sig_l, amin_l, dict(info['theta_corr']),
                                    np.asarray(errb, dtype=float),
                                    np.asarray(info['theta_hist'], dtype=float)[:, -1, :],
                                    list(estimator.idx), trk, cov_l, p_seat)
                            row.update({f'sigma_{d}': v for d, v in sig_l.items()})
                            row.update({f'cov_{a}{b}': float(cov_l[a, b])
                                        for a in range(len(cov_l))
                                        for b in range(a, len(cov_l))})
                        except Exception as exc:   # noqa: BLE001 -- diagnostics never fatal
                            log.warning('landscape/P(seat) skipped (%s)', exc)
                erra, erra_pos, erra_rot = _gt_error(T_true, T_believed)
                track6.append(erra)
                if save_plots:                     # re-saved after EVERY attempt of this trial
                    status = (f'trial {trial}/{num_trials}  |  successes {n_succ}/{trial} '
                              f'({n_succ / trial:.0%})')
                    _plot_trial_errors(os.path.join(out_dir, f'trial_{trial:03d}_errors.png'),
                                       trial, estimator.estimate_dims, track6, trackr,
                                       estimator.s_rot, live_path, trackg, trackl2, status,
                                       land)
                row.update({f'err_after_{s}': v for s, v in zip(_ERR, erra)})
                row.update({'err_after_pos_mm': erra_pos, 'err_after_rot_deg': erra_rot})
                row['converged'] = bool(erra_pos <= tol_pos_mm and erra_rot <= tol_rot_deg)
                log.info('trial %d attempt %d: gt error %.2f mm / %.2f deg -> %.2f mm / %.2f deg'
                         '%s%s%s', trial, attempt, errb_pos, errb_rot, erra_pos, erra_rot,
                         '' if not np.isfinite(p_seat) else f'  P(seat) {p_seat:.0%}',
                         '  (CONVERGED)' if row['converged'] else '',
                         '  (SUCCESS -- seated within tolerance)' if succ else '')
                rows.append(row)
                writer.writerow(row)
                fout.flush()                       # a 50-trial run must survive an abort mid-way
                os.fsync(fout.fileno())
                # P(SEAT) GATE: stop probing the moment the belief is good enough to commit --
                # the final insertion then runs with NO trajectory noise (see below).
                if basin is not None and np.isfinite(p_seat) and p_seat >= seat_gate:
                    log.info('P(seat) %.0f%% >= %.0f%% -- committing to the insertion.',
                             p_seat, 100 * seat_gate)
                    gate_open = True
                    break
                if succ or (row['converged'] and stop_conv):
                    break

            # FINAL INSERTION -- the commit. Runs when the P(seat) gate opened (or always, if
            # eval.final_insertion.enabled and the attempts ran out). ALWAYS on the NOMINAL
            # trajectory with ZERO noise: the jitter exists to gather varied contact while
            # probing, and has no place in the attempt that is meant to seat. SUCCESS is decided
            # by the SAME rule that labels the basin -- the true depth reached vs the manifold's
            # deepest insertion less seat_margin_mm -- so the gate and the verdict agree.
            if (fi_on or gate_open) and not abandoned:
                refs = [T_base_tconn @ row @ inverse(T_believed) for row in dense]
                q = robot.arm.ik(refs[0], seed_q)
                if q is None or not robot.arm.move_j(q, label=f'trial {trial} final insertion'):
                    log.warning('IK/approach failed for the final insertion of trial %d.', trial)
                else:
                    seed_q = q
                    obs, seated, lin, ang, seat6 = run_insertion(adm_final, refs, T_believed)
                    if save_obs:
                        _save_observations(os.path.join(
                            out_dir, f'trial_{trial:03d}_final_insertion_observations.csv'), obs)
                    errf, errf_pos, errf_rot = _gt_error(T_true, T_believed)
                    # depth actually reached, in the TRUE frame (the basin's own coordinate)
                    depth = float('nan')
                    seated_basin = None
                    if len(obs):
                        oa = np.asarray(obs, dtype=float).reshape(-1, 12)
                        rel = vec6_from_mats(mats_from_vec6(oa[:, :6])
                                             @ np.linalg.inv(mats_from_vec6(np.asarray(errf))))
                        depth = float(rel[:, 0].max())
                        if basin is not None:
                            seated_basin = basin.is_seated(depth)
                    frow = {'trial': trial, 'attempt': 'final_insertion',
                            'n_observations': len(obs), 'seated': seated,
                            'check_pos_mm': lin * 1000.0,
                            'check_rot_deg': float(np.degrees(ang)),
                            'success': (seated_basin if seated_basin is not None
                                        else all(abs(v) <= t for v, t in zip(seat6, succ_tol))),
                            'estimate': 'none (final insertion)', 'gate_opened': gate_open,
                            'insert_depth_mm': depth, 'seat_depth_mm':
                                (basin.seat_depth if basin is not None else ''),
                            'seated_basin': ('' if seated_basin is None else seated_basin),
                            'err_before_pos_mm': errf_pos, 'err_before_rot_deg': errf_rot}
                    if basin is not None:
                        log.info('FINAL INSERTION: depth %+.2f mm vs seat threshold %+.2f mm '
                                 '-> %s', depth, basin.seat_depth,
                                 'SEATED' if seated_basin else 'NOT seated')
                    frow.update({f'seat_{s}': v for s, v in zip(_ERR, seat6)})
                    frow.update({f'err_before_{s}': v for s, v in zip(_ERR, errf)})
                    writer.writerow(frow)
                    fout.flush()
                    os.fsync(fout.fileno())
                    log.info('trial %d final insertion: seated=%s, check %.2f mm / %.2f deg',
                             trial, seated, lin * 1000.0, float(np.degrees(ang)))

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
