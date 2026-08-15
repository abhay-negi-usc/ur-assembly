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
        final      (eval.final_insertion, default ON): ONE more guarded insertion from the
                   FINAL corrected belief, ZERO trajectory noise, optionally a different
                   stiffness -- seats-or-not, no estimation. Every collection mode ends here.
        disassemble: back at the stand-off (free space) before the next trial

Collection modes (eval.collection.mode): 'attempts' = the loop above; 'offset_sweep' = each
attempt commands a LIST of deliberate offsets (default pitch -4..+4 deg step 2), pools all
their observations, then estimates ONCE -- trajectory diversity by design, with each pass
feeding its own stop depth to the stop-signature fusion; 'peck' = a force stop only backs the
part off a few mm before advancing again (end of trajectory or a time budget ends the attempt),
so one attempt logs a SEQUENCE of contact events, each a stop signature.

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
from ..skills.manifold import (DIMS, FORCE_COLS, POSE_COLS, TORQUE_COLS,
                               mats_from_vec6, scaled12, vec6_from_mats)
from ..skills.mixture import from_energy
from ..skills import manifold_debug
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
            + [f'sigma_within_{d}' for d in dims]
            + [f'cov_{a}{b}' for a in range(len(dims)) for b in range(a, len(dims))]
            # MIXTURE + SUPPORT. `_land` columns come from the dense landscape, the bare ones
            # from the ICP finals themselves -- they answer the same question from the two
            # different things the solver produces, and disagreement between them is a finding.
            + ['n_mixture_modes', 'ambiguity', 'between_frac', 'separation', 'seeded_guesses']
            + ['n_modes_land', 'ambiguity_land', 'between_frac_land', 'separation_land',
               'support_ratio']
            + ['ranked_mode', 'p_seat_unranked']   # seat_gate.mode_ranking diagnostics
            + ['commit']                           # estimation.commit actually used this attempt
            + ['p_seat', 'gate_opened', 'insert_depth_mm', 'seat_depth_mm', 'seated_basin']
            + ['icp_inliers', 'icp_residual', 'estimate']
            + ['trust_rankavg2', 'trust_cauchy'] + [f'trust_{s}' for s in _TRUST_SIGNALS]
            + [f'err_after_{s}' for s in _ERR] + ['err_after_pos_mm', 'err_after_rot_deg']
            + ['converged', 'diverged'])


def _landscape(estimator, vec6, w6, max_pts=2600, row_cap=120, raw=None):
    """The ICP energy over a DENSE GRID of candidate corrections, in the estimator's own metric.

    The multi-start solver never builds this -- it only samples it from random starts -- so the
    landscape is what shows whether the applied correction sits in the true basin, whether the
    basin is flat (uncertain) and whether a rival mode is deeper. Returns
    (axes, E, sigma, argmin, cov, mix, support) with E shaped like the mesh, sigma per dim in
    that dim's own units (sqrt(E_min / curvature), the 2026-08 campaign's uncertainty winner),
    argmin the grid's best correction, `mix` the MIXTURE over the landscape's modes (skills/
    mixture.py -- the honest uncertainty when rivals exist) and `support` the k-th-neighbour
    distance at the argmin over the manifold's own median, so a correction picked from the EDGE
    of the map is visible as such. Cost is bounded by subsampling grid and rows."""
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
    if raw is None:                                # candidate-aware wrench (see wrench6_at)
        raw = getattr(estimator, 'last_raw', None)
        if raw is not None and len(raw[0]) != len(v6):
            raw = None                             # stale stash: skip re-basing, never guess
    sel = None
    if len(v6) > row_cap:                          # bound the cost; the shape is unaffected
        sel = np.linspace(0, len(v6) - 1, row_cap).astype(int)
        v6, w = v6[sel], w[sel]
    if raw is not None and sel is not None:
        raw = (raw[0][sel], raw[1][sel])           # keep the raw wrench aligned with the rows
    # CHUNKED over candidates: the neighbour gather is (candidates x rows, k, 12) float64, which
    # at k=64 runs to gigabytes in ONE allocation -- a diagnostic must not be able to
    # MemoryError a hardware run.
    Ym = mats_from_vec6(v6)
    Ef, Sup = np.empty(len(G)), np.empty(len(G))
    sk = getattr(estimator, 'support_k', max(estimator.interp_neighbors, 2))
    per = max(int(256e6 // max(len(v6) * sk * 12 * 8, 1)), 1)
    for lo in range(0, len(G), per):
        hi = min(lo + per, len(G))
        C = np.einsum('nij,kjl->knil', Ym, mats_from_vec6(G[lo:hi]))
        if getattr(estimator, 'wrench_follows_correction', False) and raw is not None:
            # the wrench follows each candidate into the frame that candidate claims
            wg = np.concatenate([estimator.wrench6_at(raw[0], raw[1], g) for g in G[lo:hi]],
                                axis=0)
            pts = scaled12(vec6_from_mats(C).reshape(-1, 6), wg, estimator.s_rot,
                           getattr(estimator, 'dim_w', None))
        else:
            pts = scaled12(vec6_from_mats(C), w, estimator.s_rot,
                           getattr(estimator, 'dim_w', None)).reshape(-1, 12)
        dist, nn = estimator.tree.query(pts, k=sk, workers=-1)
        dist = dist[:, None] if dist.ndim == 1 else dist
        # ONE implementation of the blend (skills/manifold.blend_residual) so the landscape and
        # the solver cannot drift apart -- per-block bandwidths included.
        kn = estimator.interp_neighbors
        d1 = (estimator.blend_residual(pts, dist[:, :kn], nn[:, :kn]) if kn > 1
              else dist[:, 0])
        Ef[lo:hi] = d1.reshape(hi - lo, len(v6)).mean(axis=1)
        # support as a DEPTH-NORMALISED ratio per row (a global reference confounds support
        # with insertion depth -- the map is dense shallow, sparse deep; see support_ref_at)
        sup_row = dist[:, -1] / estimator.support_ref_at(pts[:, 0])
        Sup[lo:hi] = sup_row.reshape(hi - lo, len(v6)).mean(axis=1)
    E = Ef.reshape(mesh[0].shape)

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
    # The MIXTURE over the landscape's modes, and the SUPPORT at the correction we picked. The
    # curvature sigma above is the width of ONE basin; when rivals exist that is not the whole
    # uncertainty, and when the argmin sits off the edge of the map it is not even the right
    # basin. Both are cheap here -- the energy is already computed.
    gth = np.stack([m.ravel() for m in mesh], axis=1)
    mix = from_energy(gth, E.ravel(), E.shape, temp=0.05, dims=list(dims))
    support = float(Sup[int(np.argmin(E))])       # already a depth-normalised ratio
    # ADD the between-mode spread to the curvature covariance rather than replacing it: the
    # curvature width is the calibrated one, and the mixture's contribution is the ambiguity the
    # curvature cannot see. Unimodal landscapes are therefore unchanged.
    cov = cov + mix.between
    for a, d in enumerate(dims):
        sigma[d] = float(min(np.sqrt(max(cov[a, a], 0.0)), axes[a][-1]))
    return axes, E, sigma, argmin, cov, mix, support


def _basin_contour(ax, basin, dims, ja, jb, xlim, ylim, levels, rest=None, n=56):
    """Overlay the SUCCESS BASIN on axes whose coordinates are ERROR in dims[ja], dims[jb].

    The basin is indexed by the PHYSICAL offset the part rides at, which is inverse(error) -- so
    the grid is built in error coordinates and inverted before the lookup, otherwise the contour
    comes out mirrored. With more than two estimated dims this is a CONDITIONAL slice: the other
    dims are held at `rest` (the current error), which is what the operator actually cares about.
    Returns the contour set, or None if anything is unavailable."""
    if basin is None:
        return None
    try:
        gx = np.linspace(xlim[0], xlim[1], n)
        gy = np.linspace(ylim[0], ylim[1], n)
        GX, GY = np.meshgrid(gx, gy, indexing='xy')
        err6 = np.zeros((GX.size, 6))
        if rest is not None:
            err6[:, :] = np.asarray(rest, dtype=float)
        err6[:, jb] = GX.ravel()                   # x axis = dims[b]
        err6[:, ja] = GY.ravel()                   # y axis = dims[a]
        p = basin.p_seat(basin.offset_of_error(err6)).reshape(GX.shape)
        cs = ax.contour(gx, gy, p, levels=sorted(levels), colors=['#55A868'],
                        linewidths=[1.1, 1.6][:len(levels)], linestyles=['dotted', 'solid'],
                        alpha=0.9, zorder=2)
        ax.clabel(cs, fmt=lambda v: f'P(seat) {v:.0%}', fontsize=6, inline=True)
        return cs
    except Exception:                              # noqa: BLE001 -- overlay is never essential
        return None


def _argmin_estimate(estimator, vec6, w6):
    """The estimate WITHOUT the multi-start ICP: build the dense landscape and take its
    minimum (estimation.commit: argmin).

    The ICP is the expensive half of an attempt -- num_initial_guesses starts x
    icp_iterations, seconds to tens of seconds -- and in argmin mode its answer is discarded,
    so it is skipped outright rather than computed and thrown away. Everything the caller
    needs still comes from the landscape, which was already being built for the plots and
    P(seat): the correction, the curvature+mixture covariance, the mode structure.

    Returns (T_corr_mm, info, land_pack), or (None, reason, None) when there is too little
    evidence. `info` mirrors the solver's dict except for the ICP-only keys (theta_hist,
    res_hist, check) which are absent -- callers must treat them as optional."""
    if len(vec6) < estimator.min_observations:
        return None, f'too few observations ({len(vec6)} < {estimator.min_observations})', None
    pack = _landscape(estimator, vec6, w6)
    axes, E, sigma, argmin, cov, mix, support = pack
    theta = [float(argmin[d]) for d in estimator.estimate_dims]
    c6 = np.zeros(6)
    c6[estimator.idx] = theta
    Emin = float(np.min(E))
    info = {
        'theta_corr': {d: float(v) for d, v in zip(estimator.estimate_dims, theta)},
        'final_residual': Emin,
        'n_observations': int(len(vec6)),
        'sigma': dict(sigma),
        'covariance': cov,
        'mixture': mix,
        'n_mixture_modes': mix.n_modes,
        'ambiguity': mix.ambiguity,
        'between_frac': mix.between_frac,
        'separation': mix.separation,
        'support_ratio': support,
        # 'inliers' has no consensus meaning here; report the share of the grid within 5% of
        # the minimum, the same flatness reading the grid solver logs.
        'inliers': int(np.sum(np.asarray(E) <= Emin * 1.05)),
        'guesses': int(np.asarray(E).size),
        'seeded_guesses': 0,
        'aggregator': 'argmin',
    }
    return mats_from_vec6(c6), info, pack


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


def _p_seat_land(basin, axes, E, idx, theta_vals, temp):
    """E_p[P(seat)] over the dense landscape's Gibbs posterior if `theta_vals` is applied.

    The robot never knows its remaining error, so the decision quantity is P(seat) AVERAGED
    over the posterior, not evaluated at the point estimate. Hypothesis 'g was the right
    correction' leaves remaining error inverse(g) (.) theta once theta is applied; the basin is
    indexed by the PHYSICAL offset of that remaining error (SuccessBasin.offset_of_error).
    Temperature is RELATIVE to the minimum energy, like everywhere else in this stack."""
    mesh = np.meshgrid(*axes, indexing='ij')
    gth = np.stack([m.ravel() for m in mesh], axis=1)
    g6 = np.zeros((len(gth), 6))
    g6[:, idx] = gth
    a6 = np.zeros(6)
    a6[idx] = np.asarray(theta_vals, dtype=float)
    rem = vec6_from_mats(np.linalg.inv(mats_from_vec6(g6)) @ mats_from_vec6(a6)[None, :, :])
    Ef = np.asarray(E, dtype=float).ravel()
    w = np.exp(-(Ef - Ef.min()) / max(temp * float(Ef.min()), 1e-12))
    return basin.p_seat_posterior(basin.offset_of_error(rem), w)


def _plot_trial_errors(path, trial, dims, err6, residuals, s_rot, live_path=None,
                       res_all=None, l2_all=None, status=None, land=None, pseat=None,
                       seat_gate=None, basin=None, icp=None, idx_e=None):
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
        ax.plot(x, l2, 'o-', color='#DD8452', zorder=2, label='L2 error')
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel(f'L2 error [mm-eq]\n(deg x {s_rot:g})')
        ax.set_xticks(x)
        ax.set_xlabel('attempt (0 = injected error, before any update)')
        # P(SEAT) per attempt on a twin axis -- the DECISION quantity next to the error it is
        # supposed to track, with the gate that ends the trial. Watching these two together is
        # how you see whether the gate is honest: P(seat) should rise as L2 error falls.
        ps = np.asarray(pseat, dtype=float) if pseat is not None else np.zeros(0)
        if len(ps) and np.isfinite(ps).any():
            axp = ax.twinx()
            axp.plot(np.arange(1, len(ps) + 1), ps, 's-', color='#4C72B0', lw=1.6, ms=5,
                     zorder=3, label='P(seat)')
            if seat_gate is not None:
                axp.axhline(seat_gate, ls='--', lw=1.2, color='#55A868', zorder=1)
                axp.annotate(f'gate {seat_gate:.0%}', (0.02, seat_gate), xycoords=('axes fraction',
                                                                                  'data'),
                             textcoords='offset points', xytext=(0, 3), fontsize=7,
                             color='#55A868')
            axp.set_ylim(0.0, 1.02)
            axp.set_ylabel('P(seat)', color='#4C72B0', fontsize=9)
            axp.tick_params(axis='y', labelcolor='#4C72B0', labelsize=8)
            last = ps[np.isfinite(ps)][-1] if np.isfinite(ps).any() else float('nan')
            axp.set_title(f'latest P(seat) = {last:.0%}', fontsize=9, loc='right',
                          color='#4C72B0')

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
            # THE MOST RECENT ICP RUN, converging: one thin line per multi-start guess, drawn in
            # the same error coordinates. Where they end shows what the aggregator averaged over;
            # whether they funnel into one place or several shows if the landscape is multi-modal.
            if icp is not None:
                th, errb_i = icp
                th = np.asarray(th, dtype=float)
                if th.ndim == 3 and th.shape[1] > 1:
                    step = max(1, len(th) // 40)   # keep the panel readable
                    for g in range(0, len(th), step):
                        pe = _as_error(errb_i, th[g], idx_e)
                        axp.plot(pe[:, b], pe[:, a], '-', color='#ff7f0e', lw=0.5, alpha=0.35,
                                 zorder=2)
                    fin = _as_error(errb_i, th[::step, -1, :], idx_e)
                    axp.scatter(fin[:, b], fin[:, a], s=8, facecolors='none',
                                edgecolors='#ff7f0e', linewidths=0.6, alpha=0.8, zorder=3,
                                label='ICP finals')
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
            # SUCCESS BASIN outline in the same error coordinates: the trajectory is trying to
            # get INSIDE this contour, not to reach the origin -- z and pitch error do not need
            # to be individually small, their COMBINATION needs to land where the part seats.
            _basin_contour(axp, basin, dims, ja, jb, (-la, la), (-lb, lb),
                           [0.5, seat_gate if seat_gate is not None else 0.9],
                           rest=err6[-1])
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
             pseat_l, mix_l) = land
            # The MIXTURE's mode centres, in the same error frame. Drawing them is the whole
            # point of the mixture: a big sigma with the modes far apart is AMBIGUITY (probe
            # differently), a big sigma with one mode is IMPRECISION (probe more).
            e_modes = (_as_error(errb_l, [c.mean for c in mix_l.components], idx_l)
                       if mix_l is not None else None)
            w_modes = [c.weight for c in mix_l.components] if mix_l is not None else []
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
                if e_modes is not None and len(e_modes) > 1:
                    for m, (em, wm) in enumerate(zip(e_modes, w_modes)):
                        axl.axvline(em[0], color='#9467bd', lw=1.0 + 2.0 * wm, alpha=0.75,
                                    zorder=3, label='mixture modes' if m == 0 else None)
                        axl.annotate(f'{wm:.0%}', (em[0], float(np.max(E))), fontsize=7,
                                     color='#9467bd', ha='center', va='top')
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
                    except np.linalg.LinAlgError:
                        pass
                # RIVAL MODES: each one's centre, sized by its posterior mass, with a thin
                # within-mode ellipse. Two well-separated modes mean the spread is a genuine
                # either/or -- averaging them lands between two answers, both wrong.
                if e_modes is not None and len(e_modes) > 1 and e_modes.shape[1] >= 2:
                    for m, (em, wm) in enumerate(zip(e_modes, w_modes)):
                        axl.plot([em[1]], [em[0]], 'D', ms=4 + 8 * wm, mfc='none',
                                 mec='#e377c2', mew=1.6, zorder=7,
                                 label='mixture modes' if m == 0 else None)
                        axl.annotate(f'{wm:.0%}', (em[1], em[0]), textcoords='offset points',
                                     xytext=(6, -9), fontsize=7, color='#e377c2')
                        cm = mix_l.components[m].cov
                        try:
                            wv, Vv = np.linalg.eigh(cm[:2, :2])
                            th = np.linspace(0, 2 * np.pi, 90)
                            pp = (Vv * np.sqrt(np.maximum(wv, 0.0))) @ np.vstack(
                                [np.cos(th), np.sin(th)])
                            axl.plot(em[1] + pp[1], em[0] + pp[0], color='#e377c2', lw=0.9,
                                     alpha=0.7, zorder=6)
                        except np.linalg.LinAlgError:
                            pass
                axl.plot([e_app[1]], [e_app[0]], 'o', ms=8, mfc='#DD8452', mec='white',
                         zorder=7, label='applied')
                axl.plot([e_arg[1]], [e_arg[0]], 'x', color='#C44E52', ms=10, mew=2, zorder=7,
                         label='grid argmin')
                axl.axhline(0.0, ls=':', lw=0.9, color='#ffffff', alpha=0.6, zorder=2)
                axl.axvline(0.0, ls=':', lw=0.9, color='#ffffff', alpha=0.6, zorder=2)
                # the basin, in the SAME error coordinates: is the estimate (and its ellipse)
                # inside the region that actually seats?
                _basin_contour(axl, basin, dims, _ERR.index(d0), _ERR.index(d1),
                               axl.get_xlim(), axl.get_ylim(),
                               [0.5, seat_gate if seat_gate is not None else 0.9],
                               rest=np.asarray(err6, dtype=float)[-1])
                axl.plot([0.0], [0.0], '*', ms=18, mfc='#55A868', mec='white', zorder=8,
                         label='truth (origin)')
                axl.set_xlabel(f'{d1} ERROR remaining [{unit[1]}]   (0 = truth)')
                axl.set_ylabel(f'{d0} ERROR remaining [{unit[0]}]   (0 = truth)')
            axl.legend(fontsize=7, loc='best')
            ttl = 'energy landscape in the ERROR frame (truth = origin)'
            if pseat_l is not None and np.isfinite(pseat_l):
                ttl += f'   |   P(seat) = {pseat_l:.0%}'
            if mix_l is not None and mix_l.n_modes > 1:
                ttl += (f'\n{mix_l.n_modes} modes: {mix_l.ambiguity:.0%} of the mass off the '
                        f'leader, separation {mix_l.separation:.1f}, '
                        f'{mix_l.between_frac:.0%} of the spread is ambiguity')
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

    # WHICH NUMBER GETS COMMITTED (estimation.commit):
    #   'aggregator'  the multi-start solver's vote (ransac / softmax) -- the production path;
    #   'argmin'      the single deepest cell of the dense energy landscape, ignoring the ICP
    #                 finals entirely. The landscape is built anyway for the plots and P(seat),
    #                 so this costs nothing extra -- it is the hardware version of the offline
    #                 argmin study (analysis/landscape_sweep), where it measured 3.3-9.6 mm
    #                 |z'| at ~50% win vs ~2.0 mm / 84% for the aggregator.
    # MATCH DIAGNOSTICS (eval.debug_match): per-attempt figures showing where the observations
    # land in the map under the COMMITTED correction versus the TRUE one, the residual split by
    # channel, which map rows the data actually resembles, and each channel's own energy
    # minimum. This is the tool for 'the argmin is not the truth' -- it separates 'the truth
    # does not fit' from 'a rival fits better' from 'one channel decides'. Only meaningful in
    # this app, where the truth is known. Costs a landscape per channel, so it is opt-in and
    # can be thinned with every_n.
    dbg = cfg.get_path('eval.debug_match') or {}
    dbg_on = bool(dbg.get('enabled', False))
    dbg_every = max(int(dbg.get('every_n', 1)), 1)
    dbg_rows = int(dbg.get('max_rows', 250))
    dbg_grid = int(dbg.get('grid_points', 41))
    # LIVE mirrors for the two match figures, alongside estimator_eval_live.png: one fixed
    # path each, atomically replaced after every diagnosed attempt. true = the experiments
    # directory; a string = an explicit directory; false disables.
    dbg_live = dbg.get('live', True)
    dbg_live = (os.path.join(cfg.get('data_dir', 'data'), 'experiments')
                if dbg_live is True else (dbg_live or None))
    if dbg_on:
        log.info('Match diagnostics ON: a debug figure every %d attempt(s) -> '
                 'trial_XXX_attempt_YY_match.png (+ _dof.png)%s', dbg_every,
                 f'; live mirrors in {dbg_live}' if dbg_live else '')
    commit = str(cfg.get_path('estimation.commit', 'aggregator')).strip().lower()
    if commit not in ('aggregator', 'argmin'):
        log.error("estimation.commit %r must be 'aggregator' or 'argmin'.", commit)
        return False                               # bad values fail HERE, pre-motion
    if commit == 'argmin':
        log.warning('estimation.commit: ARGMIN -- committing the landscape minimum, NOT the '
                    'aggregator. Measured well below the aggregator offline; this is a '
                    'diagnostic mode.')

    # ---- SUCCESS BASIN: P(seat | offset) learned from the SAME manifold, plus ONE definition
    # of success (reach within seat_margin_mm of the manifold's deepest insertion) used both to
    # gate the insertion and to score it afterwards. ----
    sg = ev.get('seat_gate', {}) or {}
    basin, seat_gate, seat_temp = None, 1.0, 0.05
    # MODE RANKING at commitment: 'energy' = apply the aggregator's estimate untouched (the
    # original); 'p_seat' = candidates are that estimate PLUS each final-mixture mode centre,
    # and the E_p[P(seat)]-argmax over the landscape posterior is committed instead
    # (motivation: the truth sat in a NON-dominant mode in 54% / 70% of the 2026-08-12
    # validation cases -- depth misranks rivals; P(seat) is what the gate acts on anyway).
    if cfg.get_path('estimation.stop_fusion.enabled'):
        log.warning('estimation.stop_fusion was REMOVED (2026-08-13). Ignored.')
    mode_rank = str(sg.get('mode_ranking', 'energy')).strip().lower()
    if mode_rank not in ('energy', 'p_seat'):
        log.error("seat_gate.mode_ranking %r must be 'energy' or 'p_seat'.", mode_rank)
        return False                               # bad values fail HERE, pre-motion
    if bool(sg.get('enabled', True)):
        try:
            basin = SuccessBasin(urconfig.resolve(cfg, cfg.get_path('estimation.manifold_csv')),
                                 estimator.estimate_dims,
                                 dict(sg, scaling_constant_deg_to_mm=estimator.s_rot))
            seat_gate = float(sg.get('p_seat_threshold', 0.95))
            seat_temp = float(sg.get('posterior_temp', 0.05))
            log.info('P(seat) gate at %.0f%%: once the posterior clears it the trial commits to '
                     'ONE insertion with NO trajectory noise.', 100 * seat_gate)
            if seat_gate > basin.p_seat_zero:
                log.warning('The gate (%.0f%%) is above P(seat) at ZERO offset (%.0f%%) -- even '
                            'a PERFECT estimate would not clear it, so trials will use all '
                            'their attempts. Lower p_seat_threshold below %.0f%%, or raise '
                            'seat_margin_mm / lower depth_reference so more trials count as '
                            'seated.%s', 100 * seat_gate, 100 * basin.p_seat_zero,
                            100 * basin.p_seat_zero,
                            '' if seat_gate <= basin.p_seat_max else
                            f' (it is also above the basin-wide best, '
                            f'{100 * basin.p_seat_max:.0f}%.)')
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
    # DECAYS: shrink the whole perturbation by noise_decay_attempt each attempt
    # ((1-f)^(k-1)) and shed it linearly along the path by noise_decay_traj (1 -> 1-f at the
    # last waypoint). Deliberate offsets live in eval.collection (offset_sweep), not here.
    tn_da = float(tn.get('noise_decay_attempt', 0.0))
    tn_dt = float(tn.get('noise_decay_traj', 0.0))
    if tn.get('alternate_pitch_deg'):
        log.warning('eval.trajectory_noise.alternate_pitch_deg was REMOVED (2026-08-13) -- '
                    'use eval.collection.mode: offset_sweep for deliberate offsets. Ignored.')
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

    # ---- OBSERVATION COLLECTION MODE (eval.collection) --------------------------------------
    # 'attempts'      the original loop: one noised insertion per attempt.
    # 'offset_sweep'  each attempt COMMANDS a list of deliberate pose offsets (default pitch
    #                 -4..+4 deg in 2 deg steps, other dims zero) and pools all their
    #                 observations BEFORE estimating once -- the probe-diversity result
    #                 (1 -> 3+ trajectories: truth-in-top-2 basin 15% -> 51%) as a collection
    #                 mode. The belief is NOT updated between sweep passes, so their evidence
    #                 fuses exactly (each pass measures the same correction inverse(E)).
    # 'peck'          on a force stop the connector does NOT fully retract: it backs off a few
    #                 mm, advances again, and keeps going until the end of the trajectory or a
    #                 time budget -- one attempt logs a whole sequence of contact events, each
    #                 with its own stop depth for the stop-signature fusion.
    col = ev.get('collection', {}) or {}
    col_mode = str(col.get('mode', 'attempts')).strip().lower()
    if col_mode not in ('attempts', 'offset_sweep', 'peck'):
        log.error("eval.collection.mode %r must be 'attempts', 'offset_sweep' or 'peck'.",
                  col_mode)
        return False                               # bad values fail HERE, pre-motion
    sweep_offsets = col.get('sweep_offsets')
    if sweep_offsets is None:
        sweep_offsets = [[0.0, 0.0, 0.0, 0.0, float(p), 0.0]
                         for p in np.arange(-4.0, 4.01, 2.0)]
    sweep_offsets = [[float(v) for v in o] for o in sweep_offsets]
    if col_mode == 'offset_sweep':
        if any(len(o) != 6 for o in sweep_offsets) or not sweep_offsets:
            log.error('eval.collection.sweep_offsets must be a non-empty list of 6-vectors '
                      '[x, y, z (m), roll, pitch, yaw (deg)].')
            return False
        log.info('Collection mode OFFSET SWEEP: %d commanded offsets per attempt, pitch %s deg.',
                 len(sweep_offsets), [round(o[4], 1) for o in sweep_offsets])
    peck_mm = float(col.get('peck_retract_mm', 5.0))
    peck_timeout_s = float(col.get('peck_timeout_s', 30.0))
    if col_mode == 'peck':
        if peck_mm <= 0 or peck_timeout_s <= 0:
            log.error('eval.collection.peck_retract_mm and peck_timeout_s must be > 0.')
            return False
        log.info('Collection mode PECK: %.1f mm back-off on force stop, %.0f s budget per '
                 'attempt.', peck_mm, peck_timeout_s)

    # ---- DIVERGENCE BOUNDS (eval.abort_bounds): terminate the TRIAL when the belief error
    # left after an update exceeds them. A diverged belief drives every later pass into contact
    # the map has never seen (the drift-out-of-distribution spiral) and, at 15 deg, toward the
    # gripper/fixture collision regime -- there is nothing left to learn from that trial.
    # EVAL-ONLY safety net: it reads the ground-truth error, which production apps do not have.
    # <= 0 disables a bound.
    ab = ev.get('abort_bounds', {}) or {}
    abort_pos_mm = float(ab.get('pos_mm', 10.0))
    abort_rot_deg = float(ab.get('rot_deg', 15.0))
    if abort_pos_mm > 0 or abort_rot_deg > 0:
        log.info('Divergence bounds: trial aborts if the post-update belief error exceeds '
                 '%.1f mm or %.1f deg.', abort_pos_mm, abort_rot_deg)

    # COMPLIANCE + guard + speeds: same shape as uncertain_sampling; the config mirrors the pick
    # app's assembly values so the estimator sees production-like observations.
    adm = AdmittanceController(robot.arm, cfg.section('compliance'))
    guard_shared = ForceGuard(robot.arm, cfg.section('force_guard'))
    # OPTIONAL FINAL INSERTION: one extra guarded assemble per trial from the FINAL corrected
    # belief under a DIFFERENT stiffness (same compliance section otherwise) -- does the
    # corrected belief actually seat? Built here so a bad stiffness list fails pre-motion.
    # Default ON (2026-08-13): EVERY collection mode ends its trial with one zero-noise
    # insertion from the final corrected belief -- the jitter/sweep/peck exist to gather
    # observations, and have no place in the attempt that is meant to seat.
    # EVERY knob that may differ for the commit insertion lives in this ONE block. The probing
    # attempts and the attempt meant to SEAT want different physics -- probing wants a light,
    # early-stopping touch that gathers varied contact; the commit wants to press home -- and
    # before this grouping those settings were spread across compliance:, force_guard: and
    # here, so a change made for probing silently changed the commit too. Anything left unset
    # (or null) inherits the shared block, so the default behaviour is unchanged.
    fi = ev.get('final_insertion', {}) or {}
    fi_on = bool(fi.get('enabled', True))
    adm_final, guard_final = None, None
    fi_settle = fi.get('settle_s')
    fi_hold = fi.get('hold_after_insertion_s')
    fi_settle = None if fi_settle is None else float(fi_settle)
    fi_hold = None if fi_hold is None else float(fi_hold)
    for k, v in (('settle_s', fi_settle), ('hold_after_insertion_s', fi_hold)):
        if v is not None and v < 0:
            log.error('eval.final_insertion.%s must be >= 0 (got %.2f).', k, v)
            return False                           # bad values fail HERE, pre-motion
    if fi_on:
        comp_final = dict(cfg.section('compliance'))
        for src, dst in (('stiffness', 'stiffness'), ('mass', 'mass'),
                         ('damping_ratio', 'damping_ratio')):
            if fi.get(src) is not None:
                comp_final[dst] = [float(v) for v in fi[src]]
        adm_final = AdmittanceController(robot.arm, comp_final)
        gsec = dict(cfg.section('force_guard'))
        over_g = {k: fi[k] for k in ('max_force_n', 'max_torque_nm', 'persistence_s')
                  if fi.get(k) is not None}
        guard_final = ForceGuard(robot.arm, {**gsec, **over_g}) if over_g else None
        log.info('Final insertion ON: stiffness %s, guard %.0f N (persistence %.2f s)%s%s.',
                 comp_final.get('stiffness'),
                 float(over_g.get('max_force_n', gsec.get('max_force_n', 0.0))),
                 float(over_g.get('persistence_s', gsec.get('persistence_s', 0.0))),
                 f', settle {fi_settle:.1f} s' if fi_settle is not None else '',
                 f', dwell {fi_hold:.1f} s' if fi_hold is not None else '')
    tare = (lambda: robot.arm.zero_ft(settle=False)) \
        if bool(cfg.get_path('compliance.tare_before', True)) else None
    settle_shared = float(cfg.get_path('compliance.settle_s', 0.5))
    # DWELL at the end of an insertion, AFTER the logged settle. Two differences from
    # settle_s, both deliberate:
    #   * NOT LOGGED -- it adds no observation rows, so lengthening the dwell cannot change
    #     what the estimator sees (settle_s does: its rows are all at the deepest contact).
    #   * UN-GUARDED -- settle_s passes the force guard, so an already-tripped guard ends that
    #     hold on its first cycle. A dwell meant to keep PRESSING (letting a stiff connector
    #     finish seating, or holding a mate steady for inspection/force reading) must not be
    #     cut short by the limit it is deliberately sitting on. The admittance spring still
    #     bounds the force: it settles at stiffness x reference penetration, no integration.
    hold_shared = float(cfg.get_path('compliance.hold_after_insertion_s', 0.0))
    if hold_shared < 0:
        log.error('compliance.hold_after_insertion_s must be >= 0 (got %.2f).', hold_shared)
        return False                               # bad values fail HERE, pre-motion
    if hold_shared > 0:
        log.info('Post-insertion dwell: %.1f s of UN-guarded hold at the stop after every '
                 'insertion (not logged as observations). eval.final_insertion may override '
                 'it for the commit.', hold_shared)
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

    def run_insertion(adm_ctl, refs, T_bel, peck=False, guard_ctl=None, settle=None,
                      hold=None):
        """One admittance-followed insertion along refs, collecting observations (same law and
        logging as cable_pick_estimate_assemble), the seated kinematic check, then the compliant
        UN-guarded retract along the believed part's own -X (a seated part is already over the
        guard limit; a guarded retract would block itself).

        peck=True (eval.collection.mode: peck): a force stop does NOT end the advance -- back
        off peck_retract_mm along the believed -X, reset the guard, and continue along the
        trajectory; the loop ends at the last waypoint or when peck_timeout_s runs out. One
        call then logs a SEQUENCE of contact events, each recorded in `stops`.

        Returns (obs, seated, lin, ang, seat6, stops) with `stops` the believed stop depth(s)
        in mm -- one per contact event (peck), one for the single stop (normal), or the deepest
        point reached when the trajectory completed without a force stop."""
        # The commit insertion may run its own guard / settle / dwell (eval.final_insertion);
        # everything else passes None and gets the shared ones.
        guard = guard_ctl if guard_ctl is not None else guard_shared
        settle_s = settle if settle is not None else settle_shared
        hold_s = hold if hold is not None else hold_shared
        obs, cnt = [], [0]

        def log_cb():
            cnt[0] += 1
            if cnt[0] % decim == 0:
                obs.append(_observe(robot, T_bel, T_base_tconn))

        adm_ctl.reset()
        adm_ctl.warmup(refs[0], tare_fn=tare)
        guard.reset()
        last_ref, seated, stops = refs[0], False, []
        t0 = time.time()
        prev, i = refs[0], 1
        while i < len(refs):
            res = adm_ctl.ramp(prev, refs[i], seg_time(prev, refs[i]), guard, on_step=log_cb)
            last_ref = refs[i]
            if res == 'seated':
                seated = True
                if obs:
                    stops.append(float(obs[-1][0]))
                if not peck:
                    log.info('Contact limit at waypoint %d/%d -- stopped advancing.',
                             i, len(refs) - 1)
                    break
                if time.time() - t0 > peck_timeout_s:
                    log.info('PECK: time budget (%.0f s) spent after %d contact(s) -- holding '
                             'here.', peck_timeout_s, len(stops))
                    break
                # back off a little, UN-guarded (we are at the guard limit by construction),
                # then continue with the NEXT waypoint -- 'continue on the trajectory'
                T_out = _retract_ref(refs[i], T_bel, peck_mm / 1000.0)
                adm_ctl.ramp(refs[i], T_out, seg_time(refs[i], T_out, rv_mm_s, rw_deg_s),
                             guard=None, on_step=log_cb)
                guard.reset()
                prev = T_out
                i += 1
                continue
            prev = refs[i]
            i += 1
        if not stops and obs:                      # ran to the end: deepest point IS the stop
            stops = [float(np.max(np.asarray(obs, dtype=float).reshape(-1, 12)[:, 0]))]
        if peck:
            log.info('PECK: %d contact event(s), stop depths %s mm.', len(stops),
                     [round(s, 1) for s in stops])
        adm_ctl.hold(last_ref, settle_s, guard, on_step=log_cb)
        if hold_s > 0:                             # dwell: un-guarded, unlogged (see above)
            log.info('   holding the stop for %.1f s.', hold_s)
            adm_ctl.hold(last_ref, hold_s, guard=None)

        # Kinematic check numbers (BELIEVED pose), for the record only.
        lin, ang = pose_error(robot.tool0() @ T_bel, T_base_tconn)
        # PHYSICAL seat pose: the TRUE connector wrt the TARGET connector at the end of the
        # hold (the part is fixtured, so T_true is exact) -- the per-DOF success measure.
        xyz, rpy = matrix_to_xyzrpy(inverse(T_base_tconn) @ robot.tool0() @ T_true)
        seat6 = list(xyz * 1000.0) + list(np.degrees(rpy))

        T_out = _retract_ref(last_ref, T_bel, retract_m)
        adm_ctl.ramp(last_ref, T_out, seg_time(last_ref, T_out, rv_mm_s, rw_deg_s), guard=None)
        adm_ctl.stop()
        return obs, seated, lin, ang, seat6, stops

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
            track6, trackr, trackg, trackl2, trackp = [inj], [], [], [], []
            acc = np.zeros((0, 12))
            seeds = None                           # rival modes carried between attempts

            abandoned = False
            gate_open = False
            diverged = False                       # set when abort_bounds terminate the trial
            for attempt in range(1, max_attempts + 1):
                errb, errb_pos, errb_rot = _gt_error(T_true, T_believed)
                # COLLECTION PASSES for this attempt: the sweep commands one insertion per
                # deliberate offset (belief FIXED across passes, so their evidence fuses
                # exactly); the other modes make a single pass. The pass bias rides on top of
                # the usual per-waypoint noise when that is enabled.
                passes = ([list(o) for o in sweep_offsets] if col_mode == 'offset_sweep'
                          else [None])
                obs, attempt_stops = [], []
                seated, lin, ang, seat6 = False, 0.0, 0.0, [0.0] * 6
                for pi, poff in enumerate(passes):
                    bias = poff                    # sweep offset, or None for a plain pass
                    if tn_on or bias is not None:
                        rows_t = traj.noised(dense, noise_rng,
                                             tn_std if tn_on else [0.0] * 6, tn_w, tn_dt,
                                             (1.0 - tn_da) ** (attempt - 1), bias)
                    else:
                        rows_t = dense
                    refs = [T_base_tconn @ row @ inverse(T_believed) for row in rows_t]

                    # To the pass start -- stiff, free space (retract/stand-off cleared it).
                    label = (f'trial {trial} attempt {attempt}'
                             + (f' sweep {pi + 1}/{len(passes)}' if poff is not None else '')
                             + ' start')
                    q = robot.arm.ik(refs[0], seed_q)
                    if q is None or not robot.arm.move_j(q, label=label):
                        log.warning('IK/approach failed; abandoning the rest of trial %d.',
                                    trial)
                        abandoned = True
                        break
                    seed_q = q

                    # ASSEMBLE under admittance (same law as the pick app), check, retract.
                    obs_i, seated, lin, ang, seat6, stops_i = run_insertion(
                        adm, refs, T_believed, peck=(col_mode == 'peck'))
                    obs.extend(obs_i)
                    attempt_stops.extend(stops_i)
                    if poff is not None:
                        log.info('  sweep %d/%d (pitch %+.1f deg, z %+.1f mm): %d obs, '
                                 'stop %s mm.', pi + 1, len(passes), poff[4], poff[2] * 1000.0,
                                 len(obs_i), [round(s, 1) for s in stops_i])
                if abandoned:
                    break
                # success is judged on the LAST pass's seat pose (for the sweep that is the
                # final commanded offset -- the record of whether probing itself ever seated)
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
                # SEED the multi-start on the previous attempt's rival modes. Random restarts
                # rediscover the DOMINANT basin easily and the runner-up only by luck, so an
                # attempt meant to tell two hypotheses apart can silently re-examine just one of
                # them. Seeds are corrections relative to the belief that produced them, so they
                # are carried through each belief update below rather than reused verbatim.
                land_pack = None
                if commit == 'argmin':
                    # NO ICP: the multi-start solver's answer would only be discarded below.
                    T_corr_mm, info, land_pack = _argmin_estimate(estimator, vec6, w6)
                else:
                    T_corr_mm, info = estimator.estimate(vec6, w6, seeds=seeds)
                seeds = None
                land, p_seat, icp_paths = None, float('nan'), None
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
                    # The LANDSCAPE moves ahead of the belief update: mode ranking picks the
                    # committed correction from it, and the diagnostics below reuse the pack
                    # (it depends only on the PRE-update observations, so nothing changes).
                    if land_pack is None and ((save_plots and plot_land) or basin is not None):
                        try:
                            land_pack = _landscape(estimator, vec6, w6)
                        except Exception as exc:   # noqa: BLE001 -- diagnostics never fatal
                            log.warning('landscape skipped (%s)', exc)
                    # ARGMIN COMMITMENT (estimation.commit: argmin): throw away the multi-start
                    # aggregator's vote and commit the single deepest cell of the dense
                    # landscape. No consensus, no mixture, no shrink -- the estimator IS the
                    # argmin. Measured offline on the 2026-08-13 AM data (analysis/
                    # landscape_sweep): |z'| 3.3-9.6 mm at ~50% win depending on the metric, vs
                    # ~2.0 mm / 84% for the aggregator, so this is a DIAGNOSTIC option -- it
                    # makes the landscape's own quality visible end-to-end on hardware.
                    row['commit'] = commit
                    if commit == 'argmin':
                        log.info('COMMIT ARGMIN (no ICP): %s.',
                                 {d: round(float(v), 2)
                                  for d, v in info['theta_corr'].items()})
                    # MODE RANKING at commitment (seat_gate.mode_ranking: p_seat): the
                    # aggregator's estimate is only the default candidate -- a rival mode of
                    # the multi-start finals that does MORE for P(seat) wins. NOTE icp_residual
                    # and icp_inliers below keep describing the aggregator's own solution.
                    row['ranked_mode'] = -1
                    if (basin is not None and mode_rank == 'p_seat' and land_pack is not None
                            and info.get('mixture') is not None
                            and info['mixture'].n_modes > 1):
                        try:
                            ax_r, E_r = land_pack[0], land_pack[1]
                            base = [info['theta_corr'][d] for d in estimator.estimate_dims]
                            ps_b = _p_seat_land(basin, ax_r, E_r, estimator.idx, base,
                                                seat_temp)
                            row['p_seat_unranked'] = ps_b
                            best = (None, ps_b, -1)
                            for m, comp in enumerate(info['mixture'].components):
                                ps_m = _p_seat_land(basin, ax_r, E_r, estimator.idx,
                                                    list(comp.mean), seat_temp)
                                if ps_m > best[1] + 1e-12:
                                    best = (list(comp.mean), ps_m, m)
                            if best[2] >= 0:
                                log.info('MODE RANKING: committing mode %d %s over the '
                                         'aggregator %s (P(seat) %.0f%% vs %.0f%%).', best[2],
                                         [round(float(v), 2) for v in best[0]],
                                         {d: round(v, 2)
                                          for d, v in info['theta_corr'].items()},
                                         100 * best[1], 100 * ps_b)
                                c6r = np.zeros(6)
                                c6r[estimator.idx] = best[0]
                                T_corr_mm = mats_from_vec6(c6r)
                                info['theta_corr'] = {
                                    d: float(v) for d, v in
                                    zip(estimator.estimate_dims, best[0])}
                                row['ranked_mode'] = best[2]
                        except Exception as exc:   # noqa: BLE001 -- ranking is best-effort
                            log.warning('mode ranking skipped (%s)', exc)
                    if dbg_on and (attempt - 1) % dbg_every == 0:
                        try:
                            # errb is the belief error BEFORE this update, so the correction
                            # that would cancel it exactly is its inverse -- the TRUTH the
                            # committed estimate is being compared against.
                            t6 = vec6_from_mats(np.linalg.inv(
                                mats_from_vec6(np.asarray(errb, dtype=float))))
                            manifold_debug.figures(
                                estimator, vec6, w6, dict(info['theta_corr']),
                                {d: float(t6[DIMS.index(d)])
                                 for d in estimator.estimate_dims},
                                os.path.join(out_dir, f'trial_{trial:03d}_attempt_'
                                                      f'{attempt:02d}_match.png'),
                                title=f'trial {trial} attempt {attempt}',
                                max_rows=dbg_rows, grid_n=dbg_grid, live_dir=dbg_live)
                        except Exception as exc:   # noqa: BLE001 -- diagnostics never fatal
                            log.warning('match diagnostics skipped (%s)', exc)
                    T_believed = T_believed @ _corr_to_m(T_corr_mm)   # believed @ corr ~= true
                    if accumulate:
                        # keep every stored row expressed in the belief JUST updated
                        acc = _rebase_rows(full, T_corr_mm) if len(full) else full
                    row['estimate'] = 'ok'
                    row.update({f'corr_{k}': v for k, v in info['theta_corr'].items()})
                    row.update({'icp_inliers': info['inliers'],
                                'icp_residual': info['final_residual'],
                                'n_mixture_modes': info.get('n_mixture_modes'),
                                'ambiguity': info.get('ambiguity'),
                                'between_frac': info.get('between_frac'),
                                'separation': info.get('separation'),
                                'seeded_guesses': info.get('seeded_guesses', 0)})
                    mix_icp = info.get('mixture')
                    if mix_icp is not None:
                        row.update({f'sigma_within_{d}': float(s) for d, s in
                                    zip(estimator.estimate_dims, mix_icp.sigma(True))})
                        if mix_icp.n_modes > 1:
                            log.info('   %d rival modes over the finals: %s | %.0f%% of the mass '
                                     'off the leader, separation %.1f, %.0f%% of the spread is '
                                     'AMBIGUITY not measurement width.', mix_icp.n_modes,
                                     ' vs '.join('(' + ', '.join(f'{v:+.1f}' for v in c.mean)
                                                 + f') w={c.weight:.2f}'
                                                 for c in mix_icp.components[:3]),
                                     100 * mix_icp.ambiguity, mix_icp.separation,
                                     100 * mix_icp.between_frac)
                        # Carry the modes into the NEXT attempt, re-expressed in the belief we
                        # are about to adopt: mode M satisfied believed @ M ~= true, and the new
                        # belief is believed @ C, so the same hypothesis is now inverse(C) @ M.
                        # Seeds only steer the multi-start, so they are pointless without it.
                        try:
                            if commit == 'argmin':
                                raise StopIteration
                            m6 = np.zeros((mix_icp.n_modes, 6))
                            m6[:, estimator.idx] = np.array([c.mean for c in mix_icp.components])
                            nxt = vec6_from_mats(np.linalg.inv(T_corr_mm)
                                                 @ mats_from_vec6(m6))[:, estimator.idx]
                            seeds = [row_ for row_ in nxt]
                        except StopIteration:
                            seeds = None               # commit: argmin -- no multi-start to seed
                        except Exception as exc:   # noqa: BLE001 -- seeding is an optimisation
                            log.debug('mode seeding skipped (%s)', exc)
                    # TRUST readout (observational): both scores + the raw signals, so the
                    # run's ground truth can score the uncertainty estimates themselves.
                    chk = info.get('check')
                    if chk:
                        row['trust_rankavg2'] = chk['rankavg2']
                        row['trust_cauchy'] = chk['cauchy']
                        row.update({f'trust_{k}': v for k, v in chk['signals'].items()
                                    if k in _TRUST_SIGNALS})
                    trackr.append(float(info['final_residual']))
                    # The per-guess populations exist only when the ICP ran (commit:
                    # aggregator). In argmin mode there are no guesses to plot, so the
                    # consensus/outcome tracks stay empty and the figures just omit them.
                    if info.get('theta_hist') is not None:
                        trackg.append(np.asarray(info['res_hist'], dtype=float)[:, -1])
                        # the whole ICP run (guesses x iterations x dims) + the belief error it
                        # was solved from, so the phase plot draws convergence in error coords
                        icp_paths = (np.asarray(info['theta_hist'], dtype=float),
                                     np.asarray(errb, dtype=float))
                        # Per-guess would-be OUTCOME: the ground-truth L2 error left if guess
                        # g's correction had been applied to the PRE-update belief (mm/deg).
                        th6 = np.zeros((len(info['theta_hist']), 6))
                        th6[:, estimator.idx] = info['theta_hist'][:, -1]
                        rem6 = vec6_from_mats(
                            mats_from_vec6(np.asarray(errb)) @ mats_from_vec6(th6))
                        trackl2.append(np.sqrt(
                            (rem6[:, :3] ** 2).sum(axis=1)
                            + ((estimator.s_rot * rem6[:, 3:]) ** 2).sum(axis=1)))
                    else:
                        trackg.append(np.zeros(0))
                        trackl2.append(np.zeros(0))
                    # The landscape the correction was picked from, with the TRUE correction
                    # (believed @ C = true, so C = inverse(err_before)) for reference.
                    # (Computed ONCE above, before the belief update, for the mode ranking.)
                    if land_pack is not None:
                        try:
                            (ax_l, E_l, sig_l, amin_l, cov_l, mix_l, sup_l) = land_pack
                            # P(SEAT) of the POSTERIOR, not of the point estimate -- evaluated
                            # at the correction ACTUALLY committed (theta_corr reflects the
                            # mode ranking when it fired).
                            if basin is not None:
                                p_seat = _p_seat_land(
                                    basin, ax_l, E_l, estimator.idx,
                                    [info['theta_corr'][d]
                                     for d in estimator.estimate_dims], seat_temp)
                                row['p_seat'] = p_seat
                            trk = np.asarray(track6, dtype=float)[:, estimator.idx]
                            # SUPPORT is logged, NOT multiplied into sigma: the 2026-08-12
                            # validation found the ratio predicts failure on the old-process
                            # eval runs but is INVERTED on the probe runs (deep seated probes
                            # read as thin support), so a multiplier would widen sigma on the
                            # best cases exactly when the process is right.
                            land = (ax_l, E_l, sig_l, amin_l, dict(info['theta_corr']),
                                    np.asarray(errb, dtype=float),
                                    (np.asarray(info['theta_hist'], dtype=float)[:, -1, :]
                                     if info.get('theta_hist') is not None
                                     else np.zeros((0, len(estimator.idx)))),
                                    list(estimator.idx), trk, cov_l, p_seat, mix_l)
                            row.update({f'sigma_{d}': v for d, v in sig_l.items()})
                            row.update({f'cov_{a}{b}': float(cov_l[a, b])
                                        for a in range(len(cov_l))
                                        for b in range(a, len(cov_l))})
                            row.update({'n_modes_land': mix_l.n_modes,
                                        'ambiguity_land': mix_l.ambiguity,
                                        'between_frac_land': mix_l.between_frac,
                                        'separation_land': mix_l.separation,
                                        'support_ratio': sup_l})
                            if sup_l > 1.5:
                                log.info('   support x%.2f: map evidence at this correction is '
                                         '%.0f%% further than typical at this depth (logged '
                                         'for analysis; does not inflate sigma -- see the '
                                         'support note).', sup_l, 100 * (sup_l - 1.0))
                        except Exception as exc:   # noqa: BLE001 -- diagnostics never fatal
                            log.warning('landscape/P(seat) skipped (%s)', exc)
                erra, erra_pos, erra_rot = _gt_error(T_true, T_believed)
                track6.append(erra)
                trackp.append(p_seat)
                if save_plots:                     # re-saved after EVERY attempt of this trial
                    status = (f'trial {trial}/{num_trials}  |  successes {n_succ}/{trial} '
                              f'({n_succ / trial:.0%})')
                    _plot_trial_errors(os.path.join(out_dir, f'trial_{trial:03d}_errors.png'),
                                       trial, estimator.estimate_dims, track6, trackr,
                                       estimator.s_rot, live_path, trackg, trackl2, status,
                                       land, trackp, seat_gate if basin is not None else None,
                                       basin, icp_paths, list(estimator.idx))
                row.update({f'err_after_{s}': v for s, v in zip(_ERR, erra)})
                row.update({'err_after_pos_mm': erra_pos, 'err_after_rot_deg': erra_rot})
                row['converged'] = bool(erra_pos <= tol_pos_mm and erra_rot <= tol_rot_deg)
                # DIVERGENCE BOUNDS: a post-update belief error beyond eval.abort_bounds ends
                # the TRIAL -- no further attempts, no final insertion (a diverged belief has
                # nothing left to teach and heads for the collision regime).
                diverged = bool((abort_pos_mm > 0 and erra_pos > abort_pos_mm)
                                or (abort_rot_deg > 0 and erra_rot > abort_rot_deg))
                row['diverged'] = diverged
                log.info('trial %d attempt %d: gt error %.2f mm / %.2f deg -> %.2f mm / %.2f deg'
                         '%s%s%s', trial, attempt, errb_pos, errb_rot, erra_pos, erra_rot,
                         '' if not np.isfinite(p_seat) else f'  P(seat) {p_seat:.0%}',
                         '  (CONVERGED)' if row['converged'] else '',
                         '  (SUCCESS -- seated within tolerance)' if succ else '')
                rows.append(row)
                writer.writerow(row)
                fout.flush()                       # a 50-trial run must survive an abort mid-way
                os.fsync(fout.fileno())
                if diverged:
                    log.error('TRIAL %d TERMINATED: belief error %.2f mm / %.2f deg exceeds '
                              'the divergence bounds (%.1f mm / %.1f deg).', trial,
                              erra_pos, erra_rot, abort_pos_mm, abort_rot_deg)
                    break
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
            if (fi_on or gate_open) and not abandoned and not diverged:
                refs = [T_base_tconn @ row @ inverse(T_believed) for row in dense]
                q = robot.arm.ik(refs[0], seed_q)
                if q is None or not robot.arm.move_j(q, label=f'trial {trial} final insertion'):
                    log.warning('IK/approach failed for the final insertion of trial %d.', trial)
                else:
                    seed_q = q
                    obs, seated, lin, ang, seat6, _ = run_insertion(
                        adm_final, refs, T_believed, guard_ctl=guard_final,
                        settle=fi_settle, hold=fi_hold)
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
