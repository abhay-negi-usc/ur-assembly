"""MANIFOLD MATCH DIAGNOSTICS -- why did the energy prefer the wrong correction?

The estimator's whole claim is: transform the observations by a candidate correction, and the
TRUE correction makes them land on the manifold. When the argmin is not the truth, exactly one
of three things went wrong, and these figures separate them:

  1. THE TRUTH DOES NOT FIT.       The observations at the true correction genuinely sit off
                                   the manifold -- calibration drift, a changed fixture, a map
                                   that never covered this offset. Look at: the overlay panels
                                   (do the green points lie in the map's support?) and the
                                   per-row residual profile (is the mismatch everywhere, or
                                   only past a certain depth?).
  2. A RIVAL FITS BETTER.          Both fit; the wrong one fits better. This is aliasing -- the
                                   part rides its offset and reproduces another offset's
                                   signature. Look at: the neighbour-vote panel (which map
                                   region do the rows actually resemble?) and the channel
                                   energies (do pose and wrench disagree about where the
                                   minimum is?).
  3. ONE CHANNEL DOMINATES.        The 12-D metric is a weighted sum; if one channel's residual
                                   is an order of magnitude larger it decides the argmin alone.
                                   Look at: the channel bar chart and the per-channel energy
                                   panels -- if the force channel's minimum is at the truth but
                                   the total's is not, the weighting is the bug, not the map.

Everything here is PURE (estimator + rows in, figures out), so it runs identically inside
estimator_eval on the robot and offline over recorded attempts.

The metric being probed is the estimator's own: pts = scaled12(observed (.) theta), matched to
the manifold KD-tree with the same soft-kNN interpolation the energy uses, so a residual read
off these plots is the number the energy actually summed.
"""

import os
import shutil

import numpy as np

from .. import log as urlog
from .manifold import DIMS, mats_from_vec6, scaled12, vec6_from_mats

log = urlog.get('manifold_debug')

# The 12-D metric's physical blocks. Splitting the residual this way is the whole point: the
# energy is one number, but it is a sum over channels that can disagree about the answer.
CHANNELS = (('translation', slice(0, 3), 'tab:blue'),
            ('rotation', slice(3, 6), 'tab:orange'),
            ('force', slice(6, 9), 'tab:green'),
            ('torque', slice(9, 12), 'tab:red'))

# The same 12 dimensions ONE AT A TIME. The block view above can hide the decisive detail: a
# block is a 3-vector norm, so a single axis inside it (say fz, or the y translation the
# fixture pins) can carry the whole residual while its two partners contribute nothing. Every
# per-DOF number here is in METRIC space -- i.e. AFTER s_rot, dim_weights and the wrench
# scalings -- because that is what the energy actually sums. A dimension whose weight is 0
# reads as a flat zero, which is itself worth seeing.
DOFS = (('x', 0, 'tab:blue'), ('y', 1, 'tab:blue'), ('z', 2, 'tab:blue'),
        ('roll', 3, 'tab:orange'), ('pitch', 4, 'tab:orange'), ('yaw', 5, 'tab:orange'),
        ('fx', 6, 'tab:green'), ('fy', 7, 'tab:green'), ('fz', 8, 'tab:green'),
        ('tx', 9, 'tab:red'), ('ty', 10, 'tab:red'), ('tz', 11, 'tab:red'))


def _blend(est, M, pts, dist, sl=None, name=None):
    """The interpolated target for one block (or the whole 12-D point), using the estimator's
    OWN kernel -- soft kNN, weights exp(-(d - d_min)/tau), NOT a raw nearest neighbour.

    Which tau: the per-block one when estimation.interp_softness is a dict (then the block's
    own distances drive the weights, exactly as blend_residual does), otherwise the joint
    bandwidth off the joint distance. Keeping this in step with skills/manifold matters --
    a diagnostic that blends differently from the energy would send debugging the wrong way."""
    if M.shape[1] <= 1:
        return M[:, 0] if sl is None else M[:, 0, sl]
    by_block = getattr(est, 'softness_by_block', None)
    if sl is not None and by_block is not None and name in getattr(est, 'tau_by_block', {}):
        d = np.linalg.norm(M[:, :, sl] - pts[:, None, sl], axis=2)
        tau = est.tau_by_block[name]
        base = d.min(axis=1, keepdims=True)
    else:
        d, tau, base = dist, est.interp_tau, dist[:, :1]
    bw = np.exp(-(d - base) / max(tau, 1e-12))
    bw /= bw.sum(axis=1, keepdims=True)
    return np.einsum('mk,mkd->md', bw, M if sl is None else M[:, :, sl])


def _raw_for(est, raw, n):
    """The raw wrench to re-base with: the caller's, else the estimator's own stash from
    prepare_observations. Same fallback the energy paths use, so a diagnostic called without
    `raw` still describes the metric the estimator is actually running."""
    if raw is None:
        raw = getattr(est, 'last_raw', None)
    if raw is not None and len(raw[0]) != n:
        return None                              # stale or subsampled: never guess alignment
    return raw


def _wrench_at(est, w6, raw, theta6):
    """The wrench the ESTIMATOR would use for this candidate (see manifold.wrench6_at)."""
    if raw is not None and getattr(est, 'wrench_follows_correction', False):
        return est.wrench6_at(raw[0], raw[1], theta6)
    return w6


def _theta6(theta, idx):
    """A 6-vector correction from either a dict {dim: value} or a sequence over `idx`."""
    t6 = np.zeros(6)
    if isinstance(theta, dict):
        for d, v in theta.items():
            t6[DIMS.index(d)] = float(v)
    else:
        t6[list(idx)] = np.asarray(theta, dtype=float).ravel()
    return t6


def match(est, vec6, w6, theta, raw=None):
    """Apply a correction and match to the manifold, exactly as the energy does.

    Returns (pts12, tgt12, resid12, pose6) -- the transformed observations in metric space, the
    soft-kNN interpolated manifold target, the signed per-row residual, and the transformed
    observation poses in PHYSICAL units (mm / deg) for plotting."""
    t6 = _theta6(theta, est.idx)
    raw = _raw_for(est, raw, len(np.asarray(vec6, dtype=float)))
    C = mats_from_vec6(np.asarray(vec6, dtype=float)) @ mats_from_vec6(t6)
    pose6 = vec6_from_mats(C)
    # The wrench rides with the candidate exactly as it does in the energy -- without this the
    # plots would describe a metric the estimator no longer uses.
    pts = scaled12(pose6, np.asarray(_wrench_at(est, w6, raw, t6), dtype=float), est.s_rot,
                   getattr(est, 'dim_w', None))
    k = max(int(est.interp_neighbors), 1)
    dist, nn = est.tree.query(pts, k=k, workers=-1)
    if dist.ndim == 1:
        dist, nn = dist[:, None], nn[:, None]
    M = est.M12[nn]
    if getattr(est, 'softness_by_block', None) is None:
        tgt = _blend(est, M, pts, dist)
    else:                                    # per-block targets, one bandwidth each
        tgt = np.empty_like(pts)
        for name, sl, _ in CHANNELS:
            tgt[:, sl] = _blend(est, M, pts, dist, sl, name)
    return pts, tgt, tgt - pts, pose6, nn


def channel_energy(est, vec6, w6, grid6, raw=None):
    """Per-channel mean residual over a grid of candidate corrections.

    Returns {channel: E} plus 'total' -- the same quantity the estimator minimises, decomposed.
    A channel whose minimum sits at the truth while the total's does not is a WEIGHTING bug; if
    every channel agrees on the wrong answer, the map or the calibration is the problem."""
    Y = mats_from_vec6(np.asarray(vec6, dtype=float))
    w = np.asarray(w6, dtype=float)
    raw = _raw_for(est, raw, len(Y))
    out = {name: np.empty(len(grid6)) for name, _, _ in CHANNELS}
    out.update({name: np.empty(len(grid6)) for name, _, _ in DOFS})
    out['total'] = np.empty(len(grid6))
    k = max(int(est.interp_neighbors), 1)
    # chunk over candidates -- the gather is (candidates x rows, k, 12) and must not blow up
    per = max(int(128e6 // max(len(vec6) * k * 12 * 8, 1)), 1)
    for lo in range(0, len(grid6), per):
        hi = min(lo + per, len(grid6))
        C = np.einsum('nij,kjl->knil', Y, mats_from_vec6(grid6[lo:hi]))
        if raw is not None and getattr(est, 'wrench_follows_correction', False):
            wg = np.concatenate([est.wrench6_at(raw[0], raw[1], g) for g in grid6[lo:hi]],
                                axis=0)
            pts = scaled12(vec6_from_mats(C).reshape(-1, 6), wg, est.s_rot,
                           getattr(est, 'dim_w', None))
        else:
            pts = scaled12(vec6_from_mats(C), w, est.s_rot,
                           getattr(est, 'dim_w', None)).reshape(-1, 12)
        dist, nn = est.tree.query(pts, k=k, workers=-1)
        if dist.ndim == 1:
            dist, nn = dist[:, None], nn[:, None]
        M = est.M12[nn]
        if getattr(est, 'softness_by_block', None) is None:
            tgt = _blend(est, M, pts, dist)
        else:
            tgt = np.empty_like(pts)
            for nm, sl2, _ in CHANNELS:
                tgt[:, sl2] = _blend(est, M, pts, dist, sl2, nm)
        r = (tgt - pts).reshape(hi - lo, len(vec6), 12)
        out['total'][lo:hi] = np.linalg.norm(r, axis=2).mean(axis=1)
        for name, sl, _ in CHANNELS:
            out[name][lo:hi] = np.linalg.norm(r[:, :, sl], axis=2).mean(axis=1)
        for name, j, _ in DOFS:                  # per-DOF: |residual| of that axis alone
            out[name][lo:hi] = np.abs(r[:, :, j]).mean(axis=1)
    return out


def grid_over(est, half=None, n=41):
    """A dense grid of candidate corrections over the estimated dims (6-vectors)."""
    dims = est.estimate_dims
    halves = [float((half or {}).get(d, est.init_range.get(d, 8.0))) or 8.0 for d in dims]
    axes = [np.linspace(-h, h, n) for h in halves]
    mesh = np.meshgrid(*axes, indexing='ij')
    G = np.zeros((mesh[0].size, 6))
    for j, m in zip(est.idx, mesh):
        G[:, j] = m.ravel()
    return G, axes, mesh[0].shape


def manifold_pose(est, rows=None):
    """The manifold's own poses in PHYSICAL units (mm / deg), recovered from metric space."""
    M = est.M12 if rows is None else est.M12[rows]
    return M[:, :6] / np.maximum(est.pose_scale6, 1e-12)


def figures(est, vec6, w6, theta_est, theta_true, out_path, title='', max_rows=250,
            grid_n=41, raw=None, live_dir=None):
    """Write the diagnostic figure comparing the ESTIMATED and TRUE corrections.

    Six panels: two physical overlays (depth vs each estimated dim) showing where the
    observations land inside the map under each correction, the per-row residual profile
    against depth, the channel decomposition, the neighbour-vote map (which part of the
    manifold the rows actually resemble), and the per-channel energy over the candidate grid
    with both corrections marked."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    v = np.asarray(vec6, dtype=float)
    w = np.asarray(w6, dtype=float)
    raw = _raw_for(est, raw, len(v))
    if len(v) > max_rows:                       # bound the cost; the shape is unaffected
        sel = np.linspace(0, len(v) - 1, max_rows).astype(int)
        v, w = v[sel], w[sel]
        if raw is not None:
            raw = (raw[0][sel], raw[1][sel])
    dims = list(est.estimate_dims)
    packs = {}
    for name, th in (('estimate', theta_est), ('truth', theta_true)):
        if th is None:
            continue
        pts, tgt, res, pose6, nn = match(est, v, w, th, raw=raw)
        packs[name] = dict(pose=pose6, res=res, nn=nn,
                           rowwise=np.linalg.norm(res, axis=1))
    if not packs:
        return None
    colors = {'estimate': 'tab:orange', 'truth': 'tab:green'}

    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5))
    Mp = manifold_pose(est)
    sub = np.linspace(0, len(Mp) - 1, min(len(Mp), 40000)).astype(int)

    # ---- A/B: physical overlays, depth (x) against each estimated dim -----------------
    for c, d in enumerate(dims[:2]):
        ax = axes[0, c]
        j = DIMS.index(d)
        ax.hexbin(Mp[sub, 0], Mp[sub, j], gridsize=60, cmap='Greys', bins='log',
                  mincnt=1)
        for name, p in packs.items():
            ax.plot(p['pose'][:, 0], p['pose'][:, j], '.', ms=4, color=colors[name],
                    label=f'obs @ {name}')
        ax.set_xlabel('insertion depth x (mm)')
        ax.set_ylabel(f'{d}')
        ax.set_title(f'A{c + 1}. where the observations LAND in the map (grey = manifold)',
                     fontsize=9)
        ax.legend(fontsize=7)
    if len(dims) == 1 and 'truth' in packs and 'estimate' in packs:
        # Spare panel: WHERE and IN WHICH CHANNEL the truth loses. Positive = the truth's
        # residual is worse than the estimate's at that depth, so this is the exact evidence
        # that voted the truth down -- read with panel C (which shows the totals).
        ax = axes[0, 1]
        pt, pe = packs['truth'], packs['estimate']
        o = np.argsort(pt['pose'][:, 0])
        xd = pt['pose'][o, 0]
        for name, sl, col in CHANNELS:
            dv = (np.linalg.norm(pt['res'][:, sl], axis=1)
                  - np.linalg.norm(pe['res'][:, sl], axis=1))[o]
            if np.allclose(dv, 0.0):
                continue                     # channel switched off (its scale is 0)
            ax.plot(xd, dv, lw=1.1, color=col, label=name)
        ax.axhline(0.0, color='k', lw=1)
        ax.set_xlabel('insertion depth x (mm)')
        ax.set_ylabel('residual(truth) - residual(estimate)')
        ax.set_title('A2. WHERE the truth loses, by channel (>0 = truth penalised here)',
                     fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # ---- C: per-row residual against depth -- WHERE the truth loses -------------------
    ax = axes[0, 2]
    for name, p in packs.items():
        o = np.argsort(p['pose'][:, 0])
        ax.plot(p['pose'][o, 0], p['rowwise'][o], '-', lw=1.2, color=colors[name],
                label=f'{name} (mean {p["rowwise"].mean():.2f})')
    ax.set_xlabel('insertion depth x (mm)')
    ax.set_ylabel('12-D residual (mm-eq)')
    ax.set_title('C. per-row match quality vs DEPTH -- which rows reject the truth?',
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ---- D: channel decomposition ----------------------------------------------------
    ax = axes[1, 0]
    names = [n for n, _, _ in DOFS]
    xpos = np.arange(len(names))
    for i, (name, p) in enumerate(packs.items()):
        vals = [np.abs(p['res'][:, j]).mean() for _, j, _ in DOFS]
        ax.bar(xpos + (i - 0.5) * 0.38, vals, 0.38, color=colors[name], label=name)
    for k, (_, _, c) in enumerate(DOFS):         # colour-code the block each DOF belongs to
        ax.get_xticklabels()
    ax.set_xticks(xpos, names, fontsize=7)
    for lbl, (_, _, c) in zip(ax.get_xticklabels(), DOFS):
        lbl.set_color(c)
    for b in (2.5, 5.5, 8.5):                    # block boundaries: trans | rot | force | torque
        ax.axvline(b, color='0.7', lw=0.8, ls=':')
    ax.set_ylabel('mean |residual| per DOF (metric units)')
    ax.set_title('D. PER-DOF residual, all 12 (label colour = block; equal bars = no '
                 'objection)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    # ---- E: neighbour votes -- what the rows actually look like -----------------------
    ax = axes[1, 1]
    jz = DIMS.index(dims[0])
    jp = DIMS.index(dims[-1]) if len(dims) > 1 else 0
    for name, p in packs.items():
        nb = manifold_pose(est, p['nn'][:, 0])
        ax.plot(nb[:, jp], nb[:, jz], '.', ms=5, color=colors[name], alpha=0.6,
                label=f'matched map rows @ {name}')
    ax.set_xlabel(f'{dims[-1] if len(dims) > 1 else "x_mm"} of the MATCHED manifold row')
    ax.set_ylabel(f'{dims[0]} of the MATCHED manifold row')
    ax.set_title('E. NEIGHBOUR VOTES: which part of the map the rows resemble', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ---- F: per-channel energy over the candidate grid --------------------------------
    ax = axes[1, 2]
    G, gaxes, shape = grid_over(est, n=grid_n)
    E = channel_energy(est, v, w, G, raw=raw)
    t_est = _theta6(theta_est, est.idx)[est.idx] if theta_est is not None else None
    t_tru = _theta6(theta_true, est.idx)[est.idx] if theta_true is not None else None
    if len(dims) == 1:
        for name, _, col in CHANNELS:
            y = E[name]
            ax.plot(gaxes[0], (y - y.min()) / max(np.ptp(y), 1e-12), lw=1.2, color=col,
                    label=name)
        y = E['total']
        ax.plot(gaxes[0], (y - y.min()) / max(np.ptp(y), 1e-12), 'k-', lw=2.5,
                label='TOTAL (what is minimised)')
        if t_tru is not None:
            ax.axvline(t_tru[0], color='tab:green', ls='--', lw=2, label='truth')
        if t_est is not None:
            ax.axvline(t_est[0], color='tab:orange', ls=':', lw=2, label='estimate')
        ax.set_xlabel(f'candidate {dims[0]}')
        ax.set_ylabel('normalised energy')
    else:
        Z = E['total'].reshape(shape)
        ext = [gaxes[1][0], gaxes[1][-1], gaxes[0][0], gaxes[0][-1]]
        ax.imshow(Z, origin='lower', aspect='auto', extent=ext, cmap='magma')
        if t_tru is not None:
            ax.plot(t_tru[1], t_tru[0], '*', c='lime', ms=18, mec='k', label='truth')
        if t_est is not None:
            ax.plot(t_est[1], t_est[0], 'x', c='w', ms=13, mew=3, label='estimate')
        ax.set_xlabel(f'candidate {dims[1]}')
        ax.set_ylabel(f'candidate {dims[0]}')
    ax.set_title('F. energy BY CHANNEL -- does any channel prefer the truth?', fontsize=9)
    ax.legend(fontsize=7)

    ox = packs[list(packs)[0]]['pose'][:, 0]
    cover = ((ox.max() - ox.min()) / max(np.ptp(Mp[:, 0]), 1e-9))
    fig.suptitle((title or 'manifold match diagnostics') +
                 f'   |   estimate=orange, truth=green   |   observations span x '
                 f'{ox.min():+.1f}..{ox.max():+.1f} mm = {cover:.0%} of the map depth, '
                 f'{len(v)} contact rows')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    # ---- companion figure: EVERY DOF on its own -------------------------------------
    # The 12 axes do not have to agree, and the block view cannot show it when they do not.
    # Each panel is one dimension's own contribution to the energy over the candidate grid,
    # with the committed estimate and the truth marked, and that dimension's mean residual at
    # both in the subtitle. Read it as a vote count: a dimension whose curve bottoms out at
    # the truth was OUTVOTED by the rest; a flat one carries no information about this
    # correction at all (and a zero-weight dimension is flat by construction).
    dof_path = os.path.splitext(out_path)[0] + '_dof.png'
    fig2, ax2 = plt.subplots(3, 4, figsize=(17, 9), sharex=(len(dims) == 1))
    for k, (name, j, col) in enumerate(DOFS):
        a = ax2.ravel()[k]
        y = np.asarray(E[name], dtype=float)
        flat = float(np.ptp(y)) <= 1e-12 * max(abs(y).max(), 1.0)
        if len(dims) == 1:
            a.plot(gaxes[0], y, color=col, lw=1.6)
            if not flat:
                a.plot(gaxes[0][int(np.argmin(y))], y.min(), 'o', color=col, ms=6)
            if t_tru is not None:
                a.axvline(t_tru[0], color='tab:green', ls='--', lw=1.6)
            if t_est is not None:
                a.axvline(t_est[0], color='tab:orange', ls=':', lw=1.6)
            a.set_xlabel(dims[0], fontsize=7)
            head = 'flat: NO information' if flat else \
                f'min @ {gaxes[0][int(np.argmin(y))]:+.2f}'
        else:
            a.imshow(y.reshape(shape), origin='lower', aspect='auto',
                     extent=[gaxes[1][0], gaxes[1][-1], gaxes[0][0], gaxes[0][-1]],
                     cmap='magma')
            if t_tru is not None:
                a.plot(t_tru[1], t_tru[0], '*', c='lime', ms=12, mec='k')
            if t_est is not None:
                a.plot(t_est[1], t_est[0], 'x', c='w', ms=9, mew=2)
            head = 'flat: NO information' if flat else 'min marked'
        bits = '   '.join(f'{nm[:3]} {np.abs(pk["res"][:, j]).mean():.3f}'
                          for nm, pk in packs.items())
        a.set_title(f'{name}:  {head}' + chr(10) + bits, fontsize=8,
                    color=(col if not flat else '0.5'))
        a.tick_params(labelsize=6)
        a.grid(alpha=0.25)
    fig2.suptitle((title or 'per-DOF energy') +
                  '   |   green dashed = truth, orange dotted = committed estimate   |   '
                  'subtitle = mean |residual| for that DOF at each')
    fig2.tight_layout()
    fig2.savefig(dof_path, dpi=110)
    plt.close(fig2)
    # LIVE mirrors: one fixed path each, atomically replaced, so an image viewer left open
    # follows the run instead of chasing per-attempt filenames.
    if live_dir:
        os.makedirs(live_dir, exist_ok=True)
        for src, name in ((out_path, 'estimator_eval_match_live.png'),
                          (dof_path, 'estimator_eval_match_dof_live.png')):
            dst = os.path.join(live_dir, name)
            tmp = dst + '.tmp'
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
    return out_path


def channel_argmins(est, vec6, w6, grid_n=41):
    """{channel: correction at that channel's own minimum} -- the numeric form of panel F,
    cheap enough to log on every attempt. If the channels disagree, the total is a
    compromise nobody voted for."""
    G, gaxes, _ = grid_over(est, n=grid_n)
    E = channel_energy(est, vec6, w6, G)
    return {name: {d: float(G[int(np.argmin(vals)), j])
                   for d, j in zip(est.estimate_dims, est.idx)}
            for name, vals in E.items()}
