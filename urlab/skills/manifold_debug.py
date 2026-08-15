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


def _theta6(theta, idx):
    """A 6-vector correction from either a dict {dim: value} or a sequence over `idx`."""
    t6 = np.zeros(6)
    if isinstance(theta, dict):
        for d, v in theta.items():
            t6[DIMS.index(d)] = float(v)
    else:
        t6[list(idx)] = np.asarray(theta, dtype=float).ravel()
    return t6


def match(est, vec6, w6, theta):
    """Apply a correction and match to the manifold, exactly as the energy does.

    Returns (pts12, tgt12, resid12, pose6) -- the transformed observations in metric space, the
    soft-kNN interpolated manifold target, the signed per-row residual, and the transformed
    observation poses in PHYSICAL units (mm / deg) for plotting."""
    t6 = _theta6(theta, est.idx)
    C = mats_from_vec6(np.asarray(vec6, dtype=float)) @ mats_from_vec6(t6)
    pose6 = vec6_from_mats(C)
    pts = scaled12(pose6, np.asarray(w6, dtype=float), est.s_rot,
                   getattr(est, 'dim_w', None))
    k = max(int(est.interp_neighbors), 1)
    dist, nn = est.tree.query(pts, k=k, workers=-1)
    if dist.ndim == 1:
        dist, nn = dist[:, None], nn[:, None]
    if k > 1:
        bw = np.exp(-(dist - dist[:, :1]) / est.interp_tau)
        bw /= bw.sum(axis=1, keepdims=True)
        tgt = np.einsum('mk,mkd->md', bw, est.M12[nn])
    else:
        tgt = est.M12[nn[:, 0]]
    return pts, tgt, tgt - pts, pose6, nn


def channel_energy(est, vec6, w6, grid6):
    """Per-channel mean residual over a grid of candidate corrections.

    Returns {channel: E} plus 'total' -- the same quantity the estimator minimises, decomposed.
    A channel whose minimum sits at the truth while the total's does not is a WEIGHTING bug; if
    every channel agrees on the wrong answer, the map or the calibration is the problem."""
    Y = mats_from_vec6(np.asarray(vec6, dtype=float))
    w = np.asarray(w6, dtype=float)
    out = {name: np.empty(len(grid6)) for name, _, _ in CHANNELS}
    out['total'] = np.empty(len(grid6))
    k = max(int(est.interp_neighbors), 1)
    # chunk over candidates -- the gather is (candidates x rows, k, 12) and must not blow up
    per = max(int(128e6 // max(len(vec6) * k * 12 * 8, 1)), 1)
    for lo in range(0, len(grid6), per):
        hi = min(lo + per, len(grid6))
        C = np.einsum('nij,kjl->knil', Y, mats_from_vec6(grid6[lo:hi]))
        pts = scaled12(vec6_from_mats(C), w, est.s_rot,
                       getattr(est, 'dim_w', None)).reshape(-1, 12)
        dist, nn = est.tree.query(pts, k=k, workers=-1)
        if dist.ndim == 1:
            dist, nn = dist[:, None], nn[:, None]
        if k > 1:
            bw = np.exp(-(dist - dist[:, :1]) / est.interp_tau)
            bw /= bw.sum(axis=1, keepdims=True)
            tgt = np.einsum('mk,mkd->md', bw, est.M12[nn])
        else:
            tgt = est.M12[nn[:, 0]]
        r = (tgt - pts).reshape(hi - lo, len(vec6), 12)
        out['total'][lo:hi] = np.linalg.norm(r, axis=2).mean(axis=1)
        for name, sl, _ in CHANNELS:
            out[name][lo:hi] = np.linalg.norm(r[:, :, sl], axis=2).mean(axis=1)
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
            grid_n=41):
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
    if len(v) > max_rows:                       # bound the cost; the shape is unaffected
        sel = np.linspace(0, len(v) - 1, max_rows).astype(int)
        v, w = v[sel], w[sel]
    dims = list(est.estimate_dims)
    packs = {}
    for name, th in (('estimate', theta_est), ('truth', theta_true)):
        if th is None:
            continue
        pts, tgt, res, pose6, nn = match(est, v, w, th)
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
    names = [n for n, _, _ in CHANNELS]
    xpos = np.arange(len(names))
    for i, (name, p) in enumerate(packs.items()):
        vals = [np.linalg.norm(p['res'][:, sl], axis=1).mean() for _, sl, _ in CHANNELS]
        ax.bar(xpos + (i - 0.5) * 0.38, vals, 0.38, color=colors[name], label=name)
    ax.set_xticks(xpos, names, fontsize=8)
    ax.set_ylabel('mean residual (mm-eq)')
    ax.set_title('D. WHICH CHANNEL rejects the truth (equal bars = no channel objects)',
                 fontsize=9)
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
    E = channel_energy(est, v, w, G)
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
