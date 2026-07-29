"""Validate the contact manifold: recover KNOWN pose offsets from validation trials with ICP.

METHOD. Both CSVs (the manifold from analysis/contact_manifold.py and a validation run from
urlab.apps.uncertain_sampling) are reduced to Nx12 point sets in a COMMON, mm-equivalent space:

    [x, y, z (mm, as-is)  |  roll, pitch, yaw (deg x scaling_constant_deg_to_mm)  |
     f_hat x scaling_constant_unit_force_to_mm  |  tau_hat x scaling_constant_unit_torque_to_mm]

Force and torque are first NORMALIZED to unit vectors (direction only -- magnitude depends on how
hard the compliance pressed, direction on the contact geometry), then scaled so a nearest-neighbour
distance mixes pose and wrench with user-controlled weights. Call these M (manifold) and V
(validation).

PERTURBATION. Per validation trial, y_true = the trial's rows. The user names the perturbed
DIMENSIONS; the trial's initial pose T_t_ctrue (first row) defines the offset
T_ctrue_coffset = inv(T_t_ctrue) @ T_zeroed, where T_zeroed equals the initial pose with the
perturbed dims set to ZERO (unperturbed dims unchanged). theta := the initial pose's perturbed-dim
values -- exactly what the offset hides. Every observation is then perturbed the same way,
[y]_i = [y_true]_i @ T_ctrue_coffset, simulating a run whose belief of the connector's initial pose
was wrong by theta. The wrench columns are left as recorded: the physical contact does not change
because our ESTIMATE of the pose is off, so the wrench is the signature the offset cannot erase.

ICP. Correspondences are found across ALL 12 dimensions (KD-tree nearest neighbour in the scaled
space); the correction transform is updated in the PERTURBED dims only (right-multiplied, like the
offset itself), for a fixed number of iterations. Many random initial guesses are run in parallel
and the final estimates are aggregated with RANSAC (consensus = mean of the largest inlier set).
theta_est = perturbed-dim values of the corrected initial pose; perfect recovery -> theta_est ==
theta_true, so the plotted per-dim ERROR converges to the dashed zero line.

OUTPUT. A timestamped folder under `output_root`: one figure per trial (per-dim error lines + the
NN residual, every guess at low opacity, RANSAC consensus bold), a CSV of per-trial offsets and
final estimation errors (signed and absolute), and config.json for provenance.

Edit CONFIG below and run:  python analysis/manifold_icp_validation.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.lines import Line2D         # noqa: E402
import seaborn as sns                       # noqa: E402

# ---------------------------------------------------------------------------- user configuration
CONFIG = {
    # ---- what & where -------------------------------------------------------------------
    'cable': 'banana',                       # connector type (names the results CSV)
    'manifold_csv': r'data/uncertain_assembly_sampling/banana/banana_connector_contact_manifold.csv',
    'validation_csv': r'data/uncertain_assembly_sampling/banana/validation_log.csv',
    'output_root': 'analysis',               # a timestamped run folder is created under this

    # ---- normalization & scaling (the common Nx12 space) --------------------------------
    # Translation stays in mm. Rotation deg -> mm-equivalent. Force/torque are unit vectors
    # scaled to mm-equivalent. These weights decide what "nearest" means in the 12-D match.
    'scaling_constant_deg_to_mm': 1.0,
    'scaling_constant_unit_force_to_mm': 0.1,
    'scaling_constant_unit_torque_to_mm': 0.1,
    'min_force_n': None,                     # drop rows with |f| < this N BEFORE normalizing
                                             # (None keeps all; zero-force rows keep a zero f_hat)

    # ---- perturbation -------------------------------------------------------------------
    # Dims of the initial connector pose to ZERO (the hidden offset ICP must recover).
    # Any subset of: x_mm y_mm z_mm roll_deg pitch_deg yaw_deg
    'perturb_dims': ['z_mm', 'pitch_deg'],

    # ---- ICP / RANSAC -------------------------------------------------------------------
    'icp_iterations': 10,
    'num_initial_guesses': 100,              # guess 0 is always the identity (no correction)
    'init_guess_range': {                    # uniform +/- range for the initial correction guesses
        'x_mm': 5.0, 'y_mm': 5.0, 'z_mm': 5.0,
        'roll_deg': 15.0, 'pitch_deg': 15.0, 'yaw_deg': 15.0,
    },
    'step_gain': 1.0,                        # fraction of the mean NN delta applied per iteration
    'ransac_iters': 200,
    'ransac_tol': 1.0,                       # inlier radius around a candidate, mm-equivalent
    # Before RANSAC votes, drop guesses whose FINAL residual exceeds gate x the best guess's --
    # a wrong local minimum can attract MANY guesses (a big cluster), but its alignment stays
    # visibly worse, so residual is the tie-breaker vote-counting alone does not have.
    'residual_gate': 1.5,                    # None disables the gate
    'random_seed': 0,                        # 0 = nondeterministic

    # ---- run control --------------------------------------------------------------------
    'trials': None,                          # None = all trials; else e.g. [1, 2, 5]
    'max_points_per_trial': 250,             # stride-subsample big trials (speed); None = all
    'dpi': 110,
}

# ---------------------------------------------------------------------------- constants
DIMS = ['x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg']
POSE_COLS = [f'connector_target_{d}' for d in DIMS]
FORCE_COLS = [f'wrench_connector_{a}' for a in ('fx', 'fy', 'fz')]
TORQUE_COLS = [f'wrench_connector_{a}' for a in ('tx', 'ty', 'tz')]

# Two identities only -- the guess cloud and the consensus -- so a CVD-safe blue/orange pair,
# with LINEWEIGHT as the secondary encoding (consensus is the only bold line). Zero line neutral.
C_GUESS = '#4C72B0'
C_CONSENSUS = '#DD8452'
C_ZERO = '#888888'


# ---------------------------------------------------------------------------- small helpers
def _unit_rows(a, scale):
    """Rows normalized to unit length then scaled; zero rows (no contact) stay zero."""
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a, axis=-1)
    out = np.zeros_like(a)
    nz = n > 1e-9
    out[nz] = a[nz] / n[nz, None] * scale
    return out


def _mats(vec6):
    """(..., 6) [x,y,z (mm), roll,pitch,yaw (deg, extrinsic XYZ -- the repo convention)] -> (..., 4, 4)."""
    v = np.asarray(vec6, dtype=float)
    shp = v.shape[:-1]
    T = np.zeros(shp + (4, 4))
    T[..., :3, :3] = (Rotation.from_euler('xyz', v.reshape(-1, 6)[:, 3:], degrees=True)
                      .as_matrix().reshape(shp + (3, 3)))
    T[..., :3, 3] = v[..., :3]
    T[..., 3, 3] = 1.0
    return T


def _vec6(T):
    """Inverse of _mats: (..., 4, 4) -> (..., 6) in mm / deg."""
    T = np.asarray(T, dtype=float)
    shp = T.shape[:-2]
    eul = (Rotation.from_matrix(T.reshape(-1, 4, 4)[:, :3, :3])
           .as_euler('xyz', degrees=True).reshape(shp + (3,)))
    return np.concatenate([T[..., :3, 3], eul], axis=-1)


def _scaled12(vec6, w6, s_rot):
    """The common 12-D point: [t (mm) | rot (deg x s_rot) | scaled unit f | scaled unit tau]."""
    v = np.asarray(vec6, dtype=float)
    w = np.broadcast_to(w6, v.shape[:-1] + (6,))
    return np.concatenate([v[..., :3], v[..., 3:] * s_rot, w], axis=-1)


def load_scaled(path, cfg, need_trial=False):
    """(vec6 (N,6) physical mm/deg, w6 (N,6) scaled unit wrench, trial ids or None)."""
    if not os.path.isfile(path):
        sys.exit(f'input CSV not found: {path}')
    df = pd.read_csv(path)
    if 'connector_target_x' in df.columns and 'connector_target_x_mm' not in df.columns:
        sys.exit(f'{path}: pre-mm log format (translations in METRES, un-suffixed) -- re-collect, '
                 'or scale x/y/z by 1000 and rename to *_mm first')
    missing = [c for c in POSE_COLS + FORCE_COLS + TORQUE_COLS if c not in df.columns]
    if missing:
        sys.exit(f'{path}: missing column(s), first: {missing[0]}')
    if need_trial and 'trial' not in df.columns:
        sys.exit(f'{path}: no `trial` column -- the validation CSV must be a raw sampler log')
    if cfg['min_force_n'] is not None:
        df = df[np.linalg.norm(df[FORCE_COLS].to_numpy(float), axis=1) >= cfg['min_force_n']]
    vec6 = df[POSE_COLS].to_numpy(float)
    w6 = np.hstack([_unit_rows(df[FORCE_COLS].to_numpy(float), cfg['scaling_constant_unit_force_to_mm']),
                    _unit_rows(df[TORQUE_COLS].to_numpy(float), cfg['scaling_constant_unit_torque_to_mm'])])
    trial = df['trial'].to_numpy(int) if need_trial else None
    return vec6, w6, trial


# ---------------------------------------------------------------------------- the solver
def solve_trial(vec6, w6, tree, M12, cfg, rng):
    """Perturb one trial, then recover the offset with multi-start ICP + RANSAC.

    Returns a dict with theta_true, per-iteration histories (all guesses), the RANSAC consensus
    estimate/errors, the inlier mask, and the number of points used."""
    dims = cfg['perturb_dims']
    idx = [DIMS.index(d) for d in dims]
    s_rot = float(cfg['scaling_constant_deg_to_mm'])

    cap = cfg['max_points_per_trial']
    if cap and len(vec6) > cap:                       # stride keeps the trajectory shape
        stride = int(np.ceil(len(vec6) / cap))
        vec6, w6 = vec6[::stride], w6[::stride]

    # The hidden offset: zero the perturbed dims of the INITIAL pose, keep the rest.
    p0 = vec6[0].copy()
    theta_true = p0[idx].copy()
    zeroed = p0.copy()
    zeroed[idx] = 0.0
    T_off = np.linalg.inv(_mats(p0)) @ _mats(zeroed)      # T_ctrue_coffset
    Y = _mats(vec6) @ T_off                               # every observation, perturbed alike

    # Multi-start corrections: guess 0 = identity, the rest uniform in the configured range.
    G, K = int(cfg['num_initial_guesses']), int(cfg['icp_iterations'])
    g6 = np.zeros((G, 6))
    for d, j in zip(dims, idx):
        g6[1:, j] = rng.uniform(-float(cfg['init_guess_range'][d]),
                                float(cfg['init_guess_range'][d]), G - 1)
    T_corr = _mats(g6)

    free = np.zeros(6, dtype=bool)
    free[idx] = True
    theta_hist = np.zeros((G, K + 1, len(idx)))
    res_hist = np.zeros((G, K))
    theta_hist[:, 0] = _vec6(Y[0][None] @ T_corr)[:, idx]

    for k in range(K):
        C = np.einsum('nij,gjk->gnik', Y, T_corr)                 # corrected poses, all guesses
        v6 = _vec6(C)
        pts = _scaled12(v6, w6, s_rot)                            # (G, N, 12)
        dist, nn = tree.query(pts.reshape(-1, 12), workers=-1)    # match across ALL dimensions
        dist, nn = dist.reshape(G, -1), nn.reshape(G, -1)
        res_hist[:, k] = dist.mean(axis=1)
        delta12 = (M12[nn] - pts).mean(axis=1)                    # mean NN delta, per guess
        delta6 = np.zeros((G, 6))
        delta6[:, :3] = delta12[:, :3]
        delta6[:, 3:] = delta12[:, 3:6] / max(s_rot, 1e-12)       # back to physical deg
        delta6[:, ~free] = 0.0                                    # update the perturbed dims ONLY
        T_corr = T_corr @ _mats(delta6 * float(cfg['step_gain']))
        theta_hist[:, k + 1] = _vec6(Y[0][None] @ T_corr)[:, idx]

    # RANSAC over the final estimates (distance in the mm-equivalent theta space). Votes are
    # counted only among RESIDUAL-GATED guesses: a wrong local minimum can capture the LARGER
    # cluster, but its final alignment stays visibly worse, so residual breaks that tie.
    scale = np.array([s_rot if d.endswith('_deg') else 1.0 for d in dims])
    finals = theta_hist[:, -1] * scale
    gate = cfg.get('residual_gate')
    keep = (res_hist[:, -1] <= res_hist[:, -1].min() * float(gate) + 1e-12) if gate else \
        np.ones(G, dtype=bool)
    kept_ids = np.flatnonzero(keep)
    best = np.zeros(G, dtype=bool)
    for _ in range(int(cfg['ransac_iters'])):
        cand = finals[kept_ids[rng.integers(len(kept_ids))]]
        inl = keep & (np.linalg.norm(finals - cand, axis=1) < float(cfg['ransac_tol']))
        if inl.sum() > best.sum():
            best = inl
    if not best.any():
        best = keep.copy()                                        # degenerate: keep the gated set
    est = finals[best].mean(axis=0)
    refit = keep & (np.linalg.norm(finals - est, axis=1) < float(cfg['ransac_tol']))   # one refit
    if refit.any():
        best = refit
        est = finals[best].mean(axis=0)
    theta_est = est / scale

    return {
        'dims': dims, 'n_points': len(vec6), 'theta_true': theta_true, 'theta_est': theta_est,
        'error': theta_est - theta_true, 'inliers': best,
        'err_hist': theta_hist - theta_true[None, None, :], 'res_hist': res_hist,
        'final_residual': float(res_hist[best, -1].mean()),
    }


# ---------------------------------------------------------------------------- plotting
def plot_trial(trial, r, out_png, cfg):
    """Per-dim theta error + NN residual vs ICP iteration: every guess faint, consensus bold."""
    sns.set_theme(style='whitegrid', context='notebook')
    dims, inl = r['dims'], r['inliers']
    D = len(dims)
    fig, axes = plt.subplots(D + 1, 1, figsize=(9.0, 2.6 * (D + 1)), sharex=True)
    axes = np.atleast_1d(axes)
    it = np.arange(r['err_hist'].shape[1])

    for j, (ax, dim) in enumerate(zip(axes[:-1], dims)):
        unit = 'deg' if dim.endswith('_deg') else 'mm'
        ax.axhline(0.0, ls='--', lw=1.0, color=C_ZERO, zorder=1)
        for g in range(r['err_hist'].shape[0]):                   # every guess, low opacity
            ax.plot(it, r['err_hist'][g, :, j], color=C_GUESS, alpha=0.07, lw=1.0, zorder=2)
        ax.plot(it, r['err_hist'][inl, :, j].mean(axis=0), color=C_CONSENSUS, lw=2.4, zorder=3)
        lim = max(float(np.abs(r['err_hist'][..., j]).max()), 1e-3) * 1.05
        ax.set_ylim(-lim, lim)                                    # centred on zero
        ax.set_ylabel(f'{dim} error [{unit}]')
        ax.set_title(f'theta[{dim}]:  true {r["theta_true"][j]:+.3f} {unit}   '
                     f'est {r["theta_est"][j]:+.3f}   err {r["error"][j]:+.3f}',
                     fontsize=10, loc='left')

    axes[0].legend(handles=[
        Line2D([], [], color=C_GUESS, alpha=0.6, lw=1.0, label=f'guesses (n={len(inl)})'),
        Line2D([], [], color=C_CONSENSUS, lw=2.4, label=f'RANSAC consensus ({int(inl.sum())} inliers)'),
    ], loc='upper right', fontsize=8, framealpha=0.9)

    ax = axes[-1]
    it_r = np.arange(1, r['res_hist'].shape[1] + 1)
    for g in range(r['res_hist'].shape[0]):
        ax.plot(it_r, r['res_hist'][g], color=C_GUESS, alpha=0.07, lw=1.0, zorder=2)
    ax.plot(it_r, r['res_hist'][inl].mean(axis=0), color=C_CONSENSUS, lw=2.4, zorder=3)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel('mean NN residual [mm-eq]')
    ax.set_xlabel('ICP iteration')

    fig.suptitle(f'trial {trial} -- ICP recovery of {", ".join(dims)}', y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=cfg['dpi'])
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main(cfg=CONFIG):
    bad = [d for d in cfg['perturb_dims'] if d not in DIMS]
    if bad:
        sys.exit(f'perturb_dims {bad} not in {DIMS}')
    seed = int(cfg.get('random_seed', 0))
    rng = np.random.default_rng(seed if seed > 0 else None)

    out_dir = os.path.join(cfg['output_root'],
                           f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_icp_{cfg["cable"]}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as fh:
        json.dump(cfg, fh, indent=2, default=str)

    Mv6, Mw6, _ = load_scaled(cfg['manifold_csv'], cfg)
    M12 = _scaled12(Mv6, Mw6, float(cfg['scaling_constant_deg_to_mm']))
    if len(M12) < 10:
        sys.exit(f'manifold has only {len(M12)} rows -- not enough to match against')
    tree = cKDTree(M12)
    print(f'manifold: {len(M12)} points ({cfg["manifold_csv"]})')

    Vv6, Vw6, Vtrial = load_scaled(cfg['validation_csv'], cfg, need_trial=True)
    trials = sorted(set(Vtrial.tolist()))
    if cfg['trials'] is not None:
        trials = [t for t in trials if t in set(cfg['trials'])]
    print(f'validation: {len(Vv6)} rows, {len(trials)} trial(s) ({cfg["validation_csv"]})')
    print(f'perturbing {cfg["perturb_dims"]}; {cfg["num_initial_guesses"]} guesses x '
          f'{cfg["icp_iterations"]} iterations, RANSAC tol {cfg["ransac_tol"]}\n')

    rows = []
    for t in trials:
        sel = Vtrial == t
        if sel.sum() < 5:
            print(f'trial {t}: only {int(sel.sum())} rows -- skipped')
            continue
        r = solve_trial(Vv6[sel], Vw6[sel], tree, M12, cfg, rng)
        plot_trial(t, r, os.path.join(out_dir, f'trial_{t:03d}_errors.png'), cfg)

        row = {'trial': t, 'n_points': r['n_points'],
               'ransac_inliers': int(r['inliers'].sum()),
               'final_residual_mm_eq': r['final_residual']}
        for j, d in enumerate(r['dims']):
            row[f'theta_true_{d}'] = r['theta_true'][j]
            row[f'theta_est_{d}'] = r['theta_est'][j]
            row[f'error_{d}'] = r['error'][j]
            row[f'abs_error_{d}'] = abs(r['error'][j])
        rows.append(row)
        err = ', '.join(f'{d}: {e:+.3f}' for d, e in zip(r['dims'], r['error']))
        print(f'trial {t}: inliers {int(r["inliers"].sum())}/{len(r["inliers"])}  error [{err}]')

    if not rows:
        sys.exit('no trials processed')
    df = pd.DataFrame(rows)
    csv_out = os.path.join(out_dir, f'{cfg["cable"]}_icp_validation_results.csv')
    df.to_csv(csv_out, index=False)

    print(f'\nmean |error| per dim: ' + ', '.join(
        f'{d}: {df[f"abs_error_{d}"].mean():.3f}' for d in cfg['perturb_dims']))
    print(f'{len(rows)} trial(s) -> {csv_out}')
    print(f'figures + config.json in {out_dir}')
    return out_dir


if __name__ == '__main__':
    main()
