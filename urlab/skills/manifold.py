"""Contact-manifold pose estimation -- localise a held part from touch, against recorded data.

The CONTACT MANIFOLD (built by analysis/contact_manifold.py from uncertain_sampling runs) pairs
"connector pose w.r.t. its target" with "wrench in the connector frame" over many recorded
insertions. If the robot's BELIEF of where the held connector sits is wrong, the observations it
collects while inserting land OFF the manifold by exactly that (rigid) belief error -- so aligning
them back onto the manifold estimates the error. This module does that alignment:

  * a common, mm-equivalent 12-D space:  [x, y, z (mm) | r, p, y (deg x s_rot) |
    unit(f) x s_force | unit(tau) x s_torque]   -- weights decide what "nearest" means;
  * multi-start ICP: nearest-neighbour matching across ALL 12 dims, the correction updated in the
    chosen `estimate_dims` only (right-multiplied, i.e. in the part's own frame);
  * residual-gated RANSAC over the finals: a wrong local minimum can capture the LARGER cluster of
    guesses, but its alignment residual stays visibly worse, so residual breaks the vote-count tie.

This is the SAME algorithm as analysis/manifold_icp_validation.py (which validates it offline
against known offsets); this copy is numpy/scipy-only so the robot apps can run it without the
pandas/seaborn analysis stack. If you change the algorithm, change both.

The returned correction T_corr satisfies: believed_pose @ T_corr ~= true_pose. A caller tracking
the in-hand estimate E = T_fingertip_connector therefore updates it as E <- E @ T_corr.
"""

import csv
import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .. import log as urlog

log = urlog.get('manifold')

DIMS = ['x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg']
POSE_COLS = [f'connector_target_{d}' for d in DIMS]
FORCE_COLS = [f'wrench_connector_{a}' for a in ('fx', 'fy', 'fz')]
TORQUE_COLS = [f'wrench_connector_{a}' for a in ('tx', 'ty', 'tz')]


def unit_rows(a, scale):
    """Rows normalized to unit length then scaled; zero rows (no contact) stay zero."""
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a, axis=-1)
    out = np.zeros_like(a)
    nz = n > 1e-9
    out[nz] = a[nz] / n[nz, None] * scale
    return out


def mats_from_vec6(v):
    """(..., 6) [x,y,z (mm), roll,pitch,yaw (deg, extrinsic XYZ)] -> (..., 4, 4), translation in mm."""
    v = np.asarray(v, dtype=float)
    shp = v.shape[:-1]
    T = np.zeros(shp + (4, 4))
    T[..., :3, :3] = (Rotation.from_euler('xyz', v.reshape(-1, 6)[:, 3:], degrees=True)
                      .as_matrix().reshape(shp + (3, 3)))
    T[..., :3, 3] = v[..., :3]
    T[..., 3, 3] = 1.0
    return T


def vec6_from_mats(T):
    """Inverse of mats_from_vec6: (..., 4, 4) -> (..., 6) in mm / deg."""
    T = np.asarray(T, dtype=float)
    shp = T.shape[:-2]
    eul = (Rotation.from_matrix(T.reshape(-1, 4, 4)[:, :3, :3])
           .as_euler('xyz', degrees=True).reshape(shp + (3,)))
    return np.concatenate([T[..., :3, 3], eul], axis=-1)


def scaled12(vec6, w6, s_rot):
    """The common 12-D point: [t (mm) | rot (deg x s_rot) | scaled unit f | scaled unit tau]."""
    v = np.asarray(vec6, dtype=float)
    w = np.broadcast_to(np.asarray(w6, dtype=float), v.shape[:-1] + (6,))
    return np.concatenate([v[..., :3], v[..., 3:] * s_rot, w], axis=-1)


class ManifoldEstimator:
    """The manifold KD-tree plus the multi-start ICP / RANSAC solver, wired from a config section.

    Reads `estimation:` (see configs/cable_pick_estimate_assemble.yaml for the schema)."""

    def __init__(self, cfg_section):
        c = cfg_section or {}
        self.s_rot = float(c.get('scaling_constant_deg_to_mm', 1.0))
        self.s_force = float(c.get('scaling_constant_unit_force_to_mm', 0.1))
        self.s_torque = float(c.get('scaling_constant_unit_torque_to_mm', 0.1))
        self.min_force_n = c.get('min_force_n')
        self.estimate_dims = list(c.get('estimate_dims', ['x_mm', 'z_mm', 'pitch_deg']))
        bad = [d for d in self.estimate_dims if d not in DIMS]
        if bad:
            raise ValueError(f'estimation.estimate_dims {bad} not in {DIMS}')
        self.idx = [DIMS.index(d) for d in self.estimate_dims]
        self.iterations = int(c.get('icp_iterations', 10))
        self.guesses = int(c.get('num_initial_guesses', 100))
        self.init_range = dict(c.get('init_guess_range', {}) or {})
        self.step_gain = float(c.get('step_gain', 1.0))
        self.ransac_iters = int(c.get('ransac_iters', 200))
        self.ransac_tol = float(c.get('ransac_tol', 1.0))
        self.residual_gate = c.get('residual_gate', 1.5)
        self.min_observations = int(c.get('min_observations', 20))
        seed = int(c.get('random_seed', 0))
        self.rng = np.random.default_rng(seed if seed > 0 else None)

        path = c.get('manifold_csv')
        if not path:
            raise ValueError('estimation.manifold_csv is required')
        self.M12 = self._load_manifold(path)
        self.tree = cKDTree(self.M12)
        log.info('Contact manifold: %d points from %s', len(self.M12), path)

    # ------------------------------------------------------------------ data
    def _load_manifold(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'estimation.manifold_csv not found: {path}')
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            missing = [c for c in POSE_COLS + FORCE_COLS + TORQUE_COLS if c not in cols]
            if missing:
                raise ValueError(f'{path}: missing column(s), first: {missing[0]} '
                                 '(pre-mm logs must be converted first)')
            v6, f, tau = [], [], []
            for rec in reader:
                try:
                    v6.append([float(rec[c]) for c in POSE_COLS])
                    f.append([float(rec[c]) for c in FORCE_COLS])
                    tau.append([float(rec[c]) for c in TORQUE_COLS])
                except (TypeError, ValueError):
                    continue                       # unparseable row -- skip, not fatal
        v6, f, tau = np.asarray(v6), np.asarray(f), np.asarray(tau)
        if self.min_force_n is not None:
            keep = np.linalg.norm(f, axis=1) >= float(self.min_force_n)
            v6, f, tau = v6[keep], f[keep], tau[keep]
        if len(v6) < 10:
            raise ValueError(f'{path}: only {len(v6)} usable manifold rows')
        w6 = np.hstack([unit_rows(f, self.s_force), unit_rows(tau, self.s_torque)])
        return scaled12(v6, w6, self.s_rot)

    def prepare_observations(self, vec6, f_raw, tau_raw):
        """Filter (min force) + normalize raw observations -> (vec6, w6) ready for estimate()."""
        vec6 = np.asarray(vec6, dtype=float)
        f_raw, tau_raw = np.asarray(f_raw, dtype=float), np.asarray(tau_raw, dtype=float)
        if self.min_force_n is not None:
            keep = np.linalg.norm(f_raw, axis=1) >= float(self.min_force_n)
            vec6, f_raw, tau_raw = vec6[keep], f_raw[keep], tau_raw[keep]
        w6 = np.hstack([unit_rows(f_raw, self.s_force), unit_rows(tau_raw, self.s_torque)])
        return vec6, w6

    # ------------------------------------------------------------------ solver
    def estimate(self, vec6, w6):
        """The belief correction from one set of observations (believed poses + their wrench).

        `vec6` (N,6): BELIEVED connector-wrt-target poses [mm, deg]. `w6` (N,6): the scaled unit
        wrench rows (from prepare_observations). Returns (T_corr, info) -- believed @ T_corr ~= true
        -- or (None, reason) when there is not enough data. info carries theta_corr (the correction
        in physical units on estimate_dims), inlier count, and the final residual."""
        if len(vec6) < self.min_observations:
            return None, f'only {len(vec6)} observations (< min_observations {self.min_observations})'

        Y = mats_from_vec6(vec6)
        G, K, idx = self.guesses, self.iterations, self.idx

        g6 = np.zeros((G, 6))
        for d, j in zip(self.estimate_dims, idx):
            r = float(self.init_range.get(d, 5.0))
            g6[1:, j] = self.rng.uniform(-r, r, G - 1)        # guess 0 stays identity
        T_corr = mats_from_vec6(g6)

        free = np.zeros(6, dtype=bool)
        free[idx] = True
        res_hist = np.zeros((G, K))
        theta_hist = np.zeros((G, K + 1, len(idx)))       # correction params per iteration (physical)
        theta_hist[:, 0] = g6[:, idx]
        for k in range(K):
            C = np.einsum('nij,gjk->gnik', Y, T_corr)
            pts = scaled12(vec6_from_mats(C), w6, self.s_rot)
            dist, nn = self.tree.query(pts.reshape(-1, 12), workers=-1)
            dist, nn = dist.reshape(G, -1), nn.reshape(G, -1)
            res_hist[:, k] = dist.mean(axis=1)
            delta12 = (self.M12[nn] - pts).mean(axis=1)
            delta6 = np.zeros((G, 6))
            delta6[:, :3] = delta12[:, :3]
            delta6[:, 3:] = delta12[:, 3:6] / max(self.s_rot, 1e-12)
            delta6[:, ~free] = 0.0
            T_corr = T_corr @ mats_from_vec6(delta6 * self.step_gain)
            theta_hist[:, k + 1] = vec6_from_mats(T_corr)[:, idx]

        # Residual-gated RANSAC over the final corrections (distance in mm-equivalent space).
        corr6 = vec6_from_mats(T_corr)                        # (G, 6) physical
        scale = np.array([self.s_rot if d.endswith('_deg') else 1.0 for d in DIMS])
        finals = corr6[:, idx] * scale[idx]
        gate = self.residual_gate
        keep = (res_hist[:, -1] <= res_hist[:, -1].min() * float(gate) + 1e-12) if gate else \
            np.ones(G, dtype=bool)
        kept_ids = np.flatnonzero(keep)
        best = np.zeros(G, dtype=bool)
        for _ in range(self.ransac_iters):
            cand = finals[kept_ids[self.rng.integers(len(kept_ids))]]
            inl = keep & (np.linalg.norm(finals - cand, axis=1) < self.ransac_tol)
            if inl.sum() > best.sum():
                best = inl
        if not best.any():
            best = keep.copy()
        est = finals[best].mean(axis=0)
        refit = keep & (np.linalg.norm(finals - est, axis=1) < self.ransac_tol)
        if refit.any():
            best = refit
            est = finals[best].mean(axis=0)

        theta = np.zeros(6)
        theta[idx] = est / scale[idx]                         # physical mm / deg, free dims only
        info = {
            'theta_corr': {d: float(theta[j]) for d, j in zip(self.estimate_dims, idx)},
            'inliers': int(best.sum()), 'guesses': G,
            'final_residual': float(res_hist[best, -1].mean()),
            'n_observations': len(vec6),
            # Per-iteration histories (all guesses) for convergence plots: correction params in
            # physical mm/deg on estimate_dims, the mean NN residual, and the RANSAC inlier mask.
            'theta_hist': theta_hist, 'res_hist': res_hist, 'inlier_mask': best,
        }
        return mats_from_vec6(theta), info
