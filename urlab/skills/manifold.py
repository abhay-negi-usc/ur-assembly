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
  * RECENCY weighting: the rigid-belief-error assumption can BREAK mid-attempt (the connector
    slips between the finger pads), so observations decay exponentially with age -- the newest,
    which describe the CURRENT in-hand pose, dominate the NN mean and the residual
    (recency_half_life_frac, a fraction of the observation window);
  * OPTIONAL soft correspondence (interp_neighbors > 1): the manifold is a FINITE sample of a
    continuous surface, so exact-NN matching can LATCH onto the single closest sample and
    quantise the correction by the local sample spacing; instead, blend the k nearest manifold
    points weighted by closeness so the match target INTERPOLATES between close-enough samples;
  * residual-gated RANSAC over the finals: a wrong local minimum can capture the LARGER cluster of
    guesses, but its alignment residual stays visibly worse, so residual breaks the vote-count tie;
  * OPTIONAL residual-softmax aggregation (aggregator: softmax): instead of the consensus vote,
    average ALL finals weighted exp(-(r - r_min)/(softmax_temp x r_min)) -- the 2026-08 offline
    ablation's winner: the residual VALUE carries more information than cluster mass, and a bad
    cluster can then never outvote a few well-aligned starts.

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
        # WRENCH REPRESENTATION (2026-08 offline wrench-lab winner): 'unit' = direction
        # only (the original); 'rawcap' = direction x saturated magnitude -- f/10 N capped
        # at 30 N, tau/1 Nm capped at 3 Nm -- so a 30 N wedge press carries a 3x larger
        # feature vector than a 10 N touch. Magnitude IS informative once the manifold
        # holds production-process data (rawcap: 0.28 mm median vs unit's 0.34 on the
        # augmented map; harmless-to-mildly-positive on the wiggle-only map). Applied
        # IDENTICALLY to the manifold rows and the observations; interp_tau self-adapts
        # because it is derived from the manifold's own spacing AFTER representation.
        rep = str(c.get('wrench_representation', 'unit')).strip().lower()
        if rep not in ('unit', 'rawcap'):
            raise ValueError(f"estimation.wrench_representation {rep!r} must be "
                             "'unit' or 'rawcap'")
        self.wrench_representation = rep               # bad values fail HERE, pre-motion
        self.rawcap_force_ref_n = float(c.get('rawcap_force_ref_n', 10.0))
        self.rawcap_torque_ref_nm = float(c.get('rawcap_torque_ref_nm', 1.0))
        self.rawcap_cap = float(c.get('rawcap_cap', 3.0))
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
        # AGGREGATOR over the multi-start finals: 'ransac' = residual-gated consensus vote (the
        # original); 'softmax' = weighted mean of ALL finals, weight exp(-(r - r_min) /
        # (softmax_temp x r_min)) -- the 2026-08 offline ablation's winner (residual outranks
        # cluster mass; ignores residual_gate / ransac_*). Temperature is RELATIVE to the best
        # residual: with 0.15, a final 15% worse than the best carries ~0.37x its weight.
        agg = str(c.get('aggregator', 'ransac')).strip().lower()
        if agg not in ('ransac', 'softmax'):
            raise ValueError(f"estimation.aggregator {agg!r} must be 'ransac' or 'softmax'")
        self.aggregator = agg                          # bad values fail HERE, pre-motion
        self.softmax_temp = float(c.get('softmax_temp', 0.15))
        # residual_gate: a float, or DISABLED via null/~ in yaml. The strings 'None'/'none'/'null'
        # also disable it -- yaml parses a bare `None` as the STRING "None" (only `null`/`~` are
        # yaml null), and float('None') would otherwise blow up MID-RUN, after the robot moved.
        gate = c.get('residual_gate', 1.5)
        if isinstance(gate, str) and gate.strip().lower() in ('none', 'null', '~', ''):
            gate = None
        self.residual_gate = None if not gate else float(gate)   # bad values fail HERE, pre-motion
        # RECENCY weighting: the in-hand pose can DRIFT during an attempt (pad slip), so newer
        # observations describe the CURRENT pose better than older ones. Exponential decay with
        # age; half-life as a FRACTION of the observation window (0.5 = moderate: the oldest
        # sample carries 0.25x the newest's weight). None/null/'None'/0 disables (uniform).
        hl = c.get('recency_half_life_frac', 0.5)
        if isinstance(hl, str) and hl.strip().lower() in ('none', 'null', '~', ''):
            hl = None
        self.recency_half_life_frac = None if not hl else float(hl)
        # INTERPOLATED correspondence (OPTIONAL, off by default): the manifold is a finite SAMPLE,
        # so exact-NN matching latches onto the single closest point. With interp_neighbors > 1
        # the match target is instead a blend of the k nearest manifold points, weighted
        # exp(-(d - d_nearest) / tau) with tau = interp_softness x the manifold's median point
        # spacing -- a point ~tau further than the nearest carries ~0.37x its weight, ~3 tau
        # carries ~0.05x, so "close enough" is relative to how dense the manifold actually is.
        # 1/None/'None' disables (exact NN, the original behaviour).
        kn = c.get('interp_neighbors', 1)
        if isinstance(kn, str) and kn.strip().lower() in ('none', 'null', '~', ''):
            kn = 1
        self.interp_neighbors = max(1, int(kn)) if kn else 1
        self.interp_softness = float(c.get('interp_softness', 1.0))
        self.min_observations = int(c.get('min_observations', 20))
        seed = int(c.get('random_seed', 0))
        self.rng = np.random.default_rng(seed if seed > 0 else None)

        path = c.get('manifold_csv')
        if not path:
            raise ValueError('estimation.manifold_csv is required')
        self.M12 = self._load_manifold(path)
        self.tree = cKDTree(self.M12)
        log.info('Contact manifold: %d points from %s', len(self.M12), path)
        self.interp_neighbors = min(self.interp_neighbors, len(self.M12))
        self.interp_tau = None
        if self.interp_neighbors > 1:
            spacing = float(np.median(self.tree.query(self.M12, k=2, workers=-1)[0][:, 1]))
            self.interp_tau = max(spacing * self.interp_softness, 1e-9)
            log.info('Interpolated matching: k=%d, tau=%.3f mm-eq (median manifold spacing %.3f)',
                     self.interp_neighbors, self.interp_tau, spacing)

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
        return scaled12(v6, self._wrench6(f, tau), self.s_rot)

    def _wrench6(self, f, tau):
        """The 6 wrench feature columns under the configured representation, scales applied."""
        if self.wrench_representation == 'rawcap':
            fm = np.linalg.norm(np.asarray(f, dtype=float), axis=-1)
            tm = np.linalg.norm(np.asarray(tau, dtype=float), axis=-1)
            wf = unit_rows(f, 1.0) * np.minimum(fm / self.rawcap_force_ref_n,
                                                self.rawcap_cap)[..., None]
            wt = unit_rows(tau, 1.0) * np.minimum(tm / self.rawcap_torque_ref_nm,
                                                  self.rawcap_cap)[..., None]
            return np.hstack([wf * self.s_force, wt * self.s_torque])
        return np.hstack([unit_rows(f, self.s_force), unit_rows(tau, self.s_torque)])

    def prepare_observations(self, vec6, f_raw, tau_raw):
        """Filter (min force) + normalize raw observations -> (vec6, w6) ready for estimate()."""
        vec6 = np.asarray(vec6, dtype=float)
        f_raw, tau_raw = np.asarray(f_raw, dtype=float), np.asarray(tau_raw, dtype=float)
        if self.min_force_n is not None:
            keep = np.linalg.norm(f_raw, axis=1) >= float(self.min_force_n)
            vec6, f_raw, tau_raw = vec6[keep], f_raw[keep], tau_raw[keep]
        return vec6, self._wrench6(f_raw, tau_raw)

    # ------------------------------------------------------------------ solver
    def estimate(self, vec6, w6):
        """The belief correction from one set of observations (believed poses + their wrench).

        `vec6` (N,6): BELIEVED connector-wrt-target poses [mm, deg]. `w6` (N,6): the scaled unit
        wrench rows (from prepare_observations). Returns (T_corr, info) -- believed @ T_corr ~= true
        -- or (None, reason) when there is not enough data. info carries theta_corr (the correction
        in physical units on estimate_dims), inlier count, and the final residual."""
        if len(vec6) < self.min_observations:
            return None, f'only {len(vec6)} observations (< min_observations {self.min_observations})'

        # RECENCY weights: observations are TIME-ORDERED (the caller logs them sequentially), and
        # the in-hand pose may have drifted mid-attempt -- the fit leans toward the newest rows.
        # (min-force filtering mostly drops the free-space START, so the survivors stay ~uniform
        # in time and the sequence index is a fair clock.)
        wts = None
        if self.recency_half_life_frac and len(vec6) > 1:
            age = (len(vec6) - 1 - np.arange(len(vec6))) / (len(vec6) - 1)   # 0 newest .. 1 oldest
            wts = np.power(0.5, age / self.recency_half_life_frac)
            wts /= wts.sum()

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
        kq = self.interp_neighbors
        for k in range(K):
            C = np.einsum('nij,gjk->gnik', Y, T_corr)
            pts = scaled12(vec6_from_mats(C), w6, self.s_rot)
            dist, nn = self.tree.query(pts.reshape(-1, 12), k=kq, workers=-1)
            if kq > 1:
                # soft correspondence: blend the close-enough neighbours so the target
                # INTERPOLATES between manifold samples instead of latching onto one
                bw = np.exp(-(dist - dist[:, :1]) / self.interp_tau)
                bw /= bw.sum(axis=1, keepdims=True)
                tgt = np.einsum('mk,mkd->md', bw, self.M12[nn])
                dist = np.linalg.norm(tgt - pts.reshape(-1, 12), axis=1)
            else:
                tgt = self.M12[nn]
            tgt, dist = tgt.reshape(G, -1, 12), dist.reshape(G, -1)
            res_hist[:, k] = np.average(dist, axis=1, weights=wts)          # recency-weighted
            delta12 = np.average(tgt - pts, axis=1, weights=wts)
            delta6 = np.zeros((G, 6))
            delta6[:, :3] = delta12[:, :3]
            delta6[:, 3:] = delta12[:, 3:6] / max(self.s_rot, 1e-12)
            delta6[:, ~free] = 0.0
            T_corr = T_corr @ mats_from_vec6(delta6 * self.step_gain)
            theta_hist[:, k + 1] = vec6_from_mats(T_corr)[:, idx]

        # Aggregate the G final corrections into ONE estimate (distances in mm-equivalent space).
        corr6 = vec6_from_mats(T_corr)                        # (G, 6) physical
        scale = np.array([self.s_rot if d.endswith('_deg') else 1.0 for d in DIMS])
        finals = corr6[:, idx] * scale[idx]
        r_fin = res_hist[:, -1]
        if self.aggregator == 'softmax':
            # Residual-softmax: EVERY final contributes, weighted by how close its residual is
            # to the best -- no vote, no gate, no discrete inlier/outlier cliff.
            r_min = float(r_fin.min())
            w = np.exp(-(r_fin - r_min) / max(self.softmax_temp * r_min, 1e-12))
            est = (finals * w[:, None]).sum(axis=0) / w.sum()
            best = r_fin <= r_min * (1.0 + self.softmax_temp)   # the >=~e^-1-weight starts (info)
            final_residual = float(np.average(r_fin, weights=w))
        else:
            # Residual-gated RANSAC over the final corrections.
            gate = self.residual_gate
            keep = (r_fin <= r_fin.min() * float(gate) + 1e-12) if gate else \
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
            final_residual = float(r_fin[best].mean())

        theta = np.zeros(6)
        theta[idx] = est / scale[idx]                         # physical mm / deg, free dims only
        info = {
            'theta_corr': {d: float(theta[j]) for d, j in zip(self.estimate_dims, idx)},
            'inliers': int(best.sum()), 'guesses': G, 'aggregator': self.aggregator,
            'final_residual': final_residual,
            'n_observations': len(vec6), 'recency_half_life_frac': self.recency_half_life_frac,
            'interp_neighbors': self.interp_neighbors,
            # Per-iteration histories (all guesses) for convergence plots: correction params in
            # physical mm/deg on estimate_dims, the mean NN residual, and the RANSAC inlier mask.
            'theta_hist': theta_hist, 'res_hist': res_hist, 'inlier_mask': best,
        }
        return mats_from_vec6(theta), info
