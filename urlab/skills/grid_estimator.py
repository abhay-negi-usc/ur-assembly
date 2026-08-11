"""EXHAUSTIVE-GRID manifold estimator with probe fusion and cheap uncertainty.

The 2026-08-11 hose information campaign (data/'test data'/ablation/hose_information_report.md)
found that on this task:

  * multi-start ICP buys nothing -- the correction space is 1-2 dims, so an EXHAUSTIVE grid is
    both cheaper and free of local-minimum / aggregator pathologies (no softmax-vs-ransac choice,
    no consensus-vs-residual tie-breaks);
  * PROBES FUSE BY SUMMING ENERGY CURVES. With belief error E and a commanded trajectory bias B,
    the logged rows are the biased path and the correction that realigns them to the manifold is
    inverse(E) REGARDLESS of B -- so several probes at different biases measure the SAME
    correction and their energies simply add. Diverse probes break aliases: on hard cases 3
    probes at alternating +/-5 deg pitch cut the 75th-percentile error from 11.08 to 1.63 deg;
  * the best uncertainty signal is the CURVATURE of the fused energy at the minimum, from just
    two extra evaluations at theta* +/- curvature_probe_deg (AUROC 0.89-0.91, beating the full
    Gibbs posterior std at 128 evaluations). +/-3 deg works; +/-1 deg does not (it measures
    interpolation noise, AUROC 0.66);
  * MODE COUNT (connected components of {E <= mode_threshold * E_min}) is a poor ranker but an
    excellent FLAG: multi-modal solutions fail 44% of the time vs 6% unimodal;
  * rows near the socket mouth carry ~3x the information of the approach rows (discriminability
    d' 0.7 -> 2.4), so rows are weighted by an informativeness-vs-depth curve.

Everything above was validated on a LEAK-FREE synthetic harness built from the manifold itself
(source trial masked out of the k-NN query). It assumes the observations were collected by the
SAME process as the manifold -- see the app's probe.settle_s note.
"""

import numpy as np

from .. import log as urlog
from .manifold import ManifoldEstimator, mats_from_vec6, scaled12, vec6_from_mats

log = urlog.get('grid_estimator')

# Informativeness vs insertion depth (mm), measured as noise-normalized discriminability d'
# between manifold trials at different offsets. Rows are weighted by d'^2.
INFO_DEPTH_MM = (-18.0, -15.0, -12.0, -10.5, -9.0, -7.5, -6.0, -4.5, -3.0)
INFO_DPRIME = (0.69, 0.73, 0.69, 1.26, 1.66, 1.63, 2.38, 2.13, 2.13)


class GridManifoldEstimator(ManifoldEstimator):
    """ManifoldEstimator with an exhaustive-grid solver, probe fusion and curvature uncertainty.

    Config: the usual `estimation:` keys plus a `grid:` subsection --
        range: per estimated dim, the half-width of the search box (mm / deg)
        step:  per estimated dim, the grid resolution
        info_weighting: 'depth' (default) | 'none'
        curvature_probe_deg: offset for the 3-point curvature (default 3.0)
        mode_threshold: sublevel factor for the mode flag (default 1.1)
    """

    def __init__(self, cfg_section):
        c = dict(cfg_section or {})
        g = dict(c.get('grid') or {})
        super().__init__(c)
        rng_cfg = dict(g.get('range') or {})
        step_cfg = dict(g.get('step') or {})
        axes = []
        for d in self.estimate_dims:
            half = float(rng_cfg.get(d, self.init_range.get(d, 8.0)))
            step = float(step_cfg.get(d, 0.25))
            if half <= 0 or step <= 0:
                raise ValueError(f'grid range/step for {d!r} must be > 0')
            n = int(round(2 * half / step)) + 1
            if n > 4001:
                raise ValueError(f'grid for {d!r} has {n} points -- raise step or lower range')
            axes.append(np.linspace(-half, half, n))
        self.grid_axes = axes
        mesh = np.meshgrid(*axes, indexing='ij')
        self.grid_shape = mesh[0].shape
        flat = np.zeros((mesh[0].size, 6))
        for j, m in zip(self.idx, mesh):
            flat[:, j] = m.ravel()
        self.grid6 = flat
        self._gridm = mats_from_vec6(flat)
        self.info_weighting = str(g.get('info_weighting', 'depth')).strip().lower()
        if self.info_weighting not in ('depth', 'none'):
            raise ValueError("grid.info_weighting must be 'depth' or 'none'")
        # Curvature probe distance, PER DIM (mm for translations, deg for rotations). The scalar
        # curvature_probe_deg is the fallback for every dim; `curvature_probe: {dim: value}`
        # overrides it. 3 deg was validated for pitch -- 1 deg does NOT work (it measures
        # interpolation noise), so anything below ~2 grid steps is rejected here.
        base = float(g.get('curvature_probe_deg', 3.0))
        per_dim = dict(g.get('curvature_probe') or {})
        self.curvature_probe = {}
        for d, ax in zip(self.estimate_dims, axes):
            v = float(per_dim.get(d, base))
            step = ax[1] - ax[0]
            if v < 2 * step:
                raise ValueError(f'grid.curvature_probe[{d!r}] = {v:g} is under two grid steps '
                                 f'({2 * step:g}) -- it would measure interpolation noise')
            if v > ax[-1]:
                raise ValueError(f'grid.curvature_probe[{d!r}] = {v:g} exceeds the grid '
                                 f'half-range {ax[-1]:g}')
            self.curvature_probe[d] = v
        self.mode_threshold = float(g.get('mode_threshold', 1.1))
        log.info('Grid estimator: %s over %s = %d candidates, info_weighting=%s.',
                 ' x '.join(f'{d}+-{a[-1]:g}@{a[1] - a[0]:g}'
                            for d, a in zip(self.estimate_dims, axes)),
                 'x'.join(str(s) for s in self.grid_shape), len(self.grid6),
                 self.info_weighting)

    # ------------------------------------------------------------------ energy
    def row_weights(self, vec6):
        """Per-observation weight from the informativeness-vs-depth curve (sums to 1)."""
        if self.info_weighting == 'none':
            return np.full(len(vec6), 1.0 / max(len(vec6), 1))
        w = np.interp(vec6[:, 0], INFO_DEPTH_MM, INFO_DPRIME,
                      left=INFO_DPRIME[0], right=INFO_DPRIME[-1]) ** 2
        s = w.sum()
        return w / s if s > 0 else np.full(len(vec6), 1.0 / max(len(vec6), 1))

    def energy(self, vec6, w6):
        """Mean (info-weighted) soft-kNN residual at every grid candidate. Returns (E, n_rows)."""
        vec6 = np.asarray(vec6, dtype=float)
        if len(vec6) < max(self.min_observations, 1):
            return None, len(vec6)
        Y = mats_from_vec6(vec6)
        C = np.einsum('nij,kjl->knil', Y, self._gridm)
        pts = scaled12(vec6_from_mats(C), w6, self.s_rot).reshape(-1, 12)
        dist, nn = self.tree.query(pts, k=self.interp_neighbors, workers=-1)
        if self.interp_neighbors > 1:
            bw = np.exp(-(dist - dist[:, :1]) / self.interp_tau)
            bw /= bw.sum(axis=1, keepdims=True)
            tgt = np.einsum('mk,mkd->md', bw, self.M12[nn])
            d1 = np.linalg.norm(tgt - pts, axis=1)
        else:
            d1 = dist if dist.ndim == 1 else dist[:, 0]
        R = d1.reshape(len(self.grid6), len(vec6))
        return R @ self.row_weights(vec6), len(vec6)

    # ------------------------------------------------------------------ uncertainty
    def _curvature(self, E):
        """3-point curvature per estimated dim at the minimum -- the campaign's winner.

        Returns (inv_curv, sigma, worst):
          inv_curv[dim] = 1/curvature  -- the RANKING metric (big = flat basin = uncertain)
          sigma[dim]    = sqrt(E_min / curvature) -- the same information in the DIM'S OWN UNITS
                          (mm or deg), which is what error bars should show
          worst         = max inv_curv over dims (one number for the CSV / gating)"""
        Eg = E.reshape(self.grid_shape)
        k = np.unravel_index(int(np.argmin(Eg)), self.grid_shape)
        Emin = float(Eg[k])
        n = len(self.estimate_dims)
        inv_curv, sigma = {}, {}
        H = np.zeros((n, n))
        probe = [max(int(round(self.curvature_probe[d] / (ax[1] - ax[0]))), 1)
                 for d, ax in zip(self.estimate_dims, self.grid_axes)]
        for a, (dim, ax) in enumerate(zip(self.estimate_dims, self.grid_axes)):
            lo, hi = list(k), list(k)
            lo[a] = max(k[a] - probe[a], 0)
            hi[a] = min(k[a] + probe[a], len(ax) - 1)
            h = 0.5 * (ax[hi[a]] - ax[lo[a]])
            curv = ((float(Eg[tuple(lo)]) - 2.0 * Emin + float(Eg[tuple(hi)])) / (h ** 2)
                    if h > 0 else 0.0)
            H[a, a] = curv
            inv_curv[dim] = float(1.0 / curv) if curv > 1e-12 else float('inf')
        # OFF-DIAGONALS: without them the per-dim sigma is the CONDITIONAL width and hides the
        # z-pitch trade-off entirely. The full covariance below reports the MARGINAL widths and,
        # through its eigenvectors, the stiff/sloppy directions the ellipse draws.
        for a in range(n):
            for b in range(a + 1, n):
                def at(sa, sb):
                    q = list(k)
                    q[a] = int(np.clip(k[a] + sa * probe[a], 0, self.grid_shape[a] - 1))
                    q[b] = int(np.clip(k[b] + sb * probe[b], 0, self.grid_shape[b] - 1))
                    return float(Eg[tuple(q)])
                ha = (self.grid_axes[a][int(np.clip(k[a] + probe[a], 0, self.grid_shape[a] - 1))]
                      - self.grid_axes[a][k[a]])
                hb = (self.grid_axes[b][int(np.clip(k[b] + probe[b], 0, self.grid_shape[b] - 1))]
                      - self.grid_axes[b][k[b]])
                if ha > 0 and hb > 0:
                    H[a, b] = H[b, a] = (at(1, 1) - at(1, -1) - at(-1, 1)
                                         + at(-1, -1)) / (4.0 * ha * hb)
        try:
            w_h, V_h = np.linalg.eigh(H)
            floor = max(1e-9, 1e-6 * max(abs(w_h).max(), 1e-9))
            cov = (V_h * (max(Emin, 1e-12) / np.maximum(w_h, floor))) @ V_h.T
        except np.linalg.LinAlgError:
            cov = np.diag([float(ax[-1]) ** 2 for ax in self.grid_axes])
        for a, (dim, ax) in enumerate(zip(self.estimate_dims, self.grid_axes)):
            sigma[dim] = float(min(np.sqrt(max(cov[a, a], 0.0)), float(ax[-1])))
        return inv_curv, sigma, cov, (max(inv_curv.values()) if inv_curv else float('inf'))

    def _modes(self, E):
        """Number of connected components of {E <= mode_threshold * E_min} (the mode FLAG)."""
        Emin = float(np.min(E))
        sub = (E <= Emin * self.mode_threshold).reshape(self.grid_shape)
        seen = np.zeros(sub.shape, dtype=bool)
        comps = 0
        nd = sub.ndim
        for start in np.argwhere(sub):
            t = tuple(start)
            if seen[t]:
                continue
            comps += 1
            stack = [t]
            seen[t] = True
            while stack:
                cur = stack.pop()
                for a in range(nd):
                    for s in (-1, 1):
                        nxt = list(cur)
                        nxt[a] += s
                        if 0 <= nxt[a] < sub.shape[a]:
                            nt = tuple(nxt)
                            if sub[nt] and not seen[nt]:
                                seen[nt] = True
                                stack.append(nt)
        return comps

    # ------------------------------------------------------------------ estimate
    def solve(self, E, n_obs):
        """(T_corr, info) from a (possibly fused) energy vector over self.grid6."""
        k = int(np.argmin(E))
        theta = self.grid6[k]
        Emin = float(E[k])
        curv, sigma, cov, worst = self._curvature(E)
        modes = self._modes(E)
        width = float(np.mean(E <= Emin * 1.2))
        info = {
            'theta_corr': {d: float(theta[j]) for d, j in zip(self.estimate_dims, self.idx)},
            'final_residual': Emin,
            'n_observations': int(n_obs),
            'candidates': int(len(self.grid6)),
            'curvature_uncertainty': curv,       # per dim: 1/curvature (big = flat = uncertain)
            'sigma': sigma,                      # per dim MARGINAL width, mm / deg
            'covariance': cov,                   # full n x n -- the ellipse and its eigen-axes
            'uncertainty': float(worst),
            'modes': int(modes),
            'multimodal': bool(modes > 1),
            'width_frac': width,
            'inliers': int(np.sum(E <= Emin * 1.05)),   # kept for log/CSV compatibility
            'guesses': int(len(self.grid6)),
            'aggregator': 'grid',
        }
        return mats_from_vec6(theta), info

    def estimate(self, vec6, w6, prior_energy=None):
        """One-shot estimate (optionally fused with a prior energy vector)."""
        E, n = self.energy(vec6, w6)
        if E is None:
            return None, f'too few observations ({n} < {self.min_observations})'
        if prior_energy is not None:
            E = E + np.asarray(prior_energy, dtype=float)
        return self.solve(E, n)
