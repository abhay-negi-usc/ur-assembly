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

TWO LATER ADDITIONS, both about what the uncertainty number is allowed to hide:

  * MIXTURE uncertainty (skills/mixture.py). A single sigma over a multi-modal landscape reports
    the DISTANCE BETWEEN RIVALS, not the width of any one of them. `solve` therefore returns the
    whole mixture and splits the spread into WITHIN (one hypothesis' precision -- more probing
    shrinks it) and BETWEEN (mass on rival hypotheses -- only a DISCRIMINATING probe shrinks it).
  * SUPPORT. The residual says how well the observations fit the map; it says nothing about
    whether the map HAS anything there. Off the edge of the manifold the k-NN target is an
    extrapolation from far-away points, the residual can look fine, and the solver happily parks
    there -- the drift-out-of-distribution failure. `support_ratio` is the distance to the k-th
    manifold neighbour at the chosen correction, over the manifold's OWN median k-th-neighbour
    distance: 1 means as well surrounded as a typical map point, 2 means the nearest evidence is
    twice as far away as normal. It inflates sigma (and can optionally penalise the energy).
"""

import numpy as np

from .. import log as urlog
from .manifold import ManifoldEstimator, mats_from_vec6, scaled12, vec6_from_mats
from .mixture import from_energy

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
        posterior_temp: Gibbs temperature for the mixture, relative to E_min (default 0.05)
        mode_min_weight: fold modes below this posterior mass into the rest (default 0.02)
        support_inflation: exponent on max(1, support_ratio) applied to sigma (default 1.0)
        support_penalty: mm-eq added to the energy per mm-eq of missing support (default 0.0)
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
        self.posterior_temp = float(g.get('posterior_temp', 0.05))
        self.mode_min_weight = float(g.get('mode_min_weight', 0.02))
        # DEFAULT 0.0 = REPORT-ONLY. The 2026-08-12 validation came back split: on the old-
        # process eval runs thin support predicts failure (AUROC 0.56, ratio >= 2 -> 72% fail),
        # but on the probe-app runs it is INVERTED (AUROC 0.38, ratio >= 2 -> 12% fail) even
        # after depth-normalising the reference -- there, high ratio marks the deep, seated,
        # HIGH-information probes. A multiplier that widens sigma on the best cases is worse
        # than no multiplier, so the ratio is logged for analysis and inflates nothing unless
        # this is explicitly raised.
        self.support_inflation = float(g.get('support_inflation', 0.0))
        self.support_penalty = float(g.get('support_penalty', 0.0))
        # PEAK-MEMORY BOUND for the energy sweep. The neighbour gather is
        # (candidates x rows, k, 12) float64 -- with 1881 candidates, 300 rows and k=64 that is
        # 3.5 GB in ONE allocation, i.e. a MemoryError mid-probe on a machine that had been
        # coping fine at 150 rows. The sweep is therefore chunked over candidates to this cap.
        self.max_query_bytes = float(g.get('max_query_bytes', 256e6))
        log.info('Grid estimator: %s over %s = %d candidates, info_weighting=%s.',
                 ' x '.join(f'{d}+-{a[-1]:g}@{a[1] - a[0]:g}'
                            for d, a in zip(self.estimate_dims, axes)),
                 'x'.join(str(s) for s in self.grid_shape), len(self.grid6),
                 self.info_weighting)
        log.info('   sigma inflated by max(1, support_ratio)^%.2g, energy penalty %.2g/mm-eq.',
                 self.support_inflation, self.support_penalty)

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
        """Mean (info-weighted) soft-kNN residual at every grid candidate.

        Returns (E, n_rows, S). S is the companion SUPPORT curve: the weighted mean of the
        per-row support RATIO -- the k-th-neighbour distance over the map's own reference AT
        THAT ROW'S DEPTH (support_ref_at). E says how well the observations fit the map; S says
        how much map was there to fit, already normalised so 1 = typical, 2 = twice as far from
        evidence as an in-distribution row at the same depth."""
        vec6 = np.asarray(vec6, dtype=float)
        if len(vec6) < max(self.min_observations, 1):
            return None, len(vec6), None
        Y = mats_from_vec6(vec6)
        rw = self.row_weights(vec6)
        n_c, n_r = len(self.grid6), len(vec6)
        E, S = np.empty(n_c), np.empty(n_c)
        per = max(int(self.max_query_bytes
                      // max(n_r * self.support_k * 12 * 8, 1)), 1)
        for lo in range(0, n_c, per):               # chunked: see max_query_bytes
            hi = min(lo + per, n_c)
            C = np.einsum('nij,kjl->knil', Y, self._gridm[lo:hi])
            pts = scaled12(vec6_from_mats(C), w6, self.s_rot).reshape(-1, 12)
            # query support_k (>= 2), so the last column is the SAME k the support reference was
            # measured at -- d1 still comes from the first interp_neighbors columns.
            dist, nn = self.tree.query(pts, k=self.support_k, workers=-1)
            dist = dist[:, None] if dist.ndim == 1 else dist
            if self.interp_neighbors > 1:
                bw = np.exp(-(dist - dist[:, :1]) / self.interp_tau)
                bw /= bw.sum(axis=1, keepdims=True)
                tgt = np.einsum('mk,mkd->md', bw, self.M12[nn])
                d1 = np.linalg.norm(tgt - pts, axis=1)
            else:
                d1 = dist[:, 0]
            E[lo:hi] = d1.reshape(hi - lo, n_r) @ rw
            # per-row RATIO against the reference at that row's depth (pts[:, 0] is x in mm) --
            # a global reference confounds support with depth (see support_ref_at).
            sup_row = dist[:, -1] / self.support_ref_at(pts[:, 0])
            S[lo:hi] = sup_row.reshape(hi - lo, n_r) @ rw
        return E, n_r, S

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
    def solve(self, E, n_obs, support=None):
        """(T_corr, info) from a (possibly fused) energy vector over self.grid6.

        `support` (optional) is the companion curve from `energy` -- when several probes were
        fused, pass their MEAN, not their sum: support is a property of the map, not evidence
        that accumulates."""
        E = np.asarray(E, dtype=float).ravel()
        sup = None if support is None else np.asarray(support, dtype=float).ravel()
        if sup is not None and self.support_penalty > 0:
            # Penalise candidates whose nearest evidence is further away than the map's own
            # spacing at that depth (sup is a RATIO; support_ref converts the excess back to
            # mm-eq so the penalty knob stays in energy units). This does not merely flag the
            # edge of the manifold, it stops the argmin sliding out there -- so it is OFF by
            # default and must be opted into.
            E = E + self.support_penalty * self.support_ref * np.maximum(sup - 1.0, 0.0)
        k = int(np.argmin(E))
        theta = self.grid6[k]
        Emin = float(E[k])
        curv, sigma, cov, worst = self._curvature(E)
        modes = self._modes(E)
        width = float(np.mean(E <= Emin * 1.2))
        mix = from_energy(self.grid6[:, self.idx], E, self.grid_shape, temp=self.posterior_temp,
                          min_weight=self.mode_min_weight, dims=self.estimate_dims)

        # SUPPORT at the chosen correction -- already a depth-normalised ratio from energy().
        ratio = 1.0 if sup is None else float(sup[k])
        infl = float(max(ratio, 1.0) ** self.support_inflation)

        # THE HEADLINE COVARIANCE = the curvature width PLUS the between-mode spread, inflated
        # by missing support. Deliberately additive rather than a wholesale swap to the mixture's
        # own covariance: the curvature width is the calibrated one (AUROC 0.89-0.91 against
        # ground truth), while the mixture's WITHIN width is just the Gibbs posterior at an
        # arbitrary temperature. So the validated number is kept for the within-mode part and the
        # mixture contributes only BETWEEN -- which is temperature-robust, being a real distance
        # between mode centres. Unimodal cases therefore report EXACTLY what they did before.
        cov_within = np.asarray(cov, dtype=float)
        cov_total = (cov_within + mix.between) * infl ** 2
        sig_within = np.sqrt(np.maximum(np.diag(cov_within), 0.0)) * infl
        sigma_out = {d: float(min(np.sqrt(max(cov_total[a, a], 0.0)), float(ax[-1])))
                     for a, (d, ax) in enumerate(zip(self.estimate_dims, self.grid_axes))}
        info = {
            'theta_corr': {d: float(theta[j]) for d, j in zip(self.estimate_dims, self.idx)},
            'final_residual': Emin,
            'n_observations': int(n_obs),
            'candidates': int(len(self.grid6)),
            'curvature_uncertainty': curv,       # per dim: 1/curvature (big = flat = uncertain)
            'sigma': sigma_out,                  # per dim MARGINAL width: within + between + support
            'sigma_curv': sigma,                 # the raw single-basin curvature width
            'sigma_within': {d: float(s) for d, s in zip(self.estimate_dims, sig_within)},
            'covariance': cov_total,             # full n x n -- the ellipse and its eigen-axes
            'cov_mixture': cov_total,            # same object; kept for the plot/CSV call sites
            'cov_between': mix.between * infl ** 2,   # the AMBIGUITY part on its own
            'uncertainty': float(worst),
            'modes': int(modes),
            'multimodal': bool(modes > 1),
            'width_frac': width,
            'mixture': mix,                      # the components themselves
            'n_mixture_modes': mix.n_modes,
            'ambiguity': mix.ambiguity,          # posterior mass NOT in the dominant mode
            'between_frac': mix.between_frac,    # share of the spread that is mode ambiguity
            'separation': mix.separation,        # top-2 gap, in units of the within-mode width
            'support_ratio': ratio,              # 1 = as surrounded as a typical map point
            'support_inflation': infl,
            'inliers': int(np.sum(E <= Emin * 1.05)),   # kept for log/CSV compatibility
            'guesses': int(len(self.grid6)),
            'aggregator': 'grid',
            'energy': E,
        }
        return mats_from_vec6(theta), info

    def estimate(self, vec6, w6, prior_energy=None):
        """One-shot estimate (optionally fused with a prior energy vector)."""
        E, n, S = self.energy(vec6, w6)
        if E is None:
            return None, f'too few observations ({n} < {self.min_observations})'
        if prior_energy is not None:
            E = E + np.asarray(prior_energy, dtype=float)
        return self.solve(E, n, S)
