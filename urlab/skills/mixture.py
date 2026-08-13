"""MULTI-MODAL uncertainty: a MIXTURE over solution modes, not one Gaussian.

Raising the multi-start count is unambiguously good for ACCURACY -- more starts means the true
basin actually gets found -- but it breaks the usual uncertainty number. The spread of the finals
then measures the DISTANCE BETWEEN RIVAL MODES rather than the width of any one of them, so a run
that nailed the right answer (and also found a decoy 8 deg away) reports a huge sigma, while a run
whose starts all fell into ONE wrong basin reports a confident, tiny one. Exactly backwards.

The fix is to stop summarising a multi-modal posterior by its mean and standard deviation and
report the mixture itself:

    p(theta) = sum_m  w_m  N(mu_m, C_m)

from which the law of total variance splits the spread into two things that mean different things:

    Cov_total  =  sum_m w_m C_m        +   sum_m w_m (mu_m - mu)(mu_m - mu)^T
                  [ WITHIN: precision ]     [ BETWEEN: ambiguity ]

WITHIN is how sharply the data pins a hypothesis -- more probing data shrinks it. BETWEEN is how
much mass sits on rival hypotheses -- more of the SAME probe does not shrink it, only a
DISCRIMINATING action does. They call for different responses, so they are reported separately.

Two constructors, because the two solvers hand back different things:

  * `from_energy` -- the exhaustive grid gives the whole landscape, so the modes are exact: label
    every cell by the local minimum its steepest-descent path reaches (a watershed), and the
    posterior mass in each basin IS that component's weight. No k to choose, no EM to converge.
  * `from_particles` -- multi-start ICP gives a weighted particle cloud (the finals, weighted by
    residual). Weighted mean-shift collapses it to its modes with one bandwidth and no k.

`separation` (the Mahalanobis gap between the top two modes, in the WITHIN metric) is the number
that says whether the ambiguity is real: two "modes" 0.5 sigma apart are one blurred basin, two
modes 6 sigma apart are a genuine either/or that no amount of averaging resolves. `split` is the
direction between them -- the thing a discriminating probe should move along.
"""

import numpy as np

DEFAULT_TEMP = 0.05          # posterior temperature, RELATIVE to the minimum energy
DEFAULT_MIN_WEIGHT = 0.02    # components below this posterior mass are folded into the rest


class Component:
    """One mode: its posterior mass, mean, covariance and the sample indices behind it."""

    __slots__ = ('weight', 'mean', 'cov', 'idx', 'w_in', 'e_min')

    def __init__(self, weight, mean, cov, idx=None, w_in=None, e_min=float('nan')):
        self.weight = float(weight)
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.idx = idx                 # indices into the source grid / particle array
        self.w_in = w_in               # their weights, normalised WITHIN this component
        self.e_min = float(e_min)      # the energy at this mode's minimum (grid only)

    def sigma(self):
        return np.sqrt(np.maximum(np.diag(self.cov), 0.0))

    def __repr__(self):
        return (f'Component(w={self.weight:.2f}, mean={np.round(self.mean, 2).tolist()}, '
                f'sigma={np.round(self.sigma(), 2).tolist()})')


class Mixture:
    """A weight-sorted list of `Component`s plus the total/within/between decomposition."""

    def __init__(self, components, dims=None):
        comps = [c for c in components if c.weight > 0]
        if not comps:
            raise ValueError('a mixture needs at least one component')
        s = sum(c.weight for c in comps)
        for c in comps:
            c.weight /= s
        self.components = sorted(comps, key=lambda c: -c.weight)
        self.dims = list(dims) if dims else None

    # ------------------------------------------------------------------ moments
    @property
    def n_modes(self):
        return len(self.components)

    @property
    def dominant(self):
        return self.components[0]

    @property
    def mean(self):
        return sum(c.weight * c.mean for c in self.components)

    @property
    def within(self):
        """Sum_m w_m C_m -- how sharply the data pins ONE hypothesis (shrinks with more data)."""
        return sum(c.weight * c.cov for c in self.components)

    @property
    def between(self):
        """Sum_m w_m (mu_m - mu)(mu_m - mu)^T -- how much mass sits on RIVAL hypotheses.

        More of the same probe does not shrink this; only a discriminating action does."""
        mu = self.mean
        return sum(c.weight * np.outer(c.mean - mu, c.mean - mu) for c in self.components)

    @property
    def cov(self):
        return self.within + self.between

    def sigma(self, within_only=False):
        c = self.within if within_only else self.cov
        return np.sqrt(np.maximum(np.diag(c), 0.0))

    @property
    def ambiguity(self):
        """Posterior mass NOT in the dominant mode: 0 = unimodal, ->1 = a coin flip."""
        return float(1.0 - self.components[0].weight)

    @property
    def between_frac(self):
        """Share of the total variance that is BETWEEN modes -- 'is my sigma ambiguity or noise?'"""
        t, b = np.trace(self.cov), np.trace(self.between)
        return float(b / t) if t > 1e-12 else 0.0

    @property
    def separation(self):
        """Mahalanobis gap between the top two modes in the WITHIN metric (inf if only one).

        Under ~1 the two 'modes' are one blurred basin and averaging them is fine; over ~3 they
        are a genuine either/or that no amount of averaging resolves -- probe instead."""
        if len(self.components) < 2:
            return 0.0
        d = self.components[0].mean - self.components[1].mean
        W = self.within
        try:
            return float(np.sqrt(max(d @ np.linalg.solve(W + 1e-9 * np.eye(len(d)), d), 0.0)))
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(d))

    @property
    def split(self):
        """Unit vector from the runner-up mode to the dominant one -- the DISCRIMINATING axis.

        A probe biased along this direction changes what the two hypotheses predict; a probe
        along the sloppy valley does not. Zero vector when the mixture is unimodal."""
        if len(self.components) < 2:
            return np.zeros_like(self.components[0].mean)
        d = self.components[0].mean - self.components[1].mean
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-12 else np.zeros_like(d)

    # ------------------------------------------------------------------ reporting
    def as_dict(self):
        dims = self.dims or [f'd{i}' for i in range(len(self.mean))]
        return {
            'n_modes': self.n_modes,
            'ambiguity': self.ambiguity,
            'between_frac': self.between_frac,
            'separation': self.separation,
            'mean': {d: float(v) for d, v in zip(dims, self.mean)},
            'sigma': {d: float(v) for d, v in zip(dims, self.sigma())},
            'sigma_within': {d: float(v) for d, v in zip(dims, self.sigma(within_only=True))},
            'modes': [{'weight': c.weight,
                       'mean': {d: float(v) for d, v in zip(dims, c.mean)},
                       'sigma': {d: float(v) for d, v in zip(dims, c.sigma())}}
                      for c in self.components],
        }

    def __repr__(self):
        return (f'Mixture({self.n_modes} modes, ambiguity {self.ambiguity:.2f}, '
                f'separation {self.separation:.1f}) ' + repr(self.components[:2]))


# ---------------------------------------------------------------------- construction
def gibbs_weights(E, temp=DEFAULT_TEMP):
    """exp(-(E - E_min) / (temp * E_min)), normalised. Temperature is RELATIVE to the best
    energy, so it means the same thing whatever units the residual came out in."""
    E = np.asarray(E, dtype=float).ravel()
    emin = float(E.min())
    w = np.exp(-(E - emin) / max(temp * abs(emin), 1e-12))
    s = w.sum()
    return w / s if s > 0 else np.full(E.size, 1.0 / E.size)


def descent_labels(E, shape):
    """Label every grid cell by the local minimum its steepest-descent path reaches.

    A watershed, so the labels PARTITION the grid: unlike a sublevel-set flood fill, no posterior
    mass is left unassigned, and the component weights are therefore real probabilities. Fully
    vectorised (pointer jumping), so it costs nothing next to the energy evaluation itself."""
    Eg = np.asarray(E, dtype=float).reshape(shape)
    idx = np.arange(Eg.size).reshape(shape)
    best, best_e = idx.copy(), Eg.copy()
    for a in range(Eg.ndim):
        for s in (-1, 1):
            nb, nb_e = np.roll(idx, s, axis=a), np.roll(Eg, s, axis=a)
            valid = np.ones(Eg.shape, dtype=bool)
            edge = [slice(None)] * Eg.ndim
            edge[a] = 0 if s == 1 else -1          # the row np.roll wrapped around
            valid[tuple(edge)] = False
            upd = valid & (nb_e < best_e)
            best, best_e = np.where(upd, nb, best), np.where(upd, nb_e, best_e)
    p = best.ravel()
    for _ in range(int(np.ceil(np.log2(max(Eg.size, 2)))) + 2):
        q = p[p]
        if np.array_equal(q, p):
            break
        p = q
    roots, labels = np.unique(p, return_inverse=True)
    return labels, roots


def _moments(pts, w):
    mu = (pts * w[:, None]).sum(axis=0)
    d = pts - mu
    return mu, (d * w[:, None]).T @ d


def from_energy(theta, E, shape, temp=DEFAULT_TEMP, min_weight=DEFAULT_MIN_WEIGHT, dims=None):
    """Mixture from an exhaustive-grid energy landscape (the modes are EXACT, no EM).

    `theta` (N, n_dims): the grid's candidate corrections in physical units; `E` (N,) their
    energy; `shape` the mesh shape. Components are the steepest-descent basins, weighted by the
    Gibbs posterior mass they hold, with the within-basin mean and covariance of that posterior."""
    theta = np.asarray(theta, dtype=float).reshape(-1, np.shape(theta)[-1])
    w = gibbs_weights(E, temp)
    labels, roots = descent_labels(E, shape)
    Ef = np.asarray(E, dtype=float).ravel()
    comps = []
    for m in range(len(roots)):
        sel = np.flatnonzero(labels == m)
        mass = float(w[sel].sum())
        if mass <= 0:
            continue
        wn = w[sel] / mass
        mu, cov = _moments(theta[sel], wn)
        comps.append(Component(mass, mu, cov, sel, wn, Ef[roots[m]]))
    comps.sort(key=lambda c: -c.weight)
    # Fold the negligible basins into the survivors rather than dropping their mass: their
    # weights are what makes `ambiguity` a probability instead of a truncated sum.
    keep = [c for c in comps if c.weight >= min_weight] or comps[:1]
    spill = sum(c.weight for c in comps if c not in keep)
    if spill > 0:
        tot = sum(c.weight for c in keep)
        for c in keep:
            c.weight += spill * c.weight / max(tot, 1e-12)
    return Mixture(keep, dims)


def from_particles(pts, w=None, bandwidth=None, min_weight=DEFAULT_MIN_WEIGHT, dims=None,
                   iters=40):
    """Mixture from a WEIGHTED particle cloud -- the multi-start ICP finals.

    Weighted mean-shift: every particle climbs the kernel density until it stops moving, and
    particles landing within half a bandwidth of each other are one mode. One knob (bandwidth,
    defaulting to a Silverman-style estimate), no k, no EM restarts. Component covariances are
    the WITHIN-mode spread of the particles, so `within` is the ICP's own repeatability and
    `between` is the mode ambiguity -- the two the raw std conflates."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    n, d = pts.shape
    w = np.full(n, 1.0 / n) if w is None else np.asarray(w, dtype=float) / max(
        float(np.sum(w)), 1e-12)
    if n == 1:
        return Mixture([Component(1.0, pts[0], np.zeros((d, d)), np.array([0]), np.array([1.0]))],
                       dims)
    if bandwidth is None:
        spread = np.sqrt(np.maximum(np.average((pts - np.average(pts, 0, w)) ** 2, 0, w), 0.0))
        bandwidth = float(max(np.mean(spread) * n ** (-1.0 / (d + 4)) * 1.06, 1e-6))
    h2 = 2.0 * bandwidth ** 2
    x = pts.copy()
    for _ in range(iters):
        k = np.exp(-((x[:, None, :] - pts[None, :, :]) ** 2).sum(-1) / h2) * w[None, :]
        s = k.sum(axis=1, keepdims=True)
        nx = np.where(s > 1e-300, (k @ pts) / np.maximum(s, 1e-300), x)
        if np.max(np.abs(nx - x)) < 1e-4 * bandwidth:
            x = nx
            break
        x = nx
    # merge the converged points: greedy, heaviest first, within half a bandwidth
    order = np.argsort(-w)
    centres, label = [], np.full(n, -1, dtype=int)
    for i in order:
        if label[i] >= 0:
            continue
        for m, c in enumerate(centres):
            if np.linalg.norm(x[i] - c) < 0.5 * bandwidth:
                label[i] = m
                break
        else:
            centres.append(x[i])
            label[i] = len(centres) - 1
    comps = []
    for m in range(len(centres)):
        sel = np.flatnonzero(label == m)
        mass = float(w[sel].sum())
        if mass <= 0:
            continue
        wn = w[sel] / mass
        mu, cov = _moments(pts[sel], wn)
        comps.append(Component(mass, mu, cov, sel, wn))
    comps.sort(key=lambda c: -c.weight)
    keep = [c for c in comps if c.weight >= min_weight] or comps[:1]
    spill = sum(c.weight for c in comps if c not in keep)
    if spill > 0:
        tot = sum(c.weight for c in keep)
        for c in keep:
            c.weight += spill * c.weight / max(tot, 1e-12)
    return Mixture(keep, dims)
