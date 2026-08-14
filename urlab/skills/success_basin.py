"""SUCCESS BASIN -- P(seat | offset) learned from the contact manifold itself.

The manifold is a set of recorded insertion trials, each executed at a known misalignment. Every
trial therefore carries a LABEL nobody had to annotate: how deep it got. Define

    seat  :=  max_x  >=  reference_depth - seat_margin_mm

with reference_depth read off the manifold's own deepest insertion, and the manifold becomes a
labelled dataset of (offset -> seated?) from which P(seat | offset) follows by a k-NN vote.

That gives the estimator something better than a parameter tolerance to aim at. Belief error in
z and pitch does not matter per se -- only whether the COMBINATION lands where the part actually
seats -- and this object encodes exactly that, including the fact that the basin is elongated
and skewed rather than a box.

Two uses, both decision-theoretic:
  * P_SEAT OF A POSTERIOR (not of a point): the estimator never knows its remaining error, only
    a distribution over it, so the honest quantity is E_p[P(seat)] = sum_i p_i P(seat | o_i).
    `p_seat_posterior` does that sum.
  * SUCCESS SCORING after the fact: `is_seated` applies the same depth rule to a real insertion,
    so the gate and the verdict use ONE definition.

SIGN CONVENTION. A belief error R (T_believed = T_true @ R) makes the robot plan as though the
part were at R, so the part physically rides at inverse(R) relative to the nominal path -- and
the manifold's trials are indexed by that PHYSICAL offset. `offset_of_error` does the inversion;
getting it backwards silently mirrors the basin, so it lives here rather than in each caller.
"""

import csv

import numpy as np
from scipy.spatial import cKDTree

from .. import log as urlog
from .manifold import DIMS, mats_from_vec6, vec6_from_mats

log = urlog.get('success_basin')

POSE_COLS = [f'connector_target_{d}' for d in DIMS]
X_COL = 'connector_target_x_mm'


class SuccessBasin:
    """P(seat | offset) over the estimated dims, from a contact-manifold CSV."""

    def __init__(self, manifold_csv, estimate_dims, cfg_section=None):
        c = dict(cfg_section or {})
        self.dims = list(estimate_dims)
        self.idx = [DIMS.index(d) for d in self.dims]
        self.seat_margin_mm = float(c.get('seat_margin_mm', 4.0))
        self.k = int(c.get('neighbors', 25))
        # deg -> mm equivalence when measuring 'nearby offsets' (matches the estimator's metric)
        self.s_rot = float(c.get('scaling_constant_deg_to_mm', 0.2))
        # Which depth counts as 'fully inserted'. 'max' = the deepest single manifold row (the
        # literal reading); a NUMBER = that percentile of the per-trial maxima. The default is a
        # PERCENTILE, not 'max', because a manifold contains crush-through and slide-past trials
        # that travel tens of mm past the real seat: on the 2026-08 hose map 'max' puts the
        # threshold at +24.9 mm, which NOTHING reaches (0% of trials seat) and the gate can never
        # open, while p95 puts it at -5.1 mm and 61% of trials seat.
        self.depth_reference = c.get('depth_reference', 95.0)
        self.min_rows = int(c.get('min_trial_rows', 30))

        pose, x = self._read(manifold_csv)
        self.offsets, self.reach = self._trials(pose, x)
        if len(self.offsets) < self.k:
            raise ValueError(f'{manifold_csv}: only {len(self.offsets)} usable trials for the '
                             f'success basin (need >= {self.k})')
        self.reference_depth = self._reference(x, self.reach)
        self.seat_depth = self.reference_depth - self.seat_margin_mm
        self.seated = self.reach >= self.seat_depth
        self._tree = cKDTree(self._scale(self.offsets))

        rate = float(self.seated.mean())
        log.info('Success basin: %d trials, reference depth %+.2f mm (%s), seat when max_x >= '
                 '%+.2f mm -> %.0f%% of the manifold seats.', len(self.offsets),
                 self.reference_depth, self.depth_reference, self.seat_depth, 100.0 * rate)
        if rate < 0.10:
            log.warning('Only %.0f%% of manifold trials count as SEATED. The reference depth is '
                        'probably being set by a crush-through/slide-past outlier -- consider '
                        'depth_reference: 99 (a percentile of the per-trial maxima) or a larger '
                        'seat_margin_mm, or the P(seat) gate will never open.', 100.0 * rate)
        # what the alternatives would give, so the choice can be made from one run's log
        alts = ', '.join(
            f'{r}: {self._rate(x, r):.0%}' for r in ('max', 99.9, 99.0, 95.0, 90.0))
        log.info('   seat rate under other depth_reference values -- %s', alts)
        # Two ceilings the caller needs BEFORE choosing a gate:
        #   p_seat_max    the best value anywhere -- a gate above it can never open at all;
        #   p_seat_zero   the value at ZERO offset, i.e. what a PERFECT belief scores. This is
        #                 the one that usually bites: it is bounded by how repeatably the process
        #                 seats (on the 2026-08 hose map, trials at near-identical offsets agree
        #                 on seated/not only 67% of the time, implying a ~83% per-trial ceiling),
        #                 so a gate above it means even a perfect estimate never commits.
        self.p_seat_max = float(self.p_seat(self.offsets).max()) if len(self.offsets) else 0.0
        self.p_seat_zero = float(self.p_seat(np.zeros((1, len(self.dims))))[0])
        log.info('   P(seat) ceilings: %.0f%% at the best offset, %.0f%% at ZERO offset (a '
                 'perfect belief). A gate above the latter never opens even when the estimate '
                 'is exact.', 100.0 * self.p_seat_max, 100.0 * self.p_seat_zero)

    # ------------------------------------------------------------------ construction
    def _read(self, path):
        pose, xs = [], []
        with open(path, newline='') as fh:
            r = csv.DictReader(fh)
            missing = [c for c in POSE_COLS if c not in (r.fieldnames or [])]
            if missing:
                raise ValueError(f'{path}: missing {missing[0]} (and {len(missing) - 1} more)')
            for rec in r:
                try:
                    pose.append([float(rec[c]) for c in POSE_COLS])
                except (TypeError, ValueError):
                    continue
        pose = np.asarray(pose, dtype=float)
        return pose, pose[:, 0]

    def _trials(self, pose, x):
        """Segment the log into insertion trials on the retract (a big backward jump in x)."""
        bounds = np.flatnonzero(np.diff(x) < -6.0) + 1
        offs, reach = [], []
        for s in np.split(np.arange(len(x)), bounds):
            if len(s) < self.min_rows:
                continue
            xs = x[s]
            shallow = s[xs <= xs.min() + 2.0]      # the approach, before contact deflects it
            if len(shallow) < 3:
                continue
            offs.append([np.median(pose[shallow, j]) for j in self.idx])
            reach.append(float(xs.max()))
        return np.asarray(offs, dtype=float), np.asarray(reach, dtype=float)

    def _reference(self, x, reach):
        r = self.depth_reference
        if isinstance(r, str) and r.strip().lower() == 'max':
            return float(np.max(x))
        return float(np.percentile(reach, float(r)))

    def _rate(self, x, ref):
        d = (float(np.max(x)) if isinstance(ref, str) else
             float(np.percentile(self.reach, float(ref))))
        return float(np.mean(self.reach >= d - self.seat_margin_mm))

    def _scale(self, o):
        o = np.atleast_2d(np.asarray(o, dtype=float))
        s = np.array([self.s_rot if d.endswith('_deg') else 1.0 for d in self.dims])
        return o * s

    # ------------------------------------------------------------------ queries
    def offset_of_error(self, err6):
        """PHYSICAL offset the part rides at, given a belief error (see the module note)."""
        e = np.atleast_2d(np.asarray(err6, dtype=float))
        o = vec6_from_mats(np.linalg.inv(mats_from_vec6(e)))
        return o[:, self.idx]

    def p_seat(self, offsets):
        """P(seat) at physical offsets (N, n_dims) -- k-NN vote over the labelled trials."""
        q = self._scale(offsets)
        _, nn = self._tree.query(q, k=min(self.k, len(self.offsets)), workers=-1)
        return self.seated[nn].mean(axis=1)

    def p_seat_posterior(self, offsets, weights):
        """E_p[P(seat)] -- the decision quantity. The robot never knows its remaining error,
        so P(seat) must be averaged over the posterior, not evaluated at a point estimate."""
        w = np.asarray(weights, dtype=float)
        if w.sum() <= 0:
            return float('nan')
        return float((self.p_seat(offsets) * (w / w.sum())).sum())

    def is_seated(self, max_x):
        """The SAME depth rule the labels use, applied to a real insertion."""
        return bool(float(max_x) >= self.seat_depth)
