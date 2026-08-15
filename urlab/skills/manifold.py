"""Contact-manifold pose estimation -- localise a held part from touch, against recorded data.

The CONTACT MANIFOLD (built by analysis/contact_manifold.py from uncertain_sampling and/or
truth-rebased estimator_eval runs) pairs
"connector pose w.r.t. its target" with "wrench in the connector frame" over many recorded
insertions. If the robot's BELIEF of where the held connector sits is wrong, the observations it
collects while inserting land OFF the manifold by exactly that (rigid) belief error -- so aligning
them back onto the manifold estimates the error. This module does that alignment:

  * a common, mm-equivalent 12-D space:  [x, y, z (mm) | r, p, y (deg x s_rot) |
    unit(f) x s_force | unit(tau) x s_torque]   -- weights decide what "nearest" means.
    Every one of the twelve carries its OWN weight on top of that: dim_weights for the six
    pose dimensions, wrench_weights for the six wrench axes, with s_rot / s_force / s_torque
    left as the pure UNIT CONVERSIONS they were;
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
    cluster can then never outvote a few well-aligned starts;
  * a MIXTURE over the finals (skills/mixture.py), because raising the guess count helps ACCURACY
    (the true basin gets found) while wrecking the usual uncertainty number: the std of the finals
    then measures the DISTANCE BETWEEN RIVAL MODES, not the width of any one of them, so a run
    that found the right answer AND a decoy looks less certain than one whose starts all fell into
    a single wrong basin. The mixture reports WITHIN (one hypothesis' precision) and BETWEEN (mass
    on rivals) separately; only the second calls for a discriminating probe rather than more data;
  * OPTIONAL SEEDED starts (`seeds=`): allocate part of the guess budget around named hypotheses
    -- the two modes of the previous solve -- so a probe meant to tell them apart actually
    re-examines both instead of re-rolling the dice.

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
from .mixture import Component, Mixture, from_particles

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


# The six WRENCH feature axes, in the order they occupy columns 6..11 of the 12-D point.
# Named separately from DIMS because they are FEATURES, not pose dimensions: nothing is ever
# estimated in them, so (unlike dim_weights) a zero here is always legal.
WRENCH_DIMS = ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')

# The four physical BLOCKS of the 12-D point. They are the granularity at which the kNN kernel
# can be given separate bandwidths (estimation.interp_softness as a dict): pose geometry and
# contact wrench are sampled at different densities and carry different noise, so one bandwidth
# for all twelve is a compromise. MEASURED (2026-08-14 BNC, analysis/bnc_tuning): translation
# wants ~5x the bandwidth the others do.
BLOCKS = (('translation', slice(0, 3)), ('rotation', slice(3, 6)),
          ('force', slice(6, 9)), ('torque', slice(9, 12)))


def scaled12(vec6, w6, s_rot, dim_w=None):
    """The common 12-D point: [t (mm) | rot (deg x s_rot) | scaled unit f | scaled unit tau].

    `dim_w` (optional, 6 per-dimension multipliers in DIMS order) weights the POSE channels ON
    TOP of the unit scaling: s_rot stays the mm <-> deg conversion, dim_w says how much each
    dimension MATTERS relative to the others (e.g. weight y down when it is fixtured, weight z
    up when it is the decisive channel). None = all ones = the original metric."""
    v = np.asarray(vec6, dtype=float)
    w = np.broadcast_to(np.asarray(w6, dtype=float), v.shape[:-1] + (6,))
    p = np.concatenate([v[..., :3], v[..., 3:] * s_rot], axis=-1)
    if dim_w is not None:
        p = p * np.asarray(dim_w, dtype=float)
    return np.concatenate([p, w], axis=-1)


class ManifoldEstimator:
    """The manifold KD-tree plus the multi-start ICP / RANSAC solver, wired from a config section.

    Reads `estimation:` (see configs/cable_pick_estimate_assemble.yaml for the schema)."""

    def __init__(self, cfg_section):
        c = cfg_section or {}
        self.s_rot = float(c.get('scaling_constant_deg_to_mm', 1.0))
        self.s_force = float(c.get('scaling_constant_unit_force_to_mm', 0.1))
        self.s_torque = float(c.get('scaling_constant_unit_torque_to_mm', 0.1))
        # PER-DIMENSION pose weights, ON TOP of the unit scaling: s_rot stays the mm <-> deg
        # CONVERSION; dim_weights say how much each dimension MATTERS relative to the others
        # (partial dicts fill with 1.0; absent = the original uniform metric). Zero is allowed
        # only on dimensions that are NOT being estimated -- a zero weight makes the channel
        # invisible to the match, and the ICP update divides by it on the estimated dims.
        dw = dict(c.get('dim_weights', {}) or {})
        bad = [k for k in dw if k not in DIMS]
        if bad:
            raise ValueError(f'estimation.dim_weights keys {bad} not in {DIMS}')
        self.dim_w = np.array([float(dw.get(d, 1.0)) for d in DIMS])
        if np.any(self.dim_w < 0):
            raise ValueError('estimation.dim_weights must be >= 0')
        # PER-AXIS wrench weights, the exact counterpart of dim_weights for the six wrench
        # feature columns: s_force / s_torque stay the block-level unit conversion, and these
        # say how much each INDIVIDUAL axis matters. The block scalars alone cannot express
        # that the axes carry different information -- on an insertion, fx (along the
        # insertion) and the two lateral axes mean physically different things, and one
        # dominating axis silently decides the match. Zero is always legal here (no wrench
        # axis is ever estimated), so an axis can be switched off outright.
        # The logged wrench lives in the BELIEVED connector frame, so a candidate correction
        # moves it exactly as it moves the pose -- the same re-basing _rebase_rows applies when
        # a correction is COMMITTED. Evaluating candidates without it scores every non-zero
        # theta with its wrench in the wrong frame (measured 2026-08-14: the force channel's
        # tracking of the truth nearly tripled, gain 0.18 -> 0.52, once this was applied).
        # Off only to reproduce pre-2026-08-14 numbers.
        self.wrench_follows_correction = bool(c.get('wrench_follows_correction', True))
        ww = dict(c.get('wrench_weights', {}) or {})
        bad = [k for k in ww if k not in WRENCH_DIMS]
        if bad:
            raise ValueError(f'estimation.wrench_weights keys {bad} not in {WRENCH_DIMS}')
        self.wrench_w = np.array([float(ww.get(k, 1.0)) for k in WRENCH_DIMS])
        if np.any(self.wrench_w < 0):
            raise ValueError('estimation.wrench_weights must be >= 0')
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
        zero_est = [d for d, j in zip(self.estimate_dims, self.idx) if self.dim_w[j] <= 0]
        if zero_est:
            raise ValueError(f'estimation.dim_weights is 0 on ESTIMATED dim(s) {zero_est} -- '
                             'the match cannot see, and the update cannot move, that dimension')
        # the EFFECTIVE per-dim scale (unit conversion x importance): physical <-> metric space
        # The EFFECTIVE per-axis wrench scale, the counterpart of pose_scale6: block scalar
        # x per-axis weight. Exposed so diagnostics and tests can read the metric that is
        # actually in force rather than re-deriving it.
        self.wrench_scale6 = self.wrench_w * np.array(
            [self.s_force] * 3 + [self.s_torque] * 3)
        self.pose_scale6 = self.dim_w * np.array([1.0, 1.0, 1.0,
                                                  self.s_rot, self.s_rot, self.s_rot])
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
        # KERNEL BANDWIDTH, tau = interp_softness x the map's median spacing. A NUMBER applies
        # one bandwidth to the joint 12-D distance (the original). A DICT keyed by block
        # (translation / rotation / force / torque) gives each block its own bandwidth AND its
        # own interpolated target, each measured against that block's own spacing so the number
        # stays dimensionless. Missing blocks default to 1.0.
        soft = c.get('interp_softness', 1.0)
        if isinstance(soft, dict):
            badb = [k for k in soft if k not in [b for b, _ in BLOCKS]]
            if badb:
                raise ValueError(f'estimation.interp_softness keys {badb} not in '
                                 f'{[b for b, _ in BLOCKS]}')
            self.softness_by_block = {b: float(soft.get(b, 1.0)) for b, _ in BLOCKS}
            if any(v < 0 for v in self.softness_by_block.values()):
                raise ValueError('estimation.interp_softness values must be >= 0')
            self.interp_softness = float(np.mean(list(self.softness_by_block.values())))
        else:
            self.softness_by_block = None
            self.interp_softness = float(soft)
        self.min_observations = int(c.get('min_observations', 20))
        # MIXTURE over the finals. `mode_bandwidth` is the mm-equivalent distance below which two
        # finals count as the SAME mode -- it defaults to ransac_tol because that is already this
        # config's answer to "how far apart is a different solution". `mode_min_weight` folds
        # negligible clusters into their neighbours so `ambiguity` stays a probability.
        self.mode_bandwidth = float(c.get('mode_bandwidth', self.ransac_tol))
        self.mode_min_weight = float(c.get('mode_min_weight', 0.02))
        # Fraction of the guess budget spent around caller-supplied `seeds`, and how tightly.
        self.seed_frac = float(c.get('seed_frac', 0.5))
        self.seed_spread = float(c.get('seed_spread', 0.25))
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
        # SUPPORT REFERENCE: how far the k-th neighbour sits for a TYPICAL manifold point. This
        # is the in-distribution scale. A candidate correction whose k-th neighbour is much
        # further away than this is being scored against EXTRAPOLATED evidence -- the residual
        # can still look small (the interpolation happily reaches out to whatever is nearest),
        # which is exactly how a solution latches onto the edge of the map and then drifts
        # further out of distribution with every update.
        # support_k is at least 2: the 1st neighbour of a MANIFOLD point is itself, so a k=1
        # reference would be identically zero and the ratio meaningless. Whatever k is used here
        # must also be the k the queries use, or the ratio compares two different quantities and
        # lands nowhere near 1 for in-distribution data.
        self.support_k = max(self.interp_neighbors, 2)
        d_all = self.tree.query(self.M12, k=self.support_k, workers=-1)[0]
        d_ref = d_all[:, -1]                        # k-th neighbour distance per map point
        self.support_ref = float(max(np.median(d_ref), 1e-9))
        # ... and the reference is DEPTH-CONDITIONAL, not one global median. Measured on the
        # 2026-08 hose data: the map is DENSE in the shallow approach region and SPARSE at
        # depth, so against a global reference a good deep insertion read as "thin support"
        # (ratio ~2) while a bad rim-stuck trial matching useless approach rows read as ~1 --
        # the metric ranked failures BACKWARDS (AUROC 0.39). Normalising each query row by the
        # median k-th-NN distance of manifold points AT ITS OWN DEPTH removes exactly that
        # confound: the ratio then measures off-distribution-ness, not insertion depth.
        xs = self.M12[:, 0]
        edges = np.quantile(xs, np.linspace(0.0, 1.0, 13))       # 12 equal-mass depth bins
        centres, meds = [], []
        for i in range(len(edges) - 1):
            m = (xs >= edges[i]) & (xs <= edges[i + 1] if i == len(edges) - 2
                                    else xs < edges[i + 1])
            if m.sum() >= 10:
                centres.append(float(np.median(xs[m])))
                meds.append(float(max(np.median(d_ref[m]), 1e-9)))
        if len(centres) >= 2:
            self._sup_x = np.asarray(centres)
            self._sup_med = np.asarray(meds)
        else:                                       # degenerate map -- fall back to global
            self._sup_x = np.array([float(xs.min()), float(xs.max())])
            self._sup_med = np.array([self.support_ref, self.support_ref])
        if self.interp_neighbors > 1:
            spacing = float(np.median(d_all[:, 1]))
            self.interp_tau = max(spacing * self.interp_softness, 1e-9)
            log.info('Interpolated matching: k=%d, tau=%.3f mm-eq (median manifold spacing %.3f)',
                     self.interp_neighbors, self.interp_tau, spacing)
            if self.softness_by_block is not None:
                # Each block needs its OWN spacing reference or a bandwidth of 1.0 would mean
                # something different in each: the blocks live at very different scales.
                self.tau_by_block = {}
                sample = self.M12[np.linspace(0, len(self.M12) - 1,
                                              min(len(self.M12), 4000)).astype(int)]
                _, nnb = self.tree.query(sample, k=min(8, len(self.M12)), workers=-1)
                for b, sl in BLOCKS:
                    db = np.linalg.norm(self.M12[nnb][:, 1:, sl] - sample[:, None, sl], axis=2)
                    db = np.where(db > 0, db, np.inf).min(axis=1)
                    db = db[np.isfinite(db)]
                    sp_b = float(np.median(db)) if len(db) else 0.0
                    self.tau_by_block[b] = max(sp_b * self.softness_by_block[b], 1e-12)
                log.info('   per-block bandwidths: %s', {b: round(v, 4) for b, v
                                                         in self.tau_by_block.items()})
                log.warning('Per-block interp_softness shapes the ENERGY paths (grid, '
                            'landscape, commit: argmin). The multi-start ICP keeps the joint '
                            'blend, because it needs a single interpolated TARGET to step '
                            'toward, not just a residual -- so aggregator and argmin will not '
                            'agree exactly while this is set.')
        log.info('Support reference: median %d-th NN distance %.3f mm-eq global, depth-'
                 'conditional %.3f..%.3f across x %.1f..%.1f mm -- a query row whose k-th '
                 'neighbour sits much further than ITS DEPTH\'s reference is extrapolating.',
                 self.support_k, self.support_ref, float(self._sup_med.min()),
                 float(self._sup_med.max()), float(self._sup_x.min()),
                 float(self._sup_x.max()))

    def support_ref_at(self, x_mm):
        """The in-distribution k-th-NN distance at insertion depth x (mm) -- the DENOMINATOR of
        the support ratio. Depth-conditional because map density varies strongly with depth."""
        return np.interp(np.asarray(x_mm, dtype=float), self._sup_x, self._sup_med,
                         left=self._sup_med[0], right=self._sup_med[-1])

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
        return scaled12(v6, self._wrench6(f, tau), self.s_rot, self.dim_w)

    def _wrench6(self, f, tau):
        """The 6 wrench feature columns: representation, block scaling, then per-axis weights."""
        if self.wrench_representation == 'rawcap':
            fm = np.linalg.norm(np.asarray(f, dtype=float), axis=-1)
            tm = np.linalg.norm(np.asarray(tau, dtype=float), axis=-1)
            wf = unit_rows(f, 1.0) * np.minimum(fm / self.rawcap_force_ref_n,
                                                self.rawcap_cap)[..., None]
            wt = unit_rows(tau, 1.0) * np.minimum(tm / self.rawcap_torque_ref_nm,
                                                  self.rawcap_cap)[..., None]
            return np.hstack([wf * self.s_force, wt * self.s_torque]) * self.wrench_w
        return (np.hstack([unit_rows(f, self.s_force), unit_rows(tau, self.s_torque)])
                * self.wrench_w)

    def wrench6_at(self, f_raw, tau_raw, theta6):
        """The wrench FEATURE re-expressed in the frame a candidate correction implies.

        Mirrors _rebase_rows exactly: with T = mats_from_vec6(theta6) and its inverse carrying
        (R, p), the wrench in the corrected frame is f' = R f, tau' = R tau + p x f'. p is the
        translation in METRES (theta is in mm), and it vanishes for a pure-rotation correction,
        which is why a pitch-only estimate needs no lever arm. The feature is rebuilt from the
        rotated RAW wrench, so saturation (rawcap) is applied to the correct vector."""
        th = np.asarray(theta6, dtype=float)
        T = mats_from_vec6(th)
        Ti = np.linalg.inv(T)
        R = Ti[:3, :3]
        p = Ti[:3, 3] / 1000.0                       # theta translation is mm; torque wants m
        f = np.asarray(f_raw, dtype=float) @ R.T
        tau = np.asarray(tau_raw, dtype=float) @ R.T + np.cross(p, f)
        return self._wrench6(f, tau)

    def blend_residual(self, pts, dist=None, nn=None):
        """Per-row residual to the soft-kNN target -- THE quantity every energy sums.

        One bandwidth (interp_softness a number): one interpolated 12-D target, residual is its
        distance. Per-block bandwidths (a dict): each block forms its own target from the SAME
        neighbours using its own distances and tau, and the residual is the L2 over blocks.
        Neighbour SELECTION stays joint in both cases -- it has to, because a candidate
        correction reaches the wrench channels only through the correspondences they share with
        pose."""
        pts = np.asarray(pts, dtype=float)
        k = max(int(self.interp_neighbors), 1)
        if dist is None or nn is None:
            dist, nn = self.tree.query(pts, k=k, workers=-1)
        if np.ndim(dist) == 1:
            dist, nn = dist[:, None], nn[:, None]
        if k <= 1:
            return dist[:, 0]
        M = self.M12[nn]
        if self.softness_by_block is None:
            bw = np.exp(-(dist - dist[:, :1]) / self.interp_tau)
            bw /= bw.sum(axis=1, keepdims=True)
            return np.linalg.norm(np.einsum('mk,mkd->md', bw, M) - pts, axis=1)
        R = M - pts[:, None, :]
        tot = np.zeros(len(pts))
        for b, sl in BLOCKS:
            tau = self.tau_by_block.get(b, self.interp_tau)
            db = np.linalg.norm(R[:, :, sl], axis=2)
            if not np.any(db > 0):
                continue                             # block switched off (zero weight/scale)
            bw = np.exp(-(db - db.min(axis=1, keepdims=True)) / tau)
            bw /= bw.sum(axis=1, keepdims=True)
            tot += np.linalg.norm(np.einsum('mk,mkd->md', bw, M[:, :, sl]) - pts[:, sl],
                                  axis=1) ** 2
        return np.sqrt(tot)

    def prepare_observations(self, vec6, f_raw, tau_raw):
        """Filter (min force) + normalize raw observations -> (vec6, w6) ready for estimate()."""
        vec6 = np.asarray(vec6, dtype=float)
        f_raw, tau_raw = np.asarray(f_raw, dtype=float), np.asarray(tau_raw, dtype=float)
        if self.min_force_n is not None:
            keep = np.linalg.norm(f_raw, axis=1) >= float(self.min_force_n)
            vec6, f_raw, tau_raw = vec6[keep], f_raw[keep], tau_raw[keep]
        # Stash the FILTERED raw wrench: candidate-aware re-basing (wrench6_at) needs the raw
        # vectors, and every caller already hands them to this method. Aligned with the
        # returned rows by construction.
        self.last_raw = (f_raw, tau_raw)
        return vec6, self._wrench6(f_raw, tau_raw)

    # ------------------------------------------------------------------ solver
    def _start_guesses(self, G, seeds=None):
        """The G initial corrections: guess 0 identity, the rest uniform in init_guess_range --
        except that when `seeds` are given, `seed_frac` of the budget is drawn TIGHTLY around
        them instead. That is what makes a disambiguating probe re-examine both live hypotheses
        rather than re-rolling the dice and possibly missing the weaker one entirely."""
        g6 = np.zeros((G, 6))
        rng_d = np.array([float(self.init_range.get(d, 5.0)) for d in self.estimate_dims])
        n_seed = 0
        seeds = [np.asarray(s, dtype=float).ravel() for s in (seeds or [])]
        seeds = [s for s in seeds if s.size == len(self.idx) and np.all(np.isfinite(s))]
        if seeds and G > 2:
            n_seed = min(int(round(self.seed_frac * (G - 1))), G - 2)
            per = max(n_seed // len(seeds), 0)
            n_seed = per * len(seeds)
            for m, s in enumerate(seeds):
                sl = slice(1 + m * per, 1 + (m + 1) * per)
                g6[sl, self.idx] = s + self.rng.normal(
                    0.0, np.maximum(self.seed_spread * rng_d, 1e-6), (per, len(self.idx)))
        for a, (d, j) in enumerate(zip(self.estimate_dims, self.idx)):
            r = rng_d[a]
            g6[1 + n_seed:, j] = self.rng.uniform(-r, r, G - 1 - n_seed)
        if seeds:
            # fancy indexing yields a COPY, so clip-and-assign (np.clip out= would be a no-op)
            g6[:, self.idx] = np.clip(g6[:, self.idx], -rng_d, rng_d)
        return g6, n_seed

    def _mixture(self, finals, r_fin, scale_idx):
        """Mixture over the multi-start finals, in PHYSICAL units.

        Fitted in the mm-equivalent space the solver works in (so one bandwidth covers mm and
        deg alike), then rescaled -- the caller wants mm and deg. Particles are weighted by the
        residual softmax whatever the aggregator is: the mixture describes the POSTERIOR, which
        is a separate question from which single number the aggregator hands back."""
        r_min = float(r_fin.min())
        w = np.exp(-(r_fin - r_min) / max(self.softmax_temp * r_min, 1e-12))
        mix = from_particles(finals, w, bandwidth=max(self.mode_bandwidth, 1e-6),
                             min_weight=self.mode_min_weight)
        inv = 1.0 / np.asarray(scale_idx, dtype=float)
        comps = [Component(c.weight, c.mean * inv, c.cov * np.outer(inv, inv), c.idx, c.w_in)
                 for c in mix.components]
        return Mixture(comps, self.estimate_dims)

    def estimate(self, vec6, w6, seeds=None, raw=None):
        """The belief correction from one set of observations (believed poses + their wrench).

        `vec6` (N,6): BELIEVED connector-wrt-target poses [mm, deg]. `w6` (N,6): the scaled unit
        wrench rows (from prepare_observations). `seeds`: optional hypotheses (each a vector over
        estimate_dims, in mm/deg) to concentrate part of the guess budget around. Returns
        `raw`: the (f, tau) the features came from -- pass it (or leave None to reuse the
        estimator's own `last_raw` from prepare_observations) so the wrench can follow each
        candidate correction into the frame that candidate claims. Returns
        (T_corr, info) -- believed @ T_corr ~= true -- or (None, reason) when there is not enough
        data. info carries theta_corr (the correction in physical units on estimate_dims), inlier
        count, the final residual, and the mixture over the finals."""
        if len(vec6) < self.min_observations:
            return None, f'only {len(vec6)} observations (< min_observations {self.min_observations})'
        if raw is None:
            raw = getattr(self, 'last_raw', None)
            if raw is not None and len(raw[0]) != len(vec6):
                raw = None                       # stale stash: never guess, just skip re-basing

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

        g6, n_seeded = self._start_guesses(G, seeds)          # guess 0 stays identity
        T_corr = mats_from_vec6(g6)

        free = np.zeros(6, dtype=bool)
        free[idx] = True
        res_hist = np.zeros((G, K))
        theta_hist = np.zeros((G, K + 1, len(idx)))       # correction params per iteration (physical)
        theta_hist[:, 0] = g6[:, idx]
        kq = self.interp_neighbors
        for k in range(K):
            C = np.einsum('nij,gjk->gnik', Y, T_corr)
            # The wrench rides with the candidate correction (wrench_follows_correction):
            # the logged rows are in the BELIEVED frame, and this guess claims the true frame
            # is believed (.) T_corr, so the wrench must be re-expressed there before it is
            # compared with the map -- the same re-basing that runs when a correction is
            # committed. Per guess, so it is inside the loop.
            wg = w6
            if self.wrench_follows_correction and raw is not None:
                wg = np.stack([self.wrench6_at(raw[0], raw[1], t6) for t6 in
                               vec6_from_mats(T_corr)]) if T_corr.ndim == 3 else \
                    self.wrench6_at(raw[0], raw[1], vec6_from_mats(T_corr[None])[0])
            pts = scaled12(vec6_from_mats(C), wg, self.s_rot, self.dim_w)
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
            # metric -> physical: divide by the EFFECTIVE per-dim scale (unit conversion x
            # dim weight); non-estimated dims are zeroed below, so their scale never divides
            delta6 = delta12[:, :6] / np.maximum(self.pose_scale6, 1e-12)
            delta6[:, ~free] = 0.0
            T_corr = T_corr @ mats_from_vec6(delta6 * self.step_gain)
            theta_hist[:, k + 1] = vec6_from_mats(T_corr)[:, idx]

        # Aggregate the G final corrections into ONE estimate (distances in mm-equivalent space).
        corr6 = vec6_from_mats(T_corr)                        # (G, 6) physical
        scale = self.pose_scale6
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
        # The MIXTURE over the finals -- see the module note on why a single std is misleading
        # here. WITHIN is the ICP's own repeatability, BETWEEN is the rival-mode ambiguity.
        try:
            mix = self._mixture(finals, r_fin, scale[idx])
            mix_info = {
                'mixture': mix, 'n_mixture_modes': mix.n_modes, 'ambiguity': mix.ambiguity,
                'between_frac': mix.between_frac, 'separation': mix.separation,
                'sigma': {d: float(s) for d, s in zip(self.estimate_dims, mix.sigma())},
                'sigma_within': {d: float(s)
                                 for d, s in zip(self.estimate_dims, mix.sigma(True))},
                'cov_mixture': mix.cov,
            }
        except Exception as exc:                              # noqa: BLE001 -- never fatal
            log.debug('mixture fit skipped (%s)', exc)
            mix_info = {'mixture': None, 'n_mixture_modes': 1, 'ambiguity': 0.0,
                        'between_frac': 0.0, 'separation': 0.0}
        info = {
            'theta_corr': {d: float(theta[j]) for d, j in zip(self.estimate_dims, idx)},
            'inliers': int(best.sum()), 'guesses': G, 'aggregator': self.aggregator,
            'final_residual': final_residual, 'seeded_guesses': int(n_seeded),
            'n_observations': len(vec6), 'recency_half_life_frac': self.recency_half_life_frac,
            'interp_neighbors': self.interp_neighbors, **mix_info,
            # Per-iteration histories (all guesses) for convergence plots: correction params in
            # physical mm/deg on estimate_dims, the mean NN residual, and the RANSAC inlier mask.
            'theta_hist': theta_hist, 'res_hist': res_hist, 'inlier_mask': best,
        }
        return mats_from_vec6(theta), info
