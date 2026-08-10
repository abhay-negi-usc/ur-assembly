"""SOLUTION CHECK -- runtime trust scoring for the contact-manifold estimate.

After every estimate, the correction is scored by CHECKING CANDIDATE SOLUTIONS against the
same observations (the 2026-08 offline uncertainty campaign's winners, validated on 147
held-out cases):

  signals (all runtime-computable, no ground truth):
    u_post    posterior std of the across-valley coordinate over N random candidate
              corrections (128 candidates match the exhaustive grid's quality: AUROC 0.84)
    u_spread  softmax-weighted across-valley std of the multi-start finals (free)
    u_split   across-valley disagreement between the estimates supported by the FIRST and
              SECOND temporal halves of the observations (halves differ systematically --
              approach vs deep contact -- unlike bootstrap resamples, which are blind)
    u_res     the final residual (meaningful on a production-process-augmented manifold)
    u_depth   -max corrected insertion depth (shallow attempts are the classic failures)
    u_cfg     (optional) across-valley disagreement with a SECOND estimator using the
              other wrench representation -- the config-ensemble member

  scores (BOTH always computed and logged; `check.method` picks the decision maker):
    rankavg2  mean of the signals' calibration-ECDF values (the deployable form of the
              offline rank-average; pool: post, spread, split [, cfg])
    cauchy    Cauchy combination (Liu & Xie 2019) of the calibration p-values over ALL
              available signals, mapped back to a [0, 1] percentile -- robust to the
              signals' mutual dependence; offline: AUROC 0.87 +/- 0.015, equal to
              rankavg2, but needs no test-batch ranks

Calibration: a JSON file of per-signal reference samples (generated offline from held-out
fixtured campaigns; see data/'test data'/ablation/uncertainty*.csv). WITHOUT the file the
check still logs raw signals; the scores are just reported as n/a.

The check is PURELY OBSERVATIONAL: the correction is ALWAYS applied and the attempt loop
runs exactly as in cable_pick_estimate_assemble -- the check only prints the belief-
correction trust (both scores + signals, labelled LOW when the deciding score exceeds
check.flag_threshold) so the OPERATOR sees it before their y/Enter/q call at the next
prompt. The across-valley coordinate is z - 0.56 * pitch (the measured lever-arm valley)
when estimating [z_mm, pitch_deg]; otherwise the first estimated dim is used.
"""

import json
import os

import numpy as np

from .. import log as urlog
from .manifold import ManifoldEstimator, mats_from_vec6, scaled12, vec6_from_mats

log = urlog.get('solution_check')

VALLEY_MM_PER_DEG = 0.56


class CheckedManifoldEstimator(ManifoldEstimator):
    """ManifoldEstimator + the candidate-check trust stack (config: estimation.check)."""

    def __init__(self, cfg_section):
        c = dict(cfg_section or {})
        chk = dict(c.get('check') or {})
        super().__init__(c)
        self.check_enabled = bool(chk.get('enabled', True))
        method = str(chk.get('method', 'cauchy')).strip().lower()
        if method not in ('cauchy', 'rankavg2'):
            raise ValueError(f"check.method {method!r} must be 'cauchy' or 'rankavg2'")
        self.check_method = method                     # bad values fail HERE, pre-motion
        self.flag_threshold = float(chk.get('flag_threshold', 0.75))
        self.n_candidates = int(chk.get('n_candidates', 128))
        self.candidate_range = dict(chk.get('candidate_range', {}) or {})
        self.calib = None
        path = chk.get('calibration_file')
        if path and os.path.isfile(path):
            with open(path) as fh:
                raw = json.load(fh)
            self.calib = {k: np.sort(np.asarray(v, dtype=float)) for k, v in raw.items()
                          if len(v) >= 10}
            log.info('Check calibration: %s (%s)', path,
                     ', '.join(f'{k}:{len(v)}' for k, v in self.calib.items()))
        elif self.check_enabled:
            log.warning('check.calibration_file missing (%s): signals will be LOGGED '
                        'but the check will never flag.', path)
        # Optional second estimator with the OTHER wrench representation (u_cfg member).
        self._alt = None
        if bool(chk.get('config_disagreement', False)):
            alt_cfg = dict(c)
            alt_cfg.pop('check', None)
            alt_cfg['wrench_representation'] = ('unit' if self.wrench_representation ==
                                                'rawcap' else 'rawcap')
            if alt_cfg['wrench_representation'] == 'unit':
                # the tuned unit-rep scales (the pre-rawcap production values)
                alt_cfg['scaling_constant_unit_force_to_mm'] = 3.0
                alt_cfg['scaling_constant_unit_torque_to_mm'] = 1.0
            self._alt = ManifoldEstimator(alt_cfg)
        self._raw = None                               # stashed by prepare_observations

    # -------------------------------------------------------------- helpers
    def _across(self, theta_by_dim):
        """Across-valley coordinate of per-dim correction values (dict dim -> value)."""
        if 'z_mm' in theta_by_dim and 'pitch_deg' in theta_by_dim:
            return theta_by_dim['z_mm'] - VALLEY_MM_PER_DEG * theta_by_dim['pitch_deg']
        return theta_by_dim[self.estimate_dims[0]]

    def _across_arr(self, arr):
        """Across-valley coordinate for an (N, n_dims) array in estimate_dims order."""
        d = {dim: arr[:, j] for j, dim in enumerate(self.estimate_dims)}
        if 'z_mm' in d and 'pitch_deg' in d:
            return d['z_mm'] - VALLEY_MM_PER_DEG * d['pitch_deg']
        return arr[:, 0]

    def prepare_observations(self, vec6, f_raw, tau_raw):
        vec6 = np.asarray(vec6, dtype=float)
        f_raw = np.asarray(f_raw, dtype=float)
        tau_raw = np.asarray(tau_raw, dtype=float)
        if self.min_force_n is not None and len(f_raw):
            keep = np.linalg.norm(f_raw, axis=1) >= float(self.min_force_n)
            self._raw = (vec6[keep], f_raw[keep], tau_raw[keep])
        else:
            self._raw = (vec6, f_raw, tau_raw)
        return super().prepare_observations(vec6, f_raw, tau_raw)

    def _signals(self, vec6, w6, T_corr, info):
        s = {'u_res': float(info['final_residual'])}
        finals = np.asarray(info['theta_hist'], dtype=float)[:, -1, :]
        r_fin = np.asarray(info['res_hist'], dtype=float)[:, -1]
        wgt = np.exp(-(r_fin - r_fin.min()) / max(self.softmax_temp * r_fin.min(), 1e-12))
        wgt /= wgt.sum()
        fa = self._across_arr(finals)
        mu = float((fa * wgt).sum())
        s['u_spread'] = float(np.sqrt((((fa - mu) ** 2) * wgt).sum()))

        # candidate check: N random corrections over the candidate box (+ the solution)
        rng = np.random.default_rng(0)
        K = max(self.n_candidates, 8)
        g6 = np.zeros((K, 6))
        for d, j in zip(self.estimate_dims, self.idx):
            r = float(self.candidate_range.get(d) or self.init_range.get(d, 5.0))
            g6[1:, j] = rng.uniform(-r, r, K - 1)
        g6[0, self.idx] = [info['theta_corr'][d] for d in self.estimate_dims]
        Y = mats_from_vec6(vec6)
        C = np.einsum('nij,kjl->knil', Y, mats_from_vec6(g6))
        pts = scaled12(vec6_from_mats(C), w6, self.s_rot).reshape(-1, 12)
        kq = self.interp_neighbors
        dist, nn = self.tree.query(pts, k=kq, workers=-1)
        if kq > 1:
            bw = np.exp(-(dist - dist[:, :1]) / self.interp_tau)
            bw /= bw.sum(axis=1, keepdims=True)
            tgt = np.einsum('mk,mkd->md', bw, self.M12[nn])
            d1 = np.linalg.norm(tgt - pts, axis=1)
        else:
            d1 = dist[:, 0] if dist.ndim > 1 else dist
        RE = d1.reshape(K, -1)                          # per-candidate, per-row energy
        E = RE.mean(axis=1)
        ca = self._across_arr(g6[:, self.idx])
        w = np.exp(-(E - E.min()) / (0.15 * max(float(E.min()), 0.1)))
        w /= w.sum()
        mu = float((ca * w).sum())
        s['u_post'] = float(np.sqrt((((ca - mu) ** 2) * w).sum()))
        n = RE.shape[1]
        if n >= 8:
            ka = int(np.argmin(RE[:, :n // 2].mean(axis=1)))
            kb = int(np.argmin(RE[:, n // 2:].mean(axis=1)))
            s['u_split'] = float(abs(ca[ka] - ca[kb]))

        xc = vec6_from_mats(Y @ np.asarray(T_corr, dtype=float))[:, 0]
        s['u_depth'] = float(-xc.max()) if len(xc) else 0.0

        if self._alt is not None and self._raw is not None:
            v6a, w6a = self._alt.prepare_observations(*self._raw)
            Ta, ia = self._alt.estimate(v6a, w6a)
            if Ta is not None:
                da = self._across(ia['theta_corr'])
                dm = self._across(info['theta_corr'])
                s['u_cfg'] = float(abs(dm - da))
        return s

    def _scores(self, signals):
        """(rankavg2, cauchy) in [0, 1]; None when calibration is unavailable."""
        if not self.calib:
            return None, None
        p_ecdf = {}
        for k, v in signals.items():
            ref = self.calib.get(k)
            if ref is not None:
                p_ecdf[k] = (np.searchsorted(ref, v, side='right') + 1.0) / (len(ref) + 2.0)
        if not p_ecdf:
            return None, None
        rank_pool = [k for k in ('u_post', 'u_spread', 'u_split', 'u_cfg') if k in p_ecdf]
        rank_score = float(np.mean([p_ecdf[k] for k in rank_pool])) if rank_pool else None
        p = np.clip(1.0 - np.array(list(p_ecdf.values())), 1e-4, 1 - 1e-4)
        stat = float(np.tan((0.5 - p) * np.pi).mean())
        cauchy_score = float(0.5 + np.arctan(stat) / np.pi)
        return rank_score, cauchy_score

    # -------------------------------------------------------------- estimate
    def estimate(self, vec6, w6):
        T_corr, info = super().estimate(vec6, w6)
        if T_corr is None or not self.check_enabled:
            return T_corr, info
        try:
            signals = self._signals(vec6, w6, T_corr, info)
            rank_score, cauchy_score = self._scores(signals)
        except Exception as exc:                        # the check must never kill a run
            log.warning('solution check failed (%s) -- correction passed through.', exc)
            return T_corr, info
        chosen = cauchy_score if self.check_method == 'cauchy' else rank_score
        low = chosen is not None and chosen > self.flag_threshold
        info['check'] = {'signals': signals, 'rankavg2': rank_score,
                         'cauchy': cauchy_score, 'method': self.check_method,
                         'score': chosen, 'flag_threshold': self.flag_threshold,
                         'flagged': low}
        # OBSERVATIONAL ONLY: the correction is applied regardless -- this line is the
        # operator's trust readout for the next y/Enter/q prompt.
        log.info('solution check [trust %s]: rankavg2=%s cauchy=%s (deciding: %s, '
                 'label threshold %.2f) correction=%s signals=%s',
                 'LOW' if low else 'ok',
                 'n/a' if rank_score is None else f'{rank_score:.2f}',
                 'n/a' if cauchy_score is None else f'{cauchy_score:.2f}',
                 self.check_method, self.flag_threshold,
                 {k: round(v, 3) for k, v in info['theta_corr'].items()},
                 {k: round(v, 3) for k, v in signals.items()})
        return T_corr, info
