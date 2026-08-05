"""Ablation: exact-NN vs INTERPOLATED correspondence in the manifold ICP, across tunings.

The contact manifold is a FINITE sample of a continuous surface, so exact nearest-neighbour
matching can LATCH onto the single closest sample and quantise the recovered correction by the
local sample spacing. The estimator's optional soft-correspondence feature (interp_neighbors /
interp_softness -- see urlab/skills/manifold.py and analysis/manifold_icp_validation.py) blends
the k nearest manifold points instead, so the match target interpolates between close-enough
samples. This script quantifies what that buys.

It re-runs the SAME validation observations once per VARIANT -- the exact-NN baseline plus a grid
of (interp_neighbors, interp_softness) tunings -- as a PAIRED comparison: the hidden offset of an
observation is data-derived (identical across variants) and every variant re-seeds the SAME rng,
so guesses and RANSAC draws match too. Any error difference is therefore the feature, not luck.

The solver under ablation IS the validated solver: manifold_icp_validation.py is imported and its
loaders and solve_trial are reused wholesale -- there is deliberately no third copy of the
algorithm to drift.

Edit CONFIG below and run:  python analysis/manifold_interp_ablation.py

OUTPUT. A timestamped folder under the base config's output_root:
    ablation_results.csv     -- one row per (variant, observation): true/est/error per dim, residual
    ablation_summary.csv     -- one row per variant: mean & median |error| per dim, mean residual
    abs_error_by_variant.png -- per-dim |error| distributions, baseline vs each tuning
    config.json              -- full provenance (base config + overrides + variants)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt             # noqa: E402
import seaborn as sns                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifold_icp_validation as base      # noqa: E402  (the solver under ablation)

# ---------------------------------------------------------------------------- user configuration
CONFIG = {
    # Everything in manifold_icp_validation.CONFIG applies (CSV paths, scaling, perturb_dims,
    # ICP/RANSAC settings, ...). Set here ONLY what this ablation changes on top of it.
    'overrides': {
        'manifold_csv': r'data/banana_map.csv',
        'validation_csv': r'data/banana_observations.csv',
        # A fixed seed is REQUIRED for the paired comparison -- 0 (nondeterministic) is coerced.
        'random_seed': 7,
    },

    # The ablation grid. Each entry is applied on top of the base config; the FIRST entry should
    # be the exact-NN baseline so every summary line reads against it. interp_softness scales the
    # blend kernel tau = softness x the manifold's median point spacing (see solve_trial).
    'variants': [
        {'label': 'exact-NN (baseline)', 'interp_neighbors': 1},
        {'label': 'k=4  soft=0.5', 'interp_neighbors': 4, 'interp_softness': 0.5},
        {'label': 'k=4  soft=1.0', 'interp_neighbors': 4, 'interp_softness': 1.0},
        {'label': 'k=8  soft=0.5', 'interp_neighbors': 8, 'interp_softness': 0.5},
        {'label': 'k=8  soft=1.0', 'interp_neighbors': 8, 'interp_softness': 1.0},
        {'label': 'k=8  soft=2.0', 'interp_neighbors': 8, 'interp_softness': 2.0},
        {'label': 'k=16 soft=1.0', 'interp_neighbors': 16, 'interp_softness': 1.0},
    ],
    'dpi': 110,
}

# Same two-identity palette as the validation plots: baseline blue, variants orange-accented only
# where a box is the baseline -- identity is carried by the axis labels, colour stays neutral.
C_BOX = '#4C72B0'
C_BASE = '#DD8452'


# ---------------------------------------------------------------------------- run
def run_variant(variant, base_cfg, Vv6, Vw6, Vtrial, groups, tree, M12, seed):
    """All observations through solve_trial under one variant's settings. Paired: fresh rng from
    the SAME seed every call, and a fresh cfg copy so the per-variant tau cache never leaks."""
    cfg = dict(base_cfg)
    cfg.update({k: v for k, v in variant.items() if k != 'label'})
    cfg.pop('_interp_tau', None)
    rng = np.random.default_rng(seed)

    rows = []
    for obs, group in enumerate(groups, start=1):
        sel = np.isin(Vtrial, group)
        label = f'{group[0]}' if len(group) == 1 else f'{group[0]}-{group[-1]}'
        if sel.sum() < 5:
            continue
        r = base.solve_trial(Vv6[sel], Vw6[sel], tree, M12, cfg, rng)
        row = {'variant': variant['label'],
               'interp_neighbors': int(cfg.get('interp_neighbors') or 1),
               'interp_softness': float(cfg.get('interp_softness', 1.0)),
               'observation': obs, 'trials': label, 'n_points': r['n_points'],
               'ransac_inliers': int(r['inliers'].sum()),
               'final_residual_mm_eq': r['final_residual']}
        for j, d in enumerate(r['dims']):
            row[f'theta_true_{d}'] = r['theta_true'][j]
            row[f'theta_est_{d}'] = r['theta_est'][j]
            row[f'error_{d}'] = r['error'][j]
            row[f'abs_error_{d}'] = abs(r['error'][j])
        rows.append(row)
    return rows


def plot_abs_errors(df, dims, out_png, cfg):
    """Per-dim |error| by variant (box + points), residual panel last. One measure across the
    variants, so one neutral colour -- the baseline is picked out by weight, not a new hue."""
    sns.set_theme(style='whitegrid', context='notebook')
    order = list(dict.fromkeys(df['variant']))
    panels = [f'abs_error_{d}' for d in dims] + ['final_residual_mm_eq']
    titles = [f'|error| {d}' for d in dims] + ['final NN residual [mm-eq]']

    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.6), dpi=cfg['dpi'])
    for ax, col, title in zip(np.atleast_1d(axes), panels, titles):
        sns.boxplot(data=df, x=col, y='variant', order=order, color=C_BOX, width=0.55,
                    fliersize=0, ax=ax)
        sns.stripplot(data=df, x=col, y='variant', order=order, color='#333333', size=3,
                      alpha=0.6, jitter=0.18, ax=ax)
        for artist, name in zip(ax.artists if ax.artists else ax.patches, order):
            if 'baseline' in name:
                artist.set_edgecolor(C_BASE)
                artist.set_linewidth(2.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=8)
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_yticklabels([])
    fig.suptitle('Manifold ICP: exact-NN baseline vs interpolated correspondence', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png)
    plt.close(fig)


def main(cfg=CONFIG):
    base_cfg = dict(base.CONFIG)
    base_cfg.update(cfg.get('overrides') or {})
    bad = [d for d in base_cfg['perturb_dims'] if d not in base.DIMS]
    if bad:
        sys.exit(f'perturb_dims {bad} not in {base.DIMS}')
    seed = int(base_cfg.get('random_seed', 0))
    if seed <= 0:
        seed = 7
        print('random_seed 0 (nondeterministic) -> forcing 7: pairing needs identical draws')
    base_cfg['random_seed'] = seed

    out_dir = os.path.join(base_cfg['output_root'],
                           f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_interp_ablation_'
                           f'{base_cfg["cable"]}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as fh:
        json.dump({'base': base_cfg, 'variants': cfg['variants']}, fh, indent=2, default=str)

    Mv6, Mw6, _ = base.load_scaled(base_cfg['manifold_csv'], base_cfg)
    M12 = base._scaled12(Mv6, Mw6, float(base_cfg['scaling_constant_deg_to_mm']))
    if len(M12) < 10:
        sys.exit(f'manifold has only {len(M12)} rows -- not enough to match against')
    tree = cKDTree(M12)
    spacing = float(np.median(tree.query(M12, k=2, workers=-1)[0][:, 1]))
    print(f'manifold: {len(M12)} points, median sample spacing {spacing:.3f} mm-eq '
          f'({base_cfg["manifold_csv"]})')

    Vv6, Vw6, Vtrial = base.load_scaled(base_cfg['validation_csv'], base_cfg, need_trial=True)
    trials = sorted(set(Vtrial.tolist()))
    if base_cfg['trials'] is not None:
        trials = [t for t in trials if t in set(base_cfg['trials'])]
    per_obs = max(1, int(base_cfg.get('trials_per_observation', 1)))
    groups = [trials[i:i + per_obs] for i in range(0, len(trials), per_obs)]
    print(f'validation: {len(Vv6)} rows -> {len(groups)} observation(s); '
          f'{len(cfg["variants"])} variant(s), seed {seed} (paired)\n')

    dims = base_cfg['perturb_dims']
    all_rows = []
    for variant in cfg['variants']:
        rows = run_variant(variant, base_cfg, Vv6, Vw6, Vtrial, groups, tree, M12, seed)
        all_rows.extend(rows)
        dfv = pd.DataFrame(rows)
        stats = ', '.join(f'{d}: {dfv[f"abs_error_{d}"].mean():.3f}' for d in dims)
        print(f'{variant["label"]:<22} mean |error| [{stats}]  '
              f'mean residual {dfv["final_residual_mm_eq"].mean():.3f}')

    if not all_rows:
        sys.exit('no observations processed')
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out_dir, 'ablation_results.csv'), index=False)

    agg = {f'abs_error_{d}': ['mean', 'median'] for d in dims}
    agg['final_residual_mm_eq'] = ['mean']
    summary = (df.groupby(['variant', 'interp_neighbors', 'interp_softness'], sort=False)
               .agg(agg))
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(out_dir, 'ablation_summary.csv'), index=False)

    plot_abs_errors(df, dims, os.path.join(out_dir, 'abs_error_by_variant.png'), cfg)
    print(f'\n{len(df)} rows -> {out_dir}')
    return out_dir


if __name__ == '__main__':
    main()
