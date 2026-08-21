"""Run figures for the estimate-while-assemble apps.

Both plots are BEST-EFFORT: a plotting problem (e.g. seaborn not installed on the robot box)
is logged and skipped, never allowed to kill a hardware run.  matplotlib is imported lazily
inside each function for the same reason.
"""

import itertools
import os

import numpy as np

from .. import log as urlog

log = urlog.get('estimate-plots')


def plot_estimate(path, dims, info):
    """Per-attempt convergence figure: one panel per estimated dim (correction vs ICP
    iteration -- every guess faint, RANSAC consensus bold, dashed zero) plus the log-scale
    NN residual."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        th, inl = info['theta_hist'], info['inlier_mask']
        res = np.maximum(info['res_hist'], 1e-6)
        sns.set_theme(style='whitegrid')
        fig, axes = plt.subplots(len(dims) + 1, 1, figsize=(9.0, 2.6 * (len(dims) + 1)),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        it = np.arange(th.shape[1])
        for j, (ax, dim) in enumerate(zip(axes[:-1], dims)):
            unit = 'deg' if dim.endswith('_deg') else 'mm'
            ax.axhline(0.0, ls='--', lw=1.0, color='#888888', zorder=1)
            for g in range(th.shape[0]):
                ax.plot(it, th[g, :, j], color='#4C72B0', alpha=0.07, lw=1.0, zorder=2)
            ax.plot(it, th[inl, :, j].mean(axis=0), color='#DD8452', lw=2.4, zorder=3)
            lim = max(float(np.abs(th[..., j]).max()), 1e-3) * 1.05
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(f'{dim} corr [{unit}]')
            ax.set_title(f'correction[{dim}] = {info["theta_corr"][dim]:+.3f} {unit}',
                         fontsize=10, loc='left')
        ax = axes[-1]
        it_r = np.arange(1, res.shape[1] + 1)
        for g in range(res.shape[0]):
            ax.plot(it_r, res[g], color='#4C72B0', alpha=0.07, lw=1.0, zorder=2)
        ax.plot(it_r, res[inl].mean(axis=0), color='#DD8452', lw=2.4, zorder=3)
        ax.set_yscale('log')
        ax.set_ylabel('mean NN residual [mm-eq]')
        ax.set_xlabel('ICP iteration')
        fig.suptitle(f'belief correction ({info["inliers"]}/{info["guesses"]} inliers, '
                     f'residual {info["final_residual"]:.3f})', y=0.995)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as exc:                       # noqa: BLE001 -- plotting is never fatal
        log.warning('estimate plot skipped (%s)', exc)


def plot_run(path, dims, corr_track, res_agg, res_all, status=None, live_path=None):
    """Run-level figure, re-saved after every estimate (+ optional atomic LIVE copy).

    No ground truth exists after a real pick, so the tracks show the BELIEF's movement: the
    cumulative applied correction per estimated dim (x = attempt, 0 = the initial in-hand
    estimate), the ICP residual per attempt (every guess faint, the aggregated pick bold,
    log y), and for 2+ dims a phase plot of the cumulative correction pairs (origin = no
    correction)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ct = np.asarray(corr_track, dtype=float)       # (attempts+1, len(dims)), row 0 = zeros
        x = np.arange(len(ct))
        n_left = len(dims)
        pairs = list(itertools.combinations(range(len(dims)), 2))
        ncols = 3 if pairs else 2
        fig = plt.figure(figsize=(4.6 * ncols + 1.0, max(2.3 * n_left, 6.0)))
        gs = fig.add_gridspec(2 * n_left, ncols, width_ratios=[1.2, 1.0, 0.95][:ncols])
        axes = []
        for i in range(n_left):
            axes.append(fig.add_subplot(gs[2 * i:2 * i + 2, 0],
                                        sharex=axes[0] if axes else None))
        for j, (ax, dim) in enumerate(zip(axes, dims)):
            unit = 'deg' if dim.endswith('_deg') else 'mm'
            ax.axhline(0.0, ls='--', lw=1.0, color='#888888', zorder=1)
            ax.plot(x, ct[:, j], 'o-', color='#4C72B0', zorder=2)
            lim = max(float(np.abs(ct[:, j]).max()), 1e-3) * 1.15
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(f'cumulative {dim} corr [{unit}]')
            ax.tick_params(labelbottom=(j == n_left - 1))
        axes[-1].set_xticks(x)
        axes[-1].set_xlabel('attempt (0 = initial in-hand estimate)')

        ax_r = fig.add_subplot(gs[:, 1])
        labelled = False
        for k, rg in enumerate(res_all or []):
            rg = np.maximum(np.asarray(rg, dtype=float), 1e-6)
            if not len(rg):
                continue
            jit = (np.arange(len(rg)) / max(len(rg) - 1, 1) - 0.5) * 0.3
            ax_r.scatter(k + 1 + jit, rg, s=7, color='#55A868', alpha=0.25, lw=0, zorder=1,
                         label=None if labelled else 'all guesses')
            labelled = True
        r = np.maximum(np.asarray(res_agg, dtype=float), 1e-6)
        ax_r.plot(np.arange(1, len(r) + 1), r, 'o-', color='#55A868', zorder=2,
                  label='aggregated')
        if labelled:
            ax_r.legend(fontsize=8)
        ax_r.set_yscale('log')
        ax_r.set_xticks(np.arange(1, len(r) + 1))
        ax_r.set_xlabel('attempt')
        ax_r.set_ylabel('ICP residual [mm-eq]')

        bounds = np.linspace(0, 2 * n_left, len(pairs) + 1).astype(int) if pairs else []
        for pi, (pa, pb) in enumerate(pairs):
            axp = fig.add_subplot(gs[bounds[pi]:bounds[pi + 1], 2])
            axp.axhline(0.0, ls=':', lw=0.8, color='#aaaaaa', zorder=1)
            axp.axvline(0.0, ls=':', lw=0.8, color='#aaaaaa', zorder=1)
            axp.plot(ct[:, pa], ct[:, pb], '-', color='#4C72B0', lw=1.0, zorder=2)
            axp.scatter(ct[1:, pa], ct[1:, pb], s=20, color='#4C72B0', zorder=3)
            axp.scatter([0.0], [0.0], s=40, marker='s', color='#DD8452', zorder=4,
                        label='initial')
            axp.scatter([ct[-1, pa]], [ct[-1, pb]], s=80, marker='*', color='#55A868',
                        zorder=5, label='latest')
            for k in range(1, len(ct)):
                axp.annotate(str(k), (ct[k, pa], ct[k, pb]), textcoords='offset points',
                             xytext=(4, 3), fontsize=7, color='#444444')
            la = max(float(np.abs(ct[:, pa]).max()), 1e-3) * 1.15
            lb = max(float(np.abs(ct[:, pb]).max()), 1e-3) * 1.15
            axp.set_xlim(-la, la)                  # symmetric: no-correction is the centre
            axp.set_ylim(-lb, lb)
            axp.set_xlabel(f'{dims[pa]} corr '
                           f'[{"deg" if dims[pa].endswith("_deg") else "mm"}]', fontsize=8)
            axp.set_ylabel(f'{dims[pb]} corr '
                           f'[{"deg" if dims[pb].endswith("_deg") else "mm"}]', fontsize=8)
            axp.tick_params(labelsize=7)
            if pi == 0:
                axp.legend(fontsize=7, loc='best')

        fig.suptitle('assembly run: belief corrections per attempt (no ground truth)', y=0.995)
        if status:
            fig.text(0.99, 0.965, status, ha='right', fontsize=9, color='#333333')
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        if live_path:
            tmp = live_path + '.tmp'               # temp + os.replace: viewers never see a
            fig.savefig(tmp, dpi=110, format='png')   # half-written PNG
            os.replace(tmp, live_path)
        plt.close(fig)
    except Exception as exc:                       # noqa: BLE001 -- plotting is never fatal
        log.warning('run plot skipped (%s)', exc)
