"""TILE GRASP PLOT -- every trial, the mean and a +-1 sigma band, per compliant phase.

Plots apps/tile_assembly_experiment runs too: the phases are taken from the data (insert, fasten,
withdraw there), and the pose figure names the pose the run zeroed on (grasped / seated).

    python -m urlab.apps.tile_grasp_plot data/experiments/tile_grasp_experiment_<stamp>
    python -m urlab.apps.tile_grasp_plot <run_dir> --align end --include-failed

Reads the attempt_NN.npz files apps/tile_grasp_experiment writes and draws, into <run_dir>/plots/:

    pose.png        the coupler pose RELATIVE TO THE GRASPED POSE (zero = where it seated), with the
                    admittance reference dashed -- the gap between them is what compliance took up
    wrench.png      the wrench at the mating point, coupler axes
    compliance.png  the admittance law's yield (adm.delta): how far it moved the arm off its
                    reference, directly
    bands.npz       the grid, mean, std and count behind every panel, for re-plotting elsewhere
    assembly_error.png  (tile assembly runs with the inspection) the tile's inspected pose in the
                    calibrated goal frame, per trial, with the mean and the +-1 sigma band

Columns are the phases (mate, lift, place, withdraw), rows the six components. Each phase is put on
a common time axis -- seconds since the phase STARTED (`--align start`, the default) or until it
ENDED (`--align end`, e.g. lined up on the lock) -- and the band is computed wherever at least two
trials have data. Only successful trials are drawn unless --include-failed.
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np

from .. import log as urlog

log = urlog.get('tile-grasp-plot')

PHASE_ORDER = ('mate', 'lift', 'place', 'insert', 'fasten', 'withdraw')
FIGURES = {
    'pose': ('pose_rel', ['x [mm]', 'y [mm]', 'z [mm]', 'roll [deg]', 'pitch [deg]', 'yaw [deg]'],
             'Coupler pose relative to the {zero} pose (dashed: admittance reference)'),
    'wrench': ('wrench', ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Tx [Nm]', 'Ty [Nm]', 'Tz [Nm]'],
               'Wrench at the mating point (coupler axes)'),
    'compliance': ('delta_mm_deg', ['x [mm]', 'y [mm]', 'z [mm]', 'rx [deg]', 'ry [deg]',
                                    'rz [deg]'],
                   'Admittance yield -- how far compliance moved the arm off its reference'),
}


def load_trials(run_dir, include_failed=False):
    """[{'attempt', 'success', arrays...}] for every attempt_NN.npz, successful ones only unless
    include_failed."""
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, 'attempt_*.npz'))):
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        meta = json.loads(str(d.pop('meta')))
        if meta.get('success') or include_failed:
            d.update(attempt=meta.get('attempt'), success=bool(meta.get('success')),
                     meta=meta)
            out.append(d)
    return out


def phase_series(trial, phase, key, align='start'):
    """(t, values (n, 6)) of one trial's phase: t in seconds since the phase's first sample, or
    until its last (`align='end'`, so t <= 0). Empty arrays when the phase is absent. Pure."""
    sel = trial['phase'] == phase
    if not np.any(sel):
        return np.zeros(0), np.zeros((0, 6))
    t = trial['t'][sel]
    t = t - (t[0] if align == 'start' else t[-1])
    return t, np.asarray(trial[key], dtype=float)[sel]


def band(trials, phase, key, align='start', dt=0.01):
    """(grid, stack (n_trials, n_grid, 6), mean, std, count) for one phase and signal. Each trial
    is interpolated onto the grid inside its own time range and NaN outside, so trials of
    different lengths line up; mean needs one trial at a grid point, std needs two. Pure."""
    series = [phase_series(tr, phase, key, align) for tr in trials]
    series = [(t, v) for t, v in series if len(t) >= 2]
    if not series:
        return np.zeros(0), np.zeros((0, 0, 6)), np.zeros((0, 6)), np.zeros((0, 6)), np.zeros(0)
    lo = min(t[0] for t, _ in series)
    hi = max(t[-1] for t, _ in series)
    # An exact-length grid ending EXACTLY on hi, and a tolerance on the range test: arange drifts,
    # and a grid point a hair past a trial's last sample would drop the very instant the phases
    # are aligned on (t = 0 under align='end').
    grid = np.linspace(lo, hi, int(round((hi - lo) / dt)) + 1)
    eps = 1e-9
    stack = np.full((len(series), len(grid), 6), np.nan)
    for i, (t, v) in enumerate(series):
        inside = (grid >= t[0] - eps) & (grid <= t[-1] + eps)
        for c in range(6):
            stack[i, inside, c] = np.interp(grid[inside], t, v[:, c])
    count = np.sum(np.isfinite(stack[:, :, 0]), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # all-NaN columns are expected
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0, ddof=1) if len(series) > 1 else np.full_like(mean, np.nan)
    std[count < 2] = np.nan
    return grid, stack, mean, std, count


ERROR_LABELS = ['x [mm]', 'y [mm]', 'z [mm]', 'roll [deg]', 'pitch [deg]', 'yaw [deg]']


def plot_assembly_errors(trials, path):
    """assembly_error.png: each component of the inspected assembly error (assy_err6) against the
    trial number, with the mean and +-1 sigma over the trials that were measured. Returns the
    path, or None when no trial carries a measurement."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    rows = [(tr['attempt'], np.asarray(tr['assy_err6'], dtype=float)) for tr in trials
            if 'assy_err6' in tr]
    if not rows:
        return None
    n = np.array([r[0] for r in rows])
    E = np.array([r[1] for r in rows]).reshape(-1, 6)
    fig, axes = plt.subplots(6, 1, figsize=(7.5, 12.5), sharex=True)
    for c, ax in enumerate(axes):
        ok = np.isfinite(E[:, c])
        if np.any(ok):
            mean = float(np.mean(E[ok, c]))
            std = float(np.std(E[ok, c], ddof=1)) if np.sum(ok) > 1 else float('nan')
            if np.isfinite(std):
                ax.axhspan(mean - std, mean + std, color='C0', alpha=0.2, lw=0,
                           label='mean $\\pm$ 1$\\sigma$')
            ax.axhline(mean, color='C0', lw=1.4,
                       label=f'mean {mean:+.2f}' + (f', $\\sigma$ {std:.2f}'
                                                    if np.isfinite(std) else ''))
            ax.plot(n[ok], E[ok, c], 'o', color='k', ms=4)
            ax.legend(loc='best', fontsize=7)
        ax.axhline(0.0, color='k', lw=0.6, alpha=0.5)
        ax.set_ylabel(ERROR_LABELS[c])
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('trial')
    axes[-1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    measured = int(np.sum(np.all(np.isfinite(E), axis=1)))
    fig.suptitle(f'Assembly error -- tile in the calibrated goal frame (zero = as calibrated); '
                 f'{measured} of {len(rows)} trial(s) measured', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_run(run_dir, align='start', include_failed=False, dt=0.01):
    """Draw pose / wrench / compliance figures and bands.npz into <run_dir>/plots. Returns the
    paths written."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    trials = load_trials(run_dir, include_failed)
    if not trials:
        raise ValueError(f'no {"" if include_failed else "successful "}trials in {run_dir}')
    # Known phases in their cycle order, then anything else in the order the data first shows it.
    seen = []
    for tr in trials:
        for ph in dict.fromkeys(tr['phase'].tolist()):
            if ph not in seen:
                seen.append(ph)
    phases = [ph for ph in PHASE_ORDER if ph in seen] + [ph for ph in seen
                                                          if ph not in PHASE_ORDER]
    zero = trials[0]['meta'].get('zero', 'grasped')
    out_dir = os.path.join(run_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    written, saved = [], {}
    xlabel = 's since phase start' if align == 'start' else 's to phase end'
    for fig_name, (key, labels, title) in FIGURES.items():
        fig, axes = plt.subplots(6, len(phases), figsize=(4.2 * len(phases), 12.5),
                                 sharex='col', squeeze=False)   # own y per panel: phases differ a lot
        for j, phase in enumerate(phases):
            grid, stack, mean, std, count = band(trials, phase, key, align, dt)
            saved[f'{fig_name}.{phase}.grid'] = grid
            saved[f'{fig_name}.{phase}.mean'] = mean
            saved[f'{fig_name}.{phase}.std'] = std
            saved[f'{fig_name}.{phase}.count'] = count
            ref = band(trials, phase, 'ref_rel', align, dt) if key == 'pose_rel' else None
            for c in range(6):
                ax = axes[c, j]
                for i in range(stack.shape[0]):
                    ax.plot(grid, stack[i, :, c], color='0.6', lw=0.6, alpha=0.5)
                if len(grid):
                    ax.fill_between(grid, mean[:, c] - std[:, c], mean[:, c] + std[:, c],
                                    color='C0', alpha=0.25, lw=0, label='mean $\\pm$ 1$\\sigma$')
                    ax.plot(grid, mean[:, c], color='C0', lw=1.6, label='mean')
                if ref is not None and len(ref[0]):
                    ax.plot(ref[0], ref[2][:, c], color='C3', lw=1.2, ls='--',
                            label='reference (mean)')
                if key != 'wrench':
                    ax.axhline(0.0, color='k', lw=0.6, alpha=0.5)
                if c == 0:
                    ax.set_title(f'{phase} (n={stack.shape[0]})')
                if j == 0:
                    ax.set_ylabel(labels[c])
                if c == 5:
                    ax.set_xlabel(xlabel)
                ax.grid(alpha=0.3)
        axes[0, 0].legend(loc='best', fontsize=7)
        fig.suptitle(f'{title.format(zero=zero)} -- {len(trials)} trial(s)', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = os.path.join(out_dir, f'{fig_name}.png')
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    bands = os.path.join(out_dir, 'bands.npz')
    np.savez_compressed(bands, align=align, attempts=[tr['attempt'] for tr in trials], **saved)
    written.append(bands)
    err = plot_assembly_errors(trials, os.path.join(out_dir, 'assembly_error.png'))
    if err:
        written.append(err)
    return written


def main():
    ap = argparse.ArgumentParser(description='Plot a tile grasp experiment: every trial, the mean '
                                             'and a +-1 sigma band, per compliant phase.')
    ap.add_argument('run_dir', help='data/experiments/tile_grasp_experiment_<stamp>')
    ap.add_argument('--align', choices=('start', 'end'), default='start',
                    help='line each phase up on its start (default) or its end')
    ap.add_argument('--include-failed', action='store_true',
                    help='also draw attempts whose pickup failed')
    args = ap.parse_args()
    for path in plot_run(args.run_dir, args.align, args.include_failed):
        print(path)


if __name__ == '__main__':
    main()
