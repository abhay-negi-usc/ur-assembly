"""SUCCESS-BASIN analysis from contact-manifold data.

The manifold CSV (connector pose wrt the target + wrench, one row per contact sample) is a
flat cloud with no per-trial grouping, so the basin is derived from the INSERTION-DEPTH
FUNNEL: x (the insertion axis) is the progress variable, and for each depth bin the envelope
of |y|, |z|, |roll|, |pitch|, |yaw| over the samples in that bin is the offset the physical
mechanics admitted at that depth. The funnel narrows toward the goal -- that narrowing IS the
success basin the hardware provides for free.

Given a GOAL TOLERANCE (translation mm + rotation deg, i.e. "seated" means every dimension
within tolerance at the goal depth), the script reports:

  * CAPTURE DEPTH x_c: the shallowest depth from which every deeper bin's envelope stays
    inside the tolerance on ALL dimensions -- once inserted past x_c, the data says the
    mechanics alone funnel the part into the tolerance box. Per-dimension capture depths
    identify the binding dimension.
  * BASIN UPPER BOUNDS: the per-dimension envelope at the FUNNEL MOUTH (the shallowest
    contact bin) -- the largest offsets from which the recorded insertions still proceeded.
  * The full per-bin envelope table (CSV) and figures: funnel profiles per dimension with the
    tolerance and capture depth marked, the joint z-pitch extent at selected depths (the
    coupled valley), the per-bin sample support, and a DEPTH MAP -- the deepest insertion
    reached per (pitch, z) cell with iso-depth contours: the basin as a surface, its diagonal
    ridge quantifying the z-pitch tradeoff (slope fitted and annotated in mm/deg).

Definitions are data-honest, not causal: an envelope says "states this far off DID occur and
insertion continued", not "any state inside is guaranteed to succeed". Free-space samples
(|F| below --min-force) are excluded so approach scatter does not inflate the mouth.

Usage:
    python analysis/success_basin.py                          # defaults below
    python analysis/success_basin.py --csv <manifold.csv> --tol-mm 1.0 --tol-deg 1.0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt             # noqa: E402

POSE_COLS = [f'connector_target_{d}' for d in
             ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')]
FORCE_COLS = [f'wrench_connector_{a}' for a in ('fx', 'fy', 'fz')]
DIMS = ('y', 'z', 'roll', 'pitch', 'yaw')            # basin dims; x is the progress variable
UNITS = ('mm', 'mm', 'deg', 'deg', 'deg')
DEFAULT_CSV = os.path.join('data', 'test data', 'banana_manifold_20260806.csv')


def load(path):
    """(pose_all (N,6) [x,y,z mm, roll,pitch,yaw deg], fmag (N,)) in FILE ORDER -- the row
    order is time order (only insert/hold phases are logged), which the initial-offset
    analysis depends on for trial segmentation."""
    pose, f = [], []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in POSE_COLS + FORCE_COLS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f'{path}: missing column(s), first: {missing[0]}')
        for rec in reader:
            try:
                pose.append([float(rec[c]) for c in POSE_COLS])
                f.append([float(rec[c]) for c in FORCE_COLS])
            except (TypeError, ValueError):
                continue
    return np.asarray(pose), np.linalg.norm(np.asarray(f), axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--tol-mm', type=float, default=1.0, help='goal tolerance, y/z [mm]')
    ap.add_argument('--tol-deg', type=float, default=1.0, help='goal tolerance, r/p/y [deg]')
    ap.add_argument('--x-goal', type=float, default=None,
                    help='goal depth [mm]; default = the deepest populated bin (data support '
                         'permitting; the true seat is x = 0)')
    ap.add_argument('--tol-x', type=float, default=None,
                    help='goal tolerance on the INSERTION axis [mm]: the goal region becomes '
                         'x >= -tol_x (within tol_x of the seat at x = 0), instead of the '
                         'deepest populated bin. Overrides --x-goal.')
    ap.add_argument('--x-band', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                    help='explicit goal BAND on x [mm], e.g. --x-band -1 -0.5: the goal '
                         'region is LO <= x <= HI. Overrides --tol-x and --x-goal.')
    ap.add_argument('--bin-mm', type=float, default=1.0, help='depth bin width [mm]')
    ap.add_argument('--percentile', type=float, default=95.0,
                    help='envelope percentile of |offset| per bin (100 = raw max)')
    ap.add_argument('--min-force', type=float, default=3.0,
                    help='contact filter [N]; 0 keeps free-space samples too')
    ap.add_argument('--min-bin-n', type=int, default=50,
                    help='bins with fewer samples are ignored (unsupported)')
    ap.add_argument('--success-x', type=float, default=None,
                    help='initial-offset analysis: a trial SUCCEEDS if its deepest sample '
                         'reaches x >= this [mm]. Default: the goal region boundary '
                         '(--x-band lo / -tol_x), else 0.5 mm short of the deepest coverage.')
    ap.add_argument('--seg-jump-mm', type=float, default=3.0,
                    help='initial-offset analysis: a backward x jump larger than this splits '
                         'trials (rows are time-ordered; retract/approach is never logged)')
    ap.add_argument('--out', default=None, help='output dir; default analysis/<timestamp>')
    args = ap.parse_args()

    pose_all, fmag = load(args.csv)
    contact = fmag >= args.min_force if args.min_force > 0 else np.ones(len(fmag), bool)
    pose = pose_all[contact]
    print(f'{len(pose)} contact samples of {len(pose_all)} rows ({args.csv})')
    x, off = pose[:, 0], np.abs(pose[:, 1:])
    tol = np.array([args.tol_mm, args.tol_mm, args.tol_deg, args.tol_deg, args.tol_deg])

    edges = np.arange(np.floor(x.min()), np.ceil(x.max()) + args.bin_mm, args.bin_mm)
    centers, counts, env, env_max = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < args.min_bin_n:
            continue
        centers.append(0.5 * (lo + hi))
        counts.append(int(m.sum()))
        env.append(np.percentile(off[m], args.percentile, axis=0))
        env_max.append(off[m].max(axis=0))
    centers, counts = np.asarray(centers), np.asarray(counts)
    env, env_max = np.asarray(env), np.asarray(env_max)   # (B, 5), shallow -> deep
    if not len(centers):
        raise SystemExit('no bin has enough samples -- lower --min-bin-n or --min-force')

    # Goal region: an explicit x band (--x-band), else x within tol_x of the seat (--tol-x),
    # else the bin containing --x-goal, else the deepest populated bin.
    if args.x_band is not None or args.tol_x is not None:
        blo, bhi = sorted(args.x_band) if args.x_band is not None else (-args.tol_x, np.inf)
        mg = (x >= blo) & (x <= bhi)
        n_goal = int(mg.sum())
        band = f'{blo:g} <= x <= {bhi:g}' if np.isfinite(bhi) else f'x >= {blo:g}'
        gi = len(centers) - 1                          # capture walk starts at the deepest bin
        x_goal = blo
        if n_goal >= args.min_bin_n:
            goal_env = np.percentile(off[mg], args.percentile, axis=0)
            goal_note = f'goal region {band} mm (n={n_goal})'
        else:
            goal_env = np.full(5, np.inf)              # unreached goal: nothing can capture
            goal_note = (f'goal region {band} mm UNREACHED -- only {n_goal} contact '
                         f'sample(s) there (< --min-bin-n {args.min_bin_n}); the deepest '
                         f'sample is x = {x.max():.2f} mm')
            print(f'GOAL UNREACHED: {goal_note}')
    else:
        gi = len(centers) - 1 if args.x_goal is None else int(np.argmin(np.abs(centers -
                                                                               args.x_goal)))
        x_goal = centers[gi]
        goal_env = env[gi]
        goal_note = f'goal depth x = {x_goal:.1f} mm (deepest populated bin)'
        if x_goal < -args.bin_mm:
            print(f'NOTE: goal depth {x_goal:.1f} mm is short of the true seat (x = 0) -- the '
                  'data does not reach seating; bounds are relative to the deepest coverage.')

    # Per-dim capture depth: within tolerance AT the goal, then walk from the goal bin toward
    # the mouth while the envelope stays inside tolerance; the shallowest bin of that
    # contiguous run is where the funnel captures.
    cap = np.full(5, np.nan)
    for d in range(5):
        j = gi
        if goal_env[d] > tol[d] or env[gi, d] > tol[d]:
            continue                                   # not within tol even AT the goal
        while j > 0 and env[j - 1, d] <= tol[d]:
            j -= 1
        cap[d] = centers[j]
    x_c = np.nanmax(cap) if np.all(np.isfinite(cap)) else np.nan

    print(f'\n{goal_note} | tolerance: +/-{args.tol_mm} mm, '
          f'+/-{args.tol_deg} deg | envelope: p{args.percentile:g} per {args.bin_mm:g} mm bin')
    print(f'{"dim":>6s} {"unit":>4s} {"mouth bound":>12s} {"mouth max":>10s} '
          f'{"@goal":>8s} {"capture x":>10s}')
    for d, (name, unit) in enumerate(zip(DIMS, UNITS)):
        c = f'{cap[d]:+9.1f}' if np.isfinite(cap[d]) else '   never'
        g = f'{goal_env[d]:8.2f}' if np.isfinite(goal_env[d]) else '     n/a'
        print(f'{name:>6s} {unit:>4s} {env[0, d]:12.2f} {env_max[0, d]:10.2f} '
              f'{g} {c:>10s}')
    if np.isfinite(x_c):
        print(f'\nCAPTURE DEPTH (all dims within tolerance from here on): x_c = {x_c:+.1f} mm '
              f'-> insert {x_goal - x_c:.1f} mm past capture to the goal')
    else:
        bad = ', '.join(DIMS[d] for d in range(5) if not np.isfinite(cap[d]))
        print(f'\nNO capture depth: {bad} exceed(s) the tolerance even at the goal depth -- '
              'loosen the tolerance or collect deeper data.')

    out = args.out or os.path.join('analysis', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'success_basin.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['x_mm', 'n'] + [f'p_{n}_{u}' for n, u in zip(DIMS, UNITS)]
                   + [f'max_{n}_{u}' for n, u in zip(DIMS, UNITS)])
        for i in range(len(centers)):
            w.writerow([f'{centers[i]:.2f}', counts[i]]
                       + [f'{v:.3f}' for v in env[i]] + [f'{v:.3f}' for v in env_max[i]])

    # ---- figures --------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 1]})
    for ax, sel, unit, t in ((axes[0], (0, 1), 'mm', args.tol_mm),
                             (axes[1], (2, 3, 4), 'deg', args.tol_deg)):
        for d, col in zip(sel, ('#4C72B0', '#DD8452', '#55A868')):
            ax.plot(centers, env[:, d], 'o-', ms=3, color=col,
                    label=f'{DIMS[d]} (p{args.percentile:g})')
            ax.plot(centers, env_max[:, d], '-', lw=0.8, alpha=0.35, color=col)
        ax.axhline(t, ls='--', lw=1.1, color='#888888', label=f'tolerance {t:g} {unit}')
        if np.isfinite(x_c):
            ax.axvline(x_c, ls=':', lw=1.2, color='#C44E52')
        ax.set_ylabel(f'|offset| envelope [{unit}]')
        ax.legend(fontsize=8)
    if np.isfinite(x_c):
        axes[0].text(x_c, axes[0].get_ylim()[1] * 0.95, f'  capture x_c = {x_c:+.1f} mm',
                     fontsize=8, color='#C44E52', va='top')
    axes[2].bar(centers, counts, width=args.bin_mm * 0.9, color='#4C72B0')
    axes[2].set_ylabel('samples')
    axes[2].set_xlabel('insertion depth x [mm]  (0 = seated goal)')
    fig.suptitle(f'Success-basin funnel: per-dimension |offset| envelope vs depth '
                 f'(tol {args.tol_mm} mm / {args.tol_deg} deg)', y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'plot_basin_funnel.png'), dpi=120)
    plt.close(fig)

    # Joint z-pitch extent at a few depths -- the coupled valley narrowing toward the goal.
    picks = np.unique(np.linspace(0, len(centers) - 1, 4).astype(int))
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    shades = plt.get_cmap('Blues')(np.linspace(0.35, 0.95, len(picks)))
    for col, i in zip(shades, picks):
        m = (x >= centers[i] - args.bin_mm / 2) & (x < centers[i] + args.bin_mm / 2)
        ax.scatter(pose[m, 4], pose[m, 2], s=4, color=col, alpha=0.4, lw=0,
                   label=f'x = {centers[i]:+.1f} mm (n={m.sum()})')
    ax.add_patch(plt.Rectangle((-args.tol_deg, -args.tol_mm), 2 * args.tol_deg,
                               2 * args.tol_mm, fill=False, ls='--', lw=1.2,
                               edgecolor='#C44E52', label='goal tolerance'))
    ax.set_xlabel('pitch [deg]')
    ax.set_ylabel('z [mm]')
    ax.set_title('Joint z-pitch extent by insertion depth (the basin valley)', fontsize=10)
    ax.legend(fontsize=8, markerscale=2.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'plot_basin_zpitch.png'), dpi=120)
    plt.close(fig)

    # ---- depth map: the DEEPEST insertion reached per (pitch, z) cell -- the basin as a
    # surface. Its diagonal ridge IS the z-pitch tradeoff: iso-depth contours say "to insert
    # past depth d you must be inside this line in (z, pitch)".
    pv, zv = pose[:, 4], pose[:, 2]
    pb = np.arange(np.floor(pv.min()), np.ceil(pv.max()) + 0.5, 0.5)       # 0.5 deg cells
    zb = np.arange(np.floor(zv.min() * 4) / 4, np.ceil(zv.max() * 4) / 4 + 0.25, 0.25)
    pi = np.clip(np.digitize(pv, pb) - 1, 0, len(pb) - 2)
    zi = np.clip(np.digitize(zv, zb) - 1, 0, len(zb) - 2)
    flat = zi * (len(pb) - 1) + pi
    depth = np.full((len(zb) - 1) * (len(pb) - 1), np.nan)
    order = np.argsort(flat)
    fs, xs = flat[order], x[order]
    starts = np.flatnonzero(np.r_[True, np.diff(fs) > 0])
    for s0, s1 in zip(starts, np.r_[starts[1:], len(fs)]):
        if s1 - s0 >= 5:                               # >= 5 samples per cell, p90 = robust max
            depth[fs[s0]] = np.percentile(xs[s0:s1], 90.0)
    depth = depth.reshape(len(zb) - 1, len(pb) - 1)
    pc = 0.5 * (pb[:-1] + pb[1:])
    zc = 0.5 * (zb[:-1] + zb[1:])

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.set_facecolor('#e6e6e6')                    # masked (no-data) cells read as grey, not white
    dm = np.ma.masked_invalid(depth)
    mesh = ax.pcolormesh(pb, zb, dm, cmap='viridis', shading='flat')
    levels = [lv for lv in (-20.0, -10.0, -5.0, -2.0) if lv > np.nanmin(depth)]
    if levels:
        csr = ax.contour(pc, zc, dm, levels=levels, colors='#111111', linewidths=0.9)
        # White label boxes: the lines cross both dark and bright cells AND the grey no-data
        # background, so no single text colour survives without a backing patch.
        for t in ax.clabel(csr, fmt=lambda v: f'{v:g} mm', fontsize=7):
            t.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1))
    # Ridge fit over the DEEP region: the z-pitch tradeoff slope. Prefer the near-seat cells
    # (>= -2 mm) -- a looser cut lets isolated off-band cells drag the PCA off the band.
    deep = np.isfinite(depth) & (depth >= -2.0)
    if deep.sum() < 8:
        deep = np.isfinite(depth) & (depth >= -5.0)
    if deep.sum() >= 8:
        pz = np.stack([np.repeat(zc, len(pc)).reshape(len(zc), len(pc))[deep],
                       np.tile(pc, (len(zc), 1))[deep]])                    # rows: z, pitch
        zm, pm = pz[0].mean(), pz[1].mean()
        w, V = np.linalg.eigh(np.cov(pz))
        slope = V[0, -1] / V[1, -1] if abs(V[1, -1]) > 1e-9 else np.inf    # mm per deg
        pspan = np.array([pz[1].min(), pz[1].max()])
        ax.plot(pspan, zm + slope * (pspan - pm), '--', color='#DD8452', lw=1.6,
                label=f'deep-region ridge: {slope:+.2f} mm/deg')
    ax.add_patch(plt.Rectangle((-args.tol_deg, -args.tol_mm), 2 * args.tol_deg,
                               2 * args.tol_mm, fill=False, ls='--', lw=1.2,
                               edgecolor='#C44E52', label='goal tolerance'))
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label('deepest reached insertion depth x [mm]  (0 = seated)', fontsize=9)
    ax.set_xlabel('pitch [deg]')
    ax.set_ylabel('z [mm]')
    ax.set_title('Success basin: deepest insertion reached per (pitch, z) '
                 '(p90 per 0.5° x 0.25 mm cell, >= 5 samples)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'plot_basin_depthmap.png'), dpi=120)
    plt.close(fig)

    # ---- INITIAL-OFFSET analysis: which INJECTED offsets still funnel to success ---------
    # The flat CSV has no trial column, but rows are time-ordered and only the insert/hold
    # phases are logged, so trials reappear as contiguous x-sweeps: a large BACKWARD jump in
    # x is a trial boundary. The INITIAL offset is the median off-axis pose over the trial's
    # PRE-CONTACT (free-space) rows -- no force yet, so compliance has not deflected anything
    # and the measured offset IS the commanded bias. Success = the trial's deepest sample
    # reaches --success-x under compliance.
    if args.success_x is not None:
        success_x = args.success_x
    elif args.x_band is not None:
        success_x = sorted(args.x_band)[0]
    elif args.tol_x is not None:
        success_x = -args.tol_x
    else:
        success_x = float(pose_all[:, 0].max()) - 0.5
    xa = pose_all[:, 0]
    cut = np.flatnonzero(np.abs(np.diff(xa)) > args.seg_jump_mm) + 1
    segs = [s for s in np.split(np.arange(len(xa)), cut) if len(s) >= 30]
    trials, n_nofree = [], 0
    for s in segs:
        fm = fmag[s]
        ci = int(np.argmax(fm >= max(args.min_force, 1e-9)))
        if fm[ci] < args.min_force or ci < 5:
            n_nofree += 1                           # never touches, or starts already loaded
            continue
        init = np.median(pose_all[s[:ci], 1:], axis=0)          # y,z,roll,pitch,yaw [mm/deg]
        trials.append((init, float(xa[s].max())))
    if len(trials) < 5:
        print(f'\nINITIAL-OFFSET analysis skipped: only {len(trials)} usable trial segments '
              f'({len(segs)} segments, {n_nofree} without a pre-contact window).')
    else:
        init = np.array([t[0] for t in trials])
        deep = np.array([t[1] for t in trials])
        succ = deep >= success_x
        print(f'\n== initial-offset basin: {len(trials)} trials segmented '
              f'({n_nofree} skipped without pre-contact rows), success = deepest '
              f'x >= {success_x:g} mm -> {int(succ.sum())}/{len(trials)} succeed ==')
        print(f'{"dim":>6s} {"unit":>4s} {"tested |max|":>13s} {"success |max|":>14s} '
              f'{"success p95":>12s}')
        for d, (name, unit) in enumerate(zip(DIMS, UNITS)):
            sv = np.abs(init[succ, d]) if succ.any() else np.array([0.0])
            print(f'{name:>6s} {unit:>4s} {np.abs(init[:, d]).max():13.2f} '
                  f'{sv.max():14.2f} {np.percentile(sv, 95):12.2f}')
        with open(os.path.join(out, 'initial_offsets.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow([f'init_{n}_{u}' for n, u in zip(DIMS, UNITS)]
                       + ['deepest_x_mm', 'success'])
            for (iv, dp), sc in zip(trials, succ):
                w.writerow([f'{v:.3f}' for v in iv] + [f'{dp:.2f}', int(sc)])

        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        ax.scatter(init[~succ, 3], init[~succ, 1], s=26, marker='x', color='#DD8452',
                   label=f'fail (deepest < {success_x:g} mm), n={int((~succ).sum())}')
        ax.scatter(init[succ, 3], init[succ, 1], s=26, color='#4C72B0',
                   label=f'success, n={int(succ.sum())}')
        if succ.sum() >= 3:
            try:
                from scipy.spatial import ConvexHull
                pts = init[succ][:, [3, 1]]
                hull = ConvexHull(pts)
                hp = pts[np.r_[hull.vertices, hull.vertices[0]]]
                ax.plot(hp[:, 0], hp[:, 1], '-', lw=1.2, color='#4C72B0', alpha=0.6,
                        label='success hull')
            except Exception:                       # noqa: BLE001 -- hull is decoration only
                pass
        ax.axhline(0.0, ls=':', lw=0.8, color='#aaaaaa')
        ax.axvline(0.0, ls=':', lw=0.8, color='#aaaaaa')
        ax.set_xlabel('initial pitch offset [deg]')
        ax.set_ylabel('initial z offset [mm]')
        ax.set_title('Allowable INITIAL offset under compliance: injected (pitch, z) vs '
                     'insertion success', fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'plot_basin_initial.png'), dpi=120)
        plt.close(fig)

        # MULTIDIMENSIONAL tolerance: a corner plot over the dims that were actually
        # perturbed (tested range above noise) -- pairwise success/fail scatters below the
        # diagonal, per-dim success-vs-fail histograms on it. The success cloud's extent IS
        # the joint tolerance; a diagonal success band in a panel = coupled tolerance.
        act = [d for d in range(5) if np.abs(init[:, d]).max() > 0.2]
        if len(act) >= 2:
            k = len(act)
            fig, axes = plt.subplots(k, k, figsize=(2.9 * k + 1.2, 2.9 * k + 0.8),
                                     squeeze=False)
            for r in range(k):
                for c in range(k):
                    ax = axes[r][c]
                    if c > r:
                        ax.axis('off')
                        continue
                    dr, dc = act[r], act[c]
                    if r == c:
                        bins = np.linspace(init[:, dr].min(), init[:, dr].max(), 21)
                        ax.hist(init[~succ, dr], bins=bins, color='#DD8452', alpha=0.6,
                                label='fail')
                        ax.hist(init[succ, dr], bins=bins, color='#4C72B0', alpha=0.6,
                                label='success')
                        if r == 0:
                            ax.legend(fontsize=7)
                    else:
                        ax.scatter(init[~succ, dc], init[~succ, dr], s=16, marker='x',
                                   color='#DD8452', alpha=0.8)
                        ax.scatter(init[succ, dc], init[succ, dr], s=16, color='#4C72B0',
                                   alpha=0.8)
                    if r == k - 1:
                        ax.set_xlabel(f'{DIMS[dc]} [{UNITS[dc]}]', fontsize=9)
                    else:
                        ax.tick_params(labelbottom=False)
                    if c == 0 and r != 0:
                        ax.set_ylabel(f'{DIMS[dr]} [{UNITS[dr]}]', fontsize=9)
                    ax.tick_params(labelsize=8)
            fig.suptitle('Multidimensional INITIAL-offset tolerance: success (blue) vs '
                         f'fail (orange x), success = deepest x >= {success_x:g} mm',
                         fontsize=10, y=0.995)
            fig.tight_layout()
            fig.savefig(os.path.join(out, 'plot_basin_tolerance.png'), dpi=120)
            plt.close(fig)

    print(f'\nwrote {out}/success_basin.csv, plot_basin_funnel.png, plot_basin_zpitch.png, '
          'plot_basin_depthmap.png' + (', plot_basin_initial.png, plot_basin_tolerance.png '
                                       '+ initial_offsets.csv' if len(trials) >= 5 else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
