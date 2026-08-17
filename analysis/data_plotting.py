"""Relative-pose plotting for assembly logs.

Reads an assembly log CSV (e.g. data/uncertain_assembly_log_YYYYMMDD_HHMMSS.csv) and plots the
RELATIVE POSE of the held part (the peg) with respect to the fixed part (the hole/target) -- the
`held_target_*` columns, which the sampler already logs as inv(T_base_target) @ T_base_tool0 @
T_tool0_held. Commanded poses (`cmd_held_target_*`) can be plotted instead via CONFIG.

MAP CSVs (e.g. data/uncertain_assembly_sampling/banana_map.csv) use a different column schema --
`connector_target_x_mm`-style names with the translation ALREADY in mm, and wrenches under
`wrench_connector_*` / `wrench_base_*` instead of `ft_tool0_*`. Both schemas are handled:
translation units follow the column SUFFIX (`x` = metres, `x_mm` = millimetres), a missing
`pose_prefix` falls back to the first known prefix present in the file, and the force columns are
auto-detected (connector-frame wrench preferred -- it is the contact-frame load).

CONVENTIONS (both differ from the raw CSV, so they are applied here explicitly):
  * Rotation is ZYX INTRINSIC -- yaw (Z), pitch (Y'), roll (X'') -- in DEGREES. By default it is
    recomputed from the logged quaternion via scipy's 'ZYX', so the convention is guaranteed rather
    than inherited. (The CSV's own yaw/pitch/roll columns come from an extrinsic-XYZ decomposition,
    which is numerically the SAME rotation -- extrinsic xyz (roll,pitch,yaw) == intrinsic ZYX
    (yaw,pitch,roll) -- so the two agree; recomputing just removes the doubt.)
  * Translation is converted from metres to MILLIMETRES.

OUTPUT: a timestamped subdirectory under `analysis/` holding the figures, plus `description.txt`
when CONFIG['description'] is set.

FIGURE COLLECTION 1 -- rotating 3D GIFs. A pose has six coordinates, so each figure spends three of
them on the plot axes and the remaining three on the point COLOUR, mapped to the RGB cube (each
channel min-max normalised over the plotted samples). Two complementary views:
    * axes = x/y/z (mm),          colour = yaw/pitch/roll -> R/G/B
    * axes = yaw/pitch/roll (deg), colour = x/y/z          -> R/G/B
Each is animated by sweeping the azimuth through 360 deg. NOTE: a direct RGB encoding is NOT
colourblind-safe and cannot be -- it is a 3-channel readout, not a categorical palette. Every
figure therefore carries a channel legend giving each channel's coordinate and its numeric range,
and the static PNG twin lets you read positions without relying on colour.

FIGURE COLLECTION 2 -- the same rotating 3D views, but coloured by CONTACT FORCE MAGNITUDE
|(fx, fy, fz)| in newtons instead of by a pose triplet. Magnitude is a sequential quantity, so it
uses a single-hue light->dark ramp with a colorbar (never a rainbow, which would band the data).
This is the view that answers "where in the relative pose does the peg actually load up?".

FIGURE COLLECTION 3 -- the x / z / pitch POINT CLOUD: the in-plane contact-manifold slice (the
three coordinates the manifold estimator corrects). Coloured by |F| when wrench columns exist,
else plotted as plain uniform-colour points. With CONFIG['show_xzpitch_window'] it ALSO opens in
an interactive window (drag to rotate) after all files are written, blocking until closed.
A second ALPHA variant of the same cloud (rot3d_xzpitch_alpha) uses much smaller markers with
per-point OPACITY following |F| (same percentile-clipped scale as the colour): barely-loaded
samples fade toward invisible, so the loaded manifold structure reads through the overplot.

ZERO-REFERENCE AXES (CONFIG['zero_axes']): the three coordinate axes through the origin, so the
ideal mate reads as a crosshair. A 3D scatter otherwise gives no cue where zero is, which is the
whole question whenever a cloud is compared against the mate -- "does this sit ON it or beside
it". Drawn AFTER the data has set the view, with the limits then restored, so adding a reference
never rescales the plot; an axis whose two companions do not bracket zero is skipped, since that
line would miss the visible origin.

Usage:
    python analysis/data_plotting.py
    python analysis/data_plotting.py --csv data/my_log.csv --description "trial B, 3 mm bias"
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib
matplotlib.use('Agg')                       # file output; a GUI backend is switched in only for
                                            # the optional interactive window at the very end
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter   # noqa: E402
from mpl_toolkits.mplot3d import Axes3D     # noqa: E402,F401  (registers the 3d projection)


# =====================================================================================
# CONFIG -- toggle everything here.
# =====================================================================================
CONFIG = {
    # ---- input / output ----------------------------------------------------------------
    'csv_path': r'configs/data/bnc_manifold_v2.csv',
    'output_root': 'analysis',       # a timestamped subdirectory is created under this
    'description': None,             # optional str -> written to description.txt

    # ---- which pose to plot ------------------------------------------------------------
    # 'held_target_'      = ACTUAL peg pose wrt the hole (assembly logs)
    # 'cmd_held_target_'  = COMMANDED peg pose wrt the hole (assembly logs)
    # 'connector_target_' = connector pose wrt the target (map CSVs)
    # If the configured prefix is absent from the file, the first known prefix that IS present
    # is used instead (logged + recorded in description.txt), so either CSV kind plots as-is.
    'pose_prefix': 'held_target_',
    'recompute_euler_from_quat': True,   # True = derive ZYX-intrinsic ypr from the quaternion

    # ---- filtering ---------------------------------------------------------------------
    # Force magnitude = |(fx, fy, fz)| of ft_tool0_*. None disables the filter entirely.
    'force_threshold_n': None,
    'force_threshold_mode': 'above',     # 'above' = keep |F| >= thr (contact); 'below' = keep <=
    'trials': None,                      # None = all trials; else e.g. [1, 2, 5]

    # These logs run to ~5e5 samples; scattering all of them across every GIF frame takes many
    # minutes and just overplots. Even stride-subsampling keeps the trajectory shape intact.
    # None = no cap (plot everything).
    'max_points': 20000,

    # ---- figure toggles ----------------------------------------------------------------
    'figures': {
        'rot3d_xyz_rgb_ypr': True,   # axes x/y/z (mm),  colour yaw/pitch/roll
        'rot3d_ypr_rgb_xyz': True,   # axes yaw/pitch/roll (deg), colour x/y/z
        'rot3d_xyz_force': True,     # axes x/y/z (mm),  colour |F| (N)
        'rot3d_ypr_force': False,    # axes yaw/pitch/roll (deg), colour |F| (N)
        'rot3d_xzpitch': True,       # axes x/z (mm) + pitch (deg) -- the contact-manifold slice;
                                     # coloured |F| (N) when wrench columns exist, else plain points
        'rot3d_xzpitch_alpha': True,  # the same slice with SMALL markers whose OPACITY follows
                                      # |F| -- loaded structure reads through the overplot
    },
    # ALSO open the x/z/pitch cloud in an interactive window (drag to rotate) after all files are
    # written. BLOCKS until the window is closed; needs a GUI backend (Qt/Tk) -- skipped with a
    # warning on a headless box (e.g. over plain ssh to the robot machine).
    'show_xzpitch_window': True,
    'save_static_png': True,         # a still PNG twin of each rotating GIF

    # ---- force-magnitude colouring -----------------------------------------------------
    # Magnitude is a SEQUENTIAL encoding: one hue, light -> dark. Never a rainbow (jet/hsv) --
    # it invents banding that is not in the data. The ramp is truncated at the light end so the
    # smallest values stay visible against the white surface.
    'force_cmap': 'Blues',
    'force_cmap_low': 0.25,
    # Contact force is heavily skewed (long tail of rare high loads), so raw min-max leaves almost
    # every point in the palest band. Clip the colour limits to this percentile window; the
    # colorbar is marked 'extend' so the clipping is visible, not silent.
    'force_clim_percentile': [0.0, 99.0],
    'force_clim': None,              # explicit [lo, hi] N -- overrides the percentiles. Set this
                                     # to fix the scale when comparing runs/filtered subsets.
    'force_norm': 'linear',          # 'linear' | 'log' ('log' spreads the low end further)

    # ---- rendering ---------------------------------------------------------------------
    'gif_frames': 72,                # azimuth steps over the full 360 deg
    'gif_fps': 12,
    'gif_elev_deg': 22.0,
    'gif_azim_start_deg': -60.0,
    # ---- zero-reference axes ----------------------------------------------------------------
    # The three coordinate axes through the ORIGIN, i.e. the ideal mate. Without them a 3D
    # scatter gives no cue where zero is, which is the whole question whenever a cloud is being
    # compared against the mate ("does this sit ON it or beside it" -- e.g. after re-centring a
    # map on its engaged density peak). Recessive but legible: a reference, never a data mark.
    # An axis is skipped when the other two do not bracket zero, since the line would then miss
    # the visible origin entirely.
    'zero_axes': True,
    'zero_axis_alpha': 0.35,
    'zero_axis_color': '#1f4e79',
    'zero_axis_width': 1.0,
    'point_size': 28,                # scatter marker area (pt^2)
    'alpha_point_size': 6,           # marker area for the ALPHA variant (much smaller)
    'alpha_range': [0.04, 0.9],      # per-point opacity at the force scale's [min, max]; the
                                     # floor keeps zero-force samples faintly present as context
    # Faint line joining consecutive samples. It is BROKEN at trial boundaries (otherwise it draws
    # a false jump from the end of one insertion to the start of the next). Most useful with a
    # single trial selected above; across many trials it mostly adds clutter, hence off by default.
    'draw_path_line': False,
    'equal_aspect': False,           # True = physically-true proportions; False = autoscale detail
    'dpi': 110,
    'figsize': (9.0, 7.0),
}

# Fixed, per-figure spec: (axis coordinate indices, colour coordinate indices).
# Pose coordinate order used throughout is [x, y, z, yaw, pitch, roll].
_COORD_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
_COORD_UNITS = ['mm', 'mm', 'mm', 'deg', 'deg', 'deg']


# =====================================================================================
# data loading
# =====================================================================================
def load_csv(path):
    """{column_name: np.ndarray} from a log CSV. Numeric where possible, else object."""
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and len(r) == len(header)]
    if not rows:
        raise ValueError(f'No data rows in {path!r}.')
    cols = list(zip(*rows))
    out = {}
    for i, name in enumerate(header):
        try:
            out[name] = np.array([float(v) if v != '' else np.nan for v in cols[i]])
        except ValueError:
            out[name] = np.array(cols[i], dtype=object)
    return out


def has_pose_columns(data, prefix):
    """True if `prefix` has a full xyz translation block in either schema (`x` or `x_mm`)."""
    return (all(prefix + s + '_mm' in data for s in 'xyz')
            or all(prefix + s in data for s in 'xyz'))


def extract_relative_pose(data, prefix, recompute_euler=True):
    """(N, 6) relative pose [x, y, z, yaw, pitch, roll] in mm and degrees (ZYX intrinsic).

    `prefix` selects the block of columns -- 'held_target_' is the held part (peg) expressed in the
    target/fixed part (hole) frame, i.e. exactly the relative pose. The translation unit follows
    the column SUFFIX: assembly logs write `<prefix>x` in metres, map CSVs write `<prefix>x_mm`
    already in millimetres."""
    if all(prefix + s + '_mm' in data for s in 'xyz'):
        xyz_mm = np.column_stack([data[prefix + s + '_mm'] for s in 'xyz'])
    elif all(prefix + s in data for s in 'xyz'):
        xyz_mm = np.column_stack([data[prefix + s] for s in 'xyz']) * 1000.0
    else:
        raise KeyError(f'CSV has no {prefix}x/y/z (or {prefix}x_mm/...) columns. Available '
                       f'prefixes look like: {sorted({k.rsplit("_", 1)[0] for k in data})}')

    quat_cols = [prefix + s for s in ('qx', 'qy', 'qz', 'qw')]
    if recompute_euler and all(c in data for c in quat_cols):
        quat = np.column_stack([data[c] for c in quat_cols])      # scipy order [x, y, z, w]
        ypr = Rotation.from_quat(quat).as_euler('ZYX', degrees=True)   # -> (yaw, pitch, roll)
    else:
        ypr = np.column_stack([data[prefix + 'yaw_deg'], data[prefix + 'pitch_deg'],
                               data[prefix + 'roll_deg']])
    return np.column_stack([xyz_mm, ypr])


# Known pose prefixes, in fallback preference order (assembly logs first, then map CSVs).
_POSE_PREFIXES = ('held_target_', 'cmd_held_target_', 'connector_target_')

# Known wrench blocks, in preference order. Assembly logs write ft_tool0_*; map CSVs write
# wrench_connector_* / wrench_base_*. The connector-frame wrench outranks the base-frame one
# because it is the load expressed at the contact.
_FORCE_PREFIXES = ('ft_tool0_', 'wrench_connector_', 'wrench_base_')


def force_magnitude(data):
    """(|(fx, fy, fz)| in N, wrench column prefix used), or (None, None) if no block is present."""
    for p in _FORCE_PREFIXES:
        cols = [p + a for a in ('fx', 'fy', 'fz')]
        if all(c in data for c in cols):
            return np.linalg.norm(np.column_stack([data[c] for c in cols]), axis=1), p
    return None, None


def build_mask(data, pose, cfg):
    """(mask, [human-readable filter notes]) -- finite samples, trial and force filters applied."""
    notes = []
    mask = np.isfinite(pose).all(axis=1)
    n_nonfinite = int((~mask).sum())
    if n_nonfinite:
        notes.append(f'dropped {n_nonfinite} row(s) with non-finite pose values')

    trials = cfg.get('trials')
    if trials is not None and 'trial' in data:
        keep = np.isin(data['trial'], np.asarray(trials, dtype=float))
        mask &= keep
        notes.append(f'trials restricted to {list(trials)}')

    thr = cfg.get('force_threshold_n')
    if thr is not None:
        fmag, fprefix = force_magnitude(data)
        if fmag is None:
            notes.append('force threshold requested but the CSV has no recognised wrench columns '
                         f'({"/".join(p + "f*" for p in _FORCE_PREFIXES)}) -- filter NOT applied')
        else:
            mode = str(cfg.get('force_threshold_mode', 'above')).lower()
            keep = fmag >= float(thr) if mode == 'above' else fmag <= float(thr)
            mask &= np.nan_to_num(keep, nan=False).astype(bool)
            notes.append(f'force magnitude ({fprefix}f*) {mode} {float(thr):.3g} N '
                         f'({int(keep.sum())}/{len(keep)} samples pass)')
    return mask, notes


# =====================================================================================
# colour: three pose coordinates -> the RGB cube
# =====================================================================================
def rgb_from_triplet(vals):
    """((N,3) rgb in [0,1], [(lo, hi) per channel]) by min-max normalising each column.

    A channel with no spread maps to 0.5 so it reads as a neutral mid-level rather than dividing
    by zero or implying a false extreme."""
    vals = np.asarray(vals, dtype=float)
    rgb = np.empty_like(vals)
    ranges = []
    for c in range(vals.shape[1]):
        lo, hi = float(np.min(vals[:, c])), float(np.max(vals[:, c]))
        span = hi - lo
        rgb[:, c] = 0.5 if span < 1e-12 else (vals[:, c] - lo) / span
        ranges.append((lo, hi))
    return np.clip(rgb, 0.0, 1.0), ranges


def _channel_legend(fig, color_idx, ranges):
    """Three thin gradient bars down the right edge: which coordinate drives R, G and B, and over
    what numeric range. Without this the colour is unreadable -- an RGB readout has no natural key."""
    names = ('R', 'G', 'B')
    for k in range(3):
        ax = fig.add_axes([0.90, 0.62 - k * 0.20, 0.022, 0.15])
        grad = np.zeros((256, 1, 3))
        grad[:, 0, k] = np.linspace(0.0, 1.0, 256)
        ax.imshow(grad, origin='lower', aspect='auto')
        ax.set_xticks([])
        lo, hi = ranges[k]
        ax.set_yticks([0, 255])
        ax.set_yticklabels([f'{lo:.2f}', f'{hi:.2f}'], fontsize=7)
        ax.yaxis.tick_right()
        ax.tick_params(length=2, pad=1, colors='#444444')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ci = color_idx[k]
        ax.set_title(f'{names[k]}: {_COORD_LABELS[ci]}\n[{_COORD_UNITS[ci]}]',
                     fontsize=7.5, pad=4, color='#222222')


# =====================================================================================
# figures
# =====================================================================================
def _style_axes(ax, axis_idx):
    """Recessive grid/panes so the data marks carry the figure."""
    ax.set_xlabel(f'{_COORD_LABELS[axis_idx[0]]} [{_COORD_UNITS[axis_idx[0]]}]', fontsize=9)
    ax.set_ylabel(f'{_COORD_LABELS[axis_idx[1]]} [{_COORD_UNITS[axis_idx[1]]}]', fontsize=9)
    ax.set_zlabel(f'{_COORD_LABELS[axis_idx[2]]} [{_COORD_UNITS[axis_idx[2]]}]', fontsize=9)
    ax.tick_params(labelsize=8, colors='#333333')
    for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane_axis.pane.set_facecolor('#ffffff')
        pane_axis.pane.set_edgecolor('#e3e3e3')
        pane_axis.pane.set_alpha(1.0)
        pane_axis._axinfo['grid'].update(color='#ededed', linewidth=0.6)


def _zero_axes(ax, cfg):
    """The three coordinate AXES through the origin -- a crosshair at the ideal mate.

    One line per axis, drawn only when the OTHER two axes both bracket zero (otherwise the line
    would not pass through the visible origin and is just a stray rule across the box). Drawn
    AFTER the scatter and after any equalisation, because the DATA must set the view; the limits
    are captured first and restored afterwards so adding a reference never rescales the plot.

    (An earlier revision also filled the three zero PLANES at alpha 0.06. They read as a haze
    over the cloud without adding location information the lines do not already give, so only the
    lines remain -- at an opacity that is legible on its own rather than as a plane accent.)"""
    if not (cfg.get('zero_axes', cfg.get('zero_planes'))):    # old key still honoured
        return
    alpha = float(cfg.get('zero_axis_alpha', 0.35))
    color = matplotlib.colors.to_rgba(cfg.get('zero_axis_color', '#1f4e79'), alpha)
    lw = float(cfg.get('zero_axis_width', 1.0))
    lims = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
    for free in range(3):
        others = [a for a in range(3) if a != free]
        if not all(lims[a][0] < 0.0 < lims[a][1] for a in others):
            continue
        p0, p1 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        p0[free], p1[free] = lims[free][0], lims[free][1]
        ax.plot(*zip(p0, p1), '-', lw=lw, color=color, zorder=1, solid_capstyle='butt')
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])


def _equalise(ax, pts):
    """Cube limits centred on the data -- physically-true proportions."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = 0.5 * (lo + hi)
    r = max(float(np.max(hi - lo)) * 0.5, 1e-6)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def _new_3d_axes(cfg, title, n_samples):
    """The shared figure skeleton: white surface, one 3D axes, title and sample count."""
    fig = plt.figure(figsize=cfg['figsize'], dpi=cfg['dpi'])
    fig.patch.set_facecolor('#ffffff')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_position([0.02, 0.03, 0.84, 0.89])
    ax.set_title(title, fontsize=11, color='#111111', pad=10)
    fig.text(0.04, 0.965, f'n = {n_samples} samples', fontsize=8, color='#666666')
    return fig, ax


def _draw_path(ax, pts, cfg, groups):
    """Optional faint polyline through consecutive samples, BROKEN at group (trial) changes."""
    if not cfg.get('draw_path_line') or len(pts) < 2:
        return
    seg = pts.astype(float)
    if groups is not None:
        # NaN rows at each trial change break the polyline instead of drawing a false jump from
        # the end of one insertion to the start of the next.
        brk = np.flatnonzero(np.diff(np.asarray(groups, dtype=float)) != 0) + 1
        seg = np.insert(seg, brk, np.nan, axis=0)
    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], '-', color='#b9bec4', lw=0.8, zorder=1)


def _render_rotating(fig, ax, out_stem, cfg):
    """Save the static PNG twin then the azimuth-sweep GIF. Returns the written paths."""
    written = []
    elev = float(cfg['gif_elev_deg'])
    azim0 = float(cfg['gif_azim_start_deg'])

    if cfg.get('save_static_png', True):
        ax.view_init(elev=elev, azim=azim0)
        png = out_stem + '.png'
        fig.savefig(png, dpi=cfg['dpi'], facecolor=fig.get_facecolor())
        written.append(png)

    frames = int(cfg['gif_frames'])

    def update(i):
        ax.view_init(elev=elev, azim=azim0 + 360.0 * i / frames)
        return ()

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    gif = out_stem + '.gif'
    anim.save(gif, writer=PillowWriter(fps=int(cfg['gif_fps'])), dpi=cfg['dpi'])
    written.append(gif)
    plt.close(fig)
    return written


def rotating_3d_figure(pose, axis_idx, color_idx, out_stem, cfg, title, groups=None):
    """Rotating 3D scatter GIF (+ optional static PNG). Axes take three pose coordinates, the RGB
    colour carries the other three. `groups` (e.g. the trial number per sample) only breaks the
    optional path line so it does not connect across trials. Returns the list of written paths."""
    pts = pose[:, list(axis_idx)]
    rgb, ranges = rgb_from_triplet(pose[:, list(color_idx)])

    fig, ax = _new_3d_axes(cfg, title, len(pts))
    _draw_path(ax, pts, cfg, groups)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rgb, s=cfg['point_size'],
               depthshade=False, edgecolors='none', zorder=2)

    _style_axes(ax, axis_idx)
    if cfg.get('equal_aspect'):
        _equalise(ax, pts)
    _zero_axes(ax, cfg)
    _channel_legend(fig, color_idx, ranges)
    return _render_rotating(fig, ax, out_stem, cfg)


def rotating_3d_plain_figure(pose, axis_idx, out_stem, cfg, title, groups=None):
    """Rotating 3D scatter with one uniform colour -- a plain point cloud for when no further
    quantity should (or can) be encoded. The single hue is a mid-tone of the force ramp so the
    figure family stays visually consistent."""
    pts = pose[:, list(axis_idx)]
    fig, ax = _new_3d_axes(cfg, title, len(pts))
    _draw_path(ax, pts, cfg, groups)
    tone = plt.get_cmap(cfg.get('force_cmap', 'Blues'))(0.65)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=tone, s=cfg['point_size'],
               depthshade=False, edgecolors='none', zorder=2)
    _style_axes(ax, axis_idx)
    if cfg.get('equal_aspect'):
        _equalise(ax, pts)
    _zero_axes(ax, cfg)
    return _render_rotating(fig, ax, out_stem, cfg)


def rotating_3d_scalar_figure(pose, axis_idx, values, out_stem, cfg, title,
                              value_label='|F|', value_unit='N', groups=None):
    """Rotating 3D scatter coloured by a SCALAR magnitude (force) rather than the RGB cube.

    Magnitude is a sequential encoding, so it gets a single-hue light->dark ramp plus a colorbar --
    never a rainbow, which would invent structure that is not in the data. The ramp is truncated at
    the light end (`force_cmap_low`) so the smallest values stay visible on a white surface."""
    pts = pose[:, list(axis_idx)]
    values = np.asarray(values, dtype=float)
    cmap, norm, vmax = _force_colour(values, cfg)

    fig, ax = _new_3d_axes(cfg, title, len(pts))
    _draw_path(ax, pts, cfg, groups)
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=values, cmap=cmap, norm=norm,
                    s=cfg['point_size'], depthshade=False, edgecolors='none', zorder=2)

    _style_axes(ax, axis_idx)
    if cfg.get('equal_aspect'):
        _equalise(ax, pts)
    _zero_axes(ax, cfg)
    _add_force_colorbar(fig, sc, values, vmax, value_label, value_unit)
    return _render_rotating(fig, ax, out_stem, cfg)


def rotating_3d_alpha_figure(pose, axis_idx, values, out_stem, cfg, title, groups=None):
    """Rotating 3D scatter with SMALL markers whose OPACITY (and colour) follow a scalar
    magnitude -- the overplot-friendly twin of rotating_3d_scalar_figure.

    Alpha rides the SAME percentile-clipped normalisation as the colour ramp, remapped into
    CONFIG['alpha_range']: barely-loaded samples fade toward (but never fully reach) invisible,
    so dense free-space clouds stop hiding the loaded manifold structure behind them. With no
    force data (values None) it falls back to small uniform points at the alpha floor + 0.3."""
    pts = pose[:, list(axis_idx)]
    size = float(cfg.get('alpha_point_size', 6))
    a_lo, a_hi = (list(cfg.get('alpha_range')) or [0.04, 0.9])[:2]

    fig, ax = _new_3d_axes(cfg, title, len(pts))
    _draw_path(ax, pts, cfg, groups)
    if values is not None:
        values = np.asarray(values, dtype=float)
        cmap, norm, vmax = _force_colour(values, cfg)
        t = np.clip(norm(np.nan_to_num(values, nan=0.0)), 0.0, 1.0)
        rgba = cmap(t)
        rgba[:, 3] = a_lo + (a_hi - a_lo) * t
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rgba, s=size,
                        depthshade=False, edgecolors='none', zorder=2)
        # The colorbar needs a mappable with the norm attached; the scatter carries raw RGBA,
        # so hand it a proxy. The bar reads for BOTH encodings (alpha co-varies with colour).
        proxy = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        proxy.set_array(values)
        _add_force_colorbar(fig, proxy, values, vmax, '|F|', 'N')
        fig.text(0.885, 0.20, f'opacity {a_lo:.2f} -> {a_hi:.2f}\nover the same |F| scale',
                 fontsize=6.5, color='#777777')
    else:
        tone = list(plt.get_cmap(cfg.get('force_cmap', 'Blues'))(0.65))
        tone[3] = min(1.0, a_lo + 0.3)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=tone, s=size,
                   depthshade=False, edgecolors='none', zorder=2)
    _style_axes(ax, axis_idx)
    if cfg.get('equal_aspect'):
        _equalise(ax, pts)
    _zero_axes(ax, cfg)
    return _render_rotating(fig, ax, out_stem, cfg)


def _force_colour(values, cfg):
    """(truncated cmap, norm, vmax) for the force views -- shared by the saved figures and the
    interactive window so both colour identically.

    Contact force is heavily skewed (a long tail of rare high-load samples), so a raw min-max
    linear scale leaves ~all points in the palest band and the ramp says nothing. Clip the limits
    to a percentile window by default; the colorbar is marked 'extend' so the clipping is visible
    rather than silent. force_clim (explicit N) overrides the percentiles."""
    base = plt.get_cmap(cfg.get('force_cmap', 'Blues'))
    low = float(cfg.get('force_cmap_low', 0.25))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc', base(np.linspace(low, 1.0, 256)))

    clim = cfg.get('force_clim')
    if clim:
        vmin, vmax = float(clim[0]), float(clim[1])
    else:
        pct = cfg.get('force_clim_percentile') or [0.0, 100.0]
        vmin, vmax = (float(np.nanpercentile(values, float(pct[0]))),
                      float(np.nanpercentile(values, float(pct[1]))))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values)) or 1.0

    norm_kind = str(cfg.get('force_norm', 'linear')).lower()
    if norm_kind == 'log':
        floor = max(vmin, float(np.nanmin(values[values > 0])) if np.any(values > 0) else 1e-3)
        norm = matplotlib.colors.LogNorm(vmin=max(floor, 1e-6), vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm, vmax


def _add_force_colorbar(fig, sc, values, vmax, value_label, value_unit):
    """The force colorbar + over-limit note, identical on the saved figure and the window."""
    n_over = int(np.sum(values > vmax))
    cax = fig.add_axes([0.90, 0.30, 0.022, 0.42])
    cb = fig.colorbar(sc, cax=cax, extend='max' if n_over else 'neither')
    cb.set_label(f'{value_label} [{value_unit}]', fontsize=8.5, color='#222222')
    cb.ax.tick_params(labelsize=7, length=2, colors='#444444')
    cb.outline.set_visible(False)
    if n_over:
        fig.text(0.885, 0.255, f'{n_over} sample(s) > {vmax:.1f} {value_unit}\nshown at the top step',
                 fontsize=6.5, color='#777777')


def show_xzpitch_window(pose, fmag, cfg, groups=None):
    """Open the x/z/pitch cloud in an INTERACTIVE window -- drag to rotate, scroll to zoom.

    Called after every file is written, so closing the window loses nothing. Needs a GUI
    matplotlib backend; on a headless box (no Qt/Tk/display) it warns and returns instead of
    crashing. BLOCKS until the window is closed."""
    plt.close('all')                 # saved figures are already closed; makes the switch silent
    for backend in ('QtAgg', 'TkAgg'):
        try:
            plt.switch_backend(backend)
            break
        except Exception:
            continue
    else:
        print('  WARNING: no interactive matplotlib backend (Qt/Tk) available -- '
              'x/z/pitch window skipped.', file=sys.stderr)
        return

    axis_idx = (0, 2, 4)
    pts = pose[:, list(axis_idx)]
    fig, ax = _new_3d_axes(cfg, 'Peg wrt hole: x / z / pitch point cloud  (drag to rotate)',
                           len(pts))
    _draw_path(ax, pts, cfg, groups)
    if fmag is not None:
        cmap, norm, vmax = _force_colour(fmag, cfg)
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=fmag, cmap=cmap, norm=norm,
                        s=cfg['point_size'], depthshade=False, edgecolors='none', zorder=2)
        _add_force_colorbar(fig, sc, fmag, vmax, '|F|', 'N')
    else:
        tone = plt.get_cmap(cfg.get('force_cmap', 'Blues'))(0.65)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=tone, s=cfg['point_size'],
                   depthshade=False, edgecolors='none', zorder=2)
    _style_axes(ax, axis_idx)
    if cfg.get('equal_aspect'):
        _equalise(ax, pts)
    _zero_axes(ax, cfg)
    ax.view_init(elev=float(cfg['gif_elev_deg']), azim=float(cfg['gif_azim_start_deg']))
    plt.show()


# =====================================================================================
# runner
# =====================================================================================
def make_output_dir(root):
    out = os.path.join(root, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out, exist_ok=True)
    return out


def write_description(out_dir, cfg, csv_path, n_total, n_plotted, notes, prefix):
    """description.txt -- the user's text plus the provenance needed to reproduce the figures."""
    if not cfg.get('description'):
        return None
    path = os.path.join(out_dir, 'description.txt')
    with open(path, 'w') as f:
        f.write(str(cfg['description']).rstrip() + '\n\n')
        f.write('--- provenance ---\n')
        f.write(f'generated:      {datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'source csv:     {os.path.abspath(csv_path)}\n')
        f.write(f'pose plotted:   {prefix}* '
                f'(held/peg relative to target/hole)\n')
        f.write('rotation:       ZYX intrinsic (yaw, pitch, roll), degrees\n')
        f.write('translation:    millimetres\n')
        f.write(f'samples:        {n_plotted} plotted of {n_total} rows\n')
        for n in notes:
            f.write(f'filter:         {n}\n')
    return path


def run(cfg):
    csv_path = cfg['csv_path']
    if not os.path.isfile(csv_path):
        print(f'ERROR: no such CSV: {csv_path}', file=sys.stderr)
        return 1

    data = load_csv(csv_path)

    # Resolve the pose prefix: the configured one if present, else the first known prefix the
    # file actually has (map CSVs name the block connector_target_* rather than held_target_*).
    prefix = cfg['pose_prefix']
    pre_notes = []
    if not has_pose_columns(data, prefix):
        fallback = next((p for p in _POSE_PREFIXES if has_pose_columns(data, p)), None)
        if fallback is None:
            print(f"ERROR: no '{prefix}*' pose columns in {csv_path} and none of the known "
                  f"prefixes ({', '.join(_POSE_PREFIXES)}) are present either.", file=sys.stderr)
            return 1
        pre_notes.append(f"no '{prefix}*' columns in this CSV -- plotted '{fallback}*' instead")
        prefix = fallback

    pose_all = extract_relative_pose(data, prefix, cfg['recompute_euler_from_quat'])
    n_total = len(pose_all)
    mask, notes = build_mask(data, pose_all, cfg)
    notes = pre_notes + notes

    # Carry an INDEX array (not a sliced copy) so the trial grouping stays aligned with the pose
    # through both the filter mask and the subsample stride.
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        print(f'ERROR: only {len(idx)} sample(s) left after filtering -- nothing to plot.',
              file=sys.stderr)
        for n in notes:
            print(f'  filter: {n}', file=sys.stderr)
        return 1

    cap = cfg.get('max_points')
    if cap is not None and len(idx) > int(cap):
        stride = int(np.ceil(len(idx) / float(cap)))
        notes.append(f'subsampled every {stride}th sample ({len(idx)} -> '
                     f'{len(idx[::stride])}) to keep rendering tractable')
        idx = idx[::stride]

    pose = pose_all[idx]
    groups = data['trial'][idx] if 'trial' in data else None
    fmag_all, force_prefix = force_magnitude(data)
    fmag = fmag_all[idx] if fmag_all is not None else None

    out_dir = make_output_dir(cfg['output_root'])
    print(f'Loaded {n_total} rows from {csv_path}')
    for n in notes:
        print(f'  filter: {n}')
    print(f'Plotting {len(pose)} samples -> {out_dir}')

    # A key absent from CONFIG['figures'] means OFF, so pruning the dict to the one figure you
    # want actually disables the rest (defaulting missing keys to True made that impossible).
    figs = cfg.get('figures', {})
    written = []
    if figs.get('rot3d_xyz_rgb_ypr'):
        written += rotating_3d_figure(
            pose, axis_idx=(0, 1, 2), color_idx=(3, 4, 5),
            out_stem=os.path.join(out_dir, 'rot3d_xyz_rgb_ypr'), cfg=cfg, groups=groups,
            title='Peg wrt hole: position axes, orientation as RGB')
    if figs.get('rot3d_ypr_rgb_xyz'):
        written += rotating_3d_figure(
            pose, axis_idx=(3, 4, 5), color_idx=(0, 1, 2),
            out_stem=os.path.join(out_dir, 'rot3d_ypr_rgb_xyz'), cfg=cfg, groups=groups,
            title='Peg wrt hole: orientation axes, position as RGB')

    want_force = figs.get('rot3d_xyz_force') or figs.get('rot3d_ypr_force')
    if want_force and fmag is None:
        print('  WARNING: force figures requested but the CSV has no recognised wrench columns '
              f'({"/".join(p + "f*" for p in _FORCE_PREFIXES)}) -- skipped.', file=sys.stderr)
    elif want_force:
        print(f'  force magnitude ({force_prefix}f*) over plotted samples: '
              f'{fmag.min():.2f} .. {fmag.max():.2f} N (mean {fmag.mean():.2f})')
        if figs.get('rot3d_xyz_force'):
            written += rotating_3d_scalar_figure(
                pose, axis_idx=(0, 1, 2), values=fmag,
                out_stem=os.path.join(out_dir, 'rot3d_xyz_force'), cfg=cfg, groups=groups,
                title='Peg wrt hole: position axes, contact force magnitude as colour')
        if figs.get('rot3d_ypr_force'):
            written += rotating_3d_scalar_figure(
                pose, axis_idx=(3, 4, 5), values=fmag,
                out_stem=os.path.join(out_dir, 'rot3d_ypr_force'), cfg=cfg, groups=groups,
                title='Peg wrt hole: orientation axes, contact force magnitude as colour')

    # x / z / pitch: the in-plane contact-manifold slice as a point cloud. Force-coloured when
    # the wrench block exists; plain uniform points otherwise (this figure never skips).
    if figs.get('rot3d_xzpitch'):
        stem = os.path.join(out_dir, 'rot3d_xzpitch')
        if fmag is not None:
            written += rotating_3d_scalar_figure(
                pose, axis_idx=(0, 2, 4), values=fmag, out_stem=stem, cfg=cfg, groups=groups,
                title='Peg wrt hole: x / z / pitch point cloud, contact force as colour')
        else:
            written += rotating_3d_plain_figure(
                pose, axis_idx=(0, 2, 4), out_stem=stem, cfg=cfg, groups=groups,
                title='Peg wrt hole: x / z / pitch point cloud')

    # The ALPHA twin of the x/z/pitch cloud: small markers, opacity following |F| (uniform faint
    # points when no wrench block exists -- like rot3d_xzpitch, this figure never skips).
    if figs.get('rot3d_xzpitch_alpha'):
        written += rotating_3d_alpha_figure(
            pose, axis_idx=(0, 2, 4), values=fmag,
            out_stem=os.path.join(out_dir, 'rot3d_xzpitch_alpha'), cfg=cfg, groups=groups,
            title='Peg wrt hole: x / z / pitch point cloud, contact force as opacity')

    desc = write_description(out_dir, cfg, csv_path, n_total, len(pose), notes, prefix)
    if desc:
        written.append(desc)
    for p in written:
        print(f'  wrote {p}')
    if not written:
        print('No figures enabled in CONFIG["figures"].')

    if cfg.get('show_xzpitch_window'):
        print('  opening the x/z/pitch window (close it to exit) ...')
        show_xzpitch_window(pose, fmag, cfg, groups)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--csv', default=None, help="override CONFIG['csv_path']")
    ap.add_argument('--out', default=None, help="override CONFIG['output_root']")
    ap.add_argument('--description', default=None, help="override CONFIG['description']")
    ap.add_argument('--force-threshold', type=float, default=None,
                    help="override CONFIG['force_threshold_n'] (N)")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.csv:
        cfg['csv_path'] = args.csv
    if args.out:
        cfg['output_root'] = args.out
    if args.description:
        cfg['description'] = args.description
    if args.force_threshold is not None:
        cfg['force_threshold_n'] = args.force_threshold
    return run(cfg)


if __name__ == '__main__':
    sys.exit(main())
