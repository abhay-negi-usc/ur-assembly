"""Marker frames of every catalogued object, drawn in that object's own frame.

configs/objects.yaml stores, per marker, the object's MATING FRAME expressed in that marker
(T_marker_grasp, marker <- grasp) -- the direction the run time wants. This plots it the other
way round: the object's mating frame at the origin and each marker at

    T_grasp_marker = inv(T_marker_grasp)

-- the picture of the physical part: where each marker is stuck relative to the feature the
coupler seats into. Every marker of an object encodes the SAME mating frame, so they all share
the one origin and a marker that was knocked or re-stuck shows up as sitting where it is not.

WHAT IS DRAWN, per object:
  * the mating frame at the origin -- the long, lettered triad; +z is the mating axis.
  * each marker as its printed black square (size_mm), with its own x/y/z triad. Corners are
    perception/aruco's object points: TL, TR, BR, BL, centred, +z OUT of the printed face. A dot
    marks corner 0 (TL), so the in-plane rotation of the print reads without the triad.
  * a hairline from the origin to each marker centre.

Axes are x red, y green, z blue (the RViz order). The legend and the origin triad's letters carry
the identity too, so it never rests on hue alone. All lengths in mm, angles in deg, rpy EXTRINSIC
XYZ (urlab/transforms.py).

A table of the same poses is printed -- the numbers behind the picture. `tilt_deg` is the angle
between the marker's face normal and the mating axis, folded to [0, 90]: 0 means the marker face
is square to the axis (facing either way).

All objects share ONE scale by default so sizes compare across subplots; every subplot rotates
together (drag any of them). --fit-each zooms each object to itself instead.

The drawing helpers below are shared with analysis/marker_assembly_frames.py.

Usage:
    python analysis/object_marker_frames.py                           # every object, one window
    python analysis/object_marker_frames.py ORU_v8 tile_1             # just these
    python analysis/object_marker_frames.py --save frames.png --no-show
    python analysis/object_marker_frames.py --file path/to/objects.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urlab.tool_frames import load_objects, objects_path    # noqa: E402
from urlab.transforms import inverse, matrix_to_xyzrpy        # noqa: E402

AXIS_COLOURS = ('#cf222e', '#1a7f37', '#0969da')               # x, y, z
INK, INK_MUTED, RULE = '#1f2328', '#656d76', '#d0d7de'
MARKER_FILL, MARKER_EDGE = '#8c959f', '#57606a'

ORIGIN_AXIS_MM = 50.0
MARKER_AXIS_FRAC = 0.75      # a marker's triad length, as a fraction of its printed side

TABLE_HEADER = ('    id  size_mm     x_mm     y_mm     z_mm  roll_deg pitch_deg   yaw_deg  dist_mm'
                '  tilt_deg')


# ---------------------------------------------------------------------------- geometry
def markers_in_origin(markers, key):
    """{id: (size_mm, T_origin_marker in mm)}, ids ascending, from catalogue markers that each
    store the ORIGIN in their own frame under `key` (marker <- origin) -- hence the inverse."""
    out = {}
    for mid, m in sorted(markers.items()):
        T = inverse(m[key])
        T[:3, 3] *= 1000.0
        out[mid] = (m['size_m'] * 1000.0, T)
    return out


def square(size_mm):
    """TL, TR, BR, BL in the marker's own frame -- ArucoDetector.object_points, in mm."""
    h = size_mm / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def apply(T, pts):
    return np.asarray(pts, dtype=float) @ T[:3, :3].T + T[:3, 3]


def triad_tips(T, length):
    return T[:3, 3] + length * T[:3, :3].T


def cube(*pose_sets):
    """(centre, side) of the box around the origin triad and every marker drawn -- what the
    axis limits have to contain."""
    pts = [np.zeros((1, 3)), triad_tips(np.eye(4), ORIGIN_AXIS_MM)]
    for poses in pose_sets:
        for size, T in poses.values():
            pts += [apply(T, square(size)), triad_tips(T, MARKER_AXIS_FRAC * size)]
    pts = np.vstack(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (lo + hi) / 2.0, float((hi - lo).max()) * 1.1


def pose_row(mid, size, T):
    """One TABLE_HEADER row. tilt = the face normal against the origin's +z, folded to [0, 90]."""
    xyz, rpy = matrix_to_xyzrpy(T)
    tilt = math.degrees(math.acos(min(1.0, abs(float(T[2, 2])))))
    return (f'{mid:4d} {size:8.2f} ' + ' '.join(f'{v:+8.2f}' for v in xyz) + ' '
            + ' '.join(f'{v:+9.2f}' for v in np.degrees(rpy))
            + f' {np.linalg.norm(xyz):8.2f} {tilt:9.2f}')


# ---------------------------------------------------------------------------- drawing
def draw_triad(ax, T, length, lw, letters=False):
    o = T[:3, 3]
    for k, (tip, colour) in enumerate(zip(triad_tips(T, length), AXIS_COLOURS)):
        ax.plot(*zip(o, tip), color=colour, lw=lw, solid_capstyle='round')
        if letters:
            ax.text(*(o + 1.18 * (tip - o)), 'xyz'[k], color=INK_MUTED, fontsize=8,
                    ha='center', va='center')


def draw_origin(ax):
    draw_triad(ax, np.eye(4), ORIGIN_AXIS_MM, lw=2.5, letters=True)
    ax.scatter([0], [0], [0], color=INK, s=14, depthshade=False)


def draw_marker(ax, T, size, label, fill=MARKER_FILL, edge=MARKER_EDGE, fill_alpha=0.3,
                ls='-'):
    """One printed marker: hairline from the origin, the square, its corner-0 dot, its triad
    and a label just past the square's -y edge (the side its own y axis does not cross)."""
    ax.plot(*zip(np.zeros(3), T[:3, 3]), color=RULE, lw=0.8)
    corners = apply(T, square(size))
    # Fill and outline separately: a collection's alpha fades its edges too, and a faint
    # outline makes the corner-0 dot look like it sits off the square.
    ax.add_collection3d(Poly3DCollection([corners], facecolor=fill, alpha=fill_alpha,
                                         edgecolor='none'))
    ax.plot(*np.vstack([corners, corners[:1]]).T, color=edge, lw=0.8, ls=ls)
    ax.scatter(*corners[0], color=INK, s=10, depthshade=False)
    draw_triad(ax, T, MARKER_AXIS_FRAC * size, lw=1.5)
    ax.text(*apply(T, [[0.0, -0.8 * size, 0.0]])[0], label, color=INK, fontsize=8,
            ha='center', va='center')


def style_axes(ax, centre, span, title):
    half = span / 2.0
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1, 1, 1))
    for axis, label in zip((ax.xaxis, ax.yaxis, ax.zaxis), ('x', 'y', 'z')):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis.line.set_color(RULE)
        axis.set_label_text(f'{label} (mm)', fontsize=7, color=INK_MUTED)
        axis.labelpad = -4
    ax.tick_params(colors=INK_MUTED, labelsize=6, pad=-2)
    ax.set_title(title, fontsize=9, color=INK, pad=-2)


# ---------------------------------------------------------------------------- figure
def add_view_args(ap):
    """The figure options both scripts take."""
    ap.add_argument('--fit-each', action='store_true',
                    help='zoom each panel to itself instead of one scale shared by all')
    ap.add_argument('--elev', type=float, default=35.0,
                    help='initial view elevation (deg); negative looks from the markers\' printed '
                         'side, roughly as the camera sees them')
    ap.add_argument('--azim', type=float, default=-60.0, help='initial view azimuth (deg)')
    ap.add_argument('--save', default=None, metavar='PNG', help='also write the figure here')
    ap.add_argument('--no-show', action='store_true', help='do not open a window')


def panel_grid(n, title, source, no_show):
    """(fig, axes): one 3D panel per item, all rotating together, with the title and the source
    file in a band fixed in INCHES so it stays put whatever the grid size."""
    if no_show:
        plt.switch_backend('Agg')
    plt.rcParams.update({'grid.color': RULE, 'grid.linewidth': 0.5})
    ncols = min(n, math.ceil(math.sqrt(1.6 * n)))
    nrows = math.ceil(n / ncols)
    w = max((4.2 if n <= 4 else 3.6) * ncols, 7.0)
    # 3D panels draw a cube, so a panel wants to be about as tall as it is wide. The legend
    # wraps to three rows on a narrow figure (see finish), which needs a deeper bottom band.
    top_in, bottom_in = 1.0, 0.5 if w >= 10.0 else 0.8
    h = nrows * min(w / ncols, 5.5) * 0.95 + top_in + bottom_in
    fig = plt.figure(figsize=(w, h), facecolor='white')
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0 - top_in / h, bottom=bottom_in / h,
                        wspace=0.0, hspace=0.12)
    axes = []
    for i in range(n):
        axes.append(fig.add_subplot(nrows, ncols, i + 1, projection='3d',
                                    shareview=axes[0] if axes else None))
    fig.suptitle(title, y=1.0 - 0.15 / h, va='top', fontsize=12, color=INK)
    shown = os.path.relpath(source)
    if shown.startswith('..'):            # outside the repo: the name is what identifies it
        shown = os.path.join('...', os.path.basename(source))
    fig.text(0.5, 1.0 - 0.45 / h, shown, ha='center', va='top', fontsize=8, color=INK_MUTED)
    return fig, axes


def finish(fig, axes, marker_handles, args):
    """Legend (axes + whatever marker styles the caller drew), initial view, save, show."""
    axes[0].view_init(elev=args.elev, azim=args.azim)
    handles = [Line2D([], [], color=c, lw=2.5, label=a) for c, a in zip(AXIS_COLOURS, 'xyz')]
    handles += marker_handles + [
        Line2D([], [], marker='o', ls='', ms=4, color=INK, label='corner 0 (top-left)'),
        Line2D([], [], color=RULE, lw=0.8, label='origin to marker centre')]
    wide = fig.get_figwidth() >= 10.0         # the same test panel_grid sized the band with
    fig.legend(handles=handles, loc='lower center', ncol=len(handles) if wide else 3,
               frameon=False, fontsize=8, labelcolor=INK)
    if args.save:
        fig.savefig(args.save, dpi=150, facecolor='white')
        print(f'\nwrote {args.save}')
    if not args.no_show:
        plt.show()


def marker_handle(label, fill=MARKER_FILL, edge=MARKER_EDGE, alpha=0.5):
    return Line2D([], [], marker='s', ls='', ms=9, markerfacecolor=fill, alpha=alpha,
                  markeredgecolor=edge, label=label)


# ---------------------------------------------------------------------------- this script
def marker_poses(obj):
    """An objects.yaml entry's markers in its mating frame (they store marker <- grasp)."""
    return markers_in_origin(obj['markers'], 'T_marker_grasp')


def print_table(name, obj, poses):
    n = len(poses)
    print(f'\n{name}  ({n} marker{"s" if n != 1 else ""}, measured '
          f'{obj["meta"].get("measured", "?")}) -- marker poses in the mating frame')
    print(TABLE_HEADER)
    for mid, (size, T) in poses.items():
        print('  ' + pose_row(mid, size, T))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('names', nargs='*', help='objects to draw (default: all of them)')
    ap.add_argument('--file', default=None, help=f'objects yaml (default: {objects_path()})')
    add_view_args(ap)
    args = ap.parse_args()

    path = args.file or objects_path()
    objects = load_objects(path=path)
    missing = [n for n in args.names if n not in objects]
    if missing:
        raise SystemExit(f'not in {path}: {", ".join(missing)} -- have: {", ".join(objects)}')
    names = args.names or list(objects)

    poses = {n: marker_poses(objects[n]) for n in names}
    for n in names:
        print_table(n, objects[n], poses[n])

    # The shared span keeps mm-per-inch equal across panels; each cube is still centred on its
    # own object so nothing is pushed off to one side.
    boxes = {n: cube(poses[n]) for n in names}
    shared = max(span for _, span in boxes.values())

    fig, axes = panel_grid(len(names), 'Marker frames in each object\'s mating frame', path,
                           args.no_show)
    for ax, n in zip(axes, names):
        draw_origin(ax)
        for mid, (size, T) in poses[n].items():
            draw_marker(ax, T, size, f'id {mid}')
        k = len(poses[n])
        centre, span = boxes[n]
        style_axes(ax, centre, span if args.fit_each else shared,
                   f'{n}\n{k} marker{"s" if k != 1 else ""} · measured '
                   f'{objects[n]["meta"].get("measured", "?")}')
    finish(fig, axes, [marker_handle('marker (printed square)')], args)


if __name__ == '__main__':
    main()
