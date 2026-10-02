"""Marker assemblies, drawn in the held object's ASSEMBLED mating frame.

configs/marker_assemblies.yaml stores, per FIXED-object marker, the held object's mating frame at
its assembled position, in that marker's own frame (T_marker_goal, marker <- goal). This plots
each assembly the other way round, with that goal at the origin:

    fixed marker:  T_goal_fixed = inv(T_marker_goal)           (marker_assemblies.yaml)
    held marker:   T_goal_held  = inv(T_held_marker_grasp)     (objects.yaml, the held_object)

The second line holds because once the held object is seated its mating frame IS the goal, so
its own markers land where they sit in the assembled pair. Together that is the picture the
calibration camera saw: both objects' markers around the one frame the coupler is driven to.
A fixed marker knocked since the calibration, or a held object whose objects.yaml entry has gone
stale (it is composed into every entry measured through it), shows up as a marker that is not
where the part is.

WHAT IS DRAWN, per assembly:
  * the goal at the origin -- the long, lettered triad; +z is the mating axis.
  * the FIXED object's markers: solid squares, labelled `fixed <id>`.
  * the HELD object's markers: lighter, dashed squares, labelled `held <id>` (--no-held hides
    them).
Squares, corner-0 dots, triads and colours are analysis/object_marker_frames.py's, whose drawing
this shares.

The table gives the same poses, plus each fixed marker's capture count and residuals -- the
spread across re-seats, i.e. how repeatably the assembly seats, plus marker noise.

Usage:
    python analysis/marker_assembly_frames.py                       # every assembly, one window
    python analysis/marker_assembly_frames.py tile_on_plate --no-held
    python analysis/marker_assembly_frames.py --save asm.png --no-show
    python analysis/marker_assembly_frames.py --file path/to/marker_assemblies.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import object_marker_frames as omf                                    # noqa: E402
from urlab.tool_frames import (load_marker_assemblies, load_objects,  # noqa: E402
                               marker_assemblies_path, objects_path)

HELD_FILL, HELD_EDGE = '#afb8c1', '#8c959f'


def _fmt(v):
    return '-' if v is None else f'{float(v):.2f}'


def print_table(name, asm, fixed, held):
    meta = asm['meta']
    n = len(fixed)
    print(f'\n{name}  (held {asm["held_object"]}, {n} fixed marker{"s" if n != 1 else ""}, '
          f'{meta.get("captures", "?")} captures, measured {meta.get("measured", "?")}) '
          '-- marker poses in the assembled mating frame')
    print('      ' + omf.TABLE_HEADER + '  residual_mm  residual_deg')
    for mid, (size, T) in fixed.items():
        m = asm['markers'][mid]['meta']
        print('  fixed ' + omf.pose_row(mid, size, T)
              + f'  {_fmt(m.get("residual_mm")):>11}  {_fmt(m.get("residual_deg")):>12}')
    for mid, (size, T) in held.items():
        print('  held  ' + omf.pose_row(mid, size, T))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('names', nargs='*', help='assemblies to draw (default: all of them)')
    ap.add_argument('--file', default=None,
                    help=f'marker assemblies yaml (default: {marker_assemblies_path()})')
    ap.add_argument('--objects', default=None,
                    help=f'objects yaml the held objects come from (default: {objects_path()})')
    ap.add_argument('--no-held', action='store_true',
                    help="draw only the fixed object's markers")
    omf.add_view_args(ap)
    args = ap.parse_args()

    path = args.file or marker_assemblies_path()
    assemblies = load_marker_assemblies(path=path)
    if not assemblies:
        raise SystemExit(f'no assemblies in {path} -- run urlab.apps.marker_assembly_calibration '
                         'first')
    missing = [n for n in args.names if n not in assemblies]
    if missing:
        raise SystemExit(f'not in {path}: {", ".join(missing)} -- have: '
                         f'{", ".join(assemblies)}')
    names = args.names or list(assemblies)
    objects = {} if args.no_held else load_objects(path=args.objects or objects_path())

    fixed, held = {}, {}
    for n in names:
        fixed[n] = omf.markers_in_origin(assemblies[n]['markers'], 'T_marker_goal')
        obj = objects.get(assemblies[n]['held_object'])
        if obj is None and not args.no_held:
            print(f'\nWARNING: {n}: held_object {assemblies[n]["held_object"]!r} is not in the '
                  'objects catalogue -- drawing the fixed markers only.')
        held[n] = omf.marker_poses(obj) if obj is not None else {}
        print_table(n, assemblies[n], fixed[n], held[n])

    boxes = {n: omf.cube(fixed[n], held[n]) for n in names}
    shared = max(span for _, span in boxes.values())

    fig, axes = omf.panel_grid(len(names), 'Marker assemblies in the assembled mating frame',
                               path, args.no_show)
    for ax, n in zip(axes, names):
        omf.draw_origin(ax)
        for mid, (size, T) in fixed[n].items():
            omf.draw_marker(ax, T, size, f'fixed {mid}')
        for mid, (size, T) in held[n].items():
            omf.draw_marker(ax, T, size, f'held {mid}', fill=HELD_FILL, edge=HELD_EDGE,
                            fill_alpha=0.15, ls='--')
        asm = assemblies[n]
        centre, span = boxes[n]
        omf.style_axes(ax, centre, span if args.fit_each else shared,
                       f'{n}\nheld {asm["held_object"]} · measured '
                       f'{asm["meta"].get("measured", "?")}')

    handles = [omf.marker_handle("fixed object's marker")]
    if not args.no_held:
        handles.append(omf.marker_handle("held object's marker", HELD_FILL, HELD_EDGE, 0.4))
    omf.finish(fig, axes, handles, args)


if __name__ == '__main__':
    main()
