"""MARKER CALIBRATION -- measure where a target sits in the frame of each fiducial around it.

WHAT IT PRODUCES is one `marker_rigs:` block for configs/frames.yaml: for every marker glued
around the fixture, the TARGET's pose in THAT MARKER's own frame. Once that exists,
bnc_assembly's `target_source: visual` can find the fixture by looking at it -- the markers are
detected, each one votes T_base_marker @ T_marker_target, and the votes are averaged -- instead of
trusting a mate recorded by hand-guiding weeks ago.

WHERE THE TARGET'S POSE COMES FROM DURING CALIBRATION: the recorded `targets:` entry. That is the
whole trade. This script does not measure the fixture; it TRANSFERS a kinematic measurement onto
the markers, so the rig is exactly as accurate as the target reading it was calibrated from --
and every visual localisation afterwards inherits that. Re-record the target (hand-guide to a good
mate, read `base_link <- <frame>` off urlab.apps.monitor) BEFORE calibrating, not after.

WHAT IT BUYS, given that, is not accuracy but INVARIANCE: after the calibration the fixture may be
unbolted and moved, and the rig still finds it, because the markers travel with it. A recorded
kinematic mate cannot survive that.

Run it with the markers ALREADY IN VIEW (hand-guide the camera, or set view_joints_deg). The
sweep is a small local ring of camera translations around wherever it starts; it is not a search.

Output: data/experiments/marker_calibration_<stamp>/ -- the yaml block, the per-marker fits, and
annotated images from every view.
Run:  python -m urlab.apps.marker_calibration --config configs/marker_calibration.yaml
"""

import csv as _csv
import os
from datetime import datetime

import numpy as np

from .. import log as urlog
from .. import tool_frames
from ..skills import marker_localize as mloc
from ..transforms import matrix_to_xyzrpy, pose_error
from ._runner import run_app

log = urlog.get('marker-calib')


def parse_markers(block, where='markers'):
    """{id: size_m} from the config's `markers:` block.

    Accepts `{7: 20.3}` (mm, the common case) or `{7: {size_mm: 20.3}}`. SIZE IS REQUIRED per id
    and never defaulted: solvePnP scales a marker's distance linearly with the side length it is
    given, so a wrong size is a silent depth error with a perfect reprojection behind it. Pure, so
    the schema is testable without a camera."""
    out = {}
    for raw_id, entry in dict(block or {}).items():
        try:
            mid = int(raw_id)
        except (TypeError, ValueError):
            raise ValueError(f'{where}: {raw_id!r} is not an integer marker id') from None
        if isinstance(entry, dict):
            e = dict(entry)
            keys = {'size_mm', 'size_m'} & set(e)
            if len(keys) != 1:
                raise ValueError(f'{where}[{mid}] needs exactly one of size_mm / size_m')
            size_m = float(e.pop('size_m')) if 'size_m' in e else float(e.pop('size_mm')) / 1000.0
            if e:
                raise ValueError(f'{where}[{mid}] has unknown key(s) {sorted(e)}')
        elif entry is None:
            raise ValueError(f'{where}[{mid}] has no size -- write the side length in mm')
        else:
            size_m = float(entry) / 1000.0
        if not size_m > 0.0:
            raise ValueError(f'{where}[{mid}] has a non-positive size')
        out[mid] = size_m
    if not out:
        raise ValueError(f'{where} is empty -- name at least one marker and its size in mm')
    return out


def build_and_run(cfg, robot, camera, args):
    tname = cfg.get('target_frame')
    frames = tool_frames.load_frames(cfg)
    targets = tool_frames.load_targets(cfg)
    if not tname or tname not in targets:
        log.error('target_frame %r needs a targets: entry in %s -- the calibration transfers THAT '
                  'recorded pose onto the markers, so it cannot run without one.',
                  tname, tool_frames.frames_path(cfg))
        return False
    if tname not in frames:
        log.error('target_frame %r has a targets: entry but no frames: entry.', tname)
        return False
    T_base_target = targets[tname]

    try:
        sizes = parse_markers(cfg.get('markers'))
        plan = mloc.ViewPlan(cfg.section('marker_views'))
    except ValueError as exc:
        log.error('%s', exc)
        return False
    # IMPORTED HERE, not at module load: perception pulls in cv2, and parse_markers / the config
    # schema are worth testing on a machine that has no OpenCV.
    from ..perception import ArucoDetector
    detector = ArucoDetector(cfg, sizes_m=sizes)

    out_dir = os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                           f'marker_calibration_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    log.info('Output: %s', out_dir)
    log.info('Calibrating %d marker%s (%s mm) against the recorded target %r.',
             len(sizes), '' if len(sizes) == 1 else 's',
             ', '.join('%d:%.1f' % (m, s * 1000.0) for m, s in sorted(sizes.items())), tname)

    # ---- GET THE MARKERS IN VIEW ------------------------------------------------------------
    # view_joints_deg is optional and exists so the SAME pose can be pinned into bnc_assembly's
    # visual localisation -- calibrating from one viewpoint and localising from a wildly different
    # one is legal but wastes the rig's best property, that both runs see the same faces.
    q_view = cfg.get('view_joints_deg')
    if q_view is not None:
        log.info('Driving to the view pose %s deg.', list(np.round(np.asarray(q_view, float), 1)))
        if not robot.arm.move_j(list(np.radians(np.asarray(q_view, dtype=float))),
                                label='marker view pose'):
            log.error('Could not reach view_joints_deg.')
            return False
    elif cfg.get('confirm_start', True) and not robot.arm.dry_run:
        input('Hand-guide the camera so ALL markers are in view, then press Enter: ')

    # ---- SWEEP + FUSE ------------------------------------------------------------------------
    log.info('MARKER SWEEP: %s.', plan.describe())

    def save_view(k, frame, poses):
        """One annotated image per view -- the only record of WHY a marker was missed."""
        try:
            import cv2
            cv2.imwrite(os.path.join(out_dir, 'view_%02d.jpg' % (k + 1)),
                        detector.draw(frame, poses))
        except Exception as exc:                       # noqa: BLE001 -- never fail a run on a jpg
            log.debug('could not save the view image: %s', exc)

    seen = mloc.sweep(robot, camera, detector, plan, wanted=set(sizes), on_view=save_view)
    missing = sorted(set(sizes) - set(seen))
    if missing:
        log.error('Marker(s) %s were never detected. They are declared in markers: but nothing '
                  'saw them -- check the ids, the dictionary (%s) and that they are in frame.',
                  ', '.join(str(m) for m in missing), cfg.get_path('aruco.dictionary'))
        if cfg.get('require_all_markers', True):
            return False
    fused = mloc.fuse_markers(seen, plan)
    if not fused:
        log.error('No marker was seen from enough views (min_views %d) to fuse.', plan.min_views)
        return False

    # ---- SOLVE: the target in each marker's frame ---------------------------------------------
    offsets = mloc.solve_offsets(fused, T_base_target)

    # THE CROSS-CHECK. Each marker's offset is exact BY CONSTRUCTION against the target it was
    # solved from, so re-deriving the target proves nothing. What DOES carry information is the
    # rig's internal geometry: the marker-to-marker distances, which no single fit can fake, and
    # -- when a previous rig exists -- how far each marker has moved since. A marker that has been
    # knocked shows up here as a metre-scale outlier or a step change, and nowhere else.
    if len(fused) > 1:
        log.info('Rig geometry (marker centre distances, mm):')
        ids = sorted(fused)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                d = float(np.linalg.norm(fused[a][0][:3, 3] - fused[b][0][:3, 3])) * 1000.0
                log.info('    %d <-> %d: %8.2f', a, b, d)
    try:
        prior = tool_frames.load_marker_rigs(cfg).get(tname)
    except ValueError:
        prior = None
    if prior:
        log.info('Change since the rig already in %s:', tool_frames.frames_path(cfg))
        for mid in sorted(offsets):
            old = prior['markers'].get(mid)
            if old is None:
                log.info('    %d: NEW.', mid)
                continue
            lin, ang = pose_error(old['T_marker_target'], offsets[mid])
            (log.warning if lin > 0.005 else log.info)(
                '    %d: %+.2f mm / %+.2f deg%s.', mid, lin * 1000.0, np.degrees(ang),
                '  <-- moved' if lin > 0.005 else '')

    # ---- WRITE ---------------------------------------------------------------------------------
    stamp = f'{datetime.now():%Y-%m-%d}'
    block = mloc.yaml_block(tname, offsets, fused, sizes,
                            dictionary=cfg.get_path('aruco.dictionary'), stamp=stamp)
    with open(os.path.join(out_dir, 'marker_rigs.yaml'), 'w') as fh:
        fh.write('# Paste into configs/frames.yaml (merge under an existing marker_rigs:).\n'
                 f'# Calibrated {stamp} against the recorded target {tname!r}.\n{block}\n')
    with open(os.path.join(out_dir, 'markers.csv'), 'w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(['marker_id', 'size_mm', 'views', 'view_spread_mm', 'view_spread_deg',
                    'marker_x_mm', 'marker_y_mm', 'marker_z_mm',
                    'marker_roll_deg', 'marker_pitch_deg', 'marker_yaw_deg',
                    'target_in_marker_x_mm', 'target_in_marker_y_mm', 'target_in_marker_z_mm',
                    'target_in_marker_roll_deg', 'target_in_marker_pitch_deg',
                    'target_in_marker_yaw_deg'])
        for mid in sorted(offsets):
            T_m, lin, ang, n = fused[mid]
            mxyz, mrpy = matrix_to_xyzrpy(T_m)
            oxyz, orpy = matrix_to_xyzrpy(offsets[mid])
            w.writerow([mid, round(sizes[mid] * 1000.0, 3), n, round(lin * 1000.0, 4),
                        round(float(np.degrees(ang)), 4)]
                       + [round(float(v) * 1000.0, 3) for v in mxyz]
                       + [round(float(np.degrees(v)), 3) for v in mrpy]
                       + [round(float(v) * 1000.0, 3) for v in oxyz]
                       + [round(float(np.degrees(v)), 3) for v in orpy])

    log.info('CALIBRATED %d marker%s. Paste this into configs/frames.yaml:\n\n%s\n',
             len(offsets), '' if len(offsets) == 1 else 's', block)
    log.info('Also written to %s', os.path.join(out_dir, 'marker_rigs.yaml'))
    if missing:
        log.warning('Marker(s) %s are NOT in the block -- they were never detected.',
                    ', '.join(str(m) for m in missing))
    return not (missing and cfg.get('require_all_markers', True))


def main():
    # with_gripper=False: nothing is grasped -- the camera looks, the arm carries it.
    run_app('Marker calibration: the target\'s pose in each fiducial\'s frame',
            'marker_calibration', build_and_run, with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
