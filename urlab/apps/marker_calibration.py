"""MARKER CALIBRATION -- measure where a target sits in the frame of each fiducial around it.

Produces one `marker_rigs:` block for configs/frames.yaml: for every marker glued around the
fixture, the TARGET's pose in THAT MARKER's own frame.  With that block in place,
bnc_assembly's `target_source: visual` can find the fixture by looking at it (each detected
marker votes T_base_marker @ T_marker_target and the votes are averaged).

The target's pose during calibration comes from the recorded `targets:` entry -- this script
does not measure the fixture, it TRANSFERS a kinematic measurement onto the markers.  The rig
is therefore exactly as accurate as that recorded mate, so re-record the target BEFORE
calibrating.  What the rig buys is INVARIANCE, not accuracy: the fixture may be unbolted and
moved afterwards and the markers travel with it.

Run it with the markers ALREADY IN VIEW (hand-guide the camera, or set view_joints_deg).  The
sweep is a small local ring of camera translations around wherever it starts; it is not a
search.  With `marker_views.servo.enabled` the camera then VISUALLY SERVOS to each marker in
turn -- centred on it at a canonical standoff, returning to the overview between markers so
the whole rig comes back into frame -- and those close, centred views join the sweep's in a
certainty-weighted fusion (closer views weigh more; nothing beyond the
max_camera_distance_mm standoff cap is used).  With `marker_views.multiview_refine` (the
default) each marker's pose is then RE-SOLVED jointly over all of its corner observations at
once, minimized in pixel space -- the same corners-solved-together principle as the run-time
rig PnP, applied where calibration can validly use it -- and the refined pose is kept only
when it reduces the reprojection error.

Output: data/experiments/marker_calibration_<stamp>/ -- the yaml block, per-marker fits, and
marker_images/, which holds EVERY image the fit was computed from, annotated with the marker
detections and the pose read from each, plus an index.csv tying each image to what it
contributed and a summary.txt of the final estimates.
Run:  python -m urlab.apps.marker_calibration --config configs/marker_calibration.yaml
"""

import csv as _csv
import os
from datetime import datetime

import numpy as np

from .. import behaviors as bt
from .. import log as urlog
from .. import tool_frames
from ..apps._common import experiment_dir, prompts_off
from ..skills import marker_localize as mloc
from ..transforms import matrix_to_xyzrpy, pose_error
from ._runner import run_app

log = urlog.get('marker-calib')

_CSV_HEADER = ['marker_id', 'size_mm', 'views', 'view_spread_mm', 'view_spread_deg',
               'marker_x_mm', 'marker_y_mm', 'marker_z_mm',
               'marker_roll_deg', 'marker_pitch_deg', 'marker_yaw_deg',
               'target_in_marker_x_mm', 'target_in_marker_y_mm', 'target_in_marker_z_mm',
               'target_in_marker_roll_deg', 'target_in_marker_pitch_deg',
               'target_in_marker_yaw_deg']


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


class _Calibration:
    """One calibration run: sweep the camera, fuse each marker, solve the target-in-marker
    offsets, and write the frames.yaml block + diagnostics."""

    def __init__(self, cfg, robot, camera, detector, sizes, plan, tname, T_base_target,
                 out_dir):
        self.cfg = cfg
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.sizes = sizes
        self.plan = plan
        self.tname = tname
        self.T_base_target = T_base_target
        self.out_dir = out_dir
        # Every image the fit is computed from, annotated + indexed under marker_images/.
        self.images = mloc.MarkerImageWriter(out_dir, detector,
                                             enabled=getattr(plan, 'save_images', True))
        self.seen = {}
        self.corner_views = []
        self.T_overview = None
        self.fused = {}
        self.offsets = {}
        self.missing = []

    # ---- get the markers in view -------------------------------------------------------------
    def move_to_view(self):
        """Drive to view_joints_deg when configured (pin the SAME pose into bnc_assembly's
        visual localisation so both runs see the same marker faces); otherwise let the operator
        hand-guide the camera."""
        q_view = self.cfg.get('view_joints_deg')
        if q_view is not None:
            log.info('Driving to the view pose %s deg.',
                     list(np.round(np.asarray(q_view, float), 1)))
            if not self.robot.move_joints(np.radians(np.asarray(q_view, dtype=float)),
                                          label='marker view pose'):
                log.error('Could not reach view_joints_deg.')
                return False
        elif (self.cfg.get('confirm_start', True) and not self.robot.arm.dry_run
              and not prompts_off(self.cfg)):
            input('Hand-guide the camera so ALL markers are in view, then press Enter: ')
        # The OVERVIEW: the one pose with every marker in frame. The sweep starts here, and
        # the servo refinement returns here between markers.
        self.T_overview = self.robot.camera()
        return True

    # ---- sweep + fuse ------------------------------------------------------------------------
    def sweep(self):
        log.info('MARKER SWEEP: %s.', self.plan.describe())
        self.seen = mloc.sweep(self.robot, self.camera, self.detector, self.plan,
                               wanted=set(self.sizes), on_view=self.images.sweep_view,
                               corner_log=self.corner_views)
        self.missing = sorted(set(self.sizes) - set(self.seen))
        if self.missing:
            log.error('Marker(s) %s were never detected. They are declared in markers: but '
                      'nothing saw them -- check the ids, the dictionary (%s) and that they '
                      'are in frame.', ', '.join(str(m) for m in self.missing),
                      self.cfg.get_path('aruco.dictionary'))
            if self.cfg.get('require_all_markers', True):
                return False
        return bool(self.seen)

    def servo_refine(self):
        """Visual-servo each swept marker (see skills/marker_localize.servo_refine); the
        refined views replace that marker's sweep views in the fusion."""
        if not self.plan.servo.enabled:
            log.info('Servo refinement disabled (marker_views.servo.enabled: false).')
            return True
        mloc.merge_refined(
            self.seen, mloc.servo_refine(self.robot, self.camera, self.detector, self.plan,
                                         self.seen, T_overview=self.T_overview,
                                         on_view=self.images.servo_view,
                                         corner_log=self.corner_views))
        return True

    def fuse(self):
        self.fused = mloc.fuse_markers(self.seen, self.plan)
        if not self.fused:
            log.error('No marker was seen from enough views (min_views %d) to fuse.',
                      self.plan.min_views)
            return False
        return True

    def multiview_refine(self):
        """Re-solve each marker jointly over ALL its corner observations (pixel-space
        least squares, camera poses from FK); a refined pose is kept only when it reduces
        the reprojection error. Optional, default ON (marker_views.multiview_refine)."""
        if not getattr(self.plan, 'multiview_refine', True):
            log.info('Multi-view refinement disabled (marker_views.multiview_refine: false).')
            return True
        refined = mloc.refine_markers_multiview(self.fused, self.corner_views, self.sizes,
                                                self.plan)
        for mid, (T_ref, _rms_px, _n) in refined.items():
            T, lin, ang, nv, w = self.fused[mid]
            self.fused[mid] = (T_ref, lin, ang, nv, w)
        if not refined:
            log.info('Multi-view refinement changed nothing -- fused poses stand.')
        return True

    # ---- solve + cross-check -----------------------------------------------------------------
    def solve(self):
        self.offsets = mloc.solve_offsets(self.fused, self.T_base_target)
        return True

    def cross_check(self):
        """Each offset is exact BY CONSTRUCTION against the target it was solved from, so what
        carries information is the rig's internal geometry (marker-to-marker distances) and --
        when a previous rig exists -- how far each marker has moved since. A knocked marker
        shows up here and nowhere else."""
        if len(self.fused) > 1:
            log.info('Rig geometry (marker centre distances, mm):')
            ids = sorted(self.fused)
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    d = float(np.linalg.norm(self.fused[a][0][:3, 3]
                                             - self.fused[b][0][:3, 3])) * 1000.0
                    log.info('    %d <-> %d: %8.2f', a, b, d)
        try:
            prior = tool_frames.load_marker_rigs(self.cfg).get(self.tname)
        except ValueError:
            prior = None
        if prior:
            log.info('Change since the rig already in %s:', tool_frames.frames_path(self.cfg))
            for mid in sorted(self.offsets):
                old = prior['markers'].get(mid)
                if old is None:
                    log.info('    %d: NEW.', mid)
                    continue
                lin, ang = pose_error(old['T_marker_target'], self.offsets[mid])
                (log.warning if lin > 0.005 else log.info)(
                    '    %d: %+.2f mm / %+.2f deg%s.', mid, lin * 1000.0, np.degrees(ang),
                    '  <-- moved' if lin > 0.005 else '')
        return True

    # ---- write -------------------------------------------------------------------------------
    def write_outputs(self):
        stamp = f'{datetime.now():%Y-%m-%d}'
        block = mloc.yaml_block(self.tname, self.offsets, self.fused, self.sizes,
                                dictionary=self.cfg.get_path('aruco.dictionary'), stamp=stamp)
        with open(os.path.join(self.out_dir, 'marker_rigs.yaml'), 'w') as fh:
            fh.write('# Paste into configs/frames.yaml (merge under an existing marker_rigs:).\n'
                     f'# Calibrated {stamp} against the recorded target {self.tname!r}.\n'
                     f'{block}\n')
        with open(os.path.join(self.out_dir, 'markers.csv'), 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(_CSV_HEADER)
            for mid in sorted(self.offsets):
                T_m, lin, ang, n = self.fused[mid][:4]
                mxyz, mrpy = matrix_to_xyzrpy(T_m)
                oxyz, orpy = matrix_to_xyzrpy(self.offsets[mid])
                w.writerow([mid, round(self.sizes[mid] * 1000.0, 3), n,
                            round(lin * 1000.0, 4), round(float(np.degrees(ang)), 4)]
                           + [round(float(v) * 1000.0, 3) for v in mxyz]
                           + [round(float(np.degrees(v)), 3) for v in mrpy]
                           + [round(float(v) * 1000.0, 3) for v in oxyz]
                           + [round(float(np.degrees(v)), 3) for v in orpy])
        # The images keep a copy of what they produced, so a folder of pictures can be
        # read on its own without the yaml.
        summary = ['marker calibration %s -- target %r' % (stamp, self.tname), '']
        for mid in sorted(self.offsets):
            T_m, lin, ang, n = self.fused[mid][:4]
            mxyz, mrpy = matrix_to_xyzrpy(T_m)
            oxyz, orpy = matrix_to_xyzrpy(self.offsets[mid])
            summary += [
                'marker %d  (%.1f mm, %d views, spread %.2f mm / %.2f deg)'
                % (mid, self.sizes[mid] * 1000.0, n, lin * 1000.0, np.degrees(ang)),
                '  marker in base_link : xyz %+8.2f %+8.2f %+8.2f mm   rpy %+7.2f %+7.2f '
                '%+7.2f deg' % (mxyz[0] * 1000.0, mxyz[1] * 1000.0, mxyz[2] * 1000.0,
                                np.degrees(mrpy[0]), np.degrees(mrpy[1]),
                                np.degrees(mrpy[2])),
                '  target in marker    : xyz %+8.2f %+8.2f %+8.2f mm   rpy %+7.2f %+7.2f '
                '%+7.2f deg' % (oxyz[0] * 1000.0, oxyz[1] * 1000.0, oxyz[2] * 1000.0,
                                np.degrees(orpy[0]), np.degrees(orpy[1]),
                                np.degrees(orpy[2]))]
        if self.missing:
            summary += ['', 'NEVER DETECTED: %s'
                        % ', '.join(str(m) for m in self.missing)]
        self.images.finish(summary)
        log.info('CALIBRATED %d marker%s. Paste this into configs/frames.yaml:\n\n%s\n',
                 len(self.offsets), '' if len(self.offsets) == 1 else 's', block)
        log.info('Also written to %s', os.path.join(self.out_dir, 'marker_rigs.yaml'))
        if self.missing:
            log.warning('Marker(s) %s are NOT in the block -- they were never detected.',
                        ', '.join(str(m) for m in self.missing))
        return True


def build_and_run(cfg, robot, camera, args):
    tname = cfg.get('target_frame')
    frames = tool_frames.load_frames(cfg)
    targets = tool_frames.load_targets(cfg)
    if not tname or tname not in targets:
        log.error('target_frame %r needs a targets: entry in %s -- the calibration transfers '
                  'THAT recorded pose onto the markers, so it cannot run without one.',
                  tname, tool_frames.frames_path(cfg))
        return False
    if tname not in frames:
        log.error('target_frame %r has a targets: entry but no frames: entry.', tname)
        return False

    try:
        sizes = parse_markers(cfg.get('markers'))
        plan = mloc.ViewPlan(cfg.section('marker_views'))
    except ValueError as exc:
        log.error('%s', exc)
        return False
    # IMPORTED HERE, not at module load: perception pulls in cv2, and parse_markers / the
    # config schema are worth testing on a machine that has no OpenCV.
    from ..perception import ArucoDetector
    detector = ArucoDetector(cfg, sizes_m=sizes)

    out_dir = experiment_dir(cfg, 'marker_calibration')
    log.info('Output: %s', out_dir)
    log.info('Calibrating %d marker%s (%s mm) against the recorded target %r.',
             len(sizes), '' if len(sizes) == 1 else 's',
             ', '.join('%d:%.1f' % (m, s * 1000.0) for m, s in sorted(sizes.items())), tname)

    cal = _Calibration(cfg, robot, camera, detector, sizes, plan, tname, targets[tname],
                       out_dir)
    # One speed factor for the whole visual-localization behavior (sweep + servo moves) --
    # the same knob bnc_assembly's runtime localization uses.
    robot.arm.set_speed_scale(float(cfg.get_path('speed.phase_scale.visual_localize', 1.0)),
                              'visual_localize')
    root = bt.sequence(
        'marker-calibration',
        bt.Action('get the markers in view', cal.move_to_view),
        bt.Action('sweep the views', cal.sweep),
        bt.Action('servo-refine each marker', cal.servo_refine),
        bt.Action('fuse per marker', cal.fuse),
        bt.Action('multi-view joint solve per marker', cal.multiview_refine),
        bt.Action('solve target-in-marker offsets', cal.solve),
        bt.Action('cross-check the rig geometry', cal.cross_check),
        bt.Action('write yaml + csv', cal.write_outputs))
    ok = bt.run_tree(root, log)
    return ok and not (cal.missing and cfg.get('require_all_markers', True))


def main():
    # with_gripper=False: nothing is grasped -- the camera looks, the arm carries it.
    run_app('Marker calibration: the target\'s pose in each fiducial\'s frame',
            'marker_calibration', build_and_run, with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
