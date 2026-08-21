"""MARKER SNAPSHOT -- grab one frame, label whatever markers are in it, save the picture.

The quick "can the camera see the markers, and where does it think they are?" check.  No
motion: the arm is only connected so the camera pose is known, which is what lets the labels
carry base_link positions as well as camera-frame ones.  Use it before a calibration to aim
the view, or after a bad localisation to see what the camera was actually looking at.

Each detected marker is labelled with its id, its range, and its pose in the camera frame --
the same annotation the calibration and assembly runs save, so a snapshot and a run artefact
read the same way.  The detected ids, their sizes and their poses are also printed.

Output: data/snapshots/ (gitignored), one timestamped jpg per capture.
Run:  python -m urlab.apps.marker_snapshot
      python -m urlab.apps.marker_snapshot --set snapshot.count=5 --set snapshot.interval_s=1
      python -m urlab.apps.marker_snapshot --config bnc_assembly
"""

import os
import time
from datetime import datetime

import numpy as np

from .. import log as urlog
from ..skills import marker_localize as mloc
from ..transforms import matrix_to_xyzrpy
from ._runner import run_app
from .marker_calibration import parse_markers

log = urlog.get('marker-snapshot')


def build_and_run(cfg, robot, camera, args):
    # Sizes come from the config's `markers:` block when it has one (marker_calibration and
    # friends), so the poses match what a real run would solve.  Without it the detector
    # falls back to aruco.marker_size_m for every id -- fine for "is it detected at all",
    # but the DEPTH it reports is only as right as that one number.
    try:
        sizes = parse_markers(cfg.get('markers')) if cfg.get('markers') else {}
    except ValueError as exc:
        log.error('%s', exc)
        return False
    if not sizes:
        log.warning('No markers: block in this config -- every id is solved at the fallback '
                    'size %.1f mm, so the reported ranges are indicative only.',
                    float(cfg.get_path('aruco.marker_size_m', 0.0203)) * 1000.0)

    # IMPORTED HERE, not at module load: perception pulls in cv2.
    import cv2

    from ..perception import ArucoDetector
    detector = ArucoDetector(cfg, sizes_m=sizes)

    snap = cfg.section('snapshot') or {}
    count = max(1, int(snap.get('count', 1)))
    interval_s = float(snap.get('interval_s', 0.0))
    out_dir = str(snap.get('out_dir', 'data/snapshots'))
    os.makedirs(out_dir, exist_ok=True)

    log.info('Snapshot: %d capture%s%s -> %s (dictionary %s).', count,
             '' if count == 1 else 's',
             '' if interval_s <= 0 else f', {interval_s:.1f} s apart', out_dir,
             cfg.get_path('aruco.dictionary'))

    for i in range(count):
        if i and interval_s > 0:
            time.sleep(interval_s)
        frame = camera.capture()
        poses = detector.detect(frame)              # camera-frame poses
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

        if poses:
            log.info('capture %d/%d: %d marker%s -- %s', i + 1, count, len(poses),
                     '' if len(poses) == 1 else 's',
                     ', '.join(str(m) for m in sorted(poses)))
            for mid, T_cm in sorted(poses.items()):
                xyz, rpy = matrix_to_xyzrpy(T_cm)
                line = ('  id %-3d %5.1f mm marker | range %7.1f mm | cam xyz %+8.1f %+8.1f '
                        '%+8.1f mm  rpy %+7.1f %+7.1f %+7.1f deg'
                        % (mid, detector.size_of(mid) * 1000.0,
                           float(np.linalg.norm(T_cm[:3, 3])) * 1000.0,
                           xyz[0] * 1000.0, xyz[1] * 1000.0, xyz[2] * 1000.0,
                           np.degrees(rpy[0]), np.degrees(rpy[1]), np.degrees(rpy[2])))
                if frame.T_base_cam is not None:
                    b = (frame.T_base_cam @ T_cm)[:3, 3] * 1000.0
                    line += ('\n      base_link xyz %+8.1f %+8.1f %+8.1f mm'
                             % (b[0], b[1], b[2]))
                log.info('%s', line)
        else:
            log.warning('capture %d/%d: NO markers detected. Check the dictionary (%s), that '
                        'the markers are in frame and in focus, and the lighting.',
                        i + 1, count, cfg.get_path('aruco.dictionary'))

        header = 'SNAPSHOT %s -- %d marker(s)' % (stamp, len(poses))
        notes = ['dictionary %s | %dx%d' % (cfg.get_path('aruco.dictionary'),
                                            camera.width, camera.height)]
        if sizes:
            notes.append('sizes mm: %s' % ', '.join('%d:%.1f' % (m, s * 1000.0)
                                                    for m, s in sorted(sizes.items())))
        img = mloc.annotate(detector, frame, poses, header, notes)
        path = os.path.join(out_dir, 'markers_%s.jpg' % stamp)
        cv2.imwrite(path, img if img is not None else frame.color)
        log.info('  saved %s', path)
    return True


def main():
    # with_gripper=False and no motion: the arm is connected only so the camera pose is known.
    run_app('Marker snapshot: capture one frame and label the markers in it',
            'marker_calibration', build_and_run, with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
