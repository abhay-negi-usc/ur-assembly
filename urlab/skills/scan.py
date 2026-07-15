"""Multi-view cable scan -- drive the camera through viewpoints and fuse SAM3 detections into a
3D connector pose.

This is the port of CablePickPlace._scan and its helpers (_hold_view, _scaled_offset, _approach,
_scan_refine). The control logic is the same, but two things are structurally better because the
detector now runs IN-PROCESS instead of over a topic:

  * ATTRIBUTION IS FREE. The ROS version agonised over crediting a detection to the right view:
    SAM3 lagged the image by an inference, so a detection arriving "now" might belong to the
    previous view or to a motion-blurred frame mid-sweep. _hold_view compared image stamps
    against the settle time to sort this out. Here the scan CAPTURES a frame while stationary,
    runs SAM3 on THAT frame, and ingests the result -- synchronously. A detection cannot belong
    to any view but the one we are standing at. The whole stamp-comparison dance is gone.

  * THE ESTIMATE IS LIVE. The ROS demo read the fused pose from an external node's TF, which only
    refreshed while the cable was in view -- hence the "estimate BEFORE returning home" hazard.
    Here the scan owns the estimator, so the pose is available the instant the last view lands
    and never ages out from under us.

THE GEOMETRY (why the scan moves the way it does) is documented in perception/connector.py. The
short version: only camera TRANSLATION adds information, orbiting the cable axis is what sharpens
the axis, and depth error grows as Z^2 so closing the range is the cheapest accuracy available.
"""

import os

import numpy as np

from .. import log as urlog
from ..transforms import (
    clamp_pose_delta, inverse, look_at, rotate_about_axis, xyzrpy_to_matrix)

log = urlog.get('scan')


class ScanConfig:
    """The scan.* config block, parsed once."""

    def __init__(self, cfg):
        s = cfg.section('scan')
        self.min_good_views = int(s.get('min_good_views', 4))
        self.max_passes = int(s.get('max_passes', 3))
        self.view_settle_s = float(s.get('view_settle_s', 0.5))
        self.require_detection = bool(s.get('require_detection', True))

        rb = s.get('relative_bounds', {}) or {}
        self.bounds_xyz = np.abs(np.asarray(rb.get('xyz', [0.08, 0.08, 0.05]), dtype=float))
        self.bounds_rpy = np.abs(np.asarray(rb.get('rpy', [0.35, 0.35, 0.10]), dtype=float))
        self.offsets = [(np.asarray(o.get('xyz', [0, 0, 0]), dtype=float),
                         np.asarray(o.get('rpy', [0, 0, 0]), dtype=float))
                        for o in (s.get('offsets', []) or [])]

        ap = s.get('approach', {}) or {}
        self.approach_enabled = bool(ap.get('enabled', True))
        self.step_m = float(ap.get('step_m', 0.010))
        self.min_distance_m = float(ap.get('min_distance_m', 0.200))
        self.nominal_distance_m = float(ap.get('nominal_distance_m', 0.23))
        self.scale_offsets = bool(ap.get('scale_offsets', True))
        self.recenter = bool(ap.get('recenter', True))

        r = s.get('refine', {}) or {}
        self.refine_enabled = bool(r.get('enabled', True))
        self.refine_orbit_deg = [float(v) for v in (r.get('orbit_deg', []) or [])]
        self.refine_max_orbit = abs(float(r.get('max_orbit_deg', 25.0)))
        self.refine_min_height = float(r.get('min_height_m', 0.12))

        self.save_images = bool(cfg.get('save_scan_images', True))
        self.images_dir = cfg.get('scan_images_subdir', 'cable_scan')


class CableScanner:
    """Runs a multi-view scan and returns the fused connector pose.

    Holds a Robot, a camera, a SAM3 detector and a ConnectorEstimator; drives the arm through
    viewpoints; returns T_base_connector or None."""

    def __init__(self, robot, camera, detector, estimator, scan_cfg, data_root='data'):
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.estimator = estimator
        self.s = scan_cfg
        self.scan_distance = None                    # camera->connector range; unknown until fit
        self.good_views = 0
        self.view_idx = 0
        from datetime import datetime
        self.image_dir = os.path.join(data_root, self.s.images_dir,
                                      datetime.now().strftime('%Y%m%d_%H%M%S'))

    # ------------------------------------------------------------------ per view
    def _hold_view(self, label):
        """Settle, capture, detect, ingest. Returns True if this view produced a detection.

        Synchronous, so a detection can only be attributed to THIS view -- the reason the ROS
        version's elaborate stamp-matching is absent here."""
        import time
        time.sleep(self.s.view_settle_s)             # let the arm come to REST before capturing
        frame = self.camera.capture()
        dets = self.detector.detect(frame)
        self._save_overlay()
        if not dets and self.s.require_detection:
            log.warning('  %s: no detection -- not a good view (%d/%d good).',
                        label, self.good_views, self.s.min_good_views)
            return False

        self.estimator.add_view(dets, frame.K, frame.T_base_cam, frame.stamp)
        self.good_views += 1
        log.info('  %s: GOOD view, %d detection(s) -- %d/%d good.',
                 label, len(dets), self.good_views, self.s.min_good_views)
        return True

    def _save_overlay(self):
        if not self.s.save_images or self.detector.last_debug is None:
            return
        try:
            import cv2
            os.makedirs(self.image_dir, exist_ok=True)
            path = os.path.join(self.image_dir, f'view_{self.view_idx:02d}.png')
            cv2.imwrite(path, self.detector.last_debug)
            log.info('  saved overlay %s', path)
        except Exception as exc:                     # noqa: BLE001 -- saving is best-effort
            log.warning('  could not save overlay: %s', exc)

    # ------------------------------------------------------------------ geometry
    def _scaled_offset(self, xyz, rpy):
        """Offset scaled to the current range, clamped to the bounds.

        The xyz scales with range; the tilts do NOT, and that is exact rather than an
        approximation: each tilt is atan(offset/distance), invariant when the offset scales with
        the distance. So a view authored at the nominal range keeps the same angular geometry --
        the same fraction of the cable in frame -- at every actual range."""
        scale = 1.0
        if self.s.scale_offsets and self.scan_distance is not None:
            scale = self.scan_distance / max(1e-6, self.s.nominal_distance_m)
        xyz = np.clip(xyz * scale, -self.s.bounds_xyz, self.s.bounds_xyz)
        rpy = np.clip(rpy, -self.s.bounds_rpy, self.s.bounds_rpy)
        return xyz, rpy

    def _approach(self, T_cam0, T_conn):
        """After a good view: re-centre the cable and step the anchor closer, using the connector
        estimate `T_conn`. Returns the new anchor (unchanged if there is no estimate to aim at)."""
        if T_conn is None:
            return T_cam0

        P, C = T_conn[:3, 3], T_cam0[:3, 3]
        v = C - P
        d = float(np.linalg.norm(v))
        if d < 1e-4:
            return T_cam0
        self.scan_distance = d

        at_floor = (not self.s.approach_enabled) or (d <= self.s.min_distance_m + 1e-4)
        d_new = d if at_floor else max(self.s.min_distance_m, d - self.s.step_m)
        C_new = P + v / d * d_new

        T_new = look_at(C_new, P, T_cam0) if self.s.recenter else _with_origin(T_cam0, C_new)
        self.scan_distance = d_new
        if at_floor:
            log.info('  at the %.0f mm view floor (%.0f mm) -- holding range.',
                     self.s.min_distance_m * 1000, d * 1000)
        else:
            log.info('  approach: %.0f -> %.0f mm; offsets now scale x%.2f.',
                     d * 1000, d_new * 1000, d_new / self.s.nominal_distance_m)
        return T_new

    def _refine(self, T_cam0):
        """Orbit the camera about the cable axis at constant standoff -- the ONLY motion that
        sharpens the axis. See perception/connector.py for why rotation-in-place would not."""
        T_conn = self.estimator.estimate()
        if T_conn is None:
            log.warning('  no estimate yet -- skipping the axis refinement.')
            return

        P = T_conn[:3, 3]
        axis = T_conn[:3, 0] / (np.linalg.norm(T_conn[:3, 0]) + 1e-12)
        r = float(np.linalg.norm(T_cam0[:3, 3] - P))
        if r < 1e-3:
            return
        log.info('  refining the axis: orbiting %s at a constant %.0f cm standoff.',
                 self.s.refine_orbit_deg, r * 100)

        for deg in self.s.refine_orbit_deg:
            if self.good_views >= self.s.min_good_views:
                break
            th = np.radians(float(np.clip(deg, -self.s.refine_max_orbit, self.s.refine_max_orbit)))
            T_orbit = rotate_about_axis(_with_origin(T_cam0, T_cam0[:3, 3]), axis, P, th)
            C = T_orbit[:3, 3]
            if float(C[2] - P[2]) < self.s.refine_min_height:
                log.warning('  orbit %+0.f deg would drop the camera below the height floor -- '
                            'skipping.', np.degrees(th))
                continue
            self.view_idx += 1
            if self.robot.move_camera(look_at(C, P, T_cam0),
                                      f'refine orbit {np.degrees(th):+.0f} deg'):
                self._hold_view(f'refine {np.degrees(th):+.0f} deg')

    def _fit_and_approach(self, T_cam0):
        """One estimate() after a good view: use it to re-centre/approach, and report whether it
        has CONVERGED enough to stop (all estimator gates pass AND >= min_good_views banked).

        Returns (new_anchor, converged_pose_or_None). The estimate is computed ONCE and serves
        both jobs. It is only attempted once there are enough views for RANSAC to possibly agree
        (min_inlier_views), so early views don't spam failed-fit warnings."""
        T_conn = (self.estimator.estimate()
                  if self.good_views >= self.estimator.min_inlier_views else None)
        T_cam0 = self._approach(T_cam0, T_conn)        # re-centre + step closer while we have a fit
        converged = T_conn if (T_conn is not None
                               and self.good_views >= self.s.min_good_views) else None
        return T_cam0, converged

    # ------------------------------------------------------------------ run
    def scan(self, confirm=None):
        """Drive the scan until the connector fit CONVERGES. Returns T_base_connector, or None.

        The scan keeps ADDING viewpoints until RANSAC finds an origin consistent across the views
        (the estimator's inlier + parallax gates pass) AND at least min_good_views detections are
        banked -- not merely until a count of detections is reached. So a fit that stalls at
        min_good_views keeps sweeping the WIDER offsets and re-sweeping, which is exactly what a
        stalled fit needs: more baseline, or more views to outvote a detection that landed on the
        wrong cable."""
        if not self.s.offsets:
            log.error('No scan.offsets configured.')
            return None
        T_cam0 = self.robot.camera()
        self.good_views = 0
        self.view_idx = 0
        n = len(self.s.offsets)
        log.info('Scanning until the fit CONVERGES (need >= %d good views AND cross-view '
                 'agreement); %d view(s) per pass, up to %d pass(es).',
                 self.s.min_good_views, n, self.s.max_passes)

        for p in range(self.s.max_passes):
            for i, (xyz, rpy) in enumerate(self.s.offsets):
                dxyz, drpy = self._scaled_offset(xyz, rpy)
                target = clamp_pose_delta(T_cam0, T_cam0 @ xyzrpy_to_matrix(dxyz, drpy),
                                          self.s.bounds_xyz, self.s.bounds_rpy)
                self.view_idx += 1
                d_txt = '' if self.scan_distance is None else f' @{self.scan_distance * 1000:.0f}mm'
                label = f'seed view {i + 1}/{n} pass {p + 1}{d_txt}'
                if confirm and not confirm(label):
                    return None
                if not self.robot.move_camera(target, label):
                    return None
                if self._hold_view(label):
                    T_cam0, converged = self._fit_and_approach(T_cam0)
                    if converged is not None:
                        log.info('Scan complete: %d good views, fit CONVERGED.', self.good_views)
                        return converged

            # A full sweep did not converge -- add the axis-refine views, then re-check.
            if self.s.refine_enabled and self.s.refine_orbit_deg:
                self._refine(T_cam0)
                if self.good_views >= self.s.min_good_views:
                    T = self.estimator.estimate()
                    if T is not None:
                        log.info('Scan complete after refine: %d good views.', self.good_views)
                        return T
            log.warning('Pass %d/%d swept, fit not converged yet (%d good views). Sweeping again '
                        'for more viewpoints / parallax.',
                        p + 1, self.s.max_passes, self.good_views)

        if self.s.require_detection and self.good_views < self.s.min_good_views:
            log.error('Only %d/%d good views after %d passes -- the detector is not seeing the '
                      'cable often enough. Check the overlays, lower sam3.confidence_floor, or '
                      'reword the prompts. Refusing to fit.',
                      self.good_views, self.s.min_good_views, self.s.max_passes)
            return None

        log.error(
            'Scan exhausted %d pass(es) with %d good views, but RANSAC never found an origin '
            'consistent across the views. This is almost always too little PARALLAX (the camera '
            'barely translated between the views that detected) or the neck landing on DIFFERENT '
            'cables across views. Fixes: widen scan.relative_bounds and scan.offsets for more '
            'baseline, start further back so the offsets subtend a larger angle, raise '
            'scan.max_passes, or check the saved overlays show the SAME connector each view.',
            self.s.max_passes, self.good_views)
        return None


def _with_origin(T, C):
    """Copy of pose T with its translation replaced by C (orientation unchanged)."""
    out = np.array(T, dtype=float)
    out[:3, 3] = C
    return out
