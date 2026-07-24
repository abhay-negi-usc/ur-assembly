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
    clamp_pose_delta, frame_from_axis, inverse, look_at, rotate_about_axis, xyzrpy_to_matrix)

log = urlog.get('scan')


class ScanConfig:
    """The scan.* config block, parsed once."""

    def __init__(self, cfg):
        s = cfg.section('scan')
        self.min_good_views = int(s.get('min_good_views', 4))
        self.max_passes = int(s.get('max_passes', 3))
        self.view_settle_s = float(s.get('view_settle_s', 0.5))
        self.require_detection = bool(s.get('require_detection', True))
        # Scan mode. 'fuse' (default) is the original behaviour: fuse per-view junction detections
        # into a pose with the ConnectorEstimator. 'reconstruction' additionally reconstructs the
        # 3D cable centreline and REFINES the junction pose from it (needs sam3.mode: junction, so
        # the cable skeleton is available). The two convergence thresholds below apply to it.
        self.mode = str(s.get('mode', 'fuse'))
        rc = cfg.section('reconstruction') if hasattr(cfg, 'section') else {}
        rc = rc or {}
        self.recon_max_reproj_px = float(rc.get('max_reproj_error_px', 3.0))
        self.recon_min_pose_shift_m = float(rc.get('min_pose_shift_m', 0.003))
        # A detected view only counts toward the good-view quota (min_good_views, the fit-trust
        # gate before the grasp) if the camera is within this range of the cable -- so the fit is
        # built from CLOSE, low-depth-error views. Detections FARTHER than this are still ingested
        # (they steer the approach in) but do not count. Should be >= approach.min_distance_m, or
        # the camera stops approaching beyond the counting range and no view ever counts.
        self.max_view_distance_m = float(s.get('max_view_distance_m', 0.250))

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
        # Rate-limit on the vision estimate BETWEEN views. The cable is one physical object, so its
        # estimated ORIGIN (and AXIS) should barely move view-to-view; a big jump is an outlier
        # detection. Clamp the per-view change so a single outlier cannot lunge the approach
        # (steering) or snap the pose. 0 = off. (Independent of the CAMERA's step_m -- this bounds
        # the ESTIMATE's motion, not the camera's.)
        self.max_view_delta_m = float(ap.get('max_view_delta_m', 0.0))
        self.max_view_delta_deg = float(ap.get('max_view_delta_deg', 0.0))

        r = s.get('refine', {}) or {}
        self.refine_enabled = bool(r.get('enabled', True))
        self.refine_orbit_deg = [float(v) for v in (r.get('orbit_deg', []) or [])]
        self.refine_max_orbit = abs(float(r.get('max_orbit_deg', 25.0)))
        self.refine_min_height = float(r.get('min_height_m', 0.12))

        self.save_images = bool(cfg.get('save_scan_images', True))
        self.images_dir = cfg.get('scan_images_subdir', 'cable_scan')
        # A STABLE-path copy of the latest overlay, overwritten (atomically) on every capture, so a
        # viewer left open on it always shows the newest view + detection without reopening files.
        # Independent of save_scan_images. Path defaults to <data_dir>/last_camera_image.png.
        self.save_last_image = bool(cfg.get('save_last_camera_image', True))
        self.last_image_path = cfg.get('last_camera_image_path', None)


class CableScanner:
    """Runs a multi-view scan and returns the fused connector pose.

    Holds a Robot, a camera, a SAM3 detector and a ConnectorEstimator; drives the arm through
    viewpoints; returns T_base_connector or None."""

    def __init__(self, robot, camera, detector, estimator, scan_cfg, data_root='data',
                 reconstructor=None):
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.estimator = estimator
        self.reconstructor = reconstructor           # CableReconstructor, only in 'reconstruction' mode
        self.mode = scan_cfg.mode
        self.s = scan_cfg
        self.scan_distance = None                    # camera->connector range; unknown until fit
        self.good_views = 0
        self.view_idx = 0
        self._recon = None                           # last Reconstruction (reconstruction mode)
        self._last_est_vid = 0                       # view id the estimator gave the current frame
        self._last_rec_vid = 0                       # ...and the reconstructor (for good-marking)
        self._last_origin = None                     # previous vision origin (for the per-view clamp)
        self._last_axis = None                       # previous connector axis (unit)
        from datetime import datetime
        self.image_dir = os.path.join(data_root, self.s.images_dir,
                                      datetime.now().strftime('%Y%m%d_%H%M%S'))
        # Stable path for the always-latest overlay copy (default: <data_dir>/last_camera_image.png).
        self.last_image_path = self.s.last_image_path or os.path.join(data_root,
                                                                      'last_camera_image.png')

    # ------------------------------------------------------------------ per view
    def _hold_view(self, label):
        """Settle, capture, detect, ingest. Returns True if this view produced a detection.

        Synchronous, so a detection can only be attributed to THIS view -- the reason the ROS
        version's elaborate stamp-matching is absent here."""
        import time
        time.sleep(self.s.view_settle_s)             # let the arm come to REST before capturing
        frame = self.camera.capture()
        self._recon = None
        dets = self._detect_and_ingest(frame)        # mode-aware: fuse vs reconstruction
        self._save_overlay()
        if not dets and self.s.require_detection:
            log.warning('  %s: no detection -- not a good view (%d/%d good).',
                        label, self.good_views, self.s.min_good_views)
            return False

        # Count (and FUSE) this view only if we can confirm the camera is CLOSE enough. The distance
        # needs an origin estimate; until there is one (the first view, or the camera beyond
        # max_range) we cannot confirm "close", so the view is not counted or fused -- but it still
        # returns True so the approach runs and closes the range. The good/far decision is pushed to
        # both estimators (mark_view), so ONLY within-distance views enter the final fit; the far
        # ones only ever steered the approach via rough_origin.
        P = self.estimator.rough_origin() if frame.T_base_cam is not None else None
        if P is not None:
            d = float(np.linalg.norm(frame.T_base_cam[:3, 3] - np.asarray(P, dtype=float)))
            good = d <= self.s.max_view_distance_m
            self._mark_good(good)
            if good:
                self.good_views += 1
                log.info('  %s: GOOD view @%.0f mm, %d detection(s) -- %d/%d good (fused).',
                         label, d * 1000, len(dets), self.good_views, self.s.min_good_views)
            else:
                log.info('  %s: detected @%.0f mm > %.0f mm max -- NOT fused (approaching in).',
                         label, d * 1000, self.s.max_view_distance_m * 1000)
        else:
            self._mark_good(False)
            log.info('  %s: detected, %d -- range unknown yet, not fused (need 2 views / in range).',
                     label, len(dets))

        # Reconstruction (and its plot) runs AFTER the good decision, so the current view's
        # good/far mark is reflected in the curve the stop rule and the figure use.
        if self.mode == 'reconstruction':
            self._recon = self.reconstructor.reconstruct()
            self._save_recon_plot()
        return True

    def _detect_and_ingest(self, frame):
        """Detect and ingest one frame. Records the estimator/reconstructor view ids (so the caller
        can mark this view good/far AFTER the distance is known) and returns the junction detections
        [(u, v, yaw), ...] the good-view/approach logic works on.

        In BOTH modes the ConnectorEstimator receives the junction point(s) -- it drives the
        approach steering (rough_origin) and the good-view distance gate. In reconstruction mode the
        detector also yields the full cable SKELETON, which the CableReconstructor ingests. Neither
        reconstructs here -- that waits until the view is marked (see _hold_view)."""
        self._last_est_vid = 0
        self._last_rec_vid = 0
        if self.mode == 'reconstruction':
            obs = self.detector.detect_cable(frame)
            dets = ([(obs['junction'][0], obs['junction'][1], obs['yaw'])]
                    if obs is not None else [])
            if obs is not None:
                self._last_rec_vid = self.reconstructor.add_view(
                    obs, frame.K, frame.T_base_cam, frame.stamp)
            self._last_est_vid = self.estimator.add_view(
                dets, frame.K, frame.T_base_cam, frame.stamp)
            return dets

        dets = self.detector.detect(frame)
        # Ingest ALWAYS (a far detection still steers the approach in via the rough origin).
        self._last_est_vid = self.estimator.add_view(dets, frame.K, frame.T_base_cam, frame.stamp)
        return dets

    def _mark_good(self, good):
        """Push the good/far decision for the current frame to both estimators, so only close views
        enter the final fusion."""
        self.estimator.mark_view(self._last_est_vid, good)
        if self.mode == 'reconstruction':
            self.reconstructor.mark_view(self._last_rec_vid, good)

    def _save_overlay(self):
        """Save the detection overlay two ways: the per-view timestamped archive file (if
        save_scan_images) AND a stable-path copy at last_image_path (if save_last_camera_image),
        overwritten each capture. The stable copy is written to a temp file and atomically renamed,
        so a viewer left open on it never reads a half-written frame."""
        dbg = self.detector.last_debug
        if dbg is None:
            return
        try:
            import cv2
            if self.s.save_images:
                os.makedirs(self.image_dir, exist_ok=True)
                path = os.path.join(self.image_dir, f'view_{self.view_idx:02d}.png')
                cv2.imwrite(path, dbg)
                log.info('  saved overlay %s', path)
            if self.s.save_last_image:
                os.makedirs(os.path.dirname(self.last_image_path) or '.', exist_ok=True)
                root, ext = os.path.splitext(self.last_image_path)
                tmp = f'{root}.tmp{ext or ".png"}'
                cv2.imwrite(tmp, dbg)
                os.replace(tmp, self.last_image_path)   # atomic swap; viewer never sees a partial file
        except Exception as exc:                     # noqa: BLE001 -- saving is best-effort
            log.warning('  could not save overlay: %s', exc)

    def _save_recon_plot(self):
        """Reconstruction mode: save the 3D cable-points figure next to the overlay (best-effort)."""
        if self.mode != 'reconstruction' or not self.s.save_images or self._recon is None:
            return
        try:
            os.makedirs(self.image_dir, exist_ok=True)
            path = os.path.join(self.image_dir, f'recon_{self.view_idx:02d}.png')
            if self.reconstructor.save_plot(path):
                log.info('  saved reconstruction plot %s', path)
        except Exception as exc:                     # noqa: BLE001 -- plotting is best-effort
            log.warning('  could not save reconstruction plot: %s', exc)

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

    def _approach(self, T_cam0, P):
        """After a good view: re-centre the cable and step the anchor closer toward the connector
        origin `P` (a base-frame 3-vector -- from the strict fit if it has converged, else a rough
        origin). Returns the new anchor (unchanged if there is no origin to aim at yet)."""
        if P is None:
            return T_cam0

        P = np.asarray(P, dtype=float)
        C = T_cam0[:3, 3]
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
        """After a good view: step the approach in, and report whether the fit has CONVERGED enough
        to stop (all estimator gates pass AND >= min_good_views banked).

        The APPROACH runs on ANY good view, not only once the strict fit converges: it steers by the
        converged origin when available, else by a ROUGH origin from the accumulated views (see
        ConnectorEstimator.rough_origin) -- so the camera starts closing in immediately (from the
        2nd view, once there is a baseline to triangulate). The strict estimate() -- which decides
        when to STOP -- is only attempted once min_inlier_views are banked, so early views don't
        spam failed-fit warnings."""
        T_conn = (self.estimator.estimate()
                  if self.good_views >= self.estimator.min_inlier_views else None)
        raw_P = T_conn[:3, 3] if T_conn is not None else self.estimator.rough_origin()
        # Rate-limit the estimate BETWEEN views so a single outlier cannot lunge the approach. The
        # origin's per-view jump is clamped; when a full pose exists, its axis rotation too, and the
        # pose is rebuilt from the clamped origin+axis (same convention as the estimator).
        P = self._rate_limit_origin(raw_P) if raw_P is not None else None
        if T_conn is not None:
            axis = self._rate_limit_axis(T_conn[:3, 0])
            T = np.eye(4)
            T[:3, :3] = frame_from_axis(axis, self.estimator.up_axis)
            T[:3, 3] = P
            T_conn = T
        T_cam0 = self._approach(T_cam0, P)             # step in on ANY detection with an origin
        return T_cam0, self._converged_pose(T_conn)

    def _rate_limit_origin(self, P):
        """Clamp the vision origin's per-view movement to max_view_delta_m (0 = off). Returns the
        clamped point and remembers it for the next view."""
        P = np.asarray(P, dtype=float)
        m = self.s.max_view_delta_m
        if m > 0 and self._last_origin is not None:
            d = P - self._last_origin
            n = float(np.linalg.norm(d))
            if n > m:
                P = self._last_origin + d / n * m
                log.warning('  vision origin jumped %.0f mm > %.0f mm cap -- clamped (outlier?).',
                            n * 1000, m * 1000)
        if m > 0:
            self._last_origin = P
        return P

    def _rate_limit_axis(self, axis):
        """Clamp the connector axis's per-view rotation to max_view_delta_deg (0 = off). Slerps the
        new axis back toward the previous one when the jump exceeds the cap; near-antiparallel jumps
        (an axis flip) are rejected outright."""
        axis = np.asarray(axis, dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        deg = self.s.max_view_delta_deg
        if deg > 0 and self._last_axis is not None:
            la = self._last_axis
            ang = float(np.arccos(float(np.clip(np.dot(axis, la), -1.0, 1.0))))
            cap = np.radians(deg)
            if ang > cap:
                s = float(np.sin(ang))
                if s < 1e-6:
                    axis = la.copy()                   # ~antiparallel flip -- keep the last axis
                else:
                    t = cap / ang
                    axis = (np.sin((1 - t) * ang) * la + np.sin(t * ang) * axis) / s
                    axis = axis / (np.linalg.norm(axis) + 1e-12)
                log.warning('  connector axis jumped %.1f deg > %.1f deg cap -- clamped (outlier?).',
                            np.degrees(ang), deg)
        if deg > 0:
            self._last_axis = axis
        return axis

    def _converged_pose(self, T_conn):
        """The pose to RETURN if the scan has CONVERGED, else None -- mode aware.

        fuse: the strict estimate (already gated) once min_good_views are banked. reconstruction:
        the reconstruction-refined pose once its cross-view reproj error AND its view-to-view origin
        shift are both under threshold (the dual stability criterion) with min_good_views banked."""
        if self.good_views < self.s.min_good_views:
            return None
        if self.mode == 'reconstruction':
            res = self._recon
            if res is None:
                return None
            if (res.reproj_rms_px <= self.s.recon_max_reproj_px
                    and res.pose_shift_m <= self.s.recon_min_pose_shift_m):
                log.info('  reconstruction CONVERGED: reproj %.2f <= %.2f px AND origin shift '
                         '%.1f <= %.1f mm.', res.reproj_rms_px, self.s.recon_max_reproj_px,
                         res.pose_shift_m * 1000, self.s.recon_min_pose_shift_m * 1000)
                return res.T
            log.info('  reconstruction not yet stable: reproj %.2f px (need <= %.2f), origin shift '
                     '%.1f mm (need <= %.1f) -- more views.', res.reproj_rms_px,
                     self.s.recon_max_reproj_px, res.pose_shift_m * 1000,
                     self.s.recon_min_pose_shift_m * 1000)
            return None
        return T_conn                                  # fuse mode: estimate() already gated

    def _final_pose(self):
        """The best available pose after a refine sweep (mode aware) -- the completion fallback."""
        if self.mode == 'reconstruction':
            res = self.reconstructor.reconstruct()
            return res.T if res is not None else None
        return self.estimator.estimate()

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
        self._last_origin = None
        self._last_axis = None
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
                    T = self._final_pose()
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
