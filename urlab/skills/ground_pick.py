"""GroundPlaneScanner -- single-image, human-in-the-loop cable pose estimation.

An alternative to the multi-view scan (scan.mode: ground_plane), for when the junction fusion is
struggling. From ONE image it detects EVERY cable's junction + ends, NUMBERS them on a saved image,
and asks the user which connector is the target. The pose is then estimated WITHOUT triangulation,
by assuming the cable lies on a GROUND PLANE parallel to the robot XY plane at a known height
(ground_plane.z_m):

  * position -- the junction pixel is back-projected onto that plane;
  * heading (yaw about base z) -- the in-plane vector from the cable's FREE end to the junction;
  * orientation -- x = heading, z = up (frame_from_axis), matching the connector-frame convention.

It returns T_base_connector just like CableScanner.scan(), so the app grasps + retries unchanged.
The user's selection is cached, so grasp retries re-estimate the SAME cable (re-detected, in case a
failed grasp nudged it) without re-prompting.
"""

import os

import numpy as np

from .. import log as urlog
from ..transforms import frame_from_axis

log = urlog.get('ground-scan')


class GroundPlaneScanner:
    """Same interface as CableScanner (scan()/reset()/.camera/.estimator) so the apps use it as a
    drop-in when scan.mode is 'ground_plane'."""

    def __init__(self, robot, camera, detector, estimator, cfg, data_root='data'):
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.estimator = estimator          # unused for the fit; kept so the app's .estimator.reset() works
        if not hasattr(detector, 'detect_all'):
            raise ValueError("scan.mode 'ground_plane' needs the junction detector's detect_all -- "
                             "set sam3.mode: junction.")
        gp = cfg.section('ground_plane')
        self.plane_z = float(gp.get('z_m', -0.758))
        self.max_cables = int(gp.get('max_cables', 8))
        self._selection = None              # cached target junction (base frame); persists across retries

        from datetime import datetime
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.image_dir = os.path.join(data_root, gp.get('images_subdir', 'ground_scan'), stamp)
        self.labeled_path = gp.get('labeled_image_path', None) or os.path.join(
            data_root, 'ground_scan_labeled.png')

    def reset(self):
        """No-op: keep the user's cable selection across grasp retries (they pick once per run)."""

    # ------------------------------------------------------------------ scan
    def scan(self, confirm=None):
        """Capture one image, let the user pick a cable (once), and return its ground-plane
        T_base_connector. None on abort / no detection."""
        frame = self.camera.capture()
        cables = self.detector.detect_all(frame, self.max_cables)
        self._save_labeled()
        if not cables:
            log.error('No cables detected -- cannot estimate a pose.')
            return None

        # Project every cable's junction + ends onto the ground plane.
        projected = []
        for cab in cables:
            Pj = self._project(cab['junction'][:2], frame)
            ends = [p for p in (self._project(e[:2], frame) for e in cab['ends']) if p is not None]
            projected.append({'junction': Pj, 'ends': ends} if (Pj is not None and ends) else None)

        target = self._choose_target(projected, len(cables), confirm)
        if target is None:
            return None

        Pj = target['junction']
        free_end = max(target['ends'], key=lambda e: float(np.linalg.norm(e - Pj)))  # farther = free tip
        heading = Pj - free_end
        heading[2] = 0.0                                    # keep the axis IN the ground plane
        if float(np.linalg.norm(heading)) < 1e-6:
            heading = np.array([1.0, 0.0, 0.0])

        T = np.eye(4)
        T[:3, :3] = frame_from_axis(heading, [0.0, 0.0, 1.0])
        T[:3, 3] = Pj
        log.info('Ground-plane pose: origin %s mm on z=%.0f mm, heading %.1f deg.',
                 (Pj * 1000).round(1), self.plane_z * 1000,
                 np.degrees(np.arctan2(heading[1], heading[0])))
        return T

    def _choose_target(self, projected, n, confirm):
        """The selected cable's projected junction/ends. First run: prompt the user and cache the
        pick. Retry: re-match the cached target to the current detections (nearest junction), so the
        SAME cable is re-estimated without re-prompting."""
        if self._selection is None:
            idx = self._prompt(n)
            if idx is None:
                return None
            if projected[idx] is None:
                log.error('Cable #%d could not be projected onto the ground plane (ray misses it).',
                          idx + 1)
                return None
            self._selection = projected[idx]['junction'].copy()
            return projected[idx]

        valid = [p for p in projected if p is not None]
        if not valid:
            log.error('No projectable cable on retry.')
            return None
        target = min(valid, key=lambda p: float(np.linalg.norm(p['junction'] - self._selection)))
        self._selection = target['junction'].copy()
        log.info('  retry: re-using the selected cable (nearest to the cached target).')
        return target

    def _project(self, uv, frame):
        """Back-project pixel (u, v) onto the ground plane z = plane_z (base frame). None if the ray
        is parallel to, or points away from, the plane."""
        if getattr(frame, 'T_base_cam', None) is None or getattr(frame, 'K', None) is None:
            return None
        C = frame.T_base_cam[:3, 3]
        g = frame.T_base_cam[:3, :3] @ (np.linalg.inv(frame.K) @ np.array([uv[0], uv[1], 1.0]))
        if abs(g[2]) < 1e-9:
            return None
        t = (self.plane_z - C[2]) / g[2]
        if t <= 0:                                          # plane is behind the camera along the ray
            return None
        return C + t * g

    def _prompt(self, n):
        """Ask the user for the target cable number (1..n). None on abort/EOF."""
        print(f'\n[ground_plane] {n} cable(s) detected -- see {self.labeled_path}')
        print(f'Enter the NUMBER (#) of the target connector, 1-{n} (q to abort):')
        while True:
            try:
                raw = input('target connector #> ').strip().lower()
            except EOFError:
                return None
            if raw in ('q', 'quit', 'abort'):
                return None
            try:
                idx = int(raw) - 1
            except ValueError:
                print(f'  enter a number 1-{n}, or q.')
                continue
            if 0 <= idx < n:
                return idx
            print(f'  out of range -- enter 1-{n}.')

    def _save_labeled(self):
        """Write the NUMBERED overlay (for the user to read) to a stable path + a per-run archive."""
        dbg = getattr(self.detector, 'last_debug', None)
        if dbg is None:
            return
        try:
            import cv2
            os.makedirs(os.path.dirname(self.labeled_path) or '.', exist_ok=True)
            root, ext = os.path.splitext(self.labeled_path)
            tmp = f'{root}.tmp{ext or ".png"}'
            cv2.imwrite(tmp, dbg)
            os.replace(tmp, self.labeled_path)
            os.makedirs(self.image_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self.image_dir, 'labeled.png'), dbg)
            log.info('  saved the numbered selection image %s', self.labeled_path)
        except Exception as exc:                     # noqa: BLE001 -- saving is best-effort
            log.warning('  could not save the labelled image: %s', exc)
