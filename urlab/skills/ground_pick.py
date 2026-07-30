"""GroundPlaneScanner -- single-image, human-in-the-loop cable pose estimation.

An alternative to the multi-view scan (scan.mode: ground_plane), for when the junction fusion is
struggling. From ONE image it detects EVERY cable's junction + ends, NUMBERS them on a saved image,
and asks the user which connector is the target. The pose is then estimated WITHOUT triangulation,
by assuming the cable lies on a GROUND PLANE parallel to the robot XY plane at a known height
(ground_plane.z_m):

  * position -- the junction pixel is back-projected onto that plane;
  * heading (yaw about base z) -- the CONNECTOR's local direction at the junction (the direction the
    detector labels and draws), projected onto the plane. This matches the multi-view connector axis
    (so the same grasp geometry aligns) and is robust for a curved cable, where the free-end->junction
    CHORD would be off by the curve angle. The free-end->junction chord is kept only as a fallback if
    the direction can't be projected;
  * orientation -- x = heading, z = up (frame_from_axis), the connector-frame convention.

At the selection prompt the user can enter 'n' to take a NEW image from a small camera translation
(useful when the junction isn't labelled in the first view). It returns T_base_connector just like
CableScanner.scan(), so the app grasps + retries unchanged. The selection is cached, so grasp
retries re-estimate the SAME cable (re-detected) without re-prompting.
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
        self.new_view_delta = float(gp.get('new_view_translation_m', 0.04))   # 'n' camera step
        self.z_step = float(gp.get('z_step_m', 0.05))       # 'z' step toward the cable (optical +z)
        self._dir_px = 24.0                 # pixels along the connector direction, to project a heading
        self._selection = None              # cached target junction (base frame); persists across retries
        self._base_view = None              # camera pose at scan start, for new-view nudges
        self._nudge_i = 0

        from datetime import datetime
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.image_dir = os.path.join(data_root, gp.get('images_subdir', 'ground_scan'), stamp)
        self.labeled_path = gp.get('labeled_image_path', None) or os.path.join(
            data_root, 'ground_scan_labeled.png')
        # Mirror the numbered selection image to the SHARED live path too (last_camera_image), so a
        # viewer left open there -- as with the multi-view scan -- shows the ground-scan labelling.
        self.save_last_image = bool(cfg.get('save_last_camera_image', True))
        self.last_image_path = cfg.get('last_camera_image_path', None) or os.path.join(
            data_root, 'last_camera_image.png')

    def reset(self):
        """No-op: keep the user's cable selection across grasp retries (they pick once per run)."""

    def reselect(self):
        """Drop the cached cable selection AND re-anchor the view base at the CURRENT camera
        pose, so the next scan() re-detects, re-numbers, and RE-PROMPTS the operator from here.
        Used by the slip recovery: a dropped cable lands somewhere new, so the cached junction
        (and the auto re-match against it) is stale."""
        self._selection = None
        self._base_view = None

    # ------------------------------------------------------------------ scan
    def scan(self, confirm=None):
        """Capture, let the user pick a cable (once; 'n' takes a new view), and return its
        ground-plane T_base_connector. None on abort / no detection."""
        if self._base_view is None and self.robot is not None:
            self._base_view = self.robot.camera().copy()

        # RETRY: reuse the cached selection (re-detect + re-match the SAME cable, no prompt).
        if self._selection is not None:
            frame = self.camera.capture()
            cables = self.detector.detect_all(frame, self.max_cables)
            self._save_labeled()
            return self._pose(self._match_cached(self._project_all(cables, frame)))

        # FIRST RUN: capture -> detect -> prompt, looping on new-view requests.
        while True:
            frame = self.camera.capture()
            cables = self.detector.detect_all(frame, self.max_cables)
            self._save_labeled()
            projected = self._project_all(cables, frame)
            choice = self._prompt(len(cables))            # int index | 'new' | 'closer' | None
            if choice is None:
                return None
            if choice == 'new':
                self._nudge_view()
                continue
            if choice == 'closer':
                self._move_toward_cable()
                continue
            if projected[choice] is None:
                print('  that cable could not be projected onto the plane -- pick another or "n".')
                continue
            self._selection = projected[choice]['junction'].copy()
            return self._pose(projected[choice])

    # ------------------------------------------------------------------ geometry
    def _project_all(self, cables, frame):
        """Each cable's ground-plane {'junction': P, 'heading': v}, or None where it can't project.
        heading = the CONNECTOR direction projected to the plane; free-end->junction chord fallback."""
        out = []
        for cab in cables:
            ju, jv, jyaw = cab['junction']
            Pj = self._project((ju, jv), frame)
            if Pj is None:
                out.append(None)
                continue
            du, dv = float(np.cos(jyaw)), float(np.sin(jyaw))
            P_along = self._project((ju + self._dir_px * du, jv + self._dir_px * dv), frame)
            heading = None
            if P_along is not None:
                heading = P_along - Pj
                heading[2] = 0.0
            if heading is None or float(np.linalg.norm(heading)) < 1e-6:   # fallback: free-end chord
                ends = [p for p in (self._project(e[:2], frame) for e in cab['ends']) if p is not None]
                if ends:
                    fe = max(ends, key=lambda e: float(np.linalg.norm(e - Pj)))
                    heading = Pj - fe
                    heading[2] = 0.0
            if heading is None or float(np.linalg.norm(heading)) < 1e-6:
                out.append(None)
                continue
            out.append({'junction': Pj, 'heading': heading})
        return out

    def _pose(self, target):
        """T_base_connector from a projected {'junction', 'heading'} (x = heading, z = up)."""
        if target is None:
            return None
        Pj, heading = target['junction'], target['heading']
        T = np.eye(4)
        T[:3, :3] = frame_from_axis(heading, [0.0, 0.0, 1.0])
        T[:3, 3] = Pj
        log.info('Ground-plane pose: origin %s mm on z=%.0f mm, heading %.1f deg (connector direction).',
                 (Pj * 1000).round(1), self.plane_z * 1000,
                 np.degrees(np.arctan2(heading[1], heading[0])))
        return T

    def _match_cached(self, projected):
        """The projected cable nearest the cached target junction (for grasp retries)."""
        valid = [p for p in projected if p is not None]
        if not valid:
            log.error('  no projectable cable on retry.')
            return None
        t = min(valid, key=lambda p: float(np.linalg.norm(p['junction'] - self._selection)))
        self._selection = t['junction'].copy()
        log.info('  retry: re-using the selected cable (nearest to the cached target).')
        return t

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

    # ------------------------------------------------------------------ user interaction
    def _prompt(self, n):
        """Ask for the target cable number (1..n), 'n' for a new view, 'z' to move toward the cable,
        or 'q' to abort. Returns a 0-based index, 'new', 'closer', or None."""
        zmm = self.z_step * 1000
        if n == 0:
            print(f'\n[ground_plane] NO cable detected in this view -- see {self.labeled_path}')
            print(f'Enter "n" for a NEW view, "z" to move {zmm:.0f} mm toward the cable, or "q":')
        else:
            print(f'\n[ground_plane] {n} cable(s) detected -- see {self.labeled_path}')
            print(f'Enter the target NUMBER (1-{n}), "n" for a new view, "z" to move {zmm:.0f} mm '
                  'closer, or "q":')
        while True:
            try:
                raw = input('target #> ').strip().lower()
            except EOFError:
                return None
            if raw in ('q', 'quit', 'abort'):
                return None
            if raw in ('n', 'new', 'view'):
                return 'new'
            if raw in ('z', 'closer', 'down'):
                return 'closer'
            try:
                idx = int(raw) - 1
            except ValueError:
                print(f'  enter a number 1-{n}, "n" (new view), "z" (closer), or "q".')
                continue
            if 0 <= idx < n:
                return idx
            print(f'  out of range -- enter 1-{n} (or "n"/"z"/"q").')

    def _nudge_view(self):
        """Move the camera a small translation (cycling N/E/S/W around the start pose) for a fresh
        image -- requested when the junction isn't labelled in the current view."""
        if self.robot is None or self._base_view is None:
            return
        d = self.new_view_delta
        ring = [(d, 0, 0), (0, d, 0), (-d, 0, 0), (0, -d, 0)]
        off = np.array(ring[self._nudge_i % len(ring)], dtype=float)
        self._nudge_i += 1
        T = self._base_view.copy()
        T[:3, 3] = T[:3, 3] + off                           # pure base-frame translation (same aim)
        log.info('  new view: translating the camera %s mm for a fresh image.',
                 (off * 1000).round(0))
        self.robot.move_camera(T, 'ground-plane new view')

    def _move_toward_cable(self):
        """Move the camera z_step TOWARD the cable, along its optical axis (+optical z = forward, the
        view direction), for a closer look. Re-anchors the new-view ring at the closer pose."""
        if self.robot is None:
            return
        T = self.robot.camera().copy()
        T[:3, 3] = T[:3, 3] + self.z_step * T[:3, 2]        # +optical z -> toward what the camera sees
        log.info('  moving %.0f mm toward the cable (closer view).', self.z_step * 1000)
        if self.robot.move_camera(T, 'ground-plane move toward cable'):
            self._base_view = self.robot.camera().copy()

    def _save_labeled(self):
        """Write the NUMBERED overlay (for the user to read) to the ground-scan path, a per-run
        archive, AND the shared last_camera_image live path so any open viewer shows it."""
        dbg = getattr(self.detector, 'last_debug', None)
        if dbg is None:
            return
        try:
            import cv2
            self._atomic_write(dbg, self.labeled_path)
            if self.save_last_image:
                self._atomic_write(dbg, self.last_image_path)      # mirror to the shared live path
            os.makedirs(self.image_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self.image_dir, f'view_{self._nudge_i:02d}.png'), dbg)
            log.info('  saved the numbered selection image %s', self.labeled_path)
        except Exception as exc:                     # noqa: BLE001 -- saving is best-effort
            log.warning('  could not save the labelled image: %s', exc)

    @staticmethod
    def _atomic_write(img, path):
        """Write `img` to `path` via a temp file + rename, so a viewer never reads a partial frame."""
        import cv2
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        root, ext = os.path.splitext(path)
        tmp = f'{root}.tmp{ext or ".png"}'
        cv2.imwrite(tmp, img)
        os.replace(tmp, path)
