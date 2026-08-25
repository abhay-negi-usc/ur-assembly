"""Locate a fixture from the fiducials bolted around it -- sweep, servo-refine, fuse, vote.

TWO APPS, ONE FILE. `apps/marker_calibration` runs the capture with the fixture at a pose it
already knows (the recorded `targets:` mate) and SOLVES for where the target sits in each
marker's frame; `apps/bnc_assembly --target_source visual` runs the same capture, reads those
poses back and solves for the fixture. Same capture, same fusion, opposite unknown -- so they
share the code and a calibration can never be fused differently from the way it is used.

WHY MULTIPLE VIEWS. A single ArUco pose is a four-point PnP solve on a small planar square,
and the ill-conditioned direction is exactly the one that matters here: rotation about an axis
IN the marker plane, which trades against depth. Two views from a few centimetres apart see
that trade-off differently, so translating the camera between captures is what actually adds
information. Capturing ten frames without moving averages the sensor noise and leaves the
geometric bias untouched.

VISUAL SERVOING (ViewPlan.servo, opt-in). After the coarse sweep the camera SERVOS to a
canonical vantage per marker -- centred on the marker at `distance_m` along its normal --
re-detecting and re-centring until two successive detections agree, then captures the
refinement views (the vantage plus an aimed parallax ring) that JOIN that marker's sweep
views in the fusion; the closeness weighting is what makes them dominate. Centred, close, fixed-distance viewing removes the vantage-dependent part of the PnP
bias, and doing it identically for every marker -- and identically at calibration and at run
time -- is what lets the remaining bias cancel. The vantage ROLL is aligned to the marker
only up to the nearest 90 deg about the view axis: pose estimation is invariant to that
rotation, so the closest quarter turn is commanded and the wrist never winds further for
nothing. Servoing to one marker takes the others out
of frame by design: the routine returns to the OVERVIEW pose (where the sweep started, all
markers in frame) between markers to reset the view.

CERTAINTY-WEIGHTED FUSION. Every view is stored with its camera-to-marker distance and fused
with weight d^-view_weight_power (closer views are more accurate: PnP translation error grows
superlinearly with range). Views beyond the `max_camera_distance_mm` standoff cap are never
commanded and never fused. Each marker's accumulated view weight is its certainty, and the
across-marker vote uses it.

JOINT PnP ESTIMATION (the default at run time). Rather than solving each marker's four
corners for its own pose and averaging the per-marker answers, every view's detected corners
of ALL rig markers are solved as ONE PnP over one rigid object -- the whole rig acting as a
single large marker, exactly the way one marker's four corners localize that marker. The
effective baseline becomes the rig's extent instead of one square's side, which is what
collapses the small-marker depth/tilt ambiguity. Per-view estimates are then fused with the
same closeness weighting single-marker views get.

WHY THE PER-MARKER VOTE STILL RUNS. Each marker with a recorded pose is an INDEPENDENT vote
on the fixture -- independent because the errors that dominate (printed size, how square it
is glued) do not correlate between markers. The vote is the CONSISTENCY GATE: a moved or
re-stuck marker stands out as a disagreeing vote and REFUSES the run, where a joint solve
would quietly absorb it into a small bias. The gate must pass before the joint estimate is
trusted (and it is the fallback when OpenCV or the corner data is unavailable).

THE FAILURE THIS IS BUILT AROUND is a marker that is right about everything except which
fixture it is on -- moved, re-stuck, re-printed at another size. It produces a confident pose
that disagrees with its neighbours, so every vote is reported individually and
`max_disagreement_mm` rejects the fused answer rather than averaging a good rig with a stale
marker.
"""

import csv as _csv
import os
import time

import numpy as np

from .. import log as urlog
from ..transforms import (average_pose, inverse, look_at, matrix_to_xyzrpy, pose_error,
                          xyzrpy_to_matrix)
from .servo import camera_on_marker

log = urlog.get('marker-loc')


class ServoPlan:
    """The `marker_views.servo:` block -- per-marker closed-loop refinement (opt-in)."""

    def __init__(self, block):
        b = dict(block or {})
        self.enabled = bool(b.get('enabled', False))
        self.distance_m = float(b.get('distance_m', 0.15))
        self.max_iterations = max(1, int(b.get('max_iterations', 4)))
        self.pos_tol_mm = float(b.get('pos_tol_mm', 0.5))
        self.ang_tol_deg = float(b.get('ang_tol_deg', 0.5))
        self.ring_mm = float(b.get('ring_mm', 25.0))
        self.ring_views = max(0, int(b.get('ring_views', 4)))


class ViewPlan:
    """The `marker_views:` config block: where to look from, and how long to sit still.

    Offsets are RELATIVE to the camera pose the sweep starts at -- the markers are assumed to
    be in view already, which is what lets this be a small local sweep instead of a search."""

    def __init__(self, block):
        b = dict(block or {})
        self.settle_s = float(b.get('settle_s', 0.4))
        self.frames_per_view = max(1, int(b.get('frames_per_view', 3)))
        self.min_views = max(1, int(b.get('min_views', 3)))
        self.min_markers = max(1, int(b.get('min_markers', 1)))
        self.require_all = bool(b.get('require_all_markers', False))
        self.aim_at_markers = bool(b.get('aim_at_markers', True))
        self.max_view_spread_mm = _opt_float(b.get('max_view_spread_mm', 3.0))
        self.max_view_spread_deg = _opt_float(b.get('max_view_spread_deg', 3.0))
        self.max_disagreement_mm = _opt_float(b.get('max_disagreement_mm', 5.0))
        self.max_disagreement_deg = _opt_float(b.get('max_disagreement_deg', 5.0))
        # RANSAC over the per-marker votes: outvote a marker that has moved, been re-stuck or
        # been reprinted, instead of failing the whole localization on the disagreement gate.
        # Needs THREE markers to mean anything -- with two, each is a consensus of one and there
        # is nothing to say which of them moved, so the all-or-nothing gate is the honest answer.
        # The inlier band defaults to the disagreement gate: a marker the run would have accepted
        # in the average is one the consensus should accept too.
        self.ransac = bool(b.get('ransac', True))
        self.ransac_inlier_mm = _opt_float(b.get('ransac_inlier_mm', self.max_disagreement_mm))
        self.ransac_inlier_deg = _opt_float(b.get('ransac_inlier_deg', self.max_disagreement_deg))
        self.ransac_min_inliers = max(2, int(b.get('ransac_min_inliers', 2)))
        # Certainty weighting: a view at distance d fuses with weight d^-power (0 = unweighted).
        self.view_weight_power = float(b.get('view_weight_power', 2.0))
        # The STANDOFF CAP: the camera never plans a view farther than this from the markers,
        # and a detection captured from beyond it is never fused. null disables.
        cap = _opt_float(b.get('max_camera_distance_mm', 500.0))
        self.max_camera_distance_m = None if cap is None else cap / 1000.0
        # JOINT PnP: solve every view's detected corners of ALL rig markers as one rigid
        # object (run-time localization only; the per-marker vote stays as the gate).
        self.joint_pnp = bool(b.get('joint_pnp', True))
        # MULTI-VIEW REFINEMENT (calibration): re-solve each marker's pose over ALL of its
        # corner observations at once, minimized in pixel space.
        self.multiview_refine = bool(b.get('multiview_refine', True))
        # Save every image the estimate was computed from, annotated + indexed, into a
        # marker_images/ subdirectory of the run's output folder.
        self.save_images = bool(b.get('save_images', True))
        self.servo = ServoPlan(b.get('servo'))
        if (self.servo.enabled and self.max_camera_distance_m is not None
                and self.servo.distance_m > self.max_camera_distance_m):
            raise ValueError(
                f'marker_views.servo.distance_m ({self.servo.distance_m:.3f} m) is beyond the '
                f'max_camera_distance_mm standoff cap ({self.max_camera_distance_m:.3f} m)')
        offsets = b.get('offsets')
        if offsets is None:
            # A default that actually adds information: a ring of camera TRANSLATIONS around the
            # start pose (the direction PnP is weakest in), not a set of re-captures in place.
            offsets = [{'xyz_mm': [0.0, 0.0, 0.0]}]
            for ang in (0.0, 90.0, 180.0, 270.0):
                r = np.radians(ang)
                offsets.append({'xyz_mm': [40.0 * float(np.cos(r)), 40.0 * float(np.sin(r)), 0.0]})
            offsets.append({'xyz_mm': [0.0, 0.0, -25.0]})
        self.offsets = [_offset_matrix(o, i) for i, o in enumerate(offsets)]

    def describe(self):
        base = ('%d view%s, %d frame%s each, %s'
                % (len(self.offsets), '' if len(self.offsets) == 1 else 's',
                   self.frames_per_view, '' if self.frames_per_view == 1 else 's',
                   'aimed at the markers' if self.aim_at_markers else 'orientation held'))
        if self.servo.enabled:
            base += (', then servo-refined per marker at %.0f mm (+%d-view ring)'
                     % (self.servo.distance_m * 1000.0, self.servo.ring_views))
        if self.joint_pnp:
            base += ', joint-PnP estimate'
        if self.max_camera_distance_m is not None:
            base += ', standoff cap %.0f mm' % (self.max_camera_distance_m * 1000.0)
        return base


def _opt_float(v):
    return None if v is None else float(v)


def _offset_matrix(o, i):
    """One camera-frame offset -> 4x4. Monitor units (mm/deg) or SI, the unit in the key name."""
    from ..config import _pose_si

    d = dict(o or {})
    unknown = set(d) - {'xyz', 'rpy', 'xyz_mm', 'rpy_deg'}
    if unknown:
        raise ValueError(f'marker_views.offsets[{i}] has unknown key(s) {sorted(unknown)}')
    p = _pose_si(d)
    return xyzrpy_to_matrix(p.get('xyz', [0.0, 0.0, 0.0]), p.get('rpy', [0.0, 0.0, 0.0]))


def _view_weight(d_m, power):
    """Closer views are more certain: weight d^-power, with the distance floored so a
    degenerate zero-range reading cannot dominate everything."""
    return float(max(d_m, 1e-3)) ** -float(power)


def _cam_position(frame, robot):
    """The camera position a detection was made from (the frame's own stamped pose wins)."""
    T = getattr(frame, 'T_base_cam', None)
    return (T if T is not None else robot.camera())[:3, 3]


def sweep(robot, camera, detector, plan, wanted=None, on_view=None, corner_log=None):
    """Drive the configured views and detect at each.
    Returns {marker_id: [(T_base_marker, camera_distance_m), ...]}.

    `on_view(index, frame, poses_in_camera)` is called once per view that captured anything --
    the calibration app uses it to save an annotated image, which is the only artefact that
    shows WHY a marker was missed rather than that it was.

    The camera pose is sampled AT CAPTURE (see perception/camera.Frame), so a detection can
    never be credited to the wrong viewpoint -- which is what makes fusing across a moving
    camera safe at all. A view that reaches nothing is SKIPPED, not fatal. Once the markers'
    whereabouts are known, any view that would sit beyond the max_camera_distance_mm standoff
    cap is PULLED IN along its view ray instead of being commanded out there."""
    T_start = robot.camera()
    aim = None
    seen = {}
    for k, off in enumerate(plan.offsets):
        T_view = T_start @ off
        if plan.aim_at_markers and aim is not None:
            # Re-point at the markers found so far. look_at only fixes the view ray; the roll
            # comes from the reference, and passing the nominal view keeps it near the start.
            T_view = look_at(T_view[:3, 3], aim, T_view)
        if aim is not None and plan.max_camera_distance_m is not None:
            v = T_view[:3, 3] - aim
            d = float(np.linalg.norm(v))
            if d > plan.max_camera_distance_m:
                T_view = T_view.copy()
                T_view[:3, 3] = aim + v * (plan.max_camera_distance_m / d)
                log.info('  view %d pulled in to the %.0f mm standoff cap.',
                         k + 1, plan.max_camera_distance_m * 1000.0)
        if k > 0 or not _at(robot.camera(), T_view):
            ok = robot.arm.move_frame_to(T_view, robot.T_tool0_cam, f'marker view {k + 1}')
            if not ok:
                log.warning('  view %d/%d: the move did not finish -- skipping it.',
                            k + 1, len(plan.offsets))
                continue
        if plan.settle_s > 0:
            time.sleep(plan.settle_s)
        poses, corners, frame = _capture_stop(camera, detector, plan.frames_per_view, wanted)
        if on_view is not None and frame is not None:
            on_view(k, frame, detector.detect(frame))
        if not poses:
            log.warning('  view %d/%d: no markers detected.', k + 1, len(plan.offsets))
            continue
        cam_p = _cam_position(frame, robot)
        for mid, T_m in poses.items():
            seen.setdefault(mid, []).append(
                (T_m, float(np.linalg.norm(T_m[:3, 3] - cam_p))))
        _log_corner_view(corner_log, corners, frame, robot)
        aim = np.mean([m[-1][0][:3, 3] for m in seen.values()], axis=0)
        log.info('  view %d/%d: marker%s %s.', k + 1, len(plan.offsets),
                 '' if len(poses) == 1 else 's', ', '.join(str(m) for m in sorted(poses)))
    return seen


def _at(T_a, T_b, tol_m=1e-3, tol_rad=np.radians(0.2)):
    lin, ang = pose_error(T_a, T_b)
    return lin <= tol_m and ang <= tol_rad


def _quarter_roll(T_marker, distance_m, T_cam_now):
    """The vantage's in-plane roll: the multiple of 90 deg about the view axis that brings
    the camera-on-marker pose closest to the camera's CURRENT attitude. ArUco pose
    estimation is invariant to rotation about the view axis, so only alignment up to the
    nearest quarter turn is worth commanding -- anything tighter is wrist travel for
    nothing. All four candidates share the same position, so the comparison is purely
    angular."""
    best_roll, best_ang = 0.0, None
    for k in range(4):
        yaw = k * np.pi / 2.0
        _lin, ang = pose_error(
            T_cam_now, camera_on_marker(T_marker, distance_m, [np.pi, 0.0, yaw]))
        if best_ang is None or ang < best_ang:
            best_roll, best_ang = yaw, ang
    return best_roll


def _capture_stop(camera, detector, n_frames, wanted=None):
    """Capture n frames at ONE stop: ({id: averaged T_base_marker}, {id: averaged (4,2)
    pixel corners}, last frame). The repeats at a stop measure sensor noise only, so they
    collapse to one view -- for the poses and for the corners alike. Corner collection is
    skipped for detectors that do not expose detect_corners."""
    poses, corners, frame = {}, {}, None
    for _ in range(n_frames):
        frame = camera.capture()
        for mid, T in detector.detect_in_base(frame).items():
            if wanted is not None and int(mid) not in wanted:
                continue
            poses.setdefault(int(mid), []).append(T)
        if hasattr(detector, 'detect_corners'):
            for mid, c in detector.detect_corners(frame).items():
                if wanted is not None and int(mid) not in wanted:
                    continue
                corners.setdefault(int(mid), []).append(np.asarray(c, dtype=float))
    return ({m: average_pose(ts)[0] for m, ts in poses.items()},
            {m: np.mean(cs, axis=0) for m, cs in corners.items()}, frame)


def _log_corner_view(corner_log, corners, frame, robot):
    """One joint-PnP input record per stop: the corners, the intrinsics, and the camera
    pose the capture was made from."""
    if corner_log is None or not corners:
        return
    T_cam = getattr(frame, 'T_base_cam', None)
    corner_log.append({'corners': corners, 'K': frame.K, 'D': frame.D,
                       'T_base_cam': T_cam if T_cam is not None else robot.camera()})


def servo_refine(robot, camera, detector, plan, seen, T_overview=None, on_view=None,
                 corner_log=None):
    """Per-marker VISUAL SERVOING refinement (ViewPlan.servo). One marker at a time:

        overview -> servo onto the marker's normal at distance_m (re-detect + re-centre
        until two successive detections agree) -> capture the vantage + an aimed parallax
        ring -> next marker

    The OVERVIEW hop between markers matters: servoing to one marker takes the others out of
    frame by design, and returning to the pose the sweep ran from is what brings the whole
    rig back into view before the next marker's servo starts.

    Returns {marker_id: [(T_base_marker, distance_m), ...]} -- ADDITIONAL views for every
    marker that refined (merge with merge_refined: they POOL with the sweep views, and the
    closeness weighting makes them dominate; replacing outright would let a small ring fall
    under min_views and silently discard the marker). A marker that will not detect from its
    vantage is left out and its sweep views stand alone.
    `on_view(marker_id, stop_index, frame, poses_in_camera)` is called per capture."""
    sv = plan.servo
    if T_overview is None:
        T_overview = robot.camera()
    refined = {}
    for mid, obs in sorted(seen.items()):
        # OVERVIEW FIRST: the previous marker's servo took this one out of frame.
        if not robot.arm.move_frame_to(T_overview, robot.T_tool0_cam,
                                       f'overview (before marker {mid})'):
            log.warning('  could not return to the overview pose -- refinement stops here; '
                        'the remaining markers keep their sweep views.')
            break
        T_est = average_pose([T for T, _d in obs])[0]
        # The roll about the view axis is chosen ONCE per marker -- the nearest quarter turn
        # to the camera's current attitude -- and held for the whole servo + ring, so the
        # views stay mutually consistent and the wrist never unwinds mid-marker.
        roll = _quarter_roll(T_est, sv.distance_m, robot.camera())

        # ---- servo: centre + square + fix the distance until the detection stops moving ----
        detected = False
        for it in range(1, sv.max_iterations + 1):
            T_cam = camera_on_marker(T_est, sv.distance_m, [np.pi, 0.0, roll])
            if not robot.arm.move_frame_to(T_cam, robot.T_tool0_cam,
                                           f'servo marker {mid} ({it}/{sv.max_iterations})'):
                log.warning('  marker %d: servo move did not finish -- keeping the sweep '
                            'views.', mid)
                detected = False
                break
            if plan.settle_s > 0:
                time.sleep(plan.settle_s)
            T_obs = _capture_stop(camera, detector, plan.frames_per_view)[0].get(int(mid))
            if T_obs is None:
                log.warning('  marker %d: not detected from the servo vantage -- keeping the '
                            'sweep views.', mid)
                detected = False
                break
            lin, ang = pose_error(T_est, T_obs)
            T_est = T_obs
            detected = True
            log.info('  marker %d: servo iteration %d moved the estimate %.2f mm / %.2f deg.',
                     mid, it, lin * 1000.0, np.degrees(ang))
            if lin * 1000.0 <= sv.pos_tol_mm and np.degrees(ang) <= sv.ang_tol_deg:
                break
        else:
            log.warning('  marker %d: servo did not settle within %d iterations -- refining '
                        'from the last vantage anyway.', mid, sv.max_iterations)
        if not detected:
            continue

        # ---- refinement views: the vantage + an aimed ring around it. The centred vantage
        # kills the lateral perspective bias; the ring restores the parallax that
        # disambiguates the planar-pose tilt a centred view alone cannot. ----
        vantage = camera_on_marker(T_est, sv.distance_m, [np.pi, 0.0, roll])
        stops = [vantage]
        for j in range(sv.ring_views):
            a = 2.0 * np.pi * j / sv.ring_views
            p = vantage[:3, 3] + vantage[:3, :3] @ np.array(
                [sv.ring_mm / 1000.0 * np.cos(a), sv.ring_mm / 1000.0 * np.sin(a), 0.0])
            stops.append(look_at(p, T_est[:3, 3], vantage))
        views = []
        for j, T_stop in enumerate(stops):
            if j > 0:                   # already AT the vantage from the servo loop
                if not robot.arm.move_frame_to(T_stop, robot.T_tool0_cam,
                                               f'servo marker {mid} ring {j}'):
                    continue
                if plan.settle_s > 0:
                    time.sleep(plan.settle_s)
            poses, corners, frame = _capture_stop(camera, detector, plan.frames_per_view)
            if on_view is not None and frame is not None:
                on_view(mid, j, frame, detector.detect(frame))
            T_obs = poses.get(int(mid))
            if T_obs is None:
                continue
            d = float(np.linalg.norm(T_obs[:3, 3] - _cam_position(frame, robot)))
            views.append((T_obs, d))
            _log_corner_view(corner_log, corners, frame, robot)
        if views:
            refined[mid] = views
            log.info('  marker %d: REFINED -- %d servoed view%s at ~%.0f mm join its %d '
                     'sweep view%s.', mid, len(views), '' if len(views) == 1 else 's',
                     sv.distance_m * 1000.0, len(obs), '' if len(obs) == 1 else 's')
        else:
            log.warning('  marker %d: no refinement view detected -- sweep views only.', mid)
    # Leave the arm at the overview, ready for whatever comes next.
    robot.arm.move_frame_to(T_overview, robot.T_tool0_cam, 'overview (refinement done)')
    return refined


# =================================================================================================
# annotated image capture -- what the estimate was actually computed from
# =================================================================================================

_INDEX_HEADER = ['image', 'stage', 'view', 'marker_id', 'range_mm',
                 'cam_x_mm', 'cam_y_mm', 'cam_z_mm',
                 'cam_roll_deg', 'cam_pitch_deg', 'cam_yaw_deg',
                 'base_x_mm', 'base_y_mm', 'base_z_mm']


def annotate(detector, frame, poses, header='', notes=()):
    """An annotated copy of `frame`: the detector's own outlines + axes, plus a text label
    per marker (id, range, camera-frame pose) and a header. `poses` are CAMERA-frame -- what
    the on_view hooks hand over. Returns None if OpenCV is unavailable."""
    try:
        import cv2
    except Exception:                              # noqa: BLE001
        return None
    img = detector.draw(frame, poses)
    corners = (detector.detect_corners(frame) if hasattr(detector, 'detect_corners') else {})
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1

    def put(text, x, y, colour):
        # drawn twice: a dark stroke under a bright fill, so the label survives whatever the
        # marker or the background happens to be
        cv2.putText(img, text, (int(x), int(y)), font, scale, (0, 0, 0), thick + 2,
                    cv2.LINE_AA)
        cv2.putText(img, text, (int(x), int(y)), font, scale, colour, thick, cv2.LINE_AA)

    for i, line in enumerate([header] + list(notes)):
        if line:
            put(str(line), 8, 20 + 18 * i, (255, 255, 255))
    for mid, T_cm in sorted(poses.items()):
        xyz, rpy = matrix_to_xyzrpy(T_cm)
        rng = float(np.linalg.norm(T_cm[:3, 3])) * 1000.0
        c = corners.get(int(mid))
        if c is not None:
            x, y = np.asarray(c, dtype=float).mean(axis=0)
            y -= 10.0
        else:                                      # no corner data: stack the labels instead
            x, y = 8.0, img.shape[0] - 24.0 - 34.0 * len(poses)
        put('id %d  %.0f mm' % (int(mid), rng), x, y, (0, 255, 255))
        put('xyz %+.1f %+.1f %+.1f  rpy %+.1f %+.1f %+.1f'
            % (xyz[0] * 1000.0, xyz[1] * 1000.0, xyz[2] * 1000.0,
               np.degrees(rpy[0]), np.degrees(rpy[1]), np.degrees(rpy[2])),
            x, y + 16.0, (0, 255, 255))
    return img


class MarkerImageWriter:
    """Saves every image the localization estimated from, annotated and indexed.

    Images land in <out_dir>/<subdir>/ and index.csv lists, per image, each marker it
    contributed and the pose read from it -- so a suspect fit can be traced back to the
    picture it came from, and a missed marker to the view that missed it. Best-effort
    throughout: an imaging problem never fails a run."""

    def __init__(self, out_dir, detector, subdir='marker_images', enabled=True):
        self.detector = detector
        self.enabled = bool(enabled) and out_dir is not None
        self.dir = os.path.join(out_dir, subdir) if self.enabled else None
        self.rows = []
        self.n = 0
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)

    def _write(self, name, stage, view, frame, poses, header, notes=()):
        if not self.enabled:
            return
        try:
            import cv2
            img = annotate(self.detector, frame, poses, header, notes)
            if img is None:
                return
            cv2.imwrite(os.path.join(self.dir, name), img)
            self.n += 1
            T_bc = getattr(frame, 'T_base_cam', None)
            if not poses:                          # a view that saw nothing is still evidence
                self.rows.append([name, stage, view, '', '', '', '', '', '', '', '',
                                  '', '', ''])
            for mid, T_cm in sorted(poses.items()):
                xyz, rpy = matrix_to_xyzrpy(T_cm)
                base = ((T_bc @ T_cm)[:3, 3] * 1000.0) if T_bc is not None else [None] * 3
                self.rows.append(
                    [name, stage, view, int(mid),
                     round(float(np.linalg.norm(T_cm[:3, 3])) * 1000.0, 2)]
                    + [round(float(v) * 1000.0, 2) for v in xyz]
                    + [round(float(np.degrees(v)), 2) for v in rpy]
                    + [None if v is None else round(float(v), 2) for v in base])
        except Exception as exc:                   # noqa: BLE001 -- imaging is never fatal
            log.debug('could not save the marker image %s: %s', name, exc)

    def sweep_view(self, k, frame, poses):
        """on_view for sweep(): the coarse overview captures."""
        self._write('sweep_%02d.jpg' % (k + 1), 'sweep', k + 1, frame, poses,
                    'SWEEP view %d -- %d marker(s)' % (k + 1, len(poses)))

    def servo_view(self, mid, j, frame, poses):
        """on_view for servo_refine(): the close, centred per-marker captures."""
        kind = 'vantage' if j == 0 else 'ring %d' % j
        self._write('servo_m%02d_%02d.jpg' % (int(mid), j), 'servo', j, frame, poses,
                    'SERVO marker %d -- %s -- %d marker(s)' % (int(mid), kind, len(poses)))

    def finish(self, summary=()):
        """Write index.csv (+ summary.txt when the caller has final estimates) and report
        where the images went."""
        if not self.enabled or not self.n:
            return
        try:
            with open(os.path.join(self.dir, 'index.csv'), 'w', newline='') as fh:
                w = _csv.writer(fh)
                w.writerow(_INDEX_HEADER)
                w.writerows(self.rows)
            if summary:
                with open(os.path.join(self.dir, 'summary.txt'), 'w') as fh:
                    fh.write(chr(10).join(str(s) for s in summary) + chr(10))
            log.info('  %d annotated marker image%s -> %s (index.csv lists what each one '
                     'contributed).', self.n, '' if self.n == 1 else 's', self.dir)
        except Exception as exc:                   # noqa: BLE001
            log.debug('could not write the marker image index: %s', exc)


def merge_refined(seen, refined):
    """Pool the servoed views WITH the sweep views. The certainty weighting (d^-power)
    already makes the close views dominate the fusion; replacing the sweep views instead
    would let a small refinement ring fall under min_views and silently discard the
    marker."""
    for mid, views in refined.items():
        seen.setdefault(mid, []).extend(views)
    return seen


def fuse_markers(seen, plan):
    """{marker_id: (T_base_marker, lin_rms_m, ang_rms_rad, n_views, weight)} for markers with
    enough views.

    Views are (pose, camera_distance_m) pairs and the fusion is CERTAINTY-WEIGHTED: a view at
    distance d carries weight d^-view_weight_power, because closer views are more accurate
    (PnP translation error grows superlinearly with range). Views beyond the standoff cap are
    dropped outright. The returned `weight` -- the marker's accumulated view weight -- is its
    certainty, which vote_target weighs the across-marker vote by.

    Markers seen from fewer than `min_views` surviving viewpoints are DROPPED rather than
    fused: a pose from one or two views carries the full PnP depth/tilt ambiguity, and
    letting it vote would spend the rig's redundancy on its worst member."""
    power = getattr(plan, 'view_weight_power', 2.0)
    cap = getattr(plan, 'max_camera_distance_m', None)
    out = {}
    for mid, obs in sorted(seen.items()):
        views = list(obs)
        if cap is not None:
            far = [v for v in views if v[1] > cap + 1e-9]
            if far:
                log.warning('  marker %d: %d view%s captured beyond the %.0f mm standoff cap '
                            '-- dropped.', mid, len(far), '' if len(far) == 1 else 's',
                            cap * 1000.0)
                views = [v for v in views if v[1] <= cap + 1e-9]
        if len(views) < plan.min_views:
            log.warning('  marker %d seen from only %d view%s (min_views %d) -- not fused.',
                        mid, len(views), '' if len(views) == 1 else 's', plan.min_views)
            continue
        w = [_view_weight(d, power) for _T, d in views]
        T, lin, ang = average_pose([T for T, _d in views], weights=w)
        level = log.info
        if ((plan.max_view_spread_mm is not None and lin * 1000.0 > plan.max_view_spread_mm)
                or (plan.max_view_spread_deg is not None
                    and np.degrees(ang) > plan.max_view_spread_deg)):
            level = log.warning
        level('  marker %d: fused from %d views (certainty-weighted, mean range %.0f mm), '
              'spread %.2f mm / %.2f deg.', mid, len(views),
              float(np.mean([d for _T, d in views])) * 1000.0, lin * 1000.0, np.degrees(ang))
        out[mid] = (T, lin, ang, len(views), float(sum(w)))
    return out


def marker_consensus(votes, weights, inlier_mm, inlier_deg, min_inliers=2):
    """RANSAC over the per-marker target votes: (inliers, rejected, why).

    EXHAUSTIVE, NOT RANDOM, and that is not a shortcut -- it is the correct algorithm here. RANSAC
    samples a MINIMAL SET, fits a hypothesis and counts agreement; the minimal set for this problem
    is ONE MARKER, because a marker plus its calibrated T_marker_target already determines the full
    6-DOF target pose. With N markers there are therefore exactly N hypotheses, so they can all be
    tried. Enumerating them is deterministic, reproducible run to run, needs no iteration count to
    tune, and is GUARANTEED to find the largest consensus -- none of which a random sampler offers.

    SCORED BY HOW MANY MARKERS AGREE, not by total weight, for the same reason the connector
    estimator scores by distinct views rather than raw detections: one marker that happens to carry
    a heavy view weight must not be able to outvote two that agree with each other. Weight breaks
    ties only, and spread breaks those.

    TWO MARKERS CANNOT ELECT AN OUTLIER. If they disagree, each is a consensus of one and there is
    nothing to say which moved -- so this refuses rather than picking the heavier. That is what
    min_inliers >= 2 means, and why a 2-marker rig gets the old all-or-nothing gate instead.
    """
    ids = sorted(votes)
    if len(ids) < 3:
        return ids, [], 'fewer than 3 markers: no consensus is possible, all kept'

    def agrees(a, b):
        d_lin, d_ang = pose_error(votes[a], votes[b])
        return ((inlier_mm is None or d_lin * 1000.0 <= inlier_mm)
                and (inlier_deg is None or np.degrees(d_ang) <= inlier_deg))

    best = None
    for h in ids:                                   # every marker is a hypothesis in turn
        inl = [m for m in ids if agrees(h, m)]
        if len(inl) < 2:
            continue
        _T, lin, ang = average_pose([votes[m] for m in inl],
                                    weights=[weights[m] for m in inl])
        # more markers > more weight > tighter spread
        key = (len(inl), sum(weights[m] for m in inl), -(lin * 1000.0 + np.degrees(ang)))
        if best is None or key > best[0]:
            best = (key, inl)

    if best is None or len(best[1]) < max(2, int(min_inliers)):
        return [], ids, ('no set of %d or more markers agrees within %s mm / %s deg'
                         % (max(2, int(min_inliers)), inlier_mm, inlier_deg))
    inl = best[1]
    return inl, [m for m in ids if m not in inl], ''


def vote_target(rig, fused, plan):
    """(T_base_target, votes) from every fused marker that the rig knows.

    Each vote is T_base_marker @ T_marker_target -- one marker's opinion of where the fixture
    is. Votes are averaged weighted by each marker's CERTAINTY (its accumulated view weight:
    more views, and closer ones, count for more). A marker's pixel-noise spread still does
    not enter the weighting -- that would favour whichever marker happened to be photographed
    most cleanly -- and once the servo refinement has run, every marker was viewed at the
    same canonical distance, so the weighting reduces to view counting."""
    votes, weights = {}, {}
    for mid, entry in fused.items():
        rig_entry = rig['markers'].get(mid)
        if rig_entry is None:
            log.warning('  marker %d is in view but not in the rig -- ignored.', mid)
            continue
        votes[mid] = entry[0] @ rig_entry['T_marker_target']
        weights[mid] = float(entry[4]) if len(entry) > 4 else 1.0
    if not votes:
        return None, votes
    missing = sorted(set(rig['markers']) - set(votes))
    if missing:
        (log.error if plan.require_all else log.warning)(
            '  rig marker(s) %s did not produce a fused pose.',
            ', '.join(str(m) for m in missing))
        if plan.require_all:
            return None, votes
    if len(votes) < plan.min_markers:
        log.error('  only %d marker%s voted (min_markers %d).',
                  len(votes), '' if len(votes) == 1 else 's', plan.min_markers)
        return None, votes

    # ---- RANSAC: drop markers that do not agree with the consensus ---------------------------
    # WHY THIS EXISTS. Before it, one marker that had been knocked, re-stuck or reprinted at the
    # wrong size failed the WHOLE localization on the disagreement gate below -- the error message
    # even said which failure it was, and then refused to proceed anyway. With three or more
    # markers the rig can outvote the bad one and carry on, which is the entire point of putting
    # several on the fixture.
    rejected = []
    if plan.ransac and len(votes) >= 3:
        inliers, rejected, why = marker_consensus(
            votes, weights, plan.ransac_inlier_mm, plan.ransac_inlier_deg, plan.ransac_min_inliers)
        if not inliers:
            log.error('MARKER RANSAC FAILED: %s. Every marker disagrees with every other, so '
                      'there is no consensus to trust -- this is a rig or calibration problem, '
                      'not noise. Re-run the marker calibration.', why)
            return None, votes
        for mid in rejected:
            d_lin, d_ang = pose_error(votes[inliers[0]], votes[mid])
            log.error('  MARKER %d REJECTED as an outlier: %.2f mm / %.2f deg from the consensus '
                      'of %d marker(s), outside the %s mm / %s deg inlier band. It has most '
                      'likely moved, been re-stuck, or been reprinted at a different size -- '
                      're-run its calibration.', mid, d_lin * 1000.0, np.degrees(d_ang),
                      len(inliers), plan.ransac_inlier_mm, plan.ransac_inlier_deg)
        if rejected and len(inliers) < plan.min_markers:
            log.error('  only %d marker(s) survived RANSAC (min_markers %d).',
                      len(inliers), plan.min_markers)
            return None, votes
        if rejected:
            log.warning('  proceeding on the %d-marker consensus %s; %s excluded.',
                        len(inliers), inliers, rejected)
        order = inliers
    else:
        order = sorted(votes)

    T, lin, ang = average_pose([votes[m] for m in order],
                               weights=[weights[m] for m in order])
    total_w = sum(weights[m] for m in order)      # over the AVERAGED set, not the rejected ones
    for mid in order:
        d_lin, d_ang = pose_error(T, votes[mid])
        log.info('    marker %d votes %+.2f mm / %+.2f deg from the fused target '
                 '(weight %.0f%%).', mid, d_lin * 1000.0, np.degrees(d_ang),
                 100.0 * weights[mid] / total_w)
    if len(order) > 1:
        level = log.info
        bad = ((plan.max_disagreement_mm is not None
                and lin * 1000.0 > plan.max_disagreement_mm)
               or (plan.max_disagreement_deg is not None
                   and np.degrees(ang) > plan.max_disagreement_deg))
        if bad:
            log.error('MARKERS DISAGREE: %d markers spread %.2f mm / %.2f deg about the fused '
                      'target, over the %s mm / %s deg gate. That is not noise to average -- one '
                      'marker has most likely moved, been re-stuck or been reprinted at a '
                      'different size. Re-run the calibration for the outlier above.',
                      len(order), lin * 1000.0, np.degrees(ang),
                      plan.max_disagreement_mm, plan.max_disagreement_deg)
            return None, votes
        level('  %d markers agree to %.2f mm / %.2f deg%s.', len(order), lin * 1000.0,
              np.degrees(ang),
              ' (after rejecting %s)' % rejected if rejected else '')
    else:
        log.warning('  only ONE marker voted -- the fused pose carries no cross-check at all. '
                    'Its view spread (above) measures pixel noise, NOT whether the marker is '
                    'where the calibration said.')
    return T, votes


def _corner_layout(size_m):
    """The four corners in the marker's own frame -- ArucoDetector.object_points' exact
    layout (TL, TR, BR, BL, centred, +Z out of the printed face)."""
    h = float(size_m) / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def rig_object_points(rig):
    """{marker_id: (4, 3) corner coordinates in the TARGET frame} -- the whole rig as ONE
    rigid object, corners carried through inverse(T_marker_target). Pure, so the geometry is
    testable without OpenCV."""
    out = {}
    for mid, entry in rig['markers'].items():
        local = _corner_layout(entry['size_m'])
        T_tm = inverse(entry['T_marker_target'])
        out[int(mid)] = (T_tm[:3, :3] @ local.T).T + T_tm[:3, 3]
    return out


def refine_markers_multiview(fused, corner_views, sizes_m, plan):
    """Per-marker MULTI-VIEW joint solve: every view's four corners of ONE marker, solved
    together for that marker's pose with the camera poses known from FK.

    This is the calibration-side counterpart of the run-time rig PnP. The rig-as-one-object
    solve cannot apply here (the rig geometry IS the unknown being measured), but the same
    corners-solved-jointly principle does: instead of averaging per-view PnP poses in pose
    space with a heuristic distance weight, minimize the REPROJECTION error in pixel space
    across all of a marker's observations at once. Pixel-space minimization weighs closer
    views more automatically (they subtend more pixels) and lets parallax resolve the
    single-view planar tilt ambiguity instead of averaging over it.

    Returns {marker_id: (T_refined, rms_px, n_views)} for markers whose solve CONVERGED AND
    reduced the reprojection error; everything else keeps its fused pose. Needs OpenCV +
    scipy; unavailable = empty dict, callers carry on with the fused poses."""
    try:
        import cv2
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
    except Exception as exc:                       # noqa: BLE001
        log.warning('multi-view refinement unavailable (%s) -- keeping the fused poses.', exc)
        return {}

    out = {}
    for mid, entry in sorted(fused.items()):
        obs = [(np.asarray(v['corners'][mid], dtype=float), v['K'], v['D'], v['T_base_cam'])
               for v in corner_views if mid in v['corners']]
        if len(obs) < 2:
            continue                               # one view is just PnP again -- no gain
        layout = _corner_layout(sizes_m[mid]).astype(np.float32)

        def residuals(T, _obs=obs, _layout=layout):
            errs = []
            for c, K, D, T_bc in _obs:
                T_cm = inverse(T_bc) @ T
                img, _ = cv2.projectPoints(
                    _layout, Rotation.from_matrix(T_cm[:3, :3]).as_rotvec(),
                    T_cm[:3, 3], K, D)
                errs.append((img.reshape(4, 2) - c).ravel())
            return np.concatenate(errs)

        def unpack(x):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
            T[:3, 3] = x[3:]
            return T

        T0 = entry[0]
        x0 = np.concatenate([Rotation.from_matrix(T0[:3, :3]).as_rotvec(), T0[:3, 3]])
        try:
            res = least_squares(lambda x: residuals(unpack(x)), x0, method='lm')
        except Exception as exc:                   # noqa: BLE001 -- a solver failure keeps fused
            log.warning('  marker %d: multi-view solve failed (%s) -- keeping the fused '
                        'pose.', mid, exc)
            continue
        T_ref = unpack(res.x)
        rms0 = float(np.sqrt(np.mean(residuals(T0) ** 2)))
        rms1 = float(np.sqrt(np.mean(residuals(T_ref) ** 2)))
        if not np.isfinite(rms1) or rms1 > rms0 + 1e-9:
            log.warning('  marker %d: multi-view solve did not improve (%.2f -> %.2f px RMS) '
                        '-- keeping the fused pose.', mid, rms0, rms1)
            continue
        d_lin, d_ang = pose_error(T0, T_ref)
        log.info('  marker %d: multi-view joint solve over %d views moved the pose '
                 '%.2f mm / %.2f deg (reprojection %.2f -> %.2f px RMS).',
                 mid, len(obs), d_lin * 1000.0, np.degrees(d_ang), rms0, rms1)
        out[mid] = (T_ref, rms1, len(obs))
    return out


def joint_pnp_views(rig, corner_views, plan):
    """One TARGET-pose estimate per view, from ALL of that view's detected rig corners
    solved as a single PnP -- the rig acting as one large marker. Returns
    [(T_base_target, camera_distance_m), ...], ready for the same certainty-weighted fusion
    single-marker views get. Views whose corners belong to no rig marker are skipped."""
    import cv2
    objs = rig_object_points(rig)
    est = []
    for v in corner_views:
        obj_pts, img_pts, n_markers = [], [], 0
        for mid, c in v['corners'].items():
            o = objs.get(int(mid))
            if o is None:
                continue
            obj_pts.append(o)
            img_pts.append(np.asarray(c, dtype=float).reshape(4, 2))
            n_markers += 1
        if not n_markers:
            continue
        obj = np.concatenate(obj_pts).astype(np.float32)
        img = np.concatenate(img_pts).astype(np.float32)
        # SQPnP: a global solver, exact-ish for any point count and geometry -- the rig's
        # corners are NOT coplanar in general (markers on different faces), so the planar
        # solvers the single-marker path uses do not apply here.
        ok, rvec, tvec = cv2.solvePnP(obj, img, v['K'], v['D'], flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        T_cam_target = np.eye(4)
        T_cam_target[:3, :3], _ = cv2.Rodrigues(rvec)
        T_cam_target[:3, 3] = np.asarray(tvec, dtype=float).flatten()
        est.append((v['T_base_cam'] @ T_cam_target, float(np.linalg.norm(tvec))))
    return est


def locate(robot, camera, detector, rig, plan, wanted=None, on_view=None,
           on_servo_view=None):
    """The whole run-time job: sweep -> (servo-refine) -> per-marker consistency GATE ->
    JOINT-PnP estimate. Returns T_base_target or None.

    The final number comes from solving every view's detected corners of ALL rig markers as
    one rigid object (rig_object_points / joint_pnp_views), fused across views with the
    closeness weighting. The per-marker vote still runs FIRST as the consistency gate -- a
    moved or re-stuck marker refuses the run there, where the joint solve would quietly
    absorb it -- and stands in as the answer when the joint solve is unavailable."""
    log.info('MARKER LOCALIZATION: %s; rig markers %s.', plan.describe(),
             ', '.join(str(m) for m in sorted(rig['markers'])))
    T_overview = robot.camera()
    corner_views = []
    seen = sweep(robot, camera, detector, plan, wanted=wanted or set(rig['markers']),
                 on_view=on_view, corner_log=corner_views)
    if not seen:
        log.error('MARKER LOCALIZATION: no rig marker was detected from any view.')
        return None
    if plan.servo.enabled:
        merge_refined(seen, servo_refine(robot, camera, detector, plan, seen,
                                         T_overview=T_overview, on_view=on_servo_view,
                                         corner_log=corner_views))
    T_vote, _votes = vote_target(rig, fuse_markers(seen, plan), plan)
    if T_vote is None:
        return None                       # the gate refused -- nothing overrides that
    if not getattr(plan, 'joint_pnp', True) or not corner_views:
        return T_vote
    try:
        est = joint_pnp_views(rig, corner_views, plan)
    except Exception as exc:              # noqa: BLE001 -- cv2 absent / solver failure
        log.warning('joint PnP unavailable (%s) -- using the per-marker vote.', exc)
        return T_vote
    if not est:
        log.warning('joint PnP produced no view estimates -- using the per-marker vote.')
        return T_vote
    w = [_view_weight(d, plan.view_weight_power) for _T, d in est]
    T_joint, lin, ang = average_pose([T for T, _d in est], weights=w)
    d_lin, d_ang = pose_error(T_vote, T_joint)
    log.info('  JOINT PnP: %d view%s, spread %.2f mm / %.2f deg; %.2f mm / %.2f deg from '
             'the per-marker vote.', len(est), '' if len(est) == 1 else 's',
             lin * 1000.0, np.degrees(ang), d_lin * 1000.0, np.degrees(d_ang))
    if ((plan.max_disagreement_mm is not None and d_lin * 1000.0 > plan.max_disagreement_mm)
            or (plan.max_disagreement_deg is not None
                and np.degrees(d_ang) > plan.max_disagreement_deg)):
        log.warning('  joint PnP and the per-marker vote disagree past the gate -- both come '
                    'from the same detections, so suspect the rig geometry (a knocked '
                    'marker) or the intrinsics. Using the JOINT estimate.')
    return T_joint


def solve_offsets(fused, T_base_target):
    """{marker_id: T_marker_target} -- the CALIBRATION, the inverse of the run-time job.

    With the fixture at a pose we already know, each fused marker gives the target in its own
    frame directly. Pure, so the arithmetic is testable without a camera."""
    return {mid: inverse(entry[0]) @ T_base_target for mid, entry in fused.items()}


def yaml_block(rig_name, offsets, fused, sizes_m, dictionary=None, stamp=None):
    """The `marker_rigs:` entry for a solved rig, as text to paste into configs/frames.yaml.

    PRINTED, NOT WRITTEN. frames.yaml is hand-maintained and heavily commented; a calibration
    that rewrote it would lose the comments and would also decide, silently, that this run was
    better than whatever is already in the file. The operator pastes."""
    lines = ['marker_rigs:', f'  {rig_name}:']
    if dictionary:
        lines.append(f'    dictionary: {dictionary}')
    lines.append('    markers:')
    for mid in sorted(offsets):
        xyz, rpy = matrix_to_xyzrpy(offsets[mid])
        _T, lin, ang, n = fused[mid][:4]
        lines += [f'      {mid}:',
                  f'        size_mm: {sizes_m[mid] * 1000.0:.2f}',
                  '        xyz_mm:  [%s]' % ', '.join('%+.2f' % (v * 1000.0) for v in xyz),
                  '        rpy_deg: [%s]' % ', '.join('%+.2f' % np.degrees(v) for v in rpy),
                  f'        views: {n}',
                  f'        residual_mm: {lin * 1000.0:.2f}',
                  f'        residual_deg: {np.degrees(ang):.2f}']
        if stamp:
            lines.append(f'        measured: {stamp}')
    return '\n'.join(lines)
