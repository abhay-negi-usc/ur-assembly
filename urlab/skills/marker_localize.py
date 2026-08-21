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
refinement views (the vantage plus an aimed parallax ring) that REPLACE that marker's sweep
views. Centred, close, fixed-distance viewing removes the vantage-dependent part of the PnP
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

WHY MULTIPLE MARKERS. Each marker with a recorded pose is an INDEPENDENT vote on the fixture
-- independent because the errors that dominate (the marker's own printed size, how square it
is glued, its detection geometry) do not correlate between two markers on different faces.
The spread ACROSS markers is therefore a real accuracy estimate.

THE FAILURE THIS IS BUILT AROUND is a marker that is right about everything except which
fixture it is on -- moved, re-stuck, re-printed at another size. It produces a confident pose
that disagrees with its neighbours, so every vote is reported individually and
`max_disagreement_mm` rejects the fused answer rather than averaging a good rig with a stale
marker.
"""

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
        # Certainty weighting: a view at distance d fuses with weight d^-power (0 = unweighted).
        self.view_weight_power = float(b.get('view_weight_power', 2.0))
        # The STANDOFF CAP: the camera never plans a view farther than this from the markers,
        # and a detection captured from beyond it is never fused. null disables.
        cap = _opt_float(b.get('max_camera_distance_mm', 500.0))
        self.max_camera_distance_m = None if cap is None else cap / 1000.0
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


def sweep(robot, camera, detector, plan, wanted=None, on_view=None):
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
        hits = {}
        frame = None
        for _ in range(plan.frames_per_view):
            frame = camera.capture()
            for mid, T in detector.detect_in_base(frame).items():
                if wanted is not None and int(mid) not in wanted:
                    continue
                hits.setdefault(int(mid), []).append(T)
        if on_view is not None and frame is not None:
            on_view(k, frame, detector.detect(frame))
        if not hits:
            log.warning('  view %d/%d: no markers detected.', k + 1, len(plan.offsets))
            continue
        cam_p = _cam_position(frame, robot)
        for mid, mats in hits.items():
            # The repeats at one stop measure sensor noise only, so they collapse to ONE view
            # here -- otherwise frames_per_view would silently weight a view by how still it was.
            T_m = average_pose(mats)[0]
            seen.setdefault(mid, []).append(
                (T_m, float(np.linalg.norm(T_m[:3, 3] - cam_p))))
        aim = np.mean([m[-1][0][:3, 3] for m in seen.values()], axis=0)
        log.info('  view %d/%d: marker%s %s.', k + 1, len(plan.offsets),
                 '' if len(hits) == 1 else 's', ', '.join(str(m) for m in sorted(hits)))
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


def _detect_one(camera, detector, mid, n_frames):
    """(averaged T_base_marker over n frames, last frame); (None, frame) if never detected."""
    mats, frame = [], None
    for _ in range(n_frames):
        frame = camera.capture()
        found = {int(m): T for m, T in detector.detect_in_base(frame).items()}
        if int(mid) in found:
            mats.append(found[int(mid)])
    return (average_pose(mats)[0] if mats else None), frame


def servo_refine(robot, camera, detector, plan, seen, T_overview=None, on_view=None):
    """Per-marker VISUAL SERVOING refinement (ViewPlan.servo). One marker at a time:

        overview -> servo onto the marker's normal at distance_m (re-detect + re-centre
        until two successive detections agree) -> capture the vantage + an aimed parallax
        ring -> next marker

    The OVERVIEW hop between markers matters: servoing to one marker takes the others out of
    frame by design, and returning to the pose the sweep ran from is what brings the whole
    rig back into view before the next marker's servo starts.

    Returns {marker_id: [(T_base_marker, distance_m), ...]} -- REPLACEMENT views for every
    marker that refined. A marker that will not detect from its vantage, or that keeps fewer
    than min_views refinement views, is left out and its sweep views stand.
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
            T_obs, _frame = _detect_one(camera, detector, mid, plan.frames_per_view)
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
            T_obs, frame = _detect_one(camera, detector, mid, plan.frames_per_view)
            if on_view is not None and frame is not None:
                on_view(mid, j, frame, detector.detect(frame))
            if T_obs is None:
                continue
            d = float(np.linalg.norm(T_obs[:3, 3] - _cam_position(frame, robot)))
            views.append((T_obs, d))
        if len(views) >= plan.min_views:
            refined[mid] = views
            log.info('  marker %d: REFINED -- %d servoed view%s at ~%.0f mm replace its %d '
                     'sweep view%s.', mid, len(views), '' if len(views) == 1 else 's',
                     sv.distance_m * 1000.0, len(obs), '' if len(obs) == 1 else 's')
        else:
            log.warning('  marker %d: only %d servoed view%s (min_views %d) -- keeping the '
                        'sweep views.', mid, len(views), '' if len(views) == 1 else 's',
                        plan.min_views)
    # Leave the arm at the overview, ready for whatever comes next.
    robot.arm.move_frame_to(T_overview, robot.T_tool0_cam, 'overview (refinement done)')
    return refined


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

    order = sorted(votes)
    T, lin, ang = average_pose([votes[m] for m in order],
                               weights=[weights[m] for m in order])
    total_w = sum(weights.values())
    for mid in order:
        d_lin, d_ang = pose_error(T, votes[mid])
        log.info('    marker %d votes %+.2f mm / %+.2f deg from the fused target '
                 '(weight %.0f%%).', mid, d_lin * 1000.0, np.degrees(d_ang),
                 100.0 * weights[mid] / total_w)
    if len(votes) > 1:
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
                      len(votes), lin * 1000.0, np.degrees(ang),
                      plan.max_disagreement_mm, plan.max_disagreement_deg)
            return None, votes
        level('  %d markers agree to %.2f mm / %.2f deg.', len(votes), lin * 1000.0,
              np.degrees(ang))
    else:
        log.warning('  only ONE marker voted -- the fused pose carries no cross-check at all. '
                    'Its view spread (above) measures pixel noise, NOT whether the marker is '
                    'where the calibration said.')
    return T, votes


def locate(robot, camera, detector, rig, plan, wanted=None):
    """The whole run-time job: sweep -> (servo-refine) -> fuse per marker -> vote.
    Returns T_base_target or None."""
    log.info('MARKER LOCALIZATION: %s; rig markers %s.', plan.describe(),
             ', '.join(str(m) for m in sorted(rig['markers'])))
    T_overview = robot.camera()
    seen = sweep(robot, camera, detector, plan, wanted=wanted or set(rig['markers']))
    if not seen:
        log.error('MARKER LOCALIZATION: no rig marker was detected from any view.')
        return None
    if plan.servo.enabled:
        seen.update(servo_refine(robot, camera, detector, plan, seen,
                                 T_overview=T_overview))
    T, _votes = vote_target(rig, fuse_markers(seen, plan), plan)
    return T


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
