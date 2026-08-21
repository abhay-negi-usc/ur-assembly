"""Locate a fixture from the fiducials bolted around it -- multi-view capture, fusion, vote.

TWO APPS, ONE FILE. `apps/marker_calibration` runs the sweep with the fixture at a pose it
already knows (the recorded `targets:` mate) and SOLVES for where the target sits in each
marker's frame; `apps/bnc_assembly --target_source visual` runs the same sweep, reads those poses
back and solves for the fixture. Same capture, same fusion, opposite unknown -- so they share the
code and a calibration can never be fused differently from the way it is used.

WHY MULTIPLE VIEWS. A single ArUco pose is a four-point PnP solve on a small planar square, and
the ill-conditioned direction is exactly the one that matters here: rotation about an axis IN the
marker plane, which trades against depth. Two views from a few centimetres apart see that
trade-off differently, so translating the camera between captures is what actually adds
information. Capturing ten frames without moving averages the sensor noise and leaves the
geometric bias untouched.

WHY MULTIPLE MARKERS. Each marker with a recorded pose is an INDEPENDENT vote on the fixture --
independent because the errors that dominate (the marker's own printed size, how square it is
glued, its detection geometry) do not correlate between two markers on different faces. The
spread ACROSS markers is therefore a real accuracy estimate, and it is the number to trust over
the spread across views of one marker, which mostly measures pixel noise.

THE FAILURE THIS IS BUILT AROUND is a marker that is right about everything except which fixture
it is on -- moved, re-stuck, re-printed at another size. It produces a confident pose that
disagrees with its neighbours, so every vote is reported individually and `max_disagreement_mm`
rejects the fused answer rather than averaging a good rig with a stale marker.
"""

import time

import numpy as np

from .. import log as urlog
from ..transforms import (average_pose, inverse, look_at, matrix_to_xyzrpy, pose_error,
                          xyzrpy_to_matrix)

log = urlog.get('marker-loc')


class ViewPlan:
    """The `marker_views:` config block: where to look from, and how long to sit still.

    Offsets are RELATIVE to the camera pose the sweep starts at -- the markers are assumed to be
    in view already, which is what lets this be a small local sweep instead of a search."""

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
        return ('%d view%s, %d frame%s each, %s'
                % (len(self.offsets), '' if len(self.offsets) == 1 else 's',
                   self.frames_per_view, '' if self.frames_per_view == 1 else 's',
                   'aimed at the markers' if self.aim_at_markers else 'orientation held'))


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


def sweep(robot, camera, detector, plan, wanted=None, on_view=None):
    """Drive the configured views and detect at each. Returns {marker_id: [T_base_marker, ...]}.

    `on_view(index, frame, poses_in_camera)` is called once per view that captured anything -- the
    calibration app uses it to save an annotated image, which is the only artefact that shows WHY
    a marker was missed rather than that it was.

    The camera pose is sampled AT CAPTURE (see perception/camera.Frame), so a detection can never
    be credited to the wrong viewpoint -- which is what makes fusing across a moving camera safe
    at all. A view that reaches nothing is SKIPPED, not fatal: an offset that puts a marker out of
    frame is a tuning problem, and the quota below is what decides whether enough got through."""
    T_start = robot.camera()
    aim = None
    seen = {}
    for k, off in enumerate(plan.offsets):
        T_view = T_start @ off
        if plan.aim_at_markers and aim is not None:
            # Re-point at the markers found so far. look_at only fixes the view ray; the roll
            # comes from the reference, and passing the nominal view keeps it near the start.
            T_view = look_at(T_view[:3, 3], aim, T_view)
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
        for mid, mats in hits.items():
            # The repeats at one stop measure sensor noise only, so they collapse to ONE view
            # here -- otherwise frames_per_view would silently weight a view by how still it was.
            seen.setdefault(mid, []).append(average_pose(mats)[0])
        aim = np.mean([m[-1][:3, 3] for m in seen.values()], axis=0)
        log.info('  view %d/%d: marker%s %s.', k + 1, len(plan.offsets),
                 '' if len(hits) == 1 else 's', ', '.join(str(m) for m in sorted(hits)))
    return seen


def _at(T_a, T_b, tol_m=1e-3, tol_rad=np.radians(0.2)):
    lin, ang = pose_error(T_a, T_b)
    return lin <= tol_m and ang <= tol_rad


def fuse_markers(seen, plan):
    """{marker_id: (T_base_marker, lin_rms_m, ang_rms_rad, n_views)} for markers with enough views.

    Markers seen from fewer than `min_views` viewpoints are DROPPED rather than fused: a pose from
    one or two views carries the full PnP depth/tilt ambiguity, and letting it vote would spend
    the rig's redundancy on its worst member."""
    out = {}
    for mid, mats in sorted(seen.items()):
        if len(mats) < plan.min_views:
            log.warning('  marker %d seen from only %d view%s (min_views %d) -- not fused.',
                        mid, len(mats), '' if len(mats) == 1 else 's', plan.min_views)
            continue
        T, lin, ang = average_pose(mats)
        level = log.info
        if ((plan.max_view_spread_mm is not None and lin * 1000.0 > plan.max_view_spread_mm)
                or (plan.max_view_spread_deg is not None
                    and np.degrees(ang) > plan.max_view_spread_deg)):
            level = log.warning
        level('  marker %d: fused from %d views, spread %.2f mm / %.2f deg.',
              mid, len(mats), lin * 1000.0, np.degrees(ang))
        out[mid] = (T, lin, ang, len(mats))
    return out


def vote_target(rig, fused, plan):
    """(T_base_target, votes) from every fused marker that the rig knows.

    Each vote is T_base_marker @ T_marker_target -- one marker's opinion of where the fixture is.
    They are averaged unweighted: a marker's view spread measures its pixel noise, not how well it
    is glued or how accurately its offset was calibrated, so weighting by it would confidently
    favour whichever marker happened to be photographed most cleanly."""
    votes = {}
    for mid, (T_marker, lin, ang, n) in fused.items():
        entry = rig['markers'].get(mid)
        if entry is None:
            log.warning('  marker %d is in view but not in the rig -- ignored.', mid)
            continue
        votes[mid] = T_marker @ entry['T_marker_target']
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

    T, lin, ang = average_pose(list(votes.values()))
    for mid, T_vote in sorted(votes.items()):
        d_lin, d_ang = pose_error(T, T_vote)
        log.info('    marker %d votes %+.2f mm / %+.2f deg from the fused target.',
                 mid, d_lin * 1000.0, np.degrees(d_ang))
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
    """The whole run-time job: sweep -> fuse per marker -> vote. Returns T_base_target or None."""
    log.info('MARKER LOCALIZATION: %s; rig markers %s.', plan.describe(),
             ', '.join(str(m) for m in sorted(rig['markers'])))
    seen = sweep(robot, camera, detector, plan, wanted=wanted or set(rig['markers']))
    if not seen:
        log.error('MARKER LOCALIZATION: no rig marker was detected from any view.')
        return None
    T, _votes = vote_target(rig, fuse_markers(seen, plan), plan)
    return T


def solve_offsets(fused, T_base_target):
    """{marker_id: T_marker_target} -- the CALIBRATION, the inverse of the run-time job.

    With the fixture at a pose we already know, each fused marker gives the target in its own
    frame directly. Pure, so the arithmetic is testable without a camera."""
    return {mid: inverse(T_marker) @ T_base_target for mid, (T_marker, _l, _a, _n) in fused.items()}


def yaml_block(rig_name, offsets, fused, sizes_m, dictionary=None, stamp=None):
    """The `marker_rigs:` entry for a solved rig, as text to paste into configs/frames.yaml.

    PRINTED, NOT WRITTEN. frames.yaml is hand-maintained and heavily commented; a calibration that
    rewrote it would lose the comments and would also decide, silently, that this run was better
    than whatever is already in the file. The operator pastes."""
    lines = ['marker_rigs:', f'  {rig_name}:']
    if dictionary:
        lines.append(f'    dictionary: {dictionary}')
    lines.append('    markers:')
    for mid in sorted(offsets):
        xyz, rpy = matrix_to_xyzrpy(offsets[mid])
        _T, lin, ang, n = fused[mid]
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
