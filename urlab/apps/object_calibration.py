"""OBJECT CALIBRATION -- where an object's mating feature sits in its marker's frame.

Produces one entry per object in configs/objects.yaml: one or more markers, each with its
printed size and the object's MATING FEATURE expressed in THAT MARKER's own frame
(marker <- grasp).  With the entry in place a pick is one multiply per marker -- the camera
measures T_base_marker, and

    T_base_grasp = T_base_marker @ T_marker_grasp

is where to drive the coupler.  Same direction, and the same reason, as frames.yaml's
marker_rigs: the object is the unknown and the marker is what the camera measures, so storing
marker <- grasp costs one multiply per pick where grasp <- marker would cost an inverse per pick
and would read as if the marker were being located, which is backwards.

WHAT IT MEASURES.  The arm is hand-guided into the object's mating feature and the coupler is
locked.  At that instant the object's mating frame and the tool's `coupler_mate` frame ARE THE
SAME FRAME, so forward kinematics gives the mate directly and nothing has to be estimated:

    T_base_grasp = T_base_tool0 @ T_tool0_coupler_mate

SEVERAL MARKERS, ONE MEASURED POSE.  Every marker on an object stores the SAME mate, written in
its own coordinates -- inverse(marker) @ mate encodes the same physical pose whichever marker it
is expressed from.  That is what lets them vote at run time and be averaged, what keeps a pick
alive when one is occluded, and what lets a marker that has been knocked be outvoted instead of
quietly dragging the answer.  No marker is special and none has to be visible on every mate: a
mate is usable as long as any declared marker saw it.

The markers are fused from a multi-view sweep taken BEFORE the mate, with the object still on the
bench.  That ordering is not incidental: it is the only part of the cycle where moving the camera
adds information, because once the object is on the coupler it travels with the camera and every
further view is geometrically the same view.

THE SWEEP IS THEN REFINED MARKER BY MARKER.  With `marker_views.servo.enabled` the camera visits
each marker in turn and servos onto its face at `servo.distance_mm` (100 mm by default), and
those close views pool with the sweep's.  It matters because solvePnP's angular error falls with
the marker's apparent size, and the sweep has to be flown far enough back to keep EVERY marker in
one frame -- which on a spread-out object is a long way.  Servoing puts the camera the same short
distance square onto each marker, so every one is measured under the geometry it is best
estimated from rather than wherever in the frame it happened to land.

THE STORED POSE IS THE MATE, UNCORRECTED.  What the coupler was actually seated in is what is
written down -- no axis is substituted, no angle is reconstructed.  Commanding it back at a pick
therefore reproduces the seating that was proven to work, which is what makes the in-hand pose
known afterwards: once mated, the object's mating frame is `coupler_mate`, exactly as it was when
the number was taken.

THE CONSEQUENCE IS ABOUT YAW.  The coupler does not constrain rotation about its own +z, so
whatever wrist angle the operator happened to seat each mate at is part of that mate.  With one
mate that is simply the pose you chose.  With several taken at DIFFERENT yaws, the mean is an
orientation none of them had -- a good mating point and a good mating axis with a fictional yaw
between them.  So seat repeat mates at a consistent wrist angle; `residual_deg` against
`residual_axis_deg` is what shows whether that happened.

Output: data/experiments/object_calibration_<stamp>/ -- the yaml entry, per-mate CSV, and
marker_images/.  The catalogue itself is MERGED (one object replaced, the rest left alone) and
the previous version is backed up into the run directory first.

Run:  python -m urlab.apps.object_calibration --set object_name=banana_jig
"""

import csv as _csv
import os
import shutil
import time
from datetime import datetime

import numpy as np

from .. import behaviors as bt
from .. import log as urlog
from .. import tool_frames
from ..apps._common import ask, experiment_dir, prompts_off
from ..robot.coupler import Coupler
from ..skills import marker_localize as mloc
from .marker_calibration import parse_markers
from ..transforms import (average_pose, inverse, matrix_to_xyzrpy, pose_error,
                          translation_matrix)
from ._runner import run_app

log = urlog.get('object-calib')

_CSV_HEADER = ['mate', 'marker_id', 'marker_views', 'marker_spread_mm',
               'marker_spread_deg',
               'grasp_in_marker_x_mm', 'grasp_in_marker_y_mm', 'grasp_in_marker_z_mm',
               'grasp_in_marker_roll_deg', 'grasp_in_marker_pitch_deg',
               'grasp_in_marker_yaw_deg', 'axis_dev_deg', 'rot_dev_deg',
               'point_dev_mm']

# ---------------------------------------------------------------------------- pure geometry
def fuse_mates(offsets):
    """(T_marker_grasp, residual_mm, residual_deg, residual_axis_deg) over the mates a marker saw.

    A PLAIN RIGID-POSE MEAN, all six degrees of freedom. The stored pose IS the mate that was
    measured -- nothing about it is reconstructed -- so every axis of the spread is evidence and
    there is nothing to leave out of the average.

    THREE RESIDUALS, because they fail differently:

        residual_mm        the mating POINT moving between mates.
        residual_axis_deg  the mating AXIS tipping. This is the one that decides whether the
                           coupler can enter at all.
        residual_deg       the FULL rotational spread. It includes the yaw about the mating
                           axis, which the coupler does not constrain -- so if this is much
                           larger than residual_axis_deg, the mates were made at different
                           wrist yaws and the mean is an average of orientations the operator
                           never repeated. That is a fact about how the mates were taken, not
                           about the object, and it is the reason to report the two apart.

    Pure, so the arithmetic is testable without a robot."""
    if not offsets:
        raise ValueError('fuse_mates needs at least one mate')
    rows = list(offsets)
    T, lin, ang = average_pose(rows)
    dev_axis = [_angle_between_deg(o[:3, 2], T[:3, 2]) for o in rows]
    return (T, lin * 1000.0, float(np.degrees(ang)),
            float(np.sqrt(np.mean(np.asarray(dev_axis) ** 2))))


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / (float(np.linalg.norm(v)) + 1e-12)


def _angle_between_deg(a, b):
    """Angle between two directions, via atan2 rather than acos(dot).

    acos LOSES PRECISION EXACTLY WHERE THIS LIVES: near zero the derivative of acos is
    unbounded, so a dot product one part in 1e12 short of 1 -- which is all it takes, since
    these vectors are normalised in floating point -- reads as a tenth of a millidegree of
    error that is not there. A good calibration spends its whole life in that regime, so the
    residual it reports would be numerical noise instead of a measurement. atan2 of the cross
    against the dot is conditioned the same at every angle."""
    a, b = _unit(a), _unit(b)
    return float(np.degrees(np.arctan2(float(np.linalg.norm(np.cross(a, b))),
                                       float(np.dot(a, b)))))


# ---------------------------------------------------------------------------- the catalogue
_HEADER = """\
# ============================================================================
# OBJECTS -- what the toolchanger coupler can pick, and how to find each one by sight.
#
# GENERATED by urlab.apps.object_calibration. It rewrites this whole file on every run, merging
# in the one object it just calibrated and leaving the others as they are -- so COMMENTS ADDED BY
# HAND HERE DO NOT SURVIVE. Re-run the calibration rather than editing a pose in place; the
# provenance fields below only mean anything if the numbers came from a measurement.
#
# Each pose is the object's MATING FEATURE expressed in its marker's own frame (marker <- grasp).
# At run time the camera measures T_base_marker and
#
#     T_base_grasp = T_base_marker @ T_marker_grasp
#
# is where to drive configs/frames.yaml's `coupler_mate`. One multiply, no inverse -- the same
# direction, and the same reason, as that file's marker_rigs.
#
# THE POSE IS THE MEASURED MATE, UNCORRECTED. What the coupler was actually seated in is what is
# written here -- no axis substituted, no angle reconstructed. Commanding it back reproduces the
# seating that was proven to work, which is what makes the in-hand pose known: once mated, the
# object's mating frame IS `coupler_mate`.
#
# THE COUPLER DOES NOT CONSTRAIN YAW about its own +z, so the wrist angle each mate was seated at
# is part of that mate. `residual_deg` much larger than `residual_axis_deg` means the mates were
# taken at different yaws and the stored yaw is their average rather than any one of them -- a
# good point and axis with a fiction between them. Re-run at a consistent wrist angle.
#
# RESIDUALS. `residual_mm` is the mating POINT spread across mates, `residual_axis_deg` the
# mating AXIS spread -- the one that decides whether the coupler can enter -- and `residual_deg`
# the full rotational spread including that unconstrained yaw.
#
# SIZES ARE LOAD-BEARING. solvePnP scales a marker's distance LINEARLY with the side length it is
# told, so a marker solved at the wrong size lands at the wrong depth with a perfect reprojection
# behind it. Measure the printed BLACK SQUARE edge to edge; the white quiet zone is not part of
# the marker.
# ============================================================================
"""


def yaml_document(objects, stamp=None):
    """The whole catalogue as text: the header above plus one block per object."""
    lines = [_HEADER.rstrip('\n'), '']
    if stamp:
        lines += [f'# Last written {stamp}.', '']
    if not objects:
        lines += ['objects: {}']
        return '\n'.join(lines) + '\n'
    lines.append('objects:')
    for name in sorted(objects):
        lines.append(_object_block(name, objects[name]))
    return '\n'.join(lines) + '\n'


def _fmt_meta(indent, meta, keys):
    out = []
    for key in keys:
        if key in meta:
            value = meta[key]
            out.append('%s%s: %s' % (indent, key,
                                     f'{value:.2f}' if isinstance(value, float) else value))
    return out


def _object_block(name, entry):
    out = [f'  {name}:', '    markers:']
    for mid in sorted(entry['markers']):
        m = entry['markers'][mid]
        xyz, rpy = matrix_to_xyzrpy(m['T_marker_grasp'])
        out += [f'      {mid}:',
                f"        size_mm: {m['size_m'] * 1000.0:.2f}",
                '        xyz_mm:  [%s]' % ', '.join('%+.2f' % (v * 1000.0) for v in xyz),
                '        rpy_deg: [%s]' % ', '.join('%+.2f' % np.degrees(v) for v in rpy)]
        out += _fmt_meta('        ', dict(m.get('meta') or {}),
                         ('mates', 'residual_mm', 'residual_deg', 'residual_axis_deg', 'note'))
    if entry.get('held_mass_kg') is not None:
        out.append(f"    held_mass_kg: {float(entry['held_mass_kg']):.3f}")
    # ASSEMBLIES ARE CARRIED THROUGH, not owned by this app. They are taught by
    # urlab.apps.coupler_assembly_calibration, and this writer rewrites the whole catalogue --
    # so anything it does not know how to emit would be silently deleted the next time an
    # object's markers were recalibrated.
    if entry.get('assemblies'):
        out.append('    assemblies:')
        for a_name in sorted(entry['assemblies']):
            a = entry['assemblies'][a_name]
            xyz, rpy = matrix_to_xyzrpy(a['T_base_assembly'])
            out += [f'      {a_name}:',
                    '        xyz_mm:  [%s]' % ', '.join('%+.2f' % (v * 1000.0) for v in xyz),
                    '        rpy_deg: [%s]' % ', '.join('%+.2f' % np.degrees(v) for v in rpy)]
            out += _fmt_meta('        ', dict(a.get('meta') or {}),
                             ('approaches', 'residual_mm', 'residual_deg', 'measured', 'note'))
    out += _fmt_meta('    ', dict(entry.get('meta') or {}),
                     ('mates', 'measured', 'note'))
    return '\n'.join(out)


def merge_catalogue(path, name, entry):
    """The catalogue with `name` replaced by `entry`, everything else carried through.

    Read back through load_objects so a hand-broken neighbour is caught HERE, before the file is
    rewritten, rather than at the next pick."""
    objects = tool_frames.load_objects(path=path) if os.path.isfile(path) else {}
    objects[name] = entry
    return objects


# ---------------------------------------------------------------------------- the run
class _ObjectCalibration:
    """One object: scan its marker, hand-mate the coupler, repeat, fuse, write the catalogue."""

    def __init__(self, cfg, robot, camera, detector, plan, name, sizes,
                 T_tool0_coupler, coupler, out_dir):
        self.cfg = cfg
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.plan = plan
        self.name = name
        self.sizes = dict(sizes)              # {marker id: printed side length, m}
        # NO REFERENCE MARKER. The stored pose is the MEASURED MATE, written into each marker's
        # own coordinates -- inverse(M_i) @ T_base_mate encodes the same physical pose whichever
        # marker M_i it is expressed from. So every marker is independent, no one of them has to
        # be visible on every mate, and a mate is usable as long as ANY declared marker saw it.
        self.T_tool0_coupler = T_tool0_coupler
        self.coupler = coupler
        self.out_dir = out_dir
        self.images = mloc.MarkerImageWriter(out_dir, detector,
                                             enabled=getattr(plan, 'save_images', True))
        self.repeats = max(1, int(cfg.get('mates', 3)))
        self.retreat_m = float(cfg.get('retreat_mm', 60.0)) / 1000.0
        self.settle_s = float(cfg.get('mate_settle_s', 0.5))

        self.q_view = None
        # The pose the sweep ran from, and the pose the servo refinement returns to between
        # markers -- servoing onto one marker takes the others out of frame by design.
        self.T_overview = None
        self.scans = []            # per mate: {marker id: the fused (T, lin, ang, n, w)}
        self.offsets = {}          # {marker id: [T_marker_grasp per mate it was seen in]}
        self.results = {}          # {marker id: (T, res_mm, res_deg, res_axis_deg)}

    # ---- collect -----------------------------------------------------------------------------
    def collect(self):
        for k in range(self.repeats):
            log.info('---- MATE %d of %d ----', k + 1, self.repeats)
            if not self._to_view(k):
                return False
            self._remember_overview()
            if not self._scan(k):
                return False
            if not self._mate(k):
                return False
        return bool(self.offsets)

    def _to_view(self, k):
        """The viewing pose: driven to `view_joints_deg` when configured, otherwise hand-guided
        once and then RETURNED TO for every later mate, so each repeat sees the marker from the
        same place and the spread reports the object rather than the operator."""
        if self.q_view is not None:
            return self.robot.move_joints(self.q_view, label='object view pose')
        q_cfg = self.cfg.get('view_joints_deg')
        if q_cfg is not None:
            self.q_view = np.radians(np.asarray(q_cfg, dtype=float))
            return self.robot.move_joints(self.q_view, label='object view pose')
        if not self._hand_guide('Push the arm until the MARKER IS IN VIEW, then press Enter '
                                '(q to abort): '):
            return False
        self.q_view = self.robot.arm.q()
        return True

    def _remember_overview(self):
        """The camera pose the sweep is about to run from.

        Captured BEFORE the sweep, not after: the servo refinement returns here between markers,
        and by the end of a sweep the camera is wherever the last offset left it -- which may be
        a pose that sees only one marker."""
        self.T_overview = self.robot.camera()
        return True

    def _hand_guide(self, prompt):
        """Software freedrive for the length of ONE operator prompt, guaranteed off again after.

        BOTH HALVES MATTER. A prompt that asks the operator to push the arm without turning
        teachMode on asks for something they cannot do -- the arm is stiff and the instruction
        is a lie. And the try/finally is not decoration: teachMode leaves the arm compliant, so
        an abort or a Ctrl-C at the prompt would otherwise return with it still on, and the next
        thing this app does is command a move into an arm that is not holding position."""
        if self.robot.arm.dry_run or prompts_off(self.cfg):
            return True
        self._freedrive(True)
        try:
            return ask(prompt)
        finally:
            self._freedrive(False)

    def _scan(self, k):
        """Fuse EVERY declared marker that the sweep sees. The object is STILL ON THE BENCH here
        -- this is the only part of the cycle where moving the camera adds geometry.

        A marker that is not seen is not fatal: occlusion is the normal reason an object carries
        several. The exception is the YAW MARKER, which defines the convention the whole mate is
        expressed in, so a mate without it cannot be used at all."""
        seen = mloc.sweep(self.robot, self.camera, self.detector, self.plan,
                          wanted=set(self.sizes),
                          on_view=lambda i, f, p: self.images.sweep_view(k * 100 + i, f, p),
                          corner_log=None)
        seen = self._servo_refine(seen, k)
        fused = mloc.fuse_markers(seen, self.plan) if seen else {}
        missing = sorted(set(self.sizes) - set(fused))
        if missing:
            log.warning('  marker(s) %s not fused on mate %d (never seen, or fewer than '
                        'min_views %d usable views). The others carry on.',
                        ', '.join(str(m) for m in missing), k + 1, self.plan.min_views)
        if not fused:
            log.error('No declared marker was fused on mate %d, so there is nothing to express '
                      'the mate in. Re-aim so at least one is clearly in view.', k + 1)
            return False
        for mid in sorted(fused):
            _T, lin, ang, n, _w = fused[mid]
            log.info('  marker %d fused from %d view%s (spread %.2f mm / %.2f deg).',
                     mid, n, '' if n == 1 else 's', lin * 1000.0, np.degrees(ang))
        self.scans.append(fused)
        return True

    def _servo_refine(self, seen, k):
        """Visit each marker in turn and look at it CLOSE UP, then pool those views with the
        sweep's.

        WHY IT IS WORTH THE TIME HERE. solvePnP's angular error falls roughly with the marker's
        apparent size, and the sweep is flown at whatever standoff keeps every marker in one
        frame -- which on an object with markers spread across it is a long way back. Servoing to
        `marker_views.servo.distance_mm` puts the camera the same short distance from EVERY
        marker, square onto its face, so each one is measured under the geometry it is best
        estimated from instead of whichever corner of the frame it happened to land in.

        THE CLOSE VIEWS POOL WITH THE SWEEP'S, they do not replace them. The closeness weighting
        (view_weight_power) makes them dominate the fusion anyway, and replacing outright would
        let a marker whose servo produced few views fall under min_views and be dropped -- worse
        than a slightly wider average.

        A marker that will not detect from its vantage keeps its sweep views and the run carries
        on; the refinement is an improvement, not a precondition."""
        if not self.plan.servo.enabled or not seen:
            return seen
        log.info('  servoing to each of %d marker%s at %.0f mm for a close look.',
                 len(seen), '' if len(seen) == 1 else 's',
                 self.plan.servo.distance_m * 1000.0)
        refined = mloc.servo_refine(
            self.robot, self.camera, self.detector, self.plan, seen,
            T_overview=self.T_overview,
            on_view=lambda mid, j, f, p: self.images.servo_view(mid, k * 100 + j, f, p),
            corner_log=None)
        mloc.merge_refined(seen, refined)
        if not refined:
            log.warning('  no marker refined -- every one kept its sweep views alone.')
        return seen

    def _mate(self, k):
        """Hand-guide the coupler into the feature, lock, and read the mate off forward
        kinematics. Locking happens with the arm RIGID and the pose is read AFTER it, because the
        locked state is the mate -- the mechanism pulls the object onto the coupler, and reading
        before that would record where the object was being offered rather than where it seats."""
        if not self._hand_guide('Seat the COUPLER in the object\'s mating feature -- keep the '
                                'object supported, it will be released again. Press Enter when '
                                'seated (q to abort): '):
            return False
        if not self.coupler.hold():
            return False
        if self.settle_s > 0:
            time.sleep(self.settle_s)

        T_base_mate = self.robot.arm.tcp_pose() @ self.T_tool0_coupler
        fused = self.scans[-1]

        # THE MEASURED MATE, STORED AS IT WAS MEASURED. No axis is substituted and no yaw is
        # reconstructed: this pose is the one the coupler was actually seated in, and writing it
        # into each marker's coordinates is all that happens to it. Commanding it back at a pick
        # therefore reproduces the seating that was proven to work -- yaw included -- which is
        # what makes the in-hand pose known afterwards.
        for mid, entry in fused.items():
            self.offsets.setdefault(mid, []).append(inverse(entry[0]) @ T_base_mate)

        first = sorted(fused)[0]
        xyz, rpy = matrix_to_xyzrpy(self.offsets[first][-1])
        log.info('  mate %d: coupler in marker %d  xyz %+8.2f %+8.2f %+8.2f mm  rpy %+7.2f '
                 '%+7.2f %+7.2f deg  (%d marker%s recorded)', k + 1, first,
                 xyz[0] * 1000.0, xyz[1] * 1000.0, xyz[2] * 1000.0, *np.degrees(rpy),
                 len(fused), '' if len(fused) == 1 else 's')

        if not self.coupler.release():
            return False
        return self._retreat(T_base_mate)

    def _retreat(self, T_base_mate):
        """Back the coupler straight out along its own -z. Any other direction levers the coupler
        against the feature it is still inside."""
        if self.retreat_m <= 0:
            return True
        T_out = T_base_mate @ translation_matrix([0.0, 0.0, -self.retreat_m])
        return self.robot.arm.move_frame_to(T_out, self.T_tool0_coupler, 'retreat off the object')

    def _freedrive(self, on):
        """teachMode on/off. Says so out loud when it goes ON, because a compliant arm is a
        different machine from the one the operator was just watching move under program
        control, and because a prompt is the wrong place to discover that by pushing."""
        rtde_c = getattr(self.robot.arm, 'rtde_c', None)
        if rtde_c is None:                        # dry run, or a read-only connection
            return True
        try:
            rtde_c.teachMode() if on else rtde_c.endTeachMode()
        except Exception as exc:                  # noqa: BLE001 -- freedrive is a convenience
            log.warning('Could not %s software freedrive (%s) -- drive the arm from the pendant '
                        'instead, then answer the prompt.', 'enter' if on else 'leave', exc)
            return False
        if on:
            log.info('SOFTWARE FREEDRIVE ON -- the arm is compliant; push it by hand.')
        return True

    # ---- fuse + check ------------------------------------------------------------------------
    def fuse(self):
        """One fused pose per marker, by the same rule for all of them.

        There is no reference marker any more: each one stores the same measured mate in its own
        coordinates, so each fuses independently and a marker that missed a mate simply has one
        fewer to average."""
        for mid in sorted(self.offsets):
            rows = self.offsets[mid]
            self.results[mid] = fuse_mates(rows)
            _T, res_mm, res_deg, res_axis = self.results[mid]
            log.info('FUSED marker %d over %d mate%s: %.2f mm, %.2f deg overall, %.2f deg on '
                     'the mating axis.', mid, len(rows), '' if len(rows) == 1 else 's',
                     res_mm, res_deg, res_axis)
        return bool(self.results)

    def cross_check(self):
        """What the numbers are allowed to be, and what they cannot tell you.

        The gates are WARNINGS by default: the honest threshold is the coupler's capture range,
        which is a property of the mechanism and not of this run, so a limit invented here would
        be a guess wearing the authority of a check."""
        cap_mm = self.cfg.get('max_residual_mm')
        cap_deg = self.cfg.get('max_residual_axis_deg')
        failed = False
        for mid in sorted(self.results):
            _T, res_mm, _res_deg, res_axis = self.results[mid]
            for value, cap, unit, what in ((res_mm, cap_mm, 'mm', 'mating point'),
                                           (res_axis, cap_deg, 'deg', 'mating axis')):
                if cap is not None and value > float(cap):
                    log.error('Marker %d: the %s spread across mates is %.2f %s, over the '
                              'configured %.2f %s.', mid, what, value, unit, float(cap), unit)
                    failed = True

        # THE YAW IS NOW EVIDENCE, NOT A CONSTRUCTION, and that changes what a big rotational
        # residual means. The coupler does not constrain yaw about its mating axis, so whatever
        # wrist angle the operator happened to use is baked into each mate. Mates taken at
        # different yaws average to an orientation none of them had -- the mean is still a
        # perfectly good POINT and AXIS, but its yaw is a fiction. The two residuals apart are
        # what make that visible.
        for mid in sorted(self.results):
            _T, _res_mm, res_deg, res_axis = self.results[mid]
            if len(self.offsets[mid]) > 1 and res_deg > 2.0 * max(res_axis, 0.05):
                log.warning('Marker %d: %.2f deg of overall rotational spread against only '
                            '%.2f deg on the mating axis -- so most of it is YAW about that '
                            'axis. The mates were seated at different wrist angles, and the '
                            'stored yaw is their average rather than any one of them. Mate at a '
                            'consistent yaw, or run mates: 1 and keep the one you meant.',
                            mid, res_deg, res_axis)

        # WHAT THE MARKERS SAY ABOUT EACH OTHER. Each one's pose is exact by construction against
        # the mate it was solved from, so the information is in whether they AGREE: every marker
        # should place the coupler in the same spot, and one that does not has been knocked,
        # re-stuck, or printed at a size other than the one declared. This is the same
        # disagreement the run-time vote gates on, measured here where it can be fixed.
        if len(self.results) > 1:
            ids = sorted(self.results)
            ref_id = ids[0]
            log.info('Marker agreement (each marker\'s coupler pose vs marker %d\'s):', ref_id)
            for mid in ids[1:]:
                shared = [k for k in range(len(self.scans))
                          if mid in self.scans[k] and ref_id in self.scans[k]]
                if not shared:
                    log.info('    %d: never seen in the same mate as %d -- nothing to compare.',
                             mid, ref_id)
                    continue
                k = shared[-1]
                here = self.scans[k][mid][0] @ self.results[mid][0]
                there = self.scans[k][ref_id][0] @ self.results[ref_id][0]
                lin, ang = pose_error(there, here)
                (log.warning if lin > 0.003 else log.info)(
                    '    %d: %.2f mm / %.2f deg from marker %d%s', mid, lin * 1000.0,
                    np.degrees(ang), ref_id,
                    '  <-- check it has not been knocked' if lin > 0.003 else '')

        mate_counts = {mid: len(rows) for mid, rows in self.offsets.items()}
        if min(mate_counts.values()) < len(self.scans):
            log.info('Mates contributing per marker: %s (of %d). A marker seen in fewer is not '
                     'wrong -- it was occluded -- but its pose rests on less evidence.',
                     ', '.join(f'{m}:{n}' for m, n in sorted(mate_counts.items())),
                     len(self.scans))
        if len(self.scans) < 2:
            log.warning('ONE MATE ONLY -- the residuals are zero by construction and mean '
                        'nothing. Set mates: >= 3 for a number that says anything, and seat '
                        'them at a consistent wrist yaw so the average means something.')
        return not failed

    # ---- write -------------------------------------------------------------------------------
    def write_outputs(self):
        stamp = f'{datetime.now():%Y-%m-%d}'
        markers = {}
        for mid in sorted(self.results):
            T, res_mm, res_deg, res_axis = self.results[mid]
            markers[mid] = {'size_m': self.sizes[mid], 'T_marker_grasp': T,
                            'meta': {'mates': len(self.offsets[mid]),
                                     'residual_mm': round(res_mm, 2),
                                     'residual_deg': round(res_deg, 2),
                                     'residual_axis_deg': round(res_axis, 2)}}
        entry = {'markers': markers, 'held_mass_kg': self.cfg.get('held_mass_kg'),
                 'meta': {'mates': len(self.scans), 'measured': stamp}}

        block = _object_block(self.name, entry)
        with open(os.path.join(self.out_dir, 'object.yaml'), 'w') as fh:
            fh.write(f'# The entry for {self.name!r}, calibrated {stamp}.\n'
                     f'# Merged into {tool_frames.objects_path(self.cfg)} by the same run.\n'
                     f'objects:\n{block}\n')
        self._write_csv()

        path = tool_frames.objects_path(self.cfg)
        if not bool(self.cfg.get('write_catalogue', True)):
            log.info('write_catalogue is off -- %s not touched. The entry:\n\n%s\n', path, block)
            return True
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(self.out_dir, 'objects.yaml.bak'))
        try:
            merged = merge_catalogue(path, self.name, entry)
        except ValueError as exc:
            log.error('%s -- refusing to rewrite the catalogue over a file that does not load.',
                      exc)
            return False
        with open(path, 'w') as fh:
            fh.write(yaml_document(merged, stamp=stamp))
        log.info('WROTE %s (%d object%s). This run:\n\n%s\n', path, len(merged),
                 '' if len(merged) == 1 else 's', block)
        return True

    def _write_csv(self):
        """One row per (mate, marker) -- the whole run, long-form, for when a number in the
        yaml looks wrong and the question is which mate or which marker produced it."""
        with open(os.path.join(self.out_dir, 'mates.csv'), 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(_CSV_HEADER)
            for mid in sorted(self.offsets):
                T_mean = self.results[mid][0]
                axis = _unit(T_mean[:3, 2])
                mate_no = 0
                for scan in self.scans:
                    if mid not in scan:
                        continue
                    T_off = self.offsets[mid][mate_no]
                    mate_no += 1
                    xyz, rpy = matrix_to_xyzrpy(T_off)
                    _Tm, lin, ang, n = scan[mid][:4]
                    dev_axis = _angle_between_deg(T_off[:3, 2], axis)
                    _lin, dev_rot = pose_error(T_mean, T_off)
                    dev_mm = float(np.linalg.norm(T_off[:3, 3] - T_mean[:3, 3])) * 1000.0
                    w.writerow([mate_no, mid, n, round(lin * 1000.0, 4),
                                round(float(np.degrees(ang)), 4)]
                               + [round(float(v) * 1000.0, 3) for v in xyz]
                               + [round(float(np.degrees(v)), 3) for v in rpy]
                               + [round(dev_axis, 4), round(float(np.degrees(dev_rot)), 4),
                                  round(dev_mm, 4)])


# ---------------------------------------------------------------------------- entry point
def build_and_run(cfg, robot, camera, args):
    name = cfg.get('object_name')
    if not name:
        log.error('object_name is required -- it is the key this run writes into %s. '
                  "Pass --set object_name=<name>.", tool_frames.objects_path(cfg))
        return False
    if cfg.get('marker') is not None:
        log.error("`marker:` is gone -- an object may carry SEVERAL markers now, so they are "
                  'declared as a `markers:` map of id -> printed size in mm:\n\n'
                  '    markers:\n      %s: %s\n\n'
                  'One entry is still perfectly valid; the plural is so a second never needs a '
                  'schema change.', cfg.get_path('marker.id', 0),
                  cfg.get_path('marker.size_mm', 40.0))
        return False
    try:
        sizes = parse_markers(cfg.get('markers'))
        T_tool0_coupler = tool_frames.coupler_mate(cfg)
        plan = mloc.ViewPlan(cfg.section('marker_views'))
    except ValueError as exc:
        log.error('%s', exc)
        return False

    # Imported here, not at module load: perception pulls in cv2, and the geometry and the
    # catalogue schema are worth testing on a machine that has no OpenCV.
    from ..perception import ArucoDetector
    detector = ArucoDetector(cfg, sizes_m=sizes)

    out_dir = experiment_dir(cfg, 'object_calibration')
    log.info('Output: %s', out_dir)
    log.info('Calibrating object %r from %d marker%s (%s), %d mate%s.',
             name, len(sizes), '' if len(sizes) == 1 else 's',
             ', '.join('%d:%.1fmm' % (m, sz * 1000.0) for m, sz in sorted(sizes.items())),
             int(cfg.get('mates', 3)), '' if int(cfg.get('mates', 3)) == 1 else 's')
    log.info('Coupler mate at %s mm along tool0.',
             np.round(T_tool0_coupler[:3, 3] * 1000.0, 2).tolist())

    coupler = Coupler(cfg)
    cal = _ObjectCalibration(cfg, robot, camera, detector, plan, name, sizes,
                             T_tool0_coupler, coupler, out_dir)
    robot.arm.set_speed_scale(float(cfg.get_path('speed.phase_scale.visual_localize', 1.0)),
                              'visual_localize')
    root = bt.sequence(
        'object-calibration',
        bt.Action('collect the mates', cal.collect),
        bt.Action('fuse the mates', cal.fuse),
        bt.Action('cross-check the spread and the yaw conditioning', cal.cross_check),
        bt.Action('write objects.yaml', cal.write_outputs))
    try:
        return bt.run_tree(root, log)
    finally:
        coupler.close()


def main():
    # with_gripper=False: the coupler does the holding, and the 2F-85 may not even be fitted.
    run_app("Object calibration: the mating feature's pose in its marker's frame",
            'object_calibration', build_and_run, with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
