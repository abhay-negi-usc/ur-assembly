"""TILE ASSEMBLY -- assemble a held tile onto a fixed one, located by its markers.

    python -m urlab.apps.tile_assembly --set marker_assembly=tile_on_plate

A COPY OF apps/coupler_marker_assemble, with its own config (configs/tile_assembly.yaml), so the
tile cycle can be changed without touching that app. Everything below describes the cycle as it
was copied.

The same cycle as apps/coupler_pick_assemble, with one thing changed: the destination is found
BY SIGHT every run instead of being a taught base_link pose. The fixed object can be moved,
re-bolted or swapped for another copy, and the run follows it.

THE RUN, in order:

    1. LOOK AT THE FIXED OBJECT from its view pose (`fixed_view_joints_deg`, or the joints the
       calibration saw it from) and locate it from its markers -- sweep, close servo, vote,
       joint PnP, the same estimate a pick uses. FIRST, AND WITH THE ARM EMPTY: once the held
       object is on the coupler it occludes the camera.
    2. Drive to the pick view (`pick_view_joints_deg`, or back to where the run started) and
       LOCATE the held object (configs/objects.yaml), mate, lock, lift -- exactly as
       coupler_pick_place does.
    3. CARRY it: to `assembly_joints_deg` if set, then through `approach_path` -- waypoints
       written as offsets from the final assembly pose, in the coupler frame there -- to the
       insertion standoff. INSERT under admittance, preload, optionally clamp the cleat,
       release, withdraw.

THE WHOLE ASSEMBLY IS ONE MULTIPLY. configs/marker_assemblies.yaml stores, for every marker on
the fixed object, the held object's MATING FRAME at its assembled position, in that marker's own
frame (urlab.apps.marker_assembly_calibration measures it). The camera measures
T_base_fixed_marker, so

    T_base_coupler_goal = T_base_fixed_marker @ T_marker_goal

and every fixed marker in view is an independent vote on it, gated and averaged exactly like a
marker rig.

WHY THE HELD OBJECT'S MARKERS DROP OUT OF THE ASSEMBLY. Marker to marker, the chain is

    fixed marker -> held marker -> held mating frame -> coupler

but the last link is mechanical: once the coupler has seated in the mating feature and locked,
the object's mating frame IS `coupler_mate`. So the run needs the held object's markers only to
FIND it for the pick. The calibration composes held marker -> mating frame once, from the
objects.yaml entry, and stores the result -- the marker-to-marker poses are printed at startup
for reference.

AN OPTIONAL `goal_offset` (xyz_mm, rpy_deg) corrects where the held object is assembled, in the
goal's own coupler frame: T_goal = T_located @ offset. That frame is attached to the fixed
object, so the correction travels with it, and everything after -- approach path, insertion,
fastening, retract -- follows the offset goal.

THE APPROACH PATH IS RELATIVE TO THE GOAL, IN ITS COUPLER FRAME: a waypoint at xyz_mm
[0, 0, -250] is 250 mm back along the assembled object's mating axis, wherever the camera found
the fixed object. Like coupler_pick_assemble's path, the LAST waypoint is the insertion standoff
-- motion.assembly_standoff is derived from it, so the two cannot disagree -- and it must be at
the assembled attitude (no rpy), because the insertion is a straight line, not a rotation.
Waypoints are IK + joint moves, so the arm's path BETWEEN them is not a straight line; add
waypoints where clearance matters.

FASTENING (`fastening.enabled`). The held object carries a coupling screw, turned by the
screwdriver on the multi-toolchanger board. With the object picked and lifted clear, the
screwdriver runs `predrive_sequence` (screwdrive_predrive: the bolt about 80% out, for
alignment); once the insertion has preloaded, it runs `fasten_sequence`
(screwdrive_predrive_to_bolt: home at low speed) BEFORE anything releases. The screw draws the
two parts together, so with `compliant: true` (the default) the insertion law holds the object
compliantly while it turns -- a rigid arm would turn any remaining gap into force on itself --
and a force-guard trip stops the screwdriver. The coupler then releases and the end effector
retracts by `fastening.retract` (200 mm along its own -z by default), which replaces
motion.final_retract for a fastened assembly.

BOLT LOCALIZATION (`fastening.mode: bolt_localization`, off by default). For testing a bolt-first
method: after the pick the screwdriver runs `bolt_localization.attach_sequence` (the bolt FULLY
out) instead of the predrive. At the end of the approach path, under its own law
(`bolt_localization_compliance`), the tool presses gently along the bolt (`drop_axis`, tool +z)
until it feels the fixed object, then oscillates slowly along `search_axis` (tool x) with a
growing amplitude until it DROPS -- advances along the bolt by `drop_mm` -- which is the bolt
falling into the hole. The reference freezes there and, under the same law, the screwdriver
runs detach then attach. That replaces both the compliant insertion and the fasten sequence;
release and retract follow as usual. Every search cycle is written to bolt_localization.csv.

THE INSERTION AXIS. `motion.assembly_standoff` is where the compliant insertion begins and its
axis, reversed, is the direction of insertion (see mating_direction). Besides `coupler` and
`base`, it may be written `frame: fixed_marker` -- the axis in one fixed marker's own frame
(`marker: <id>`, default the lowest id), e.g. [0, 0, 1] for "come in along that marker's
normal". It is converted to the coupler frame at the goal once, at startup, through the
calibration; both are attached to the fixed object, so the conversion is a constant.
"""

import csv as _csv
import math
import os
import threading

import numpy as np

from .. import behaviors as bt
from .. import log as urlog
from .. import tool_frames
from ..config import _is_derived
from ..robot.admittance import AdmittanceController
from ..robot.screwdriver import Screwdriver
from ..skills import marker_localize as mloc
from ..skills import wiggle as wg
from ..transforms import fmt_pose, inverse, matrix_to_xyzrpy, xyzrpy_to_matrix
from ._common import ask, prompts_off
from ._runner import run_app
from .coupler_pick_assemble import AssembleCycle
from ._common import experiment_dir
from .coupler_pick_place import compliance_blocks, offset_pose, parse_offset
from .coupler_pick_place import build_and_run as _cycle_build_and_run

log = urlog.get('tile-asm')

# Joint poses the run can be given, each six angles in degrees.
JOINT_KEYS = ('fixed_view_joints_deg', 'pick_view_joints_deg', 'assembly_joints_deg')
# What `joint_move_speed` may set -- the canonical `speed:` keys (URArm.parse_limits).
SPEED_KEYS = ('max_joint_velocity_deg_s', 'max_joint_acceleration_deg_s2',
              'max_cartesian_translation_mm_s', 'max_cartesian_rotation_deg_s')
# What `transition_joints_deg` keys are spelled from: <from>_to_<to>. `start` is wherever the
# run was started.
POSE_NAMES = ('start', 'fixed_view', 'pick_view', 'assembly')

# Legs that start or end at the DESTINATION, where the fixed object's frame means something. The
# pick legs happen before the fixed object is involved, so a fixed-marker axis there is refused.
MARKER_FRAME_LEGS = ('assembly_standoff', 'final_retract')


# ---------------------------------------------------------------------------- pure geometry
def resolve_marker_assembly(cfg, catalogue=None):
    """(name, entry) for the run's `marker_assembly`. Raises with the catalogued names listed --
    there is no default destination, for the same reason coupler_pick_assemble has none."""
    path = tool_frames.marker_assemblies_path(cfg)
    if catalogue is None:
        catalogue = tool_frames.load_marker_assemblies(cfg)
    name = cfg.get('marker_assembly')
    known = ', '.join(sorted(catalogue)) or '(none)'
    if not name:
        raise ValueError(f'marker_assembly is required. Calibrated in {path}: {known}. '
                         'Calibrate one with urlab.apps.marker_assembly_calibration.')
    if name not in catalogue:
        raise KeyError(f'{name!r} is not in {path}. Calibrated: {known}. Calibrate it with '
                       f'urlab.apps.marker_assembly_calibration --set marker_assembly={name}')
    return name, catalogue[name]


def fixed_rig(entry, dictionary=None):
    """The fixed object's markers as the `rig` skills/marker_localize.locate() consumes.

    The target every marker stores is the GOAL -- the coupler pose that assembles the held
    object -- so locate() returns T_base_coupler_goal directly."""
    return {'dictionary': entry.get('dictionary') or dictionary,
            'markers': {int(mid): {'size_m': float(m['size_m']),
                                   'T_marker_target': m['T_marker_goal'],
                                   'meta': dict(m.get('meta') or {})}
                        for mid, m in entry['markers'].items()}}


def marker_to_marker(T_fixed_goal, T_held_grasp):
    """T_fixed_marker_held_marker at the assembled position, from the two catalogue links.

    The goal is the held MATING frame in the fixed marker; the held marker sits at
    inverse(T_held_marker_grasp) from that same frame. Pure, so the chain is testable."""
    return np.asarray(T_fixed_goal, dtype=float) @ inverse(T_held_grasp)


def marker_leg(block, where, entry, default_mm=100.0):
    """A `frame: fixed_marker` leg, converted to the coupler frame AT THE GOAL.

    The axis is written in one fixed marker's frame -- `marker: <id>`, default the lowest. The coupler frame at the goal is the held
    mating frame at assembly, and the calibration stores exactly that pose in the marker, so

        axis_coupler = R_marker_goal^T @ axis_marker

    -- a constant, because both frames are attached to the fixed object. Converting once here
    leaves the rest of the cycle with an ordinary coupler-frame leg. Pure, so it is testable."""
    b = dict(block)
    marker_id = b.pop('marker', None)
    markers = entry['markers']
    mid = min(markers) if marker_id is None else int(marker_id)
    if mid not in markers:
        raise ValueError(f'{where}.marker {mid} is not one of the fixed object\'s markers '
                         f'{sorted(markers)}')
    spec = parse_offset(dict(b, frame='coupler'), where, default_mm)
    axis = np.asarray(markers[mid]['T_marker_goal'], dtype=float)[:3, :3].T @ spec['axis']
    spec['axis'] = axis / float(np.linalg.norm(axis))
    spec['from_marker'] = mid
    return spec


def parse_coupler_path(raw):
    """`approach_path` from the config: [{name, xyz (m), rpy (rad)}], offsets in the GOAL's
    coupler frame.

    Same schema and checks as an objects.yaml approach path (tool_frames._approach_path) --
    the config loader's derived SI siblings (xyz, rpy) are dropped first, since they are not
    something the operator wrote."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        raw = [{k: v for k, v in wp.items() if not _is_derived(v)} if isinstance(wp, dict)
               else wp for wp in raw]
    return tool_frames._approach_path(raw, 'approach_path:')


def parse_transitions(raw):
    """{(from, to): [q (rad), ...]} from `transition_joints_deg`.

    Each key is <from>_to_<to> over POSE_NAMES; each value is a list of six-angle joint poses
    in degrees, visited in order on the way (one bare six-angle list is one pose). Pure."""
    out = {}
    for key, value in dict(raw or {}).items():
        where = f'transition_joints_deg.{key}'
        parts = str(key).split('_to_')
        if (len(parts) != 2 or parts[0] not in POSE_NAMES or parts[1] not in POSE_NAMES
                or parts[0] == parts[1]):
            raise ValueError(f'{where}: expected <from>_to_<to>, two different names from '
                             f'{", ".join(POSE_NAMES)}')
        try:
            qs = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            qs = np.zeros((0, 0))
        if qs.ndim == 1:
            qs = qs[None, :]
        if qs.ndim != 2 or qs.shape[1] != 6 or not len(qs):
            raise ValueError(f'{where} must be a list of joint poses, six angles each in degrees, '
                             f'got {value!r}')
        out[(parts[0], parts[1])] = [np.radians(q) for q in qs]
    return out


def run_transitions(has_fixed, has_pick, has_assembly):
    """The (from, to) joint moves a run makes, in order, given which named poses it has. Pure.

        fixed view set   start -> fixed_view, then fixed_view -> pick_view (or -> start)
        no fixed view    start -> pick_view, when a pick view is set
        assembly set     pick_view (or start, where the pick happened) -> assembly"""
    moves = []
    if has_fixed:
        moves += [('start', 'fixed_view'), ('fixed_view', 'pick_view' if has_pick else 'start')]
    elif has_pick:
        moves.append(('start', 'pick_view'))
    if has_assembly:
        moves.append(('pick_view' if has_pick else 'start', 'assembly'))
    return moves


FASTENED_RETRACT_MM = 200.0
FASTENING_MODES = ('standard', 'bolt_localization')

# bolt_localization: key -> (default, SI scale, rule). Lengths are written in mm.
_BOLT_NUMBERS = {
    'press_force_n': (5.0, 1.0, 'nonneg'),      # 0 = do not press, just search
    'press_step_mm': (0.5, 1e-3, 'pos'),
    'press_max_mm': (30.0, 1e-3, 'pos'),
    'settle_s': (0.2, 1.0, 'pos'),
    'frequency_hz': (0.2, 1.0, 'pos'),
    'amplitude_start_mm': (0.5, 1e-3, 'nonneg'),
    'amplitude_growth_mm': (0.5, 1e-3, 'nonneg'),   # per oscillation cycle
    'amplitude_max_mm': (10.0, 1e-3, 'pos'),
    'max_time_s': (120.0, 1.0, 'pos'),
    'drop_mm': (1.0, 1e-3, 'pos'),
}
_BOLT_OTHER = {'attach_sequence': 'screwdrive_attach', 'detach_sequence': 'screwdrive_detach',
               'drop_axis': [0.0, 0.0, 1.0], 'search_axis': [1.0, 0.0, 0.0], 'save_trace': True}


def parse_bolt_localization(block):
    """The `bolt_localization:` block, validated, lengths in metres and axes unit. Pure.

    The search axis may not lie along the drop axis: the search motion would then itself read as
    a drop."""
    b = {k: v for k, v in dict(block or {}).items() if not _is_derived(v)}
    unknown = sorted(set(b) - set(_BOLT_NUMBERS) - set(_BOLT_OTHER))
    if unknown:
        raise ValueError(f'bolt_localization has unknown key(s) {unknown}')
    out = {}
    for key, (default, scale, rule) in _BOLT_NUMBERS.items():
        v = float(b.get(key, default))
        if (rule == 'pos' and not v > 0.0) or (rule == 'nonneg' and v < 0.0):
            raise ValueError(f'bolt_localization.{key} must be '
                             f'{"positive" if rule == "pos" else "zero or more"}, got {v}')
        out[key.rsplit('_', 1)[0] if scale != 1.0 else key] = v * scale
    for key in ('drop_axis', 'search_axis'):
        a = np.asarray(b.get(key, _BOLT_OTHER[key]), dtype=float)
        if a.shape != (3,) or float(np.linalg.norm(a)) < 1e-9:
            raise ValueError(f'bolt_localization.{key} must be a non-zero direction of three '
                             f'numbers, got {a.tolist()}')
        out[key] = a / float(np.linalg.norm(a))
    if abs(float(out['drop_axis'] @ out['search_axis'])) > 0.9:
        raise ValueError('bolt_localization.search_axis lies along drop_axis -- the search would '
                         'read its own motion as the drop')
    out['attach'] = str(b.get('attach_sequence') or _BOLT_OTHER['attach_sequence'])
    out['detach'] = str(b.get('detach_sequence') or _BOLT_OTHER['detach_sequence'])
    out['save_trace'] = bool(b.get('save_trace', True))
    return out


def search_amplitude(cycle, p):
    """The bolt search's amplitude (m) for oscillation cycle `cycle` (0, 1, ...): amplitude_start,
    growing by amplitude_growth per cycle, capped at amplitude_max. Pure."""
    return min(p['amplitude_start'] + p['amplitude_growth'] * cycle, p['amplitude_max'])


def rotation_taking_x_to(axis):
    """A 4x4 rotation whose x axis is `axis` -- conjugating a wiggle by it turns an oscillation
    along local x into one along `axis`. Pure."""
    x = np.asarray(axis, dtype=float) / float(np.linalg.norm(axis))
    helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(helper, x)
    y /= float(np.linalg.norm(y))
    R = np.eye(4)
    R[:3, :3] = np.column_stack([x, y, np.cross(x, y)])
    return R


class _DropWatch:
    """The `guard` handed to wiggle.run for the bolt search: ends a cycle when the real force
    guard trips OR the tool has advanced `drop` along the drop axis, and records which -- plus a
    trace row per servo cycle."""

    def __init__(self, guard, measured, force_along, p0, drop_dir, drop, rows):
        self.guard, self.measured, self.force_along = guard, measured, force_along
        self.p0, self.drop_dir, self.drop, self.rows = p0, drop_dir, drop, rows
        self.tripped = self.dropped = False
        self.t = self.offset = self.amp = self.advance = self.force = 0.0

    def check(self):
        if self.guard.check():
            self.tripped = True
            return True
        self.advance = float(np.dot(self.measured()[:3, 3] - self.p0, self.drop_dir))
        self.force = self.force_along(self.drop_dir)
        self.rows.append(['search', round(self.t, 4), round(self.amp * 1000.0, 3),
                          round(self.offset * 1000.0, 3), '', round(self.advance * 1000.0, 3),
                          round(self.force, 3)])
        if self.advance >= self.drop:
            self.dropped = True
        return self.dropped


def parse_goal_offset(block):
    """`goal_offset:` -> a 4x4 applied to the located goal (T_goal @ offset), or None when it is
    absent or all zeros. xyz_mm / rpy_deg, in the goal's own coupler frame; the rotation is
    about the offset point's own axes. Unknown keys are refused rather than ignored. Pure."""
    b = {k: v for k, v in dict(block or {}).items() if not _is_derived(v)}
    unknown = sorted(set(b) - {'xyz_mm', 'rpy_deg'})
    if unknown:
        raise ValueError(f'goal_offset has unknown key(s) {unknown} -- allowed: xyz_mm, rpy_deg')
    xyz = np.asarray(b.get('xyz_mm') or [0.0, 0.0, 0.0], dtype=float)
    rpy = np.asarray(b.get('rpy_deg') or [0.0, 0.0, 0.0], dtype=float)
    for key, v in (('xyz_mm', xyz), ('rpy_deg', rpy)):
        if v.shape != (3,):
            raise ValueError(f'goal_offset.{key} must be three numbers, got {v.tolist()}')
    if not xyz.any() and not rpy.any():
        return None
    return xyzrpy_to_matrix(xyz / 1000.0, np.radians(rpy))


def parse_joint_speed(block):
    """`joint_move_speed:` -> the per-move caps for the named joint moves, or None to use the
    global limits. Unknown keys are refused: parse_limits would silently ignore a typo and the
    move would run at the slow default. Pure."""
    b = {k: v for k, v in dict(block or {}).items() if not _is_derived(v)}
    unknown = sorted(set(b) - set(SPEED_KEYS))
    if unknown:
        raise ValueError(f'joint_move_speed has unknown key(s) {unknown} -- allowed: '
                         f'{", ".join(SPEED_KEYS)}')
    bad = [k for k, v in b.items() if not float(v) > 0.0]
    if bad:
        raise ValueError(f'joint_move_speed: {", ".join(bad)} must be positive')
    return b or None


def parse_fastening(block):
    """The `fastening:` block -> {predrive, fasten, compliant, retract, mode}, or None when it is
    off. `retract` is the raw leg block (see fastening_retract_leg). Pure."""
    b = dict(block or {})
    if not bool(b.get('enabled', False)):
        return None
    mode = str(b.get('mode') or 'standard')
    if mode not in FASTENING_MODES:
        raise ValueError(f'fastening.mode must be one of {", ".join(FASTENING_MODES)}, '
                         f'not {mode!r}')
    return {'predrive': str(b.get('predrive_sequence') or 'screwdrive_predrive'),
            'fasten': str(b.get('fasten_sequence') or 'screwdrive_predrive_to_bolt'),
            'compliant': bool(b.get('compliant', True)),
            'retract': dict(b.get('retract') or {}),
            'mode': mode}


def fastening_retract_leg(block, entry):
    """The withdrawal after a fastened assembly, as a final_retract leg: by default 200 mm along
    the end effector's own -z (coupler frame), the way it came in. `frame: fixed_marker` is
    accepted, as on the other destination legs. Pure."""
    where = 'fastening.retract'
    if dict(block).get('frame') == 'fixed_marker':
        return marker_leg(block, where, entry, FASTENED_RETRACT_MM)
    return parse_offset(block, where, FASTENED_RETRACT_MM)


def coupler_waypoint_pose(T_goal, wp):
    """One waypoint as an absolute coupler pose: T_goal @ offset, the offset in the goal's own
    coupler frame. Position and attitude both travel with the fixed object. Pure."""
    return np.asarray(T_goal, dtype=float) @ xyzrpy_to_matrix(
        np.asarray(wp['xyz'], dtype=float), np.asarray(wp.get('rpy', np.zeros(3)), dtype=float))


def coupler_standoff_from_path(path):
    """The last waypoint as the `assembly_standoff` leg in the coupler frame, or None.

    The compliant insertion runs from the standoff to the goal along a straight line at constant
    attitude, so a last waypoint with a rotation would leave a jump between where the carry ends
    and where the insertion starts -- refused rather than silently dropped. Pure."""
    if not path:
        return None
    last = path[-1]
    if float(np.linalg.norm(last.get('rpy', np.zeros(3)))) > 1e-9:
        raise ValueError(f'approach_path: the last waypoint ({last["name"]!r}) is the insertion '
                         'standoff, so it must be at the assembled attitude -- drop its rpy_deg, '
                         'or add a waypoint after it that has none')
    d = np.asarray(last['xyz'], dtype=float)
    n = float(np.linalg.norm(d))
    return {'distance_m': n, 'axis': d / n, 'frame': 'coupler'}


# ---------------------------------------------------------------------------- the run
class TileAssemblyCycle(AssembleCycle):
    """The assemble cycle with a destination located by the fixed object's markers."""

    RUN_NAME = 'tile_assembly'

    def __init__(self, cfg, robot, camera, detector, plan, name, obj, coupler):
        # BEFORE the base constructor: it parses the motion legs, and _parse_leg needs the
        # calibration to read a fixed-marker axis.
        self.masm_name, self.masm = resolve_marker_assembly(cfg)
        super().__init__(cfg, robot, camera, detector, plan, name, obj, coupler)
        clash = sorted(set(self.masm['markers']) & set(obj['markers']))
        if clash:
            raise ValueError(f'marker id(s) {clash} are on BOTH the fixed object and {name!r}. '
                             'A detection could then be credited to either, so the two '
                             'estimates would contaminate each other -- give them distinct ids.')
        dictionary = cfg.get_path('aruco.dictionary')
        if self.masm['dictionary'] and dictionary and self.masm['dictionary'] != dictionary:
            raise ValueError(f'{self.masm_name!r} was calibrated with {self.masm["dictionary"]} '
                             f'but aruco.dictionary is {dictionary} -- one detector dictionary '
                             'serves both objects, so they must match.')
        self.fixed_rig = fixed_rig(self.masm, dictionary)
        from ..perception import ArucoDetector
        self.fixed_detector = ArucoDetector(cfg, sizes_m=tool_frames.marker_sizes(self.fixed_rig))
        block = cfg.section('fixed_marker_views')
        self.fixed_plan = mloc.ViewPlan(block) if block else plan
        self.fixed_images = (mloc.MarkerImageWriter(self.out_dir, self.fixed_detector,
                                                    subdir='fixed_marker_images')
                             if self.out_dir else None)
        self.T_goal = None
        self._approach_speed_scale()
        self._joint_caps()
        offset = self._goal_offset()
        if offset is not None:
            xyz, rpy = matrix_to_xyzrpy(offset)
            log.info('goal_offset: the target is moved by xyz %s mm, rpy %s deg in its own coupler '
                     'frame.', np.round(xyz * 1000.0, 2).tolist(),
                     np.round(np.degrees(rpy), 2).tolist())
        for key in JOINT_KEYS:
            q = cfg.get(key)
            if q is not None and np.asarray(q, dtype=float).shape != (6,):
                raise ValueError(f'{key} must be six joint angles in degrees, got {q!r}')
        # TRANSITIONS between the named joint poses, and a warning for any the run never makes.
        self.transitions = parse_transitions(cfg.get('transition_joints_deg'))
        made = run_transitions(
            self._joints('fixed_view_joints_deg', self.masm['view_joints']) is not None,
            cfg.get('pick_view_joints_deg') is not None, cfg.get('assembly_joints_deg') is not None)
        unused = sorted(set(self.transitions) - set(made))
        if unused:
            log.warning('transition_joints_deg: %s will never be used -- with the joint poses '
                        'configured, this run moves %s.',
                        ', '.join(f'{a}_to_{b}' for a, b in unused),
                        ', '.join(f'{a}_to_{b}' for a, b in made) or 'between none of them')
        # THE APPROACH, relative to the goal. Set AFTER the base constructor, which reads only
        # the objects.yaml kind of path (base-frame offsets) and was handed none.
        self.approach_path = parse_coupler_path(cfg.get('approach_path'))
        derived = coupler_standoff_from_path(self.approach_path)
        if derived is not None:
            self.legs[self.TARGET_STANDOFF] = derived
            log.info('approach_path: %d waypoint%s off the goal (coupler frame). The last, %r, IS '
                     'the insertion standoff, so motion.%s is derived from it and not read: %.0f '
                     'mm along %s.', len(self.approach_path),
                     '' if len(self.approach_path) == 1 else 's', self.approach_path[-1]['name'],
                     self.TARGET_STANDOFF, derived['distance_m'] * 1000.0,
                     np.round(derived['axis'], 3).tolist())
        # FASTENING. Connected LAST, once everything else has validated: opening the port resets
        # the board, and a run refused for a config typo should not have touched it.
        self.fastening = parse_fastening(cfg.section('fastening'))
        self.screwdriver = self.bolt = None
        if self.fastening:
            retract = fastening_retract_leg(self.fastening['retract'], self.masm)
            if self._bolt_mode():
                self.bolt = parse_bolt_localization(cfg.section('bolt_localization'))
                # ITS OWN LAW, so the search can be retuned without touching the insertion's.
                # Absent, it is the insertion law's parameters.
                self.adm_bolt = AdmittanceController(
                    robot.arm, cfg.section('bolt_localization_compliance')
                    or compliance_blocks(cfg)[2])
                sequences = (self.bolt['attach'], self.bolt['detach'])
            else:
                sequences = (self.fastening['predrive'], self.fastening['fasten'])
            self.screwdriver = Screwdriver(cfg, sequences=sequences)
            # THE WITHDRAWAL CHANGES WITH IT: after the release, back off by fastening.retract
            # instead of motion.final_retract. Parsed before connecting, with everything else.
            self.legs['final_retract'] = retract
            if self.bolt:
                b = self.bolt
                log.info('Fastening ON, BOLT LOCALIZATION mode: %r after the pick; at the end of '
                         'the approach press %.1f N along %s, oscillate along %s at %.2f Hz '
                         '(%.1f -> %.1f mm, +%.1f mm/cycle, %.0f s max) until a %.1f mm drop, '
                         'then %r + %r.', b['attach'], b['press_force_n'],
                         b['drop_axis'].round(3).tolist(), b['search_axis'].round(3).tolist(),
                         b['frequency_hz'], b['amplitude_start'] * 1000.0,
                         b['amplitude_max'] * 1000.0, b['amplitude_growth'] * 1000.0,
                         b['max_time_s'], b['drop'] * 1000.0, b['detach'], b['attach'])
            else:
                log.info('Fastening ON: %r after the pick, %r once inserted (%s), then release '
                         'and retract %.0f mm along %s (%s frame).', self.fastening['predrive'],
                         self.fastening['fasten'],
                         'compliant' if self.fastening['compliant'] else 'arm held RIGID',
                         self.legs['final_retract']['distance_m'] * 1000.0,
                         np.round(self.legs['final_retract']['axis'], 3).tolist(),
                         self.legs['final_retract']['frame'])

    # ---- the destination, by sight -----------------------------------------------------------
    def _assembly_entry(self):
        """No taught pose to look up: the destination is measured in preamble()."""
        meta = self.masm['meta']
        log.info('Marker assembly %r: %r onto %d fixed marker%s (%s), calibrated %s from %s '
                 'capture(s).', self.masm_name, self.masm['held_object'],
                 len(self.masm['markers']), '' if len(self.masm['markers']) == 1 else 's',
                 ', '.join(str(m) for m in sorted(self.masm['markers'])),
                 meta.get('measured', '?'), meta.get('captures', '?'))
        return {'T_base_assembly': None, 'approach_path': [], 'meta': meta}

    def _parse_leg(self, name, block):
        b = dict(block or {})
        if b.get('frame') != 'fixed_marker':
            return super()._parse_leg(name, b)
        where = f'motion.{name}'
        if name not in MARKER_FRAME_LEGS:
            raise ValueError(f'{where}.frame: fixed_marker only means something at the '
                             f'destination ({", ".join(MARKER_FRAME_LEGS)}) -- this leg happens '
                             'before the fixed object is involved')
        spec = marker_leg(b, where, self.masm)
        log.info('%s: the axis is read in fixed marker %d\'s frame -> %s in the coupler frame '
                 'at the goal.', where, spec['from_marker'], np.round(spec['axis'], 3).tolist())
        return spec

    def preamble(self, confirm):
        return [bt.Action('look at the fixed object', self.locate_fixed, confirm=confirm)]

    def locate_fixed(self):
        """Locate the fixed object FIRST, with the arm still empty, then go back to the pick view.

        The pick view is where the run started unless `pick_view_joints_deg` says otherwise --
        the same convention as coupler_pick_place, which locates from wherever it is started."""
        q_fixed = self._joints('fixed_view_joints_deg', self.masm['view_joints'])
        q_pick = self._joints('pick_view_joints_deg', None)
        to_pick = 'pick_view' if q_pick is not None else 'start'
        if q_fixed is not None:
            if q_pick is None:
                q_pick = np.asarray(self.robot.arm.q(), dtype=float)
            if not self._transit('start', 'fixed_view'):
                return False
            log.info('Driving to the fixed object\'s view pose %s deg.',
                     np.round(np.degrees(q_fixed), 1).tolist())
            if not self.robot.move_joints(q_fixed, label='fixed object view', guard=self.guard,
                                          caps=self._joint_caps()):
                log.error('Could not reach the fixed object\'s view pose.')
                return False
        else:
            log.info('No fixed view pose (fixed_view_joints_deg, or one recorded by the '
                     'calibration) -- locating the fixed object from where the arm is.')
        servo = self.fixed_plan.servo
        if servo.enabled:
            log.info('Locating the fixed object: sweep, then %s onto each of %d marker%s at '
                     '%.0f mm.', 'ONE square-on view' if servo.single_view else 'a close servo',
                     len(self.fixed_rig['markers']),
                     '' if len(self.fixed_rig['markers']) == 1 else 's',
                     servo.distance_m * 1000.0)
        imgs = self.fixed_images
        T = mloc.locate(self.robot, self.camera, self.fixed_detector, self.fixed_rig,
                        self.fixed_plan, on_view=imgs.sweep_view if imgs else None,
                        on_servo_view=imgs.servo_view if imgs else None)
        if imgs:
            imgs.finish([f'marker assembly {self.masm_name}',
                         'GOAL ' + (fmt_pose(T) if T is not None else 'not located')])
        if T is None:
            log.error('Could not locate the fixed object -- marker(s) %s were not seen well '
                      'enough, or disagreed past the gate. Nothing has been picked.',
                      ', '.join(str(m) for m in sorted(self.fixed_rig['markers'])))
            return False
        offset = self._goal_offset()
        if offset is not None:
            log.info('LOCATED %s', fmt_pose(T))
            T = T @ offset
            log.info('  + goal_offset (coupler frame at the goal).')
        self.T_goal = T
        log.info('GOAL   %s  (the coupler pose that assembles %r)', fmt_pose(T), self.name)
        if q_pick is not None:
            if not self._transit('fixed_view' if q_fixed is not None else 'start', to_pick):
                return False
            if not self.robot.move_joints(q_pick, label='back to the pick view',
                                          guard=self.guard, caps=self._joint_caps()):
                log.error('Could not return to the pick view.')
                return False
        return True

    def _via(self, a, b):
        """[(label, q)] -- the transition poses configured for the move a -> b, in order."""
        qs = self.transitions.get((a, b), [])
        return [('transition %d/%d: %s -> %s' % (i, len(qs), a, b), q)
                for i, q in enumerate(qs, start=1)]

    def _transit(self, a, b):
        """Joint-move through the a -> b transition poses (none configured: nothing to do)."""
        for label, q in self._via(a, b):
            log.info('  %s: %s deg', label, np.round(np.degrees(q), 1).tolist())
            if not self.robot.move_joints(q, label=label, guard=self.guard,
                                          caps=self._joint_caps()):
                log.error('%s did not finish.', label)
                return False
        return True

    def _joints(self, key, fallback):
        q = self.cfg.get(key)
        return np.radians(np.asarray(q, dtype=float)) if q is not None else fallback

    def carry(self):
        """Object in hand: `assembly_joints_deg`, then the approach waypoints off the goal.

        With no path this is the one straight hop to the standoff, as before. Free space, so
        guarded moves throughout -- the compliant leg starts at the last waypoint.

        `confirm_approach_steps` pauses before EVERY move here, printing where the arm is and
        where it goes next, so a path can be walked one waypoint at a time while it is new.

        THE APPROACH PATH RUNS SLOWER: `approach_speed_scale` scales the global speed limits for
        its waypoint moves only -- they thread the object in next to the fixed one -- and the
        previous scale is restored however the carry ends."""
        steps = self._approach_steps()
        if self.approach_path:
            log.info('Approaching along %d waypoint%s (offsets from the assembly goal, in its '
                     'coupler frame).', len(self.approach_path),
                     '' if len(self.approach_path) == 1 else 's')
        scale = self._approach_speed_scale()
        restore = None
        current = f'retracted from the pick, {self.name!r} in hand'
        try:
            for label, kind, target, on_path in steps:
                where = ('joints %s deg' % np.round(np.degrees(target), 1).tolist()
                         if kind == 'joints' else fmt_pose(target))
                log.info('  %s -> %s', label, where)
                if not self._confirm_approach(current, label, where):
                    log.info('Stopped by the operator at %r, before %r.', current, label)
                    return False
                if on_path and restore is None and scale != 1.0:
                    restore = self.robot.arm.speed_scale
                    self.robot.arm.set_speed_scale(scale, 'approach_path')
                if kind == 'joints':
                    ok = self.robot.move_joints(target, label=label, guard=self.guard,
                                                caps=self._joint_caps())
                else:
                    ok = self._move_to(target, label)
                if not ok:
                    log.error('%s did not finish.', label)
                    return False
                current = label
            return True
        finally:
            if restore is not None:
                self.robot.arm.set_speed_scale(restore, 'after the approach path')

    def _goal_offset(self):
        """`goal_offset` as a 4x4 (T_goal @ offset), or None for none."""
        return parse_goal_offset(self.cfg.section('goal_offset'))

    def _joint_caps(self):
        """The speed limits for moves to the named joint poses and their transitions."""
        return parse_joint_speed(self.cfg.section('joint_move_speed'))

    def _approach_speed_scale(self):
        """`approach_speed_scale`: a fraction of the global speed limits for the approach path."""
        scale = float(self.cfg.get('approach_speed_scale', 1.0) or 1.0)
        if not scale > 0.0:
            raise ValueError(f'approach_speed_scale must be positive, got {scale}')
        return scale

    def _approach_steps(self):
        """[(label, 'joints' | 'pose', target, on_path)] -- every move the carry makes, in
        order. `on_path` marks the approach_path waypoints, which run at approach_speed_scale."""
        steps = []
        q = self._joints('assembly_joints_deg', None)
        if q is not None:
            # The pick happened at the pick view, or at the start pose when there is none.
            src = 'pick_view' if self.cfg.get('pick_view_joints_deg') is not None else 'start'
            steps += [(label, 'joints', qv, False) for label, qv in self._via(src, 'assembly')]
            steps.append(('assembly joints', 'joints', q, False))
        n = len(self.approach_path)
        for i, wp in enumerate(self.approach_path, start=1):
            steps.append(('approach %d/%d: %s' % (i, n, wp['name']), 'pose',
                          coupler_waypoint_pose(self.T_place, wp), True))
        if not n:
            steps.append((f'carry to the {self.TARGET_WORD} standoff', 'pose',
                          offset_pose(self.T_place, self.legs[self.TARGET_STANDOFF]), False))
        return steps

    def _confirm_approach(self, current, label, where):
        """Print the current and next step and wait for Enter. Off for a dry run, under
        --no-prompts, and with confirm_approach_steps: false."""
        if (not bool(self.cfg.get('confirm_approach_steps', True)) or prompts_off(self.cfg)
                or self.robot.arm.dry_run):
            return True
        log.info('  CURRENT: %s', current)
        log.info('  NEXT:    %s -> %s', label, where)
        return ask('  Enter to move, q to stop: ')

    # ---- fastening ---------------------------------------------------------------------------
    def _bolt_mode(self):
        return bool(getattr(self, 'fastening', None)) and \
            self.fastening.get('mode') == 'bolt_localization'

    def steps_after_lift(self, confirm):
        if not self.fastening:
            return []
        if self._bolt_mode():
            return [bt.Action(f'ATTACH: drive the bolt fully out ({self.bolt["attach"]})',
                              self.attach_bolt, confirm=confirm)]
        return [bt.Action(f'PREDRIVE the fastening screw ({self.fastening["predrive"]})',
                          self.predrive, confirm=confirm)]

    def steps_after_insertion(self, confirm):
        # Bolt localization screws the bolt in INSIDE set_down, under the search's own law.
        if not self.fastening or self._bolt_mode():
            return []
        return [bt.Action(f'FASTEN the screw home ({self.fastening["fasten"]})', self.fasten,
                          confirm=confirm)]

    def attach_bolt(self):
        """Bolt localization: the bolt fully out, object in hand and lifted clear."""
        return self._screw(self.bolt['attach'])

    def set_down(self):
        """The compliant insertion -- or, in bolt-localization mode, the bolt search instead."""
        if self._bolt_mode():
            return self.localize_bolt()
        return super().set_down()

    # ---- bolt localization ---------------------------------------------------------------------
    def localize_bolt(self):
        """Press, search for the hole, and screw in -- one servo session, one law, one tare.

        STARTS AT THE END OF THE APPROACH PATH with nothing touching, so the F/T is tared there,
        exactly as the insertion would. The press and search directions are the reference's own
        drop_axis / search_axis, fixed for the whole search because its attitude never changes."""
        p, adm = self.bolt, self.adm_bolt
        if self.robot.arm.dry_run:
            log.info('DRY RUN: the bolt search needs contact, so it is skipped -- running %r then '
                     '%r.', p['detach'], p['attach'])
            return self._screw(p['detach']) and self._screw(p['attach'])
        ref = lambda T: T @ inverse(self.T_tool0_coupler)       # noqa: E731 -- one expression
        T_start = offset_pose(self.T_place, self.legs[self.TARGET_STANDOFF])
        drop = T_start[:3, :3] @ p['drop_axis']
        adm.reset()
        adm.warmup(ref(T_start), tare_fn=((lambda: self.robot.arm.zero_ft(settle=False))
                                          if self.tare_before else None))
        self.guard.reset()
        self._active_law = adm
        log.info('BOLT LOCALIZATION under bolt_localization_compliance (S=%.0f/%.0f/%.0f N/m), '
                 'guarded at %.0f N.', adm.S[0], adm.S[1], adm.S[2], self.guard.max_force)
        rows = []
        try:
            T_press, p0 = self._press_bolt(adm, ref, T_start, drop, rows)
            if T_press is None:
                return False
            T_drop = self._search_bolt(adm, ref, T_press, p0, drop, rows)
            if T_drop is None:
                return False
            log.info('Holding the drop pose under the same law while the screwdriver runs %r '
                     'then %r.', p['detach'], p['attach'])
            return self._hold_while(
                adm, ref(T_drop), lambda: self._screw(p['detach']) and self._screw(p['attach']),
                f'bolt: {p["detach"]} + {p["attach"]}')
        finally:
            self._active_law = None
            self.robot.arm.servo_stop()
            self._write_bolt_trace(rows)

    def _measured(self):
        return self.robot.arm.tcp_pose() @ self.T_tool0_coupler

    def _force_along(self, direction):
        """Contact force along `direction`: arm.wrench() is the force ON the tool, so pressing
        along +direction reads as a push back along -direction; negated, pressing is positive."""
        return -float(np.dot(self.robot.arm.wrench()[:3], direction))

    def _press_bolt(self, adm, ref, T_start, drop, rows):
        """Step the reference along the drop axis until the bolt presses with press_force_n.
        (T_reference, contact position) or (None, None)."""
        p = self.bolt
        if p['press_force_n'] <= 0.0:
            log.info('  press_force_n is 0 -- searching from the end of the path without pressing.')
            return T_start, self._measured()[:3, 3]
        travelled, T_ref = 0.0, T_start
        while travelled < p['press_max']:
            travelled = min(travelled + p['press_step'], p['press_max'])
            T_next = np.array(T_start, dtype=float)
            T_next[:3, 3] = T_start[:3, 3] + drop * travelled
            if adm.ramp(ref(T_ref), ref(T_next), p['settle_s'], self.guard) == 'seated':
                log.error('The force guard tripped while pressing the bolt down -- stopping.')
                return None, None
            T_ref = T_next
            force = self._force_along(drop)
            rows.append(['press', '', '', '', round(travelled * 1000.0, 3), '', round(force, 3)])
            if force >= p['press_force_n']:
                log.info('  bolt contact: %.2f N after %.1f mm along the drop axis.', force,
                         travelled * 1000.0)
                return T_ref, self._measured()[:3, 3]
        log.error('The bolt never pressed on anything: %.2f N after the full %.0f mm along the '
                  'drop axis. Is drop_axis the bolt\'s direction, and is the fixed object where '
                  'it was located?', self._force_along(drop), p['press_max'] * 1000.0)
        return None, None

    def _search_bolt(self, adm, ref, T_press, p0, drop, rows):
        """Oscillate along the search axis with a growing amplitude until the tool advances along
        the drop axis by drop_mm. Returns the coupler reference at that moment, or None.

        THE WAVEFORM IS skills/wiggle's, one cycle per wiggle.run call with the amplitude raised
        for each: every cycle starts and ends at zero offset, so the steps join without a jump,
        and the shared run() paces it off the wall clock so the delivered frequency is the one
        configured. The drop detector rides in as its guard."""
        p = self.bolt
        f = p['frequency_hz']
        wig = wg.Wiggle([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [f, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0] * 6, label='bolt search')
        wig.validate(rate_hz=adm.rate)
        R = rotation_taking_x_to(p['search_axis'])
        R_inv = inverse(R)
        search = T_press[:3, :3] @ p['search_axis']
        watch = _DropWatch(self.guard, self._measured, self._force_along, p0, drop, p['drop'],
                           rows)
        period = 1.0 / f

        def anchor(D):
            return ref(T_press @ R @ D @ R_inv)

        def on_ref(cur, t, _tapered):
            watch.t = n * period + t
            watch.offset = float(np.dot((cur @ self.T_tool0_coupler)[:3, 3] - T_press[:3, 3],
                                        search))

        n = 0
        for n in range(max(1, math.ceil(p['max_time_s'] * f))):
            watch.amp = search_amplitude(n, p)
            status, last = wg.run(adm, wig, anchor, period, 1.0 / adm.rate, guard=watch,
                                  scale=watch.amp * 1000.0, on_ref=on_ref)
            if status == 'seated':
                if watch.tripped:
                    log.error('The force guard tripped during the bolt search -- stopping.')
                    return None
                log.info('  DROP: %.2f mm along the drop axis at %.1f s (amplitude %.1f mm, '
                         '%+.1f mm along the search axis), %.2f N left -- the bolt is in.',
                         watch.advance * 1000.0, watch.t, watch.amp * 1000.0,
                         watch.offset * 1000.0, watch.force)
                return last @ self.T_tool0_coupler
            log.info('  search cycle %d: amplitude %.1f mm, %.2f mm along the drop axis, %.2f N.',
                     n + 1, watch.amp * 1000.0, watch.advance * 1000.0, watch.force)
        log.error('No drop within %.0f s (amplitude reached %.1f mm). The bolt did not find the '
                  'hole -- check the located goal, goal_offset, or widen amplitude_max_mm.',
                  p['max_time_s'], search_amplitude(n, p) * 1000.0)
        return None

    def _write_bolt_trace(self, rows):
        if not rows or not self.bolt.get('save_trace', True):
            return
        try:
            out = getattr(self, 'out_dir', None) or experiment_dir(self.cfg, self.RUN_NAME)
            path = os.path.join(out, 'bolt_localization.csv')
            with open(path, 'w', newline='') as fh:
                w = _csv.writer(fh)
                w.writerow(['phase', 't_s', 'amplitude_mm', 'offset_mm', 'press_mm',
                            'advance_mm', 'force_n'])
                w.writerows(rows)
            log.info('Bolt localization trace -> %s', path)
        except OSError as exc:
            log.warning('Could not write the bolt localization trace: %s', exc)

    def predrive(self):
        """Run the bolt out for alignment -- object in hand, lifted clear, nothing touching."""
        return self._screw(self.fastening['predrive'])

    def fasten(self):
        """Drive the screw home with the object inserted and preloaded, before any release."""
        seq = self.fastening['fasten']
        if self.robot.arm.dry_run:
            return self._screw(seq)
        if not self.fastening['compliant']:
            log.warning('fastening.compliant is off -- the arm is held RIGID while %r draws the '
                        'parts together, so any gap left becomes force on the arm.', seq)
            return self._screw(seq)
        return self._compliant_while(lambda: self._screw(seq), f'fasten ({seq})')

    def _screw(self, seq):
        ok = self.screwdriver.run(seq)
        if not ok:
            log.error('The screwdriver did not complete %r. The object is still on the coupler; '
                      'nothing has been released.', seq)
        return ok

    def _compliant_while(self, work, what):
        """Run `work` on a worker thread while the INSERTION law holds the object where it is.

        IN CONTACT, so no tare and a rebase: the spring is re-referenced onto where the object
        actually sits, and yields as the screw draws the parts together. The worker is the only
        thread that talks to the screwdriver; a force-guard trip here only asks it to stop
        (Screwdriver.stop), and the arm stays compliant until it has."""
        adm = self.adm_insert
        ref = lambda T: T @ inverse(self.T_tool0_coupler)       # noqa: E731 -- one expression
        T_hold = adm.rebase(self.robot.arm.tcp_pose() @ self.T_tool0_coupler)
        adm.warmup(ref(T_hold))
        self.guard.reset()
        log.info('%s under the INSERTION law (S=%.0f/%.0f/%.0f N/m), guarded at %.0f N.', what,
                 adm.S[0], adm.S[1], adm.S[2], self.guard.max_force)
        try:
            return self._hold_while(adm, ref(T_hold), work, what)
        finally:
            self.robot.arm.servo_stop()

    def _hold_while(self, adm, T_ref_cmd, work, what):
        """Inside an active servo session: hold `T_ref_cmd` under `adm` while `work` runs on a
        worker thread. A guard trip only asks the screwdriver to stop, and the hold carries on
        until it has. The caller owns the session (warmup before, servo_stop after)."""
        result = {}
        worker = threading.Thread(target=lambda: result.__setitem__('ok', work()),
                                  name='screwdriver', daemon=True)
        tripped = False
        worker.start()
        try:
            while worker.is_alive():
                status = adm.hold(T_ref_cmd, 0.1, None if tripped else self.guard)
                if status == 'seated' and not tripped:
                    tripped = True
                    log.error('The force guard tripped during %s -- stopping the screwdriver; '
                              'the arm stays compliant until it has.', what)
                    self.screwdriver.stop()
        finally:
            if worker.is_alive():               # an exception out of the hold loop
                self.screwdriver.stop()
                self.robot.arm.servo_stop()
            worker.join(timeout=10.0)
        delta = adm.delta
        log.info('  %s: the law yielded %.1f mm / %.2f deg.', what,
                 float(np.linalg.norm(delta[:3])) * 1000.0,
                 float(np.degrees(np.linalg.norm(delta[3:]))))
        return bool(result.get('ok')) and not tripped

    def teardown(self):
        """Close the screwdriver too (closing resets its board, which stops the motor)."""
        try:
            return super().teardown()
        finally:
            if getattr(self, 'screwdriver', None) is not None:
                self.screwdriver.close()

    def _target_pose(self, T_pick):
        """Where the fixed object's markers said the coupler has to be. Ignores the pick."""
        if self.T_goal is None:
            raise ValueError('the fixed object has not been located -- preamble() must run '
                             'before the pick')
        return self.T_goal


# ---------------------------------------------------------------------------- entry point
def build_and_run(cfg, robot, camera, args):
    # Fail on a missing or mismatched assembly BEFORE anything moves.
    try:
        name, entry = resolve_marker_assembly(cfg)
        objects = tool_frames.load_objects(cfg)
    except (ValueError, KeyError) as exc:
        log.error('%s', exc.args[0] if exc.args else exc)
        return False
    held = entry['held_object']
    if not cfg.get('object_name'):
        cfg['object_name'] = held
    elif cfg['object_name'] != held:
        log.error('%r was calibrated for %r, not object_name=%r. Leave object_name unset to '
                  'take it from the assembly.', name, held, cfg['object_name'])
        return False
    if held not in objects:
        log.error('%r assembles %r, which is not in %s.', name, held,
                  tool_frames.objects_path(cfg))
        return False
    log.info('Marker to marker at the assembled position (fixed -> held, for reference):')
    for fid in sorted(entry['markers']):
        for hid in sorted(objects[held]['markers']):
            xyz, rpy = matrix_to_xyzrpy(marker_to_marker(
                entry['markers'][fid]['T_marker_goal'],
                objects[held]['markers'][hid]['T_marker_grasp']))
            log.info('  %d -> %d: xyz %s mm  rpy %s deg', fid, hid,
                     np.round(xyz * 1000.0, 2).tolist(), np.round(np.degrees(rpy), 2).tolist())
    return _cycle_build_and_run(cfg, robot, camera, args, cycle=TileAssemblyCycle)


def main():
    run_app('Tile assembly: locate a fixed tile by its markers, pick a tile, assemble onto it',
            'tile_assembly', build_and_run, with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
