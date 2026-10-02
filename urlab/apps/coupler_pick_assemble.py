"""COUPLER PICK AND ASSEMBLE -- find a catalogued object, pick it, drive it into its assembly.

    python -m urlab.apps.coupler_pick_assemble --set object_name=ORU_v7 \\
                                               --set assembly_name=rack_slot_1

The same cycle as apps/coupler_pick_place, with one thing changed: where the object ends up. That
is the whole difference, and it is why this file is short -- the locating, mating, preloading,
locking, payload handling, two compliance laws and tare timing are all inherited rather than
copied. See CouplerCycle.

A SET-DOWN IS A DELTA; AN ASSEMBLY IS A PLACE. Placing puts the object somewhere relative to
where it was found, so a delta is right: the pick pose is not known until the camera looks, and
an absolute pose would need re-teaching whenever the object started somewhere new. An assembly
is the opposite -- the FIXTURE does not move with the part. Where the object came from says
nothing about where it has to go, so the target is an absolute pose, taught once by
apps/coupler_assembly_calibration and read out of configs/objects.yaml.

WHICH MEANS THE ERRORS COMPOSE DIFFERENTLY. A placement inherits the pick's error and then adds
nothing: get the pick wrong by 2 mm and the object lands 2 mm off, still square in the hand. An
assembly inherits the pick error and drives it into a fixture that does not know about it --
the object is held 2 mm off where the arm believes, and the assembly pose is exact, so the 2 mm
shows up as interference at the fixture. That is what the compliant insertion and the preload
are for, and why `assembly_preload.force_n` is worth setting deliberately rather than leaving at
the touchdown default.

THE ASSEMBLY TARGET IS KINEMATIC. It was taught by hand-guiding and is exactly as good as the
cell staying put -- see apps/coupler_assembly_calibration for what that costs and what the
alternative is.

AND IT IS REACHED ALONG A TAUGHT PATH, not in one straight line. A standoff describes a
destination reachable from anywhere; a fixture with structure around it -- a bracket to get past,
a deck to come in over -- is not one, and the straight line to its standoff goes through the
obstacle. The route is `approach_path` on the assembly in configs/objects.yaml: base-frame
offsets from the assembled pose, walked in order, carried at the attitude the object is going to
be assembled in. It belongs to the STATION and not to this app, because two assemblies on the
same bench need two different ways in.

THE LAST WAYPOINT IS THE INSERTION STANDOFF, and `motion.assembly_standoff` is derived from it
rather than read. They are one fact -- where the compliant insertion begins -- and the standoff
also fixes the insertion AXIS, so two settings that disagreed would have the arm thread its path,
hop somewhere else, and press home along a line it never approached on.

THE DESTINATION CAN HOLD ON TO THE OBJECT. `cleat_toolchanger:` is a second board, on the cleat
the part mounts to, and when it is configured the run hands over rather than just letting go: the
cleat is opened before the pick, clamps once the insertion has preloaded, and is re-checked --
all BEFORE the end-effector coupler releases. Between the two the object is held twice, which is
harmless; the other order holds it none. The cleat is then left holding and latched, because the
point of the run is that the part stays mounted after it.
"""

import numpy as np

from .. import log as urlog
from .. import tool_frames
from ..robot.coupler import Coupler
from ..transforms import fmt_pose, xyzrpy_to_matrix
from ._runner import run_app
from .coupler_pick_place import DEFAULT_LEG_MM, CouplerCycle
from .coupler_pick_place import build_and_run as _cycle_build_and_run

log = urlog.get('coupler-asm')

# The assembly standoff replaces the place standoff; everything else is the same trip.
LEGS = ('mate_standoff', 'pick_retract', 'assembly_standoff', 'final_retract')
DEFAULT_LEG_MM = dict(DEFAULT_LEG_MM, assembly_standoff=100.0)


# The cleat -- or whatever the destination's own clamp is -- lives in its own config block, so
# the two boards can differ in port, latch and bypass without either inheriting the other's.
CLEAT_SECTION = 'cleat_toolchanger'


def waypoint_pose(T_target, wp):
    """One approach waypoint as an absolute pose: a BASE-FRAME offset from the assembled pose.

    THE OFFSET IS IN BASE AXES AND THE ATTITUDE IS THE TARGET'S. That pairing is the readable
    one for threading a part into a fixture: the numbers are "250 mm up and 250 mm back" off a
    pose already on the monitor, while the object stays at the attitude it is going to be
    assembled in -- the one attitude already proven to fit through the gap. `rpy_deg` re-orients
    the tool AT the waypoint, about base axes and about the waypoint's own origin, so a rotation
    never drags the position with it.

    Pure, so the convention is testable without a robot."""
    T = np.array(T_target, dtype=float)
    T[:3, 3] = T[:3, 3] + np.asarray(wp['xyz'], dtype=float)
    rpy = np.asarray(wp.get('rpy', [0.0, 0.0, 0.0]), dtype=float)
    if float(np.linalg.norm(rpy)) > 1e-12:
        T[:3, :3] = xyzrpy_to_matrix(np.zeros(3), rpy)[:3, :3] @ T[:3, :3]
    return T


def standoff_from_path(path):
    """The last waypoint, as the `assembly_standoff` motion leg -- or None with no path.

    ONE SETTING, NOT TWO. The last waypoint is where the compliant insertion begins and
    motion.assembly_standoff is where the compliant insertion begins, so they are the same fact
    written down twice -- and two independent settings can disagree. That disagreement is not a
    tidiness problem: the standoff also DEFINES the insertion axis (see mating_direction), so an
    approach path that ends 50 mm out along base +x while the standoff says 100 mm along the
    coupler's -z would have the arm thread its path, hop somewhere else entirely, and then press
    home along an axis it never approached on. Deriving one from the other makes that
    unrepresentable.

    Pure, so the derivation is testable."""
    if not path:
        return None
    d = np.asarray(path[-1]['xyz'], dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:                     # tool_frames rejects this already; belt and braces
        raise ValueError('the approach path ends at the assembled pose, leaving the insertion '
                         'no direction to travel along')
    return {'distance_m': n, 'axis': d / n, 'frame': 'base'}


class AssembleCycle(CouplerCycle):
    """The pick-and-place cycle with a taught assembly pose as the destination."""

    LEGS = LEGS
    TARGET_STANDOFF = 'assembly_standoff'
    TARGET_PRELOAD = 'assembly_preload'
    TARGET_WORD = 'assemble'
    RUN_NAME = 'coupler_pick_assemble'
    PREPARE_TARGET_LABEL = 'power and OPEN the cleat coupler'
    SECURE_LABEL = 'CLAMP the cleat onto the assembled object'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # RESOLVED IN THE CONSTRUCTOR, not at locate() time, so a missing assembly or a broken
        # approach path stops the run BEFORE the camera sweep -- which with the servo refinement
        # on is the better part of a minute -- rather than after it.
        self.entry = self._assembly_entry()
        self.approach_path = self.entry.get('approach_path') or []
        # The tool's attitude at the destination. Known here because a taught assembly pose
        # carries it, and the shared startup check needs it to read a base-frame insertion axis
        # against tool-frame stiffnesses. See insertion_axis_is_stiff.
        # None when the destination is only known once the camera has looked (a marker-relative
        # assembly, apps/coupler_marker_assemble).
        T_asm = self.entry.get('T_base_assembly')
        self.target_rotation = None if T_asm is None else np.asarray(T_asm, dtype=float)[:3, :3]
        derived = standoff_from_path(self.approach_path)
        if derived is not None:
            self.legs[self.TARGET_STANDOFF] = derived
            log.info('The approach path\'s last waypoint (%r) IS the insertion standoff, so '
                     'motion.%s is derived from it and not read: %.0f mm along %s in base.',
                     self.approach_path[-1]['name'], self.TARGET_STANDOFF,
                     derived['distance_m'] * 1000.0, np.round(derived['axis'], 3).tolist())
        self.cleat = self._build_cleat()

    # ---- the destination ---------------------------------------------------------------------
    @property
    def secures_target(self):
        """True when a cleat coupler is configured -- see CouplerCycle.secures_target."""
        return self.cleat is not None

    def _build_cleat(self):
        block = self.cfg.section(CLEAT_SECTION)
        if not block or not bool(block.get('enabled', True)):
            return None
        return Coupler(self.cfg, section=CLEAT_SECTION, label='cleat')

    def prepare_target(self):
        """Power the cleat's actuation and open its jaws, before the object is anywhere near it.

        A CLEAT THAT IS ALREADY SHUT CANNOT BE ENTERED, and the board makes that more likely than
        it sounds: setup() adopts the sensor's opinion as its state, so a cleat that booted with
        anything in front of its probe comes up clamped. Opening here also makes the clamp at the
        end a real servo transition rather than a no-op that still answers "confirmed!"."""
        if not self.cleat.prepare_to_mate():
            log.error('The CLEAT coupler could not be powered and opened, so the object would be '
                      'driven into a clamp that is already shut. Nothing has moved.')
            return False
        if not self.cleat.latched:
            log.warning('The cleat is NOT latched, so closing its port at the end of this run '
                        'RESETS its board -- and a reset re-decides the clamp from a single '
                        'sensor reading taken microseconds after power-on. The object this run '
                        'assembles can therefore be dropped by the teardown. Set '
                        '%s.latch: true.', CLEAT_SECTION)
        return True

    def secure_target(self):
        """Clamp the cleat onto the object while the end-effector coupler is still holding it.

        THIS IS THE HANDOVER, and the run's job here is to refuse to proceed if it did not
        happen. A refused clamp means the cleat is gripping nothing -- the part did not reach its
        seat, or stopped short on a chamfer -- and the very next steps release the coupler and
        pull away. Failing here leaves the object still on the coupler, at the assembled pose,
        which is the state it can be backed out of."""
        if not self.cleat.hold():
            log.error('The CLEAT would not clamp %r. The end-effector coupler is NOT releasing, '
                      'so the object is still held -- back it out along the approach path once '
                      'you know why.', self.name)
            return False
        again = self.cleat.verify()
        if again is False:
            log.error('The cleat confirmed the clamp and then its sensor disagreed on a '
                      're-check. %r is not reliably held by the cleat; refusing to let go of '
                      'it.', self.name)
            return False
        if again is None:
            log.warning('The cleat reports clamped but NOTHING VERIFIED IT -- no board, or the '
                        'sensor is bypassed. The release below assumes the cleat has the object.')
        log.info('The cleat is holding %r. It is now held by BOTH mechanisms, which is the only '
                 'safe moment for the coupler to let go.', self.name)
        return True

    def teardown(self):
        """Leave the cleat holding, and say so. Nothing here releases it.

        The cleat is the whole point of the assembly: the run ends with the object mounted, so
        the teardown must not be the thing that unmounts it. With `latch: true` the port closes
        without dropping DTR and the board keeps the servo and the relay where they are."""
        if self.cleat is None:
            return True
        if not self.cleat.latched and not self.cleat.manual:
            log.warning('Closing the cleat\'s port now resets its board, which re-decides the '
                        'clamp from one sensor reading. Support %r if it must not fall.',
                        self.name)
        self.cleat.close()
        return True

    # ---- the approach --------------------------------------------------------------------------
    def carry(self):
        """Thread the taught approach path, instead of one straight hop to the standoff.

        WHY A PATH AND NOT A LEG. A standoff is a single offset, which describes a destination
        that can be reached in a straight line from anywhere the object happens to be. A fixture
        surrounded by its own structure -- a bracket to get past, a deck to come in over -- cannot
        be, and the straight line to its standoff goes through the obstacle. The waypoints are
        that route, catalogued with the station they belong to.

        FREE SPACE, SO PLAIN GUARDED MOVES. Nothing on the way in should touch anything; the
        guard is armed the whole way precisely because that assumption is the one worth checking,
        and the compliant leg starts at the last waypoint."""
        if not self.approach_path:
            return super().carry()
        n = len(self.approach_path)
        log.info('Approaching %r along %d taught waypoint%s (base-frame offsets from the '
                 'assembled pose, carried at the assembled attitude).',
                 self.cfg.get('assembly_name'), n, '' if n == 1 else 's')
        for i, wp in enumerate(self.approach_path, start=1):
            T = waypoint_pose(self.T_place, wp)
            label = 'approach %d/%d: %s' % (i, n, wp['name'])
            log.info('  %s -> %s', label, fmt_pose(T))
            if not self._move_to(T, label):
                return False
        return True

    # ---- the target ----------------------------------------------------------------------------
    def _assembly_entry(self):
        """The catalogue's entry for the named assembly: pose, approach path and provenance.

        RAISES rather than falling back to anything. An assembly the catalogue has never been
        taught has no sensible default -- driving the object to the pick pose, or to some delta
        from it, would be a confident move to a place nobody chose."""
        name = self.cfg.get('assembly_name')
        assemblies = self.obj.get('assemblies') or {}
        if not name:
            raise ValueError(
                'assembly_name is required. Taught for %r: %s. Teach one with '
                'urlab.apps.coupler_assembly_calibration.'
                % (self.name, ', '.join(sorted(assemblies)) or '(none)'))
        if name not in assemblies:
            raise KeyError(
                '%r has no assembly %r. Taught: %s. Teach it with '
                'urlab.apps.coupler_assembly_calibration --set object_name=%s '
                '--set assembly_name=%s'
                % (self.name, name, ', '.join(sorted(assemblies)) or '(none)', self.name, name))
        entry = assemblies[name]
        meta = entry.get('meta') or {}
        log.info('Assembly %r: taught %s from %s approach(es) (%s mm / %s deg).', name,
                 meta.get('measured', '?'), meta.get('approaches', '?'),
                 meta.get('residual_mm', '?'), meta.get('residual_deg', '?'))
        if int(meta.get('approaches', 0) or 0) < 2:
            log.warning('That assembly was taught from a SINGLE approach, so its residuals are '
                        'zero by construction and say nothing about how repeatably the part '
                        'seats. Re-teach with approaches: >= 3 before trusting it.')
        return entry

    def _target_pose(self, T_pick):
        """The taught assembly pose, straight out of the catalogue. Ignores where it was picked."""
        return self.entry['T_base_assembly']


def build_and_run(cfg, robot, camera, args):
    # Fail on a missing assembly BEFORE the camera sweep rather than after it -- locating takes
    # the better part of a minute with the servo refinement on, and a typo in the name should
    # not cost that.
    name, assembly = cfg.get('object_name'), cfg.get('assembly_name')
    try:
        catalogue = tool_frames.load_objects(cfg)
    except ValueError as exc:
        log.error('%s', exc)
        return False
    obj = catalogue.get(name or '')
    if obj is not None and assembly and assembly not in (obj.get('assemblies') or {}):
        log.error('%r has no assembly %r. Taught: %s. Teach it with '
                  'urlab.apps.coupler_assembly_calibration.', name, assembly,
                  ', '.join(sorted(obj.get('assemblies') or {})) or '(none)')
        return False
    entry = (obj or {}).get('assemblies', {}).get(assembly or '') or {}
    if entry and not entry.get('approach_path'):
        log.info('Assembly %r has no approach_path, so the object goes to its standoff in ONE '
                 'straight line from wherever it was picked. That is right for a fixture in open '
                 'space and wrong for one with structure around it -- add waypoints under the '
                 'assembly in configs/objects.yaml if the straight line goes through something.',
                 assembly)
    return _cycle_build_and_run(cfg, robot, camera, args, cycle=AssembleCycle)


def main():
    run_app('Coupler pick and assemble: find a catalogued object, pick it, assemble it',
            'coupler_pick_assemble', build_and_run, with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
