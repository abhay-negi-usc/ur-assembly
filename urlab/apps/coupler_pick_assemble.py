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
"""

import numpy as np

from .. import log as urlog
from .. import tool_frames
from ._runner import run_app
from .coupler_pick_place import DEFAULT_LEG_MM, CouplerCycle
from .coupler_pick_place import build_and_run as _cycle_build_and_run

log = urlog.get('coupler-asm')

# The assembly standoff replaces the place standoff; everything else is the same trip.
LEGS = ('mate_standoff', 'pick_retract', 'assembly_standoff', 'final_retract')
DEFAULT_LEG_MM = dict(DEFAULT_LEG_MM, assembly_standoff=100.0)


class AssembleCycle(CouplerCycle):
    """The pick-and-place cycle with a taught assembly pose as the destination."""

    LEGS = LEGS
    TARGET_STANDOFF = 'assembly_standoff'
    TARGET_PRELOAD = 'assembly_preload'
    TARGET_WORD = 'assemble'

    def _target_pose(self, T_pick):
        """The taught assembly pose, straight out of the catalogue. Ignores where it was picked.

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
        return entry['T_base_assembly']


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
    return _cycle_build_and_run(cfg, robot, camera, args, cycle=AssembleCycle)


def main():
    run_app('Coupler pick and assemble: find a catalogued object, pick it, assemble it',
            'coupler_pick_assemble', build_and_run, with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
