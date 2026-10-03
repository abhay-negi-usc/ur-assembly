"""DISASSEMBLE TILE -- grasp an assembled, cantilevered tile, unscrew it, pull it off.

    python -m urlab.apps.disassemble_tile --set object_name=tile_1

The pick half of coupler_pick_place, then the reverse of coupler_marker_assemble's fastening:

    1. LOCATE the tile by its own markers (configs/objects.yaml), exactly as coupler_pick_place
       does -- from wherever the run is started, so start it with the tile's markers in view.
    2. ALIGN at `motion.mate_standoff` and MATE under admittance, then push to
       `mate_preload.force_n`. LOW FORCES THROUGHOUT, because the tile is CANTILEVERED off the
       fixture: whatever the coupler pushes with bends the tile's mount, and the tile deflects
       away from the coupler instead of seating in it. `force_guard` is low for the same reason;
       it is the guard for the whole run, so it caps the unscrewing and the pull too.
    3. LOCK, and take the tile's mass as payload, with NO tare (see
       coupler_pick_place.take_payload). From here the arm carries the tile's weight, so the
       screw comes out unloaded and the tile does not drop when it lets go.
    4. UNSCREW: `screwdriver.detach_sequence` (screwdrive_detach) while the loaded law holds the
       tile compliantly where it is. A force-guard trip stops the screwdriver.
    5. DISASSEMBLE: a compliant pull along `motion.disassemble` -- by default 100 mm along the end
       effector's own -X. A guard trip here is a FAILURE, not contact: a tile that will not come
       off is still screwed in or jammed, and the run stops with it held where the trip left it.

THE RUN ENDS HOLDING THE TILE at the pulled-off pose, coupler locked and powered. Closing the
coupler's port resets its board, which can open it (robot/coupler.py), so the config sets
`toolchanger.latch: true`; with it off, the run warns before anything moves.
"""

import numpy as np

from .. import behaviors as bt
from .. import log as urlog
from .. import tool_frames
from ..robot.coupler import Coupler
from ..robot.screwdriver import Screwdriver
from ..skills import marker_localize as mloc
from ..transforms import inverse, pose_error
from ._common import prompts_off
from ._runner import run_app
from .coupler_marker_assemble import MarkerAssembleCycle
from .coupler_pick_place import (CouplerCycle, describe_leg, offset_pose, parse_offset,
                                 step_gate)

log = urlog.get('disassemble')

DISASSEMBLE_MM = 100.0
DISASSEMBLE_AXIS = [-1.0, 0.0, 0.0]          # the end effector's own -X


class DisassembleCycle(CouplerCycle):
    """Locate, mate, lock and take the payload exactly as CouplerCycle does; then unscrew and
    pull off instead of carrying and placing."""

    LEGS = ('mate_standoff', 'disassemble')
    TARGET_WORD = 'pull-off'
    RUN_NAME = 'disassemble_tile'

    # The screwdriver hold is coupler_marker_assemble's, borrowed rather than copied: the same
    # worker thread, and the same stop-the-screwdriver-on-a-trip.
    _hold_while = MarkerAssembleCycle._hold_while
    _screw = MarkerAssembleCycle._screw

    def __init__(self, cfg, robot, camera, detector, plan, name, obj, coupler):
        super().__init__(cfg, robot, camera, detector, plan, name, obj, coupler)
        sd = cfg.section('screwdriver')
        self.detach_seq = str(sd.get('detach_sequence') or 'screwdrive_detach')
        self.compliant_detach = bool(sd.get('compliant', True))
        # Connected LAST: opening the port resets the board, and a run refused for a config typo
        # should not have touched it.
        self.screwdriver = Screwdriver(cfg, section='screwdriver', sequences=(self.detach_seq,))

    def _parse_leg(self, name, block):
        if name != 'disassemble':
            return super()._parse_leg(name, block)
        b = dict(block or {})
        b.setdefault('axis', DISASSEMBLE_AXIS)
        return parse_offset(b, 'motion.disassemble', DISASSEMBLE_MM)

    def _target_pose(self, T_pick):
        """Where the pull ends: the grasp pose moved along motion.disassemble. Being the cycle's
        target, it is logged at locate() and checked against min_grasp_z_mm before anything
        moves."""
        return offset_pose(T_pick, self.legs['disassemble'])

    def steps(self, confirm):
        """The run, in order. The one-off gate before the first motion toward the tile is NOT
        silenced by --yes, as in coupler_pick_place."""
        return [
            bt.Action('locate the tile', self.locate, confirm=confirm),
            bt.Action('power and open the coupler', self.prepare_coupler, confirm=confirm),
            bt.OperatorGate(self.robot, 'About to approach and GRASP the tile. Clear of the arm? '
                            '(Enter / q): ', label='before the grasp',
                            skip=prompts_off(self.cfg) or confirm is not None),
            bt.Action('align at the mate standoff', self.approach, confirm=confirm),
            bt.Action('mate with the tile', self.descend_and_mate, confirm=confirm),
            bt.Action('lock the coupler', self.lock, confirm=confirm),
            bt.Action('take the payload', self.take_payload),
            bt.Action(f'unscrew the tile ({self.detach_seq})', self.detach, confirm=confirm),
            bt.Action('pull the tile off', self.disassemble, confirm=confirm),
            bt.Action('report', self.report)]

    # ---- unscrew -----------------------------------------------------------------------------
    def detach(self):
        """Back the screw out with the arm carrying the tile.

        IN CONTACT, so no tare and a rebase: the spring is re-referenced onto where the tile
        actually sits, and the law lets it move as the screw backs out rather than holding it
        rigidly against the fixture. The worker thread is the only one that talks to the
        screwdriver; a force-guard trip only asks it to stop."""
        seq = self.detach_seq
        if self.robot.arm.dry_run:
            return self._screw(seq)
        if not self.compliant_detach:
            log.warning('screwdriver.compliant is off -- the arm is held RIGID while %r backs '
                        'the screw out, so any misalignment becomes force on the tile.', seq)
            return self._screw(seq)
        adm = self._adm()
        ref = lambda T: T @ inverse(self.T_tool0_coupler)       # noqa: E731 -- one expression
        T_hold = adm.rebase(self.robot.arm.tcp_pose() @ self.T_tool0_coupler)
        adm.warmup(ref(T_hold))
        self.guard.reset()
        log.info('Unscrewing (%s) under the %s law (S=%.0f/%.0f/%.0f N/m), guarded at %.0f N.',
                 seq, 'LOADED' if self.holding else 'free', adm.S[0], adm.S[1], adm.S[2],
                 self.guard.max_force)
        try:
            return self._hold_while(adm, ref(T_hold), lambda: self._screw(seq),
                                    f'unscrew ({seq})')
        finally:
            self.robot.arm.servo_stop()

    # ---- pull off ----------------------------------------------------------------------------
    def disassemble(self):
        """Pull the tile off along motion.disassemble, under the loaded law.

        `in_contact`, like coupler_pick_place's lift: the tile is still seated on the fixture
        when this starts, so no tare, and the ramp starts from where the arm IS.

        A GUARD TRIP IS A FAILURE HERE. _compliant treats a trip as contact reached and holds,
        which is right for a mate and wrong for a pull: the tile has not come off."""
        T_to = offset_pose(self.T_pick, self.legs['disassemble'])
        if not self._compliant(self.T_pick, T_to, 'pull the tile off', in_contact=True):
            return False
        if self.guard.tripped_by:
            log.error('The pull tripped the force guard (%s) after %.1f of %.0f mm -- the tile '
                      'has NOT come off: it is still screwed in or jammed. It is held where the '
                      'trip left it, coupler locked.', self.guard.tripped_by,
                      self._pulled_mm(), self.legs['disassemble']['distance_m'] * 1000.0)
            return False
        return True

    def _pulled_mm(self):
        """How far the coupler is from the grasp pose ALONG the pull axis, in mm."""
        spec = self.legs['disassemble']
        axis = (self.T_pick[:3, :3] @ spec['axis'] if spec['frame'] == 'coupler'
                else spec['axis'])
        here = self.robot.arm.tcp_pose() @ self.T_tool0_coupler
        return float(np.dot(here[:3, 3] - self.T_pick[:3, 3], axis)) * 1000.0

    # ---- reporting ---------------------------------------------------------------------------
    def report(self):
        here = self.robot.arm.tcp_pose() @ self.T_tool0_coupler
        lin, ang = pose_error(self.T_place, here)
        spec = self.legs['disassemble']
        log.info('DISASSEMBLED %r: pulled %.1f of %.0f mm along %s (%s frame); %.1f mm / %.2f '
                 'deg from the commanded pull-off pose.', self.name, self._pulled_mm(),
                 spec['distance_m'] * 1000.0, np.round(spec['axis'], 3).tolist(), spec['frame'],
                 lin * 1000.0, np.degrees(ang))
        log.info('HOLDING %r, coupler locked. %s', self.name,
                 'The coupler is LATCHED, so it stays locked and powered after this exits.'
                 if self.coupler.latched else
                 'The coupler is NOT latched: closing its port resets the board, which can open '
                 'it and drop the tile.')
        return True

    def teardown(self):
        """Close the screwdriver (closing resets its board, which stops the motor)."""
        try:
            return super().teardown()
        finally:
            if getattr(self, 'screwdriver', None) is not None:
                self.screwdriver.close()


# ---------------------------------------------------------------------------- entry point
def build_and_run(cfg, robot, camera, args):
    name = cfg.get('object_name')
    try:
        catalogue = tool_frames.load_objects(cfg)
        plan = mloc.ViewPlan(cfg.section('marker_views'))
    except ValueError as exc:
        log.error('%s', exc)
        return False
    if name not in catalogue:
        log.error('object_name must name a catalogued object. In %s: %s.',
                  tool_frames.objects_path(cfg), ', '.join(sorted(catalogue)) or '(none)')
        return False
    obj = catalogue[name]

    from ..perception import ArucoDetector
    detector = ArucoDetector(cfg, sizes_m={int(mid): m['size_m']
                                           for mid, m in obj['markers'].items()})
    coupler = Coupler(cfg)
    step = step_gate(cfg, robot)
    try:
        job = DisassembleCycle(cfg, robot, camera, detector, plan, name, obj, coupler)
    except (ValueError, KeyError) as exc:
        log.error('%s', exc)
        coupler.close()
        return False
    log.info('Disassembling %r (%d marker%s, %s kg).', name, len(obj['markers']),
             '' if len(obj['markers']) == 1 else 's', obj.get('held_mass_kg'))
    for leg in job.LEGS:
        log.info('  %s', describe_leg(leg, job.legs[leg]))
    log.info('  CANTILEVERED: mate preload %.1f N; force guard %.0f N / %.1f Nm for the whole '
             'run.', job.preload_mate['force_n'], job.guard.max_force, job.guard.max_torque)
    if not coupler.latched:
        log.warning('toolchanger.latch is off, and this run ENDS HOLDING THE TILE. Closing the '
                    'port at exit resets the coupler board, which re-decides the clamp from one '
                    'sensor reading and can open it. Set toolchanger.latch: true.')
    root = bt.sequence('disassemble-tile', *job.steps(step))
    try:
        return bt.run_tree(root, log)
    finally:
        robot.arm.servo_stop()
        job.teardown()
        # NOT switching the coupler motor off: the tile is still held.
        coupler.close()


def main():
    run_app('Disassemble tile: grasp a cantilevered tile, unscrew it, pull it off',
            'disassemble_tile', build_and_run, with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
