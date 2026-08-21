"""CALIBRATION CHECK -- probe the recorded target repeatedly and measure where contact happens.

Answers: is the recorded mate (frames.yaml targets: entry) still where the physical socket is,
and how repeatable is the approach?  Run it whenever a frame, fixture or grasp may have moved --
it is a two-minute test that catches calibration drift before it poisons an experiment.

Per cycle:
    standoff   the held connector backed off standoff_distance_m along its own -X from the
               recorded target (free space)
    approach   a SLOW straight-line advance along the connector's +X, under admittance, to
               slightly past the recorded target (overshoot_m) -- contact is guaranteed
    contact    the force guard trips; hold settle_s, then record the CONTACT POSE (held
               connector wrt the recorded target, mm / deg) and the settled wrench
    retract    compliant, un-guarded, straight back out.  For mates the robot cannot back out
               of on its own: `retract: false` ends the run AT the contact after one probe;
               `retract: prompt` waits for the operator to press the unlock button and confirm
               with ENTER before retracting, so multi-cycle checks still work.

After `cycles` repetitions: per-axis mean / std / min / max of the contact pose and the DRIFT
(slope of contact x vs cycle).  Contact x should sit near the physical face-to-face offset and
stay put; a trend means something is moving (fixture, grasp, or calibration).

TWO FRAMES, both from configs/frames.yaml: `held_frame` is the part in the fingers (a frames:
entry, tool0 -> part) and `target_frame` is the recorded mate it is probed against (a targets:
entry, base_link <- mate).  `target_frame` defaults to `held_frame`.  The reported contact pose
is always the held frame w.r.t. the target frame.

Output: data/experiments/calibration_check_<stamp>/contacts.csv + a printed summary.
Run:  python -m urlab.apps.calibration_check --config configs/calibration_check.yaml
"""

import csv as _csv
import os
import time

import numpy as np

from .. import behaviors as bt
from .. import log as urlog
from .. import tool_frames
from ..apps._common import experiment_dir, seg_time, tare_fn
from ..robot import AdmittanceController, ForceGuard
from ..transforms import inverse, matrix_to_xyzrpy, translation_matrix
from ._runner import run_app

log = urlog.get('calibration-check')

_CSV_HEADER = ['cycle', 'contact_x_mm', 'contact_y_mm', 'contact_z_mm',
               'contact_roll_deg', 'contact_pitch_deg', 'contact_yaw_deg',
               'force_n', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz',
               'tripped_by', 'reached_end', 'duration_s']


def line_rows(standoff_m, overshoot_m, resolution_m):
    """Straight-line waypoints along the connector's +X, RELATIVE TO THE TARGET frame: from
    -standoff (backed out) to +overshoot (just past the recorded mate). Pure geometry, so the
    smoke tests can check it without a robot."""
    n = max(int(np.ceil((standoff_m + overshoot_m) / max(resolution_m, 1e-5))), 2)
    xs = np.linspace(-abs(standoff_m), abs(overshoot_m), n + 1)
    return [translation_matrix([float(x), 0.0, 0.0]) for x in xs]


def _parse_retract(cfg):
    """'auto' | 'never' | 'prompt' from the config's `retract` key, or None on a bad value."""
    retract = cfg.get('retract', True)
    retract = (retract.strip().lower() if isinstance(retract, str)
               else ('auto' if retract else 'never'))
    if retract not in ('auto', 'never', 'prompt'):
        log.error("retract must be true, false or 'prompt' (got %r).", cfg.get('retract'))
        return None
    return retract


class _ProbeRun:
    """The shared state of one calibration-check run: the probe references, the compliant
    controller, the contact log, and the primitive actions each cycle's tree is built from."""

    def __init__(self, cfg, robot, refs, T_held, T_base_tconn, retract, out_dir):
        self.robot = robot
        self.refs = refs
        self.T_held = T_held
        self.T_base_tconn = T_base_tconn
        self.retract = retract
        self.adm = AdmittanceController(robot.arm, cfg.section('compliance'))
        self.guard = ForceGuard(robot.arm, cfg.section('force_guard'))
        self.tare = tare_fn(robot, cfg.section('compliance'))
        self.v_mm_s = float(cfg.get_path('speed.approach_translation_mm_s', 2.0))
        self.rv_mm_s = float(cfg.get_path('speed.retract_translation_mm_s', 20.0))
        self.settle_s = float(cfg.get_path('compliance.settle_s', 1.0))
        self.min_seg_s = 1.0 / self.adm.rate

        self.seed = {'q': robot.arm.q()}
        self.contacts = []
        self.stopped = False                # set when a prompt-retract loses its terminal
        self.ramp_result = {}               # last_ref / tripped / reached_end, per cycle
        self._t0 = 0.0

        self._fout = open(os.path.join(out_dir, 'contacts.csv'), 'w', newline='')
        self._writer = _csv.writer(self._fout)
        self._writer.writerow(_CSV_HEADER)

    def close(self):
        self._fout.close()

    def approach_time(self, A, B):
        return seg_time(A, B, self.v_mm_s, min_s=self.min_seg_s)

    def retract_time(self, A, B):
        return seg_time(A, B, self.rv_mm_s, min_s=self.min_seg_s)

    def start_cycle(self):
        self._t0 = time.time()

    def contact_pose(self):
        """Held connector wrt the recorded target, [x..yaw] in mm / deg, plus the wrench."""
        xyz, rpy = matrix_to_xyzrpy(inverse(self.T_base_tconn) @ self.robot.tool0()
                                    @ self.T_held)
        w = self.robot.arm.wrench()
        return list(xyz * 1000.0) + list(np.degrees(rpy)), list(w)

    def record_contact(self, cyc, cycles):
        pose6, w = self.contact_pose()
        fmag = float(np.linalg.norm(w[:3]))
        reached_end = self.ramp_result.get('reached_end', False)
        self.contacts.append(pose6)
        self._writer.writerow([cyc] + [f'{v:.3f}' for v in pose6] + [f'{fmag:.2f}']
                              + [f'{v:.2f}' for v in w]
                              + [self.guard.tripped_by or '', reached_end,
                                 f'{time.time() - self._t0:.1f}'])
        self._fout.flush()
        log.info('cycle %2d/%d: contact x %+7.2f mm  (y %+.2f, z %+.2f mm | r %+.2f, '
                 'p %+.2f, y %+.2f deg)  |f| %.1f N%s', cyc, cycles, pose6[0], pose6[1],
                 pose6[2], pose6[3], pose6[4], pose6[5], fmag,
                 '  [REACHED END -- no contact before the overshoot!]' if reached_end else '')

    def do_retract(self, cyc, cycles):
        """Back out to the standoff, honouring the configured retract mode."""
        last_ref = self.ramp_result.get('last_ref', self.refs[0])
        if self.retract == 'never':
            self.adm.stop()
            log.warning('retract: false -- the arm is LEFT AT THE CONTACT. Press the '
                        'unlock/release button before commanding any robot motion.')
            return True
        if self.retract == 'prompt':
            self.adm.stop()             # hold position while the human is at the connector
            try:
                input(f'   cycle {cyc}/{cycles}: press the unlock/release button, '
                      'then hit ENTER to retract... ')
            except EOFError:
                log.warning("retract: prompt needs an interactive terminal and stdin is "
                            "closed. Leaving the arm AT the contact; press the unlock/release "
                            "button before commanding any motion.")
                self.stopped = True     # skip the remaining cycles, arm stays put
                return True
            self.adm.reset()
            self.adm.warmup(last_ref)   # NO tare: the sensor is loaded at the contact
        self.adm.ramp(last_ref, self.refs[0], self.retract_time(last_ref, self.refs[0]),
                      guard=None)
        self.adm.stop()
        return True

    def cycle_tree(self, cyc, cycles):
        """One probe cycle as a behavior sequence; skipped wholesale once `stopped` is set."""
        run = self
        probe = bt.sequence(
            f'cycle {cyc}',
            bt.Action(f'cycle {cyc}: start', lambda: run.start_cycle()),
            bt.MoveToPose(run.robot, run.refs[0], f'cycle {cyc} standoff', run.seed),
            bt.Warmup(run.adm, run.refs[0], tare=run.tare),
            bt.AdmittanceRamp(run.adm, run.refs, run.approach_time,
                              f'cycle {cyc}: approach', guard=run.guard,
                              result=run.ramp_result),
            bt.Hold(run.adm, lambda: run.ramp_result['last_ref'], run.settle_s,
                    f'cycle {cyc}: settle at the stop'),
            bt.Action(f'cycle {cyc}: record contact',
                      lambda: run.record_contact(cyc, cycles)),
            bt.Action(f'cycle {cyc}: retract', lambda: run.do_retract(cyc, cycles)),
        )
        return bt.selector(f'cycle {cyc} (or stopped)',
                           bt.Check('stopped earlier', lambda: run.stopped), probe)


def _summarise(contacts):
    if len(contacts) < 2:
        return
    C = np.asarray(contacts, dtype=float)
    names = ['x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg']
    log.info('--- calibration summary over %d contacts ---', len(C))
    for j, nm in enumerate(names):
        log.info('   %-10s mean %+8.3f  std %6.3f  min %+8.3f  max %+8.3f', nm,
                 C[:, j].mean(), C[:, j].std(), C[:, j].min(), C[:, j].max())
    slope = float(np.polyfit(np.arange(len(C)), C[:, 0], 1)[0])
    log.info('   contact-x drift: %+.3f mm/cycle (%+.2f mm over the run). Repeatability '
             '(x std): %.3f mm.', slope, slope * (len(C) - 1), C[:, 0].std())
    if abs(slope) * (len(C) - 1) > 0.5:
        log.warning('Contact x DRIFTED more than 0.5 mm over the run -- fixture, grasp or '
                    'frame is moving.')


def build_and_run(cfg, robot, camera, args):
    held_name = cfg.get('held_frame')
    frames = tool_frames.load_frames(cfg)
    targets = tool_frames.load_targets(cfg)
    try:
        # held_frame = tool0 -> the part in the fingers; target_frame = base_link <- the
        # recorded mate it is probed against. target_frame null = held_frame.
        T_held, T_base_tconn, target_name = tool_frames.resolve_held_and_target(
            frames, targets, held_name, cfg.get('target_frame'), tool_frames.frames_path(cfg))
    except ValueError as exc:
        log.error('%s', exc)
        return False

    cycles = int(cfg.get('cycles', 10))
    standoff_m = float(cfg.get('standoff_distance_m', 0.03))
    overshoot_m = float(cfg.get('overshoot_m', 0.005))
    res_m = float(cfg.get('resolution_m', 0.001))
    retract = _parse_retract(cfg)
    if retract is None:
        return False
    if cycles < 1 or standoff_m <= 0 or overshoot_m < 0:
        log.error('cycles must be >= 1, standoff_distance_m > 0, overshoot_m >= 0.')
        return False
    if retract == 'never' and cycles > 1:
        log.warning('retract: false leaves the arm AT the contact, so only ONE probe can run '
                    '(%d cycles requested). Capping to 1.', cycles)
        cycles = 1

    refs = [T_base_tconn @ r @ inverse(T_held)
            for r in line_rows(standoff_m, overshoot_m, res_m)]
    out_dir = experiment_dir(cfg, 'calibration_check')
    run = _ProbeRun(cfg, robot, refs, T_held, T_base_tconn, retract, out_dir)

    log.info('Calibration check: holding %r, probing the recorded target %r%s.', held_name,
             target_name, '' if target_name == held_name else ' (NOT the held frame)')
    log.info('  %d cycles, %.0f mm standoff -> %.1f mm past the recorded target at %.1f mm/s, '
             'guard %.0f N (persistence %.2f s), retract %s.', cycles, standoff_m * 1000.0,
             overshoot_m * 1000.0, run.v_mm_s, run.guard.max_force, run.guard.persistence_s,
             retract)
    log.info('Output: %s', out_dir)

    root = bt.sequence(
        'calibration-check',
        bt.MoveToPose(robot, refs[0], 'calibration-check standoff', run.seed),
        *[run.cycle_tree(cyc, cycles) for cyc in range(1, cycles + 1)])
    try:
        ok = bt.run_tree(root, log)
    finally:
        run.close()

    _summarise(run.contacts)
    return ok and len(run.contacts) == cycles


def main():
    # with_gripper=False: the connector is fixtured between the closed fingers.
    run_app('Calibration check: repeated slow probes of the recorded target',
            'calibration_check', build_and_run, with_gripper=False)


if __name__ == '__main__':
    main()
