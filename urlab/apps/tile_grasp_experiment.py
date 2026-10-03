"""TILE GRASP EXPERIMENT -- pick and place one tile over and over, recording what compliance absorbs.

    python -m urlab.apps.tile_grasp_experiment --set experiment.trials=10
    python -m urlab.apps.tile_grasp_plot data/experiments/tile_grasp_experiment_<stamp>

Every attempt is apps/coupler_pick_place's cycle, with the tile put back where it was grasped --
or, with `experiment.randomize_place` (the default), at a random base_link x/y offset of up to
+-x_mm / +-y_mm from the FIRST grasp (a fixed anchor, so the tile cannot random-walk out of view),
at that grasp's attitude:

    view pose -> locate -> open coupler -> align -> MATE (compliant, preload) -> lock -> payload
    -> LIFT (compliant) -> operator: was the grasp + pickup successful?
         yes -> tare (tile hanging free) -> carry -> PLACE (compliant, preload) at the place
                target -> release -> payload back -> WITHDRAW (compliant)
         no  -> payload back -> retract -> wait for the operator (the coupler opens at the start
                of the next attempt, once they have dealt with the tile)

WHAT IS RECORDED: every servo cycle of the four compliant phases (mate, lift, place, withdraw):
the coupler pose MEASURED, the admittance REFERENCE and the pose COMMANDED, the law's YIELD
(adm.delta: how far compliance moved the arm off its reference -- what it is compensating), and
the WRENCH at the mating point in coupler axes (arm.wrench_in, so it is independent of where the
flange is). Poses are also stored RELATIVE TO THE GRASPED POSE -- the coupler pose measured at
lock -- so every trial passes through zero there, and the reference shows where the camera
thought the tile was. One .npz per attempt plus trials.csv; tile_grasp_plot draws every successful
trial with the mean and a +-1 sigma band.

SPEED: `experiment.speed_scale` multiplies configs/robot.yaml's speed limits for every move that
does not set its own -- the free-space moves, the return to the view pose, the camera's sweep and
servo -- and NOT the compliant legs, which keep their own pace (compliant_speed_mm_s): the phases
being measured do not change with it.

TARING (the cycle's own, the same in every attempt):
    mate      tared at the mate standoff -- coupler empty, nothing touching
    lift      NOT re-tared: the tile still rests on the bench, which carries its weight, and a zero
              taken against that becomes a phantom the moment it lifts (CouplerCycle.lift)
    after the operator confirms the pickup: tared with the tile hanging free
    place     tared at the place standoff -- tile hanging free, nothing touching
    withdraw  NOT re-tared: it starts pressed in by the place preload
A failed pickup is never tared (it would zero against an empty or mis-held coupler).
"""

import csv as _csv
import json
import os
import time

import numpy as np

from .. import behaviors as bt
from .. import log as urlog
from .. import tool_frames
from ..robot.coupler import Coupler
from ..skills import marker_localize as mloc
from ..transforms import inverse, matrix_to_xyzrpy, pose_error
from ._common import ask, experiment_dir, prompts_off
from ._runner import run_app
from .coupler_pick_place import CouplerCycle, offset_pose, parse_offset

log = urlog.get('tile-grasp')

# CouplerCycle's labels for its compliant legs -> the phase names the data and plots use.
PHASES = {'mate with the coupling feature': 'mate', 'retract with the object': 'lift',
          'place the object': 'place', 'withdraw from the object': 'withdraw'}
PHASE_ORDER = ('mate', 'lift', 'place', 'withdraw')

_TRIALS_HEADER = ['attempt', 'success', 'outcome', 'samples', 'camera_err_mm', 'camera_err_deg',
                  'camera_dx_mm', 'camera_dy_mm', 'camera_dz_mm', 'place_offset_x_mm',
                  'place_offset_y_mm', 'place_err_mm', 'place_err_deg',
                  'mate_peak_n', 'lift_peak_n', 'place_peak_n', 'withdraw_peak_n']


def random_place(T_anchor, rng, x_m, y_m):
    """(T_place, (dx, dy)): T_anchor moved by a uniform random base_link offset within +-x_m,
    +-y_m, at T_anchor's attitude. Pure given `rng`."""
    dx, dy = float(rng.uniform(-x_m, x_m)), float(rng.uniform(-y_m, y_m))
    T = np.array(T_anchor, dtype=float)
    T[:3, 3] = T[:3, 3] + np.array([dx, dy, 0.0])
    return T, (dx, dy)


def relative_pose6(T_ref_frame, Ts):
    """(N, 6): each pose in `Ts` relative to T_ref_frame, as [x y z mm, roll pitch yaw deg] in
    T_ref_frame's axes. NaN rows stay NaN. Pure."""
    inv = inverse(T_ref_frame)
    out = np.full((len(Ts), 6), np.nan)
    for i, T in enumerate(Ts):
        if np.all(np.isfinite(T)):
            xyz, rpy = matrix_to_xyzrpy(inv @ T)
            out[i] = np.concatenate([np.asarray(xyz) * 1000.0, np.degrees(rpy)])
    return out


class TrialRecorder:
    """Collects one attempt's servo-cycle samples and events, and writes them out.

    THE DEFAULTS ARE THE GRASP EXPERIMENT'S; apps/tile_assembly_experiment passes its own:
        phase_of(what) -> phase name     (the cycle's leg label -> 'mate', 'insert', ...)
        zero, zero_label                 which recorded pose the relative poses are zeroed on
        pose_names                       the discrete poses saved as T_<name>
        summarize(attempt, success, outcome, arrays, poses, recorder) -> row, and its header"""

    def __init__(self, robot, T_tool0_coupler, clock=time.monotonic, phase_of=None, zero='grasp',
                 zero_label='grasped', pose_names=('pick', 'grasp', 'place_target', 'placed'),
                 summarize=None, header=None):
        self.robot, self.Tc, self.clock = robot, np.asarray(T_tool0_coupler, dtype=float), clock
        self.phase_of = phase_of or (lambda what: PHASES.get(what, what))
        self.zero, self.zero_label, self.pose_names = zero, zero_label, tuple(pose_names)
        self.summarize = summarize or (lambda attempt, success, outcome, a, poses, rec:
                                       summary_row(attempt, success, outcome, a, poses,
                                                   rec.place_offset))
        self.header = list(header or _TRIALS_HEADER)
        self.start(0)

    def start(self, attempt):
        self.attempt, self.t0 = attempt, self.clock()
        self.rows, self.events, self.poses = [], [], {}
        self.extras = {}          # extra per-trial arrays, saved into the .npz as given
        self.place_offset = None
        self._phase = None

    def event(self, name, **extra):
        self.events.append({'t': self.clock() - self.t0, 'name': name, **extra})

    def pose(self, name, T):
        self.poses[name] = None if T is None else np.array(T, dtype=float)

    def on_servo_step(self, what, adm):
        """CouplerCycle.on_servo_step: one row per servo cycle of a compliant leg."""
        phase = self.phase_of(what)
        if phase != self._phase:
            self._phase = phase
            self.event(f'{phase} start')
        T_flange = self.robot.arm.tcp_pose()
        T_meas = T_flange @ self.Tc
        wrench = np.asarray(self.robot.arm.wrench_in(T_meas, T_flange), dtype=float)
        ref, cmd = adm.last_ref, adm.last_cmd
        nan4 = np.full((4, 4), np.nan)
        self.rows.append((self.clock() - self.t0, phase, T_meas,
                          ref @ self.Tc if ref is not None else nan4,
                          cmd @ self.Tc if cmd is not None else nan4,
                          np.asarray(adm.delta, dtype=float), wrench))

    def arrays(self):
        """The samples as arrays: t, phase, T_meas, T_ref, T_cmd, delta (N, 6), wrench (N, 6)."""
        if not self.rows:
            z = np.zeros((0, 4, 4))
            return {'t': np.zeros(0), 'phase': np.zeros(0, dtype='U16'), 'T_meas': z,
                    'T_ref': z, 'T_cmd': z, 'delta': np.zeros((0, 6)), 'wrench': np.zeros((0, 6))}
        t, ph, Tm, Tr, Tcm, d, w = zip(*self.rows)
        return {'t': np.asarray(t), 'phase': np.asarray(ph, dtype='U16'), 'T_meas': np.stack(Tm),
                'T_ref': np.stack(Tr), 'T_cmd': np.stack(Tcm), 'delta': np.stack(d),
                'wrench': np.stack(w)}

    def save(self, out_dir, success, outcome, meta):
        """attempt_NN.npz -- raw poses, the zero-relative ones, wrench, yield, events -- and a
        row of trials.csv. Returns the summary row."""
        a = self.arrays()
        T_zero = self.poses.get(self.zero)
        if T_zero is not None:
            a['pose_rel'] = relative_pose6(T_zero, a['T_meas'])
            a['ref_rel'] = relative_pose6(T_zero, a['T_ref'])
            a['cmd_rel'] = relative_pose6(T_zero, a['T_cmd'])
        else:
            a['pose_rel'] = a['ref_rel'] = a['cmd_rel'] = np.full((len(a['t']), 6), np.nan)
        # The law's yield in mm / deg (tool0 axes = coupler axes: coupler_mate has no rotation).
        a['delta_mm_deg'] = np.concatenate([a['delta'][:, :3] * 1000.0,
                                            np.degrees(a['delta'][:, 3:])], axis=1)
        for name in self.pose_names:
            T = self.poses.get(name)
            a[f'T_{name}'] = T if T is not None else np.full((4, 4), np.nan)
        a.update({k: np.asarray(v) for k, v in self.extras.items()})
        info = dict(meta, attempt=self.attempt, success=bool(success), outcome=outcome,
                    events=self.events, zero=self.zero_label)
        path = os.path.join(out_dir, f'attempt_{self.attempt:02d}.npz')
        np.savez_compressed(path, meta=json.dumps(info, default=float), **a)
        row = self.summarize(self.attempt, success, outcome, a, self.poses, self)
        csv_path = os.path.join(out_dir, 'trials.csv')
        new = not os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as fh:
            w = _csv.writer(fh)
            if new:
                w.writerow(self.header)
            w.writerow([row.get(k, '') for k in self.header])
        log.info('Attempt %d saved: %s (%d samples).', self.attempt, path, len(a['t']))
        return row


def summary_row(attempt, success, outcome, a, poses, place_offset=None):
    """One trials.csv row: the camera's error against the grasp (what the mate absorbed), the
    commanded place offset, the place error against its target, and the peak force of each
    phase. Pure."""
    row = {'attempt': attempt, 'success': int(bool(success)), 'outcome': outcome,
           'samples': len(a['t'])}
    if place_offset is not None:
        row.update(place_offset_x_mm=round(place_offset[0] * 1000.0, 3),
                   place_offset_y_mm=round(place_offset[1] * 1000.0, 3))
    T_pick, T_grasp, T_placed = poses.get('pick'), poses.get('grasp'), poses.get('placed')
    T_target = poses.get('place_target', T_grasp)
    if T_pick is not None and T_grasp is not None:
        lin, ang = pose_error(T_grasp, T_pick)
        d = (inverse(T_grasp) @ T_pick)[:3, 3] * 1000.0
        row.update(camera_err_mm=round(lin * 1000.0, 3), camera_err_deg=round(np.degrees(ang), 3),
                   camera_dx_mm=round(d[0], 3), camera_dy_mm=round(d[1], 3),
                   camera_dz_mm=round(d[2], 3))
    if T_placed is not None and T_target is not None:
        lin, ang = pose_error(T_target, T_placed)
        row.update(place_err_mm=round(lin * 1000.0, 3), place_err_deg=round(np.degrees(ang), 3))
    for phase in PHASE_ORDER:
        sel = a['phase'] == phase
        if np.any(sel):
            row[f'{phase}_peak_n'] = round(float(np.max(np.linalg.norm(a['wrench'][sel, :3],
                                                                      axis=1))), 3)
    return {k: row.get(k, '') for k in _TRIALS_HEADER}


class GraspTrialCycle(CouplerCycle):
    """The pick-and-place cycle, recording, with the place pose = the GRASPED pose."""

    RUN_NAME = 'tile_grasp_experiment'
    # Set by build_and_run: {'x_m', 'y_m'} to randomize the place, or None to put it back where it
    # was grasped; the generator; and the anchor, which is the FIRST grasp.
    place_random = None
    rng = None
    place_anchor = None

    def _target_pose(self, T_pick):
        # A placeholder until the grasp: lock() replaces it with the pose actually seated in.
        return T_pick

    def locate(self):
        ok = super().locate()
        if ok:
            self.recorder.pose('pick', self.T_pick)
        return ok

    def lock(self):
        """Lock, then take the GRASPED POSE -- the coupler pose measured now -- as the zero of the
        recorded poses, and choose the place target: the grasp itself, or (randomize_place) a
        random x/y offset from the FIRST grasp."""
        ok = super().lock()
        if ok:
            T_grasp = self.robot.arm.tcp_pose() @ self.T_tool0_coupler
            self.recorder.pose('grasp', T_grasp)
            self.recorder.event('grasp (locked)')
            if self.place_random:
                if self.place_anchor is None:
                    self.place_anchor = T_grasp
                self.T_place, offset = random_place(self.place_anchor, self.rng,
                                                    self.place_random['x_m'],
                                                    self.place_random['y_m'])
                self.recorder.place_offset = offset
                self.recorder.event('place target', dx_mm=offset[0] * 1000.0,
                                    dy_mm=offset[1] * 1000.0)
                log.info('Place target: %+.1f / %+.1f mm (base x / y) from the first grasp.',
                         offset[0] * 1000.0, offset[1] * 1000.0)
            else:
                self.T_place = T_grasp
            self.recorder.pose('place_target', self.T_place)
        return ok

    def settle_after_lift(self):
        ok = super().settle_after_lift()
        self.recorder.event('tare (tile hanging free)',
                            wrench=np.asarray(self.robot.arm.wrench(), float).tolist())
        return ok

    def record_placed(self):
        """The pose the tile was set down at, before letting go."""
        self.recorder.pose('placed', self.robot.arm.tcp_pose() @ self.T_tool0_coupler)
        return True

    def retract_failed_pick(self, leg):
        """After an unsuccessful pickup: back straight off along `leg` from where the arm is."""
        here = self.robot.arm.tcp_pose() @ self.T_tool0_coupler
        return self._move_to(offset_pose(here, leg), 'retract after the failed pickup')


def ask_pickup_ok(cfg, robot):
    """'y' = the grasp + pickup succeeded, 'n' = it did not, 'q' = stop the experiment. Unattended
    (dry run, --no-prompts) counts as success."""
    if robot.arm.dry_run or prompts_off(cfg):
        return 'y'
    while True:
        try:
            ans = input('  Was the GRASP + PICKUP successful? [y = yes, place it / n = no / '
                        'q = stop the experiment]: ').strip().lower()
        except EOFError:
            return 'y'
        if ans in ('y', 'yes', 'n', 'no', 'q', 'quit'):
            return ans[0]


def experiment_settings(cfg):
    """`experiment:` -> {trials, max_attempts, pause_between_trials, plot_at_end, failed_retract,
    speed_scale, randomize_place ({x_m, y_m, seed} or None), confirm_first_mate}.
    `trials` counts SUCCESSFUL pick-and-places; failed pickups are attempts, capped by
    max_attempts. Pure."""
    e = dict(cfg.section('experiment') or {})
    trials = int(e.get('trials', 10))
    if trials < 1:
        raise ValueError('experiment.trials must be at least 1')
    max_attempts = int(e.get('max_attempts') or 2 * trials)
    if max_attempts < trials:
        raise ValueError(f'experiment.max_attempts ({max_attempts}) is below trials ({trials})')
    speed_scale = float(e.get('speed_scale') or 1.0)
    if not speed_scale > 0.0:
        raise ValueError(f'experiment.speed_scale must be positive, got {speed_scale}')
    rp = dict(e.get('randomize_place') or {})
    randomize = None
    if bool(rp.get('enabled', True)):
        randomize = {'x_m': float(rp.get('x_mm', 50.0)) / 1000.0,
                     'y_m': float(rp.get('y_mm', 50.0)) / 1000.0,
                     'seed': None if rp.get('seed') is None else int(rp['seed'])}
        if randomize['x_m'] < 0.0 or randomize['y_m'] < 0.0:
            raise ValueError('experiment.randomize_place x_mm / y_mm must not be negative')
    return {'trials': trials, 'max_attempts': max_attempts, 'speed_scale': speed_scale,
            'confirm_first_mate': e.get('confirm_first_mate') is not False,   # null = on
            'randomize_place': randomize,
            'pause_between_trials': bool(e.get('pause_between_trials', False)),
            'plot_at_end': bool(e.get('plot_at_end', True)),
            'failed_retract': parse_offset(e.get('failed_retract'), 'experiment.failed_retract',
                                           100.0)}


def apply_speed_scale(robot, scale):
    """Scale robot.yaml's limits for every following move without its own caps; returns the scale
    it replaced, to restore afterwards. The compliant legs are paced separately and unaffected."""
    previous = float(getattr(robot.arm, 'speed_scale', 1.0))
    robot.arm.set_speed_scale(scale, 'tile_grasp_experiment')
    return previous


def run_attempt(cfg, robot, job, rec, attempt, first, q_view, step, failed_retract,
                confirm_first_mate=True):
    """One attempt. Returns 'success', 'failed_pickup', 'locate_failed', 'stopped' or 'error'.
    The first attempt starts at the view pose already; later ones drive back to it."""
    rec.start(attempt)
    if not first:
        if not robot.move_joints(q_view, label='back to the view pose', guard=job.guard):
            return 'error'
    if not bt.run_tree(bt.sequence(f'attempt {attempt}: locate',
                                   bt.Action('locate the tile', job.locate)), log):
        return 'locate_failed'
    pick = bt.sequence(
        f'attempt {attempt}: pick',
        bt.Action('power and open the coupler', job.prepare_coupler, confirm=step),
        bt.OperatorGate(robot, 'About to approach and MATE. Clear of the arm? (Enter / q): ',
                        label='before the first mate',
                        skip=not first or not confirm_first_mate or prompts_off(cfg)),
        bt.Action('align at the mate standoff', job.approach, confirm=step),
        bt.Action('mate with the tile', job.descend_and_mate, confirm=step),
        bt.Action('lock the coupler', job.lock, confirm=step),
        bt.Action('take the payload', job.take_payload),
        bt.Action('lift the tile', job.lift, confirm=step))
    if not bt.run_tree(pick, log):
        return 'error'

    answer = ask_pickup_ok(cfg, robot)
    rec.event(f'operator: pickup {"ok" if answer == "y" else "FAILED" if answer == "n" else "stop"}')
    if answer == 'q':
        return 'stopped'
    if answer == 'n':
        fail = bt.sequence(
            f'attempt {attempt}: failed pickup',
            bt.Action('payload back to the tool alone', job.drop_payload),
            bt.Action('retract', lambda: job.retract_failed_pick(failed_retract),
                      confirm=step))
        return 'failed_pickup' if bt.run_tree(fail, log) else 'error'

    place = bt.sequence(
        f'attempt {attempt}: place',
        bt.Action('tare with the tile hanging free', job.settle_after_lift),
        bt.Action('carry to the place standoff', job.carry, confirm=step),
        bt.Action('place the tile at the grasped pose', job.set_down, confirm=step),
        bt.Action('record the placed pose', job.record_placed),
        bt.Action('release the coupler', job.unlock, confirm=step),
        bt.Action('payload back to the tool alone', job.drop_payload),
        bt.Action('withdraw', job.withdraw, confirm=step))
    return 'success' if bt.run_tree(place, log) else 'error'


def build_and_run(cfg, robot, camera, args):
    try:
        settings = experiment_settings(cfg)
        catalogue = tool_frames.load_objects(cfg)
        plan = mloc.ViewPlan(cfg.section('marker_views'))
    except ValueError as exc:
        log.error('%s', exc)
        return False
    name = cfg.get('object_name')
    if name not in catalogue:
        log.error('object_name %r is not in %s. Catalogued: %s.', name,
                  tool_frames.objects_path(cfg), ', '.join(sorted(catalogue)) or '(none)')
        return False
    obj = catalogue[name]

    from ..perception import ArucoDetector
    from .coupler_pick_place import step_gate
    detector = ArucoDetector(cfg, sizes_m={int(m): v['size_m'] for m, v in obj['markers'].items()})
    coupler = Coupler(cfg)
    try:
        job = GraspTrialCycle(cfg, robot, camera, detector, plan, name, obj, coupler)
    except (ValueError, KeyError) as exc:
        log.error('%s', exc)
        coupler.close()
        return False
    run_dir = job.out_dir or experiment_dir(cfg, GraspTrialCycle.RUN_NAME)
    job.out_dir = run_dir
    rec = TrialRecorder(robot, job.T_tool0_coupler)
    job.recorder, job.on_servo_step = rec, rec.on_servo_step
    rand = settings['randomize_place']
    job.place_random = rand
    job.rng = np.random.default_rng(rand['seed'] if rand else None)
    q_view = np.asarray(robot.arm.q(), dtype=float)
    step = step_gate(cfg, robot)
    meta = {'object': name, 'start_joints_deg': np.degrees(q_view).tolist(),
            'speed_scale': settings['speed_scale'],
            'randomize_place': ({'x_mm': rand['x_m'] * 1000.0, 'y_mm': rand['y_m'] * 1000.0,
                                 'seed': rand['seed'], 'anchor': 'the first grasp, base_link x/y'}
                                if rand else None),
            'compliant_speed_mm_s': cfg.get('compliant_speed_mm_s'),
            'compliance': {k: cfg.section(k) for k in ('compliance', 'compliance_loaded',
                                                       'compliance_insert') if cfg.section(k)},
            'mate_preload': cfg.section('mate_preload'),
            'place_preload': cfg.section('place_preload'),
            'tare': {'mate': 'mate standoff, coupler empty',
                     'lift': 'none -- the mate standoff zero; bench carries the tile',
                     'place': 'after the operator-confirmed pickup, tile hanging free; again at '
                              'the place standoff',
                     'withdraw': 'none -- the place standoff zero'},
            'frames': {'pose_rel': 'coupler pose in the GRASPED pose frame [mm, deg]',
                       'wrench': 'at the mating point, coupler axes [N, Nm]',
                       'delta_mm_deg': 'admittance yield, tool0 = coupler axes [mm, deg]'}}
    log.info('TILE GRASP EXPERIMENT: %d successful trial(s) of %r, at most %d attempts. Data -> '
             '%s', settings['trials'], name, settings['max_attempts'], run_dir)

    done = attempts = 0
    ok_run = True
    previous_scale = apply_speed_scale(robot, settings['speed_scale'])
    try:
        while done < settings['trials'] and attempts < settings['max_attempts']:
            attempts += 1
            if job.images is not None:
                # One image folder per attempt -- the writer names views by index, so a shared
                # folder would be overwritten every time.
                job.images = mloc.MarkerImageWriter(run_dir, detector,
                                                    subdir=f'marker_images/attempt_{attempts:02d}')
            log.info('==== ATTEMPT %d (%d of %d successful so far) ====', attempts, done,
                     settings['trials'])
            outcome = run_attempt(cfg, robot, job, rec, attempts, attempts == 1, q_view, step,
                                  settings['failed_retract'], settings['confirm_first_mate'])
            robot.arm.servo_stop()
            rec.save(run_dir, outcome == 'success', outcome, meta)
            if outcome == 'success':
                done += 1
                if (settings['pause_between_trials'] and done < settings['trials']
                        and not robot.arm.dry_run and not prompts_off(cfg)
                        and not ask('  Trial done. Enter for the next one (q to stop): ')):
                    break
                continue
            if outcome == 'failed_pickup':
                if robot.arm.dry_run or prompts_off(cfg):
                    continue
                if not ask('  Pickup marked UNSUCCESSFUL and the arm has retracted. Deal with the '
                           'tile (remove it from the coupler if it is hanging -- the coupler '
                           'OPENS at the start of the next attempt), then Enter to continue the '
                           'experiment (q to stop): '):
                    break
                continue
            if outcome == 'locate_failed':
                log.error('Attempt %d: the tile was not located; nothing moved toward it.',
                          attempts)
                if (robot.arm.dry_run or prompts_off(cfg)
                        or not ask('  Enter to retry (q to stop): ')):
                    ok_run = False
                    break
                continue
            # 'stopped' by the operator, or a step failed: stop here, with whatever is held still
            # held -- a failed step means the state is not one to carry on from automatically.
            ok_run = outcome == 'stopped'
            log.error('Attempt %d ended with %r -- stopping the experiment.', attempts, outcome)
            break
        log.info('EXPERIMENT DONE: %d successful trial(s) in %d attempt(s). Data: %s', done,
                 attempts, run_dir)
        if done and ok_run:
            job.park_coupler()
            robot.move_joints(q_view, label='back to the view pose', guard=job.guard)
        if settings['plot_at_end'] and done:
            try:
                from .tile_grasp_plot import plot_run
                for path in plot_run(run_dir):
                    log.info('Plot: %s', path)
            except Exception as exc:                # noqa: BLE001 -- plots never fail a run
                log.warning('Could not plot the run (%s). Plot it later with: python -m '
                            'urlab.apps.tile_grasp_plot %s', exc, run_dir)
        return ok_run and done >= settings['trials']
    finally:
        robot.arm.servo_stop()
        robot.arm.set_speed_scale(previous_scale, 'restored')
        coupler.close()


def main():
    # with_gripper=False: the coupler does the holding.
    run_app('Tile grasp experiment: repeated pick and place, recording pose, wrench and compliance',
            'tile_grasp_experiment', build_and_run, with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
