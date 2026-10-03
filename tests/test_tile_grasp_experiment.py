"""TILE GRASP EXPERIMENT -- the recording, the outcome branches and the plots, without a robot.

What would spoil an experiment silently: poses not zeroed on the grasp, a tare taken against an
empty or mis-held coupler, a failed pickup that still gets "placed", and trials of different
lengths averaged as if they lined up.
"""

import csv
import json
import os
import tempfile

import numpy as np
import pytest

from urlab import config as C
from urlab.apps import tile_grasp_experiment as tg
from urlab.apps import tile_grasp_plot as tp
from urlab.apps.coupler_pick_place import CouplerCycle
from urlab.transforms import xyzrpy_to_matrix


def _pose(xyz_mm, rpy_deg):
    return xyzrpy_to_matrix(np.array(xyz_mm, float) / 1000.0, np.radians(rpy_deg))


T_GRASP = _pose([600, -100, 50], [180, 0, 30])


# ---------------------------------------------------------------------------- frames
def test_poses_are_zero_at_the_grasp_and_read_in_its_axes():
    rel = tg.relative_pose6(T_GRASP, [T_GRASP, T_GRASP @ _pose([0, 0, -5], [0, 0, 2]),
                                      np.full((4, 4), np.nan)])
    assert np.allclose(rel[0], 0.0)
    assert np.allclose(rel[1], [0, 0, -5, 0, 0, 2], atol=1e-9), 'in the GRASP frame, mm / deg'
    assert np.all(np.isnan(rel[2]))


# ---------------------------------------------------------------------------- the recorder
class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 0.008
        return self.t


class _Arm:
    dry_run = False

    def __init__(self):
        self.T = np.array(T_GRASP)

    def tcp_pose(self):
        return self.T

    def wrench_in(self, T, T_flange=None):
        return np.array([0.1, 0.2, -3.0, 0.0, 0.01, 0.0])

    def wrench(self):
        return np.zeros(6)


class _Adm:
    def __init__(self):
        self.last_ref = np.array(T_GRASP)
        self.last_cmd = np.array(T_GRASP)
        self.delta = np.array([0.001, 0.0, -0.002, 0.0, 0.0, np.radians(1.0)])


def _recorder():
    robot = type('R', (), {'arm': _Arm()})()
    return tg.TrialRecorder(robot, np.eye(4), clock=_Clock()), robot


def test_every_servo_step_is_a_row_and_phases_are_named():
    rec, _robot = _recorder()
    rec.start(3)
    adm = _Adm()
    for what in ['mate with the coupling feature'] * 3 + ['retract with the object'] * 2:
        rec.on_servo_step(what, adm)
    a = rec.arrays()
    assert list(a['phase']) == ['mate'] * 3 + ['lift'] * 2
    assert a['T_meas'].shape == (5, 4, 4) and a['wrench'].shape == (5, 6)
    assert [e['name'] for e in rec.events] == ['mate start', 'lift start']
    assert np.all(np.diff(a['t']) > 0)


def test_an_attempt_is_saved_with_grasp_relative_poses_and_a_summary_row():
    rec, _robot = _recorder()
    out = tempfile.mkdtemp()
    rec.start(1)
    rec.pose('pick', T_GRASP @ _pose([1.5, -0.5, 2.0], [0, 0, 0]))   # the camera was 2.6 mm off
    adm = _Adm()
    for _ in range(4):
        rec.on_servo_step('mate with the coupling feature', adm)
    rec.pose('grasp', T_GRASP)
    rec.pose('placed', T_GRASP @ _pose([0.3, 0, 0], [0, 0, 0]))
    row = rec.save(out, True, 'success', {'object': 'tile_6'})
    with np.load(os.path.join(out, 'attempt_01.npz')) as z:
        assert np.allclose(z['pose_rel'], 0.0), 'the arm sat AT the grasp in this fake'
        assert np.allclose(z['delta_mm_deg'][0], [1.0, 0.0, -2.0, 0.0, 0.0, 1.0])
        meta = json.loads(str(z['meta']))
    assert meta['success'] and meta['attempt'] == 1 and meta['object'] == 'tile_6'
    assert row['camera_err_mm'] == pytest.approx(np.linalg.norm([1.5, -0.5, 2.0]), abs=1e-3)
    assert row['camera_dz_mm'] == pytest.approx(2.0, abs=1e-6)
    assert row['place_err_mm'] == pytest.approx(0.3, abs=1e-6)
    assert row['mate_peak_n'] == pytest.approx(np.linalg.norm([0.1, 0.2, -3.0]), abs=1e-3)
    rows = list(csv.DictReader(open(os.path.join(out, 'trials.csv'))))
    assert len(rows) == 1 and rows[0]['outcome'] == 'success'


def test_an_attempt_that_never_grasped_saves_without_a_zero():
    rec, _robot = _recorder()
    out = tempfile.mkdtemp()
    rec.start(2)
    rec.on_servo_step('mate with the coupling feature', _Adm())
    rec.save(out, False, 'error', {})
    with np.load(os.path.join(out, 'attempt_02.npz')) as z:
        assert np.all(np.isnan(z['pose_rel'])) and np.all(np.isnan(z['T_grasp']))


# ---------------------------------------------------------------------------- the outcome branches
class _Job:
    """Records which cycle steps ran, in order."""

    def __init__(self):
        self.calls = []
        self.guard = None
        for name in ('locate', 'prepare_coupler', 'approach', 'descend_and_mate', 'lock',
                     'take_payload', 'lift', 'settle_after_lift', 'carry', 'set_down',
                     'record_placed', 'unlock', 'drop_payload', 'withdraw'):
            setattr(self, name, self._step(name))

    def _step(self, name):
        return lambda: self.calls.append(name) or True

    def retract_failed_pick(self, leg):
        self.calls.append(f'retract {leg["distance_m"] * 1000:.0f}')
        return True


def _attempt(monkeypatch, answer, first=True):
    job = _Job()
    robot = type('R', (), {'arm': type('A', (), {'dry_run': False})(),
                           'moves': [], 'move_joints': None})()
    robot.move_joints = lambda q, label='', guard=None: robot.moves.append(label) or True
    rec, _r = _recorder()
    cfg = C.load('tile_grasp_experiment', ['skip_prompts=true'])
    monkeypatch.setattr(tg, 'ask_pickup_ok', lambda cfg, robot: answer)
    leg = {'distance_m': 0.1, 'axis': np.array([0.0, 0.0, -1.0]), 'frame': 'coupler'}
    outcome = tg.run_attempt(cfg, robot, job, rec, 2, first, np.zeros(6), None, leg)
    return outcome, job.calls, robot.moves, rec


def test_a_successful_pickup_is_tared_free_then_placed_at_the_grasp(monkeypatch):
    outcome, calls, moves, rec = _attempt(monkeypatch, 'y')
    assert outcome == 'success'
    assert calls == ['locate', 'prepare_coupler', 'approach', 'descend_and_mate', 'lock',
                     'take_payload', 'lift', 'settle_after_lift', 'carry', 'set_down',
                     'record_placed', 'unlock', 'drop_payload', 'withdraw']
    assert moves == [], 'the first attempt starts at the view pose already'
    assert any(e['name'] == 'operator: pickup ok' for e in rec.events)


def test_a_failed_pickup_is_never_tared_or_placed_and_retracts(monkeypatch):
    outcome, calls, _moves, rec = _attempt(monkeypatch, 'n')
    assert outcome == 'failed_pickup'
    assert calls[-2:] == ['drop_payload', 'retract 100']
    assert 'settle_after_lift' not in calls, 'a tare against a failed pickup would be wrong'
    assert not {'carry', 'set_down', 'unlock'} & set(calls)
    assert any(e['name'] == 'operator: pickup FAILED' for e in rec.events)


def test_stop_and_later_attempts(monkeypatch):
    outcome, calls, _m, _r = _attempt(monkeypatch, 'q')
    assert outcome == 'stopped' and calls[-1] == 'lift'
    _o, _c, moves, _r = _attempt(monkeypatch, 'y', first=False)
    assert moves == ['back to the view pose'], 'later attempts drive back to the view first'


def test_the_settings_count_successes_and_cap_attempts():
    s = tg.experiment_settings(C.load('tile_grasp_experiment', ['experiment.trials=5',
                                                                'experiment.max_attempts=null']))
    assert s['trials'] == 5 and s['max_attempts'] == 10, 'null = twice the trials'
    with pytest.raises(ValueError, match='trials'):
        tg.experiment_settings(C.load('tile_grasp_experiment', ['experiment.trials=0']))
    with pytest.raises(ValueError, match='below trials'):
        tg.experiment_settings(C.load('tile_grasp_experiment', ['experiment.trials=5',
                                                                'experiment.max_attempts=3']))


# ---------------------------------------------------------------------------- the hook
def test_the_cycle_calls_the_recorder_on_every_compliant_servo_step():
    """CouplerCycle._compliant -> watch() -> on_servo_step(what, adm), every cycle."""
    assert CouplerCycle.on_servo_step is None, 'off unless someone records'
    job = CouplerCycle.__new__(CouplerCycle)
    job.cfg = C.load('tile_grasp_experiment')
    job.T_tool0_coupler, job.tare_before, job.holding, job.settle_s = np.eye(4), True, False, 0.0
    job._active_law = None
    seen = []
    job.on_servo_step = lambda what, adm: seen.append(what)

    class Adm:
        S = np.ones(6)
        delta = np.zeros(6)

        def reset(self):
            pass

        def warmup(self, T, tare_fn=None):
            pass

        def ramp(self, T0, T1, duration, guard=None, on_step=None):
            for _ in range(5):
                on_step()
            return 'done'

        def hold(self, T, seconds, guard=None, on_step=None):
            on_step()
            return 'done'

    job.adm = job.adm_loaded = job.adm_insert = Adm()
    job.guard = type('G', (), {'max_force': 60.0, 'max_torque': 8.0, 'reset': lambda s: None})()
    job.robot = type('R', (), {'arm': type('A', (), {'wrench': lambda s: np.zeros(6),
                                                      'servo_stop': lambda s: None,
                                                      'zero_ft': lambda s, settle=True: None})()})()
    assert job._compliant(np.eye(4), _pose([0, 0, -10], [0, 0, 0]), 'mate with the coupling '
                          'feature')
    assert seen == ['mate with the coupling feature'] * 6       # 5 ramp cycles + 1 hold


# ---------------------------------------------------------------------------- the plots
def _trial(attempt, n, offset, success=True):
    t = np.arange(n) * 0.008
    pose = np.zeros((n, 6))
    pose[:, 2] = -10.0 + 12.5 * t + offset                      # the same ramp, shifted
    return {'t': t, 'phase': np.array(['mate'] * n), 'pose_rel': pose, 'ref_rel': pose * 0,
            'wrench': np.zeros((n, 6)), 'delta_mm_deg': np.zeros((n, 6)), 'attempt': attempt,
            'success': success}


def test_the_band_lines_trials_up_and_needs_two_for_a_sigma():
    a, b = _trial(1, 101, 0.0), _trial(2, 51, 1.0)               # 0.8 s and 0.4 s long
    grid, stack, mean, std, count = tp.band([a, b], 'mate', 'pose_rel', 'start', dt=0.008)
    assert stack.shape[0] == 2 and grid[0] == pytest.approx(0.0)
    both = count == 2
    assert np.any(both) and np.any(count == 1)
    assert np.all(np.isfinite(std[both, 2])) and np.all(np.isnan(std[count == 1, 2]))
    assert np.allclose(mean[both, 2] - stack[0, both, 2], 0.5, atol=0.06)
    # aligned on the END, the two line up at t = 0 instead
    g_end, *_rest, c_end = tp.band([a, b], 'mate', 'pose_rel', 'end', dt=0.008)
    assert g_end[-1] == pytest.approx(0.0, abs=1e-9) and c_end[-1] == 2


def test_plotting_a_run_writes_the_three_figures_and_the_bands(monkeypatch):
    out = tempfile.mkdtemp()
    rec, robot = _recorder()
    for k, dz in ((1, 0.0), (2, 0.4), (3, -0.3)):
        rec.start(k)
        adm = _Adm()
        for i in range(30):
            robot.arm.T = T_GRASP @ _pose([0, 0, -3 + 0.1 * i + dz], [0, 0, 0])
            rec.on_servo_step('mate with the coupling feature', adm)
        for i in range(20):
            rec.on_servo_step('place the object', adm)
        rec.pose('grasp', T_GRASP)
        rec.save(out, True, 'success', {})
    rec.start(4)
    rec.on_servo_step('mate with the coupling feature', _Adm())
    rec.save(out, False, 'failed_pickup', {})
    assert len(tp.load_trials(out)) == 3 and len(tp.load_trials(out, include_failed=True)) == 4
    paths = tp.plot_run(out)
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ['bands.npz', 'compliance.png', 'pose.png', 'wrench.png']
    with np.load(os.path.join(out, 'plots', 'bands.npz')) as z:
        assert 'pose.mate.mean' in z.files and 'wrench.place.std' in z.files
        assert int(z['pose.mate.count'].max()) == 3


# ---------------------------------------------------------------------------- speed
def test_the_speed_scale_is_read_validated_and_applied_with_a_restore():
    assert tg.experiment_settings(C.load('tile_grasp_experiment',
                                         ['experiment.speed_scale=1.5']))['speed_scale'] == 1.5
    assert tg.experiment_settings(C.load('tile_grasp_experiment',
                                         ['experiment.speed_scale=null']))['speed_scale'] == 1.0
    with pytest.raises(ValueError, match='speed_scale'):
        tg.experiment_settings(C.load('tile_grasp_experiment', ['experiment.speed_scale=-1']))

    class Arm:
        speed_scale = 0.8

        def set_speed_scale(self, scale, phase=''):
            self.speed_scale = scale
    robot = type('R', (), {'arm': Arm()})()
    assert tg.apply_speed_scale(robot, 1.5) == 0.8 and robot.arm.speed_scale == 1.5



# ---------------------------------------------------------------------------- randomized place
def test_a_random_place_stays_within_bounds_at_the_anchors_attitude():
    rng = np.random.default_rng(0)
    offsets = []
    for _ in range(200):
        T, (dx, dy) = tg.random_place(T_GRASP, rng, 0.05, 0.05)
        offsets.append((dx, dy))
        assert np.allclose(T[:3, :3], T_GRASP[:3, :3]) and T[2, 3] == pytest.approx(T_GRASP[2, 3])
        assert np.allclose(T[:2, 3] - T_GRASP[:2, 3], [dx, dy])
    o = np.array(offsets)
    assert np.all(np.abs(o) <= 0.05) and o.std(axis=0).min() > 0.02, 'spread over the box'
    T_a, _ = tg.random_place(T_GRASP, np.random.default_rng(7), 0.05, 0.05)
    T_b, _ = tg.random_place(T_GRASP, np.random.default_rng(7), 0.05, 0.05)
    assert np.allclose(T_a, T_b), 'a seed repeats the sequence'


class _Coupler:
    def hold(self):
        return True

    def verify(self):
        return True


def _locker(randomize):
    job = tg.GraspTrialCycle.__new__(tg.GraspTrialCycle)
    job.coupler, job.T_tool0_coupler = _Coupler(), np.eye(4)
    job.robot = type('R', (), {'arm': _Arm()})()
    job.recorder = tg.TrialRecorder(job.robot, np.eye(4), clock=_Clock())
    job.place_random = randomize
    job.rng = np.random.default_rng(3)
    job.place_anchor = None
    return job


def test_place_offsets_are_from_the_first_grasp_not_the_last():
    job = _locker({'x_m': 0.05, 'y_m': 0.05, 'seed': 3})
    assert job.lock()
    first = np.array(job.place_anchor)
    assert np.allclose(first, T_GRASP)
    job.robot.arm.T = T_GRASP @ _pose([30, -20, 0], [0, 0, 0])     # the tile was picked elsewhere
    job.recorder.start(2)
    assert job.lock()
    assert np.allclose(job.place_anchor, first), 'the anchor does not move'
    dx, dy = job.recorder.place_offset
    assert np.allclose(job.T_place[:2, 3], first[:2, 3] + [dx, dy])
    assert np.allclose(job.recorder.poses['grasp'], job.robot.arm.T), 'zero is still THIS grasp'


def test_without_randomizing_the_tile_goes_back_where_it_was_grasped():
    job = _locker(None)
    assert job.lock()
    assert np.allclose(job.T_place, T_GRASP) and job.recorder.place_offset is None


def test_the_place_error_is_measured_against_the_place_target():
    a = {'t': np.zeros(1), 'phase': np.array(['place']), 'wrench': np.zeros((1, 6))}
    target = T_GRASP @ _pose([40, 10, 0], [0, 0, 0])
    poses = {'pick': T_GRASP, 'grasp': T_GRASP, 'place_target': target,
             'placed': target @ _pose([0.5, 0, 0], [0, 0, 0])}
    row = tg.summary_row(1, True, 'success', a, poses, place_offset=(0.04, 0.01))
    assert row['place_err_mm'] == pytest.approx(0.5, abs=1e-6)
    assert row['place_offset_x_mm'] == pytest.approx(40.0) and row['place_offset_y_mm'] == 10.0


def test_randomize_place_is_on_by_default_and_validated():
    assert tg.experiment_settings(C.load('tile_grasp_experiment', ['experiment.randomize_place=null'])
                                  )['randomize_place'] == {'x_m': 0.05, 'y_m': 0.05, 'seed': None}
    off = C.load('tile_grasp_experiment', ['experiment.randomize_place.enabled=false'])
    assert tg.experiment_settings(off)['randomize_place'] is None
    with pytest.raises(ValueError, match='negative'):
        tg.experiment_settings(C.load('tile_grasp_experiment',
                                      ['experiment.randomize_place.x_mm=-1']))


# ---------------------------------------------------------------------------- prompts
def test_with_the_gates_off_the_only_question_is_the_pickup_check(monkeypatch):
    asked = []
    monkeypatch.setattr('builtins.input', lambda prompt='': asked.append(prompt) or 'y')
    job = _Job()
    robot = type('R', (), {'arm': type('A', (), {'dry_run': False})()})()
    robot.move_joints = lambda q, label='', guard=None: True
    rec, _r = _recorder()
    cfg = C.load('tile_grasp_experiment', ['skip_prompts=false', 'confirm_each_step=false'])
    leg = {'distance_m': 0.1, 'axis': np.array([0.0, 0.0, -1.0]), 'frame': 'coupler'}
    assert tg.run_attempt(cfg, robot, job, rec, 1, True, np.zeros(6), None, leg,
                          confirm_first_mate=False) == 'success'
    assert len(asked) == 1 and 'PICKUP' in asked[0]
    assert tg.experiment_settings(C.load('tile_grasp_experiment',
                                         ['experiment.confirm_first_mate=null'])
                                  )['confirm_first_mate'] is True, 'on unless switched off'

