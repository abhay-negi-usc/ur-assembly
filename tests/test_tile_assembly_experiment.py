"""TILE ASSEMBLY EXPERIMENT -- the hand-loaded trial, its recording and its plots, without a robot.

The failures worth catching: poses not zeroed on the seat, a trial that skips the operator's
"let go" before anything moves, a refused lock that is treated as loaded, a fasten hold that
goes unrecorded, and an inspection that mixes the two parts' markers, reports the error in the
wrong frame, or fails the trial when it cannot see.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from urlab import config as C
from urlab.apps import tile_assembly_experiment as ta
from urlab.apps import tile_grasp_plot as tp
from urlab.apps.tile_grasp_experiment import TrialRecorder
from urlab.transforms import xyzrpy_to_matrix


def _pose(xyz_mm, rpy_deg):
    return xyzrpy_to_matrix(np.array(xyz_mm, float) / 1000.0, np.radians(rpy_deg))


T_SEAT = _pose([-1050, 360, 330], [180, 0, -3])


def test_the_legs_map_to_insert_fasten_withdraw():
    assert ta.assembly_phase('assemble the object') == 'insert'
    assert ta.assembly_phase('fasten (screwdrive_predrive_to_bolt)') == 'fasten'
    assert ta.assembly_phase('withdraw from the object') == 'withdraw'


def test_the_summary_is_the_goals_error_against_the_seat():
    a = {'t': np.zeros(2), 'phase': np.array(['insert', 'fasten']),
         'wrench': np.array([[0, 0, -3.0, 0, 0, 0], [0, 4.0, 0, 0, 0, 0]])}
    goal = T_SEAT @ _pose([1.0, -2.0, 0.5], [0, 0, 0])
    row = ta.assembly_summary(1, True, 'success', a, {'goal': goal, 'seated': T_SEAT})
    assert row['goal_err_mm'] == pytest.approx(np.linalg.norm([1.0, -2.0, 0.5]), abs=1e-3)
    assert (row['goal_dx_mm'], row['goal_dy_mm']) == (pytest.approx(1.0), pytest.approx(-2.0))
    assert row['insert_peak_n'] == 3.0 and row['fasten_peak_n'] == 4.0


def test_the_load_pose_defaults_to_the_pick_view():
    cfg = C.load('tile_assembly_experiment', ['experiment.load_joints_deg=null',
                                              'pick_view_joints_deg=[1, 2, 3, 4, 5, 6]'])
    assert np.allclose(np.degrees(ta.experiment_settings(cfg)['load_joints']), [1, 2, 3, 4, 5, 6])
    own = C.load('tile_assembly_experiment', ['experiment.load_joints_deg=[9, 8, 7, 6, 5, 4]'])
    assert np.allclose(np.degrees(ta.experiment_settings(own)['load_joints']), [9, 8, 7, 6, 5, 4])
    with pytest.raises(ValueError, match='six'):
        ta.experiment_settings(C.load('tile_assembly_experiment',
                                      ['experiment.load_joints_deg=[1, 2]']))
    with pytest.raises(ValueError, match='trials'):
        ta.experiment_settings(C.load('tile_assembly_experiment', ['experiment.trials=0']))


# ---------------------------------------------------------------------------- the trial
class _Job:
    """Records which cycle steps ran, in order."""

    secures_target = False

    def __init__(self):
        self.calls, self.guard = [], None
        for name in ('locate_goal', 'load_tile', 'tare_free', 'carry', 'set_down',
                     'record_seated', 'unlock', 'record_released', 'drop_payload', 'withdraw',
                     'inspect'):
            setattr(self, name, self._step(name))

    def _step(self, name):
        return lambda: self.calls.append(name) or True

    def steps_after_lift(self, confirm):
        from urlab import behaviors as bt
        return [bt.Action('predrive', self._step('predrive'))]

    def steps_after_insertion(self, confirm):
        from urlab import behaviors as bt
        return [bt.Action('fasten', self._step('fasten'))]


def _robot(dry=True):
    robot = type('R', (), {'arm': type('A', (), {'dry_run': dry})(), 'moves': []})()
    robot.move_joints = lambda q, label='', guard=None, caps=None: robot.moves.append(label) or True
    return robot


def test_a_trial_locates_loads_predrives_tares_inserts_fastens_releases_inspects_and_goes_home():
    job, robot = _Job(), _robot()
    rec = TrialRecorder(robot, np.eye(4))
    cfg = C.load('tile_assembly_experiment')
    assert ta.run_trial(cfg, robot, job, rec, 1, np.zeros(6), np.ones(6), None) == 'done'
    assert job.calls == ['locate_goal', 'load_tile', 'predrive', 'tare_free', 'carry', 'set_down',
                         'record_seated', 'fasten', 'unlock', 'record_released', 'drop_payload',
                         'withdraw', 'inspect']
    assert robot.moves == ['load pose', 'home']


def test_a_fixture_not_found_never_moves_toward_it():
    job, robot = _Job(), _robot()
    job.locate_goal = lambda: False
    rec = TrialRecorder(robot, np.eye(4))
    assert ta.run_trial(C.load('tile_assembly_experiment'), robot, job, rec, 1, np.zeros(6),
                        np.ones(6), None) == 'locate_failed'
    assert robot.moves == [] and job.calls == []


# ---------------------------------------------------------------------------- the load
class _LoadJob(ta.AssemblyTrialCycle):
    def __init__(self, locks, answers, dry=False):
        self.cfg = C.load('tile_assembly_experiment', ['skip_prompts=false'])
        self.robot = _robot(dry)
        self.recorder = TrialRecorder(self.robot, np.eye(4))
        self._locks, self.calls = iter(locks), []
        self._answers = iter(answers)

    def prepare_coupler(self):
        self.calls.append('open')
        return True

    def lock(self):
        self.calls.append('lock')
        return next(self._locks)

    def take_payload(self):
        self.calls.append('payload')
        return True


def _load(monkeypatch, locks, answers):
    job = _LoadJob(locks, answers)
    asked = []
    monkeypatch.setattr(ta, 'ask', lambda prompt: asked.append(prompt) or next(job._answers))
    return job, job.load_tile(), asked


def test_the_load_waits_for_the_tile_then_for_hands_off(monkeypatch):
    job, ok, asked = _load(monkeypatch, [True], [True, True])
    assert ok and job.calls == ['open', 'lock', 'payload']
    assert 'Place the TILE' in asked[0] and 'LET GO' in asked[1], 'nothing moves before "let go"'


def test_a_refused_lock_opens_and_asks_again(monkeypatch):
    job, ok, asked = _load(monkeypatch, [False, True], [True, True, True, True])
    assert ok and job.calls == ['open', 'lock', 'open', 'lock', 'payload']
    assert 'refused' in asked[1]


def test_stopping_at_the_load_is_recorded_as_a_stop(monkeypatch):
    job, ok, _asked = _load(monkeypatch, [True], [False])
    assert not ok and job.calls == ['open']
    assert job.recorder.events[-1]['name'] == 'stopped'


# ---------------------------------------------------------------------------- recording
def test_poses_are_zeroed_on_the_seat():
    arm = type('A', (), {'dry_run': False})()
    arm.T = T_SEAT @ _pose([0, 0, -3], [0, 0, 0])
    arm.tcp_pose = lambda: arm.T
    arm.wrench_in = lambda T, Tf=None: np.zeros(6)
    robot = type('R', (), {'arm': arm})()
    rec = TrialRecorder(robot, np.eye(4), phase_of=ta.assembly_phase, zero='seated',
                        zero_label='seated', pose_names=('goal', 'seated', 'released'),
                        summarize=ta.assembly_summary, header=ta._HEADER)
    rec.start(1)
    adm = type('Adm', (), {'last_ref': T_SEAT, 'last_cmd': T_SEAT, 'delta': np.zeros(6)})()
    rec.on_servo_step('assemble the object', adm)
    rec.pose('seated', T_SEAT)
    rec.pose('goal', T_SEAT @ _pose([1, 0, 0], [0, 0, 0]))
    out = tempfile.mkdtemp()
    row = rec.save(out, True, 'success', {})
    with np.load(os.path.join(out, 'attempt_01.npz')) as z:
        assert np.allclose(z['pose_rel'][0], [0, 0, -3, 0, 0, 0], atol=1e-9)
        assert list(z['phase']) == ['insert'] and json.loads(str(z['meta']))['zero'] == 'seated'
        assert np.all(np.isfinite(z['T_goal'])) and 'T_grasp' not in z.files
    assert row['goal_err_mm'] == pytest.approx(1.0)
    header = open(os.path.join(out, 'trials.csv')).readline().strip().split(',')
    assert header == ta._HEADER


def test_the_fasten_hold_reports_every_cycle_to_the_recorder():
    from urlab.apps.tile_assembly import TileAssemblyCycle
    job = TileAssemblyCycle.__new__(TileAssemblyCycle)
    seen = []
    job.on_servo_step = lambda what, adm: seen.append(what)
    job.guard = object()
    job.robot = type('R', (), {'arm': type('A', (), {'servo_stop': lambda s: None})()})()

    class Adm:
        delta = np.zeros(6)
        holds = 0

        def hold(self, T, seconds, guard=None, on_step=None):
            Adm.holds += 1
            if on_step:
                on_step()
            return 'done'
    import time
    ok = job._hold_while(Adm(), np.eye(4), lambda: time.sleep(0.05) or True, 'fasten (x)')
    assert ok and len(seen) == Adm.holds > 0 and set(seen) == {'fasten (x)'}


def test_the_plotter_draws_the_assembly_phases_and_names_the_seat():
    out = tempfile.mkdtemp()
    arm = type('A', (), {'dry_run': False})()
    arm.tcp_pose = lambda: T_SEAT
    arm.wrench_in = lambda T, Tf=None: np.zeros(6)
    robot = type('R', (), {'arm': arm})()
    rec = TrialRecorder(robot, np.eye(4), phase_of=ta.assembly_phase, zero='seated',
                        zero_label='seated', pose_names=('goal', 'seated', 'released'),
                        summarize=ta.assembly_summary, header=ta._HEADER)
    adm = type('Adm', (), {'last_ref': T_SEAT, 'last_cmd': T_SEAT, 'delta': np.zeros(6)})()
    for k in (1, 2):
        rec.start(k)
        for what, n in (('assemble the object', 20), ('fasten (screwdrive_predrive_to_bolt)', 15),
                        ('withdraw from the object', 10)):
            for _ in range(n):
                rec.on_servo_step(what, adm)
        rec.pose('seated', T_SEAT)
        rec.save(out, True, 'success', {})
    tp.plot_run(out)
    with np.load(os.path.join(out, 'plots', 'bands.npz')) as z:
        assert {'pose.insert.mean', 'pose.fasten.mean', 'pose.withdraw.mean'} <= set(z.files)


# ---------------------------------------------------------------------------- the inspection
T_GOAL = _pose([-1050, 360, 330], [180, 0, -3])
HUB, TILE = (41, 46), (3, 4, 5)


def test_the_assembly_error_is_the_tile_in_the_goal_frame():
    off = [1.0, -2.0, 0.5], [0.0, 0.0, 3.0]
    err = ta.assembly_error(T_GOAL, T_GOAL @ _pose(*off))
    assert np.allclose(err, off[0] + off[1], atol=1e-6)
    assert np.all(np.isnan(ta.assembly_error(T_GOAL, None)))


def test_the_summary_carries_the_inspected_error_and_the_markers_seen():
    a = {'t': np.zeros(1), 'phase': np.array(['insert']), 'wrench': np.zeros((1, 6))}
    tile = T_GOAL @ _pose([0.3, 0, -0.4], [0, 1.0, 0])
    rec = type('Rec', (), {'extras': {'inspect_hub_ids': np.array([41, 46]),
                                      'inspect_tile_ids': np.array([3, 5])}})()
    row = ta.assembly_summary(1, True, 'success', a,
                              {'seated': T_GOAL, 'inspect_goal': T_GOAL, 'inspect_tile': tile}, rec)
    assert row['assy_err_mm'] == pytest.approx(0.5, abs=1e-3)
    assert (row['assy_dx_mm'], row['assy_dz_mm']) == (pytest.approx(0.3), pytest.approx(-0.4))
    assert row['assy_dpitch_deg'] == pytest.approx(1.0)
    assert row['inspect_hub_markers'] == '41 46' and row['inspect_tile_markers'] == '3 5'
    assert set(row) <= set(ta._HEADER)


def _rigs():
    hub = {'markers': {m: {'size_m': 0.03, 'T_marker_target': np.eye(4)} for m in HUB}}
    tile = {'markers': {m: {'size_m': 0.02, 'T_marker_target': np.eye(4)} for m in TILE}}
    return hub, tile


def test_the_inspection_sweeps_once_for_both_parts_and_estimates_each_on_its_own(monkeypatch):
    hub_rig, tile_rig = _rigs()
    calls = {}

    def sweep(robot, camera, detector, plan, wanted=None, on_view=None, corner_log=None):
        calls['wanted'] = set(wanted)
        corner_log += ['view'] * 3
        return {m: [(np.eye(4), 0.4)] for m in (41, 46, 3, 4, 7)}    # 7: a stray
    monkeypatch.setattr(ta.mloc, 'sweep', sweep)
    monkeypatch.setattr(ta.mloc, 'fuse_markers', lambda seen, plan: {
        m: (_pose([m, 0, 0], [0, 0, 0]), 0.0, 3) for m in seen})
    monkeypatch.setattr(ta.mloc, 'refine_markers_multiview', lambda fused, cv, sizes, plan: (
        calls.setdefault('sizes', sizes), {41: (_pose([99, 0, 0], [0, 0, 0]), 0.1, 3)})[1])
    estimated = []

    def estimate(rig, fused, corner_views, plan):
        estimated.append((rig, sorted(fused), len(corner_views)))
        return T_GOAL if rig is hub_rig else T_GOAL @ _pose([1, 0, 0], [0, 0, 0])
    monkeypatch.setattr(ta.mloc, 'estimate_target', estimate)
    robot = type('R', (), {'camera': lambda self: np.eye(4)})()
    plan = ta.mloc.ViewPlan({'servo': {'enabled': False}})
    res = ta.inspect_assembly(robot, None, None, plan, hub_rig, tile_rig)
    assert calls['wanted'] == set(HUB) | set(TILE)
    assert calls['sizes'] == {41: 0.03, 46: 0.03, 3: 0.02, 4: 0.02, 5: 0.02}
    assert estimated == [(hub_rig, [41, 46], 3), (tile_rig, [3, 4], 3)], 'no cross-talk, no stray'
    assert sorted(res['hub']) == [41, 46] and sorted(res['tile']) == [3, 4]
    assert np.allclose(res['hub'][41], _pose([99, 0, 0], [0, 0, 0])), 'the refined pose is kept'
    assert np.allclose(ta.assembly_error(res['T_goal'], res['T_tile']), [1, 0, 0, 0, 0, 0])


def test_a_part_with_no_markers_seen_is_not_estimated(monkeypatch):
    hub_rig, tile_rig = _rigs()
    monkeypatch.setattr(ta.mloc, 'sweep', lambda *a, **k: {41: [(np.eye(4), 0.4)]})
    monkeypatch.setattr(ta.mloc, 'fuse_markers', lambda seen, plan: {
        m: (np.eye(4), 0.0, 3) for m in seen})
    monkeypatch.setattr(ta.mloc, 'refine_markers_multiview', lambda *a: {})
    monkeypatch.setattr(ta.mloc, 'estimate_target', lambda rig, fused, cv, plan: T_GOAL)
    robot = type('R', (), {'camera': lambda self: np.eye(4)})()
    res = ta.inspect_assembly(robot, None, None, ta.mloc.ViewPlan({}), hub_rig, tile_rig)
    assert res['T_goal'] is not None and res['T_tile'] is None and res['tile'] == {}


def test_the_inspection_pose_defaults_to_where_the_assembly_was_calibrated():
    over = ['experiment.inspection.joints_deg=null', 'fixed_view_joints_deg=[1, 2, 3, 4, 5, 6]']
    cfg = C.load('tile_assembly_experiment', over)
    calib = np.radians([9, 9, 9, 9, 9, 9])
    assert np.allclose(ta.inspection_settings(cfg, calib)['joints'], calib)
    assert np.allclose(np.degrees(ta.inspection_settings(cfg, None)['joints']), [1, 2, 3, 4, 5, 6])
    own = C.load('tile_assembly_experiment',
                 ['experiment.inspection.joints_deg=[6, 5, 4, 3, 2, 1]',
                  'experiment.inspection.via_joints=[[1, 1, 1, 1, 1, 1]]'])
    got = ta.inspection_settings(own, calib)
    assert np.allclose(np.degrees(got['joints']), [6, 5, 4, 3, 2, 1])
    assert len(got['via']) == 1 and np.allclose(np.degrees(got['via'][0]), 1)
    with pytest.raises(ValueError, match='six'):
        ta.inspection_settings(C.load('tile_assembly_experiment',
                                      ['experiment.inspection.joints_deg=[1, 2]']))


class _InspectJob(ta.AssemblyTrialCycle):
    def __init__(self, via=()):
        self.robot = _robot(False)
        self.guard, self.camera, self.inspect_detector, self.inspect_images = None, None, None, None
        self.fixed_rig, self.held_rig = _rigs()
        self.inspection = {'enabled': True, 'joints': np.zeros(6), 'via': list(via),
                           'plan': ta.mloc.ViewPlan({}), 'save_images': False}
        self.recorder = TrialRecorder(self.robot, np.eye(4))
        self.recorder.start(1)
        self.recorder.pose('seated', T_GOAL @ _pose([0, 0, 1], [0, 0, 0]))

    def _joint_caps(self):
        return None


def test_the_inspection_drives_out_and_records_the_error_and_every_marker(monkeypatch):
    job = _InspectJob(via=[np.ones(6)])
    tile = T_GOAL @ _pose([0.5, 0, 0], [0, 0, 2])
    monkeypatch.setattr(ta, 'inspect_assembly', lambda *a: {
        'T_goal': T_GOAL, 'T_tile': tile, 'views': 6,
        'hub': {46: np.eye(4), 41: np.eye(4)}, 'tile': {3: np.eye(4)}})
    assert job.inspect()
    assert job.robot.moves == ['inspection via 1/1', 'inspection pose']
    x = job.recorder.extras
    assert np.allclose(x['assy_err6'], [0.5, 0, 0, 0, 0, 2], atol=1e-6)
    assert np.allclose(x['tile_vs_seated6'][:3], [0.5, 0, -1], atol=1e-6)
    assert list(x['inspect_hub_ids']) == [41, 46] and x['inspect_hub_T'].shape == (2, 4, 4)
    assert x['inspect_tile_T'].shape == (1, 4, 4)
    assert np.allclose(job.recorder.poses['inspect_tile'], tile)


def test_an_assembly_that_cannot_be_seen_is_saved_as_nan_not_failed(monkeypatch):
    job = _InspectJob()
    monkeypatch.setattr(ta, 'inspect_assembly', lambda *a: {
        'T_goal': None, 'T_tile': None, 'views': 6, 'hub': {}, 'tile': {}})
    assert job.inspect()
    assert np.all(np.isnan(job.recorder.extras['assy_err6']))
    assert job.recorder.extras['inspect_hub_T'].shape == (0, 4, 4)


def test_an_unreachable_inspection_pose_fails_the_step(monkeypatch):
    job = _InspectJob()
    job.robot.move_joints = lambda q, label='', guard=None, caps=None: False
    monkeypatch.setattr(ta, 'inspect_assembly', lambda *a: pytest.fail('looked from nowhere'))
    assert not job.inspect()


def _saved_run(errors):
    """A run dir whose trials carry the given (assy_err6, success) pairs."""
    out = tempfile.mkdtemp()
    arm = type('A', (), {'dry_run': False})()
    arm.tcp_pose = lambda: T_GOAL
    arm.wrench_in = lambda T, Tf=None: np.zeros(6)
    robot = type('R', (), {'arm': arm})()
    rec = TrialRecorder(robot, np.eye(4), phase_of=ta.assembly_phase, zero='seated',
                        zero_label='seated',
                        pose_names=('goal', 'seated', 'released', 'inspect_goal', 'inspect_tile'),
                        summarize=ta.assembly_summary, header=ta._HEADER)
    adm = type('Adm', (), {'last_ref': T_GOAL, 'last_cmd': T_GOAL, 'delta': np.zeros(6)})()
    for k, (err, ok) in enumerate(errors, start=1):
        rec.start(k)
        for _ in range(5):
            rec.on_servo_step('assemble the object', adm)
        rec.pose('seated', T_GOAL)
        rec.extras['assy_err6'] = np.asarray(err, dtype=float)
        rec.save(out, ok, 'success' if ok else 'failed', {})
    return out


def test_the_spread_is_over_the_successful_measured_trials():
    run = _saved_run([([1, 0, 0, 0, 0, 1], True), ([3, 0, 0, 0, 0, 3], True),
                      ([np.nan] * 6, True), ([50, 0, 0, 0, 0, 0], False)])
    stats = ta.assembly_error_stats(run)
    mean, std, lo, hi, n = stats['dx_mm']
    assert (mean, lo, hi, n) == (2.0, 1.0, 3.0, 2) and std == pytest.approx(np.sqrt(2.0))
    assert ta.assembly_error_stats(run, include_failed=True)['dx_mm'][4] == 3
    path = ta.write_error_summary(run, stats)
    lines = open(path).read().splitlines()
    assert lines[0] == 'component,mean,std,min,max,n' and lines[1].startswith('dx_mm,2.0,')


def test_the_plotter_draws_the_assembly_error_per_trial():
    run = _saved_run([([1, 0, 0, 0, 0, 1], True), ([3, 0, 0, 0, 0, 3], True)])
    written = tp.plot_run(run)
    assert os.path.join(run, 'plots', 'assembly_error.png') in written
