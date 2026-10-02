"""MARKER-RELATIVE ASSEMBLY and its visual calibration, without a robot or a camera.

The failures worth catching are the ones that would drive a part somewhere plausible and wrong:
a goal that does not follow the fixed object when it moves, an insertion axis read in the wrong
frame, and a catalogue rewrite that loses or garbles an entry.
"""

import os
import tempfile

import numpy as np
import pytest

from urlab import config as C
from urlab import tool_frames
from urlab.apps.coupler_marker_assemble import (MarkerAssembleCycle, fixed_rig, marker_leg,
                                                marker_to_marker, parse_transitions,
                                                resolve_marker_assembly, run_transitions)
from urlab.apps.coupler_pick_assemble import AssembleCycle
from urlab.apps.coupler_pick_place import CouplerCycle
from urlab.apps.marker_assembly_calibration import (_MarkerAssemblyCalibration,
                                                    goal_in_fixed_markers, joints_deg,
                                                    yaml_document)
from urlab.skills import marker_localize as mloc
from urlab.transforms import inverse, pose_error, xyzrpy_to_matrix


def _pose(xyz_mm, rpy_deg):
    return xyzrpy_to_matrix(np.array(xyz_mm, float) / 1000.0, np.radians(rpy_deg))


def _fused(T):
    return (T, 0.0, 0.0, 3, 1.0)


# A fixed object with two markers, a held object with one, assembled.
T_BASE_FIXED = {10: _pose([600, -100, 20], [0, 0, 30]), 11: _pose([700, 50, 25], [2, -1, 120])}
T_HELD_GRASP = _pose([40, -12, -3], [180, 2, -88])         # objects.yaml: grasp in held marker
T_BASE_GOAL = _pose([650, -40, 90], [178, 1, 15])           # the held mating frame, assembled
T_BASE_HELD_MARKER = T_BASE_GOAL @ inverse(T_HELD_GRASP)


def _entry(goals=None, view_joints=None):
    goals = goals or goal_in_fixed_markers(
        T_BASE_GOAL, {mid: _fused(T) for mid, T in T_BASE_FIXED.items()})
    return {'held_object': 'tile_1', 'dictionary': 'DICT_4X4_50', 'view_joints': view_joints,
            'markers': {mid: {'size_m': 0.04525, 'T_marker_goal': T, 'meta': {'captures': 3}}
                        for mid, T in goals.items()},
            'meta': {'captures': 3, 'measured': '2026-10-01'}}


# ---------------------------------------------------------------------------- the geometry
def test_calibrating_then_locating_reproduces_the_goal_and_follows_a_moved_fixture():
    """THE WHOLE POINT. Calibrate with the objects assembled, move the fixed object anywhere, and
    the goal the run computes from its markers is the assembled pose moved with it."""
    held_goal = T_BASE_HELD_MARKER @ T_HELD_GRASP              # what calibration solves
    entry = _entry(goal_in_fixed_markers(
        held_goal, {mid: _fused(T) for mid, T in T_BASE_FIXED.items()}))
    rig = fixed_rig(entry)
    plan = mloc.ViewPlan({'min_views': 1})
    for M in (np.eye(4), _pose([-150, 230, 5], [3, -2, 70])):
        moved = {mid: _fused(M @ T) for mid, T in T_BASE_FIXED.items()}
        T_goal, votes = mloc.vote_target(rig, moved, plan)
        lin, ang = pose_error(M @ T_BASE_GOAL, T_goal)
        assert lin < 1e-9 and ang < 1e-6, 'the goal did not follow the fixed object'
        assert sorted(votes) == [10, 11], 'every fixed marker should vote'


def test_marker_to_marker_is_the_relative_pose_of_the_two_markers():
    entry = _entry()
    for mid, T_fm in T_BASE_FIXED.items():
        T_fh = marker_to_marker(entry['markers'][mid]['T_marker_goal'], T_HELD_GRASP)
        assert np.allclose(T_fh, inverse(T_fm) @ T_BASE_HELD_MARKER)


def test_a_fixed_marker_axis_points_the_same_way_in_the_room_after_conversion():
    """`frame: fixed_marker` is converted to the coupler frame at the goal. Whatever the fixed
    object's pose, the converted axis must point where the marker-frame axis points."""
    entry = _entry()
    for mid in (10, 11):
        spec = marker_leg({'distance_mm': 80.0, 'axis': [0, 0, 1], 'frame': 'fixed_marker',
                           'marker': mid}, 'motion.assembly_standoff', entry)
        assert spec['frame'] == 'coupler' and spec['from_marker'] == mid
        assert spec['distance_m'] == pytest.approx(0.08)
        R_goal = T_BASE_GOAL[:3, :3]
        assert np.allclose(R_goal @ spec['axis'], T_BASE_FIXED[mid][:3, :3] @ [0, 0, 1],
                           atol=1e-9)


def test_the_fixed_marker_defaults_to_the_lowest_id_and_an_unknown_one_is_refused():
    entry = _entry()
    spec = marker_leg({'axis': [0, 0, 1]}, 'w', entry)
    assert spec['from_marker'] == 10
    with pytest.raises(ValueError, match='not one of'):
        marker_leg({'axis': [0, 0, 1], 'marker': 99}, 'w', entry)
    with pytest.raises(ValueError, match='zero'):
        marker_leg({'axis': [0, 0, 0]}, 'w', entry)


# Pinned so the tests do not depend on whatever the shipped config currently says.
_PINNED = ['fixed_view_joints_deg=null', 'pick_view_joints_deg=null', 'assembly_joints_deg=null',
           'approach_path=[]', 'transition_joints_deg={}', 'approach_speed_scale=1.0',
           'joint_move_speed=null', 'goal_offset=null']


def _cycle(entry=None, cfg_over=()):
    job = MarkerAssembleCycle.__new__(MarkerAssembleCycle)
    job.cfg = C.load('coupler_marker_assemble', _PINNED + list(cfg_over))
    job.masm_name, job.masm = 'tile_on_plate', entry or _entry()
    job.name = 'tile_1'
    job.T_goal = None
    job.transitions = parse_transitions(job.cfg.get('transition_joints_deg'))
    return job


def test_a_fixed_marker_axis_is_refused_on_the_pick_legs():
    """Before the pick the fixed object is not involved, so its frame means nothing there."""
    job = _cycle()
    with pytest.raises(ValueError, match='destination'):
        job._parse_leg('mate_standoff', {'axis': [0, 0, 1], 'frame': 'fixed_marker'})
    assert job._parse_leg('assembly_standoff',
                          {'axis': [0, 0, 1], 'frame': 'fixed_marker'})['frame'] == 'coupler'
    assert job._parse_leg('pick_retract', None)['frame'] == 'coupler'      # the default leg


def test_the_target_is_the_located_goal_and_ignores_the_pick():
    job = _cycle()
    with pytest.raises(ValueError, match='not been located'):
        job._target_pose(np.eye(4))
    job.T_goal = T_BASE_GOAL
    for pick in (_pose([400, 0, 50], [180, 0, 0]), _pose([100, 300, 80], [175, 5, 90])):
        assert np.allclose(job._target_pose(pick), T_BASE_GOAL)


def test_the_marker_cycle_is_the_assemble_cycle_with_a_preamble():
    assert issubclass(MarkerAssembleCycle, AssembleCycle)
    assert CouplerCycle.preamble(CouplerCycle.__new__(CouplerCycle), None) == []
    steps = _cycle().preamble(None)
    assert len(steps) == 1 and steps[0].name == 'look at the fixed object'


# ---------------------------------------------------------------------------- the fixed view
class _Arm:
    dry_run = True

    def __init__(self, q):
        self._q = list(q)
        self.speed_scale = 1.0
        self.scales = []

    def set_speed_scale(self, scale, phase=''):
        self.speed_scale = scale
        self.scales.append((scale, phase))

    def q(self):
        return list(self._q)


class _Robot:
    def __init__(self, q=(0, -90, 90, -90, -90, 0)):
        self.arm = _Arm(np.radians(q))
        self.moves = []

    def move_joints(self, q, label='move', guard=None, caps=None):
        self.moves.append((label, np.degrees(np.asarray(q, float)).round(3).tolist()))
        self.caps = getattr(self, 'caps', []) + [caps]
        self.arm._q = list(q)
        return True


def _locating(monkeypatch, job, robot, found=T_BASE_GOAL):
    calls = []

    def fake_locate(rbt, camera, detector, rig, plan, **kw):
        calls.append(('locate', [m[0] for m in robot.moves]))
        job.located_with = plan
        assert rig is job.fixed_rig and detector is job.fixed_detector
        return found
    monkeypatch.setattr(mloc, 'locate', fake_locate)
    job.robot, job.camera, job.guard = robot, None, None
    job.fixed_rig, job.fixed_detector, job.fixed_images = fixed_rig(job.masm), object(), None
    job.fixed_plan = mloc.ViewPlan(job.cfg.section('marker_views'))
    return calls


def test_the_fixed_object_is_seen_from_the_calibrated_view_then_the_arm_returns(monkeypatch):
    view = np.radians([10, -80, 100, -110, -90, 5])
    job = _cycle(_entry(view_joints=view))
    robot = _Robot()
    calls = _locating(monkeypatch, job, robot)
    assert job.locate_fixed()
    assert [m[0] for m in robot.moves] == ['fixed object view', 'back to the pick view']
    assert robot.moves[0][1] == pytest.approx([10, -80, 100, -110, -90, 5])
    assert robot.moves[1][1] == pytest.approx([0, -90, 90, -90, -90, 0])   # where it started
    assert calls == [('locate', ['fixed object view'])], 'located before it went back'
    assert np.allclose(job.T_goal, T_BASE_GOAL)


def test_the_config_view_overrides_the_calibrated_one(monkeypatch):
    job = _cycle(_entry(view_joints=np.zeros(6)),
                 ['fixed_view_joints_deg=[1,2,3,4,5,6]', 'pick_view_joints_deg=[6,5,4,3,2,1]'])
    robot = _Robot()
    _locating(monkeypatch, job, robot)
    assert job.locate_fixed()
    assert robot.moves[0][1] == pytest.approx([1, 2, 3, 4, 5, 6])
    assert robot.moves[1][1] == pytest.approx([6, 5, 4, 3, 2, 1])


def test_no_view_anywhere_locates_from_where_the_arm_is(monkeypatch):
    job = _cycle(_entry(view_joints=None))
    robot = _Robot()
    _locating(monkeypatch, job, robot)
    assert job.locate_fixed() and robot.moves == []


def test_a_fixed_object_not_found_stops_before_the_pick(monkeypatch):
    job = _cycle(_entry(view_joints=np.zeros(6)))
    robot = _Robot()
    _locating(monkeypatch, job, robot, found=None)
    assert not job.locate_fixed()
    assert job.T_goal is None
    assert [m[0] for m in robot.moves] == ['fixed object view'], 'no return trip after a miss'


# ---------------------------------------------------------------------------- the catalogue
def _write(text):
    fd, path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as fh:
        fh.write(text)
    return path


def test_an_assembly_round_trips_through_the_catalogue():
    view = np.radians([10.5, -80, 100, -110, -90, 5])
    entry = _entry(view_joints=view)
    path = _write(yaml_document({'tile_on_plate': entry}, stamp='2026-10-01'))
    back = tool_frames.load_marker_assemblies(path=path)['tile_on_plate']
    assert back['held_object'] == 'tile_1' and back['dictionary'] == 'DICT_4X4_50'
    assert np.allclose(back['view_joints'], view, atol=1e-6)
    for mid, m in entry['markers'].items():
        lin, ang = pose_error(m['T_marker_goal'], back['markers'][mid]['T_marker_goal'])
        assert lin < 1e-5 and np.degrees(ang) < 0.01            # 0.01 mm / 0.01 deg printing
        assert back['markers'][mid]['size_m'] == pytest.approx(0.04525)
    assert back['meta']['captures'] == 3


def test_an_absent_or_empty_catalogue_is_empty_not_an_error():
    assert tool_frames.load_marker_assemblies(path='/nonexistent/marker_assemblies.yaml') == {}
    assert tool_frames.load_marker_assemblies(path=_write(yaml_document({}))) == {}
    tool_frames.load_marker_assemblies()                  # the shipped one loads, whatever it holds


@pytest.mark.parametrize('body, match', [
    ('    markers: {10: {size_mm: 40, xyz_mm: [0,0,0], rpy_deg: [0,0,0]}}', 'held_object'),
    ('    held_object: t\n    markers: {10: {xyz_mm: [0,0,0], rpy_deg: [0,0,0]}}', 'size_mm'),
    ('    held_object: t\n    markers: {10: {size_mm: 40, xyz_m: [0,0,0]}}', 'unknown'),
    ('    held_object: t\n    markers: {10: {size_mm: 40, xyz_mm: [0,0,0]}}\n    colour: red',
     'unknown'),
    ('    held_object: t\n    view_joints_deg: [1, 2, 3]\n'
     '    markers: {10: {size_mm: 40, xyz_mm: [0,0,0]}}', 'six'),
    ('    held_object: t\n    markers: {}', 'no markers'),
])
def test_a_malformed_entry_is_refused_not_guessed(body, match):
    path = _write('marker_assemblies:\n  a:\n' + body + '\n')
    with pytest.raises(ValueError, match=match):
        tool_frames.load_marker_assemblies(path=path)


def test_the_assembly_name_is_required_and_a_typo_lists_what_exists():
    cat = {'tile_on_plate': _entry()}
    with pytest.raises(ValueError, match='tile_on_plate'):
        resolve_marker_assembly(C.load('coupler_marker_assemble', ['marker_assembly=null']), cat)
    with pytest.raises(KeyError, match='tile_on_plate'):
        resolve_marker_assembly(C.load('coupler_marker_assemble', ['marker_assembly=nope']), cat)
    name, entry = resolve_marker_assembly(
        C.load('coupler_marker_assemble', ['marker_assembly=tile_on_plate']), cat)
    assert name == 'tile_on_plate' and entry is cat['tile_on_plate']


def test_writing_merges_into_the_catalogue_and_backs_up_the_old_one():
    """A calibration replaces ITS entry and leaves every other assembly alone."""
    other = _entry()
    path = _write(yaml_document({'other_one': other}, stamp='2026-09-30'))
    out_dir = tempfile.mkdtemp()
    cal = _MarkerAssemblyCalibration.__new__(_MarkerAssemblyCalibration)
    cal.cfg = C.load('marker_assembly_calibration', [f'marker_assemblies_file={path}'])
    cal.name, cal.held_name, cal.out_dir = 'tile_on_plate', 'tile_1', out_dir
    cal.fixed_sizes = {10: 0.04525}
    cal.q_view = np.radians([1, 2, 3, 4, 5, 6])
    T = _pose([1, 2, 3], [4, 5, 6])
    cal.offsets, cal.scans, cal.goals = {10: [T]}, [{10: _fused(T)}], [T_BASE_GOAL]
    cal.results = {10: (T, 0.0, 0.0)}
    assert cal.write_outputs()
    back = tool_frames.load_marker_assemblies(path=path)
    assert sorted(back) == ['other_one', 'tile_on_plate']
    assert np.allclose(back['tile_on_plate']['markers'][10]['T_marker_goal'], T, atol=1e-5)
    assert np.allclose(back['tile_on_plate']['view_joints'], cal.q_view, atol=1e-6)
    for f in ('marker_assemblies.yaml.bak', 'marker_assembly.yaml', 'captures.csv'):
        assert os.path.isfile(os.path.join(out_dir, f)), f


def _calibrating(q_start, captures=2):
    robot = _Robot()
    cal = _MarkerAssemblyCalibration.__new__(_MarkerAssemblyCalibration)
    cal.cfg = C.load('marker_assembly_calibration')
    cal.cfg['view_joints_deg'] = [1, 2, 3, 4, 5, 6]
    cal.robot, cal.held_name, cal.captures = robot, 'tile_1', captures
    cal.q_start = None if q_start is None else np.radians(q_start)
    cal.q_view, cal.offsets = None, {}
    cal._capture = lambda k: cal.offsets.setdefault(10, []).append(k) or True
    return cal, robot


def test_the_start_pose_comes_first_and_once_then_the_view_every_capture():
    cal, robot = _calibrating([10, -100, 90, -80, -90, 5])
    assert cal.to_start() and cal._collect()
    assert robot.moves == [('start pose', [10, -100, 90, -80, -90, 5]),
                           ('assembly view pose', [1, 2, 3, 4, 5, 6]),
                           ('assembly view pose', [1, 2, 3, 4, 5, 6])]


def test_no_start_pose_begins_where_the_arm_is():
    cal, robot = _calibrating(None)
    assert cal.to_start()
    assert robot.moves == []


def test_a_start_pose_that_is_not_six_joints_is_refused():
    assert joints_deg({}, 'start_joints_deg') is None
    assert joints_deg({'start_joints_deg': None}, 'start_joints_deg') is None
    assert np.allclose(joints_deg({'start_joints_deg': [90, 0, 0, 0, 0, 0]}, 'start_joints_deg'),
                       [np.pi / 2, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match='6 joint angles'):
        joints_deg({'start_joints_deg': [1, 2, 3]}, 'start_joints_deg')


def test_the_shipped_configs_carry_what_the_apps_read():
    cfg = C.load('coupler_marker_assemble')
    assert cfg['cleat_toolchanger']['enabled'] is False
    for key in ('compliance', 'compliance_loaded', 'compliance_insert', 'force_guard',
                'mate_preload', 'assembly_preload', 'marker_views', 'toolchanger'):
        assert cfg.section(key), key
    cal = C.load('marker_assembly_calibration')
    assert cal['fixed_markers'] and cal['held_object']


# ---------------------------------------------------------------------------- the approach
from urlab.apps.coupler_marker_assemble import (coupler_standoff_from_path,  # noqa: E402
                                                coupler_waypoint_pose, parse_coupler_path)
from urlab.apps.coupler_pick_place import offset_pose  # noqa: E402


def _path(*wps):
    return parse_coupler_path([dict(w) for w in wps])


def test_a_waypoint_is_an_offset_in_the_goals_own_coupler_frame():
    """[0, 0, -250] is 250 mm back along the ASSEMBLED mating axis, however the goal is tilted,
    at the goal's attitude; rpy turns the tool about the waypoint's own axes, not the goal's."""
    wp, = _path({'xyz_mm': [0, 0, -250]})
    T = coupler_waypoint_pose(T_BASE_GOAL, wp)
    assert np.allclose(T[:3, 3], T_BASE_GOAL[:3, 3] - 0.25 * T_BASE_GOAL[:3, 2])
    assert np.allclose(T[:3, :3], T_BASE_GOAL[:3, :3])
    turned, = _path({'xyz_mm': [0, 0, -250], 'rpy_deg': [0, 0, 90]})
    T2 = coupler_waypoint_pose(T_BASE_GOAL, turned)
    assert np.allclose(T2[:3, 3], T[:3, 3]), 'a rotation must not drag the position'
    assert np.allclose(T2[:3, :3], T_BASE_GOAL[:3, :3] @ _pose([0, 0, 0], [0, 0, 90])[:3, :3])


def test_the_path_follows_the_fixed_object_when_it_moves():
    M = _pose([-150, 230, 5], [3, -2, 70])
    for wp in _path({'xyz_mm': [-300, 0, -200]}, {'xyz_mm': [-100, 0, 0]}):
        assert np.allclose(coupler_waypoint_pose(M @ T_BASE_GOAL, wp),
                           M @ coupler_waypoint_pose(T_BASE_GOAL, wp))


def test_the_last_waypoint_IS_the_insertion_standoff():
    path = _path({'xyz_mm': [-300, 0, -200]}, {'xyz_mm': [-100, 0, 0]})
    leg = coupler_standoff_from_path(path)
    assert leg['frame'] == 'coupler' and leg['distance_m'] == pytest.approx(0.1)
    assert np.allclose(offset_pose(T_BASE_GOAL, leg), coupler_waypoint_pose(T_BASE_GOAL, path[-1]))
    assert coupler_standoff_from_path([]) is None


def test_a_rotated_last_waypoint_is_refused():
    """The insertion is a straight line at constant attitude -- a rotated standoff would leave a
    jump between the end of the carry and the start of the insertion."""
    with pytest.raises(ValueError, match='assembled attitude'):
        coupler_standoff_from_path(_path({'xyz_mm': [0, 0, -100], 'rpy_deg': [0, 0, 10]}))
    # a rotation EARLIER in the path is fine
    coupler_standoff_from_path(_path({'xyz_mm': [0, 0, -200], 'rpy_deg': [0, 0, 10]},
                                     {'xyz_mm': [0, 0, -100]}))


def test_the_path_parses_from_the_config_and_typos_are_refused():
    cfg = C.load('coupler_marker_assemble',
                 ['approach_path=[{name: a, xyz_mm: [-300, 0, -200]}, {xyz_mm: [-100, 0, 0]}]'])
    path = parse_coupler_path(cfg['approach_path'])
    assert [w['name'] for w in path] == ['a', 'waypoint 2']
    assert np.allclose(path[0]['xyz'], [-0.3, 0.0, -0.2])
    assert parse_coupler_path(None) == [] and parse_coupler_path([]) == []
    with pytest.raises(ValueError, match='unknown'):
        parse_coupler_path([{'xyz_mn': [0, 0, -100]}])
    with pytest.raises(ValueError, match='ends AT'):
        parse_coupler_path([{'xyz_mm': [0, 0, 0]}])


def _carrying(cfg_over=(), path=()):
    job = _cycle(cfg_over=cfg_over)
    job.robot, job.guard, job.T_place = _Robot(), None, T_BASE_GOAL
    job.approach_path = list(path)
    job.legs = {'assembly_standoff': coupler_standoff_from_path(job.approach_path)
                or {'distance_m': 0.1, 'axis': np.array([0.0, 0.0, -1.0]), 'frame': 'coupler'}}
    moved = []
    job._move_to = lambda T, label: moved.append((label, T)) or True
    return job, moved


def test_the_carry_goes_to_the_assembly_joints_then_walks_the_path_in_order():
    path = _path({'name': 'beside', 'xyz_mm': [-300, 0, -200]},
                 {'name': 'standoff', 'xyz_mm': [-100, 0, 0]})
    job, moved = _carrying(['assembly_joints_deg=[90, -80, 100, -110, -90, 0]'], path)
    assert job.carry()
    assert [m[0] for m in job.robot.moves] == ['assembly joints']
    assert job.robot.moves[0][1] == pytest.approx([90, -80, 100, -110, -90, 0])
    assert [m[0] for m in moved] == ['approach 1/2: beside', 'approach 2/2: standoff']
    assert np.allclose(moved[-1][1], offset_pose(T_BASE_GOAL, job.legs['assembly_standoff'])), (
        'the carry must end exactly where the insertion begins')


def test_no_path_and_no_joints_is_the_one_straight_hop():
    job, moved = _carrying()
    assert job.carry()
    assert job.robot.moves == []
    assert len(moved) == 1
    assert np.allclose(moved[0][1], offset_pose(T_BASE_GOAL, job.legs['assembly_standoff']))


# ---------------------------------------------------------------------------- stepping the approach
from urlab.apps import coupler_marker_assemble as _mod  # noqa: E402


def _stepping(monkeypatch, answers, cfg_over=()):
    path = _path({'name': 'beside', 'xyz_mm': [-300, 0, -200]},
                 {'name': 'standoff', 'xyz_mm': [-100, 0, 0]})
    job, moved = _carrying(['assembly_joints_deg=[90, -80, 100, -110, -90, 0]'] + list(cfg_over),
                           path)
    job.robot.arm.dry_run = False                 # the pause is off in a dry run
    events = []
    job.robot.move_joints = (lambda q, label='', guard=None, caps=None:
                             events.append(('move', label)) or True)
    job._move_to = lambda T, label: events.append(('move', label)) or True
    replies = iter(answers)
    monkeypatch.setattr(_mod, 'ask', lambda prompt: events.append(('ask',)) or next(replies))
    return job, events


def test_every_approach_move_waits_for_enter_first(monkeypatch):
    job, events = _stepping(monkeypatch, [True, True, True])
    assert job.carry()
    assert events == [('ask',), ('move', 'assembly joints'),
                      ('ask',), ('move', 'approach 1/2: beside'),
                      ('ask',), ('move', 'approach 2/2: standoff')]


def test_the_pause_names_where_the_arm_is_and_where_it_goes_next(monkeypatch):
    job, _events = _stepping(monkeypatch, [True, True, True])
    seen = []
    monkeypatch.setattr(job, '_confirm_approach',
                        lambda current, label, where: seen.append((current, label)) or True)
    assert job.carry()
    assert seen == [("retracted from the pick, 'tile_1' in hand", 'assembly joints'),
                    ('assembly joints', 'approach 1/2: beside'),
                    ('approach 1/2: beside', 'approach 2/2: standoff')]


def test_q_stops_before_the_move_with_nothing_moved(monkeypatch):
    job, events = _stepping(monkeypatch, [True, False])
    assert not job.carry()
    assert events == [('ask',), ('move', 'assembly joints'), ('ask',)], (
        'declining must stop BEFORE the waypoint move, not after it')


@pytest.mark.parametrize('cfg_over, dry', [(['confirm_approach_steps=false'], False),
                                           (['skip_prompts=true'], False),
                                           ([], True)])
def test_the_pause_is_off_when_asked_unattended_or_dry(monkeypatch, cfg_over, dry):
    job, events = _stepping(monkeypatch, [], cfg_over)
    job.robot.arm.dry_run = dry
    assert job.carry()
    assert ('ask',) not in events and len(events) == 3


# ---------------------------------------------------------------------------- transitions
def test_transitions_parse_any_pair_and_refuse_typos():
    t = parse_transitions({'fixed_view_to_pick_view': [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]],
                           'pick_view_to_assembly': [10, 20, 30, 40, 50, 60],   # one bare pose
                           'assembly_to_fixed_view': [[0, 0, 0, 0, 0, 0]]})
    assert np.allclose(np.degrees(t[('fixed_view', 'pick_view')][1]), [6, 5, 4, 3, 2, 1])
    assert len(t[('pick_view', 'assembly')]) == 1
    assert parse_transitions(None) == {} and parse_transitions({}) == {}
    for bad in ({'fixed_to_pick_view': [[0] * 6]}, {'pick_view_to_pick_view': [[0] * 6]},
                {'pick_view-assembly': [[0] * 6]}):
        with pytest.raises(ValueError, match='<from>_to_<to>'):
            parse_transitions(bad)
    for bad in ([[1, 2, 3]], [], [[0] * 6, [0] * 5]):
        with pytest.raises(ValueError, match='six angles'):
            parse_transitions({'start_to_fixed_view': bad})


@pytest.mark.parametrize('has, moves', [
    ((True, True, True), [('start', 'fixed_view'), ('fixed_view', 'pick_view'),
                          ('pick_view', 'assembly')]),
    ((True, False, True), [('start', 'fixed_view'), ('fixed_view', 'start'),
                           ('start', 'assembly')]),
    ((False, True, False), [('start', 'pick_view')]),
    ((False, False, False), []),
])
def test_the_run_makes_exactly_these_moves(has, moves):
    assert run_transitions(*has) == moves


def test_the_fixed_view_trip_goes_through_its_transitions(monkeypatch):
    job = _cycle(_entry(view_joints=np.radians([10, 0, 0, 0, 0, 0])), [
        'pick_view_joints_deg=[20, 0, 0, 0, 0, 0]',
        'transition_joints_deg={start_to_fixed_view: [[1, 0, 0, 0, 0, 0]], '
        'fixed_view_to_pick_view: [[2, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0]]}'])
    job.transitions = parse_transitions(job.cfg['transition_joints_deg'])
    robot = _Robot()
    calls = _locating(monkeypatch, job, robot)
    assert job.locate_fixed()
    assert [(m[0], m[1][0]) for m in robot.moves] == [
        ('transition 1/1: start -> fixed_view', 1), ('fixed object view', 10),
        ('transition 1/2: fixed_view -> pick_view', 2),
        ('transition 2/2: fixed_view -> pick_view', 3), ('back to the pick view', 20)]
    assert calls[0][1][-1] == 'fixed object view', 'located at the view, before leaving it'


def test_without_a_pick_view_the_return_uses_fixed_view_to_start(monkeypatch):
    job = _cycle(_entry(view_joints=np.radians([10, 0, 0, 0, 0, 0])), [
        'transition_joints_deg={fixed_view_to_start: [[5, 0, 0, 0, 0, 0]]}'])
    job.transitions = parse_transitions(job.cfg['transition_joints_deg'])
    robot = _Robot()
    _locating(monkeypatch, job, robot)
    assert job.locate_fixed()
    assert [m[0] for m in robot.moves] == ['fixed object view',
                                           'transition 1/1: fixed_view -> start',
                                           'back to the pick view']


def test_the_assembly_transitions_are_approach_steps_and_pause(monkeypatch):
    job, events = _stepping(monkeypatch, [True] * 5, [
        'pick_view_joints_deg=[20, 0, 0, 0, 0, 0]',
        'transition_joints_deg={pick_view_to_assembly: [[1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]]}'])
    job.transitions = parse_transitions(job.cfg['transition_joints_deg'])
    assert job.carry()
    assert events == [('ask',), ('move', 'transition 1/2: pick_view -> assembly'),
                      ('ask',), ('move', 'transition 2/2: pick_view -> assembly'),
                      ('ask',), ('move', 'assembly joints'),
                      ('ask',), ('move', 'approach 1/2: beside'),
                      ('ask',), ('move', 'approach 2/2: standoff')]


def test_the_fixed_object_is_located_with_single_view_when_configured(monkeypatch):
    """servo.single_view reaches the fixed localization, as it does the pick (one ViewPlan)."""
    for flag in (True, False):
        job = _cycle(_entry(view_joints=None),
                     ['marker_views.servo.enabled=true',
                      f'marker_views.servo.single_view={str(flag).lower()}'])
        _locating(monkeypatch, job, _Robot())
        assert job.locate_fixed()
        assert job.located_with.servo.enabled and job.located_with.servo.single_view is flag


# ---------------------------------------------------------------------------- fastening
import threading  # noqa: E402
import time  # noqa: E402

from urlab.apps.coupler_marker_assemble import parse_fastening  # noqa: E402
from urlab.robot import screwdriver as sdmod  # noqa: E402


def test_fastening_is_off_unless_enabled_and_defaults_to_the_two_sequences():
    assert parse_fastening(None) is None and parse_fastening({'enabled': False}) is None
    assert parse_fastening({'enabled': True}) == {
        'predrive': 'screwdrive_predrive', 'fasten': 'screwdrive_predrive_to_bolt',
        'compliant': True, 'retract': {}, 'mode': 'standard'}
    assert parse_fastening({'enabled': True, 'fasten_sequence': 'screwdrive_attach',
                            'compliant': False})['fasten'] == 'screwdrive_attach'


def test_the_screwdriver_sequences_are_checked_against_the_board_config():
    sdmod.check_sequences(['screwdrive_predrive', 'screwdrive_predrive_to_bolt'])
    with pytest.raises(ValueError, match='screwdrive_predrive'):      # lists what exists
        sdmod.check_sequences(['screwdrive_predriv'])


def test_a_dry_run_screwdriver_checks_names_then_only_logs():
    cfg = C.load('coupler_marker_assemble', ['robot.dry_run=true', 'fastening.enabled=true'])
    sd = sdmod.Screwdriver(cfg, sequences=('screwdrive_predrive',))
    assert sd.tc is None and sd.run('screwdrive_predrive') is True
    sd.close()
    with pytest.raises(ValueError, match='no sequence'):
        sdmod.Screwdriver(cfg, sequences=('screwdrive_nope',))


def test_stop_is_seen_by_the_running_sequence_as_a_keypress():
    class Keys:
        active = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def pressed(self):
            return False
    event = threading.Event()
    with sdmod._StopOrKey(event, Keys()) as w:
        assert not w.pressed()
        event.set()
        assert w.pressed()


def test_a_failed_sequence_returns_false_rather_than_raising():
    mtc, _ = sdmod._mtc()

    class Session:
        def execute(self, tokens):
            raise mtc.ToolChangerError('aborted at step 2')
    sd = sdmod.Screwdriver.__new__(sdmod.Screwdriver)
    sd.dry_run, sd.session, sd._stop = False, Session(), threading.Event()
    assert sd.run('screwdrive_predrive') is False


def _fastening_cycle(compliant=True):
    job = _cycle()
    job.fastening = parse_fastening({'enabled': True, 'compliant': compliant})
    return job


def test_the_fastening_steps_appear_only_when_it_is_on():
    off = _cycle()
    off.fastening = None
    assert off.steps_after_lift(None) == [] and off.steps_after_insertion(None) == []
    on = _fastening_cycle()
    assert [a.name for a in on.steps_after_lift(None)] == [
        'PREDRIVE the fastening screw (screwdrive_predrive)']
    assert [a.name for a in on.steps_after_insertion(None)] == [
        'FASTEN the screw home (screwdrive_predrive_to_bolt)']


class _Adm:
    S = np.array([3000.0, 3000.0, 3000.0, 4.0, 4.0, 4.0])
    delta = np.zeros(6)

    def __init__(self, trip_after=None):
        self.trip_after, self.holds = trip_after, 0

    def rebase(self, T):
        return T

    def warmup(self, T, tare_fn=None):
        pass

    def hold(self, T, seconds, guard=None, on_step=None):
        self.holds += 1
        time.sleep(0.005)
        if guard is not None and self.trip_after is not None and self.holds >= self.trip_after:
            return 'seated'
        return 'done'


class _ScrewArm:
    dry_run = False

    def __init__(self):
        self.stopped = 0

    def tcp_pose(self):
        return np.eye(4)

    def servo_stop(self):
        self.stopped += 1


class _Screw:
    """Turns for `seconds`, or until stop()."""

    def __init__(self, seconds=0.1):
        self.seconds, self.event, self.ran = seconds, threading.Event(), []

    def run(self, name):
        self.ran.append(name)
        return not self.event.wait(self.seconds)

    def stop(self):
        self.event.set()


class _Guard:
    max_force = 60.0

    def reset(self):
        pass

    def check(self):
        return False


def _fastener(compliant=True, trip_after=None, seconds=0.1):
    job = _fastening_cycle(compliant)
    job.adm_insert, job.guard = _Adm(trip_after), _Guard()
    job.robot = type('R', (), {'arm': _ScrewArm()})()
    job.T_tool0_coupler = np.eye(4)
    job.screwdriver = _Screw(seconds)
    return job


def test_fastening_holds_the_object_compliantly_until_the_screw_is_home():
    job = _fastener(seconds=0.1)
    assert job.fasten()
    assert job.screwdriver.ran == ['screwdrive_predrive_to_bolt']
    assert job.adm_insert.holds > 1, 'the law must keep servoing while the screw turns'
    assert job.robot.arm.stopped == 1, 'the servo session must end once it is done'


def test_a_guard_trip_while_fastening_stops_the_screwdriver_and_fails():
    job = _fastener(trip_after=3, seconds=5.0)
    t0 = time.time()
    assert not job.fasten()
    assert job.screwdriver.event.is_set(), 'the screwdriver was not asked to stop'
    assert time.time() - t0 < 2.0, 'it should stop promptly, not run the sequence out'
    assert job.robot.arm.stopped == 1


def test_non_compliant_fastening_just_runs_the_sequence():
    job = _fastener(compliant=False)
    assert job.fasten()
    assert job.adm_insert.holds == 0 and job.robot.arm.stopped == 0


def test_the_predrive_runs_its_own_sequence():
    job = _fastener()
    assert job.predrive()
    assert job.screwdriver.ran == ['screwdrive_predrive']


from urlab.apps.coupler_marker_assemble import fastening_retract_leg  # noqa: E402


def test_after_fastening_the_end_effector_retracts_200mm_along_its_own_minus_z():
    leg = fastening_retract_leg({}, _entry())
    assert leg['frame'] == 'coupler' and leg['distance_m'] == pytest.approx(0.2)
    assert np.allclose(leg['axis'], [0, 0, -1])
    # the withdraw backs off from the goal: 200 mm back along the assembled coupler's -z
    T = offset_pose(T_BASE_GOAL, leg)
    assert np.allclose(T[:3, 3], T_BASE_GOAL[:3, 3] - 0.2 * T_BASE_GOAL[:3, 2])
    assert np.allclose(T[:3, :3], T_BASE_GOAL[:3, :3])


def test_the_fastened_retract_reads_the_config_block_and_its_frames():
    cfg = C.load('coupler_marker_assemble', ['fastening.enabled=true'])
    leg = fastening_retract_leg(parse_fastening(cfg.section('fastening'))['retract'], _entry())
    assert leg['distance_m'] == pytest.approx(0.2) and leg['frame'] == 'coupler'
    assert fastening_retract_leg({'distance_mm': 150.0}, _entry())['distance_m'] == \
        pytest.approx(0.15)
    m = fastening_retract_leg({'axis': [0, 0, 1], 'frame': 'fixed_marker'}, _entry())
    assert m['frame'] == 'coupler' and m['distance_m'] == pytest.approx(0.2)
    with pytest.raises(ValueError, match='fastening.retract'):
        fastening_retract_leg({'axis': [0, 0, 0]}, _entry())


# ---------------------------------------------------------------------------- approach speed
def _speed_events(monkeypatch, answers, cfg_over=()):
    job, events = _stepping(monkeypatch, answers, ['approach_speed_scale=0.25'] + list(cfg_over))
    arm = job.robot.arm

    def set_scale(scale, phase=''):
        arm.speed_scale = scale
        events.append(('scale', scale))
    arm.set_speed_scale = set_scale
    return job, events


def test_only_the_approach_path_runs_at_the_lower_speed(monkeypatch):
    job, events = _speed_events(monkeypatch, [True] * 3)
    assert job.carry()
    moves_and_scales = [e for e in events if e[0] != 'ask']
    assert moves_and_scales == [('move', 'assembly joints'),             # normal speed
                                ('scale', 0.25),
                                ('move', 'approach 1/2: beside'),
                                ('move', 'approach 2/2: standoff'),
                                ('scale', 1.0)]                          # restored after
    assert job.robot.arm.speed_scale == 1.0


def test_the_speed_is_restored_even_when_the_approach_is_stopped(monkeypatch):
    job, events = _speed_events(monkeypatch, [True, True, False])        # q before waypoint 2
    assert not job.carry()
    assert ('scale', 0.25) in events and events[-1] == ('scale', 1.0)
    assert job.robot.arm.speed_scale == 1.0


def test_full_speed_changes_nothing_and_a_bad_scale_is_refused(monkeypatch):
    job, events = _stepping(monkeypatch, [True] * 3)                     # pinned at 1.0
    assert job.carry()
    assert job.robot.arm.scales == []
    with pytest.raises(ValueError, match='approach_speed_scale'):
        _cycle(cfg_over=['approach_speed_scale=-0.5'])._approach_speed_scale()
    assert C.load('coupler_marker_assemble').get('approach_speed_scale') is not None


# ---------------------------------------------------------------------------- joint move speed
from urlab.apps.coupler_marker_assemble import parse_joint_speed  # noqa: E402
from urlab.robot.arm import parse_limits  # noqa: E402


def test_the_joint_move_speed_block_parses_and_typos_are_refused():
    assert parse_joint_speed(None) is None and parse_joint_speed({}) is None
    caps = parse_joint_speed({'max_joint_velocity_deg_s': 60.0,
                              'max_cartesian_translation_mm_s': 250.0})
    jv, ja, cv, cr = parse_limits(caps, (0.1, 0.2, 0.3, 0.4))
    assert jv == pytest.approx(np.radians(60.0)) and cv == pytest.approx(0.25)
    assert (ja, cr) == (0.2, 0.4), 'absent keys inherit the global limits'
    with pytest.raises(ValueError, match='unknown'):
        parse_joint_speed({'max_joint_velocity_deg': 60.0})
    with pytest.raises(ValueError, match='positive'):
        parse_joint_speed({'max_joint_velocity_deg_s': 0.0})
    shipped = C.load('coupler_marker_assemble').section('joint_move_speed')
    assert shipped is None or parse_joint_speed(shipped) is not None or shipped == {}


def test_every_named_joint_move_carries_the_joint_speed(monkeypatch):
    fast = 'joint_move_speed={max_joint_velocity_deg_s: 60.0, max_cartesian_translation_mm_s: 250.0}'
    job = _cycle(_entry(view_joints=np.radians([10, 0, 0, 0, 0, 0])), [
        fast, 'pick_view_joints_deg=[20, 0, 0, 0, 0, 0]',
        'transition_joints_deg={fixed_view_to_pick_view: [[2, 0, 0, 0, 0, 0]]}'])
    job.transitions = parse_transitions(job.cfg['transition_joints_deg'])
    robot = _Robot()
    _locating(monkeypatch, job, robot)
    assert job.locate_fixed()
    assert len(robot.caps) == 3
    assert all(c == {'max_joint_velocity_deg_s': 60.0, 'max_cartesian_translation_mm_s': 250.0}
               for c in robot.caps)
    # ... and the assembly joints in the carry
    job2, _moved = _carrying(['assembly_joints_deg=[90, -80, 100, -110, -90, 0]', fast])
    assert job2.carry()
    assert job2.robot.caps == [{'max_joint_velocity_deg_s': 60.0,
                                'max_cartesian_translation_mm_s': 250.0}]


# ---------------------------------------------------------------------------- goal offset
from urlab.apps.coupler_marker_assemble import parse_goal_offset  # noqa: E402


def test_the_goal_offset_is_optional_and_typos_are_refused():
    assert parse_goal_offset(None) is None and parse_goal_offset({}) is None
    assert parse_goal_offset({'xyz_mm': [0, 0, 0], 'rpy_deg': [0, 0, 0]}) is None
    assert np.allclose(parse_goal_offset({'xyz_mm': [1, 2, 3]}), _pose([1, 2, 3], [0, 0, 0]))
    with pytest.raises(ValueError, match='unknown'):
        parse_goal_offset({'xyz': [1, 2, 3]})
    with pytest.raises(ValueError, match='three numbers'):
        parse_goal_offset({'rpy_deg': [1, 2]})
    parse_goal_offset(C.load('coupler_marker_assemble').section('goal_offset'))   # shipped loads


def test_the_offset_is_applied_in_the_goals_own_frame_and_the_path_follows(monkeypatch):
    job = _cycle(_entry(view_joints=None),
                 ['goal_offset={xyz_mm: [0, 0, -5], rpy_deg: [0, 0, 2]}'])
    _locating(monkeypatch, job, _Robot())
    assert job.locate_fixed()
    expected = T_BASE_GOAL @ _pose([0, 0, -5], [0, 0, 2])
    assert np.allclose(job.T_goal, expected)
    lin, _ang = pose_error(T_BASE_GOAL, job.T_goal)
    assert lin == pytest.approx(0.005), '5 mm back along the goal\'s own z'
    # the approach path is relative to the OFFSET goal
    wp, = _path({'xyz_mm': [-100, 0, 0]})
    assert np.allclose(coupler_waypoint_pose(job.T_goal, wp),
                       expected @ _pose([-100, 0, 0], [0, 0, 0]))


def test_no_offset_leaves_the_located_goal_untouched(monkeypatch):
    job = _cycle(_entry(view_joints=None))
    _locating(monkeypatch, job, _Robot())
    assert job.locate_fixed() and np.allclose(job.T_goal, T_BASE_GOAL)


# ---------------------------------------------------------------------------- bolt localization
import csv  # noqa: E402

from urlab.apps.coupler_marker_assemble import (parse_bolt_localization,  # noqa: E402
                                                rotation_taking_x_to, search_amplitude)


class _BoltWorld:
    """The admittance law AND the arm, over a surface with a hole in it.

    The reference drives a spring of stiffness k along the drop axis (z of the start pose); the
    surface stops the bolt `surface_mm` below the start, except within `half_mm` of `hole_mm`
    along x, where it can go `depth_mm` deeper. Force = k * how far the reference is past where
    the bolt is held."""
    rate = 125.0
    S = np.array([2000.0, 2000.0, 2000.0, 20.0, 20.0, 20.0])
    delta = np.zeros(6)
    dry_run = False
    # wiggle.run reads adm.arm.dry_run to choose a VIRTUAL clock, so the search runs at test
    # speed. (robot.arm is this object, whose own dry_run stays False so the search happens.)
    arm = type('VirtualClock', (), {'dry_run': True})()

    def __init__(self, T_start, surface_mm=3.0, hole_mm=4.0, half_mm=0.4, depth_mm=5.0,
                 k=2000.0):
        self.T0, self.T_ref = np.array(T_start), np.array(T_start)
        self.surface, self.hole, self.half, self.depth, self.k = (
            surface_mm / 1e3, hole_mm / 1e3, half_mm / 1e3, depth_mm / 1e3, k)
        self.ramps = self.holds = self.stopped = self.tared = 0

    # -- the law
    def reset(self):
        pass

    def rebase(self, T):
        return T

    def warmup(self, T, tare_fn=None):
        self.T_ref = np.array(T)
        if tare_fn:
            tare_fn()

    def ramp(self, T0, T1, duration, guard=None, on_step=None):
        self.ramps += 1
        self.T_ref = np.array(T1)
        return 'seated' if guard is not None and guard.check() else 'done'

    def hold(self, T, seconds, guard=None, on_step=None):
        self.holds += 1
        time.sleep(0.002)
        return self.ramp(T, T, seconds, guard)

    # -- the arm
    def _state(self):
        rel = self.T_ref[:3, 3] - self.T0[:3, 3]
        x, z = float(rel @ self.T0[:3, 0]), float(rel @ self.T0[:3, 2])
        limit = self.surface + (self.depth if abs(x - self.hole) <= self.half else 0.0)
        z_meas = min(z, limit)
        return x, z, z_meas

    def tcp_pose(self):
        x, _z, z_meas = self._state()
        T = np.array(self.T0)
        T[:3, 3] = self.T0[:3, 3] + x * self.T0[:3, 0] + z_meas * self.T0[:3, 2]
        return T

    def wrench(self):
        _x, z, z_meas = self._state()
        return np.concatenate([-self.k * (z - z_meas) * self.T0[:3, 2], np.zeros(3)])

    def servo_stop(self):
        self.stopped += 1

    def zero_ft(self, settle=True):
        self.tared += 1


_LEG = {'distance_m': 0.01, 'axis': np.array([-1.0, 0.0, 0.0]), 'frame': 'coupler'}
_FAST = dict(frequency_hz=1.0, amplitude_growth_mm=1.0, max_time_s=30.0)


def _bolt_job(world_kw=(), **params):
    job = _cycle()
    job.fastening = parse_fastening({'enabled': True, 'mode': 'bolt_localization'})
    job.bolt = parse_bolt_localization(dict(_FAST, **params))
    job.T_tool0_coupler, job.T_place, job.tare_before = np.eye(4), np.eye(4), True
    job.legs = {'assembly_standoff': _LEG}
    world = _BoltWorld(offset_pose(job.T_place, _LEG), **dict(world_kw))
    job.adm_bolt, job.guard = world, _Guard()
    job.robot = type('R', (), {'arm': world})()
    job.screwdriver, job.out_dir = _Screw(0.05), tempfile.mkdtemp()
    return job, world


def test_bolt_localization_is_off_by_default_and_its_block_validates():
    assert parse_fastening({'enabled': True})['mode'] == 'standard'
    with pytest.raises(ValueError, match='fastening.mode'):
        parse_fastening({'enabled': True, 'mode': 'bolt-localisation'})
    p = parse_bolt_localization({})
    assert (p['attach'], p['detach']) == ('screwdrive_attach', 'screwdrive_detach')
    assert np.allclose(p['drop_axis'], [0, 0, 1]) and np.allclose(p['search_axis'], [1, 0, 0])
    assert p['drop'] == pytest.approx(0.001) and p['press_force_n'] == 5.0
    for bad, match in (({'drop_mn': 1.0}, 'unknown'), ({'drop_mm': 0.0}, 'positive'),
                       ({'search_axis': [0, 0, 1]}, 'along drop_axis'),
                       ({'drop_axis': [0, 0, 0]}, 'non-zero')):
        with pytest.raises(ValueError, match=match):
            parse_bolt_localization(bad)
    parse_bolt_localization(C.load('coupler_marker_assemble').section('bolt_localization'))


def test_the_search_amplitude_grows_per_cycle_and_is_capped():
    p = parse_bolt_localization({'amplitude_start_mm': 1.0, 'amplitude_growth_mm': 2.0,
                                 'amplitude_max_mm': 6.0})
    assert [round(search_amplitude(n, p) * 1000, 6) for n in range(5)] == [1, 3, 5, 6, 6]


def test_the_search_wiggle_is_carried_onto_the_search_axis():
    for axis in ([1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]):
        R = rotation_taking_x_to(axis)
        unit = np.asarray(axis, float) / np.linalg.norm(axis)
        assert np.allclose(R[:3, 0], unit) and np.allclose(R[:3, :3].T @ R[:3, :3], np.eye(3))
        D = _pose([3, 0, 0], [0, 0, 0])                           # 3 mm along local x
        assert np.allclose((R @ D @ np.linalg.inv(R))[:3, 3], 0.003 * unit)


def test_bolt_mode_attaches_after_the_pick_and_searches_instead_of_inserting(monkeypatch):
    job, _world = _bolt_job()
    assert [a.name for a in job.steps_after_lift(None)] == [
        'ATTACH: drive the bolt fully out (screwdrive_attach)']
    assert job.steps_after_insertion(None) == [], 'the screwing happens inside the search step'
    called = []
    monkeypatch.setattr(job, 'localize_bolt', lambda: called.append(1) or True)
    assert job.set_down() and called == [1]


def test_the_search_finds_the_hole_then_detaches_and_attaches_under_the_same_law():
    job, world = _bolt_job()
    assert job.localize_bolt()
    assert world.tared == 1, 'tared at the end of the path, before anything touches'
    assert job.screwdriver.ran == ['screwdrive_detach', 'screwdrive_attach']
    assert world.holds > 0, 'the screwing must run under the SAME law, still servoing'
    x, _z, z_meas = world._state()
    assert abs(x - world.hole) <= world.half + 1e-9, 'the reference froze over the hole'
    assert z_meas > world.surface, 'the bolt is down in it'
    assert world.stopped >= 1
    rows = list(csv.DictReader(open(os.path.join(job.out_dir, 'bolt_localization.csv'))))
    assert rows[0]['phase'] == 'press' and rows[-1]['phase'] == 'search'
    assert float(rows[-1]['advance_mm']) >= 1.0


def test_no_hole_within_the_time_limit_fails_without_screwing():
    job, world = _bolt_job(world_kw={'hole_mm': 50.0}, max_time_s=5.0, amplitude_max_mm=3.0)
    assert not job.localize_bolt()
    assert job.screwdriver.ran == [] and world.stopped == 1


def test_nothing_to_press_on_fails_before_searching():
    job, world = _bolt_job(world_kw={'surface_mm': 500.0}, press_max_mm=10.0)
    assert not job.localize_bolt()
    assert job.screwdriver.ran == []
    rows = list(csv.DictReader(open(os.path.join(job.out_dir, 'bolt_localization.csv'))))
    assert {r['phase'] for r in rows} == {'press'}, 'it must not search without contact'
