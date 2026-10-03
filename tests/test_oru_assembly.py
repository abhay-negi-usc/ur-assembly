"""ORU ASSEMBLY -- the pick half is CouplerCycle (tested with coupler_pick_place); what is new is
the joining: plan first, pick, then kinematic_assembly's trajectory, and nothing released."""

import numpy as np
import pytest

from urlab import config as C
from urlab.apps import oru_assembly as oa
from urlab.skills import trajectory as traj


class _Arm:
    dry_run = True

    def __init__(self, reachable=True):
        self.reachable, self.moves = reachable, []

    def q(self):
        return [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

    def ik(self, T, qnear=None):
        return list(qnear if qnear is not None else self.q()) if self.reachable else None

    def move_j(self, q, label='move', **kw):
        self.moves.append(label)
        return True

    def wrench(self):
        return np.zeros(6)

    def zero_ft(self, **kw):
        pass

    def force_mode(self, *a, **kw):
        self.moves.append('force_mode')

    def end_force_mode(self):
        pass

    def add_guard(self, guard):
        pass

    def clear_guards(self):
        pass


def _robot(**kw):
    return type('R', (), {'arm': _Arm(**kw)})()


def test_the_config_carries_both_halves_and_latches_the_coupler():
    cfg = C.load('oru_assembly')
    for key in ('aruco', 'motion', 'compliance', 'compliance_loaded', 'mate_preload',
                'force_guard', 'marker_views', 'toolchanger'):
        assert cfg.section(key), key
    # WHERE IT GOES is the taught assembly, not a hand-entered pose
    assert cfg.get('assembly_name')
    assert cfg.get('assembled_pose') is None and cfg.get('held_object_pose') is None
    assert cfg.get_path('toolchanger.latch') is True, (
        'the run ends with the ORU held -- an unlatched close can drop it')
    assert cfg.get('trajectory_csv') == 'oru_assembly_trajectory.csv'


def test_the_trajectory_is_its_own_copy_and_ends_assembled():
    cfg = C.load('oru_assembly')
    mats = traj.load_csv(C.resolve(cfg, cfg['trajectory_csv']))
    assert len(mats) > 1 and np.allclose(mats[-1], np.eye(4)), 'the last row must be identity'


def test_the_assembly_half_is_kinematic_assemblys_sequence():
    cfg = C.load('oru_assembly', ['control_mode=position', 'disassemble_after=true',
                                  'settle_s=0.0', 'use_standoff=true'])
    robot = _robot()
    p = oa.plan(cfg, robot, np.eye(4))
    assert p is not None
    assert oa.assemble(cfg, robot, p, confirm=None)
    n = len(p['waypoint_q'])
    assert robot.arm.moves == (['stand-off'] + [f'waypoint {i}' for i in range(1, n + 1)]
                               + [f'disassemble {i}' for i in range(n - 2, -1, -1)]
                               + ['stand-off', 'home'])


def _no_coupler(monkeypatch):
    def no_coupler(*a, **kw):
        raise AssertionError('the coupler was opened before the plan was known to be good')
    monkeypatch.setattr(oa, 'Coupler', no_coupler)


def test_an_unreachable_trajectory_stops_the_run_before_anything_is_picked(monkeypatch):
    _no_coupler(monkeypatch)
    cfg = C.load('oru_assembly', ['object_name=mini_ORU', 'assembly_name=cleat'])
    assert oa.build_and_run(cfg, _robot(reachable=False), None, None) is False


def test_a_missing_assembly_stops_the_run_and_says_what_is_taught(monkeypatch, caplog):
    _no_coupler(monkeypatch)
    cfg = C.load('oru_assembly', ['object_name=mini_ORU', 'assembly_name=nope'])
    assert oa.build_and_run(cfg, _robot(), None, None) is False
    obj = {'assemblies': {'cleat': {}}}
    with pytest.raises(KeyError, match='cleat'):
        oa.resolve_assembly(cfg, 'mini_ORU', obj)


def test_the_path_is_anchored_on_the_taught_assembly_through_coupler_mate():
    """The last row lands the coupler frame exactly on the taught assembly: tool0 there is the
    assembly with coupler_mate undone."""
    from urlab import tool_frames
    from urlab.transforms import inverse, xyzrpy_to_matrix
    cfg = C.load('oru_assembly', ['settle_s=0.0', 'cleat_toolchanger.enabled=false'])
    T_asm = xyzrpy_to_matrix([-1.05, 0.36, 0.33], np.radians([-179.0, 0.8, -2.8]))
    seen = []
    robot = _robot()
    robot.arm.ik = lambda T, qnear=None: seen.append(np.array(T)) or [0.0] * 6
    assert oa.plan(cfg, robot, T_asm) is not None
    T_tool0_coupler = tool_frames.coupler_mate(cfg)
    assert np.allclose(seen[-1], T_asm @ inverse(T_tool0_coupler)), (
        'the final waypoint must put coupler_mate on the taught assembly')


def test_the_place_pose_is_the_pick_pose():
    job = oa.PickAndReturn.__new__(oa.PickAndReturn)
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    assert job._target_pose(T) is T
    assert issubclass(oa.PickAndReturn, oa.CouplerCycle)
    assert oa.PickAndReturn.TARGET_PRELOAD == 'place_preload'


# ---------------------------------------------------------------------------- the stand-off
def test_the_stand_off_is_only_for_a_trajectory_that_is_the_mate_alone():
    cfg = C.load('oru_assembly', ['use_standoff=auto'])
    assert oa.use_standoff(cfg, 1) is True and oa.use_standoff(cfg, 5) is False
    assert oa.use_standoff(C.load('oru_assembly', ['use_standoff=true']), 5) is True
    assert oa.use_standoff(C.load('oru_assembly', ['use_standoff=false']), 1) is False
    with pytest.raises(ValueError, match='use_standoff'):
        oa.use_standoff(C.load('oru_assembly', ['use_standoff=sometimes']), 3)


def test_without_a_stand_off_the_route_is_entered_and_left_at_its_first_row():
    cfg = C.load('oru_assembly', ['use_standoff=false', 'disassemble_after=true',
                                  'settle_s=0.0'])
    robot = _robot()
    p = oa.plan(cfg, robot, np.eye(4))
    assert p['q_standoff'] is None
    assert oa.assemble(cfg, robot, p, confirm=None)
    n = len(p['waypoint_q'])
    assert robot.arm.moves == ([f'waypoint {i}' for i in range(1, n + 1)]
                               + [f'disassemble {i}' for i in range(n - 2, -1, -1)] + ['home'])


def test_no_stand_off_and_no_disassembly_stops_at_the_mate():
    """A joint move home from the mate would drag the held part out of its fixture."""
    cfg = C.load('oru_assembly', ['use_standoff=false', 'disassemble_after=false',
                                  'settle_s=0.0'])
    robot = _robot()
    p = oa.plan(cfg, robot, np.eye(4))
    assert oa.assemble(cfg, robot, p, confirm=None)
    n = len(p['waypoint_q'])
    assert robot.arm.moves == [f'waypoint {i}' for i in range(1, n + 1)]


# ---------------------------------------------------------------------------- assembly preload
from urlab.transforms import xyzrpy_to_matrix  # noqa: E402


def test_the_preload_pushes_along_the_last_segment_of_the_route():
    rows = [xyzrpy_to_matrix([0.3, 0.2, 0.0], [0, 0, 0]),
            xyzrpy_to_matrix([-0.015, 0.0, 0.0], [0, 0, 0]), np.eye(4)]
    assert np.allclose(oa.insertion_back_axis(rows, [0, 0, -1]), [-1, 0, 0])   # push = +x
    # a tilted last row: the direction is read in ITS frame
    tilted = [xyzrpy_to_matrix([0, 0, -0.02], [0, 0, 0]),
              xyzrpy_to_matrix([0, 0, 0], np.radians([0, 90, 0]))]
    assert np.allclose(oa.insertion_back_axis(tilted, [0, 0, -1]), [1, 0, 0], atol=1e-9)
    assert np.allclose(oa.insertion_back_axis([np.eye(4)], [0, 0, -2]), [0, 0, -1])  # one row
    with pytest.raises(ValueError, match='same point'):
        oa.insertion_back_axis([np.eye(4), np.eye(4)], [0, 0, -1])


class _PreloadJob:
    def __init__(self):
        self.legs, self.calls, self._active_law = {}, [], None
        self.T_tool0_coupler = np.eye(4)
        self.guard = type('G', (), {'reset': lambda self: None})()
        self.adm_insert = type('A', (), {'rebase': lambda self, T: T, 'last_ref': None,
                                         'warmup': lambda self, T, tare_fn=None: None})()


def _preload_robot():
    arm = _Arm()
    arm.dry_run, arm.stopped = False, 0
    arm.tcp_pose = lambda: np.eye(4)
    arm.servo_stop = lambda: setattr(arm, 'stopped', arm.stopped + 1)
    return type('R', (), {'arm': arm})()


def test_the_preload_runs_at_the_mate_before_backing_out_and_a_miss_stops_there():
    cfg = C.load('oru_assembly', ['use_standoff=false', 'disassemble_after=true',
                                  'settle_s=0.0'])
    robot = _robot()
    p = oa.plan(cfg, robot, np.eye(4))
    n = len(p['waypoint_q'])
    assert oa.assemble(cfg, robot, p, confirm=None,
                       preload=lambda: robot.arm.moves.append('PRELOAD') or True)
    assert robot.arm.moves[n] == 'PRELOAD', 'right after the last waypoint'
    assert robot.arm.moves[n + 1].startswith('disassemble')
    robot2 = _robot()
    assert not oa.assemble(cfg, robot2, p, confirm=None, preload=lambda: False)
    assert not any(m.startswith('disassemble') or m == 'home' for m in robot2.arm.moves)


# ---------------------------------------------------------------------------- the put-back
def test_with_the_put_back_the_assembly_does_not_go_home():
    cfg = C.load('oru_assembly', ['use_standoff=false', 'disassemble_after=true',
                                  'settle_s=0.0'])
    robot = _robot()
    p = oa.plan(cfg, robot, np.eye(4))
    assert oa.assemble(cfg, robot, p, confirm=None, go_home=False)
    assert robot.arm.moves[-1] == 'disassemble 0' and 'home' not in robot.arm.moves


def test_the_put_back_uses_coupler_pick_places_place_preload():
    ours, theirs = C.load('oru_assembly'), C.load('coupler_pick_place')
    keys = ('enabled', 'force_n', 'max_travel_mm', 'step_mm', 'settle_s', 'persistence_s')
    assert {k: ours['place_preload'].get(k) for k in keys} == \
        {k: theirs['place_preload'].get(k) for k in keys}
    assert ours.get('return_to_pick') is True


def _steps(cfg_over):
    cfg = C.load('oru_assembly', ['robot.dry_run=true', 'toolchanger.enabled=false']
                 + list(cfg_over))
    cap = {}
    real = oa.bt.run_tree
    oa.bt.run_tree = lambda root, log: cap.setdefault('n', [c.name for c in root.children]) and True
    try:
        from urlab.robot import Robot
        robot = Robot(cfg, with_gripper=False)
        try:
            assert oa.build_and_run(cfg, robot, None, None)
        finally:
            robot.close()
    finally:
        oa.bt.run_tree = real
    return cap['n']


def test_the_run_ends_by_putting_the_part_back_where_it_was_picked():
    names = _steps(['cleat_toolchanger.enabled=false', 'return_to_pick=true',
                    'disassemble_after=true'])
    tail = names[names.index('assemble along the trajectory') + 1:]
    assert tail == ['carry back to the pick position', 'place the object', 'release the coupler',
                    'drop the payload', 'withdraw', 'switch the coupler motor off', 'report',
                    'move home']


def test_no_disassembly_means_no_put_back():
    for over in (['return_to_pick=false'], ['return_to_pick=true', 'disassemble_after=false']):
        names = _steps(['cleat_toolchanger.enabled=false'] + over)
        assert names[-1] == 'assemble along the trajectory', over


# ---------------------------------------------------------------------------- torque preload
def test_the_torque_preload_block_parses_and_refuses_typos():
    t = oa.parse_torque_preload({'torque_nm': 0.5, 'axis': [0, 0, 2]})
    assert t['torque_nm'] == 0.5 and np.allclose(t['axis'], [0, 0, 1])
    assert t['max_rotation'] == pytest.approx(np.radians(10.0))
    assert oa.parse_torque_preload(None) is None
    assert oa.parse_torque_preload({'enabled': False, 'torque_nm': 0.5}) is None
    for bad, match in (({'torque_nm': 0.5, 'axes': [0, 0, 1]}, 'unknown'),
                       ({'axis': [0, 0, 0]}, 'non-zero'), ({'torque_nm': 0.0}, 'positive')):
        with pytest.raises(ValueError, match=match):
            oa.parse_torque_preload(bad)
    oa.parse_torque_preload(C.load('oru_assembly').section('assembly_torque_preload'))


def test_rotation_about_turns_about_the_axis_without_moving():
    R = oa.rotation_about([0, 0, 1], np.radians(90))
    assert np.allclose(R[:3, :3] @ [1, 0, 0], [0, 1, 0]) and np.allclose(R[:3, 3], 0)
    assert np.allclose(oa.rotation_about([1, 0, 0], 0.0), np.eye(4))


# ---------------------------------------------------------------------------- simultaneous preload
class _ContactWorld:
    """Law + arm against a seat: past `stop_mm` along `d` a linear spring (k N/m), past `stop_deg`
    about `axis` a rotational one (kr Nm/rad). `couple` N per rad of twist RELIEVES the force --
    the coupling a contact off the mating point produces. Records every reference."""

    def __init__(self, d, axis, stop_mm=1.0, stop_deg=2.0, k=2000.0, kr=15.0, couple=0.0,
                 trip_after=None):
        self.d, self.axis = np.asarray(d, float), np.asarray(axis, float)
        self.stop, self.stop_r, self.k, self.kr = stop_mm / 1e3, np.radians(stop_deg), k, kr
        self.couple, self.trip_after = couple, trip_after
        self.T_ref, self.refs, self.last_ref = np.eye(4), [], None

    def ramp(self, T0, T1, duration, guard=None, on_step=None):
        self.T_ref = self.last_ref = np.array(T1)
        self.refs.append(self.T_ref)
        if on_step:
            on_step()
        return 'seated' if self.trip_after and len(self.refs) >= self.trip_after else 'done'

    def hold(self, T, seconds, guard=None, on_step=None):
        for _ in range(3):
            res = self.ramp(T, T, seconds / 3, guard, on_step)
        return res

    def travelled(self):
        return float(self.T_ref[:3, 3] @ self.d)

    def turned(self):
        R = self.T_ref[:3, :3]
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
        return float(np.arctan2(v @ self.axis, (np.trace(R) - 1.0) / 2.0))

    def tcp_pose(self):
        return self.T_ref

    def wrench_in(self, T):
        f = max(0.0, self.k * (self.travelled() - self.stop) - self.couple * self.turned())
        t = self.kr * max(0.0, self.turned() - self.stop_r)
        return np.concatenate([-f * self.d, -t * self.axis])      # the REACTION on the tool


def _combined(world, pre_over=None, tq_over=None, force=True, torque=True):
    job = type('J', (), {})()
    job.T_tool0_coupler, job.guard = np.eye(4), object()
    robot = type('R', (), {'arm': world})()
    pre = oa.parse_preload(dict({'force_n': 5.0, 'max_travel_mm': 10.0}, **(pre_over or {})),
                           'assembly_preload') if force else None
    tq = oa.parse_torque_preload(dict({'torque_nm': 0.5, 'axis': list(world.axis)},
                                      **(tq_over or {}))) if torque else None
    return oa.combined_preload(job, robot, world, pre, tq, np.eye(4), world.d)


def test_force_and_torque_advance_together_and_both_hold():
    world = _ContactWorld(d=[1, 0, 0], axis=[0, 1, 0])
    assert _combined(world)
    first = world.refs[0]
    assert first[0, 3] > 0 and abs(world.turned()) > 0
    R0 = first[:3, :3]
    assert not np.allclose(R0, np.eye(3)), 'the FIRST step must already both push AND twist'
    f, t = -world.wrench_in(None)[:3] @ world.d, -world.wrench_in(None)[3:] @ world.axis
    assert f >= 5.0 and t >= 0.5
    assert world.travelled() == pytest.approx(0.001 + 5.0 / 2000.0, abs=0.0006)
    assert np.degrees(world.turned()) == pytest.approx(2.0 + np.degrees(0.5 / 15.0), abs=0.6)


def test_a_twist_that_relieves_the_force_is_answered_by_pushing_more():
    """Coupled: every degree of twist takes 0.2 N off the push. The force must keep advancing
    until BOTH are met at once -- the gap the sequential version left unchecked."""
    world = _ContactWorld(d=[1, 0, 0], axis=[0, 1, 0], couple=0.2 * 180 / np.pi)
    assert _combined(world)
    f = -world.wrench_in(None)[:3] @ world.d
    assert f >= 5.0
    uncoupled = 0.001 + 5.0 / 2000.0
    assert world.travelled() > uncoupled, 'it had to push further to make up what the twist took'


def test_either_alone_works_through_the_same_loop():
    w1 = _ContactWorld(d=[1, 0, 0], axis=[0, 1, 0])
    assert _combined(w1, torque=False) and w1.turned() == pytest.approx(0.0)
    w2 = _ContactWorld(d=[1, 0, 0], axis=[0, 1, 0])
    assert _combined(w2, force=False) and w2.travelled() == pytest.approx(0.0)


def test_running_out_of_either_travel_or_a_trip_fails():
    no_stop = _ContactWorld(d=[1, 0, 0], axis=[0, 1, 0], stop_deg=90.0)
    assert not _combined(no_stop, tq_over={'max_rotation_deg': 5.0})
    assert np.degrees(no_stop.turned()) == pytest.approx(5.0)
    far = _ContactWorld(d=[1, 0, 0], axis=[0, 1, 0], stop_mm=50.0)
    assert not _combined(far, pre_over={'max_travel_mm': 8.0})
    tripping = _ContactWorld(d=[1, 0, 0], axis=[0, 1, 0], trip_after=3)
    assert not _combined(tripping)


def test_the_mate_preload_runs_combined_in_one_session(monkeypatch):
    cfg = C.load('oru_assembly', ['assembly_preload.enabled=true', 'assembly_preload.force_n=5.0',
                                  'assembly_torque_preload={torque_nm: 0.5, axis: [0, 1, 0]}'])
    job, robot = _PreloadJob(), _preload_robot()
    seen = []
    monkeypatch.setattr(oa, 'combined_preload',
                        lambda j, r, adm, pre, tq, T_at, d: seen.append((pre['force_n'],
                                                                         tq['torque_nm'],
                                                                         list(d))) or True)
    assert oa.preload_at_mate(cfg, robot, job, {'insert_back_axis': np.array([-1.0, 0.0, 0.0])})
    assert seen == [(5.0, 0.5, [1.0, 0.0, 0.0])] and robot.arm.stopped == 1
    assert job._active_law is None
    off = C.load('oru_assembly', ['assembly_preload.enabled=false',
                                  'assembly_torque_preload.enabled=false'])
    job2, robot2 = _PreloadJob(), _preload_robot()
    seen.clear()
    assert oa.preload_at_mate(off, robot2, job2, {'insert_back_axis': np.array([-1.0, 0, 0])})
    assert seen == [] and robot2.arm.stopped == 0


# ---------------------------------------------------------------------------- the cleat mount
def test_with_the_cleat_the_part_is_clamped_then_released_then_the_arm_retracts():
    names = _steps(['cleat_toolchanger.enabled=true', 'return_to_pick=true',
                    'exit_along_route=true'])
    assert 'power and OPEN the cleat' not in names, 'it opens INSIDE the trajectory, not up front'
    tail = names[names.index('assemble along the trajectory') + 1:]
    assert tail == ['CLAMP the cleat onto the part', 'release the coupler', 'drop the payload',
                    'retract from the mounted part', 'switch the coupler motor off',
                    'retrace the route out (raised)', 'move home'], (
        'clamp BEFORE release, and no put-back once it is mounted')


def test_the_mount_ends_the_assembly_at_the_mate_after_the_preload():
    cfg = C.load('oru_assembly', ['use_standoff=false', 'disassemble_after=true',
                                  'settle_s=0.0'])
    robot = _robot()
    p = oa.plan(cfg, robot, np.eye(4))
    assert oa.assemble(cfg, robot, p, confirm=None, end_at_mate=True,
                       preload=lambda: robot.arm.moves.append('PRELOAD') or True)
    n = len(p['waypoint_q'])
    assert robot.arm.moves == [f'waypoint {i}' for i in range(1, n + 1)] + ['PRELOAD']


class _Cleat:
    latched, manual = True, False

    def __init__(self, hold=True, verify=True, prepare=True):
        self._hold, self._verify, self._prepare, self.calls = hold, verify, prepare, []

    def hold(self):
        self.calls.append('hold')
        return self._hold

    def verify(self):
        self.calls.append('verify')
        return self._verify

    def prepare_to_mate(self):
        self.calls.append('open')
        return self._prepare


def test_the_clamp_is_rechecked_and_a_refusal_keeps_the_part_on_the_coupler():
    assert oa.clamp_cleat(_Cleat(), 'mini_ORU')
    assert oa.clamp_cleat(_Cleat(verify=None), 'mini_ORU'), 'unverified is a warning, not a stop'
    refused = _Cleat(hold=False)
    assert not oa.clamp_cleat(refused, 'mini_ORU') and refused.calls == ['hold']
    assert not oa.clamp_cleat(_Cleat(verify=False), 'mini_ORU')
    assert oa.open_cleat(_Cleat()) and not oa.open_cleat(_Cleat(prepare=False))


def test_the_retract_backs_out_of_the_seat_from_where_the_arm_is():
    seen = []
    job = type('J', (), {})()
    job._compliant = lambda T_from, T_to, what, in_contact=False: seen.append(
        (np.array(T_from), np.array(T_to), in_contact)) or True
    T_seat = np.diag([1.0, -1.0, -1.0, 1.0])
    leg = {'distance_m': 0.4, 'axis': np.array([0.0, 0.0, -1.0]), 'frame': 'coupler'}
    assert oa.retract_from_seat(job, {'T_seat': T_seat, 'retract_leg': leg})
    T_from, T_to, in_contact = seen[0]
    assert in_contact, 'it starts pressed in by the preload'
    assert np.allclose(T_to[:3, 3], T_seat[:3, 3] - 0.4 * T_seat[:3, 2]), '400 mm out along -z'


def test_the_cleat_opens_right_before_the_last_waypoint():
    cfg = C.load('oru_assembly', ['use_standoff=false', 'settle_s=0.0'])
    robot = _robot()
    p = oa.plan(cfg, robot, np.eye(4))
    n = len(p['waypoint_q'])
    assert oa.assemble(cfg, robot, p, confirm=None, end_at_mate=True,
                       before_last=lambda: robot.arm.moves.append('OPEN CLEAT') or True)
    assert robot.arm.moves == ([f'waypoint {i}' for i in range(1, n)] + ['OPEN CLEAT']
                               + [f'waypoint {n}'])


def test_a_cleat_that_will_not_open_stops_short_of_the_seat():
    cfg = C.load('oru_assembly', ['use_standoff=false', 'settle_s=0.0'])
    robot = _robot()
    p = oa.plan(cfg, robot, np.eye(4))
    n = len(p['waypoint_q'])
    assert not oa.assemble(cfg, robot, p, confirm=None, end_at_mate=True,
                           before_last=lambda: False)
    assert robot.arm.moves == [f'waypoint {i}' for i in range(1, n)], 'no final move'


def test_a_cleat_never_opened_is_never_clamped():
    cleat = _Cleat()
    assert not oa.clamp_cleat(cleat, 'mini_ORU', opened=False) and cleat.calls == []


# ---------------------------------------------------------------------------- home
def test_home_is_the_start_pose_unless_pinned():
    q0 = [0.1, -1.5, 1.4, -1.6, -1.5, 0.2]
    cfg = C.load('oru_assembly', ['home_joints_deg=null'])
    assert np.allclose(oa.home_joints(cfg, q0), q0)
    pinned = C.load('oru_assembly', ['home_joints_deg=[0, -90, 90, -90, -90, 0]'])
    assert np.allclose(np.degrees(oa.home_joints(pinned, q0)), [0, -90, 90, -90, -90, 0])
    with pytest.raises(ValueError, match='six'):
        oa.home_joints(C.load('oru_assembly', ['home_joints_deg=[1, 2, 3]']), q0)


def test_the_plan_carries_the_home_pose():
    robot = _robot()
    p = oa.plan(C.load('oru_assembly', ['home_joints_deg=null']), robot, np.eye(4))
    assert np.allclose(p['q_home'], robot.arm.q())


# ---------------------------------------------------------------------------- the way out
def test_the_way_out_is_the_route_reversed_and_raised_by_the_retract():
    rows = [xyzrpy_to_matrix([0.5, 0.2, -0.1], [0, 0, 0]),
            xyzrpy_to_matrix([0.5, 0.2, 0.0], [0, 0, 0]),
            xyzrpy_to_matrix([-0.015, 0.0, 0.0], [0, 0, 0]), np.eye(4)]
    lift = np.array([0.0, 0.0, 0.4])
    out = oa.raised_route(np.eye(4), rows, lift, np.eye(4))
    assert len(out) == 3, 'the seat itself is left out -- the retract is the raised seat'
    for T, row in zip(out, reversed(rows[:-1])):
        assert np.allclose(T[:3, 3], row[:3, 3] + lift)


def test_the_way_out_is_planned_before_the_pick_and_can_be_switched_off():
    on = C.load('oru_assembly', ['cleat_toolchanger.enabled=true', 'exit_along_route=true',
                                 'use_standoff=false'])
    robot = _robot()
    p = oa.plan(on, robot, np.eye(4))
    n_rows = len(p['waypoint_q'])
    assert len(p['exit_q']) == n_rows - 1
    off = C.load('oru_assembly', ['cleat_toolchanger.enabled=true', 'exit_along_route=false',
                                  'use_standoff=false'])
    assert oa.plan(off, _robot(), np.eye(4))['exit_q'] == []
    assert oa.plan(C.load('oru_assembly', ['cleat_toolchanger.enabled=false']), _robot(),
                   np.eye(4))['exit_q'] == []


def test_an_unreachable_way_out_stops_the_plan():
    cfg = C.load('oru_assembly', ['cleat_toolchanger.enabled=true', 'exit_along_route=true',
                                  'use_standoff=false'])
    robot = _robot()
    calls = {'n': 0}
    n_rows = len(oa.plan(cfg, _robot(), np.eye(4))['waypoint_q'])

    def ik(T, qnear=None):
        calls['n'] += 1
        return None if calls['n'] > n_rows else [0.0] * 6     # the route fine, the exit not
    robot.arm.ik = ik
    assert oa.plan(cfg, robot, np.eye(4)) is None


def test_the_exit_moves_run_in_order_and_a_decline_stops_them():
    moves = []
    robot = type('R', (), {})()
    robot.move_joints = lambda q, label='', guard=None: moves.append(label) or True
    p = {'exit_q': [[1] * 6, [2] * 6, [3] * 6]}
    assert oa.exit_along_route(robot, p, None, None)
    assert moves == ['exit 1/3: waypoint 3, raised', 'exit 2/3: waypoint 2, raised',
                     'exit 3/3: waypoint 1, raised']
    moves.clear()
    answers = iter([True, False])
    assert not oa.exit_along_route(robot, p, lambda label: next(answers), None)
    assert moves == ['exit 1/3: waypoint 3, raised']


def test_without_the_way_out_the_mount_goes_straight_home():
    names = _steps(['cleat_toolchanger.enabled=true', 'exit_along_route=false'])
    assert names[-2:] == ['switch the coupler motor off', 'move home']


def test_the_mount_retract_is_its_own_leg_and_defaults_to_final_retract():
    cfg = C.load('oru_assembly', ['mount_retract={distance_mm: 200.0, axis: [0, 0, -1], '
                                  'frame: coupler}'])
    assert oa.mount_retract_leg(cfg)['distance_m'] == pytest.approx(0.2)
    fallback = C.load('oru_assembly', ['mount_retract=null'])
    assert oa.mount_retract_leg(fallback)['distance_m'] == pytest.approx(
        fallback['motion']['final_retract']['distance_mm'] / 1000.0)


def test_a_retract_past_the_reach_stops_the_plan_and_says_so(caplog):
    cfg = C.load('oru_assembly', ['cleat_toolchanger.enabled=true', 'use_standoff=false'])
    n_rows = len(oa.plan(cfg, _robot(), np.eye(4))['waypoint_q'])
    robot = _robot()
    calls = {'n': 0}

    def ik(T, qnear=None):
        calls['n'] += 1
        return None if calls['n'] == n_rows + 1 else [0.0] * 6   # the retract pose fails
    robot.arm.ik = ik
    with caplog.at_level('ERROR'):
        assert oa.plan(cfg, robot, np.eye(4)) is None
    assert 'mount_retract' in caplog.text

