"""DISASSEMBLE TILE, without a robot, a camera or a screwdriver.

The failures worth catching: a pull along the wrong axis, a pull that tripped the guard being
reported as a disassembly, the steps out of order (unscrewing before the lock, pulling before the
unscrew), and a shipped config that drops the tile at exit or pushes on it as hard as an assembly.
"""

import numpy as np
import pytest

from urlab import config as C
from urlab.apps.disassemble_tile import DisassembleCycle
from urlab.transforms import translation_matrix, xyzrpy_to_matrix

T_PICK = xyzrpy_to_matrix([0.5, -0.2, 0.3], np.radians([180.0, 0.0, 30.0]))


def _job(cfg_over=()):
    job = DisassembleCycle.__new__(DisassembleCycle)
    job.cfg = C.load('disassemble_tile', list(cfg_over))
    job.name = 'tile_1'
    job.legs = {n: job._parse_leg(n, job.cfg.section('motion').get(n)) for n in job.LEGS}
    job.detach_seq = 'screwdrive_detach'
    job.T_tool0_coupler = np.eye(4)
    job.T_pick = T_PICK
    return job


class _Guard:
    max_force, max_torque = 15.0, 4.0

    def __init__(self, trip=None):
        self.trip = trip
        self.tripped_by = None

    def reset(self):
        self.tripped_by = None


class _Arm:
    dry_run = True

    def __init__(self, T_tcp):
        self.T_tcp = T_tcp

    def tcp_pose(self):
        return self.T_tcp


def test_the_pull_defaults_to_100_mm_along_the_end_effectors_minus_x():
    job = _job()
    leg = job._parse_leg('disassemble', None)
    assert leg['distance_m'] == pytest.approx(0.1)
    assert np.allclose(leg['axis'], [-1, 0, 0]) and leg['frame'] == 'coupler'
    assert job._parse_leg('disassemble', {'distance_mm': 50.0, 'axis': [0, 1, 0]})['axis'][1] == 1
    # the mate standoff keeps the coupler_pick_place default, straight back along -z
    assert np.allclose(job._parse_leg('mate_standoff', None)['axis'], [0, 0, -1])


def test_the_pull_off_pose_is_the_grasp_moved_along_its_own_minus_x():
    job = _job()
    T = job._target_pose(T_PICK)
    assert np.allclose(T, T_PICK @ translation_matrix([-0.1, 0.0, 0.0]))
    assert np.allclose(T[:3, :3], T_PICK[:3, :3]), 'a pull does not rotate'


def test_the_steps_lock_and_take_the_payload_before_unscrewing_then_pull():
    job = _job()
    job.robot = type('R', (), {'arm': _Arm(T_PICK)})()
    names = [s.name for s in job.steps(None)]
    assert names == ['locate the tile', 'power and open the coupler', 'before the grasp',
                     'align at the mate standoff', 'mate with the tile', 'lock the coupler',
                     'take the payload', 'unscrew the tile (screwdrive_detach)',
                     'pull the tile off', 'report']


def _pull(job, trip, end):
    """disassemble() with _compliant replaced by a move that ends at `end` and trips or not."""
    job.guard = _Guard()
    job.robot = type('R', (), {'arm': _Arm(T_PICK)})()
    seen = {}

    def fake_compliant(T_from, T_to, what, in_contact=False, **kw):
        seen.update(T_to=T_to, in_contact=in_contact)
        job.guard.reset()
        job.guard.tripped_by = trip
        job.robot.arm.T_tcp = end
        return True
    job._compliant = fake_compliant
    return job.disassemble(), seen


def test_a_clean_pull_succeeds_and_starts_from_where_the_arm_is():
    job = _job()
    ok, seen = _pull(job, None, T_PICK @ translation_matrix([-0.1, 0, 0]))
    assert ok
    assert seen['in_contact'] is True, 'the tile is still seated: no tare, start from reality'
    assert np.allclose(seen['T_to'], job._target_pose(T_PICK))
    assert job._pulled_mm() == pytest.approx(100.0)


def test_a_guard_trip_during_the_pull_is_a_failure_not_contact():
    job = _job()
    ok, _seen = _pull(job, 'force 16.0 N >= 15.0 N', T_PICK @ translation_matrix([-0.012, 0, 0]))
    assert ok is False
    assert job._pulled_mm() == pytest.approx(12.0)


def test_a_dry_run_unscrew_runs_the_detach_sequence_once():
    job = _job()
    job.robot = type('R', (), {'arm': _Arm(T_PICK)})()
    ran = []
    job.screwdriver = type('S', (), {'run': lambda self, seq: ran.append(seq) or True})()
    assert job.detach()
    assert ran == ['screwdrive_detach']


def test_the_shipped_config_is_gentle_and_keeps_hold_of_the_tile():
    cfg = C.load('disassemble_tile')
    asm = C.load('coupler_marker_assemble')
    assert cfg.get_path('toolchanger.latch') is True, 'the run ends holding the tile'
    assert cfg.get_path('mate_preload.force_n') < asm.get_path('mate_preload.force_n')
    assert cfg.get_path('force_guard.max_force_n') < asm.get_path('force_guard.max_force_n')
    assert cfg.get_path('screwdriver.detach_sequence') == 'screwdrive_detach'
    from urlab.robot.screwdriver import check_sequences
    check_sequences(['screwdrive_detach'])


def test_the_shipped_config_wiggles_the_mate_by_5_mm_across_the_axis():
    from urlab.apps.coupler_pick_place import parse_mate_wiggle
    cfg = C.load('disassemble_tile')
    approach_s = cfg.get_path('motion.mate_standoff.distance_mm') / cfg.get('compliant_speed_mm_s')
    mw = parse_mate_wiggle(cfg.section('mate_wiggle'), 125.0, approach_s)
    assert mw is not None, 'mate_wiggle is enabled for the disassembly'
    assert mw['wiggle'].amp == [5.0, 5.0, 0.0, 0.0, 0.0, 0.0]
