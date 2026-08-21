"""Offline tests for the behavior-tree layer (urlab/behaviors) and the shared app helpers
(urlab/apps/_common).  No robot, no camera -- pure control-flow and math."""

import numpy as np

from urlab import behaviors as bt
from urlab.apps._common import fmt_dur, seg_time
from urlab.transforms import translation_matrix, xyzrpy_to_matrix


def test_action_semantics():
    """An Action leaf FAILS only on an explicit False -- None (a do-and-return-nothing step)
    counts as success, matching the StepRunner contract the apps migrated from."""
    ran = []
    assert bt.run_tree(bt.Action('none is ok', lambda: ran.append(1)))
    assert bt.run_tree(bt.Action('true is ok', lambda: True))
    assert not bt.run_tree(bt.Action('false fails', lambda: False))
    assert ran == [1]


def test_sequence_stops_at_first_failure():
    ran = []
    root = bt.sequence(
        'seq',
        bt.Action('a', lambda: ran.append('a')),
        bt.Action('b', lambda: False),
        bt.Action('c', lambda: ran.append('c')))
    assert not bt.run_tree(root)
    assert ran == ['a'], 'the step after a failure must not run'


def test_selector_skips_the_rest_once_a_child_succeeds():
    """The skip-if-done idiom calibration_check uses: a Check that succeeds short-circuits
    the work child."""
    ran = []
    root = bt.selector('or', bt.Check('already done', lambda: True),
                       bt.Action('work', lambda: ran.append(1)))
    assert bt.run_tree(root) and ran == []
    root = bt.selector('or', bt.Check('not done', lambda: False),
                       bt.Action('work', lambda: ran.append(1)))
    assert bt.run_tree(root) and ran == [1]


def test_retry_reruns_a_failing_child():
    calls = []

    def flaky():
        calls.append(1)
        return len(calls) >= 3

    assert bt.run_tree(bt.retry(bt.Action('flaky', flaky), 5))
    assert len(calls) == 3
    calls.clear()
    assert not bt.run_tree(bt.retry(bt.Action('flaky', lambda: (calls.append(1), False)[1]), 2))
    assert len(calls) == 2


def test_confirm_gate_blocks_a_declined_step():
    ran = []
    root = bt.Action('gated', lambda: ran.append(1), confirm=lambda label: False)
    assert not bt.run_tree(root) and ran == []
    root = bt.Action('gated', lambda: ran.append(1), confirm=lambda label: True)
    assert bt.run_tree(root) and ran == [1]


def test_library_lookup():
    """The behavior dictionary: every entry is constructible by name, and an unknown name
    refuses with the known names in the message."""
    leaf = bt.make('say', 'hello')
    assert bt.run_tree(leaf)
    assert bt.make('action', 'x', lambda: True).name == 'x'
    try:
        bt.make('teleport')
    except KeyError as exc:
        assert 'teleport' in str(exc) and 'say' in str(exc)
    else:
        raise AssertionError('unknown behavior names must be rejected')


def test_seg_time_paces_both_translation_and_rotation():
    A = np.eye(4)
    B = translation_matrix([0.010, 0.0, 0.0])              # 10 mm
    assert abs(seg_time(A, B, 5.0) - 2.0) < 1e-6           # 10 mm at 5 mm/s
    C = xyzrpy_to_matrix([0, 0, 0], [0, 0, np.radians(10)])
    assert abs(seg_time(A, C, 5.0, 2.0) - 5.0) < 1e-3      # 10 deg at 2 deg/s dominates
    assert seg_time(A, A, 5.0, min_s=0.008) == 0.008       # never shorter than a servo cycle


def test_fmt_dur():
    assert fmt_dur(62) == '1:02'
    assert fmt_dur(3661) == '1:01:01'
    assert fmt_dur(-5) == '0:00'


def test_gripper_repl_grammar():
    from urlab.apps.gripper_control import parse_command
    assert parse_command(' Q ', 0, 255) == ('quit', None)
    assert parse_command('open', 3, 250) == ('move', 3)
    assert parse_command('close', 3, 250) == ('move', 250)
    assert parse_command('999', 0, 255) == ('move', 255), 'targets clamp to the count range'
    assert parse_command('-1', 0, 255) == ('move', 0)
    assert parse_command('fault', 0, 255) == ('clear', None)
    assert parse_command('wat', 0, 255) == ('help', None)


def test_calibration_check_retract_modes():
    from urlab.apps.calibration_check import _parse_retract

    class FakeCfg(dict):
        def get(self, k, d=None):
            return super().get(k, d)

    assert _parse_retract(FakeCfg()) == 'auto'
    assert _parse_retract(FakeCfg(retract=True)) == 'auto'
    assert _parse_retract(FakeCfg(retract=False)) == 'never'
    assert _parse_retract(FakeCfg(retract=' Prompt ')) == 'prompt'
    assert _parse_retract(FakeCfg(retract='sideways')) is None
