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


# ---- the Robot class: frame registry + motion primitives (dry-run arm, poses injected) ----

def _dry_robot():
    from urlab import config as urconfig
    from urlab.robot import Robot
    cfg = urconfig.load('cartesian')
    cfg.set_path('robot.dry_run', True)
    return Robot(cfg, with_gripper=False)


def _pin_flange(robot, T):
    """Pin the live base_link->tool0 edge (and tool0()) to a known pose for deterministic
    math."""
    robot.frames.set_live(robot.base_frame, robot.tip_frame, lambda: T)
    robot.arm.tcp_pose = lambda: T


def test_frame_registry_resolves_chains_in_any_reference():
    from urlab.transforms import inverse, xyzrpy_to_matrix

    r = _dry_robot()
    try:
        Tp = xyzrpy_to_matrix([0.4, 0.1, 0.3], [0.0, 0.0, np.pi / 2])
        _pin_flange(r, Tp)
        Tt = translation_matrix([0.0, 0.0, 0.1])
        Tf = translation_matrix([1.0, 0.0, 0.0])
        Ts = translation_matrix([0.0, 0.2, 0.0])
        r.register_frame('tip2', Tt)                        # rides on the arm
        r.register_frame('fixture', Tf, parent='base_link')  # bolted to the world
        r.register_frame('slot', Ts, parent='fixture')       # chains recursively

        assert np.allclose(r.pose('slot'), Tf @ Ts)
        assert np.allclose(r.pose('tip2'), Tp @ Tt), 'live FK must be on the path'
        assert np.allclose(r.pose('tip2', reference='fixture'), inverse(Tf) @ Tp @ Tt)
        assert r.is_tool_attached('tip2') and not r.is_tool_attached('slot')

        # 6-vector poses [xyz m, rpy rad] are accepted; targets live in their own registry
        r.register_frame('six', [0.0, 0.0, 0.05, 0.0, 0.0, 0.0], parent='base_link')
        assert np.allclose(r.pose('six'), translation_matrix([0.0, 0.0, 0.05]))
        r.register_target('slot', translation_matrix([2.0, 0.0, 0.0]))
        assert np.allclose(r.target('slot')[:3, 3], [2.0, 0.0, 0.0])

        # built-in frames refuse re-registration (a tool0->tool0 entry once hung the walk)
        for builtin in ('tool0', 'base_link'):
            try:
                r.register_frame(builtin, np.eye(4))
            except ValueError:
                continue
            raise AssertionError(f'{builtin} must refuse registration')
        try:
            r.register_frame('orphan', np.eye(4), parent='never_registered')
        except KeyError:
            pass
        else:
            raise AssertionError('an unknown parent must be rejected')
    finally:
        r.close()


def test_move_cartesian_back_solves_the_flange_pose():
    from urlab.transforms import inverse

    r = _dry_robot()
    try:
        Tp = translation_matrix([0.4, 0.0, 0.3])
        _pin_flange(r, Tp)
        Tt = translation_matrix([0.0, 0.0, 0.1])
        Tf = translation_matrix([1.0, 0.0, 0.0])
        r.register_frame('tip2', Tt)
        r.register_frame('fixture', Tf, parent='base_link')
        sent = []
        r.arm.move_l = lambda T, label='': (sent.append(np.array(T)), True)[1]

        target = translation_matrix([0.05, 0.0, 0.0])
        assert r.move_cartesian(target, frame='tip2', reference='fixture',
                                interpolation='lin')
        assert np.allclose(sent[-1], Tf @ target @ inverse(Tt)), \
            'T_base_tool0 = T_base_ref @ target @ inv(T_tool0_frame)'

        # a world-fixed frame cannot be the MOVING frame
        try:
            r.move_cartesian(np.eye(4), frame='fixture', interpolation='lin')
        except ValueError:
            pass
        else:
            raise AssertionError('moving a world-fixed frame must be refused')

        # ptp seeds and re-seeds the IK branch
        r.arm.ik = lambda T, qnear=None: [0.1, 0.2, 0.3, 0.4, 0.5, qnear is not None]
        r.arm.move_j = lambda q, label='': True
        seed = {}
        assert r.move_cartesian(target, interpolation='ptp', seed=seed)
        assert seed['q'][-1] is False        # first solve had no seed
        assert r.move_cartesian(target, interpolation='ptp', seed=seed)
        assert seed['q'][-1] is True         # second solve was seeded
    finally:
        r.close()


def test_move_relative_expresses_the_delta_in_the_named_frame():
    from urlab.transforms import inverse, xyzrpy_to_matrix

    r = _dry_robot()
    try:
        Tp = xyzrpy_to_matrix([0.4, 0.0, 0.3], [np.pi, 0.0, 0.0])
        _pin_flange(r, Tp)
        Toff = translation_matrix([0.0, 0.0, 0.02])
        Tf = xyzrpy_to_matrix([1.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2])
        r.register_frame('probe', Toff)
        r.register_frame('fixture', Tf, parent='base_link')
        sent = []
        r.arm.move_l = lambda T, label='': (sent.append(np.array(T)), True)[1]
        D = translation_matrix([0.0, 0.0, 0.03])

        assert r.move_relative(D, expressed_in='probe')
        assert np.allclose(sent[-1], Tp @ Toff @ D @ inverse(Toff)), \
            'a tool-frame jog post-multiplies about that frame'
        assert r.move_relative(D, expressed_in='fixture')
        assert np.allclose(sent[-1], Tf @ D @ inverse(Tf) @ Tp), \
            "a world-frame jog uses the reference frame's axes with the arm as the pivot"
        assert r.move_relative(D, expressed_in='base_link')
        assert np.allclose(sent[-1], D @ Tp)
    finally:
        r.close()


def test_chain_builds_from_specs_and_nodes():
    ran = []
    root = bt.chain(
        'mixed',
        bt.Action('inline node', lambda: ran.append('node')),
        ('say', 'plain spec'),
        ('action', 'kwargs spec', lambda: ran.append('kw'), {}),
    )
    assert bt.run_tree(root)
    assert ran == ['node', 'kw']
    try:
        bt.chain('bad', 42)
    except TypeError:
        pass
    else:
        raise AssertionError('a non-spec step must be rejected')


# ---- marker localization: certainty weighting, the standoff cap, and the servo loop ----

class _MarkerPlan:
    """The ViewPlan fields the pure fusion functions read."""
    min_views, min_markers, require_all = 2, 1, False
    max_view_spread_mm = max_view_spread_deg = None
    max_disagreement_mm = max_disagreement_deg = None
    view_weight_power = 2.0
    max_camera_distance_m = 0.5
    settle_s = 0.0
    frames_per_view = 1


def test_fusion_weights_closer_views_more():
    """A close view must dominate a far one (weight d^-2), and a view beyond the 500 mm
    standoff cap must be dropped outright."""
    from urlab.skills import marker_localize as mloc

    T_true = translation_matrix([1.0, 0.0, 0.0])
    T_off = translation_matrix([1.0, 0.010, 0.0])           # a 10 mm-wrong far view
    fused = mloc.fuse_markers({7: [(T_true, 0.1), (T_off, 0.4)]}, _MarkerPlan)
    err_mm = abs(fused[7][0][1, 3]) * 1000.0
    # weights 100 : 6.25 -> the wrong far view contributes ~0.6 mm, not the unweighted 5 mm
    assert err_mm < 1.0, f'closer views must dominate; got {err_mm:.2f} mm of pull'
    assert fused[7][3] == 2 and fused[7][4] > 0

    # beyond the cap: dropped -- and min_views then bites
    fused = mloc.fuse_markers({7: [(T_true, 0.1), (T_off, 0.6)]}, _MarkerPlan)
    assert 7 not in fused, 'a single surviving view (min_views 2) must not fuse'


def test_vote_weighs_markers_by_their_certainty():
    from urlab.skills import marker_localize as mloc

    rig = {'markers': {7: {'T_marker_target': np.eye(4)},
                       8: {'T_marker_target': np.eye(4)}}}
    # marker 7: heavy (close views); marker 8: light (far views), 10 mm disagreeing
    fused = {7: (translation_matrix([1.0, 0.0, 0.0]), 0.0, 0.0, 3, 300.0),
             8: (translation_matrix([1.0, 0.010, 0.0]), 0.0, 0.0, 3, 6.0)}
    T, votes = mloc.vote_target(rig, fused, _MarkerPlan)
    assert len(votes) == 2
    pull_mm = abs(T[1, 3]) * 1000.0
    assert pull_mm < 0.5, (
        f'the light marker must barely move the vote; got {pull_mm:.2f} mm (unweighted = 5)')


def test_servo_refine_centres_each_marker_and_returns_to_the_overview():
    """The servo loop: overview -> per-marker vantage at the canonical distance -> ring ->
    overview. The fake detector reports a fixed truth, so one servo step centres it and the
    refinement views land on the truth at ~distance_m."""
    from urlab.skills import marker_localize as mloc
    from urlab.transforms import xyzrpy_to_matrix

    truth = {5: xyzrpy_to_matrix([0.6, 0.0, 0.2], [0.0, np.pi / 2, 0.0]),
             6: xyzrpy_to_matrix([0.6, 0.1, 0.2], [0.0, np.pi / 2, 0.0])}
    T_overview = xyzrpy_to_matrix([0.3, 0.05, 0.2], [0.0, np.pi / 2, 0.0])

    class FakeArm:
        def __init__(self):
            self.T_cam = np.array(T_overview)
            self.labels = []

        def move_frame_to(self, T, T_tool0_frame, label):
            self.T_cam = np.array(T)
            self.labels.append(label)
            return True

    class FakeRobot:
        def __init__(self):
            self.arm = FakeArm()
            self.T_tool0_cam = np.eye(4)

        def camera(self):
            return self.arm.T_cam

    class FakeCamera:
        def __init__(self, robot):
            self.robot = robot

        def capture(self):
            class F:
                T_base_cam = np.array(self.robot.arm.T_cam)
            return F()

    class FakeDetector:
        def detect_in_base(self, frame):
            return dict(truth)

        def detect(self, frame):
            return {}

    plan = _MarkerPlan()
    plan.servo = mloc.ServoPlan({'enabled': True, 'distance_m': 0.15, 'max_iterations': 3,
                                 'pos_tol_mm': 0.5, 'ang_tol_deg': 0.5,
                                 'ring_mm': 25.0, 'ring_views': 2})
    robot = FakeRobot()
    seen = {mid: [(Tm @ translation_matrix([0.002, 0.001, 0.0]), 0.35)]
            for mid, Tm in truth.items()}
    refined = mloc.servo_refine(robot, FakeCamera(robot), FakeDetector(), plan, seen,
                                T_overview=T_overview)

    assert sorted(refined) == [5, 6]
    for mid, views in refined.items():
        assert len(views) == 3                      # vantage + 2 ring stops
        for T_v, d in views:
            assert np.allclose(T_v, truth[mid])
            assert 0.10 <= d <= 0.20, f'refinement views must sit near distance_m, got {d}'
    # the overview reset: before EACH marker and once at the end
    overview_hops = [l for l in robot.arm.labels if l.startswith('overview')]
    assert len(overview_hops) == 3, robot.arm.labels
    order = [l for l in robot.arm.labels if 'marker 6' in l or 'before marker' in l]
    assert order[0].startswith('overview (before marker 5)'.split(' 5')[0]), robot.arm.labels
