"""Offline tests for the behavior-tree layer (urlab/behaviors) and the shared app helpers
(urlab/apps/_common).  No robot, no camera -- pure control-flow and math."""

import os

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


def test_servo_vantage_roll_snaps_to_the_nearest_quarter_turn():
    """Pose estimation is invariant to rotation about the view axis, so the servo aligns the
    camera to the marker only up to the nearest 90 deg -- never winding the wrist further."""
    from urlab.skills.marker_localize import _quarter_roll
    from urlab.skills.servo import camera_on_marker
    from urlab.transforms import pose_error, xyzrpy_to_matrix

    T_marker = xyzrpy_to_matrix([0.6, 0.0, 0.2], [0.0, np.pi / 2, 0.0])
    for cur_roll_deg, want_deg in ((10.0, 0.0), (85.0, 90.0), (170.0, 180.0),
                                   (-100.0, -90.0), (44.0, 0.0), (46.0, 90.0)):
        T_now = camera_on_marker(T_marker, 0.15, [np.pi, 0.0, np.radians(cur_roll_deg)])
        roll = _quarter_roll(T_marker, 0.15, T_now)
        got = np.degrees(roll) % 360.0
        assert got == want_deg % 360.0, (cur_roll_deg, got, want_deg)
        # ...and the commanded vantage is then within 45 deg of the current attitude
        _l, ang = pose_error(T_now, camera_on_marker(T_marker, 0.15, [np.pi, 0.0, roll]))
        assert np.degrees(ang) <= 45.0 + 1e-6


def test_rig_object_points_carries_the_aruco_corner_layout_into_the_target_frame():
    """The joint-PnP object model: each marker's four corners (TL, TR, BR, BL, +Z out of the
    face -- ArucoDetector.object_points' exact layout) expressed in the TARGET frame through
    inverse(T_marker_target)."""
    from urlab.skills.marker_localize import rig_object_points
    from urlab.transforms import inverse, xyzrpy_to_matrix

    s = 0.02
    h = s / 2.0
    layout = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    T_mt8 = xyzrpy_to_matrix([0.05, -0.02, 0.01], [0.0, 0.0, np.pi / 2])
    rig = {'markers': {7: {'size_m': s, 'T_marker_target': np.eye(4)},
                       8: {'size_m': s, 'T_marker_target': T_mt8}}}
    pts = rig_object_points(rig)
    # marker 7 sits AT the target (identity offset): corners are the raw layout;
    # marker 8 is displaced + rotated: corners ride through the inverse offset.
    assert np.allclose(pts[7], layout)
    T_tm8 = inverse(T_mt8)
    want8 = (T_tm8[:3, :3] @ layout.T).T + T_tm8[:3, 3]
    assert np.allclose(pts[8], want8)
    assert pts[8].shape == (4, 3)


def test_capture_stops_log_joint_pnp_corner_views():
    """The servo capture path must record, per stop, the corners + intrinsics + camera pose
    the joint PnP solves from -- and skip detectors without corner support."""
    from urlab.skills import marker_localize as mloc
    from urlab.transforms import xyzrpy_to_matrix

    truth = {5: xyzrpy_to_matrix([0.6, 0.0, 0.2], [0.0, np.pi / 2, 0.0])}
    T_overview = xyzrpy_to_matrix([0.3, 0.0, 0.2], [0.0, np.pi / 2, 0.0])
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])

    class FakeArm:
        def __init__(self):
            self.T_cam = np.array(T_overview)

        def move_frame_to(self, T, T_tool0_frame, label):
            self.T_cam = np.array(T)
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
            F.K, F.D = K, np.zeros(5)
            return F()

    class FakeDetector:
        def detect_in_base(self, frame):
            return dict(truth)

        def detect(self, frame):
            return {}

        def detect_corners(self, frame):
            return {5: np.arange(8, dtype=float).reshape(4, 2)}

    plan = _MarkerPlan()
    plan.servo = mloc.ServoPlan({'enabled': True, 'distance_m': 0.15, 'max_iterations': 2,
                                 'ring_mm': 25.0, 'ring_views': 1})
    robot = FakeRobot()
    corner_views = []
    refined = mloc.servo_refine(robot, FakeCamera(robot), FakeDetector(), plan,
                                {5: [(truth[5], 0.35)]}, T_overview=T_overview,
                                corner_log=corner_views)
    assert 5 in refined
    assert len(corner_views) == 2          # the vantage + one ring stop
    for v in corner_views:
        assert set(v['corners']) == {5} and v['corners'][5].shape == (4, 2)
        assert v['K'] is K and v['T_base_cam'].shape == (4, 4)

    class PlainDetector:                   # no detect_corners: logs nothing, still refines
        detect_in_base = FakeDetector.detect_in_base
        detect = FakeDetector.detect

    corner_views = []
    refined = mloc.servo_refine(robot, FakeCamera(robot), PlainDetector(), plan,
                                {5: [(truth[5], 0.35)]}, T_overview=T_overview,
                                corner_log=corner_views)
    assert 5 in refined and corner_views == []


def test_joint_pnp_recovers_the_target_from_synthetic_corners():
    """End to end with real PnP (skipped where OpenCV is absent): project a 3-marker rig's
    corners through a known camera, solve jointly, recover the target pose."""
    import pytest
    cv2 = pytest.importorskip('cv2')
    from urlab.skills.marker_localize import joint_pnp_views, rig_object_points
    from urlab.transforms import inverse, pose_error, xyzrpy_to_matrix

    T_base_target = xyzrpy_to_matrix([0.6, 0.1, -0.1], np.radians([5.0, -3.0, 40.0]))
    rig = {'markers': {}}
    for mid, (off, rot) in {7: ([0.0, -0.06, 0.03], [90.0, 0.0, 0.0]),
                            8: ([0.05, -0.06, -0.02], [90.0, 0.0, 25.0]),
                            9: ([-0.05, -0.06, 0.01], [90.0, 0.0, -25.0])}.items():
        T_tm = xyzrpy_to_matrix(off, np.radians(rot))
        rig['markers'][mid] = {'size_m': 0.0203, 'T_marker_target': inverse(T_tm)}
    K = np.array([[615.0, 0, 424], [0, 615.0, 240], [0, 0, 1]], dtype=float)
    D = np.zeros(5)
    T_base_cam = xyzrpy_to_matrix([0.45, 0.05, 0.05], np.radians([-90.0, 0.0, 30.0]))

    T_cam_target = inverse(T_base_cam) @ T_base_target
    objs = rig_object_points(rig)
    corners = {}
    rvec, _ = cv2.Rodrigues(T_cam_target[:3, :3])
    for mid, pts in objs.items():
        img, _ = cv2.projectPoints(pts.astype(np.float32), rvec, T_cam_target[:3, 3], K, D)
        corners[mid] = img.reshape(4, 2)

    est = joint_pnp_views(rig, [{'corners': corners, 'K': K, 'D': D,
                                 'T_base_cam': T_base_cam}], _MarkerPlan())
    assert len(est) == 1
    lin, ang = pose_error(est[0][0], T_base_target)
    assert lin * 1000.0 < 0.5 and np.degrees(ang) < 0.1, (lin * 1000.0, np.degrees(ang))


def test_commanded_speeds_are_clamped_to_the_controller_ceiling():
    """ur_rtde REJECTS an out-of-range speed rather than clamping it -- moveJ raises
    ValueError('The value is not within [0;3.14]') and the run dies wherever it happened.
    Since speed.phase_scale multiplies the global caps, any scale past ~2x the shipped 90
    deg/s joint cap lands over the UR's own pi rad/s ceiling. Clamping is the honest
    behaviour: the controller limits internally anyway, so the exception only cost the run.
    """
    from urlab.robot.arm import (RTDE_MAX_JOINT_ACCEL, RTDE_MAX_JOINT_VELOCITY,
                                 RTDE_MAX_TOOL_VELOCITY, clamp_rtde)

    assert clamp_rtde(2.36, RTDE_MAX_JOINT_VELOCITY, 'v', 'ok') == 2.36, 'under = untouched'
    assert clamp_rtde(6.28, RTDE_MAX_JOINT_VELOCITY, 'v', 'retract') == RTDE_MAX_JOINT_VELOCITY
    assert clamp_rtde(-1.0, RTDE_MAX_JOINT_VELOCITY, 'v', 'x') == 0.0, 'never negative'
    assert clamp_rtde(999.0, RTDE_MAX_TOOL_VELOCITY, 'v', 'x') == RTDE_MAX_TOOL_VELOCITY
    assert clamp_rtde(999.0, RTDE_MAX_JOINT_ACCEL, 'a', 'x') == RTDE_MAX_JOINT_ACCEL

    # every phase_scale in the shipped config, resolved against the ceiling -- reported rather
    # than asserted, since a scale over the ceiling is now safe (clamped + warned), not fatal
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'configs',
                                           'bnc_assembly.yaml')))
    jv = np.radians(float(cfg['speed']['max_joint_velocity_deg_s']))
    over = {n: round(jv * float(s), 2)
            for n, s in (cfg['speed'].get('phase_scale') or {}).items()
            if jv * float(s) > RTDE_MAX_JOINT_VELOCITY}
    assert all(v > RTDE_MAX_JOINT_VELOCITY for v in over.values())   # the arithmetic holds
    assert isinstance(over, dict)


def test_skip_prompts_silences_everything_except_the_cable_labelling():
    """`skip_prompts` / --no-prompts is the STRONGER switch: --yes only silences the per-step
    gates and deliberately keeps the interlocks (reset, pre-contact stand-off, the operator's
    success call), while this silences those too. The cable labelling is exempt by design --
    which cable to pick is an input with no sane default, not a confirmation."""
    from urlab import config as urconfig
    from urlab.apps._cable import make_confirm
    from urlab.apps._common import prompts_off

    plain = urconfig.load('bnc_assembly')
    assert prompts_off(plain) is False, 'the shipped config must still ask'

    # --yes / confirm_each_step: false is NOT the same switch
    yes_only = urconfig.load('bnc_assembly', ['confirm_each_step=false'])
    assert prompts_off(yes_only) is False, (
        '--yes must leave the interlocks asking -- that is the whole difference between the '
        'two switches')

    off = urconfig.load('bnc_assembly', ['skip_prompts=true'])
    assert prompts_off(off) is True
    assert make_confirm(off) is None, 'the per-step confirm callback must be disabled too'

    # the CLI flag sets both keys
    parser = urconfig.arg_parser('t', 'bnc_assembly')
    args = parser.parse_args(['--no-prompts'])
    cfg = urconfig.from_args(args)
    assert prompts_off(cfg) is True and cfg.get('confirm_each_step') is False

    # the cable labelling prompt is NOT routed through the switch
    src = open(os.path.join(os.path.dirname(__file__), '..', 'urlab', 'skills',
                            'ground_pick.py'), encoding='utf-8').read()
    assert 'prompts_off' not in src and "input('target #> ')" in src, (
        'the cable selection must keep asking regardless -- a run that guessed which cable to '
        'grab would pick an arbitrary one')


def test_operator_gate_skip_bypasses_the_prompt():
    """The pre-contact gate must pass without touching stdin when skipping is requested (a
    prompt with closed stdin is the classic unattended-run hang)."""
    class FakeArm:
        dry_run = False

    class FakeRobot:
        arm = FakeArm()

    def boom(*_a, **_k):
        raise AssertionError('must not prompt when skip=True')

    gate = bt.OperatorGate(FakeRobot(), 'ready? ', label='gate', skip=True)
    import builtins
    real, builtins.input = builtins.input, boom
    try:
        assert bt.run_tree(gate)
    finally:
        builtins.input = real


def test_no_undefined_names_anywhere_in_the_package():
    """A name used but never imported is INVISIBLE to an import check: the module loads fine
    and raises NameError only when that line runs -- which in this codebase means partway
    through a hardware run, after the arm has already moved.

    Two real instances motivated this (both 2026-08-21): bnc_assembly used pickup_pitch_rad
    without importing it (NameError after the pick), and estimator_eval's eval_config.json
    dump referenced an `fh` that was never opened -- swallowed by a bare except, so that file
    had silently never been written. Neither was reachable by importing the module.
    """
    import io

    import pytest
    pytest.importorskip('pyflakes', reason='pyflakes guards undefined names; pip install -r '
                                           'requirements/dev.txt')
    from pyflakes import api as pyflakes_api
    from pyflakes import messages as pfm
    from pyflakes.reporter import Reporter

    class Collect(Reporter):
        def __init__(self):
            super().__init__(io.StringIO(), io.StringIO())
            self.found = []

        def flake(self, message):
            if isinstance(message, (pfm.UndefinedName, pfm.UndefinedLocal,
                                    pfm.UndefinedExport)):
                self.found.append('%s:%d: %s' % (message.filename, message.lineno,
                                                 message.message % message.message_args))

    root = os.path.join(os.path.dirname(__file__), '..', 'urlab')
    reporter = Collect()
    n = 0
    for dirpath, _dirs, files in os.walk(root):
        if '__pycache__' in dirpath:
            continue
        for f in files:
            if f.endswith('.py'):
                pyflakes_api.checkPath(os.path.join(dirpath, f), reporter)
                n += 1
    assert n > 20, f'only scanned {n} files -- the walk is not finding the package'
    assert not reporter.found, (
        'undefined name(s) -- these raise NameError at RUN time, not import time:\n  '
        + '\n  '.join(reporter.found))






def test_disassembly_walks_the_state_ladder_backwards():
    """Assembly walks engaged -> seated -> locked; disassembly must walk locked -> seated ->
    engaged -> removed, and each rung may only be claimed by the step that undoes it. The
    failure this prevents is reporting a connector 'removed' that was never unlocked."""
    from urlab.apps.bnc_assembly import (CLOCK_STATES, UNCLOCK_STATES, _advance_state,
                                         _retreat_state)

    assert CLOCK_STATES == ('engaged', 'seated', 'locked')
    assert UNCLOCK_STATES == ('locked', 'seated', 'engaged', 'removed')
    assert _retreat_state('locked', 'locked') == 'seated'
    assert _retreat_state('seated', 'seated') == 'engaged'
    assert _retreat_state('engaged', 'engaged') == 'removed'

    # it must REFUSE to skip a rung, in either direction
    for state, expected in (('locked', 'seated'), ('seated', 'locked'), ('engaged', 'locked')):
        try:
            _retreat_state(state, expected)
        except AssertionError:
            continue
        raise AssertionError(f'retreating from {state!r} as {expected!r} must not be allowed')

    # and the two ladders are exact mirrors over the rungs they share
    for a, b in zip(CLOCK_STATES, reversed(UNCLOCK_STATES[:-1])):
        assert a == b, 'the ladders must describe the same three states'
    assert _advance_state('engaged', 'engaged') == 'seated'      # assembly still works


def test_disassembly_reverses_the_assembly_rotations_about_the_socket_axis():
    """The geometry the disassembly depends on: a rotation of -theta about the socket axis
    line must exactly undo a rotation of +theta about that same line, leaving the arm where it
    started -- and a point ON the axis must not translate at all through either.

    This is what makes 'turn the collar back by -rotation_deg' and 'turn the connector back
    through the ACHIEVED sweep' correct rather than approximately correct."""
    from urlab.transforms import inverse, pose_error, rotate_about_axis, xyzrpy_to_matrix

    # a socket frame with an arbitrary (non-axis-aligned) orientation, as a real mate has
    T_clk = xyzrpy_to_matrix([0.048, 1.087, -0.155], np.radians([-1.02, 0.62, 94.17]))
    axis, point = T_clk[:3, 0], T_clk[:3, 3]
    T_tool0 = xyzrpy_to_matrix([0.1, 0.95, -0.15], np.radians([12.0, -80.0, 30.0]))

    for deg in (120.0, 60.0, -75.0):
        th = np.radians(deg)
        fwd = rotate_about_axis(T_tool0, axis, point, th)
        back = rotate_about_axis(fwd, axis, point, -th)
        lin, ang = pose_error(T_tool0, back)
        assert lin * 1000.0 < 1e-6 and np.degrees(ang) < 1e-4, (
            f'{deg} deg then -{deg} deg must return the arm exactly where it started')

    # a point ON the axis is unmoved by the turn -- which is why the connector spins in place
    # rather than being dragged through an arc
    on_axis = np.eye(4)
    on_axis[:3, 3] = point + 0.03 * (axis / np.linalg.norm(axis))
    turned = rotate_about_axis(on_axis, axis, point, np.radians(120.0))
    assert float(np.linalg.norm(turned[:3, 3] - on_axis[:3, 3])) < 1e-12

    # and the extraction direction is the exact reverse of the insertion axis
    axn = axis / np.linalg.norm(axis)
    assert abs(float(np.dot(-axn, T_clk[:3, 0] / np.linalg.norm(T_clk[:3, 0]))) + 1.0) < 1e-12


def test_disassembly_config_is_coherent():
    """The shipped block must parse, default OFF, and declare every knob the app reads."""
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'configs',
                                           'bnc_assembly.yaml')))
    d = cfg['assembly']['disassembly']
    assert isinstance(d['enabled'], bool), 'enabled must be a plain bool'
    # NO unclock_connector: the axial grasp holds the collar and the connector body together,
    # so the single reverse turn releases both -- a separate bayonet rotation would turn a part
    # that is already free.
    assert 'unclock_connector' not in d, 'the separate bayonet rotation was removed'
    for k in ('unlock_collar', 'extract_mm', 'place'):
        assert k in d, f'disassembly must declare {k}'
    assert float(d['extract_mm']) > 0
    # CYCLES: a second pass needs the cell put back, so the loop is only meaningful with
    # disassembly AND place on. The app clamps rather than looping into an already-mated
    # socket; this pins the config half of that contract.
    assert 'cycles' in d and int(d['cycles']) >= 1
    if int(d['cycles']) > 1:
        assert d['enabled'] and d['place']['enabled'], (
            'cycles > 1 needs disassembly and place enabled -- otherwise the second pass has '
            'nothing to pick and nowhere to start from')
    src = open(os.path.join(os.path.dirname(__file__), '..', 'urlab', 'apps',
                            'bnc_assembly.py'), encoding='utf-8').read()
    assert 'T_ftip_conn_nominal' in src, (
        'each cycle must reset the in-hand belief -- a fresh grasp has a fresh error, and '
        "carrying the last cycle's correction starts the insertion confidently wrong")
    assert 'scanner.reselect()' in src, (
        'each cycle must drop the cached junction selection -- the cable was PLACED, so it is '
        'not where it was picked from')
    p = d['place']
    for k in ('enabled', 'clearance_mm', 'retreat_mm'):
        assert k in p, f'disassembly.place must declare {k}'
    # the release height must clear the ground, and the rise must clear the released cable
    assert float(p['clearance_mm']) > 0, 'releasing AT the ground would press the cable into it'
    assert float(p['retreat_mm']) > float(p['clearance_mm'])






def test_ground_contact_recovery_rises_where_the_empty_rule_would_drop():
    """A full-closure stall is AMBIGUOUS -- the pads met nothing, which is either 'above the
    part, drop onto it' or 'below it, bottomed on the table'. For the BNC it is the second, so
    the reseat must go UP. Getting the sign wrong here presses the fingers harder into the
    work surface on every retry, which is exactly what the rule exists to stop."""
    from urlab import config as urconfig
    from urlab.skills.pick import GraspRecovery

    r = GraspRecovery(urconfig.load('bnc_assembly'))
    assert r.ground_counts == 190 and r.ground_tol >= 1
    assert r.ground_rise > 0, 'the ground reseat must be a RISE'
    # it must be small: a rise past the part clears it entirely on the next try
    assert r.ground_rise < r.empty_drop, (
        'the ground rise should be a fraction of a diameter, smaller than the empty drop')

    # A DISTINCT reading, at the opposite end of the travel from `empty`: an early/wide
    # stall (blocked) versus a full closure on nothing (above the part). Confusing the two
    # inverts the correction, which is the whole point of the rule.
    assert abs(r.ground_counts - r.closed_counts) > r.tol, (
        'the ground stall must not collide with the closed count -- they mean opposite things')
    # WIDER than the success band. The bands may touch at the edge -- the success check runs
    # FIRST, so a count inside the connector band is returned 'ok' before the ground rule is
    # ever consulted -- but the ground stall must not sit INSIDE it.
    assert r.ground_counts < r.connector_lo, (
        'the ground stall must be wider than the connector success band')
    assert abs(r.ground_counts - r.edge_counts) > r.ground_tol, 'not the free-closure point'

    # OPT-IN: cables that did not declare it keep the old drop-on-empty behaviour
    for other in ('cable_pick_estimate_assemble', 'cable_pick_place'):
        assert GraspRecovery(urconfig.load(other)).ground_counts is None, (
            f'{other} must be unaffected -- the rule is per-cable, not a fleet default')

    # the branch is ordered BEFORE empty in the source, which is what makes it take effect
    src = open(os.path.join(os.path.dirname(__file__), '..', 'urlab', 'skills', 'pick.py'),
               encoding='utf-8').read()
    assert src.index('GROUND CONTACT (%d ~ %d)') < src.index('EMPTY close (%d ~ closed %d)'), (
        'the ground rule is checked before the empty rule -- harmless while the two counts are '
        'distinct, and what makes the rise win if they are ever set to overlap')
    assert 'world_delta = translation_matrix([0.0, 0.0, self.ground_rise])' in src, (
        'the ground rise is applied in BASE_LINK (world up), not along the fingertip -- they '
        'coincide only at a square approach, and the pick is no longer square')







class _FakeFrame:
    def __init__(self, T_base_cam):
        self.color = np.zeros((60, 80, 3), dtype=np.uint8)
        self.K = np.array([[100.0, 0, 40], [0, 100.0, 30], [0, 0, 1]])
        self.D = np.zeros(5)
        self.T_base_cam = T_base_cam


class _FakeDrawDetector:
    """Enough of ArucoDetector for the image writer: draw() + detect_corners()."""

    def draw(self, frame, poses=None):
        return frame.color.copy()

    def detect_corners(self, frame):
        return {7: np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])}


def test_marker_image_writer_indexes_every_capture(tmp_path):
    """The writer records one index row per marker per image (and a row for a view that saw
    nothing), and is a silent no-op when disabled or when OpenCV is missing."""
    from urlab.skills.marker_localize import MarkerImageWriter
    from urlab.transforms import translation_matrix

    out = str(tmp_path)
    w = MarkerImageWriter(out, _FakeDrawDetector(), enabled=True)
    assert os.path.isdir(os.path.join(out, 'marker_images'))
    frame = _FakeFrame(translation_matrix([1.0, 0.0, 0.0]))
    w.sweep_view(0, frame, {7: translation_matrix([0.0, 0.0, 0.3])})
    w.servo_view(7, 0, frame, {7: translation_matrix([0.0, 0.0, 0.25])})
    w.servo_view(7, 1, frame, {})                  # a view that detected nothing
    w.finish(['done'])

    try:
        import cv2                                 # noqa: F401
    except ImportError:
        assert w.n == 0 and w.rows == [], 'without OpenCV the writer must no-op silently'
        return
    assert w.n == 3
    names = [r[0] for r in w.rows]
    assert names == ['sweep_01.jpg', 'servo_m07_00.jpg', 'servo_m07_01.jpg']
    assert [r[1] for r in w.rows] == ['sweep', 'servo', 'servo']
    assert w.rows[0][3] == 7 and abs(w.rows[0][4] - 300.0) < 1e-6, 'id + range in mm'
    assert w.rows[2][3] == '', 'a view that saw nothing still gets a row'
    # base_link position = T_base_cam @ T_cam_marker, in mm
    assert abs(w.rows[0][11] - 1000.0) < 1e-6 and abs(w.rows[0][13] - 300.0) < 1e-6
    d = os.path.join(out, 'marker_images')
    for f in ('sweep_01.jpg', 'servo_m07_00.jpg', 'index.csv', 'summary.txt'):
        assert os.path.exists(os.path.join(d, f)), f

    # disabled: no directory, no rows, no exception
    w2 = MarkerImageWriter(str(tmp_path / 'off'), _FakeDrawDetector(), enabled=False)
    w2.sweep_view(0, frame, {7: np.eye(4)})
    w2.finish()
    assert not os.path.exists(str(tmp_path / 'off')) and w2.n == 0


def test_multiview_refine_keeps_fused_poses_without_enough_corner_data():
    """A marker with fewer than two corner observations (or no cv2/scipy) must be left on
    its fused pose -- an empty result, never an exception."""
    from urlab.skills.marker_localize import refine_markers_multiview

    fused = {7: (np.eye(4), 0.001, 0.001, 3, 100.0)}
    assert refine_markers_multiview(fused, [], {7: 0.02}, _MarkerPlan()) == {}
    one_view = [{'corners': {7: np.zeros((4, 2))}, 'K': np.eye(3), 'D': np.zeros(5),
                 'T_base_cam': np.eye(4)}]
    assert refine_markers_multiview(fused, one_view, {7: 0.02}, _MarkerPlan()) == {}


def test_multiview_refine_recovers_the_marker_from_synthetic_corners():
    """End to end with the real solver (skipped where OpenCV is absent): project one
    marker's corners through three known cameras, start from a deliberately wrong fused
    pose, and the joint pixel-space solve must land back on the truth."""
    import pytest
    cv2 = pytest.importorskip('cv2')
    from urlab.skills.marker_localize import _corner_layout, refine_markers_multiview
    from urlab.transforms import inverse, pose_error, xyzrpy_to_matrix

    T_true = xyzrpy_to_matrix([0.55, 0.05, 0.02], np.radians([88.0, 2.0, 15.0]))
    K = np.array([[615.0, 0, 424], [0, 615.0, 240], [0, 0, 1]], dtype=float)
    D = np.zeros(5)
    size = 0.0203
    cams = [xyzrpy_to_matrix([0.40, 0.05 + dy, 0.05], np.radians([-90.0, 0.0, 20.0]))
            for dy in (-0.04, 0.0, 0.05)]
    layout = _corner_layout(size).astype(np.float32)
    views = []
    for T_bc in cams:
        T_cm = inverse(T_bc) @ T_true
        rvec, _ = cv2.Rodrigues(T_cm[:3, :3])
        img, _ = cv2.projectPoints(layout, rvec, T_cm[:3, 3], K, D)
        views.append({'corners': {7: img.reshape(4, 2)}, 'K': K, 'D': D, 'T_base_cam': T_bc})

    T_init = T_true @ xyzrpy_to_matrix([0.003, -0.002, 0.004], np.radians([1.5, -1.0, 2.0]))
    fused = {7: (T_init, 0.001, 0.001, 3, 100.0)}
    refined = refine_markers_multiview(fused, views, {7: size}, _MarkerPlan())
    assert 7 in refined
    T_ref, rms_px, n = refined[7]
    assert n == 3 and rms_px < 0.1
    lin, ang = pose_error(T_ref, T_true)
    init_lin, _ = pose_error(T_init, T_true)
    assert lin * 1000.0 < 0.1 and np.degrees(ang) < 0.05, (lin * 1000.0, np.degrees(ang))
    assert lin < init_lin / 10.0, 'the joint solve must land far closer than the wrong start'


def test_gripper_warmup_sequence():
    """The session warm-up strokes: open, full close, two partial cycles, end fully open."""
    from urlab.robot.gripper import Robotiq2F85

    calls = []

    class G(Robotiq2F85):
        def __init__(self):                 # no hardware -- warmup only needs go_to
            pass

        def go_to(self, counts, label='', wait=True):
            calls.append(counts)
            return True

    assert G().warmup()
    assert calls == [0, 255, 150, 255, 150, 255, 0]
    calls.clear()

    class GFail(G):
        def go_to(self, counts, label='', wait=True):
            calls.append(counts)
            return counts != 150            # stall on the first partial open

    assert not GFail().warmup()
    assert calls == [0, 255, 150], 'the warm-up must stop at the failing stroke'


# ---------------------------------------------------------------------------------------------
# THE GRASP IS ONE TRANSFORM. `pickup.fingertip_in_connector` is the target FINGERTIP frame in
# the DETECTED CONNECTOR frame, and the arm is commanded to detected_connector @ that. These pin
# the two things that has to keep true: it aims the approach, and the in-hand belief it implies
# still describes where the part physically is.
# ---------------------------------------------------------------------------------------------
_T_TOOL0_FTIP_RPY = (np.pi, 0.0, -np.pi / 2)      # configs/bnc_assembly.yaml fingertip_grasp
_T_TOOL0_FTIP_XYZ = (0.0, 0.0, 0.183)


def _bnc_frames():
    """(tool0->fingertip, junction_in_fingertip, nominal in-hand belief, a detected junction).

    The detected junction has x = the connector axis and z = the ground normal, which is what
    skills/scan's frame_from_axis builds."""
    from urlab.transforms import xyzrpy_to_matrix
    return (xyzrpy_to_matrix(list(_T_TOOL0_FTIP_XYZ), list(_T_TOOL0_FTIP_RPY)),
            xyzrpy_to_matrix([0, 0, 0], [0, 0, np.pi]),              # cables.yaml bnc
            xyzrpy_to_matrix([-0.0457, 0.0, 0.0075], [0, 0, np.pi]),  # frames.yaml fingerpads
            xyzrpy_to_matrix([0.5, 0.0, 0.005], [0, 0, 0]))


def test_the_rpy_pitch_of_fingertip_in_connector_aims_the_whole_approach():
    """ONE NUMBER SETS BOTH ALIGNMENTS. With the rotation written as a pure pitch about the
    connector frame's y, tool0 +Z comes off the connector +X and tool0 -Y comes off the
    connector +Z by the SAME angle, 90 - |pitch| -- so -90 is exactly axial and -60 is 30 deg
    off both. That coincidence is why the old pitch_deg/roll_deg pair collapsed into this: the
    roll was only ever there to undo junction_in_fingertip's yaw, and here it cancels for free.
    """
    from urlab.skills.pick import grasp_pose
    from urlab.transforms import inverse, xyzrpy_to_matrix

    T_tool0_ftip, _T_fj, _nom, J = _bnc_frames()
    conn_x, conn_z = J[:3, 0], J[:3, 2]

    def angle(u, v):
        return float(np.degrees(np.arccos(np.clip(float(np.dot(u, v)), -1.0, 1.0))))

    for pitch in (0.0, -30.0, -60.0, -75.0, -90.0):
        G = xyzrpy_to_matrix([0.010, 0.0, 0.0], [0.0, np.radians(pitch), 0.0])
        T_tool0 = grasp_pose(J, G) @ inverse(T_tool0_ftip)
        want = 90.0 - abs(pitch)
        assert abs(angle(T_tool0[:3, 2], conn_x) - want) < 1e-6, (
            f'pitch {pitch}: tool0 +Z is {angle(T_tool0[:3, 2], conn_x):.2f} deg off the '
            f'connector axis, want {want}')
        assert abs(angle(-T_tool0[:3, 1], conn_z) - want) < 1e-6, (
            f'pitch {pitch}: tool0 -Y is {angle(-T_tool0[:3, 1], conn_z):.2f} deg off the '
            f'connector +Z, want {want} -- the two must track together')

    # AND THE FLANGE GEOMETRY THAT FOLLOWS: 183*sin(90-|pitch|) is both its height above the
    # bite point and its distance off the connector axis. Ground clearance and the wrist-twist
    # geometry are the same number pulling opposite ways, which is the whole design tension.
    for pitch in (-60.0, -75.0, -90.0):
        G = xyzrpy_to_matrix([0.0, 0.0, 0.0], [0.0, np.radians(pitch), 0.0])
        T_tool0 = grasp_pose(J, G) @ inverse(T_tool0_ftip)
        rel = np.asarray(T_tool0[:3, 3]) - np.asarray(J[:3, 3])
        expect = 0.183 * np.sin(np.radians(90.0 - abs(pitch)))
        assert abs(float(rel[2]) - expect) < 1e-9, 'flange height above the bite point'
        assert abs(float(np.linalg.norm(rel[1:])) - expect) < 1e-9, 'flange offset off the axis'


def test_held_belief_describes_where_the_part_physically_is():
    """THE ROUND TRIP that keeps the pick and the belief one pair. Grasp the part with any
    fingertip_in_connector, and the belief the app plans with must put the connector exactly
    where the part actually is -- because both come from that one transform. A hand-edited
    frames.yaml, or a grasp knob changed without the belief, breaks this."""
    from urlab.skills.pick import grasp_pose, held_belief
    from urlab.transforms import inverse, pose_error, xyzrpy_to_matrix

    _T_tool0_ftip, T_fj, nominal, J = _bnc_frames()
    T_junction_conn = inverse(T_fj) @ nominal            # a property of the PART

    for xyz in ([0, 0, 0], [0.010, 0.0, 0.0], [0.010, -0.002, 0.009]):
        for rpy in ([0, 0, 0], [0, -60, 0], [0, -90, 0], [8, -75, -5]):
            G = xyzrpy_to_matrix(xyz, np.radians(rpy))
            believed = grasp_pose(J, G) @ held_belief(nominal, T_fj, G)
            lin, ang = pose_error(believed, J @ T_junction_conn)
            # pose_error's arccos loses digits near identity, hence 1e-4 deg not an exact zero
            assert lin * 1000.0 < 1e-6 and np.degrees(ang) < 1e-4, (
                f'xyz {xyz} rpy {rpy}: the belief is {lin * 1000:.6f} mm / '
                f'{np.degrees(ang):.6f} deg from where the part actually is')

    # the NOMINAL grip is the fixed point: fingertip_in_connector = inverse(junction_in_fingertip)
    # is "grip exactly as frames.yaml describes", and must leave the belief untouched.
    assert np.allclose(held_belief(nominal, T_fj, inverse(T_fj)), nominal)


def test_fingertip_in_connector_translation_is_read_in_the_connector_frame():
    """The xyz must move the bite point along the CONNECTOR's own axes -- +x along the barrel,
    +z up off the ground -- and must not touch the approach direction. That independence is
    what makes the ground-clearance fix (z) safe to tune without re-aiming the gripper."""
    from urlab.skills.pick import grasp_pose
    from urlab.transforms import xyzrpy_to_matrix

    _t, _fj, _nom, J = _bnc_frames()
    for rpy in ([0, 0, 0], [0, -60, 0], [0, -90, 0]):
        R = np.radians(rpy)
        base = grasp_pose(J, xyzrpy_to_matrix([0, 0, 0], R))
        for axis, vec in ((0, [0.012, 0, 0]), (1, [0, 0.012, 0]), (2, [0, 0, 0.012])):
            moved = grasp_pose(J, xyzrpy_to_matrix(vec, R))
            step = moved[:3, 3] - base[:3, 3]
            assert np.allclose(step, J[:3, :3] @ np.asarray(vec)), (
                f'rpy {rpy}: an xyz step must land along the CONNECTOR frame axis {axis}')
            assert np.allclose(moved[:3, :3], base[:3, :3]), (
                'the translation must not rotate the approach')


def test_held_junction_in_fingertip_is_just_the_inverse():
    """Downstream users (the kinematic-assembly target) need the ACTUAL junction-in-fingertip.
    The fingertip was commanded to connector @ G, so from the fingertip the connector is at
    inverse(G) -- and expressing it that way is what stops the sign being re-derived, and
    dropped, at each call site."""
    from urlab.skills.pick import held_junction_in_fingertip
    from urlab.transforms import inverse, xyzrpy_to_matrix

    for xyz, rpy in (([0, 0, 0], [0, 0, 0]), ([0.01, 0, 0.009], [0, -60, 0]),
                     ([0.01, -0.002, 0], [5, -90, 12])):
        G = xyzrpy_to_matrix(xyz, np.radians(rpy))
        assert np.allclose(held_junction_in_fingertip(G), inverse(G))


def test_belief_offset_moves_the_belief_and_never_the_grasp():
    """THE SEPARATION this knob exists for: `pickup.belief_offset_mm` is the MEASURED seating
    residual, and it must move only what we think we are holding -- never where the fingers go.
    Everything else in the pickup block is the command; this one alone is the belief."""
    from urlab.skills.pick import belief_offset_m, grasp_pose, offset_belief
    from urlab.transforms import xyzrpy_to_matrix

    _t, _fj, nominal, J = _bnc_frames()
    G = xyzrpy_to_matrix([0.010, 0.0, 0.0], [0.0, np.radians(-60.0), 0.0])

    # the BELIEF moves, in the FINGERTIP frame, orientation untouched
    shifted = offset_belief(nominal, np.array([0.0, 0.0, -0.0012]))
    assert np.allclose(shifted[:3, :3], nominal[:3, :3]), 'a seating residual is a translation'
    assert np.allclose(shifted[:3, 3] - nominal[:3, 3], [0.0, 0.0, -0.0012])

    # the GRASP cannot even see it -- grasp_pose is not a function of the belief
    assert np.allclose(grasp_pose(J, G), grasp_pose(J, G))

    from urlab import config as urconfig

    cfg = urconfig.load('bnc_assembly')
    v = belief_offset_m(cfg)
    assert v.shape == (3,), 'belief_offset_mm is a 3-vector in the fingertip frame'


def test_the_connector_axis_height_comes_from_the_greatest_diameter():
    """The scan reports the junction ON the ground plane; the part is a solid resting on it, so
    its axis is HALF ITS GREATEST DIAMETER up. Getting this wrong aims the pads at the floor,
    which is exactly what drove the gripper into the bench."""
    from urlab.robot.gripper_kinematics import width_from_counts
    from urlab.skills.pick import connector_axis_height_m

    from urlab import config as urconfig

    cfg = urconfig.load('bnc_assembly')
    assert bool(cfg.get_path('pickup.rests_on_ground_plane', True)), (
        'the bnc connector is picked off the bench -- the correction must be on')
    r = connector_axis_height_m(cfg)
    assert r > 0.0, 'neither a measured diameter nor a counts band -- the pads would aim low'

    # DERIVED FROM THE MEASURED BAND when no diameter is declared: the LOW count is the FAT end
    # (more obstruction = less closed), so it is the one that gives the greatest diameter.
    if not cfg.get_path('grasp_check.connector_diameter_mm'):
        counts = cfg.get_path('grasp_check.connector_counts')
        groove = cfg.get_path('gripper.groove_depth_mm')
        kw = {} if groove is None else {'groove_depth_m': float(groove) / 1000.0}
        assert abs(r - width_from_counts(min(counts), **kw) / 2.0) < 1e-12
        assert 0.004 < r < 0.015, (
            f'{r * 1000:.2f} mm is not a plausible BNC radius -- check the counts band')

    # a measured diameter WINS over the derived one
    cfg2 = urconfig.load('bnc_assembly')
    cfg2.set_path('grasp_check.connector_diameter_mm', [9.0, 20.0])
    assert abs(connector_axis_height_m(cfg2) - 0.010) < 1e-12, 'max(20 mm)/2'


def test_the_bnc_config_asks_for_the_tilted_axial_grasp():
    """The pickup block must describe the grasp with the ONE transform, and no leftovers of the
    trio it replaced -- a stale pitch_deg would read as live tuning and change nothing."""
    from urlab.skills.pick import fingertip_in_connector
    from urlab.transforms import matrix_to_xyzrpy

    from urlab import config as urconfig

    cfg = urconfig.load('bnc_assembly')
    pick = cfg.section('pickup')
    for gone in ('pitch_deg', 'roll_deg', 'grip_offset', 'grip_offset_mm', 'height_from_model'):
        assert gone not in pick, (
            f'pickup.{gone} is superseded by fingertip_in_connector -- leaving it in the file '
            'reads as a live knob that silently does nothing')
    assert 'fingertip_in_connector' in pick, 'the grasp target must be declared'

    xyz, rpy = matrix_to_xyzrpy(fingertip_in_connector(cfg))
    deg = np.degrees(rpy)
    assert abs(deg[0]) < 1e-9 and abs(deg[2]) < 1e-9, (
        f'rpy {np.round(deg, 2).tolist()}: keep the approach a PURE PITCH about the connector '
        'frame y -- that is what makes both tool0 alignments track one number')
    assert -90.0 <= deg[1] <= -30.0, (
        f'rpy y is {deg[1]:.1f}; the tilted/axial grasp wants a large negative pitch (the '
        'gripper pointing down the cable), not a near-square one')
    # z stays 0: the ground-plane lift is a property of the PART and lives in the estimate
    assert abs(xyz[2]) < 1e-9, (
        'the barrel-radius lift belongs to rests_on_ground_plane, not here -- setting both '
        'doubles the correction')
