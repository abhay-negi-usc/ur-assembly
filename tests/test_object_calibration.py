"""OBJECT CALIBRATION -- the geometry and the catalogue schema, without a robot or a camera.

The app's arithmetic is pure on purpose (choose_yaw_reference / canonical_grasp_frame /
fuse_mates / yaml_document), because the interesting failure modes here are silent ones: a yaw
convention that flips between runs, a residual that reports more error than the evidence
supports, a catalogue rewrite that eats a neighbour. None of those need hardware to catch.
"""

import os
import tempfile

import numpy as np
import pytest

from urlab import config as C
from urlab import tool_frames
from urlab.apps import object_calibration as oc
from urlab.apps.object_calibration import fuse_mates, merge_catalogue, yaml_document
from urlab.transforms import inverse, xyzrpy_to_matrix

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')


def _pose(xyz_mm, rpy_deg):
    return xyzrpy_to_matrix(np.array(xyz_mm, float) / 1000.0, np.radians(rpy_deg))


# ---------------------------------------------------------------------------- the tool frame
def test_the_coupler_mate_is_on_the_tool_axis_and_only_there():
    """Straight out of the flange along +z: the coupler picks down the tool axis, so the
    frame is a PURE TRANSLATION. A rotation here would mean the mating axis is not the
    tool axis, and every standoff and retract in the cell reads that axis.

    The LENGTH is a measurement of the hardware and is deliberately not pinned -- re-machine
    or re-measure the coupler and this test should not care."""
    T = tool_frames.coupler_mate(C.load('object_calibration'))
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-9), 'the coupler frame is rotated'
    assert np.allclose(T[:2, 3], [0.0, 0.0], atol=1e-9), 'the coupler is off the axis'
    assert T[2, 3] > 0.0, 'the engagement end must be OUT of the flange'


def test_a_missing_coupler_frame_is_an_error_not_identity():
    """Identity would silently put the mating point AT the flange and drive every pick 55 mm too
    deep -- into the object, not onto it."""
    with tempfile.TemporaryDirectory() as tmp:
        frames = os.path.join(tmp, 'frames.yaml')
        with open(frames, 'w') as fh:
            fh.write('frames: {}\n')
        cfg = C.Config({'frames_file': frames, '_config_dir': tmp,
                        '_config_path': os.path.join(tmp, 'x.yaml')})
        with pytest.raises(ValueError, match='coupler_mate'):
            tool_frames.coupler_mate(cfg)


# ------------------------------------------------------------------ the stored pose IS the mate
def test_the_stored_pose_is_the_measured_mate_uncorrected():
    """THE CONTRACT. inverse(marker) @ mate, and nothing else -- no axis substituted from the
    marker, no angle reconstructed. Commanding it back must reproduce the seated pose exactly."""
    T_marker = _pose([90.0, 210.0, 10.0], [0.0, 0.0, 25.0])
    T_mate = _pose([100.0, 200.0, 50.0], [176.0, 3.0, 41.0])     # deliberately not square
    stored = inverse(T_marker) @ T_mate
    assert np.allclose(T_marker @ stored, T_mate, atol=1e-12), 'the mate was not reproduced'


def test_the_operators_yaw_survives_into_the_stored_pose():
    """The counterpart of the old canonicalisation test, inverted. Yaw about the mating axis is
    no longer corrected away, so two mates seated at different wrist angles are DIFFERENT stored
    poses -- which is the point: the pose that worked is the pose that is kept."""
    T_marker = _pose([90.0, 210.0, 10.0], [0.0, 0.0, 0.0])
    base = _pose([100.0, 200.0, 50.0], [180.0, 0.0, 0.0])
    stored = []
    for yaw_deg in (0.0, 47.0, -120.0):
        spun = base @ xyzrpy_to_matrix([0, 0, 0], [0.0, 0.0, np.radians(yaw_deg)])
        stored.append(inverse(T_marker) @ spun)
    for T in stored[1:]:
        assert not np.allclose(T, stored[0]), 'the wrist yaw was corrected away after all'
    # ... but the mating POINT and AXIS are untouched by a spin about that axis.
    for T in stored[1:]:
        assert np.allclose(T[:3, 3], stored[0][:3, 3], atol=1e-12)
        assert np.allclose(T[:3, 2], stored[0][:3, 2], atol=1e-9)


def test_every_marker_encodes_the_same_mate():
    """No marker is special any more. Each stores the same physical pose in its own frame, so
    each votes for the same answer -- and none has to be present on every mate."""
    T_mate = _pose([100.0, 200.0, 50.0], [176.0, 3.0, 41.0])
    markers = {0: _pose([90.0, 210.0, 10.0], [0.0, 0.0, 25.0]),
               1: _pose([-40.0, 60.0, 20.0], [0.0, 0.0, -70.0]),
               2: _pose([10.0, -70.0, 40.0], [0.0, 0.0, 160.0])}
    for M in markers.values():
        assert np.allclose(M @ (inverse(M) @ T_mate), T_mate, atol=1e-12)


# ---------------------------------------------------------------------------- the catalogue
def _entry(marker_id=31, size_m=0.0388, extra=()):
    markers = {marker_id: {'size_m': size_m,
                           'T_marker_grasp': _pose([12.4, -38.1, 5.2], [-90.0, 0.0, 180.0]),
                           'meta': {'mates': 3, 'residual_mm': 0.42,
                                    'residual_deg': 0.55, 'residual_axis_deg': 0.31}}}
    for i, mid in enumerate(extra, start=1):
        markers[mid] = {'size_m': size_m,
                        'T_marker_grasp': _pose([12.4 + 10 * i, -38.1, 5.2], [-90.0, 0.0, 180.0]),
                        'meta': {'mates': 3, 'residual_mm': 0.5,
                                 'residual_deg': 0.6, 'residual_axis_deg': 0.4}}
    return {'markers': markers, 'held_mass_kg': 0.4,
            'meta': {'mates': 3, 'measured': '2026-09-22'}}


def test_a_written_catalogue_reads_back_to_the_same_pose():
    """Round trip through the file, because the yaml is written by hand-rolled formatting and a
    dropped sign or a millimetre/metre slip would not raise -- it would just pick wrong."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'objects.yaml')
        with open(path, 'w') as fh:
            fh.write(yaml_document({'banana_jig': _entry()}))
        back = tool_frames.load_objects(path=path)
        assert set(back) == {'banana_jig'}
        got = back['banana_jig']
        assert set(got['markers']) == {31}
        assert got['markers'][31]['size_m'] == pytest.approx(0.0388)
        assert got['held_mass_kg'] == pytest.approx(0.4)
        assert np.allclose(got['markers'][31]['T_marker_grasp'],
                           _entry()['markers'][31]['T_marker_grasp'], atol=1e-5)
        assert got['meta']['mates'] == 3
        assert got['markers'][31]['meta']['residual_deg'] == pytest.approx(0.55)


def test_several_markers_round_trip_independently():
    """Each marker keeps its OWN offset -- they encode the same grasp frame from different
    coordinates, so writing them all through one entry must not collapse them together."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'objects.yaml')
        with open(path, 'w') as fh:
            fh.write(yaml_document({'jig': _entry(31, extra=(32, 33))}))
        got = tool_frames.load_objects(path=path)['jig']
        assert set(got['markers']) == {31, 32, 33}
        poses = [got['markers'][m]['T_marker_grasp'][:3, 3] for m in (31, 32, 33)]
        assert not np.allclose(poses[0], poses[1]), 'two markers collapsed to one pose'
        assert set(got['meta']) == {'mates', 'measured'}, 'no yaw convention is stored'


def test_calibrating_one_object_leaves_the_others_alone():
    """The catalogue is rewritten whole on every run. If the merge dropped a neighbour, the next
    pick of THAT object would fail with the object sitting right there in front of the camera."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'objects.yaml')
        with open(path, 'w') as fh:
            fh.write(yaml_document({'a': _entry(31), 'b': _entry(32)}))
        merged = merge_catalogue(path, 'b', _entry(99))
        assert set(merged) == {'a', 'b'}
        assert set(merged['a']['markers']) == {31}, 'a bystander was rewritten'
        assert set(merged['b']['markers']) == {99}, 'the calibrated object was not replaced'


def test_an_empty_catalogue_is_valid_and_loads():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'objects.yaml')
        with open(path, 'w') as fh:
            fh.write(yaml_document({}))
        assert tool_frames.load_objects(path=path) == {}


def test_the_shipped_catalogue_loads():
    """configs/objects.yaml is machine-written; it must still parse under the strict loader."""
    assert isinstance(tool_frames.load_objects(C.load('object_calibration')), dict)


@pytest.mark.parametrize('broken, match', [
    ('objects:\n  x:\n    xyz_mm: [0,0,0]\n', 'no markers'),
    ('objects:\n  x:\n    markers: {}\n', 'no markers'),
    ('objects:\n  x:\n    markers:\n      3: {xyz_mm: [0,0,0], rpy_deg: [0,0,0]}\n', 'size_mm'),
    ('objects:\n  x:\n    markers:\n      3: {size_mm: 0, xyz_mm: [0,0,0], rpy_deg: [0,0,0]}\n',
     'non-positive'),
    ('objects:\n  x:\n    markers:\n      3: {size_mm: 20, xyz_m: [0,0,0]}\n', 'unknown key'),
    ('objects:\n  x:\n    markers:\n      3: {size_mm: 20}\n', 'no pose keys'),
    ('objects:\n  x:\n    markers:\n      3: {size_mm: 20, xyz_mm: [0,0,0], rpy_deg: [0,0,0]}\n'
     '    bogus: 1\n', 'unknown key'),
])
def test_a_broken_entry_fails_loudly(broken, match):
    """Every one of these would otherwise place the coupler somewhere plausible and wrong -- a
    missing size is a depth error, a typo'd unit key collapses the pose to identity."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'objects.yaml')
        with open(path, 'w') as fh:
            fh.write(broken)
        with pytest.raises(ValueError, match=match):
            tool_frames.load_objects(path=path)


# ---------------------------------------------------------------------------- hand guiding
class _FakeRtde:
    def __init__(self):
        self.calls = []

    def teachMode(self):
        self.calls.append('on')

    def endTeachMode(self):
        self.calls.append('off')


class _FakeArm:
    dry_run = False

    def __init__(self):
        self.rtde_c = _FakeRtde()


class _FakeRobot:
    def __init__(self):
        self.arm = _FakeArm()


def _guider(cfg=None):
    cal = oc._ObjectCalibration.__new__(oc._ObjectCalibration)
    cal.robot = _FakeRobot()
    cal.cfg = C.Config(cfg or {})
    return cal


def test_a_hand_guide_prompt_actually_enables_freedrive(monkeypatch):
    """REGRESSION. The view prompt used to ask the operator to push the arm without turning
    teachMode on -- so the arm was stiff and the instruction was impossible to follow."""
    cal = _guider()
    monkeypatch.setattr(oc, 'ask', lambda _p: True)
    assert cal._hand_guide('push it: ') is True
    assert cal.robot.arm.rtde_c.calls == ['on', 'off'], (
        'freedrive must be ON for the prompt and OFF again afterwards')


def test_freedrive_is_released_when_the_operator_aborts(monkeypatch):
    cal = _guider()
    monkeypatch.setattr(oc, 'ask', lambda _p: False)
    assert cal._hand_guide('push it: ') is False
    assert cal.robot.arm.rtde_c.calls == ['on', 'off'], 'an abort left the arm compliant'


def test_freedrive_is_released_when_the_prompt_is_interrupted(monkeypatch):
    """Ctrl-C at the prompt must not walk out with teachMode still on -- the next thing the app
    does is command a move, into an arm that is not holding position."""
    def boom(_p):
        raise KeyboardInterrupt

    cal = _guider()
    monkeypatch.setattr(oc, 'ask', boom)
    with pytest.raises(KeyboardInterrupt):
        cal._hand_guide('push it: ')
    assert cal.robot.arm.rtde_c.calls == ['on', 'off'], 'an interrupt left the arm compliant'


def test_an_unattended_run_never_enters_freedrive(monkeypatch):
    """skip_prompts means nobody is standing there. Going compliant with no operator is the one
    case where freedrive is actively unsafe."""
    monkeypatch.setattr(oc, 'ask', lambda _p: pytest.fail('should not prompt'))
    cal = _guider({'skip_prompts': True})
    assert cal._hand_guide('push it: ') is True
    assert cal.robot.arm.rtde_c.calls == []


# ---------------------------------------------------------------------------- the close look
def test_both_apps_servo_to_each_marker_at_the_configured_distance():
    """The close look is on by default in BOTH apps, at the same standoff, so a pick is solved
    under the same viewing geometry the calibration was measured under."""
    from urlab.skills import marker_localize as mloc

    assert mloc.ServoPlan({'enabled': True}).distance_m == pytest.approx(0.150)
    for name in ('object_calibration', 'coupler_pick_place', 'coupler_pick_assemble'):
        plan = mloc.ViewPlan(C.load(name).section('marker_views'))
        assert plan.servo.enabled is True, f'{name} does not servo'
        # The RANGE is a tuning value -- close enough to resolve the marker, far enough
        # to keep it in frame -- so only the structural bound is pinned: reachable
        # inside the standoff cap, which ViewPlan itself refuses to violate.
        assert 0.0 < plan.servo.distance_m < plan.max_camera_distance_m, name


def test_the_servo_standoff_is_inside_the_camera_distance_cap():
    """ViewPlan refuses a servo range beyond max_camera_distance_mm, because a vantage the sweep
    would never fly to cannot be reached. Pinned so neither number can drift past the other."""
    from urlab.skills import marker_localize as mloc

    for name in ('object_calibration', 'coupler_pick_place',
                 'coupler_pick_assemble'):
        block = dict(C.load(name).section('marker_views'))
        plan = mloc.ViewPlan(block)
        assert plan.servo.distance_m < plan.max_camera_distance_m, name
        # Override the SI key, not the mm one: a loaded block carries BOTH (the loader adds
        # the sibling) and distance_m takes precedence, so setting only distance_mm here would
        # be silently ignored -- the very trap ServoPlan was just taught to avoid.
        block['servo'] = dict(block['servo'],
                              distance_m=plan.max_camera_distance_m + 0.001)
        with pytest.raises(ValueError, match='standoff cap'):
            mloc.ViewPlan(block)


def test_a_servo_block_written_in_mm_alone_is_honoured():
    """REGRESSION. Every config writes `distance_mm`, and that only reached ServoPlan because
    the config loader adds an SI sibling. A ViewPlan built from a hand-made dict -- a test, or a
    caller assembling a block -- silently got the 0.15 m default instead, which is a servo
    standoff 50 mm further out than asked for and looks like nothing."""
    from urlab.skills.marker_localize import ServoPlan

    assert ServoPlan({'enabled': True, 'distance_mm': 100.0}).distance_m == pytest.approx(0.100)
    assert ServoPlan({'enabled': True, 'distance_m': 0.2}).distance_m == pytest.approx(0.200)
    assert ServoPlan({'enabled': True}).distance_m == pytest.approx(0.150), 'default moved'


def test_the_calibration_servos_after_the_sweep_and_pools_the_views(monkeypatch):
    """Close views must POOL with the sweep's, not replace them -- replacing could drop a marker
    whose servo produced few views below min_views, which is worse than a wider average."""
    from urlab.apps import object_calibration as oc
    from urlab.skills import marker_localize as mloc

    order = []
    T = _pose([400.0, 0.0, 50.0], [0.0, 0.0, 0.0])
    cal = oc._ObjectCalibration.__new__(oc._ObjectCalibration)
    cal.cfg = C.load('object_calibration')
    cal.plan = mloc.ViewPlan(cal.cfg.section('marker_views'))
    # DISTANCES DERIVED, NOT PINNED. fuse_markers drops any view beyond
    # max_camera_distance_mm, so a hardcoded sweep range would make this test fail whenever
    # that cap is retuned -- which says nothing about whether pooling works.
    near = cal.plan.servo.distance_m
    far = cal.plan.max_camera_distance_m * 0.9
    monkeypatch.setattr(mloc, 'sweep',
                        lambda *a, **k: (order.append('sweep'), {0: [(T, far)] * 4})[1])
    monkeypatch.setattr(mloc, 'servo_refine',
                        lambda *a, **k: (order.append('servo'), {0: [(T, near)] * 3})[1])
    cal.sizes, cal.yaw_marker, cal.scans = {0: 0.04}, 0, []
    cal.robot = cal.camera = cal.detector = None
    cal.T_overview = np.eye(4)
    cal.images = type('I', (), {'sweep_view': lambda *a: None,
                                'servo_view': lambda *a: None})()

    assert cal._scan(0) is True
    assert order == ['sweep', 'servo'], 'the servo must follow the sweep, not replace it'
    assert cal.scans[0][0][3] == 7, 'the close views did not pool with the sweep views'


# ---------------------------------------------------------------------------- the schema
def test_no_yaw_convention_survives_in_the_schema():
    """The correction is gone: there is no reference marker and no stored yaw convention, so
    neither key may reappear in the catalogue or its allowed metadata."""
    assert not any(k.startswith('yaw') or k.startswith('roll')
                   for k in tool_frames.OBJECT_META_KEYS)
    assert 'residual_deg' in tool_frames.OBJECT_META_KEYS, 'full rotation spread must be kept'
    for obj in tool_frames.load_objects().values():
        assert not any(k.startswith(('yaw', 'roll')) for k in obj['meta'])


# ---------------------------------------------------------------------------- the close look
def test_both_apps_servo_to_each_marker_at_the_configured_distance():
    """The close look is on by default in BOTH apps, at the same standoff, so a pick is solved
    under the same viewing geometry the calibration was measured under."""
    from urlab.skills import marker_localize as mloc

    assert mloc.ServoPlan({'enabled': True}).distance_m == pytest.approx(0.150)
    for name in ('object_calibration', 'coupler_pick_place', 'coupler_pick_assemble'):
        plan = mloc.ViewPlan(C.load(name).section('marker_views'))
        assert plan.servo.enabled is True, f'{name} does not servo'
        # The RANGE is a tuning value -- close enough to resolve the marker, far enough to
        # keep it in frame -- so only the structural bound is pinned: it has to be
        # reachable inside the standoff cap, which ViewPlan itself refuses to violate.
        assert 0.0 < plan.servo.distance_m < plan.max_camera_distance_m, name


def test_the_servo_standoff_is_inside_the_camera_distance_cap():
    """ViewPlan refuses a servo range beyond max_camera_distance_mm, because a vantage the sweep
    would never fly to cannot be reached. Pinned so neither number can drift past the other."""
    from urlab.skills import marker_localize as mloc

    for name in ('object_calibration', 'coupler_pick_place'):
        block = dict(C.load(name).section('marker_views'))
        plan = mloc.ViewPlan(block)
        assert plan.servo.distance_m < plan.max_camera_distance_m, name
        # Override the SI key, not the mm one: a loaded block carries BOTH (the loader adds
        # the sibling) and distance_m takes precedence, so setting only distance_mm here would
        # be silently ignored -- the very trap ServoPlan was just taught to avoid.
        block['servo'] = dict(block['servo'],
                              distance_m=plan.max_camera_distance_m + 0.001)
        with pytest.raises(ValueError, match='standoff cap'):
            mloc.ViewPlan(block)


def test_a_servo_block_written_in_mm_alone_is_honoured():
    """REGRESSION. Every config writes `distance_mm`, and that only reached ServoPlan because
    the config loader adds an SI sibling. A ViewPlan built from a hand-made dict -- a test, or a
    caller assembling a block -- silently got the 0.15 m default instead, which is a servo
    standoff 50 mm further out than asked for and looks like nothing."""
    from urlab.skills.marker_localize import ServoPlan

    assert ServoPlan({'enabled': True, 'distance_mm': 100.0}).distance_m == pytest.approx(0.100)
    assert ServoPlan({'enabled': True, 'distance_m': 0.2}).distance_m == pytest.approx(0.200)
    assert ServoPlan({'enabled': True}).distance_m == pytest.approx(0.150), 'default moved'


def test_the_calibration_servos_after_the_sweep_and_pools_the_views(monkeypatch):
    """Close views must POOL with the sweep's, not replace them -- replacing could drop a marker
    whose servo produced few views below min_views, which is worse than a wider average."""
    from urlab.apps import object_calibration as oc
    from urlab.skills import marker_localize as mloc

    order = []
    T = _pose([400.0, 0.0, 50.0], [0.0, 0.0, 0.0])
    cal = oc._ObjectCalibration.__new__(oc._ObjectCalibration)
    cal.cfg = C.load('object_calibration')
    cal.plan = mloc.ViewPlan(cal.cfg.section('marker_views'))
    # DISTANCES DERIVED, NOT PINNED. fuse_markers drops any view beyond
    # max_camera_distance_mm, so a hardcoded sweep range would make this test fail whenever
    # that cap is retuned -- which says nothing about whether pooling works.
    near = cal.plan.servo.distance_m
    far = cal.plan.max_camera_distance_m * 0.9
    monkeypatch.setattr(mloc, 'sweep',
                        lambda *a, **k: (order.append('sweep'), {0: [(T, far)] * 4})[1])
    monkeypatch.setattr(mloc, 'servo_refine',
                        lambda *a, **k: (order.append('servo'), {0: [(T, near)] * 3})[1])
    cal.sizes, cal.yaw_marker, cal.scans = {0: 0.04}, 0, []
    cal.robot = cal.camera = cal.detector = None
    cal.T_overview = np.eye(4)
    cal.images = type('I', (), {'sweep_view': lambda *a: None,
                                'servo_view': lambda *a: None})()

    assert cal._scan(0) is True
    assert order == ['sweep', 'servo'], 'the servo must follow the sweep, not replace it'
    assert cal.scans[0][0][3] == 7, 'the close views did not pool with the sweep views'


# ---------------------------------------------------------------------------- the axes




# ---------------------------------------------------------------------------- servo routing
class _RoutingArm:
    def __init__(self, fail=()):
        self.fail = set(fail)
        self.moves = []

    def move_frame_to(self, T, T_tool, label='move'):
        self.moves.append(label)
        if label in self.fail:
            self.fail.discard(label)
            return False
        return True


class _RoutingRobot:
    def __init__(self, fail=()):
        self.arm = _RoutingArm(fail)
        self.T_tool0_cam = np.eye(4)

    def camera(self):
        return _pose([400.0, 0.0, 400.0], [180.0, 0.0, 0.0])


class _RoutingCam:
    def capture(self):
        return type('F', (), {'K': None, 'D': None, 'T_base_cam': np.eye(4)})()


def _routing_run(fail=(), ids=(0, 1, 2)):
    from urlab.skills import marker_localize as mloc
    markers = {i: _pose([400.0 + 60 * i, 20.0 * i, 50.0], [0.0, 0.0, 15.0 * i]) for i in ids}

    class Det:
        def detect_in_base(self, f):
            return dict(markers)

        def detect(self, f):
            return {}

    robot = _RoutingRobot(fail)
    plan = mloc.ViewPlan(C.load('object_calibration').section('marker_views'))
    seen = {m: [(T, 0.35)] * 4 for m, T in markers.items()}
    out = mloc.servo_refine(robot, _RoutingCam(), Det(), plan, seen,
                            T_overview=robot.camera())
    return robot.arm.moves, out


def test_the_servo_goes_straight_from_one_marker_to_the_next():
    """REGRESSION. It used to return to the sweep's overview pose before every marker, costing a
    full extra traverse per marker. The vantage is an ABSOLUTE pose computed from the sweep
    estimate, so the camera does not need the marker in frame before setting off."""
    moves, out = _routing_run()
    assert set(out) == {0, 1, 2}
    hops = [m for m in moves if m.startswith('overview') and 'done' not in m]
    assert hops == [], f'still detouring via the overview: {hops}'


def test_the_arm_is_still_parked_at_the_overview_when_it_finishes():
    """Whatever runs next expects the whole object in frame again, not a close-up of the last
    marker."""
    moves, _out = _routing_run()
    assert moves[-1].startswith('overview'), moves[-1]


def test_a_failed_direct_hop_recovers_through_the_overview():
    """The one thing the detour genuinely bought was reachability -- a straight path between two
    close vantages can cross a wrist limit the long way round does not. So it is tried on
    failure, rather than before every move on the chance it might be needed."""
    moves, out = _routing_run(fail=('servo marker 1 (1/4)',))
    assert any('recovery' in m for m in moves), 'the overview recovery never fired'
    assert set(out) == {0, 1, 2}, f'a marker was lost to a reachable-by-detour vantage: {out}'


def test_the_first_marker_has_no_previous_vantage_to_recover_from():
    """Nothing has moved yet, so a failure there is not a bad short hop -- the overview IS where
    the arm already is, and retrying through it would just repeat the same move."""
    moves, _out = _routing_run(fail=('servo marker 0 (1/4)',), ids=(0, 1))
    assert not any('recovery' in m for m in moves), moves
