"""COUPLER PICK AND PLACE -- the geometry and the payload arithmetic, without a robot.

Everything here is a silent failure mode if it is wrong: a place delta applied in the wrong
frame puts the object somewhere plausible, and a mis-combined payload biases every wrench
reading rather than raising anything.
"""

import numpy as np
import pytest

from urlab import config as C
from urlab import tool_frames
from urlab.apps.coupler_pick_place import (DEFAULT_LEG_MM, LEGS, combined_payload, object_rig,
                                           offset_pose, parse_offset, place_pose)
from urlab.transforms import inverse, xyzrpy_to_matrix


def _pose(xyz_mm, rpy_deg):
    return xyzrpy_to_matrix(np.array(xyz_mm, float) / 1000.0, np.radians(rpy_deg))


# ---------------------------------------------------------------------------- the legs
def test_the_default_leg_backs_straight_out_along_the_mating_axis():
    """Any other direction levers the coupler against the hole it is still inside."""
    T = _pose([100.0, 200.0, 50.0], [180.0, 0.0, 30.0])
    back = offset_pose(T, parse_offset(None, 'x', 80.0))
    assert np.allclose(back[:3, :3], T[:3, :3]), 'a standoff must not rotate the tool'
    moved = back[:3, 3] - T[:3, 3]
    assert np.allclose(moved, -0.08 * T[:3, 2]), 'the offset must be along the pose OWN -z'
    assert np.linalg.norm(moved) == pytest.approx(0.08)


def test_a_tilted_mating_axis_still_backs_out_along_itself():
    """The whole point of frame: coupler -- the retreat follows the TOOL, not base +z."""
    T = _pose([0.0, 0.0, 0.0], [0.0, 90.0, 0.0])          # mating axis now along base +x
    back = offset_pose(T, parse_offset({'distance_mm': 50.0}, 'x'))
    assert np.allclose(back[:3, 3], [-0.05, 0.0, 0.0], atol=1e-9)


def test_a_base_framed_leg_goes_straight_up_whatever_the_tool_is_doing():
    """'Lift straight up' means base +z, and must not follow a tilted mating axis."""
    T = _pose([0.0, 0.0, 0.0], [0.0, 90.0, 0.0])
    spec = parse_offset({'distance_mm': 100.0, 'axis': [0, 0, 1], 'frame': 'base'}, 'x')
    up = offset_pose(T, spec)
    assert np.allclose(up[:3, 3], [0.0, 0.0, 0.1], atol=1e-9)
    assert np.allclose(up[:3, :3], T[:3, :3]), 'a translation must not rotate the tool'


def test_the_axis_is_a_direction_and_is_normalised():
    """A length in `axis` would multiply the distance -- 100 mm along [0,0,-2] is still 100 mm."""
    T = np.eye(4)
    a = offset_pose(T, parse_offset({'distance_mm': 100.0, 'axis': [0, 0, -2]}, 'x'))
    b = offset_pose(T, parse_offset({'distance_mm': 100.0, 'axis': [0, 0, -1]}, 'x'))
    assert np.allclose(a, b)


def test_the_conops_defaults_are_the_ones_asked_for():
    """250 mm to mate, 100 mm for each of the three short hops, all along the coupler's -z."""
    assert DEFAULT_LEG_MM['mate_standoff'] == 250.0
    assert {DEFAULT_LEG_MM[k] for k in LEGS if k != 'mate_standoff'} == {100.0}
    for name in LEGS:
        spec = parse_offset(None, name, DEFAULT_LEG_MM[name])
        assert spec['frame'] == 'coupler'
        assert np.allclose(spec['axis'], [0.0, 0.0, -1.0])


@pytest.mark.parametrize('block, match', [
    ({'axis': [0.0, 0.0, 0.0]}, 'zero'),
    ({'frame': 'world'}, 'frame'),
    ({'distance_mm': -10.0}, 'negative'),
    ({'axis': [0.0, 1.0]}, 'three numbers'),
    ({'distance': 0.1}, 'unknown key'),
])
def test_a_broken_leg_is_refused(block, match):
    """A zero axis collapses the leg to no motion, which for a standoff means the compliant mate
    starts already inside the feature. A typo'd unit key would do the same silently."""
    with pytest.raises(ValueError, match=match):
        parse_offset(block, 'motion.x')


def test_a_leg_accepts_the_loaders_own_si_sibling():
    """REGRESSION. config._normalise_units adds a `distance_m` beside every `distance_mm` at
    load time, so a block written in mm arrives carrying BOTH. A strict reader that had not been
    told rejected the app's own shipped config on startup."""
    spec = parse_offset({'distance_mm': 250.0, 'distance_m': 0.25}, 'x')
    assert spec['distance_m'] == pytest.approx(0.25)
    assert parse_offset({'distance_m': 0.25}, 'x')['distance_m'] == pytest.approx(0.25)


def test_the_shipped_config_parses_every_leg():
    """That it PARSES and is usable -- not that it matches the defaults. The distances are
    tuning values (the mate standoff in particular gets shortened once the cell is trusted, to
    cut the slow compliant leg), and pinning them here would fail on every retune while saying
    nothing about whether the config is valid. The DEFAULTS are pinned separately, against
    parse_offset's own fallbacks, which is where they actually live."""
    cfg = C.load('coupler_pick_place')
    motion = cfg.section('motion')
    assert set(motion) == set(LEGS), 'a leg in the config that the app does not read'
    for name in LEGS:
        spec = parse_offset(motion[name], name, DEFAULT_LEG_MM[name])
        assert spec['distance_m'] > 0.0, f'{name} would collapse to no motion'
        assert np.linalg.norm(spec['axis']) == pytest.approx(1.0)
    # The mate standoff is where the compliant descent starts, so it has to leave room for the
    # preload push that follows it.
    mate_mm = parse_offset(motion['mate_standoff'], 'x')['distance_m'] * 1000.0
    from urlab.apps.coupler_pick_place import parse_preload
    assert mate_mm > parse_preload(cfg.section('mate_preload'))['max_travel_m'] * 1000.0, (
        'the mate standoff is shorter than the preload may travel past the target')


# ---------------------------------------------------------------------------- place delta
def test_a_base_relative_place_moves_along_the_robot_axes():
    """'300 mm to the left' means base axes, and must not depend on how the object is oriented."""
    offset = {'xyz': [0.0, 0.3, 0.0], 'rpy': [0.0, 0.0, 0.0]}
    for rpy in ([180.0, 0.0, 0.0], [180.0, 0.0, 90.0], [0.0, 90.0, 45.0]):
        pick = _pose([100.0, 200.0, 50.0], rpy)
        place = place_pose(pick, offset, 'base')
        assert np.allclose(place[:3, 3] - pick[:3, 3], [0.0, 0.3, 0.0], atol=1e-9), rpy
        assert np.allclose(place[:3, :3], pick[:3, :3], atol=1e-9), 'orientation must be kept'


def test_an_object_relative_place_moves_along_the_objects_own_axes():
    """'Lift it 50 mm along its own mating axis' -- which is NOT base z when the object is
    tilted, and is the reason the two conventions cannot be collapsed into one."""
    pick = _pose([0.0, 0.0, 0.0], [0.0, 90.0, 0.0])       # mating axis along base +x
    place = place_pose(pick, {'xyz': [0.0, 0.0, 0.05], 'rpy': [0.0, 0.0, 0.0]}, 'object')
    assert np.allclose(place[:3, 3], [0.05, 0.0, 0.0], atol=1e-9)


def test_the_two_conventions_really_differ():
    """If these ever agreed for a tilted object, one of them would be implemented wrong."""
    pick = _pose([10.0, 20.0, 30.0], [30.0, 40.0, 50.0])
    offset = {'xyz': [0.1, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.2]}
    assert not np.allclose(place_pose(pick, offset, 'base'),
                           place_pose(pick, offset, 'object'))


def test_an_object_relative_yaw_spins_about_the_mating_axis():
    """The coupler does not constrain yaw, so 'turn it a quarter turn' is a legal request --
    it must spin about the object's OWN z and leave the mating point where it is."""
    pick = _pose([100.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    place = place_pose(pick, {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, np.pi / 2]}, 'object')
    assert np.allclose(place[:3, 3], pick[:3, 3], atol=1e-9), 'a spin must not translate'
    spin = inverse(pick) @ place
    assert np.allclose(spin[:3, 2], [0.0, 0.0, 1.0], atol=1e-9), 'the axis must be unmoved'


def test_a_zero_delta_places_it_back_where_it_was_picked():
    pick = _pose([10.0, 20.0, 30.0], [180.0, 0.0, 15.0])
    for how in ('base', 'object'):
        assert np.allclose(place_pose(pick, {}, how), pick, atol=1e-12)


def test_an_unknown_place_frame_is_refused():
    """Silently picking one would move the object somewhere plausible and wrong."""
    with pytest.raises(ValueError, match='relative_to'):
        place_pose(np.eye(4), {'xyz': [0.1, 0, 0]}, 'world')


# ---------------------------------------------------------------------------- payload
def test_the_combined_payload_is_a_mass_weighted_mean_not_just_more_mass():
    """Adding the object's mass at the TOOL's CoG would claim its weight acts where it does
    not; the compensation residual then shows up as an orientation-dependent force bias."""
    tool = {'mass_kg': 1.0, 'cog_m': [0.0, 0.0, 0.0]}
    out = combined_payload(tool, 3.0, [0.0, 0.0, 0.100])
    assert out['mass_kg'] == pytest.approx(4.0)
    assert np.allclose(out['cog_m'], [0.0, 0.0, 0.075]), 'CoG must move toward the heavier part'


def test_carrying_nothing_leaves_the_tool_payload_untouched():
    tool = {'mass_kg': 1.3, 'cog_m': [-0.026, 0.028, 0.030]}
    for held in (None, 0, 0.0):
        out = combined_payload(tool, held, [0.0, 0.0, 0.055])
        assert out['mass_kg'] == pytest.approx(1.3)
        assert np.allclose(out['cog_m'], tool['cog_m'])


def test_the_real_tool_and_the_real_object_combine_sanely():
    """The shipped numbers, DERIVED from the configs rather than spelled out -- the tool mass is
    a property of whatever is bolted on today and the object mass is a measurement, so pinning
    either would make this test fail on a hardware change it has no opinion about."""
    cfg = C.load('coupler_pick_place')
    tool = cfg.section('robot').get('payload', {})
    obj = tool_frames.load_objects(cfg)['ORU']
    coupler_z = tool_frames.coupler_mate(cfg)[2, 3]
    out = combined_payload(tool, obj['held_mass_kg'], tool_frames.coupler_mate(cfg)[:3, 3])

    assert out['mass_kg'] == pytest.approx(tool['mass_kg'] + obj['held_mass_kg'])
    # The combined CoG must land strictly BETWEEN the tool's and the object's, and -- because
    # the object outweighs the tool here -- nearer the object's.
    lo, hi = sorted((tool['cog_m'][2], coupler_z))
    assert lo < out['cog_m'][2] < hi, 'the combined CoG left the segment between the two'
    assert obj['held_mass_kg'] > tool['mass_kg'], 'premise of the next assertion'
    assert abs(out['cog_m'][2] - coupler_z) < abs(out['cog_m'][2] - tool['cog_m'][2]), (
        'the heavier body must pull the combined CoG toward itself')


# ---------------------------------------------------------------------------- the rig shim
def test_the_catalogue_entry_becomes_a_rig_that_locate_can_use():
    """locate() returns T_base_marker @ T_marker_target, so feeding it each marker's mating
    offset as the 'target' makes it return the coupler pose directly -- the whole pick."""
    obj = tool_frames.load_objects(C.load('coupler_pick_place'))['ORU']
    rig = object_rig(obj, 'DICT_4X4_50')
    assert rig['dictionary'] == 'DICT_4X4_50'
    assert set(rig['markers']) == set(obj['markers'])
    for mid, m in obj['markers'].items():
        entry = rig['markers'][mid]
        assert np.allclose(entry['T_marker_target'], m['T_marker_grasp'])
        assert entry['size_m'] == pytest.approx(m['size_m'])


def test_every_marker_on_an_object_becomes_a_voter():
    """The point of several markers: each one is an independent vote for the SAME grasp frame,
    which is what lets the run-time RANSAC reject one that has been knocked."""
    obj = {'markers': {7: {'size_m': 0.02, 'T_marker_grasp': _pose([10, 0, 0], [0, 0, 0])},
                       8: {'size_m': 0.03, 'T_marker_grasp': _pose([0, 10, 0], [0, 0, 0])},
                       9: {'size_m': 0.04, 'T_marker_grasp': _pose([0, 0, 10], [0, 0, 0])}}}
    rig = object_rig(obj)
    assert set(rig['markers']) == {7, 8, 9}
    assert {m['size_m'] for m in rig['markers'].values()} == {0.02, 0.03, 0.04}, (
        'per-marker sizes were collapsed -- solvePnP would solve two of them at the wrong depth')


def test_the_rig_shim_survives_the_joint_pnp_object_points():
    """rig_object_points() is what the run-time joint solve is built on; a shim that did not
    satisfy it would fail deep inside the solver on the robot instead of here."""
    pytest.importorskip('cv2')
    from urlab.skills.marker_localize import rig_object_points

    obj = tool_frames.load_objects(C.load('coupler_pick_place'))['ORU']
    pts = rig_object_points(object_rig(obj))
    assert set(pts) == set(obj['markers'])
    for mid, corners in pts.items():
        assert corners.shape == (4, 3)
        side = np.linalg.norm(corners[0] - corners[1])
        assert side == pytest.approx(obj['markers'][mid]['size_m'], rel=1e-6), (
            f'marker {mid} lost its printed size')


# ---------------------------------------------------------------------------- config wiring
def test_the_shipped_config_is_soft_and_guarded():
    """The defaults the user asked for: admittance on, soft translational stiffness, and a hard
    guard above it. A stiff default would lever the coupler against a camera-found feature."""
    cfg = C.load('coupler_pick_place')
    comp = cfg.section('compliance')
    assert comp['enabled'] is True
    assert min(comp['stiffness']) > 0.0, 'a zero stiffness is force control, not admittance'
    assert min(comp['damping_ratio']) >= 1.0, 'an underdamped law with 3 kg on it will ring'
    assert cfg.get_path('force_guard.enabled') is True
    assert cfg.get_path('force_guard.max_force_n') > 0.0
    # NO SOFTNESS CEILING IS PINNED HERE. The stiffness is a TUNING value -- raised when the
    # object gets pushed around on the descent, lowered when the coupler jams on entry -- and a
    # magic number in a test would just have to be edited every time it is tuned. What is
    # structural is the RELATIONSHIP: the law has to be able to yield a usable distance before
    # the guard stops the arm, or the compliance is decorative.
    yield_mm = cfg.get_path('force_guard.max_force_n') / min(comp['stiffness'][:3]) * 1000.0
    clamp_mm = cfg.get_path('compliance.max_delta_mm')
    assert min(yield_mm, clamp_mm) >= 10.0, (
        f'the law can only yield {min(yield_mm, clamp_mm):.0f} mm before the guard or the clamp '
        'stops it -- too little for a camera-found feature')


def test_a_configured_default_object_is_one_that_actually_exists():
    """object_name may be left null (the app then lists what is available and stops) or pinned
    to a default. What it must never be is a name the catalogue does not have -- that is a typo
    that only shows up when someone runs the cell."""
    cfg = C.load('coupler_pick_place')
    name = cfg.get('object_name')
    if name is None:
        return
    catalogue = tool_frames.load_objects(cfg)
    assert name in catalogue, (
        f'object_name is {name!r}, which is not in the catalogue. Have: '
        f'{sorted(catalogue)}')


# ---------------------------------------------------------------------------- the interlock
class _FakeBoard:
    """Shaped like toolchanger.ToolChanger.

    The contract that matters: hold/release/status return True when the proximity sensor AGREES
    with the commanded state and FALSE on an emergency stop -- 'asked to hold but nothing is
    gripped'. They do not raise for that; a False IS the refusal.
    """

    port = '/dev/fake'
    booted = True

    def __init__(self, hold=True, release=True, status=True, boom=False, motor_on=False):
        self._hold, self._release, self._status, self._boom = hold, release, status, boom
        self.motor_on = motor_on          # the RELAY, which the wire command TOGGLES
        self.calls = []

    def motor(self):
        self.calls.append('motor')
        if self._boom:
            raise RuntimeError('serial timeout')
        self.motor_on = not self.motor_on
        return self.motor_on

    def _answer(self, what, value):
        self.calls.append(what)
        if self._boom:
            raise RuntimeError('serial timeout')
        return value

    def hold(self):
        return self._answer('hold', self._hold)

    def release(self):
        return self._answer('release', self._release)

    def status(self):
        return self._answer('status', self._status)

    def close(self):
        pass


def _coupler(board=None, **over):
    from urlab.robot.coupler import Coupler
    cfg = C.load('coupler_pick_place', ['toolchanger.enabled=false'])
    cfg.update(over)
    c = Coupler(cfg)
    c.device = board
    return c


def test_a_refused_hold_is_a_failure_not_a_success():
    """REGRESSION. The wrapper used to discard the driver's return value and report every hold
    as verified. A refused hold means the coupler is gripping NOTHING -- treating it as success
    sets a payload for an absent object, lifts air, and carries a phantom 30 N bias into every
    force reading afterwards."""
    c = _coupler(_FakeBoard(hold=False))
    assert c.hold() is False, 'an emergency stop must not read as a successful lock'
    assert c.verified is False


def test_a_confirmed_hold_is_verified():
    c = _coupler(_FakeBoard(hold=True))
    assert c.hold() is True
    assert c.verified is True, 'the board agreed -- that is what verified means'


def test_a_refused_release_is_a_failure():
    """The sensor still sees the object: it has NOT let go, and withdrawing would drag it."""
    c = _coupler(_FakeBoard(release=False))
    assert c.release() is False
    assert c.verified is False


def test_a_serial_failure_is_never_reported_as_a_grip():
    c = _coupler(_FakeBoard(boom=True))
    assert c.hold() is False
    assert c.verified is False


def test_verify_says_none_without_a_board_and_never_false():
    """None ('nobody checked') and False ('checked, and it is not held') must not collapse into
    each other -- the first is a warning and the second stops the run."""
    assert _coupler(None).verify() is None
    assert _coupler(_FakeBoard(status=True)).verify() is True
    assert _coupler(_FakeBoard(status=False)).verify() is False


def test_the_pick_refuses_to_lift_when_the_lock_was_refused():
    """The whole point of the interlock, at the level that matters: no lock, no lift."""
    job = cpp_job(_FakeBoard(hold=False))
    assert job.lock() is False


def test_the_pick_refuses_to_lift_when_the_recheck_disagrees():
    """hold() confirms at the instant the servo stops; a part resting where the probe could see
    it can settle out in the moment after. The second look is what catches that."""
    job = cpp_job(_FakeBoard(hold=True, status=False))
    assert job.lock() is False


def test_the_pick_proceeds_when_both_checks_agree():
    job = cpp_job(_FakeBoard(hold=True, status=True))
    assert job.lock() is True


def cpp_job(board):
    from urlab.apps import coupler_pick_place as cpp
    job = cpp.CouplerCycle.__new__(cpp.CouplerCycle)
    job.coupler = _coupler(board)
    return job


# ---------------------------------------------------------------------------- stepping
class _GateArm:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run


class _GateRobot:
    def __init__(self, dry_run=False):
        self.arm = _GateArm(dry_run)


def test_the_step_gate_is_off_by_default_and_on_when_asked():
    from urlab.apps import coupler_pick_place as cpp
    assert cpp.step_gate(C.Config({}), _GateRobot()) is None
    assert cpp.step_gate(C.Config({'confirm_each_step': True}), _GateRobot()) is not None


def test_a_dry_run_never_waits_for_a_human():
    """A simulated run must not hang on a prompt nobody is there to answer."""
    from urlab.apps import coupler_pick_place as cpp
    cfg = C.Config({'confirm_each_step': True})
    assert cpp.step_gate(cfg, _GateRobot(dry_run=True)) is None


def test_no_prompts_beats_confirm_each_step():
    """--no-prompts is the stronger switch: it silences gates that --yes deliberately keeps."""
    from urlab.apps import coupler_pick_place as cpp
    cfg = C.Config({'confirm_each_step': True, 'skip_prompts': True})
    assert cpp.step_gate(cfg, _GateRobot()) is None


def test_declining_a_step_fails_it_rather_than_skipping_it():
    """bt.Action treats a refused confirm as FAILURE, which stops the sequence. Skipping instead
    would let a run release an object the arm never carried to the place pose."""
    from urlab import behaviors as bt
    ran = []
    act = bt.Action('place the object', lambda: ran.append(1), confirm=lambda _label: False)
    act.setup()
    assert act.update() == bt.core.Status.FAILURE
    assert not ran, 'the action ran despite being declined'


def test_the_gated_steps_are_the_ones_that_move():
    """The payload updates and the report move nothing, so they are deliberately not gated."""
    import inspect
    from urlab.apps import coupler_pick_place as cpp
    src = inspect.getsource(cpp.build_and_run)
    # `carry` and `set_down` are wired with an f-string label spanning a line break, so match
    # the call rather than the exact one-line spelling.
    for moving in ('job.locate', 'job.approach', 'job.descend_and_mate', 'job.lock',
                   'job.lift', 'job.carry', 'job.set_down', 'job.unlock', 'job.withdraw'):
        call = src[src.index(moving):src.index(moving) + 120]
        assert 'confirm=step' in call, f'{moving} is not gated'
    for still in ('job.take_payload', 'job.drop_payload', 'job.report',
                  'job.settle_after_lift'):
        assert f'{still}, confirm=step' not in src, f'{still} moves nothing but is gated'


# ---------------------------------------------------------------------------- coupler_actuate
def test_the_actuate_app_refuses_to_run_without_a_board():
    """A coupler check that passes because a human pressed Enter has tested the human."""
    from urlab.apps import coupler_actuate as ca
    cfg = C.load('coupler_actuate', ['toolchanger.enabled=false'])
    assert ca.build_and_run(cfg, _coupler(None)) is False


def test_the_actuate_app_runs_the_manual_path_when_told_to():
    from urlab.apps import coupler_actuate as ca
    cfg = C.load('coupler_actuate', ['toolchanger.enabled=false', 'require_driver=false',
                                     'skip_prompts=true'])
    assert ca.build_and_run(cfg, _coupler(None, skip_prompts=True)) is True


def test_the_actuate_app_locks_then_releases_in_that_order():
    """The full sequence, including the two steps the failed pick was missing: the relay is
    powered and the jaws forced open BEFORE the first hold, and the relay is only switched off
    at the end, once nothing is held."""
    from urlab.apps import coupler_actuate as ca
    board = _FakeBoard()
    cfg = C.load('coupler_actuate', ['skip_prompts=true'])
    assert ca.build_and_run(cfg, _coupler(board)) is True
    assert board.calls == ['status',            # where does the board think it stands
                           'motor', 'release',  # power it, and break any boot-time latch
                           'hold', 'release',   # the cycle itself
                           'motor'], board.calls
    assert board.motor_on is False, 'the relay was left on'


def test_the_actuate_app_reports_a_refused_lock_as_a_failure():
    """Locking an empty coupler SHOULD be refused -- and that refusal must exit non-zero, or the
    interlock reads as working when it is the thing being tested."""
    from urlab.apps import coupler_actuate as ca
    board = _FakeBoard(hold=False)
    cfg = C.load('coupler_actuate', ['skip_prompts=true'])
    assert ca.build_and_run(cfg, _coupler(board)) is False
    # The prepare step legitimately releases first; what must not happen is a release AFTER the
    # refused hold, nor the relay being cut while the state is unknown.
    assert board.calls[board.calls.index('hold'):] == ['hold'], board.calls
    assert board.motor_on is True, 'the relay was cut after a failure, possibly under a load'


def test_the_actuate_app_cycles_as_many_times_as_asked():
    from urlab.apps import coupler_actuate as ca
    board = _FakeBoard()
    cfg = C.load('coupler_actuate', ['skip_prompts=true', 'cycles=3'])
    assert ca.build_and_run(cfg, _coupler(board)) is True
    assert board.calls.count('hold') == 3
    assert board.calls.count('release') == 4, 'three cycles plus the one that opens the jaws'


def test_stopping_before_the_release_leaves_the_coupler_locked():
    """Deliberate: the operator may be holding the tool, and a coupler that lets go because
    someone typed 'q' is a dropped tool."""
    from urlab.apps import coupler_actuate as ca
    board = _FakeBoard()
    answers = iter([True, False])           # yes to lock, no to release
    assert ca.cycle(_coupler(board), 1, 1, lambda _p: next(answers)) is False
    assert board.calls == ['hold'], 'it released anyway'


# ---------------------------------------------------------------------------- the motor relay
def test_the_motor_is_driven_to_a_state_not_pulsed():
    """REGRESSION for the failed pick. The wire command TOGGLES, so a blind call is a coin flip
    -- which is why the relay 'sometimes defaults to on'. Driving it must converge either way."""
    for already_on in (False, True):
        board = _FakeBoard(motor_on=already_on)
        c = _coupler(board)
        c._motor_on = None                      # state unknown, as after a bannerless boot
        assert c.set_motor(True) is True
        assert board.motor_on is True, f'relay did not end ON (started {already_on})'
        assert board.calls.count('motor') <= 2, 'more than two round trips to set one relay'


def test_a_known_relay_state_costs_no_round_trip():
    board = _FakeBoard(motor_on=False)
    c = _coupler(board)
    c._motor_on = False                         # what the boot banner tells us
    assert c.set_motor(False) is True
    assert board.calls.count('motor') == 0


def test_preparing_to_mate_powers_the_relay_and_opens_the_jaws():
    """THE BUG. A board that booted with something in front of the probe comes up believing it
    is locked, and changeStatus() only moves the servo when the status CHANGES -- so a hold in
    that state confirms without clamping. The release first is what makes the hold real."""
    board = _FakeBoard(motor_on=False)
    c = _coupler(board)
    c._motor_on = False
    assert c.prepare_to_mate() is True
    assert board.calls == ['motor', 'release'], board.calls
    assert board.motor_on is True


def test_preparing_to_mate_fails_loudly_if_the_relay_will_not_switch():
    class _Stuck(_FakeBoard):
        def motor(self):
            self.calls.append('motor')
            return False                        # never comes on

    board = _Stuck()
    c = _coupler(board)
    c._motor_on = None
    assert c.prepare_to_mate() is False
    assert 'release' not in board.calls, 'it opened the jaws despite having no power'


def test_the_motor_can_be_switched_off_for_a_mechanism_without_one():
    board = _FakeBoard()
    c = _coupler(board)
    c.use_motor = False
    assert c.set_motor(True) is True
    assert board.calls.count('motor') == 0


def test_the_pick_powers_the_coupler_before_it_goes_near_the_object():
    """Order matters: the relay and the jaws are sorted out BEFORE the arm approaches, so a
    dead relay stops the run with nothing moved rather than halfway into a feature."""
    import inspect
    from urlab.apps import coupler_pick_place as cpp
    src = inspect.getsource(cpp.build_and_run)
    assert src.index('job.prepare_coupler') < src.index('job.approach') < src.index(
        'job.descend_and_mate'), 'the coupler must be powered and opened before the approach'
    assert src.index('job.unlock') < src.index('job.park_coupler'), (
        'the motor must not be cut while the object is still held')


# ---------------------------------------------------------------------------- multi-marker
def _voting_rig():
    """Three markers bolted around one grasp frame, each storing it in its own coordinates."""
    from urlab.transforms import inverse
    G = np.eye(4)
    placements = {0: _pose([90.0, 20.0, 0.0], [0.0, 0.0, 15.0]),
                  1: _pose([-80.0, 60.0, 0.0], [0.0, 0.0, -70.0]),
                  2: _pose([10.0, -100.0, 0.0], [0.0, 0.0, 160.0])}
    obj = {'markers': {m: {'size_m': 0.04, 'T_marker_grasp': inverse(P) @ G}
                       for m, P in placements.items()}}
    return object_rig(obj), placements, G


def _fused(placements, T_base_obj, ids, knock=None, knock_mm=8.0):
    out = {}
    for m in ids:
        P = placements[m]
        if m == knock:
            P = P @ _pose([knock_mm, 0.0, 0.0], [0.0, 0.0, 6.0])
        out[m] = (T_base_obj @ P, 0.0004, np.radians(0.2), 6, 1.0)
    return out


def _vote(ids, knock=None, knock_mm=8.0):
    from urlab.skills import marker_localize as mloc
    from urlab.transforms import pose_error
    rig, placements, G = _voting_rig()
    plan = mloc.ViewPlan(C.load('coupler_pick_place').section('marker_views'))
    T_base_obj = _pose([500.0, 0.0, 50.0], [180.0, 0.0, 20.0])
    T, votes = mloc.vote_target(rig, _fused(placements, T_base_obj, ids, knock, knock_mm), plan)
    if T is None:
        return None, len(votes)
    lin, _ang = pose_error(T_base_obj @ G, T)
    return lin * 1000.0, len(votes)


def test_every_marker_votes_and_they_agree():
    err, n = _vote([0, 1, 2])
    assert n == 3 and err == pytest.approx(0.0, abs=1e-6)


def test_an_occluded_marker_is_not_fatal():
    """The reason to put several on an object: one out of shot must not stop the pick."""
    err, n = _vote([0, 1])
    assert n == 2 and err == pytest.approx(0.0, abs=1e-6)


def test_three_markers_outvote_a_knocked_one():
    """RANSAC rejects the odd one out and the run carries on, exactly right."""
    err, _n = _vote([0, 1, 2], knock=2)
    assert err == pytest.approx(0.0, abs=1e-6), 'the knocked marker dragged the answer'


def test_two_markers_that_disagree_refuse_rather_than_average():
    """REGRESSION on the gate VALUE. It measures spread ABOUT THE MEAN, so with two markers a
    knock of X mm shows up as only X/2. The library default of 5 mm -- which this config used to
    inherit silently -- let an 8 mm knock through as a 4 mm pick error. Nothing can say which of
    two markers moved, so refusing is the only honest answer."""
    err, _n = _vote([0, 1], knock=1, knock_mm=8.0)
    assert err is None, 'an 8 mm knock across two markers was averaged instead of refused'


def test_the_gate_is_tight_enough_to_halve():
    """Explicitly pinned, not inherited: with two markers the gate catches knocks of about twice
    its value, so it has to sit well under the error the coupler cannot absorb."""
    plan_cfg = C.load('coupler_pick_place').section('marker_views')
    assert plan_cfg['max_disagreement_mm'] <= 3.0, 'the gate drifted back toward the default'
    assert plan_cfg['max_disagreement_deg'] <= 3.0
    assert plan_cfg['require_all_markers'] is False, 'occlusion must not be fatal'
    assert plan_cfg['min_markers'] == 1, 'single-marker objects are still in the catalogue'
    assert plan_cfg['joint_pnp'] is True, 'the wide baseline is the point of several markers'


# ---------------------------------------------------------------------------- mate preload
def test_the_mating_direction_is_the_standoff_axis_reversed():
    """One axis, two signs. Configuring them apart would let the arm approach along one line and
    preload along another."""
    from urlab.apps.coupler_pick_place import mating_direction
    spec = parse_offset({'distance_mm': 250.0, 'axis': [0, 0, -1]}, 'x')
    mate = mating_direction(spec)
    assert np.allclose(mate['axis'], -spec['axis'])
    assert mate['frame'] == spec['frame'], 'the frame must carry over, or the two disagree'


def test_the_preload_defaults_are_on_at_one_newton():
    from urlab.apps.coupler_pick_place import parse_preload
    d = parse_preload(None)
    assert d['enabled'] is True and d['force_n'] == pytest.approx(1.0)
    shipped = parse_preload(C.load('coupler_pick_place').section('mate_preload'))
    assert shipped['enabled'] is True and shipped['force_n'] == pytest.approx(1.0)


@pytest.mark.parametrize('block, match', [
    ({'force_n': 0.0}, 'positive'),
    ({'force_n': -1.0}, 'positive'),
    ({'step_mm': 0.0}, 'positive'),
    ({'max_travel_mm': 0.0}, 'positive'),
    ({'threshold_n': 1.0}, 'unknown key'),
])
def test_a_broken_preload_block_is_refused(block, match):
    from urlab.apps.coupler_pick_place import parse_preload
    with pytest.raises(ValueError, match=match):
        parse_preload(block)


class _PreloadArm:
    """Reports a contact force that grows with how far the reference has been pushed."""

    dry_run = False

    def __init__(self, newtons_per_mm=1.0, cap=None):
        self.n_per_mm, self.cap = newtons_per_mm, cap
        self.pushed_mm = 0.0
        self.tares = []
        self.events = []

    def wrench(self):
        f = self.pushed_mm * self.n_per_mm
        if self.cap is not None:
            f = min(f, self.cap)
        # EXTERNAL force ON the tool: pushing down into the feature is felt pushing back UP.
        self.events.append(('read', round(f, 3)))
        return np.array([0.0, 0.0, +f, 0.0, 0.0, 0.0])

    def zero_ft(self, settle=True):
        self.tares.append(self.pushed_mm)
        self.events.append(('tare', round(self.pushed_mm, 3)))

    def servo_stop(self):
        pass


def _preload_job(arm, **over):
    from urlab.apps import coupler_pick_place as cpp
    job = cpp.CouplerCycle.__new__(cpp.CouplerCycle)
    cfg = C.load('coupler_pick_place')
    job.cfg = cfg
    job.robot = type('R', (), {'arm': arm})()
    job.T_tool0_coupler = np.eye(4)
    job.legs = {n: parse_offset({'distance_mm': 250.0, 'axis': [0, 0, -1]}, 'x')
                for n in ('mate_standoff', 'place_standoff')}
    job.preload_mate = dict(cpp.parse_preload(cfg.section('mate_preload')), **over)
    job.preload_place = dict(cpp.parse_preload(cfg.section('place_preload')), **over)
    job.obj = {'held_mass_kg': 3.3}
    job.guard = type('G', (), {'reset': lambda self: None})()

    class _Adm:
        S = np.array([1000.0] * 3 + [8.0] * 3)

        def ramp(self, a, b, *args, **kw):
            # HOW FAR the reference advanced, whatever axis it went along -- the mate and the
            # placement travel on different legs, so a z-only stub would silently measure
            # nothing for one of them.
            arm.pushed_mm += float(np.linalg.norm(b[:3, 3] - a[:3, 3])) * 1000.0
            arm.events.append(('ramp', round(arm.pushed_mm, 3)))
            return 'done'

    job.adm = job.adm_loaded = job.adm_insert = _Adm()
    job.holding = False              # the preload happens before the object is on
    job._active_law = None           # set by _compliant; the push falls back to _adm()
    return job


def test_the_push_stops_as_soon_as_the_preload_is_measured():
    """1 N at 1 N/mm is one step past the target -- it must not keep shoving to max travel."""
    arm = _PreloadArm(newtons_per_mm=1.0)
    job = _preload_job(arm)
    # +z of the tool is the mating direction here, so the reaction is along -z: flip the stub.
    arm.wrench = lambda: np.array([0.0, 0.0, -arm.pushed_mm, 0.0, 0.0, 0.0])
    assert job._push_to_preload(np.eye(4), lambda: None, job.preload_mate,
                                'mate_standoff', 'mate') is True
    assert arm.pushed_mm == pytest.approx(1.0, abs=0.51), arm.pushed_mm
    assert arm.pushed_mm < job.preload_mate['max_travel_m'] * 1000.0, 'it pushed to the limit'


def test_no_contact_fails_rather_than_lifting_air():
    """Running out of travel without the force means nothing was there. Continuing would lock
    onto nothing, set a payload for an absent object and carry it nowhere."""
    arm = _PreloadArm(newtons_per_mm=0.0)
    job = _preload_job(arm)
    arm.wrench = lambda: np.zeros(6)
    assert job._push_to_preload(np.eye(4), lambda: None, job.preload_mate,
                                'mate_standoff', 'mate') is False
    assert arm.pushed_mm == pytest.approx(job.preload_mate['max_travel_m'] * 1000.0, abs=0.51)


def test_disabling_the_preload_skips_the_push_entirely():
    arm = _PreloadArm()
    job = _preload_job(arm, enabled=False)
    assert job._push_to_preload(np.eye(4), lambda: None, job.preload_mate,
                                'mate_standoff', 'mate') is True
    assert arm.pushed_mm == 0.0 and arm.events == []


def test_the_side_load_of_a_jammed_entry_does_not_count_as_preload():
    """The force is PROJECTED onto the mating axis. A magnitude would call 5 N of lateral jam a
    met preload while the coupler is hung up on a lip instead of seated."""
    arm = _PreloadArm(newtons_per_mm=0.0)
    job = _preload_job(arm)
    arm.wrench = lambda: np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])   # all lateral
    assert job._push_to_preload(np.eye(4), lambda: None, job.preload_mate,
                                'mate_standoff', 'mate') is False, (
        'a pure side load was accepted as seating pressure')


# ---------------------------------------------------------------------------- loaded compliance
def test_the_payload_mass_reaches_the_controller():
    """The catalogue mass must actually be applied, or every wrench after the pick carries the
    object's weight as external force and the guard and the law both act on a wrong number."""
    cfg = C.load('coupler_pick_place')
    obj = tool_frames.load_objects(cfg)[cfg.get('object_name')]
    assert obj['held_mass_kg'] == pytest.approx(3.3), 'the ORU mass is not in the catalogue'
    tool = cfg.section('robot').get('payload', {})
    out = combined_payload(tool, obj['held_mass_kg'], tool_frames.coupler_mate(cfg)[:3, 3])
    assert out['mass_kg'] == pytest.approx(tool['mass_kg'] + 3.3)


def test_the_loaded_law_is_slower_than_the_free_one():
    """THE OSCILLATION FIX. The virtual mass is a design parameter and does not follow the
    payload, so a law tuned for a bare coupler outruns the loaded arm. Raising it lowers the
    law's bandwidth (sqrt(S/M)) until the reference stops asking for accelerations the arm
    cannot track."""
    from urlab.apps.coupler_pick_place import compliance_blocks, law_bandwidth_hz
    free, loaded, _insert = compliance_blocks(C.load('coupler_pick_place'))
    f_hz = law_bandwidth_hz(free['mass'], free['stiffness'])[0]
    l_hz = law_bandwidth_hz(loaded['mass'], loaded['stiffness'])[0]
    assert l_hz < f_hz, 'the loaded law is not slower than the free one'
    assert loaded['mass'][0] >= 3.6, 'the virtual mass is under the real mass it has to move'
    assert min(loaded['damping_ratio']) >= min(free['damping_ratio'])


def test_the_loaded_block_falls_back_to_the_free_one():
    """Absent, the loaded law IS the free one -- the old behaviour, fine for a light object."""
    from urlab.apps.coupler_pick_place import compliance_blocks
    cfg = C.load('coupler_pick_place')
    cfg['compliance_loaded'] = None
    free, loaded, _insert = compliance_blocks(cfg)
    assert loaded is free


def test_the_law_switches_exactly_when_the_mass_does():
    """The gains must change at the same instant the payload does -- one set being a compromise
    for both states is what a single block forces."""
    from urlab.apps import coupler_pick_place as cpp
    job = cpp.CouplerCycle.__new__(cpp.CouplerCycle)
    job.cfg = C.load('coupler_pick_place')
    job.adm, job.adm_loaded = object(), object()
    job.holding = False
    assert job._adm() is job.adm
    job.holding = True
    assert job._adm() is job.adm_loaded, 'the loaded law was not selected while holding'


def test_bandwidth_is_the_textbook_second_order_formula():
    from urlab.apps.coupler_pick_place import law_bandwidth_hz
    assert law_bandwidth_hz([1.0], [4.0 * np.pi ** 2])[0] == pytest.approx(1.0)
    assert law_bandwidth_hz([4.0], [4.0 * np.pi ** 2])[0] == pytest.approx(0.5), (
        'quadrupling the virtual mass must halve the bandwidth'
    )


# ---------------------------------------------------------------------------- place preload
def test_the_placement_preload_is_on_with_a_usable_threshold():
    """The DEFAULT is pinned against parse_preload's own fallback; the shipped force is a
    TUNING value -- how hard to press before letting go depends on the object and the surface,
    and the run prints what fraction of the weight it corresponds to. Pinning it here would fail
    on every retune while saying nothing about whether the feature works."""
    from urlab.apps.coupler_pick_place import parse_preload
    assert parse_preload(None, 'place_preload')['force_n'] == pytest.approx(1.0)
    cfg = C.load('coupler_pick_place')
    shipped = parse_preload(cfg.section('place_preload'), 'place_preload')
    assert shipped['enabled'] is True and shipped['force_n'] > 0.0
    # It has to be reachable inside the travel the block allows, at the loaded stiffness.
    from urlab.apps.coupler_pick_place import compliance_blocks
    stiffness = compliance_blocks(cfg)[1]['stiffness'][0]
    needed_mm = shipped['force_n'] / stiffness * 1000.0
    assert needed_mm <= shipped['max_travel_m'] * 1000.0, (
        f'{shipped["force_n"]} N needs ~{needed_mm:.1f} mm of reference travel at '
        f'{stiffness:.0f} N/m, but max_travel_mm is only '
        f'{shipped["max_travel_m"] * 1000.0:.1f}')


def test_the_placement_travels_along_the_place_standoff_reversed():
    """Its own leg, not the mate's -- the two standoffs are configured separately and a
    placement that pushed along the MATE axis would drive sideways into the bench."""
    from urlab.apps.coupler_pick_place import mating_direction
    cfg = C.load('coupler_pick_place')
    place = parse_offset(cfg.section('motion')['place_standoff'], 'place_standoff')
    assert np.allclose(mating_direction(place)['axis'], -place['axis'])


def test_the_placement_push_uses_the_place_leg_and_the_loaded_law():
    """While placing, the object is on the coupler -- so the LOADED compliance must be in use,
    and the travel must follow the place standoff."""
    arm = _PreloadArm(newtons_per_mm=1.0)
    job = _preload_job(arm)
    job.holding = True
    job.legs['place_standoff'] = parse_offset({'distance_mm': 100.0, 'axis': [0, -1, 0]}, 'x')
    # travel is +y, so the surface pushes back along -y
    arm.wrench = lambda: np.array([0.0, -arm.pushed_mm, 0.0, 0.0, 0.0, 0.0])
    assert job._push_to_preload(np.eye(4), lambda: None, job.preload_place,
                                'place_standoff', 'placement') is True
    assert job._adm() is job.adm_loaded, 'the free law was used while carrying the object'


def test_the_placement_reports_what_fraction_of_the_weight_landed():
    """1 N against a 32 N object is a KISS, not a set-down. Invisible unless printed, and the
    difference decides whether releasing there drops the object."""
    arm = _PreloadArm()
    job = _preload_job(arm)
    job.holding = True
    note = job._weight_note(1.0)
    assert '3%' in note and '32.4 N' in note, note
    assert job._weight_note(16.2).startswith(' The surface is taking 50%'), job._weight_note(16.2)
    job.holding = False
    assert job._weight_note(1.0) == '', 'the mate must not claim to be weighing anything'


def test_both_preloads_share_one_implementation():
    """Same manoeuvre, opposite ends of the cycle. Two copies would drift apart."""
    import inspect
    from urlab.apps import coupler_pick_place as cpp
    src = inspect.getsource(cpp.CouplerCycle)
    assert src.count('def _push_to_preload') == 1
    assert "'mate_standoff', 'mate'" in src
    assert 'self.TARGET_STANDOFF,' in src and 'self.TARGET_WORD' in src, (
        'the target-side push must go through the same hook the subclass overrides')


# ---------------------------------------------------------------------------- tare timing
class _TareArm:
    dry_run = False

    def __init__(self):
        self.tares = []
        self.stage = 'resting'

    def zero_ft(self, settle=True):
        self.tares.append(self.stage)

    def set_payload(self, p):
        pass

    def wrench(self):
        return np.zeros(6)

    def servo_stop(self):
        pass


def _tare_job(arm):
    from urlab.apps import coupler_pick_place as cpp
    job = cpp.CouplerCycle.__new__(cpp.CouplerCycle)
    job.cfg = C.load('coupler_pick_place')
    job.robot = type('R', (), {'arm': arm})()
    job.name, job.obj = 'x', {'held_mass_kg': 3.3}
    job.T_tool0_coupler = np.eye(4)
    job.tare_before, job.holding = True, False
    # take_payload logs both laws' bandwidths, so the stubs need M and S.
    law = type('L', (), {'M': np.array([5.0] * 6), 'S': np.array([1000.0] * 6)})
    job.adm, job.adm_loaded = law(), law()
    return job


def test_the_pick_does_not_tare_while_the_object_is_still_supported():
    """REGRESSION, and the cause of the angled lift. Locked on but still resting, the surface is
    carrying the object -- with the payload already set for the airborne case that reads as a
    real ~32 N. Zeroing it declares that force to be nothing, so the instant the object lifts
    clear the law sees the whole weight as a phantom and yields 32 mm to it."""
    arm = _TareArm()
    job = _tare_job(arm)
    job.take_payload()
    assert arm.tares == [], 'take_payload tared while the object was still on the bench'


def test_the_release_does_not_tare_while_the_coupler_is_still_pressed():
    """The mirror: released but still down in the object, carrying the placement preload."""
    arm = _TareArm()
    job = _tare_job(arm)
    job.holding = True
    job.drop_payload()
    assert arm.tares == [], 'drop_payload tared against the residual preload'


def test_the_tare_happens_once_the_object_is_hanging_free():
    """The one moment in the cycle it genuinely is -- and everything downstream reads against
    this zero, so it also absorbs whatever the assumed payload CoG got wrong."""
    arm = _TareArm()
    job = _tare_job(arm)
    arm.stage = 'airborne'
    assert job.settle_after_lift() is True
    assert arm.tares == ['airborne']


def test_the_legs_that_start_in_contact_skip_their_tare():
    """_compliant tares at the pose it starts from, so the legs that begin already touching have
    to opt out -- otherwise the fix above is undone one line later."""
    import inspect
    from urlab.apps import coupler_pick_place as cpp
    for meth in ('lift', 'withdraw'):
        src = inspect.getsource(getattr(cpp.CouplerCycle, meth))
        assert 'tare=False' in src, f'{meth} still tares against a force that is already there'
    for meth in ('descend_and_mate', 'set_down'):
        src = inspect.getsource(getattr(cpp.CouplerCycle, meth))
        assert 'tare=False' not in src, f'{meth} must tare -- it starts clear of contact'


def test_the_tare_runs_between_the_lift_and_the_carry():
    import inspect
    from urlab.apps import coupler_pick_place as cpp
    src = inspect.getsource(cpp.build_and_run)
    assert src.index('job.lift') < src.index('job.settle_after_lift') < src.index('job.carry')
