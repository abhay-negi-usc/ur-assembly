"""COUPLER PICK-AND-ASSEMBLE and the assembly calibration that feeds it, without a robot.

The interesting failures here are the ones that would drive a 3 kg part into a fixture: a target
that silently falls back to something, an assembly name that does not exist, and a catalogue
rewrite that eats a taught pose.
"""

import os
import tempfile

import numpy as np
import pytest

from urlab import config as C
from urlab import tool_frames
from urlab.apps.coupler_assembly_calibration import fuse_approaches
from urlab.apps.coupler_pick_assemble import AssembleCycle
from urlab.apps.coupler_pick_place import CouplerCycle
from urlab.apps.object_calibration import merge_catalogue, yaml_document
from urlab.transforms import xyzrpy_to_matrix


def _pose(xyz_mm, rpy_deg):
    return xyzrpy_to_matrix(np.array(xyz_mm, float) / 1000.0, np.radians(rpy_deg))


def _obj(assemblies=None):
    return {'markers': {7: {'size_m': 0.04, 'T_marker_grasp': _pose([10, 0, 0], [0, 0, 0]),
                            'meta': {}}},
            'held_mass_kg': 3.3,
            'assemblies': assemblies or {},
            'meta': {'mates': 3}}


def _cycle(cfg_over=(), obj=None, resolve=True):
    """An AssembleCycle with only the fields the destination logic touches.

    `resolve` mirrors what __init__ does: look the assembly up ONCE, in the constructor, so a
    typo stops the run before the camera sweep. Tests about the lookup refusing pass False and
    call _assembly_entry() themselves."""
    job = AssembleCycle.__new__(AssembleCycle)
    job.cfg = C.load('coupler_pick_assemble', list(cfg_over))
    job.name = 'thing'
    job.obj = obj if obj is not None else _obj()
    if resolve:
        job.entry = job._assembly_entry()
        job.approach_path = job.entry.get('approach_path') or []
    return job


# ---------------------------------------------------------------------------- the target
def test_the_assembly_target_is_absolute_not_a_delta_from_the_pick():
    """The whole difference from a set-down. A fixture does not move with the part, so where the
    object was found says nothing about where it has to go."""
    T_asm = _pose([600.0, -200.0, 120.0], [180.0, 0.0, 45.0])
    job = _cycle(['assembly_name=slot'], _obj({'slot': {'T_base_assembly': T_asm, 'meta': {}}}))
    for pick in (_pose([400, 0, 50], [180, 0, 0]), _pose([100, 300, 80], [175, 5, 90])):
        assert np.allclose(job._target_pose(pick), T_asm), (
            'the assembly target moved with the pick pose')


def test_a_missing_assembly_name_refuses_rather_than_defaulting():
    """There is no sensible default. Driving to the pick pose, or some delta from it, would be a
    confident move to a place nobody chose."""
    job = _cycle(['assembly_name='], _obj({'slot': {'T_base_assembly': np.eye(4), 'meta': {}}}),
                 resolve=False)
    with pytest.raises(ValueError, match='assembly_name'):
        job._assembly_entry()


def test_an_unknown_assembly_name_says_what_is_taught():
    job = _cycle(['assembly_name=nope'],
                 _obj({'slot_a': {'T_base_assembly': np.eye(4), 'meta': {}}}), resolve=False)
    with pytest.raises(KeyError, match='slot_a'):
        job._assembly_entry()


def test_an_object_with_no_assemblies_at_all_refuses():
    job = _cycle(['assembly_name=slot'], _obj(), resolve=False)
    with pytest.raises(KeyError, match='none'):
        job._assembly_entry()


def test_the_lookup_happens_in_the_constructor_not_at_locate_time():
    """A typo in assembly_name must cost nothing. locate() runs a camera sweep with per-marker
    servo refinement -- the better part of a minute -- and resolving the destination after it
    would charge that for a misspelling."""
    import inspect
    src = inspect.getsource(AssembleCycle.__init__)
    assert '_assembly_entry()' in src, 'the destination is resolved lazily again'


# ---------------------------------------------------------------------------- the shared cycle
def test_the_assemble_app_reuses_the_pick_place_cycle():
    """One implementation, one set of tare-timing and compliance fixes. A copy would mean fixing
    the next bug in both."""
    assert issubclass(AssembleCycle, CouplerCycle)
    for shared in ('locate', 'descend_and_mate', 'lock', 'take_payload', 'lift',
                   'settle_after_lift', 'set_down', 'unlock', 'withdraw',
                   '_push_to_preload', '_compliant', '_adm'):
        assert getattr(AssembleCycle, shared) is getattr(CouplerCycle, shared), (
            f'{shared} was overridden -- it should be inherited unchanged')
    # The subclass changes WHERE the object goes, HOW it gets there, and WHO holds it at the end.
    # Everything that touches force, tare timing or compliance is inherited.
    for changed in ('_target_pose', 'carry', 'prepare_target', 'secure_target', 'teardown'):
        assert getattr(AssembleCycle, changed) is not getattr(CouplerCycle, changed), (
            f'{changed} is no longer specialised for an assembly')


def test_the_assemble_cycle_stands_off_its_own_leg_and_preload():
    assert AssembleCycle.TARGET_STANDOFF == 'assembly_standoff'
    assert AssembleCycle.TARGET_PRELOAD == 'assembly_preload'
    cfg = C.load('coupler_pick_assemble')
    assert set(cfg.section('motion')) == set(AssembleCycle.LEGS)
    assert cfg.section('assembly_preload')['enabled'] is True
    assert cfg.get('place') is None, 'a set-down delta has no meaning for an assembly'


def test_the_assemble_config_keeps_everything_the_cycle_needs():
    """It is derived from the pick-and-place config, so the shared machinery must survive."""
    cfg = C.load('coupler_pick_assemble')
    for block in ('compliance', 'compliance_loaded', 'force_guard', 'mate_preload',
                  'marker_views', 'toolchanger', 'motion'):
        assert cfg.section(block), f'{block} was lost when the config was derived'


# ---------------------------------------------------------------------------- the calibration
def test_fusing_approaches_is_a_full_six_dof_mean():
    """Nothing is unconstrained at an assembly -- the fixture decides the whole pose -- so every
    axis of the spread is evidence."""
    a = _pose([100.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    b = _pose([102.0, 0.0, 0.0], [0.0, 0.0, 4.0])
    T, res_mm, res_deg = fuse_approaches([a, b])
    assert np.allclose(T[:3, 3] * 1000.0, [101.0, 0.0, 0.0], atol=1e-6)
    assert res_mm == pytest.approx(1.0, abs=1e-6)
    assert res_deg == pytest.approx(2.0, abs=1e-6)


def test_one_approach_has_no_error_bar():
    _T, res_mm, res_deg = fuse_approaches([np.eye(4)])
    assert res_mm == 0.0 and res_deg == 0.0


def test_fusing_nothing_is_an_error():
    with pytest.raises(ValueError, match='at least one'):
        fuse_approaches([])


# ---------------------------------------------------------------------------- round trip
def test_a_taught_assembly_round_trips_through_the_catalogue():
    T = _pose([600.0, -200.0, 120.0], [179.0, 1.0, 45.0])
    entry = _obj({'slot': {'T_base_assembly': T,
                           'meta': {'approaches': 3, 'residual_mm': 0.4,
                                    'residual_deg': 0.3, 'measured': '2026-09-25'}}})
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'objects.yaml')
        with open(path, 'w') as fh:
            fh.write(yaml_document({'thing': entry}))
        back = tool_frames.load_objects(path=path)['thing']
        assert set(back['assemblies']) == {'slot'}
        got = back['assemblies']['slot']
        assert np.allclose(got['T_base_assembly'], T, atol=1e-5)
        assert got['meta']['approaches'] == 3


def test_recalibrating_an_objects_markers_does_not_eat_its_assemblies():
    """THE DANGEROUS ONE. object_calibration rewrites the whole catalogue, so anything its
    writer does not know how to emit disappears -- and a taught assembly costs a bench session
    to replace."""
    T = _pose([600.0, -200.0, 120.0], [179.0, 1.0, 45.0])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'objects.yaml')
        with open(path, 'w') as fh:
            fh.write(yaml_document({'thing': _obj({'slot': {'T_base_assembly': T, 'meta': {}}})}))
        # a fresh marker calibration for the SAME object, carrying its assemblies through
        reloaded = tool_frames.load_objects(path=path)['thing']
        merged = merge_catalogue(path, 'thing', reloaded)
        with open(path, 'w') as fh:
            fh.write(yaml_document(merged))
        after = tool_frames.load_objects(path=path)['thing']
        assert set(after['assemblies']) == {'slot'}, 'the taught assembly was wiped'
        assert np.allclose(after['assemblies']['slot']['T_base_assembly'], T, atol=1e-5)


def test_the_shipped_catalogue_still_loads_with_the_new_section():
    for obj in tool_frames.load_objects().values():
        assert isinstance(obj['assemblies'], dict)


# ---------------------------------------------------------------------------- insertion law
def test_the_fallback_assembly_axis_is_read_in_the_end_effector_frame():
    """`frame: coupler` IS the end-effector frame: coupler_mate is a pure translation from tool0,
    so the two share their axes exactly. The insertion direction is therefore the tool's own +z,
    wherever the tool is pointing -- not a direction in the room. This is the leg used by an
    assembly that has no approach path; one that has a path derives the leg from it instead."""
    from urlab.apps.coupler_pick_place import mating_direction, parse_offset
    cfg = C.load('coupler_pick_assemble')
    leg = parse_offset(cfg.section('motion')['assembly_standoff'], 'assembly_standoff')
    assert leg['frame'] == 'coupler', 'the assembly axis left the end-effector frame'
    assert np.allclose(mating_direction(leg)['axis'], [0.0, 0.0, 1.0])
    R = tool_frames.coupler_mate(cfg)[:3, :3]
    assert np.allclose(R, np.eye(3)), (
        'coupler_mate is rotated wrt tool0, so "the end effector frame" is now ambiguous')


def test_the_final_retract_backs_out_along_the_pick_axis():
    """Straight back out the way the coupler came down onto the object -- away from the mating
    feature and the marker. Read in the COUPLER frame and not in base, so it stays the pick axis
    if the assembly is ever re-taught at a different attitude."""
    from urlab.apps.coupler_pick_place import parse_offset
    cfg = C.load('coupler_pick_assemble')
    leg = parse_offset(cfg.section('motion')['final_retract'], 'final_retract')
    assert leg['frame'] == 'coupler'
    assert np.allclose(leg['axis'], [0.0, 0.0, -1.0])
    # At the shipped attitude that is very nearly straight up in base -- worth pinning as a
    # SANITY bound, not as a value: a retreat that had become sideways or downward would drag the
    # part out of the fixture the cleat is now holding it in.
    obj = tool_frames.load_objects(cfg).get(cfg.get('object_name')) or {}
    entry = (obj.get('assemblies') or {}).get(cfg.get('assembly_name'))
    if entry is not None:
        up = np.asarray(entry['T_base_assembly'], float)[:3, :3] @ leg['axis']
        assert up[2] > 0.9, ('the final retract is not lifting off the part -- it points %s in '
                             'base' % np.round(up, 3).tolist())


def _shipped_travel_in_tool():
    """The direction the shipped config's part actually travels, in the COUPLER's own axes.

    Derived from whatever object and assembly the config names rather than from a hard-coded
    pair, so retuning either does not turn these tests into a record of what they used to be.
    Returns None when that assembly has no approach path -- then the fallback leg is already in
    coupler axes and there is nothing to convert."""
    from urlab.apps.coupler_pick_assemble import standoff_from_path
    from urlab.apps.coupler_pick_place import (insertion_axis_in_tool, mating_direction,
                                               parse_offset)
    cfg = C.load('coupler_pick_assemble')
    obj = tool_frames.load_objects(cfg).get(cfg.get('object_name'))
    entry = (obj or {}).get('assemblies', {}).get(cfg.get('assembly_name')) or {}
    leg = standoff_from_path(entry.get('approach_path') or [])
    if leg is None:
        leg = parse_offset(cfg.section('motion')['assembly_standoff'], 'assembly_standoff')
        return insertion_axis_in_tool(mating_direction(leg))
    R = np.asarray(entry['T_base_assembly'], float)[:3, :3]
    return insertion_axis_in_tool(mating_direction(leg), R_base_tool=R)


def test_the_insertion_gets_its_own_shape_of_compliance():
    """THE GIVE GOES WHERE THE PART DOES NOT TRAVEL.

    The rule used to be stated as "z is the stiff axis", which only held while every insertion
    went straight down the coupler. The shipped cleat is entered SIDEWAYS, so the invariant has
    to be written against the travel direction itself: the softest tool axis must be the one the
    part moves along least, or the law is compliant where the push is wanted."""
    from urlab.apps.coupler_pick_place import compliance_blocks
    _free, loaded, insert = compliance_blocks(C.load('coupler_pick_assemble'))
    assert insert is not loaded, 'the insertion reuses the carry law'
    S = np.asarray(insert['stiffness'][:3], dtype=float)
    a = np.abs(_shipped_travel_in_tool())
    assert int(np.argmin(S)) == int(np.argmin(a)), (
        'the softest axis is not the one the part travels least along -- the law gives where '
        'the push is wanted and resists where the give is')
    along = float((a ** 2) @ S)
    assert along > 2.0 * float(S[int(np.argmin(a))]), (
        'the push direction is no stiffer than the axis meant to be soft')
    assert insert['mass'][0] == loaded['mass'][0], (
        'the virtual mass must stay at the loaded value -- the object is still on the coupler '
        'and the oscillation it causes does not care which leg is running')


def test_the_insertion_law_falls_back_when_no_block_is_given():
    from urlab.apps.coupler_pick_place import compliance_blocks
    cfg = C.load('coupler_pick_assemble')
    cfg['compliance_insert'] = None
    _free, loaded, insert = compliance_blocks(cfg)
    assert insert is loaded


def test_a_misaligned_stiffness_profile_is_detected():
    """The law runs in the tool0 frame, so an insertion down a different axis makes the profile
    exactly backwards -- compliant where the force is wanted, rigid where the give is. Nothing
    about the numbers says so."""
    from urlab.apps.coupler_pick_place import insertion_axis_is_stiff
    soft_lateral = [400.0, 400.0, 2000.0]
    assert insertion_axis_is_stiff({'axis': np.array([0.0, 0.0, 1.0])}, soft_lateral)
    assert not insertion_axis_is_stiff({'axis': np.array([1.0, 0.0, 0.0])}, soft_lateral)
    # an isotropic law has no profile to point the wrong way
    assert insertion_axis_is_stiff({'axis': np.array([1.0, 0.0, 0.0])}, [1000.0] * 3)


def test_a_base_frame_travel_axis_is_never_compared_to_tool_frame_gains():
    """THE CHECK MUST NOT BE THE THING THAT LIES. The law runs in tool0, so a leg written
    `frame: base` is in different axes entirely -- comparing it as though it were not would
    produce a confident verdict about nothing. Without the destination's orientation the honest
    answer is silence."""
    from urlab.apps.coupler_pick_place import insertion_axis_in_tool, insertion_axis_is_stiff
    spec = {'axis': np.array([1.0, 0.0, 0.0]), 'frame': 'base'}
    assert insertion_axis_in_tool(spec) is None
    assert insertion_axis_is_stiff(spec, [400.0, 400.0, 2000.0]), (
        'it produced a verdict on a base axis it could not convert')
    # Given the orientation it converts: a tool pitched 90 deg about y points its own +z along
    # base +x, so a base +x travel is travel along the tool's +z.
    R = xyzrpy_to_matrix(np.zeros(3), np.radians([0.0, 90.0, 0.0]))[:3, :3]
    assert np.allclose(R[:3, 2], [1.0, 0.0, 0.0], atol=1e-9)
    got = insertion_axis_in_tool(spec, R_base_tool=R)
    assert np.allclose(got, [0.0, 0.0, 1.0], atol=1e-9)
    assert insertion_axis_is_stiff(spec, [400.0, 400.0, 2000.0], R_base_tool=R)


def test_a_diagonal_insertion_is_reported_as_unshapeable_not_as_a_reorder():
    """Along a 45-degree axis in the tool's x-y plane the effective gain is (Sx + Sy) / 2 -- and
    the perpendicular direction IN THAT PLANE gets the identical number. No choice of three
    values separates them, so telling someone to reorder the gains sends them after something
    unreachable. The two cases have to be distinguishable."""
    from urlab.apps.coupler_pick_place import axis_is_diagonal
    assert axis_is_diagonal([0.71, 0.71, 0.0])
    assert not axis_is_diagonal([1.0, 0.0, 0.0])
    assert not axis_is_diagonal([0.0, 0.0, 1.0])
    # and the shipped cleat is the diagonal case, which is why its check cannot be cleared
    assert axis_is_diagonal(_shipped_travel_in_tool())


def test_the_insertion_leg_actually_uses_that_law():
    """And the preload push inside it pushes against the SAME spring -- a push computed against
    the carry stiffness would reach the force at the wrong reference travel."""
    import inspect
    from urlab.apps import coupler_pick_place as cpp
    assert 'law=self.adm_insert' in inspect.getsource(cpp.CouplerCycle.set_down)
    push = inspect.getsource(cpp.CouplerCycle._push_to_preload)
    assert 'self._active_law' in push, 'the preload push ignores the leg law'
    for other in ('descend_and_mate', 'lift', 'carry', 'withdraw'):
        assert 'law=' not in inspect.getsource(getattr(cpp.CouplerCycle, other)), (
            f'{other} is not an insertion and must use the payload-state law')


# ---------------------------------------------------------------------------- never let go
class _HoldBoard:
    port, booted = '/dev/fake', True

    def __init__(self, held=True):
        self.held = held
        self.calls = []
        self.bypassed = False
        self.motor_on = False

    def hold(self):
        self.calls.append('hold')
        return self.held

    def release(self):
        self.calls.append('release')
        self.held = False
        return True

    def status(self):
        self.calls.append('status')
        return self.held

    def motor(self):
        self.calls.append('motor')
        self.motor_on = not self.motor_on
        return self.motor_on

    def close(self):
        self.calls.append('close')


def _cal(board=None, cfg_over=()):
    from urlab.apps import coupler_assembly_calibration as ca
    from urlab.robot.coupler import Coupler
    cfg = C.load('coupler_assembly_calibration',
                 ['toolchanger.enabled=false'] + list(cfg_over))
    coupler = Coupler(cfg)
    coupler.device = board
    job = ca._AssemblyCalibration.__new__(ca._AssemblyCalibration)
    job.cfg, job.coupler, job.name = cfg, coupler, 'thing'
    job.robot = type('R', (), {'arm': type('A', (), {
        'dry_run': False, 'set_payload': lambda s, p: None})()})()
    return job


def test_this_app_latches_the_board_so_closing_the_port_cannot_open_the_coupler():
    """THE DROP HAZARD. Closing the port drops DTR, which resets the board -- and setup() ends
    with changeServo(checkTool()), re-deciding the clamp from ONE reading taken microseconds
    after power-on. An untrustworthy probe opens the coupler on the way out."""
    cfg = C.load('coupler_assembly_calibration')
    assert cfg.get_path('toolchanger.latch') is True, (
        'without latch, every exit from a run holding a part can drop it')


def test_the_run_does_not_release_by_default():
    """The part stays on the coupler when the run ends -- right when the next thing is another
    calibration or a pick, and never a surprise."""
    board = _HoldBoard()
    job = _cal(board)
    assert job.hand_back() is True
    assert 'release' not in board.calls, 'it let go without being asked to'


def test_releasing_is_gated_on_somebody_holding_the_part(monkeypatch):
    from urlab.apps import coupler_assembly_calibration as ca
    board = _HoldBoard()
    job = _cal(board, ['release_after=true'])
    monkeypatch.setattr(ca, 'ask', lambda _p: False)         # operator declines
    assert job.hand_back() is True
    assert 'release' not in board.calls, 'it released despite the operator declining'

    monkeypatch.setattr(ca, 'ask', lambda _p: True)          # operator has it
    assert job.hand_back() is True
    assert 'release' in board.calls


def test_a_pose_is_never_recorded_off_an_empty_coupler():
    """Not a near-miss -- a plausible number for the wrong thing, which would be averaged in and
    quietly drag the taught assembly."""
    job = _cal(_HoldBoard(held=False))
    assert job._still_held(0) is False
    assert _cal(_HoldBoard(held=True))._still_held(0) is True


def test_nothing_in_the_teardown_releases():
    """Every exit -- a failed step, a declined prompt, Ctrl-C -- can happen with the part on the
    coupler. A teardown that let go would drop it from wherever the arm was left."""
    import inspect
    from urlab.apps import coupler_assembly_calibration as ca
    src = inspect.getsource(ca.build_and_run)
    tail = src[src.index('finally:'):]
    assert '.release()' not in tail, 'the teardown releases the part'
    assert 'coupler.close()' in tail
    # ... and the only release in the whole module is the gated hand-back.
    mod = inspect.getsource(ca)
    assert mod.count('.release()') == 1, 'more than one place lets go'
    assert '.release()' in inspect.getsource(ca._AssemblyCalibration.hand_back)


# ---------------------------------------------------------------------------- the approach path
def test_a_waypoint_is_a_base_frame_offset_carried_at_the_assembled_attitude():
    """The pairing that makes the numbers readable: offsets in base axes, so they can be read off
    the same monitor reading that taught the pose, while the object stays at the attitude it is
    going to be assembled in -- the one already proven to fit through the gap."""
    from urlab.apps.coupler_pick_assemble import waypoint_pose
    T = _pose([600.0, -200.0, 120.0], [180.0, 0.0, 45.0])
    got = waypoint_pose(T, {'name': 'up', 'xyz': np.array([-0.25, 0.0, 0.25])})
    assert np.allclose(got[:3, 3] * 1000.0, [350.0, -200.0, 370.0], atol=1e-6)
    assert np.allclose(got[:3, :3], T[:3, :3]), 'the attitude changed on the way in'


def test_a_waypoint_rotation_turns_the_tool_without_moving_it():
    from urlab.apps.coupler_pick_assemble import waypoint_pose
    T = _pose([600.0, -200.0, 120.0], [180.0, 0.0, 45.0])
    wp = {'name': 'turned', 'xyz': np.array([0.1, 0.0, 0.0]),
          'rpy': np.radians([0.0, 0.0, 30.0])}
    got = waypoint_pose(T, wp)
    assert np.allclose(got[:3, 3] * 1000.0, [700.0, -200.0, 120.0], atol=1e-6), (
        'the rotation dragged the position with it')
    assert not np.allclose(got[:3, :3], T[:3, :3])


def test_the_last_waypoint_IS_the_insertion_standoff():
    """ONE SETTING, NOT TWO. The standoff also fixes the insertion AXIS, so a path ending
    somewhere the standoff leg does not agree with would have the arm thread its route, hop
    elsewhere, and press home along a line it never approached on."""
    from urlab.apps.coupler_pick_assemble import standoff_from_path
    from urlab.apps.coupler_pick_place import mating_direction, offset_pose
    path = [{'name': 'a', 'xyz': np.array([-0.25, 0.0, 0.25])},
            {'name': 'b', 'xyz': np.array([0.05, 0.0, 0.0])}]
    leg = standoff_from_path(path)
    assert leg['frame'] == 'base'
    assert leg['distance_m'] == pytest.approx(0.05)
    assert np.allclose(leg['axis'], [1.0, 0.0, 0.0])
    T = _pose([600.0, -200.0, 120.0], [180.0, 0.0, 45.0])
    from urlab.apps.coupler_pick_assemble import waypoint_pose
    assert np.allclose(offset_pose(T, leg), waypoint_pose(T, path[-1])), (
        'the compliant insertion does not begin where the approach path ends')
    # ... and the push home runs back down that same line.
    assert np.allclose(mating_direction(leg)['axis'], [-1.0, 0.0, 0.0])


def test_no_path_means_no_derived_standoff_and_the_configured_leg_stands():
    from urlab.apps.coupler_pick_assemble import standoff_from_path
    assert standoff_from_path([]) is None


def test_a_path_that_ends_at_the_assembly_is_refused():
    """The last waypoint is the standoff, and the vector from it to the assembly is the insertion
    axis -- a zero one leaves the insertion no direction and no distance."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'objects.yaml')
        with open(path, 'w') as fh:
            fh.write('objects:\n  thing:\n    markers:\n      7:\n        size_mm: 40.0\n'
                     '        xyz_mm: [0, 0, 0]\n        rpy_deg: [0, 0, 0]\n'
                     '    assemblies:\n      slot:\n        xyz_mm: [600, 0, 100]\n'
                     '        rpy_deg: [180, 0, 0]\n        approach_path:\n'
                     '          - xyz_mm: [0, 0, 100]\n          - xyz_mm: [0, 0, 0]\n')
        with pytest.raises(ValueError, match='no direction'):
            tool_frames.load_objects(path=path)


def test_a_waypoint_typo_is_refused_rather_than_zero_filled():
    """The same rule as every other pose in the catalogue: a silently-dropped key would move a
    waypoint to the assembled pose itself, which is inside the fixture."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'objects.yaml')
        with open(path, 'w') as fh:
            fh.write('objects:\n  thing:\n    markers:\n      7:\n        size_mm: 40.0\n'
                     '        xyz_mm: [0, 0, 0]\n        rpy_deg: [0, 0, 0]\n'
                     '    assemblies:\n      slot:\n        xyz_mm: [600, 0, 100]\n'
                     '        rpy_deg: [180, 0, 0]\n        approach_path:\n'
                     '          - xyz_m: [0, 0, 0.1]\n')
        with pytest.raises(ValueError, match='unknown key'):
            tool_frames.load_objects(path=path)


def test_an_approach_path_survives_a_marker_recalibration():
    """THE SAME DANGER AS THE TAUGHT POSE. object_calibration rewrites the whole catalogue, and
    an approach path is hand-written -- exactly the thing nobody thinks to back up before
    re-teaching a marker."""
    T = _pose([600.0, -200.0, 120.0], [179.0, 1.0, 45.0])
    path_in = [{'name': 'above deck', 'xyz': np.array([-0.25, 0.0, 0.25]),
                'rpy': np.zeros(3)},
               {'name': 'standoff', 'xyz': np.array([0.05, 0.0, 0.0]),
                'rpy': np.radians([0.0, 0.0, 5.0])}]
    entry = _obj({'slot': {'T_base_assembly': T, 'approach_path': path_in, 'meta': {}}})
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'objects.yaml')
        with open(p, 'w') as fh:
            fh.write(yaml_document({'thing': entry}))
        merged = merge_catalogue(p, 'thing', tool_frames.load_objects(path=p)['thing'])
        with open(p, 'w') as fh:
            fh.write(yaml_document(merged))
        after = tool_frames.load_objects(path=p)['thing']['assemblies']['slot']['approach_path']
        assert [w['name'] for w in after] == ['above deck', 'standoff']
        for before, got in zip(path_in, after):
            assert np.allclose(before['xyz'], got['xyz'], atol=1e-6)
            assert np.allclose(before['rpy'], got['rpy'], atol=1e-6)


def test_the_shipped_cleat_path_ends_where_the_insertion_starts():
    """End to end on the real catalogue: whatever object and assembly the config names, the
    derived standoff and the last waypoint must be the same pose."""
    from urlab.apps.coupler_pick_assemble import standoff_from_path, waypoint_pose
    from urlab.apps.coupler_pick_place import offset_pose
    cfg = C.load('coupler_pick_assemble')
    obj = tool_frames.load_objects(cfg).get(cfg.get('object_name'))
    entry = (obj or {}).get('assemblies', {}).get(cfg.get('assembly_name'))
    if entry is None or not entry.get('approach_path'):
        pytest.skip('the shipped config names an assembly with no approach path')
    T = entry['T_base_assembly']
    leg = standoff_from_path(entry['approach_path'])
    assert np.allclose(offset_pose(T, leg), waypoint_pose(T, entry['approach_path'][-1]))


# ---------------------------------------------------------------------------- the handover
def _assemble_with_cleat(board, cfg_over=()):
    """An AssembleCycle carrying a fake cleat board, and nothing else wired up."""
    from urlab.robot.coupler import Coupler
    cfg = C.load('coupler_pick_assemble',
                 ['cleat_toolchanger.enabled=false'] + list(cfg_over))
    job = AssembleCycle.__new__(AssembleCycle)
    job.cfg, job.name = cfg, 'thing'
    job.cleat = Coupler(cfg, section='cleat_toolchanger', label='cleat')
    job.cleat.device = board
    job.cleat.latched = True
    return job


def test_the_cleat_clamps_BEFORE_the_coupler_lets_go():
    """THE ORDER IS THE WHOLE POINT. Between the cleat taking hold and the coupler releasing the
    object is held twice, which is harmless. The other order holds it none."""
    import inspect
    from urlab.apps import coupler_pick_place as cpp
    src = inspect.getsource(cpp.build_and_run)
    assert src.index('job.secure_target') < src.index('job.unlock'), (
        'the coupler releases before the destination has taken hold')
    assert src.index('job.prepare_target') < src.index('job.approach'), (
        'the destination clamp is opened after the arm has already set off')


def test_a_refused_cleat_clamp_stops_the_run_with_the_object_still_held():
    """A refused clamp means the part did not reach its seat. The next steps release and pull
    away, so this has to fail rather than warn."""
    job = _assemble_with_cleat(_HoldBoard(held=False))
    assert job.secure_target() is False
    assert 'hold' in job.cleat.device.calls
    assert 'release' not in job.cleat.device.calls, 'it let go of a part the cleat refused'


def test_a_clamp_that_the_sensor_then_disagrees_with_also_stops():
    """hold() confirms at the instant the servo finished; a part merely resting where the probe
    can see it settles out in the moment after, and the next step is letting go of it."""
    class _Flaky(_HoldBoard):
        def status(self):
            self.calls.append('status')
            return False

    job = _assemble_with_cleat(_Flaky(held=True))
    assert job.secure_target() is False


def test_a_good_clamp_is_re_checked_and_then_accepted():
    job = _assemble_with_cleat(_HoldBoard(held=True))
    assert job.secure_target() is True
    assert job.cleat.device.calls == ['hold', 'status']


def test_opening_the_cleat_powers_it_and_forces_the_jaws_open():
    """A cleat that booted with anything in front of its probe comes up clamped, and a part
    cannot be slid into a clamp that is already shut -- nor would a later hold() move anything."""
    job = _assemble_with_cleat(_HoldBoard(held=True))
    assert job.prepare_target() is True
    assert 'motor' in job.cleat.device.calls, 'the cleat relay was never powered'
    assert 'release' in job.cleat.device.calls, 'the cleat jaws were never opened'


def test_nothing_in_the_assemble_teardown_releases_the_cleat():
    """The point of the run is that the part stays mounted after it."""
    job = _assemble_with_cleat(_HoldBoard(held=True))
    assert job.teardown() is True
    assert 'release' not in job.cleat.device.calls
    assert 'close' in job.cleat.device.calls


def test_the_shipped_cleat_is_latched_or_the_teardown_drops_the_part():
    """Closing the port drops DTR, which resets the board -- and setup() re-decides the clamp
    from ONE reading taken microseconds after power-on. On the CLEAT that reset happens at the
    end of a SUCCESSFUL run, with the assembled object in it and the arm already withdrawn."""
    cfg = C.load('coupler_pick_assemble')
    if not cfg.section('cleat_toolchanger'):
        pytest.skip('no cleat configured')
    assert cfg.get_path('cleat_toolchanger.latch') is True


def test_the_two_couplers_are_not_the_same_board():
    """They run identical firmware and answer identically over the wire, so the only thing that
    tells them apart is which port they are reached on. Pointing both at one board would have the
    'cleat' confirm a clamp it never made, and the release after it would drop the part."""
    cfg = C.load('coupler_pick_assemble')
    if not cfg.section('cleat_toolchanger'):
        pytest.skip('no cleat configured')
    assert (cfg.get_path('cleat_toolchanger.port')
            != cfg.get_path('toolchanger.port')), 'both couplers point at the same board'


def test_a_plain_set_down_gains_no_cleat_steps():
    """A bench does not hold on to anything, and a no-op step that still asks for a confirmation
    is the noise that trains people to hit Enter unread."""
    assert CouplerCycle.secures_target is False
    job = _assemble_with_cleat(_HoldBoard())
    job.cleat = None
    assert job.secures_target is False


def test_the_carry_walks_every_waypoint_in_order_and_stops_at_the_insertion_standoff():
    """END TO END on the legs: the approach visits each taught waypoint and hands off to the
    compliant insertion exactly where it stopped. A gap between the two would be an unannounced
    free-space hop -- through whatever the waypoints were threaded around."""
    from urlab.apps.coupler_pick_assemble import waypoint_pose
    from urlab.apps.coupler_pick_place import offset_pose
    T_asm = _pose([600.0, -200.0, 120.0], [180.0, 0.0, 45.0])
    path = [{'name': 'above deck', 'xyz': np.array([-0.25, 0.0, 0.25]), 'rpy': np.zeros(3)},
            {'name': 'past bracket', 'xyz': np.array([0.05, 0.0, 0.25]), 'rpy': np.zeros(3)},
            {'name': 'standoff', 'xyz': np.array([0.05, 0.0, 0.0]), 'rpy': np.zeros(3)}]
    job = _cycle(['assembly_name=slot'],
                 _obj({'slot': {'T_base_assembly': T_asm, 'approach_path': path, 'meta': {}}}))
    from urlab.apps.coupler_pick_assemble import standoff_from_path
    job.T_place = T_asm
    job.legs = {'assembly_standoff': standoff_from_path(path)}
    went = []
    job._move_to = lambda T, label: went.append((label, T)) or True
    assert job.carry() is True
    assert [w['name'] in label for (label, _T), w in zip(went, path)] == [True] * 3
    for (_label, T), w in zip(went, path):
        assert np.allclose(T, waypoint_pose(T_asm, w))
    # the compliant insertion begins exactly where the last move ended
    assert np.allclose(went[-1][1], offset_pose(T_asm, job.legs['assembly_standoff']))


def test_an_assembly_with_no_path_still_takes_the_one_straight_hop():
    T_asm = _pose([600.0, -200.0, 120.0], [180.0, 0.0, 45.0])
    job = _cycle(['assembly_name=slot'],
                 _obj({'slot': {'T_base_assembly': T_asm, 'meta': {}}}))
    assert job.approach_path == []
    went = []
    job.T_place = T_asm
    job.legs = {'assembly_standoff': {'distance_m': 0.1, 'axis': np.array([0.0, 0.0, -1.0]),
                                      'frame': 'coupler'}}
    job._move_to = lambda T, label: went.append(label) or True
    assert job.carry() is True
    assert len(went) == 1, 'a pathless assembly should be one hop to the standoff'
