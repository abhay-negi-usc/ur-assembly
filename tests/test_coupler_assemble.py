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


def _cycle(cfg_over=(), obj=None):
    job = AssembleCycle.__new__(AssembleCycle)
    job.cfg = C.load('coupler_pick_assemble', list(cfg_over))
    job.name = 'thing'
    job.obj = obj if obj is not None else _obj()
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
    job = _cycle([], _obj({'slot': {'T_base_assembly': np.eye(4), 'meta': {}}}))
    with pytest.raises(ValueError, match='assembly_name'):
        job._target_pose(np.eye(4))


def test_an_unknown_assembly_name_says_what_is_taught():
    job = _cycle(['assembly_name=nope'],
                 _obj({'slot_a': {'T_base_assembly': np.eye(4), 'meta': {}}}))
    with pytest.raises(KeyError, match='slot_a'):
        job._target_pose(np.eye(4))


def test_an_object_with_no_assemblies_at_all_refuses():
    job = _cycle(['assembly_name=slot'], _obj())
    with pytest.raises(KeyError, match='none'):
        job._target_pose(np.eye(4))


# ---------------------------------------------------------------------------- the shared cycle
def test_the_assemble_app_reuses_the_pick_place_cycle():
    """One implementation, one set of tare-timing and compliance fixes. A copy would mean fixing
    the next bug in both."""
    assert issubclass(AssembleCycle, CouplerCycle)
    for shared in ('locate', 'descend_and_mate', 'lock', 'take_payload', 'lift',
                   'settle_after_lift', 'carry', 'set_down', 'unlock', 'withdraw',
                   '_push_to_preload', '_compliant', '_adm'):
        assert getattr(AssembleCycle, shared) is getattr(CouplerCycle, shared), (
            f'{shared} was overridden -- it should be inherited unchanged')
    assert AssembleCycle._target_pose is not CouplerCycle._target_pose, (
        'the destination is the ONLY thing the subclass changes')


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
def test_the_assembly_axis_is_read_in_the_end_effector_frame():
    """`frame: coupler` IS the end-effector frame: coupler_mate is a pure translation from tool0,
    so the two share their axes exactly. The insertion direction is therefore the tool's own +z,
    wherever the tool is pointing -- not a direction in the room."""
    from urlab.apps.coupler_pick_place import mating_direction, parse_offset
    cfg = C.load('coupler_pick_assemble')
    leg = parse_offset(cfg.section('motion')['assembly_standoff'], 'assembly_standoff')
    assert leg['frame'] == 'coupler', 'the assembly axis left the end-effector frame'
    assert np.allclose(mating_direction(leg)['axis'], [0.0, 0.0, 1.0])
    R = tool_frames.coupler_mate(cfg)[:3, :3]
    assert np.allclose(R, np.eye(3)), (
        'coupler_mate is rotated wrt tool0, so "the end effector frame" is now ambiguous')


def test_the_insertion_gets_its_own_shape_of_compliance():
    """Soft ACROSS the travel axis so a chamfer can guide the part, stiffer ALONG it so the
    preload push reaches its force in a sane amount of reference travel."""
    from urlab.apps.coupler_pick_place import compliance_blocks
    _free, loaded, insert = compliance_blocks(C.load('coupler_pick_assemble'))
    assert insert is not loaded, 'the insertion reuses the carry law'
    S = insert['stiffness']
    assert S[2] > 2.0 * max(S[0], S[1]), 'the travel axis is not the stiff one'
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


def test_the_shipped_assembly_profile_matches_its_own_travel_axis():
    from urlab.apps.coupler_pick_place import (compliance_blocks, insertion_axis_is_stiff,
                                               mating_direction, parse_offset)
    cfg = C.load('coupler_pick_assemble')
    leg = parse_offset(cfg.section('motion')['assembly_standoff'], 'assembly_standoff')
    insert = compliance_blocks(cfg)[2]
    assert insertion_axis_is_stiff(mating_direction(leg), insert['stiffness'])


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
