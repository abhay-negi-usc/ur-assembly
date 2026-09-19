"""BncSpec: the typed view of configs/bnc_assembly.yaml (per-behaviour schema, 2026-08-28).

Defaults live on the dataclasses -- the single source of truth; the yaml declares only what it
tunes; unknown keys are rejected with their dotted path; the units-normaliser's derived SI
siblings and explicit nulls never reach the parser."""

import yaml

from urlab import config as C
from urlab.config import parse_block
from urlab.domain import BncSpec, ContactSpec, EngageSpec, strip_derived_units


def test_the_shipped_config_loads_field_for_field():
    """Every value the yaml declares lands on its spec field unchanged (and nothing else --
    absent keys read as the dataclass defaults)."""
    cfg = C.load('bnc_assembly')
    sp = BncSpec.from_config(cfg)

    en = cfg.section('engage')
    assert sp.engage.travel_mm == float(en['travel_mm'])
    assert sp.engage.max_axial_force_n == en['max_axial_force_n']
    assert sp.engage.compliance == strip_derived_units(en['compliance'])
    assert sp.engage.contact.enabled == en['contact']['enabled']
    assert sp.engage.confirm.max_wiggle_mm == en['confirm']['max_wiggle_mm']
    assert sp.engage.fail_retract.distance_mm == en['fail_retract']['distance_mm']

    cc = cfg.section('connector_clocking')
    assert sp.connector_clocking.enabled == cc['enabled']
    assert sp.connector_clocking.force_guard == strip_derived_units(cc['force_guard'])
    cl = cfg.section('collar_clocking')
    assert sp.collar_clocking.collar_offset_mm == float(cl['collar_offset_mm'])
    assert sp.collar_clocking.seat_push.force_n == float(cl['seat_push']['force_n'])

    run = cfg.section('run')
    assert sp.run.target_frame == run['target_frame']
    assert sp.run.insertion_mode == run['insertion_mode']
    assert sp.run.gate_between_behaviors == run['gate_between_behaviors']
    assert sp.run.success_pos_mm == float(run['success_pos_mm'])
    tr = cfg.section('trajectory')
    assert sp.trajectory.csv == tr['csv']
    assert sp.trajectory.standoff['distance_mm'] == tr['standoff']['distance_mm']
    cr = cfg.section('clocking_retract')
    assert sp.clocking_retract.gripper_distance_mm == float(cr['gripper_distance_mm'])
    vt = cfg.section('visual_target')
    assert sp.visual_target.return_home_after == vt['return_home_after']
    assert sp.disassembly.place_scatter.enabled == \
        cfg.get_path('disassembly.place_scatter.enabled')


def test_the_raw_yaml_matches_what_the_spec_parsed():
    """The FILE (no loader magic) declares the same values the spec carries -- so a hand edit
    to the yaml is guaranteed to be what the run uses."""
    import os
    raw = yaml.safe_load(open(os.path.join(os.path.dirname(C.__file__), '..', 'configs',
                                           'bnc_assembly.yaml'), encoding='utf-8'))
    sp = BncSpec.from_config(C.load('bnc_assembly'))
    assert sp.engage.preload_mm == float(raw['engage']['preload_mm'])
    assert sp.celebrate.enabled == bool(raw['celebrate'].get('enabled', False))
    assert sp.tug_verify.pull_force_n == float(raw['tug_verify']['pull_force_n'])
    assert sp.reorient_recovery.place_offsets['z_mm'] == \
        raw['reorient_recovery']['place_offsets']['z_mm']


def test_unknown_keys_are_rejected_with_the_dotted_path():
    try:
        parse_block(BncSpec, {'engage': {'travel_mmm': 5.0}}, 'bnc')
    except ValueError as exc:
        assert 'travel_mmm' in str(exc) and 'bnc.engage' in str(exc)
    else:
        raise AssertionError('a typo key must be rejected, not silently dropped')


def test_strip_derived_units_removes_siblings_and_nulls_only():
    """The loader plants X_m/X_rad/xyz/rpy siblings for every X_mm/X_deg/xyz_mm/rpy_deg key;
    those and explicit nulls must never reach the parser -- everything else must."""
    block = {'travel_mm': 5.0, 'travel_m': 0.005,          # derived sibling -> dropped
             'angle_deg': 90.0, 'angle_rad': 1.5707,       # derived sibling -> dropped
             'xyz_mm': [1, 2, 3], 'xyz': [0.001, 0.002, 0.003],
             'rpy_deg': [0, 0, 90], 'rpy': [0, 0, 1.57],
             'speed_mm_s': 3.0, 'speed_m_s': 0.003,
             'settle_s': None,                             # explicit null -> dropped (inherit)
             'plain_m': 0.5,                               # no _mm sibling -> KEPT
             'nested': {'depth_mm': 2.0, 'depth_m': 0.002, 'keep': True}}
    assert strip_derived_units(block) == {
        'travel_mm': 5.0, 'angle_deg': 90.0, 'xyz_mm': [1, 2, 3], 'rpy_deg': [0, 0, 90],
        'speed_mm_s': 3.0, 'plain_m': 0.5, 'nested': {'depth_mm': 2.0, 'keep': True}}


def test_defaults_live_on_the_dataclass_alone():
    """An empty config gives pure defaults -- the yaml declares only what it tunes."""

    class _Empty:
        @staticmethod
        def get(_k, default=None):
            return default

    sp = BncSpec.from_config(_Empty())
    assert sp.engage == EngageSpec()
    assert sp.engage.contact == ContactSpec()
    assert sp.run.max_attempts == 5 and sp.run.log_decimation == 5
    assert sp.trajectory.csv == 'assembly_trajectory.csv'
    assert sp.celebrate.enabled is False and sp.celebrate.when == 'last'
