"""The configs are written in mm/deg; the code runs in m/rad. `config.load()` bridges the two ONCE.

These tests pin the MECHANISM, not any tuned number -- a value someone retunes must never break
them. What they defend is the failure this design exists to prevent: a length silently off by
1000x because one read site was missed, or a number that quietly stopped being a number.
"""

import glob
import math
import os
import re

import pytest

from urlab import config as C
from urlab.config import _normalise_units

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')


def test_mm_and_deg_keys_gain_their_si_siblings():
    """The whole point: yaml says mm/deg, the reader asks for m/rad and gets the right number."""
    cfg = _normalise_units({'standoff_mm': 50.0, 'sweep_deg': 90.0,
                            'speed': {'max_mm_s': 250.0, 'turn_deg_s': 180.0},
                            'pose': {'xyz_mm': [10.0, 0.0, -5.0], 'rpy_deg': [180.0, 0.0, -90.0]}})
    assert cfg['standoff_m'] == pytest.approx(0.05)
    assert cfg['sweep_rad'] == pytest.approx(math.pi / 2)
    assert cfg['speed']['max_m_s'] == pytest.approx(0.25)          # nested, and a compound suffix
    assert cfg['speed']['turn_rad_s'] == pytest.approx(math.pi)
    assert cfg['pose']['xyz'] == pytest.approx([0.010, 0.0, -0.005])
    assert cfg['pose']['rpy'] == pytest.approx([math.pi, 0.0, -math.pi / 2])
    # The mm/deg keys SURVIVE: plenty of readers take mm directly, and clobbering them would
    # break every one of those instead of the SI readers.
    assert cfg['standoff_mm'] == 50.0 and cfg['pose']['xyz_mm'] == [10.0, 0.0, -5.0]


def test_the_pose_triples_do_not_also_mint_a_suffixed_spelling():
    """`xyz_mm` means `xyz`, NOT `xyz_m`. A second unread spelling is just another thing to get
    stale, and `_pose_si` would then see a pair it did not create."""
    cfg = _normalise_units({'xyz_mm': [1.0, 2.0, 3.0], 'rpy_deg': [0.0, 0.0, 90.0]})
    assert 'xyz' in cfg and 'rpy' in cfg
    assert 'xyz_m' not in cfg and 'rpy_rad' not in cfg


def test_writing_one_quantity_in_two_units_is_refused():
    """The error this design exists for: preferring one silently turns 90 mm into 90 m. Equal
    values are refused too -- `[0,0,0]` twice is still a file that cannot say what it means."""
    for bad in ({'lift_mm': 50.0, 'lift_m': 0.05},          # consistent, still ambiguous authoring
                {'lift_mm': 50.0, 'lift_m': 999.0},         # outright contradictory
                {'xyz_mm': [0, 0, 0], 'xyz': [0, 0, 0]}):
        with pytest.raises(ValueError, match='(?i)both|one unit'):
            _normalise_units(dict(bad))


def test_a_unit_key_holding_something_that_is_not_a_number_is_loud():
    """YAML 1.1 reads `1e+02` as a STRING (it wants `1.0e+02`). Passing that through quietly is the
    dangerous case: the key keeps its mm spelling, never gains an SI sibling, and the reader falls
    back to its default -- a standoff becoming some other distance with nothing logged."""
    for bad in ({'standoff_mm': '1e+02'}, {'xyz_mm': [1.0, '2e+01', 3.0]},
                {'blk': {'lift_mm': 'oops'}}):
        with pytest.raises(ValueError, match='(?i)not a number|non-numeric'):
            _normalise_units(dict(bad))


def test_names_that_end_in_a_unit_without_being_that_quantity_are_left_alone():
    """Three real shapes from these configs. `curvature_min_per_m` is an INVERSE length -- scaling
    it as if it were a length would be exactly backwards."""
    cfg = _normalise_units({'curvature_min_per_m': 5.0,
                            'trajectory_angles_deg': False,          # a FLAG, not an angle
                            'joint_limits_deg': {'wrist_3': [-180.0, 180.0]},
                            'station': {'depth_mm': {'lower': 0.0, 'upper': 10.0}}})
    assert cfg['curvature_min_per_m'] == 5.0 and 'curvature_min_per_mm' not in cfg
    assert cfg['trajectory_angles_deg'] is False and 'trajectory_angles_rad' not in cfg
    assert cfg['joint_limits_deg'] == {'wrist_3': [-180.0, 180.0]}
    assert 'joint_limits_rad' not in cfg and 'depth_m' not in cfg['station']


def test_normalising_twice_changes_nothing():
    """Re-running must not read its own output as a hand-written duplicate."""
    once = _normalise_units({'lift_mm': 50.0, 'p': {'xyz_mm': [1.0, 2.0, 3.0]}})
    twice = _normalise_units(dict(once))
    assert twice['lift_m'] == once['lift_m'] == pytest.approx(0.05)
    assert twice['p']['xyz'] == pytest.approx([0.001, 0.002, 0.003])


def test_an_explicit_override_in_si_units_beats_the_file():
    """`--set` naming the SI spelling is a deliberate override, not the two-spellings mistake, so
    it must WIN -- and leave the mm sibling agreeing with it rather than contradicting it."""
    cfg = C.load('cartesian', ['linear_step_m=0.05'])
    assert cfg.get('linear_step_m') == pytest.approx(0.05)
    assert cfg.get('linear_step_mm') == pytest.approx(50.0)
    cfg2 = C.load('cartesian', ['linear_step_mm=30.0'])           # the mm spelling works too
    assert cfg2.get('linear_step_mm') == pytest.approx(30.0)
    assert cfg2.get('linear_step_m') == pytest.approx(0.03)


def test_every_shipped_config_is_written_in_mm_and_deg():
    """The convention itself. A new config that slips back to `_m`/`_rad` fails HERE, rather than
    on the robot -- the loader would still read it, so nothing else would notice."""
    si = re.compile(r'^\s*(?:-\s+)?([A-Za-z_][A-Za-z0-9_]*(?:_m|_rad|_m_s|_rad_s|_m_s2|_rad_s2))\s*:')
    exempt = ('_per_m', '_per_mm', '_per_deg', '_per_rad')
    offenders = []
    for path in sorted(glob.glob(os.path.join(CONFIG_DIR, '*.yaml'))):
        for n, line in enumerate(open(path, encoding='utf-8'), 1):
            m = si.match(line)
            if m and not m.group(1).endswith(exempt):
                offenders.append(f'{os.path.basename(path)}:{n} {m.group(1)}')
    assert not offenders, 'configs must use mm/deg key names:\n  ' + '\n  '.join(offenders)


def test_bare_xyz_and_rpy_are_not_used_in_configs():
    """A bare `xyz:`/`rpy:` carries no unit in its name, which is the one thing this convention
    refuses -- it reads as m/rad only if you already know the rule."""
    bare = re.compile(r'^\s*(?:-\s+)?(xyz|rpy)\s*:')
    offenders = [f'{os.path.basename(p)}:{n}'
                 for p in sorted(glob.glob(os.path.join(CONFIG_DIR, '*.yaml')))
                 for n, line in enumerate(open(p, encoding='utf-8'), 1) if bare.match(line)]
    assert not offenders, 'poses must be xyz_mm/rpy_deg:\n  ' + '\n  '.join(offenders)


def test_every_config_still_loads():
    """Cheap end-to-end: normalisation runs over every shipped file, so a bad value anywhere in
    the set raises here instead of when that demo is next run on hardware."""
    for path in sorted(glob.glob(os.path.join(CONFIG_DIR, '*.yaml'))):
        C.load(path)
