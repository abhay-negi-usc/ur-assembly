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


# --------------------------------------------------------------------- per-axis maps
# `dim_weights: {x_mm: 1.0, ..., yaw_deg: 1.0}` keys a map BY AXIS. The suffix names the axis, not
# the value -- an x weight of 1.0 is not "1 mm" -- and the code validates these key sets against
# its own DIMS vocabulary. Injecting `x_m` beside `x_mm` there is meaningless AND fatal: it reached
# the robot as `ValueError: estimation.dim_weights keys ['x_m', ...] not in ['x_mm', ...]`.

AXIS_SI_SIBLINGS = ('x_m', 'y_m', 'z_m', 'roll_rad', 'pitch_rad', 'yaw_rad')


def _dicts(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _dicts(v)
    elif isinstance(node, list):
        for v in node:
            yield from _dicts(v)


def test_the_axis_vocabulary_matches_the_code_that_validates_it():
    """config.py hard-codes the axis labels; the skills validate against their own DIMS. If those
    ever drift apart the exemption silently stops covering a real block, so pin them together."""
    from urlab.apps import insertion_tester, wiggle_sampling
    from urlab.config import _AXIS_LABELS
    from urlab.skills import manifold, wiggle
    for owner, dims in (('manifold', manifold.DIMS), ('wiggle', wiggle.DIMS),
                        ('wiggle_sampling', wiggle_sampling.DIMS),
                        ('insertion_tester', insertion_tester._DIMS)):
        assert set(dims) == set(_AXIS_LABELS), f'{owner}.DIMS drifted from config._AXIS_LABELS'


AXIS_LENGTHS = {'x_mm', 'y_mm', 'z_mm', 'x_m', 'y_m', 'z_m'}
AXIS_ANGLES = {'roll_deg', 'pitch_deg', 'yaw_deg', 'roll_rad', 'pitch_rad', 'yaw_rad'}
AXIS_ANY = AXIS_LENGTHS | AXIS_ANGLES


@pytest.mark.parametrize('path', sorted(glob.glob(os.path.join(CONFIG_DIR, '*.yaml'))),
                         ids=lambda p: os.path.basename(p))
def test_no_config_gains_a_bogus_axis_key(path):
    """After loading, a per-axis map must contain ONLY axis labels -- the exact shape that crashed
    CheckedManifoldEstimator on the robot with `dim_weights keys ['x_m', ...] not in ['x_mm', ...]`.

    Identified independently of the loader's own predicate: a map whose keys are drawn purely from
    the axis vocabulary AND span both a length and an angle. `ground_plane` (a plain depth sitting
    beside `max_cables`) does not match, so its legitimate `z_m` sibling is not flagged.
    """
    for d in _dicts(C.load(path)):
        keys = set(d)
        if not keys <= AXIS_ANY or not (keys & AXIS_LENGTHS) or not (keys & AXIS_ANGLES):
            continue
        bogus = keys & set(AXIS_SI_SIBLINGS)
        assert not bogus, (f'{os.path.basename(path)}: per-axis map gained SI twins '
                           f'{sorted(bogus)} beside {sorted(keys - bogus)}')


def test_a_plain_depth_still_gets_its_si_sibling():
    """The counter-case the exemption must NOT swallow. `ground_plane.z_mm` is a depth, not an axis
    map, and `ground_plane.z_m` is read with a default -- so losing it would be SILENT."""
    cfg = C.load('bnc_assembly')
    assert cfg.get_path('ground_plane.z_m') == pytest.approx(
        cfg.get_path('ground_plane.z_mm') / 1000.0)
    # ... and it must keep working even if the depth were the block's only key.
    lone = _normalise_units({'ground_plane': {'z_mm': -760.0}})
    assert lone['ground_plane']['z_m'] == pytest.approx(-0.76)


def test_a_real_axis_map_is_left_exactly_as_written():
    """Both halves: a 6-DOF map is untouched, and a mixed block that merely CONTAINS axis names
    (place_scatter has `enabled`/`seed` too) is not mistaken for one."""
    weights = {'x_mm': 1.0, 'y_mm': 1.0, 'z_mm': 1.0,
               'roll_deg': 1.0, 'pitch_deg': 1.0, 'yaw_deg': 1.0}
    assert _normalise_units({'dim_weights': dict(weights)})['dim_weights'] == weights
    assert _normalise_units({'w': {'z_mm': 2.0, 'roll_deg': 5.0}})['w'] == {'z_mm': 2.0,
                                                                           'roll_deg': 5.0}
    scatter = _normalise_units({'s': {'enabled': True, 'seed': 7, 'x_mm': 50.0, 'yaw_deg': 30.0}})
    assert scatter['s']['x_m'] == pytest.approx(0.05)      # not an axis map -- siblings are fine
