"""The engage runs at its OWN compliance, set once and static for the whole behaviour.

Stiffness is the knob that decides how far the connector gives under load: an admittance loop
holds a force by deflecting `force / stiffness`, so the commanded pose backs off the contact by
exactly that much. These pin the override plumbing and that relationship.
"""

import numpy as np
import pytest

from urlab import config as C
from urlab.apps.bnc_assembly import _compliance_override
from urlab.robot.admittance import AdmittanceController

SHARED = {'stiffness': [500.0] * 3 + [10.0] * 3, 'mass': [5.0] * 3 + [0.5] * 3,
          'damping_ratio': [1.0] * 6, 'reference_rate_hz': 125.0}


def test_the_engage_block_sets_its_own_stiffness():
    stiff = [4000.0, 500.0, 500.0, 10.0, 10.0, 10.0]
    comp = _compliance_override(SHARED, {'stiffness': stiff})
    assert comp['stiffness'] == stiff
    assert comp['mass'] == SHARED['mass'], 'unset keys must inherit, not reset to a default'
    assert comp['reference_rate_hz'] == 125.0, 'the rest of the shared section carries over'


def test_null_and_absent_both_inherit():
    """`stiffness: null` is how the config says "use the shared one" -- not "use zero"."""
    for block in ({'stiffness': None, 'mass': None, 'damping_ratio': None}, {}, None):
        comp = _compliance_override(SHARED, block)
        assert comp['stiffness'] == SHARED['stiffness']
        assert comp['damping_ratio'] == SHARED['damping_ratio']


def test_overriding_one_maneuver_does_not_retune_the_others():
    """The shared section is read by every other maneuver, so the merge must not mutate it."""
    before = [list(SHARED['stiffness']), list(SHARED['mass'])]
    _compliance_override(SHARED, {'stiffness': [9999.0] * 6, 'mass': [1.0] * 6})
    assert [SHARED['stiffness'], SHARED['mass']] == before


@pytest.mark.parametrize('stiffness_n_m, expect_give_mm', [(500.0, 10.0), (2000.0, 2.5),
                                                           (5000.0, 1.0)])
def test_stiffness_sets_how_far_the_connector_gives(stiffness_n_m, expect_give_mm):
    """The whole point of the knob: give = force / stiffness. Pinned as a MECHANISM, so retuning
    the shipped numbers cannot break it."""
    force_n = 5.0

    class Arm:
        dry_run = False

        def wrench(self):
            return np.array([-force_n, 0.0, 0.0, 0.0, 0.0, 0.0])

        def servo_l(self, T, dt, lookahead, gain):
            pass

        def servo_stop(self):
            pass

    comp = _compliance_override(SHARED, {'stiffness': [stiffness_n_m] * 3 + [10.0] * 3,
                                         'mass': [1.0] * 3 + [0.1] * 3})
    adm = AdmittanceController(Arm(), comp)
    adm.reset()
    adm.hold(np.eye(4), 4.0)
    assert abs(adm.delta[0]) * 1000.0 == pytest.approx(expect_give_mm, rel=0.05)


def test_the_shipped_engage_block_declares_a_stiffness():
    """It must be an explicit 6-vector: inheriting silently is what made the give invisible."""
    en = C.load('bnc_assembly').get_path('assembly.engage') or {}
    assert 'stiffness' in en, 'assembly.engage must declare its own stiffness'
    assert len(en['stiffness']) == 6, '[x, y, z, rx, ry, rz]'
    give_mm = float(en['max_axial_force_n']) / float(en['stiffness'][0]) * 1000.0
    assert give_mm > 0
