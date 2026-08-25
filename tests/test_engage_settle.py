"""The end-of-engage settle must not unload the contact it just made.

`T_cmd = T_ref @ Delta`. When engage stops on its force criterion, Delta IS the contact deflection
(w/S). Re-referencing the hold to the MEASURED pose with Delta zeroed does not jump in that first
cycle -- which is why it reads as safe -- but it discards the equilibrium, so the integrator spends
the settle re-deriving it and walks the connector w/S back out of the socket.

This pins the MECHANISM, so it survives retuning of the force limit or the stiffness.
"""

import numpy as np
import pytest

from urlab.robot.admittance import AdmittanceController

STIFF_N_M = 500.0
CONTACT_N = 5.0                      # the axial reaction still present when engage exits
EXPECTED_RETREAT_M = CONTACT_N / STIFF_N_M       # 10 mm -- what a fresh zero must re-derive


class FakeArm:
    """Reports a constant axial contact force and records every commanded pose."""

    dry_run = False

    def __init__(self):
        self.commanded = []

    def wrench(self):
        # Pushing back along -x on the tool: the reaction to an insertion along +x.
        return np.array([-CONTACT_N, 0.0, 0.0, 0.0, 0.0, 0.0])

    def servo_l(self, T, dt, lookahead, gain):
        self.commanded.append(np.array(T, dtype=float))

    def servo_stop(self):
        pass


def _controller(arm):
    return AdmittanceController(arm, {
        'mass': [1.0] * 3 + [0.1] * 3,
        'stiffness': [STIFF_N_M] * 3 + [10.0] * 3,
        'damping_ratio': [1.0] * 6,
        'reference_rate_hz': 125.0,
        'max_delta_m': 0.05,
    })


def _x_travel(arm):
    """Signed x displacement, in the tool frame, from first commanded pose to last."""
    return float(arm.commanded[-1][0, 3] - arm.commanded[0][0, 3])


def _loaded_engage_end():
    """Drive the controller to the loaded equilibrium, as the end of engage leaves it."""
    arm = FakeArm()
    adm = _controller(arm)
    ref = np.eye(4)
    ref[0, 3] = 0.200                       # a reference deliberately DEEPER than the part can go
    adm.reset()
    adm.hold(ref, 3.0)                      # long enough to reach S*Delta = w
    settled = np.array(adm._delta[:3])
    assert settled[0] == pytest.approx(-EXPECTED_RETREAT_M, rel=0.05), \
        'the fixture must actually reach the loaded equilibrium before the settle is tested'
    return adm, ref, arm


def test_settling_on_the_live_reference_keeps_the_preload():
    """The fix: hold the reference engage ended on, Delta intact. The command must not move."""
    adm, ref, _ = _loaded_engage_end()
    arm = FakeArm()
    adm.arm = arm
    adm.hold(ref, 3.0)                      # what the settle now does
    assert abs(_x_travel(arm)) < 1e-4, \
        f'settling on the live reference moved the tool {_x_travel(arm) * 1000:.2f} mm'


def test_resetting_onto_the_measured_pose_walks_back_out():
    """The bug, pinned so it cannot come back: zeroing Delta and re-referencing to where the arm
    IS starts from the right place and unloads by the full w/S over the settle -- a slow RAMP out,
    not a step, which is exactly why it survived inspection and a previous round of fixes."""
    adm, _, _ = _loaded_engage_end()
    arm = FakeArm()
    adm.arm = arm
    stay = adm._command_pose(np.eye(4))     # "where the tool actually is", as tool0() would report
    adm.reset()
    adm.hold(stay, 3.0)

    # No step: the integrator restarts from zero, so cycle one moves a fraction of a mm. The
    # damage accumulates over the settle instead, which is what makes it hard to spot.
    first_step = float(arm.commanded[0][0, 3] - stay[0, 3])
    assert abs(first_step) < 0.05 * EXPECTED_RETREAT_M,         f'expected a gradual ramp, not a jump; first cycle moved {first_step * 1000:.2f} mm'
    assert _x_travel(arm) == pytest.approx(-EXPECTED_RETREAT_M, rel=0.05), \
        'the retreat should be the whole contact deflection, re-derived from the fresh zero'


def test_the_two_differ_by_the_whole_contact_deflection():
    """Stated as the quantity that matters on the bench: how far the connector comes out."""
    adm_a, ref, _ = _loaded_engage_end()
    arm_a = FakeArm()
    adm_a.arm = arm_a
    adm_a.hold(ref, 3.0)

    adm_b, _, _ = _loaded_engage_end()
    arm_b = FakeArm()
    adm_b.arm = arm_b
    stay = adm_b._command_pose(np.eye(4))
    adm_b.reset()
    adm_b.hold(stay, 3.0)

    assert abs(_x_travel(arm_a) - _x_travel(arm_b)) == pytest.approx(EXPECTED_RETREAT_M, rel=0.05)
