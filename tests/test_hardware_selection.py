"""HARDWARE SELECTION, without hardware: how URArm connects, what a CB3 arm does in place of the
e-Series-only calls, and which gripper / camera a config builds.

Every DEFAULT must reproduce exactly what the UR10e cell has always done -- adding a cell must not
change the one that is running experiments. ur_rtde is replaced by recorders, so these run anywhere.
"""

import sys
import types

import numpy as np
import pytest

from urlab.config import Config
from urlab.robot import Robot
from urlab.robot.arm import ArmError, URArm
from urlab.robot.gripper import GripperError, Robotiq2F85, make_gripper


class _Recorder:
    """Stands in for an ur_rtde interface: keeps its constructor args and the names it was called
    with. getActualTCPForce returns whatever the test put in `tcp_force`."""
    FLAG_USE_EXT_UR_CAP = 0x02

    def __init__(self, *args):
        self.args = args
        self.calls = []
        self.tcp_force = [0.0] * 6
        type(self).instances.append(self)

    def getTCPOffset(self):
        return [0.0] * 6

    def getActualTCPForce(self):
        return list(self.tcp_force)

    def __getattr__(self, name):
        def call(*_a, **_k):
            self.calls.append(name)
            return True
        return call


@pytest.fixture
def rtde(monkeypatch):
    class Control(_Recorder):
        instances = []

    class Receive(_Recorder):
        instances = []

    for name, attr, cls in (('rtde_control', 'RTDEControlInterface', Control),
                            ('rtde_receive', 'RTDEReceiveInterface', Receive)):
        mod = types.ModuleType(name)
        setattr(mod, attr, cls)
        monkeypatch.setitem(sys.modules, name, mod)
    return Control, Receive


def _arm(**robot):
    return URArm(Config({'robot': {'ip': '10.0.0.5', **robot}}))


# ---------------------------------------------------------------------------- connection
def test_the_default_connection_is_the_call_the_ur10e_always_made(rtde):
    Control, Receive = rtde
    _arm()
    assert Control.instances[-1].args == ('10.0.0.5',)
    assert Receive.instances[-1].args == ('10.0.0.5',)


def test_a_stated_rtde_rate_reaches_both_interfaces(rtde):
    Control, Receive = rtde
    _arm(rtde_frequency_hz=125)
    assert Control.instances[-1].args == ('10.0.0.5', 125.0)
    assert Receive.instances[-1].args == ('10.0.0.5', 125.0)


def test_urcap_mode_passes_the_flag_and_the_port_to_the_control_interface_only(rtde):
    Control, Receive = rtde
    _arm(use_ext_ur_cap=True, ur_cap_port=50002)
    # -1 = "the controller's maximum rate", ur_rtde's own default for the slot before the flags.
    assert Control.instances[-1].args == ('10.0.0.5', -1.0, 0x02, 50002)
    assert Receive.instances[-1].args == ('10.0.0.5',)


def test_a_urcap_port_without_urcap_mode_changes_nothing(rtde):
    """ur5-robot.yaml records the port for a later switch; on its own it must stay inert."""
    Control, _ = rtde
    _arm(ur_cap_port=50002)
    assert Control.instances[-1].args == ('10.0.0.5',)


# ---------------------------------------------------------------------------- CB3 vs e-Series
def test_an_unknown_series_is_refused(rtde):
    with pytest.raises(ArmError, match='robot.series'):
        _arm(series='cb2')


def test_e_series_tares_on_the_controller(rtde):
    arm = _arm()
    arm.zero_ft(settle=False)
    assert 'zeroFtSensor' in arm.rtde_c.calls
    arm.rtde_r.tcp_force = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    # The controller does the subtracting on e-Series; we only bridge UR base -> base_link.
    assert np.allclose(arm.wrench(), [-1.0, -2.0, 3.0, -0.1, -0.2, 0.3])


def test_cb3_tares_in_software_and_still_bridges_to_base_link(rtde):
    arm = _arm(series='cb3')
    arm.rtde_r.tcp_force = [5.0, -4.0, 30.0, 0.5, 0.4, -0.3]     # tool weight etc. at the tare
    arm.zero_ft(settle=False)
    assert 'zeroFtSensor' not in arm.rtde_c.calls               # e-Series-only ("G5")
    assert np.allclose(arm.wrench(), 0.0)
    arm.rtde_r.tcp_force = [6.0, -2.0, 33.0, 4.5, 5.4, 5.7]     # + [1, 2, 3, 4, 5, 6]
    # x/y NEGATED, z kept: the Rz(pi) bridge applies after the tare, as on e-Series. Some axes
    # wrong = a FRAME error -- see the README's symptom key.
    assert np.allclose(arm.wrench(), [-1.0, -2.0, 3.0, -4.0, -5.0, 6.0])


@pytest.mark.parametrize('series, scaled', [('e_series', True), ('cb3', False)])
def test_force_mode_sets_gain_scaling_only_on_e_series(rtde, series, scaled):
    arm = _arm(series=series)
    arm.force_mode(np.eye(4), [0, 0, 1, 0, 0, 0], [0.0] * 6, [0.1] * 6)
    assert 'forceModeSetDamping' in arm.rtde_c.calls
    assert 'forceMode' in arm.rtde_c.calls
    assert ('forceModeSetGainScaling' in arm.rtde_c.calls) is scaled


# ---------------------------------------------------------------------------- gripper / camera
def _dry(**blocks):
    return Config({'robot': {'dry_run': True}, **blocks})


def test_the_default_gripper_is_the_robotiq():
    assert isinstance(make_gripper(_dry()), Robotiq2F85)


def test_gripper_type_none_builds_a_robot_with_no_gripper():
    assert make_gripper(_dry(gripper={'type': 'none'})) is None
    robot = Robot(_dry(gripper={'type': 'none'}))
    assert robot.gripper is None
    robot.close()                                  # must not trip over the missing gripper


def test_an_unknown_gripper_is_refused_by_name():
    with pytest.raises(GripperError, match='no_such_gripper'):
        make_gripper(_dry(gripper={'type': 'no_such_gripper'}))


def test_the_default_camera_is_the_realsense():
    from urlab.perception import make_camera
    from urlab.perception.camera import RealSenseCamera
    assert isinstance(make_camera(_dry()), RealSenseCamera)


def test_an_unknown_camera_is_refused_by_name():
    from urlab.perception import make_camera
    with pytest.raises(ValueError, match='unset'):
        make_camera(_dry(camera={'type': 'unset'}))
