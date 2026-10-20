"""KEYBOARD TELEOP, without a robot: the jog convention and the key parser.

The jog frame is the thing most likely to be silently wrong -- a tool-frame jog that moves along
base axes looks right whenever the tool happens to be square to the base.
"""

import numpy as np
import pytest

from urlab import config as C
from urlab.apps.teleop import Teleop, jog, split_keys
from urlab.robot import Robot
from urlab.transforms import BASE_LINK_FROM_UR_BASE, inverse, xyzrpy_to_matrix


def _yawed():
    # A tool pointing down and yawed 90 deg, so no tool axis coincides with a base axis.
    return xyzrpy_to_matrix([0.5, 0.2, 0.3], np.radians([180.0, 0.0, 90.0]))


def test_base_translation_ignores_attitude():
    T = _yawed()
    out = jog(T, 0, 0.01, 'base')
    assert np.allclose(out[:3, 3] - T[:3, 3], [0.01, 0.0, 0.0])
    assert np.allclose(out[:3, :3], T[:3, :3])


def test_tool_translation_follows_tool_axes():
    T = _yawed()
    out = jog(T, 2, 0.01, 'tool')
    assert np.allclose(out[:3, 3] - T[:3, 3], 0.01 * T[:3, 2])     # along the tool's own z
    assert np.allclose(out[:3, :3], T[:3, :3])


def test_rotations_pivot_on_the_frame_origin():
    T = _yawed()
    for frame in ('base', 'tool'):
        out = jog(T, 5, np.radians(10.0), frame)
        assert np.allclose(out[:3, 3], T[:3, 3])


def test_base_rotation_premultiplies_tool_rotation_postmultiplies():
    T = _yawed()
    a = np.radians(10.0)
    Rz = xyzrpy_to_matrix([0, 0, 0], [0, 0, a])[:3, :3]
    assert np.allclose(jog(T, 5, a, 'base')[:3, :3], Rz @ T[:3, :3])
    assert np.allclose(jog(T, 5, a, 'tool')[:3, :3], T[:3, :3] @ Rz)


def test_tool_z_rotation_keeps_the_tool_axis():
    T = _yawed()
    out = jog(T, 5, np.radians(30.0), 'tool')
    assert np.allclose(out[:3, 2], T[:3, 2])


def _in_ur_base(T):
    """A base_link pose as the pendant's Base view shows it."""
    return inverse(BASE_LINK_FROM_UR_BASE) @ T


def test_base_jog_in_ur_base_follows_the_pendant_axes():
    T, R_ub = _yawed(), BASE_LINK_FROM_UR_BASE[:3, :3]
    out = jog(T, 0, 0.01, 'base', R_ub)
    assert np.allclose(_in_ur_base(out)[:3, 3] - _in_ur_base(T)[:3, 3], [0.01, 0.0, 0.0])
    assert np.allclose(out[:3, 3] - T[:3, 3], [-0.01, 0.0, 0.0])    # backwards in base_link
    a = np.radians(10.0)
    Rx = xyzrpy_to_matrix([0, 0, 0], [a, 0, 0])[:3, :3]
    out = jog(T, 3, a, 'base', R_ub)
    assert np.allclose(_in_ur_base(out)[:3, :3], Rx @ _in_ur_base(T)[:3, :3])
    assert np.allclose(out[:3, 3], T[:3, 3])


def test_tool_jog_ignores_the_base_jog_frame():
    T, R_ub = _yawed(), BASE_LINK_FROM_UR_BASE[:3, :3]
    for axis in range(6):
        assert np.allclose(jog(T, axis, 0.01, 'tool', R_ub), jog(T, axis, 0.01, 'tool'))


@pytest.fixture
def robot():
    r = Robot(C.load('teleop', ['robot.dry_run=true']), with_gripper=False)
    yield r
    r.close()


def test_the_base_jog_frame_defaults_to_base_link(robot):
    teleop = Teleop(C.load('teleop', ['robot.dry_run=true']), robot)
    assert teleop.base_name == 'base_link'
    assert np.allclose(teleop.R_base, np.eye(3))


@pytest.mark.parametrize('name, error', [('tool0', ValueError), ('ur_bsae', KeyError)])
def test_a_base_jog_frame_on_the_arm_or_unknown_is_refused(robot, name, error):
    cfg = C.load('teleop', ['robot.dry_run=true', f'teleop.base_jog_frame={name}'])
    with pytest.raises(error, match=name):
        Teleop(cfg, robot)


def test_split_keys_drops_escape_sequences():
    # An arrow key's '[' must not reach the stiffness handler.
    assert split_keys('w\x1b[Aa') == ['w', 'a']
    assert split_keys('\x1b[1;5C]') == [']']
    assert split_keys('\x1bOPx') == ['x']
    assert split_keys('\x1b') == []
    assert split_keys('[]{}') == ['[', ']', '{', '}']
