"""KEYBOARD TELEOP, without a robot: the jog convention and the key parser.

The jog frame is the thing most likely to be silently wrong -- a tool-frame jog that moves along
base axes looks right whenever the tool happens to be square to the base.
"""

import numpy as np

from urlab.apps.teleop import jog, split_keys
from urlab.transforms import xyzrpy_to_matrix


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


def test_split_keys_drops_escape_sequences():
    # An arrow key's '[' must not reach the stiffness handler.
    assert split_keys('w\x1b[Aa') == ['w', 'a']
    assert split_keys('\x1b[1;5C]') == [']']
    assert split_keys('\x1bOPx') == ['x']
    assert split_keys('\x1b') == []
    assert split_keys('[]{}') == ['[', ']', '{', '}']
