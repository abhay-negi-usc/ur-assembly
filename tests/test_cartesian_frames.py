"""The cartesian jog's motion frames, against what the pendant shows.

urlab works in ROS base_link; the pendant (and RTDE) in the UR `base` frame, turned 180 deg about Z.
Jogging in 'world' therefore runs x/y BACKWARDS against the pendant's Base view -- 'ur_base' is the
frame that matches it. These tests read the jog targets back in RTDE's own frame, i.e. as the
pendant would display them.

    python -m pytest tests/test_cartesian_frames.py -q
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urlab import config as C
from urlab.apps import cartesian
from urlab.robot import Robot
from urlab.transforms import BASE_LINK_FROM_UR_BASE, matrix_to_rtde

LIN, ANG = 0.02, np.radians(10.0)


@pytest.fixture
def robot():
    r = Robot(C.load('cartesian', ['robot.dry_run=true']), with_gripper=False)
    yield r
    r.close()


def _pendant(T):
    """A base_link pose as the pendant shows it: [x, y, z] and the rotation, in UR `base`."""
    p = matrix_to_rtde(T)
    return np.array(p[:3]), Rotation.from_rotvec(p[3:]).as_matrix()


def test_ur_base_is_registered_as_base_link_turned_about_z(robot):
    np.testing.assert_allclose(robot.pose('ur_base'), BASE_LINK_FROM_UR_BASE, atol=1e-12)
    assert not robot.is_tool_attached('ur_base')


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_a_ur_base_translation_moves_along_the_pendant_base_axis(robot, axis):
    T_home = robot.tool0()
    R = cartesian._frame_axes(robot, 'ur_base')
    target = cartesian._offset(T_home, R, axis, +1, LIN, ANG)
    expected = np.zeros(3)
    expected[axis] = LIN
    np.testing.assert_allclose(_pendant(target)[0] - _pendant(T_home)[0], expected, atol=1e-9)


@pytest.mark.parametrize('axis', [3, 4, 5])
def test_a_ur_base_rotation_turns_about_the_pendant_base_axis(robot, axis):
    T_home = robot.tool0()
    R = cartesian._frame_axes(robot, 'ur_base')
    target = cartesian._offset(T_home, R, axis, +1, LIN, ANG)
    (p0, R0), (p1, R1) = _pendant(T_home), _pendant(target)
    rotvec = np.zeros(3)
    rotvec[axis - 3] = ANG
    np.testing.assert_allclose(R1 @ R0.T, Rotation.from_rotvec(rotvec).as_matrix(), atol=1e-9)
    np.testing.assert_allclose(p1, p0, atol=1e-12)                # turns about tool0, in place


def test_world_is_base_link_and_runs_x_y_backwards_on_the_pendant(robot):
    """Kept for older configs -- and why the UR5 config does not use it."""
    T_home = robot.tool0()
    target = cartesian._offset(T_home, cartesian._frame_axes(robot, 'world'), 0, +1, LIN, ANG)
    np.testing.assert_allclose(_pendant(target)[0] - _pendant(T_home)[0], [-LIN, 0, 0], atol=1e-9)


def test_tool0_axes_are_the_home_orientation(robot):
    np.testing.assert_allclose(cartesian._frame_axes(robot, 'tool0'), robot.tool0()[:3, :3])


def test_an_unknown_motion_frame_stops_the_run_before_any_move(robot, monkeypatch):
    moves = []
    monkeypatch.setattr(robot, 'move_tool0', lambda T, label: moves.append(label) or True)
    cfg = {'motion_frames': ['ur_base', 'tolo0'], 'confirm_each_step': False}
    with pytest.raises(KeyError, match='tolo0'):
        cartesian.build_and_run(cfg, robot, None, None)
    assert moves == []


def test_rotation_labels_name_the_axis_they_turn_about():
    assert cartesian._label('tool0', 0, +1) == 'tool0 x+'
    assert cartesian._label('tool0', 3, +1) == 'tool0 R+ (about tool0 x)'
    assert cartesian._label('ur_base', 4, -1) == 'ur_base P- (about ur_base y)'
    assert cartesian._label('ur_base', 5, +1) == 'ur_base Y+ (about ur_base z)'
