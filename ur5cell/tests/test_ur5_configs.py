"""The UR5 cell's configs resolve to the UR5 -- its IP, its controller generation, its frames file --
and never to the UR10e's machine or calibration files that live elsewhere in the same repo.

The failure this guards is SILENT: a UR5 config that forgot `robot_config:` would load the UR10e's
hand-eye from configs/frames.yaml and run without complaint.

    python -m pytest ur5cell/tests -q
"""

import glob
import os

import numpy as np
import pytest

import ur5cell
from urlab import config as C
from urlab import tool_frames
from urlab.robot import Robot

MACHINE_FILES = {'ur5-robot.yaml', 'ur5-camera.yaml', 'ur5-frames.yaml'}


def _all():
    return sorted(glob.glob(os.path.join(ur5cell.CONFIG_DIR, '*.yaml')))


def _demo_configs():
    return [p for p in _all() if os.path.basename(p) not in MACHINE_FILES]


def test_every_ur5_config_file_is_prefixed():
    """`ur5-` on every file, so a UR5 config can never be mistaken for a UR10e one in an editor."""
    for p in _all():
        assert os.path.basename(p).startswith('ur5-'), p


@pytest.mark.parametrize('path', _demo_configs(), ids=os.path.basename)
def test_a_ur5_config_resolves_to_the_ur5(path):
    cfg = C.load(path)
    assert cfg.get('robot_config') == 'ur5-robot.yaml'
    assert cfg.get_path('robot.ip') == '192.168.10.125'
    assert cfg.get_path('robot.series') == 'cb3'
    frames = os.path.realpath(tool_frames.frames_path(cfg))
    assert os.path.dirname(frames) == os.path.realpath(ur5cell.CONFIG_DIR), frames
    assert frames != os.path.realpath(tool_frames.DEFAULT_PATH), 'fell back to the UR10e frames'


@pytest.mark.parametrize('path', _demo_configs(), ids=os.path.basename)
def test_a_ur5_robot_builds_without_hardware(path):
    robot = Robot(C.load(path, ['robot.dry_run=true']))
    assert robot.arm.series == 'cb3'
    assert type(robot.gripper).__name__ == 'Robotiq2F85'
    robot.close()


@pytest.mark.parametrize('path', _demo_configs(), ids=os.path.basename)
def test_the_robot_fingertip_and_grasp_are_the_catalogue_frames(path):
    """ur5-robot.yaml NAMES them, so the spacer -> gripper chain is the only copy of the numbers."""
    cfg = C.load(path, ['robot.dry_run=true'])
    assert cfg.section('fingertip_grasp') == {'frame': 'fingertip'}
    assert cfg.section('grasp_tcp_offset') == {'frame': 'grasp'}
    frames = tool_frames.load_frames(cfg)
    robot = Robot(cfg, with_gripper=False)
    assert np.allclose(robot.T_tool0_fingertip, frames['fingertip'])
    assert np.allclose(robot.T_tool0_grasp, frames['grasp'])
    robot.close()


def test_the_camera_and_gripper_ride_on_the_spacer():
    """Mounted TO the spacer: anything later fitted between flange and spacer moves both."""
    import yaml
    with open(os.path.join(ur5cell.CONFIG_DIR, 'ur5-frames.yaml')) as fh:
        raw = yaml.safe_load(fh)['frames']
    assert raw['camera']['parent'] == 'spacer'
    assert raw['gripper']['parent'] == 'spacer'
    assert raw['fingertip']['parent'] == raw['grasp']['parent'] == 'gripper'


def test_the_cartesian_jog_frames_are_ones_the_ur5_robot_knows():
    """ur5-cartesian names ur_base (the pendant's Base axes), not world (base_link, x/y negated)."""
    from urlab.apps import cartesian
    cfg = C.load(ur5cell.config_path('ur5-cartesian'), ['robot.dry_run=true'])
    assert 'world' not in cfg['motion_frames']
    robot = Robot(cfg, with_gripper=False)
    for frame in cfg['motion_frames']:
        cartesian._frame_axes(robot, frame)       # raises on a frame the robot does not know
    robot.close()


def test_teleop_base_jogs_in_the_pendant_base_axes():
    from urlab.apps.teleop import Teleop
    cfg = C.load(ur5cell.config_path('ur5-teleop'), ['robot.dry_run=true'])
    robot = Robot(cfg, with_gripper=False)
    teleop = Teleop(cfg, robot)
    assert teleop.base_name == 'ur_base'
    assert np.allclose(teleop.R_base, np.diag([-1.0, -1.0, 1.0]))
    robot.close()


def test_config_path_names_a_file_in_this_cell():
    assert ur5cell.config_path('ur5-monitor') == os.path.join(ur5cell.CONFIG_DIR, 'ur5-monitor.yaml')
    assert os.path.isfile(ur5cell.config_path('ur5-monitor.yaml'))


# ---------------------------------------------------------------------------- the UR5 entry points
@pytest.fixture
def clean_env(monkeypatch):
    """use_ur5_configs writes os.environ; this makes monkeypatch restore it after the test."""
    monkeypatch.delenv(C.CONFIG_DIR_ENV, raising=False)


def test_use_ur5_configs_points_urlab_at_this_cell(clean_env):
    ur5cell.use_ur5_configs(argv=['prog'])
    assert C.config_dir() == os.path.realpath(ur5cell.CONFIG_DIR)
    assert C.load('ur5-cartesian').get_path('robot.ip') == '192.168.10.125'


def test_through_the_ur5_entry_points_a_ur10e_config_name_is_refused(clean_env):
    """cartesian.yaml is the UR10e's (robot.ip .112): from a UR5 entry point it must not load."""
    ur5cell.use_ur5_configs(argv=['prog'])
    with pytest.raises(FileNotFoundError, match='URLAB_CONFIG_DIR'):
        C.load('cartesian')


@pytest.mark.parametrize('argv, expected', [
    (['prog', '--once'], ['prog', '--config', 'ur5-monitor', '--once']),
    (['prog', '--config', 'ur5-teleop'], ['prog', '--config', 'ur5-teleop']),
    (['prog', '--config=ur5-teleop'], ['prog', '--config=ur5-teleop']),
])
def test_the_default_config_is_the_ur5_one_unless_given(clean_env, argv, expected):
    assert ur5cell.use_ur5_configs('ur5-monitor', argv=list(argv)) == expected


@pytest.mark.parametrize('app', ['monitor', 'cartesian', 'teleop'])
def test_every_wrapper_has_its_ur5_config(app):
    assert os.path.isfile(ur5cell.config_path(f'ur5-{app}'))
