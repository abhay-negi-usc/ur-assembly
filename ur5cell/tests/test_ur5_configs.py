"""The UR5 cell's configs resolve to the UR5 -- its IP, its controller generation, its frames file --
and never to the UR10e's machine or calibration files that live elsewhere in the same repo.

The failure this guards is SILENT: a UR5 config that forgot `robot_config:` would load the UR10e's
hand-eye from configs/frames.yaml and run without complaint.

    python -m pytest ur5cell/tests -q
"""

import glob
import os

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
    assert robot.gripper is None                   # gripper.type: none until one is mounted
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
