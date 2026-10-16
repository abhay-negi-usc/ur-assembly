"""THE CONFIGS FOLDER, for a project that imports urlab: $URLAB_CONFIG_DIR redirects bare config
names and the default catalogues; unset, everything resolves in this repo's configs/ as it always has.

The failure this guards is the dangerous one: another project running `--config cartesian` and
getting THIS repo's cartesian.yaml -- whose robot.ip is the UR10e.
"""

import logging
import os

import pytest

from urlab import config as C
from urlab import tool_frames


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv(C.CONFIG_DIR_ENV, raising=False)
    monkeypatch.setattr(C, '_warned_foreign', False)


@pytest.fixture
def other_project(tmp_path, monkeypatch):
    """A stand-in for another project's configs folder, selected via the env var."""
    d = tmp_path / 'configs'
    d.mkdir()
    (d / 'demo.yaml').write_text('robot:\n  ip: 10.9.9.9\n')
    monkeypatch.setenv(C.CONFIG_DIR_ENV, str(d))
    monkeypatch.setattr(C, '_warned_foreign', False)
    return d


def test_unset_is_this_repos_configs(no_env):
    assert C.config_dir() == C.CONFIG_DIR
    assert tool_frames.frames_path() == tool_frames.DEFAULT_PATH
    assert C.load('cartesian')['_config_dir'] == C.CONFIG_DIR


def test_set_redirects_bare_names_and_every_default_catalogue(other_project):
    cfg = C.load('demo')
    assert cfg['_config_path'] == str(other_project / 'demo.yaml')
    assert cfg.get_path('robot.ip') == '10.9.9.9'
    assert tool_frames.frames_path(cfg) == str(other_project / 'frames.yaml')
    assert tool_frames.objects_path(cfg) == str(other_project / 'objects.yaml')
    assert tool_frames.marker_assemblies_path(cfg) == str(other_project / 'marker_assemblies.yaml')


def test_a_bare_name_never_falls_back_to_this_repos_configs(other_project):
    """cartesian.yaml exists HERE, not there: it must be an error, not the UR10e's file."""
    with pytest.raises(FileNotFoundError) as exc:
        C.load('cartesian')
    msg = str(exc.value)
    assert str(other_project) in msg and C.CONFIG_DIR_ENV in msg, msg


def test_a_missing_catalogue_is_loud_rather_than_the_ur10e_calibration(other_project):
    """No frames.yaml in the other project: hand_eye must fail, never read this repo's."""
    with pytest.raises(FileNotFoundError, match='frames'):
        tool_frames.hand_eye(C.load('demo'))


def test_a_config_given_by_path_resolves_its_references_beside_itself(other_project, tmp_path):
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    (elsewhere / 'x.yaml').write_text('frames_file: my-frames.yaml\n')
    cfg = C.load(str(elsewhere / 'x.yaml'))
    assert tool_frames.frames_path(cfg) == str(elsewhere / 'my-frames.yaml')


def test_a_set_but_missing_folder_is_an_error_that_names_the_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(C.CONFIG_DIR_ENV, str(tmp_path / 'nope'))
    with pytest.raises(FileNotFoundError, match=C.CONFIG_DIR_ENV):
        C.load('cartesian')


def test_home_is_expanded(tmp_path, monkeypatch):
    d = tmp_path / 'cfgs'
    d.mkdir()
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv(C.CONFIG_DIR_ENV, '~/cfgs')
    assert C.config_dir() == str(d)


def test_running_from_another_project_without_the_variable_warns_once(no_env, tmp_path,
                                                                    monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)                    # outside the urlab repo
    with caplog.at_level(logging.WARNING, logger='config'):
        C.load('cartesian')
        tool_frames.frames_path()
    warnings = [r for r in caplog.records if C.CONFIG_DIR_ENV in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    assert 'README' in warnings[0].getMessage()


def test_running_inside_this_repo_does_not_warn(no_env, monkeypatch, caplog):
    monkeypatch.chdir(C.REPO_ROOT)
    with caplog.at_level(logging.WARNING, logger='config'):
        C.load('cartesian')
        tool_frames.frames_path()
    assert not [r for r in caplog.records if C.CONFIG_DIR_ENV in r.getMessage()]
    assert os.path.realpath(os.getcwd()) == os.path.realpath(C.REPO_ROOT)
