"""tests/ checks THIS repo's configs (the UR10e cell's). A shell left with $URLAB_CONFIG_DIR pointing
elsewhere -- e.g. at ur5cell/configs after a UR5 session -- must not change what these tests load."""

import pytest


@pytest.fixture(autouse=True)
def _this_repos_configs(monkeypatch):
    monkeypatch.delenv('URLAB_CONFIG_DIR', raising=False)
