"""ur5cell -- the UR5 (CB3.1) cell: its apps, configs and tests, on top of the shared `urlab` core.

The core is the same library the UR10e cell runs; this package holds only what is THIS cell's --
its machine files (configs/ur5-*.yaml) and its task apps. See ur5cell/README.md.
"""

import os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')


def config_path(name):
    """ur5cell/configs/<name>.yaml as an absolute path -- the `default_config` a UR5 app hands
    run_app, since urlab's bare-name lookup only searches the UR10e's configs/."""
    return os.path.join(CONFIG_DIR, name if name.endswith('.yaml') else name + '.yaml')
