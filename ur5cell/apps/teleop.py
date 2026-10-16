"""urlab's teleop app, ON THE UR5:  python -m ur5cell.apps.teleop [same options as urlab.apps.teleop]

Configs resolve in ur5cell/configs only ($URLAB_CONFIG_DIR is set for this run), and --config
defaults to ur5-teleop -- so this entry point cannot load a UR10e config. See ur5cell.use_ur5_configs.
"""

import sys

import ur5cell

if __name__ == '__main__':
    sys.exit(ur5cell.run_urlab_app('urlab.apps.teleop', 'ur5-teleop'))
