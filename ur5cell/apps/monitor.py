"""urlab's monitor app, ON THE UR5:  python -m ur5cell.apps.monitor [same options as urlab.apps.monitor]

Configs resolve in ur5cell/configs only ($URLAB_CONFIG_DIR is set for this run), and --config
defaults to ur5-monitor -- so this entry point cannot load a UR10e config. See ur5cell.use_ur5_configs.
"""

import sys

import ur5cell

if __name__ == '__main__':
    sys.exit(ur5cell.run_urlab_app('urlab.apps.monitor', 'ur5-monitor'))
