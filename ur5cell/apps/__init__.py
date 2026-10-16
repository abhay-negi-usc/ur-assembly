"""UR5 apps.

    python -m ur5cell.apps.monitor | cartesian | teleop      urlab's generic apps, on the UR5

Each calls ur5cell.use_ur5_configs() before urlab reads a config: $URLAB_CONFIG_DIR -> ur5cell/configs
(so a bare --config can only name a UR5 config) and --config defaulting to the ur5-<app> one.

A NEW UR5 app: start from urlab/apps/behavior_demo.py, call ur5cell.use_ur5_configs() at the top
of main(), and give run_app a ur5- config name.
"""
