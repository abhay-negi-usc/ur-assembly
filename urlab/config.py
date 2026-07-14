"""Config loading -- replaces ROS parameters.

The old stack had two overlapping mechanisms: a yaml file loaded by the node, plus `--ros-args
-p name:=value` overrides for a handful of hand-picked keys that someone had remembered to
declare. Here there is one: the yaml, with `--set key=value` able to reach ANY key, nested ones
included:

    python -m urlab.apps.cable_pick_place --set scan.approach.min_distance_m=0.15 --set debug=true

Values parse as YAML, so `true`, `0.15`, `[1, 0, 0]` and `null` all mean what they look like.
"""

import argparse
import os

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')


class Config(dict):
    """dict with dotted access: cfg.get_path('scan.approach.step_m', 0.01)."""

    def get_path(self, path, default=None):
        node = self
        for key in path.split('.'):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return default if node is None else node

    def set_path(self, path, value):
        keys = path.split('.')
        node = self
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

    def section(self, path):
        """A sub-dict, or {} -- so `cfg.section('scan').get('dwell_s', 5)` never raises."""
        return self.get_path(path, {}) or {}


def load(name_or_path, overrides=()):
    """Load configs/<name>.yaml (or an explicit path) and apply `key=value` overrides."""
    path = name_or_path
    if not os.path.isfile(path):
        path = os.path.join(CONFIG_DIR, name_or_path)
        if not path.endswith(('.yaml', '.yml')):
            path += '.yaml'
    if not os.path.isfile(path):
        raise FileNotFoundError(f'No config at {name_or_path!r} or {path!r}')

    with open(path, 'r') as f:
        cfg = Config(yaml.safe_load(f) or {})
    cfg['_config_path'] = os.path.abspath(path)
    cfg['_config_dir'] = os.path.dirname(os.path.abspath(path))

    for item in overrides:
        if '=' not in item:
            raise ValueError(f'--set expects key=value, got {item!r}')
        key, _, raw = item.partition('=')
        cfg.set_path(key.strip(), yaml.safe_load(raw))
    return cfg


def resolve(cfg, path):
    """Resolve a possibly-relative path from a config against the config file's own directory --
    so `trajectory_csv: assembly_trajectory.csv` finds the file sitting next to the yaml."""
    if os.path.isabs(path):
        return path
    return os.path.join(cfg.get('_config_dir', CONFIG_DIR), path)


def arg_parser(description, default_config):
    """The CLI every app shares: --config, --set, --robot-ip, --dry-run, --yes, --debug."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', default=default_config,
                   help=f'config name in configs/ or a path (default: {default_config})')
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                   help='override any config key, nested with dots (repeatable)')
    p.add_argument('--robot-ip', default=None, help='overrides robot.ip')
    p.add_argument('--dry-run', action='store_true',
                   help='plan and print every move without connecting to the robot')
    p.add_argument('--yes', '-y', action='store_true',
                   help='skip the per-step confirmation prompts')
    p.add_argument('--debug', action='store_true', help='verbose pose/delta logging')
    return p


def from_args(args):
    """Config from parsed args, with the CLI flags folded in as overrides."""
    cfg = load(args.config, args.overrides)
    if args.robot_ip:
        cfg.set_path('robot.ip', args.robot_ip)
    if args.dry_run:
        cfg.set_path('robot.dry_run', True)
    if args.yes:
        cfg.set_path('confirm_each_step', False)
    if args.debug:
        cfg.set_path('debug', True)
    return cfg
