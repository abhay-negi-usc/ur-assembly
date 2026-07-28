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

    _apply_overrides(cfg, overrides)      # 1st pass: a `--set cable=...` can select the profile
    apply_cable_profile(cfg)              # override gripper/grasp-check counts for the chosen cable
    _apply_overrides(cfg, overrides)      # 2nd pass: an explicit `--set` still wins over the profile
    return cfg


def _apply_overrides(cfg, overrides):
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'--set expects key=value, got {item!r}')
        key, _, raw = item.partition('=')
        cfg.set_path(key.strip(), yaml.safe_load(raw))


def apply_cable_profile(cfg):
    """If the config selects a cable (`cable: <name>`), load the cables file (default cables.yaml,
    resolved next to the config) and OVERRIDE the gripper endpoints + grasp-check counts from that
    entry -- so one demo config runs any cable by naming it. No-op if `cable` is unset. Schema +
    the exact key mapping are documented in configs/cables.yaml."""
    name = cfg.get('cable')
    if not name:
        return cfg
    cables_path = resolve(cfg, cfg.get('cables_file', 'cables.yaml'))
    if not os.path.isfile(cables_path):
        raise FileNotFoundError(f'cable {name!r} selected but no cables file at {cables_path!r}')
    with open(cables_path, 'r') as f:
        db = yaml.safe_load(f) or {}
    entry = (db.get('cables') or {}).get(name)
    if entry is None:
        have = ', '.join((db.get('cables') or {}).keys()) or '(none)'
        raise KeyError(f'cable {name!r} not in {cables_path} (have: {have})')

    shared = db.get('gripper') or {}                          # gripper params live SOLELY here now
    if 'port' in shared:
        cfg.set_path('gripper.port', shared['port'])
    if 'open' in shared:
        cfg.set_path('gripper.open_counts', int(shared['open']))
    if 'closed' in shared:                                    # closed on nothing = the empty band
        cfg.set_path('gripper.closed_counts', int(shared['closed']))
        cfg.set_path('grasp_check.empty_counts', int(shared['closed']))
    if 'speed' in shared:
        cfg.set_path('gripper.speed_counts', int(shared['speed']))
    if 'force' in shared:
        cfg.set_path('gripper.force_counts', int(shared['force']))
    # The grasp TARGET is the CONNECTOR, so its count range is the SUCCESS band; the cable (thinner,
    # HIGHER count) is a miss above it, and anything thicker (<= band) is a miss below it.
    conn = entry.get('connector')
    if conn:
        lo, hi = int(min(conn)), int(max(conn))
        cfg.set_path('grasp_check.connector_counts', [lo, hi])
        cfg.set_path('grasp_check.groove_counts', (lo + hi) // 2)     # success reference (midpoint)
        cfg.set_path('grasp_check.faces_max_counts', lo - 1)          # <= this  = miss (too thick)
        cfg.set_path('grasp_check.groove_max_counts', hi)            # >  this  = miss (the cable)
    if 'cable' in entry:
        cfg.set_path('grasp_check.cable_counts', int(entry['cable']))   # reference: cable grab = miss
    # Fingertip target offset from the junction, POSITIVE along the connector axis (x, toward the
    # connector's END): connector_grasp is the fingertip pose relative to the connector frame.
    off = float(entry.get('junction_offset_m', 0.0))
    cfg.set_path('connector_grasp.xyz', [off, 0.0, 0.0])
    # Held-connector calibration (connector_holder -> connector) for the assembly / uncertain_sampling.
    if entry.get('connector_in_holder'):
        cfg.set_path('connector_in_holder', entry['connector_in_holder'])
    if entry.get('target_connector_pose'):                    # (future) the assembly target
        cfg.set_path('assembly.target_connector_pose', entry['target_connector_pose'])
    cfg['_cable_profile'] = name
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
