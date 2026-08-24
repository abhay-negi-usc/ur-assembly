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

    _apply_common(cfg)                    # base layer: top-level blocks absent here come from _common
    _apply_overrides(cfg, overrides)      # 1st pass: a `--set cable=...` can select the profile
    apply_cable_profile(cfg)              # override gripper/grasp-check counts for the chosen cable
    _apply_overrides(cfg, overrides)      # 2nd pass: an explicit `--set` still wins over the profile
    return cfg


COMMON_FILE = '_common.yaml'


def _apply_common(cfg):
    """Fill TOP-LEVEL keys the config does not define from the _common.yaml sitting next to it --
    the shared robot-box definition (robot, camera, speed), written ONCE instead of repeated in
    every demo config.

    The merge is deliberately WHOLE-BLOCK, not per-key: a config that defines `speed:` owns the
    entire block. Merging key-by-key would mix schema generations inside one block (a config's
    legacy `joint_acceleration_rad_s2` silently fighting an inherited
    `max_joint_acceleration_deg_s2`), which is worse than repeating a block. Loading _common.yaml
    itself, or a config in a directory without one, is a no-op."""
    if os.path.basename(cfg.get('_config_path', '')) == COMMON_FILE:
        return cfg
    path = os.path.join(cfg.get('_config_dir', CONFIG_DIR), COMMON_FILE)
    if not os.path.isfile(path):
        return cfg
    with open(path, 'r') as f:
        common = yaml.safe_load(f) or {}
    for key, value in common.items():
        if key not in cfg:
            cfg[key] = value
    return cfg


def _apply_overrides(cfg, overrides):
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'--set expects key=value, got {item!r}')
        key, _, raw = item.partition('=')
        cfg.set_path(key.strip(), yaml.safe_load(raw))


def _pose_si(pose):
    """A cables.yaml pose block -> the repo-standard {xyz: m, rpy: rad} that from_cfg expects.

    Calibration poses are read straight off the `monitor`, which prints mm and deg -- so they may be
    written in those units, with the unit IN THE KEY NAME so it can never be mistaken for the m/rad
    convention used everywhere else:

        xyz_mm: [90.71, 1073.48, -184.20]      rpy_deg: [-0.79, -0.25, 88.92]

    `xyz`/`rpy` (m/rad) are still accepted. Specifying BOTH units for one axis triple is an error --
    silently preferring one would turn a 90 mm offset into 90 m, or 88 deg into 88 rad."""
    import math

    p = dict(pose or {})
    for si, alt, scale in (('xyz', 'xyz_mm', 1e-3), ('rpy', 'rpy_deg', math.pi / 180.0)):
        if alt in p:
            if si in p:
                raise ValueError(f'pose block sets both {si!r} and {alt!r}; use one unit, not both')
            p[si] = [float(v) * scale for v in p.pop(alt)]
    return p


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
    # PHYSICAL DIMENSIONS ARE PREFERRED: <cable>.connector_diameter_mm derives the band through
    # the calibrated gripper kinematics + the fingertip groove depth (gripper.groove_depth_mm),
    # so re-measuring a connector means re-measuring a DIAMETER with calipers, not gripper
    # counts. Raw count keys remain the fallback (and the logged reference when both exist).
    import math

    conn = entry.get('connector')
    d_conn = entry.get('connector_diameter_mm')
    if d_conn:
        from .robot.gripper_kinematics import GROOVE_DEPTH_M, counts_from_width
        groove = (float(shared['groove_depth_mm']) / 1000.0
                  if shared.get('groove_depth_mm') is not None else GROOVE_DEPTH_M)
        d_lo, d_hi = sorted(float(v) for v in d_conn)
        margin = int(entry.get('band_margin_counts', 1))         # widen for stall-force spread
        lo = int(math.floor(counts_from_width(d_hi / 1000.0, groove))) - margin  # thickest end
        hi = int(math.ceil(counts_from_width(d_lo / 1000.0, groove))) + margin   # thinnest end
        if conn:
            print(f'  cable {name!r}: connector band from diameters {d_lo}-{d_hi} mm -> '
                  f'[{lo}, {hi}] counts (measured reference: {sorted(int(v) for v in conn)})')
        cfg.set_path('grasp_check.connector_diameter_mm', [d_lo, d_hi])   # for payload-width checks
        conn = [lo, hi]
    if conn:
        lo, hi = int(min(conn)), int(max(conn))
        cfg.set_path('grasp_check.connector_counts', [lo, hi])
        cfg.set_path('grasp_check.groove_counts', (lo + hi) // 2)     # success reference (midpoint)
        cfg.set_path('grasp_check.faces_max_counts', lo - 1)          # <= this  = miss (too thick)
        cfg.set_path('grasp_check.groove_max_counts', hi)            # >  this  = miss (the cable)
    if entry.get('cable_diameter_mm') is not None:
        cfg.set_path('grasp_check.cable_diameter_mm', float(entry['cable_diameter_mm']))
    if 'cable' in entry:
        cfg.set_path('grasp_check.cable_counts', int(entry['cable']))   # reference: cable grab = miss
    # Grasp geometry: junction_in_fingertip = the JUNCTION pose wrt the FINGERTIP at the grasp
    # (T_fingertip_junction). The pick poses the fingertip so the DETECTED junction lands exactly
    # there before closing: fingertip target = detected junction @ inverse(junction_in_fingertip).
    # Replaces both junction_offset_m and the demo configs' connector_grasp (its inverse).
    if 'junction_offset_m' in entry:
        off = float(entry['junction_offset_m'])
        raise ValueError(
            f'cables.yaml {name!r}: junction_offset_m is replaced by junction_in_fingertip '
            f'(the junction pose wrt the fingertip; the old offset converts to '
            f'junction_in_fingertip: {{xyz: [{-off}, 0.0, 0.0], rpy: [0.0, 0.0, 0.0]}}).')
    if entry.get('junction_in_fingertip'):
        cfg.set_path('junction_in_fingertip', _pose_si(entry['junction_in_fingertip']))
    # A COLOURED TAG on this cable, so the ground-plane scan can pick it out of several by itself.
    # `none` is written out per cable rather than left absent, because "no tag" is a decision worth
    # seeing in the file. NOTE yaml parses bare `none` as the STRING 'none' (only null/~ are None),
    # so it is normalised here -- a config saying "no tag" must never become a colour called "none".
    tag = entry.get('tag_color')
    if isinstance(tag, str) and tag.strip().lower() in ('none', 'null', 'nil', 'off', ''):
        tag = None
    cfg.set_path('cable_tag.color', tag)
    # The HOLDER chain (connector_in_holder / connector_holder_target) is RETIRED: held connectors
    # are specified directly wrt tool0 in configs/frames.yaml (frames: + targets:). Fail loudly so
    # a stale cables.yaml entry cannot silently feed a target nothing reads any more.
    for obsolete in ('connector_in_holder', 'connector_holder_target'):
        if entry.get(obsolete):
            raise ValueError(
                f'cables.yaml {name!r}: {obsolete} is retired -- the held connector is now a '
                f'configs/frames.yaml frame (frames: tool0 -> connector; targets: the recorded '
                f'mate), selected by name (held_frame / assembly.target_frame).')
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
    p.add_argument('--no-prompts', action='store_true',
                   help='ask NOTHING except the cable labelling -- also skips the reset, '
                        'stand-off and success prompts that --yes deliberately keeps')
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
    if getattr(args, 'no_prompts', False):
        cfg.set_path('confirm_each_step', False)
        cfg.set_path('skip_prompts', True)
    if args.debug:
        cfg.set_path('debug', True)
    return cfg
