"""Config loading -- replaces ROS parameters.

The old stack had two overlapping mechanisms: a yaml file loaded by the node, plus `--ros-args
-p name:=value` overrides for a handful of hand-picked keys that someone had remembered to
declare. Here there is one: the yaml, with `--set key=value` able to reach ANY key, nested ones
included:

    python -m urlab.apps.cable_pick_place --set scan.approach.min_distance_mm=150 --set debug=true

Values parse as YAML, so `true`, `0.15`, `[1, 0, 0]` and `null` all mean what they look like.

UNITS: every yaml in configs/ is written in **mm and deg**, with the unit IN THE KEY NAME
(`standoff_mm`, `sweep_deg`, `max_speed_mm_s`, `xyz_mm`, `rpy_deg`). The code works in m and rad,
so `load()` converts once, at the end, adding the SI-named sibling of every such key
(`standoff_mm` -> `standoff_m`, `rpy_deg` -> `rpy`). Both spellings are readable afterwards, which
is why `--set` accepts either:

    python -m urlab.apps.cable_pick_place --set scan.approach.min_distance_mm=150

See `_normalise_units` for the exact rules -- including the three shapes that keep a unit in the
name WITHOUT being a quantity to scale (a flag, a per-joint mapping, a sweep spec).
"""

import argparse
import math
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
    forced = _apply_overrides(cfg, overrides)  # 2nd pass: an explicit `--set` beats the profile
    _normalise_units(cfg, forced=forced)  # mm/deg in the file -> m/rad siblings for the code
    return cfg


# ---------------------------------------------------------------------------- units
# CONFIGS ARE WRITTEN IN mm AND deg. The code is written in m and rad, because that is what the
# maths and the RTDE interface use. Rather than convert at ~250 read sites -- where one missed
# division silently turns a 50 mm standoff into 50 m -- the conversion happens ONCE, here, at load
# time: every `<name>_mm` key gains a `<name>_m` sibling holding the same value in metres, and
# likewise `_deg` -> `_rad`, `_mm_s` -> `_m_s`, `xyz_mm` -> `xyz`, `rpy_deg` -> `rpy`.
#
# ADDITIVE, NEVER DESTRUCTIVE: the mm/deg key stays exactly as written, so the many readers that
# already read `_mm` and divide by 1000 themselves are untouched. A reader asking for either unit
# gets the right number, which is what lets the configs be converted without touching the code.
#
# WRITING BOTH SPELLINGS IS AN ERROR, not a silent preference -- the same rule _pose_si has always
# enforced for poses, and for the same reason: quietly choosing one turns a 90 mm offset into 90 m.
_UNIT_SUFFIXES = (
    ('_mm_s2', '_m_s2', 1e-3), ('_deg_s2', '_rad_s2', math.pi / 180.0),
    ('_mm_s', '_m_s', 1e-3), ('_deg_s', '_rad_s', math.pi / 180.0),
    ('_mm', '_m', 1e-3), ('_deg', '_rad', math.pi / 180.0),
)
# Keys that END in a unit suffix but are NOT that quantity. `per_m` is an INVERSE length (1/m):
# scaling it as if it were a length would be exactly backwards.
_UNIT_EXEMPT = ('_per_m', '_per_mm', '_per_deg', '_per_rad')
# The pose triples are handled by their own rule below, whose SI names carry no suffix (`xyz`, not
# `xyz_m`) -- letting the generic rule near them would mint a second, unread spelling.
_POSE_KEYS = ('xyz_mm', 'rpy_deg')

# PER-AXIS MAPS: `dim_weights: {x_mm: 1.0, ..., yaw_deg: 1.0}`, wiggle amplitudes, search ranges.
# Here the suffix names the AXIS, not the value -- an x weight of 1.0 is not "1 mm", and the code
# validates these key sets against its own DIMS vocabulary (urlab/skills/manifold.py,
# urlab/skills/wiggle.py), so injecting `x_m` beside `x_mm` both means nothing and CRASHES the
# validator with an unknown axis.
#
# The signature is a map whose keys are ALL axis labels AND which spans lengths and angles
# together -- the shape of a 6-DOF map, and the thing no scalar quantity block looks like.
# Deliberately not "any key is an axis label": `ground_plane: {z_mm: ...}` is a plain depth whose
# `z_m` sibling IS read, and it must keep getting one even if it were the block's only key.
_AXIS_LENGTHS = ('x_mm', 'y_mm', 'z_mm')
_AXIS_ANGLES = ('roll_deg', 'pitch_deg', 'yaw_deg')
_AXIS_LABELS = frozenset(_AXIS_LENGTHS + _AXIS_ANGLES)


def _is_axis_map(node):
    keys = set(node)
    if not keys or not keys <= _AXIS_LABELS:
        return False
    return bool(keys & set(_AXIS_LENGTHS)) and bool(keys & set(_AXIS_ANGLES))


class _DerivedFloat(float):
    """A float THIS MODULE computed from a mm/deg key, not one a human wrote in the yaml.

    The marker rides on the value's TYPE rather than an extra dict key, so it is invisible to
    everything downstream -- it is a float to arithmetic, numpy, json and `==` alike -- while still
    letting the both-units check tell "the sibling we derived" from "written twice by hand"."""


class _DerivedList(list):
    """See _DerivedFloat -- the list flavour, for xyz/rpy triples."""


def _is_derived(value):
    return isinstance(value, (_DerivedFloat, _DerivedList))


def _convert_units(value, scale, where=''):
    """Scale a scalar or a flat list of scalars.

    A key whose NAME declares a unit but whose value is not a number is an error, not something to
    pass through quietly. That silence is the dangerous case: `1e+02` is a STRING to YAML 1.1 (it
    wants `1.0e+02`), so a mistyped number would keep its mm spelling, never gain its SI sibling,
    and the reader would fall back to a default -- a 100 mm standoff becoming whatever the default
    said, with nothing logged. Raising turns that into a startup failure naming the key."""
    if value is None or isinstance(value, (bool, dict)):
        # NOT every key ending `_deg`/`_mm` is a quantity to scale:
        #   trajectory_angles_deg: false        -- a FLAG naming the unit of a CSV's columns
        #   joint_limits_deg: {wrist_3: [...]}  -- a MAPPING, read in degrees as written
        #   depth_mm: {lower:, upper:, ...}     -- a sweep SPEC, read in mm as written
        # Nothing reads an SI sibling of these, so pass them through untouched.
        return value
    if isinstance(value, (int, float)):
        return _DerivedFloat(float(value) * scale)
    if isinstance(value, (list, tuple)):
        bad = [v for v in value if isinstance(v, bool) or not isinstance(v, (int, float))]
        if bad:
            raise ValueError(f'{where or "value"} declares a unit in its name but contains '
                             f'non-numeric entries {bad!r} (a YAML number needs a decimal point '
                             f'in exponent form: 1.0e+02, not 1e+02)')
        return _DerivedList(float(v) * scale for v in value)
    raise ValueError(f'{where or "value"} declares a unit in its name but is not a number: '
                     f'{value!r} (a YAML number needs a decimal point in exponent form: '
                     f'1.0e+02, not 1e+02)')


def _normalise_units(node, _path='', forced=()):
    """Walk the config tree and add the SI sibling of every mm/deg key, in place.

    `forced` holds paths an explicit `--set` wrote. `--set linear_step_m=0.05` against a file that
    says `linear_step_mm: 30.0` is not the two-spellings mistake -- it is a deliberate override, so
    it WINS and the mm sibling is rewritten to match rather than the load being refused."""
    if isinstance(node, dict):
        if _is_axis_map(node):
            return node                       # per-axis map: the suffix names the AXIS, not a value
        for key in list(node):
            _normalise_units(node[key], f'{_path}.{key}' if _path else str(key), forced)
        for key in list(node):
            if not isinstance(key, str) or key.endswith(_UNIT_EXEMPT) or key in _POSE_KEYS:
                continue
            for suffix, si_suffix, scale in _UNIT_SUFFIXES:
                if not key.endswith(suffix):
                    continue
                si = key[:-len(suffix)] + si_suffix
                si_path = f'{_path}.{si}' if _path else si
                if si in node and not _is_derived(node[si]):
                    if si_path in forced:                 # an explicit --set in SI units wins
                        back = _convert_units(node[si], 1.0 / scale, si_path)
                        node[key] = back                  # keep the mm spelling consistent with it
                        break
                    raise ValueError(
                        f'{_path or "<root>"}: both {key!r}={node[key]!r} and {si!r}={node[si]!r} '
                        f'are set. They are the same quantity in different units -- silently '
                        f'preferring one would turn a 90 mm value into 90 m. Keep the mm/deg '
                        f'spelling and delete the other (or pass it as --set {si_path}=... to '
                        f'override deliberately).')
                converted = _convert_units(node[key], scale, f'{_path}.{key}' if _path else key)
                if converted is not node[key]:
                    node[si] = converted
                break
        # The pose triples, whose SI names carry no suffix at all.
        for alt, si, scale in (('xyz_mm', 'xyz', 1e-3), ('rpy_deg', 'rpy', math.pi / 180.0)):
            if alt not in node:
                continue
            if si in node and not _is_derived(node[si]):
                if (f'{_path}.{si}' if _path else si) in forced:
                    node[alt] = _convert_units(node[si], 1.0 / scale)
                    continue
                raise ValueError(
                    f'{_path or "<root>"}: pose sets both {si!r}={node[si]!r} and '
                    f'{alt!r}={node[alt]!r}. Use one unit, not both.')
            node[si] = _convert_units(node[alt], scale, f'{_path}.{alt}' if _path else alt)
    elif isinstance(node, list):
        for item in node:
            _normalise_units(item, _path)
    return node


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
    """Apply `key=value` overrides; returns the set of paths touched, so unit normalisation can
    tell an EXPLICIT `--set linear_step_m=0.05` (which must win) from a file that carelessly wrote
    the same quantity twice (which must raise)."""
    touched = set()
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'--set expects key=value, got {item!r}')
        key, _, raw = item.partition('=')
        cfg.set_path(key.strip(), yaml.safe_load(raw))
        touched.add(key.strip())
    return touched


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
            # A pose from config.load() ALREADY carries the SI sibling _normalise_units derived,
            # so its mere presence is not a conflict; a HAND-WRITTEN second unit still is.
            if si in p and not _is_derived(p[si]):
                raise ValueError(f'pose block sets both {si!r} and {alt!r}; use one unit, not both')
            p[si] = _DerivedList(float(v) * scale for v in p.pop(alt))
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
