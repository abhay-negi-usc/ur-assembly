"""Tool0-attached frames from ONE shared yaml -- configs/frames.yaml.

Every demo config declares its own frame sections (fingertip_grasp, hand_eye, ...), and every
script that wanted to display a frame had to list it by name. frames.yaml is the single source
instead: a flat `frames:` mapping of name -> {parent, pose}, pose in the repo-standard xyz/rpy
(m / rad, extrinsic XYZ) or monitor units xyz_mm/rpy_deg (the unit lives in the KEY -- see
config._pose_si). `parent` chains frames (default tool0); the loader flattens every chain to
tool0 and fails LOUDLY on unknown parents, cycles, mixed-unit pose blocks, unknown pose keys
(a typo like 'xyz_m' would otherwise silently place the frame at its parent), and entries with
no pose at all -- a frames typo should stop a run at startup, never silently misplace a frame.

Adding a frame is one yaml entry, no code:

    banana_connector_finger_holder:
      xyz_mm:  [0.0, 0.0, 159.0]
      rpy_deg: [180.0, 0.0, -90.0]

Generic consumers (urlab.apps.monitor) iterate load_frames() and show whatever the file holds.

The yaml's `targets:` section records BASE_LINK poses of catalogued frames (base_link <- frame,
e.g. an assembly mate read off the monitor after hand-guiding); load_targets() serves them. A
consumer that names a frame (uncertain_sampling's `held_frame:`) pulls the tool0-attached frame
AND its recorded target from this one file.

(Not to be confused with urlab.frames.FrameGraph -- the LIVE transform tree between base_link,
tool0, camera, and observed frames. This module is the static catalogue of frames bolted to the
tool; a FrameGraph consumer can register these as static edges under tool0.)

The Robot facade still reads the legacy per-config sections; check_drift() warns when a loaded
config's sections disagree with frames.yaml, so the duplication cannot rot silently while the
facade migrates.
"""

import os

import numpy as np

from . import log as urlog
from .config import CONFIG_DIR, _pose_si, resolve
from .transforms import from_cfg, inverse, matrix_to_xyzrpy

log = urlog.get('tool-frames')

DEFAULT_PATH = os.path.join(CONFIG_DIR, 'frames.yaml')
ROOT = 'tool0'
POSE_KEYS = {'xyz', 'rpy', 'xyz_mm', 'rpy_deg'}
# marker_rigs: entries carry a pose PLUS provenance. Sizes are load-bearing (solvePnP scales the
# translation linearly with the side length it is told); the rest is for the operator.
MARKER_SIZE_KEYS = {'size_mm', 'size_m'}
MARKER_META_KEYS = {'views', 'residual_mm', 'residual_deg', 'measured', 'note'}

# frames.yaml name -> the legacy per-config section that still feeds the Robot facade.
LEGACY_SECTIONS = {'fingertip': 'fingertip_grasp', 'camera': 'hand_eye',
                   'grasp': 'grasp_tcp_offset'}


def frames_path(cfg=None):
    """The frames yaml for this run: a config's `frames_file` (resolved beside that config) when
    set, else the shared configs/frames.yaml."""
    if cfg is not None and cfg.get('frames_file'):
        return resolve(cfg, cfg['frames_file'])
    return DEFAULT_PATH


def _read(p):
    import yaml

    if not os.path.isfile(p):
        raise FileNotFoundError(f'frames yaml not found: {p}')
    with open(p, 'r') as fh:
        return yaml.safe_load(fh) or {}


def _pose(e, where):
    """Validated pose block -> 4x4. from_cfg IGNORES unknown keys and zero-fills missing ones,
    so a typo ('xyz_m', 'rpy_de') would silently collapse the pose to identity -- reject
    anything unexpected, and require an explicit pose (write zeros for an intentional
    identity)."""
    unknown = set(e) - POSE_KEYS
    if unknown:
        raise ValueError(f'{where} has unknown key(s) {sorted(unknown)} -- '
                         f'allowed: {sorted(POSE_KEYS)}')
    if not e:
        raise ValueError(f'{where} has no pose keys -- write explicit zeros for an '
                         'intentional identity')
    return from_cfg(_pose_si(e))


def load_frames(cfg=None, path=None):
    """{name: T_tool0_frame} for EVERY frame in the frames yaml, parent chains flattened;
    includes 'tool0' itself (identity)."""
    p = path or frames_path(cfg)
    raw = _read(p).get('frames') or {}
    if ROOT in raw:
        raise ValueError(f'{p}: {ROOT!r} is the root frame -- it cannot be (re)defined')

    local, parent = {}, {}
    for name, entry in raw.items():
        e = dict(entry or {})
        parent[name] = str(e.pop('parent', ROOT))
        local[name] = _pose(e, f'{p}: frame {name!r}')

    frames = {ROOT: np.eye(4)}

    def to_tool0(name, trail):
        if name in frames:
            return frames[name]
        if name not in local:
            raise ValueError(f'{p}: frame {trail[-2]!r} has unknown parent {name!r}')
        if name in trail[:-1]:
            raise ValueError(f'{p}: parent cycle: {" -> ".join(trail)}')
        frames[name] = to_tool0(parent[name], trail + (parent[name],)) @ local[name]
        return frames[name]

    for name in local:
        to_tool0(name, (name,))
    return frames


def load_targets(cfg=None, path=None):
    """{name: T_base_frame} from the frames yaml `targets:` section -- the RECORDED base_link
    pose of a catalogued frame (base_link <- frame), e.g. an assembly mate.

    MEASURE: hand-guide to the mate, read the `base_link <- <frame>` line off urlab.apps.monitor,
    paste it under targets: (monitor units). Every target name must have a matching `frames:`
    entry -- a target for an undeclared frame is a typo, not a definition."""
    p = path or frames_path(cfg)
    doc = _read(p)
    declared = set(doc.get('frames') or {}) | {ROOT}
    targets = {}
    for name, entry in (doc.get('targets') or {}).items():
        if name not in declared:
            raise ValueError(f'{p}: target {name!r} has no matching frames: entry')
        targets[name] = _pose(dict(entry or {}), f'{p}: target {name!r}')
    return targets


def load_marker_rigs(cfg=None, path=None):
    """{target_name: rig} from the frames yaml `marker_rigs:` section.

    A RIG is the fiducials bolted around one fixture, each carrying the TARGET's pose in ITS OWN
    frame (marker <- target). That direction is the useful one: at run time the camera measures
    T_base_marker, and T_base_marker @ T_marker_target is the target, so every marker in view is
    an independent vote on where the fixture is and they can simply be averaged. Storing
    target <- marker instead would need an inverse per marker per run and would read as if the
    markers were being located, which is backwards -- the fixture is the unknown.

    Shape:

        marker_rigs:
          bnc_connector_in_fingerpads:      # a frames: name -- what the rig locates
            dictionary: DICT_4X4_50         # optional; absent = the app's aruco.dictionary
            markers:
              7:
                size_mm: 20.3               # REQUIRED -- see ArucoDetector on why
                xyz_mm:  [...]              # the target, expressed in marker 7's frame
                rpy_deg: [...]

    Returns {name: {'dictionary': str|None,
                    'markers': {id: {'size_m': float, 'T_marker_target': 4x4, 'meta': {...}}}}}.

    Fails LOUDLY on a rig for an undeclared frame, a non-integer id, a missing or non-positive
    size, and unknown keys -- the same rule the rest of this file follows, because a marker typo
    would otherwise put the whole fixture somewhere plausible and wrong."""
    p = path or frames_path(cfg)
    doc = _read(p)
    declared = set(doc.get('frames') or {}) | {ROOT}
    rigs = {}
    for name, entry in (doc.get('marker_rigs') or {}).items():
        if name not in declared:
            raise ValueError(f'{p}: marker_rigs {name!r} has no matching frames: entry')
        e = dict(entry or {})
        dictionary = e.pop('dictionary', None)
        raw_markers = e.pop('markers', None)
        if e:
            raise ValueError(f'{p}: marker_rigs {name!r} has unknown key(s) {sorted(e)} -- '
                             "allowed: ['dictionary', 'markers']")
        if not raw_markers:
            raise ValueError(f'{p}: marker_rigs {name!r} declares no markers')
        markers = {}
        for raw_id, m_entry in raw_markers.items():
            try:
                mid = int(raw_id)
            except (TypeError, ValueError):
                raise ValueError(f'{p}: marker_rigs {name!r} has non-integer marker id '
                                 f'{raw_id!r}') from None
            m = dict(m_entry or {})
            where = f'{p}: marker_rigs {name!r} marker {mid}'
            sizes = MARKER_SIZE_KEYS & set(m)
            if not sizes:
                raise ValueError(f'{where} has no size_mm -- solvePnP scales the marker\'s '
                                 'distance linearly with the side length, so an undeclared size '
                                 'is a silent depth error, not a missing default')
            if len(sizes) > 1:
                raise ValueError(f'{where} sets both size_mm and size_m; use one unit')
            size_m = float(m.pop('size_m')) if 'size_m' in m else float(m.pop('size_mm')) / 1000.0
            if not size_m > 0.0:
                raise ValueError(f'{where} has a non-positive size')
            meta = {k: m.pop(k) for k in list(m) if k in MARKER_META_KEYS}
            markers[mid] = {'size_m': size_m, 'T_marker_target': _pose(m, where), 'meta': meta}
        rigs[name] = {'dictionary': dictionary, 'markers': markers}
    return rigs


def marker_sizes(rig):
    """{marker_id: size_m} for a rig -- what ArucoDetector(sizes_m=) wants."""
    return {mid: m['size_m'] for mid, m in rig['markers'].items()}


def resolve_held_and_target(frames, targets, held_name, target_name=None, path=None):
    """(T_tool0_held, T_base_target, target_name) for an app that drives a HELD part at a
    RECORDED mate.

    TWO LOOKUPS, TWO SECTIONS. `held_name` is a `frames:` entry -- where the part sits w.r.t.
    tool0, i.e. what the arm is carrying. `target_name` is a `targets:` entry -- the recorded
    base_link pose that part is driven to. `target_name=None` means "the same name", which is the
    usual case and the behaviour before the key existed.

    They are separate because they answer different questions: one held part can be probed against
    several recorded mates (a second socket, a re-measured one), and the same mate can be
    approached with a different frame declared on the part. Requiring one name to satisfy both
    sections forced a new catalogue entry for either.

    NOTE the two names should still describe the SAME POINT on the part. The apps report the held
    frame's pose w.r.t. the target frame, so naming different points offsets every number they
    print by exactly the difference between them.

    Raises ValueError carrying the operator-facing reason. Pure, so the rule is testable without a
    robot or a catalogue file."""
    tgt = target_name or held_name
    where = f' in {path}' if path else ''
    if not held_name:
        raise ValueError('held_frame is required.')
    if held_name not in frames:
        raise ValueError(f'held_frame {held_name!r} needs a frames: entry{where}.')
    if tgt not in targets:
        raise ValueError(
            f'target_frame {tgt!r} needs a targets: entry{where}'
            + ('' if target_name else
               f' (it defaults to held_frame, so either add a targets: entry for {held_name!r} '
               f'or set target_frame to a name that has one)') + '.')
    return frames[held_name], targets[tgt], tgt


def check_drift(frames, cfg, tol_mm=0.5, tol_deg=0.2):
    """Warn for every frame where frames.yaml and the loaded config's LEGACY section disagree.

    The Robot facade still reads the sections, so silent drift would mean the monitor displays
    one pose while the robot commands another. Returns the list of drifted frame names."""
    drifted = []
    for name, section in LEGACY_SECTIONS.items():
        blk = cfg.section(section) if cfg is not None else {}
        if name not in frames or not blk:
            continue
        d = inverse(from_cfg(_pose_si(dict(blk)))) @ frames[name]
        xyz, rpy = matrix_to_xyzrpy(d)
        d_mm = float(np.linalg.norm(xyz)) * 1000.0
        d_deg = float(np.degrees(np.max(np.abs(rpy))))
        if d_mm > tol_mm or d_deg > tol_deg:
            drifted.append(name)
            log.warning('frames.yaml %r differs from config section %r by %.2f mm / %.2f deg -- '
                        'the Robot facade uses the SECTION; align them.', name, section, d_mm, d_deg)
    return drifted
