"""Tool0-attached frames from ONE shared yaml -- configs/frames.yaml.

Every demo config used to declare its own frame sections (fingertip_grasp, hand_eye, ...), and
every script that wanted to display a frame had to list it by name. frames.yaml is the single
source instead: a flat `frames:` mapping of name -> {parent, pose}, pose in the repo-standard
xyz/rpy (m / rad, extrinsic XYZ) or monitor units xyz_mm/rpy_deg (the unit lives in the KEY -- see
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

# The coupler's engagement datum, and the objects catalogue keyed off it.
COUPLER_FRAME = 'coupler_mate'
OBJECTS_FILE = 'objects.yaml'
# objects: entries carry a pose PLUS provenance PLUS the held mass. Same split as the marker
# rigs: the loader validates what it uses and carries the rest through untouched.
OBJECT_META_KEYS = {'mates', 'views', 'approaches', 'residual_mm', 'residual_deg',
                    'residual_axis_deg', 'measured', 'note'}

# frames.yaml name -> the legacy per-config section that still feeds the Robot facade.
# NOTE 'camera' is deliberately absent: the hand-eye calibration has no per-config section any
# more (see hand_eye()), so there is no second value for check_drift to disagree with.
LEGACY_SECTIONS = {'fingertip': 'fingertip_grasp', 'grasp': 'grasp_tcp_offset'}


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


def objects_path(cfg=None):
    """The objects catalogue for this run: a config's `objects_file` (resolved beside that
    config) when set, else the shared configs/objects.yaml."""
    if cfg is not None and cfg.get('objects_file'):
        return resolve(cfg, cfg['objects_file'])
    return os.path.join(CONFIG_DIR, OBJECTS_FILE)


def coupler_mate(cfg=None):
    """tool0 -> the toolchanger coupler's engagement end, from the frames catalogue.

    +z is the mating axis. This is the TOOL-side half of a mate; the object-side half lives in
    objects.yaml, per object, relative to that object's marker. A missing entry is an error and
    not an identity, for the same reason hand_eye() refuses one: from_cfg({}) would silently put
    the mating point AT the flange, 55 mm behind where the parts actually touch, and every pick
    would drive that far too deep."""
    frames = load_frames(cfg)
    if COUPLER_FRAME not in frames:
        raise ValueError(f"no {COUPLER_FRAME!r} frame in {frames_path(cfg)} -- it is the "
                         'coupler\'s engagement end (tool0 -> mating point) and nothing else '
                         'defines it')
    return frames[COUPLER_FRAME]


def load_objects(cfg=None, path=None):
    """{name: object} from the objects catalogue -- what the coupler can pick, and how to find
    each one by sight.

    An object carries ONE OR MORE markers, and each one holds the pose of the object's MATING
    FEATURE in that marker's own frame (marker <- grasp). That direction is the useful one: at
    run time the camera measures T_base_marker, and T_base_marker @ T_marker_grasp is where to
    drive the coupler -- one multiply per marker, no inverse. Storing grasp <- marker would
    invert per pick and would read as if the marker were being located, which is backwards; the
    object is the unknown.

    EVERY MARKER ENCODES THE SAME GRASP FRAME, expressed in its own coordinates. That is what
    lets several of them vote at run time and be averaged, and what lets a pick survive one of
    them being occluded. A marker that has been knocked or re-stuck disagrees with the others
    and is outvoted there rather than quietly dragging the answer.

    Shape (see configs/objects.yaml, which urlab.apps.object_calibration writes):

        objects:
          banana_jig:
            markers:
              31:
                size_mm: 38.80       # REQUIRED -- see ArucoDetector on why
                xyz_mm:  [...]       # the MATING FEATURE, expressed in marker 31's frame
                rpy_deg: [...]
              32:
                size_mm: 38.80
                xyz_mm:  [...]
                rpy_deg: [...]
            held_mass_kg: 0.4        # optional; the payload to set once it is on the coupler

    Returns {name: {'markers': {id: {'size_m', 'T_marker_grasp', 'meta'}},
                    'held_mass_kg': float|None, 'meta': {...}}}.

    Fails LOUDLY on a missing or non-positive marker size, a non-integer id, a missing pose, an
    object with no markers at all and unknown keys -- the same rule as the rest of this file,
    because a typo here puts the coupler somewhere plausible and wrong rather than nowhere."""
    p = path or objects_path(cfg)
    doc = _read(p)
    objects = {}
    for name, entry in (doc.get('objects') or {}).items():
        e = dict(entry or {})
        where = f'{p}: object {name!r}'
        raw_markers = e.pop('markers', None)
        if not raw_markers:
            raise ValueError(f'{where} declares no markers: -- an object is found by sight, so '
                             'it needs at least one')
        markers = {}
        for raw_id, m_entry in raw_markers.items():
            try:
                mid = int(raw_id)
            except (TypeError, ValueError):
                raise ValueError(f'{where} has non-integer marker id {raw_id!r}') from None
            m = dict(m_entry or {})
            m_where = f'{where} marker {mid}'
            sizes = MARKER_SIZE_KEYS & set(m)
            if not sizes:
                raise ValueError(f'{m_where} has no size_mm -- solvePnP scales the marker\'s '
                                 'distance linearly with the side length, so an undeclared '
                                 'size is a silent depth error, not a missing default')
            if len(sizes) > 1:
                raise ValueError(f'{m_where} sets both size_mm and size_m; use one unit')
            size_m = float(m.pop('size_m')) if 'size_m' in m else float(m.pop('size_mm')) / 1000.0
            if not size_m > 0.0:
                raise ValueError(f'{m_where} has a non-positive size')
            meta = {k: m.pop(k) for k in list(m) if k in OBJECT_META_KEYS}
            markers[mid] = {'size_m': size_m, 'T_marker_grasp': _pose(m, m_where), 'meta': meta}

        mass = e.pop('held_mass_kg', None)
        if mass is not None:
            mass = float(mass)
            if not mass >= 0.0:
                raise ValueError(f'{where} has a negative held_mass_kg')

        # ASSEMBLIES: recorded base_link poses of the object's mating frame at an assembled
        # position -- i.e. where `coupler_mate` has to end up for the part to be seated in its
        # fixture. KINEMATIC, so exactly as good as the cell staying put: unbolt the fixture and
        # they are wrong, with nothing to notice it. That is the trade a taught pose makes, and
        # the alternative (a marker rig on the fixture) is what frames.yaml's marker_rigs are
        # for. One object may have several, keyed by name.
        assemblies = {}
        for a_name, a_entry in dict(e.pop('assemblies', None) or {}).items():
            a = dict(a_entry or {})
            a_where = f'{where} assembly {a_name!r}'
            a_meta = {k: a.pop(k) for k in list(a) if k in OBJECT_META_KEYS}
            assemblies[str(a_name)] = {'T_base_assembly': _pose(a, a_where), 'meta': a_meta}

        meta = {k: e.pop(k) for k in list(e) if k in OBJECT_META_KEYS}
        if e:
            raise ValueError(f'{where} has unknown key(s) {sorted(e)}')
        objects[name] = {'markers': markers, 'held_mass_kg': mass, 'assemblies': assemblies,
                         'meta': meta}
    return objects


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

def hand_eye(cfg=None):
    """tool0 -> camera, from the frames catalogue's `camera` entry. THE ONLY SOURCE.

    Configs used to be able to declare their own `hand_eye:` block, and several did, so the one
    number that every detection depends on was spelled four different ways across configs/ and
    silently differed by 9 mm between them. There is no override now: the calibration is measured
    geometry, it belongs with the other measured geometry, and a per-config copy is a way for the
    cell to be wrong without anyone noticing.

    A MISSING `camera` entry is an error, not an identity: from_cfg({}) would quietly put the
    camera at the flange and every detection would be off by the length of the bracket."""
    frames = load_frames(cfg)
    if 'camera' not in frames:
        raise ValueError(f"no 'camera' frame in {frames_path(cfg)} -- it is the hand-eye "
                         'calibration (tool0 -> camera) and nothing else defines it')
    return frames['camera']


def aruco_defaults(cfg=None):
    """The `aruco:`-shaped dict derived from the frames catalogue's marker_rigs.

    The rigs already record the CALIBRATION -- dictionary and per-marker printed size -- so a
    config does not have to repeat them (and drift, which is worse). Returns
    {'dictionary': ..., 'marker_sizes_m': {id: size}, 'marker_size_m': <the common size>};
    empty dict if the catalogue has no rigs."""
    rigs = (_read(frames_path(cfg)).get('marker_rigs') or {})
    dictionaries, sizes = set(), {}
    for rig in rigs.values():
        if rig.get('dictionary'):
            dictionaries.add(str(rig['dictionary']))
        for mid, m in (rig.get('markers') or {}).items():
            if isinstance(m, dict) and m.get('size_mm') is not None:
                sizes[int(mid)] = float(m['size_mm']) / 1000.0
    if not dictionaries and not sizes:
        return {}
    if len(dictionaries) > 1:
        raise ValueError(f'marker_rigs disagree on the dictionary: {sorted(dictionaries)} -- '
                         f'set aruco.dictionary in the config to pick one')
    out = {'marker_sizes_m': sizes}
    if dictionaries:
        out['dictionary'] = next(iter(dictionaries))
    if sizes and len(set(sizes.values())) == 1:
        out['marker_size_m'] = next(iter(sizes.values()))
    return out
