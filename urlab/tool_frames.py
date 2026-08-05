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

# frames.yaml name -> the legacy per-config section that still feeds the Robot facade.
LEGACY_SECTIONS = {'fingertip': 'fingertip_grasp', 'camera': 'hand_eye',
                   'grasp': 'grasp_tcp_offset', 'connector_holder': 'connector_holder'}


def frames_path(cfg=None):
    """The frames yaml for this run: a config's `frames_file` (resolved beside that config) when
    set, else the shared configs/frames.yaml."""
    if cfg is not None and cfg.get('frames_file'):
        return resolve(cfg, cfg['frames_file'])
    return DEFAULT_PATH


def load_frames(cfg=None, path=None):
    """{name: T_tool0_frame} for EVERY frame in the frames yaml, parent chains flattened;
    includes 'tool0' itself (identity)."""
    import yaml

    p = path or frames_path(cfg)
    if not os.path.isfile(p):
        raise FileNotFoundError(f'frames yaml not found: {p}')
    with open(p, 'r') as fh:
        raw = (yaml.safe_load(fh) or {}).get('frames') or {}
    if ROOT in raw:
        raise ValueError(f'{p}: {ROOT!r} is the root frame -- it cannot be (re)defined')

    local, parent = {}, {}
    for name, entry in raw.items():
        e = dict(entry or {})
        parent[name] = str(e.pop('parent', ROOT))
        # from_cfg IGNORES unknown keys and zero-fills missing ones, so a typo ('xyz_m',
        # 'rpy_de') would silently place the frame AT ITS PARENT. Reject anything unexpected,
        # and require an explicit pose -- an intentional identity is written as zeros.
        unknown = set(e) - POSE_KEYS
        if unknown:
            raise ValueError(f'{p}: frame {name!r} has unknown key(s) {sorted(unknown)} -- '
                             f"allowed: 'parent' plus {sorted(POSE_KEYS)}")
        if not e:
            raise ValueError(f'{p}: frame {name!r} has no pose keys -- write explicit zeros '
                             'for an intentional identity')
        local[name] = from_cfg(_pose_si(e))

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
