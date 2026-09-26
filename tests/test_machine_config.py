"""The machine files: robot.yaml / camera.yaml, referenced EXPLICITLY from every config.

    robot_config: robot.yaml          # the arm/host: robot:, speed:, base/tip frames, compute:
    camera_config: camera.yaml        # the sensor: camera:, camera_frame

Whole-block fill of anything the config does not define; a config that defines a block owns it
wholesale; `--set` beats everything. The camera-MOUNT calibration (hand-eye) and the marker-rig
calibration (aruco) come from configs/frames.yaml ONLY -- no config block overrides either.
"""

import glob
import os
import tempfile

import numpy as np
import pytest

from urlab import config as C
from urlab import tool_frames
from urlab.perception.sam3_backend import resolve_device

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')


# The machine files themselves, plus the CATALOGUES -- frames.yaml, cables.yaml, objects.yaml.
# A catalogue is measured data read straight off disk by its own loader (tool_frames._read,
# apply_cable_profile); it never goes through config.load, so it has no machine to declare and a
# `robot_config:` line in one is inert. objects.yaml is MACHINE-WRITTEN besides, so it could not
# carry one even if it wanted to.
NOT_DEMO_CONFIGS = {'robot.yaml', 'camera.yaml', 'frames.yaml', 'cables.yaml', 'objects.yaml'}


def test_every_config_declares_its_machine():
    """Explicit reference, not directory magic: each demo config names robot.yaml/camera.yaml."""
    for path in sorted(glob.glob(os.path.join(CONFIG_DIR, '*.yaml'))):
        base = os.path.basename(path)
        if base in NOT_DEMO_CONFIGS:
            continue
        cfg = C.load(path)
        assert cfg.get('robot_config'), f'{base} does not declare robot_config'
        assert cfg.get_path('robot.ip'), f'{base} resolved no robot.ip'
        assert cfg.get_path('speed.max_joint_velocity_deg_s') is not None, base
        assert cfg.get_path('compute.device') == 'auto', base
        assert cfg.get_path('camera.serial_no'), f'{base} resolved no camera'


def test_a_config_block_owns_the_machine_block_wholesale():
    """A config CAN still override the machine -- whole-block, same rule as ever."""
    cfg = C.load('bnc_assembly', ['robot.ip=10.0.0.1'])
    assert cfg.get_path('robot.ip') == '10.0.0.1', '--set must beat the machine file'


def test_a_dangling_machine_reference_is_loud():
    """robot_config naming a missing file must fail at load, not resolve to no robot block."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'demo.yaml')
        with open(p, 'w') as fh:
            fh.write('robot_config: nope.yaml\nx: 1\n')
        with pytest.raises(FileNotFoundError, match='nope.yaml'):
            C.load(p)


def test_hand_eye_comes_from_the_frames_catalogue_for_every_config():
    """ONE calibration, no per-config blocks and no override path -- every config that images
    resolves the SAME tool0 -> camera, because they all read frames.yaml's `camera` entry.

    NOTHING IS PINNED TO A LITERAL HERE, deliberately. The hand-eye is a MEASUREMENT and moves
    whenever the camera is recalibrated or remounted; a test that spelled the number out would
    fail on every recalibration and would teach whoever is holding the calipers to edit the
    expectation until it passes. What has to hold is AGREEMENT -- that the catalogue is the only
    source and everything reads it -- and that is what is asserted. (A hardcoded [-9, -80, 31]
    in tests/test_behaviors.py went on passing for exactly this reason long after nothing used
    that value.)"""
    catalogue = tool_frames.load_frames(C.load('bnc_assembly'))['camera']
    for name in ('bnc_assembly', 'pick_place', 'visual_servo', 'cable_pick_place',
                 'cable_pick_assemble', 'marker_calibration'):
        assert np.allclose(tool_frames.hand_eye(C.load(name)), catalogue, atol=1e-12), (
            f'{name} resolved a hand-eye that is not the catalogue entry')


def test_a_config_cannot_override_the_hand_eye():
    """The override was how the 9 mm split between cells happened. A stray `hand_eye:` block is
    now INERT -- the catalogue wins -- so a leftover copy cannot quietly re-open the split."""
    cfg = C.load('pick_place')
    catalogue = tool_frames.load_frames(cfg)['camera']
    cfg['hand_eye'] = {'xyz_mm': [99.0, 99.0, 99.0], 'rpy_deg': [0.0, 0.0, 0.0]}
    assert np.allclose(tool_frames.hand_eye(cfg), catalogue, atol=1e-12)


def test_no_config_still_declares_a_hand_eye_block():
    """The blocks are gone from configs/; this fails if one is reintroduced by a copy-paste."""
    import yaml
    offenders = []
    for path in sorted(glob.glob(os.path.join(CONFIG_DIR, '*.yaml'))):
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        if isinstance(doc, dict) and 'hand_eye' in doc:
            offenders.append(os.path.basename(path))
    assert not offenders, f'hand_eye: belongs only in frames.yaml, found in {offenders}' 


def test_hand_eye_missing_from_the_catalogue_is_an_error_not_identity():
    """from_cfg({}) would put the camera AT the flange and every detection off by the bracket.
    With no config-level block left, the catalogue is the only thing that can supply it."""
    with tempfile.TemporaryDirectory() as tmp:
        frames = os.path.join(tmp, 'frames.yaml')
        with open(frames, 'w') as fh:
            fh.write('frames: {}\n')
        cfg = C.Config({'frames_file': frames, '_config_dir': tmp,
                        '_config_path': os.path.join(tmp, 'x.yaml')})
        with pytest.raises(ValueError, match='hand-eye'):
            tool_frames.hand_eye(cfg)


def test_aruco_defaults_derive_from_the_marker_rigs():
    """Dictionary + per-marker printed sizes come from the rig CALIBRATION in frames.yaml."""
    a = tool_frames.aruco_defaults(C.load('bnc_assembly'))
    assert a['dictionary'] == 'DICT_4X4_50'
    assert a['marker_sizes_m'][24] == pytest.approx(0.0388)
    assert a['marker_sizes_m'][25] == pytest.approx(0.0388)


# ---------------------------------------------------------------- compute.device
def _cuda():
    import torch
    return torch.cuda.is_available()


def test_compute_device_resolution_rules():
    dev = resolve_device('auto')
    assert dev == ('cuda' if _cuda() else 'cpu')
    assert resolve_device(None) == dev, 'unset behaves as auto'
    assert resolve_device('cpu') == 'cpu', 'cpu always wins (GPU may be owned by other software)'
    if _cuda():
        assert resolve_device('cuda') == 'cuda'
    else:
        with pytest.raises(RuntimeError, match='compute.device'):
            resolve_device('cuda')
    with pytest.raises(ValueError, match="'auto', 'cpu' or 'cuda'"):
        resolve_device('gpu')
