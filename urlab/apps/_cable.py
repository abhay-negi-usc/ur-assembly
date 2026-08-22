"""Shared wiring for the cable demos -- assemble the scan pipeline from a config.

The three cable apps (pick-place, touch, pick-assemble) all start the same way: a camera, a SAM3
detector (neck or tip), a connector estimator, and a CableScanner over them. This builds that
stack so each app is just the sequence that follows the scan.
"""

from ..perception import CableReconstructor, ConnectorEstimator
from ..perception.sam3 import make_detector
from ..skills.scan import CableScanner, ScanConfig


def build_scanner(cfg, robot, camera):
    """(scanner, detector, estimator) wired from the config's scan/sam3/connector_estimator blocks.

    In scan.mode 'reconstruction' the scanner also gets a CableReconstructor, which refines the
    junction pose from the reconstructed 3D cable centreline. That needs the cable SKELETON, which
    only the junction detector (sam3.mode: junction) exposes -- validated here."""
    detector = make_detector(cfg)
    estimator = ConnectorEstimator(cfg)
    scan_cfg = ScanConfig(cfg)

    # Ground-plane manual-select mode: single image, user picks the cable, pose from a known ground
    # plane (no multi-view fit). Drop-in scanner with the same scan()/reset()/.camera/.estimator API.
    if scan_cfg.mode == 'ground_plane':
        from ..skills.ground_pick import GroundPlaneScanner
        scanner = GroundPlaneScanner(robot, camera, detector, estimator, cfg,
                                     data_root=cfg.get('data_dir', 'data'))
        return scanner, detector, estimator

    reconstructor = None
    if scan_cfg.mode == 'reconstruction':
        if not hasattr(detector, 'detect_cable'):
            raise ValueError(
                "scan.mode 'reconstruction' needs the cable skeleton, which only the junction "
                "detector provides -- set sam3.mode: junction (got %r)."
                % cfg.get_path('sam3.mode', 'junction'))
        reconstructor = CableReconstructor(cfg)

    # Two-phase 'cable-end' scan (opt-in): a second estimator triangulates the connector ENDPOINT
    # that Phase 1 steers on. Needs the junction detector (it exposes the trace via detect_both).
    end_estimator = None
    if scan_cfg.cable_end_enabled:
        if not hasattr(detector, 'detect_both'):
            raise ValueError(
                "cable_end.enabled needs the junction detector's endpoint -- set sam3.mode: junction "
                "(got %r)." % cfg.get_path('sam3.mode', 'junction'))
        end_estimator = ConnectorEstimator(cfg)

    scanner = CableScanner(robot, camera, detector, estimator, scan_cfg,
                           data_root=cfg.get('data_dir', 'data'), reconstructor=reconstructor,
                           end_estimator=end_estimator)
    return scanner, detector, estimator


def make_confirm(cfg):
    """A confirm(label) callback, or None if confirmation is disabled."""
    from ._common import prompts_off
    if cfg.get('confirm_each_step', True) is False or prompts_off(cfg):
        return None

    def confirm(label):
        try:
            return input(f'[{label}] Enter to proceed (q to abort): ').strip().lower() != 'q'
        except EOFError:
            return True
    return confirm
