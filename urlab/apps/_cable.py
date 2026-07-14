"""Shared wiring for the cable demos -- assemble the scan pipeline from a config.

The three cable apps (pick-place, touch, pick-assemble) all start the same way: a camera, a SAM3
detector (neck or tip), a connector estimator, and a CableScanner over them. This builds that
stack so each app is just the sequence that follows the scan.
"""

from ..perception import ConnectorEstimator
from ..perception.sam3 import make_detector
from ..skills.scan import CableScanner, ScanConfig


def build_scanner(cfg, robot, camera):
    """(scanner, detector, estimator) wired from the config's scan/sam3/connector_estimator blocks."""
    detector = make_detector(cfg)
    estimator = ConnectorEstimator(cfg)
    scan_cfg = ScanConfig(cfg)
    scanner = CableScanner(robot, camera, detector, estimator, scan_cfg,
                           data_root=cfg.get('data_dir', 'data'))
    return scanner, detector, estimator


def make_confirm(cfg):
    """A confirm(label) callback, or None if confirmation is disabled."""
    if cfg.get('confirm_each_step', True) is False:
        return None

    def confirm(label):
        try:
            return input(f'[{label}] Enter to proceed (q to abort): ').strip().lower() != 'q'
        except EOFError:
            return True
    return confirm
