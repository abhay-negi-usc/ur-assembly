"""Shared wiring for the cable demos -- assemble the scan pipeline from a config.

The three cable apps (pick-place, touch, pick-assemble) all start the same way: a camera, a SAM3
detector and the ground-plane scanner over it. This builds that
stack so each app is just the sequence that follows the scan.
"""

from ..perception.sam3 import make_detector


def build_scanner(cfg, robot, camera):
    """(scanner, detector, None) wired from the config's sam3/ground_plane blocks.

    ONE scan method (2026-08-28): the single-image GROUND-PLANE estimate. The multi-view
    CableScanner + ConnectorEstimator (orbit the camera, triangulate the junction from many
    views) and the reconstruction mode were REMOVED -- every shipped config ran ground_plane,
    and the orbit machinery was hundreds of lines of alternative nobody exercised. The third
    return stays for the callers' tuple shape; it is always None now."""
    from ..skills.ground_pick import GroundPlaneScanner
    detector = make_detector(cfg)
    scanner = GroundPlaneScanner(robot, camera, detector, None, cfg,
                                 data_root=str(cfg.get('data_root', 'data')))
    return scanner, detector, None


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
