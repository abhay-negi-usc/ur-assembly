"""SAM3 cable/connector detection -- an in-process adapter over sam3-abhay's cable_neck_core.

The ROS versions (cable_neck_ros_node.py, cable_tip_ros_node.py) were two separate PROCESSES,
each holding its own copy of SAM3 on a 6 GB GPU, talking to the demo over a PoseArray topic whose
`position.x/y` were secretly pixel coordinates. All of that goes away: the detector is an object
the demo calls, and it returns (u, v, yaw) tuples that say what they are.

cable_neck_core.py itself is imported UNCHANGED -- it never depended on ROS. It stays in
sam3-abhay because that is where the model, the venv and the GPU workarounds live; point
`sam3.repo_path` at it.

TWO DETECTORS, because SAM3 mislabels the connector as "cable" often enough to matter:

  NeckDetector -- finds the cable/connector junction. Iterates over CONNECTOR masks, so a
      mislabelled connector starves it: zero necks, zero detections, scan fails.

  TipDetector -- finds the cable's free END. CLASSIFICATION-FREE: it unions the cable and
      connector masks and takes the geodesic diameter of the result, so it does not care what
      SAM3 called anything. It survives the exact failure that kills the neck detector, which is
      why the touch demo uses it.

Both emit the same (u, v, yaw) tuple, so the estimator fuses either without knowing the
difference.
"""

import os
import sys

import numpy as np

from .. import log as urlog

log = urlog.get('sam3')


def _count(x):
    """Length for a logged count -- tolerant of cable_neck_core returning either a list of masks
    or an already-counted int for the *_raw fields (it varies), and of None."""
    if x is None:
        return 0
    try:
        return len(x)
    except TypeError:
        return int(x) if isinstance(x, (int, float)) else 0


def _load_core(repo_path):
    """Import cable_neck_core from the sam3-abhay checkout."""
    scripts = os.path.join(repo_path, 'scripts')
    if not os.path.isdir(scripts):
        raise FileNotFoundError(
            f'No scripts/ under {repo_path!r}. Set sam3.repo_path in the config to your '
            f'sam3-abhay checkout.')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import cable_neck_core
    return cable_neck_core


class _Base:
    def __init__(self, cfg):
        s = cfg.section('sam3')
        self.repo_path = s.get('repo_path', '/abhay_ws/sam3-abhay')
        self.cable_prompt = s.get('cable_prompt', 'cable')
        self.connector_prompt = s.get('connector_prompt', 'connector')
        self.threshold = float(s.get('threshold', 0.5))
        self.connector_threshold = s.get('connector_threshold', None)
        self.mislabel_overlap = float(s.get('mislabel_overlap', 0.6))
        self.adaptive = bool(s.get('adaptive', True))
        self.confidence_floor = float(s.get('confidence_floor', 0.2))
        self.dry_run = bool(cfg.get_path('robot.dry_run', False))
        self.last_debug = None

        if self.dry_run:
            log.warning('DRY RUN: SAM3 is not loaded; the detector returns nothing.')
            self.core = self.detector = None
            return

        self.core = _load_core(self.repo_path)
        log.info('Loading SAM3 (first call takes ~30 s; inference is 1-2 s/frame on a GTX 1060)...')
        self.detector = self.core.NeckDetector(
            cable_prompt=self.cable_prompt,
            connector_prompt=self.connector_prompt,
            threshold=self.threshold,
            connector_threshold=self.connector_threshold,
            mislabel_overlap=self.mislabel_overlap)
        log.info('SAM3 ready on %s.', self.detector.device)

    def _pil(self, frame):
        from PIL import Image
        return Image.fromarray(frame.rgb)


class NeckDetector(_Base):
    """Cable/connector junction ("neck") detection."""

    def detect(self, frame):
        """[(u, v, yaw_rad), ...]. yaw is in the PIXEL frame (u right, v DOWN), so the neck's
        direction is (cos yaw, sin yaw) in image coordinates -- which is what the estimator's
        back-projection expects."""
        if self.dry_run:
            return []

        pil = self._pil(frame)
        if self.adaptive:
            # The confidence threshold is only a post-hoc FILTER on per-mask scores -- the forward
            # pass is identical at every threshold. So one inference at the floor yields every
            # candidate with its score, and sweeping thresholds afterwards is FREE. This is what
            # makes the adaptive search affordable: it costs one inference, not one per threshold.
            res = self.detector.detect_adaptive(
                pil, floor=self.confidence_floor, mislabel_overlap=self.mislabel_overlap)
            log.info('  adaptive: thr_cable=%.2f thr_conn=%.2f eff=%.2f (%d combos), '
                     'cables=%d connectors=%d necks=%d',
                     res.get('thr_cable', -1), res.get('thr_conn', -1), res.get('eff_conf', -1),
                     res.get('combos_tried', 0), _count(res.get('cables_raw')),
                     _count(res.get('connectors_raw')), _count(res.get('necks')))
        else:
            res = self.detector.detect(pil)
            log.info('  cables=%d connectors=%d necks=%d dropped=%d',
                     _count(res.get('cables_raw')), _count(res.get('connectors_raw')),
                     _count(res.get('necks')), res.get('n_dropped', 0))

        self.last_debug = self._overlay(frame, res)
        out = []
        for neck in res.get('necks', []):
            u, v = neck['neck']
            dx, dy = neck['direction']
            out.append((float(u), float(v), float(np.arctan2(dy, dx))))
        return out

    def _overlay(self, frame, res):
        import cv2
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        return self.core.render_overlay(
            bgr, res.get('cleaned_cables', []), res.get('conn_masks', []), res.get('necks', []))


class TipDetector(_Base):
    """Cable TIP detection -- classification-free, so it survives SAM3 mislabelling."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.curve_px = int(cfg.get_path('sam3.curve_px', 40))

    def detect(self, frame):
        """[(u, v, yaw_rad)] -- at most one, since a cable has one connector end."""
        if self.dry_run:
            return []

        res = self.detector.detect_tip(self._pil(frame), curve_px=self.curve_px)
        self.last_debug = self._overlay(frame, res)

        tip = res.get('tip')
        if tip is None:
            log.info('  no tip found.')
            return []
        dx, dy = res['direction']
        log.info('  tip at (%.0f, %.0f), end chosen by %s.', tip[0], tip[1],
                 'connector-mask' if res.get('used_connector') else 'thicker-end fallback')
        return [(float(tip[0]), float(tip[1]), float(np.arctan2(dy, dx)))]

    def _overlay(self, frame, res):
        import cv2
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        return self.core.render_tip_overlay(bgr, res)


def make_detector(cfg):
    """NeckDetector or TipDetector, per `sam3.mode` ('neck' | 'tip')."""
    mode = cfg.get_path('sam3.mode', 'neck')
    if mode == 'tip':
        return TipDetector(cfg)
    if mode == 'neck':
        return NeckDetector(cfg)
    raise ValueError(f"sam3.mode must be 'neck' or 'tip', got {mode!r}")
