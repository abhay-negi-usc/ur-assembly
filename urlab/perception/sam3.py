"""SAM3 cable/connector detection -- in-process adapters over sam3-abhay's detectors.

The ROS versions were separate PROCESSES, each holding its own copy of SAM3 on a 6 GB GPU, talking
to the demo over a PoseArray topic whose `position.x/y` were secretly pixel coordinates. All of
that goes away: the detector is an object the demo calls, and it returns (u, v, yaw) tuples that
say what they are. The sam3-abhay modules are imported UNCHANGED; point `sam3.repo_path` at the
checkout (that is where the model, the venv and the GPU workarounds live).

FOUR names, THREE methods -- selected by `sam3.mode`:

  neck     -- cable/connector junction, from cable_neck_core.NeckDetector. Iterates over CONNECTOR
              masks, so a mislabelled connector starves it. Has an adaptive-threshold mode.
  junction -- the SAME junction, but the DIAMETER-PROFILING method (cable_neck_diameter.
              JunctionDetector): it unions the cable+connector masks, traces the assembly, and puts
              the junction where the constant-diameter cable run ends -- classification-free, one
              junction per frame, NO adaptive mode. "neck" and "junction" are the same physical
              point by two different methods; the name in the log tells you which is live.
  tip      -- the cable's free END (cable_neck_core.detect_tip). Also classification-free.

All emit the SAME (u, v, yaw) tuple, so the ConnectorEstimator fuses any of them unchanged.

Only ONE SAM3 model is ever loaded: each detector builds exactly one segmentation backend (the
junction method reuses NeckDetector internally for segmentation), so the 6 GB card is not doubled.
"""

import importlib
import os
import sys

import numpy as np

from .. import log as urlog

log = urlog.get('sam3')


def _count(x):
    """Length for a logged count -- tolerant of the sam3-abhay detectors returning either a list of
    masks or an already-counted int for the *_raw fields (it varies), and of None."""
    if x is None:
        return 0
    try:
        return len(x)
    except TypeError:
        return int(x) if isinstance(x, (int, float)) else 0


def _num(x, default=-1.0):
    """A float for logging. detect_adaptive returns thr_cable/thr_conn/eff_conf as None when it
    finds no valid neck combo, and %.2f can't format None -- coalesce to a sentinel."""
    return float(x) if isinstance(x, (int, float)) else default


def _import_from_repo(repo_path, module_name):
    """Import a module from the sam3-abhay scripts/ checkout."""
    scripts = os.path.join(repo_path, 'scripts')
    if not os.path.isdir(scripts):
        raise FileNotFoundError(
            f'No scripts/ under {repo_path!r}. Set sam3.repo_path in the config to your '
            f'sam3-abhay checkout.')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(module_name)


class _Base:
    """Common config + lifecycle. Subclasses build their own detector in _build() -- the base does
    NOT create one, so the junction method (which wraps its own NeckDetector) never loads SAM3
    twice."""

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
        self.core = None
        self.detector = None

        if self.dry_run:
            log.warning('DRY RUN: SAM3 is not loaded; the detector returns nothing.')
            return

        log.info('Loading SAM3 (first call takes ~30 s; inference is 1-2 s/frame on a GTX 1060)...')
        self.core, self.detector = self._build()
        log.info('SAM3 ready on %s.', self.detector.device)

    def _build(self):
        """Return (module, detector). Subclass hook."""
        raise NotImplementedError

    def _neck_backend(self):
        """cable_neck_core + its NeckDetector -- shared by the neck and tip methods."""
        core = _import_from_repo(self.repo_path, 'cable_neck_core')
        det = core.NeckDetector(
            cable_prompt=self.cable_prompt, connector_prompt=self.connector_prompt,
            threshold=self.threshold, connector_threshold=self.connector_threshold,
            mislabel_overlap=self.mislabel_overlap)
        return core, det

    def _pil(self, frame):
        from PIL import Image
        return Image.fromarray(frame.rgb)


class NeckDetector(_Base):
    """Cable/connector junction ("neck") detection -- cable_neck_core.NeckDetector."""

    def _build(self):
        return self._neck_backend()

    def detect(self, frame):
        """[(u, v, yaw_rad), ...]. yaw is in the PIXEL frame (u right, v DOWN), so the neck's
        direction is (cos yaw, sin yaw) in image coordinates -- what the estimator expects."""
        if self.dry_run:
            return []

        pil = self._pil(frame)
        if self.adaptive:
            # The confidence threshold is only a post-hoc FILTER on per-mask scores -- the forward
            # pass is identical at every threshold. So one inference at the floor yields every
            # candidate with its score, and sweeping thresholds afterwards is FREE.
            res = self.detector.detect_adaptive(
                pil, floor=self.confidence_floor, mislabel_overlap=self.mislabel_overlap)
            log.info('  adaptive: thr_cable=%.2f thr_conn=%.2f eff=%.2f (%d combos), '
                     'cables=%d connectors=%d necks=%d',
                     _num(res.get('thr_cable')), _num(res.get('thr_conn')),
                     _num(res.get('eff_conf')), _count(res.get('combos_tried')),
                     _count(res.get('cables_raw')), _count(res.get('connectors_raw')),
                     _count(res.get('necks')))
        else:
            res = self.detector.detect(pil)
            log.info('  cables=%d connectors=%d necks=%d dropped=%d',
                     _count(res.get('cables_raw')), _count(res.get('connectors_raw')),
                     _count(res.get('necks')), _count(res.get('n_dropped')))

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


class JunctionDetector(_Base):
    """Cable/connector JUNCTION detection by DIAMETER PROFILING -- cable_neck_diameter.JunctionDetector.

    Same physical point as the neck, a different method: classification-free (unions both prompts),
    ONE junction per frame, NO adaptive-threshold mode. `sam3.min_contrast` drops a junction whose
    connector/cable diameter ratio is too weak to be a real cable->connector step (0 = keep all)."""

    def __init__(self, cfg):
        # Read before super().__init__ -- _build (called from it) needs work_dim.
        self.min_contrast = float(cfg.get_path('sam3.min_contrast', 0.0))
        self.work_dim = int(cfg.get_path('sam3.work_dim', 1024))
        super().__init__(cfg)
        if not self.dry_run and self.adaptive:
            log.info('  (junction method has no adaptive-threshold mode; sam3.adaptive ignored.)')

    def _build(self):
        core = _import_from_repo(self.repo_path, 'cable_neck_diameter')
        det = core.JunctionDetector(
            cable_prompt=self.cable_prompt, connector_prompt=self.connector_prompt,
            threshold=self.threshold, connector_threshold=self.connector_threshold,
            work_dim=self.work_dim)
        return core, det

    def detect(self, frame):
        """[(u, v, yaw_rad)] -- at most one junction per frame."""
        if self.dry_run:
            return []

        res = self.detector.detect(self._pil(frame))
        self.last_debug = self._overlay(frame, res)

        out, dropped = [], 0
        for j in res.get('junctions', []):
            contrast = float(j.get('contrast', 0.0))
            if self.min_contrast > 0.0 and contrast < self.min_contrast:
                dropped += 1                            # a flat diameter profile isn't a real junction
                continue
            u, v = j['junction']
            dx, dy = j['direction']
            out.append((float(u), float(v), float(np.arctan2(dy, dx))))
        log.info('  junctions=%d (dropped %d < contrast %.1f) | cables=%d connectors=%d',
                 len(out), dropped, self.min_contrast,
                 _count(res.get('cables_raw')), _count(res.get('connectors_raw')))
        return out

    def _overlay(self, frame, res):
        import cv2
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        return self.core.render_overlay(bgr, res.get('assembly'), res.get('result'))


class TipDetector(_Base):
    """Cable TIP detection -- classification-free, so it survives SAM3 mislabelling."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.curve_px = int(cfg.get_path('sam3.curve_px', 40))

    def _build(self):
        return self._neck_backend()

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


_MODES = {'neck': NeckDetector, 'junction': JunctionDetector, 'tip': TipDetector}


def make_detector(cfg):
    """The detector for `sam3.mode` ('neck' | 'junction' | 'tip')."""
    mode = cfg.get_path('sam3.mode', 'neck')
    if mode not in _MODES:
        raise ValueError(f"sam3.mode must be one of {sorted(_MODES)}, got {mode!r}")
    return _MODES[mode](cfg)
