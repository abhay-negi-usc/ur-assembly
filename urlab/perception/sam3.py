"""SAM3 cable/connector detection -- in-process adapters over sam3-abhay's detectors.

The ROS versions were separate PROCESSES, each holding its own copy of SAM3 on a 6 GB GPU, talking
to the demo over a PoseArray topic whose `position.x/y` were secretly pixel coordinates. All of
that goes away: the detector is an object the demo calls, and it returns (u, v, yaw) tuples that
say what they are.

WHERE THE LINE IS. Every line of detection code is in THIS repo -- the mask geometry
(perception/junction.py, perception/neck.py, perception/cable_trace_graph.py) AND the SAM3 wrapper
that produces the masks (perception/sam3_backend.py, which owns the per-prompt thresholds and the
pre-Ampere GPU workarounds). What is EXTERNAL is the sam3 LIBRARY and its weights, a dependency
like torch: `import sam3`, resolving its own tokenizer and checkpoint. So there is no
`sam3.repo_path` any more -- if the import fails, INSTALL sam3 (pip install -e <checkout>) rather
than pointing a config at a directory. Detection code used to live in that checkout, where the test
suite could not reach it and a fix only got to the robot by pulling a second repo.

FOUR names, THREE methods -- selected by `sam3.mode`:

  neck     -- cable/connector junction, from cable_neck_core.NeckDetector. Iterates over CONNECTOR
              masks, so a mislabelled connector starves it. Has an adaptive-threshold mode.
  junction -- the SAME junction, but the DIAMETER-PROFILING method (perception.junction): it unions
              the cable+connector masks, traces the assembly, and puts the junction where the
              constant-diameter cable run ends -- classification-free, one junction per frame, NO
              adaptive mode. "neck" and "junction" are the same physical point by two different
              methods; the name in the log tells you which is live. `sam3.trace` picks its
              centreline tracer: 'graph' (default) or 'geodesic' (the original).
  tip      -- the cable's free END (cable_neck_core.detect_tip). Also classification-free.

All emit the SAME (u, v, yaw) tuple, so the ConnectorEstimator fuses any of them unchanged.

Only ONE SAM3 model is ever loaded: each detector builds exactly one segmentation backend (the
junction method reuses NeckDetector for segmentation), so the 6 GB card is not doubled.
"""

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


class _Base:
    """Common config + lifecycle. Subclasses build their own detector in _build() -- the base does
    NOT create one, so the junction method (which wraps its own NeckDetector) never loads SAM3
    twice."""

    def __init__(self, cfg):
        s = cfg.section('sam3')
        self.cable_prompt = s.get('cable_prompt', 'cable')
        self.connector_prompt = s.get('connector_prompt', 'connector')
        self.threshold = float(s.get('threshold', 0.5))
        self.connector_threshold = s.get('connector_threshold', None)
        self.mislabel_overlap = float(s.get('mislabel_overlap', 0.6))
        self.adaptive = bool(s.get('adaptive', True))
        self.confidence_floor = float(s.get('confidence_floor', 0.2))
        # Opacity of the drawn overlay (lines/arrows/dots/mask tint) over the raw image, 0..1.
        # 1.0 = the detector's overlay unchanged; lower fades every drawn marker toward the raw
        # frame so the cable underneath stays visible. Applied in _blend, after the detector renders.
        self.overlay_opacity = float(np.clip(s.get('overlay_opacity', 1.0), 0.0, 1.0))
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
        """The neck geometry + the SAM3 segmentation backend -- shared by all three methods.

        The junction method takes only the SEGMENTATION from this and swaps in its own geometry
        module; the neck and tip methods use both halves."""
        from . import neck
        from .sam3_backend import Sam3Backend
        det = Sam3Backend(
            cable_prompt=self.cable_prompt, connector_prompt=self.connector_prompt,
            threshold=self.threshold, connector_threshold=self.connector_threshold,
            mislabel_overlap=self.mislabel_overlap)
        return neck, det

    def _pil(self, frame):
        from PIL import Image
        return Image.fromarray(frame.rgb)

    def _blend(self, orig_bgr, vis):
        """Fade the rendered overlay `vis` toward the raw `orig_bgr` by overlay_opacity (1.0 = the
        detector's overlay unchanged). Since render_overlay draws everything -- mask tint AND the
        markers -- on a copy of the image, blending the whole result back is what lowers the marker
        opacity without touching the sam3-abhay render code."""
        if vis is None or self.overlay_opacity >= 1.0:
            return vis
        import cv2
        return cv2.addWeighted(vis, self.overlay_opacity,
                               orig_bgr, 1.0 - self.overlay_opacity, 0.0)

    @staticmethod
    def _raw_bgr(frame):
        """The RAW camera image as a fresh BGR array -- the base our markers draw on, so the overlay
        is raw + markers rather than the SAM3 mask-tinted render."""
        import cv2
        return cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)


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
        vis = self.core.render_overlay(
            bgr.copy(), res.get('cleaned_cables', []), res.get('conn_masks', []),
            res.get('necks', []))
        return self._blend(bgr, vis)


class JunctionDetector(_Base):
    """Cable/connector JUNCTION detection by DIAMETER PROFILING -- geometry from perception.junction.

    Same physical point as the neck, a different method: classification-free (unions both prompts),
    ONE junction per frame, NO adaptive-threshold mode. `sam3.min_contrast` drops a junction whose
    connector/cable diameter ratio is too weak to be a real cable->connector step (0 = keep all).

    THE SPLIT. SAM3 supplies the cable + connector MASKS and nothing else; every measurement made
    on those masks is `perception.junction`, in this repo. Previously the whole method came out of
    the sam3-abhay checkout, so the geometry could not be tested here and reached the robot only by
    pulling a second repo. `sam3.trace` picks which centreline tracer that geometry uses --
    'graph' (default, walks through a self-crossing) or 'geodesic' (the original)."""

    def __init__(self, cfg):
        # Read before super().__init__ -- _build (called from it) needs work_dim.
        self.min_contrast = float(cfg.get_path('sam3.min_contrast', 0.0))
        self.work_dim = int(cfg.get_path('sam3.work_dim', 1024))
        self.trace = str(cfg.get_path('sam3.trace', 'graph'))
        if self.trace not in ('graph', 'geodesic'):
            raise ValueError(f"sam3.trace must be 'graph' or 'geodesic', got {self.trace!r}")
        super().__init__(cfg)
        if not self.dry_run and self.adaptive:
            log.info('  (junction method has no adaptive-threshold mode; sam3.adaptive ignored.)')

    def _build(self):
        """SAM3 for segmentation only; the junction geometry is ours."""
        from . import junction
        _core, det = self._neck_backend()      # NeckDetector: the torch model + its GPU setup
        log.info('  junction geometry: urlab.perception.junction, trace=%s.', self.trace)
        return junction, det

    def _detect_raw(self, frame):
        """Segment with SAM3, union both prompts, and run the LOCAL junction geometry.

        The union is what makes the method classification-free: it never has to decide which mask
        is the cable and which the connector, only where the one shape changes diameter."""
        cable_masks, conn_masks = self.detector._segment_both(self._pil(frame))
        h, w = frame.rgb.shape[:2]
        assembly = np.zeros((h, w), dtype=bool)
        for m in cable_masks:
            assembly |= m
        for m in conn_masks:
            assembly |= m
        res = self.core.compute_junction(assembly, work_dim=self.work_dim, trace=self.trace)
        return dict(junctions=[res] if res is not None else [], result=res, assembly=assembly,
                    cables_raw=len(cable_masks), connectors_raw=len(conn_masks))

    def detect(self, frame):
        """[(u, v, yaw_rad)] -- at most one junction per frame."""
        if self.dry_run:
            return []

        res = self._detect_raw(frame)
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

    def detect_junctions(self, frame, top_n=2):
        """Junction candidates -- ONE per top_n largest assembly component (the MULTI-CABLE case),
        as [(u, v, yaw), ...]. The stock detect() only ever returns the LARGEST component's junction,
        so a second visible cable's connector is invisible to it; this splits the assembly mask and
        finds a junction per component. ALL candidates are ingested -- RANSAC (in the estimator)
        decides which is the real connector; the losers are what it decides against. Drawn
        'conn 1/2/...' (numbered per cable) on the overlay."""
        if self.dry_run:
            return []
        res = self._detect_raw(frame)
        self.last_debug = self._raw_bgr(frame)        # markers on the RAW image, not the SAM3 render
        cands = self._component_junctions(res.get('assembly'), top_n)
        if self.last_debug is not None:
            h, w = frame.rgb.shape[:2]
            self._overlay_markers(self.last_debug, cands, [], w, h)   # low opacity, small font
        log.info('  junction candidates=%d (top-%d components) | cables=%d connectors=%d',
                 len(cands), top_n, _count(res.get('cables_raw')), _count(res.get('connectors_raw')))
        return cands

    def _component_junctions(self, assembly, top_n=2):
        """Junction of each of the top_n largest connected components of the assembly mask, as
        [(u, v, yaw), ...] ordered largest-first. One SAM3 pass already produced `assembly`; the
        per-component geometry (compute_junction) is cheap, so re-running it per component is how the
        multi-cable case gets a junction PER cable rather than only the largest one's."""
        if assembly is None:
            return []
        from scipy import ndimage
        m = np.asarray(assembly, dtype=bool)
        lbl, n = ndimage.label(m, structure=np.ones((3, 3), np.uint8))
        if n == 0:
            return []
        sizes = ndimage.sum(m, lbl, index=np.arange(1, n + 1))
        out = []
        for i in np.argsort(sizes)[::-1][:max(1, int(top_n))]:
            j = self.core.compute_junction(lbl == (i + 1), work_dim=self.work_dim,
                                       trace=self.trace)
            if j is None:
                continue
            if self.min_contrast > 0.0 and float(j.get('contrast', 0.0)) < self.min_contrast:
                continue
            u, v = j['junction']
            dx, dy = j['direction']
            out.append((float(u), float(v), float(np.arctan2(dy, dx))))
        return out

    def _overlay_markers(self, vis, cands, ends, w, h):
        """Draw ALL of urlab's markers -- the 'conn 1/2/...' junction candidates and any cable-END
        A/B -- on a COPY and blend back at LOW opacity, so every annotation fades together (it does
        not obscure the image) and the labels don't clash. `ends` is [(label, col, (u,v,yaw)|None)].
        The font is kept small on purpose (see _draw_end)."""
        if vis is None:
            return
        layer = vis.copy()
        self._draw_candidates(layer, cands, w, h)
        for label, col, end in ends:
            if end is not None:
                self._draw_end(layer, end[0], end[1], end[2], label=label, col=col)
        import cv2
        a = min(float(self.overlay_opacity), 0.5)      # secondary annotation -> at most half opacity
        cv2.addWeighted(layer, a, vis, 1.0 - a, 0.0, dst=vis)

    def detect_all(self, frame, max_cables=8):
        """For the GROUND-PLANE / manual-select mode: detect EVERY cable's junction AND its ends --
        one entry per connected component (largest first, up to max_cables) -- and NUMBER them on the
        overlay so the user can pick the target. Returns
        [{'junction': (u, v, yaw), 'ends': [(u, v, yaw), ...]}, ...]."""
        if self.dry_run:
            return []
        res = self._detect_raw(frame)
        self.last_debug = self._raw_bgr(frame)        # numbered markers on the RAW image, not SAM3
        assembly = res.get('assembly')
        if assembly is None:
            return []
        from scipy import ndimage
        m = np.asarray(assembly, dtype=bool)
        lbl, n = ndimage.label(m, structure=np.ones((3, 3), np.uint8))
        if n == 0:
            return []
        sizes = ndimage.sum(m, lbl, index=np.arange(1, n + 1))
        h, w = frame.rgb.shape[:2]
        cables = []
        for i in np.argsort(sizes)[::-1][:max(1, int(max_cables))]:
            j = self.core.compute_junction(lbl == (i + 1), work_dim=self.work_dim,
                                       trace=self.trace)
            if j is None:
                continue
            if self.min_contrast > 0.0 and float(j.get('contrast', 0.0)) < self.min_contrast:
                continue
            u, v = j['junction']
            dx, dy = j['direction']
            end_a, end_b = self._assembly_ends(j, w, h)
            cables.append({'junction': (float(u), float(v), float(np.arctan2(dy, dx))),
                           'ends': [e for e in (end_a, end_b) if e is not None]})
        if self.last_debug is not None:
            self._draw_enumerated(self.last_debug, cables, w, h)
        log.info('  detect_all: %d cable(s) numbered for selection.', len(cables))
        return cables

    def _draw_enumerated(self, vis, cables, w, h):
        """Draw each cable NUMBERED at its junction (#1, #2, ...) plus its ends, so the user can read
        the numbers and pick a target. Full opacity -- the numbers must be legible."""
        for i, cab in enumerate(cables, start=1):
            ju, jv, jyaw = cab['junction']
            for e in cab['ends']:
                self._draw_end(vis, e[0], e[1], e[2], label='end', col=(255, 255, 0))
            self._draw_end(vis, ju, jv, jyaw, label=f'#{i}', col=(0, 255, 0))

    def _draw_candidates(self, vis, cands, w, h):
        """Draw junction candidates labelled 'conn 1/2/...' (a per-cable ID, 1 = nearest image
        centre) on `vis` so a second cable's connector is visible alongside the primary junction
        marker. Numbers, NOT letters -- A/B is reserved for the two ENDS of a cable. No blend here;
        _overlay_markers fades all the markers together."""
        if not cands or vis is None:
            return
        cx, cy = w / 2.0, h / 2.0
        ordered = sorted(cands, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
        for idx, c in enumerate(ordered, start=1):     # conn 1, conn 2, ... (cable ID)
            self._draw_end(vis, c[0], c[1], c[2], label=f'conn {idx}', col=(0, 200, 255))

    def detect_both(self, frame):
        """(junction_candidates, end_dets) from ONE SAM3 pass -- for the two-phase 'cable_end' scan.

        junction_candidates is [(u, v, yaw), ...], ONE per top-N assembly component (the MULTI-CABLE
        case), drawn 'conn 1/2/...' (numbered per cable); ALL are ingested and RANSAC decides which is
        the real connector. The traced assembly (largest component) has TWO ends, labelled
        'cable end A/B' by IMAGE-CENTRE proximity -- 'A' the end CLOSER to centre (tracked/approached),
        'B' the FARTHER -- both drawn
        so the two-ends failure is visible. end_dets is [(u, v, yaw)] for END A only: Phase 1 steers
        on A, and because it also requires cross-view consistency, an A that flips between the two
        ends simply won't agree across views -> no approach (a built-in guard on the ambiguous case)."""
        if self.dry_run:
            return [], []
        res = self._detect_raw(frame)
        self.last_debug = self._raw_bgr(frame)        # markers on the RAW image, not the SAM3 render
        j = res.get('result')
        if j is None:
            log.info('  no junction/endpoint this view.')
            return [], []

        jcands = self._component_junctions(res.get('assembly'), top_n=2)

        h, w = frame.rgb.shape[:2]
        end_a, end_b = self._assembly_ends(j, w, h)
        if self.last_debug is not None:
            self._overlay_markers(self.last_debug, jcands,   # conn 1/2 + cable ends, low opacity
                                  [('cable end A', (255, 255, 0), end_a),
                                   ('cable end B', (0, 165, 255), end_b)], w, h)
        log.info('  junction candidates=%d endA=%d endB=%d | cables=%d connectors=%d', len(jcands),
                 1 if end_a else 0, 1 if end_b else 0,
                 _count(res.get('cables_raw')), _count(res.get('connectors_raw')))
        return jcands, ([end_a] if end_a is not None else [])   # Phase 1 steers on cable end A

    @staticmethod
    def _draw_end(vis, u, v, yaw, label='cable end', col=(255, 255, 0)):
        """Draw a marker (dot + short arrow + small label) at (u, v) in `col` (BGR). Small font and
        arrow on purpose -- several of these share the frame, so oversized ones clash. Blended to low
        opacity by the caller (_overlay_markers), so it does not obscure the image."""
        import cv2
        wh = vis.shape[:2]
        diag = float(np.hypot(wh[1], wh[0]))
        thick = max(1, int(0.002 * diag))
        r = max(3, int(0.004 * diag))
        p0 = (int(round(u)), int(round(v)))
        L = 0.055 * diag                               # shorter arrow -- less clutter
        p1 = (int(round(u + np.cos(yaw) * L)), int(round(v + np.sin(yaw) * L)))
        cv2.arrowedLine(vis, p0, p1, col, thick, cv2.LINE_AA, tipLength=0.22)
        cv2.circle(vis, p0, r, col, -1, cv2.LINE_AA)
        cv2.circle(vis, p0, r, (0, 0, 0), max(1, thick // 2), cv2.LINE_AA)
        fscale = 0.00055 * diag                        # ~half the previous size (labels were clashing)
        fthick = max(1, thick // 2)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, fthick)
        ox, oy = p0[0] + r + 4, max(th + 4, p0[1] - r)
        cv2.rectangle(vis, (ox - 3, oy - th - 4), (ox + tw + 3, oy + bl), (0, 0, 0), -1)
        cv2.putText(vis, label, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, fscale, col, fthick, cv2.LINE_AA)
        return vis

    @staticmethod
    def _assembly_ends(j, w, h):
        """Both endpoints of the traced assembly as (u, v, yaw), returned (A, B): A the end CLOSER to
        the image centre, B the FARTHER. The direction is the PCA principal axis of a WINDOW of the
        centreline near each endpoint (a least-squares line over many points, far steadier than a
        two-point secant tangent), signed to point OUTWARD -- from the interior toward the tip, i.e.
        out of the cable/connector END. (None, None) if the trace is too short."""
        try:
            path = np.asarray(j['_path'], dtype=float)     # (M,2) small-image (y,x), tip -> tip
            inv = 1.0 / float(j['_scale'])
        except (KeyError, TypeError, ValueError):
            return None, None
        m = len(path)
        if m < 4:
            return None, None
        cx, cy = w / 2.0, h / 2.0
        span = int(np.clip(0.2 * m, 4, 25))               # window of path points for the PCA

        def endpoint(e, seg):
            ey, ex = path[e]
            c = seg.mean(axis=0)
            _, _, vt = np.linalg.svd(seg - c, full_matrices=False)
            major = vt[0]                                  # (dy, dx) principal axis (unsigned)
            if float(np.dot(major, path[e] - c)) < 0:      # point from the interior OUT toward the tip
                major = -major
            return (float(ex * inv), float(ey * inv), float(np.arctan2(major[0], major[1])))

        e0 = endpoint(0, path[0:span + 1])
        e1 = endpoint(m - 1, path[m - 1 - span:m])
        d0 = (e0[0] - cx) ** 2 + (e0[1] - cy) ** 2
        d1 = (e1[0] - cx) ** 2 + (e1[1] - cy) ** 2
        return (e0, e1) if d0 <= d1 else (e1, e0)          # A = closer to centre, B = farther

    def detect_cable(self, frame):
        """The junction PLUS the ordered CABLE-side centreline -- for the reconstruction scan mode.

        Returns {'junction': (u,v), 'yaw': rad, 'skeleton': (N,2) full-res (u,v)} or None. The
        skeleton is the traced centreline restricted to the CABLE (thin) side of the junction,
        ordered from the junction (index 0) OUTWARD toward the free end, and recentred onto each
        cross-section's midline (a geodesic trace hugs the inside of a bend; the graph tracer's
        medial axis already sits centred, so the shift is small there). It is built entirely from
        artefacts compute_junction already exposes (_path/_normals/_half_plus/_half_minus/_scale/
        _junction_k), so perception/junction.py needs nothing added for it.

        NOTE it does NOT yet consult `_crossing`: on a self-crossing cable the skeleton handed to
        the reconstruction still contains the fused samples. That only matters in
        scan.mode 'reconstruction'.

        Only this detector implements detect_cable: the reconstruction needs the full centreline,
        which the diameter-profiling method traces but the neck/tip methods do not."""
        if self.dry_run:
            return None
        res = self._detect_raw(frame)
        self.last_debug = self._overlay(frame, res)
        j = res.get('result')
        if j is None:
            log.info('  no junction -- no cable observation for reconstruction.')
            return None

        path = np.asarray(j['_path'], dtype=float)       # (M,2) small-image (y,x), tip A -> tip B
        normals = np.asarray(j['_normals'], dtype=float)
        d_plus = np.asarray(j['_half_plus'], dtype=float)
        d_minus = np.asarray(j['_half_minus'], dtype=float)
        dia = np.asarray(j['_dia'], dtype=float)
        k = int(j['_junction_k'])
        inv = 1.0 / float(j['_scale'])                   # small-image px -> full-res px
        M = len(path)
        if M < 6 or k < 0 or k > M - 1:
            log.info('  junction trace too short for a cable skeleton.')
            return None

        # The cable is the THIN, constant-diameter side of the junction; the connector is thicker.
        left_thin = float(np.mean(dia[:max(1, k)])) if k > 0 else np.inf
        right_thin = float(np.mean(dia[k + 1:])) if k < M - 1 else np.inf
        cable_left = left_thin <= right_thin
        idx = range(k, -1, -1) if cable_left else range(k, M)   # from the junction outward

        recentred = path + 0.5 * (d_plus - d_minus)[:, None] * normals   # onto the cross-section
        skel = np.array([[recentred[i, 1] * inv, recentred[i, 0] * inv] for i in idx])  # (u,v) full
        u, v = j['junction']
        skel[0] = [float(u), float(v)]                   # anchor index 0 exactly on the junction
        dx, dy = j['direction']
        log.info('  cable skeleton: %d points on the %s side of the junction.',
                 len(skel), 'left/A' if cable_left else 'right/B')
        return {'junction': (float(u), float(v)), 'yaw': float(np.arctan2(dy, dx)),
                'skeleton': skel}

    def _overlay(self, frame, res):
        import cv2
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        vis = self.core.render_overlay(bgr.copy(), res.get('assembly'), res.get('result'))
        return self._blend(bgr, vis)


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
        vis = self.core.render_tip_overlay(bgr.copy(), res)
        return self._blend(bgr, vis)


_MODES = {'neck': NeckDetector, 'junction': JunctionDetector, 'tip': TipDetector}


def make_detector(cfg):
    """The detector for `sam3.mode` ('neck' | 'junction' | 'tip')."""
    mode = cfg.get_path('sam3.mode', 'junction')
    if mode not in _MODES:
        raise ValueError(f"sam3.mode must be one of {sorted(_MODES)}, got {mode!r}")
    return _MODES[mode](cfg)
