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

ONE method. The junction detector: SAM3 segments with the fixed prompts "cable" and
"connector", the masks are unioned, the graph tracer follows the strand through its own
crossings, and the slope selector puts the junction on the cable-side flank of the thickest
diameter transition. The neck and tip methods, the geodesic-tracer option and the
longest-run selector were REMOVED as configuration (2026-08-27): every shipped config used
junction/graph/slope, and alternatives-as-config meant every option had to be revalidated
after every change. The geometry functions still accept the old strategies as ARGUMENTS
(compute_junction(trace=...), find_junction_index(select=...)) so tests can compare against
them; they are just not reachable from YAML.


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


class _Base:
    """Common config + lifecycle. Subclasses build their own detector in _build() -- the base does
    NOT create one, so the junction method (which wraps its own NeckDetector) never loads SAM3
    twice."""

    def __init__(self, cfg):
        s = cfg.section('sam3')
        # The prompts are FIXED. They are part of the method, not tuning: the junction geometry
        # assumes the union of exactly these two semantic classes, and every mask-band constant
        # downstream was measured against them.
        self.cable_prompt = 'cable'
        self.connector_prompt = 'connector'
        self.threshold = float(s.get('threshold', 0.5))
        self.connector_threshold = s.get('connector_threshold', None)
        self.mislabel_overlap = float(s.get('mislabel_overlap', 0.6))
        # `sam3.adaptive` and `sam3.confidence_floor` are GONE. They belonged to the retired
        # NECK method: adaptive re-ran segmentation over a grid of per-prompt confidence
        # thresholds until a cable/connector pair produced a valid neck, and confidence_floor
        # was the lowest threshold that search was allowed to try. The junction method never
        # had an adaptive mode (it unions the masks and needs no classification), so the keys
        # were read and ignored -- now they are neither.
        # Opacity of the drawn overlay (lines/arrows/dots/mask tint) over the raw image, 0..1.
        # 1.0 = the detector's overlay unchanged; lower fades every drawn marker toward the raw
        # frame so the cable underneath stays visible. Applied in _blend, after the detector renders.
        self.overlay_opacity = float(np.clip(s.get('overlay_opacity', 1.0), 0.0, 1.0))
        self.dry_run = bool(cfg.get_path('robot.dry_run', False))
        # Where the model runs: 'auto' | 'cpu' | 'cuda'. A MACHINE fact, so it lives in
        # configs/robot.yaml (compute:) with the robot IP.
        self.compute_device = str(cfg.get_path('compute.device', 'auto'))
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
            mislabel_overlap=self.mislabel_overlap, device=self.compute_device)
        log.info('  SAM3 device: %s (compute.device: %s).', det.device, self.compute_device)
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
        # graph tracing and slope selection are THE method, not options (module docstring).
        self.trace = 'graph'
        # HOW the junction is picked off the diameter profile. 'slope' finds the cable-side flank
        # of the thickest transition; 'longest_run' is the original (find the longest constant
        # stretch, take whichever end borders a rise). See perception/junction.py for why the
        # default changed. `connector_peak_min` is how many times the cable diameter a feature
        # must reach to count as a connector at all -- it also decides how many junctions a cable
        # reports, since every qualifying feature gets one.
        from . import junction as _j
        self.select = 'slope'
        self.peak_min = float(cfg.get_path('sam3.connector_peak_min', _j.CONNECTOR_PEAK_MIN))
        super().__init__(cfg)

    def _build(self):
        """SAM3 for segmentation only; the junction geometry is ours."""
        from . import junction
        _core, det = self._neck_backend()      # NeckDetector: the torch model + its GPU setup
        log.info('  junction geometry: urlab.perception.junction, graph trace + slope '
                 'selection (connector >= %.2f x cable).', self.peak_min)
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
        res = self.core.compute_junction(assembly, work_dim=self.work_dim, trace=self.trace,
                                         select=self.select, peak_min=self.peak_min)
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
                                       trace=self.trace, select=self.select,
                                       peak_min=self.peak_min)
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
        one entry per connected component (largest first, up to max_cables). Returns
        [{'junction': (u, v, yaw), 'ends': [...], 'cable_mask': HxW bool, 'size': px}, ...].

        `cable_mask` is the CABLE-side pixels of that component (the connector excluded) -- what a
        tag is stuck to, and the region tag matching scores. It is per-junction, so colour found
        elsewhere in the frame cannot be credited to this cable.

        NUMBERING IS NOT DONE HERE. The caller may re-order these (tag matching ranks them by how
        well each wears the configured colour), and the numbers drawn on the image have to be the
        numbers the user is asked to choose between -- so labelling is `label_cables`, called after
        the order is settled. This still labels in detection order, so a caller that does not
        re-order gets the old behaviour."""
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
            comp = lbl == (i + 1)
            j = self.core.compute_junction(comp, work_dim=self.work_dim, trace=self.trace,
                                           select=self.select, peak_min=self.peak_min)
            if j is None:
                continue
            if self.min_contrast > 0.0 and float(j.get('contrast', 0.0)) < self.min_contrast:
                continue
            u, v = j['junction']
            dx, dy = j['direction']
            end_a, end_b = self._assembly_ends(j, w, h)
            try:
                band = self.core.cable_side_mask(j, (h, w))
                cable_px = None if band is None else (band & comp)
            except Exception as exc:                   # noqa: BLE001 -- selection must still work
                log.warning('  could not isolate the cable-side pixels (%s); tag matching will '
                            'skip this cable.', exc)
                cable_px = None
            cables.append({'junction': (float(u), float(v), float(np.arctan2(dy, dx))),
                           'ends': [e for e in (end_a, end_b) if e is not None],
                           'cable_mask': cable_px,
                           'size': int(comp.sum())})
        self.label_cables(frame, cables)
        log.info('  detect_all: %d cable(s) found.', len(cables))
        return cables

    def label_cables(self, frame, cables):
        """Redraw the selection overlay so the numbers match the CURRENT order of `cables`.

        Separate from detection because the order is the caller's decision -- and a picture whose
        numbers disagree with the prompt is worse than no picture."""
        if self.dry_run:
            return
        self.last_debug = self._raw_bgr(frame)
        if self.last_debug is None:
            return
        h, w = frame.rgb.shape[:2]
        self._draw_enumerated(self.last_debug, cables, w, h)

    def _draw_enumerated(self, vis, cables, w, h):
        """Draw each cable NUMBERED at its junction (#1, #2, ...) plus its ends, so the user can read
        the numbers and pick a target. Full opacity -- the numbers must be legible.

        A cable carrying a tag score shows the NUMBER of its pixels wearing the colour, and the one
        that CLEARED the threshold is drawn in the tag's own colour. Seeing the number the ranking
        was made on is what lets a wrong pick be diagnosed -- 30 px on the intended cable means the
        tag is shadowed, occluded or too far away, which is a different problem from 900 px on the
        wrong one."""
        for i, cab in enumerate(cables, start=1):
            ju, jv, jyaw = cab['junction']
            for e in cab['ends']:
                self._draw_end(vis, e[0], e[1], e[2], label='end', col=(255, 255, 0))
            label, col = f'#{i}', (0, 255, 0)
            if cab.get('tag_score') is not None:
                label = f'#{i} tag {int(cab["tag_score"])}px'
                if cab.get('tag_pass'):
                    col = cab.get('tag_bgr') or (0, 0, 255)
            self._draw_end(vis, ju, jv, jyaw, label=label, col=col)

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


def make_detector(cfg):
    """The cable/connector junction detector. There is exactly one method now -- see the module
    docstring for what was removed and why. `sam3.mode`, if a config still carries it, is
    ignored rather than an error, so an old config keeps loading."""
    return JunctionDetector(cfg)
