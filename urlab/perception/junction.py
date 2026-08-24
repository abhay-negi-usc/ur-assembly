"""Cable<->connector JUNCTION geometry -- VENDORED from sam3-abhay/scripts/cable_neck_diameter.py.

WHY IT LIVES HERE. The geometry is pure numpy/OpenCV and is the part of the detector this repo
actually iterates on, but it used to be imported at runtime out of a sibling checkout
(`sam3.repo_path`), so it was neither version-controlled with the app nor testable in CI, and a
fix only reached the robot by pulling a second repo. It is now ours: the SAM3 *segmentation*
still comes from that checkout (it is a torch model with weights and cannot be vendored), but
everything downstream of the mask is here.

WHAT IT DOES. Given ONE boolean mask of the cable+connector assembly, trace its centreline,
measure the true local width along it, and put the junction where the constant-diameter cable
ends and the connector's thicker body begins.

TWO TRACERS, selectable with `trace=` (config: sam3.trace):

  'geodesic'  the ORIGINAL. Endpoints are the shape's geodesic diameter and the path is the
              shortest route between them. Exact for a cable in a simple arc. Where the cable
              crosses itself it SHORTCUTS -- the shortest route skips a whole lobe, which can
              leave the connector untraced -- and it cannot report that a crossing happened, so
              the width spike where two strands fuse is indistinguishable from a connector.
  'graph'     the DEFAULT. Builds the skeleton's graph and walks THROUGH each crossing by
              pairing branches on tangent continuity, flagging every contaminated sample so the
              junction search can ignore it. See cable_trace_graph.

With trace='geodesic' and no crossing flags this module reproduces the original bit for bit --
`find_junction_index` short-circuits every crossing-aware branch on an all-False mask -- so the
two are directly comparable on the same image.
"""

import math

import cv2
import numpy as np
from scipy import ndimage

from .cable_trace_graph import trace_centerline_graph

# BGR colors for cv2 drawing (kept consistent with cable_neck_core).
COL_CABLE = (60, 60, 231)     # red-ish mask tint
COL_PATH = (255, 200, 0)      # cyan/blue traced centreline
COL_JUNCTION = (0, 215, 255)  # amber junction dot
COL_ARROW = (0, 255, 0)       # green connector-ward arrow
COL_DIA = (255, 255, 255)     # white diameter chord
COL_OUTLINE = (255, 0, 255)   # magenta detected cable outline (both sides)


# --------------------------------------------------------------------------- mask primitives
# pca_axis / centroid are vendored from cable_neck_core for the same reason as the rest: they
# are four lines of numpy, and importing them was the last thing tying this file to torch.
def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])          # (x, y)


def pca_axis(mask):
    """Return (major_unit_vec (x,y), elongation) for a boolean mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return np.array([1.0, 0.0]), 1.0
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    major = evecs[:, 0]
    elong = math.sqrt(max(evals[0], 1e-9) / max(evals[1], 1e-9))
    return major / (np.linalg.norm(major) + 1e-9), elong


def largest_component(mask):
    """Largest 8-connected component of a boolean mask (drops unrelated blobs)."""
    lbl, n = ndimage.label(mask, structure=np.ones((3, 3), np.uint8))
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndimage.sum(mask, lbl, index=np.arange(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def _geodesic_dist(sub, seed):
    """Geodesic (within-mask, 8-connected) integer distance from `seed=(y,x)`.

    Vectorised layered dilation on the bounding box -- the same trick as
    cable_neck_core._geodesic_bfs. -1 marks unreachable / outside-mask pixels.
    """
    dist = np.full(sub.shape, -1, dtype=np.int32)
    sy, sx = int(seed[0]), int(seed[1])
    if not (0 <= sy < sub.shape[0] and 0 <= sx < sub.shape[1] and sub[sy, sx]):
        return dist
    cur = np.zeros(sub.shape, dtype=bool)
    cur[sy, sx] = True
    dist[cur] = 0
    k = np.ones((3, 3), np.uint8)
    d = 0
    while cur.any():
        d += 1
        grown = cv2.dilate(cur.astype(np.uint8), k).astype(bool)
        nxt = grown & sub & (dist < 0)
        if not nxt.any():
            break
        dist[nxt] = d
        cur = nxt
    return dist


def _farthest(dist):
    """(y, x) of the maximum reachable geodesic distance in a field from _geodesic_dist."""
    idx = int(np.argmax(np.where(dist >= 0, dist, -1)))
    return np.unravel_index(idx, dist.shape)


def trace_centerline(mask):
    """Trace the shape's length as an ordered list of (y, x) centreline pixels.

    Endpoints = the geodesic diameter (BFS from any seed -> farthest A; BFS from A ->
    farthest B). Path = greedy descent of the A-rooted distance field from B back to A,
    so consecutive pixels are 8-neighbours and strictly decreasing in distance. Returns
    (path Nx2 int array, distA field) or (None, None) if the shape is too small.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 10:
        return None, None
    dseed = _geodesic_dist(mask, (ys[0], xs[0]))
    A = _farthest(dseed)
    distA = _geodesic_dist(mask, A)
    B = _farthest(distA)

    # Greedy descent B -> A. At each step move to the 8-neighbour inside the mask with the
    # smallest geodesic distance; that strictly decreases toward the source A (dist 0).
    H, W = mask.shape
    y, x = int(B[0]), int(B[1])
    path = [(y, x)]
    guard = int(distA[y, x]) + 5
    while distA[y, x] > 0 and guard > 0:
        guard -= 1
        best = None
        best_d = distA[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and distA[ny, nx] >= 0:
                    if distA[ny, nx] < best_d:
                        best_d = distA[ny, nx]
                        best = (ny, nx)
        if best is None:
            break
        y, x = best
        path.append((y, x))
    path.reverse()  # order from tip A -> tip B
    return np.asarray(path, dtype=np.int32), distA


def path_normals(path, half=4):
    """Unit normals (ny, nx) perpendicular to a LOCAL tangent at each path pixel.

    The tangent is a short secant p[i+half] - p[i-half] (a few pixels each way). It must
    stay LOCAL: smoothing the tangent over a long window flattens a curved/coiled cable, so
    the "normal" then points partly ALONG the cable and the width ray runs down its length.
    A short secant is stable against single-pixel staircase jitter yet follows real bends.
    """
    n = len(path)
    p = path.astype(np.float64)
    if n < 3:
        return np.tile(np.array([0.0, 1.0]), (n, 1))
    fwd = np.minimum(np.arange(n) + half, n - 1)
    bwd = np.maximum(np.arange(n) - half, 0)
    t = p[fwd] - p[bwd]                       # (n, 2) local secant (dy, dx)
    tn = np.hypot(t[:, 0], t[:, 1]) + 1e-9
    ty, tx = t[:, 0] / tn, t[:, 1] / tn
    return np.stack([-tx, ty], axis=1)        # 90-deg rotation of the tangent


def _march_to_edge(mask, path, normals, sign, max_r):
    """For each path pixel, distance (px) travelled along sign*normal before leaving the mask.

    Vectorised across all path pixels: one step advances every still-inside ray by 1 px.
    Out-of-bounds counts as outside. Rays stop at the FIRST background pixel, so a ray never
    jumps a gap into a neighbouring strand -- that is what makes this a true local width.
    """
    n = len(path)
    H, W = mask.shape
    d = np.zeros(n)
    active = np.ones(n, dtype=bool)
    for t in range(1, max_r + 1):
        pts = path + sign * t * normals
        yi = np.round(pts[:, 0]).astype(np.int64)
        xi = np.round(pts[:, 1]).astype(np.int64)
        cand = active & (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
        hit = np.zeros(n, dtype=bool)
        hit[cand] = mask[yi[cand], xi[cand]]
        active &= hit
        if not active.any():
            break
        d[active] = t
    return d


def perp_width(mask, path, normals, max_r):
    """True local cable WIDTH at each path pixel: cast rays both ways along the normal.

    width = d_plus + d_minus + 1 (the +1 is the centre pixel). Unlike 2*EDT (distance to the
    NEAREST edge, in any direction), this measures the cross-section ACROSS the cable's
    travel direction, so it neither under-reads at bends nor over-reads where the medial
    axis sits inside a merged blob. Returns (width, d_plus, d_minus).
    """
    d_plus = _march_to_edge(mask, path, normals, +1, max_r)
    d_minus = _march_to_edge(mask, path, normals, -1, max_r)
    return d_plus + d_minus + 1.0, d_plus, d_minus


def diameter_profile(path, width, smooth_frac=0.03):
    """Arc length + smoothed diameter profile from a raw per-pixel width array.

    Returns (arclen Nx float in px, diameter N float in px). `arclen` accounts for the
    sqrt(2) cost of diagonal steps. Diameter is smoothed with a moving average whose window
    is `smooth_frac` of the path length (odd, >= 3) to tame mask-edge jitter.
    """
    yy, xx = path[:, 0], path[:, 1]
    step = np.hypot(np.diff(yy), np.diff(xx))
    arclen = np.concatenate([[0.0], np.cumsum(step)])
    dia = np.asarray(width, dtype=np.float64)

    w = max(3, int(smooth_frac * len(dia)) | 1)  # odd window
    if len(dia) >= w:
        kernel = np.ones(w) / w
        dia = np.convolve(dia, kernel, mode="same")
        # convolve's edges are averaged over fewer real samples; renormalise them
        norm = np.convolve(np.ones_like(dia), kernel, mode="same")
        dia = dia / norm
    return arclen, dia


def _cable_baseline(dia, ignore=None):
    """Robust estimate of the cable's constant diameter d0.

    The cable is the THIN, CONSTANT-diameter portion; the connector is thicker. So the
    cable level is the mode of the profile's LOWER half (values <= median are
    cable-dominated -- the connector sits above the median). A histogram mode over that
    half ignores the free-tip taper (a few low outliers) and the connector entirely.

    `ignore` (crossing samples) is dropped first: where two strands are fused the width is
    not the cable's, and letting those samples into the median shifts the half the mode is
    taken over.
    """
    if ignore is not None and ignore.any() and (~ignore).sum() >= 4:
        dia = dia[~ignore]
    med = float(np.median(dia))
    lower = dia[dia <= med]
    if lower.size < 4:
        lower = dia
    hist, edges = np.histogram(lower, bins=min(30, max(6, lower.size // 8)))
    k = int(np.argmax(hist))
    return max(0.5 * (edges[k] + edges[k + 1]), 1e-3)


def _contiguous_runs(flag):
    """List of (start, end_inclusive) index ranges where boolean `flag` is True."""
    runs = []
    i, n = 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _spread_crossing(crossing, dia, hi, d0):
    """Grow each crossing flag over the width spike it causes, so the whole contaminated
    stretch is marked -- not just the pixels the walk bridged.

    The graph tracer knows WHERE two strands are fused, but not how far the damage reaches:
    a perpendicular ray is already reading both strands well before the centrelines meet, and
    how early depends on the crossing angle (shallow = long). So grow outward from each flag
    while the profile is still above the cable band.

    CAPPED at ~3 cable widths each way, because the growth cannot be allowed to run into a
    genuine connector and swallow it -- if a connector lies ACROSS the cable, its fat region
    is contiguous with the crossing's, and an uncapped growth would mark the real junction as
    contaminated and hide it.
    """
    out = np.array(crossing, dtype=bool)
    n = len(out)
    fat = dia > hi
    cap = max(4, int(round(3.0 * d0)))
    for a, b in _contiguous_runs(np.array(crossing, dtype=bool)):
        i = a
        while i > 0 and fat[i - 1] and (a - i) < cap:
            i -= 1
            out[i] = True
        j = b
        while j < n - 1 and fat[j + 1] and (j - b) < cap:
            j += 1
            out[j] = True
    return out


def find_junction_index(dia, tol_frac=0.30, crossing=None):
    """Index of the cable<->connector JUNCTION = the END of the constant-diameter cable.

    Per the cable geometry: the cable has a CONSTANT diameter d0; the connector is thicker.
    So the junction is simply where the constant-diameter run ends and the profile departs
    upward into the connector. Directly:

      1. estimate the cable baseline d0 (mode of the thin half -- see _cable_baseline),
      2. mark every pixel "at cable diameter": |d - d0| <= tol (tol = tol_frac * d0),
      3. the cable is the LONGEST contiguous constant-diameter run [i0, i1] -- this ignores
         short constant patches inside a connector or a stubby free tip,
      4. the junction is the END of that run that BORDERS the connector, i.e. where the
         diameter just beyond the run rises above d0 (not a free cable tip, which drops
         below d0). If BOTH ends border a connector (cable between two connectors), take
         the end next to the THICKER connector.

    The junction thus sits exactly at the last constant-diameter pixel of the cable, not
    somewhere up the connector's ramp. Returns (index, info). `contrast` = connector/cable
    diameter ratio is the confidence signal (~1 => no real diameter change).

    `crossing` (from cable_trace_graph, per path sample) marks where the traced strand passes
    THROUGH another strand. Those samples are not measurements of anything: two fused strands
    have no background between them, so the perpendicular ray runs through both and the width
    spikes. Without the flags that spike is indistinguishable from a connector, and step 3
    above makes it worse -- the spike SPLITS the constant run in two, so the "cable" becomes
    whichever half is longer, and step 4 then places the junction at the crossing whenever the
    crossing reads thicker than the connector (which a shallow crossing does easily, since its
    apparent width goes as 1/sin of the crossing angle). Worse still, `contrast` is measured on
    whichever bump won, so the wrong answer comes back looking CONFIDENT.

    So crossings are treated as TRANSPARENT: they neither split the cable run nor contribute a
    diameter. The cable is allowed to continue straight through, which is what it physically
    does, and the junction is then found among real measurements only.
    """
    n = len(dia)
    crossing = (np.zeros(n, dtype=bool) if crossing is None
                else np.asarray(crossing, dtype=bool)[:n])
    d0 = _cable_baseline(dia, crossing)
    tol = max(2.0, tol_frac * d0)
    hi = d0 + tol
    if crossing.any():
        crossing = _spread_crossing(crossing, dia, hi, d0)
    # A crossing counts as in-band so the run BRIDGES it rather than being cut in two.
    inband = (np.abs(dia - d0) <= tol) | crossing

    runs = _contiguous_runs(inband)
    if not runs:                                   # degenerate: no constant run at all
        peak_idx = int(np.argmax(dia))
        return peak_idx, dict(
            junction_k=peak_idx, peak_idx=peak_idx, cable_baseline=float(d0),
            thin_diameter=float(d0), thick_diameter=float(np.max(dia)),
            contrast=float(np.max(dia) / max(d0, 1e-6)), connector_on_right=True)

    i0, i1 = max(runs, key=lambda r: r[1] - r[0])  # the cable = longest constant run
    w = max(2, int(0.02 * n))
    real = ~crossing                               # samples that measure a single strand

    def borders_connector(end, step):             # does the profile rise past `hi` beyond `end`?
        a = end + step
        b = a + step * w
        lo_, hi_ = (min(a, b), max(a, b))
        sl = slice(max(0, lo_), min(n, hi_ + 1))
        seg = dia[sl][real[sl]]                   # a fused crossing is not a connector
        return seg.size > 0 and float(np.median(seg)) > hi

    def peak(sl):
        seg = dia[sl][real[sl]]
        return float(np.max(seg)) if seg.size else 0.0

    left_conn = borders_connector(i0, -1)
    right_conn = borders_connector(i1, +1)
    left_peak = peak(slice(0, i0)) if i0 > 0 else 0.0
    right_peak = peak(slice(i1 + 1, n)) if i1 < n - 1 else 0.0

    if left_conn and right_conn:
        connector_on_right = right_peak >= left_peak
    elif right_conn:
        connector_on_right = True
    elif left_conn:
        connector_on_right = False
    else:                                          # no connector adjacent -> nearest global peak
        connector_on_right = int(np.argmax(dia)) >= i1

    j = i1 if connector_on_right else i0

    # The junction must sit on a REAL measurement. It normally does -- with crossings folded
    # into the run they are interior, not at its ends -- but a crossing right at the cable's
    # end (a connector lying across the cable) can put one here. Back off along the cable to
    # the last honest sample and say so, rather than reporting a fused width as the junction.
    on_crossing = bool(crossing[j])
    if on_crossing:
        step = -1 if connector_on_right else +1
        k = j
        while 0 <= k < n and crossing[k]:
            k += step
        if 0 <= k < n:
            j = k

    thick = right_peak if connector_on_right else left_peak
    thick = max(thick, d0)
    contrast = thick / max(d0, 1e-6)
    info = dict(
        junction_k=int(j),
        peak_idx=int(np.argmax(np.where(real, dia, -np.inf))) if real.any() else int(np.argmax(dia)),
        cable_baseline=float(d0),
        thin_diameter=float(d0),
        thick_diameter=float(thick),
        contrast=float(contrast),
        connector_on_right=bool(connector_on_right),
        n_crossing_px=int(crossing.sum()),
        junction_on_crossing=on_crossing,
    )
    return int(j), info


def connector_direction(small, junction_yx, normal_k, conn_tip_yx, min_area=15):
    """Direction from the CONNECTOR's principal axis -- the same rule the neck node uses.

    The cable tangent at the junction is a poor orientation: it follows the floppy cable's
    local bend. The connector is a rigid, usually elongated body whose PCA major axis is a
    stable pointing direction. To get the connector region without a classification, we CUT
    the assembly across the junction cross-section (a barrier line perpendicular to the
    cable; it easily severs the thin cable at the junction) and keep the component on the
    connector side (the one containing the connector-end tip). Then, exactly as compute_necks:
      * elongated connector (elong > 1.3): direction = major axis, signed junction->centroid,
      * else: direction = junction -> connector-centroid vector.
    Returns (dx, dy) in the pixel frame, or None if no usable connector region was found.
    """
    H, W = small.shape
    Lp = int(math.hypot(H, W))
    ny, nx = float(normal_k[0]), float(normal_k[1])
    cut = (small.astype(np.uint8) * 255)
    p1 = (int(round(junction_yx[1] - nx * Lp)), int(round(junction_yx[0] - ny * Lp)))
    p2 = (int(round(junction_yx[1] + nx * Lp)), int(round(junction_yx[0] + ny * Lp)))
    cv2.line(cut, p1, p2, 0, thickness=3)               # sever the assembly at the junction

    lbl, nc = ndimage.label(cut > 0)
    if nc == 0:
        return None
    ty = min(max(int(round(conn_tip_yx[0])), 0), H - 1)
    tx = min(max(int(round(conn_tip_yx[1])), 0), W - 1)
    comp_id = int(lbl[ty, tx])
    if comp_id == 0:
        return None
    conn = lbl == comp_id
    if int(conn.sum()) < min_area:
        return None

    major, elong = pca_axis(conn)                       # (x, y) unit, elongation
    conn_c = centroid(conn)                              # (x, y)
    junction_xy = np.array([junction_yx[1], junction_yx[0]])
    v_body = conn_c - junction_xy
    if float(np.linalg.norm(v_body)) < 1e-6:
        return None
    if elong > 1.3:
        d = major.copy()
        if float(np.dot(d, v_body)) < 0:
            d = -d
    else:
        d = v_body / np.linalg.norm(v_body)
    return float(d[0]), float(d[1])


def compute_junction(assembly_mask, work_dim=1024, min_area_frac=0.0004, trace='graph'):
    """Full diameter-based JUNCTION detection on a boolean assembly mask (cable U connector).

    Runs the geometry on a downscaled copy for speed, then maps everything back to
    full-resolution pixels. Returns a dict (key 'junction'=[u,v], ...) or None if nothing
    usable. This is the junction method; cf. compute_necks (the neck method).
    """
    H, W = assembly_mask.shape
    if assembly_mask.sum() < max(64, int(min_area_frac * H * W)):
        return None

    scale = min(1.0, work_dim / float(max(H, W)))
    if scale < 1.0:
        sw, sh = max(1, int(W * scale)), max(1, int(H * scale))
        small = cv2.resize(assembly_mask.astype(np.uint8), (sw, sh),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        small = assembly_mask.copy()

    small = largest_component(small)
    if small.sum() < 20:
        return None
    # close pin-holes so the distance transform / trace don't see spurious thin spots
    small = cv2.morphologyEx(small.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((3, 3), np.uint8)).astype(bool)

    # Trace the strand through its own crossings (skeleton graph), and fall back to the
    # geodesic diameter if that yields nothing. The geodesic path SHORTCUTS a loop -- it takes
    # the cheapest route through the crossing, skipping a whole lobe, which can leave the
    # connector untraced entirely -- and it cannot report where the crossings were. See
    # cable_trace_graph.
    crossing, trace_info = None, {}
    path = None
    if trace == 'graph':
        path, crossing, trace_info = trace_centerline_graph(small)
    # COVERAGE GATE. A mask ragged enough that thinning sprouts more branches than the spur
    # pruning can clear leaves a shattered skeleton, and the longest strand is then a FRAGMENT.
    # That is worse than the geodesic shortcut it replaced: both are short, but the shortcut at
    # least runs tip to tip. Measured on synthetic loops across resolution, cable width and
    # boundary noise, a whole strand covers >= 0.93 of the skeleton and a fragment <= 0.53 --
    # nothing lands between 0.56 and 0.93, so the threshold is not finely balanced.
    if path is not None and trace_info.get("coverage", 1.0) < 0.75:
        path = None
    if path is None or len(path) < 10:
        path, _distA = trace_centerline(small)
        crossing = None
        trace_info = dict(trace_info, fallback="geodesic")
    trace_info.setdefault("trace", trace)
    if path is None or len(path) < 10:
        return None

    # TRUE local width: cast rays perpendicular to the cable's travel direction and measure
    # how far they reach before leaving the mask (see perp_width) -- NOT 2*EDT (nearest edge).
    normals = path_normals(path)
    max_r = max(8, int(0.25 * max(small.shape)))
    width, d_plus, d_minus = perp_width(small, path, normals, max_r)
    arclen, dia = diameter_profile(path, width)
    k, info = find_junction_index(dia, crossing=crossing)

    inv = 1.0 / scale  # small-image px -> full-res px
    # Recentre the origin onto the MIDDLE of the cable cross-section. The traced centreline
    # can hug one edge (a geodesic path cuts the inside of a bend), so shift along the local
    # normal by half the difference of the two ray reaches: +d_plus and -d_minus edges ->
    # their midpoint is at offset (d_plus - d_minus)/2.
    center_off = 0.5 * (d_plus[k] - d_minus[k])
    junction_yx = path[k].astype(np.float64) + center_off * normals[k]

    # Orientation = the CONNECTOR's principal axis (same rule as the neck node / compute_necks),
    # which is far more stable than the floppy cable's tangent at the junction.
    conn_tip = path[-1] if info["connector_on_right"] else path[0]
    cdir = connector_direction(small, junction_yx, normals[k], conn_tip)
    if cdir is not None:
        dx, dy = cdir  # pixel frame, already signed junction -> connector
    else:
        # fallback: local cable tangent toward the connector side
        span = max(2, int(0.04 * len(path)))
        a = max(0, k - span)
        b = min(len(path) - 1, k + span)
        tangent = (path[b] - path[a]).astype(np.float64)  # (dy, dx)
        if not info["connector_on_right"]:
            tangent = -tangent
        nrm = float(np.linalg.norm(tangent))
        if nrm < 1e-6:
            tangent = np.array([0.0, 1.0])
            nrm = 1.0
        tangent /= nrm
        dy, dx = tangent

    junction_uv = [float(junction_yx[1] * inv), float(junction_yx[0] * inv)]  # (u, v)
    return dict(
        junction=junction_uv,
        direction=[float(dx), float(dy)],
        angle_deg=float(math.degrees(math.atan2(-dy, dx))),
        cable_diameter_px=float(info["thin_diameter"] * inv),
        connector_diameter_px=float(info["thick_diameter"] * inv),
        contrast=float(info["contrast"]),
        arc_length_px=float(arclen[-1] * inv),
        junction_arc_frac=float(arclen[k] / max(arclen[-1], 1e-6)),
        # Self-crossing report. n_crossings > 0 means the cable overlaps itself somewhere on
        # the traced strand; junction_on_crossing means the junction had to be backed off a
        # fused stretch, so treat the result as low confidence.
        n_crossings=int(trace_info.get("n_crossings", 0)),
        junction_on_crossing=bool(info.get("junction_on_crossing", False)),
        trace_fallback=bool(trace_info.get("fallback")),
        # small-scale artefacts for rendering / plotting
        _scale=scale,
        _path=path,
        _arclen=arclen,
        _dia=dia,
        _normals=normals,
        _half_plus=d_plus,    # ray reach along +normal (edge distance per path px)
        _half_minus=d_minus,  # ray reach along -normal
        _junction_k=k,
        _crossing=crossing,   # per-path-sample: this width reads TWO fused strands, not one
        _mask_small=small,
    )


def cable_outline(res):
    """Two rails tracing the detected cable boundary from the perpendicular ray hits.

    Each rail sits exactly where the width ray reached the mask edge: `centre + d_plus*n`
    and `centre - d_minus*n`. Returns (left Nx2, right Nx2) in (y, x). Overlay these to see
    precisely what cross-section the tracer measured -- and where it departs from the cable.
    """
    p = res["_path"].astype(np.float64)
    normals = res["_normals"]
    left = p + res["_half_plus"][:, None] * normals
    right = p - res["_half_minus"][:, None] * normals
    return left, right


# ---------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------
def render_overlay(bgr, assembly_mask, res, alpha=0.35):
    """Annotated BGR: mask tint, traced centreline, junction dot, diameter chord, arrow."""
    H, W = bgr.shape[:2]
    diag = math.hypot(W, H)
    thick = max(2, int(0.003 * diag))
    dot_r = max(5, int(0.006 * diag))

    vis = bgr.astype(np.float32)
    vis[assembly_mask] = (1 - alpha) * vis[assembly_mask] + alpha * np.array(COL_CABLE, np.float32)
    vis = vis.clip(0, 255).astype(np.uint8)

    if res is None:
        cv2.putText(vis, "NO JUNCTION", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.0016 * diag, (0, 0, 255), thick, cv2.LINE_AA)
        return vis

    inv = 1.0 / res["_scale"]
    path_full = (res["_path"][:, ::-1].astype(np.float64) * inv).astype(np.int32)  # -> (x, y)

    # detected cable outline: both boundary rails at the perpendicular ray hits
    left, right = cable_outline(res)
    for rail in (left, right):
        rail_xy = (rail[:, ::-1] * inv).astype(np.int32)  # (y,x)->(x,y), to full res
        cv2.polylines(vis, [rail_xy], False, COL_OUTLINE, max(1, thick // 2), cv2.LINE_AA)

    cv2.polylines(vis, [path_full], False, COL_PATH, max(1, thick // 2), cv2.LINE_AA)

    u, v = res["junction"]
    dx, dy = res["direction"]
    p0 = (int(round(u)), int(round(v)))

    # diameter chord: perpendicular to the path, length = connector diameter at the junction
    perp = np.array([-dy, dx])
    r = 0.5 * res["connector_diameter_px"]
    c0 = (int(round(u - perp[0] * r)), int(round(v - perp[1] * r)))
    c1 = (int(round(u + perp[0] * r)), int(round(v + perp[1] * r)))
    cv2.line(vis, c0, c1, COL_DIA, max(1, thick // 2), cv2.LINE_AA)

    L = 0.12 * diag
    p1 = (int(round(u + dx * L)), int(round(v + dy * L)))
    cv2.arrowedLine(vis, p0, p1, COL_ARROW, thick, cv2.LINE_AA, tipLength=0.22)
    cv2.circle(vis, p0, dot_r, COL_JUNCTION, -1, cv2.LINE_AA)
    cv2.circle(vis, p0, dot_r, (0, 0, 0), max(1, thick // 2), cv2.LINE_AA)

    lbl = (f"junction {res['angle_deg']:+.0f}deg  "
           f"dia {res['cable_diameter_px']:.0f}->{res['connector_diameter_px']:.0f}px  "
           f"x{res['contrast']:.1f}")
    fscale = 0.0011 * diag
    fthick = max(2, thick // 2)
    (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fscale, fthick)
    ox, oy = p0[0] + dot_r + 6, max(th + 6, p0[1] - dot_r)
    cv2.rectangle(vis, (ox - 4, oy - th - 6), (ox + tw + 4, oy + bl), (0, 0, 0), -1)
    cv2.putText(vis, lbl, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, fscale, COL_ARROW,
                fthick, cv2.LINE_AA)
    return vis


def save_profile_plot(res, path_png, title=""):
    """Diameter-vs-arclength plot with the detected junction marked (debug aid)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inv = 1.0 / res["_scale"]
    s = res["_arclen"] * inv
    d = res["_dia"] * inv
    k = res["_junction_k"]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    # Shade the self-crossings FIRST, under the curve: a spike inside a shaded band is two
    # fused strands, not a connector, and seeing that is the whole point of this plot.
    xing = res.get("_crossing")
    if xing is not None and np.any(xing):
        for a, b in _contiguous_runs(np.asarray(xing, dtype=bool)):
            ax.axvspan(s[a], s[b], color="#ff7f0e", alpha=0.18, lw=0,
                       label="_" if a else "self-crossing")
    ax.plot(s, d, color="#1f77b4", lw=1.6, label="diameter d(s)")
    ax.axvline(s[k], color="#d62728", lw=1.8, ls="--", label="junction")
    ax.axhline(res["cable_diameter_px"], color="#7f7f7f", lw=0.9, ls=":")
    ax.axhline(res["connector_diameter_px"], color="#7f7f7f", lw=0.9, ls=":")
    ax.set_xlabel("arc length along cable (px)")
    ax.set_ylabel("diameter (px)")
    ax.set_title(title + f"  contrast x{res['contrast']:.1f}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)
