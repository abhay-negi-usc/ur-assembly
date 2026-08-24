"""Cable NECK + free-END geometry -- VENDORED from sam3-abhay/scripts/cable_neck_core.py.

The companion to junction.py, and here for the same reason: this is pure numpy/OpenCV measurement
made ON a mask, so it belongs with the app that iterates on it rather than in a sibling checkout
that the test suite cannot reach.

TWO METHODS live here, both classification-free of the JUNCTION method's union trick:

  compute_necks -- pairs each CONNECTOR mask with the cable pixels touching it; the neck is the
      centroid of that contact. Iterating over CONNECTORS is its weakness: SAM3 labelling the whole
      assembly "cable" leaves no connector mask, and then there are no necks at all however good
      the cable mask is. That is what sam3.adaptive and the split thresholds exist to fight.
  compute_tip -- the cable's free END, by geodesic distance from the neck end of the mask.

"neck" and "junction" name the SAME physical point by different measurements; sam3.mode picks one.

centroid/pca_axis are imported from junction.py rather than copied again.
"""

import math

import cv2
import numpy as np
from scipy import ndimage

from .junction import centroid, pca_axis

# BGR draw colours, as in cable_neck_core.
COL_CABLE = (60, 60, 231)     # red-ish
COL_CONN = (113, 204, 46)     # green-ish
COL_NECK = (0, 215, 255)      # amber dot
COL_ARROW = (255, 200, 0)     # cyan/blue arrow

__all__ = ['masks_from_output', 'masks_scores_from_output', 'merge_overlapping', 'compute_necks',
           'compute_tip', 'render_overlay', 'render_tip_overlay', 'centroid', 'pca_axis']


def masks_from_output(out):
    """Return list of boolean HxW masks from a Sam3 processor output."""
    return [m.squeeze().cpu().numpy() > 0.5 for m in out["masks"]]


def masks_scores_from_output(out):
    """Boolean masks + their confidence scores, sorted by score DESCENDING.

    The scores are what make an adaptive threshold possible: Sam3Processor applies the threshold as a
    plain filter (`keep = out_probs > confidence_threshold`), so the forward pass does not depend on it
    at all. Run once at a low floor, keep the scores, and any threshold can then be evaluated in
    software for free. See NeckDetector.detect_adaptive.
    """
    masks = [m.squeeze().cpu().numpy() > 0.5 for m in out["masks"]]
    if "scores" in out and out["scores"] is not None and len(out["scores"]) == len(masks):
        scores = [float(s) for s in out["scores"].detach().cpu().numpy().reshape(-1)]
    else:                                     # shouldn't happen, but never lose the masks over it
        scores = [1.0] * len(masks)
    order = sorted(range(len(masks)), key=lambda i: -scores[i])
    return [masks[i] for i in order], [scores[i] for i in order]


def merge_overlapping(masks, iou_thr=0.25, ov_thr=0.55):
    """Greedily union masks that are duplicate detections of the same object
    (high IoU, or one largely contained in the other). Iterates to convergence
    so chains of overlapping detections collapse into one."""
    merged = [m.copy() for m in masks if m.sum() > 0]
    changed = True
    while changed:
        changed = False
        out = []
        for m in merged:
            hit = None
            for i, mm in enumerate(out):
                inter = int((m & mm).sum())
                if inter == 0:
                    continue
                iou = inter / int((m | mm).sum())
                ov = inter / min(int(m.sum()), int(mm.sum()))
                if iou > iou_thr or ov > ov_thr:
                    hit = i
                    break
            if hit is None:
                out.append(m)
            else:
                out[hit] = out[hit] | m
                changed = True
        merged = out
    return merged


def compute_necks(cable_masks, conn_masks_raw, H, W, mislabel_overlap=0.6):
    """Geometry only. Given boolean cable/connector masks, return a dict with the
    cleaned masks and one neck per connector.

    Returns dict(conn_masks, cleaned_cables, necks, n_dropped). Each neck is a
    dict(connector, cables, neck=[u,v], direction=[dx,dy], angle_deg, length, tip=[u,v]).
    """
    diag = math.hypot(W, H)
    band = max(4, int(0.012 * diag))                 # contact band width (px)
    min_contact = max(30, int((0.004 * diag) ** 2 / 4))
    min_cable_area = int(0.0003 * H * W)
    max_gap = 0.06 * diag                             # bridge cable<->connector gaps

    # merge duplicate detections of the same physical connector
    conn_masks = merge_overlapping(conn_masks_raw)
    conn_union = np.zeros((H, W), dtype=bool)
    for cm in conn_masks:
        conn_union |= cm

    # de-duplicate mislabels + subtract connector pixels from cables
    cleaned_cables = []
    n_dropped = 0
    for cm in cable_masks:
        area = int(cm.sum())
        if area == 0:
            continue
        if (cm & conn_union).sum() / area > mislabel_overlap:
            n_dropped += 1                            # this "cable" is a connector
            continue
        stub = cm & ~conn_union
        if stub.sum() >= min_cable_area:
            cleaned_cables.append(stub)

    cables_union = np.zeros((H, W), dtype=bool)
    for cable in cleaned_cables:
        cables_union |= cable

    necks = []
    for cj, conn in enumerate(conn_masks):
        if conn.sum() < 20:
            continue
        dist_to_conn = ndimage.distance_transform_edt(~conn)
        conn_c = centroid(conn)
        major, elong = pca_axis(conn)

        contact = cables_union & (dist_to_conn <= band)
        contact_cables = [ci for ci, c in enumerate(cleaned_cables)
                          if (c & (dist_to_conn <= band)).sum() >= min_contact]

        if contact.sum() < min_contact:
            # fallback: bridge a small gap (intermediate hardware in between)
            if cables_union.any():
                dmin = float(dist_to_conn[cables_union].min())
                if dmin <= max_gap:
                    contact = cables_union & (dist_to_conn <= dmin + band)
                    contact_cables = [ci for ci, c in enumerate(cleaned_cables)
                                      if (c & (dist_to_conn <= dmin + band)).sum() > 0]
            if contact.sum() == 0:
                continue

        neck = centroid(contact)  # (x, y)
        v_body = conn_c - neck
        if np.linalg.norm(v_body) < 1e-6:
            continue
        if elong > 1.3:
            direction = major.copy()
            if np.dot(direction, v_body) < 0:
                direction = -direction
        else:
            direction = v_body / np.linalg.norm(v_body)

        ys, xs = np.nonzero(conn)
        proj = (xs - neck[0]) * direction[0] + (ys - neck[1]) * direction[1]
        L = max(float(np.percentile(proj, 95)), 0.15 * diag)
        tip = neck + direction * L
        angle = math.degrees(math.atan2(-direction[1], direction[0]))

        necks.append(dict(
            connector=cj,
            cables=contact_cables,
            neck=[float(neck[0]), float(neck[1])],
            direction=[float(direction[0]), float(direction[1])],
            angle_deg=round(angle, 2),
            length=float(L),
            tip=[float(tip[0]), float(tip[1])],
        ))

    return dict(conn_masks=conn_masks, cleaned_cables=cleaned_cables,
                necks=necks, n_dropped=n_dropped)


def render_overlay(bgr, cleaned_cables, conn_masks, necks, alpha=0.35):
    """Return an annotated BGR image: faint masks + neck dot + orientation arrow."""
    H, W = bgr.shape[:2]
    diag = math.hypot(W, H)
    thick = max(2, int(0.0035 * diag))
    dot_r = max(4, int(0.007 * diag))

    vis = bgr.astype(np.float32)
    for cm in cleaned_cables:
        vis[cm] = (1 - alpha) * vis[cm] + alpha * np.array(COL_CABLE, np.float32)
    for cm in conn_masks:
        vis[cm] = (1 - alpha) * vis[cm] + alpha * np.array(COL_CONN, np.float32)
    vis = vis.clip(0, 255).astype(np.uint8)

    for nk in necks:
        u, v = nk["neck"]
        dx, dy = nk["direction"]
        L = nk["length"]
        p0 = (int(round(u)), int(round(v)))
        p1 = (int(round(u + dx * L)), int(round(v + dy * L)))
        cv2.arrowedLine(vis, p0, p1, COL_ARROW, thick, tipLength=0.18,
                        line_type=cv2.LINE_AA)
        cv2.circle(vis, p0, dot_r, COL_NECK, -1, cv2.LINE_AA)
        cv2.circle(vis, p0, dot_r, (0, 0, 0), max(1, thick // 2), cv2.LINE_AA)
        lx = int(u + dx * L * 0.62) + dot_r
        ly = int(v + dy * L * 0.62)
        lbl = f"C{nk['connector']}:{nk['angle_deg']:+.0f}deg"
        fscale = 0.0009 * diag
        fthick = max(2, thick // 2)
        (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fscale, fthick)
        cv2.rectangle(vis, (lx - 4, ly - th - 6), (lx + tw + 4, ly + bl + 2),
                      (0, 0, 0), -1)
        cv2.putText(vis, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, fscale,
                    COL_ARROW, fthick, cv2.LINE_AA)
    return vis


# ===================================================================================================
# CONNECTOR TIP -- classification-free
# ===================================================================================================
# SAM3 routinely labels the WHOLE assembly "cable" and returns no connector mask at all. That starves
# any connector-first pipeline: compute_necks() iterates over CONNECTOR masks, so zero connectors =>
# zero necks, no matter how good the cable mask is. The tip pipeline below therefore does not trust
# the cable/connector classification AT ALL -- it unions both prompts into one 'cable_and_connector'
# object and works purely on that object's SHAPE.

def _geodesic_bfs(mask, seed):
    """Geodesic (within-mask) pixel distance from `seed`; -1 where unreachable.

    Geodesic, NOT Euclidean: a cable is a CURVE, so two points can be adjacent in the image yet far
    apart along the cable (think of a bend that doubles back). Euclidean distance would happily jump
    the gap. Implemented as a layered dilation on the mask's bounding box -- cheap, and vectorised so
    it costs a small fraction of one SAM3 inference.
    """
    out = np.full(mask.shape, -1, dtype=np.int32)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = mask[y0:y1, x0:x1]
    dist = np.full(sub.shape, -1, dtype=np.int32)
    cur = np.zeros(sub.shape, dtype=bool)
    sy, sx = int(seed[0]) - y0, int(seed[1]) - x0
    if not (0 <= sy < sub.shape[0] and 0 <= sx < sub.shape[1] and sub[sy, sx]):
        return out
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
    out[y0:y1, x0:x1] = dist
    return out


def compute_tip(cable_masks, conn_masks_raw, H, W, curve_px=40, min_area_frac=0.0004):
    """Locate the CONNECTOR TIP from the UNION of the cable and connector segmentations.

    Steps:
      1. union both prompts -> one 'cable_and_connector' mask; keep its largest connected component,
      2. find that shape's two ENDS as its geodesic diameter (BFS from any seed -> farthest = A;
         BFS from A -> farthest = B),
      3. decide which end is the connector:
           * if SAM3 DID produce a connector mask, take the end nearest it (strongest evidence), else
           * take the THICKER end -- a connector is fatter than the cable it terminates. Thickness is
             the mean distance-transform value over the first `curve_px` of curve from that end,
      4. walk BACK along the curve from the tip by `curve_px` (the predefined curve length) and take
         `tip - back` as the connector AXIS. A fixed arc length gives a far more stable direction than
         the local tangent at the very tip, which is dominated by mask noise.

    Returns dict(mask, tip=[u,v], back=[u,v], direction=[dx,dy] (pixel frame, tip-ward), angle_deg,
    thickness_px, used_connector) -- or None if nothing usable was found.
    """
    combined = np.zeros((H, W), dtype=bool)
    for m in cable_masks:
        combined |= m
    conn_union = np.zeros((H, W), dtype=bool)
    for m in conn_masks_raw:
        conn_union |= m
    combined |= conn_union                       # <-- the classification is deliberately ignored here

    min_area = max(64, int(min_area_frac * H * W))
    if combined.sum() < min_area:
        return None

    lbl, n = ndimage.label(combined)
    if n == 0:
        return None
    sizes = ndimage.sum(combined, lbl, index=list(range(1, n + 1)))
    comp = (lbl == (int(np.argmax(sizes)) + 1))   # the cable assembly; drops unrelated blobs
    if comp.sum() < min_area:
        return None

    # --- the two ends: geodesic diameter ---
    ys, xs = np.nonzero(comp)
    dA = _geodesic_bfs(comp, (ys[0], xs[0]))
    A = np.unravel_index(int(np.argmax(np.where(dA >= 0, dA, -1))), comp.shape)
    dB = _geodesic_bfs(comp, A)
    B = np.unravel_index(int(np.argmax(np.where(dB >= 0, dB, -1))), comp.shape)

    edt = ndimage.distance_transform_edt(comp)

    def thickness_of(end):
        de = _geodesic_bfs(comp, end)
        sel = (de >= 0) & (de <= curve_px)
        return float(edt[sel].mean()) if sel.any() else 0.0

    # --- which end is the connector? ---
    if conn_union.any():
        cy, cx = np.nonzero(conn_union)
        cc = np.array([cy.mean(), cx.mean()])
        tip = A if (np.linalg.norm(np.array(A) - cc)
                    <= np.linalg.norm(np.array(B) - cc)) else B
        used_connector = True
    else:
        tip = A if thickness_of(A) >= thickness_of(B) else B
        used_connector = False

    # --- axis: walk back along the curve by the predefined length ---
    dT = _geodesic_bfs(comp, tip)
    reach = dT[dT >= 0]
    if reach.size == 0:
        return None
    lo, hi = max(1, curve_px - 3), curve_px + 3
    band = (dT >= lo) & (dT <= hi)
    if not band.any():                            # cable shorter than curve_px: use its far end
        band = (dT == int(reach.max()))
    by, bx = np.nonzero(band)
    back = np.array([by.mean(), bx.mean()])       # (y, x) centroid of the band -> stable centreline pt

    v = np.array(tip, dtype=float) - back         # (dy, dx), pointing from the body OUT to the tip
    nrm = float(np.linalg.norm(v))
    if nrm < 1e-6:
        return None
    dy, dx = v / nrm

    return dict(
        mask=comp,
        tip=[float(tip[1]), float(tip[0])],       # [u, v]
        back=[float(back[1]), float(back[0])],    # [u, v]
        direction=[float(dx), float(dy)],         # pixel frame (x right, y down), tip-ward
        angle_deg=float(math.degrees(math.atan2(-dy, dx))),
        thickness_px=float(thickness_of(tip)),
        used_connector=used_connector,
    )


def render_tip_overlay(bgr, res, alpha=0.35):
    """Overlay the combined mask, the walked-back curve segment, the tip and the axis arrow."""
    vis = bgr.copy()
    if res.get("mask") is not None:
        ov = vis.copy()
        ov[res["mask"]] = COL_CABLE
        vis = cv2.addWeighted(ov, alpha, vis, 1.0 - alpha, 0)
    if not res.get("tip"):
        cv2.putText(vis, "NO TIP", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                    cv2.LINE_AA)
        return vis
    u, v = (int(round(c)) for c in res["tip"])
    bu, bv = (int(round(c)) for c in res["back"])
    dx, dy = res["direction"]
    cv2.line(vis, (bu, bv), (u, v), COL_CONN, 2, cv2.LINE_AA)          # the curve segment measured
    cv2.arrowedLine(vis, (u, v), (int(u + 70 * dx), int(v + 70 * dy)),
                    COL_ARROW, 3, cv2.LINE_AA, tipLength=0.25)         # the connector axis
    cv2.circle(vis, (bu, bv), 4, COL_CONN, -1, cv2.LINE_AA)
    cv2.circle(vis, (u, v), 7, COL_NECK, -1, cv2.LINE_AA)
    cv2.circle(vis, (u, v), 9, (0, 0, 0), 2, cv2.LINE_AA)
    src = "conn-mask" if res.get("used_connector") else "thicker-end"
    cv2.putText(vis, f"TIP {res['angle_deg']:+.0f}deg  thick={res.get('thickness_px', 0):.0f}px  [{src}]",
                (u + 14, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_ARROW, 2, cv2.LINE_AA)
    return vis
