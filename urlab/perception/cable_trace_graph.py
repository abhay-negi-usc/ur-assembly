"""Skeleton-graph cable tracing -- walk THROUGH a self-crossing instead of shortcutting it.

WHY THIS EXISTS. junction.trace_centerline traces the shape's GEODESIC DIAMETER:
BFS to the farthest point A, BFS from A to the farthest B, then descend the distance field
back. That is exact and cheap for a cable lying in a simple arc, and wrong in two ways the
moment the cable crosses over itself:

  * IT SHORTCUTS. A geodesic path takes the cheapest route through the crossing, so an entire
    lobe of a loop is never traced. If the connector sits on the skipped lobe it never enters
    the diameter profile at all, and the traced "tips" (used to decide which side the
    connector is on) are not the physical tips.
  * IT HIDES THE CROSSING. The path emerges with no record that it passed through a place
    where two strands are fused, so the width spike there is indistinguishable from a real
    cable->connector diameter step -- which is exactly how a crossing gets reported as the
    junction, with HIGH contrast, because contrast is measured on whichever bump won.

So instead of one shortest path, build the skeleton's GRAPH: branches between nodes, with the
degree->=3 nodes marked. At a crossing a cable passes STRAIGHT THROUGH -- that is a physical
fact about a stiff-ish cable, and it is the only signal available, since the two strands are
photometrically identical. So pair the branches meeting at a node by TANGENT CONTINUITY and
walk from one to its partner. The strand comes out whole, in order, and every place the walk
crossed a node is FLAGGED, so the diameter profile downstream knows which samples are
contaminated rather than having to guess.

WHAT THIS DOES NOT FIX. The width measured AT a crossing is still wrong -- two fused strands
have no background between them, so a perpendicular ray runs through both. This module's job
is to say WHERE that happens; suppressing it is find_junction_index's (see the `crossing`
argument in junction.py). A tight hairpin whose two legs have MERGED into one blob is genuinely
ambiguous -- the branches enter the node nearly parallel rather than nearly opposite, so no
pairing is made and the walk stops there. Refusing is the right answer: nothing in the image
says which way the cable went.

Pure numpy + scipy on purpose -- no OpenCV, no torch, no SAM3 -- so it can be exercised on
synthetic masks in the test suite with no GPU and no model checkout.
"""

import numpy as np
from scipy import ndimage

_K8 = np.ones((3, 3), np.uint8)


# ---------------------------------------------------------------------------------------
# thinning
# ---------------------------------------------------------------------------------------
_RING = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _tables():
    """Precompute, for all 256 neighbourhood patterns, what each pass may delete.

    Bit k of the code is _RING[k] -- N, NE, E, SE, S, SW, W, NW, i.e. Zhang-Suen's P2..P9.
    Doing this once turns each thinning pass into a gather plus a table lookup, instead of two
    dozen whole-image array operations.
    """
    zs0 = np.zeros(256, dtype=bool)
    zs1 = np.zeros(256, dtype=bool)
    solo = np.zeros(256, dtype=bool)
    adj = [[max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 for b in _RING] for a in _RING]
    for c in range(256):
        bit = [(c >> k) & 1 for k in range(8)]
        B = sum(bit)
        A = sum(1 for k in range(8) if bit[k] == 0 and bit[(k + 1) % 8] == 1)
        base = (2 <= B <= 6) and A == 1
        # P2 P4 P6 / P4 P6 P8  and  P2 P4 P8 / P2 P6 P8
        zs0[c] = base and not (bit[0] and bit[2] and bit[4]) \
            and not (bit[2] and bit[4] and bit[6])
        zs1[c] = base and not (bit[0] and bit[2] and bit[6]) \
            and not (bit[0] and bit[4] and bit[6])
        # Are the set neighbours already 8-connected to one another, without the centre?
        on = [k for k in range(8) if bit[k]]
        if len(on) >= 2:
            seen, stack = {on[0]}, [on[0]]
            while stack:
                i = stack.pop()
                for j in on:
                    if j not in seen and adj[i][j]:
                        seen.add(j)
                        stack.append(j)
            solo[c] = len(seen) == len(on)
    return zs0, zs1, solo


_ZS0, _ZS1, _SOLO = _tables()


def _codes(flat, idx, offs):
    """The 8-bit neighbourhood code of every pixel in `idx` (indices into a padded flat image)."""
    code = np.zeros(len(idx), dtype=np.uint8)
    for k, off in enumerate(offs):
        code |= flat[idx + off].astype(np.uint8) << k
    return code


def thin(mask, max_iter=200):
    """Zhang-Suen thinning -> a 1-px-wide skeleton, 8-connected, topology preserved.

    Hand-rolled rather than skimage.morphology.skeletonize because scikit-image is only in
    sam3's `notebooks`/`train` extras, not the base install -- this file must import on a plain
    deployment.

    Works on the FOREGROUND INDICES, not the whole image. A cable is a percent or two of its
    frame, so a whole-image pass spends 98% of its time on background that can never change;
    at work_dim 1024 that difference is a second per call, and compute_junction is called once
    per visible cable.

    The skeleton is the MEDIAL AXIS, which matters beyond the graph: it sits in the middle of
    the cross-section, whereas a geodesic path hugs the INSIDE of a bend. The perpendicular
    width ray therefore starts centred, and the bend over-read shrinks.
    """
    pad = np.pad(np.asarray(mask, dtype=bool), 1)
    flat = pad.ravel()
    W = pad.shape[1]
    offs = np.array([dy * W + dx for dy, dx in _RING], dtype=np.intp)

    idx = np.flatnonzero(flat)
    for _ in range(int(max_iter)):
        changed = False
        for table in (_ZS0, _ZS1):
            if idx.size == 0:
                break
            kill = table[_codes(flat, idx, offs)]
            if kill.any():
                flat[idx[kill]] = False
                idx = idx[~kill]
                changed = True
        if not changed:
            break
    return _drop_redundant(pad, flat, idx, offs)[1:-1, 1:-1]


def _drop_redundant(pad, flat, idx, offs):
    """Delete pixels Zhang-Suen leaves behind at a staircase, and ONLY those.

    Zhang-Suen does not fully thin a diagonal: at every step it leaves a doubled pixel, so a
    plain arc comes out with a degree-3 pixel at each stair. That is fatal here -- the graph
    reads each one as a branch node and shatters the cable into dozens of stubs.

    A pixel is redundant exactly when its foreground neighbours are already 8-connected to each
    other WITHOUT it (so deleting it joins nothing up) and it is not a line end. The ring-order
    crossing number A used above cannot see this: it walks N, NE, E ... so it reads N and W as
    separated by a background NW, when N and W are in fact diagonal neighbours. _SOLO tests real
    8-adjacency among the neighbours instead.

    Endpoints (one neighbour) are never redundant by that test, so a curve is not eaten from its
    tips inward, and a genuine Y/X node has arms that are NOT mutually adjacent, so it survives
    -- which is the point, since those nodes are what the graph is built from.

    The candidates are found in one vectorised sweep but deleted ONE AT A TIME, re-testing
    against the live image: the two pixels of a stair are each redundant GIVEN the other, and a
    parallel pass would take both and break the curve.
    """
    for _ in range(8):
        if idx.size == 0:
            break
        cand = idx[_SOLO[_codes(flat, idx, offs)]]
        if cand.size == 0:
            break
        for i in cand:                       # sequential: re-test against the live image
            code = 0
            for k, off in enumerate(offs):
                if flat[i + off]:
                    code |= 1 << k
            if _SOLO[code]:
                flat[i] = False
        idx = idx[flat[idx]]
    return pad


def _nbr_count(sk):
    """Number of 8-connected skeleton neighbours of each skeleton pixel (0 off-skeleton)."""
    a = sk.astype(np.uint8)
    return (ndimage.convolve(a, _K8, mode="constant", cval=0) - a) * a


# ---------------------------------------------------------------------------------------
# skeleton -> graph
# ---------------------------------------------------------------------------------------
def _order_run(pixels):
    """Order a set of degree<=2 skeleton pixels into a path (N,2), tip -> tip.

    `pixels` is an (N,2) int array forming a simple 8-connected curve (possibly a closed
    loop). Starts at an end pixel when there is one, else anywhere (a loop has no end).
    """
    if len(pixels) <= 2:
        return np.asarray(pixels, dtype=np.int32)
    have = {(int(y), int(x)) for y, x in pixels}

    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and (y + dy, x + dx) in have]

    ends = [p for p in have if len(nbrs(p)) <= 1]
    start = min(ends) if ends else min(have)
    path, seen, cur = [start], {start}, start
    while True:
        nxt = [q for q in nbrs(cur) if q not in seen]
        if not nxt:
            break
        # Prefer a 4-neighbour: on a staircase the diagonal can skip a pixel and strand it.
        cur = min(nxt, key=lambda q: (abs(q[0] - cur[0]) + abs(q[1] - cur[1]), q))
        seen.add(cur)
        path.append(cur)
    return np.asarray(path, dtype=np.int32)


def prune_spurs(sk, min_len):
    """Delete short dead-end twigs, iteratively, and return the cleaned skeleton.

    Thinning sprouts a spur wherever the shape has a blunt corner -- the connector's flat end
    reliably grows two or three. Every spur is a false tip and a false degree-3 node, so the
    graph is meaningless until they are gone. A twig is a spur if it runs from a tip to a NODE
    (not to another tip -- that would be the whole cable) and is shorter than `min_len`.

    PADDED BY ONE, because the walk below indexes y+-1 / x+-1 raw. A cable that runs OUT OF FRAME
    puts skeleton pixels on the last row or column, and stepping off the array there is an
    IndexError rather than a wrong answer -- so it survived every synthetic test, where the cable
    was always drawn clear of the border, and crashed on the first real frame.
    """
    sk = np.pad(np.asarray(sk, dtype=bool), 1)
    for _ in range(12):
        deg = _nbr_count(sk)
        tips = list(zip(*np.nonzero(sk & (deg == 1))))
        if len(tips) <= 2:
            break                                  # a simple curve: nothing to prune
        doomed = []
        for tip in tips:
            run, cur, prev = [tip], tip, None
            while True:
                y, x = cur
                nxt = [(y + dy, x + dx)
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                       if (dy or dx) and sk[y + dy, x + dx] and (y + dy, x + dx) != prev]
                if len(nxt) != 1:
                    break                          # a node (>1) or a dead end (0)
                prev, cur = cur, nxt[0]
                if deg[cur] >= 3:
                    break
                run.append(cur)
                if len(run) > min_len:
                    break
            if len(run) <= min_len and deg[cur] >= 3:
                doomed.extend(run)
        if not doomed:
            break
        if len(tips) - len({t for t in tips if t in set(doomed)}) < 2:
            break                                  # refuse to prune the shape into nothing
        for y, x in doomed:
            sk[y, x] = False
    return sk[1:-1, 1:-1]


class SkeletonGraph:
    """Branches (1-px curves) plus the node clusters they meet at.

    branches[i]   ordered (N,2) pixels of branch i
    ends[i]       (node_id_at_start, node_id_at_end); -1 means a FREE TIP
    node_px[n]    (M,2) pixels of node cluster n -- what the walk must bridge across
    """

    def __init__(self, branches, ends, node_px):
        self.branches = branches
        self.ends = ends
        self.node_px = node_px


def _build(sk, node_mask):
    """Split `sk` into branches at `node_mask` and record which node each branch end meets."""
    node_lbl, n_nodes = ndimage.label(node_mask, structure=_K8)
    br_lbl, n_br = ndimage.label(sk & ~node_mask, structure=_K8)
    node_px = [np.argwhere(node_lbl == i + 1) for i in range(n_nodes)]

    branches, ends = [], []
    for i in range(n_br):
        path = _order_run(np.argwhere(br_lbl == i + 1))
        if len(path) == 0:
            continue

        def node_at(p):
            y, x = int(p[0]), int(p[1])
            sl = node_lbl[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            hit = sl[sl > 0]
            return int(hit[0]) - 1 if hit.size else -1

        branches.append(path)
        ends.append((node_at(path[0]), node_at(path[-1])))
    return SkeletonGraph(branches, ends, node_px)


def build_graph(sk, width_px, edt=None):
    """Build the branch/node graph, absorbing OVERLAP REGIONS into single nodes.

    Thinning turns a shallow X into a pair of Y's joined by a short fat stub -- the stretch
    where the two strands are fused and the skeleton has nowhere to be but between them. Left
    as two separate degree-3 nodes, the pairing sees three branches at each and can only guess.
    So a connecting branch is ABSORBED into its neighbours (making one node of the whole
    crossing) when it is either
      * SHORT -- under about two cable widths, the ordinary near-perpendicular X, or
      * FAT -- its medial-axis radius runs well over the cable's, which is what a region of
        two overlapping strands looks like however long it is (the shallow-crossing case,
        where the stub can be many widths long).
    """
    deg = _nbr_count(sk)
    seed = sk & (deg >= 3)
    if not seed.any():
        return _build(sk, np.zeros_like(sk))

    r = max(1, int(round(0.5 * width_px)))
    node_mask = ndimage.binary_dilation(seed, _K8, iterations=r) & sk

    # Second pass: fold the fused stubs in, then re-split. `edt` must be the distance transform
    # of the MASK -- the skeleton's own transform is ~1 px everywhere and says nothing about how
    # thick the shape is there, which is the whole signal the fat test reads.
    g = _build(sk, node_mask)
    absorb = np.zeros_like(sk)
    for path, (a, b) in zip(g.branches, g.ends):
        if a < 0 or b < 0 or a == b:
            continue                                        # a free tip, or already one node
        fuse = len(path) <= max(3, int(round(2.0 * width_px)))
        if not fuse and edt is not None:
            rad = float(np.median(edt[path[:, 0], path[:, 1]]))
            fuse = rad > 0.65 * width_px                    # ~1.3x the cable RADIUS
        if fuse:
            absorb[path[:, 0], path[:, 1]] = True
    if absorb.any():
        return _build(sk, node_mask | absorb)
    return g


# ---------------------------------------------------------------------------------------
# pairing branches through a node
# ---------------------------------------------------------------------------------------
def _end_dir(path, at_start, span):
    """Unit direction (dy, dx) of a branch end, pointing AWAY from the node it meets.

    A least-squares line over a window, not a two-point secant: the last pixel or two of a
    thinned branch are pulled toward the node blob, and a secant reads that pull as the
    cable's direction.
    """
    seg = (path[:span] if at_start else path[-span:][::-1]).astype(np.float64)
    if len(seg) < 2:
        return None
    c = seg.mean(axis=0)
    _, _, vt = np.linalg.svd(seg - c, full_matrices=False)
    d = vt[0]
    if float(np.dot(d, seg[-1] - seg[0])) < 0:
        d = -d
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else None


def pair_by_continuity(dirs, min_straight=0.5):
    """Pair branch ends that CONTINUE through each other. Returns {i: j} (symmetric).

    Two ends continue if their away-directions are close to opposite, scored -u_i . u_j
    (+1 = dead straight). `min_straight` = 0.5 admits a bend of up to 60 deg across the node.

    The 4-branch case -- an X, the one this module exists for -- is settled EXACTLY: there are
    only three ways to perfectly match four ends, so all three are scored and the best wins.
    Anything else is greedy over the sorted pairs, which is what a T (one branch genuinely
    ends) or a messy multi-node needs: the straight pair is taken and the odd branch is simply
    left unpaired rather than forced onto a partner.
    """
    k = len(dirs)
    ok = [i for i in range(k) if dirs[i] is not None]
    if len(ok) < 2:
        return {}

    def sc(i, j):
        return -float(np.dot(dirs[i], dirs[j]))

    best = {}
    if len(ok) == 4:
        a, b, c, d = ok
        options = (((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c)))
        pairs = max(options, key=lambda m: sum(sc(i, j) for i, j in m))
        for i, j in pairs:
            if sc(i, j) >= min_straight:
                best[i] = j
                best[j] = i
        return best

    cand = sorted(((sc(i, j), i, j) for n, i in enumerate(ok) for j in ok[n + 1:]),
                  reverse=True)
    for s, i, j in cand:
        if s < min_straight or i in best or j in best:
            continue
        best[i] = j
        best[j] = i
    return best


# ---------------------------------------------------------------------------------------
# walking
# ---------------------------------------------------------------------------------------
def _route(region, p0, p1):
    """Shortest 8-connected path from p0 to p1 staying inside `region`, ENDPOINTS EXCLUDED.

    Used to cross a node cluster: the bridge has to lie inside the mask (the width ray is cast
    from every path pixel), so it is routed through the cluster rather than drawn straight.
    """
    ys, xs = np.nonzero(region)
    if not len(ys):
        return []
    y0, y1 = int(min(ys.min(), p0[0], p1[0])), int(max(ys.max(), p0[0], p1[0]))
    x0, x1 = int(min(xs.min(), p0[1], p1[1])), int(max(xs.max(), p0[1], p1[1]))
    sub = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    sub[ys - y0, xs - x0] = True
    a = (int(p0[0]) - y0, int(p0[1]) - x0)
    b = (int(p1[0]) - y0, int(p1[1]) - x0)
    sub[a] = sub[b] = True

    prev = {a: None}
    frontier = [a]
    while frontier and b not in prev:
        nxt = []
        for y, x in frontier:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if (dy or dx) and 0 <= q[0] < sub.shape[0] and 0 <= q[1] < sub.shape[1] \
                            and sub[q] and q not in prev:
                        prev[q] = (y, x)
                        nxt.append(q)
        frontier = nxt
    if b not in prev:
        return []
    out, cur = [], prev[b]
    while cur is not None and cur != a:
        out.append((cur[0] + y0, cur[1] + x0))
        cur = prev[cur]
    return out[::-1]


def _strand(g, partner, start, used, shape):
    """Walk one strand from branch end `start=(bi, e)`, through every paired node."""
    path, cross, cur = [], [], start
    while cur is not None and cur[0] not in used:
        bi, e = cur
        used.add(bi)
        seg = g.branches[bi] if e == 0 else g.branches[bi][::-1]
        if path:
            node = g.ends[bi][e]
            region = np.zeros(shape, dtype=bool)
            if node >= 0:
                px = g.node_px[node]
                region[px[:, 0], px[:, 1]] = True
            bridge = _route(region, path[-1], seg[0])
            path.extend(bridge)
            cross.extend([True] * len(bridge))
        path.extend([tuple(p) for p in seg])
        cross.extend([False] * len(seg))
        cur = partner.get((bi, 1 - e))
    return np.asarray(path, dtype=np.int32), np.asarray(cross, dtype=bool)


def trace_strands(mask, min_straight=0.5):
    """Every maximal strand of `mask`, each walked THROUGH its crossings.

    Returns ([(path (N,2) int, crossing (N,) bool), ...] longest-first, the width px the graph
    was built at, the pruned skeleton's pixel count). An empty list means no usable skeleton.
    """
    mask = np.asarray(mask, dtype=bool)
    sk = thin(mask)
    if sk.sum() < 4:
        return [], 0.0, 0

    edt = ndimage.distance_transform_edt(mask)
    width_px = float(2.0 * np.median(edt[sk])) if sk.any() else 0.0
    width_px = max(width_px, 2.0)

    sk = prune_spurs(sk, min_len=max(3, int(round(1.5 * width_px))))
    n_skel = int(sk.sum())
    g = build_graph(sk, width_px, edt)
    if not g.branches:
        return [], width_px

    # Pair the branch ends meeting at each node by tangent continuity.
    span = max(4, int(round(2.0 * width_px)))
    at_node = {}
    for bi, (a, b) in enumerate(g.ends):
        for e, node in ((0, a), (1, b)):
            if node >= 0:
                at_node.setdefault(node, []).append((bi, e))
    partner = {}
    for node, members in at_node.items():
        dirs = [_end_dir(g.branches[bi], e == 0, span) for bi, e in members]
        for i, j in pair_by_continuity(dirs, min_straight).items():
            partner[members[i]] = members[j]

    starts = [(bi, e) for bi, (a, b) in enumerate(g.ends)
              for e, node in ((0, a), (1, b)) if node < 0]
    used, out = set(), []
    for s in starts:                                   # tip-rooted strands first
        if s[0] in used:
            continue
        p, c = _strand(g, partner, s, used, mask.shape)
        if len(p) >= 4:
            out.append((p, c))
    for bi in range(len(g.branches)):                  # then anything left (closed loops)
        if bi in used:
            continue
        p, c = _strand(g, partner, (bi, 0), used, mask.shape)
        if len(p) >= 4:
            out.append((p, c))

    out.sort(key=lambda pc: float(np.hypot(*np.diff(pc[0], axis=0).T).sum()) if len(pc[0]) > 1
             else 0.0, reverse=True)
    return out, width_px, n_skel


def trace_centerline_graph(mask, min_straight=0.5):
    """The longest strand of `mask`, walked through its crossings.

    Drop-in for junction.trace_centerline's path, PLUS the crossing flags it cannot
    produce. Returns (path (N,2) int32, crossing (N,) bool, info) or (None, None, info) if the
    shape has no usable skeleton -- the caller falls back to the geodesic trace.

    LONGEST, not "the geodesic diameter": with the crossings walked through rather than cut,
    the longest strand IS the cable, and it reaches both physical tips.
    """
    strands, width_px, n_skel = trace_strands(mask, min_straight)
    info = dict(n_strands=len(strands), width_px=float(width_px), skeleton_px=int(n_skel))
    if not strands:
        return None, None, info
    path, crossing = strands[0]
    info["n_crossings"] = int(np.count_nonzero(np.diff(crossing.astype(np.int8)) > 0))
    info["crossing_px"] = int(crossing.sum())
    # How much of the skeleton the winning strand actually accounts for. Near 1 when the walk
    # came out whole; LOW means the skeleton shattered (a mask so ragged that thinning sprouted
    # more branches than pruning could clear), and the strand is a fragment. That is the signal
    # compute_junction gates the fallback on -- a fragment is worse than a geodesic shortcut,
    # because it is short AND gives no hint that it is short.
    info["coverage"] = float(len(path) / max(1, n_skel))
    return path, crossing, info
