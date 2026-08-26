"""The vendored cable<->connector junction geometry.

These masks are DRAWN, so the true centreline and the true junction are known exactly, which is
what makes the two tracers comparable: the same shape goes into both and the answers are scored
against the same ground truth.

The graph tracer is numpy/scipy only and always runs. compute_junction needs cv2 (it is a faithful
vendoring of the sam3-abhay geometry, drawing and morphology included), so those tests skip where
OpenCV is absent -- the same convention the ArUco tests use.
"""

import numpy as np
import pytest
from scipy import ndimage

from urlab.perception.cable_trace_graph import thin, trace_centerline_graph

N = 420
K = np.ones((3, 3), bool)


# ------------------------------------------------------------------ synthetic cables
def stroke(pts, radius, n=N):
    """Rasterise a polyline (M,2 in y,x) as everything within `radius` of it."""
    seeds = np.zeros((n, n), dtype=bool)
    p = np.asarray(pts, dtype=float)
    for a, b in zip(p[:-1], p[1:]):
        for t in np.linspace(0, 1, max(2, int(np.hypot(*(b - a)) * 2))):
            q = a + t * (b - a)
            y, x = int(round(q[0])), int(round(q[1]))
            if 0 <= y < n and 0 <= x < n:
                seeds[y, x] = True
    return ndimage.distance_transform_edt(~seeds) <= radius


def arclen(p):
    p = np.asarray(p, dtype=float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def figure_eight(b=45.0, t0=0.18, n=600):
    """An OPEN cable that crosses itself once: x = 150 cos t, y = b sin 2t, trimmed to two tips.
    Smaller `b` = shallower crossing angle."""
    t = np.linspace(t0, 2 * np.pi - t0, n)
    return np.stack([b * np.sin(2 * t) + 210.0, 150.0 * np.cos(t) + 210.0], axis=1)


def cable_with_connector(pts, cable_r=5, conn_r=12, conn_len=55):
    """`pts` plus a fat barrel on the last point, laid radially outward so it stands clear.
    Returns (mask, true junction (u, v))."""
    tip = np.asarray(pts, float)[-1]
    out = tip - np.asarray(pts, float).mean(axis=0)
    out = out / np.linalg.norm(out)
    m = stroke(pts, cable_r) | stroke(np.stack([tip, tip + conn_len * out]), conn_r)
    return m, (float(tip[1]), float(tip[0]))


# ------------------------------------------------------------------ the tracer
def test_thinning_leaves_a_clean_one_pixel_skeleton():
    """Zhang-Suen alone leaves a doubled pixel at every staircase step, and each one reads as a
    degree-3 branch node -- which shatters the graph. The simple-point pass must clear them."""
    arc = np.stack([210 + 120 * np.sin(np.linspace(0, 2.6, 300)),
                    210 + 120 * np.cos(np.linspace(0, 2.6, 300))], axis=1)
    sk = thin(stroke(arc, 5))
    a = sk.astype(np.uint8)
    deg = (ndimage.convolve(a, K.astype(np.uint8), mode='constant') - a) * a
    assert int((sk & (deg >= 3)).sum()) == 0, 'a plain arc must have NO branch nodes'
    assert int((sk & (deg == 1)).sum()) == 2, 'an open arc has exactly two tips'


@pytest.mark.parametrize('b,label', [(75.0, 'steep'), (40.0, 'medium'), (22.0, 'shallow')])
def test_the_graph_tracer_walks_through_a_self_crossing(b, label):
    """The whole point: recover the WHOLE cable, not the shortcut across the loop.

    A geodesic path takes the cheapest route through the crossing and so misses roughly half of a
    figure-eight. The graph walk must come back with the cable essentially intact, as ONE strand,
    having flagged the crossing -- twice, because the strand passes through it once per lobe."""
    pts = figure_eight(b)
    path, crossing, info = trace_centerline_graph(stroke(pts, 5))
    assert path is not None, f'{label}: no strand traced'
    assert info['n_strands'] == 1, f'{label}: the cable came back in pieces'
    assert 0.95 <= arclen(path) / arclen(pts) <= 1.15, f'{label}: recovered the wrong length'
    assert info['n_crossings'] == 2, f'{label}: a figure-eight is entered twice'
    assert info['coverage'] > 0.9, f'{label}: the strand should account for the whole skeleton'

    # ... and the flags must sit ON the crossing, which for this curve is the centre.
    ys, xs = path[crossing][:, 0], path[crossing][:, 1]
    assert np.hypot(ys.mean() - 210, xs.mean() - 210) < 12, f'{label}: flags are off the crossing'


def test_a_cable_with_no_crossing_is_traced_without_flagging_one():
    """No false positives: a plain arc and a straight cable must report zero crossings, or the
    junction search would start suppressing real measurements."""
    for name, pts in (('arc', np.stack([210 + 120 * np.sin(np.linspace(0, 2.6, 300)),
                                        210 + 120 * np.cos(np.linspace(0, 2.6, 300))], axis=1)),
                      ('straight', np.stack([np.full(300, 210.0),
                                             np.linspace(60, 340, 300)], axis=1))):
        path, crossing, info = trace_centerline_graph(stroke(pts, 5))
        assert path is not None and info['n_strands'] == 1, name
        assert info['n_crossings'] == 0, f'{name}: invented a crossing'
        assert not crossing.any(), f'{name}: flagged samples on a shape with no overlap'


def test_the_trace_reaches_both_tips_of_a_looped_cable():
    """A shortcut leaves the connector untraced when it sits on the skipped lobe. Every physical
    tip must be on the strand, or the diameter profile cannot contain the connector at all."""
    pts = figure_eight(45.0)
    path, _crossing, _info = trace_centerline_graph(stroke(pts, 5))
    assert path is not None
    for tip in (pts[0], pts[-1]):
        d = float(np.min(np.hypot(path[:, 0] - tip[0], path[:, 1] - tip[1])))
        assert d < 12.0, f'tip {tip.round(0)} is {d:.0f} px from the trace'


def test_a_cable_running_out_of_frame_is_traced_without_stepping_off_the_array():
    """REGRESSION: every fixture here draws the cable clear of the border, so nothing caught that
    the spur walk indexed y+-1 / x+-1 raw. A real frame has the cable leaving the image, which puts
    skeleton pixels on the last row/column -- and that was an IndexError, not a wrong answer, so it
    took down the whole app on the first live frame.

    Each edge is exercised separately: a bug guarding only one axis passes a single-edge test."""
    def touches_edge(a):
        return bool(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())

    # A cable GRAZING the frame -- mostly outside it, leaving a thin sliver along the border --
    # is what puts a skeleton pixel there. A cable merely running OUT of frame does not: the
    # blunt end reads as an exposed end and thinning pulls the tip back about half a width.
    lo, hi = 0.0, float(N - 1)
    cases = {
        'graze right': np.stack([np.linspace(60, 360, 300), np.full(300, hi)], axis=1),
        'graze left': np.stack([np.linspace(60, 360, 300), np.full(300, lo)], axis=1),
        'graze bottom': np.stack([np.full(300, hi), np.linspace(60, 360, 300)], axis=1),
        'graze top': np.stack([np.full(300, lo), np.linspace(60, 360, 300)], axis=1),
    }
    covered = []
    for name, pts in cases.items():
        mask = stroke(pts, 5)
        covered.append(touches_edge(thin(mask)))   # not every edge does: thinning is not
        path, _crossing, info = trace_centerline_graph(mask)      # left/right symmetric
        assert path is not None and info['n_strands'] >= 1, name
    assert any(covered), 'no graze fixture reaches the border any more -- coverage has been lost'

    # And with a branch, so prune_spurs actually walks (it does nothing on a 2-tip curve, which
    # is what the walk that crashed lives inside).
    m = np.zeros((N, N), dtype=bool)
    m[100:300, -1] = True                      # a one-pixel sliver hard against the border
    m[195:205, 260:] = True                    # joined to a cable running inward -> three tips
    m |= stroke(np.stack([np.full(200, 200.0), np.linspace(60, 265, 200)], axis=1), 5)
    assert touches_edge(thin(m)), 'branch fixture puts no skeleton pixel on the border'
    path, _crossing, info = trace_centerline_graph(m)             # must not raise
    assert path is not None, 'branched border shape traced nothing'


def test_coverage_reports_a_shattered_skeleton():
    """Coverage is what compute_junction gates its fallback on, so it has to actually fall when
    the skeleton fragments. A mask chewed to pieces must not come back claiming full coverage."""
    rng = np.random.default_rng(0)
    m = stroke(figure_eight(45.0), 5)
    for _ in range(3):                       # ragged far beyond anything SAM3 produces
        m = (ndimage.binary_dilation(m, K) & (rng.random(m.shape) < 0.5)) \
            | ndimage.binary_erosion(m, K)
    lbl, k = ndimage.label(m, structure=K)
    m = lbl == (1 + int(np.argmax(ndimage.sum(m, lbl, index=np.arange(1, k + 1)))))
    path, _c, info = trace_centerline_graph(m)
    if path is not None:
        assert info['coverage'] < 0.75, 'a shattered skeleton must trip the fallback gate'


# ------------------------------------------------------------------ the junction decision
def test_geodesic_tracing_reproduces_the_original_on_a_simple_cable():
    """trace='geodesic' is the ORIGINAL method, kept so the two are comparable on one image. On a
    cable with no crossing both tracers must agree closely -- if they disagree here, the
    disagreement on a loop says nothing."""
    pytest.importorskip('cv2')
    from urlab.perception.junction import compute_junction
    pts = np.stack([np.full(300, 210.0), np.linspace(60, 300, 300)], axis=1)
    mask, true_uv = cable_with_connector(pts)
    a = compute_junction(mask, work_dim=1024, trace='geodesic')
    b = compute_junction(mask, work_dim=1024, trace='graph')
    assert a is not None and b is not None
    assert a['n_crossings'] == 0 and b['n_crossings'] == 0
    sep = np.hypot(a['junction'][0] - b['junction'][0], a['junction'][1] - b['junction'][1])
    assert sep < 12.0, f'the tracers disagree by {sep:.0f} px on a cable with no crossing'


@pytest.mark.parametrize('b', [75.0, 40.0, 22.0])
def test_a_self_crossing_is_no_longer_reported_as_the_junction(b):
    """THE REGRESSION THIS EXISTS FOR.

    Two fused strands have no background between them, so the width ray reads both and spikes.
    The original method splits the constant-diameter run on that spike and then places the
    junction at whichever bump is thicker -- and a shallow crossing is thicker than a connector,
    since its apparent width goes as 1/sin of the crossing angle. Worse, `contrast` is measured on
    the winning bump, so the wrong answer comes back looking as confident as a right one; that is
    why sam3.min_contrast cannot filter this and why the fix had to be in the tracing."""
    pytest.importorskip('cv2')
    from urlab.perception.junction import compute_junction
    mask, true_uv = cable_with_connector(figure_eight(b))

    old = compute_junction(mask, work_dim=1024, trace='geodesic', select='longest_run')
    new = compute_junction(mask, work_dim=1024, trace='graph')
    assert old is not None and new is not None

    def err(r):
        return float(np.hypot(r['junction'][0] - true_uv[0], r['junction'][1] - true_uv[1]))

    assert err(new) < 30.0, f'graph tracing put the junction {err(new):.0f} px out'
    assert err(old) > 100.0, ('geodesic tracing AND the longest-run selector together are expected '
                              'to fail here -- if they stopped, this fixture no longer reproduces '
                              'the bug')
    assert new['n_crossings'] >= 1, 'the crossing must be reported'

    # The SELECTOR fixes this independently of the tracer: slope selection measures the transition
    # and treats the crossing as transparent, so it lands on the connector even down the geodesic
    # shortcut that defeats longest_run. Neither fix now depends on the other.
    geo_slope = compute_junction(mask, work_dim=1024, trace='geodesic', select='slope')
    assert geo_slope is not None and err(geo_slope) < 30.0, (
        'slope selection should survive the geodesic shortcut '
        f'(got {err(geo_slope):.0f} px out)')
