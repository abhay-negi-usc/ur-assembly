"""CableReconstructor -- multi-view 3D reconstruction of the cable centreline, used to REFINE the
junction pose that the grasp is taken at.

WHY THIS EXISTS (see also the design notes in connector.py):

  The ConnectorEstimator triangulates the junction as an ISOLATED point from one SAM3 junction
  pixel per view. Its two weak quantities are (a) the depth along the viewing ray (error ~ Z^2 /
  baseline) and (b) the out-of-image-plane TILT of the axis, which the 2D-line null space
  under-constrains. Both improve if we stop treating the junction as a lone point and instead treat
  it as the ENDPOINT of the whole cable curve:

    * depth transfer -- the cable is one continuous curve. Stretches of it that happen to have good
      depth conditioning constrain the fit, and because the curve is smooth that good depth
      propagates along arc length into the junction region, even where the junction pixel itself
      sits in a locally ill-conditioned (tangent-to-epipolar) spot.
    * measured tangent -- the axis becomes the 3D tangent of the fitted curve at the junction, so
      its depth component (the tilt) is real, not inferred from 2D lines.
    * measured roll (optional) -- a CURVED cable has an osculating plane near the junction, a real
      geometric feature that can pin the roll the estimator otherwise leaves to up_axis. Straight
      cable -> no plane -> fall back to up_axis.

THE HARD PART is correspondence: a smooth cable is featureless, so "the same physical point" cannot
be matched across views by appearance. Two things make it tractable here:

  1. The JUNCTION is a true landmark (SAM3 finds it), so it anchors one end of the curve.
  2. From the anchored junction we MARCH outward along the reference view's skeleton; each new
     sample's epipolar line meets each other view's skeleton at the NEXT crossing past the previous
     sample's (monotone-ordering constraint). That kills the multiple-intersection ambiguity and
     the anchored start fixes the absolute correspondence.

The camera poses are KNOWN (robot FK + hand-eye), so epipolar geometry is exact and scale is
fixed -- this is correspondence + triangulation, not SfM.

The output frame convention is IDENTICAL to ConnectorEstimator's (x = cable axis pointing INTO the
connector, z ~ up, y = z x x), so a reconstruction pose is a drop-in replacement for estimate() and
every grasp offset (junction_in_fingertip / fingertip_grasp) in the config still applies unchanged.
"""

import numpy as np

from .. import log as urlog
from ..transforms import frame_from_axis
from .connector import triangulate

log = urlog.get('cable-recon')


# ---------------------------------------------------------------------- polyline helpers (pixels)
def _clean_polyline(pts):
    """Drop consecutive duplicate points so the arc length is strictly increasing."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-6:
            keep.append(i)
    return pts[keep]


def _arclen(pts):
    """Cumulative arc length (same units as pts) along a polyline, starting at 0."""
    if len(pts) < 2:
        return np.zeros(len(pts))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _resample(pts, arc, s):
    """Linear-interpolate a polyline (pts, its arc field) at arc-length positions `s` -> Mx2."""
    u = np.interp(s, arc, pts[:, 0])
    v = np.interp(s, arc, pts[:, 1])
    return np.column_stack([u, v])


def _line_crossings(a, b, skel, arc):
    """Where the infinite 2D line through a,b crosses the polyline `skel`.

    Returns [(uv, arc_at_crossing), ...]. The line's normal is n=(dy,-dx) for direction (dx,dy);
    a segment is crossed where the signed distance n.(q-a) changes sign between its endpoints."""
    d = np.asarray(b, float) - np.asarray(a, float)
    nrm = float(np.hypot(d[0], d[1]))
    if nrm < 1e-9:
        return []
    n = np.array([d[1], -d[0]]) / nrm
    f = (skel - np.asarray(a, float)) @ n          # signed distance of each vertex to the line
    out = []
    for j in range(len(skel) - 1):
        f0, f1 = f[j], f[j + 1]
        if f0 == 0.0:
            out.append((skel[j].copy(), float(arc[j])))
        if (f0 < 0) != (f1 < 0) and (f1 - f0) != 0.0:
            t = f0 / (f0 - f1)
            uv = skel[j] + t * (skel[j + 1] - skel[j])
            s = arc[j] + t * (arc[j + 1] - arc[j])
            out.append((uv, float(s)))
    return out


def _point_polyline_dist(uv, skel):
    """Shortest distance from a 2D point to a polyline (min over its segments)."""
    p = np.asarray(uv, float)
    seg0 = skel[:-1]
    seg1 = skel[1:]
    d = seg1 - seg0
    L2 = np.einsum('ij,ij->i', d, d)
    L2 = np.where(L2 < 1e-12, 1e-12, L2)
    t = np.clip(np.einsum('ij,ij->i', p - seg0, d) / L2, 0.0, 1.0)
    proj = seg0 + t[:, None] * d
    return float(np.min(np.linalg.norm(proj - p, axis=1)))


# ---------------------------------------------------------------------- per-view record
class _View:
    """One captured view's cable skeleton, lifted into base-frame rays on demand."""

    def __init__(self, K, T_base_cam, junction_uv, yaw, skel_uv, stamp, view_id):
        self.K = np.asarray(K, dtype=float)
        self.Kinv = np.linalg.inv(self.K)
        self.C = np.asarray(T_base_cam[:3, 3], dtype=float)     # camera centre in base
        self.R = np.asarray(T_base_cam[:3, :3], dtype=float)    # base <- camera
        self.junction = np.asarray(junction_uv, dtype=float)
        self.yaw = float(yaw)
        self.skel = _clean_polyline(skel_uv)                    # (N,2) (u,v), idx0=junction outward
        self.arc = _arclen(self.skel)
        self.stamp = stamp
        self.view = view_id

    def ray(self, uv):
        """Unit base-frame viewing ray through pixel uv."""
        g = self.R @ (self.Kinv @ np.array([uv[0], uv[1], 1.0]))
        return g / (np.linalg.norm(g) + 1e-12)

    def project(self, Xw):
        """Pixel of base-frame point Xw, or None if behind the camera."""
        Xc = self.R.T @ (np.asarray(Xw, float) - self.C)
        if Xc[2] <= 1e-6:
            return None
        uv = self.K @ (Xc / Xc[2])
        return uv[:2]


# ---------------------------------------------------------------------- result
class Reconstruction:
    """The refined junction pose plus the diagnostics the scan's stop rule needs."""

    def __init__(self, T, origin, axis, reproj_rms_px, pose_shift_m,
                 n_views, n_points, mean_support, kappa_per_m, used_plane):
        self.T = T
        self.origin = origin
        self.axis = axis
        self.reproj_rms_px = reproj_rms_px
        self.pose_shift_m = pose_shift_m
        self.n_views = n_views
        self.n_points = n_points
        self.mean_support = mean_support
        self.kappa_per_m = kappa_per_m
        self.used_plane = used_plane


# ---------------------------------------------------------------------- reconstructor
class CableReconstructor:
    """Ingests per-view cable skeletons and reconstructs the 3D centreline to refine the junction."""

    def __init__(self, cfg):
        c = cfg.section('reconstruction') if hasattr(cfg, 'section') else {}
        c = c or {}
        self.min_views = int(c.get('min_views', 3))
        self.max_history = int(c.get('max_history', 40))
        self.n_samples = int(c.get('samples', 32))
        self.max_range = float(c.get('max_range_m', 0.80))
        self.min_support_views = int(c.get('min_support_views', 2))   # total rays (incl. reference)
        self.near_m = float(c.get('depth_near_m', 0.05))              # ref-ray depth search bounds
        self.far_m = float(c.get('depth_far_m', self.max_range))
        self.epi_back_tol_px = float(c.get('epipolar_back_tol_px', 3.0))  # monotone-march slack
        self.junction_span_m = float(c.get('junction_span_m', 0.04))  # local fit window near junction
        self.use_curve_plane_roll = bool(c.get('use_curve_plane_roll', False))
        self.curvature_min = float(c.get('curvature_min_per_m', 5.0))  # 1/radius to trust the plane
        up_default = cfg.get_path('connector_estimator.up_axis', [0.0, 0.0, 1.0])
        self.up_axis = np.asarray(c.get('up_axis', up_default), dtype=float)

        self.views = []
        self._view_id = 0
        self._last_origin = None
        self._last_curve = None       # reconstructed 3D curve (base frame), for save_plot
        self._last_T = None           # refined junction pose, for save_plot
        # Views marked GOOD (within the distance threshold). reconstruct() uses only these when any
        # are marked -- close views have both better skeletons and lower depth error. Empty -> use
        # all (offline/direct use).
        self.good_view_ids = set()

    def reset(self):
        self.views = []
        self._view_id = 0
        self._last_origin = None
        self._last_curve = None
        self._last_T = None
        self.good_view_ids = set()

    def mark_view(self, view_id, good=True):
        """Mark (or unmark) a view as GOOD -- close enough for reconstruction to trust. `view_id` is
        what add_view returned."""
        if not view_id:
            return
        if good:
            self.good_view_ids.add(view_id)
        else:
            self.good_view_ids.discard(view_id)

    def _fuse_views(self):
        """Views reconstruct() should use: the good (within-distance) ones if any are marked, else
        all of them."""
        if self.good_view_ids:
            v = [v for v in self.views if v.view in self.good_view_ids]
            if v:
                return v
        return self.views

    @property
    def n(self):
        return len(self.views)

    # ------------------------------------------------------------------ ingestion
    def add_view(self, obs, K, T_base_cam, stamp):
        """Ingest one view's cable observation. `obs` is the dict from JunctionDetector.detect_cable:
        {'junction': (u,v), 'yaw': rad, 'skeleton': (N,2) (u,v) from the junction outward}. Returns
        the view id, or 0 if the observation is unusable (no skeleton, or no camera pose)."""
        if obs is None or T_base_cam is None:
            return 0
        skel = obs.get('skeleton')
        if skel is None or len(skel) < 4:
            return 0
        self._view_id += 1
        self.views.append(_View(K, T_base_cam, obs['junction'], obs['yaw'], skel, stamp,
                                self._view_id))
        if len(self.views) > self.max_history:
            self.views = self.views[-self.max_history:]
        return self._view_id

    # ------------------------------------------------------------------ reconstruction
    def reconstruct(self):
        """Reconstruct the cable and return a Reconstruction (refined junction pose + diagnostics),
        or None if there are too few views or the correspondence/fit does not yield a curve."""
        self._last_curve = None
        self._last_T = None
        views = self._fuse_views()                           # good (within-distance) views only
        if len(views) < self.min_views:
            return None

        ref = max(views, key=lambda v: v.arc[-1])            # longest skeleton = most cable sampled
        others = [v for v in views if v is not ref]
        if ref.arc[-1] < 1e-3 or not others:
            return None

        # Sample the reference skeleton by arc length, from the junction outward.
        s_ref = np.linspace(0.0, ref.arc[-1], self.n_samples)
        ref_pts = _resample(ref.skel, ref.arc, s_ref)

        # Anchor: the junction 3D point from every good view's junction ray (each a true landmark).
        X_j = triangulate([v.C for v in views], [v.ray(v.junction) for v in views])

        # Monotone epipolar march outward from the anchored junction.
        last_arc = {v.view: 0.0 for v in others}
        pts, support = [], []
        for p in ref_pts:
            g0 = ref.ray(p)
            centers, rays = [ref.C], [g0]
            for v in others:
                uv = self._crossing(ref, g0, v, last_arc)
                if uv is not None:
                    centers.append(v.C)
                    rays.append(v.ray(uv))
            if len(rays) < self.min_support_views:
                pts.append(None)
                support.append(len(rays))
                continue
            X = triangulate(centers, rays)
            if X is None or not self._plausible(X, ref):
                pts.append(None)
                support.append(0)
                continue
            pts.append(X)
            support.append(len(rays))

        if X_j is not None and self._plausible(X_j, ref):
            pts[0] = X_j                                     # the anchor is the best origin sample
            support[0] = len(views)

        valid = [(pts[i], support[i]) for i in range(len(pts)) if pts[i] is not None]
        if len(valid) < 3:
            log.info('  reconstruction: only %d/%d samples triangulated -- not enough curve yet.',
                     len(valid), len(pts))
            return None
        curve = np.array([X for X, _ in valid])

        origin, tangent, kappa, plane_n = self._local_frame(curve)
        if origin is None:
            return None

        # Frame convention (identical to ConnectorEstimator): x points INTO the connector, i.e.
        # opposite the cable tangent that marches AWAY from the junction along the skeleton.
        axis = -tangent
        used_plane = False
        up = self.up_axis
        if self.use_curve_plane_roll and plane_n is not None and kappa >= self.curvature_min:
            up = plane_n if float(np.dot(plane_n, self.up_axis)) >= 0 else -plane_n
            used_plane = True

        T = np.eye(4)
        T[:3, :3] = frame_from_axis(axis, up)
        T[:3, 3] = origin

        reproj = self._reproj_rms(curve, views)
        shift = (float(np.linalg.norm(origin - self._last_origin))
                 if self._last_origin is not None else float('inf'))
        self._last_origin = origin
        self._last_curve = curve
        self._last_T = T
        mean_support = float(np.mean([s for _, s in valid]))

        log.info('  reconstruction: %d views, %d curve points (mean %.1f views/point), '
                 'reproj RMS %.2f px, kappa %.1f /m%s, origin shift %.1f mm.',
                 len(views), len(valid), mean_support, reproj, kappa,
                 ' (plane roll)' if used_plane else '',
                 shift * 1000 if np.isfinite(shift) else float('nan'))
        return Reconstruction(T, origin, axis, reproj, shift, len(views), len(valid),
                              mean_support, kappa, used_plane)

    # ------------------------------------------------------------------ internals
    def _crossing(self, ref, g0, v, last_arc):
        """The next monotone epipolar crossing of the reference ray (ref.C, g0) with view v's
        skeleton, past v's last accepted arc position. Returns the crossing pixel or None."""
        a = v.project(ref.C + self.near_m * g0)
        b = v.project(ref.C + self.far_m * g0)
        if a is None or b is None:
            return None
        crossings = _line_crossings(a, b, v.skel, v.arc)
        if not crossings:
            return None
        la = last_arc[v.view]
        ahead = [(uv, s) for (uv, s) in crossings if s >= la - self.epi_back_tol_px]
        if not ahead:
            return None
        uv, s = min(ahead, key=lambda c: c[1])           # nearest crossing at or past last position
        last_arc[v.view] = s
        return uv

    def _plausible(self, X, ref):
        """In front of the reference camera and within max_range of it."""
        Xc = ref.R.T @ (np.asarray(X, float) - ref.C)
        return bool(Xc[2] > 1e-3 and np.linalg.norm(X - ref.C) <= self.max_range)

    def _local_frame(self, curve):
        """Fit the curve near the junction (curve[0]) and return (origin, unit tangent, curvature
        [1/m], osculating-plane normal or None). Parametrised by metric arc length from the
        junction; quadratic where enough points fall in the window, else linear."""
        seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        t = np.concatenate([[0.0], np.cumsum(seg)])
        window = max(self.junction_span_m, t[1] if len(t) > 1 else self.junction_span_m)
        sel = np.where(t <= window)[0]
        if len(sel) < 2:
            sel = np.arange(min(len(t), 3))
        ts, Xs = t[sel], curve[sel]
        deg = 2 if len(sel) >= 3 else 1

        coeffs = [np.polyfit(ts, Xs[:, c], deg) for c in range(3)]   # highest power first
        origin = np.array([np.polyval(cc, 0.0) for cc in coeffs])
        d1 = np.array([cc[-2] for cc in coeffs])                     # dX/dt at t=0
        n1 = float(np.linalg.norm(d1))
        if n1 < 1e-9:
            return None, None, None, None
        tangent = d1 / n1

        kappa, plane_n = 0.0, None
        if deg == 2:
            d2 = np.array([2.0 * cc[0] for cc in coeffs])            # d2X/dt2
            perp = d2 - float(np.dot(d2, tangent)) * tangent
            kappa = float(np.linalg.norm(perp)) / (n1 * n1)
            pn = np.cross(tangent, d2)
            npn = float(np.linalg.norm(pn))
            plane_n = pn / npn if npn > 1e-9 else None
        return origin, tangent, kappa, plane_n

    def _reproj_rms(self, curve, views):
        """RMS pixel distance of the reconstructed curve, reprojected into every (good) view, from
        that view's detected skeleton -- the cross-view consistency error the stop rule watches.
        Logs the per-view and overall error EACH time it is computed."""
        per_view = []
        sq_all, n_all = 0.0, 0
        for v in views:
            sq, n = 0.0, 0
            for X in curve:
                uv = v.project(X)
                if uv is None:
                    continue
                d = _point_polyline_dist(uv, v.skel)
                sq += d * d
                n += 1
            per_view.append((v.view, float(np.sqrt(sq / n)) if n else float('inf')))
            sq_all += sq
            n_all += n
        total = float(np.sqrt(sq_all / n_all)) if n_all else float('inf')
        log.info('  reprojection error: %.2f px RMS over %d view(s) [%s]', total, len(per_view),
                 ', '.join(f'v{vid}={r:.2f}' for vid, r in per_view))
        return total

    # ------------------------------------------------------------------ visualisation
    def save_plot(self, path, azimuths=(-60, 30), elev=22.0):
        """Save a figure with two 3D views (different azimuths) of the reconstructed cable points,
        plotted in ROBOT BASE-FRAME axes. The axis DIRECTIONS are base x/y/z; the origin is not the
        base origin -- the view is framed on the cable (offset in translation) so the points fill it.
        The junction pose is drawn as an RGB triad (x=red into the connector, y=green, z=blue).

        Returns True if a figure was written, False if there is no current reconstruction. Mirrors
        the junction overlay: one image per view, saved next to it."""
        if self._last_curve is None or self._last_T is None:
            return False
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

        curve = self._last_curve
        T = self._last_T
        O = T[:3, 3]
        L = 0.03                                       # triad arm length (m)
        pts_all = np.vstack([curve, O]) if curve.size else O[None, :]

        fig = plt.figure(figsize=(11.5, 5.2))
        for i, az in enumerate(azimuths):
            ax = fig.add_subplot(1, len(azimuths), i + 1, projection='3d')
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], '-', color='#9aa0a6', lw=1.2, zorder=1)
            ax.scatter(curve[:, 0], curve[:, 1], curve[:, 2],
                       c=np.arange(len(curve)), cmap='viridis', s=16, depthshade=False, zorder=2)
            ax.scatter([O[0]], [O[1]], [O[2]], color='red', s=45, marker='*',
                       label='junction', zorder=3)
            for col, k in zip(('#d62728', '#2ca02c', '#1f77b4'), range(3)):  # x,y,z triad
                d = T[:3, k] * L
                ax.plot([O[0], O[0] + d[0]], [O[1], O[1] + d[1]], [O[2], O[2] + d[2]],
                        color=col, lw=2.2)
            ax.set_xlabel('base X (m)')
            ax.set_ylabel('base Y (m)')
            ax.set_zlabel('base Z (m)')
            ax.set_title(f'azim {az:+.0f} deg')
            ax.view_init(elev=elev, azim=az)
            _set_equal_cube(ax, pts_all)
        fig.suptitle('Cable reconstruction (base-frame axes; origin offset onto the cable)')
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return True


def _set_equal_cube(ax, pts):
    """EQUAL SCALING on all three axes -- 1 m in X == 1 m in Y == 1 m in Z on screen, so the cable is
    not distorted. Equal-range cube limits centred on the data (keeping the base-frame axis
    DIRECTIONS, origin offset onto the cable) PLUS an equal aspect so the box is a true cube."""
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    c = 0.5 * (lo + hi)
    r = max(float(np.max(hi - lo)) * 0.5, 0.02)      # at least a 4 cm cube
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_aspect('equal')                       # equal DATA scaling (matplotlib >= 3.6)
    except (ValueError, NotImplementedError):
        try:
            ax.set_box_aspect((1, 1, 1))             # cubic box fallback (matplotlib >= 3.3)
        except Exception:                            # noqa: BLE001 -- older matplotlib
            pass
