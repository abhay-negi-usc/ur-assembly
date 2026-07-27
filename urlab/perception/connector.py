"""ConnectorEstimator -- multi-view fusion of 2D neck/tip detections into a 3D connector pose.

This is the port of connector_pose_node.py. The math is unchanged (it was already pure numpy);
what changes is that the camera pose now arrives WITH each detection instead of being fetched
afterwards from a tf buffer, which is what made the whole class of staleness bug possible.

THE GEOMETRY, because it dictates how the scan must move:

  origin -- triangulated from the viewing rays. Needs the camera to TRANSLATE between views; a
      baseline perpendicular to the viewing ray is what makes it well conditioned.

  axis -- the null space of the stacked back-projected plane normals. Each view says "the axis
      lies in THIS plane"; two non-parallel planes intersect in a line, and that line is the axis.

  CAMERA ROTATION CONTRIBUTES NOTHING. Spinning the camera in place changes neither the ray nor
      the plane -- it is the same photon bundle relabelled. Only TRANSLATION adds information,
      and specifically translation along the connector's y (perpendicular to both the axis and
      the viewing ray) is what sharpens the axis. Translation ALONG the axis adds nothing to it.
      This is why the scan orbits rather than merely rotating, and it is why more viewpoints of
      the same kind stop helping.

  roll about the axis is NOT OBSERVABLE. A cylinder looks the same spun about its own axis.
      `up_axis` pins it -- that is a convention, not a measurement.

OUTLIER REJECTION. Background cables and connectors are REAL objects producing perfectly valid
rays; nothing about a single view marks them as wrong. Only cross-view agreement separates them,
so RANSAC scores a hypothesis by HOW MANY DISTINCT VIEWS support it -- not how many detections.
One frame full of spurious masks therefore cannot outvote a genuine multi-view cluster, which a
raw-count score would let it do.
"""

import math

import numpy as np

from .. import log as urlog
from ..transforms import frame_from_axis, inverse

log = urlog.get('connector')


def ray_point_distance(C, g, P):
    """Perpendicular distance from point P to the ray (C, g). g must be a unit vector."""
    w = np.asarray(P, dtype=float) - np.asarray(C, dtype=float)
    return float(np.linalg.norm(w - np.dot(w, g) * g))


def triangulate(centers, rays):
    """Least-squares point closest to a bundle of rays.

    Minimises sum_k ||(I - g_k g_k^T)(P - C_k)||^2 -- the squared perpendicular distance to each
    ray. Singular (returns None) when the rays are parallel, which is exactly the degenerate case
    of a camera that rotated but never translated."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for C, g in zip(centers, rays):
        P_perp = np.eye(3) - np.outer(g, g)
        A += P_perp
        b += P_perp @ np.asarray(C, dtype=float)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


class Detection:
    """One 2D neck/tip observation, lifted into 3D rays and planes at ingestion time."""

    def __init__(self, u, v, yaw, K, T_base_cam, view_id, stamp):
        self.view = view_id
        self.stamp = stamp
        self.p = np.array([u, v], dtype=float)
        self.d = np.array([math.cos(yaw), math.sin(yaw)])      # pixel-frame direction (v is DOWN)
        self.C = np.array(T_base_cam[:3, 3], dtype=float)      # camera centre in base
        self.R = np.array(T_base_cam[:3, :3], dtype=float)     # base <- camera
        self.K = K

        # The viewing ray through the detected pixel, in the base frame.
        g = self.R @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))
        self.g = g / (np.linalg.norm(g) + 1e-12)

        # The image line through p with direction d is l = [dy, -dx, dx*v - dy*u]: check that
        # l . [x, y, 1] = dy(x-u) - dx(y-v), which vanishes exactly when (x-u, y-v) is parallel
        # to d. A point X_c projects as x ~ K X_c, so the line constraint l^T x = 0 becomes
        # (K^T l) . X_c = 0 -- i.e. K^T l is the normal of the plane through the camera centre
        # containing that image line. The 3D axis lies in EVERY such plane.
        l = np.array([self.d[1], -self.d[0], self.d[0] * v - self.d[1] * u])
        n = self.R @ (K.T @ l)
        self.n = n / (np.linalg.norm(n) + 1e-12)


class ConnectorEstimator:
    """Accumulates detections across views and fits the connector's origin + axis."""

    def __init__(self, cfg):
        c = cfg.section('connector_estimator')
        self.max_history = int(c.get('max_history', 60))
        self.min_views = int(c.get('min_views', 2))
        self.min_inlier_views = int(c.get('min_inlier_views', 3))
        self.min_parallax_deg = float(c.get('min_parallax_deg', 3.0))
        self.ransac_iters = int(c.get('ransac_iters', 200))
        self.inlier_dist = float(c.get('inlier_dist_m', 0.010))
        self.max_range = float(c.get('max_range_m', 0.80))
        self.up_axis = np.array(c.get('up_axis', [0.0, 0.0, 1.0]), dtype=float)
        # Workspace box for the triangulated origin. Defaults match the dev connector_pose_node
        # (its launch never overrode these, so it ran the [-10, 10] node default) -- effectively no
        # box; the max_range_m gate does the real far-clutter rejection. Tighten these only if you
        # want to reject a specific region, and keep z_min well below your work surface.
        self.ws_min = np.array(c.get('workspace_min', [-10.0, -10.0, -10.0]), dtype=float)
        self.ws_max = np.array(c.get('workspace_max', [10.0, 10.0, 10.0]), dtype=float)

        self.history = []
        self._view_id = 0
        self._rng = np.random.default_rng(0)
        # Views the caller has marked GOOD (close enough for a low-depth-error ray). The final fit
        # (estimate) fuses ONLY these when any are marked -- a far view's depth error grows as Z^2,
        # so fusing it pulls the origin off as hard as a close one. rough_origin still uses ALL
        # views (it only needs the direction, and the far early views are what steer the approach
        # in). Empty -> no marking in use (e.g. direct/offline use) -> estimate fuses everything.
        self.good_view_ids = set()
        # Validation gate (reject_dist_m; 0 = off). Once a CONFIDENT estimate is established, an
        # incoming detection whose viewing ray misses the estimate origin by more than this is a
        # DIFFERENT / background cable -- reject it before it enters the history, so it cannot drift
        # the steered scan off the target (RANSAC resists a lone outlier only when the good cluster
        # is tight; a noisy junction cluster + the recenter feedback loop can wander otherwise).
        # Set > inlier_dist_m so genuine refinement passes; < the target<->clutter separation.
        # `_gate_origin` is the reference; estimate() sets it on success (before that, no gating).
        self.reject_dist = float(c.get('reject_dist_m', 0.0))
        self._gate_origin = None
        self._last_inlier_views = set()          # views RANSAC last kept (cached for save_plot)

    # ------------------------------------------------------------------ ingestion
    def add_view(self, detections, K, T_base_cam, stamp):
        """Ingest one captured frame's detections.

        Every detection is kept as a candidate -- there is deliberately no per-frame "pick the
        most likely one" heuristic. An earlier version picked the detection nearest the image
        centre, which latched onto whichever background object happened to be centred and then
        confirmed itself across views. Deciding which is real is RANSAC's job, and it needs the
        losers to decide against.

        EXCEPTION -- the validation gate: once a confident estimate exists (see reject_dist), a
        detection whose ray misses the established origin by more than reject_dist is rejected here
        (a background cable), so it never enters the history or steers the next view. A frame whose
        detections are all gated out consumes no view id (returns 0)."""
        if not detections:
            return 0
        vid = self._view_id + 1
        kept, rejected = [], 0
        for (u, v, yaw) in detections:
            d = Detection(u, v, yaw, K, T_base_cam, vid, stamp)
            if (self.reject_dist > 0.0 and self._gate_origin is not None
                    and ray_point_distance(d.C, d.g, self._gate_origin) > self.reject_dist):
                rejected += 1
                continue
            kept.append(d)
        if rejected:
            log.info('  gated out %d detection(s) > %.0f mm from the established estimate '
                     '(likely a background cable).', rejected, self.reject_dist * 1000)
        if not kept:
            return 0
        self._view_id = vid
        self.history.extend(kept)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return self._view_id

    @property
    def n_views(self):
        return len({d.view for d in self.history})

    def mark_view(self, view_id, good=True):
        """Mark (or unmark) a view as GOOD -- close enough for the final fit to trust its depth.
        `view_id` is the value add_view returned. estimate() fuses only marked views (see
        good_view_ids)."""
        if not view_id:
            return
        if good:
            self.good_view_ids.add(view_id)
        else:
            self.good_view_ids.discard(view_id)

    def _fuse_history(self):
        """Detections the FINAL fit should fuse: the good (within-distance) views ONCE there are
        enough of them to fit (>= min_inlier_views), else everything.

        The threshold matters while the camera is still APPROACHING (the cable-end scan starts far,
        so early views are all "far"/unmarked): switching to the good subset the instant the FIRST
        close view lands would drop the fit to one view and fail it, stalling the approach at the
        distance boundary. Staying on all views until enough good ones accumulate avoids that -- and
        with no marking at all it fuses everything, as before (offline/direct use)."""
        if self.good_view_ids:
            h = [d for d in self.history if d.view in self.good_view_ids]
            if len({d.view for d in h}) >= self.min_inlier_views:
                return h
        return self.history

    def reset(self):
        self.history = []
        self._view_id = 0
        self.good_view_ids = set()
        self._gate_origin = None
        self._last_inlier_views = set()

    # ------------------------------------------------------------------ fitting
    def _in_workspace(self, P):
        # The finite check is UNCONDITIONAL. In the ROS version it sat behind `if max_range > 0`,
        # so setting max_range_m: 0 to disable the range gate also silently disabled the NaN
        # guard, letting a degenerate triangulation through as a "valid" pose.
        if P is None or not np.isfinite(P).all():
            return False
        return bool(np.all(P >= self.ws_min) and np.all(P <= self.ws_max))

    def _ransac(self, history=None):
        """(origin, inlier detections) or (None, []). Scored by DISTINCT VIEWS, not detections.
        Operates on `history` (defaults to the full ingested set; estimate passes the good-views
        subset)."""
        history = self.history if history is None else history
        by_view = {}
        for i, d in enumerate(history):
            by_view.setdefault(d.view, []).append(i)
        views = sorted(by_view)
        if len(views) < 2:
            return None, []

        best_P, best_inliers, best_score, best_count = None, [], 0, 0
        for _ in range(self.ransac_iters):
            va, vb = self._rng.choice(len(views), size=2, replace=False)
            ia = int(self._rng.choice(by_view[views[va]]))
            ib = int(self._rng.choice(by_view[views[vb]]))
            a, b = history[ia], history[ib]

            P = triangulate([a.C, b.C], [a.g, b.g])
            if not self._in_workspace(P):
                continue
            # Range gate against BOTH sampled cameras (the ROS version checked only the first,
            # so a hypothesis absurdly far from the second camera could still pass).
            if self.max_range > 0.0 and (np.linalg.norm(P - a.C) > self.max_range
                                         or np.linalg.norm(P - b.C) > self.max_range):
                continue

            inliers = [i for i, d in enumerate(history)
                       if ray_point_distance(d.C, d.g, P) <= self.inlier_dist]
            score = len({history[i].view for i in inliers})
            if (score, len(inliers)) > (best_score, best_count):
                best_P, best_inliers, best_score, best_count = P, inliers, score, len(inliers)

        return best_P, [history[i] for i in best_inliers]

    def rough_origin(self):
        """Best-effort connector origin (base frame) for STEERING the scan's approach BEFORE the
        strict fit converges. It is the RANSAC origin WITHOUT the inlier-count or parallax gates
        that estimate() enforces -- rough (the depth along the ray is poorly constrained at low
        parallax), but the DIRECTION from the camera to the cable is sound, which is all the
        approach needs to step and re-centre. None if there are <2 views or no origin survives the
        workspace/range gates. The final grasp still comes from the strict estimate()."""
        if self.n_views < 2:
            return None
        P, _ = self._ransac()
        return P

    def estimate(self):
        """Fit the connector pose in base_link. Returns a 4x4, or None with a logged reason.

        Refusing to publish is a FEATURE. Every gate below corresponds to a way the fit can be
        confidently wrong rather than merely noisy, and a wrong connector pose sends the gripper
        somewhere real.

        Fuses only the GOOD (within-distance) views when any are marked -- far views, whose depth
        error grows as Z^2, are kept out of the fit (they still steered the approach via
        rough_origin). With no marking it fuses everything, as before."""
        hist = self._fuse_history()
        n_views = len({d.view for d in hist})
        if n_views < self.min_views:
            log.info('Need %d views, have %d (within distance).', self.min_views, n_views)
            return None

        P, inliers = self._ransac(hist)
        if P is None:
            log.warning('RANSAC found no consistent origin across views.')
            return None

        n_inlier_views = len({d.view for d in inliers})
        if n_inlier_views < self.min_inlier_views:
            log.warning('Only %d/%d views agree (need %d) -- the detections do not converge on '
                        'one object. Likely a background cable is being tracked.',
                        n_inlier_views, self.n_views, self.min_inlier_views)
            return None

        # Parallax gate. Rays that are nearly parallel triangulate to a point whose depth is
        # essentially unconstrained -- the fit "succeeds" with a huge, invisible error bar.
        rays = [d.g for d in inliers]
        max_ang = 0.0
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                ang = math.degrees(math.acos(float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))))
                max_ang = max(max_ang, ang)
        if max_ang < self.min_parallax_deg:
            log.warning('Parallax only %.1f deg (need %.1f). The camera has not translated enough '
                        'relative to the target -- rotating in place adds NO information.',
                        max_ang, self.min_parallax_deg)
            return None

        # Refit the origin on the inliers only.
        P = triangulate([d.C for d in inliers], [d.g for d in inliers])
        if not self._in_workspace(P):
            log.warning('Refit origin is outside the workspace.')
            return None

        # Axis = null space of the stacked plane normals. The axis lies in every back-projected
        # plane, so it is orthogonal to every normal; the smallest singular vector IS that
        # direction.
        N = np.array([d.n for d in inliers])
        _, sv, Vt = np.linalg.svd(N)
        axis = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-12)

        # Sign: the null space fixes the LINE, not which way along it points away from the cable.
        # Reproject a step along +axis into each inlier view and see whether it moves the way the
        # observed 2D arrow points. Majority wins.
        vote = 0.0
        for d in inliers:
            Xc = d.R.T @ (P - d.C)
            Xc2 = d.R.T @ (P + 0.01 * axis - d.C)
            if Xc[2] <= 1e-6 or Xc2[2] <= 1e-6:        # behind the camera; it cannot vote
                continue
            px1 = d.K @ (Xc / Xc[2])
            px2 = d.K @ (Xc2 / Xc2[2])
            vote += float((px2 - px1)[:2] @ d.d)
        if vote < 0:
            axis = -axis

        T = np.eye(4)
        T[:3, :3] = frame_from_axis(axis, self.up_axis)
        T[:3, 3] = P
        self._gate_origin = P            # establish/update the validation-gate reference
        self._last_inlier_views = {d.view for d in inliers}   # cached for save_plot (read-only)

        # Conditioning report. sv[-1]/sv[-2] near 1 means the null space is not well separated --
        # the planes are nearly parallel and the axis is poorly determined even though a number
        # came out. Worth seeing, because a confident-looking axis with no support is the failure
        # mode that wastes the most time on the robot.
        cond = float(sv[-1] / (sv[-2] + 1e-12)) if len(sv) >= 2 else float('nan')
        depth = float((inliers[0].R.T @ (P - inliers[0].C))[2])
        log.info('Connector fit: %d detections / %d views, parallax %.1f deg, '
                 'axis conditioning %.3f (lower is better), depth %.3f m.',
                 len(inliers), n_inlier_views, max_ang, cond, depth)
        return T

    def fit_readonly(self):
        """A READ-ONLY fit for OBSERVERS (plots, debug overlays): the strict estimate if it
        converges, else the rough origin -- computed WITHOUT perturbing the estimator. Restores the
        shared RANSAC RNG and the validation-gate reference and silences the fit log, so an observer
        never changes the scan's own fit. Returns (T_or_None, inlier_views, is_rough)."""
        if not self.history:
            return None, set(), False
        import logging
        rng_state = self._rng.bit_generator.state
        gate = self._gate_origin
        prev_level = log.level
        log.setLevel(logging.ERROR)
        try:
            T = self.estimate()
            if T is not None:
                return T, set(self._last_inlier_views), False   # cached inliers; no 2nd RANSAC
            P = self.rough_origin()
            if P is None:
                return None, set(), False
            Tr = np.eye(4)
            Tr[:3, 3] = P
            return Tr, set(), True
        finally:
            log.setLevel(prev_level)
            self._rng.bit_generator.state = rng_state       # restore -> observer changed no RNG
            self._gate_origin = gate                        # restore -> observer moved no gate ref

    def save_plot(self, path, azimuths=(-60, 30), elev=22.0):
        """Save a 3D figure (two azimuths, ROBOT BASE-FRAME axes) of the junction FUSION -- the
        fuse-mode analogue of the reconstruction plot. It draws:

          * every view's back-projected RAY (camera centre -> through the detected junction pixel),
          * the CUMULATIVE estimate ORIGIN (red star) with its axis triad (x=red into the connector,
            y=green, z=blue),
          * the camera centres,

        coloured by role: the LATEST detection (orange, thick), RANSAC inliers (green), and outliers
        / not-fused views (grey). So you literally watch the rays converge on the estimate as views
        accumulate -- a lone outlier ray stands out, and a poorly-conditioned fit shows as rays that
        do not meet at a point. Returns True if written, False if there is no usable estimate yet.

        Uses fit_readonly(), so plotting never perturbs the scan's own fit."""
        T, inlier_views, _ = self.fit_readonly()      # READ-ONLY: no RNG/gate/log side effects
        if T is None:
            return False
        P = T[:3, 3]

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

        latest = max((d.view for d in self.history), default=0)
        pts = np.vstack([np.array([d.C for d in self.history]), P[None, :]])
        L = 0.03                                       # triad arm length (m)

        fig = plt.figure(figsize=(11.5, 5.2))
        for i, az in enumerate(azimuths):
            ax = fig.add_subplot(1, len(azimuths), i + 1, projection='3d')
            for d in self.history:
                t = float(np.dot(P - d.C, d.g))        # param of the closest point on the ray to P
                end = d.C + d.g * max(t, 1e-3)         # draw the ray up to (about) the estimate
                if d.view == latest:
                    col, lw, z = '#ff7f0e', 2.6, 5     # latest detection -- orange, thick
                elif d.view in inlier_views:
                    col, lw, z = '#2ca02c', 1.1, 3     # RANSAC inlier -- green
                else:
                    col, lw, z = '#b8b8b8', 0.8, 2     # outlier / not fused -- grey
                ax.plot([d.C[0], end[0]], [d.C[1], end[1]], [d.C[2], end[2]],
                        color=col, lw=lw, zorder=z)
                ax.scatter([d.C[0]], [d.C[1]], [d.C[2]], color=col, s=10, depthshade=False, zorder=z)
            ax.scatter([P[0]], [P[1]], [P[2]], color='red', s=55, marker='*', zorder=6)
            for col, k in zip(('#d62728', '#2ca02c', '#1f77b4'), range(3)):   # x,y,z axis triad
                v = T[:3, k] * L
                ax.plot([P[0], P[0] + v[0]], [P[1], P[1] + v[1]], [P[2], P[2] + v[2]],
                        color=col, lw=2.2, zorder=6)
            ax.set_xlabel('base X (m)')
            ax.set_ylabel('base Y (m)')
            ax.set_zlabel('base Z (m)')
            ax.set_title(f'azim {az:+.0f} deg')
            ax.view_init(elev=elev, azim=az)
            _set_equal_cube(ax, pts)
        tag = 'estimate' if inlier_views else 'rough -- not yet consistent'
        fig.suptitle(f'Junction fusion -- {self.n_views} views, {len(inlier_views)} inliers '
                     f'[{tag}]   (orange=latest, green=inlier, grey=outlier; * = estimate, '
                     f'RGB triad = axis)')
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return True


def _set_equal_cube(ax, pts):
    """EQUAL SCALING on all three axes -- 1 m in X == 1 m in Y == 1 m in Z on screen, so the geometry
    is not distorted. Equal-range cube limits centred on the data (keeping the base-frame axis
    DIRECTIONS, origin offset onto the scene) PLUS an equal aspect so the box is a true cube."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = 0.5 * (lo + hi)
    r = max(float(np.max(hi - lo)) * 0.5, 0.02)        # at least a 4 cm cube
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_aspect('equal')                         # equal DATA scaling (matplotlib >= 3.6)
    except (ValueError, NotImplementedError):
        try:
            ax.set_box_aspect((1, 1, 1))               # cubic box fallback (matplotlib >= 3.3)
        except Exception:                              # noqa: BLE001 -- older matplotlib
            pass


class ConnectorTracker:
    """Ties a detector to the estimator and publishes the result into the frame graph."""

    def __init__(self, camera, detector, estimator, frames, cfg):
        self.camera = camera
        self.detector = detector
        self.estimator = estimator
        self.frames = frames
        self.base_frame = cfg.get('base_frame', 'base_link')
        self.connector_frame = cfg.get('connector_frame', 'connector')

    def observe(self, frame=None):
        """Capture (or reuse) a frame, detect, ingest. Returns the number of detections."""
        frame = frame if frame is not None else self.camera.capture()
        if frame.T_base_cam is None:
            raise ValueError('Frame has no camera pose; construct the camera with pose_fn=.')
        dets = self.detector.detect(frame)
        self.estimator.add_view(dets, frame.K, frame.T_base_cam, frame.stamp)
        return len(dets), frame

    def publish(self):
        """Fit and publish base_link -> connector. Returns the 4x4 or None."""
        T = self.estimator.estimate()
        if T is not None:
            # Stamped NOW, not with the image time. The connector is a STATIC OBJECT pose: the
            # fit is current as of this instant, even though its inputs are seconds old. Stamping
            # it with the (already stale) image time is what made every downstream consumer see a
            # fresh estimate as expired.
            self.frames.set_observed(self.base_frame, self.connector_frame, T)
        return T

    def read(self, max_age_s=None):
        """The last published connector pose, if it is fresh enough."""
        T = self.frames.lookup(self.base_frame, self.connector_frame, max_age=max_age_s)
        if T is None:
            age = self.frames.age(self.base_frame, self.connector_frame)
            if age is None:
                log.warning("No '%s' has ever been published -- the estimator has not converged.",
                            self.connector_frame)
            else:
                log.warning("'%s' is STALE (%.1f s old, limit %.1f s).",
                            self.connector_frame, age, max_age_s)
        return T
