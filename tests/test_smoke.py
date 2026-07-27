"""Smoke + math tests for the ROS-free layers -- no robot, no camera, no torch.

Run: python -m pytest tests/ -q     (or: python tests/test_smoke.py)

Covers the parts that are pure computation and therefore fully testable offline: the transform
conventions (the thing most likely to be silently wrong), the FrameGraph staleness semantics, the
config loader, and the connector-fusion geometry against a synthetic ground truth.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urlab import transforms as T   # noqa: E402
from urlab.frames import FrameGraph   # noqa: E402


def approx(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol)


# ------------------------------------------------------------------ transforms
def test_xyzrpy_roundtrip():
    xyz = [0.1, -0.2, 0.3]
    rpy = [0.3, -0.5, 1.2]
    M = T.xyzrpy_to_matrix(xyz, rpy)
    xyz2, rpy2 = T.matrix_to_xyzrpy(M)
    assert approx(xyz, xyz2) and approx(rpy, rpy2)


def test_inverse_matches_numpy():
    M = T.xyzrpy_to_matrix([0.4, 0.1, -0.2], [1.1, -0.3, 0.7])
    assert approx(T.inverse(M), np.linalg.inv(M), 1e-9)
    assert approx(T.inverse(M) @ M, np.eye(4), 1e-9)


def test_extrinsic_xyz_convention():
    # Extrinsic XYZ: R = Rz(yaw) @ Ry(pitch) @ Rx(roll). A pure yaw about +Z sends +X -> +Y.
    R = T.xyzrpy_to_matrix([0, 0, 0], [0, 0, np.pi / 2])[:3, :3]
    assert approx(R @ [1, 0, 0], [0, 1, 0], 1e-9)


def test_rtde_base_link_bridge():
    # base_link and UR base differ by Rz(pi). A UR-base point on +x lands on base_link -x.
    M = T.rtde_to_matrix([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert approx(M[:3, 3], [-1.0, 0.0, 0.0], 1e-9)
    # And the round trip is exact.
    pose = [0.3, -0.2, 0.5, 0.1, -0.2, 1.3]
    assert approx(T.matrix_to_rtde(T.rtde_to_matrix(pose)), pose, 1e-9)


def test_rotvec_roundtrip():
    pose = [0.4, 0.0, 0.4, 1.2, -0.3, 0.8]
    M = T.rtde_to_matrix(pose, ur_base=True)
    assert approx(T.matrix_to_rtde(M, ur_base=True), pose, 1e-9)


def test_look_at_points_at_target():
    T_ref = np.eye(4)
    eye = np.array([0.0, 0.0, 0.5])
    target = np.array([0.1, 0.0, 0.0])
    M = T.look_at(eye, target, T_ref)
    z = M[:3, 2]                                  # optical axis
    expect = (target - eye) / np.linalg.norm(target - eye)
    assert approx(z, expect, 1e-9)
    assert approx(np.linalg.det(M[:3, :3]), 1.0, 1e-9)   # right-handed


def test_rotate_about_axis_preserves_radius():
    T0 = T.xyzrpy_to_matrix([0.3, 0.0, 0.2], [0, 0, 0])
    P = np.array([0.0, 0.0, 0.2])
    axis = np.array([0.0, 0.0, 1.0])
    T1 = T.rotate_about_axis(T0, axis, P, np.radians(37))
    assert approx(np.linalg.norm(T1[:3, 3] - P), np.linalg.norm(T0[:3, 3] - P), 1e-9)


def test_frame_from_axis_orthonormal():
    R = T.frame_from_axis([1.0, 0.5, 0.0], [0, 0, 1])
    assert approx(R.T @ R, np.eye(3), 1e-9)
    assert approx(R[:, 0], [1.0, 0.5, 0.0] / np.linalg.norm([1.0, 0.5, 0.0]), 1e-9)


def test_transform_wrench_cross_term():
    # A pure force offset by a lever arm produces a torque; magnitude of force is preserved.
    f = np.array([0.0, 0.0, -10.0])
    tau = np.zeros(3)
    T_ba = T.translation_matrix([0.1, 0.0, 0.0])
    fb, taub = T.transform_wrench(f, tau, T_ba)
    assert approx(fb, f, 1e-9)                    # pure translation: force unchanged
    assert approx(taub, np.cross([0.1, 0, 0], f), 1e-9)


def test_clamp_pose_delta():
    T_ref = T.xyzrpy_to_matrix([0.4, 0.0, 0.4], [0, 0, 0])
    far = T_ref @ T.xyzrpy_to_matrix([0.5, 0, 0], [0, 0, 0])   # 0.5 m out in ref-x
    clamped = T.clamp_pose_delta(T_ref, far, [0.08, 0.08, 0.05], [0.3, 0.3, 0.3])
    rel = T.inverse(T_ref) @ clamped
    assert rel[0, 3] <= 0.08 + 1e-9


# ------------------------------------------------------------------ frame graph
def test_framegraph_chain():
    g = FrameGraph()
    g.set_static('a', 'b', T.translation_matrix([1, 0, 0]))
    g.set_static('b', 'c', T.translation_matrix([0, 1, 0]))
    M = g.lookup('a', 'c')
    assert approx(M[:3, 3], [1, 1, 0], 1e-9)
    # And the reverse walks the same edges inverted.
    assert approx(g.lookup('c', 'a')[:3, 3], [-1, -1, 0], 1e-9)


def test_framegraph_live_never_stale():
    g = FrameGraph()
    g.set_live('a', 'b', lambda: T.translation_matrix([2, 0, 0]))
    assert g.age('a', 'b') == 0.0
    assert g.lookup('a', 'b', max_age=0.001) is not None   # live edges never expire


def test_framegraph_observed_staleness():
    import time
    g = FrameGraph()
    g.set_observed('a', 'b', np.eye(4), stamp=time.monotonic() - 5.0)
    assert g.lookup('a', 'b', max_age=10.0) is not None
    assert g.lookup('a', 'b', max_age=2.0) is None         # older than the budget -> rejected
    assert g.lookup('a', 'b') is not None                  # no budget -> never rejected


# ------------------------------------------------------------------ config
def test_config_dotted_and_override():
    from urlab import config as C
    cfg = C.load('cartesian')
    assert cfg.get('base_frame') == 'base_link'
    cfg2 = C.load('cartesian', ['linear_step_m=0.05', 'robot.dry_run=true'])
    assert cfg2.get('linear_step_m') == 0.05
    assert cfg2.get_path('robot.dry_run') is True


# ------------------------------------------------------------------ fusion geometry
def test_connector_fusion_recovers_synthetic_axis():
    """A synthetic cable at a known pose, seen from several translated views, should be recovered
    (origin within a mm, axis within a couple of degrees)."""
    from urlab.config import Config
    from urlab.perception.connector import ConnectorEstimator

    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    P_true = np.array([0.5, 0.0, 0.2])            # connector origin in base
    axis_true = np.array([1.0, 0.2, 0.0])
    axis_true = axis_true / np.linalg.norm(axis_true)

    est = ConnectorEstimator(Config({'connector_estimator': {
        'min_inlier_views': 3, 'min_parallax_deg': 1.0, 'inlier_dist_m': 0.02,
        'max_range_m': 2.0, 'up_axis': [0, 0, 1]}}))

    # Cameras looking down (-z world) from above, translated laterally for parallax.
    for dx in (-0.08, -0.04, 0.0, 0.04, 0.08):
        C = np.array([0.5 + dx, 0.0, 0.6])
        # Optical frame: z toward the target (down), x right, y down.
        T_bc = T.look_at(C, P_true, np.eye(4))
        # Project the origin and a point along the axis to get (u, v) and the pixel-frame yaw.
        Rcw = T_bc[:3, :3].T
        def proj(Xw):
            Xc = Rcw @ (Xw - C)
            uv = K @ (Xc / Xc[2])
            return uv[:2]
        p0 = proj(P_true)
        p1 = proj(P_true + 0.03 * axis_true)
        yaw = np.arctan2(*(p1 - p0)[::-1])        # atan2(dy, dx)
        est.add_view([(p0[0], p0[1], yaw)], K, T_bc, 0.0)

    M = est.estimate()
    assert M is not None, 'fusion refused a clean synthetic case'
    assert np.linalg.norm(M[:3, 3] - P_true) < 0.005, f'origin off by {M[:3,3]-P_true}'
    axis_est = M[:3, 0]
    cos = abs(float(np.dot(axis_est, axis_true)))
    assert cos > np.cos(np.radians(5)), f'axis off by {np.degrees(np.arccos(cos)):.1f} deg'


def test_cable_reconstruction_recovers_curve():
    """A synthetic (slightly tilted) cable, seen from several translated views, should reconstruct:
    the junction origin within a cm and the axis -- INCLUDING its out-of-plane tilt -- within a few
    degrees. This is the quantity the point estimator is weakest on, so it is what the test pins."""
    from urlab.config import Config
    from urlab.perception.cable_recon import CableReconstructor

    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    P_j = np.array([0.5, 0.0, 0.2])                    # junction (connector end) in base
    dir_cable = np.array([1.0, 0.2, 0.3])             # cable heads out with a real depth component
    dir_cable = dir_cable / np.linalg.norm(dir_cable)
    s = np.linspace(0.0, 0.15, 20)
    cable3d = P_j[None, :] + s[:, None] * dir_cable[None, :]   # idx 0 = junction, outward

    rec = CableReconstructor(Config({
        'reconstruction': {'min_views': 3, 'samples': 24, 'junction_span_m': 0.05,
                           'max_reproj_error_px': 5.0},
        'connector_estimator': {'up_axis': [0, 0, 1]}}))

    for dx in (-0.08, -0.04, 0.0, 0.04, 0.08):        # lateral translation for parallax
        C = np.array([0.5 + dx, 0.0, 0.6])
        T_bc = T.look_at(C, P_j, np.eye(4))
        Rcw = T_bc[:3, :3].T

        def proj(Xw, Rcw=Rcw, C=C):
            Xc = Rcw @ (Xw - C)
            uv = K @ (Xc / Xc[2])
            return uv[:2]

        skel = np.array([proj(X) for X in cable3d])   # ordered from junction outward
        obs = {'junction': (float(skel[0, 0]), float(skel[0, 1])), 'yaw': 0.0, 'skeleton': skel}
        rec.add_view(obs, K, T_bc, 0.0)

    res = rec.reconstruct()
    assert res is not None, 'reconstruction refused a clean synthetic cable'
    assert np.linalg.norm(res.origin - P_j) < 0.01, f'origin off by {res.origin - P_j}'
    axis_true = -dir_cable                            # frame x points INTO the connector
    cos = abs(float(np.dot(res.axis, axis_true)))
    assert cos > np.cos(np.radians(6)), f'axis off by {np.degrees(np.arccos(cos)):.1f} deg'


def test_connector_estimator_fuses_only_marked_good_views():
    """When views are marked good/far, estimate() must fuse ONLY the good ones -- a far, biased view
    left unmarked should not move the origin."""
    from urlab.config import Config
    from urlab.perception.connector import ConnectorEstimator

    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    P_true = np.array([0.5, 0.0, 0.2])
    axis_true = np.array([1.0, 0.0, 0.0]) / 1.0
    est = ConnectorEstimator(Config({'connector_estimator': {
        'min_inlier_views': 3, 'min_parallax_deg': 1.0, 'inlier_dist_m': 0.02,
        'max_range_m': 3.0, 'up_axis': [0, 0, 1]}}))

    def add(C, P, good):
        T_bc = T.look_at(C, P, np.eye(4))
        Rcw = T_bc[:3, :3].T
        def proj(Xw):
            Xc = Rcw @ (Xw - C)
            return (K @ (Xc / Xc[2]))[:2]
        p0, p1 = proj(P), proj(P + 0.03 * axis_true)
        yaw = float(np.arctan2(*(p1 - p0)[::-1]))
        vid = est.add_view([(p0[0], p0[1], yaw)], K, T_bc, 0.0)
        est.mark_view(vid, good)

    for dx in (-0.06, -0.02, 0.02, 0.06):             # good, close views of the TRUE origin
        add(np.array([0.5 + dx, 0.0, 0.6]), P_true, good=True)
    # A far view whose rays are consistent with a DIFFERENT (biased) origin -- must be excluded.
    P_bias = P_true + np.array([0.10, 0.0, 0.0])
    for dx in (-0.05, 0.05):
        add(np.array([0.5 + dx, 0.0, 1.2]), P_bias, good=False)

    M = est.estimate()
    assert M is not None
    assert np.linalg.norm(M[:3, 3] - P_true) < 0.01, \
        f'far unmarked view leaked into the fit: origin {M[:3,3]} vs {P_true}'


def test_connector_estimator_validation_gate_rejects_background():
    """Once a confident estimate is established, a detection of a DIFFERENT (background) cable is
    gated out of the history; a consistent detection still lands."""
    from urlab.config import Config
    from urlab.perception.connector import ConnectorEstimator

    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    P_true = np.array([0.5, 0.0, 0.2])
    axis = np.array([1.0, 0.0, 0.0])
    est = ConnectorEstimator(Config({'connector_estimator': {
        'min_inlier_views': 3, 'min_parallax_deg': 1.0, 'inlier_dist_m': 0.02,
        'max_range_m': 3.0, 'reject_dist_m': 0.05, 'up_axis': [0, 0, 1]}}))

    def view_of(P, dx):
        C = np.array([0.5 + dx, 0.0, 0.6])
        T_bc = T.look_at(C, P, np.eye(4))
        Rcw = T_bc[:3, :3].T
        def proj(X):
            Xc = Rcw @ (X - C)
            return (K @ (Xc / Xc[2]))[:2]
        p0, p1 = proj(P), proj(P + 0.03 * axis)
        return (float(p0[0]), float(p0[1]), float(np.arctan2(*(p1 - p0)[::-1]))), K, T_bc

    for dx in (-0.06, -0.02, 0.02, 0.06):                # establish the estimate on the true cable
        det, k, tbc = view_of(P_true, dx)
        est.add_view([det], k, tbc, 0.0)
    assert est.estimate() is not None                    # confident fit -> gate armed
    n = est.n_views

    det_bg, k, tbc = view_of(P_true + np.array([0.30, 0.0, 0.0]), 0.0)   # a cable 30 cm away
    assert est.add_view([det_bg], k, tbc, 0.0) == 0, 'background detection should be gated out'
    assert est.n_views == n, 'gated detection must not enter the history'

    det_ok, k, tbc = view_of(P_true, 0.0)                # a consistent detection still lands
    assert est.add_view([det_ok], k, tbc, 0.0) != 0


def test_estimator_ransac_picks_real_connector_over_background():
    """Ingest ALL cable junctions each view (one per cable) and let RANSAC decide: the connector
    seen consistently across views wins; a background cable seen in only one view is outvoted."""
    from urlab.config import Config
    from urlab.perception.connector import ConnectorEstimator

    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    P_real = np.array([0.5, 0.0, 0.2])
    est = ConnectorEstimator(Config({'connector_estimator': {
        'min_inlier_views': 3, 'min_parallax_deg': 1.0, 'inlier_dist_m': 0.02,
        'max_range_m': 3.0, 'up_axis': [0, 0, 1]}}))

    def det(P, dx):
        C = np.array([0.5 + dx, 0.0, 0.6])
        T_bc = T.look_at(C, P, np.eye(4))
        Rcw = T_bc[:3, :3].T
        p = (K @ (Rcw @ (np.asarray(P, float) - C)))[:2] / (Rcw @ (P - C))[2]
        return (float(p[0]), float(p[1]), 0.0), K, T_bc

    # Each view sees the REAL connector plus a DIFFERENT phantom (background) point -- both ingested.
    for i, dx in enumerate((-0.06, -0.02, 0.02, 0.06)):
        real, k, tbc = det(P_real, dx)
        phantom, _, _ = det(P_real + np.array([0.0, 0.20 + 0.03 * i, 0.0]), dx)  # inconsistent
        est.add_view([real, phantom], k, tbc, 0.0)

    T_fit = est.estimate()
    assert T_fit is not None
    assert np.linalg.norm(T_fit[:3, 3] - P_real) < 0.02, 'RANSAC should lock onto the real connector'


class _FakeGripper:
    """Scripted gripper: each close() reports the next count in `on_close`; go_to sets the count."""

    def __init__(self, on_close):
        self.on_close = list(on_close)
        self.pos = 0
        self._i = 0
        self.dry_run = False

    def close(self, label='close'):
        self.pos = self.on_close[min(self._i, len(self.on_close) - 1)]
        self._i += 1
        return True

    def go_to(self, counts, label='', wait=True):
        self.pos = int(counts)
        return True

    def position(self):
        return self.pos

    def grasp_result(self, groove_counts, empty_counts, faces_max_counts, tolerance=1,
                     detect_empty=True):
        if self.pos <= faces_max_counts:
            return 'missed'
        if detect_empty and self.pos >= empty_counts - tolerance:
            return 'empty'
        return 'ok'


class _FakeRobot:
    def __init__(self, gripper):
        self.gripper = gripper
        self.moves = []

    def move_fingertip(self, T, label='move'):
        self.moves.append((label, np.array(T, dtype=float)))
        return True


def _recovery_cfg(**recovery):
    from urlab.config import Config
    return Config({
        'approach_axis': [0.0, 0.0, 1.0], 'approach_distance_m': 0.10,
        'grasp_check': {'enabled': True, 'groove_counts': 225, 'empty_counts': 228,
                        'faces_max_counts': 223, 'tolerance_counts': 1, 'detect_empty': True,
                        'settle_s': 0.0,
                        'recovery': {'enabled': True, 'increment_m': 0.0005, 'loose_counts': 215,
                                     'faces_counts': 223, 'faces_band_counts': 0, **recovery}}})


def test_grasp_recovery_blind_retry_succeeds():
    """A miss that the blind loose->close retry fixes returns 'ok' with NO arm motion. Success is
    now the GROOVE count (225), not full closure."""
    from urlab.skills.pick import GraspCheck, GraspGeometry, GraspRecovery
    cfg = _recovery_cfg()
    g = _FakeGripper(on_close=[220, 225])         # faces miss, then the blind retry seats it (groove)
    robot = _FakeRobot(g)
    rec = GraspRecovery(cfg)
    res = rec.grasp_with_recovery(robot, GraspGeometry(cfg), GraspCheck(cfg))
    assert res == 'ok', res
    assert robot.moves == [], 'blind retry must not move the arm'


def test_grasp_recovery_faces_mode_moves_away_from_cable():
    """A faces miss nudges the arm AWAY from the cable (+approach_axis) and reseats at the groove."""
    from urlab.skills.pick import GraspCheck, GraspGeometry, GraspRecovery
    cfg = _recovery_cfg()
    g = _FakeGripper(on_close=[220, 220, 225])    # miss, blind-retry miss, then reseat succeeds
    robot = _FakeRobot(g)
    geom = GraspGeometry(cfg)
    res = GraspRecovery(cfg).grasp_with_recovery(robot, geom, GraspCheck(cfg))
    assert res == 'ok', res
    assert len(robot.moves) == 1, 'exactly one corrective move expected'
    # approach_axis is +z, so 'toward the cable' is -z; faces moves AWAY -> +z in the grasp frame.
    assert geom.T_base_grasp[2, 3] > 0, f'faces mode should move away (+z): {geom.T_base_grasp[2,3]}'
    assert abs(geom.T_base_grasp[2, 3] - 0.0005) < 1e-9


def test_grasp_check_empty_is_not_recovered():
    """A full-closure count (228) is EMPTY -- returned straight through, no reseat attempts."""
    from urlab.skills.pick import GraspCheck, GraspGeometry, GraspRecovery
    cfg = _recovery_cfg()
    g = _FakeGripper(on_close=[228])              # closed fully on nothing
    robot = _FakeRobot(g)
    res = GraspRecovery(cfg).grasp_with_recovery(robot, GraspGeometry(cfg), GraspCheck(cfg))
    assert res == 'empty', res
    assert robot.moves == [], 'empty is not reseated -- there is no cable to reseat'


def test_grasp_recovery_classify_directions():
    """The faces/tips classifier and the 'toward the cable' direction (kept for other fingertips)."""
    from urlab.skills.pick import GraspGeometry, GraspRecovery
    cfg = _recovery_cfg(faces_counts=220, faces_band_counts=1)   # re-enable a tips band
    rec = GraspRecovery(cfg)
    assert rec._classify(220) == 'faces' and rec._classify(221) == 'faces'
    assert rec._classify(224) == 'tips'
    toward = rec._toward(GraspGeometry(cfg))     # approach_axis +z -> toward the cable is -z
    assert toward[2] < 0


def test_grasp_recovery_gives_up_after_max_tries():
    """A grasp that never seats returns 'missed' after the blind retry + max_tries corrections."""
    from urlab.skills.pick import GraspCheck, GraspGeometry, GraspRecovery
    cfg = _recovery_cfg(max_tries=3)
    g = _FakeGripper(on_close=[220])              # always a faces miss
    robot = _FakeRobot(g)
    res = GraspRecovery(cfg).grasp_with_recovery(robot, GraspGeometry(cfg), GraspCheck(cfg))
    assert res == 'missed', res
    assert len(robot.moves) == 3, 'should attempt exactly max_tries corrective moves'


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception as exc:                  # noqa: BLE001
            failed += 1
            print(f'FAIL {fn.__name__}: {exc}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
