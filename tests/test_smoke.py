"""Smoke + math tests for the ROS-free layers -- no robot, no camera, no torch.

Run: python -m pytest tests/ -q     (or: python tests/test_smoke.py)

Covers the parts that are pure computation and therefore fully testable offline: the transform
conventions (the thing most likely to be silently wrong), the FrameGraph staleness semantics, the
config loader, and the connector-fusion geometry against a synthetic ground truth.
"""

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from urlab import transforms as T   # noqa: E402
from urlab.frames import FrameGraph   # noqa: E402


def approx(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol)


def test_sources_parse_and_have_no_duplicated_blocks():
    """Every source PARSES, and no block of lines is immediately repeated.

    Both failure modes have actually happened here: committing the same work independently on the
    dev box and the robot box, then merging, makes git take BOTH copies of each changed hunk. The
    result is silently duplicated statements (harmless-looking) and unbalanced brackets (a hard
    syntax error) -- twice, in files that a hardware run imports. Line endings are not the cause
    and normalising them does not prevent it; only noticing does, so this test does the noticing."""
    import ast

    src_dirs = [os.path.join(ROOT, 'urlab'), os.path.join(ROOT, 'analysis'),
                os.path.join(ROOT, 'tests')]
    files = []
    for d in src_dirs:
        for base, _, names in os.walk(d):
            files += [os.path.join(base, n) for n in names if n.endswith('.py')]
    assert files, 'no sources found -- the walk paths are wrong'

    bad_syntax, dups = [], []
    for f in files:
        text = open(f, encoding='utf-8').read()
        try:
            ast.parse(text, f)
        except SyntaxError as exc:
            bad_syntax.append(f'{os.path.relpath(f, ROOT)}:{exc.lineno}: {exc.msg}')
            continue
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            hit = 0
            # Longest-first so a big duplicated hunk is reported once, not as many small ones.
            for n in range(40, 1, -1):
                if i + 2 * n > len(lines):
                    continue
                blk = lines[i:i + n]
                if blk == lines[i + n:i + 2 * n] and sum(1 for s in blk if s.strip()) >= 2:
                    dups.append(f'{os.path.relpath(f, ROOT)}:{i + 1}-{i + n} repeated '
                                f'immediately ({n} lines): {blk[0].strip()[:60]!r}')
                    hit = 2 * n
                    break
            i += hit if hit else 1

    assert not bad_syntax, 'sources fail to parse:\n  ' + '\n  '.join(bad_syntax)
    assert not dups, ('duplicated line blocks -- almost certainly a merge that took both copies '
                      'of the same change:\n  ' + '\n  '.join(dups))


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


def test_seed_connector_gate_rejects_background_first_view():
    """Seeding the gate from the centred detection rejects a background cable on the FIRST view --
    before RANSAC (which ties two equally-seen cables) could steer toward the wrong one."""
    import types
    from urlab.config import Config
    from urlab.perception.connector import ConnectorEstimator
    from urlab.skills.scan import CableScanner

    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    est = ConnectorEstimator(Config({'connector_estimator': {
        'min_inlier_views': 3, 'min_parallax_deg': 1.0, 'inlier_dist_m': 0.02,
        'max_range_m': 3.0, 'reject_dist_m': 0.05, 'up_axis': [0, 0, 1]}}))
    sc = CableScanner.__new__(CableScanner)
    sc.estimator = est
    sc.s = types.SimpleNamespace(nominal_distance_m=0.23)

    C = np.array([0.5, 0.0, 0.6])
    T_bc = T.look_at(C, np.array([0.5, 0.0, 0.2]), np.eye(4))
    Rcw = T_bc[:3, :3].T

    def det(P):
        p = K @ (Rcw @ (np.asarray(P, float) - C))
        return (float(p[0] / p[2]), float(p[1] / p[2]), 0.0)

    target = det([0.5, 0.0, 0.2])            # projects to the image centre
    bg = det([0.5, 0.15, 0.2])               # a cable 15 cm to the side

    sc._seed_connector_gate([bg, target], K, T_bc)    # background listed first -> centre pick = target
    assert est._gate_origin is not None
    est.add_view([bg, target], K, T_bc, 0.0)          # ingest both this view
    assert est.n_views == 1 and len(est.history) == 1, 'background must be gated out on view 1'


def test_cable_profile_applies_counts():
    """Selecting a cable overrides the gripper endpoints, grasp-check band, and grasp offset. The
    grasp TARGET is the CONNECTOR: its range is the success band, the cable count is a miss above it."""
    from urlab.config import CONFIG_DIR, Config, apply_cable_profile

    cfg = Config({'cable': 'banana', '_config_dir': CONFIG_DIR})
    apply_cable_profile(cfg)
    assert cfg.get_path('gripper.port') == '/dev/ttyUSB0'              # ALL gripper params from cables.yaml
    assert cfg.get_path('gripper.open_counts') == 3
    assert cfg.get_path('gripper.closed_counts') == 228
    assert cfg.get_path('gripper.speed_counts') == 255
    assert cfg.get_path('gripper.force_counts') == 150
    assert cfg.get_path('grasp_check.empty_counts') == 228
    # The banana band is now DERIVED from connector_diameter_mm via the calibrated gripper model
    # + groove depth -- and must land on the measured reference band.
    assert cfg.get_path('grasp_check.connector_counts') == [207, 213]  # SUCCESS band (the connector)
    assert cfg.get_path('grasp_check.faces_max_counts') == 206         # <= this = miss (too thick)
    assert cfg.get_path('grasp_check.groove_max_counts') == 213        # > this (< empty) = miss (cable)
    assert cfg.get_path('grasp_check.groove_counts') == 210            # band midpoint
    assert cfg.get_path('grasp_check.cable_counts') == 225
    assert cfg.get_path('grasp_check.connector_diameter_mm') == [9.55, 10.7]  # payload-width check
    assert cfg.get_path('grasp_check.cable_diameter_mm') == 3.66
    # junction_in_fingertip: the junction pose wrt the fingertip at the grasp -- monitor units
    # (xyz_mm/rpy_deg) must be converted to m/rad, replacing junction_offset_m/connector_grasp.
    jf = cfg.get_path('junction_in_fingertip')
    assert set(jf) == {'xyz', 'rpy'}, 'xyz_mm/rpy_deg must be converted away, not passed through'
    assert all(abs(v) < 0.1 for v in jf['xyz']), 'xyz must be METRES (mm would be ~1000x)'
    # The HOLDER chain is retired: no cable may still promote the old keys.
    assert cfg.get_path('connector_in_holder') is None
    assert cfg.get_path('connector_holder_target') is None

    bnc = Config({'cable': 'bnc', '_config_dir': CONFIG_DIR})
    apply_cable_profile(bnc)
    assert bnc.get_path('grasp_check.faces_max_counts') == 196
    assert bnc.get_path('grasp_check.groove_max_counts') == 206

    # The RETIRED junction_offset_m must fail LOUDLY with the conversion recipe -- a silently
    # ignored offset would grasp at the junction itself.
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, 'cables.yaml'), 'w') as fh:
            fh.write('cables:\n  x:\n    junction_offset_m: 0.01\n')
        try:
            apply_cable_profile(Config({'cable': 'x', '_config_dir': tmp}))
            raise AssertionError('junction_offset_m must raise, not be ignored')
        except ValueError as exc:
            assert 'junction_in_fingertip' in str(exc)
        # The retired HOLDER keys must fail loudly too -- a stale entry would otherwise feed a
        # target nothing reads any more.
        with open(os.path.join(tmp, 'cables.yaml'), 'w') as fh:
            fh.write('cables:\n  x:\n    connector_holder_target: {xyz_mm: [1, 2, 3]}\n')
        try:
            apply_cable_profile(Config({'cable': 'x', '_config_dir': tmp}))
            raise AssertionError('connector_holder_target must raise, not be ignored')
        except ValueError as exc:
            assert 'frames.yaml' in str(exc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # unknown cable -> a clear error; unset -> no-op.
    try:
        apply_cable_profile(Config({'cable': 'nope', '_config_dir': CONFIG_DIR}))
        assert False, 'expected KeyError for an unknown cable'
    except KeyError:
        pass
    assert apply_cable_profile(Config({})).get('gripper') is None


def test_grasp_result_connector_band():
    """With a connector-target band, only the connector range is 'ok'; the cable (thinner) is a miss."""
    from urlab.robot.gripper import Robotiq2F85
    g = Robotiq2F85.__new__(Robotiq2F85)
    g.dry_run = False

    def result(pos):
        g._read = lambda: {'pos': pos, 'obj': 3}
        return g.grasp_result(210, 231, 206, tolerance=1, detect_empty=True, groove_max_counts=213)

    assert result(210) == 'ok'        # connector seated (in the band)
    assert result(207) == 'ok'
    assert result(226) == 'missed'    # grabbed the CABLE (thinner, above the band)
    assert result(231) == 'empty'     # closed on nothing
    assert result(200) == 'missed'    # too thick (below the band)


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

    def open(self, label='open'):
        self.pos = 3
        return True

    def position(self):
        return self.pos

    def grasp_result(self, groove_counts, empty_counts, faces_max_counts, tolerance=1,
                     detect_empty=True, groove_max_counts=None):
        if self.pos <= faces_max_counts:
            return 'missed'
        if detect_empty and self.pos >= empty_counts - tolerance:
            return 'empty'
        if groove_max_counts is not None and self.pos > groove_max_counts:
            return 'missed'
        return 'ok'


class _FakeRobot:
    def __init__(self, gripper):
        self.gripper = gripper
        self.moves = []

    def move_fingertip(self, T, label='move'):
        self.moves.append((label, np.array(T, dtype=float)))
        return True


def _recovery_cfg(**recovery):
    """A connector-target grasp_check: connector band [207,213] = success, cable 226 / closed 231
    are the two miss states."""
    from urlab.config import Config
    return Config({
        'grasp_check': {'enabled': True, 'connector_counts': [207, 213], 'cable_counts': 226,
                        'empty_counts': 231, 'faces_max_counts': 206, 'groove_max_counts': 213,
                        'groove_counts': 210, 'tolerance_counts': 1, 'detect_empty': True,
                        'settle_s': 0.0,
                        'recovery': {'enabled': True, 'finger_width_m': 0.02278,
                                     'cable_shift_fraction': 0.8, 'empty_drop_m': 0.003, **recovery}}})


def test_grasp_recovery_connector_success():
    """A close in the connector band is 'ok' with NO arm motion."""
    from urlab.skills.pick import GraspCheck, GraspGeometry, GraspRecovery
    cfg = _recovery_cfg()
    g = _FakeGripper(on_close=[210])              # inside [207, 213]
    robot = _FakeRobot(g)
    res = GraspRecovery(cfg).grasp_with_recovery(robot, GraspGeometry(cfg), GraspCheck(cfg))
    assert res == 'ok', res
    assert robot.moves == []


def test_grasp_recovery_cable_shifts_toward_connector():
    """A CABLE grab (226) opens, shifts +x by 0.8*finger_width toward the connector end, and reseats."""
    from urlab.skills.pick import GraspCheck, GraspGeometry, GraspRecovery
    cfg = _recovery_cfg()
    g = _FakeGripper(on_close=[226, 210])         # cable, then the +x reseat seats the connector
    robot = _FakeRobot(g)
    geom = GraspGeometry(cfg)
    res = GraspRecovery(cfg).grasp_with_recovery(robot, geom, GraspCheck(cfg))
    assert res == 'ok', res
    assert len(robot.moves) == 1
    assert abs(geom.T_base_grasp[0, 3] - 0.8 * 0.02278) < 1e-9    # +x only
    assert abs(geom.T_base_grasp[1, 3]) < 1e-12 and abs(geom.T_base_grasp[2, 3]) < 1e-12


def test_grasp_recovery_edge_pinch_moves_deeper():
    """A stall at ~the FREE-CLOSURE counts (216 from the calibrated model: separation ~0, held
    width at the 2*groove floor -- thinner than any connector) is an EDGE PINCH: the grooves
    closed past the connector's fat section, the grasp is too shallow. Directed reseat: -z, IN
    toward the connector, then retry."""
    from urlab.skills.pick import GraspCheck, GraspGeometry, GraspRecovery
    cfg = _recovery_cfg()
    g = _FakeGripper(on_close=[216, 210])         # edge pinch, then the -z reseat seats it
    robot = _FakeRobot(g)
    geom = GraspGeometry(cfg)
    res = GraspRecovery(cfg).grasp_with_recovery(robot, geom, GraspCheck(cfg))
    assert res == 'ok', res
    assert len(robot.moves) == 1
    assert abs(geom.T_base_grasp[2, 3] + 0.003) < 1e-9    # -z only (deeper onto the connector)
    assert abs(geom.T_base_grasp[0, 3]) < 1e-12 and abs(geom.T_base_grasp[1, 3]) < 1e-12


def test_grasp_recovery_empty_drops_toward_ground():
    """An EMPTY close (231) opens, drops 3 mm in -z (toward the object), and reseats."""
    from urlab.skills.pick import GraspCheck, GraspGeometry, GraspRecovery
    cfg = _recovery_cfg()
    g = _FakeGripper(on_close=[231, 210])         # empty, then the -z reseat seats the connector
    robot = _FakeRobot(g)
    geom = GraspGeometry(cfg)
    res = GraspRecovery(cfg).grasp_with_recovery(robot, geom, GraspCheck(cfg))
    assert res == 'ok', res
    assert len(robot.moves) == 1
    assert abs(geom.T_base_grasp[2, 3] - (-0.003)) < 1e-9         # -z only
    assert abs(geom.T_base_grasp[0, 3]) < 1e-12


def test_grasp_recovery_unexpected_is_blind_retry():
    """A count that is neither connector/cable/closed (218) opens + retries with NO move."""
    from urlab.skills.pick import GraspCheck, GraspGeometry, GraspRecovery
    cfg = _recovery_cfg()
    g = _FakeGripper(on_close=[218, 210])         # between connector(213) and cable(226) -> blind, then ok
    robot = _FakeRobot(g)
    res = GraspRecovery(cfg).grasp_with_recovery(robot, GraspGeometry(cfg), GraspCheck(cfg))
    assert res == 'ok', res
    assert robot.moves == []


def test_grasp_recovery_gives_up_after_max_tries():
    """A grasp that never seats returns 'missed' after exactly max_tries reseats."""
    from urlab.skills.pick import GraspCheck, GraspGeometry, GraspRecovery
    cfg = _recovery_cfg(max_tries=3)
    g = _FakeGripper(on_close=[226])              # always the cable
    robot = _FakeRobot(g)
    res = GraspRecovery(cfg).grasp_with_recovery(robot, GraspGeometry(cfg), GraspCheck(cfg))
    assert res == 'missed', res
    assert len(robot.moves) == 3, 'exactly max_tries corrective moves'


def test_admittance_yields_along_external_push():
    """COMPLIANCE, not resistance: an external push must move the tool ALONG the push, on EVERY
    axis. A per-axis split (x complies, z opposes) is the signature of a frame error, not a sign
    error -- a wrong sign inverts all axes together."""
    from urlab.robot.admittance import AdmittanceController

    class _FakeArm:
        dry_run = False

        def __init__(self, w):
            self._w = np.asarray(w, dtype=float)
            self.commanded = None

        def wrench(self):
            return self._w                        # EXTERNAL force on the tool, in base_link

        def servo_l(self, T, dt, lookahead=0.1, gain=300):
            self.commanded = T

    T_ref = np.eye(4)                             # tool0 aligned with base_link (R = I)
    for axis in range(3):
        push = np.zeros(6)
        push[axis] = 5.0                          # +5 N along this base_link axis
        arm = _FakeArm(push)
        adm = AdmittanceController(arm, {'reference_rate_hz': 125.0})
        adm.reset()
        for _ in range(20):
            adm._step(T_ref, 1.0 / 125.0)
        assert adm._delta[axis] > 0, f'axis {axis} must yield ALONG the push, not against it'
        assert arm.commanded[axis, 3] > 0, f'axis {axis}: commanded pose must move along the push'


def test_retract_backs_out_along_the_connector_axis():
    """Linear peg-in-hole escape: the retract must translate the HELD PART along ITS OWN -X, so a
    tilted (perturbed) connector backs out along its own axis rather than the target's. Distance is
    a magnitude -- a negative config value must not drive INTO the socket."""
    from urlab.apps.uncertain_sampling import _retract_ref
    from urlab.transforms import xyzrpy_to_matrix

    T_tool0_held = xyzrpy_to_matrix([0.0, -0.045, 0.010], [math.pi, 0.0, -math.pi / 2])
    d = 0.05

    # Tool0 reference chosen so the HELD PART sits pitched 30 deg about base Y.
    T_conn = xyzrpy_to_matrix([0.4, 0.2, 0.3], np.radians([0.0, 30.0, 0.0]))
    T_ref = T_conn @ T.inverse(T_tool0_held)

    out = _retract_ref(T_ref, T_tool0_held, d)
    conn_out = out @ T_tool0_held                       # where the connector ended up
    moved = conn_out[:3, 3] - T_conn[:3, 3]

    assert np.allclose(conn_out[:3, :3], T_conn[:3, :3], atol=1e-12), 'retract must not rotate'
    assert np.isclose(np.linalg.norm(moved), d), f'must travel exactly {d} m'
    assert np.allclose(moved / d, -T_conn[:3, 0], atol=1e-9), \
        "travel must be along the CONNECTOR's own -X"
    assert not np.allclose(moved / d, [-1.0, 0.0, 0.0], atol=1e-3), \
        'a 30 deg tilt must move it off the target/base -X'
    # magnitude only: a negative distance must retract, never insert
    assert np.allclose(_retract_ref(T_ref, T_tool0_held, -d), out)


def _write_log(path, rows, cols=None):
    """A minimal uncertain_sampling log: the manifold columns plus some cell-specific ones."""
    import csv as _csv
    from analysis.contact_manifold import MANIFOLD_COLS
    cols = cols if cols is not None else (['trial', 'timestamp'] + MANIFOLD_COLS + ['tool0_base_x_mm'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, 0.0) for c in cols])


def test_contact_manifold_extracts_and_concatenates():
    """The manifold keeps ONLY the frame-invariant pair (connector-wrt-target pose + wrench in the
    connector frame) and concatenates runs. Cell-specific columns must not leak in."""
    import csv as _csv
    import shutil
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.contact_manifold import MANIFOLD_COLS, build, expand_inputs

    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, 'banana')
        _write_log(os.path.join(d, 'run_a.csv'),
                   [{'trial': 1, 'connector_target_x_mm': 1.0, 'wrench_connector_fx': 3.0,
                     'tool0_base_x_mm': 999.0}] * 3)
        _write_log(os.path.join(d, 'run_b.csv'),
                   [{'trial': 1, 'connector_target_x_mm': 2.0, 'wrench_connector_fx': 0.1}] * 2)

        paths = expand_inputs([d])
        assert len(paths) == 2, paths

        out = os.path.join(tmp, 'banana_connector_contact_manifold.csv')
        assert build(paths, out) == 5, 'all rows from both runs'
        with open(out, newline='') as fh:
            recs = list(_csv.DictReader(fh))
        assert list(recs[0]) == MANIFOLD_COLS, 'exactly the manifold columns, in order'
        assert 'tool0_base_x_mm' not in recs[0], 'cell-specific columns must NOT leak in'
        assert len(recs) == 5

        # A force filter keeps only contact samples (|f| >= threshold).
        assert build(paths, out, min_force=1.0) == 3

        # An older-format log is SKIPPED, not fatal -- the rest of the batch still builds.
        _write_log(os.path.join(d, 'old.csv'), [{'held_target_x': 1.0}], cols=['held_target_x'])
        assert build(expand_inputs([d]), out) == 5, 'stale file skipped, good files still written'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_contact_manifold_never_ingests_its_own_output():
    """A rebuild must be IDEMPOTENT. The manifold lands beside its inputs and shares their columns,
    so a directory sweep that picked it up would silently DOUBLE every sample on each re-run."""
    import shutil
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.contact_manifold import build, expand_inputs

    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, 'banana')
        _write_log(os.path.join(d, 'run_a.csv'), [{'connector_target_x_mm': 1.0}] * 4)
        out = os.path.join(d, 'banana_connector_contact_manifold.csv')

        assert build(expand_inputs([d]), out) == 4
        assert build(expand_inputs([d]), out) == 4, 'rebuild doubled the data (self-ingestion)'
        assert out not in expand_inputs([d]), 'sweep must exclude existing manifolds'
        # Naming one explicitly is still allowed (deliberately merging manifolds).
        assert expand_inputs([out]) == [out]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_contact_manifold_ingests_estimator_eval_runs():
    """estimator_eval observation files are logged in the BELIEVED frame; the builder must rebase
    them into the TRUE frame via the run's own trials.csv (err_before_* = the belief error E during
    that attempt) and emit the full 16-column manifold schema (quaternion included). trials.csv /
    summary.csv are the rebase's metadata and must never be ingested as samples."""
    import csv as _csv
    import shutil
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.contact_manifold import (ERR_BEFORE_COLS, EVAL_OBS_COLS, MANIFOLD_COLS,
                                           _rebase_to_truth, build, expand_inputs)

    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, 'estimator_eval_20260810_000000')
        os.makedirs(d)
        # Attempt 1 ran with a +2 mm z belief error; the final insertion with the belief perfect.
        with open(os.path.join(d, 'trials.csv'), 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(['trial', 'attempt'] + ERR_BEFORE_COLS)
            w.writerow([1, 1, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
            w.writerow([1, 'final_insertion', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        with open(os.path.join(d, 'trial_001_attempt_01_observations.csv'), 'w',
                  newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(EVAL_OBS_COLS)
            w.writerow([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            w.writerow([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])  # free space
        with open(os.path.join(d, 'trial_001_final_insertion_observations.csv'), 'w',
                  newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(EVAL_OBS_COLS)
            w.writerow([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        with open(os.path.join(d, 'summary.csv'), 'w', newline='') as fh:
            fh.write('attempt,n\nfinal,1\n')

        paths = expand_inputs([d])
        assert all(os.path.basename(p).endswith('_observations.csv') for p in paths), \
            'trials.csv / summary.csv must be swept out as metadata'
        out = os.path.join(tmp, 'eval_contact_manifold.csv')
        assert build(paths, out) == 3
        with open(out, newline='') as fh:
            recs = list(_csv.DictReader(fh))
        assert list(recs[0]) == MANIFOLD_COLS, 'eval rows must land in the full manifold schema'
        # +2 mm z belief error: rel_true = rel_logged @ inverse(E) -> z = 5 - 2 = 3 mm. The wrench
        # moves to the true connector frame: rotation is identity, so f is unchanged and tau picks
        # up the metre lever arm, p x f = [0, 0, 0.002] x [10, 0, 0] = [0, 0.02, 0] Nm.
        r = recs[0]
        assert abs(float(r['connector_target_z_mm']) - 3.0) < 1e-6
        assert abs(float(r['connector_target_qw']) - 1.0) < 1e-9, 'identity rotation -> qw = 1'
        assert abs(float(r['wrench_connector_fx']) - 10.0) < 1e-6
        assert abs(float(r['wrench_connector_ty']) - 0.02) < 1e-6
        # The error-free final insertion passes through untouched.
        assert abs(float(recs[2]['connector_target_x_mm']) - 1.0) < 1e-6
        # The force filter applies to eval rows too (drops the 0.1 N free-space row).
        assert build(paths, out, min_force=1.0) == 2

        # Rebase self-consistency with a rotation-ful error: rebasing by E then by inverse(E)
        # must return the original rows exactly -- pins the pose AND wrench frame math.
        from scipy.spatial.transform import Rotation as _R
        rows = np.random.default_rng(3).normal(size=(5, 12)) * [5, 5, 5, 8, 8, 8, 10, 10, 10,
                                                                1, 1, 1]
        e6 = [1.0, -2.0, 3.0, 4.0, -5.0, 6.0]
        Rm = _R.from_euler('xyz', e6[3:], degrees=True).as_matrix()
        e6_inv = (list(-Rm.T @ np.asarray(e6[:3]))
                  + list(_R.from_matrix(Rm.T).as_euler('xyz', degrees=True)))
        fwd = _rebase_to_truth(rows, e6)           # (N, 16): [xyz | quat | yaw,pitch,roll | f,tau]
        as12 = lambda a: np.hstack([a[:, :3], a[:, [9, 8, 7]], a[:, 10:16]])   # noqa: E731
        back = _rebase_to_truth(as12(fwd), e6_inv)
        assert np.allclose(as12(back), rows, atol=1e-9), 'rebase(E) then rebase(inv(E)) != id'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_grasp_descent_is_speed_paced():
    """pickup.descent_translation_mm_s / _rotation_deg_s derive the compliant ramp duration from
    the ACTUAL distance -- so touchdown speed no longer changes silently with approach_distance_m.
    With neither set, the legacy fixed descent_time_s still applies."""
    from urlab.config import Config
    from urlab.skills.pick import GraspController

    g = GraspController(Config({'pickup': {'mode': 'compliance',
                                           'descent_translation_mm_s': 50.0,
                                           'descent_rotation_deg_s': 30.0}}))
    A = np.eye(4)
    B = T.translation_matrix([0.0, 0.0, -0.10])           # 100 mm descent
    assert np.isclose(g._duration(A, B), 2.0), '100 mm at 50 mm/s must take 2 s'
    C = T.xyzrpy_to_matrix([0, 0, 0], [0, np.radians(60.0), 0])
    assert np.isclose(g._duration(A, C), 2.0), '60 deg at 30 deg/s must take 2 s'
    both = T.translation_matrix([0, 0, -0.05]) @ C        # 1 s of travel, 2 s of rotation
    assert np.isclose(g._duration(A, both), 2.0), 'the slower axis must set the duration'
    assert g._duration(A, A) == 0.1, 'zero-length move floors at 0.1 s'

    legacy = GraspController(Config({'pickup': {'mode': 'compliance', 'descent_time_s': 3.0}}))
    assert legacy._duration(A, B) == 3.0, 'no speed keys -> the legacy fixed duration'

    # Without pickup descent keys the GLOBAL cartesian limits pace the descent/lift, so a config
    # with only the four-key speed: block needs no pickup speed duplicates.
    glob = GraspController(Config({'pickup': {'mode': 'compliance'},
                                   'speed': {'max_cartesian_translation_mm_s': 25.0,
                                             'max_cartesian_rotation_deg_s': 30.0}}))
    assert np.isclose(glob._duration(A, B), 4.0), '100 mm at the global 25 mm/s must take 4 s'

    # speed.phase_scale: the descent runs at the 'pickup' scale, the lift at the 'lift' scale.
    sc = GraspController(Config({'pickup': {'mode': 'compliance'},
                                 'speed': {'max_cartesian_translation_mm_s': 25.0,
                                           'max_cartesian_rotation_deg_s': 30.0,
                                           'phase_scale': {'pickup': 1.0, 'lift': 0.5,
                                                           'assemble': 0.2}}}))
    assert (sc.pickup_scale, sc.lift_scale) == (1.0, 0.5)
    assert np.isclose(sc._duration(A, B, sc.pickup_scale), 4.0)
    assert np.isclose(sc._duration(A, B, sc.lift_scale), 8.0), 'half speed = double ramp time'

    # The LIFT must not re-tare by default: it starts IN CONTACT, and a tare there turns the
    # ground reaction into a phantom downward force at liftoff (the arm chases it into the ground).
    assert g.tare_before is True and g.tare_before_lift is False


def test_speed_limits_global_and_assembly_blocks():
    """EXACTLY four speed limits (joint vel deg/s, joint accel deg/s^2, cartesian translation
    mm/s, cartesian rotation deg/s), in two variations: the global speed: block and an
    assembly.speed override whose absent keys inherit the global values. Legacy spellings
    (rad/s, rad/s^2, m/s) still parse for the older configs."""
    from urlab.config import Config
    from urlab.robot.arm import URArm, parse_limits

    arm = URArm(Config({'robot': {'dry_run': True},
                        'speed': {'max_joint_velocity_deg_s': 30.0,
                                  'max_joint_acceleration_deg_s2': 30.0,
                                  'max_cartesian_translation_mm_s': 25.0,
                                  'max_cartesian_rotation_deg_s': 30.0}}))
    assert np.isclose(arm.max_joint_vel, np.radians(30.0))
    assert np.isclose(arm.joint_accel, np.radians(30.0)), 'accel key is in deg/s^2'
    assert np.isclose(arm.max_cart_vel, 0.025), 'moveL speed comes from the mm/s key'
    assert np.isclose(arm.max_cart_rot, np.radians(30.0))

    # A partial mapping override: set keys win, absent keys inherit the global limits.
    base = (arm.max_joint_vel, arm.joint_accel, arm.max_cart_vel, arm.max_cart_rot)
    jv, ja, cv, cr = parse_limits({'max_joint_velocity_deg_s': 15.0,
                                   'max_cartesian_translation_mm_s': 5.0}, base)
    assert np.isclose(jv, np.radians(15.0)) and np.isclose(cv, 0.005)
    assert np.isclose(ja, np.radians(30.0)) and np.isclose(cr, np.radians(30.0))

    # A bare NUMBER as caps scales ALL FOUR global limits -- the speed.phase_scale mechanism.
    jv, ja, cv, cr = arm._limits(0.2)
    assert np.isclose(jv, np.radians(6.0)) and np.isclose(ja, np.radians(6.0))
    assert np.isclose(cv, 0.005) and np.isclose(cr, np.radians(6.0))

    # The arm-level PHASE scale (set_speed_scale) applies whenever no per-move caps is given;
    # an explicit caps REPLACES it rather than compounding.
    assert arm.speed_scale == 1.0
    arm.set_speed_scale(0.8, 'scan')
    jv, ja, cv, cr = arm._limits(None)
    assert np.isclose(jv, np.radians(24.0)) and np.isclose(cv, 0.02)
    jv, ja, cv, cr = arm._limits(0.2)
    assert np.isclose(jv, np.radians(6.0)) and np.isclose(cv, 0.005), \
        'explicit caps must replace the phase scale, not multiply it'
    arm.set_speed_scale(1.0)

    # Legacy spellings keep their meaning (and the legacy m/s wins over mm/s for moveL).
    jv, ja, cv, cr = parse_limits({'max_joint_velocity_rad_s': 1.0,
                                   'joint_acceleration_rad_s2': 2.0,
                                   'max_cartesian_velocity_m_s': 0.5,
                                   'max_cartesian_translation_mm_s': 10.0})
    assert (jv, ja, cv) == (1.0, 2.0, 0.5)


def test_lift_slip_check_and_retry_perturbation():
    """pick: (1) lift_verified RE-CLOSES after a small partial lift -- a slipped cable lets the
    fingers run to empty -> 'slipped' (no full lift); a held connector stalls in band -> 'ok' and
    the lift completes. The re-close is essential: the fingers HOLD their stalled position when
    the part vanishes, so a bare position read cannot see the slip. (2) retry_offset_x steps
    0, +d, -d, +2d, -2d along the junction x -- the fixed-point breaker for outer retries."""
    from urlab.config import Config
    from urlab.skills.pick import (GraspCheck, GraspController, GraspGeometry, retry_offset_x)

    gc = _recovery_cfg().section('grasp_check')
    gc['lift_check'] = {'enabled': True, 'height_m': 0.02}
    cfg = Config({'grasp_check': gc})
    check, grasp, geom = GraspCheck(cfg), GraspController(cfg), GraspGeometry(cfg)
    assert np.isclose(check.slip_raise_m, 0.10), 'slip recovery rises 10 cm by default'
    geom.T_base_grasp = np.eye(4)

    held = _FakeRobot(_FakeGripper(on_close=[210]))       # re-close stalls in the band: still held
    assert grasp.lift_verified(held, geom, check) == 'ok'
    assert len(held.moves) == 2, 'partial slip-check lift, then the full lift'
    assert np.isclose(held.moves[0][1][2, 3], 0.02), 'first raise is the small slip-check lift'
    assert np.isclose(held.moves[1][1][2, 3], 0.10), 'then the full lift height'

    slipped = _FakeRobot(_FakeGripper(on_close=[231]))    # re-close runs on to EMPTY: slipped out
    assert grasp.lift_verified(slipped, geom, check) == 'slipped'
    assert len(slipped.moves) == 1, 'no full lift after a detected slip'

    # verify_cable_held: the SAME re-close principle without any arm motion (stand-off + per-
    # attempt checks). Held stalls in band -> True; gone runs to empty -> False; no moves either way.
    from urlab.skills.pick import verify_cable_held
    still = _FakeRobot(_FakeGripper(on_close=[210]))
    assert verify_cable_held(still, check, 'stand-off') is True and still.moves == []
    gone = _FakeRobot(_FakeGripper(on_close=[231]))
    assert verify_cable_held(gone, check, 'stand-off') is False and gone.moves == []

    d = 0.003
    assert [retry_offset_x(a, d) for a in range(5)] == [0.0, d, -d, 2 * d, -2 * d]
    assert retry_offset_x(3, 0.0) == 0.0, 'step 0 disables the perturbation'


def test_gripper_gap_to_forward_relation():
    """robot/gripper_kinematics: the CALIBRATED circle model (R=55.0 mm, apex offset 6.2 mm,
    linear counts->angle) must reproduce the 2026-07-30 measured table -- (encoder, gap mm,
    delta-height mm) -- within the fit tolerance, saturate past pad contact (~counts 216), and
    advance the tips ~12.8 mm over the full stroke."""
    from urlab.robot import gripper_kinematics as gk

    measured = [(3, 83.56, 93.66), (50, 67.03, 99.03), (100, 47.80, 104.15),
                (150, 27.19, 105.90), (200, 7.36, 106.42), (230, 0.00, 106.42)]
    for c, gap_mm, z_mm in measured:
        assert abs(gk.gap_from_counts(c) * 1000 - gap_mm) < 0.7, f'gap mismatch at counts {c}'
        assert abs(gk.pad_forward_from_counts(c) * 1000 - z_mm) < 0.5, f'z mismatch at counts {c}'

    # Full-stroke advance: the measured 12.76 mm, not the Menagerie-derived 28 mm.
    adv = gk.pad_forward_from_counts(230) - gk.pad_forward_from_counts(3)
    assert 0.012 < adv < 0.0135, 'tips must ADVANCE ~12.8 mm from open to closed'

    # Saturation: past pad contact (~counts 216) the geometry freezes -- 230 = 216, not further.
    assert gk.gap_from_counts(230) == 0.0
    assert np.isclose(gk.pad_forward_from_counts(230),
                      gk.pad_forward_from_counts(gk.COUNTS_CLOSED))
    assert 210 < gk.COUNTS_CLOSED < 222

    # z(gap) closed form sits ON the fitted circle. z is monotonic in counts only UP TO the
    # apex (~counts 186); from there to pad contact it dips < 0.5 mm (the measured rows are
    # flat there -- within noise), then freezes.
    gaps = np.linspace(0.0, gk.GAP_MAX_M, 25)
    r2 = ((gaps / 2 - gk.APEX_LATERAL_M) ** 2
          + (np.array([gk.pad_forward_from_gap(g) for g in gaps]) - gk.DATUM_FORWARD_M) ** 2)
    assert np.allclose(r2, gk.R_M ** 2), 'z(gap) must stay ON the calibrated circle'
    zc = np.array([gk.pad_forward_from_counts(c) for c in range(0, 187, 3)])
    assert np.all(np.diff(zc) > 0.0), 'z must rise monotonically up to the apex'
    dip = zc[-1] - gk.pad_forward_from_counts(gk.COUNTS_CLOSED)
    assert 0.0 <= dip < 0.0005, 'the post-apex dip must stay under half a millimetre'

    # counts_from_gap is the exact inverse of gap_from_counts on the live stroke.
    for c in (3, 50, 100, 150, 200):
        assert abs(gk.counts_from_gap(gk.gap_from_counts(c)) - c) < 0.01
    assert np.isclose(gk.counts_from_gap(0.0), gk.COUNTS_CLOSED)

    # WIDTH conversions (groove-aware): the banana connector diameters land inside the measured
    # grasp band; a bare cable (thinner than 2*groove) saturates to free closure.
    assert abs(gk.counts_from_width(0.0107) - 208.6) < 0.5
    assert abs(gk.counts_from_width(0.00955) - 211.4) < 0.5
    assert np.isclose(gk.counts_from_width(0.00366), gk.COUNTS_CLOSED)
    assert abs(gk.width_from_counts(gk.counts_from_width(0.0107)) - 0.0107) < 1e-9

    # PICKUP HEIGHT (connector on the ground plane): the banana grasp target rises d_max/2 above
    # the plane plus the (tiny, near-closure) advance vs the closed-calibrated fingertip frame.
    d_max = 0.0107
    s_grasp = d_max - 2.0 * gk.GROOVE_DEPTH_M
    dz = d_max / 2.0 + gk.pad_forward_from_gap(s_grasp) - gk.pad_forward_from_gap(0.0)
    assert 0.005 < dz < 0.006, 'banana pickup height must be ~5.5 mm above the ground plane'

    try:
        gk.pad_forward_from_gap(0.2)
        raise AssertionError('a gap beyond the stroke must raise')
    except ValueError:
        pass


def test_cartesian_bound_ignores_pose_model_mismatch_at_target():
    """The moveJ cartesian bound divides tool travel by joint travel. When the arm is ALREADY at
    the target (dj ~ encoder residual) the measured 'tool travel' is just the constant few-mm
    disagreement between getActualTCPPose and getForwardKinematics -- an impossible >2 m/rad lever
    -- and an ungated bound collapses to the 1e-3 floor ('moveJ 0.00 rad/s'). The gate must skip
    the bound there, yet still apply it to physically real moves."""
    from urlab.robot.arm import URArm

    arm = URArm.__new__(URArm)                       # no hardware: exercise _speeds alone
    arm.max_joint_vel = np.radians(30.0)
    arm.joint_accel = np.radians(30.0)
    arm.max_cart_vel = 0.025
    arm.max_cart_rot = np.radians(30.0)
    arm.speed_scale = 1.0

    # Already at home (1e-4 rad residual), pose sources disagreeing by a constant 5 mm.
    arm.q = lambda: [1e-4, 0, 0, 0, 0, 0]
    arm.fk = lambda q: np.eye(4)
    T_mismatch = T.translation_matrix([0.005, 0.0, 0.0])
    arm.tcp_pose = lambda: T_mismatch
    speed, accel, which = arm._speeds([0.0] * 6)
    assert which == 'joint-velocity' and np.isclose(speed, arm.max_joint_vel), \
        'a no-op move must not collapse to the floor on pose-model mismatch'

    # A real move (1 rad swinging the tool 0.5 m -- lever 0.5 m/rad) IS still bounded.
    arm.q = lambda: [0.0] * 6
    arm.fk = lambda q: T.translation_matrix([0.5, 0.0, 0.0])
    arm.tcp_pose = lambda: np.eye(4)
    speed, accel, which = arm._speeds([1.0, 0, 0, 0, 0, 0])
    assert which == 'cartesian-translation' and np.isclose(speed, 1.0 / 0.5 * 0.025), \
        'a real swing must still be paced by the cartesian translation cap'


def test_manifold_estimator_recovers_belief_error():
    """skills/manifold: observations whose POSE columns carry a rigid belief error (right-multiplied,
    like an in-hand grasp error) but whose WRENCH is the true contact signature must yield a
    correction T_corr with believed @ T_corr ~= true -- i.e. D @ T_corr ~= identity."""
    import csv as _csv
    import shutil
    import tempfile

    from urlab.skills.manifold import (FORCE_COLS, POSE_COLS, TORQUE_COLS, ManifoldEstimator,
                                       mats_from_vec6, vec6_from_mats)

    tmp = tempfile.mkdtemp()
    try:
        # Manifold: one linear -X insertion at the TRUE pose (z=0, p=0), constant contact direction.
        u_f, u_t = np.array([0.6, 0.0, -0.8]), np.array([0.0, 1.0, 0.0])
        rows = []
        for x in np.linspace(-20.0, 0.0, 80):
            rows.append([x, 0.0, 0.0, 0.0, 0.0, 0.0] + list(5.0 * u_f) + list(0.5 * u_t))
        path = os.path.join(tmp, 'manifold.csv')
        with open(path, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(rows)

        est = ManifoldEstimator({'manifold_csv': path, 'estimate_dims': ['z_mm', 'pitch_deg'],
                                 'icp_iterations': 10, 'num_initial_guesses': 30,
                                 'random_seed': 5})

        # Believed observations = true poses right-multiplied by the hidden belief error D. D is
        # built as the inverse of an in-dims transform so the exact correction LIVES in the free
        # dims (an arbitrary trans+rot D has a small coupled x-component, e.g. 2.5*sin(6 deg), that
        # a z+pitch-only correction cannot express -- by design, not a defect).
        true6 = np.array([r[:6] for r in rows])[::2]
        D = T.inverse(mats_from_vec6([0.0, 0.0, -2.5, 0.0, 6.0, 0.0]))
        obs6 = vec6_from_mats(mats_from_vec6(true6) @ D)
        f_raw = np.tile(5.0 * u_f, (len(obs6), 1))
        tau_raw = np.tile(0.5 * u_t, (len(obs6), 1))

        vec6, w6 = est.prepare_observations(obs6, f_raw, tau_raw)
        T_corr, info = est.estimate(vec6, w6)
        assert T_corr is not None, info
        undone = vec6_from_mats(D @ T_corr)               # perfect correction -> identity
        assert np.all(np.abs(undone) < 0.2), f'D @ T_corr != I: {np.round(undone, 3)}'
        assert info['final_residual'] < 0.5, info

        # Too few observations must SKIP (None + reason), never guess from thin data.
        none_corr, reason = est.estimate(vec6[:3], w6[:3])
        assert none_corr is None and 'observations' in reason

        # RECENCY weighting: if the in-hand pose DRIFTS mid-attempt (the connector slips between
        # the finger pads), the newest observations must dominate -- the correction lands nearer
        # the CURRENT belief error than the stale one. First half at (z -1, pitch +2), second
        # half at (z -4, pitch +9): the estimate must sit past the midpoint, toward the new.
        D_old = T.inverse(mats_from_vec6([0.0, 0.0, -1.0, 0.0, 2.0, 0.0]))
        D_new = T.inverse(mats_from_vec6([0.0, 0.0, -4.0, 0.0, 9.0, 0.0]))
        half = len(true6) // 2
        obs_drift = np.vstack([vec6_from_mats(mats_from_vec6(true6[:half]) @ D_old),
                               vec6_from_mats(mats_from_vec6(true6[half:]) @ D_new)])
        v6d, w6d = est.prepare_observations(obs_drift, f_raw, tau_raw)
        T_corr_d, info_d = est.estimate(v6d, w6d)
        assert T_corr_d is not None, info_d
        tc = info_d['theta_corr']
        assert tc['z_mm'] < -2.5 and tc['pitch_deg'] > 5.5, \
            f'recency weighting must pull the estimate toward the NEWEST in-hand pose: {tc}'

        # SEEDED starts: seed_frac of the budget must be drawn TIGHTLY around the given
        # hypotheses (so a rival mode is genuinely re-examined next attempt, not left to the
        # luck of uniform restarts), guess 0 must stay identity, and every start must stay
        # inside init_guess_range even when a seed sits at the box edge.
        est.init_range = {'z_mm': 5.0, 'pitch_deg': 8.0}
        g6s, n_seeded = est._start_guesses(40, seeds=[[-2.0, 6.0], [5.0, -8.0]])
        # seed_frac 0.5 of the 39 non-identity guesses, split evenly -> 19//2 = 9 per seed... but
        # rounded: round(0.5 * 39) = 20, per-seed 10. Guess 0 stays identity regardless.
        assert n_seeded == 20 and np.all(g6s[0] == 0.0), n_seeded
        near_a = np.linalg.norm(g6s[1:11][:, [2, 4]] - [-2.0, 6.0], axis=1)
        assert np.median(near_a) < 3.0, 'first seed block must cluster on the first hypothesis'
        assert np.all(np.abs(g6s[:, 2]) <= 5.0 + 1e-9) and np.all(np.abs(g6s[:, 4]) <= 8.0 + 1e-9)
        _, info_s = est.estimate(vec6, w6, seeds=[[-2.5, 6.0]])
        assert info_s['seeded_guesses'] > 0
        assert 'mixture' in info_s and info_s['n_mixture_modes'] >= 1

        # PER-DIMENSION WEIGHTS (estimation.dim_weights): applied ON TOP of the unit scaling
        # (s_rot stays the mm <-> deg conversion). (a) the metric must actually scale -- the
        # manifold's z column doubles under z_mm: 2; (b) recovery must survive a reweighting
        # (it changes the METRIC, not the answer); (c) bad configs fail at construction.
        wcfg = {'manifold_csv': path, 'estimate_dims': ['z_mm', 'pitch_deg'],
                'icp_iterations': 10, 'num_initial_guesses': 30, 'random_seed': 5,
                'dim_weights': {'z_mm': 2.0, 'pitch_deg': 0.5}}
        west = ManifoldEstimator(wcfg)
        assert np.allclose(west.M12[:, 2], 2.0 * est.M12[:, 2])
        assert np.allclose(west.M12[:, 4], 0.5 * est.M12[:, 4])
        assert np.allclose(west.M12[:, 0], est.M12[:, 0]), 'unweighted dims must not move'
        wv6, ww6 = west.prepare_observations(obs6, f_raw, tau_raw)
        T_corr_w, info_w = west.estimate(wv6, ww6)
        assert T_corr_w is not None, info_w
        undone_w = vec6_from_mats(D @ T_corr_w)
        assert np.all(np.abs(undone_w) < 0.3), f'reweighted recovery: {np.round(undone_w, 3)}'
        for bad in ({'dim_weights': {'bogus_dim': 1.0}},
                    {'dim_weights': {'z_mm': -1.0}},
                    {'dim_weights': {'z_mm': 0.0}}):   # zero on an ESTIMATED dim
            try:
                ManifoldEstimator({**wcfg, **bad})
            except ValueError:
                pass
            else:
                raise AssertionError(f'{bad} must raise at construction')
        # ... but zero on a NON-estimated dim is legitimate (make the channel invisible)
        ok0 = ManifoldEstimator({**wcfg, 'dim_weights': {'y_mm': 0.0}})
        assert ok0.dim_w[1] == 0.0

        # residual_gate: None disables it (like manifold_icp_validation) -- INCLUDING the STRING
        # 'None', because yaml parses a bare `None` as a string (only null/~ are yaml null) and
        # float('None') used to blow up MID-RUN, after the robot had already moved.
        for gate in (None, 'None', 'null', 0):
            gated = ManifoldEstimator({'manifold_csv': path, 'residual_gate': gate,
                                       'estimate_dims': ['z_mm', 'pitch_deg'],
                                       'icp_iterations': 5, 'num_initial_guesses': 10,
                                       'random_seed': 5})
            assert gated.residual_gate is None, f'{gate!r} must DISABLE the gate'
            corr, info2 = gated.estimate(vec6, w6)
            assert corr is not None, f'gate {gate!r}: estimate must run gateless, not crash'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mixture_separates_ambiguity_from_precision():
    """skills\\mixture: a multi-modal posterior must not be summarised by one mean and one std.

    The failure it exists to prevent: more multi-start guesses IMPROVE accuracy (the true basin
    gets found) but blow up the reported sigma, because the spread of the finals then measures
    the DISTANCE BETWEEN RIVAL MODES rather than the width of any one of them. So a run that
    found the right answer AND a decoy reads as less certain than one whose starts all fell into
    a single wrong basin -- backwards, and the number a gate would act on.

    What must hold:
      (a) the law of total variance -- total = within + between, exactly;
      (b) two separated clusters read as AMBIGUITY (between_frac high, separation large) while
          one diffuse cluster of the SAME total spread reads as IMPRECISION (between_frac ~ 0);
      (c) the component weights are the posterior mass, so `ambiguity` is a probability;
      (d) the grid constructor PARTITIONS the landscape (watershed), so nothing is dropped and
          a double well is found as two modes with the deeper one carrying more mass;
      (e) `split` points from the runner-up to the leader -- the direction a discriminating
          probe must move along."""
    from urlab.skills.mixture import descent_labels, from_energy, from_particles

    rng = np.random.default_rng(3)

    # (b/c) two well-separated clusters, 70/30 by mass.
    a = rng.normal([0.0, 0.0], 0.2, (140, 2))
    b = rng.normal([6.0, 0.0], 0.2, (60, 2))
    mix = from_particles(np.vstack([a, b]), bandwidth=1.0, dims=['z_mm', 'pitch_deg'])
    assert mix.n_modes == 2, mix
    assert abs(mix.dominant.weight - 0.7) < 0.05, mix.dominant.weight
    assert abs(mix.ambiguity - 0.3) < 0.05, mix.ambiguity
    assert mix.between_frac > 0.9, mix.between_frac
    assert mix.separation > 5.0, mix.separation
    assert abs(np.linalg.norm(mix.split) - 1.0) < 1e-6, mix.split
    # the leader sits at 0 and the runner-up at +6, so the split points in -z
    assert mix.split[0] < -0.99, mix.split

    # (a) law of total variance, exactly.
    assert np.allclose(mix.cov, mix.within + mix.between)
    # ... and the WITHIN width is the cluster width (0.2), not the 6-unit gap.
    assert mix.sigma(within_only=True)[0] < 0.5 < mix.sigma()[0]

    # (b) one DIFFUSE cluster with a comparable total spread must read as IMPRECISION instead:
    # same sigma, but nothing to disambiguate, so a probe should gather more of the same.
    diffuse = from_particles(rng.normal([0.0, 0.0], [2.6, 0.2], (200, 2)), bandwidth=1.0)
    assert diffuse.n_modes == 1 and diffuse.ambiguity == 0.0, diffuse
    assert diffuse.between_frac < 1e-9
    assert diffuse.sigma()[0] > 1.5, diffuse.sigma()

    # (d) grid constructor: a double well, the left one deeper. Watershed labels must PARTITION
    # the grid (every cell assigned exactly once) and the deeper well must carry more mass.
    # The 5% depth gap is deliberate -- that is the scale rival residual minima actually differ
    # by, and at posterior_temp 0.05 a much deeper rival is correctly weighted out of existence.
    ax = np.linspace(-10.0, 10.0, 201)
    E = np.minimum((ax + 5.0) ** 2 * 0.05 + 1.0, (ax - 5.0) ** 2 * 0.05 + 1.05)
    labels, roots = descent_labels(E, (201,))
    assert labels.shape == (201,) and len(roots) == 2 and set(np.unique(labels)) == {0, 1}
    gm = from_energy(ax[:, None], E, (201,), temp=0.05, dims=['pitch_deg'])
    assert gm.n_modes == 2, gm
    assert abs(gm.dominant.mean[0] + 5.0) < 0.5, gm.dominant.mean
    assert gm.dominant.weight > 0.5 and abs(sum(c.weight for c in gm.components) - 1.0) < 1e-9
    assert gm.separation > 2.0, gm.separation

    # a SINGLE well must stay unimodal -- a mode count that inflates on smooth landscapes would
    # make the ambiguity flag useless.
    single = from_energy(ax[:, None], (ax ** 2) * 0.05 + 1.0, (201,), temp=0.05)
    assert single.n_modes == 1 and single.between_frac < 1e-9, single

    # `as_dict` is what reaches the CSV/log: it must carry both widths, not just the total.
    d = gm.as_dict()
    assert d['n_modes'] == 2 and 'sigma_within' in d and len(d['modes']) == 2


def test_mode_pose_config_is_wired():
    """configs/mode_pose_estimator_eval.yaml: the two stages must keep their intended
    asymmetry (mode = wrench-HEAVY + pose down-weighted, pose = wrench-light), share one grid
    (only estimation_shared may define it), and hold a settle long enough for the wrench the
    mode stage lives on."""
    import yaml

    with open(os.path.join(ROOT, 'configs', 'mode_pose_estimator_eval.yaml')) as fh:
        cfg = yaml.safe_load(fh)
    sh, em, ep = (cfg['estimation_shared'], cfg['estimation_mode'], cfg['estimation_pose'])
    assert 'grid' in sh and 'estimate_dims' in sh
    for stage in (em, ep):
        assert 'grid' not in stage and 'estimate_dims' not in stage, \
            'stages must share estimation_shared\'s grid'
    # per-dim search bounds (estimator_eval-style init_guess_range, or grid.range) must give
    # every ESTIMATED dim a positive half-width, and stay wide enough to contain any injected
    # error plus a couple of imperfect updates
    igr = sh.get('init_guess_range') or {}
    grng = (sh.get('grid') or {}).get('range') or {}
    for d in sh['estimate_dims']:
        half = float(grng.get(d, igr.get(d, 8.0)))
        assert half > 0, f'estimated dim {d} needs a positive search half-width'
    assert float(em['scaling_constant_unit_force_to_mm']) \
        > float(ep['scaling_constant_unit_force_to_mm']), \
        'the MODE stage must weight the wrench harder than the POSE stage'
    dw = em.get('dim_weights') or {}
    assert dw and max(float(v) for v in dw.values()) < 1.0, \
        'the mode stage must down-weight the (aliased) pose channels'
    assert float(cfg['compliance']['settle_s']) >= 1.2, \
        'stage 1 lives on the wrench -- it must be SETTLED (>= ~1.2 s)'
    assert 0 < float(cfg['pose_stage']['alpha']) <= 0.8
    with open(os.path.join(ROOT, 'configs', 'frames.yaml')) as fh:
        frames = yaml.safe_load(fh)
    held = cfg['held_frame']
    assert held in frames['frames'] and held in frames['targets']


def test_two_stage_mode_then_pose_escapes_wrong_global_minimum():
    """apps\\mode_pose_estimator_eval.two_stage: the pathology it exists for, synthesized.

    A DENSE decoy region of the map makes the pose-light energy's GLOBAL minimum land in the
    wrong mode (soft-kNN energy is lower where sampling is denser -- what should be a local
    minimum becomes global). The wrench signature still separates the modes. The MODE GATE
    was removed (2026-08-13: restricting stage 2 measurably hurt on real data), so what the
    two-stage must deliver here is the DIAGNOSTIC that catches the pathology: the pose-only
    argmin fooled, the wrench-heavy stage-1 partition putting the truth in its TOP-mass mode,
    and the mode centre disagreeing loudly with the fooled pose argmin -- the discrepancy an
    operator (or a later gate, once the classifier earns it) acts on."""
    import csv as _csv
    import shutil
    import tempfile

    from urlab.apps.mode_pose_estimator_eval import (merged_estimation, truth_mode_rank,
                                                     two_stage)
    from urlab.skills.grid_estimator import GridManifoldEstimator
    from urlab.skills.manifold import (FORCE_COLS, POSE_COLS, TORQUE_COLS, mats_from_vec6,
                                       vec6_from_mats)

    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(4)
    try:
        # map: pitch offsets -12..12, wrench fy encodes the offset (mode-identifying). The
        # DECOY is a SAMPLING-DENSITY asymmetry -- the user's observed mechanism: the
        # [-8, -4] band is sampled 7.5x denser in x than the truth's side, so its kNN
        # residuals are genuinely SMALLER and the pose metric's global minimum lands there
        # ('incorrect parameters better explain the observations'). The wrench refuses it.
        rows = []
        for p in np.arange(-12.0, 12.01, 1.0):
            n_x = 90 if -8.0 <= p <= -4.0 else (12 if p >= 0.0 else 30)
            for x in np.linspace(-20.0, -4.0, n_x):
                f = [-4.5 + rng.normal(0, 0.02), -0.25 * p + rng.normal(0, 0.02),
                     rng.normal(0, 0.02)]
                rows.append([x + rng.normal(0, 0.05), 0.0, 0.0, 0.0,
                             p + rng.normal(0, 0.05), 0.0] + f + [0.0, 0.0, 0.0])
        path = os.path.join(tmp, 'manifold.csv')
        with open(path, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(rows)

        base = dict(manifold_csv=path, estimate_dims=['pitch_deg'], min_observations=5,
                    wrench_representation='rawcap', min_force_n=1.0, interp_neighbors=8,
                    scaling_constant_deg_to_mm=1.0,
                    grid={'range': {'pitch_deg': 14.0}, 'step': {'pitch_deg': 0.5},
                          'curvature_probe_deg': 3.0})
        est_mode = GridManifoldEstimator({**base, 'scaling_constant_unit_force_to_mm': 4.0,
                                          'dim_weights': {'pitch_deg': 0.3}})
        est_pose = GridManifoldEstimator({**base, 'scaling_constant_unit_force_to_mm': 0.1})

        # observations: physical offset +4 (wrench fy = -1.0) over the FULL depth range --
        # the real insertion went deeper than the map's coverage of its own mode. Belief
        # error -4 deg: the believed pitch column reads ~0, the CORRECT correction is +4.
        xs = np.linspace(-20.0, -4.0, 40)
        src = np.stack([xs + rng.normal(0, 0.05, 40), np.zeros(40), np.zeros(40),
                        np.zeros(40), 4.0 + rng.normal(0, 0.05, 40), np.zeros(40),
                        -4.5 + rng.normal(0, 0.02, 40), -1.0 + rng.normal(0, 0.02, 40),
                        rng.normal(0, 0.02, 40), np.zeros(40), np.zeros(40),
                        np.zeros(40)], axis=1)
        D = mats_from_vec6([0.0, 0.0, 0.0, 0.0, -4.0, 0.0])
        bel = vec6_from_mats(mats_from_vec6(src[:, :6]) @ D)
        obs = np.hstack([bel, src[:, 6:9] @ D[:3, :3], np.zeros((len(src), 3))])
        e6 = np.array([0.0, 0.0, 0.0, 0.0, -4.0, 0.0])   # err_before

        # the single pose-metric argmin must be SEDUCED by the dense decoy band...
        vp, wp = est_pose.prepare_observations(obs[:, :6], obs[:, 6:9], obs[:, 9:12])
        E_p, _, _ = est_pose.energy(vp, wp)
        th_single = float(est_pose.grid6[int(np.argmin(E_p))][4])
        assert abs(th_single - 4.0) > 3.0, \
            f'the decoy must fool the pose-only argmin (got {th_single:+.1f}, wanted wrong)'
        # ... while the wrench-heavy stage-1 DIAGNOSTIC catches it: truth in the TOP mode,
        # and the mode centre pointing at +4 while the fooled pose argmin sits elsewhere.
        th6, info = two_stage(est_mode, est_pose, obs)
        assert th6 is not None, info
        assert truth_mode_rank(info['mixture'], est_mode, e6) == 1, \
            'the wrench-heavy stage must put the truth in its TOP mode'
        assert abs(info['mode_centre']['pitch_deg'] - 4.0) <= 2.5, \
            f"the mode centre must point near +4 (got {info['mode_centre']})"
        assert abs(info['pose_argmin_global']['pitch_deg']
                   - info['mode_centre']['pitch_deg']) > 3.0, \
            'the diagnostic must expose the pose-vs-mode disagreement'

        # config plumbing: per-stage grid/estimate_dims overrides must be rejected
        class FakeCfg(dict):
            def section(self, k):
                return self.get(k, {})
        try:
            merged_estimation(FakeCfg(estimation_shared=dict(base),
                                      estimation_mode={'grid': {}}, estimation_pose={}))
        except ValueError:
            pass
        else:
            raise AssertionError('per-stage grid override must raise')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_grid_estimator_support_flags_thin_evidence():
    """skills/grid_estimator SUPPORT: observations taken far off the manifold must read as
    thin support -- the drift-out-of-distribution failure the residual alone cannot see. On-map
    observations must calibrate to ratio ~1, off-map wrenches must at least double it and widen
    the reported sigma when the (opt-in) inflation multiplier is on."""
    import csv as _csv
    import shutil
    import tempfile

    from urlab.skills.grid_estimator import GridManifoldEstimator
    from urlab.skills.manifold import (FORCE_COLS, POSE_COLS, TORQUE_COLS, mats_from_vec6,
                                       vec6_from_mats)

    # support: a manifold covering pitch in [-12, 12] only. Observations that sit ON it must
    # report support ~1; observations driven far outside it must report a materially larger
    # ratio even though the RESIDUAL cannot tell the difference on its own.
    tmp = tempfile.mkdtemp()
    try:
        rows = []
        for p in np.arange(-12.0, 12.01, 1.0):
            for x in np.linspace(-20.0, -4.0, 40):
                f = 5.0 * np.array([-0.9, 0.0, 0.0]) + np.array([0.0, -0.25 * p, 0.0])
                rows.append([x, 0.0, 0.0, 0.0, p, 0.0] + list(f) + [0.0, 0.0, 0.0])
        path = os.path.join(tmp, 'manifold.csv')
        with open(path, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(rows)
        est = GridManifoldEstimator({
            'manifold_csv': path, 'estimate_dims': ['pitch_deg'], 'min_observations': 5,
            'scaling_constant_unit_force_to_mm': 1.5, 'scaling_constant_unit_torque_to_mm': 0.0,
            'wrench_representation': 'rawcap', 'min_force_n': 1.0, 'interp_neighbors': 8,
            # support_inflation is 0 (report-only) by default after the 2026-08-12 validation;
            # the MECHANISM is what this test checks, so it turns the multiplier on.
            'grid': {'range': {'pitch_deg': 6.0}, 'step': {'pitch_deg': 0.5},
                     'curvature_probe_deg': 3.0, 'support_inflation': 1.0}})
        assert est.support_ref > 0

        def obs_at(true_pitch, err_pitch):
            src = np.array([r for r in rows if abs(r[4] - true_pitch) < 1e-9])
            D = mats_from_vec6([0.0, 0.0, 0.0, 0.0, err_pitch, 0.0])
            return (vec6_from_mats(mats_from_vec6(src[:, :6]) @ D), src[:, 6:9] @ D[:3, :3],
                    np.zeros((len(src), 3)))

        _, info_in = est.estimate(*est.prepare_observations(*obs_at(0.0, 2.0)))
        # CALIBRATION: observations that lie ON the map must score ~1, not "somewhat far". The
        # ratio is only interpretable -- and only usable as a sigma multiplier -- if 1 really
        # does mean "as well surrounded as a typical manifold point".
        assert 0.7 < info_in['support_ratio'] < 1.3, info_in['support_ratio']
        # OFF the map: a wrench direction AND magnitude the manifold never contains at any pitch.
        pose, f, tau = obs_at(0.0, 2.0)
        f_off = np.tile([0.0, 0.0, 30.0], (len(f), 1))
        _, info_out = est.estimate(*est.prepare_observations(pose, f_off, tau))
        assert info_out['support_ratio'] > 2.0 * info_in['support_ratio'], \
            f"off-manifold observations must read as thin support: {info_in['support_ratio']:.2f}" \
            f" -> {info_out['support_ratio']:.2f}"
        # ... and thin support must WIDEN the reported sigma, which is the whole point: the gate
        # sees a number that already knows the answer came from the edge of the map.
        assert info_out['support_inflation'] > 1.0
        assert set(info_in) >= {'mixture', 'ambiguity', 'between_frac', 'separation',
                                'sigma_within', 'cov_mixture'}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_grid_estimator_fuses_probes_and_flags_uncertainty():
    """skills\\grid_estimator: the exhaustive-grid estimator must (a) recover a rigid belief error
    without multi-start, (b) FUSE probes by adding their energies -- the algebra the probe app
    relies on -- (c) weight rows by informativeness vs depth, and (d) report the campaign's
    uncertainty signals: 1/curvature at the minimum and the multi-modality flag."""
    import csv as _csv
    import shutil
    import tempfile

    from urlab.skills.grid_estimator import GridManifoldEstimator
    from urlab.skills.manifold import (FORCE_COLS, POSE_COLS, TORQUE_COLS, mats_from_vec6,
                                       vec6_from_mats)

    tmp = tempfile.mkdtemp()
    try:
        # Manifold: insertions at several pitch offsets, each with a pitch-DEPENDENT contact
        # direction, so pitch is identifiable from the wrench (the real manifold's structure).
        rows = []
        for p in np.arange(-12.0, 12.01, 1.0):
            for x in np.linspace(-20.0, -4.0, 40):
                f = 5.0 * np.array([-0.9, 0.0, 0.0]) + np.array([0.0, -0.25 * p, 0.0])
                rows.append([x, 0.0, 0.0, 0.0, p, 0.0] + list(f) + [0.0, 0.0, 0.0])
        path = os.path.join(tmp, 'manifold.csv')
        with open(path, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(rows)

        cfg = {'manifold_csv': path, 'estimate_dims': ['pitch_deg'], 'min_observations': 5,
               'scaling_constant_unit_force_to_mm': 1.5, 'scaling_constant_unit_torque_to_mm': 0.0,
               'wrench_representation': 'rawcap', 'min_force_n': 1.0,
               'grid': {'range': {'pitch_deg': 14.0}, 'step': {'pitch_deg': 0.25},
                        'curvature_probe_deg': 3.0, 'mode_threshold': 1.1}}
        est = GridManifoldEstimator(cfg)
        assert len(est.grid6) == 113, len(est.grid6)

        def observe(true_pitch, err_pitch):
            """Rows the robot would log at `true_pitch` with belief error `err_pitch`."""
            src = np.array([r for r in rows if abs(r[4] - true_pitch) < 1e-9])
            D = mats_from_vec6([0.0, 0.0, 0.0, 0.0, err_pitch, 0.0])
            pose = vec6_from_mats(mats_from_vec6(src[:, :6]) @ D)
            f = src[:, 6:9] @ D[:3, :3]
            return pose, f, np.zeros_like(f)

        # (a) single probe recovers the correction: believed @ T_corr ~= true, so corr ~= -err.
        pose, f, tau = observe(0.0, 4.0)
        v6, w6 = est.prepare_observations(pose, f, tau)
        T_corr, info = est.estimate(v6, w6)
        assert T_corr is not None, info
        assert abs(info['theta_corr']['pitch_deg'] + 4.0) <= 0.5, info['theta_corr']
        assert info['aggregator'] == 'grid' and info['candidates'] == 113
        assert info['modes'] >= 1 and 'pitch_deg' in info['curvature_uncertainty']
        # sigma is the same information in the DIM'S OWN UNITS (what the plot's bars show), so it
        # must be finite, positive, and inside the searched box for a well-determined estimate.
        assert 0.0 < info['sigma']['pitch_deg'] < 14.0, info['sigma']

        # (b) FUSION: the same belief error probed at two DIFFERENT true pitches (i.e. two
        # commanded biases) must fuse by ADDING energies and still yield that one correction.
        E1, n1, S1 = est.energy(*est.prepare_observations(*observe(-5.0, 4.0)))
        E2, n2, S2 = est.energy(*est.prepare_observations(*observe(+5.0, 4.0)))
        _, fused = est.solve(E1 + E2, n1 + n2, 0.5 * (S1 + S2))
        assert abs(fused['theta_corr']['pitch_deg'] + 4.0) <= 0.5, fused['theta_corr']
        assert fused['n_observations'] == n1 + n2
        # SUPPORT travels with the energy and is a RATIO against the map's own k-th-neighbour
        # distance -- for observations that lie ON the manifold it must be about 1, not 5.
        assert S1 is not None and S1.shape == E1.shape
        assert 0.2 < fused['support_ratio'] < 2.5, fused['support_ratio']

        # (c) depth weighting: deep rows outweigh approach rows (d' 0.7 -> 2.4), and the weights
        # are a normalized distribution either way.
        wd = est.row_weights(v6)
        assert abs(wd.sum() - 1.0) < 1e-9
        deep = v6[:, 0] >= -8.0
        if deep.any() and (~deep).any():
            assert wd[deep].mean() > wd[~deep].mean() * 2.0, 'deep rows must dominate'
        flat = GridManifoldEstimator(dict(cfg, grid=dict(cfg['grid'], info_weighting='none')))
        assert np.allclose(flat.row_weights(v6), 1.0 / len(v6))

        # (d) uncertainty: against a manifold whose wrench does NOT vary with pitch, every
        # correction matches equally well (the aliasing case) -- the basin flattens, so
        # 1/curvature must blow up relative to the identifiable manifold above.
        alias_rows = [[r[0], 0.0, 0.0, 0.0, r[4], 0.0, -4.5, 0.0, 0.0, 0.0, 0.0, 0.0]
                      for r in rows]
        apath = os.path.join(tmp, 'alias.csv')
        with open(apath, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(alias_rows)
        alias = GridManifoldEstimator(dict(cfg, manifold_csv=apath))
        src = np.array([r for r in alias_rows if abs(r[4]) < 1e-9])
        D = mats_from_vec6([0.0, 0.0, 0.0, 0.0, 4.0, 0.0])
        apose = vec6_from_mats(mats_from_vec6(src[:, :6]) @ D)
        av6, aw6 = alias.prepare_observations(apose, src[:, 6:9] @ D[:3, :3],
                                              np.zeros((len(src), 3)))
        _, alias_info = alias.estimate(av6, aw6)
        assert alias_info['uncertainty'] > 5.0 * info['uncertainty'], \
            f"aliased manifold must read as far more uncertain: {info['uncertainty']:.3g} " \
            f"vs {alias_info['uncertainty']:.3g}"
        assert alias_info['sigma']['pitch_deg'] > info['sigma']['pitch_deg']

        # (e) TWO estimated dims: the grid is the product, and every per-dim output follows.
        two = GridManifoldEstimator(dict(
            cfg, estimate_dims=['z_mm', 'pitch_deg'],
            grid={'range': {'z_mm': 8.0, 'pitch_deg': 14.0},
                  'step': {'z_mm': 0.5, 'pitch_deg': 0.5},
                  'curvature_probe': {'z_mm': 3.0, 'pitch_deg': 3.0}, 'mode_threshold': 1.1}))
        assert two.grid_shape == (33, 57) and len(two.grid6) == 33 * 57
        _, i2 = two.estimate(*two.prepare_observations(pose, f, tau))
        assert set(i2['theta_corr']) == {'z_mm', 'pitch_deg'}
        assert set(i2['sigma']) == {'z_mm', 'pitch_deg'}
        assert abs(i2['theta_corr']['pitch_deg'] + 4.0) <= 1.0, i2['theta_corr']

        # Too few observations must SKIP, never guess.
        none_corr, reason = est.estimate(v6[:2], w6[:2])
        assert none_corr is None and 'observations' in reason

        # A bad grid config must fail at CONSTRUCTION, before any robot motion -- including a
        # curvature probe too small to clear interpolation noise (the 1-deg failure mode).
        for bad in ({'range': {'pitch_deg': 0.0}}, {'step': {'pitch_deg': 0.0}},
                    {'curvature_probe': {'pitch_deg': 0.25}},
                    {'curvature_probe': {'pitch_deg': 99.0}}):
            try:
                GridManifoldEstimator(dict(cfg, grid=dict(cfg['grid'], **bad)))
            except ValueError:
                pass
            else:
                raise AssertionError(f'grid {bad} must raise at construction')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_success_basin_labels_gates_and_signs():
    """skills\\success_basin: the manifold's own trials are labelled by how deep they got, giving
    P(seat | offset). Three things must hold or the gate is silently wrong:
      (a) the depth rule labels trials and scores real insertions IDENTICALLY;
      (b) the belief-error -> PHYSICAL-offset inversion (getting it backwards mirrors the basin);
      (c) P(seat) of a POSTERIOR, which is the decision quantity -- the robot never knows its
          remaining error, only a distribution over it."""
    import csv as _csv
    import shutil
    import tempfile

    from urlab.skills.manifold import FORCE_COLS, POSE_COLS, TORQUE_COLS
    from urlab.skills.success_basin import SuccessBasin

    tmp = tempfile.mkdtemp()
    try:
        # Trials at many offsets: those with |z| <= 3 insert to x = -4, the rest wedge at -12.
        # Each trial is a separate run of x from -20 up, so the segmenter sees the resets.
        rows = []
        for z in np.arange(-8, 8.01, 0.5):
            deep = -4.0 if abs(z) <= 3.0 else -12.0
            for x in np.linspace(-20.0, deep, 40):
                rows.append([x, 0.0, float(z), 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        path = os.path.join(tmp, 'm.csv')
        with open(path, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(rows)

        b = SuccessBasin(path, ['z_mm'], {'depth_reference': 95.0, 'seat_margin_mm': 4.0,
                                          'neighbors': 5})
        assert len(b.offsets) == len(np.arange(-8, 8.01, 0.5)), len(b.offsets)
        # (a) ONE definition: the labels and is_seated() must agree on the same numbers.
        assert b.is_seated(b.seat_depth + 0.1) and not b.is_seated(b.seat_depth - 0.1)
        assert b.seated[np.abs(b.offsets[:, 0]) <= 2.5].all(), 'aligned trials must count seated'
        assert not b.seated[np.abs(b.offsets[:, 0]) >= 5.0].any(), 'wedged trials must not'
        # P(seat) tracks the physical structure
        p = b.p_seat(np.array([[0.0], [7.0]]))
        assert p[0] > 0.9 > p[1], p

        # (b) SIGN: belief error +z means the robot plans too far +z, so the part rides at -z.
        off = b.offset_of_error([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
        assert abs(off[0, 0] + 5.0) < 1e-6, off

        # (c) POSTERIOR P(seat): a posterior concentrated on a good offset must beat a diffuse
        # one that puts mass out in the wedge region, even with the same argmax.
        offs = np.array([[0.0], [7.0]])
        sharp = b.p_seat_posterior(offs, [1.0, 0.0])
        diffuse = b.p_seat_posterior(offs, [0.5, 0.5])
        assert sharp > diffuse, (sharp, diffuse)
        assert abs(diffuse - 0.5 * (p[0] + p[1])) < 1e-9

        # 'max' on a map with a crush-through outlier makes the gate un-openable -- the reason
        # the default is a percentile. One deep row is enough to demonstrate it.
        rows.append([40.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        path2 = os.path.join(tmp, 'm2.csv')
        with open(path2, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(rows)
        bmax = SuccessBasin(path2, ['z_mm'], {'depth_reference': 'max', 'neighbors': 5})
        assert bmax.seated.mean() < 0.05, 'a single deep outlier must sink the seat rate'
        bpct = SuccessBasin(path2, ['z_mm'], {'depth_reference': 95.0, 'neighbors': 5})
        assert bpct.seated.mean() > 0.3, 'the percentile default must stay usable'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_calibration_check_line_and_config():
    """apps/calibration_check: the probe line must run from -standoff to +overshoot along the
    connector's +X, monotone, at the requested resolution; and the shipped config must keep the
    guard INSTANT (persistence smears the contact pose by the extra travel) and the approach
    slow -- the trip point IS the measurement."""
    import yaml

    from urlab.apps.calibration_check import line_rows

    rows = line_rows(0.030, 0.005, 0.001)
    xs = [r[0, 3] for r in rows]
    assert abs(xs[0] + 0.030) < 1e-9 and abs(xs[-1] - 0.005) < 1e-9
    assert all(b > a for a, b in zip(xs, xs[1:])), 'the advance must be monotone in x'
    assert all(abs(r[1, 3]) < 1e-12 and abs(r[2, 3]) < 1e-12 for r in rows), \
        'a calibration probe moves along x ONLY'
    assert len(rows) >= 2 and len(line_rows(0.002, 0.0, 0.01)) >= 2

    with open(os.path.join(os.path.dirname(__file__), '..', 'configs',
                           'calibration_check.yaml')) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg['cycles'] >= 1 and cfg['standoff_distance_m'] > 0
    assert float(cfg['force_guard'].get('persistence_s', 0.0)) == 0.0, \
        'calibration probes need an INSTANT guard -- persistence smears the contact pose'
    assert float(cfg['speed']['approach_translation_mm_s']) <= 5.0, \
        'the approach must stay slow: the trip point is the measurement'
    assert float(cfg['force_guard']['max_force_n']) <= 20.0


def test_force_guard_persistence_debounces_transients():
    """robot/guard: with persistence_s set, the guard must IGNORE a spike shorter than the
    window, TRIP on a sustained press, and RESET its clock when the wrench drops under the
    limit between two shorter presses (two 0.3 s presses != one 0.6 s press)."""
    import time as _time

    from urlab.robot.guard import ForceGuard

    class FakeArm:
        def __init__(self):
            self.w = np.zeros(6)

        def wrench(self):
            return self.w

    arm = FakeArm()
    g = ForceGuard(arm, {'max_force_n': 10.0, 'persistence_s': 0.15})
    over = np.array([20.0, 0, 0, 0, 0, 0])

    # a short spike must NOT trip
    arm.w = over
    assert not g(), 'first over-limit sample must start the clock, not trip'
    arm.w = np.zeros(6)
    assert not g()
    # ... and the clock must have RESET: a new press starts from zero
    arm.w = over
    assert not g()
    _time.sleep(0.08)
    arm.w = np.zeros(6)
    assert not g()
    arm.w = over
    assert not g(), 'two sub-window presses must not add up across a gap'
    # a sustained press MUST trip, and report the duration
    _time.sleep(0.17)
    assert g(), 'the limit held past persistence_s -- the guard must trip'
    assert g.tripped_by and 'force' in g.tripped_by
    # reset clears the clock too
    g.reset()
    arm.w = over
    assert not g()
    # persistence 0 keeps the ORIGINAL instant behaviour
    g0 = ForceGuard(arm, {'max_force_n': 10.0})
    arm.w = over
    assert g0(), 'persistence_s absent/0 must trip on the first sample, as before'


def test_estimator_eval_collection_config():
    """configs/estimator_eval.yaml eval.collection must match the app's schema: a known mode,
    positive peck parameters, and -- when sweep_offsets is explicit -- 6-vectors. The app
    validates pre-motion; this catches a broken config before it reaches the robot box."""
    import yaml
    with open(os.path.join(os.path.dirname(__file__), '..', 'configs',
                           'estimator_eval.yaml')) as fh:
        cfg = yaml.safe_load(fh)
    col = cfg['eval'].get('collection') or {}
    assert str(col.get('mode', 'attempts')) in ('attempts', 'offset_sweep', 'peck'), col
    assert float(col.get('peck_retract_mm', 5.0)) > 0
    assert float(col.get('peck_timeout_s', 30.0)) > 0
    so = col.get('sweep_offsets')
    if so is not None:
        assert so and all(len(o) == 6 for o in so), \
            'sweep_offsets must be 6-vectors [x, y, z (m), roll, pitch, yaw (deg)]'
    # the zero-noise commit after the last attempt must not be silently disabled
    fi = cfg['eval'].get('final_insertion') or {}
    assert fi.get('enabled', True), \
        'eval.final_insertion must stay enabled: every collection mode ends with one ' \
        'zero-noise insertion from the final belief'
    # divergence bounds: present and sane (a diverged belief must terminate the trial well
    # before the >15 deg gripper/fixture collision regime)
    abt = cfg['eval'].get('abort_bounds') or {}
    assert 0 < float(abt.get('pos_mm', 10.0)) <= 20.0, abt
    assert 0 < float(abt.get('rot_deg', 15.0)) <= 15.0, abt
    # estimation.commit selects WHICH number is applied: the aggregator's vote or the raw
    # landscape argmin. Only those two exist, and the app rejects anything else pre-motion.
    commit = str(cfg['estimation'].get('commit', 'aggregator')).strip().lower()
    assert commit in ('aggregator', 'argmin'), commit
    src = open(os.path.join(os.path.dirname(__file__), '..', 'urlab', 'apps',
                            'estimator_eval.py')).read()
    assert "'aggregator', 'argmin'" in src, \
        'estimator_eval must validate estimation.commit pre-motion'
    assert "commit == 'argmin'" in src and '_argmin_estimate' in src, \
        'commit: argmin must commit the dense landscape argmin, not the ICP finals'
    # commit: argmin must SKIP the multi-start solver, not run it and throw the answer away --
    # the ICP is the expensive half of an attempt (num_initial_guesses x icp_iterations).
    body = src.split('def _argmin_estimate')[1].split('\ndef ')[0]
    assert 'estimator.estimate(' not in body and '.estimate(' not in body, \
        '_argmin_estimate must not invoke the ICP solver'
    assert '# NO ICP' in src, 'the estimate call must branch on commit BEFORE running the ICP'
    # ICP-only diagnostics must be optional, or argmin mode crashes on its first attempt
    assert "info.get('theta_hist') is not None" in src, \
        'theta_hist/res_hist are ICP-only and must be guarded'
    assert "+ ['commit']" in src, 'trials.csv must record which commit path ran'
    # Match diagnostics: opt-in, and the TRUTH passed in must be the inverse of the belief
    # error (believed @ C = true), not the error itself -- a sign slip here would draw a
    # mirrored 'truth' and send the debugging in the wrong direction.
    assert 'manifold_debug.figures(' in src and 'eval.debug_match' in src, \
        'estimator_eval must be able to emit match diagnostics'
    call = src.split('manifold_debug.figures(')[0][-600:]
    assert 'np.linalg.inv(' in call and 'mats_from_vec6' in call, \
        'the diagnostic truth must be inverse(err_before), not err_before'
    dm = cfg['eval'].get('debug_match') or {}
    assert dm.get('enabled') in (True, False), dm
    assert int(dm.get('every_n', 1)) >= 1 and int(dm.get('grid_points', 41)) >= 5, dm
    # The post-insertion dwell must stay UN-GUARDED and UNLOGGED: guarded, it would end on its
    # first cycle (the guard is already tripped at a contact stop); logged, it would flood the
    # observation set with duplicate deepest-contact rows and change the estimate.
    for app in ('estimator_eval.py', 'mode_pose_estimator_eval.py'):
        s = open(os.path.join(os.path.dirname(__file__), '..', 'urlab', 'apps', app)).read()
        assert 'hold_after_insertion_s' in s, app
        assert 'adm_ctl.hold(last_ref, hold_s, guard=None)' in s, \
            f'{app}: the dwell must be un-guarded'
        assert 'adm_ctl.hold(last_ref, hold_s, guard=None)\n' in s and \
            'on_step' not in s.split('adm_ctl.hold(last_ref, hold_s')[1][:80], \
            f'{app}: the dwell must not log observations'
    for cf in ('estimator_eval.yaml', 'mode_pose_estimator_eval.yaml'):
        with open(os.path.join(os.path.dirname(__file__), '..', 'configs', cf)) as fh:
            comp = yaml.safe_load(fh)['compliance']
        assert float(comp.get('hold_after_insertion_s', 0.0)) >= 0.0, cf


def test_manifold_interpolation_reduces_latching():
    """skills\\manifold interp_neighbors: the manifold is a FINITE sample of a continuous surface,
    so exact-NN matching LATCHES onto the single closest sample -- observations lying BETWEEN
    samples get dragged onto a sample, quantising the correction by the sample spacing. The
    optional soft correspondence blends close-enough samples so the target INTERPOLATES; on a
    coarse manifold the recovered correction must land far closer to the truth."""
    import csv as _csv
    import shutil
    import tempfile

    from urlab.skills.manifold import FORCE_COLS, POSE_COLS, TORQUE_COLS, ManifoldEstimator

    tmp = tempfile.mkdtemp()
    try:
        # COARSE manifold: the ridge z = x sampled every 2 mm -- the continuous surface exists
        # BETWEEN the samples. Constant contact direction (contributes nothing to the NN match).
        u_f, u_t = np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])
        rows = [[x, 0.0, x, 0.0, 0.0, 0.0] + list(5.0 * u_f) + list(0.5 * u_t)
                for x in np.arange(-10.0, 10.0 + 1e-9, 2.0)]
        path = os.path.join(tmp, 'coarse_manifold.csv')
        with open(path, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(rows)

        # Observations ON the continuous ridge, clustered BETWEEN samples 0 and 2 (x ~ 0.7):
        # the true correction is ZERO. z is the only estimated dim, so sliding along the ridge
        # cannot absorb the error -- any nonzero z correction IS the latching artifact.
        xs = np.linspace(0.6, 0.8, 30)
        obs6 = np.column_stack([xs, np.zeros_like(xs), xs] + [np.zeros_like(xs)] * 3)
        f_raw = np.tile(5.0 * u_f, (len(obs6), 1))
        tau_raw = np.tile(0.5 * u_t, (len(obs6), 1))

        cfg = {'manifold_csv': path, 'estimate_dims': ['z_mm'], 'icp_iterations': 15,
               'num_initial_guesses': 30, 'random_seed': 5}
        errs = {}
        for kn in (1, 4):
            est = ManifoldEstimator({**cfg, 'interp_neighbors': kn})
            v6, w6 = est.prepare_observations(obs6, f_raw, tau_raw)
            T_corr, info = est.estimate(v6, w6)
            assert T_corr is not None, info
            errs[kn] = abs(info['theta_corr']['z_mm'])
        # Exact NN latches onto the sample at (0,0): ~0.7 mm of pure quantisation error.
        assert errs[1] > 0.4, f'expected the latching artifact on the coarse manifold: {errs}'
        # Interpolation blends the neighbours and lands near the true (zero) correction.
        assert errs[4] < 0.2 and errs[4] < errs[1] / 2.0, \
            f'interpolation must beat exact NN on off-sample observations: {errs}'

        # 'None'/1 keep the feature OFF (exact NN), matching the residual_gate yaml convention.
        off = ManifoldEstimator({**cfg, 'interp_neighbors': 'None'})
        assert off.interp_neighbors == 1 and off.interp_tau is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tool_frames_shared_yaml_source():
    """configs/frames.yaml is the SINGLE source of tool0-attached frames (urlab/tool_frames.py):
    one yaml entry defines a frame (monitor units welcome), parent chains flatten to tool0,
    typos fail loudly, and drift against a config's legacy sections warns instead of rotting."""
    import shutil
    import tempfile

    from urlab.config import Config
    from urlab.tool_frames import check_drift, load_frames, load_targets
    from urlab.transforms import matrix_to_xyzrpy

    # The repo file: every catalogued frame present, tool0 the identity root, and the
    # banana_connector_finger_holder entry exactly as specified (159 mm +Z, rpy 180/0/-90 deg).
    frames = load_frames()
    assert np.allclose(frames['tool0'], np.eye(4))
    for name in ('fingertip', 'camera', 'grasp', 'banana_connector_finger_holder'):
        assert name in frames, f'missing frame {name!r}'
    assert 'connector_holder' not in frames, 'the holder chain is retired'
    xyz, rpy = matrix_to_xyzrpy(frames['banana_connector_finger_holder'])
    d = np.degrees(rpy)
    assert np.allclose(xyz, [0.0, 0.0, 0.159])
    assert np.isclose(abs(d[0]), 180.0) and np.isclose(d[1], 0.0) and np.isclose(d[2], -90.0)

    # targets: the recorded base_link <- frame poses (the mate for uncertain_sampling's
    # held_frame); every target must pair with a declared frame. The recorded NUMBERS are the
    # user's to re-measure whenever the mate moves, so assert the UNIT round-trip against the
    # yaml itself (monitor mm/deg -> m/rad), not a hard-coded pose.
    import yaml
    targets = load_targets()
    with open(os.path.join(ROOT, 'configs', 'frames.yaml')) as fh:
        raw = yaml.safe_load(fh)['targets']['banana_connector_finger_holder']
    xyz, rpy = matrix_to_xyzrpy(targets['banana_connector_finger_holder'])
    assert np.allclose(xyz * 1000.0, raw['xyz_mm'])
    assert np.allclose(np.degrees(rpy), raw['rpy_deg'])

    # No drift: a config whose legacy sections MATCH the catalogue stays silent.
    cfg = Config({'fingertip_grasp': {'xyz': [0.0, 0.0, 0.183],
                                      'rpy': [3.14159, 0.0, -1.5708]}})
    assert check_drift(frames, cfg) == []
    # Drift: a section that disagrees is CALLED OUT by name (the facade reads the section).
    cfg_bad = Config({'fingertip_grasp': {'xyz': [0.0, 0.0, 0.190],
                                          'rpy': [3.14159, 0.0, -1.5708]}})
    assert check_drift(frames, cfg_bad) == ['fingertip']

    tmp = tempfile.mkdtemp()
    try:
        # Parent CHAINING flattens to tool0: b sits 50 mm along a's z, a 100 mm along tool0's z.
        chain = os.path.join(tmp, 'chain.yaml')
        with open(chain, 'w') as fh:
            fh.write('frames:\n'
                     '  a: {xyz: [0.0, 0.0, 0.1], rpy: [0.0, 0.0, 0.0]}\n'
                     '  b: {parent: a, xyz_mm: [0.0, 0.0, 50.0], rpy_deg: [0.0, 0.0, 0.0]}\n')
        got = load_frames(path=chain)
        assert np.allclose(got['b'][:3, 3], [0.0, 0.0, 0.15])

        # Typos fail LOUDLY at load: unknown parent, parent cycle, redefined root.
        for body, why in (
                ('frames:\n  b: {parent: nope, xyz: [0, 0, 0.1], rpy: [0, 0, 0]}\n', 'unknown parent'),
                ('frames:\n'
                 '  c: {parent: d, xyz: [0, 0, 0.1], rpy: [0, 0, 0]}\n'
                 '  d: {parent: c, xyz: [0, 0, 0.1], rpy: [0, 0, 0]}\n', 'parent cycle'),
                ('frames:\n  tool0: {xyz: [0, 0, 0.1], rpy: [0, 0, 0]}\n', 'redefined root'),
                # from_cfg zero-fills unknown/missing keys, so these would otherwise SILENTLY
                # place the frame at its parent -- the classic quiet unit bug.
                ('frames:\n  e: {xyz_m: [0, 0, 0.1], rpy: [0, 0, 0]}\n', 'typoed pose key'),
                ('frames:\n  f: {}\n', 'no pose keys'),
                ('frames:\n  g: {xyz: [0, 0, 0.1], rpy: [0, 0, 0]}\n'
                 'targets:\n  gg: {xyz: [0, 0, 0.1], rpy: [0, 0, 0]}\n', 'target sans frame'),
                ('frames:\n  h: {xyz: [0, 0, 0.1], rpy: [0, 0, 0]}\n'
                 'targets:\n  h: {xyz_m: [0, 0, 0.1]}\n', 'typoed target key')):
            bad = os.path.join(tmp, 'bad.yaml')
            with open(bad, 'w') as fh:
                fh.write(body)
            try:
                load_frames(path=bad)
                load_targets(path=bad)     # target-level typos surface here
                assert False, f'expected ValueError for {why}'
            except ValueError:
                pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pose_block_accepts_monitor_units():
    """Calibration poses are pasted off the monitor, which prints mm/deg. `xyz_mm`/`rpy_deg` convert
    to the repo-standard m/rad; the unit lives in the KEY so it cannot be confused. Mixing both units
    for one triple must RAISE -- silently preferring one turns 90 mm into 90 m."""
    from urlab.config import _pose_si

    got = _pose_si({'xyz_mm': [90.71, 1073.48, -184.20], 'rpy_deg': [-0.79, -0.25, 88.92]})
    assert np.allclose(got['xyz'], [0.09071, 1.07348, -0.18420])
    assert np.allclose(got['rpy'], np.radians([-0.79, -0.25, 88.92]))
    assert 'xyz_mm' not in got and 'rpy_deg' not in got, 'the mm/deg keys must be consumed'

    si = {'xyz': [0.1, 0.0, 0.0], 'rpy': [0.0, 0.0, 1.5]}      # m/rad still works unchanged
    assert _pose_si(si) == si
    assert _pose_si({}) == {} and _pose_si(None) == {}

    for bad in ({'xyz': [0, 0, 0], 'xyz_mm': [0, 0, 0]}, {'rpy': [0, 0, 0], 'rpy_deg': [0, 0, 0]}):
        try:
            _pose_si(bad)
            assert False, f'expected ValueError for mixed units: {bad}'
        except ValueError:
            pass


def test_perturb_frames_model_different_error_sources():
    """The two perturb frames are physically different and must stay distinguishable.

    'connector' = IN-HAND error: the robot runs its nominal motion, so a pitch bias tilts the PART
    but leaves TRAVEL along the target's axis. 'target' = SOCKET error: the whole approach is
    rigidly misaimed, so travel rotates onto the part's OWN axis. Both tilt the part identically --
    only the travel direction tells them apart, which is why it is asserted here."""
    from urlab.skills.trajectory import perturb
    from urlab.transforms import xyzrpy_to_matrix

    # Ideal path: -5 cm -> 0 along the target's X, identity rotation (mate = last row = identity).
    dense = [xyzrpy_to_matrix([-s, 0, 0], [0, 0, 0]) for s in np.linspace(0.05, 0.0, 6)]
    pitch = 30.0
    R_bias = xyzrpy_to_matrix([0, 0, 0], np.radians([0, pitch, 0]))
    bias = R_bias                          # the trial's chosen delta (grid point or random draw)
    zeros = [0.0] * 6
    rng = np.random.default_rng(0)

    def travel_dir(path):
        v = path[-1][:3, 3] - path[0][:3, 3]
        return v / np.linalg.norm(v)

    in_hand = perturb(dense, bias, zeros, zeros, rng, frame='connector')
    assert np.allclose(travel_dir(in_hand), [1.0, 0.0, 0.0], atol=1e-9), \
        'in-hand error must leave travel on the TARGET axis (the robot moves nominally)'
    assert np.allclose(in_hand[-1][:3, :3], R_bias[:3, :3]), \
        'the part itself must still be tilted by the bias'

    socket = perturb(dense, bias, zeros, zeros, rng, frame='target')
    assert np.allclose(travel_dir(socket), R_bias[:3, 0], atol=1e-9), \
        "socket error must rotate travel onto the part's OWN axis"

    # 'held' is an alias for 'connector'; an unknown frame must fail loudly, not silently pick one.
    assert np.allclose(perturb(dense, bias, zeros, zeros, rng, frame='held'), in_hand)
    try:
        perturb(dense, bias, zeros, zeros, rng, frame='base')
        assert False, 'expected ValueError for an unknown perturb frame'
    except ValueError:
        pass


def test_uncertainty_is_uniform_within_its_bounds():
    """`uncertainty` is per-DOF ABSOLUTE lower/upper bounds (not half-widths, so a range need not be
    centred on zero). Draws fill the range and NEVER leave it -- these are hard limits, not sigmas."""
    from urlab.skills.trajectory import random_delta
    from urlab.transforms import matrix_to_xyzrpy

    lower = [0.0, 0.0, -0.005, 0.0, -15.0, 0.0]
    upper = [0.0, 0.0, 0.005, 0.0, 15.0, 0.0]
    rng = np.random.default_rng(3)
    draws = []
    for _ in range(4000):
        xyz, rpy = matrix_to_xyzrpy(random_delta(lower, upper, rng))
        draws.append(np.concatenate([xyz, np.degrees(rpy)]))
    d = np.asarray(draws)
    assert np.all(d >= np.array(lower) - 1e-9) and np.all(d <= np.array(upper) + 1e-9), \
        'a draw left its bounds -- these are HARD limits'
    for i in (2, 4):                       # the active DOFs fill their range, both ends
        assert d[:, i].min() < lower[i] * 0.97 and d[:, i].max() > upper[i] * 0.97
    for i in (0, 1, 3, 5):                 # lower == upper == 0 -> never moves
        assert np.all(d[:, i] == 0.0), f'DOF {i} has a zero-width range and must never move'

    # An ASYMMETRIC range must be honoured -- the old half-width form could not express this.
    off = [random_delta([0, 0, -0.004, 0, 0, 0], [0, 0, -0.001, 0, 0, 0], rng)[2, 3]
           for _ in range(500)]
    assert min(off) >= -0.004 - 1e-9 and max(off) <= -0.001 + 1e-9, 'asymmetric range not honoured'

    try:
        random_delta([0.0] * 6, [-1.0] + [0.0] * 5, rng)
        assert False, 'expected ValueError when upper < lower'
    except ValueError:
        pass


def test_grid_sweep_is_ordered_and_exhaustive():
    """Grid mode must step through the range IN ORDER -- deterministic, every combination exactly
    once, endpoints included -- not sample it randomly. The trial count IS len(grid)."""
    from urlab.skills.trajectory import grid_deltas
    from urlab.transforms import matrix_to_xyzrpy

    lower = [0.0, 0.0, -0.005, 0.0, -15.0, 0.0]
    upper = [0.0, 0.0, 0.005, 0.0, 15.0, 0.0]
    res = [0.0, 0.0, 0.005, 0.0, 15.0, 0.0]

    grid = grid_deltas(lower, upper, res)
    assert len(grid) == 9, f'3 z-steps x 3 pitch-steps = 9, got {len(grid)}'

    pts = []
    for g in grid:
        xyz, rpy = matrix_to_xyzrpy(g)
        pts.append((round(xyz[2], 6), round(math.degrees(rpy[1]), 4)))

    # EXACT order: z is the outer loop, pitch the inner (itertools.product, last axis fastest).
    expect = [(z, p) for z in (-0.005, 0.0, 0.005) for p in (-15.0, 0.0, 15.0)]
    assert pts == expect, f'grid not in sweep order:\n got {pts}\n want {expect}'
    assert len(set(pts)) == len(pts), 'a grid point repeated'
    assert grid_deltas(lower, upper, res)[0] is not None
    assert [tuple(map(lambda v: round(v, 6), p)) for p in pts][0] == (-0.005, -15.0), \
        'sweep must start at the lower corner'

    # Degenerate DOFs cost nothing; a spanning DOF with no step is a loud error, not a silent 1 point.
    assert len(grid_deltas([0.0] * 6, [0.0] * 6, [0.0] * 6)) == 1
    try:
        grid_deltas([0.0] * 6, [0.0, 0.0, 0.01, 0.0, 0.0, 0.0], [0.0] * 6)
        assert False, 'expected ValueError for a spanning DOF with zero resolution'
    except ValueError:
        pass

    # A range that is not a whole multiple of the step still hits BOTH endpoints.
    g2 = grid_deltas([0.0] * 6, [0.0, 0.0, 0.010, 0.0, 0.0, 0.0], [0.0, 0.0, 0.004, 0.0, 0.0, 0.0])
    zs = [round(matrix_to_xyzrpy(g)[0][2], 6) for g in g2]
    assert zs[0] == 0.0 and zs[-1] == 0.010, f'endpoints not hit exactly: {zs}'


def test_wrench_is_bridged_into_base_link():
    """getActualTCPForce() speaks UR `base`; the rest of urlab speaks ROS base_link (Rz(pi) apart).
    arm.wrench() must apply that bridge, or x/y silently come out NEGATED while z is fine -- which
    inverts exactly SOME of the admittance axes and mislabels the logged wrench columns."""
    from urlab.robot.arm import URArm

    class _FakeRtdeR:
        @staticmethod
        def getActualTCPForce():
            return [1.0, 2.0, 3.0, 0.4, 0.5, 0.6]     # in UR `base`

    arm = URArm.__new__(URArm)                        # no hardware: exercise wrench() alone
    arm.dry_run = False
    arm.rtde_r = _FakeRtdeR()
    w = arm.wrench()
    # Rz(pi): (x, y, z) -> (-x, -y, z), applied to BOTH the force and the torque triple.
    assert np.allclose(w, [-1.0, -2.0, 3.0, -0.4, -0.5, 0.6]), w
    assert np.isclose(np.linalg.norm(w[:3]), np.linalg.norm([1.0, 2.0, 3.0])), \
        'a pure rotation must preserve magnitude (so the force guard is unaffected)'


def test_common_yaml_is_the_shared_base_layer():
    """configs/_common.yaml is LOADED as the base under every config: top-level blocks a config
    does not define are inherited; a block the config defines is owned WHOLESALE (no per-key
    merge -- schema generations must never mix inside one speed: block); --set beats both; a
    directory without a _common.yaml inherits nothing."""
    import shutil
    import tempfile
    from urlab import config as C
    from urlab import tool_frames

    # Repo wiring: a demo config without a camera: block inherits the shared device...
    cfg = C.load('estimator_eval')
    assert cfg.get_path('camera.serial_no') == '218622272137'
    # ...and the pick app's mate comes from the frames catalogue (target_frame -> targets:).
    pick = C.load('cable_pick_estimate_assemble')
    assert pick.get_path('assembly.target_frame') in tool_frames.load_targets(pick)

    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, '_common.yaml'), 'w') as fh:
            fh.write('robot: {ip: 1.2.3.4, dry_run: true}\nspeed: {a: 1, b: 2}\nextra: 7\n')
        with open(os.path.join(tmp, 'demo.yaml'), 'w') as fh:
            fh.write('speed: {a: 9}\n')
        cfg = C.load(os.path.join(tmp, 'demo.yaml'))
        assert cfg.get_path('robot.ip') == '1.2.3.4'       # absent block -> inherited
        assert cfg.get('extra') == 7                       # top-level scalars inherit too
        assert cfg.get_path('speed.a') == 9                # defined block is owned...
        assert cfg.get_path('speed.b') is None, 'WHOLE-BLOCK ownership: no per-key merge'
        assert C.load(os.path.join(tmp, 'demo.yaml'),
                      ['robot.ip=9.9.9.9']).get_path('robot.ip') == '9.9.9.9'
        # _common.yaml itself must load flat, not recurse into itself.
        assert C.load(os.path.join(tmp, '_common.yaml')).get_path('speed.b') == 2
        os.remove(os.path.join(tmp, '_common.yaml'))
        assert C.load(os.path.join(tmp, 'demo.yaml')).get('robot') is None, \
            'no _common.yaml in the directory -> nothing inherited'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_estimator_eval_ground_truth_algebra():
    """The eval harness scores the estimator against a KNOWN truth, so its algebra must close the
    loop exactly: the injected right-multiplied belief error reads back verbatim through
    _gt_error (identity = perfect belief), and the estimator's own correction convention
    (mm translation, believed @ corr ~= true) applied with the exact inverse zeroes it."""
    from urlab.apps.estimator_eval import _corr_to_m, _gt_error
    from urlab.skills.trajectory import delta_from
    from urlab.transforms import xyzrpy_to_matrix

    T_true = xyzrpy_to_matrix([0.0, 0.0, 0.159], [math.pi, 0.0, -math.pi / 2])   # the banana holder
    inj = [0.003, 0.0, -0.002, 0.0, 4.0, 0.0]              # +3 mm x, -2 mm z, +4 deg pitch
    T_bel = T_true @ delta_from(inj)

    vec, pos_mm, rot_deg = _gt_error(T_true, T_bel)
    assert np.allclose(vec, [3.0, 0.0, -2.0, 0.0, 4.0, 0.0], atol=1e-9), vec
    assert np.isclose(pos_mm, math.hypot(3.0, 2.0)) and np.isclose(rot_deg, 4.0)
    # ... and a perfect belief scores zero.
    assert _gt_error(T_true, T_true)[1] < 1e-12

    # A PERFECT correction, in the estimator's return convention (translation in mm): the belief
    # update T_bel @ _corr_to_m(T_corr_mm) must land back on the truth exactly.
    T_corr_mm = T.inverse(delta_from(inj))
    T_corr_mm[:3, 3] *= 1000.0
    _, pos2, rot2 = _gt_error(T_true, T_bel @ _corr_to_m(T_corr_mm))
    assert pos2 < 1e-9 and rot2 < 1e-9, (pos2, rot2)


def test_estimator_eval_trial_error_plot_renders():
    """The per-trial error figure must actually render (the plotter is best-effort at runtime --
    an exception is swallowed with a warning, so only a file-exists check catches a broken plot).
    Data shape: index 0 = injected error, then one point per attempt."""
    import shutil
    import tempfile
    from urlab.apps.estimator_eval import _plot_trial_errors

    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'trial_001_errors.png')
        err6 = [[4.0, 0.0, -3.0, 0.0, 5.0, 0.0],           # injected
                [1.2, 0.0, -0.8, 0.0, 1.5, 0.0],           # after attempt 1
                [0.3, 0.0, 0.2, 0.0, 0.4, 0.0]]            # after attempt 2
        residuals = [0.42, float('nan')]                   # attempt 2's estimation was skipped
        rng = np.random.default_rng(0)
        res_all = [0.42 + 0.3 * rng.random(50),            # every guess, attempt 1
                   np.zeros(0)]                            # skipped -> no guesses
        l2_all = [2.0 * rng.random(50), np.zeros(0)]       # each guess's would-be L2 outcome
        _plot_trial_errors(path, 1, ['x_mm', 'z_mm', 'pitch_deg'], err6, residuals, 0.2,
                           None, res_all, l2_all)
        assert os.path.isfile(path) and os.path.getsize(path) > 0, \
            'plot did not render (the runtime warning path swallowed an error)'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bounds_deltas_hit_extremes():
    """eval.mode 'bounds' tests each nonzero uncertainty endpoint one DOF at a time: two
    deltas per two-sided DOF, one for a degenerate (lower == upper) DOF, nothing for all-zero
    DOFs -- and each delta moves ONLY its own DOF."""
    from urlab.skills import trajectory as traj
    from urlab.transforms import matrix_to_xyzrpy

    lo = [-0.005, 0.0, -0.003, 0.0, -5.0, 2.0]
    hi = [0.005, 0.0, 0.003, 0.0, 5.0, 2.0]
    ds = traj.bounds_deltas(lo, hi)
    assert len(ds) == 7                            # x: 2, z: 2, pitch: 2, yaw (lo == hi): 1
    xyz, rpy = matrix_to_xyzrpy(ds[0])             # first = x lower endpoint, nothing else
    assert np.isclose(xyz[0], -0.005) and np.allclose(xyz[1:], 0.0) and np.allclose(rpy, 0.0)
    xyz, rpy = matrix_to_xyzrpy(ds[-1])            # last = the degenerate yaw endpoint
    assert np.isclose(np.degrees(rpy[2]), 2.0) and np.allclose(xyz, 0.0)
    assert not traj.bounds_deltas([0.0] * 6, [0.0] * 6)   # all-zero bounds -> no trials

    # SIMULTANEOUS: every corner of the box -- all active DOFs at an extreme at once.
    # x/z/pitch two-sided -> 2^3 corners; the degenerate yaw rides along in every one.
    dc = traj.bounds_deltas(lo, hi, simultaneous=True)
    assert len(dc) == 8
    for d in dc:
        xyz, rpy = matrix_to_xyzrpy(d)
        assert np.isclose(abs(xyz[0]), 0.005) and np.isclose(abs(xyz[2]), 0.003)
        assert np.isclose(abs(np.degrees(rpy[1])), 5.0, atol=0.05)
        assert np.isclose(np.degrees(rpy[2]), 2.0, atol=0.05)
    assert not traj.bounds_deltas([0.0] * 6, [0.0] * 6, simultaneous=True)


def test_trajectory_noise_decay_and_bias():
    """traj.noised(): with zero std and a pitch bias, the delta is deterministic -- the bias
    times the per-attempt `scale`, shed linearly along the path by `decay_traj` (full at the
    first row, x(1 - f) at the last)."""
    from urlab.skills import trajectory as traj
    from urlab.transforms import matrix_to_xyzrpy

    rows = [np.eye(4)] * 5
    out = traj.noised(rows, np.random.default_rng(0), [0.0] * 6, 1,
                      decay_traj=0.5, scale=0.5, bias6=[0, 0, 0, 0, 4.0, 0])
    p = [np.degrees(matrix_to_xyzrpy(m)[1][1]) for m in out]
    assert np.isclose(p[0], 2.0), p          # 4 deg x scale 0.5 at the first waypoint
    assert np.isclose(p[-1], 1.0), p         # x (1 - 0.5) by the last waypoint
    assert np.all(np.diff(p) < 0)            # monotone linear shed in between


def test_estimator_eval_accumulation_rebase():
    """accumulate_observations re-projects stored rows into the updated belief. A logged pose
    is inverse(T_target) @ tool0 @ T_believed_old with T_believed_new = T_believed_old @ T_corr,
    so rel_new = rel_old @ T_corr exactly (mm); the wrench columns change frame by the same
    T_corr, i.e. transform_wrench with T_ba = inverse(T_corr) (metres for the cross term). The
    vectorized helper must match those reference implementations row by row."""
    from urlab.apps.estimator_eval import _corr_to_m, _rebase_rows
    from urlab.skills.manifold import mats_from_vec6, vec6_from_mats
    from urlab.transforms import inverse, transform_wrench

    rng = np.random.default_rng(4)
    rows = np.hstack([rng.uniform(-20, 20, (7, 3)), rng.uniform(-10, 10, (7, 3)),
                      rng.uniform(-8, 8, (7, 6))])
    T_corr = mats_from_vec6(np.array([1.5, -0.5, 2.0, 0.0, 3.0, 0.0]))    # mm / deg
    out = _rebase_rows(rows, T_corr)
    T_ba = inverse(_corr_to_m(T_corr))
    for r, o in zip(rows, out):
        assert np.allclose(o[:6], vec6_from_mats(mats_from_vec6(r[:6]) @ T_corr))
        f_b, tau_b = transform_wrench(r[6:9], r[9:12], T_ba)
        assert np.allclose(o[6:9], f_b) and np.allclose(o[9:12], tau_b)


def test_estimator_eval_config_is_wired_to_the_catalogue():
    """configs/estimator_eval.yaml must parse, name a held_frame present in BOTH the catalogue's
    frames: and targets: (the app refuses to move otherwise -- the frame IS the ground truth),
    and be structurally sound. The tuned NUMBERS (trial counts, bounds, dims) are the user's
    live experiment knobs -- assert their SHAPE, not their values, so tuning never breaks CI."""
    from urlab import config as C
    from urlab import tool_frames
    from urlab.skills.manifold import DIMS

    cfg = C.load(os.path.join(ROOT, 'configs', 'estimator_eval.yaml'))
    held = cfg.get('held_frame')
    assert held in tool_frames.load_frames(cfg), held
    assert held in tool_frames.load_targets(cfg), held

    ev = cfg.section('eval')
    assert int(ev.get('num_trials')) > 0 and int(ev.get('max_attempts')) > 0
    lo, hi = ev['perturbation']['lower'], ev['perturbation']['upper']   # [m x3, deg x3]
    assert len(lo) == 6 and len(hi) == 6
    assert all(a <= b for a, b in zip(lo, hi)), 'perturbation lower must not exceed upper'
    dims = cfg.get_path('estimation.estimate_dims')
    assert dims and all(d in DIMS for d in dims), dims
    agg = str(cfg.get_path('estimation.aggregator', 'ransac')).lower()
    assert agg in ('ransac', 'softmax'), agg
    # The trials.csv schema must name every estimated dim's correction and the ground-truth
    # error columns the summary aggregates (err_after_<dim> matches estimate_dims by name).
    from urlab.apps.estimator_eval import _fieldnames
    fields = _fieldnames(dims)
    for d in dims:
        assert f'corr_{d}' in fields and f'err_after_{d}' in fields


def test_manifold_rawcap_wrench_representation():
    """'rawcap' must scale the force feature by SATURATED magnitude (f/10 N capped at 3),
    identically in the manifold load and prepare_observations, self-adapt interp_tau, and
    still solve the synthetic alignment; bad names must fail at construction (pre-motion)."""
    import csv as _csv
    import shutil
    import tempfile

    from urlab.skills.manifold import (FORCE_COLS, POSE_COLS, TORQUE_COLS, ManifoldEstimator,
                                       mats_from_vec6, vec6_from_mats)

    tmp = tempfile.mkdtemp()
    try:
        u_f, u_t = np.array([0.6, 0.0, -0.8]), np.array([0.0, 1.0, 0.0])
        rows = []
        for x in np.linspace(-20.0, 0.0, 80):
            rows.append([x, 0.0, 0.0, 0.0, 0.0, 0.0] + list(5.0 * u_f) + list(0.5 * u_t))
        path = os.path.join(tmp, 'manifold.csv')
        with open(path, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(rows)

        base_cfg = {'manifold_csv': path, 'estimate_dims': ['z_mm', 'pitch_deg'],
                    'icp_iterations': 10, 'num_initial_guesses': 30, 'random_seed': 5,
                    'scaling_constant_unit_force_to_mm': 1.5,
                    'scaling_constant_unit_torque_to_mm': 0.0}
        est = ManifoldEstimator({**base_cfg, 'wrench_representation': 'rawcap'})
        # |f| = 5 N -> magnitude factor min(5/10, 3) = 0.5 -> force-feature norm 0.5 * 1.5.
        fn = np.linalg.norm(est.M12[:, 6:9], axis=1)
        assert np.allclose(fn, 0.75, atol=1e-6), fn[:3]
        # torque dropped: s_torque 0 -> zero columns
        assert np.allclose(est.M12[:, 9:12], 0.0)
        # saturation: a 100 N observation row caps at 3 (not 10)
        v6s, w6s = est.prepare_observations(np.zeros((2, 6)),
                                            np.array([100.0 * u_f, 5.0 * u_f]),
                                            np.array([0.5 * u_t, 0.5 * u_t]))
        assert np.isclose(np.linalg.norm(w6s[0, :3]), 3.0 * 1.5, atol=1e-6)
        assert np.isclose(np.linalg.norm(w6s[1, :3]), 0.5 * 1.5, atol=1e-6)
        # the synthetic belief error must still be recovered under rawcap
        true6 = np.array([r[:6] for r in rows])[::2]
        D = T.inverse(mats_from_vec6([0.0, 0.0, -2.5, 0.0, 6.0, 0.0]))
        obs6 = vec6_from_mats(mats_from_vec6(true6) @ D)
        f_raw = np.tile(5.0 * u_f, (len(obs6), 1))
        tau_raw = np.tile(0.5 * u_t, (len(obs6), 1))
        v6, w6 = est.prepare_observations(obs6, f_raw, tau_raw)
        T_corr, info = est.estimate(v6, w6)
        assert T_corr is not None, info
        undone = vec6_from_mats(D @ T_corr)
        assert np.all(np.abs(undone) < 0.3), f'rawcap correction off: {np.round(undone, 3)}'
        # unknown representation fails at CONSTRUCTION, never mid-run
        try:
            ManifoldEstimator({**base_cfg, 'wrench_representation': 'sqrtmag'})
            raise AssertionError('bad wrench_representation must raise')
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_solution_check_scores_and_gate():
    """The check must compute BOTH scores and label trust, but stay OBSERVATIONAL: the
    correction is ALWAYS applied (identical loop flow to the base app), and without
    calibration the scores read None while the estimate still passes through."""
    import csv as _csv
    import json as _json
    import shutil
    import tempfile

    from urlab.skills.manifold import FORCE_COLS, POSE_COLS, TORQUE_COLS, mats_from_vec6, \
        vec6_from_mats
    from urlab.skills.solution_check import CheckedManifoldEstimator

    tmp = tempfile.mkdtemp()
    try:
        u_f, u_t = np.array([0.6, 0.0, -0.8]), np.array([0.0, 1.0, 0.0])
        rows = []
        for x in np.linspace(-20.0, 0.0, 80):
            rows.append([x, 0.0, 0.0, 0.0, 0.0, 0.0] + list(5.0 * u_f) + list(0.5 * u_t))
        path = os.path.join(tmp, 'manifold.csv')
        with open(path, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(rows)
        calib = os.path.join(tmp, 'calib.json')
        with open(calib, 'w') as fh:                     # wide reference spreads
            _json.dump({k: list(np.linspace(0.0, 50.0, 40)) for k in
                        ('u_post', 'u_spread', 'u_split', 'u_res', 'u_depth')}, fh)

        cfg = {'manifold_csv': path, 'estimate_dims': ['z_mm', 'pitch_deg'],
               'icp_iterations': 8, 'num_initial_guesses': 20, 'random_seed': 5,
               'check': {'enabled': True, 'method': 'cauchy', 'flag_threshold': 0.99,
                         'n_candidates': 32, 'calibration_file': calib}}
        est = CheckedManifoldEstimator(cfg)
        true6 = np.array([r[:6] for r in rows])[::2]
        D = T.inverse(mats_from_vec6([0.0, 0.0, -1.5, 0.0, 3.0, 0.0]))
        obs6 = vec6_from_mats(mats_from_vec6(true6) @ D)
        f_raw = np.tile(5.0 * u_f, (len(obs6), 1))
        tau_raw = np.tile(0.5 * u_t, (len(obs6), 1))
        v6, w6 = est.prepare_observations(obs6, f_raw, tau_raw)
        T_corr, info = est.estimate(v6, w6)
        assert T_corr is not None, info
        chk = info['check']
        assert chk['rankavg2'] is not None and chk['cauchy'] is not None
        assert 0.0 <= chk['rankavg2'] <= 1.0 and 0.0 <= chk['cauchy'] <= 1.0
        assert set(chk['signals']) >= {'u_post', 'u_spread', 'u_res', 'u_depth'}
        assert chk['flagged'] is False

        # OBSERVATIONAL: even a threshold below any score must NOT block the correction --
        # the trust label flips to LOW but the estimate is applied, exactly like the base app.
        est.flag_threshold = -1.0
        T2, info2 = est.estimate(v6, w6)
        assert T2 is not None and info2['check']['flagged'] is True
        # method selection only changes WHICH score carries the label
        est.check_method = 'rankavg2'
        T3, info3 = est.estimate(v6, w6)
        assert T3 is not None and info3['check']['score'] == info3['check']['rankavg2']

        # WITHOUT calibration: signals logged, scores None, label never LOW (even at -1)
        cfg2 = dict(cfg)
        cfg2['check'] = {**cfg['check'], 'calibration_file': None, 'flag_threshold': -1.0}
        est2 = CheckedManifoldEstimator(cfg2)
        v6b, w6b = est2.prepare_observations(obs6, f_raw, tau_raw)
        T4, info4 = est2.estimate(v6b, w6b)
        assert T4 is not None and info4['check']['score'] is None
        assert info4['check']['flagged'] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_check_app_config_is_wired():
    """The check app's yaml must name the augmented manifold, the rawcap representation and
    a valid check block; the app module must expose the wrapped build_and_run."""
    import yaml

    with open(os.path.join(ROOT, 'configs', 'cable_pick_estimate_assemble_check.yaml')) as fh:
        cfg = yaml.safe_load(fh)
    est = cfg['estimation']
    assert str(est['manifold_csv']).endswith('banana_manifold_augmented.csv')
    assert est.get('wrench_representation') == 'rawcap'
    assert float(est['scaling_constant_unit_torque_to_mm']) == 0.0
    chk = est['check']
    assert str(chk['method']) in ('cauchy', 'rankavg2')
    assert 'on_flag' not in chk, 'the check is observational -- no gating key'
    assert 0.0 < float(chk['flag_threshold']) <= 1.0
    assert int(chk['n_candidates']) >= 8
    from urlab.apps import cable_pick_estimate_assemble_check as app
    assert callable(app.build_and_run) and callable(app.main)

    # estimator_eval carries the SAME estimator + check stack: the trust scores must have
    # trials.csv columns (next to the ground truth -- that is where they get validated), and its
    # estimation block must stay STRUCTURALLY valid. The manifold FILE is deliberately not
    # pinned: this config gets retargeted per connector (banana augmented map, hose map, ...),
    # so pinning a filename here only breaks the suite whenever the rig changes parts.
    from urlab.apps.estimator_eval import _fieldnames
    f = _fieldnames(['z_mm', 'pitch_deg'])
    for col in ('trust_rankavg2', 'trust_cauchy', 'trust_u_post', 'trust_u_split'):
        assert col in f, col
    with open(os.path.join(ROOT, 'configs', 'estimator_eval.yaml')) as fh:
        ecfg = yaml.safe_load(fh)
    eest = ecfg['estimation']
    assert str(eest['manifold_csv']).endswith('.csv')
    assert 'configs/data' in str(eest['manifold_csv']).replace('\\', '/')
    assert eest.get('wrench_representation') in ('rawcap', 'unit')
    assert str(eest['check']['method']) in ('cauchy', 'rankavg2')
    assert 'on_flag' not in eest['check']


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
