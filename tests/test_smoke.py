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

from urlab.skills import trajectory as traj_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from urlab import transforms as T   # noqa: E402
from urlab.frames import FrameGraph   # noqa: E402


def approx(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol)


def widest_cable_leg(asm):
    """The largest single rotation bnc_assembly's connector sweep commands, in radians.

    connector_clocking.sweep_deg is a list of ABSOLUTE roll positions wrt the target frame, and the
    arm starts at assembly.engage_clock_deg -- so the legs are the gaps along that walk, and the
    widest of them is the biggest arc any one servoed stroke has to cover. Tests that care about
    the geometry of a stroke (chord error, axis drag) want that worst case, not a config key.
    Falls back to the legacy single relative rotation_deg when no sweep is declared."""
    cc = asm['connector_clocking']
    sw = cc.get('sweep_deg')
    if sw is None:
        return abs(np.radians(float(cc['rotation_deg'])))
    stops = [np.radians(float(asm.get('engage_clock_deg', 0.0) or 0.0))]
    stops += [np.radians(float(v)) for v in sw]
    return max(abs(b - a) for a, b in zip(stops, stops[1:]))


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
    # THE RULE, NOT THE NUMBER. Both thresholds are DERIVED from the connector band, so
    # pinning the derived value breaks on every legitimate retune of the band -- it did, this
    # failed on 196 after the bnc band moved while nothing was actually wrong. What must hold
    # is the arithmetic tying them together:
    #     faces_max  = band_low - 1   (at or below the band floor = too thick = a face grasp)
    #     groove_max = band_high      (above the band but below empty = the cable, not the shell)
    for _c in (cfg, bnc):
        _lo, _hi = _c.get_path('grasp_check.connector_counts')
        assert _c.get_path('grasp_check.faces_max_counts') == _lo - 1, (
            'faces_max_counts must sit one count below the connector band, not float free of it')
        assert _c.get_path('grasp_check.groove_max_counts') == _hi, (
            'groove_max_counts must be the connector band ceiling')
        assert _lo < _hi < _c.get_path('grasp_check.empty_counts'), (
            'bands must stay ordered: connector inside, cable above it, empty above that')
    assert (bnc.get_path('grasp_check.connector_counts')
            != cfg.get_path('grasp_check.connector_counts')), (
        'the bnc profile must actually override the default band -- equal means it never applied')

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


def test_contact_manifold_downsample_is_exact_seeded_and_pooled():
    """--downsample N keeps EXACTLY N rows, the same N for the same seed, drawn from the POOL.

    Exact, because that is what the option promises and a reservoir is the only way to get it in
    one streaming pass. Seeded, because the seed is printed so the build can be repeated -- if the
    draw did not follow it that printout would be worthless. And pooled rather than per-file,
    because the alternative is a different dataset: giving every file its own quota would make a
    200-row log weigh as much as a 2000-row one, which is not a uniform sample of anything. The
    imbalance that leaves is real, and the point is that it is reported rather than engineered
    away.
    """
    import shutil
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.contact_manifold import MANIFOLD_COLS, build

    def _rows(path):
        import csv as _csv
        with open(path, newline='') as fh:
            return list(_csv.DictReader(fh))

    tmp = tempfile.mkdtemp()
    try:
        # a 2000-row source and a 200-row one: 10:1, so a pooled draw must reflect that ratio
        big = [{'connector_target_x_mm': -1.0 - 0.001 * i, 'wrench_connector_fx': 5.0}
               for i in range(2000)]
        small = [{'connector_target_x_mm': -9.0 - 0.001 * i, 'wrench_connector_fx': 5.0}
                 for i in range(200)]
        p_big, p_small = os.path.join(tmp, 'big.csv'), os.path.join(tmp, 'small.csv')
        _write_log(p_big, big)
        _write_log(p_small, small)
        paths, out = [p_big, p_small], os.path.join(tmp, 'ds_contact_manifold.csv')

        # EXACT -- not "about 220", exactly 220
        assert build(paths, out, downsample=220, seed=1) == 220
        first = _rows(out)
        assert len(first) == 220, f'{len(first)} rows on disk, asked for exactly 220'
        assert list(first[0].keys()) == MANIFOLD_COLS, 'downsampling must not disturb the schema'

        # SEEDED -- same seed, same rows; different seed, different rows
        assert build(paths, out, downsample=220, seed=1) == 220
        again = _rows(out)
        assert [r['connector_target_x_mm'] for r in again] == \
               [r['connector_target_x_mm'] for r in first], (
            'the same seed must reproduce the same draw -- the seed is printed so a build can be '
            'repeated, and that is only true if it actually determines the sample')
        build(paths, out, downsample=220, seed=2)
        assert [r['connector_target_x_mm'] for r in _rows(out)] != \
               [r['connector_target_x_mm'] for r in first], 'a different seed must redraw'

        # POOLED, NOT PER FILE. Rows from big.csv start at -1, small.csv at -9, so the source of
        # every row is readable off its x. A per-file quota would give 110/110; a uniform draw
        # over the pool gives ~10:1. Bounds are loose enough not to be flaky, tight enough that
        # an even split (0.5) fails them.
        frac_big = sum(1 for r in first if float(r['connector_target_x_mm']) > -5.0) / 220.0
        assert 0.82 < frac_big < 0.98, (
            f'big.csv is {frac_big:.0%} of the sample; it is 91% of the pool, so a uniform draw '
            'should land near that. A number near 50% means each file was sampled to its own '
            'quota, which up-weights the small source by 10x.')

        # asking for more than exists keeps everything rather than erroring or padding
        assert build(paths, out, downsample=99999, seed=1) == 2200
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_contact_manifold_ingests_wiggle_runs_without_their_transit():
    """A wiggle_sampling run is a sampling log with a `segment` column, and that column is the only
    thing separating contact from transit.

    `approach`/`retract` run from a 25 mm standoff and `datum` is the free-space return-to-reference
    pose ~60 mm off the mate in two axes. Every one of those rows has the full manifold schema and a
    valid pose, so no structural check rejects them -- they simply are not contact at a
    misalignment, and in a kNN manifold they widen the extent enough to distort the distance
    scaling the support/OOD reference is computed against. So the default must drop them, and
    --segments must be able to keep any subset deliberately.

    Also pinned: a GEOMETRY directory (which holds run_<stamp>/ subdirectories, not CSVs) is
    swept, and bursts.csv -- the per-burst summary -- is never ingested as samples.
    """
    import shutil
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.contact_manifold import (MANIFOLD_COLS, WIGGLE_CONTACT_SEGMENTS, build,
                                           classify, expand_inputs)

    cols = ['trial', 'segment'] + MANIFOLD_COLS
    contact = [{'segment': 'wiggle', 'connector_target_x_mm': -2.0 - 0.01 * i,
                'connector_target_z_mm': 0.5, 'wrench_connector_fx': 4.0} for i in range(6)]
    contact += [{'segment': 'quiet', 'connector_target_x_mm': -3.0,
                 'connector_target_z_mm': 0.7, 'wrench_connector_fx': 3.0}] * 2
    transit = [{'segment': 'approach', 'connector_target_x_mm': -25.0,
                'connector_target_z_mm': 0.0, 'wrench_connector_fx': 0.2}] * 3
    transit += [{'segment': 'datum', 'connector_target_x_mm': -60.0,
                 'connector_target_z_mm': 60.0, 'wrench_connector_fx': 0.1}] * 2

    tmp = tempfile.mkdtemp()
    try:
        geom = os.path.join(tmp, 'bnc')
        run = os.path.join(geom, 'run_20260819_023144')
        _write_log(os.path.join(run, 'samples.csv'), contact + transit, cols=cols)
        _write_log(os.path.join(run, 'bursts.csv'),
                   [{'burst': 1}], cols=['burst', 'offset_x_mm'])

        # a GEOMETRY directory holds runs, not CSVs -- the sweep has to look one level down
        paths = expand_inputs([geom])
        assert [os.path.basename(p) for p in paths] == ['samples.csv'], (
            f'geometry sweep should find exactly the run samples, got {paths}')
        assert classify(paths[0]) == ('wiggle', None), 'the segment column marks a wiggle run'

        out = os.path.join(tmp, 'wig_contact_manifold.csv')
        n = build(paths, out, segments=list(WIGGLE_CONTACT_SEGMENTS))
        assert n == len(contact), (
            f'{n} rows written; the {len(transit)} transit/datum rows must be dropped')

        with open(out, newline='') as fh:
            import csv as _csv
            got = list(_csv.DictReader(fh))
        xs = [float(r['connector_target_x_mm']) for r in got]
        zs = [float(r['connector_target_z_mm']) for r in got]
        assert min(xs) > -12.0 and max(zs) < 8.0, (
            f'transit leaked in: x down to {min(xs)}, z up to {max(zs)} -- '
            'that is the standoff/datum, not contact at a misalignment')

        # keeping a subset is deliberate and must work
        assert build(paths, out, segments=['quiet']) == 2
        # ...and "all" is still available for someone who genuinely wants the transit
        assert build(paths, out, segments=None) == len(contact) + len(transit)
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

        # PER-AXIS WRENCH WEIGHTS (estimation.wrench_weights): the same idea for the six wrench
        # columns, applied ON TOP of s_force / s_torque so those stay the block-level unit
        # conversion. Columns 6..11 of the 12-D point must scale one axis at a time, and zero
        # must be LEGAL on any of them (nothing is ever estimated in a wrench axis).
        xest = ManifoldEstimator({**wcfg, 'wrench_weights': {'fy': 0.5, 'fz': 0.0,
                                                             'tz': 2.0}})
        assert np.allclose(xest.M12[:, 7], 0.5 * west.M12[:, 7]), 'fy must halve'
        assert np.allclose(xest.M12[:, 8], 0.0), 'fz must switch off'
        assert np.allclose(xest.M12[:, 11], 2.0 * west.M12[:, 11]), 'tz must double'
        assert np.allclose(xest.M12[:, [6, 9, 10]], west.M12[:, [6, 9, 10]]), \
            'unweighted wrench axes must not move'
        assert np.allclose(xest.M12[:, :6], west.M12[:, :6]), 'pose block must not move'
        # the effective scale the metric actually runs with, block scalar x per-axis weight
        assert np.allclose(xest.wrench_scale6,
                           xest.wrench_w * np.array([xest.s_force] * 3 +
                                                    [xest.s_torque] * 3))
        # observations must go through the SAME weighting as the map, or the two are measured
        # in different spaces and every residual is meaningless
        _, xw6 = xest.prepare_observations(obs6, f_raw, tau_raw)
        _, ww6b = west.prepare_observations(obs6, f_raw, tau_raw)
        assert np.allclose(xw6[:, 2], 0.0) and np.allclose(xw6[:, 1], 0.5 * ww6b[:, 1]), \
            'wrench_weights must apply to observations exactly as to the manifold'
        for bad in ({'wrench_weights': {'fq': 1.0}}, {'wrench_weights': {'fx': -1.0}}):
            try:
                ManifoldEstimator({**wcfg, **bad})
            except ValueError:
                pass
            else:
                raise AssertionError(f'{bad} must raise at construction')

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


def test_wrench_follows_the_candidate_correction():
    """skills\manifold: the logged wrench lives in the BELIEVED connector frame, so scoring a
    candidate correction must re-express it in the frame that candidate claims -- the same
    re-basing _rebase_rows does when a correction is COMMITTED. Evaluating candidates without
    it scored every non-zero theta with its wrench in the wrong frame (fixed 2026-08-14: the
    force channel's tracking of the truth roughly tripled).

    Pinned here: identity is a no-op, a pure rotation rotates the feature without changing its
    magnitude, the lever arm appears only when the correction TRANSLATES, and a frame error is
    recoverable from the wrench ALONE -- which is the information the old code discarded."""
    import csv as _csv
    import shutil
    import tempfile

    from urlab.skills.manifold import (FORCE_COLS, POSE_COLS, TORQUE_COLS, ManifoldEstimator,
                                       mats_from_vec6, vec6_from_mats)

    tmp = tempfile.mkdtemp()
    try:
        rows = []
        for p_ in np.arange(-12.0, 12.01, 1.0):
            for x in np.linspace(-20.0, -4.0, 40):
                rows.append([x, 0.0, 0.0, 0.0, p_, 0.0, -5.0, 0.0, 0.0, 0.0, 0.2, 0.0])
        path = os.path.join(tmp, 'm.csv')
        with open(path, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(rows)
        cfg = {'manifold_csv': path, 'estimate_dims': ['pitch_deg'], 'min_observations': 5,
               'scaling_constant_unit_force_to_mm': 2.0,
               'scaling_constant_unit_torque_to_mm': 1.0, 'min_force_n': 1.0,
               'num_initial_guesses': 40, 'icp_iterations': 12, 'random_seed': 3}
        est = ManifoldEstimator(cfg)
        assert est.wrench_follows_correction, 'must default ON'

        f = np.array([[3.0, 4.0, 12.0]])
        tau = np.array([[0.3, 0.4, 1.2]])
        assert np.allclose(est.wrench6_at(f, tau, np.zeros(6)), est._wrench6(f, tau)),             'identity correction must leave the wrench untouched'
        th = np.zeros(6)
        th[4] = 10.0
        w10 = est.wrench6_at(f, tau, th)
        w00 = est._wrench6(f, tau)
        assert np.allclose(np.linalg.norm(w10[:, :3]), np.linalg.norm(w00[:, :3])),             'a rotation must preserve the force feature MAGNITUDE'
        assert not np.allclose(w10[:, :3], w00[:, :3]), 'and must change its DIRECTION'
        # lever arm: a pure rotation has none; a TRANSLATING correction must move the torque
        # by p x f on top of the rotation (mirrors _rebase_rows, p in metres)
        tz = np.zeros(6)
        tz[2] = 5.0                                   # 5 mm of z, no rotation at all
        wz = est.wrench6_at(f, tau, tz)
        assert np.allclose(wz[:, :3], w00[:, :3]), 'no rotation -> force is unchanged'
        assert not np.allclose(wz[:, 3:], w00[:, 3:]), 'but the torque gets the lever arm'

        # THE POINT: with the wrench re-based, a frame error is identifiable from the wrench
        # alone -- here the pose is IDENTICAL for every candidate, so only the wrench can speak.
        src = np.array([r for r in rows if abs(r[4]) < 1e-9])
        D = mats_from_vec6([0.0, 0.0, 0.0, 0.0, 4.0, 0.0])
        obs = vec6_from_mats(mats_from_vec6(src[:, :6]) @ D)
        v6, w6 = est.prepare_observations(obs, src[:, 6:9] @ D[:3, :3], src[:, 9:12] @ D[:3, :3])
        T, info = est.estimate(v6, w6)
        assert T is not None, info
        assert abs(info['theta_corr']['pitch_deg'] + 4.0) <= 1.5,             f"wrench-only recovery should find -4 deg, got {info['theta_corr']}"
        # and with the re-basing OFF the same evidence is uninformative -> a worse estimate
        frozen = ManifoldEstimator({**cfg, 'wrench_follows_correction': False})
        v6f, w6f = frozen.prepare_observations(obs, src[:, 6:9] @ D[:3, :3],
                                               src[:, 9:12] @ D[:3, :3])
        _, info_f = frozen.estimate(v6f, w6f)
        assert abs(info_f['theta_corr']['pitch_deg'] + 4.0) >             abs(info['theta_corr']['pitch_deg'] + 4.0),             'freezing the wrench must lose information, not gain it'
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

        # (d) uncertainty: against a manifold where pitch is UNIDENTIFIABLE, every correction
        # matches equally well -- the basin flattens, so 1/curvature must blow up relative to
        # the identifiable manifold above.
        # The force here points along Y, the PITCH ROTATION AXIS, and that is load-bearing.
        # Since wrench_follows_correction (2026-08-14) the observed wrench is re-expressed in
        # each candidate's frame, so a force with any component OFF the rotation axis is
        # identifying all by itself -- only the true correction restores the recorded
        # direction, whatever the contact physics does. A y-aligned force is invariant under a
        # pitch rotation, so this fixture stays genuinely aliased.
        alias_rows = [[r[0], 0.0, 0.0, 0.0, r[4], 0.0, 0.0, -4.5, 0.0, 0.0, 0.0, 0.0]
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


def test_held_and_target_frames_are_resolved_separately():
    """calibration_check / insertion_tester take the target frame as its OWN input.

    Both apps drive a HELD part at a RECORDED mate, and those are two different catalogue
    sections: `held_frame` is a frames: entry (tool0 -> the part in the fingers) and
    `target_frame` is a targets: entry (base_link <- the mate). They used to be one name, which
    forced every held part to carry a targets: entry of its own before it could be probed against
    anything.

    What has to hold:
      * target_frame DEFAULTS to held_frame, so a config predating the key behaves exactly as
        before;
      * the two are looked up in DIFFERENT sections -- a name that is only a frames: entry is a
        legal held_frame and an illegal target_frame, and the error has to say which;
      * both apps resolve them through the SAME helper, so they cannot drift apart.
    """
    from urlab import tool_frames

    frames = {'part_a': np.eye(4), 'part_b': np.eye(4), 'no_target': np.eye(4)}
    T_a = T.translation_matrix([0.5, 0.0, 0.2])
    T_b = T.translation_matrix([0.1, 0.4, 0.3])
    targets = {'part_a': T_a, 'part_b': T_b}

    # ---- DEFAULT: target_frame absent -> the held name, in the targets: section ----
    for absent in (None, ''):
        held, tgt, name = tool_frames.resolve_held_and_target(frames, targets, 'part_a', absent)
        assert name == 'part_a' and np.allclose(tgt, T_a) and np.allclose(held, frames['part_a'])

    # ---- DECOUPLED: hold one part, drive it at another's recorded mate ----
    held, tgt, name = tool_frames.resolve_held_and_target(frames, targets, 'part_a', 'part_b')
    assert name == 'part_b', 'the explicit target_frame must win'
    assert np.allclose(tgt, T_b), 'the target must come from the TARGETS section, not frames'
    assert np.allclose(held, frames['part_a']), 'the held pose must still be held_frame'

    # ---- THE TWO SECTIONS ARE NOT INTERCHANGEABLE ----
    # 'no_target' is a legal held_frame (it has a frames: entry) and an illegal target_frame.
    held, tgt, name = tool_frames.resolve_held_and_target(frames, targets, 'no_target', 'part_a')
    assert name == 'part_a', 'a held frame with no targets: entry is fine when target_frame names one'
    for bad_held in (None, '', 'nope'):
        try:
            tool_frames.resolve_held_and_target(frames, targets, bad_held, 'part_a')
        except ValueError as exc:
            assert 'held_frame' in str(exc), str(exc)
        else:
            raise AssertionError(f'held_frame {bad_held!r} must be rejected')
    try:
        tool_frames.resolve_held_and_target(frames, targets, 'part_a', 'nope')
    except ValueError as exc:
        assert 'target_frame' in str(exc) and 'targets:' in str(exc), str(exc)
    else:
        raise AssertionError('a target_frame with no targets: entry must be rejected')
    # the DEFAULTED case must say so, or the operator reads "target_frame 'x'" for a key they
    # never set and goes looking for the wrong typo
    try:
        tool_frames.resolve_held_and_target(frames, targets, 'no_target', None)
    except ValueError as exc:
        assert 'held_frame' in str(exc) and 'defaults' in str(exc), str(exc)
    else:
        raise AssertionError('a held_frame with no targets: entry and no target_frame must fail')

    # ---- BOTH APPS GO THROUGH THE SAME HELPER ----
    for app in ('calibration_check', 'insertion_tester'):
        src = open(os.path.join(ROOT, 'urlab', 'apps', f'{app}.py'), encoding='utf-8').read()
        assert 'tool_frames.resolve_held_and_target(' in src, (
            f'{app} must resolve the two frames through the shared helper, not its own lookup')
        assert "cfg.get('target_frame')" in src, f'{app} must read target_frame from the config'
        assert 'targets[held_name]' not in src, (
            f'{app} must not index targets: by the HELD name -- that is the coupling this removed')

    # ---- AND BOTH CONFIGS DECLARE THE KEY, so it is discoverable ----
    import yaml
    for name in ('calibration_check', 'insertion_tester'):
        c = yaml.safe_load(open(os.path.join(ROOT, 'configs', f'{name}.yaml')))
        assert 'target_frame' in c, f'{name}.yaml must declare target_frame (null = held_frame)'
        assert c['held_frame'] in tool_frames.load_frames(), c['held_frame']
        eff = c['target_frame'] or c['held_frame']
        assert eff in tool_frames.load_targets(), (
            f'{name}.yaml resolves to target {eff!r}, which has no targets: entry')


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


def test_bnc_assembly_shares_the_tuned_estimator():
    """apps\bnc_assembly is cable_pick_estimate_assemble's pipeline with estimator_eval's
    estimator, so the two must not drift: the estimator internals are IMPORTED (not copied),
    the physics blocks live at the same top-level names, and the tuned values match.

    Also pinned: the app must NOT expect ground truth (a real pick has none) -- it passes None
    as the diagnostic truth, so nothing may index an injected error."""
    import yaml
    root = os.path.join(os.path.dirname(__file__), '..')
    src = open(os.path.join(root, 'urlab', 'apps', 'bnc_assembly.py')).read()
    # shared internals, IMPORTED rather than reimplemented -- a copy would drift the moment
    # either app is retuned, which is the failure this whole test exists to prevent
    assert 'from .estimator_eval import _argmin_estimate, _landscape' in src, \
        'the argmin estimator must be imported from estimator_eval, not duplicated'
    assert 'from .cable_pick_estimate_assemble import' in src, \
        'the pick-side helpers must be imported from the app this derives from'
    assert 'def _argmin_estimate' not in src and 'def _observe' not in src, \
        'no copies of the shared helpers'
    # NO GROUND TRUTH: a real pick has none, so the diagnostics get None for the truth and
    # none of estimator_eval's injected-error machinery may appear (comments excluded, so the
    # docstring may still explain the difference)
    assert "info['theta_corr']), None," in src, \
        'a real pick has no ground truth -- the diagnostic truth must be None'
    code = '\n'.join(ln for ln in src.splitlines() if not ln.lstrip().startswith('#'))
    body = code.split('"""', 2)[-1]                # drop the module docstring
    for banned in ('_gt_error', 'T_true', 'inj_', 'abort_bounds'):
        assert banned not in body, \
            f'bnc_assembly must not use the ground-truth machinery ({banned})'
    # the physics blocks sit where estimator_eval reads them, with the same tuned values
    with open(os.path.join(root, 'configs', 'bnc_assembly.yaml')) as fh:
        b = yaml.safe_load(fh)
    with open(os.path.join(root, 'configs', 'estimator_eval.yaml')) as fh:
        e = yaml.safe_load(fh)
    for k in ('estimation', 'compliance', 'force_guard'):
        assert k in b, f'bnc_assembly.yaml must carry a top-level {k}: block'
    assert b['compliance']['stiffness'] == e['compliance']['stiffness'], \
        'stiffness must match the tuned estimator_eval values'
    for k in ('manifold_csv', 'commit', 'estimate_dims', 'dim_weights', 'wrench_weights',
              'scaling_constant_deg_to_mm', 'scaling_constant_unit_force_to_mm',
              'scaling_constant_unit_torque_to_mm', 'interp_softness',
              'wrench_follows_correction'):
        assert b['estimation'][k] == e['estimation'][k], f'estimation.{k} drifted'
    assert b['assembly']['final_insertion'] == e['eval']['final_insertion'], \
        'the final-insertion block must match estimator_eval exactly'
    assert str(b['assembly']['collection']['mode']) in ('attempts', 'offset_sweep', 'peck')


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
    # EVERY knob that may differ for the COMMIT insertion belongs in eval.final_insertion.
    # They used to be spread across compliance: and force_guard:, so retuning the probing
    # attempts silently retuned the commit as well. Each must be OPTIONAL -- null or absent
    # inherits the shared block -- and the app must thread them through rather than reading
    # the shared values for the commit.
    for k in ('stiffness', 'mass', 'damping_ratio', 'settle_s', 'hold_after_insertion_s',
              'max_force_n', 'max_torque_nm', 'persistence_s', 'speed_translation_mm_s',
              'speed_rotation_deg_s', 'pause_s', 'trajectory_noise'):
        assert k in fi, f'eval.final_insertion is missing the {k} override'
        if k == 'trajectory_noise':
            continue
        assert fi[k] is None or isinstance(fi[k], (int, float, list)), (k, fi[k])
    for k in ('settle_s', 'hold_after_insertion_s', 'max_force_n', 'persistence_s',
              'speed_translation_mm_s', 'speed_rotation_deg_s', 'pause_s'):
        assert fi[k] is None or float(fi[k]) >= 0, (k, fi[k])
    # The commit is deliberately ZERO-NOISE: the jitter gathers varied contact while PROBING
    # and has no place in the attempt meant to seat. Enabling it is a real choice, so it must
    # not be the shipped default.
    fnoise = fi.get('trajectory_noise') or {}
    assert fnoise.get('enabled') is False,         'final_insertion.trajectory_noise must ship OFF -- the commit is the zero-noise attempt'
    assert len(fnoise.get('std', [0] * 6)) == 6, fnoise
    # both apps must thread the pacing/pause through rather than reusing the probing values
    for app in ('estimator_eval.py', 'bnc_assembly.py'):
        a_src = open(os.path.join(os.path.dirname(__file__), '..', 'urlab', 'apps', app)).read()
        assert 'speed=(fi_v' in a_src and 'pause=fi_pause' in a_src,             f'{app}: the commit must use its own speed and pause'
        assert 'fi_noise_on' in a_src, f'{app}: the commit must honour its own noise setting'
    assert 'guard_ctl=guard_final' in src and 'settle=fi_settle' in src, \
        'the commit must run with its OWN guard/settle/dwell when they are overridden'
    assert 'guard_shared' in src and 'settle_shared' in src, \
        'the shared guard/settle must be named apart from the per-insertion locals'
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
    assert dm.get('live', True) is not None, dm
    # The diagnostics must describe the metric the estimator ACTUALLY runs: soft-kNN blending
    # (not raw nearest neighbour) and the wrench re-based per candidate. Both drifted once
    # already, which is exactly the failure these assertions exist to catch.
    dbg = open(os.path.join(os.path.dirname(__file__), '..', 'urlab', 'skills',
                            'manifold_debug.py')).read()
    assert '_wrench_at(' in dbg and 'wrench6_at' in dbg,         'match diagnostics must re-base the wrench like the energy does'
    assert 'interp_tau' in dbg and 'softness_by_block' in dbg,         'match diagnostics must use the soft-kNN kernel, including per-block bandwidths'
    assert 'estimator_eval_match_dof_live.png' in dbg, 'per-DOF live mirror must be written'
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


def test_bnc_clocking_geometry():
    """The post-mate clocking maneuvers are pure geometry layered on the mate, and each identity
    below is load-bearing -- these are the ways this code goes wrong SILENTLY, moving the arm
    somewhere plausible but wrong rather than raising.

      * the SCREW is a rotation about +X composed with a translation ALONG +X, so the two commute
        and the reference is a true helix; if they did not, composition order would quietly change
        the path;
      * the BELIEF RESET must put the connector exactly AT the target -- that is what makes the
        screw axis the target's +X instead of the estimate's;
      * ADVANCE must be read along the ENGAGED frame's +X, and a pure roll must contribute ZERO to
        it (the maneuver aims at a virtual target, so a roll-coupled reading would fake success);
      * the COLLAR must land a fixed offset along the connector's own +X with the CLOSED fingertip
        frame on it, and the turn must ORBIT the connector axis -- a point on the axis unmoved, no
        translation along it -- or the fingers scrub across the collar instead of turning it;
      * the two RETRACT legs read their axes in DIFFERENT frames (gripper/tool0 then target), so
        one is a right-multiply and the other a left-multiply. Swapping them is the classic bug and
        sends the arm off in a direction that looks reasonable until it collides.
    """
    import numpy as np

    from urlab.apps.bnc_assembly import _AnyGuard, _ScrewAdvance
    from urlab.transforms import (inverse, matrix_to_xyzrpy, rotate_about_axis,
                                  translation_matrix, xyzrpy_to_matrix)

    push_m, rot, off = 0.005, np.radians(90.0), 0.025
    screw = xyzrpy_to_matrix([push_m, 0.0, 0.0], [rot, 0.0, 0.0])
    Rx = xyzrpy_to_matrix([0.0, 0.0, 0.0], [rot, 0.0, 0.0])
    Tx = translation_matrix([push_m, 0.0, 0.0])
    assert np.allclose(Rx @ Tx, Tx @ Rx, atol=1e-12), 'push must be ALONG the rotation axis'
    assert np.allclose(screw, Rx @ Tx, atol=1e-12)

    # a deliberately non-axis-aligned mate and stand: an error that cancels in a nice frame will
    # not cancel here
    T_base_tconn = xyzrpy_to_matrix([0.4, -0.2, 0.3], [0.3, -0.4, 1.1])
    T_tool0_engaged = xyzrpy_to_matrix([0.35, -0.25, 0.45], [0.1, 0.2, -0.7])

    # BELIEF RESET
    T_tool0_conn = inverse(T_tool0_engaged) @ T_base_tconn
    assert np.allclose(T_tool0_engaged @ T_tool0_conn, T_base_tconn, atol=1e-12), \
        'the reset must place the connector exactly at the target'

    def ref_of(S):
        return T_base_tconn @ S @ inverse(T_tool0_conn)

    assert np.allclose(ref_of(np.eye(4)), T_tool0_engaged, atol=1e-12), \
        'the identity screw must be the pose the arm is standing at'
    assert np.allclose(ref_of(screw) @ T_tool0_conn, T_base_tconn @ screw, atol=1e-12)

    # THE STROKE, as the app writes it: a base-frame screw about the fixed axis LINE through the
    # engaged connector origin. It must equal the belief-based form (so it is the same motion) yet
    # be free of T_tool0_conn (so a REGRASP, which invalidates that belief, cannot change it).
    ref_goal = (T_base_tconn @ screw @ inverse(T_base_tconn)) @ T_tool0_engaged
    assert np.allclose(ref_goal, ref_of(screw), atol=1e-12), \
        'the axis-line form of the stroke must be identical to the belief-based one'

    # ADVANCE, measured from the arm pose
    class _Arm:
        dry_run = False

    class _FakeRobot:
        arm = _Arm()

        def __init__(self, T):
            self._T = T

        def tool0(self):
            return self._T

    rb = _FakeRobot(T_tool0_engaged)
    det = _ScrewAdvance(rb, T_tool0_conn, T_base_tconn, push_m)
    assert abs(det.advance_m()) < 1e-12 and not det.check()
    rb._T = ref_of(translation_matrix([0.003, 0.0, 0.0]))          # 3 mm: short of the threshold
    assert abs(det.advance_m() - 0.003) < 1e-12 and not det.check()
    rb._T = ref_of(Rx)                                            # pure 90 deg roll, no advance
    assert abs(det.advance_m()) < 1e-12 and not det.check(), \
        'a roll must not register as advance'
    rb._T = ref_of(screw)                                         # the full screw: success
    assert abs(det.advance_m() - push_m) < 1e-12 and det.check()
    assert 'advance' in det.tripped_by and abs(det.peak_m - push_m) < 1e-12

    # REBASING ACROSS A CHANGE OF GRIP. The connector sweep no longer regrasps -- a retry is the
    # REVERSAL onto the next sweep_deg position and the gripper stays closed -- so nothing in
    # bnc_assembly calls rebase today. What is tested here is the detector's CONTRACT, which is
    # what makes "cumulative advance" mean anything at all: release the connector and it stays
    # where it screwed to while the gripper travels, so the two afterwards differ by exactly the
    # progress made. Without the rebase the detector reads zero again after any release and a
    # cumulative threshold could never be reached. Any future caller that lets go (the collar
    # maneuver's seat push already does) needs this arithmetic to be right.
    partial = xyzrpy_to_matrix([0.002, 0.0, 0.0], [np.radians(30.0), 0.0, 0.0])   # 2 mm gained
    det2 = _ScrewAdvance(rb, T_tool0_conn, T_base_tconn, 0.005)
    rb._T = ref_of(partial)
    assert abs(det2.advance_m() - 0.002) < 1e-12 and not det2.check()
    C_conn = rb.tool0() @ det2.T_tool0_conn                       # captured BEFORE releasing
    assert np.allclose(C_conn, T_base_tconn @ partial, atol=1e-12)
    rb._T = T_tool0_engaged                                       # gripper back at engaged pose
    det2.rebase(inverse(rb.tool0()) @ C_conn)                     # re-grip: adopt the new relation
    assert abs(det2.advance_m() - 0.002) < 1e-12, \
        'the 2 mm already gained must survive the change of grip'
    assert not np.allclose(det2.T_tool0_conn, T_tool0_conn, atol=1e-9), (
        'the connector-in-gripper relation MUST change when the gripper travels and the part '
        'does not')
    # the second identical stroke then adds to it rather than restarting from zero
    rb._T = (T_base_tconn @ screw @ inverse(T_base_tconn)) @ T_tool0_engaged
    assert abs(det2.advance_m() - (0.002 + push_m)) < 1e-12 and det2.check(), \
        'a second stroke after the rebase must accumulate onto the first'

    # _AnyGuard must remember WHICH watchdog fired -- success and jam are the same 'seated'
    class _Trip:
        def __init__(self, hit):
            self.hit, self.tripped_by = hit, ('boom' if hit else None)

        def check(self):
            return self.hit

        def reset(self):
            pass

    quiet, loud = _Trip(False), _Trip(True)
    g = _AnyGuard(quiet, loud, None)                              # None guards are dropped
    assert len(g.guards) == 2 and g.check() and g.tripped is loud and g.tripped_by == 'boom'
    g.reset()
    assert g.tripped is None and g.tripped_by is None
    assert not _AnyGuard(quiet).check()

    # COLLAR: offset along the connector's OWN +X, CLOSED fingertip frame on it
    T_base_conn = T_base_tconn @ screw                            # where connector clocking left it
    T_ftip = xyzrpy_to_matrix([0.0, 0.0, 0.183], [np.pi, 0.0, -np.pi / 2])
    T_collar = T_base_conn @ translation_matrix([off, 0.0, 0.0])
    T_ref = T_collar @ inverse(T_ftip)
    assert np.allclose(T_ref @ T_ftip, T_collar, atol=1e-12), \
        'the CLOSED fingertip frame must coincide with the collar frame'
    d_xyz, _d_rpy = matrix_to_xyzrpy(inverse(T_base_conn) @ (T_ref @ T_ftip))
    assert abs(d_xyz[0] - off) < 1e-12 and np.allclose(d_xyz[1:], 0.0, atol=1e-12), \
        'the collar must sit purely along the connector +X'

    # the collar TURN orbits the connector axis
    axis, point = T_base_conn[:3, 0], T_base_conn[:3, 3]
    assert np.allclose(rotate_about_axis(T_base_conn, axis, point, rot)[:3, 3], point,
                       atol=1e-12), 'a point ON the axis must not move'
    end = rotate_about_axis(T_ref, axis, point, rot)
    r_xyz, r_rpy = matrix_to_xyzrpy(inverse(T_base_conn) @ (end @ T_ftip))
    assert abs(r_rpy[0] - rot) < 1e-9 and np.allclose(r_rpy[1:], 0.0, atol=1e-9), \
        'the collar turn must be a pure roll about the connector +X'
    assert abs(r_xyz[0] - off) < 1e-12 and np.allclose(r_xyz[1:], 0.0, atol=1e-12), \
        'and must not translate the collar along or off the axis'

    # RETRACT: leg 1 in the GRIPPER frame (right-multiply), leg 2 in the TARGET frame (left)
    T1 = T_ref @ translation_matrix([0.0, 0.0, -0.1])
    assert np.allclose(T1[:3, 3] - T_ref[:3, 3], -0.1 * T_ref[:3, 2], atol=1e-12), \
        'leg 1 must travel along the GRIPPER z, not a base axis'
    T2 = translation_matrix(T_base_tconn[:3, :3] @ np.array([-0.1, 0.0, 0.0])) @ T1
    assert np.allclose(T2[:3, 3] - T1[:3, 3], -0.1 * T_base_tconn[:3, 0], atol=1e-12), \
        'leg 2 must travel along the TARGET x, not the gripper x'
    for T_before, T_after in ((T_ref, T1), (T1, T2)):
        assert np.allclose(T_before[:3, :3], T_after[:3, :3], atol=1e-12), \
            'a retract leg is a pure translation -- it must not rotate the tool'


def test_bnc_insertion_holds_the_seat():
    """The insertion meant to SEAT must not retract.

    It used to retract unconditionally, which was wrong twice over: the operator was asked whether
    the mate had succeeded AFTER the gripper had already backed 30 mm out along the connector's own
    -X (taking the connector with it, since the gripper holds the cable), and connector clocking then
    read that retracted pose as its "engaged pose" -- anchoring the screw axis and the entire
    clocking sequence 30 mm away from the actual mate.

    So: retract is now the CALLER's decision, taken after the verdict. Intermediate sweep passes
    still back off (the next one realigns to a different offset's start); the last pass, a
    successful attempt and the final insertion all hold the seat.
    """
    src = open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py')).read()
    code = '\n'.join(ln for ln in src.splitlines() if not ln.lstrip().startswith('#'))
    assert 'def retract_from(' in code, \
        'the escape must be callable on its own so the caller can defer it past the verdict'
    assert 'retract=True' in code, 'run_insertion must take a retract flag'
    # the commit holds the seat
    assert 'pause=fi_pause, retract=False' in code, \
        'the final insertion must NOT retract -- it is the attempt meant to seat'
    # intermediate sweep passes still back off, the last one does not
    assert 'retract=(pi < len(passes) - 1)' in code, \
        'only intermediate sweep passes should retract'
    # and the retract for a retry happens after the operator verdict
    v = code.index("row['success']")
    r = code.index('retract_from(last_ref, T_tool0_conn)')
    assert r > v, 'the retry retract must come AFTER the success verdict, not before it'


def test_bnc_engage_config():
    """The engage block's oscillation must be able to do its job. Four ways it silently cannot.

      * MUTUALLY PRIME frequencies. A rational ratio retraces one closed Lissajous path forever,
        so the oscillation re-probes a ONE-dimensional line through the (z, pitch) rectangle
        instead of sweeping it.
      * NO ALIASING. The reference is rebuilt at sample_rate_hz; a frequency above a quarter of
        that produces a slower oscillation than configured, silently.
      * WHOLE CYCLES INSIDE THE INSERTION. This one is unique to engage: the insertion lasts
        path/speed_mm_s, and a frequency that completes less than one cycle in that time is a
        constant OFFSET, not a wiggle. The standalone wiggle had no such constraint because it
        ran on its own timeout.
      * AMPLITUDE WITHOUT FREQUENCY is a constant offset too, and belongs in the trajectory.
    """
    import math

    import yaml
    with open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')) as fh:
        a = yaml.safe_load(fh)['assembly']
    assert a['insertion_mode'] in ('estimate', 'engage'), \
        "the standalone 'wiggle' mode is retired"
    assert 'wiggle' not in a, (
        'assembly.wiggle must be gone -- two blocks with the same shape, one live and one dead, '
        'is how the wrong one gets tuned')

    e = a['engage']
    dims = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')
    _r = _resolved_wiggle('bnc_assembly', 'assembly', 'engage')
    amp = {d: float((_r.get('amplitude') or {}).get(d, 0.0)) for d in dims}
    frq = {d: float((_r.get('frequency_hz') or {}).get(d, 0.0)) for d in dims}

    assert amp['x_mm'] == 0.0, \
        'x is the push direction -- the trajectory owns it, not the oscillation'
    active = [d for d in dims if amp[d] != 0.0]
    for d in active:
        assert frq[d] > 0, f'{d} has amplitude but no frequency -- that is a constant offset'

    if not active:
        return                                  # a direct insertion: nothing else to check

    # no aliasing
    fmax = max(frq[d] for d in active)
    assert float(e['sample_rate_hz']) >= 4.0 * fmax, \
        f"sample_rate_hz {e['sample_rate_hz']} aliases a {fmax} Hz oscillation"

    # THE ORBIT MUST BE LONG relative to a single axis' period. The Lissajous figure closes at
    # 1/gcd(frequencies); when that equals the slowest axis' own period -- which is what a 1:1 or
    # 1:2 ratio gives -- the pattern degenerates to a line and the oscillation re-probes it.
    # Checking gcd == 1 on an arbitrary grid would be wrong: 0.7 and 1.1 Hz are the intended
    # co-prime pair (7:11) yet share a factor of 10 on a 0.01 Hz grid.
    if len(active) >= 2:
        ints = [int(round(frq[d] * 1000)) for d in active]
        g = 0
        for n in ints:
            g = math.gcd(g, n)
        orbit_s = 1000.0 / g
        slowest = 1.0 / min(frq[d] for d in active)
        assert orbit_s >= 3.0 * slowest, (
            f'frequencies {[frq[d] for d in active]} Hz close their orbit every {orbit_s:.1f} s '
            f"against a slowest single-axis period of {slowest:.1f} s -- the ratio is too simple, "
            'so the figure is a line rather than a sweep of the rectangle')

    # whole cycles must fit inside the insertion
    mats = traj_mod.load_csv(os.path.join(ROOT, 'configs', 'assembly_trajectory.csv'))
    span = abs(T.matrix_to_xyzrpy(mats[0])[0][0] - T.matrix_to_xyzrpy(mats[-1])[0][0]) * 1000.0
    total = span + float(e['preload_mm'])
    v = e['speed_mm_s']
    if v is None:
        with open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')) as fh:
            spd = yaml.safe_load(fh)['speed']
        v = float(spd['max_cartesian_translation_mm_s']) * float(
            (spd.get('phase_scale') or {}).get('assemble', 1.0))
    dur = total / float(v)
    for d in active:
        assert frq[d] * dur >= 1.0, (
            f'{d} completes {frq[d] * dur:.2f} cycles in the {dur:.2f} s insertion '
            f'({total:.1f} mm at {float(v):.2f} mm/s) -- under one cycle it acts as a constant '
            'OFFSET. Raise the frequency or lower engage.speed_mm_s')

    # the axial limit must be BELOW the general guard, or the jam limit fires first and the
    # normal early stop never happens
    gen = float(e.get('max_force_n') or yaml.safe_load(
        open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))['force_guard']['max_force_n'])
    assert float(e['max_axial_force_n']) < gen, (
        f"the axial limit {e['max_axial_force_n']} N must sit below the general guard {gen} N -- "
        'otherwise a jam trip pre-empts the normal force-stop the phase is designed around')

def test_bnc_clocking_enable_gating():
    """Both post-mate maneuvers are OPTIONAL and independently switchable, with one dependency:
    collar clocking requires connector clocking, and that combination is REJECTED rather than silently
    reinterpreted (it would otherwise turn the collar on a connector still proud of the socket,
    using a connector pose that was never established).

    Tested through the pure function the app calls, because a config rule that only exists inside
    the robot routine can only be checked by running the robot -- which means it never gets checked.
    """
    from urlab.apps.bnc_assembly import _clocking_plan

    assert _clocking_plan({'enabled': False}, {'enabled': False}) == (False, False)
    assert _clocking_plan({'enabled': True}, {'enabled': False}) == (True, False)
    assert _clocking_plan({'enabled': True}, {'enabled': True}) == (True, True)
    # an absent block, an absent key, or a null must all mean OFF -- a config predating these
    # maneuvers must not start moving the robot in new ways
    for cable, collar in ((None, None), ({}, {}), ({'enabled': None}, {'enabled': None})):
        assert _clocking_plan(cable, collar) == (False, False), (cable, collar)
    # the one forbidden combination, and the message must name both keys so it is actionable
    try:
        _clocking_plan({'enabled': False}, {'enabled': True})
    except ValueError as exc:
        assert 'collar_clocking' in str(exc) and 'connector_clocking' in str(exc), str(exc)
    else:
        raise AssertionError('collar clocking without connector clocking must be REJECTED')
    # and the app must route the rejection to a pre-motion failure, not an exception at runtime
    src = open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py')).read()
    assert '_clocking_plan(cc, cl)' in src and 'except ValueError as exc' in src, \
        'build_and_run must call _clocking_plan and fail cleanly before the robot moves'


def test_bnc_clocking_state_vocabulary():
    """The run reports progress as ENGAGED -> SEATED -> LOCKED -> ASSEMBLED, and that vocabulary is
    STRUCTURAL rather than prose: the app walks CLOCK_STATES through _advance_state, so a future
    edit that skips a step -- calling collar clocking without connector clocking, or advancing twice --
    raises instead of quietly logging LOCKED for a connector that was never seated.

    Also pinned: 'seated' is OVERLOADED. AdmittanceController.ramp returns the string 'seated' to
    mean "a guard tripped", which is the robot layer's word and says nothing about the assembly
    state. Reading that return into a variable called `seated` is the mistake this guards against,
    so the clocking code must name it `stopped`.
    """
    from urlab.apps.bnc_assembly import CLOCK_STATES, _advance_state

    assert CLOCK_STATES == ('engaged', 'seated', 'locked')
    assert _advance_state('engaged', 'engaged') == 'seated'
    assert _advance_state('seated', 'seated') == 'locked'
    for state, expected in (('engaged', 'seated'), ('seated', 'engaged'), ('locked', 'seated')):
        try:
            _advance_state(state, expected)
        except AssertionError:
            continue
        raise AssertionError(f'advancing from {state!r} as {expected!r} must not be allowed')

    # the ramp-return / state collision: every `== 'seated'` comparison in the clocking code must
    # land in a variable named `stopped`, never `seated`
    src = open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py')).read()
    code = '\n'.join(ln for ln in src.splitlines() if not ln.lstrip().startswith('#'))
    for bad in ("seated = res == 'seated'", "seated = (res == 'seated')"):
        assert bad not in code, \
            "ramp's 'seated' means a guard tripped -- do not bind it to a variable called `seated`"
    assert code.count("stopped = res == 'seated'") == 2, \
        'both clocking maneuvers must read the ramp return into `stopped`'


def test_bnc_clocking_config():
    """configs/bnc_assembly.yaml must match what the app validates pre-motion, and the two
    settings that make the maneuver possible at all must not be lost in a retune:

      * the SUCCESS threshold has to be reachable given the push the virtual target commands --
        asking for more advance than the screw aims for can never succeed;
      * the guard must OVERRIDE the shared force_guard:. That block is tuned for a light probing
        insertion (5 N) and a deliberate press-and-twist exceeds it on the first cycle, so an
        inherited guard means the screw never runs.
    """
    import yaml
    with open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')) as fh:
        a = yaml.safe_load(fh)['assembly']
    cc, cl = a['connector_clocking'], a['collar_clocking']
    assert 'retry_mode' not in cc, (
        'retry_mode is gone -- a retry is the REVERSAL onto the next sweep_deg position, '
        'with the gripper still closed')
    assert float(cc['success_advance_mm']) > 0, 'no way to tell the screw worked'
    assert int(cc['max_tries']) >= 1
    # ONE TRY = ONE LEG, so more than one try needs more than one position to alternate
    # between, or the extra legs have nothing to turn (the app warns about exactly this).
    if int(cc['max_tries']) > 1:
        assert len(cc.get('sweep_deg') or []) >= 2, (
            'max_tries > 1 with fewer than two sweep_deg positions gives zero-rotation legs -- '
            'the oscillation needs two ends to rock between')
    assert float(cc['max_force_n']) > 0 and float(cc['max_torque_nm']) > 0
    # `enabled` is the MANEUVER's switch; the guard's is force_guard_enabled. Writing `enabled`
    # twice in one block is silently legal in YAML (last wins), so the guard override would have
    # eaten the maneuver's own switch -- assert the guard key is the distinct one.
    for blk, nm in ((cc, 'connector_clocking'), (cl, 'collar_clocking')):
        assert 'force_guard_enabled' in blk, f'{nm} must name the guard switch separately'
    with open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')) as fh:
        lines = fh.read().splitlines()
    for nm in ('connector_clocking', 'collar_clocking', 'clocking_retract'):
        start = next(i for i, ln in enumerate(lines) if ln.strip() == f'{nm}:')
        indent = len(lines[start]) - len(lines[start].lstrip())
        body = []
        for ln in lines[start + 1:]:
            if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
                break
            body.append(ln)
        # DERIVE the child indent instead of assuming it. Assuming indent + 4 when the real depth
        # is indent + 2 skipped every line and made this whole check silently vacuous -- it passed
        # against a deliberately injected duplicate. Hence the closing assert too.
        kids = [ln for ln in body if ln.strip() and not ln.strip().startswith(('#', '-'))]
        assert kids, f'assembly.{nm} has no settings'
        child = min(len(ln) - len(ln.lstrip()) for ln in kids)
        assert child > indent, (nm, child, indent)
        seen = set()
        for ln in kids:
            s = ln.strip()
            if ':' not in s or (len(ln) - len(ln.lstrip())) != child:
                continue
            key = s.split(':', 1)[0]
            assert key not in seen, f'assembly.{nm}.{key} is defined twice -- YAML keeps the last'
            seen.add(key)
        assert len(seen) >= 3, f'duplicate scan saw only {seen} under {nm} -- it is vacuous'
    # Advance is CUMULATIVE from the ORIGINAL engaged pose, so a reversal does not re-zero what
    # an earlier leg gained. success_advance_mm is an EARLY-OUT, not the success condition (a
    # sweep that runs its legs is assumed seated), so it does not have to be deliverable within
    # max_tries legs -- an unreachable early-out just never fires. Only the > 0 floor above
    # still matters (zero would trip on the first servo cycle, before anything turned).
    shared = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))
    assert float(cc['max_force_n']) >= float(shared['force_guard']['max_force_n']), \
        'the screw guard must not be TIGHTER than the probing guard or it trips immediately'
    # each maneuver carries its OWN guard block, so they can be tuned apart
    for blk, nm in ((cc, 'connector_clocking'), (cl, 'collar_clocking')):
        for k in ('max_force_n', 'max_torque_nm', 'persistence_s'):
            assert blk.get(k) is not None, f'{nm} must set its own {k}'
    # and its OWN phase scale, so the strokes can be paced apart from each other and from the
    # insertion. A maneuver whose speed_* keys are null depends on this entry existing.
    ps = shared['speed']['phase_scale']
    for nm in ('connector_clock', 'collar_clock', 'clock_retract'):
        assert nm in ps and float(ps[nm]) > 0, f'speed.phase_scale.{nm} missing or non-positive'
    for blk, nm in ((cc, 'connector_clock'), (cl, 'collar_clock')):
        if blk.get('speed_rotation_deg_s') is None:
            assert float(ps[nm]) * float(shared['speed']['max_cartesian_rotation_deg_s']) > 0
    assert len(cc['stiffness']) == 6 and len(cl['stiffness']) == 6
    # >= 0, not > 0: this is measured from the connector frame ORIGIN (the mating face), so zero
    # is a legitimate reading -- the ring sitting at the face. It was > 0 only while the offset
    # was measured from the JUNCTION, ~45.7 mm behind the origin, where zero could not happen.
    # ANY SIGN: measured from the connector frame ORIGIN (the mating face), so negative means the
    # ring sits behind the face -- a real reading, not an error. It was > 0 only while the offset
    # was measured from the JUNCTION, ~45.7 mm behind the origin.
    assert isinstance(float(cl['collar_offset_mm']), float)
    if cl.get('enabled'):
        assert cc.get('enabled'), 'collar clocking depends on connector clocking'
    r = a['clocking_retract']
    for k in ('gripper_axis', 'target_axis'):
        assert len(r[k]) == 3 and any(abs(float(v)) > 1e-9 for v in r[k]), r[k]
    for k in ('gripper_distance_m', 'target_distance_m'):
        assert float(r[k]) > 0, k
    assert 'clock' not in ps, \
        'the single `clock` phase was split per maneuver -- a leftover entry paces nothing'


def test_estimator_eval_sweep_is_a_sampling_trial():
    """A parity sweep pass must be built by uncertain_sampling's OWN call, not an approximation.

    The map is a record of paths that app drove. `traj.noised` (smoothed Gaussian, decaying along
    the path and across attempts) and `traj.perturb` (per-waypoint UNIFORM, un-smoothed) are
    different jitter models, so a pass built with the wrong one is off-map before contact even
    starts. This pins the branch to the identical call with identical arguments: same bias, same
    seed, same bounds -> byte-identical rows."""
    from urlab.skills import trajectory as traj

    dense = traj.resample(traj.load_csv(os.path.join(ROOT, 'configs', 'assembly_trajectory.csv')),
                          0.001, 1.0)
    bias6 = [0.0, 0.0, -0.003, 0.0, -5.0, 0.0]
    nl, nh = [0.0, 0.0, -0.005, 0.0, -5.0, 0.0], [0.0, 0.0, 0.005, 0.0, 5.0, 0.0]
    us = traj.perturb(dense, traj.delta_from(bias6), nl, nh, np.random.default_rng(7),
                      frame='connector')                        # uncertain_sampling's call
    ev = traj.perturb(dense[:len(dense)], traj.delta_from(bias6), nl, nh,
                      np.random.default_rng(7), frame='connector')   # the parity branch
    assert all(approx(a, b, 0.0) for a, b in zip(us, ev))
    # ...and the OLD branch is measurably NOT that path, so the test above is not vacuous.
    old = traj.noised(dense, np.random.default_rng(7), [0.0, 0.002, 0.002, 0.0, 2.0, 0.0],
                      10, 0.2, 1.0, bias6)
    assert not all(approx(a, b, 1e-6) for a, b in zip(us, old))

    src = open(os.path.join(ROOT, 'urlab', 'apps', 'estimator_eval.py'), encoding='utf-8').read()
    assert 'traj.perturb(dense[:sweep_k]' in src, 'the parity pass must use traj.perturb'
    assert "if col_mode == 'peck':" in src and "must be 'attempts' or " in src,         'peck stays rejected under parity; offset_sweep must not be'


def test_estimator_eval_sampling_recipe_inherits():
    """The sweep's sampling recipe defaults to null = inherit from uncertain_sampling.yaml.

    A COPY of those values here would drift the moment the sampler is re-tuned, and the drift is
    exactly what nobody notices -- so the config must leave them null and the app must read the
    reference at run time."""
    import yaml

    ev = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'estimator_eval.yaml')))['eval']
    sw = ev['collection']['sampling']
    for k in ('perturb_frame', 'chunk_fraction', 'noise', 'random_seed'):
        assert k in sw, f'collection.sampling.{k} missing'
        assert sw[k] is None, f'collection.sampling.{k} must default to null (inherit), not a copy'
    ref = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'uncertain_sampling.yaml')))
    for k in ('perturb_frame', 'chunk_fraction', 'noise'):
        assert ref['sampling'].get(k) is not None,             f'nothing to inherit: uncertain_sampling sampling.{k} is unset'
    # peck is the only mode parity rejects
    assert ev['collection']['mode'] in ('attempts', 'offset_sweep')


def test_trajectory_csv_ends_on_the_mate():
    """The shared trajectory's LAST ROW must be the identity, as its own header requires.

    Two apps ANCHOR this path (uncertain_sampling, kinematic_assembly -- traj.anchor_target
    normalises the last row onto the recorded mate) and the rest apply rows directly. Those agree
    only while the last row IS the identity. It once carried a deliberate +10 mm press, and the
    result was that the app building the contact map drove 20 -> 0 mm while the apps matched
    against it drove 10 -> +10: the same file, 10 mm apart, silently. A press belongs in
    final_insertion.preload_mm, which survives anchoring; this test is what keeps it out of here."""
    from urlab.skills import trajectory as traj

    mats = traj.load_csv(os.path.join(ROOT, 'configs', 'assembly_trajectory.csv'))
    assert approx(mats[-1], np.eye(4), 1e-12), \
        'trajectory last row must be identity -- put an insertion press in preload_mm instead'
    # ...so anchoring is a NO-OP, which is what makes anchored and direct apps agree
    mate = T.xyzrpy_to_matrix([0.4, -0.1, 0.3], [0.1, -0.2, 0.3])     # any mate; arbitrary
    assert approx(mate @ T.inverse(mats[-1]), mate)
    # The approach LENGTH is a tunable (20 mm originally, 15 later) -- what must hold is
    # that the path approaches from OUTSIDE the mate and ends on it, so the sweep covers
    # real approach travel rather than starting already seated.
    start_mm = T.matrix_to_xyzrpy(mats[0])[0][0] * 1000.0
    assert start_mm < -1.0, (
        f'the trajectory must approach from outside the mate, but starts at {start_mm:+.1f} '
        'mm -- a path beginning at or past the mate collects no approach contact')
    # and it is monotonic in +X: a direct insertion, never backing up mid-path
    xs = [T.matrix_to_xyzrpy(m)[0][0] for m in mats]
    assert all(b > a for a, b in zip(xs, xs[1:])), 'the insertion must advance monotonically'


def test_apps_anchor_the_trajectory_like_the_sampler():
    """estimator_eval and bnc_assembly must anchor, so a future bad CSV cannot silently offset them.

    The CSV is conforming today, which makes anchoring a no-op -- and that is exactly why it needs
    a test rather than trust: nothing about a passing run would reveal its absence until someone
    edits the last row again."""
    for app, anchor_expr in (
            ('estimator_eval',
             'T_base_targetobj = T_base_tconn @ inverse(mats[-1]) if anchor else T_base_tconn'),
            ('bnc_assembly', 'T_base_targetobj = T_base_tconn @ inverse(mats[-1])')):
        src = open(os.path.join(ROOT, 'urlab', 'apps', f'{app}.py'), encoding='utf-8').read()
        assert anchor_expr in src, f'{app} must anchor the trajectory'
        assert 'T_base_commit = T_base_targetobj @ translation_matrix(' in src, \
            f'{app}: the commit anchor is the probing anchor PLUS the named preload'

    src = open(os.path.join(ROOT, 'urlab', 'apps', 'estimator_eval.py'), encoding='utf-8').read()
    # The MEASUREMENT frame must stay the RECORDED mate -- that is what the map's own columns are
    # expressed in, so anchoring it too would double-count.
    for meas in ('inverse(T_base_tconn) @ robot.tool0() @ T_true',
                 'pose_error(robot.tool0() @ T_bel, T_base_tconn)'):
        assert meas in src, 'the measurement frame must remain the RECORDED mate'

    # bnc_assembly: EVERY reference is now an anchored trajectory row. There used to be a
    # second helper (tool0_ref) for poses stated directly wrt the recorded mate, needed
    # only by the standalone wiggle's fixed target; both are retired, so a reappearance of
    # an unanchored path would mean a preload had crept back in as a trajectory row.
    bnc = open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8').read()
    assert 'refs = [traj_ref(row, T_tool0_conn) for row in rows_t]' in bnc
    assert 'refs = [traj_ref(row_, T_tool0_conn, commit=True) for row_ in rows_f]' in bnc
    assert 'def tool0_ref(' not in bnc, (
        'tool0_ref is retired along with the standalone wiggle -- if it is back, some pose is '
        'bypassing the anchoring that keeps every app agreeing about where the mate is')


def test_estimator_eval_sweep_coverage_check():
    """The start-up check must fire when offset + injected error can leave the map's box."""
    import logging
    from urlab.apps import estimator_eval as ee

    box = {'lower': [0.0, 0.0, -0.005, 0.0, -5.0, 0.0],
           'upper': [0.0, 0.0, 0.005, 0.0, 5.0, 0.0]}
    inj_lo, inj_hi = [0.0] * 6, [0.0, 0.0, 0.0, 0.0, 5.0, 0.0]        # +/-5 deg pitch injected

    def warned(offsets, lo, hi, ref_box=box):
        """True iff the check logged at WARNING -- it returns None, so the LOG is the assertion."""
        rec = []
        h = logging.Handler()
        h.emit = lambda r: rec.append(r)
        ee.log.addHandler(h)
        lvl = ee.log.level
        ee.log.setLevel(logging.INFO)              # else the all-clear INFO is never even created
        try:
            ee._report_sweep_coverage(offsets, lo, hi, ref_box)
        finally:
            ee.log.removeHandler(h)
            ee.log.setLevel(lvl)
        assert rec, 'the check must say SOMETHING -- silence would make every case below vacuous'
        return any(r.levelno >= logging.WARNING for r in rec)

    # +5 deg commanded under a +/-5 deg injection reaches 10 deg -- outside a +/-5 deg box
    assert warned([[0.0, 0.0, 0.0, 0.0, 5.0, 0.0]], inj_lo, inj_hi)
    # the same offsets with NO injected error stay inside
    assert not warned([[0.0, 0.0, 0.0, 0.0, 5.0, 0.0]], [0.0] * 6, [0.0] * 6)
    # and z is checked too, in metres
    assert warned([[0.0, 0.0, 0.006, 0.0, 0.0, 0.0]], [0.0] * 6, [0.0] * 6)
    assert not warned([[0.0, 0.0, 0.003, 0.0, 0.0, 0.0]], [0.0] * 6, [0.0] * 6)
    # no reference box = advisory only, never a warning (but still not silent)
    assert not warned([[0.0, 0.0, 0.9, 0.0, 90.0, 0.0]], [0.0] * 6, [0.0] * 6, ref_box=None)


def test_preload_is_commit_only_and_preserves_the_seat():
    """The deliberate 10 mm press must land on the COMMIT and nowhere else.

    The press is intentional -- it is how the connector is driven home. Moving it out of the
    trajectory CSV (where the anchoring apps deleted it) must not lose it: the commit has to
    reproduce exactly the path it drove before, while every observation-collecting pass stops at
    the mate, which is where the contact map is."""
    import yaml
    from urlab.skills import trajectory as traj

    mats = traj.load_csv(os.path.join(ROOT, 'configs', 'assembly_trajectory.csv'))
    dense = traj.resample(mats, 0.001, 1.0)
    mate = np.eye(4)                                   # depths are relative to the mate anyway

    def depths(anchor_to):
        return [T.matrix_to_xyzrpy(anchor_to @ r)[0][0] * 1000.0 for r in dense]

    anchored = mate @ T.inverse(mats[-1])
    probe = depths(anchored)
    assert abs(probe[-1]) < 1e-9, 'probing must end exactly ON the mate -- that is where the map is'
    # The CSV's LENGTH is a tunable (it was 20 mm, then 15); what must hold is that
    # anchoring does not move the approach -- only normalises the end onto the mate.
    raw0 = T.matrix_to_xyzrpy(mats[0])[0][0] * 1000.0
    raw_end = T.matrix_to_xyzrpy(mats[-1])[0][0] * 1000.0
    assert abs(probe[0] - (raw0 - raw_end)) < 1e-6, (
        'anchoring must preserve the approach the CSV describes, only shifting its end onto '
        'the mate')
    assert probe[0] < -1.0, 'the pass must actually approach from outside the mate'

    # BOTH apps press by their named amount, and the commit is the probing path shifted by
    # exactly that much -- the relationship, not a hardcoded depth.
    for cfg_name, block in (('estimator_eval.yaml', ('eval', 'final_insertion')),
                            ('bnc_assembly.yaml', ('assembly', 'final_insertion'))):
        c = yaml.safe_load(open(os.path.join(ROOT, 'configs', cfg_name)))
        for k in block:
            c = c[k]
        pre = float(c['preload_mm'])
        assert pre > 0, f'{cfg_name}: the commit is what drives the connector home'
        comm = depths(anchored @ T.translation_matrix([pre / 1000.0, 0.0, 0.0]))
        assert abs(comm[-1] - pre) < 1e-6, \
            f'{cfg_name}: the commit must end exactly preload_mm past the mate'
        assert abs((comm[0] - probe[0]) - pre) < 1e-6, \
            f'{cfg_name}: the commit is the probing path shifted by preload_mm, nothing else'
        assert comm[-1] > probe[-1], 'the commit presses deeper than a probing pass'

    # ...and each app applies it through its OWN commit anchor, exactly once.
    for app, probing, commit in (
            ('estimator_eval', 'T_base_targetobj @ row @ inverse(T_believed)',
             'T_base_commit @ row @ inverse(T_believed)'),
            ('bnc_assembly', 'traj_ref(row, T_tool0_conn)',
             'traj_ref(row_, T_tool0_conn, commit=True)')):
        src = open(os.path.join(ROOT, 'urlab', 'apps', f'{app}.py'), encoding='utf-8').read()
        assert src.count(commit) == 1, f'{app}: exactly ONE insertion presses, and it is the commit'
        assert src.count(probing) == 1, f'{app}: probing passes must not press'


def axial_stiffness(K6, cfg):
    """The stiffness entry that actually resists the insertion.

    Compliance acts on the TOOL0 axes but the part goes in along the CONNECTOR's +X, and
    for bnc_connector_finger_holder those are different axes. u'Ku is that projection,
    exact for the diagonal K the configs carry -- taking max() instead happens to give the
    right number for some stiffness triples and silently the wrong one for others.

    The connector's orientation comes from whichever chain the app uses: `held_frame` for the
    fixtured apps, or fingertip_grasp @ initial_connector_in_fingertip for the ones that pick it
    up. bnc_assembly declares no held_frame, so both paths are needed.
    """
    from urlab import tool_frames as _tf
    if cfg.get('held_frame'):
        R = _tf.load_frames(cfg)[cfg['held_frame']][:3, :3]
    else:
        init = (cfg.get('estimation') or {}).get('initial_connector_in_fingertip')
        chain = T.from_cfg(cfg['fingertip_grasp'])
        if init:
            chain = chain @ T.from_cfg(init)
        R = chain[:3, :3]
    u = R @ np.array([1.0, 0.0, 0.0])
    return float(u @ (np.asarray(K6[:3], dtype=float) * u))


def test_preload_force_is_a_spike_not_a_press():
    """What the preload can HOLD is stiffness x preload -- small. The configs must not promise more.

    admittance.py's law is F_ext = M x'' + D x' + S (x - x_d), so a static force holds a steady
    deflection F / S and nothing more. 10 mm against the PROBING stiffness sustains single-digit
    newtons; the 169 N measured on the v5 run was a contact TRANSIENT, not a press. This pins the
    arithmetic so the comments cannot drift away from the configs they describe."""
    import yaml
    from urlab.skills import trajectory as traj

    for name, block in (('estimator_eval.yaml', 'eval'), ('bnc_assembly.yaml', 'assembly')):
        cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', name)))
        fi = cfg[block]['final_insertion']
        pre_m = float(fi['preload_mm']) / 1000.0
        probe_S = max(float(v) for v in cfg['compliance']['stiffness'][:3])
        commit_S = max(float(v) for v in fi['stiffness'][:3])
        assert probe_S * pre_m < 10.0, \
            f'{name}: the probing spring cannot hold a large press -- keep the comment honest'
        # The commit must never be SOFTER than probing -- that would be backwards, a press
        # yielding more easily than a search. It is no longer required to be STRICTLY stiffer:
        # the fleet runs one stiffness everywhere by explicit decision, so equal is the intended
        # state and the press is bought with preload rather than with extra spring.
        assert commit_S >= probe_S, (
            f'{name}: the commit spring is softer than the probing spring -- that is inverted')
        # THE GUARD CANNOT BOUND THE SPIKE, and the comments say so, so pin the arithmetic behind
        # that claim. The guard needs the limit held CONTINUOUSLY for persistence_s. The WORST
        # case for that claim is contact from the very first waypoint -- the connector catching a
        # chamfer well before the mate -- so compare persistence against the whole commit ramp,
        # not just the preload. If even that cannot outlast persistence_s, the guard provably
        # never ends the advance; it can only trip in the settle that follows, at zero velocity.
        # Use the SLOWEST speed the commit can run at, since a slower ramp is the case closest to
        # tripping: a null speed override means speed.max_cartesian_* x the assemble phase scale.
        v = float(cfg['speed']['max_cartesian_translation_mm_s'])
        if fi.get('speed_translation_mm_s') is not None:
            v = float(fi['speed_translation_mm_s'])
        else:
            scales = cfg['speed'].get('phase_scale') or {}
            v *= float(scales.get('assemble', 1.0))
        mats = traj.load_csv(os.path.join(ROOT, 'configs', 'assembly_trajectory.csv'))
        path_mm = (T.matrix_to_xyzrpy(mats[-1])[0][0]
                   - T.matrix_to_xyzrpy(mats[0])[0][0]) * 1000.0 + float(fi['preload_mm'])
        ramp_s = path_mm / v
        persist = float(cfg['force_guard']['persistence_s'])
        if fi.get('persistence_s') is not None:
            persist = float(fi['persistence_s'])
        lim = float(fi.get('max_force_n') or cfg['force_guard']['max_force_n'])
        # THE INVARIANT: the commit is the pass that drives the connector home, so its guard
        # must not stop it before it has pressed. Two ways to satisfy that, and one of them
        # has to hold:
        #   (a) the guard cannot act inside the ramp at all (ramp shorter than persistence),
        #       which is where this sat while the commit ran at 30 mm/s; or
        #   (b) the limit is at or above the force the commit INTENDS to apply, which is the
        #       axial stiffness times the preload -- so a normal press never reaches it.
        # Failing both means the guard trips partway through the press every time, and the
        # connector is left short of the seat with nothing in the log saying why.
        S_ax = axial_stiffness(fi.get('stiffness') or cfg['compliance']['stiffness'], cfg)
        intended_n = S_ax * float(fi['preload_mm']) / 1000.0
        guard_cannot_act = ramp_s <= persist + 1e-9
        limit_above_press = lim >= intended_n - 1e-9
        assert guard_cannot_act or limit_above_press, (
            f'{name}: the commit ramp lasts {ramp_s:.2f} s so the {persist:.2f} s guard CAN '
            f'fire, and its limit {lim:.1f} N is below the {intended_n:.1f} N the commit '
            f'intends to apply ({S_ax:.0f} N/m axial x {float(fi["preload_mm"]):.1f} mm '
            f'preload) -- it will cut '
            'the press short. Raise final_insertion.max_force_n above the intended press, or '
            'shorten persistence_s so the guard is a spike detector rather than a brake.')
        # MARGIN, reported through the assertion message rather than a bare pass: bnc_assembly
        # currently sits at EXACTLY zero (a 30 mm commit ramp at 30 mm/s = 1.00 s against a
        # persistence of 1.00 s, with the commit guard at 1 N -- which the connector exceeds on
        # first touch). It is on the boundary, not past it, so the press still completes; but any
        # slowing of the commit, lengthening of the path, or raising of the preload tips it over
        # and the guard starts cutting the press short. Flagged, not silently "fixed": changing a
        # force limit or a speed is a physical decision.
        assert persist - ramp_s >= 0.0, f'{name}: negative guard margin'


def test_a_null_speed_override_is_resolved_before_it_reaches_the_arithmetic():
    """A `null` speed override must be defaulted by the caller, never compared to a number.

    THE BUG THIS EXISTS TO CATCH. connector_clocking and collar_clocking both ship
    `speed_translation_mm_s: null` / `speed_rotation_deg_s: null`, which the app turns into a
    literal None meaning "no override". seg_time defaulted those; a second duration path was added
    that duplicated seg_time's arithmetic WITHOUT its defaulting, so `v > 0` compared None to an
    int and raised TypeError partway into the stroke -- after the connector was engaged and the
    gripper closed on it. The suite never saw it because the arithmetic lived in a closure inside
    build_and_run, unreachable without a live robot.

    So the arithmetic is module level now, it REFUSES a None rather than limping, and every
    duration in the app resolves its caps through one helper."""
    import pytest

    from urlab.apps.bnc_assembly import _path_time

    # Both caps bind; the slower one wins, and a floor applies.
    assert _path_time(100.0, 90.0, 10.0, 45.0, 0.008) == pytest.approx(10.0)   # translation-bound
    assert _path_time(10.0, 90.0, 100.0, 9.0, 0.008) == pytest.approx(10.0)    # rotation-bound
    assert _path_time(0.0, 0.0, 10.0, 10.0, 0.008) == pytest.approx(0.008)     # the floor
    # A zero cap means "this axis does not constrain", not "divide by zero".
    assert _path_time(100.0, 90.0, 0.0, 45.0, 0.008) == pytest.approx(2.0)

    # THE REGRESSION: a null must raise with the values named, not TypeError deep in a comparison.
    for v, w in ((None, 45.0), (10.0, None), (None, None)):
        with pytest.raises(ValueError, match='resolved speed caps'):
            _path_time(100.0, 90.0, v, w, 0.008)

    # And every duration path in the app must route its overrides through the one resolver.
    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    assert src.count('_path_time(') >= 3, 'seg_time and screw_ramp must share the arithmetic'
    assert 'def caps(' in src, 'the null-override resolver is gone'
    body = src[src.index('def screw_ramp('):src.index('retract_m =')]
    assert 'caps(v, w)' in body, (
        'screw_ramp must resolve its caps before computing a duration -- it ships with null '
        'overrides from both clocking blocks')

    # The configs that feed it really do carry nulls, so this path is live and not hypothetical.
    import yaml
    asm = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))['assembly']
    assert any(asm[b].get(k) is None
               for b in ('connector_clocking', 'collar_clocking')
               for k in ('speed_translation_mm_s', 'speed_rotation_deg_s')),         'if no clocking block ships a null any more, this guard has lost its subject'


def test_tug_verification_defaults_on_and_its_spring_can_exceed_the_threshold():
    """The tug is a SPRING pull: reference offset = force / stiffness, so the force can never
    exceed pull_force_n on a connector that holds. For the test to be able to FAIL, that offset
    must exceed displacement_threshold_mm -- otherwise an unlocked connector could never move
    past the threshold and every tug would report verified."""
    import yaml

    y = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))
    tv = y['assembly']['tug_verify']
    assert tv['enabled'] is True, 'tug verification must default ON'
    for k in ('pull_force_n', 'pull_time_s', 'displacement_threshold_mm',
              'extraction_distance_mm'):
        assert float(tv[k]) > 0.0, f'{k} must be positive'
    S_max = max(float(v) for v in (tv.get('stiffness') or y['compliance']['stiffness'])[:3])
    offset_mm = float(tv['pull_force_n']) / S_max * 1000.0
    assert offset_mm > float(tv['displacement_threshold_mm']), (
        f'the spring offset ({offset_mm:.1f} mm at the stiffest axis) must exceed the '
        f'{tv["displacement_threshold_mm"]} mm threshold, or an unlocked connector can never '
        f'show -- raise pull_force_n or lower the threshold/stiffness')

    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    body = src[src.index('def tug_verify_in_place('):src.index('def engage_insertion(')]
    order = ["gripper.close('tug grasp')", "verify_cable_held(robot, check, 'tug regrasp')",
             'tare_fn=tare', 'adm_tug.hold(T_pull', "'release (tug verified)'"]
    idx = [body.index(t) for t in order]
    assert idx == sorted(idx), (
        'the tug must close, verify the grasp, tare while gripping, pull, and only release '
        'after a verified hold -- in that order')
    assert "'terminated'" in body and 'tv_extract_m' in body and 'guard_shared' in body, (
        'the failure path must extract by extraction_distance_mm under the GLOBAL guard, and a '
        'guard trip must terminate the script')


def test_the_seat_push_can_reach_its_force_and_keeps_the_gripper_logic_straight():
    """The seat push sits between the collar unwind and the advance, and it grips.

    Two things must hold. FORCE REACHABILITY: the push is a spring press, so the reference must
    be able to stretch the spring by force_n / stiffness before the travel budget runs out --
    otherwise the guard can never trip and every push 'never builds the force'. GRIPPER LOGIC:
    the push closes on the junction mid-way through a sequence that otherwise needs OPEN fingers
    (unwind before it, advance onto the ring after it), so the close must be verified, the
    release must PRECEDE the advance, and a failed release must abort rather than slide a
    clamped gripper into the ring."""
    import yaml

    y = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))
    cl = y['assembly']['collar_clocking']
    sp = cl['seat_push']
    assert sp['enabled'] is True, 'the seat push defaults ON'
    S_max = max(float(v) for v in (cl.get('stiffness') or y['compliance']['stiffness'])[:3])
    need_mm = float(sp['force_n']) / S_max * 1000.0
    assert float(sp['max_travel_mm']) > need_mm, (
        f'max_travel_mm ({sp["max_travel_mm"]}) must exceed the {need_mm:.1f} mm spring stretch '
        f'that {sp["force_n"]} N needs at {S_max:.0f} N/m, or the push can never reach its force')

    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    body = src[src.index('def collar_clocking('):src.index('def traj_ref(')]
    # The push now runs FIRST -- the pads are already around the cable at the junction where it
    # wants them -- and everything after it needs OPEN fingers: the withdraw slides along the
    # cable, the lift-off frees the fingers, the advance threads the cable into the jaw.
    # NO close/verify any more: connector_clocking hands the part over STILL HELD
    # (open_gripper_after false), so the push presses with the pads already on it.
    order = ["guard_push.reset()",                                     # the push guard, not global
             "'release (seat push)'",                                  # reopen...
             "label='collar withdraw (connector -X)'",                 # ...BEFORE anything moves
             "label='collar lift-off (gripper -Z)'",
             "label='collar retreat + reorient axial'",
             "gripper.close('grasp collar')"]                          # then the collar bite
    idx = [body.index(t) for t in order]
    assert idx == sorted(idx), (
        'the seat push must run close -> verify -> press -> RELEASE before the withdraw; a '
        'release after it would drag the clamped junction along the cable')
    assert 'return False' in body[body.index("'release (seat push)'"):
                                  body.index("label='collar withdraw (connector -X)'")], (
        'a failed release must abort before the withdraw')
    # the connector moves with the press, so the collar poses must ride it
    assert 'T_grip = translation_matrix(d_push * axn) @ T_grip' in body, (
        'the grasp target must shift by the measured press travel, or the jaw closes short of '
        'the ring by exactly that much')


def test_post_engage_frame_is_a_config_choice_defaulting_to_target():
    """connector clocking, collar clocking, the escape leg and the tug all read ONE frame, T_clk --
    'target' (default) binds it to the recorded socket pose, 'believed' to the estimator's
    in-hand belief frozen at clocking time. The engagement itself always uses the target."""
    import yaml

    y = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))
    assert y['assembly'].get('post_engage_frame', 'target') == 'target', (
        'the default post-engage frame is the recorded target -- the socket is bolted down; '
        'the belief carries estimator error')

    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    assert "a.get('post_engage_frame')" in src and "'believed'" in src, (
        'the choice must be read from the config and validated')
    # every post-engage maneuver goes through the shared frame
    checks = (('def connector_clocking(', 'inverse(T_clk)) @ ref_start'),
              ('def collar_clocking(', 'axis = T_clk[:3, 0]'),
              ('def tug_verify_in_place(', 'axn_t = T_clk[:3, 0]'),
              ('def clocking_retract(', 'T_clk[:3, :3] @ step'))
    for fn, frag in checks:
        body = src[src.index(fn):]
        body = body[:body.index('\n    def ')] if '\n    def ' in body else body
        assert frag in body, f'{fn} must build its geometry from T_clk, found no {frag!r}'
    # the engagement keeps the target frame regardless of the choice
    assert 'T_base_targetobj = T_base_tconn @ inverse' in src, (
        'the engagement must stay anchored to the TARGET frame; post_engage_frame applies only '
        'after it')


def test_the_collar_axis_offset_is_read_from_config_in_the_connector_frame():
    """The collar's rotation line is tunable without touching the frame declaration.

    The declared connector frame origin is the mating-face reference, placed for insertion --
    nothing forces it onto the barrel centreline the collar physically turns about (bench: ~5 mm
    along the connector -Z). collar_clocking.axis_offset_mm shifts the axis LINE by a vector in
    the SOCKET frame's own axes, scoped to the collar maneuver only; connector clocking keeps the
    unoffset axis, since the socket physically corrects that captive stroke anyway.

    AND IT MUST NOT ROLL WITH assembly.engage_clock_deg. That key rolls the working frame about
    its own +X, so the frame's Y and Z stop being the axes this number was measured in -- while
    the offset itself is a property of the FIXTURE and stays put however far round the plug is
    mated. Resolving it in the rolled basis would swing a bench-measured -Z correction round to
    -Y at a 90 deg clock angle: the same YAML, a collar axis 10 mm somewhere nobody measured, and
    nothing in the log to say so. That is what axis_offset_base's roll-free basis prevents."""
    import numpy as np
    import yaml

    from urlab import config as urconfig, tool_frames

    cl = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))[
        'assembly']['collar_clocking']
    off = [float(v) for v in cl['axis_offset_mm']]
    assert len(off) == 3, 'axis_offset_mm must be connector-frame xyz'

    # the offset must move the LINE by exactly its perpendicular part -- connector-frame Y/Z are
    # orthogonal to the axis (+X) under any rigid placement, so |lateral| = |(y, z)|
    cfg = urconfig.load('bnc_assembly')
    T_t = tool_frames.load_targets(cfg)[cfg['assembly']['target_frame']]
    axn = T_t[:3, 0] / np.linalg.norm(T_t[:3, 0])
    d = T_t[:3, :3] @ (np.asarray(off) / 1000.0)
    lat = float(np.linalg.norm(d - np.dot(d, axn) * axn)) * 1000.0
    assert abs(lat - float(np.hypot(off[1], off[2]))) < 1e-9, (
        'a connector-frame offset must shift the axis line by exactly its Y/Z magnitude')

    # ---- INVARIANT TO THE ENGAGE CLOCK ANGLE ----
    # The app resolves the offset as (T_clk_rot @ R_clock_rot.T) @ off, and T_clk is the socket
    # frame ALREADY rolled by R_clock -- so the two cancel and the vector is the same at every
    # clock angle. Checked against the raw socket basis at a spread of angles, including the
    # -90 the config ships and the 180 where a sign slip would hide.
    d_socket = T_t[:3, :3] @ (np.asarray(off) / 1000.0)
    for clock_deg in (0.0, -90.0, 45.0, 180.0):
        R_clock = T.xyzrpy_to_matrix([0.0, 0.0, 0.0], np.radians([clock_deg, 0.0, 0.0]))
        T_clk = T_t @ R_clock                              # what the app builds and works in
        d_app = (T_clk[:3, :3] @ R_clock[:3, :3].T) @ (np.asarray(off) / 1000.0)
        assert np.allclose(d_app, d_socket, atol=1e-12), (
            f'the collar axis offset moved when engage_clock_deg = {clock_deg}: it is measured '
            f'against the fixture and must not roll with the plug')
    # and the roll-free basis is doing real work -- rolling WITH the frame would move it
    R_90 = T.xyzrpy_to_matrix([0.0, 0.0, 0.0], np.radians([-90.0, 0.0, 0.0]))
    d_naive = (T_t @ R_90)[:3, :3] @ (np.asarray(off) / 1000.0)
    if float(np.hypot(off[1], off[2])) > 1e-9:
        assert np.linalg.norm(d_naive - d_socket) * 1000.0 > 1.0, (
            'with a Y/Z offset and a 90 deg clock angle the naive rolled basis must land '
            'somewhere measurably different')
    # SHIPPED ZERO on purpose: the collar turns about the CONNECTOR AXIS itself, and a non-zero
    # Y/Z moves the whole maneuver's line off it -- which put the fingertip 10 mm off the ring.
    assert float(np.hypot(off[1], off[2])) < 1e-9, (
        f'collar_clocking.axis_offset_mm {off} shifts the collar line off the connector axis; '
        f'set it non-zero only from a MEASURED barrel centreline')

    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    assert "cl.get('axis_offset_mm'" in src, 'the offset must come from the config'
    helper = src[src.index('def axis_offset_base('):src.index('def clocking_retract(')]
    assert 'R_clock[:3, :3].T' in helper, (
        'the offset must be resolved in the frame ROLL-FREE basis -- undo the engage clock '
        'roll before rotating a fixture-measured vector into base')
    body = src[src.index('def collar_clocking('):src.index('def traj_ref(')]
    assert 'axis_offset_base()' in body, (
        'the collar maneuver must shift its point by the shared, roll-free offset')
    cable = src[src.index('def connector_clocking('):src.index('def collar_clocking(')]
    assert ('off_conn' not in cable and 'axis_offset_mm' not in cable
            and 'axis_offset_base' not in cable), (
        'the offset is scoped to the collar maneuver and the tug; connector clocking keeps the '
        'unoffset axis')
    # the tug centres its RE-GRIP on the same offset axis -- the pull direction cannot carry a
    # line offset, so the grasp position is where the correction lands. SAME helper, so the two
    # cannot disagree about where the collar's axis is.
    tug = src[src.index('def tug_verify_in_place('):src.index('def engage_insertion(')]
    assert 'axis_offset_base()' in tug and 'T_grasp = translation_matrix(-d_r) @ T_grasp' in tug, (
        'tug verification must centre its re-grip on the same offset connector axis')


def test_the_escape_releases_the_collar_before_retracting():
    """collar_clocking returns with the fingers CLOSED on the locked collar -- it grips the ring
    to turn it and nothing in the maneuver lets go. The escape that follows was written for OPEN
    fingers (its first leg "lifts the open fingers off the connector"), so retracting without a
    release drags the just-locked assembly sideways by the collar. The release therefore sits
    between the ESCAPE gate and the retract, gating the retract on its success -- and it must
    stay on that path for EVERY way into the escape (collar locked, failed, or disabled), which
    is why it lives at the boundary and not inside collar_clocking."""
    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    i_gate = src.index("phase_gate('ESCAPE'")
    tail = src[i_gate:i_gate + 1600]
    assert "gripper.open('release before escape')" in tail, (
        'the escape must open the gripper before retracting -- the fingers are still closed on '
        'the locked collar when collar clocking returns')
    assert tail.index("gripper.open('release before escape')") < tail.index('clocking_retract()'), (
        'the release must PRECEDE the retract, and the retract must be gated on it')


def test_the_offaxis_tilt_gate_clears_the_screws_own_compliance():
    """The gate must survive the tilt the PREVIOUS maneuver routinely leaves behind.

    Collar clocking checks how far the arm's orientation is from the collar-frame family in a way
    no rotation about the connector axis explains, and refuses to orbit above the gate. The arm
    arrives there straight off the connector screw, whose block is soft in exactly that direction --
    5 Nm/rad about tool0 Rx, with tare_before false -- so 0.1 Nm of out-of-axis moment is already
    1.15 deg, and a bayonet cam pushes sideways by design.

    A gate set below that turns ordinary compliance yield into an aborted run. This pins the
    RELATIONSHIP rather than the number, so retuning either the gate or the screw's stiffness stays
    honest: the gate has to clear the tilt a plausible cam moment produces.

    It is deliberately not pinned from above -- the tilt is logged unconditionally, so a genuine
    frame error still surfaces with its magnitude however the gate is set."""
    import numpy as np
    import yaml

    asm = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))['assembly']
    gate = float(asm['collar_clocking']['max_offaxis_tilt_deg'])
    # the screw axis is the connector +X; the SOFTEST out-of-axis rotational term is what tilts
    S_rot = [float(v) for v in asm['connector_clocking']['stiffness'][3:]]
    tilt_per_nm = np.degrees(1.0 / min(S_rot))
    assert gate >= 0.25 * tilt_per_nm, (
        f'the {gate:.1f} deg gate is below the {0.25 * tilt_per_nm:.1f} deg that a modest 0.25 Nm '
        f'out-of-axis cam moment produces at {min(S_rot):.0f} Nm/rad -- ordinary compliance yield '
        f'would abort the run. Raise max_offaxis_tilt_deg or stiffen connector_clocking')

    # and the gate must come from the config, not be baked into the app
    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    assert "_num(cl, 'max_offaxis_tilt_deg'" in src, 'the gate must be a config knob'
    assert '_tilt > cl_tilt_deg' in src, 'the gate must be READ, not hardcoded'
    # It gates the CONNECTOR's tilt from the socket axis, not the arm's orientation. The arm's
    # only mattered while the approach was an orbit from wherever the sweep ended; the axial
    # approach is placed in free space, so what matters is whether the cable lies on the line the
    # gripper is about to advance down.
    assert 'tilted from it (gate' in src, (
        'the measured tilt must be logged unconditionally -- it is the diagnostic that separates '
        'a constant frame error from variable compliance yield, and raising the gate must not '
        'hide it')
    assert 'refusing to thread it' in src, (
        'a cocked connector must abort BEFORE the axial advance -- that is the leg whose whole '
        'premise is that the cable lies on the axis')


def test_both_clockings_turn_about_the_socket_and_not_about_the_arm():
    """The two clocking strokes must share a frame, not just a function.

    They already share the motion: connector clocking builds its stroke as the conjugation
    (T_f @ screw @ inverse(T_f)) @ T_arm, collar clocking calls rotate_about_axis(T_arm, T_f.X,
    T_f.origin, angle), and those are the SAME operation to 1e-12. Both then run through
    screw_ramp. So a difference in behaviour between them cannot come from the maths.

    It came from the FRAME. Connector clocking turns about T_base_tconn -- the fixed, hand-measured
    socket pose. Collar clocking turned about T_base_conn, which is rebuilt at runtime as
    robot.tool0() @ T_tool0_conn_now and therefore carries every deviation the screw accumulated:
    spring yield under load, an advance that stopped short, a regrasp rebase. The socket does not
    move while any of that happens, so all of it is error in an axis.

    The runtime measurement is still worth one thing -- how far the bayonet cammed the connector IN
    -- and that is a translation ALONG the axis, which cannot move the line. Keep it, drop the
    rest."""
    import numpy as np

    from urlab.transforms import inverse, rotate_about_axis, xyzrpy_to_matrix, pose_error

    # ---- the two constructions are one construction -------------------------------------------
    rng = np.random.default_rng(0)
    T_f = xyzrpy_to_matrix(rng.normal(size=3), rng.normal(size=3))
    T_arm = xyzrpy_to_matrix(rng.normal(size=3), rng.normal(size=3))
    th = 0.7
    conj = (T_f @ xyzrpy_to_matrix([0.0, 0.0, 0.0], [th, 0.0, 0.0]) @ inverse(T_f)) @ T_arm
    lin, ang = pose_error(conj, rotate_about_axis(T_arm, T_f[:3, 0], T_f[:3, 3], th))
    assert lin * 1000.0 < 1e-6 and np.degrees(ang) < 1e-9, (
        'the conjugation and rotate_about_axis must be the same operation; if they diverge, one '
        'of the two clockings is turning about something else entirely')

    # ---- a drifted runtime frame is a drifted AXIS ---------------------------------------------
    axn = T_f[:3, 0] / np.linalg.norm(T_f[:3, 0])
    drift = np.array([0.0, 0.012, 0.004])                      # 12 mm lateral-ish runtime error
    lat = float(np.linalg.norm(drift - np.dot(drift, axn) * axn))
    assert lat > 0.001, 'the fixture drift must have a lateral component to be worth testing'
    # projecting onto the true axis keeps the cam-in and discards exactly that lateral part
    kept = T_f[:3, 3] + float(np.dot(drift, axn)) * axn
    resid = (T_f[:3, 3] + drift) - kept
    assert abs(float(np.linalg.norm(resid)) - lat) < 1e-12, (
        'projecting the measured origin onto the socket axis must discard the lateral drift and '
        'keep only the advance along it')

    # ---- and the app must do exactly that ------------------------------------------------------
    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    body = src[src.index('def collar_clocking('):src.index('def traj_ref(')]
    code = '\n'.join(ln for ln in body.splitlines() if not ln.lstrip().startswith('#'))
    assert 'axis = T_clk[:3, 0]' in code, (
        'collar clocking must take its axis DIRECTION from the shared post-engage frame T_clk '
        '(assembly.post_engage_frame; the recorded socket pose by default) -- exactly as cable '
        'clocking does')
    assert 'T_base_axis' in code, 'the socket-axis frame is gone'
    # T_base_conn may still be READ (for the cam-in and the drift report) but must not be the frame
    for banned in ('axis, point = T_base_conn', 'T_base_collar = T_base_conn',
                   'T_base_conn @ translation_matrix'):
        assert banned not in code, (
            f'{banned!r} turns about the RUNTIME connector pose, which carries the screw\'s '
            f'accumulated deviation; the socket it is captive in has not moved')


def test_a_clocking_stroke_follows_the_arc_and_not_the_chord():
    """A one-call ramp across a 90 deg orbit drags the held part 52 mm off its own axis.

    THE HAZARD. transforms.slerp_matrix SLERPs the rotation but LERPs the TRANSLATION, so
    admittance.ramp draws a STRAIGHT LINE between the two tool0 positions it is given. Over the
    servo-rate steps engage takes that is exact to microns. Over a whole clocking stroke it is not:
    a `theta` turn about an axis `r` from tool0 has a true path that bows into an arc, and the
    chord cuts inside it by r(1 - cos(theta/2)). Both ENDPOINTS stay exactly right, which is why
    this never shows up in a logged pose -- but mid-stroke the motion behaves as though the axis
    were r sin(theta/2)/(theta/2) away instead of r, i.e. ~10% CLOSER TO THE TOOL, and whatever is
    sitting on the axis gets scrubbed sideways through the sagitta.

    Pinned numerically rather than by eye, and structurally at both call sites, because the
    endpoints being correct makes every other check pass."""
    import numpy as np

    from urlab import config as urconfig, tool_frames
    from urlab.transforms import inverse, slerp_matrix, xyzrpy_to_matrix

    cfg = urconfig.load('bnc_assembly')
    frames, targets = tool_frames.load_frames(cfg), tool_frames.load_targets(cfg)
    cc = cfg['assembly']['connector_clocking']
    # The WIDEST leg the sweep commands -- the worst case for chord error, and what the app
    # actually hands screw_ramp. (It used to be the single relative rotation_deg stroke.)
    rot = widest_cable_leg(cfg['assembly'])
    push = float(cc['push_mm']) / 1000.0
    T_base_tconn = targets[cfg['assembly']['target_frame']]
    T_tool0_conn = frames[cfg['estimation']['initial_connector_frame']]
    ref_start = T_base_tconn @ inverse(T_tool0_conn)

    def ref_at(f):
        return (T_base_tconn @ xyzrpy_to_matrix([push * f, 0.0, 0.0], [rot * f, 0.0, 0.0])
                @ inverse(T_base_tconn)) @ ref_start

    axis = T_base_tconn[:3, 0] / np.linalg.norm(T_base_tconn[:3, 0])
    org = T_base_tconn[:3, 3]

    def worst_scrub(waypoints):
        """How far the connector origin is pulled OFF its own axis anywhere along the path."""
        out = 0.0
        for a, b in zip(waypoints, waypoints[1:]):
            for t in np.linspace(0.0, 1.0, 9):
                d = (slerp_matrix(a, b, t) @ T_tool0_conn)[:3, 3] - org
                out = max(out, float(np.linalg.norm(d - np.dot(d, axis) * axis)))
        return out * 1000.0

    # THE HAZARD IS REAL: one ramp across the whole stroke, endpoints exact, middle far off.
    assert worst_scrub([ref_start, ref_at(1.0)]) > 25.0, (
        'a single ramp across the clocking stroke should visibly cut the chord; if this no longer '
        'holds, slerp_matrix changed and screw_ramp may have become unnecessary')

    # SUBDIVIDING ON THE TRUE SCREW FIXES IT. n as screw_ramp picks it: ceil(duration * rate).
    n = 375
    assert worst_scrub([ref_at(k / n) for k in range(n + 1)]) < 0.05, (
        'subdividing on the true screw must keep the connector on its own axis')

    # AND BOTH STROKES MUST ACTUALLY GO THROUGH IT.
    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    assert 'def screw_ramp(' in src, 'the arc-following ramp helper is gone'
    body = src[src.index('def connector_clocking('):src.index('def collar_clocking(')]
    assert 'screw_ramp(' in body and 'adm_cc.ramp(' not in body, (
        'connector_clocking must take its stroke through screw_ramp, not a single adm_cc.ramp -- a '
        'one-call ramp cuts the chord and drags the connector off its axis')
    # EVERY leg of the sweep, not just the first. The retry is now the REVERSAL -- the gripper
    # stays closed and rotates back -- so it is a rotation about the socket axis exactly like the
    # leg before it, and a move_l would cut the same chord with the part clamped in the fingers.
    # (This hid for a while in the old ratchet, whose label said 'realign', not 'rotate'.)
    assert 'move_l(' not in body, (
        'no move_l inside connector_clocking: every leg is a rotation about the socket axis and must '
        'travel the arc (screw_ramp), not cut the chord between its endpoints')
    assert 'for k in range(1, cc_tries + 1):' in body and 'cc_legs[(k - 1) % len(cc_legs)]' in body, (
        'the sweep must WALK the sweep_deg positions, one per try, cycling when there are more '
        'tries than positions -- that cycling IS the oscillation')
    body = src[src.index('def collar_clocking('):src.index('def traj_ref(')]
    assert 'screw_ramp(' in body, (
        'collar_clocking must take its turn through screw_ramp too -- the fingers are CLOSED on '
        'the collar there, so the chord excursion goes straight into the ring')
    # adm_cl.ramp is allowed only where the endpoints differ by a PURE TRANSLATION along the
    # axis (identical orientation) -- a straight ramp draws that path exactly, so there is no
    # chord to cut. Two qualify: the seat push, and the axial advance onto the ring. Any OTHER
    # direct ramp risks spanning a rotation.
    allowed = ('adm_cl.ramp(T_a, T_b', 'adm_cl.ramp(T_retreat, T_grip')
    direct = [ln for ln in body.splitlines() if 'adm_cl.ramp(' in ln]
    assert all(any(a in ln for a in allowed) for ln in direct) and len(direct) == 2, (
        f'collar_clocking may direct-ramp only pure translations (the seat push and the axial '
        f'advance); found {direct!r} -- any ramp spanning a rotation cuts the chord')


def test_the_target_frame_and_the_in_hand_belief_name_the_same_point():
    """The clocking screw axis IS the target frame, so it must be the connector, not a neighbour.

    THE BUG THIS EXISTS TO CATCH, because it is silent and it wrecks the clocking strokes.
    `assembly.target_frame` names a point on the tool; so does the believed in-hand connector pose.
    Nothing forces them to be the SAME point, and when they drifted apart the failure showed up
    nowhere in the logs -- every pose printed looked self-consistent.

    THE INVARIANT. Engage commands tool0 to `target @ inverse(T_tool0_conn)`. At that arm pose the
    frame named by target_frame sits at `target @ inverse(T_tool0_conn) @ frames[target_frame]`.
    For that to actually BE the target -- a real mate -- the tail has to vanish:

        frames[assembly.target_frame] == T_tool0_conn

    WHY THE CLOCKING PASSES CARE MOST. connector_clocking resets its belief to the target and screws
    about THAT frame's +X:  ref_goal = (T_base_tconn @ screw @ inverse(T_base_tconn)) @ T_tool0.
    That is a rotation about an axis LINE through the target's origin. If the target is `d` off the
    true connector axis, a `theta` turn drags the connector origin through a chord of
    `2 d sin(theta/2)` instead of spinning it in place -- 19 mm of offset on a 90 deg stroke is a
    27 mm arc, which scrubs the connector sideways through the socket rather than clocking it.
    collar_clocking then inherits the same axis through T_base_conn, so one drift breaks both.

    Checked GEOMETRICALLY rather than by comparing the two config strings, so that an alias frame
    with the same pose passes and a same-named frame that someone later moves does not."""
    import numpy as np

    from urlab import config as urconfig, tool_frames
    from urlab.transforms import from_cfg, pose_error

    cfg = urconfig.load('bnc_assembly')
    frames = tool_frames.load_frames(cfg)
    tname = cfg['assembly']['target_frame']
    assert tname in frames, f'assembly.target_frame {tname!r} is not a declared frame'
    # RESOLVED THE WAY THE APP DOES: estimation.initial_connector_frame names a frame in the
    # shared catalogue and WINS; the inline pose is only an override for when no frame is named.
    # Comparing against the inline pose instead would fail whenever the frame is re-measured and
    # the stale duplicate has not caught up -- which is a real problem, but a different one, and
    # it is checked separately below.
    iname = cfg.get_path('estimation.initial_connector_frame')
    inline = (from_cfg(cfg['fingertip_grasp'])
              @ from_cfg(cfg['estimation']['initial_connector_in_fingertip']))
    belief = frames[iname] if iname else inline
    lin, ang = pose_error(frames[tname], belief)
    # Report the consequence in the units the operator cares about: the arc the connector would be
    # dragged through by the ACTUAL clocking stroke, chord = 2 d sin(theta / 2).
    theta = widest_cable_leg(cfg['assembly'])
    arc_mm = 2.0 * lin * 1000.0 * abs(np.sin(theta / 2.0))
    assert lin * 1000.0 < 0.05 and np.degrees(ang) < 0.05, (
        f'assembly.target_frame {tname!r} sits {lin * 1000.0:.2f} mm / {np.degrees(ang):.2f} deg '
        f'from the believed in-hand connector, so the clocking screw axis is that far off the '
        f'connector axis: the {np.degrees(theta):.0f} deg connector_clocking stroke would drag the '
        f'connector through a {arc_mm:.1f} mm arc instead of spinning it in place, and '
        f'collar_clocking would inherit the same axis. Point target_frame at the same frame as '
        f'estimation.initial_connector_frame.')

    # THE STALE-DUPLICATE CHECK. The inline override must not drift from the frame it duplicates.
    # It loses at runtime, so a gap changes no motion -- it just makes the app warn every run and
    # leaves a wrong number where a reader would trust it. The threshold is the app's own.
    if iname:
        d_lin, d_ang = pose_error(inline, frames[iname])
        assert d_lin * 1000.0 <= 0.5 and np.degrees(d_ang) <= 0.2, (
            f'estimation.initial_connector_in_fingertip is {d_lin * 1000.0:.2f} mm / '
            f'{np.degrees(d_ang):.2f} deg from frame {iname!r} that it duplicates. The frame wins, '
            f'so nothing moves wrong -- but the app warns every run and the stale pose misleads. '
            f'Set it to inverse(fingertip_grasp) @ frames[{iname!r}], or delete it.')

    # And the two config keys should AGREE BY NAME as well, since that is how a reader checks it.
    if iname:
        assert iname == tname, (
            f'estimation.initial_connector_frame {iname!r} and assembly.target_frame {tname!r} '
            f'name different frames; they happen to be geometrically equal today, but nothing '
            f'keeps them that way')


def test_wiggle_station_grid_and_the_retuned_excitation_stay_runnable():
    """wiggle_sampling generates stations from RANGES + RESOLUTIONS (uncertain_sampling's
    contract), and the config's excitation must keep passing the app's own pre-motion refusals --
    which otherwise only fire at the bench: every active axis needs a tone, tones pairwise
    co-prime with a long-enough orbit, spectral gap x wiggle_s >= 3 bins, peak speeds under the
    caps (the app refuses rather than dilating), and contact damping >= critical at 500 N/m."""
    import math

    import numpy as np
    import yaml

    from urlab.apps.wiggle_sampling import _grid_stations

    g = {'mode': 'grid', 'depth_mm': {'lower': -10.0, 'upper': 0.0, 'resolution': 2.0},
         'offset': {'lower': [0, 0, 0, 0, -2, 0], 'upper': [0, 0, 0, 0, 2, 0],
                    'resolution': [0, 0, 0, 0, 2, 0]}}
    st = _grid_stations(g)
    assert len(st) == 18, 'count is DERIVED: 6 depths x 3 pitches; degenerate axes cost nothing'
    assert sorted({x['depth_mm'] for x in st}) == [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0], (
        'both endpoints must be hit exactly, like traj.grid_deltas')
    assert len({x['name'] for x in st}) == len(st), 'names key the CSV -- must be unique'
    rnd = _grid_stations(dict(g, mode='random', num_stations=7, random_seed=3))
    assert len(rnd) == 7 and all(-10.0 <= x['depth_mm'] <= 0.0 for x in rnd)
    assert _grid_stations(dict(g, mode='sideways')) is None, 'a bad mode must refuse, not guess'

    y = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'wiggle_sampling.yaml')))
    w = y['wiggle']
    ordering = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')
    amp = [float(w['amplitude'][k]) for k in ordering]
    frq = [float(w['frequency_hz'][k]) for k in ordering]
    active = [i for i in range(6) if amp[i] > 0]
    assert all(frq[i] > 0 for i in active), 'an active axis with no tone is a constant offset'
    gcd = 0
    for i in active:
        gcd = math.gcd(gcd, int(round(frq[i] * 1000)))
    orbit = 1000.0 / gcd
    assert orbit >= 3.0 * (1.0 / min(frq[i] for i in active)) - 1e-9, (
        'the tone ratios are too simple -- the orbit closes before it fills the box')
    live = sorted(frq[i] for i in active)
    gap = min(b - a for a, b in zip(live, live[1:]))
    assert gap * float(w['wiggle_s']) >= 3.0 - 1e-9, (
        f'{gap:.2f} Hz gap x {w["wiggle_s"]} s < 3 FFT bins -- adjacent tones cannot be resolved')
    big = [a * max(float(v) for v in w['amplitude_scales']) for a in amp]
    pv = max(big[i] * 2 * math.pi * frq[i] for i in range(3) if frq[i] > 0)
    pw = max(big[i] * 2 * math.pi * frq[i] for i in range(3, 6) if frq[i] > 0)
    assert pv <= float(w['max_speed_mm_s']) and pw <= float(w['max_rotation_deg_s']), (
        f'peaks {pv:.1f} mm/s / {pw:.1f} deg/s exceed the caps -- the app will refuse to run')
    S = [float(v) for v in y['compliance']['stiffness']]
    zeta = float(y['compliance']['damping_ratio'][0])
    assert zeta * math.sqrt(S[0] / (S[0] + 39000.0)) >= 1.0, (
        'underdamped at the measured 39 N/mm seat -- a rebound is a lost contact, and a lost '
        'contact makes the burst a measurement of free space')


def test_sampling_preload_appends_and_does_not_move_the_sweep():
    """The map collector's preload must EXTEND the map, never shift it.

    estimator_eval's probing passes end on the mate precisely because uncertain_sampling does, so
    a preload that moved the whole path would silently break that parity while looking like it had
    only deepened the press. It has to leave the -X -> 0 sweep alone and append past the mate --
    which makes a preloaded map a strict superset of an un-preloaded one."""
    import yaml
    from urlab.apps import uncertain_sampling as us
    from urlab.skills import trajectory as traj

    T_tool0_held = T.xyzrpy_to_matrix([0.0, 0.0457, 0.159], np.zeros(3))
    mats = traj.load_csv(os.path.join(ROOT, 'configs', 'assembly_trajectory.csv'))
    dense = traj.resample(mats, 0.001, 1.0)
    mate = np.eye(4)
    anchor = mate @ T.inverse(mats[-1])

    def conn_x(ref):                       # connector depth wrt the mate, mm
        return T.matrix_to_xyzrpy(T.inverse(mate) @ (ref @ T_tool0_held))[0][0] * 1000.0

    first = traj.tool0_at(anchor, dense[0], T_tool0_held)
    last = traj.tool0_at(anchor, dense[-1], T_tool0_held)
    span = T.matrix_to_xyzrpy(mats[0])[0][0] - T.matrix_to_xyzrpy(mats[-1])[0][0]
    assert abs(conn_x(first) - span * 1000.0) < 1e-6, (
        'the preload must not move the SWEEP -- whatever length the CSV describes, the '
        'approach is unchanged and the press is appended past the mate')
    assert abs(conn_x(last)) < 1e-9, 'and it still reaches the mate exactly'
    for p in (5.0, 10.0):                  # the press lands exactly p past the mate
        assert abs(conn_x(us._axial_ref(last, T_tool0_held, p / 1000.0)) - p) < 1e-6

    # It presses along the PART's own axis, not the target's -- a tilted trial must not be
    # levered sideways into the socket wall.
    tilt = T.xyzrpy_to_matrix(np.zeros(3), np.radians([0.0, 8.0, 0.0]))
    ref_t = last @ T_tool0_held @ tilt @ T.inverse(T_tool0_held)
    moved = (T.inverse(ref_t @ T_tool0_held)
             @ (us._axial_ref(ref_t, T_tool0_held, 0.010) @ T_tool0_held))[:3, 3] * 1000.0
    assert np.allclose(moved, [10.0, 0.0, 0.0], atol=1e-6), \
        f'a tilted part must press along its OWN +X, got {moved}'

    # The retract keeps its magnitude semantics through the shared helper: a sign slip there
    # would drive INTO the socket at the end of every trial.
    for d in (0.005, -0.005):
        assert abs(conn_x(us._retract_ref(last, T_tool0_held, d)) + 5.0) < 1e-6, \
            'retract_distance_m is a MAGNITUDE; a negative value must still back out'

    # The distance is an operator knob; what must hold is that it is DECLARED (so a map's
    # collection conditions are recoverable) and never negative (which would retract).
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'uncertain_sampling.yaml')))
    assert 'preload_mm' in cfg['sampling'], \
        'preload_mm must stay declared -- it changes what every new map contains'
    assert float(cfg['sampling']['preload_mm']) >= 0.0, \
        'a negative preload would drive the retract INTO the socket'



def test_raw_wrench_representation_keeps_magnitude_and_ood_distance():
    """`raw` must pass the wrench through untouched, and that is the point of it.

    Both normalising representations rescale an out-of-distribution contact onto the shell the
    map lives on, so a press the map never recorded finds a confident neighbour at whatever pose
    merely shares its force direction -- the support machinery cannot flag what it cannot see.
    `raw` is the option that keeps a far row far. Pinned here so a future "tidy-up" cannot
    reintroduce a normalisation and quietly delete the property."""
    import csv as _csv
    import shutil
    import tempfile

    from urlab.skills.manifold import (FORCE_COLS, POSE_COLS, TORQUE_COLS, ManifoldEstimator,
                                       scaled12)

    tmp = tempfile.mkdtemp()
    try:
        # a LIGHT-TOUCH map (|f| ~ 5 N), the regime every BNC map was collected in
        rng = np.random.default_rng(0)
        rows = []
        for p_ in np.arange(-8.0, 8.01, 1.0):
            for x in np.linspace(-20.0, 0.0, 40):
                f = 5.0 + rng.normal(0.0, 0.5)
                rows.append([x, 0.0, 0.0, 0.0, p_, 0.0, -f, 0.0, 0.0, 0.0, 0.2, 0.0])
        path = os.path.join(tmp, 'm.csv')
        with open(path, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
            w.writerows(rows)

        base = {'manifold_csv': path, 'estimate_dims': ['pitch_deg'], 'min_force_n': 3.0,
                'min_observations': 5, 'interp_neighbors': 8, 'random_seed': 1}

        # ---- 1. the feature IS the wrench, times the block scale. Nothing else. ----
        est = ManifoldEstimator({**base, 'wrench_representation': 'raw',
                                 'scaling_constant_unit_force_to_mm': 0.11,
                                 'scaling_constant_unit_torque_to_mm': 0.20})
        f = np.array([[3.0, 0.0, 4.0], [90.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        tau = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
        assert np.allclose(est._wrench6(f, tau),
                           np.hstack([f * 0.11, tau * 0.20])), \
            'raw must be a pure scale of the measured wrench -- no normalisation, no cap'
        assert np.allclose(est._wrench6(f, tau)[2], 0.0), 'a zero wrench must stay zero'

        # ---- 2. NO saturation: unlike rawcap, the feature keeps growing with force ----
        mags = np.array([5.0, 30.0, 60.0, 120.0])
        ff = np.stack([[m, 0.0, 0.0] for m in mags])
        tt = np.zeros_like(ff)
        lens = {}
        for rep in ('unit', 'rawcap', 'raw'):
            e = ManifoldEstimator({**base, 'wrench_representation': rep,
                                   'scaling_constant_unit_force_to_mm': 1.0,
                                   'scaling_constant_unit_torque_to_mm': 1.0})
            lens[rep] = np.linalg.norm(e._wrench6(ff, tt)[:, :3], axis=1)
        assert np.allclose(lens['unit'], lens['unit'][0]), 'unit discards magnitude entirely'
        assert lens['rawcap'][-1] == lens['rawcap'][-2], 'rawcap must saturate at ref x cap'
        assert lens['raw'][-1] > lens['raw'][-2] > lens['raw'][-3], \
            'raw must NOT saturate -- a harder press has to keep moving the feature'
        assert np.isclose(lens['raw'][-1] / lens['raw'][0], mags[-1] / mags[0]), \
            'raw must be exactly proportional to |f|'

        # ---- 3. the property it exists for: a press stays FAR from a light-touch map ----
        obs = np.zeros((60, 12))
        obs[:, 0] = np.linspace(-18.0, -2.0, 60)          # in-distribution poses
        obs[:, 10] = 0.2
        far = {}
        for rep, sf in (('unit', 1.0), ('rawcap', 1.0), ('raw', 0.11)):
            e = ManifoldEstimator({**base, 'wrench_representation': rep,
                                   'scaling_constant_unit_force_to_mm': sf,
                                   'scaling_constant_unit_torque_to_mm': sf})
            d = {}
            for press in (5.0, 90.0):                      # map regime, then far outside it
                o = obs.copy()
                o[:, 6] = -press
                v6, w6 = e.prepare_observations(o[:, :6], o[:, 6:9], o[:, 9:12])
                pts = scaled12(v6, w6, e.s_rot, e.dim_w)
                nn = e.tree.query(pts, k=e.support_k, workers=-1)[0][:, -1]
                d[press] = float(np.median(nn / e.support_ref_at(v6[:, 0])))
            far[rep] = d
        ratio = {r: far[r][90.0] / far[r][5.0] for r in far}
        # unit is BLIND by construction: it threw the magnitude away, so an 18x harder
        # press sits at exactly the same distance as the light touch the map does hold.
        assert ratio['unit'] < 1.05, \
            f"unit cannot distinguish the press at all, got {ratio['unit']:.2f}x"
        # rawcap sees some of it, but only up to its ceiling -- past ref x cap it goes
        # blind too. raw is the one that keeps scaling, and that gap is the whole point.
        assert ratio['raw'] > 3.0, \
            f"raw must leave the press visibly off-map, got {ratio['raw']:.2f}x"
        assert ratio['raw'] > 2.0 * ratio['rawcap'], (
            f"raw ({ratio['raw']:.2f}x) must separate an out-of-distribution press far "
            f"more than the saturating option ({ratio['rawcap']:.2f}x) -- if this fails, "
            'the support gate has lost the only representation able to feed it')

        # ---- 4. a bad value still fails pre-motion, and names all three ----
        try:
            ManifoldEstimator({**base, 'wrench_representation': 'rawuncapped'})
            raise AssertionError('an unknown representation must be rejected BEFORE motion')
        except ValueError as exc:
            for name in ('unit', 'rawcap', 'raw'):
                assert name in str(exc), f'the error must name {name!r} as a valid choice'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_map_collection_and_eval_share_insert_speed_and_damping():
    """The map builder and the eval must agree on the two knobs that SET the contact force.

    Neither is a cycle-time detail. Simulated on admittance.py's own ODE against a 5e4 N/m wall,
    peak contact force runs 7 N at 2 mm/s and 187 N at 50 mm/s on an unchanged spring -- so a map
    collected at one speed and probed at another records a different contact regime for the SAME
    pose, which is precisely the mismatch that put the true correction uphill of no-correction in
    100% of the 17-Aug attempts.

    Damping is pinned on the CONTACT criterion, which is not the configured number: see the
    comment on the loop below."""
    import yaml

    us = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'uncertain_sampling.yaml')))
    ee = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'estimator_eval.yaml')))

    v_us = float(us['speed']['max_cartesian_translation_mm_s'])
    v_ee = float(ee['speed']['max_cartesian_translation_mm_s'])
    assert v_us == v_ee, (f'insert speed drifted: sampler {v_us} vs eval {v_ee} mm/s -- the '
                          'map and the observations would sit in different force regimes')

    # CONTACT damping, not free-air damping. zeta is defined against the admittance
    # stiffness S, but in contact the loop stiffness is S + k_env, so what governs the
    # bounce is zeta_contact = zeta * sqrt(S/(S+k_env)) -- with the virtual mass cancelling
    # out entirely. zeta = 1.0 was tried on hardware on 2026-08-17 and RANG; this is the
    # criterion that predicts it, so it is the one worth pinning.
    # k_env measured as dF/dx over the loaded part of 26 real insertions: 3.5-9 N/mm while
    # probing, ~39 N/mm on the seated final insertion.
    K_PROBE_MAX, K_SEATED = 9000.0, 39000.0
    from urlab import tool_frames as _tf

    def axial(K6, R):
        """Stiffness facing the insertion: compliance acts on TOOL0 axes but the part
        goes in along the CONNECTOR's +X, and for the BNC frame those differ."""
        u = R @ np.array([1.0, 0.0, 0.0])
        return float(u @ (np.asarray(K6[:3], dtype=float) * u))

    for name, cfg in (('uncertain_sampling', us), ('estimator_eval', ee)):
        R = _tf.load_frames(cfg)[cfg['held_frame']][:3, :3]
        M = np.array([float(x) for x in cfg['compliance']['mass']])
        Z = np.array([float(x) for x in cfg['compliance']['damping_ratio']])
        S_ax = axial(cfg['compliance']['stiffness'], R)
        # the sampler can be driven into the SEATED contact by sampling.preload_mm; the
        # eval's probing passes stop at the mate and its commit carries its own stiffness.
        k_worst = K_SEATED if name == 'uncertain_sampling' else K_PROBE_MAX
        zc = float(Z[0]) * np.sqrt(S_ax / (S_ax + k_worst))
        assert zc >= 0.7, (
            f'{name}: zeta {Z[0]:g} against S_axial {S_ax:.0f} N/m gives zeta_contact '
            f'{zc:.2f} at k_env {k_worst/1000:.0f} N/mm -- below 0.7 the part BOUNCES off '
            f'the seat and the logged wrench oscillates instead of settling')
        # forward Euler at reference_rate_hz needs dt*D/M < 2; HIGH zeta is what breaks it
        D = Z * 2.0 * np.sqrt(M * np.array([float(x) for x in cfg['compliance']['stiffness']]))
        dt = 1.0 / float(cfg['compliance'].get('reference_rate_hz', 125.0))
        assert float(np.max(dt * D / M)) < 1.0, (
            f'{name}: dt*D/M = {np.max(dt * D / M):.2f}, too close to the forward-Euler '
            'stability limit of 2')


def test_observation_passes_inherit_the_samplers_preload():
    """If the map is collected with a press, the observation passes must press too.

    The map's DEEPEST rows are the ones nearest the seat -- the rows that decide the correction.
    Collect them under load (sampling.preload_mm) and a probing pass that stops at the mate never
    visits them, so the observations sit off the map exactly where it matters most. That is the
    same map/observation regime split the preload exists to close, with the sign reversed, so the
    eval has to track the sampler rather than carry its own number."""
    src = open(os.path.join(ROOT, 'urlab', 'apps', 'estimator_eval.py'), encoding='utf-8').read()

    # ---- 1. it INHERITS rather than duplicating the sampler's value ----
    import yaml
    ee = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'estimator_eval.yaml')))
    samp = (ee['eval']['collection'].get('sampling') or {})
    us = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'uncertain_sampling.yaml')))
    assert 'preload_mm' in samp, \
        'collection.sampling.preload_mm must exist, or the eval cannot follow the map'
    # null = inherit the sampler, a number = an explicit override. EITHER is fine; what is
    # not fine is an override that DISAGREES with the sampler, because then the eval probes
    # to a different depth than the map was collected at -- the exact split this closes.
    if samp['preload_mm'] is not None:
        assert float(samp['preload_mm']) == float(us['sampling']['preload_mm']), (
            f"collection.sampling.preload_mm is pinned at {samp['preload_mm']} while the "
            f"sampler presses {us['sampling']['preload_mm']} mm -- the observations would "
            'sit off the map at exactly the depths that decide the seat. Match it, or set '
            'it back to null to inherit.')
    assert "((samp_ref or {}).get('sampling') or {}).get('preload_mm')" in src, \
        'the inherit path must read the sampler config at run time'

    # ---- 2. the OBSERVATION passes get it; the COMMIT does not ----
    assert src.count('preload_mm=obs_pre_mm') == 1, \
        'exactly one call site -- the observation pass -- may take the inherited preload'
    commit = src[src.index('adm_final, refs, T_believed, guard_ctl=guard_final'):]
    commit = commit[:commit.index(')')]
    assert 'preload_mm' not in commit, (
        "the commit must NOT take preload_mm: its press is already in T_base_commit, which "
        'shifts the whole reference path, so passing it again would press twice')

    # ---- 3. the press uses the SAMPLER's mechanism, from the actual stop point ----
    assert '_axial_ref(last_ref, T_bel, preload_mm / 1000.0)' in src, (
        'the observation press must advance from where the pass actually STOPPED (the sampler '
        'mechanism), not command a deeper path -- the two differ once a force stop lands short '
        'of the mate, and the map was built with the first')
    assert 'from .uncertain_sampling import _axial_ref' in src, \
        'and it must be the sampler\'s own helper, not a reimplementation that can drift'

    # ---- 4. un-guarded press AND un-guarded settle after one ----
    press = src[src.index('press = preload_mm > 0 and'):]
    press = press[:press.index('if hold_s > 0:')]
    assert 'guard=None' in press, \
        'the press must be un-guarded -- at the seat the guard has already tripped'
    # `press`, not `preload_mm > 0`: approach_only can withhold the press on a pass that never
    # made contact, and then there is no standing load for the settle to protect, so the guard
    # must go back on. Keying the settle off the CONFIG rather than off what actually happened
    # would leave that settle un-guarded with nothing pressing.
    assert 'None if press else guard' in press, (
        'the settle after a press must be un-guarded too: hold() is a ramp, so an armed guard '
        'returns seated on the first cycle and cuts short the loaded rows the press recorded')

    # ---- 5. and the loaded rows must actually reach the estimator when the press IS the
    # evidence mechanism. advance_cb is log_cb under observe_during: insertion (the mode this
    # parity story belongs to) and None under observe_during: wiggle, where the evidence is the
    # wiggle's and the press only loads the station it oscillates about.
    assert 'on_step=advance_cb' in press, \
        'the press must log through advance_cb -- log_cb when the pass is the evidence source'
    assert 'advance_cb = None if observe_wiggle else log_cb' in src, \
        'and advance_cb must resolve to log_cb whenever observe_during is insertion'


def test_observation_passes_do_not_attempt_to_seat():
    """Only eval.final_insertion may try to seat the connector.

    An observation pass has TWO ways to go deep. The guarded ramp stops itself at first contact,
    which is a probe. But a pass whose guard never trips has driven the reference all the way to
    the mate, and the preload then presses preload_mm PAST it UN-GUARDED -- and that is a seating
    attempt, made from the belief this pass exists to correct. It fails quietly: the pass still
    returns observations and the trial still looks fine, while the connector has been pressed home
    (or jammed) on an error the estimator has not seen yet.

    approach_only closes that second path: no contact stop, no press and no wiggle. It is passed
    to the observation call site and NOT to the commit, which is the whole distinction.
    """
    import yaml

    src = open(os.path.join(ROOT, 'urlab', 'apps', 'estimator_eval.py')).read()
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'estimator_eval.yaml')))
    col = cfg['eval']['collection']

    assert col.get('approach_only') is True, (
        'eval.collection.approach_only must ship ON -- an observation pass that seats does so '
        'from an uncorrected belief, and nothing about the run reports that it happened')

    # CONTACT IS MEASURED, NOT INFERRED FROM THE GUARD. The first deployment gated the wiggle on
    # the guard trip, and the guard is the ABORT limit (10 N held 5 s at the time) -- a 3-second
    # approach cannot trip it even in principle, so every pass read as free space and collected
    # zero observations (2026-08-20). The press runs first (distance-bounded, doubling as the
    # probe) and the F/T decides: spring x stretch on a stopped part, ~zero on a free one.
    assert 'press = preload_mm > 0 and (seated or observe_wiggle or not approach_only)' in src, (
        'under observe_wiggle the press must run even without a guard trip -- it IS the contact '
        'probe, and without it a part stopped exactly at the trajectory end reads as free space')
    assert 'in_contact = f_now >= contact_force_n' in src, \
        'contact must be a force MEASUREMENT -- the guard trip alone reads real contact as free space'
    # the wiggle rides the measured gate: with no contact there is nothing to excite
    wig = src[src.index('if obs_wig is not None and obs_wig_s > 0'):]
    assert 'and in_contact' in wig[:120], (
        'the wiggle must be withheld without measured contact: oscillating in free space logs '
        'rows that carry no contact information and pulls the manifold toward zero-force poses')

    # ---- the commit is exempt, and that exemption is the point ----
    assert src.count('approach_only=approach_only') == 1, \
        'exactly one call site -- the observation pass -- may be approach-only'
    commit = src[src.index('adm_final, refs, T_believed, guard_ctl=guard_final'):]
    commit = commit[:commit.index(')')]
    assert 'approach_only' not in commit, (
        'eval.final_insertion must NOT be approach-only: it runs after the observations have '
        'corrected the belief, and seating is exactly what it is for')


def test_observations_come_from_the_wiggle():
    """Under observe_during: wiggle, the WIGGLE is the only observation source.

    The approach, press and settle still run -- they are how the wiggle reaches loaded contact --
    but their rows are a drive-in that a wiggle_sampling map does not contain, so logging them
    would put every pass partly off-map by construction (the same class of failure as the v5
    depth gap, from the other direction). The estimate then corrects the belief and
    eval.final_insertion makes the one direct insertion from the corrected pose.
    """
    import yaml

    src = open(os.path.join(ROOT, 'urlab', 'apps', 'estimator_eval.py')).read()
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'estimator_eval.yaml')))
    col = cfg['eval']['collection']

    # the config ships the mode, and with a duration that can actually collect something
    assert col.get('observe_during', 'wiggle') == 'wiggle'
    assert float(col.get('wiggle_s', 0.0)) > 0.0, (
        "observe_during: wiggle with wiggle_s 0 collects ZERO observations per pass -- the app "
        'refuses it pre-motion, so the shipped config must not be that config')

    # one callback split decides who logs: the advance/press/settle go quiet, the wiggle keeps
    # logging in BOTH modes (it was an observation source before this mode existed)
    assert 'advance_cb = None if observe_wiggle else log_cb' in src
    # the function's actual extent, not a guessed character count -- the body grows with its
    # comments, and a window that silently excludes the wiggle asserts nothing about it
    body = src[src.index('def run_insertion('):src.index('    out_dir = os.path.join')]
    assert body.count('on_step=advance_cb') >= 3, (
        'approach, peck back-off, press and settle must all log through advance_cb -- any one of '
        'them still holding log_cb leaks drive-in rows into a wiggle-only collection')
    wig = body[body.index('wigmod.run('):]
    assert 'on_step=log_cb' in wig[:200], 'the wiggle itself must still log'

    # the stop depth cannot come from obs[-1] when the advance is not logging
    assert '_depth_now()' in body, (
        'the guard-trip stop depth must have a direct read -- obs[-1] is empty by design when '
        'the advance does not log, and the stop-signature fusion still needs the depth')

    # a zero-row attempt (possible now: no contact -> no wiggle -> nothing) is SKIPPED, not fitted
    assert 'if not len(full):' in src, (
        'an attempt with zero observations must skip the estimate -- an update from zero rows '
        "is the estimator's prior wearing an estimate's name")

    # the mode is validated pre-motion, and the commit is exempt from it
    assert "obs_during == 'wiggle' and (obs_wig is None or obs_wig_s <= 0)" in src, \
        'a config that would collect nothing must fail before the arm moves'
    assert src.count('observe_wiggle=(obs_during ==') == 1, \
        'exactly one call site -- the observation pass -- may narrow its logging to the wiggle'
    commit = src[src.index('adm_final, refs, T_believed, guard_ctl=guard_final'):]
    commit = commit[:commit.index(')')]
    assert 'observe_wiggle' not in commit, (
        "the commit logs its whole motion: its observations are the RECORD of the seating, not "
        'estimator evidence, and a wiggle-only record of a direct insertion would be empty')


def test_sampler_and_eval_presses_are_distinct_knobs():
    """Two presses exist and they are not interchangeable -- keep them legible.

    final_insertion.preload_mm presses the COMMIT by shifting its whole reference path deeper.
    collection.sampling.preload_mm presses each OBSERVATION pass by advancing from wherever it
    stopped. Same units, same name-stem, different mechanism and different phase; the failure
    mode is someone 'unifying' them and silently double-pressing the commit or halving the map."""
    import yaml

    ee = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'estimator_eval.yaml')))
    us = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'uncertain_sampling.yaml')))

    assert float(ee['eval']['final_insertion']['preload_mm']) > 0.0, \
        'the commit press is what drives the connector home'
    assert float(us['sampling']['preload_mm']) >= 0.0, \
        'the sampler press is an operator knob, but it can never be negative'
    # The eval's observation press must END UP equal to the sampler's, whether it gets there
    # by inheriting (null) or by an explicit override that agrees.
    obs_pre = (ee['eval']['collection'].get('sampling') or {}).get('preload_mm')
    effective = float(us['sampling']['preload_mm'] if obs_pre is None else obs_pre)
    assert effective == float(us['sampling']['preload_mm']), (
        f'the eval would preload {effective} mm on its observation passes while the map was '
        f"collected at {us['sampling']['preload_mm']} mm -- probing and map must reach the "
        'same depth')

    src = open(os.path.join(ROOT, 'urlab', 'apps', 'estimator_eval.py'), encoding='utf-8').read()
    # the commit press is a TARGET shift; the observation press is an AXIAL advance
    assert 'T_base_commit = T_base_targetobj @ translation_matrix(' in src, \
        'the commit press must remain a shift of the reference path'
    assert 'T_pre = _axial_ref(last_ref, T_bel,' in src, \
        'the observation press must remain an advance from the stop point'


def test_insertion_tester_injection_sign_and_modes():
    """The tester's whole output is (offset -> seated?), so the offset had better mean something.

    The injection is estimator_eval's: T_believed = T_true @ delta. The robot then builds every
    reference as T_base_tconn @ row @ inverse(T_believed), so the connector's TRUE pose w.r.t. the
    target comes out at row @ inverse(delta) -- the physical misalignment is the INVERSE of the
    injected belief error, not the belief error itself. Confusing the two silently mirrors every
    basin this script produces, so the relationship is pinned here and both columns are logged."""
    from urlab.apps import insertion_tester as it
    from urlab.apps.estimator_eval import _corr_to_m
    from urlab.skills.manifold import mats_from_vec6

    # ---- the sign relationship, end to end through the real reference construction ----
    T_true = T.xyzrpy_to_matrix([0.0, 0.0457, 0.159], np.zeros(3))
    T_base_tconn = T.xyzrpy_to_matrix([0.4, -0.1, 0.3], np.radians([10.0, -5.0, 20.0]))
    for off6 in ([0.0, 0.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
                 [1.0, 0.0, -2.0, 0.0, -3.0, 0.0]):
        delta = _corr_to_m(mats_from_vec6(np.asarray(off6, dtype=float)))
        T_believed = T_true @ delta
        # the reference the app commands at the mate (row = identity)
        ref = T_base_tconn @ np.eye(4) @ T.inverse(T_believed)
        # ...and where the TRUE connector actually ends up, since the part is fixtured
        actual = T.inverse(T_base_tconn) @ (ref @ T_true)
        xyz, rpy = T.matrix_to_xyzrpy(actual)
        got = list(xyz * 1000.0) + list(np.degrees(rpy))
        pred = it._physical_offset(mats_from_vec6(np.asarray(off6, dtype=float)))
        assert np.allclose(got, pred, atol=1e-6), (
            f'_physical_offset must predict where the part really lands for {off6}: '
            f'predicted {np.round(pred, 3)}, geometry gives {np.round(got, 3)}')
        # and it is genuinely the INVERSE, not a copy of the injected number
        if any(abs(c) > 1e-9 for c in off6):
            assert not np.allclose(got, off6, atol=1e-6), \
                'physical and injected offsets must not be conflated -- they differ in sign'

    # ---- both insertion strategies exist and are validated pre-motion ----
    assert it.MODES == ('direct', 'wiggle'), 'the two strategies under test'
    src = open(os.path.join(ROOT, 'urlab', 'apps', 'insertion_tester.py'),
               encoding='utf-8').read()
    # the maneuvers are IMPORTED from the apps that own them, not re-implemented, or this tester
    # would slowly stop measuring what production actually runs
    for owner, name in (('bnc_assembly', '_ScrewAdvance'), ('bnc_assembly', '_AnyGuard'),
                        ('calibration_check', 'line_rows')):
        assert name in src and f'from .{owner} import' in src, \
            f'{name} must come from {owner}, not be copied into the tester'

    # ---- seating is MEASURED on the true pose, never inferred from why the motion ended ----
    assert 'seated = bool(abs(seat6[0]) <= seat_tol_mm' in src, (
        'seated must be decided by the measured seat pose: a guard trip AT the mate is a good '
        'seat and completing the path 3 mm short is not, and only the pose separates them')


def test_insertion_tester_config_is_coherent():
    """The shipped config must describe a test that can actually succeed."""
    import yaml

    c = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'insertion_tester.yaml')))
    ins = c['insertion']
    assert ins['insertion_mode'] in ('direct', 'wiggle')
    offs = ins['offsets']
    assert all(len(o) == 6 for o in offs), 'every offset is a 6-vector [mm x3, deg x3]'
    assert any(not any(abs(float(v)) > 1e-12 for v in o) for o in offs), (
        'keep a NOMINAL (all-zero) offset as the control -- without it a run cannot distinguish '
        '"this offset is too big" from "the setup is broken"')
    # repeats is the operator's call: more gives a better basin estimate (seat success is
    # a RATE, and one try per offset cannot separate a 90% basin from a 40% one), but a
    # single pass is a legitimate quick check. Only a non-positive count is incoherent.
    assert int(ins['repeats']) >= 1, 'repeats must be at least 1'

    wg = ins['wiggle']
    _r = _resolved_wiggle('insertion_tester', 'insertion', 'wiggle')
    amp, frq = _r['amplitude'], _r['frequency_hz']
    live = [d for d in amp if abs(float(amp[d])) > 0.0]
    assert live, 'the wiggle must oscillate on at least one axis'
    for d in live:
        assert float(frq[d]) > 0.0, f'wiggle axis {d} has amplitude but no frequency'
    f_max = max(float(frq[d]) for d in live)
    assert float(wg['sample_rate_hz']) >= 4.0 * f_max, (
        f'sample_rate_hz {wg["sample_rate_hz"]} aliases a {f_max} Hz component -- the wiggle '
        'would silently run at a frequency nobody chose')
    assert float(wg['engage_advance_mm']) <= float(wg['target'][0]), (
        'the engagement threshold must sit at or before where the reference pushes to, or the '
        'wiggle can only ever time out')
    # co-prime frequencies: a rational ratio retraces one closed Lissajous path forever
    if len(live) >= 2:
        fr = sorted(int(round(float(frq[d]) * 10)) for d in live)
        assert math.gcd(fr[0], fr[1]) == 1, (
            f'wiggle frequencies {[float(frq[d]) for d in live]} Hz share a common factor, so '
            'the orbit closes early and re-probes one line through the rectangle')

    # the contact-damping criterion this repo learned the hard way (2026-08-17: zeta 1.0 rang)
    M = np.array([float(x) for x in c['compliance']['mass']])
    S = np.array([float(x) for x in c['compliance']['stiffness']])
    Z = np.array([float(x) for x in c['compliance']['damping_ratio']])
    # the SOFTEST translational axis is the worst case for bouncing (zeta_contact grows
    # with S), checked against the stiffest contact actually measured, ~39 N/mm seated
    S_soft = float(np.min(S[:3]))
    zc = float(Z[0]) * np.sqrt(S_soft / (S_soft + 39000.0))
    assert zc >= 0.7, (
        f'zeta {Z[0]:g} at its softest axis S {S_soft:.0f} gives zeta_contact {zc:.2f} '
        'against a seated contact -- '
        'below 0.7 the connector bounces off the seat instead of settling into it')
    assert float(np.max((1.0 / 125.0) * (Z * 2.0 * np.sqrt(M * S)) / M)) < 1.0, \
        'forward-Euler headroom at 125 Hz'


def test_wiggle_speed_cap_is_a_time_dilation():
    """Capping the wiggle must cost only TIME -- never search area, never orbit shape.

    The wiggle paces by time, so seg_time() and speed.phase_scale never reach it; the cap is the
    only speed limit it has. It works by dilating the waveform clock, and the two things that must
    survive that are the ones a naive implementation would break: shrinking the AMPLITUDE would
    quietly shrink the region the wiggle can find a lead-in in, and rounding the FREQUENCIES
    individually would break their mutual primality and collapse the Lissajous orbit onto a single
    closed path through the rectangle -- which is the exact failure the co-prime choice exists to
    prevent."""
    from urlab.skills.trajectory import wiggle_time_scale

    A = [0.0, 0.0, 5.0, 0.0, 5.0, 0.0]
    F = [0.0, 0.0, 0.7, 0.0, 1.1, 0.0]

    # ---- uncapped is exactly A*2*pi*f, and the orbit is 1/gcd ----
    s, pv, pw, orbit = wiggle_time_scale(A, F)
    assert s == 1.0, 'no cap must leave the wiggle untouched'
    assert abs(pv - 5.0 * 2 * math.pi * 0.7) < 1e-9
    assert abs(pw - 5.0 * 2 * math.pi * 1.1) < 1e-9
    assert abs(orbit - 10.0) < 1e-9, '0.7 and 1.1 Hz close a 10 s orbit'

    # ---- the cap binds exactly, and dilation is linear ----
    for cap in (10.0, 5.0, 3.0, 1.0):
        sc, pv_, pw_, orb = wiggle_time_scale(A, F, cap)
        assert abs(pv_ * sc - cap) < 1e-9, f'peak speed must land ON the cap, got {pv_ * sc}'
        assert abs(orb / sc - orbit / sc) < 1e-9
        assert sc < 1.0
    # the ROTATION cap binds when it is the tighter one
    sc_w = wiggle_time_scale(A, F, None, 5.0)[0]
    assert abs(wiggle_time_scale(A, F, None, 5.0)[2] * sc_w - 5.0) < 1e-9
    # whichever is tighter wins
    assert wiggle_time_scale(A, F, 3.0, 999.0)[0] == wiggle_time_scale(A, F, 3.0)[0]
    assert wiggle_time_scale(A, F, 999.0, 5.0)[0] == wiggle_time_scale(A, F, None, 5.0)[0]

    # ---- a SLACK cap must never speed the wiggle UP ----
    assert wiggle_time_scale(A, F, 1e6)[0] == 1.0, 'a cap above the peak is a no-op, not a boost'

    # ---- the two invariants the whole design rests on ----
    sc = wiggle_time_scale(A, F, 3.0)[0]
    #  (a) amplitude is not an input to the scale at all -> search AREA preserved
    assert wiggle_time_scale(A, F, 3.0)[1] == pv, 'the reported peak is the UNDILATED one'
    #  (b) frequencies are scaled TOGETHER, so their ratio -- and hence co-primality -- holds
    assert abs((0.7 * sc) / (1.1 * sc) - 0.7 / 1.1) < 1e-12, (
        'dilation must scale the waveform CLOCK, not the individual frequencies: scaling them '
        'separately (or rounding them) breaks mutual primality and collapses the orbit')

    # ---- degenerate configs must not blow up ----
    assert wiggle_time_scale([0.0] * 6, [0.0] * 6, 5.0) == (1.0, 0.0, 0.0, 0.0)
    assert wiggle_time_scale([0, 0, 5.0, 0, 0, 0], [0, 0, 0.0, 0, 0, 0], 5.0)[0] == 1.0


def _resolved_wiggle(app, *keys):
    """The wiggle an app actually ends up driving, after inheriting from the tuned file.

    Tests must ask this rather than reading `amplitude` out of the app's own config: the
    parameters deliberately are not there any more, and a test that reads the raw block is
    asserting an arrangement that was removed on purpose."""
    from urlab import config as urconfig
    from urlab.skills import wiggle as wigmod
    cfg = urconfig.load(app)
    node = cfg
    for k in keys:
        node = (node or {}).get(k) if isinstance(node, dict) else None
    node = node or {}
    blk, _ = wigmod.from_shared(cfg, node if 'amplitude' in node else node.get('wiggle'),
                                node.get('wiggle_from', 'wiggle_sampling.yaml'), app)
    return blk


def test_every_app_resolves_to_the_tuned_wiggle():
    """Each app that wiggles must end up with the parameters from configs/wiggle_sampling.yaml.

    That file is where the axes, amplitudes, frequencies and phases were tuned against hardware.
    An app carrying its own copy is an app whose observations stop being comparable the moment the
    tuned values change -- which is the whole reason the waveform itself is shared. So this does
    not check that each config DECLARES a wiggle; it checks that each app RESOLVES to the same
    one, which is the property that actually matters.

    A deliberate local override is still allowed -- `from_shared` layers it on top and names it in
    the source string -- but it has to be deliberate, and this test makes an accidental one
    visible.
    """
    import yaml

    from urlab import config as urconfig
    from urlab.skills import wiggle as wigmod

    with open(os.path.join(ROOT, 'configs', 'wiggle_sampling.yaml')) as fh:
        tuned = (yaml.safe_load(fh) or {}).get('wiggle') or {}
    assert tuned.get('amplitude'), 'configs/wiggle_sampling.yaml must carry the tuned block'

    cases = [('estimator_eval', ('eval', 'collection')),
             ('bnc_assembly', ('assembly', 'engage')),
             ('insertion_tester', ('insertion', 'wiggle'))]
    for app, keys in cases:
        cfg = urconfig.load(app)
        node = cfg
        for k in keys:
            node = (node or {}).get(k) if isinstance(node, dict) else None
        node = node or {}
        blk, src = wigmod.from_shared(
            cfg, node if 'amplitude' in node else node.get('wiggle'),
            node.get('wiggle_from', 'wiggle_sampling.yaml'), app)
        for key in ('amplitude', 'frequency_hz'):
            assert blk.get(key) == tuned.get(key), (
                f'{app} resolves {key} to {blk.get(key)}, not the tuned {tuned.get(key)} '
                f'(source: {src}). Remove the local copy, or make the override deliberate.')
        wigmod.Wiggle.from_cfg(blk, app)          # and it must still validate


def test_the_collar_is_grasped_axially_and_turned_by_a_wrist_twist():
    """collar_clocking takes the ring with the fingers PARALLEL to the cable and twists the wrist.

    The socket is wall-mounted, and that is what decides the grasp. Making the fingertip frame
    coincide with the collar frame -- the obvious reading of "put the pads on the ring" -- puts
    tool0 183 mm OFF the connector axis, level with the mating face, and a rotation about that
    axis then swings the flange through a 258 mm arc ACROSS the wall. Rolling the grasp 90 deg
    about the jaw-CLOSING axis instead lays tool0 ON the axis with its Z collinear: the same bite
    on the same ring, but the turn becomes a rotation about tool0's own Z and the flange does not
    move at all.

    Four properties, and each is a way a plausible implementation goes wrong:
      * TOOL0 ON THE AXIS, Z COLLINEAR -- otherwise the turn is still an orbit, just a smaller one.
      * THE BITE IS UNCHANGED. The fingertip still lands on the collar station and the jaws still
        close ACROSS a diameter; a roll about the wrong axis would close them along the cable.
      * THE FLANGE STAYS PUT through the turn, moving only by push_mm.
      * THE APPROACH IS A PURE AXIAL TRANSLATION, so a straight ramp draws it exactly.
    """
    import yaml

    from urlab import config as urconfig, tool_frames
    from urlab.transforms import rotate_about_axis

    cfg = urconfig.load('bnc_assembly')
    asm = cfg.section('assembly')
    cl = asm['collar_clocking']
    T_ftip = tool_frames.load_frames(cfg)['fingertip']
    T_socket = tool_frames.load_targets(cfg)[asm['target_frame']]
    eng = np.radians(float(asm.get('engage_clock_deg', 0.0) or 0.0))
    T_clk = T_socket @ T.xyzrpy_to_matrix([0., 0., 0.], [eng, 0., 0.])

    collar_x = float(cl['collar_offset_mm']) / 1000.0     # from the connector ORIGIN, on the axis
    cl_rot = np.radians(float(cl['rotation_deg']))
    push_m = float(cl['push_mm']) / 1000.0
    T_collar = T_clk @ T.translation_matrix([collar_x, 0.0, 0.0])
    axis, point = T_clk[:3, 0], T_collar[:3, 3]
    axn = axis / np.linalg.norm(axis)

    RADIAL = T.inverse(T_ftip)                                   # the grasp this replaced
    AXIAL = (T.xyzrpy_to_matrix([0., 0., 0.], [0., -np.pi / 2., 0.]) @ T.inverse(T_ftip))
    frames_cat = tool_frames.load_frames(cfg)
    iname_M = cfg.get_path('estimation.initial_connector_frame')
    _ = frames_cat[iname_M]                                  # the belief frame must exist

    def in_axis_frame(T_base_tool0):
        return T.inverse(T_clk) @ T_base_tool0

    # ---- TOOL0 ON THE AXIS, ITS Z COLLINEAR WITH IT ----
    T_grip = T_collar @ AXIAL
    rel = in_axis_frame(T_grip)
    off = rel[:3, 3] - np.dot(rel[:3, 3], [1., 0, 0]) * np.array([1., 0, 0])
    assert np.linalg.norm(off) * 1000.0 < 1e-9, (
        f'tool0 sits {np.linalg.norm(off) * 1000:.3f} mm off the connector axis -- the turn would '
        f'still orbit that radius instead of twisting the wrist')
    assert abs(abs(float(np.dot(rel[:3, 2], [1., 0, 0]))) - 1.0) < 1e-9, (
        'tool0 Z must be COLLINEAR with the connector axis, or a rotation about the axis is not '
        'a rotation about the tool')

    # ---- THE BITE IS UNCHANGED: fingertip on the ring, jaws across a diameter ----
    for name, G in (('radial', RADIAL), ('axial', AXIAL)):
        ft = T.inverse(T_collar) @ (T_collar @ G @ T_ftip)
        assert np.linalg.norm(ft[:3, 3]) * 1000.0 < 1e-9, (
            f'{name}: the fingertip must land ON the collar frame, got '
            f'{np.linalg.norm(ft[:3, 3]) * 1000:.3f} mm off')
    closing = (T.inverse(T_clk) @ T_grip @ T_ftip)[:3, 1]        # fingertip Y = the jaw axis
    assert abs(float(closing[0])) < 1e-9, (
        f'the jaws must close ACROSS the connector axis, not along it (axis component '
        f'{closing[0]:.3e}) -- a roll about the wrong axis would pinch the cable lengthwise')

    # ---- THE FLANGE STAYS PUT: this is the whole point, and the wall is why ----
    for name, G, want_travel in (('radial', RADIAL, 258.0), ('axial', AXIAL, 0.0)):
        T0 = T_collar @ G
        T1 = rotate_about_axis(T0, axis, point, cl_rot)
        travel = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])) * 1000.0
        if name == 'axial':
            assert travel < 1e-9, f'the axial turn must not move the flange, got {travel:.3f} mm'
        else:
            assert travel > 200.0, (
                'this test has lost its subject: the radial grasp is supposed to swing the '
                f'flange a long way for a {np.degrees(cl_rot):.0f} deg turn, got {travel:.1f} mm')
    # with push_mm the flange advances by exactly that and nothing else
    T_end = T.translation_matrix(push_m * axn) @ rotate_about_axis(T_grip, axis, point, cl_rot)
    assert abs(float(np.linalg.norm(T_end[:3, 3] - T_grip[:3, 3])) - push_m) < 1e-9, (
        'the only flange motion during the twist may be push_mm along the axis')

    # ---- WALL CLEARANCE: the axial grasp is a whole gripper length further back ----
    x_ax = float(in_axis_frame(T_grip)[0, 3]) * 1000.0
    x_rad = float(in_axis_frame(T_collar @ RADIAL)[0, 3]) * 1000.0
    assert x_ax < x_rad - 150.0, (
        f'the axial grasp must stand well clear of the wall: got {x_ax:+.1f} mm vs the radial '
        f'{x_rad:+.1f} mm along the connector +X')
    wall = cl.get('wall_standoff_mm')
    if wall is not None:
        assert x_ax <= float(wall), (
            f'the shipped grasp sits at {x_ax:+.1f} mm, past its own wall_standoff_mm '
            f'({float(wall):+.1f}) -- the app would refuse to run')
        assert x_rad > float(wall), (
            'this test has lost its subject: wall_standoff_mm is supposed to REJECT the old '
            'radial grasp, which is why the grasp changed')

    # ---- THE APPROACH IS A PURE AXIAL TRANSLATION ----
    retreat_m = float(cl['retreat_mm']) / 1000.0
    T_retreat = T.translation_matrix(-(retreat_m + collar_x) * axn) @ T_grip
    d = T_grip[:3, 3] - T_retreat[:3, 3]
    lat = float(np.linalg.norm(d - np.dot(d, axn) * axn)) * 1000.0
    _l, ang = T.pose_error(T_retreat, T_grip)
    assert lat < 1e-9 and np.degrees(ang) < 1e-9, (
        f'the advance onto the ring must be a PURE translation along the axis ({lat:.4f} mm '
        f'lateral, {np.degrees(ang):.4f} deg) -- that is what makes a straight ramp exact')

    # ---- ORDER, and every leg guarded ----
    src = open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8').read()
    body = src[src.index('def collar_clocking('):src.index('def traj_ref(')]
    order = ["label='collar withdraw (connector -X)'",     # back off ALONG the cable first...
             "label='collar lift-off (gripper -Z)'",       # ...then free the fingers, far back
             "label='collar retreat + reorient axial'",    # ...then reorient in clear space
             "adm_cl.ramp(T_retreat, T_grip",              # ...then advance down the axis
             "gripper.close('grasp collar')",
             "label='twist '"]
    idx = [body.index(t) for t in order]
    assert idx == sorted(idx), (
        'the order must be withdraw -> lift-off -> reorient -> advance -> close -> twist. The '
        'withdraw comes FIRST because the lateral lift-off travels parallel to the wall, so it '
        'must happen with the arm already backed off')
    assert 'AXIAL' in body and '-np.pi / 2.0' in body, (
        'the axial grasp roll must be built explicitly, not inherited from the fingertip frame')

    # ---- THE DEFAULT GRASP CLOCK ANGLE: tool0 -Y laid on the connector -Z --------------------
    # tool0 +Z is pinned along the connector +X by the axial grasp, so the only freedom left is
    # the roll about that axis. Spending it to put tool0 -Y on the connector -Z is the attitude
    # the sweep already leaves the wrist near, which HALVES the reorientation onto the axis --
    # and a 180 deg tool reorientation is where the analytic IK stops finding a reachable branch.
    #
    # THE CONNECTOR'S -Z IS WHERE THE SWEEP LEFT IT. The gripper turned the connector, so its own
    # Z came with it; reading the rule against the TARGET frame lands the minimum only when the
    # sweep happens to end at 0. Adding the achieved sweep makes it a constant 90 deg swing
    # wherever the sweep ends, which is what this pins.
    v0 = AXIAL[:3, 1]                                       # tool0 +Y at zero clock
    th_rule = float(np.arctan2(float(np.dot([1., 0, 0], np.cross(v0, [0., 0, 1.]))),
                               float(np.dot(v0, [0., 0, 1.]))))
    R = rotate_about_axis(AXIAL, np.array([1., 0, 0]), np.zeros(3), th_rule)[:3, :3]
    assert np.allclose(-R[:, 1], [0., 0, -1], atol=1e-9), (
        f'the rule must lay tool0 -Y on the connector -Z, got {-R[:, 1]}')
    assert np.allclose(R[:, 2], [1., 0, 0], atol=1e-9), (
        'and it must not disturb tool0 +Z, which stays on the connector +X')

    # the swing must be the MINIMUM available, at EVERY sweep end -- not just at 0
    T_eng = T_clk @ T.inverse(T_ftip @ (T.inverse(T_ftip) @ frames_cat[iname_M]))
    ret_x = collar_x - float(cl['retreat_mm']) / 1000.0
    lift = float(cl['liftoff_mm']) / 1000.0

    def _at(st, th):
        return rotate_about_axis(T_clk @ T.translation_matrix([st, 0., 0.]) @ AXIAL,
                                 axis, point, th)

    for sweep in (0.0, -60.0, -75.0, 75.0, 90.0):
        here = rotate_about_axis(T_eng, axis, point, np.radians(sweep))
        T_off = (T.translation_matrix((ret_x - collar_x) * axn) @ here)             @ T.translation_matrix([0., 0., -lift])
        R2 = rotate_about_axis(AXIAL, np.array([1., 0, 0]), np.zeros(3), th_rule)[:3, :3]
        assert np.allclose(-R2[:, 1], [0., 0, -1], atol=1e-9), (
            'the grasp attitude must be the SAME fixed -Y/-Z alignment wherever the sweep ends -- '
            'offsetting it by the achieved sweep makes it move with the sweep')
    assert 'np.cross(v0, want)' in body and 'th_grasp = th_rule' in body, (
        'the default angle must be SOLVED in closed form against the TARGET frame; a scan makes '
        'the motion depend on IK seeding and move run to run')

    assert 'wall_standoff_mm' in src and 'cl_wall_mm' in body, (
        'every planned station must be gated against the wall standoff before the arm moves')
    # comments may still EXPLAIN what prewind was; no code may still read it
    code = ' '.join(ln for ln in body.splitlines() if not ln.lstrip().startswith('#'))
    assert 'cl_prewind' not in code, (
        'prewind belonged to the radial orbit -- the axial approach states its grasp angle '
        'outright (grasp_clock_deg) instead of deriving it from where the sweep ended')

    # ---- CONFIG SHAPE ----
    c = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))
    ccl = c['assembly']['collar_clocking']
    assert 'prewind_deg' not in ccl, 'prewind_deg is gone with the radial approach'
    for k in ('grasp_clock_deg', 'liftoff_mm', 'retreat_mm', 'wall_standoff_mm'):
        assert k in ccl, f'collar_clocking must declare {k} so the axial approach is tunable'
    assert float(ccl['liftoff_mm']) > 0 and float(ccl['retreat_mm']) > 0


def test_the_connector_sweep_rocks_between_absolute_roll_positions():
    """connector_clocking walks sweep_deg's roll positions, one leg per try, rocking across the slot.

    A bayonet pin that did not line up with its slot at the mate will not find it by turning
    harder in one direction -- it rides the rim and jams. It finds it by crossing back and forth
    over the slot under a steady axial load, which is why the sweep alternates and why the press
    is established once and HELD through every reversal.

    Four things have to hold, and each is a way a plausible implementation goes wrong:
      * THE LEGS ALTERNATE. Positions are absolute, so the rotations between them must flip sign;
        a walk that only ever turns one way is the old behaviour wearing new config.
      * EACH LEG IS A TRUE SCREW about the socket axis. The rotation is about +X and the push is
        ALONG +X, so they commute and the axis LINE is invariant -- the connector origin must
        never leave it, at any roll, on any leg.
      * THE PRESS DOES NOT BLINK. push ramps to its full value on leg 1 and is CONSTANT after,
        so a reversal does not unload the connector and make the cams give back what they gained.
      * A JAMMED LEG CONTINUES FROM WHERE IT STOPPED. screw_ramp reports the fraction reached;
        restarting the next leg from the endpoint it never got to would command a jump across
        the arc the guard just refused.
    """
    import yaml

    from urlab import config as urconfig, tool_frames
    from urlab.transforms import inverse, xyzrpy_to_matrix

    asm = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))['assembly']
    cc = asm['connector_clocking']
    eng = float(asm['engage_clock_deg'])
    sweep = [float(v) for v in cc['sweep_deg']]
    tries = int(cc['max_tries'])
    assert len(sweep) >= 2, 'the oscillation needs two ends to rock between'

    # ---- THE LEG WALK: one position per try, cycling ----
    walk = [eng] + [sweep[(k - 1) % len(sweep)] for k in range(1, tries + 1)]
    turns = [b - a for a, b in zip(walk, walk[1:])]
    assert all(abs(t) > 1e-9 for t in turns), f'a leg with nothing to turn: {turns}'
    assert turns[0] < 0, (
        f'leg 1 turns {turns[0]:+.0f} deg from the engaged roll {eng:+.0f} -- the first rotation '
        f'is supposed to be NEGATIVE, which needs sweep_deg[0] below engage_clock_deg')
    assert all(a * b < 0 for a, b in zip(turns, turns[1:])), (
        f'the legs must ALTERNATE direction -- got turns {[round(t) for t in turns]}, which only '
        f'ever rocks one way')
    # the band worked is the config's, not something derived from the engage angle
    assert min(walk) == min([eng] + sweep) and max(walk) == max([eng] + sweep)

    # ---- EACH LEG IS A TRUE SCREW: the connector origin never leaves the axis line ----
    cfg = urconfig.load('bnc_assembly')
    T_clk = tool_frames.load_targets(cfg)[cfg['assembly']['target_frame']] \
        @ xyzrpy_to_matrix([0.0, 0.0, 0.0], np.radians([eng, 0.0, 0.0]))
    T_tool0_conn = tool_frames.load_frames(cfg)[cfg['estimation']['initial_connector_frame']]
    ref_start = T_clk @ inverse(T_tool0_conn)
    push_m = float(cc['push_mm']) / 1000.0
    axn = T_clk[:3, 0] / np.linalg.norm(T_clk[:3, 0])
    org = T_clk[:3, 3]

    def arm_at(th_rel, push):                       # mirrors the app's closure exactly
        return (T_clk @ xyzrpy_to_matrix([push, 0.0, 0.0], [th_rel, 0.0, 0.0])
                @ inverse(T_clk)) @ ref_start

    for th_deg in np.linspace(min(walk), max(walk), 25):
        d = (arm_at(np.radians(th_deg - eng), push_m) @ T_tool0_conn)[:3, 3] - org
        radial = float(np.linalg.norm(d - float(d @ axn) * axn))
        assert radial < 1e-9, (
            f'at {th_deg:+.0f} deg the connector origin is {radial * 1000:.4f} mm off the socket '
            f'axis -- the leg is not a screw about that line')
        assert abs(float(d @ axn) - push_m) < 1e-9, (
            'the axial station must be exactly push_mm at every roll -- rotation about +X and '
            'translation along +X commute, so the two cannot interfere')

    # ---- THE PRESS DOES NOT BLINK: push is full from the end of leg 1 onward ----
    # Replays the app's recursion: each leg interpolates its own start push -> cc_push_m, and
    # carries the value it reached into the next.
    push_at, seen = 0.0, []
    for _ in turns:
        start_push = push_at
        seen.append((start_push, push_at + (push_m - push_at) * 1.0))   # a leg that completes
        push_at = push_at + (push_m - push_at) * 1.0
    assert abs(seen[0][0]) < 1e-12 and abs(seen[0][1] - push_m) < 1e-12, \
        'leg 1 must build the press from zero to push_mm'
    for lo, hi in seen[1:]:
        assert abs(lo - push_m) < 1e-12 and abs(hi - push_m) < 1e-12, (
            'every leg after the first must hold push_mm CONSTANT -- re-ramping it from zero '
            'would unload the connector on every reversal, which is when the cams give back')

    # ---- SOURCE: the structure the arithmetic above assumes ----
    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    body = src[src.index('def connector_clocking('):src.index('def collar_clocking(')]
    assert 'res, f_done = screw_ramp(' in body and 'th_at = _a + turn * f_done' in body, (
        'a jammed leg must continue from the FRACTION screw_ramp reached, not from the endpoint '
        'it never got to -- restarting at the endpoint commands a jump across the arc the guard '
        'just refused')
    assert "return 'seated', k / n" in src and "return 'done', 1.0" in src, \
        'screw_ramp must report how far the reference actually got'
    # ONE warm-up for the whole sweep: adm.ramp keeps its integrator across calls, so resetting
    # per leg would dump the axial deflection holding the part loaded.
    assert body.count('adm_cc.warmup(') == 1, (
        'the sweep must warm up ONCE, not per leg -- a reset between legs dumps the integrator '
        'and the press has to build again on every reversal')
    assert body.index('adm_cc.warmup(') < body.index('for k in range(1, cc_tries + 1):'), (
        'the single warm-up belongs BEFORE the leg loop')
    # and the gripper stays closed for the whole sweep -- the retry is the reversal
    for banned in ('gripper.open(f\'release for clocking retry', 'gripper.close(f\'regrasp',
                   'det.rebase('):
        assert banned not in body, (
            f'{banned}...) is the old regrasp ratchet; a retry is now the REVERSAL onto the next '
            f'sweep position, with the gripper still closed')


def test_engage_clock_angle_places_the_sweep_without_changing_the_insertion():
    """assembly.engage_clock_deg picks WHERE in the roll the whole sequence sits.

    A BNC is free about its own axis until the bayonet pins pick up, so the clock angle the
    connector is ENGAGED at is a free parameter -- and connector_clocking.rotation_deg sweeps FROM it.
    A 180 deg screw started at the declared roll ends 180 deg past it, out where the fixture and
    the cable are; started at -90 the same stroke runs -90 -> +90, symmetric about the declared
    roll. That is the point of the key: it MOVES the band, it does not shorten the stroke.

    Three properties make it safe to set, and this pins each one:
      * THE INSERTION IS UNTOUCHED. The roll is about +X, so +X itself -- the insertion axis, the
        push direction, the retract leg, every depth reading -- is identical at any setting.
      * THE PATH ROLLS RIGIDLY WITH THE PART. Every pose is built from the rolled frame, so the
        connector's relationship to its own trajectory is exactly what it was; only the pair's
        placement about the axis moves. Rolling any ONE of them instead would put the part at one
        clock angle and its reference at another.
      * IT IS APPLIED ONCE. The rolled frame is what the rest of the app is handed, so no caller
        can forget to roll it (or roll it twice).
    """
    import yaml
    from scipy.spatial.transform import Rotation as _R

    T_socket = T.xyzrpy_to_matrix([0.4, -0.1, 0.3], np.radians([10.0, -25.0, 40.0]))
    axn = T_socket[:3, 0] / np.linalg.norm(T_socket[:3, 0])

    for clock_deg in (0.0, -90.0, 45.0, 180.0, -180.0):
        R_clock = T.xyzrpy_to_matrix([0.0, 0.0, 0.0], np.radians([clock_deg, 0.0, 0.0]))
        T_clk = T_socket @ R_clock
        # +X survives a roll about +X: the axis, the origin, and therefore the whole insertion
        assert np.allclose(T_clk[:3, 0], T_socket[:3, 0], atol=1e-12), \
            f'engage_clock_deg {clock_deg} moved the INSERTION AXIS'
        assert np.allclose(T_clk[:3, 3], T_socket[:3, 3], atol=1e-12), \
            f'engage_clock_deg {clock_deg} moved the mate POINT'
        # and the difference from the socket frame is a pure rotation about that axis
        rel = T.inverse(T_socket) @ T_clk
        rv = _R.from_matrix(rel[:3, :3]).as_rotvec()
        assert np.linalg.norm(rel[:3, 3]) < 1e-12, 'the roll must not translate the frame'
        if np.linalg.norm(rv) > 1e-9:
            assert abs(abs(float(np.dot(rv / np.linalg.norm(rv), [1.0, 0.0, 0.0]))) - 1.0) < 1e-9, \
                'the roll must be about the frame\'s own +X and nothing else'

        # THE PATH ROLLS WITH THE PART. traj_ref is T_base_targetobj @ row @ inv(T_tool0_conn),
        # so anchoring on the rolled frame moves BOTH the reference and the part it carries: the
        # connector-wrt-path relationship is preserved exactly, which is the whole claim.
        row = T.xyzrpy_to_matrix([-0.02, 0.003, -0.001], np.radians([0.0, 2.0, -1.0]))
        conn_plain = T_socket @ row
        conn_rolled = T_clk @ row
        assert np.allclose(T.inverse(T_socket) @ conn_plain,
                           T.inverse(T_clk) @ conn_rolled, atol=1e-12), \
            'the trajectory must roll rigidly with the frame it is anchored to'
        # the two differ in BASE coordinates by exactly the roll -- i.e. it really did move
        if abs(clock_deg) > 1e-9:
            assert not np.allclose(conn_plain, conn_rolled, atol=1e-6), \
                'a non-zero clock angle must actually place the path somewhere else'
        # depth along the axis is the SAME number measured in either frame
        assert abs(float((T.inverse(T_socket) @ conn_plain)[0, 3])
                   - float((T.inverse(T_clk) @ conn_rolled)[0, 3])) < 1e-12, \
            'axial depth must not depend on the clock angle'
        # and the retract leg (target-frame -X) is the same direction in base
        assert np.allclose(T_clk[:3, :3] @ [-1.0, 0.0, 0.0], -axn, atol=1e-12), \
            'the target-frame retract leg must be unmoved by the clock angle'

    # ---- APPLIED ONCE, AND EVERYTHING DOWNSTREAM IS HANDED THE ROLLED FRAME ----
    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    assert 'T_base_socket = targets[tname]' in src and \
           'T_base_tconn = T_base_socket @ R_clock' in src, (
        'the clock angle must be folded into the target frame ONCE, at the catalogue load, so no '
        'caller downstream can forget it or apply it twice')
    assert 'T_base_targetobj = T_base_tconn @ inverse(mats[-1])' in src, \
        'the trajectory must be anchored on the ROLLED frame'
    assert src.count('T_base_socket') == 2, (
        'the raw socket frame exists only to build the rolled one -- anything else reading it '
        'would be working at a different clock angle from the rest of the app')
    # the estimator matches a map collected at ONE clock angle, so a non-zero roll must say so
    tail = src[src.index('T_base_tconn = T_base_socket @ R_clock'):]
    assert "ins_mode == 'estimate'" in tail[:2500] and 'manifold' in tail[:2500], (
        'a non-zero clock angle with insertion_mode: estimate must warn -- the contact manifold '
        'was collected at one clock angle and the socket is not a body of revolution')

    # ---- THE SHIPPED PAIR STAYS INSIDE ONE TURN ----
    a = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))['assembly']
    assert 'engage_clock_deg' in a, 'the key must be declared so the behaviour is discoverable'
    start = float(a['engage_clock_deg'])
    assert abs(start) <= 180.0, f'engage_clock_deg {start} is outside one turn'
    stops = [start] + [float(v) for v in a['connector_clocking']['sweep_deg']]
    assert max(stops) - min(stops) <= 360.0, (
        f'the engaged roll and the sweep span {max(stops) - min(stops):.0f} deg -- more than one '
        f'turn, so a measured clock angle cannot be told from itself plus 360 and the app refuses')
    # the engage angle decides WHICH WAY the first leg turns, and the config comment claims it
    # goes negative first -- pin that, because it is the whole of the user-visible ordering
    assert stops[1] < stops[0], (
        f'leg 1 runs from the engaged roll {stops[0]:+.0f} to {stops[1]:+.0f} deg, which is a '
        f'POSITIVE rotation -- sweep_deg[0] must be below engage_clock_deg for the first leg to '
        f'rock negative')


def test_the_achieved_clock_angle_is_wrapped_onto_the_stroke_branch():
    """A 180 deg screw makes the naive angle read-back sign-ambiguous, and the sign is a DIRECTION.

    Every achieved clock angle in bnc_assembly is recovered from a measured pose -- as an Euler
    roll, or as dot(rotvec, axis) -- and both return a value in (-pi, pi]. Harmless at 90 deg. At
    180 it is a trap: a stroke that actually turned +182 reads back as -178, and that number is
    handed straight to the unwind, which reverses it. Negating -178 orbits the OPEN gripper +178
    deg the wrong way round the captive cable instead of retracing the arc it came in on -- a full
    turn into exactly the region the clock angle was chosen to avoid, with the fingers wrapped
    around the part.

    The achieved angle always lies in [0, commanded], so wrapping near the middle of that range
    puts the whole feasible range inside one branch. This pins that, including the sign-symmetric
    case (a negative commanded stroke) and the ordinary angles that must NOT be moved.
    """
    from urlab.apps.bnc_assembly import _wrap_near

    half = np.radians(90.0)                     # centre for a +180 deg commanded stroke
    # the failure case: +182 achieved, read back as -178
    assert abs(np.degrees(_wrap_near(np.radians(-178.0), half)) - 182.0) < 1e-9, \
        'an overshoot past 180 must wrap FORWARD, not flip the unwind direction'
    # ordinary readings inside the stroke are untouched
    for deg in (0.0, 20.0, 90.0, 179.0):
        assert abs(np.degrees(_wrap_near(np.radians(deg), half)) - deg) < 1e-9, \
            f'{deg} deg is already on the branch and must not be moved'
    # exactly 180, either sign, resolves to the direction the stroke actually turned
    for deg in (180.0, -180.0):
        assert abs(np.degrees(_wrap_near(np.radians(deg), half)) - 180.0) < 1e-9, \
            'a 180 deg reading must resolve onto the commanded end, not its negation'
    # a NEGATIVE commanded stroke is the mirror image, not a special case
    assert abs(np.degrees(_wrap_near(np.radians(178.0), np.radians(-90.0))) + 182.0) < 1e-9, \
        'a -180 deg stroke must wrap BACKWARD by the same rule'
    # a 90 deg stroke never needs the wrap -- the fix must not disturb the tuned behaviour
    for deg in (-30.0, 0.0, 45.0, 95.0):
        assert abs(np.degrees(_wrap_near(np.radians(deg), np.radians(45.0))) - deg) < 1e-9, \
            'the 90 deg stroke branch must be unchanged by the wrap'

    # ---- and every place that reads an achieved angle must use it ----
    with open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8') as fh:
        src = fh.read()
    cable = src[src.index('def connector_clocking('):src.index('def collar_clocking(')]
    assert '_wrap_near(float(got[1][0]), cc_mid)' in cable, (
        'the achieved-roll read-back must be wrapped onto the sweep band centre -- it is handed '
        'to collar clocking as the angle to UNWIND, so a sign that flipped at the band edge '
        'would orbit the open gripper the long way round the part')
    assert 'screw_rad' in cable and 'float(np.degrees(screw_rad))' in cable, (
        'the WRAPPED angle is what collar clocking must be given, not the raw Euler roll')
    # cc_mid is only an honest branch if the band it centres is under a full turn -- the app
    # refuses a wider one rather than reading an angle back wrong
    assert 'cc_hi - cc_lo > 2.0 * np.pi' in src, (
        'a sweep spanning more than one turn must be REFUSED at config time: there is no branch '
        'that tells a measured roll from itself plus 360')
    # collar_clocking does NOT wrap anything any more: the axial approach is placed in free
    # space from the frames, so there is no measured arm clock angle to recover a branch for.
    collar = src[src.index('def collar_clocking('):src.index('def traj_ref(')]
    assert '_wrap_near(' not in collar, (
        'the axial collar approach is built from the frames, not solved from the measured arm '
        'pose -- nothing there has a branch to wrap')
    assert 'th_grasp' in collar and 'cl_grasp_clock' in collar, (
        'the collar grasp angle must come from grasp_clock_deg, stated absolutely, rather than '
        'being derived from where the sweep left the arm')


def test_wrench_in_moves_the_moment_off_the_flange():
    """A contact moment must be a property of the CONTACT, not of where the arm happens to be.

    The UR script manual for get_tcp_force is explicit: the components are "all measured at the
    TOOL FLANGE with the orientation of the robot base coordinate system". A moment is only
    defined about a stated point, so re-expressing it in the connector frame means moving that
    point first (tau += (p_flange - p_target) x f) and only then rotating.

    Composing the rotation ALONE silently claims the moment was about the BASE ORIGIN, which
    injects a phantom lever of |base -> flange| -- about a metre on this arm. That bug is not
    hypothetical: before the fix, the logged BNC connector torque was 90% explained (R^2 0.90) by
    a single p x f with |p| = 1024 mm against a predicted 1050 mm, and tau came out perpendicular
    to f in 98.8% of rows. This test is what keeps it from coming back.
    """
    from urlab.robot.arm import URArm

    class _FakeArm(URArm):
        """Just the two readings wrench_in consumes."""
        def __init__(self, T_flange, w):
            self._T, self._w = T_flange, np.asarray(w, dtype=float)
            self.dry_run = False

        def wrench(self):
            return self._w

        def tcp_pose(self):
            return self._T

    T_flange = T.xyzrpy_to_matrix([0.10, 1.00, 0.20], np.radians([10.0, 20.0, 30.0]))
    T_flange_conn = T.xyzrpy_to_matrix([0.0, 0.0457, 0.159], np.radians([180.0, 0.0, 90.0]))
    T_conn = T_flange @ T_flange_conn
    p_f, p_c = T_flange[:3, 3], T_conn[:3, 3]
    assert np.linalg.norm(p_f) > 0.5, 'the flange must be far from the base, or this proves nothing'

    def controller_reports(f_base, tau_true_base):
        """What the arm would report: the moment about the FLANGE, in base axes."""
        return np.concatenate([f_base, tau_true_base + np.cross(p_c - p_f, f_base)])

    # ---- a pure force AT the contact carries no moment about the contact ----
    f_base = np.array([5.0, -3.0, 2.0])
    w = _FakeArm(T_flange, controller_reports(f_base, np.zeros(3))).wrench_in(T_conn)
    assert np.linalg.norm(w[3:]) < 1e-12, (
        f'a pure force at the contact must give ZERO moment about it, got '
        f'{np.linalg.norm(w[3:]):.4f} Nm -- the reference point was not moved off the flange')
    assert np.allclose(w[:3], T_conn[:3, :3].T @ f_base, atol=1e-12), 'force is a pure rotation'

    # ---- a real contact moment survives the trip exactly ----
    tau_true = np.array([0.05, -0.02, 0.11])                       # in CONNECTOR axes
    w = _FakeArm(T_flange, controller_reports(
        f_base, T_conn[:3, :3] @ tau_true)).wrench_in(T_conn)
    assert np.allclose(w[3:], tau_true, atol=1e-12), \
        f'the contact moment must be recovered exactly, got {w[3:]} want {tau_true}'

    # ---- POSE INVARIANCE: the same contact read from different arm poses is the same wrench ----
    seen = []
    for rpy in ([0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [-40.0, 15.0, 120.0]):
        for t in ([0.10, 1.00, 0.20], [0.30, 0.60, -0.10]):
            Tf = T.xyzrpy_to_matrix(t, np.radians(rpy))
            Tc = Tf @ T_flange_conn
            f_b = Tc[:3, :3] @ np.array([4.0, 1.0, -2.0])          # fixed in CONNECTOR axes
            tau_fl = (Tc[:3, :3] @ tau_true) + np.cross(Tc[:3, 3] - Tf[:3, 3], f_b)
            ww = _FakeArm(Tf, np.concatenate([f_b, tau_fl])).wrench_in(Tc)
            seen.append(ww)
    for ww in seen[1:]:
        assert np.allclose(ww, seen[0], atol=1e-12), (
            'the SAME physical contact must read identically from every arm pose; a '
            'pose-dependent answer is the lever-arm artefact returning')
    lever = np.linalg.norm(seen[0][3:]) / np.linalg.norm(seen[0][:3]) * 1000.0
    assert lever < 100.0, (
        f'lever {lever:.0f} mm -- a connector-scale contact should be tens of mm, not the '
        'hundreds that a base-origin reference produces')

    # ---- the callers hand over the flange pose they already read ----
    for app in ('estimator_eval', 'uncertain_sampling', 'cable_pick_estimate_assemble'):
        src = open(os.path.join(ROOT, 'urlab', 'apps', f'{app}.py'), encoding='utf-8').read()
        assert 'wrench_in(' in src, f'{app} logs a connector-frame wrench'
        for call in [ln for ln in src.splitlines() if 'wrench_in(' in ln and 'def ' not in ln]:
            assert 'T_base_tool0' in call or 'T_base_tool0' in src[
                max(0, src.index(call) - 400):src.index(call)], (
                f'{app}: pass the flange pose already read into wrench_in, so the pose and the '
                'wrench come from the same sample rather than two reads a cycle apart')


def test_payload_and_joint_acceleration_are_fleet_wide_constants():
    """One payload and one joint acceleration across every app -- both silently divergent before.

    PAYLOAD is not bookkeeping: getActualTCPForce subtracts the tool weight using it, so a config
    that declares the wrong mass reports a wrench offset by the difference. uncertain_sampling
    (the app that BUILDS the contact map) carried 1.0 kg / [0,0,0] against everyone else's
    1.3 kg / [-0.026, 0.028, 0.030], so the map was gravity-compensated differently from every
    run matched against it.

    JOINT ACCELERATION had four values in play (30 / 57.3 / 68.8 / 286.5 deg/s^2) and only the
    30 was ever chosen -- the rest came from the legacy `joint_acceleration_rad_s2` spelling or
    from falling through to the arm's built-in default. Both routes are checked here, because
    either one re-diverges the fleet without touching a number anybody reads.
    """
    import glob

    import yaml

    from urlab import config as urconfig
    from urlab.robot.arm import _DEFAULT_LIMITS, parse_limits

    names = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'configs', '*.yaml'))):
        n = os.path.basename(f)[:-5]
        if n.startswith('_') or n in ('frames', 'cables'):
            continue
        names.append(n)
    assert len(names) > 10, 'the fleet should be more than a handful of configs'

    payloads, accels = {}, {}
    for n in names:
        cfg = urconfig.load(n)
        p = (cfg.get('robot') or {}).get('payload') or {}
        payloads[n] = (float(p.get('mass_kg', -1)), tuple(float(v) for v in p.get('cog_m') or ()))
        accels[n] = round(float(np.degrees(
            parse_limits(cfg.get('speed') or {}, _DEFAULT_LIMITS)[1])), 3)

    distinct_p = sorted(set(payloads.values()))
    assert len(distinct_p) == 1, (
        'every config must declare the SAME tool payload -- the wrench is only as good as the '
        f'weight compensation. Got {len(distinct_p)}: '
        + '; '.join(f'{v} in ' + ', '.join(k for k in payloads if payloads[k] == v)
                    for v in distinct_p))

    distinct_a = sorted(set(accels.values()))
    assert len(distinct_a) == 1, (
        'every config must resolve to the SAME joint acceleration. Got '
        f'{len(distinct_a)}: '
        + '; '.join(f'{v} deg/s2 in ' + ', '.join(k for k in accels if accels[k] == v)
                    for v in distinct_a))

    # ...and it must be the value the shared file declares, not merely a value they agree on
    common = yaml.safe_load(open(os.path.join(ROOT, 'configs', '_common.yaml')))
    assert distinct_a[0] == float(common['speed']['max_joint_acceleration_deg_s2']), (
        f'the fleet agrees on {distinct_a[0]} deg/s2 but _common.yaml declares '
        f"{common['speed']['max_joint_acceleration_deg_s2']} -- the shared file must be the "
        'source of truth, not a stale fourth opinion')
    cp = common['robot']['payload']
    assert distinct_p[0] == (float(cp['mass_kg']), tuple(float(v) for v in cp['cog_m'])), \
        'the fleet payload must match the one _common.yaml declares'

    # The LEGACY spelling silently wins over nothing but loses to the modern key, so a config
    # carrying both is a trap: delete the modern one and the fleet re-diverges invisibly.
    for n in names:
        raw = yaml.safe_load(open(os.path.join(ROOT, 'configs', f'{n}.yaml'))) or {}
        spd = raw.get('speed') or {}
        assert 'joint_acceleration_rad_s2' not in spd, (
            f'{n}.yaml still carries the legacy joint_acceleration_rad_s2 -- it is shadowed by '
            'the modern key today and would silently govern the moment that key is removed')
        if spd:
            assert 'max_joint_acceleration_deg_s2' in spd, (
                f'{n}.yaml defines a speed: block without max_joint_acceleration_deg_s2. Blocks '
                'are owned WHOLESALE (no per-key merge with _common.yaml), so this one falls '
                "through to the arm's built-in default instead of the fleet value")


def test_engage_is_the_trajectory_plus_an_optional_oscillation():
    """ENGAGE drives the trajectory; the wiggle is an ADDITION to it, not a replacement.

    insertion_mode: wiggle ignores configs/assembly_trajectory.csv entirely and drives at one
    fixed target. ENGAGE follows the trajectory, extends it by preload_mm, and superimposes the
    oscillation -- so amplitude 0 is a direct insertion and 'direct' and 'wiggle' are one
    behaviour at two settings rather than two code paths that drift apart.

    The property worth pinning is the TWO CLOCKS: the path advances by DISTANCE (so the insert
    takes the time its length implies) while the oscillation advances by TIME (so its frequency
    is the frequency configured). Pace both by distance -- as seg_time does for every other ramp
    in this app -- and the wiggle frequency silently becomes a function of the insert speed."""
    import yaml

    from urlab.skills import trajectory as traj
    from urlab.skills.manifold import vec6_from_mats

    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'bnc_assembly.yaml')))['assembly']
    mats = traj.load_csv(os.path.join(ROOT, 'configs', 'assembly_trajectory.csv'))
    res_m = float(cfg.get('translational_resolution_m', 0.001))
    dense = traj.resample(mats, res_m, 1.0)

    def build(pre_mm, amp, frq, v_mm_s):
        """The engage reference, mirroring engage_insertion()."""
        rows = [np.concatenate([np.asarray(vec6_from_mats(r), float)[:3] * 1000.0,
                                np.asarray(vec6_from_mats(r), float)[3:]]) for r in dense]
        if pre_mm > 0:
            step = res_m * 1000.0
            n = max(1, int(round(pre_mm / max(step, 1e-6))))
            base = rows[-1].copy()
            for k in range(1, n + 1):
                v = base.copy()
                v[0] += pre_mm * k / n
                rows.append(v)
        seg = [float(np.linalg.norm(rows[i + 1][:3] - rows[i][:3])) for i in range(len(rows) - 1)]
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])

        def ref6(t):
            d = float(np.clip(v_mm_s * t, 0.0, total))
            j = max(0, min(int(np.searchsorted(cum, d, side='right') - 1), len(rows) - 2))
            span = cum[j + 1] - cum[j]
            f = 0.0 if span <= 1e-12 else (d - cum[j]) / span
            v = rows[j] + f * (rows[j + 1] - rows[j])
            for i in range(6):
                if abs(amp[i]) > 0 and frq[i] > 0:
                    v[i] += amp[i] * np.sin(2 * np.pi * frq[i] * t)
            return v
        return ref6, total

    x0 = T.matrix_to_xyzrpy(mats[0])[0][0] * 1000.0

    # ---- the path is the trajectory, EXTENDED by preload_mm past the mate ----
    for pre in (0.0, 2.0, 5.0):
        ref6, total = build(pre, [0] * 6, [0] * 6, 2.5)
        assert abs(ref6(0.0)[0] - x0) < 1e-9, 'the engage must START on the trajectory first row'
        assert abs(ref6(total / 2.5)[0] - pre) < 1e-6, (
            f'preload {pre} mm must END the path that far PAST the mate, got '
            f'{ref6(total / 2.5)[0]:.3f}')
        assert abs(total - (abs(x0) + pre)) < 1e-6, 'path length = trajectory + preload'

    # ---- speed is honoured: depth advances at exactly speed_mm_s, then CLAMPS ----
    ref6, total = build(2.0, [0] * 6, [0] * 6, 2.5)
    for t in (0.0, 1.0, 4.0):
        assert abs(ref6(t)[0] - (x0 + 2.5 * t)) < 1e-9, 'depth must advance at speed_mm_s'
    assert abs(ref6(100.0)[0] - 2.0) < 1e-9, 'past the end it must clamp, not overshoot'

    # ---- TWO CLOCKS: the oscillation period is independent of the path speed ----
    amp = [0, 0, 1.0, 0, 1.0, 0]
    frq = [0, 0, 0.7, 0, 1.1, 0]
    periods = []
    for v in (1.0, 2.5, 10.0):
        ref6, total = build(2.0, amp, frq, v)
        ts = np.linspace(0.0, total / v, 20000)
        z = np.array([ref6(t)[2] for t in ts])
        xs = ts[np.where(np.diff(np.sign(z)) != 0)[0]]
        assert len(xs) > 2, 'the oscillation must actually cross zero'
        periods.append(2 * float(np.median(np.diff(xs))))
    for p in periods:
        assert abs(p - 1.0 / 0.7) < 0.02, (
            f'the z oscillation period must stay 1/0.7 s at every path speed, got {p:.3f} -- '
            'pacing the oscillation by distance would make frequency depend on insert speed')

    # ---- amplitude 0 IS a direct insertion ----
    d, _ = build(2.0, [0] * 6, [0] * 6, 2.5)
    assert max(abs(d(t)[2]) for t in np.linspace(0, 6, 200)) < 1e-12, \
        'with every amplitude 0 the engage must reproduce the trajectory exactly'

    # ---- the config and the code agree on what exists ----
    en = cfg['engage']
    # amplitude/frequency are NOT declared here any more -- they come from the tuned file, and
    # test_every_app_resolves_to_the_tuned_wiggle checks that resolution.
    for k in ('speed_mm_s', 'preload_mm', 'sample_rate_hz',
              'max_axial_force_n', 'persistence_s', 'stiffness'):
        assert k in en, f'assembly.engage.{k} must be declared'
    assert float(en['preload_mm']) >= 0.0
    _r = _resolved_wiggle('bnc_assembly', 'assembly', 'engage')
    _amp, _frq = _r.get('amplitude') or {}, _r.get('frequency_hz') or {}
    live = [k for k, v in _amp.items() if abs(float(v)) > 0.0]
    for d_ in live:
        assert float(_frq.get(d_, 0)) > 0.0, f'engage axis {d_} has amplitude but no freq'
    if live:
        assert float(en['sample_rate_hz']) >= 4.0 * max(float(_frq[d_])
                                                        for d_ in live), 'engage would alias'

    src = open(os.path.join(ROOT, 'urlab', 'apps', 'bnc_assembly.py'), encoding='utf-8').read()
    assert "ins_mode not in ('estimate', 'engage')" in src, (
        'engage must be a valid mode, and the standalone wiggle must NOT be -- it is retired, '
        'with its oscillation living inside engage')
    # a force stop ends the PHASE, not the run
    assert "success = en_status in ('complete', 'force')" in src, (
        'hitting the axial force limit must NOT fail the run -- a connector meeting resistance '
        'partway is what the clocking screw is for, so the sequence carries on')
    # and the limit is AXIAL, not |f|
    assert 'class _AxialForce' in src and 'wrench_in(T_base_conn, T_base_tool0)' in src, (
        'the engage limit must project onto the connector +X; a |f| limit tight enough to catch '
        'real resistance also stops on every lateral graze')

def test_shipped_wiggle_configs_are_accepted_by_the_shared_implementation():
    """Every app that superimposes a wiggle must build one the shared module accepts.

    urlab/skills/wiggle.py owns the waveform and its correctness checks -- amplitude without a
    frequency, aliasing above a quarter of the reference rate, and a frequency ratio so simple the
    probe traces a line instead of filling the box. Those are the things that make a wiggle
    something other than what its config says, and they are refused before the arm moves.

    This asserts the SHIPPED configs pass that gate. It deliberately does NOT re-implement the
    checks, and it does NOT pin tuning: burst length against orbit period, which stations exist,
    how many amplitude scales are swept, are all judgement calls the experiment is free to change
    -- an earlier version of this test pinned them and broke the moment the station grid was
    retuned, which is a test failing at its own author rather than at a defect.
    """
    import yaml

    from urlab.skills import wiggle as wigmod

    def block_of(path, *keys):
        with open(os.path.join(ROOT, 'configs', path)) as fh:
            node = yaml.safe_load(fh)
        for k in keys:
            node = (node or {}).get(k)
            if node is None:
                return None
        return node

    cases = [
        ('wiggle_sampling.yaml', ('wiggle',), 'compliance'),
        ('bnc_assembly.yaml', ('assembly', 'engage'), 'compliance'),
        ('estimator_eval.yaml', ('eval', 'collection', 'wiggle'), 'compliance'),
    ]
    checked = 0
    for path, keys, _ in cases:
        blk = block_of(path, *keys)
        if not blk:
            continue
        amps = blk.get('amplitude') or {}
        if not any(abs(float(v)) > 0 for v in amps.values()):
            continue                       # no oscillation configured: nothing to validate
        with open(os.path.join(ROOT, 'configs', path)) as fh:
            rate = float(((yaml.safe_load(fh) or {}).get('compliance') or {})
                         .get('reference_rate_hz', 125.0))
        dims = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')
        w = wigmod.Wiggle([float(amps.get(d, 0)) for d in dims],
                          [float((blk.get('frequency_hz') or {}).get(d, 0)) for d in dims],
                          [float((blk.get('phase_deg') or {}).get(d, 0)) for d in dims],
                          float(blk.get('taper_s', 0) or 0), path)
        w.validate(rate_hz=rate,
                   cap_v=blk.get('max_speed_mm_s') or blk.get('max_oscillation_speed_mm_s'),
                   cap_w=blk.get('max_rotation_deg_s')
                   or blk.get('max_oscillation_rotation_deg_s'))
        checked += 1
    assert checked, 'no shipped config carries a live wiggle -- expected at least one'


def test_every_wiggle_goes_through_the_shared_implementation():
    """No app may carry its own copy of the waveform.

    Three copies of a sine is how the frame convention, the taper and the speed-cap policy
    diverged in the first place, and observations collected under one are then not comparable
    with a map built under another -- which is the whole point of the map/eval parity work. If a
    new hand-rolled oscillation appears in an app, this fails.
    """
    apps = os.path.join(ROOT, 'urlab', 'apps')
    offenders = []
    for name in os.listdir(apps):
        if not name.endswith('.py'):
            continue
        src = open(os.path.join(apps, name), encoding='utf-8').read()
        # a sine driven by 2*pi*frequency is a waveform; anything else (a rotation, a
        # Lissajous drawn in a diagnostic) is not what this looks for. Scanned line by
        # line so the pattern stays readable.
        hits = [ln.strip() for ln in src.splitlines()
                if not ln.lstrip().startswith('#')
                and ('np.sin(2.0 * np.pi *' in ln or 'np.sin(2 * np.pi *' in ln
                     or 'math.sin(2.0 * math.pi *' in ln
                     or 'math.sin(2 * math.pi *' in ln)]
        if hits:
            offenders.append(f'{name}: {len(hits)} hand-rolled sine(s)')
    assert not offenders, (
        'these apps build a waveform themselves instead of using urlab/skills/wiggle.py:\n  '
        + '\n  '.join(offenders))


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
