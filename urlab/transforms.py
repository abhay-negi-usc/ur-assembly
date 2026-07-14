"""Rigid-transform math. Pure numpy + scipy -- no ROS, no robot, no camera.

Every pose in urlab is a 4x4 homogeneous matrix `T_a_b` ("b expressed in a"), so composition
reads left to right:  T_a_c = T_a_b @ T_b_c.

CONVENTIONS -- get these wrong and everything else is silently wrong:

  * rpy is EXTRINSIC XYZ ("sxyz" in tf_transformations, lowercase 'xyz' in scipy):
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    This is what the ROS stack used, so every yaml angle in configs/ carries over untouched.
    CAD tools usually report INTRINSIC XYZ instead -- numerically different unless two of the
    three angles are zero.

  * quaternions are [x, y, z, w] (scipy's order, and the old tf_transformations' order).

  * RTDE speaks a different dialect on BOTH counts, so the two conversions below are the only
    places where robot poses cross into urlab:
      - rotation as an axis-angle ROTATION VECTOR, not rpy   -> rtde_to_matrix / matrix_to_rtde
      - translations in the UR `base` frame, not ROS `base_link`, which differ by a 180 deg
        turn about Z                                          -> BASE_LINK_FROM_UR_BASE
    Keeping both conversions here (and nowhere else) is what lets the rest of the library --
    and every config value measured in base_link under the ROS stack -- stay unchanged.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# UR joint order. Same order the pendant, the old ROS /joint_states, and RTDE's getActualQ()
# all use -- so a joint vector never needs reordering anywhere in this library.
UR_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]


# ------------------------------------------------------------------ construction
def xyzrpy_to_matrix(xyz, rpy):
    """4x4 from a translation and an EXTRINSIC-XYZ rpy triple (radians)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', np.asarray(rpy, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def matrix_to_xyzrpy(T):
    """(xyz, rpy) from a 4x4. Inverse of xyzrpy_to_matrix."""
    return (np.array(T[:3, 3], dtype=float),
            Rotation.from_matrix(T[:3, :3]).as_euler('xyz'))


def xyzquat_to_matrix(xyz, quat):
    """4x4 from a translation and an [x, y, z, w] quaternion."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def matrix_to_quat(T):
    """[x, y, z, w] quaternion from a 4x4."""
    return Rotation.from_matrix(T[:3, :3]).as_quat()


def translation_matrix(vec):
    """4x4 pure translation."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(vec, dtype=float)
    return T


def from_cfg(d, default_xyz=(0.0, 0.0, 0.0), default_rpy=(0.0, 0.0, 0.0)):
    """4x4 from a config dict {xyz: [...], rpy: [...]}. Missing keys fall back to the defaults,
    so an absent section yields identity rather than a KeyError."""
    d = d or {}
    return xyzrpy_to_matrix(d.get('xyz', default_xyz), d.get('rpy', default_rpy))


def inverse(T):
    """Inverse of a RIGID 4x4 -- transpose the rotation, negate the rotated translation.

    Exact and ~10x faster than np.linalg.inv, but it is only correct for rigid transforms (it
    assumes R is orthonormal). Everything in urlab is rigid; if you ever feed this a scaled or
    sheared matrix it will return nonsense without complaining."""
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


# ------------------------------------------------------------------ RTDE interop
# ROS `base_link` -> UR `base`: a 180 deg turn about Z. RTDE reports and accepts poses in the UR
# `base` frame; every config value in this repo was measured in ROS `base_link` under the old
# stack. This constant is the whole bridge -- it is its own inverse (Rz(pi) squared = I).
BASE_LINK_FROM_UR_BASE = xyzrpy_to_matrix([0.0, 0.0, 0.0], [0.0, 0.0, np.pi])


def rtde_to_matrix(pose, ur_base=False):
    """RTDE pose [x, y, z, rx, ry, rz] (axis-angle rotvec, UR `base` frame) -> 4x4.

    Returns the pose in ROS `base_link` by default -- the frame every config in this repo is
    measured in. Pass ur_base=True to get the raw UR-base pose instead."""
    p = np.asarray(pose, dtype=float)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[3:6]).as_matrix()
    T[:3, 3] = p[0:3]
    return T if ur_base else BASE_LINK_FROM_UR_BASE @ T


def matrix_to_rtde(T, ur_base=False):
    """4x4 (in ROS `base_link` unless ur_base) -> RTDE pose [x, y, z, rx, ry, rz] in UR `base`."""
    T = T if ur_base else BASE_LINK_FROM_UR_BASE @ T   # Rz(pi) is its own inverse
    return list(T[:3, 3]) + list(Rotation.from_matrix(T[:3, :3]).as_rotvec())


# ------------------------------------------------------------------ pose comparison
def pose_error(T_cur, T_des):
    """(linear error [m], angular error [rad]) between two poses.

    The angle is the geodesic 2*arccos(|q_cur . q_des|); the abs() picks the shortest path
    across the quaternion double cover, so it never reports ~2*pi for a small rotation."""
    lin = float(np.linalg.norm(T_des[:3, 3] - T_cur[:3, 3]))
    q_cur, q_des = matrix_to_quat(T_cur), matrix_to_quat(T_des)
    dot = min(1.0, abs(float(np.dot(q_cur, q_des))))
    return lin, float(2.0 * np.arccos(dot))


def slerp_matrix(T0, T1, alpha):
    """Interpolate between two poses: LERP the translation, SLERP the rotation."""
    key = Rotation.from_quat([matrix_to_quat(T0), matrix_to_quat(T1)])
    T = np.eye(4)
    T[:3, :3] = Slerp([0.0, 1.0], key)(float(alpha)).as_matrix()
    T[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    return T


def step_toward(T_cur, T_des, gain, max_lin, max_ang):
    """One clamped proportional step from T_cur toward T_des -- the servo/PBVS inner step.

    Translation and rotation are clamped INDEPENDENTLY (translate by at most max_lin, rotate by
    at most max_ang), so a target that is far in one and close in the other still converges."""
    p_cur, p_des = T_cur[:3, 3], T_des[:3, 3]
    step = (p_des - p_cur) * gain
    n = float(np.linalg.norm(step))
    if n > max_lin:
        step = step / n * max_lin

    _, ang = pose_error(T_cur, T_des)
    frac = 0.0 if ang < 1e-6 else min(gain, max_ang / ang)
    T = slerp_matrix(T_cur, T_des, frac)
    T[:3, 3] = p_cur + step
    return T


# ------------------------------------------------------------------ camera geometry
def look_at(eye, target, T_ref):
    """Camera pose (4x4) at `eye` whose optical axis points at `target`.

    OPTICAL convention -- x right, y down, z FORWARD along the view ray. (This is the convention
    of camera1_color_optical_frame and of every pixel-space computation in perception/; a robot
    z-up frame here would silently mirror every detection.)

    T_ref only breaks the roll degree of freedom -- the spin about the view ray, which looking at
    a point cannot determine. Its y axis is used as the 'down' hint; passing the camera's own
    current pose therefore keeps roll as close to unchanged as the new view direction allows."""
    eye = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - eye
    n = float(np.linalg.norm(z))
    if n < 1e-9:
        return np.array(T_ref, dtype=float)
    z /= n

    x = np.cross(T_ref[:3, 1], z)              # right = down_hint x forward
    if float(np.linalg.norm(x)) < 1e-6:        # view ray parallel to the hint: pick any x _|_ z
        x = np.cross([0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= float(np.linalg.norm(x))
    y = np.cross(z, x)                         # down = forward x right

    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = eye
    return T


def frame_from_axis(axis, up):
    """Frame (3x3) whose x IS `axis` and whose z is as close to `up` as orthogonality allows.

    Used to turn a measured connector axis into a full pose. The roll about the axis is NOT
    observable from the multi-view fit -- `up` is what pins it, and it is a choice, not a
    measurement."""
    x = np.asarray(axis, dtype=float)
    x = x / (float(np.linalg.norm(x)) + 1e-12)

    y = np.cross(np.asarray(up, dtype=float), x)
    if float(np.linalg.norm(y)) < 1e-6:        # axis parallel to up: any perpendicular will do
        alt = [1.0, 0.0, 0.0] if abs(x[0]) < 0.9 else [0.0, 1.0, 0.0]
        y = np.cross(alt, x)
    y /= float(np.linalg.norm(y))

    z = np.cross(x, y)
    z /= float(np.linalg.norm(z))
    return np.column_stack([x, y, z])


def rotate_about_axis(T, axis, point, angle):
    """Rigidly orbit the pose T about the line (point, axis) by `angle` radians.

    Rotates the pose's ORIENTATION as well as its position, so a camera orbiting a cable keeps
    pointing the same way relative to it."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-12)
    R = Rotation.from_rotvec(axis * float(angle)).as_matrix()

    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = np.asarray(point, dtype=float) + R @ (T[:3, 3] - np.asarray(point, dtype=float))
    return out


def clamp_pose_delta(T_ref, T, bounds_xyz, bounds_rpy):
    """Clamp T so it stays within +/- bounds of T_ref, measured IN T_REF'S OWN FRAME.

    The bound is on the relative pose inv(T_ref) @ T, so `bounds_xyz` is read in the reference
    frame's axes (for the scan, that means camera x/y/z: sideways, up-down, along the view ray)."""
    rel = inverse(T_ref) @ T
    xyz, rpy = matrix_to_xyzrpy(rel)
    xyz = np.clip(xyz, -np.abs(bounds_xyz), np.abs(bounds_xyz))
    rpy = np.clip(rpy, -np.abs(bounds_rpy), np.abs(bounds_rpy))
    return T_ref @ xyzrpy_to_matrix(xyz, rpy)


# ------------------------------------------------------------------ wrenches
def transform_wrench(f, tau, T_ba):
    """Re-express a wrench measured in frame A into frame B, given T_ba.

    f_B = R . f_A ;  tau_B = R . tau_A + p x f_B

    The cross term is the one people forget: moving a wrench to a new origin adds the moment of
    the force about the offset. Forces are frame-invariant in magnitude, torques are NOT."""
    R, p = T_ba[:3, :3], T_ba[:3, 3]
    f_b = R @ np.asarray(f, dtype=float)
    tau_b = R @ np.asarray(tau, dtype=float) + np.cross(p, f_b)
    return f_b, tau_b


# ------------------------------------------------------------------ formatting
def fmt_pose(T, label=''):
    """One-line 'xyz=[...] m rpy=[...] deg' for logs."""
    xyz, rpy = matrix_to_xyzrpy(T)
    deg = np.degrees(rpy)
    s = (f'xyz=[{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}] m  '
         f'rpy=[{deg[0]:+.1f}, {deg[1]:+.1f}, {deg[2]:+.1f}] deg')
    return f'{label}: {s}' if label else s


def fmt_delta(T_cur, T_des):
    """'dist=.. mm, angle=.. deg' summary of the gap between two poses."""
    lin, ang = pose_error(T_cur, T_des)
    return f'dist={lin * 1000:.1f} mm, angle={np.degrees(ang):.1f} deg'
