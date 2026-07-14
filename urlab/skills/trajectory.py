"""Trajectory tooling for the kinematic-assembly and uncertain-sampling demos.

All pure numpy, lifted verbatim from KinematicAssembly / UncertainAssemblySampling (the audit
flagged these as already ROS-free): CSV loading, resampling, the target-frame anchoring trick,
per-trial perturbation, and the closest-pose disassembly snap.

THE ANCHORING TRICK (worth understanding before you touch it). The ground truth is the
ASSEMBLED tool0 pose in base -- you jog the arm to a good mate and read it off the robot. The
target OBJECT frame is never measured; it is back-derived so that the LAST trajectory row maps
EXACTLY onto that assembled pose:

    T_base_targetobj = assembled_pose @ held_object_pose @ inv(traj[-1])

Every row then maps to a tool0 command via
    T_base_tool0(row) = T_base_targetobj @ T_targetobj_held(row) @ inv(T_tool0_held)
and the last row collapses back to assembled_pose by construction. That self-consistency is the
whole point -- do not try to measure the target frame independently.
"""

import csv
import math

import numpy as np

from .. import log as urlog
from ..transforms import (
    inverse, matrix_to_quat, matrix_to_xyzrpy, slerp_matrix, xyzrpy_to_matrix)

log = urlog.get('trajectory')


def load_csv(path, angles_deg=False):
    """Load an (x, y, z, roll, pitch, yaw) trajectory as a list of 4x4 T_targetobj_held.

    Skips blanks, '#' comments, and any row whose first six cells do not all parse as floats --
    which is how the header line is silently dropped."""
    mats = []
    with open(path, 'r') as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith('#'):
                continue
            try:
                vals = [float(x) for x in row[:6]]
            except ValueError:
                continue
            if len(vals) < 6:
                continue
            rpy = [math.radians(a) for a in vals[3:6]] if angles_deg else vals[3:6]
            mats.append(xyzrpy_to_matrix(vals[0:3], rpy))
    return mats


def anchor_target(assembled_pose, held_object_pose, traj_last):
    """T_base_targetobj, back-derived so traj[-1] maps exactly onto the assembled pose."""
    T_base_held_assembled = assembled_pose @ held_object_pose
    return T_base_held_assembled @ inverse(traj_last)


def tool0_at(T_base_targetobj, T_targetobj_held, T_tool0_held):
    """A trajectory row (held-obj-vs-target) reduced to a tool0 pose in base."""
    return T_base_targetobj @ T_targetobj_held @ inverse(T_tool0_held)


def resample(mats, res_t=0.001, res_r_deg=1.0):
    """Densify so consecutive poses are within res_t metres and res_r_deg degrees.

    n = max(1, ceil(dp / res_t), ceil(dtheta / res_r)) subdivisions per segment; LERP position,
    SLERP orientation. The geodesic angle is 2*arccos(|qa.qb|) (shortest path)."""
    res_r = math.radians(res_r_deg)
    dense = [mats[0]]
    for a, b in zip(mats[:-1], mats[1:]):
        dp = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
        dot = min(1.0, abs(float(np.dot(matrix_to_quat(a), matrix_to_quat(b)))))
        dtheta = float(2.0 * np.arccos(dot))
        n = 1
        if res_t > 0:
            n = max(n, int(np.ceil(dp / res_t)))
        if res_r > 0:
            n = max(n, int(np.ceil(dtheta / res_r)))
        for k in range(1, n + 1):
            dense.append(slerp_matrix(a, b, k / n))
    return dense


def random_delta(bounds, rng):
    """A random pose perturbation from per-dimension half-widths [x,y,z (m), r,p,y (deg)]."""
    b = np.asarray(bounds, dtype=float)
    t = rng.uniform(-1.0, 1.0, 3) * b[0:3]
    r = np.radians(rng.uniform(-1.0, 1.0, 3) * b[3:6])
    return xyzrpy_to_matrix(t, r)


def perturb(dense, bias_bounds, noise_bounds, rng):
    """perturbed[i] = bias @ noise_i @ dense[i], in the TARGET-OBJECT frame.

    bias is ONE draw for the whole trial (a systematic offset); noise is a fresh draw per waypoint
    (jitter). Both act about the target origin, so a rotational bias also translates the part by
    (R_bias - I) @ p -- replicate exactly if you want data matching the ROS runs."""
    bias = random_delta(bias_bounds, rng)
    return [bias @ random_delta(noise_bounds, rng) @ c for c in dense]


def closest_index(actual, dense, rot_weight_mm_per_deg=1.0):
    """Index of the dense pose closest to `actual`, by cost = |dp|_mm + w * dtheta_deg."""
    p = actual[:3, 3]
    qa = matrix_to_quat(actual)
    best_i, best_cost = 0, float('inf')
    for i, d in enumerate(dense):
        dt_mm = float(np.linalg.norm(d[:3, 3] - p)) * 1000.0
        dot = min(1.0, abs(float(np.dot(qa, matrix_to_quat(d)))))
        dr_deg = math.degrees(2.0 * math.acos(dot))
        cost = dt_mm + rot_weight_mm_per_deg * dr_deg
        if cost < best_cost:
            best_i, best_cost = i, cost
    return best_i


def pose_fields(T):
    """[x, y, z, qx, qy, qz, qw, yaw_deg, pitch_deg, roll_deg] for a CSV log row."""
    q = matrix_to_quat(T)
    _, rpy = matrix_to_xyzrpy(T)
    roll, pitch, yaw = rpy
    return [T[0, 3], T[1, 3], T[2, 3], q[0], q[1], q[2], q[3],
            math.degrees(yaw), math.degrees(pitch), math.degrees(roll)]


def ik_chain(arm, poses, seed):
    """IK a sequence of tool0 poses, each seeded from the previous solution -- keeps the whole path
    on one IK branch. Returns the joint list, or None on the first unreachable pose."""
    out = []
    q = list(seed)
    for i, pose in enumerate(poses):
        q = arm.ik(pose, q)
        if q is None:
            log.error('IK failed at waypoint %d/%d.', i + 1, len(poses))
            return None
        out.append(q)
    return out
